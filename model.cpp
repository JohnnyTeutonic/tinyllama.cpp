#include "model.h"
#include "logger.h"
#include <stdexcept>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <cblas.h> // Include CBLAS header for BLAS calls
#include <omp.h> // Include OpenMP header
#ifdef _WIN32
#include <windows.h>
#endif
#ifdef __has_include
#  if __has_include(<openssl/sha.h>)
#    include <openssl/sha.h>
#    define HAVE_OPENSSL_SHA
#  endif
#endif
#include <cstdint> // For uintptr_t, uint16_t

// --- START: New Helper Functions for BFloat16 ---

// Convert a single bfloat16 (uint16_t) to float32
float bfloat16_to_float32(uint16_t b16) {
    uint32_t f = ((uint32_t)b16) << 16;
    float f32;
    std::memcpy(&f32, &f, sizeof(float));
    return f32;
}

// Convert raw bytes (vector<uint8_t>) to vector<uint16_t> for bfloat16 storage
static std::vector<uint16_t> uint8_vector_to_uint16_vector(const std::vector<uint8_t>& bytes, size_t numel) {
    if (bytes.size() != numel * 2) {
        throw std::runtime_error("Byte vector size mismatch for bfloat16 conversion");
    }
    std::vector<uint16_t> out(numel);
    for (size_t i = 0; i < numel; ++i) {
        // Assuming little-endian storage in bytes (consistent with original bfloat16_to_float32 logic)
        out[i] = (bytes[2 * i + 1] << 8) | bytes[2 * i];
    }
    return out;
}

// --- END: New Helper Functions for BFloat16 ---

// --- START: Argmax Helper ---
// Find the index of the maximum element in a vector
int argmax(const std::vector<float>& v) {
    if (v.empty()) {
        throw std::runtime_error("Cannot perform argmax on empty vector");
    }
    return std::distance(v.begin(), std::max_element(v.begin(), v.end()));
}
// --- END: Argmax Helper ---

// RMSNorm: Match PyTorch's likely calculation order
void rmsnorm(const std::vector<float>& x, const std::vector<uint16_t>& weight, float eps, std::vector<float>& out) {
    size_t N_full = x.size();
    if (N_full == 0) { 
        Logger::error("RMSNorm Error: Input vector x is empty.");
        return; 
    }

    // 1. Calculate sum of squares (use double for stability)
    double ss = 0.0;
    #pragma omp parallel for reduction(+:ss)
    for (size_t i = 0; i < N_full; ++i) {
        ss += double(x[i]) * double(x[i]);
    }

    // 2. Calculate RMS = sqrt(mean_sq)
    double mean_sq = ss / double(N_full);
    double rms_val = std::sqrt(mean_sq); // Calculate sqrt directly

    // 3. Apply the formula: output = x * (weight / (rms + eps))
    #pragma omp parallel for
    for (size_t i = 0; i < N_full; ++i) {
        double weight_f = bfloat16_to_float32(weight[i]);
        // Calculate scaling factor using double for intermediate precision
        double scale = weight_f / (rms_val + double(eps)); 
        double scaled_x = double(x[i]) * scale;
        out[i] = float(scaled_x); // Cast final result back to float
    }
}

// SiLU activation: y = x * sigmoid(x)
static void silu(const std::vector<float>& x, std::vector<float>& out) {
    for (size_t i = 0; i < x.size(); ++i) {
        out[i] = x[i] / (1.0f + std::exp(-x[i]));
    }
}

// RoPE: apply rotary positional embedding to Q or K (in-place)
void apply_rope(std::vector<float>& x, int num_heads, int head_dim, int pos, float rope_theta) {
    // x: [num_heads, head_dim]
    // Only even/odd pairs are rotated
    float theta = rope_theta;
    // Parallelize the loop over heads
    #pragma omp parallel for
    for (int h = 0; h < num_heads; ++h) {
        for (int i = 0; i < head_dim; i += 2) {
            // Ensure floating point division for frequency calculation
            float freq = std::pow(theta, -((float)i) / head_dim);
            float angle = pos * freq;
            float x0 = x[h * head_dim + i];
            float x1 = x[h * head_dim + i + 1];
            x[h * head_dim + i]     = x0 * std::cos(angle) - x1 * std::sin(angle);
            x[h * head_dim + i + 1] = x0 * std::sin(angle) + x1 * std::cos(angle);
        }
    }
}

// Softmax over a vector (in-place)
void softmax(std::vector<float>& x) {
    if (x.empty()) return;
    float maxv = x[0];
    for (size_t i = 1; i < x.size(); ++i) {
         if (x[i] > maxv) maxv = x[i];
    }
    // Accumulate in double for sum (matches PyTorch)
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = std::exp(x[i] - maxv);
        sum += double(x[i]);
    }
    #pragma omp parallel for
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = float(double(x[i]) / sum);
    }
}

// Helper: log vector stats (min, max, mean, all finite)
static void log_vec_stats(const std::string& name, const std::vector<float>& v) {
    float minv = *std::min_element(v.begin(), v.end());
    float maxv = *std::max_element(v.begin(), v.end());
    float mean = std::accumulate(v.begin(), v.end(), 0.0f) / v.size();
    bool all_finite = std::all_of(v.begin(), v.end(), [](float x) { return std::isfinite(x); });
    Logger::info(name + ": min=" + std::to_string(minv) + ", max=" + std::to_string(maxv) + ", mean=" + std::to_string(mean) + ", all_finite=" + (all_finite ? "yes" : "no"));
}

// Helper function to write vector<float> to a binary file
static bool write_vector_to_file(const std::string& filename, const std::vector<float>& vec) {
    // --- START: Log vector received by writer ---
    std::string vec_writer_vals;
    int N_log_writer = std::min(10, (int)vec.size());
    for (int i = 0; i < N_log_writer; ++i) vec_writer_vals += (i ? " " : "") + std::to_string(vec[i]);
    // --- END: Log vector received by writer ---
    // --- START: Log vector data address in writer ---
    Logger::info("write_vector_to_file Enter: Address of vec.data() on entry: " + std::to_string(reinterpret_cast<uintptr_t>(vec.data())));
    // --- END: Log vector data address in writer ---

    std::ofstream outfile(filename, std::ios::binary);
    if (!outfile) {
        Logger::error("Failed to open file for writing: " + filename);
        return false;
    }
    outfile.write(reinterpret_cast<const char*>(vec.data()), vec.size() * sizeof(float));
    if (!outfile) {
        Logger::error("Failed to write data to file: " + filename);
        return false;
    }
    Logger::info("Successfully wrote vector to " + filename);
    return true;
}

// Helper: load a [num_tokens, hidden_size] float32 .bin file (row-major)
static std::vector<std::vector<float>> load_rmsnorm_bin(const std::string& filename, int num_tokens, int hidden_size) {
    std::ifstream infile(filename, std::ios::binary);
    if (!infile) throw std::runtime_error("Failed to open " + filename);
    std::vector<float> flat(num_tokens * hidden_size);
    infile.read(reinterpret_cast<char*>(flat.data()), flat.size() * sizeof(float));
    if (!infile) throw std::runtime_error("Failed to read all data from " + filename);
    std::vector<std::vector<float>> result(num_tokens, std::vector<float>(hidden_size));
    for (int t = 0; t < num_tokens; ++t)
        for (int h = 0; h < hidden_size; ++h)
            result[t][h] = flat[t * hidden_size + h];
    return result;
}

// Parse ModelConfig from nlohmann::json
ModelConfig parse_model_config(const nlohmann::json& json) {
    ModelConfig cfg;
    cfg.hidden_size = json.value("hidden_size", 0);
    cfg.intermediate_size = json.value("intermediate_size", 0);
    cfg.num_attention_heads = json.value("num_attention_heads", 0);
    cfg.num_key_value_heads = json.value("num_key_value_heads", 0);
    cfg.num_hidden_layers = json.value("num_hidden_layers", 0);
    cfg.vocab_size = json.value("vocab_size", 0);
    cfg.max_position_embeddings = json.value("max_position_embeddings", 0);
    cfg.rms_norm_eps = json.value("rms_norm_eps", 1e-5f);
    cfg.rope_theta = json.value("rope_theta", 10000.0f);
    cfg.hidden_act = json.value("hidden_act", "silu");
    cfg.torch_dtype = json.value("torch_dtype", "bfloat16");
    cfg.bos_token_id = json.value("bos_token_id", 1);
    cfg.eos_token_id = json.value("eos_token_id", 2);
    return cfg;
}

// --- START: New MatVec for BFloat16 Weights ---
// Matrix multiplication: out = mat [M,N] * vec [N] -> [M]
// mat: uint16_t (bfloat16), vec: float, out: float
void matvec_bf16_f32(const std::vector<uint16_t>& mat, const std::vector<float>& vec, std::vector<float>& out, int M, int N) {
    if (mat.size() != M * N) {
        throw std::runtime_error("Matrix size mismatch in matvec_bf16_f32");
    }
    if (vec.size() != N) {
        throw std::runtime_error("Vector size mismatch in matvec_bf16_f32");
    }
    if (out.size() != M) {
         out.resize(M); // Ensure output vector has correct size
    }
    // Accumulate in double for maximum stability (matches PyTorch)
    #pragma omp parallel for
    for (int i = 0; i < M; ++i) {
        double sum = 0.0;
        for (int j = 0; j < N; ++j) {
            float mat_val = bfloat16_to_float32(mat[i * N + j]);
            sum += double(mat_val) * double(vec[j]);
        }
        out[i] = float(sum);
    }
}
// --- END: New MatVec for BFloat16 Weights ---

// TinyLlamaModel constructor: load all weights from safetensors into uint16_t vectors
TinyLlamaModel::TinyLlamaModel(const ModelConfig& config, const SafeTensorsLoader& loader)
    : config_(config)
{
    int hs = config.hidden_size;
    int is = config.intermediate_size;
    int nhl = config.num_hidden_layers;
    int vs = config.vocab_size;
    int n_heads = config.num_attention_heads;
    int n_kv_heads = config.num_key_value_heads;
    int kv_dim = (hs / n_heads) * n_kv_heads;
    Logger::info("K/V projection kv_dim: " + std::to_string(kv_dim));

    // Load weights directly into uint16_t vectors
    try {
        embed_tokens = uint8_vector_to_uint16_vector(loader.get_tensor_bytes("model.embed_tokens.weight"), vs * hs);
    } catch (const std::exception& e) {
        Logger::error("Missing model.embed_tokens.weight: " + std::string(e.what()));
    }
    try {
        lm_head = uint8_vector_to_uint16_vector(loader.get_tensor_bytes("lm_head.weight"), vs * hs);
    } catch (const std::exception& e) {
        Logger::error("Missing lm_head.weight: " + std::string(e.what()));
    }
    try {
        final_norm = uint8_vector_to_uint16_vector(loader.get_tensor_bytes("model.norm.weight"), hs);
    } catch (const std::exception& e) {
        Logger::error("Missing model.norm.weight: " + std::string(e.what()));
    }

    layers.resize(nhl);
    for (int i = 0; i < nhl; ++i) {
        auto& lw = layers[i];
        std::string prefix = "model.layers." + std::to_string(i) + ".";
        try {
            lw.input_layernorm = uint8_vector_to_uint16_vector(loader.get_tensor_bytes(prefix + "input_layernorm.weight"), hs);
            lw.post_attention_layernorm = uint8_vector_to_uint16_vector(loader.get_tensor_bytes(prefix + "post_attention_layernorm.weight"), hs);
            lw.q_proj = uint8_vector_to_uint16_vector(loader.get_tensor_bytes(prefix + "self_attn.q_proj.weight"), hs * hs);
            lw.k_proj = uint8_vector_to_uint16_vector(loader.get_tensor_bytes(prefix + "self_attn.k_proj.weight"), hs * kv_dim);
            lw.v_proj = uint8_vector_to_uint16_vector(loader.get_tensor_bytes(prefix + "self_attn.v_proj.weight"), hs * kv_dim);
            lw.o_proj = uint8_vector_to_uint16_vector(loader.get_tensor_bytes(prefix + "self_attn.o_proj.weight"), hs * hs);
            lw.gate_proj = uint8_vector_to_uint16_vector(loader.get_tensor_bytes(prefix + "mlp.gate_proj.weight"), is * hs);
            lw.up_proj = uint8_vector_to_uint16_vector(loader.get_tensor_bytes(prefix + "mlp.up_proj.weight"), is * hs);
            lw.down_proj = uint8_vector_to_uint16_vector(loader.get_tensor_bytes(prefix + "mlp.down_proj.weight"), hs * is);

            // Log expected dimensions and loaded size for Layer 0 Q/K proj
            if (i == 0) {
                int expected_q_M = hs;
                int expected_q_N = hs;
                size_t expected_q_size = (size_t)expected_q_M * expected_q_N;
                Logger::info("C++ Layer 0 q_proj expected M, N: " + std::to_string(expected_q_M) + ", " + std::to_string(expected_q_N));
                Logger::info("C++ Layer 0 q_proj loaded size: " + std::to_string(lw.q_proj.size()) + " (Expected: " + std::to_string(expected_q_size) + ")");
                if (lw.q_proj.size() != expected_q_size) {
                    Logger::error("C++ Layer 0 q_proj SIZE MISMATCH!");
                }

                int expected_k_M = kv_dim;
                int expected_k_N = hs;
                size_t expected_k_size = (size_t)expected_k_M * expected_k_N;
                 Logger::info("C++ Layer 0 k_proj expected M, N: " + std::to_string(expected_k_M) + ", " + std::to_string(expected_k_N));
                 Logger::info("C++ Layer 0 k_proj loaded size: " + std::to_string(lw.k_proj.size()) + " (Expected: " + std::to_string(expected_k_size) + ")");
                 if (lw.k_proj.size() != expected_k_size) {
                     Logger::error("C++ Layer 0 k_proj SIZE MISMATCH!");
                 }
            }

            // Print first 5 raw uint16_t values of q_proj for this layer (for basic check)
            std::ostringstream oss;
            oss << "C++ layer " << i << " q_proj first 5 (uint16_t): ";
            for (int j = 0; j < 5 && j < lw.q_proj.size(); ++j) oss << lw.q_proj[j] << " ";
            Logger::info(oss.str());
        } catch (const std::exception& e) {
            Logger::error("Missing or malformed weights in layer " + std::to_string(i) + ": " + e.what());
        }
    }
    Logger::info("All model weights loaded (as bfloat16/uint16_t).");

    // Diagnostics still need update later to handle uint16_t if needed, 
    // or only log float activations
    // Logger::info("Embedding shape: ...");
    // Logger::log_vector_stats("embed_tokens", embed_tokens, 10); // Error: expects float
    // ... other diagnostics ...
}

// Forward pass - Updated for KVCache store/retrieve
std::vector<float> TinyLlamaModel::forward(int input_id, int token_idx, KVCache* cache, ForwardDiagCallback diag_cb) {
    // Check if cache is valid and initialized (basic check)
    if (cache && cache->layers.empty() && config_.num_hidden_layers > 0) {
        // This shouldn't happen if main initializes it, but as a safeguard:
        Logger::error("KVCache passed to forward but layers vector is empty!");
        // Handle error appropriately, maybe throw or return empty
        return {};
    }

    // --- Config dimensions ---
    int hs = config_.hidden_size;
    int is = config_.intermediate_size;
    int nhl = config_.num_hidden_layers;
    int vs = config_.vocab_size;
    int n_heads = config_.num_attention_heads;
    int n_kv_heads = config_.num_key_value_heads;
    int head_dim = hs / n_heads;
    // IMPORTANT: Need max sequence length for cache indexing
    int max_seq_len = config_.max_position_embeddings;
    float eps = config_.rms_norm_eps;
    float rope_theta = config_.rope_theta;
    // 'pos' is the index where the *current* token's KV will be stored
    int pos = cache ? cache->seq_len : 0; // Use the current length as the position index

    // Ensure pos is within bounds
    if (cache && pos >= max_seq_len) {
         Logger::error("KVCache position exceeds max_seq_len!");
         return {}; // Or handle sequence length limit
    }

    // --- Logging setup (remains same) ---
    // std::ofstream cpp_log_stream;
    // const std::string cpp_log_filename = "layer_0_cpp_outputs.log";
    // bool opened_log_stream = false;
    // if (token_idx == 0) { 
    //     cpp_log_stream.open(cpp_log_filename, std::ios::trunc);
    //     if (!cpp_log_stream) {
    //         Logger::error("Failed to open C++ log file: " + cpp_log_filename);
    //     } else {
    //         opened_log_stream = true;
    //         Logger::info("Opened " + cpp_log_filename + " for Layer 0 logging (truncating).");
    //     }
    // }
    // auto log_to_file = [&](const std::string& key, const std::vector<float>& vec) {
    //      if (opened_log_stream) { /* ... */ }
    // };

    // 1. Embedding lookup (remains same)
    std::vector<float> x(hs);
    for (int i = 0; i < hs; ++i) {
        x[i] = bfloat16_to_float32(embed_tokens[input_id * hs + i]);
    }
    Logger::info("Embedding lookup complete (converted to float32).");
    log_vec_stats("embedding (float32)", x);
    if (!std::all_of(x.begin(), x.end(), [](float v){ return std::isfinite(v); })) {
        Logger::error("NaN/Inf detected in embedding vector after lookup!");
    }
    if (diag_cb) diag_cb(-1, "embedding", x);
    // if (input_id == 1 && opened_log_stream) { /* ... log x ... */ }

    // 2. Transformer blocks
    for (int l = 0; l < nhl; ++l) {
        const auto& lw = layers[l];

        // if (l == 0) log_to_file("x (Layer Input)", x);

        // a) Input RMSNorm (remains same)
        std::vector<float> x_norm(hs);
        rmsnorm(x, lw.input_layernorm, eps, x_norm);
        if (l == 0) {
            std::stringstream ss_x_norm_raw;
            ss_x_norm_raw << "Layer 0 RMSNorm1 Output (Input to Proj) first 5: ["; // Changed label back
            for(int i=0; i < 5 && i < x_norm.size(); ++i) ss_x_norm_raw << (i > 0 ? " " : "") << x_norm[i];
            ss_x_norm_raw << "]";
            Logger::info(ss_x_norm_raw.str());
        }
        if (!std::all_of(x_norm.begin(), x_norm.end(), [](float v){ return std::isfinite(v); })) {
            Logger::error("NaN/Inf detected in x_norm after input RMSNorm, layer " + std::to_string(l));
        }
        // if (l == 0) log_to_file("x_norm (Input RMSNorm Out)", x_norm);

        // b) Attention Q, K, V projections (remains same)
        std::vector<float> q(hs), k_current(head_dim * n_kv_heads), v_current(head_dim * n_kv_heads);
        matvec_bf16_f32(lw.q_proj, x_norm, q, hs, hs);
        matvec_bf16_f32(lw.k_proj, x_norm, k_current, head_dim * n_kv_heads, hs);
        matvec_bf16_f32(lw.v_proj, x_norm, v_current, head_dim * n_kv_heads, hs);
        // Log Q/K raw projection outputs before RoPE for Layer 0
        if (l == 0) {
            std::stringstream ss_q_proj_raw;
            ss_q_proj_raw << "Layer 0 Q Projection Output (Pre-RoPE) first 5: [";
            for(int i=0; i < 5 && i < q.size(); ++i) ss_q_proj_raw << (i > 0 ? " " : "") << q[i];
            ss_q_proj_raw << "]";
            Logger::info(ss_q_proj_raw.str());

            std::stringstream ss_k_proj_raw;
            ss_k_proj_raw << "Layer 0 K Projection Output (Pre-RoPE) first 5: [";
            for(int i=0; i < 5 && i < k_current.size(); ++i) ss_k_proj_raw << (i > 0 ? " " : "") << k_current[i];
            ss_k_proj_raw << "]";
            Logger::info(ss_k_proj_raw.str());
        }
        // Log Q/K/V projection stats
        if (l == 0) {
            log_vec_stats("Layer " + std::to_string(l) + " Q projection", q);
            log_vec_stats("Layer " + std::to_string(l) + " K projection", k_current);
            log_vec_stats("Layer " + std::to_string(l) + " V projection", v_current);
            // Q before RoPE (logging raw values above now)
            // Logger::info("Layer " + std::to_string(l) + " Q before RoPE shape: [" + std::to_string(q.size()) + "] first 5: " + std::to_string(q[0]) + " " + std::to_string(q[1]) + " " + std::to_string(q[2]) + " " + std::to_string(q[3]) + " " + std::to_string(q[4]));
        }
        // RoPE (operates on float Q/K copies)
        std::vector<float> q_heads(q); // Full Q projection
        apply_rope(q_heads, n_heads, head_dim, pos, rope_theta);
        std::vector<float> k_rope_current = k_current;
        apply_rope(k_rope_current, n_kv_heads, head_dim, pos, rope_theta);
        // Q after RoPE
        Logger::info("Layer " + std::to_string(l) + " Q after RoPE shape: [" + std::to_string(q_heads.size()) + "] first 5: " + std::to_string(q_heads[0]) + " " + std::to_string(q_heads[1]) + " " + std::to_string(q_heads[2]) + " " + std::to_string(q_heads[3]) + " " + std::to_string(q_heads[4]));
        // K after RoPE
        Logger::info("Layer " + std::to_string(l) + " K after RoPE shape: [" + std::to_string(k_rope_current.size()) + "] first 5: " + std::to_string(k_rope_current[0]) + " " + std::to_string(k_rope_current[1]) + " " + std::to_string(k_rope_current[2]) + " " + std::to_string(k_rope_current[3]) + " " + std::to_string(k_rope_current[4]));
        // Store current K, V to cache
        if (cache) {
            size_t cache_offset_base = (size_t)pos * head_dim;
            size_t layer_stride = (size_t)max_seq_len * head_dim;
            for (int kvh = 0; kvh < n_kv_heads; ++kvh) {
                size_t head_offset = (size_t)kvh * layer_stride;
                size_t current_k_offset = (size_t)kvh * head_dim;
                size_t current_v_offset = (size_t)kvh * head_dim;
                if (head_offset + cache_offset_base + head_dim <= cache->layers[l].k.size()) {
                    std::memcpy(&cache->layers[l].k[head_offset + cache_offset_base],
                                &k_rope_current[current_k_offset],
                                head_dim * sizeof(float));
                    std::memcpy(&cache->layers[l].v[head_offset + cache_offset_base],
                                &v_current[current_v_offset],
                                head_dim * sizeof(float));
                } else {
                    Logger::error("Cache write out of bounds! Layer=" + std::to_string(l) + " Pos=" + std::to_string(pos));
                }
            }
        }

        // Attention Calculation using Cache
        int kv_dim = head_dim * n_kv_heads;
        std::vector<float> attn_out(hs, 0.0f);
        #pragma omp parallel for
        for (int h = 0; h < n_heads; ++h) {
            if (h >= n_kv_heads) {
                // For heads beyond n_kv_heads, set output to zero (already zero-initialized)
                continue;
            }
            int kvh = h;
            float* current_q_head = q_heads.data() + h * head_dim;
            // Log Q head for Layer 0, Head 0
            if (l == 0 && h == 0) {
                 std::stringstream ss_q_head;
                 ss_q_head << "Layer 0 Q Head 0, Pos 0 (Input to Score) first 5: [";
                 for(int i = 0; i < 5 && i < head_dim; ++i) ss_q_head << (i > 0 ? " " : "") << current_q_head[i];
                 ss_q_head << "]";
                 Logger::info(ss_q_head.str());
            }
            std::vector<float> scores(pos + 1);
            size_t cache_layer_stride = (size_t)max_seq_len * head_dim;
            size_t cache_head_offset = (size_t)kvh * cache_layer_stride;
            for (int t = 0; t <= pos; ++t) {
                float score = 0.0f;
                float* cached_k_timestep = cache ? cache->layers[l].k.data() + cache_head_offset + t * head_dim : nullptr;
                 // Log K head for Layer 0, Head 0, Timestep 0
                if (l == 0 && h == 0 && t == 0) {
                     float* k_head_ptr = cache ? cached_k_timestep : k_rope_current.data() + kvh * head_dim;
                     std::stringstream ss_k_head;
                     ss_k_head << "Layer 0 K Head 0, Pos 0 (Input to Score) first 5: [";
                     if (k_head_ptr) {
                         for(int i = 0; i < 5 && i < head_dim; ++i) ss_k_head << (i > 0 ? " " : "") << k_head_ptr[i];
                     } else {
                         ss_k_head << "ERROR_K_HEAD_NULL"; // Should not happen for t=0 unless no cache and k_rope_current is bad
                     }
                     ss_k_head << "]";
                     Logger::info(ss_k_head.str());
                }
                if (!cached_k_timestep && cache) {
                    Logger::error("Cache read K out of bounds! Layer=" + std::to_string(l) + " Pos=" + std::to_string(t));
                    score = -std::numeric_limits<float>::infinity();
                } else {
                    for (int i = 0; i < head_dim; ++i) {
                        float k_val = cache ? cached_k_timestep[i] : k_rope_current[kvh * head_dim + i];
                        score += current_q_head[i] * k_val;
                    }
                }
                scores[t] = score / std::sqrt(float(head_dim));
            }
            // Log scores before softmax for Layer 0, Head 0
            if (l == 0 && h == 0) {
                std::stringstream ss_scores_before;
                ss_scores_before << "Layer 0 Attention Scores (Head 0, Pos 0, Before Softmax): [";
                for(size_t i = 0; i < scores.size(); ++i) ss_scores_before << (i > 0 ? " " : "") << scores[i];
                ss_scores_before << "]";
                Logger::info(ss_scores_before.str());
            }
            softmax(scores);
            // Log probabilities after softmax for Layer 0, Head 0
            if (l == 0 && h == 0) {
                std::stringstream ss_scores_after;
                ss_scores_after << "Layer 0 Attention Probs (Head 0, Pos 0): [";
                for(size_t i = 0; i < scores.size(); ++i) ss_scores_after << (i > 0 ? " " : "") << scores[i];
                ss_scores_after << "]";
                Logger::info(ss_scores_after.str());
            }
            float* attn_out_head = attn_out.data() + h * head_dim;
            std::fill(attn_out_head, attn_out_head + head_dim, 0.0f);
            for (int t = 0; t <= pos; ++t) {
                float* cached_v_timestep = cache ? cache->layers[l].v.data() + cache_head_offset + t * head_dim : nullptr;
                if (!cached_v_timestep && cache) {
                    Logger::error("Cache read V out of bounds! Layer=" + std::to_string(l) + " Pos=" + std::to_string(t));
                    continue;
                }
                float score_t = scores[t];
                for (int i = 0; i < head_dim; ++i) {
                    float v_val = cache ? cached_v_timestep[i] : v_current[kvh * head_dim + i];
                    attn_out_head[i] += score_t * v_val;
                }
            }
            // Log first 5 elements of attn_out_head for Layer 0, Head 0
            if (l == 0 && h == 0) {
                std::stringstream ss_attn_out_head;
                ss_attn_out_head << "Layer 0 Attention Out (Head 0, Pos 0, Before o_proj) first 5: [";
                for(int i = 0; i < 5 && i < head_dim; ++i) ss_attn_out_head << (i > 0 ? " " : "") << attn_out_head[i];
                ss_attn_out_head << "]";
                Logger::info(ss_attn_out_head.str());
            }
        }
        // Log attention output stats
        if (l == 0) {
            log_vec_stats("Layer " + std::to_string(l) + " attn_out (Before o_proj)", attn_out);
        }
        // Output projection
        std::vector<float> attn_proj(hs);
        matvec_bf16_f32(lw.o_proj, attn_out, attn_proj, hs, hs);
        // Log projected attention output stats
        if (l == 0) {
             log_vec_stats("Layer " + std::to_string(l) + " attn_out (Projected)", attn_proj);
        }
        // Residual addition
        for (int i = 0; i < hs; ++i) x[i] += attn_proj[i];
        log_vec_stats("Layer " + std::to_string(l) + " post-attn residual", x);
        // d) Post-attention RMSNorm
        std::vector<float> ffn_norm(hs);
        rmsnorm(x, lw.post_attention_layernorm, eps, ffn_norm);
        // e) MLP
        std::vector<float> gate(is), up(is), silu_gate(is), ffn_out(hs);
        matvec_bf16_f32(lw.gate_proj, ffn_norm, gate, is, hs);
        matvec_bf16_f32(lw.up_proj, ffn_norm, up, is, hs);
        silu(gate, silu_gate);
        for (int i = 0; i < is; ++i) up[i] *= silu_gate[i];
        matvec_bf16_f32(lw.down_proj, up, ffn_out, hs, is);
        log_vec_stats("Layer " + std::to_string(l) + " MLP output", ffn_out);
        // Residual addition after MLP
        for (int i = 0; i < hs; ++i) x[i] += ffn_out[i];
        log_vec_stats("Layer " + std::to_string(l) + " post-MLP residual", x);
    }

    // Final RMSNorm
    std::vector<float> x_norm(hs);
    rmsnorm(x, final_norm, eps, x_norm);
    if (!std::all_of(x_norm.begin(), x_norm.end(), [](float v){ return std::isfinite(v); })) {
        Logger::error("NaN/Inf detected in final RMSNorm output");
    }

    // LM head projection
    std::vector<float> logits(vs);
    matvec_bf16_f32(lm_head, x_norm, logits, vs, hs);
    if (!std::all_of(logits.begin(), logits.end(), [](float v){ return std::isfinite(v); })) {
        Logger::error("NaN/Inf detected in logits before return");
    }
    // Log final logits stats before return
    log_vec_stats("final logits", logits);
    Logger::info("Forward pass complete.");
    return logits;
} 