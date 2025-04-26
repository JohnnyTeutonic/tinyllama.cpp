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
std::vector<uint16_t> uint8_vector_to_uint16_vector(const std::vector<uint8_t>& bytes, size_t numel) {
    if (bytes.size() != numel * 2) {
        throw std::runtime_error("Byte vector size mismatch for bfloat16 conversion");
    }
    std::vector<uint16_t> out(numel);
    for (size_t i = 0; i < numel; ++i) {
        // --- Revert to Little-Endian storage assumption --- 
        out[i] = (bytes[2 * i + 1] << 8) | bytes[2 * i];
        // Big-Endian Attempt: out[i] = (bytes[2 * i] << 8) | bytes[2 * i + 1]; 
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

// Softmax over a vector (in-place) - Use float accumulation
void softmax(std::vector<float>& x) {
    if (x.empty()) return;
    float maxv = x[0];
    #pragma omp parallel for reduction(max:maxv) // Ensure max finding is parallel safe if needed
    for (size_t i = 1; i < x.size(); ++i) {
         if (x[i] > maxv) maxv = x[i];
    }
    // --- CHANGE: Accumulate sum in float ---
    float sum = 0.0f;
    #pragma omp parallel for reduction(+:sum)
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = expf(x[i] - maxv); // Use global namespace expf
        sum += x[i]; // Accumulate as float
    }
    // --- CHANGE: Divide by float sum ---
    #pragma omp parallel for
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = x[i] / sum; // Divide by float sum
    }
}

// SiLU activation: y = x * sigmoid(x) - Match PyTorch structure
static void silu(const std::vector<float>& x, std::vector<float>& out) {
    #pragma omp parallel for
    for (size_t i = 0; i < x.size(); ++i) {
        float sigmoid_x = 1.0f / (1.0f + std::exp(-x[i])); // Calculate sigmoid explicitly
        out[i] = x[i] * sigmoid_x; // Multiply
    }
}

// RoPE: apply rotary positional embedding to Q or K (in-place)
void apply_rope(std::vector<float>& x, int num_heads, int head_dim, int pos, const std::vector<float>& freqs) {
    // --- START: RoPE Debug Logging --- 
    bool should_log = (pos == 15); // Log only for position 15
    // --- END: RoPE Debug Logging --- 
    #pragma omp parallel for // --- RE-ENABLED ---
    for (int h = 0; h < num_heads; ++h) {
        for (int i = 0; i < head_dim; i += 2) {
            // --- START: RoPE Debug Logging --- 
            if (should_log && h < 2 && i < 4) { // Log only for first 2 heads, first 2 pairs
                float freq = freqs[i / 2];
                float angle = pos * freq;
                float x0_in = x[h * head_dim + i];
                float x1_in = x[h * head_dim + i + 1];
                std::stringstream ss_rope_dbg;
                ss_rope_dbg << "C++ RoPE Debug (P15 H" << h << " I" << i << "): "
                            << "x0_in=" << x0_in << " x1_in=" << x1_in 
                            << " freq=" << freq << " angle=" << angle;
                Logger::info(ss_rope_dbg.str());
            }
            // --- END: RoPE Debug Logging --- 
            if (i / 2 >= freqs.size()) { // Bounds check
                Logger::error("RoPE frequency index out of bounds!");
                continue;
            }
            float freq = freqs[i / 2]; // Use precomputed frequency
            float angle = pos * freq;
            float x0 = x[h * head_dim + i];
            float x1 = x[h * head_dim + i + 1];
            x[h * head_dim + i]     = x0 * std::cos(angle) - x1 * std::sin(angle);
            x[h * head_dim + i + 1] = x0 * std::sin(angle) + x1 * std::cos(angle);
        }
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

            // --- START: Verify o_proj loading for Layer 0 --- 
            if (i == 0) {
                 size_t expected_o_proj_size = (size_t)hs * hs;
                 Logger::info("C++ Layer 0 o_proj loaded size: " + std::to_string(lw.o_proj.size()) + " (Expected: " + std::to_string(expected_o_proj_size) + ")");
                 if (lw.o_proj.size() == expected_o_proj_size) {
                     std::stringstream oss_oproj;
                     oss_oproj << "C++ Layer 0 o_proj first 5 (uint16_t): ";
                     for (int j = 0; j < 5; ++j) oss_oproj << lw.o_proj[j] << " ";
                     Logger::info(oss_oproj.str());
                 } else {
                     Logger::error("C++ Layer 0 o_proj SIZE MISMATCH after loading!");
                 }
            }
            // --- END: Verify o_proj loading --- 

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

    // Precompute RoPE frequencies
    int head_dim = hs / n_heads;
    precomputed_freqs_.resize(head_dim / 2);
    float theta = config_.rope_theta;
    for (int i = 0; i < head_dim; i += 2) {
        precomputed_freqs_[i / 2] = std::pow(theta, -((float)i) / head_dim);
    }
    Logger::info("Precomputed RoPE frequencies.");
}

// --- ADDED: Embedding Lookup Method --- 
std::vector<float> TinyLlamaModel::lookup_embedding(int token_id) {
    int hs = config_.hidden_size;
    int vs = config_.vocab_size;
    if (token_id < 0 || token_id >= vs) {
        Logger::error("Token ID out of bounds in lookup_embedding: " + std::to_string(token_id));
        // Return an empty or zero vector, or throw an exception
        return std::vector<float>(hs, 0.0f); 
    }
    
    std::vector<float> embedding(hs);
    size_t offset = (size_t)token_id * hs;
    if (offset + hs > embed_tokens.size()) {
         Logger::error("Embedding offset out of bounds in lookup_embedding for token: " + std::to_string(token_id));
         return std::vector<float>(hs, 0.0f);
    }

    for (int i = 0; i < hs; ++i) {
        embedding[i] = bfloat16_to_float32(embed_tokens[offset + i]);
    }
    // Optional: Add logging here if needed
    // Logger::info("Performed embedding lookup for token: " + std::to_string(token_id));
    return embedding;
}
// --- END ADDED --- 

// Forward pass - Updated for KVCache store/retrieve
// --- CHANGE: Pass state vector x by reference, remove input_id --- 
std::vector<float> TinyLlamaModel::forward(std::vector<float>& x, int pos, KVCache* cache, ForwardDiagCallback diag_cb) {
    // Config dimensions (move to top for full function scope)
    int hs = config_.hidden_size;
    int is = config_.intermediate_size;
    int nhl = config_.num_hidden_layers;
    int vs = config_.vocab_size;
    int n_heads = config_.num_attention_heads;
    int n_kv_heads = config_.num_key_value_heads;
    int head_dim = hs / n_heads;
    int max_seq_len = config_.max_position_embeddings;
    float eps = config_.rms_norm_eps;

    // Check if cache is valid and initialized (basic check)
    if (cache && cache->layers.empty() && config_.num_hidden_layers > 0) {
        // This shouldn't happen if main initializes it, but as a safeguard:
        Logger::error("KVCache passed to forward but layers vector is empty!");
        // Handle error appropriately, maybe throw or return empty
        return {};
    }

    // --- REMOVED: Embedding lookup is now handled by the caller ---
    // std::vector<float> x(hs); 
    // for (int i = 0; i < hs; ++i) { 
    //    x[i] = bfloat16_to_float32(embed_tokens[input_id * hs + i]); 
    // } 
    // Logger::info("Embedding lookup complete (converted to float32)."); 
    // ... (removed logging related to embedding lookup inside forward) ...
    // log_vec_stats("embedding (float32)", x);
    // if (!std::all_of(x.begin(), x.end(), [](float v){ return std::isfinite(v); })) {
    //     Logger::error("NaN/Inf detected in embedding vector after lookup!");
    // }
    // if (diag_cb) diag_cb(-1, "embedding", x);
    // --- END REMOVED --- 

    // --- Add initial state logging for pos=1 --- 
    if (pos == 1) { // Log the state vector x *as received* for pos=1
        try {
            if (x.empty()) {
                Logger::error("C++ Input state x is empty before L0 P1 stats calculation!");
            } else {
                float min_val = *std::min_element(x.begin(), x.end());
                float max_val = *std::max_element(x.begin(), x.end());
                double sum = std::accumulate(x.begin(), x.end(), 0.0); // Use double for sum
                float mean_val = static_cast<float>(sum / x.size());

                Logger::info("--- C++ Forward Input State Debug (Pos 1) --- ");
                Logger::info("C++ Input x (Start of Forward, Pos 1) shape: [" + std::to_string(x.size()) + "]");
                Logger::info("C++ Input x (Start of Forward, Pos 1) mean: " + std::to_string(mean_val));
                Logger::info("C++ Input x (Start of Forward, Pos 1) min: " + std::to_string(min_val));
                Logger::info("C++ Input x (Start of Forward, Pos 1) max: " + std::to_string(max_val));
                 // Log first 5 elements
                 std::stringstream ss_x_start;
                 ss_x_start << "C++ Input x (Start of Forward, Pos 1) first 5: [";
                 for(int i=0; i < 5 && i < x.size(); ++i) ss_x_start << (i > 0 ? " " : "") << x[i];
                 ss_x_start << "]";
                 Logger::info(ss_x_start.str());
                Logger::info("--- End C++ Forward Input State Debug --- ");
            }
        } catch (const std::exception& e) {
            Logger::error("C++ Exception during input state stats calculation (Pos 1): " + std::string(e.what()));
        } catch (...) {
            Logger::error("C++ Unknown exception during input state stats calculation (Pos 1).");
        }
    }
    // --- END Add initial state logging --- 

    // 2. Transformer blocks (operates on the input vector 'x')
    for (int l = 0; l < nhl; ++l) {
        const auto& lw = layers[l];

        // --- DETAILED LOGGING INITIALISED (Layer 0, Pos 0) ---
        if (l == 0 && pos == 0) {
            Logger::info("=== DETAILED LOGGING INITIALISED (Layer 0, Pos 0) ===");
        }

        // --- START: Log x input to RMSNorm1 (L0 P0) --- 
        if (l == 0 && pos == 0) {
            std::stringstream ss_x_rms_inp;
            ss_x_rms_inp << "x vector (Input to RMSNorm1, L0 P0) first 5: [";
            for(int i=0; i < 5 && i < x.size(); ++i) ss_x_rms_inp << (i > 0 ? " " : "") << x[i];
            ss_x_rms_inp << "]";
            Logger::info(ss_x_rms_inp.str());
        }
        // --- END: Log x --- 

        // a) Input RMSNorm (remains same)
        std::vector<float> x_norm(hs);
        rmsnorm(x, lw.input_layernorm, eps, x_norm);
        if (l == 0 && pos == 0) {
            std::stringstream ss_x_norm_raw;
            ss_x_norm_raw << "Layer 0 RMSNorm1 Output (Input to Proj, Pos 0) first 5: [";
            for(int i=0; i < 5 && i < x_norm.size(); ++i) ss_x_norm_raw << (i > 0 ? " " : "") << x_norm[i];
            ss_x_norm_raw << "]";
            Logger::info(ss_x_norm_raw.str());
        }
        // --- START: Log x_norm input to Q proj (L0 P0) --- 
        if (l == 0 && pos == 0) {
            std::stringstream ss_x_norm_q_inp;
            ss_x_norm_q_inp << "x_norm vector (Input to Q proj, L0 P0) first 5: [";
            for(int i=0; i < 5 && i < x_norm.size(); ++i) ss_x_norm_q_inp << (i > 0 ? " " : "") << x_norm[i];
            ss_x_norm_q_inp << "]";
            Logger::info(ss_x_norm_q_inp.str());
        }
        // --- END: Log x_norm --- 
        if (!std::all_of(x_norm.begin(), x_norm.end(), [](float v){ return std::isfinite(v); })) {
            Logger::error("NaN/Inf detected in x_norm after input RMSNorm, layer " + std::to_string(l));
        }

        // b) Attention Q, K, V projections
        std::vector<float> q(hs), k_current(head_dim * n_kv_heads), v_current(head_dim * n_kv_heads);
        matvec_bf16_f32(lw.q_proj, x_norm, q, hs, hs);
        matvec_bf16_f32(lw.k_proj, x_norm, k_current, head_dim * n_kv_heads, hs);
        matvec_bf16_f32(lw.v_proj, x_norm, v_current, head_dim * n_kv_heads, hs);
        // Log Q/K raw projection outputs before RoPE for Layer 0, Pos 0
        if (l == 0 && pos == 0) {
            std::stringstream ss_q_proj_raw;
            ss_q_proj_raw << "Layer 0 Q Projection Output (Pre-RoPE, Pos 0) first 5: [";
            for(int i=0; i < 5 && i < q.size(); ++i) ss_q_proj_raw << (i > 0 ? " " : "") << q[i];
            ss_q_proj_raw << "]";
            Logger::info(ss_q_proj_raw.str());

            std::stringstream ss_k_proj_raw;
            ss_k_proj_raw << "Layer 0 K Projection Output (Pre-RoPE, Pos 0) first 5: [";
            for(int i=0; i < 5 && i < k_current.size(); ++i) ss_k_proj_raw << (i > 0 ? " " : "") << k_current[i];
            ss_k_proj_raw << "]";
            Logger::info(ss_k_proj_raw.str());
        }
        // Log Q/K/V projection stats for Layer 0, Pos 0
        if (l == 0 && pos == 0) {
            log_vec_stats("Layer 0 Q projection (Pos 0)", q);
            log_vec_stats("Layer 0 K projection (Pos 0)", k_current);
            log_vec_stats("Layer 0 V projection (Pos 0)", v_current);
        }
        // RoPE (operates on float Q/K copies) - uses precomputed freqs
        std::vector<float> q_heads(q); // Full Q projection

        // --- START: Log q vector and q_heads copy before apply_rope (L0, P0) ---
        if (l == 0 && pos == 0) {
             std::stringstream ss_q_pre_copy;
             ss_q_pre_copy << "q vector (Before q_heads copy, L0 P0) first 5: [";
             for(int i=0; i < 5 && i < q.size(); ++i) ss_q_pre_copy << (i > 0 ? " " : "") << q[i];
             ss_q_pre_copy << "]";
             Logger::info(ss_q_pre_copy.str());
        }
        // --- END: Log q --- 

        // --- START: Log q_heads after copy (L0, P0) --- 
        if (l == 0 && pos == 0) {
             std::stringstream ss_q_heads_pre_rope;
             ss_q_heads_pre_rope << "q_heads vector (Before apply_rope, L0 P0) first 5: [";
             for(int i=0; i < 5 && i < q_heads.size(); ++i) ss_q_heads_pre_rope << (i > 0 ? " " : "") << q_heads[i];
             ss_q_heads_pre_rope << "]";
             Logger::info(ss_q_heads_pre_rope.str());
        }
        // --- END: Log q_heads --- 

        apply_rope(q_heads, n_heads, head_dim, pos, precomputed_freqs_); // Pass precomputed freqs
        std::vector<float> k_rope_current = k_current;
        apply_rope(k_rope_current, n_kv_heads, head_dim, pos, precomputed_freqs_); // Pass precomputed freqs

        // Log Q/K after RoPE for pos=0 and pos=15 (first generated token)
        if (l == 0) {
            if (pos == 0) {
                 // Log Q after RoPE for pos=0
                 std::stringstream ss_q_rope0;
                 ss_q_rope0 << "Layer 0 Q after RoPE (Pos 0) shape: [" + std::to_string(q_heads.size()) + "] first 5: " + std::to_string(q_heads[0]) + " " + std::to_string(q_heads[1]) + " " + std::to_string(q_heads[2]) + " " + std::to_string(q_heads[3]) + " " + std::to_string(q_heads[4]);
                 Logger::info(ss_q_rope0.str());
                 // Log K after RoPE for pos=0
                 std::stringstream ss_k_rope0;
                 ss_k_rope0 << "Layer 0 K after RoPE (Pos 0) shape: [" + std::to_string(k_rope_current.size()) + "] first 5: " + std::to_string(k_rope_current[0]) + " " + std::to_string(k_rope_current[1]) + " " + std::to_string(k_rope_current[2]) + " " + std::to_string(k_rope_current[3]) + " " + std::to_string(k_rope_current[4]);
                 Logger::info(ss_k_rope0.str());
            } else if (pos == 15) { // Log for the first generated token (now potentially loaded from file)
                 // Log Q after RoPE for pos=15
                 std::stringstream ss_q_rope15;
                 ss_q_rope15 << "Layer 0 Q after RoPE (Pos 15 - Loaded) shape: [" + std::to_string(q_heads.size()) + "] first 5: " + std::to_string(q_heads[0]) + " " + std::to_string(q_heads[1]) + " " + std::to_string(q_heads[2]) + " " + std::to_string(q_heads[3]) + " " + std::to_string(q_heads[4]);
                 Logger::info(ss_q_rope15.str());
                 // Log K after RoPE for pos=15
                 std::stringstream ss_k_rope15;
                 ss_k_rope15 << "Layer 0 K after RoPE (Pos 15 - Loaded) shape: [" + std::to_string(k_rope_current.size()) + "] first 5: " + std::to_string(k_rope_current[0]) + " " + std::to_string(k_rope_current[1]) + " " + std::to_string(k_rope_current[2]) + " " + std::to_string(k_rope_current[3]) + " " + std::to_string(k_rope_current[4]);
                 Logger::info(ss_k_rope15.str());
            }
        }
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

        // --- START: Verify Cache Store/Retrieve for L0, Pos 15 --- 
        if (cache && l == 0 && pos == 15) {
             size_t cache_offset_verify = (size_t)pos * head_dim; // Base offset for this position
             size_t layer_stride_verify = (size_t)max_seq_len * head_dim;
             size_t head0_offset_verify = 0 * layer_stride_verify; // Offset for head 0 within the layer
             size_t final_k_offset = head0_offset_verify + cache_offset_verify;
             size_t final_v_offset = head0_offset_verify + cache_offset_verify; // V cache has same structure

             if (final_k_offset + 5 <= cache->layers[l].k.size() && final_v_offset + 5 <= cache->layers[l].v.size()) {
                 std::stringstream ss_k_cache, ss_v_cache, ss_k_src, ss_v_src;
                 
                 // Log K retrieved from cache
                 ss_k_cache << "L0 P15 K Cache Readback (H0) first 5: [";
                 for (int i = 0; i < 5; ++i) ss_k_cache << (i > 0 ? " " : "") << cache->layers[l].k[final_k_offset + i];
                 ss_k_cache << "]";
                 Logger::info(ss_k_cache.str());

                 // Log V retrieved from cache
                 ss_v_cache << "L0 P15 V Cache Readback (H0) first 5: [";
                 for (int i = 0; i < 5; ++i) ss_v_cache << (i > 0 ? " " : "") << cache->layers[l].v[final_v_offset + i];
                 ss_v_cache << "]";
                 Logger::info(ss_v_cache.str());

                 // Log original K source (head 0 slice)
                 size_t k_src_offset = 0 * head_dim; // Head 0 offset in k_rope_current
                 ss_k_src << "L0 P15 K Source (H0) first 5: [";
                 for (int i = 0; i < 5; ++i) ss_k_src << (i > 0 ? " " : "") << k_rope_current[k_src_offset + i];
                 ss_k_src << "]";
                 Logger::info(ss_k_src.str());

                 // Log original V source (head 0 slice)
                 size_t v_src_offset = 0 * head_dim; // Head 0 offset in v_current
                 ss_v_src << "L0 P15 V Source (H0) first 5: [";
                 for (int i = 0; i < 5; ++i) ss_v_src << (i > 0 ? " " : "") << v_current[v_src_offset + i];
                 ss_v_src << "]";
                 Logger::info(ss_v_src.str());
             } else {
                 Logger::error("Cache verification indices out of bounds!");
             }
        }
        // --- END: Verify Cache Store/Retrieve --- 

        // Attention Calculation using Cache
        int kv_dim = head_dim * n_kv_heads;
        std::vector<float> attn_out(hs, 0.0f);

        // --- START: Log q_heads before parallel loop --- 
        if (l == 0 && pos == 15) {
             std::stringstream ss_q_heads_verify;
             ss_q_heads_verify << "C++ q_heads vector (L0 P15, before parallel loop) first 5: [";
             for(int i=0; i<5 && i<q_heads.size(); ++i) ss_q_heads_verify << (i>0?" ":"") << q_heads[i];
             ss_q_heads_verify << "]";
             Logger::info(ss_q_heads_verify.str());
        }
        // --- END: Log q_heads --- 

        // --- Temporarily REMOVE OpenMP --- 
        // #pragma omp parallel for 
            for (int h = 0; h < n_heads; ++h) {
            if (h >= n_kv_heads) {
                // For heads beyond n_kv_heads, set output to zero (already zero-initialized)
                continue;
            }
            int kvh = h;
            float* current_q_head = q_heads.data() + h * head_dim;
            float* attn_out_head = attn_out.data() + h * head_dim;
            std::fill(attn_out_head, attn_out_head + head_dim, 0.0f); // Initialize output for the head
            
            // --- START: Force Load K/V Cache for L0, H0, Pos 15 --- 
            std::vector<float> k_cache_ref; // Holds loaded K cache
            std::vector<float> v_cache_ref; // Holds loaded V cache
            bool use_ref_cache = false;
            if (l == 0 && h == 0 && pos == 15) {
                std::string k_ref_filename = "k_cache_layer0_head0_0to15_ref.bin";
                std::string v_ref_filename = "v_cache_layer0_head0_0to15_ref.bin";
                size_t expected_cache_elements = (pos + 1) * head_dim;
                size_t expected_cache_bytes = expected_cache_elements * sizeof(float);

                std::ifstream k_ref_file(k_ref_filename, std::ios::binary);
                std::ifstream v_ref_file(v_ref_filename, std::ios::binary);

                if (k_ref_file && v_ref_file) {
                    k_cache_ref.resize(expected_cache_elements);
                    v_cache_ref.resize(expected_cache_elements);

                    k_ref_file.read(reinterpret_cast<char*>(k_cache_ref.data()), expected_cache_bytes);
                    v_ref_file.read(reinterpret_cast<char*>(v_cache_ref.data()), expected_cache_bytes);

                    if (k_ref_file && v_ref_file) {
                        Logger::info("Successfully loaded reference K and V caches for L0, H0, Pos 0-15.");
                        use_ref_cache = true;
                    } else {
                        Logger::error("FAILED to read full data from reference K/V cache files.");
                    }
                } else {
                    Logger::error("FAILED TO OPEN reference K/V cache files: " + k_ref_filename + " or " + v_ref_filename);
                }
            }
            // --- END: Force Load K/V Cache --- 

            // --- START: Add Attention Calculation Logic --- 
            // Vector to store attention scores for this head for all positions up to pos
            std::vector<float> attn_scores(pos + 1);

            // Calculate attention scores (Q @ K^T / sqrt(head_dim))
            size_t layer_stride_attn = (size_t)max_seq_len * head_dim;
            size_t head_offset_attn = (size_t)kvh * layer_stride_attn;
            for (int t = 0; t <= pos; ++t) {
                float* cached_k;
                size_t cache_offset_attn = (size_t)t * head_dim;
                cached_k = &cache->layers[l].k[head_offset_attn + cache_offset_attn];
                // Log Q/K for Layer 0, Pos 0, Head 0, t=0
                if (l == 0 && h == 0 && pos == 0 && t == 0) {
                    std::stringstream ss_q, ss_k;
                    ss_q << "[C++] L0 H0 P0 Q: ";
                    for(int i=0; i<5 && i<head_dim; ++i) ss_q << (i>0?" ":"") << current_q_head[i];
                    Logger::info(ss_q.str());
                    ss_k << "[C++] L0 H0 P0 K: ";
                    for(int i=0; i<5 && i<head_dim; ++i) ss_k << (i>0?" ":"") << cached_k[i];
                    Logger::info(ss_k.str());
                }
                double score = 0.0;
                for (int i = 0; i < head_dim; ++i) {
                    score += (double)current_q_head[i] * (double)cached_k[i];
                }
                attn_scores[t] = (float)(score / std::sqrt((double)head_dim));
            }
            // Log attention scores before softmax
            if (l == 0 && h == 0 && pos == 0) {
                std::stringstream ss;
                ss << "[C++] L0 H0 P0 attn_scores (before softmax): ";
                for(int i=0; i<attn_scores.size(); ++i) ss << (i>0?" ":"") << attn_scores[i];
                Logger::info(ss.str());
            }
            softmax(attn_scores); // Operates in-place
            // Log attention scores after softmax
            if (l == 0 && h == 0 && pos == 0) {
                std::stringstream ss;
                ss << "[C++] L0 H0 P0 attn_probs (after softmax): ";
                for(int i=0; i<attn_scores.size(); ++i) ss << (i>0?" ":"") << attn_scores[i];
                Logger::info(ss.str());
            }
            // Weighted sum of values (Scores @ V)
            for (int t = 0; t <= pos; ++t) {
                float* cached_v;
                size_t cache_offset_attn = (size_t)t * head_dim;
                cached_v = &cache->layers[l].v[head_offset_attn + cache_offset_attn];
                float weight = attn_scores[t];
                // Log V and weight for Layer 0, Pos 0, Head 0, t=0
                if (l == 0 && h == 0 && pos == 0 && t == 0) {
                    std::stringstream ss_v, ss_w;
                    ss_v << "[C++] L0 H0 P0 V: ";
                    for(int i=0; i<5 && i<head_dim; ++i) ss_v << (i>0?" ":"") << cached_v[i];
                    Logger::info(ss_v.str());
                    ss_w << "[C++] L0 H0 P0 attn_weight: " << weight;
                    Logger::info(ss_w.str());
                }
                for (int i = 0; i < head_dim; ++i) {
                    attn_out_head[i] += weight * cached_v[i];
                }
            }
            // Log attention output for Layer 0, Pos 0, Head 0
            if (l == 0 && h == 0 && pos == 0) {
                std::stringstream ss_out;
                ss_out << "[C++] L0 H0 P0 attn_out_head: ";
                for(int i=0; i<5 && i<head_dim; ++i) ss_out << (i>0?" ":"") << attn_out_head[i];
                Logger::info(ss_out.str());
            }
            // --- END: Add Attention Calculation Logic --- 

            // Log first 5 elements of attn_out_head (potentially loaded) for Layer 0, Head 0
            if (l == 0 && h == 0) {
                if (pos == 0) {
                    std::stringstream ss_attn_out_head;
                    ss_attn_out_head << "Layer 0 Attention Out (Head 0, Pos 0, Before o_proj) first 5: [";
                    for(int i = 0; i < 5 && i < head_dim; ++i) ss_attn_out_head << (i > 0 ? " " : "") << attn_out_head[i];
                    ss_attn_out_head << "]";
                    Logger::info(ss_attn_out_head.str());
                 } else if (pos == 15) {
                     std::stringstream ss_attn_out_head_15;
                     ss_attn_out_head_15 << "Layer 0 Attention Out (Head 0, Pos 15, Before o_proj) first 5: [";
                     for(int i = 0; i < 5 && i < head_dim; ++i) ss_attn_out_head_15 << (i > 0 ? " " : "") << attn_out_head[i];
                     ss_attn_out_head_15 << "]";
                     Logger::info(ss_attn_out_head_15.str());
                 }
            }
        }

        // Log first 5 elements of C++ calculated attn_out_head for Layer 0, Head 0 at pos=15
        // Note: This logging happens *after* the parallel loop completes, reflecting the final C++ calculated state.
        if (l == 0 && pos == 15) {
            std::stringstream ss_cpp_attn_out_head_15;
            ss_cpp_attn_out_head_15 << "C++ Layer 0 Attention Out (Head 0, Pos 15, Before o_proj) first 5: [";
            float* head_ptr_log = attn_out.data(); // attn_out_head is potentially out of scope
            for(int i = 0; i < 5 && i < head_dim; ++i) ss_cpp_attn_out_head_15 << (i > 0 ? " " : "") << head_ptr_log[i];
            ss_cpp_attn_out_head_15 << "]";
            Logger::info(ss_cpp_attn_out_head_15.str());
        }

        // Log attention output stats
        if (l == 0) {
            // --- START: Log full C++ attn_out vector at pos=15 --- 
            if (pos == 15) {
                 std::stringstream ss_full_attn_out_15;
                 ss_full_attn_out_15 << "C++ Layer 0 Full attn_out (Pos 15, Before o_proj) first 5: [";
                 for(int i = 0; i < 5 && i < attn_out.size(); ++i) ss_full_attn_out_15 << (i > 0 ? " " : "") << attn_out[i];
                 ss_full_attn_out_15 << "]";
                 Logger::info(ss_full_attn_out_15.str());
            }
            // --- END: Log full C++ attn_out vector --- 
            log_vec_stats("Layer " + std::to_string(l) + " attn_out (Before o_proj)", attn_out);
        }

        // Output projection
        std::vector<float> attn_proj(hs);
        matvec_bf16_f32(lw.o_proj, attn_out, attn_proj, hs, hs);
        // Log projected attention output stats
        if (l == 0) {
             // Log stats regardless of position for layer 0
             log_vec_stats("Layer " + std::to_string(l) + " attn_out (Projected, Pos " + std::to_string(pos) + ")", attn_proj);
             // Log first 5 values for specific positions
            if (pos == 0) {
                 std::stringstream ss_proj0;
                 ss_proj0 << "Layer 0 attn_out (Projected, Pos 0) first 5: [";
                 for(int i=0; i < 5 && i < attn_proj.size(); ++i) ss_proj0 << (i > 0 ? " " : "") << attn_proj[i];
                 ss_proj0 << "]";
                 Logger::info(ss_proj0.str());
            } else if (pos == 15) {
                 std::stringstream ss_proj15;
                 ss_proj15 << "Layer 0 attn_out (Projected, Pos 15) first 5: [";
                 for(int i=0; i < 5 && i < attn_proj.size(); ++i) ss_proj15 << (i > 0 ? " " : "") << attn_proj[i];
                 ss_proj15 << "]";
                 Logger::info(ss_proj15.str());
            }
        }
        // Residual addition after MLP
        for (int i = 0; i < hs; ++i) x[i] += attn_proj[i];
        log_vec_stats("Layer " + std::to_string(l) + " post-MLP residual", x);
        
        // --- DETAILED LOGGING INITIALISED (Layer 0, Pos 1) ---
        if (l == 0 && pos == 1) {
            Logger::info("=== DETAILED LOGGING INITIALISED (Layer 0, Pos 1) ===");
        }

        // --- START: Log x input to RMSNorm1 (L0 P1) --- 
        if (l == 0 && pos == 1) {
            std::stringstream ss_x_rms_inp;
            ss_x_rms_inp << "x vector (Input to RMSNorm1, L0 P1) first 5: [";
            for(int i=0; i < 5 && i < x.size(); ++i) ss_x_rms_inp << (i > 0 ? " " : "") << x[i];
            ss_x_rms_inp << "]";
            Logger::info(ss_x_rms_inp.str());
        }
        // --- END: Log x --- 

        // a) Input RMSNorm (remains same)
        // x_norm is already declared above
        rmsnorm(x, lw.input_layernorm, eps, x_norm);
        if (l == 0 && pos == 1) {
            std::stringstream ss_x_norm_raw;
            ss_x_norm_raw << "Layer 0 RMSNorm1 Output (Input to Proj, Pos 1) first 5: [";
            for(int i=0; i < 5 && i < x_norm.size(); ++i) ss_x_norm_raw << (i > 0 ? " " : "") << x_norm[i];
            ss_x_norm_raw << "]";
            Logger::info(ss_x_norm_raw.str());
        }
        // --- START: Log x_norm input to Q proj (L0 P1) --- 
        if (l == 0 && pos == 1) {
            std::stringstream ss_x_norm_q_inp;
            ss_x_norm_q_inp << "x_norm vector (Input to Q proj, L0 P1) first 5: [";
            for(int i=0; i < 5 && i < x_norm.size(); ++i) ss_x_norm_q_inp << (i > 0 ? " " : "") << x_norm[i];
            ss_x_norm_q_inp << "]";
            Logger::info(ss_x_norm_q_inp.str());
        }
        // --- END: Log x_norm --- 

        // b) Attention Q, K, V projections
        // q, k_current, v_current are already declared above
        matvec_bf16_f32(lw.q_proj, x_norm, q, hs, hs);
        matvec_bf16_f32(lw.k_proj, x_norm, k_current, head_dim * n_kv_heads, hs);
        matvec_bf16_f32(lw.v_proj, x_norm, v_current, head_dim * n_kv_heads, hs);
        // Log Q/K raw projection outputs before RoPE for Layer 0, Pos 1
        if (l == 0 && pos == 1) {
            std::stringstream ss_q_proj_raw;
            ss_q_proj_raw << "Layer 0 Q Projection Output (Pre-RoPE, Pos 1) first 5: [";
            for(int i=0; i < 5 && i < q.size(); ++i) ss_q_proj_raw << (i > 0 ? " " : "") << q[i];
            ss_q_proj_raw << "]";
            Logger::info(ss_q_proj_raw.str());

            std::stringstream ss_k_proj_raw;
            ss_k_proj_raw << "Layer 0 K Projection Output (Pre-RoPE, Pos 1) first 5: [";
            for(int i=0; i < 5 && i < k_current.size(); ++i) ss_k_proj_raw << (i > 0 ? " " : "") << k_current[i];
            ss_k_proj_raw << "]";
            Logger::info(ss_k_proj_raw.str());
        }
        // Log Q/K/V projection stats for Layer 0, Pos 1
        if (l == 0 && pos == 1) {
            log_vec_stats("Layer 0 Q projection (Pos 1)", q);
            log_vec_stats("Layer 0 K projection (Pos 1)", k_current);
            log_vec_stats("Layer 0 V projection (Pos 1)", v_current);
        }
        // RoPE (operates on float Q/K copies) - uses precomputed freqs
        // q_heads is already declared above
        q_heads = q; // assign q to q_heads

        // --- START: Log q vector and q_heads copy before apply_rope (L0, P1) ---
        if (l == 0 && pos == 1) {
             std::stringstream ss_q_pre_copy;
             ss_q_pre_copy << "q vector (Before q_heads copy, L0 P1) first 5: [";
             for(int i=0; i < 5 && i < q.size(); ++i) ss_q_pre_copy << (i > 0 ? " " : "") << q[i];
             ss_q_pre_copy << "]";
             Logger::info(ss_q_pre_copy.str());
        }
        // --- END: Log q --- 

        // --- START: Log q_heads after copy (L0, P1) --- 
        if (l == 0 && pos == 1) {
             std::stringstream ss_q_heads_pre_rope;
             ss_q_heads_pre_rope << "q_heads vector (Before apply_rope, L0 P1) first 5: [";
             for(int i=0; i < 5 && i < q_heads.size(); ++i) ss_q_heads_pre_rope << (i > 0 ? " " : "") << q_heads[i];
             ss_q_heads_pre_rope << "]";
             Logger::info(ss_q_heads_pre_rope.str());
        }
        // --- END: Log q_heads --- 

        apply_rope(q_heads, n_heads, head_dim, pos, precomputed_freqs_); // Pass precomputed freqs
        // k_rope_current is already declared above
        k_rope_current = k_current;
        apply_rope(k_rope_current, n_kv_heads, head_dim, pos, precomputed_freqs_); // Pass precomputed freqs

        // Log Q/K after RoPE for pos=1
        if (l == 0 && pos == 1) {
            std::stringstream ss_q_rope1;
            ss_q_rope1 << "Layer 0 Q after RoPE (Pos 1) shape: [" + std::to_string(q_heads.size()) + "] first 5: " + std::to_string(q_heads[0]) + " " + std::to_string(q_heads[1]) + " " + std::to_string(q_heads[2]) + " " + std::to_string(q_heads[3]) + " " + std::to_string(q_heads[4]);
            Logger::info(ss_q_rope1.str());
            std::stringstream ss_k_rope1;
            ss_k_rope1 << "Layer 0 K after RoPE (Pos 1) shape: [" + std::to_string(k_rope_current.size()) + "] first 5: " + std::to_string(k_rope_current[0]) + " " + std::to_string(k_rope_current[1]) + " " + std::to_string(k_rope_current[2]) + " " + std::to_string(k_rope_current[3]) + " " + std::to_string(k_rope_current[4]);
            Logger::info(ss_k_rope1.str());
        }
        // --- DETAILED LOGGING ENDED (Layer 0, Pos 1) ---
        if (l == 0 && pos == 1) {
            Logger::info("=== DETAILED LOGGING ENDED (Layer 0, Pos 1) ===");
        }
    } // End of layer loop

    // --- START: Log Final Norm Input Stats (pos=13) --- 
    if (pos == 13) {
        try {
            if (x.empty()) {
                 Logger::error("C++ vector x is empty before final norm input stats!");
            } else {
                float min_val = *std::min_element(x.begin(), x.end());
                float max_val = *std::max_element(x.begin(), x.end());
                double sum = std::accumulate(x.begin(), x.end(), 0.0);
                float mean_val = static_cast<float>(sum / x.size());
                Logger::info("--- C++ Final Norm Input Debug (Pos 13) ---");
                Logger::info("C++ x (Input to Final Norm, Pos 13) shape: [" + std::to_string(x.size()) + "]");
                Logger::info("C++ x (Input to Final Norm, Pos 13) mean: " + std::to_string(mean_val));
                Logger::info("C++ x (Input to Final Norm, Pos 13) min: " + std::to_string(min_val));
                Logger::info("C++ x (Input to Final Norm, Pos 13) max: " + std::to_string(max_val));
                Logger::info("--- End C++ Final Norm Input Debug ---");
            }
        } catch (const std::exception& e) {
             Logger::error("C++ Exception during final norm input stats calc: " + std::string(e.what()));
        } catch (...) {
             Logger::error("C++ Unknown exception during final norm input stats calc.");
        }
    }
    // --- END: Log Final Norm Input Stats --- 

    // Final RMSNorm
    std::vector<float> x_norm(hs);
    rmsnorm(x, final_norm, eps, x_norm);
    if (!std::all_of(x_norm.begin(), x_norm.end(), [](float v){ return std::isfinite(v); })) {
        Logger::error("NaN/Inf detected in final RMSNorm output");
    }
    // --- START: Log Final Norm Output Stats (pos=13) --- 
    if (pos == 13) {
        try {
            if (x_norm.empty()) {
                 Logger::error("C++ vector x_norm is empty before final norm output stats!");
            } else {
                float min_val = *std::min_element(x_norm.begin(), x_norm.end());
                float max_val = *std::max_element(x_norm.begin(), x_norm.end());
                double sum = std::accumulate(x_norm.begin(), x_norm.end(), 0.0);
                float mean_val = static_cast<float>(sum / x_norm.size());
                Logger::info("--- C++ Final Norm Output Debug (Pos 13) ---");
                Logger::info("C++ x_norm (Output of Final Norm, Pos 13) shape: [" + std::to_string(x_norm.size()) + "]");
                Logger::info("C++ x_norm (Output of Final Norm, Pos 13) mean: " + std::to_string(mean_val));
                Logger::info("C++ x_norm (Output of Final Norm, Pos 13) min: " + std::to_string(min_val));
                Logger::info("C++ x_norm (Output of Final Norm, Pos 13) max: " + std::to_string(max_val));
                Logger::info("--- End C++ Final Norm Output Debug ---");
            }
        } catch (const std::exception& e) {
             Logger::error("C++ Exception during final norm output stats calc: " + std::string(e.what()));
        } catch (...) {
             Logger::error("C++ Unknown exception during final norm output stats calc.");
        }
    }
    // --- END: Log Final Norm Output Stats --- 

    // LM head projection
    std::vector<float> logits(vs);
    matvec_bf16_f32(lm_head, x_norm, logits, vs, hs);
    if (!std::all_of(logits.begin(), logits.end(), [](float v){ return std::isfinite(v); })) {
        Logger::error("NaN/Inf detected in logits before return");
    }
    // --- START: Log Final Logits Stats (pos=13) ---
    // Note: Existing log_vec_stats already provides this, but we add explicit checks/labels for clarity 
    if (pos == 13) {
        try {
            if (logits.empty()) {
                 Logger::error("C++ vector logits is empty before final logits stats!");
            } else {
                float min_val = *std::min_element(logits.begin(), logits.end());
                float max_val = *std::max_element(logits.begin(), logits.end());
                double sum = std::accumulate(logits.begin(), logits.end(), 0.0);
                float mean_val = static_cast<float>(sum / logits.size());
                Logger::info("--- C++ Final Logits Debug (Pos 13) ---");
                Logger::info("C++ Logits (Pos 13) shape: [" + std::to_string(logits.size()) + "]");
                Logger::info("C++ Logits (Pos 13) mean: " + std::to_string(mean_val));
                Logger::info("C++ Logits (Pos 13) min: " + std::to_string(min_val));
                Logger::info("C++ Logits (Pos 13) max: " + std::to_string(max_val));
                Logger::info("--- End C++ Final Logits Debug ---");
            }
        } catch (const std::exception& e) {
             Logger::error("C++ Exception during final logits stats calc: " + std::string(e.what()));
        } catch (...) {
             Logger::error("C++ Unknown exception during final logits stats calc.");
        }
    }
    // --- END: Log Final Logits Stats --- 
    // Log final logits stats before return
    log_vec_stats("final logits", logits);
    Logger::info("Forward pass complete.");
    return logits;
} 