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
static float bfloat16_to_float32(uint16_t b16) {
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

// RMSNorm: y = x / sqrt(mean(x^2) + eps) * weight
// Accepts float input/output, uint16_t (bfloat16) weights
static void rmsnorm(const std::vector<float>& x, const std::vector<uint16_t>& weight, float eps, std::vector<float>& out) {
    size_t N_full = x.size();
    if (N_full == 0) {
        Logger::error("RMSNorm Error: Input vector x is empty.");
        return;
    }
    double ss = 0.0;
    for (size_t i = 0; i < N_full; ++i) {
        ss += double(x[i]) * double(x[i]);
    }
    double mean_sq = ss / double(N_full);
    double denom = 1.0 / std::sqrt(mean_sq + double(eps));
    // Apply norm and weight (converting weight to float on the fly)
    for (size_t i = 0; i < N_full; ++i) {
        out[i] = float(double(x[i]) * denom) * bfloat16_to_float32(weight[i]);
    }
    // Logging (optional, kept for consistency, logs converted weight)
    // ... (existing logging using bfloat16_to_float32(weight[i]) if needed) ...
}

// SiLU activation: y = x * sigmoid(x)
static void silu(const std::vector<float>& x, std::vector<float>& out) {
    for (size_t i = 0; i < x.size(); ++i) {
        out[i] = x[i] / (1.0f + std::exp(-x[i]));
    }
}

// RoPE: apply rotary positional embedding to Q or K (in-place)
static void apply_rope(std::vector<float>& x, int num_heads, int head_dim, int pos, float rope_theta) {
    // x: [num_heads, head_dim]
    // Only even/odd pairs are rotated
    float theta = rope_theta;
    for (int h = 0; h < num_heads; ++h) {
        for (int i = 0; i < head_dim; i += 2) {
            float freq = std::pow(theta, -float(i) / head_dim);
            float angle = pos * freq;
            float x0 = x[h * head_dim + i];
            float x1 = x[h * head_dim + i + 1];
            x[h * head_dim + i]     = x0 * std::cos(angle) - x1 * std::sin(angle);
            x[h * head_dim + i + 1] = x0 * std::sin(angle) + x1 * std::cos(angle);
        }
    }
}

// Softmax over a vector (in-place)
static void softmax(std::vector<float>& x) {
    float maxv = *std::max_element(x.begin(), x.end());
    float sum = 0.0f;
    for (float& v : x) { v = std::exp(v - maxv); sum += v; }
    for (float& v : x) v /= sum;
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
static void matvec_bf16_f32(const std::vector<uint16_t>& mat, const std::vector<float>& vec, std::vector<float>& out, int M, int N) {
    if (mat.size() != M * N) {
        throw std::runtime_error("Matrix size mismatch in matvec_bf16_f32");
    }
    if (vec.size() != N) {
        throw std::runtime_error("Vector size mismatch in matvec_bf16_f32");
    }
    if (out.size() != M) {
         out.resize(M); // Ensure output vector has correct size
    }

    #pragma omp parallel for
    for (int i = 0; i < M; ++i) {
        float sum = 0.0f; // Accumulate in float32
        for (int j = 0; j < N; ++j) {
            float mat_val = bfloat16_to_float32(mat[i * N + j]);
            sum += mat_val * vec[j];
        }
        out[i] = sum;
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

// Forward pass - Updated for bfloat16 weight handling
std::vector<float> TinyLlamaModel::forward(int input_id, int token_idx, KVCache* cache, ForwardDiagCallback diag_cb) {
    int hs = config_.hidden_size;
    int is = config_.intermediate_size;
    int nhl = config_.num_hidden_layers;
    int vs = config_.vocab_size;
    int n_heads = config_.num_attention_heads;
    int n_kv_heads = config_.num_key_value_heads;
    int head_dim = hs / n_heads;
    int kv_dim = head_dim * n_kv_heads;
    float eps = config_.rms_norm_eps;
    float rope_theta = config_.rope_theta;
    int pos = cache ? cache->seq_len : 0;

    // --- File stream setup (remains the same) ---
    std::ofstream cpp_log_stream;
    const std::string cpp_log_filename = "layer_0_cpp_outputs.log";
    bool opened_log_stream = false;
    if (token_idx == 0) { 
        cpp_log_stream.open(cpp_log_filename, std::ios::trunc);
        if (!cpp_log_stream) {
            Logger::error("Failed to open C++ log file: " + cpp_log_filename);
        } else {
            opened_log_stream = true;
            Logger::info("Opened " + cpp_log_filename + " for Layer 0 logging (truncating).");
        }
    }
    // Helper lambda remains the same (logs float vectors)
    auto log_to_file = [&](const std::string& key, const std::vector<float>& vec) {
         if (opened_log_stream) { /* ... */ }
    };

    // 1. Embedding lookup (Convert bfloat16 weights to float activations)
    std::vector<float> x(hs);
    for (int i = 0; i < hs; ++i) {
        x[i] = bfloat16_to_float32(embed_tokens[input_id * hs + i]);
    }
    Logger::info("Embedding lookup complete (converted to float32).");
    log_vec_stats("embedding (float32)", x);
    if (diag_cb) diag_cb(-1, "embedding", x);
    if (input_id == 1 && opened_log_stream) { /* ... log x ... */ }

    // 2. Transformer blocks
    for (int l = 0; l < nhl; ++l) {
        const auto& lw = layers[l]; // lw now contains uint16_t vectors

        if (l == 0) log_to_file("x (Layer Input)", x);

        // a) RMSNorm (weights are uint16_t, input/output are float)
        std::vector<float> x_norm(hs);
        rmsnorm(x, lw.input_layernorm, eps, x_norm);
        if (l == 0) log_to_file("x_norm (Input RMSNorm Out)", x_norm);

        // b) Attention
        std::vector<float> q(hs), k(kv_dim), v(kv_dim); // Activations remain float
        // Use new matvec for bfloat16 weights
        matvec_bf16_f32(lw.q_proj, x_norm, q, hs, hs);
        matvec_bf16_f32(lw.k_proj, x_norm, k, kv_dim, hs);
        matvec_bf16_f32(lw.v_proj, x_norm, v, kv_dim, hs);
        if (l == 0) { log_to_file("Q", q); log_to_file("K", k); log_to_file("V", v); }

        // RoPE (operates on float Q/K copies)
        std::vector<float> q_heads(q);
        std::vector<float> k_heads(k);
        apply_rope(q_heads, n_heads, head_dim, pos, rope_theta);
        apply_rope(k_heads, n_kv_heads, head_dim, pos, rope_theta);
        if (l == 0) { log_to_file("Q_rope", q_heads); log_to_file("K_rope", k_heads); }

        // Attention Score/Output (operates on floats)
        std::vector<float> attn_out(hs, 0.0f);
        if (cache) { /* ... existing cache/attention logic (uses float q_heads, k_heads, v) ... */ }
        else { /* ... existing no-cache logic ... */ }
        if (l == 0) { log_to_file("attn_out", attn_out); /* Logging lw.o_proj needs thought */ }

        // Output projection (Use new matvec, replaces cblas_sgemv)
        std::vector<float> attn_proj(hs);
        if (l == 0) { /* ... existing logging setup ... */ }
        matvec_bf16_f32(lw.o_proj, attn_out, attn_proj, hs, hs);
        if (l == 0) { log_to_file("attn_proj", attn_proj); }

        // c) Residual connection (operates on float)
        for (int i = 0; i < hs; ++i) x[i] = (x[i] + attn_proj[i]);
        if (l == 0) log_to_file("x (after attn residual)", x);

        // d) Post-attention RMSNorm (weights are uint16_t, input/output are float)
        std::vector<float> ffn_norm(hs);
        rmsnorm(x, lw.post_attention_layernorm, eps, ffn_norm);
        if (l == 0) log_to_file("ffn_norm (Post Attn RMSNorm Out)", ffn_norm);

        // e) MLP (Use new matvec for bfloat16 weights)
        std::vector<float> gate(is), up(is), silu_gate(is), ffn_out(hs);
        matvec_bf16_f32(lw.gate_proj, ffn_norm, gate, is, hs);
        matvec_bf16_f32(lw.up_proj, ffn_norm, up, is, hs);
        if (l == 0) { log_to_file("gate_proj", gate); log_to_file("up_proj", up); }
        silu(gate, silu_gate); // Operates on float
        if (l == 0) log_to_file("silu_gate", silu_gate);
        for (int i = 0; i < is; ++i) up[i] *= silu_gate[i]; // Elementwise product (float)
        if (l == 0) log_to_file("up * silu_gate", up);
        matvec_bf16_f32(lw.down_proj, up, ffn_out, hs, is);
        if (l == 0) log_to_file("ffn_out (MLP Down Proj)", ffn_out);

        // f) Residual connection (operates on float)
        for (int i = 0; i < hs; ++i) x[i] = (x[i] + ffn_out[i]);
        if (l == 0) { 
             log_to_file("x (Layer Output / after ffn residual)", x);
             // ... existing end-of-layer-0 logging ...
        }
    }
    if (cache) cache->seq_len += 1;
    Logger::info("Transformer blocks complete.");

    // 3. Final RMSNorm (weights are uint16_t, input/output are float)
    std::vector<float> x_norm(hs);
    rmsnorm(x, final_norm, eps, x_norm);
    log_vec_stats("final RMSNorm", x_norm);
    if (diag_cb) diag_cb(-1, "final RMSNorm", x_norm);

    // 4. LM head projection (Use new matvec for bfloat16 weights)
    std::vector<float> logits(vs);
    matvec_bf16_f32(lm_head, x_norm, logits, vs, hs);
    log_vec_stats("logits", logits);
    if (diag_cb) diag_cb(-1, "logits", logits);

    Logger::info("Forward pass complete.");
    return logits; // Logits remain float
} 