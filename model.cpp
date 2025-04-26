#include "model.h"
#include "logger.h"
#include <stdexcept>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <fstream>
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
#include <numeric> // For std::accumulate
#include <cassert> // For assert()
#include <iostream> // Include for std::cout in logging
#include <torch/torch.h>

// --- PASTE: Definition of logging helper moved here ---
// (Declaration is in model.h)
void log_vector_summary(const std::string& name, const std::vector<float>& v, int head_count) {
    if (v.empty()) {
        Logger::info(name + ": EMPTY");
        return;
    }
    std::stringstream ss;
    ss << name << ": size=" << v.size() << ", first " << std::min((int)v.size(), head_count) << ": [";
    for(int i = 0; i < std::min((int)v.size(), head_count); ++i) {
        ss << (i > 0 ? " " : "") << std::fixed << std::setprecision(4) << v[i];
    }
    ss << "]";
    // Add basic stats
    float minv = *std::min_element(v.begin(), v.end());
    float maxv = *std::max_element(v.begin(), v.end());
    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    float mean = sum / v.size();
    bool all_finite = std::all_of(v.begin(), v.end(), [](float x){ return std::isfinite(x); });
    ss << ", min=" << minv << ", max=" << maxv << ", mean=" << mean << ", finite=" << (all_finite ? "yes" : "NO");
    Logger::info(ss.str());
}
// --- END PASTE ---

// --- START: New Helper Functions for BFloat16 ---

// Improved BFloat16 to Float32 conversion with proper handling of special values
float bfloat16_to_float32(uint16_t bf16) {
    // Special case handling for important IEEE-754 values
    if (bf16 == 0) return 0.0f; // Positive zero
    if (bf16 == 0x8000) return -0.0f; // Negative zero
    
    // Check for NaN patterns (exponent all 1s, non-zero mantissa)
    bool is_nan = ((bf16 & 0x7F80) == 0x7F80) && ((bf16 & 0x007F) != 0);
    if (is_nan) return std::numeric_limits<float>::quiet_NaN();
    
    // Check for Infinity patterns (exponent all 1s, zero mantissa)
    if ((bf16 & 0x7F80) == 0x7F80 && (bf16 & 0x007F) == 0) {
        return (bf16 & 0x8000) ? -std::numeric_limits<float>::infinity() : 
                                  std::numeric_limits<float>::infinity();
    }
    
    // Normal conversion using bit operations with endianness safety
    uint32_t bits = static_cast<uint32_t>(bf16) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(float));
    
    return result;
}

// Helper to convert a vector of BF16 to F32 efficiently
std::vector<float> bfloat16_vector_to_float32(const std::vector<uint16_t>& bf16_vec) {
    std::vector<float> f32_vec(bf16_vec.size());
    
    #pragma omp parallel for
    for (size_t i = 0; i < bf16_vec.size(); ++i) {
        f32_vec[i] = bfloat16_to_float32(bf16_vec[i]);
    }
    
    return f32_vec;
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

// Replace matvec_bf16_f32 with libtorch matmul
// mat: [M, N], vec: [N] -> out: [M]
torch::Tensor matvec(const torch::Tensor& mat, const torch::Tensor& vec) {
    return torch::matmul(mat, vec);
}

// Replace RMSNorm with libtorch version
torch::Tensor rmsnorm(const torch::Tensor& x, const torch::Tensor& weight, float eps) {
    auto norm = x.norm(2, -1, true);
    return x * (weight / (norm / std::sqrt(x.size(-1)) + eps));
}

// Replace softmax with libtorch version
torch::Tensor softmax(const torch::Tensor& x) {
    return torch::softmax(x, -1);
}

// SiLU activation: y = x * sigmoid(x) - Match PyTorch structure
static void silu(const std::vector<float>& x, std::vector<float>& out) {
    #pragma omp parallel for
    for (size_t i = 0; i < x.size(); ++i) {
        float sigmoid_x = 1.0f / (1.0f + std::exp(-x[i])); // Calculate sigmoid explicitly
        out[i] = x[i] * sigmoid_x; // Multiply
    }
}

// Fix for RoPE implementation with better position handling
void apply_rope(std::vector<float>& x, int num_heads, int head_dim, int pos, 
                const std::vector<std::pair<float, float>>& freqs_cis) {
    // Validate input
    if (head_dim % 2 != 0) {
        Logger::error("apply_rope: head_dim must be even, got " + std::to_string(head_dim));
        return;
    }
    
    // Required vector size check
    const int dim_half = head_dim / 2;
    const size_t expected_size = static_cast<size_t>(num_heads * head_dim);
    if (x.size() != expected_size) {
        Logger::error("apply_rope: Input vector size mismatch. Expected " + 
                     std::to_string(expected_size) + ", got " + std::to_string(x.size()));
        return;
    }
    
    // Validate freqs_cis has enough elements
    if (freqs_cis.size() < static_cast<size_t>(dim_half)) {
        Logger::error("apply_rope: freqs_cis too small. Expected at least " + 
                     std::to_string(dim_half) + ", got " + std::to_string(freqs_cis.size()));
        return;
    }
    
    // Apply rotation for each head
    for (int h = 0; h < num_heads; ++h) {
        const int head_start = h * head_dim;
        
        for (int d = 0; d < dim_half; ++d) {
            // Get indices for the feature dimension and its pair
            const int i = head_start + d;
            const int i_pair = head_start + d + dim_half;
            
            // Ensure indices are in bounds
            if (i >= static_cast<int>(x.size()) || i_pair >= static_cast<int>(x.size())) {
                Logger::error("apply_rope: Index out of bounds: i=" + std::to_string(i) + 
                             ", i_pair=" + std::to_string(i_pair) + 
                             ", x.size()=" + std::to_string(x.size()));
                continue;
            }
            
            // Get the original values
            const float x_i = x[i];
            const float x_i_pair = x[i_pair];
            
            // Get position-specific rotation factors
            const float cos_pos = freqs_cis[d].first;
            const float sin_pos = freqs_cis[d].second;
            
            // Apply the rotation
            x[i] = x_i * cos_pos - x_i_pair * sin_pos;
            x[i_pair] = x_i * sin_pos + x_i_pair * cos_pos;
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

// Helper to log the first few elements of a raw float pointer
static void log_raw_float_pointer(const std::string& name, const float* ptr, size_t count = 5) {
    if (!ptr) {
        Logger::info(name + ": NULL Pointer");
        return;
    }
    std::stringstream ss;
    ss << name << " first " << count << ": [";
    for (size_t i = 0; i < count; ++i) {
        ss << (i > 0 ? " " : "") << std::fixed << std::setprecision(6) << ptr[i];
    }
    ss << "]";
    Logger::info(ss.str());
}

// KVCache initialization method to add to the existing struct
void KVCache::initialize(int num_layers, int max_seq_len, int num_kv_heads, int head_dim) {
    layers.resize(num_layers);
    
    // Calculate the size needed for each layer's K and V cache
    // Structure: sequence_position → kv_head → head_dimension
    size_t cache_size_per_layer = static_cast<size_t>(max_seq_len) * 
                                 static_cast<size_t>(num_kv_heads) * 
                                 static_cast<size_t>(head_dim);
    
    if (cache_size_per_layer == 0) {
        throw std::runtime_error("KVCache: Calculated cache size is zero. Check parameters: " +
                           std::to_string(max_seq_len) + " * " + 
                           std::to_string(num_kv_heads) + " * " + 
                           std::to_string(head_dim));
    }
    
    // Allocate and zero-initialize all cache tensors
    for (int l = 0; l < num_layers; ++l) {
        layers[l].k.resize(cache_size_per_layer, 0.0f);
        layers[l].v.resize(cache_size_per_layer, 0.0f);
    }
    
    Logger::info("KVCache initialized with dimensions: " +
               std::to_string(num_layers) + " layers, " +
               std::to_string(max_seq_len) + " sequence length, " +
               std::to_string(num_kv_heads) + " KV heads, " + 
               std::to_string(head_dim) + " head dimension");
    
    seq_len = 0;
}

// Helper to convert std::vector<float> to torch::Tensor
inline torch::Tensor vec_to_tensor(const std::vector<float>& v, std::vector<int64_t> shape = {}) {
    if (shape.empty()) shape = {static_cast<int64_t>(v.size())};
    return torch::from_blob((void*)v.data(), shape, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}

// Helper to convert std::vector<uint16_t> (BF16) to torch::Tensor (float32)
inline torch::Tensor bf16vec_to_tensor(const std::vector<uint16_t>& v, std::vector<int64_t> shape = {}) {
    std::vector<float> f32(v.size());
    for (size_t i = 0; i < v.size(); ++i) f32[i] = bfloat16_to_float32(v[i]);
    if (shape.empty()) shape = {static_cast<int64_t>(f32.size())};
    return torch::from_blob(f32.data(), shape, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}

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
            lw.k_proj = uint8_vector_to_uint16_vector(loader.get_tensor_bytes(prefix + "self_attn.k_proj.weight"), kv_dim * hs);
            lw.v_proj = uint8_vector_to_uint16_vector(loader.get_tensor_bytes(prefix + "self_attn.v_proj.weight"), kv_dim * hs);
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

    // Precompute RoPE cos/sin values
    int head_dim = hs / n_heads;
    int max_seq_len = config.max_position_embeddings;
    precomputed_freqs_cis_.resize((max_seq_len * head_dim) / 2);
    float theta = config_.rope_theta;
    for (int pos = 0; pos < max_seq_len; ++pos) {
    for (int i = 0; i < head_dim; i += 2) {
            float freq = std::pow(theta, -((float)i) / head_dim);
            float angle = pos * freq;
            float cos_val = std::cos(angle);
            float sin_val = std::sin(angle);
            precomputed_freqs_cis_[(pos * head_dim / 2) + (i / 2)] = {cos_val, sin_val};
            // Log first few values for pos=1
            if (pos == 1 && (i/2) < 5) { // Log first 5 pairs for pos=1
                 Logger::info("C++ RoPE Precompute (Pos=1, FreqDim=" + std::to_string(i/2) + "): cos=" + std::to_string(cos_val) + " sin=" + std::to_string(sin_val));
            }
        }
    }
    Logger::info("Precomputed RoPE cos/sin frequencies.");
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

// --- RESTORED: Token-by-Token + KVCache Forward ---
std::vector<float> TinyLlamaModel::forward(std::vector<float>& x, int pos, KVCache* cache, const std::vector<int>* attention_mask) {
    // Config dimensions
    int hs = config_.hidden_size;
    int is = config_.intermediate_size;
    int nhl = config_.num_hidden_layers;
    int vs = config_.vocab_size;
    int n_heads = config_.num_attention_heads;
    int n_kv_heads = config_.num_key_value_heads;
    int head_dim = hs / n_heads;
    int max_seq_len = config_.max_position_embeddings;
    float eps = config_.rms_norm_eps;

    // --- Flags for detailed logging --- 
    bool log_initial = (pos == 0); 
    // bool log_details_l0_p1 = (pos == 1 && l == 0); // Restore this target

    // if (log_initial) Logger::info("--- TROUBLESHOOTING INITIAL (C++) START pos=0 ---");
    
    if (pos >= max_seq_len) {
        Logger::error("Position index exceeds max_position_embeddings");
        return {};
    }
    if (!cache) {
        Logger::error("KVCache is required for token-by-token forward pass");
        return {};
    }

    // 2. Process through all transformer layers
    for (int l = 0; l < nhl; ++l) {
        // --- Restore logging target to L0 P1 --- 
        bool log_target_layer = (pos == 1 && l == 0); 
        const auto& lw = layers[l];
        std::vector<float> x_resid1 = x; 
        // --- Use L0 P1 flag --- 
        if (log_target_layer) log_vector_summary("TROUBLESHOOTING L0 P1 Input", x_resid1);

        // a) Input RMSNorm
        torch::Tensor x_tensor = vec_to_tensor(x, {hs});
        torch::Tensor w_tensor = bf16vec_to_tensor(lw.input_layernorm, {hs});
        torch::Tensor x_norm_tensor = rmsnorm(x_tensor, w_tensor, eps);
        std::vector<float> x_norm(x_norm_tensor.template data_ptr<float>(), x_norm_tensor.template data_ptr<float>() + hs);
        // --- Use L0 P1 flag --- 
        if (log_target_layer) log_vector_summary("TROUBLESHOOTING L0 P1 Output RMSNorm1", x_norm);

        // b) Q, K, V projections
        torch::Tensor q_tensor = matvec(bf16vec_to_tensor(lw.q_proj, {hs, hs}), x_norm_tensor);
        torch::Tensor k_current_tensor = matvec(bf16vec_to_tensor(lw.k_proj, {n_kv_heads * head_dim, hs}), x_norm_tensor);
        torch::Tensor v_current_tensor = matvec(bf16vec_to_tensor(lw.v_proj, {n_kv_heads * head_dim, hs}), x_norm_tensor);
        std::vector<float> q(q_tensor.template data_ptr<float>(), q_tensor.template data_ptr<float>() + hs);
        std::vector<float> k_current(k_current_tensor.template data_ptr<float>(), k_current_tensor.template data_ptr<float>() + n_kv_heads * head_dim);
        std::vector<float> v_current(v_current_tensor.template data_ptr<float>(), v_current_tensor.template data_ptr<float>() + n_kv_heads * head_dim);
        // --- Use L0 P1 flag --- 
        if (log_target_layer) {
            log_vector_summary("TROUBLESHOOTING L0 P1 Q Proj Output", q);
            log_vector_summary("TROUBLESHOOTING L0 P1 K Proj Output", k_current);
            log_vector_summary("TROUBLESHOOTING L0 P1 V Proj Output", v_current);
        }

        // c) RoPE for Q, K
        std::vector<float> q_before_rope = q; // Copy for logging
        apply_rope(q, n_heads, head_dim, pos, precomputed_freqs_cis_);
        std::vector<float> k_before_rope = k_current; // Copy for logging
        apply_rope(k_current, n_kv_heads, head_dim, pos, precomputed_freqs_cis_);
        // --- Use L0 P1 flag --- 
        if (log_target_layer) {
            log_vector_summary("TROUBLESHOOTING L0 P1 Q BEFORE RoPE", q_before_rope);
            log_vector_summary("TROUBLESHOOTING L0 P1 Q AFTER RoPE", q);
            log_vector_summary("TROUBLESHOOTING L0 P1 K BEFORE RoPE", k_before_rope);
            log_vector_summary("TROUBLESHOOTING L0 P1 K AFTER RoPE", k_current);
        }

        // Write to KV Cache
            for (int kvh = 0; kvh < n_kv_heads; ++kvh) {
            // Calculate offsets into current K and V vectors
                size_t current_k_offset = (size_t)kvh * head_dim;
                size_t current_v_offset = (size_t)kvh * head_dim;
            
            // Calculate write position in cache (matching the read pattern used in attention)
            // Structure: pos → kv_head → head_dim
            size_t write_offset = pos * n_kv_heads * head_dim + kvh * head_dim;
            
            // Bounds check before writing
            if (write_offset + head_dim <= cache->layers[l].k.size()) {
                // Write K and V vectors to cache
                std::memcpy(&cache->layers[l].k[write_offset], &k_current[current_k_offset], head_dim * sizeof(float));
                std::memcpy(&cache->layers[l].v[write_offset], &v_current[current_v_offset], head_dim * sizeof(float));
             } else {
                Logger::error("KVCache write out of bounds: write_offset=" + 
                             std::to_string(write_offset) + ", cache size=" + 
                             std::to_string(cache->layers[l].k.size()));
             }
        }

        // d) Attention
        std::vector<float> attn_out(hs, 0.0f);
        if (pos > 0) {
            int current_seq_len = pos + 1; // Include current position
            if (current_seq_len > 0) { 
                #pragma omp parallel for
            for (int h = 0; h < n_heads; ++h) {
                    const float* q_ptr = q.data() + h * head_dim;
                    std::vector<float> scores(current_seq_len);
                    
                    // Determine which KV head this Q head maps to for Grouped Query Attention (GQA)
                    int kv_head_idx = h / (n_heads / n_kv_heads); // This ensures proper GQA head mapping
                    
                    // Calculate Scores
                    for (int j = 0; j < current_seq_len; ++j) {
                        // Calculate cache position for current KV head and sequence position
                        size_t cache_pos_offset = j * n_kv_heads * head_dim + kv_head_idx * head_dim;
                        
                        // Bounds checking
                        if (cache_pos_offset + head_dim <= cache->layers[l].k.size()) {
                            const float* k_ptr = &cache->layers[l].k[cache_pos_offset];
                            
                            // Calculate attention score (dot product of Q and K)
                            float score = 0.0f;
                            for (int d = 0; d < head_dim; ++d) {
                                score += q_ptr[d] * k_ptr[d];
                            }
                            score /= std::sqrt(head_dim); // Scale by sqrt(head_dim)
                            scores[j] = score;
                } else {
                            Logger::error("Attention K access out of bounds: cache_pos_offset=" + 
                                         std::to_string(cache_pos_offset) + ", cache size=" + 
                                         std::to_string(cache->layers[l].k.size()));
                            scores[j] = -1e9f; // Large negative value to avoid NaN after softmax
                        }
                    }
                    
                    // Convert scores vector to tensor and apply softmax
                    torch::Tensor scores_tensor = vec_to_tensor(scores);
                    torch::Tensor probs_tensor = softmax(scores_tensor);
                    // Convert probs back to vector if needed, or use tensor directly
                    std::vector<float> probs(probs_tensor.template data_ptr<float>(), probs_tensor.template data_ptr<float>() + current_seq_len);
                    log_vector_stats("Attention Scores Post-Softmax (L" + std::to_string(l) + " H" + std::to_string(h) + " P" + std::to_string(pos) + ")", probs);

                    // Compute weighted sum of V vectors using the probabilities
                    std::vector<float> head_attn_out(head_dim, 0.0f);
                    for (int d = 0; d < head_dim; ++d) {
                        double val = 0.0; // Use double for accumulation
                        for (int j = 0; j < current_seq_len; ++j) {
                            // Use the same cache position calculation for V
                            size_t cache_pos_offset = j * n_kv_heads * head_dim + kv_head_idx * head_dim;
                            
                            if (cache_pos_offset + head_dim <= cache->layers[l].v.size()) {
                                const float* v_ptr = &cache->layers[l].v[cache_pos_offset];
                                // Use the probability from the softmax result
                                val += static_cast<double>(probs[j]) * static_cast<double>(v_ptr[d]); 
                } else {
                                // Error already logged during K access, skip V access
                            }
                        }
                        head_attn_out[d] = static_cast<float>(val);
                    }
                    
                    // Copy this head's output to the correct position in attn_out
                    std::memcpy(attn_out.data() + h * head_dim, head_attn_out.data(), head_dim * sizeof(float));
                }
            }
        } else {
            // For pos=0, no attention is calculated (output is all zeros)
            // existing code...
        }

        // e) Output projection
        std::vector<float> attn_out_copy = attn_out; 
        torch::Tensor attn_proj_tensor = matvec(bf16vec_to_tensor(lw.o_proj, {hs, hs}), vec_to_tensor(attn_out, {hs}));
        std::vector<float> attn_proj(attn_proj_tensor.template data_ptr<float>(), attn_proj_tensor.template data_ptr<float>() + hs);
        log_vector_stats("Attention Projection Output (L" + std::to_string(l) + " P" + std::to_string(pos) + ")", attn_proj);

        // f) First residual connection
        std::vector<float> x_resid1_copy = x_resid1;
        for (int i = 0; i < hs; ++i) x[i] = x_resid1[i] + attn_proj[i];
        // --- Use L0 P1 flag --- 
        if (log_target_layer) { 
            log_vector_summary("TROUBLESHOOTING L0 P1 State BEFORE 1st Residual", x_resid1_copy);
            log_vector_summary("TROUBLESHOOTING L0 P1 State AFTER 1st Residual", x);
        }

        // g) MLP block
        std::vector<float> x_resid2 = x; 
        // --- Use L0 P1 flag --- 
        if (log_target_layer) log_vector_summary("TROUBLESHOOTING L0 P1 Input to MLP", x_resid2);
        torch::Tensor x_norm2_tensor = rmsnorm(vec_to_tensor(x, {hs}), bf16vec_to_tensor(lw.post_attention_layernorm, {hs}), eps);
        std::vector<float> x_norm2(x_norm2_tensor.template data_ptr<float>(), x_norm2_tensor.template data_ptr<float>() + hs);
        log_vector_stats("Input to MLP RMSNorm (L" + std::to_string(l) + " P" + std::to_string(pos) + ")", x_norm2);

        std::vector<float> gate(is), up(is);
        torch::Tensor gate_proj_tensor = matvec(bf16vec_to_tensor(lw.gate_proj, {is, hs}), vec_to_tensor(x_norm2, {is}));
        torch::Tensor up_proj_tensor = matvec(bf16vec_to_tensor(lw.up_proj, {is, hs}), vec_to_tensor(x_norm2, {is}));
        std::vector<float> gate_orig = gate; 
        for (int d = 0; d < is; ++d) {
            gate[d] = (gate[d] / (1.0f + std::exp(-gate[d]))) * up[d]; // SiLU * Up
        }
        if (log_target_layer) {
            log_vector_summary("TROUBLESHOOTING L0 P1 Gate BEFORE SiLU*Up", gate_orig);
            log_vector_summary("TROUBLESHOOTING L0 P1 Gate * Up", gate);
        }
        torch::Tensor mlp_out_tensor = matvec(bf16vec_to_tensor(lw.down_proj, {hs, is}), vec_to_tensor(gate, {hs}));
        std::vector<float> mlp_out(mlp_out_tensor.template data_ptr<float>(), mlp_out_tensor.template data_ptr<float>() + hs);
        log_vector_stats("MLP Output (L" + std::to_string(l) + " P" + std::to_string(pos) + ")", mlp_out);
        
        // h) Second residual connection
        std::vector<float> x_resid2_copy = x_resid2; 
        for (int i = 0; i < hs; ++i) x[i] = x_resid2[i] + mlp_out[i];
        if (log_target_layer) { 
            log_vector_summary("TROUBLESHOOTING L0 P1 State BEFORE 2nd Residual", x_resid2_copy);
            log_vector_summary("TROUBLESHOOTING L0 P1 State AFTER 2nd Residual", x);
            Logger::info("--- TROUBLESHOOTING L0 P1 C++ Layer End ---");
        }
    } // End layer loop

    // 3. Final RMSNorm
    std::vector<float> x_final_norm_input = x; // Copy for logging
    torch::Tensor x_final_norm_tensor = rmsnorm(vec_to_tensor(x, {hs}), bf16vec_to_tensor(final_norm, {hs}), eps);
    std::vector<float> x_norm_final(x_final_norm_tensor.template data_ptr<float>(), x_final_norm_tensor.template data_ptr<float>() + hs);
    if (log_initial) { 
        log_vector_summary("TROUBLESHOOTING L0 P1 Input to Final RMSNorm (P1)", x_final_norm_input);
        log_vector_summary("TROUBLESHOOTING L0 P1 Output of Final RMSNorm (P1)", x_norm_final);
    }

    // 4. Output projection to logits
    torch::Tensor logits_tensor = matvec(bf16vec_to_tensor(lm_head, {vs, hs}), x_final_norm_tensor);
    std::vector<float> logits(logits_tensor.template data_ptr<float>(), logits_tensor.template data_ptr<float>() + vs);
    if (log_initial) {
        log_vector_summary("TROUBLESHOOTING L0 P1 Final Logits (P1)", logits, 10);
    }

    if (log_initial) Logger::info("--- TROUBLESHOOTING L0 P1 (C++) END pos=1 ---");
    return logits; 
}

// --- Get Vocab Size --- 
int TinyLlamaModel::get_vocab_size() const {
    return config_.vocab_size;
} 