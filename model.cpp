#include "model.h"
#include "logger.h"
#ifdef HAS_CUDA
#include "cuda_kernels.h" // Include CUDA kernel declarations
#endif
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

// --- START: Argmax Helper ---
// Find the index of the maximum element in a vector
int argmax(const std::vector<float>& v) {
    if (v.empty()) {
        // throw std::runtime_error("Cannot perform argmax on empty vector");
        Logger::error("Cannot perform argmax on empty vector"); // Log instead of throwing
        return -1; // Return an invalid index
    }
    return std::distance(v.begin(), std::max_element(v.begin(), v.end()));
}
// --- END: Argmax Helper ---

// mat: [M, N], vec: [N] -> out: [M]
// torch::Tensor matvec(const torch::Tensor& mat, const torch::Tensor& vec) {
//     return torch::matmul(mat, vec);
// }

// --- START: C++ Vector RMSNorm ---
static void rmsnorm_vector(const std::vector<float>& x, const std::vector<float>& weight, std::vector<float>& out, float eps) {
#ifdef HAS_CUDA
    // Logger::info("Using CUDA RMSNorm"); // Optional: Log which version is used
    // Call the CUDA wrapper function
    rmsnorm_vector_cuda(x, weight, out, x.size(), eps); 
#else
    // Logger::info("Using CPU RMSNorm (OpenMP)"); // Optional: Log which version is used
    // Original OpenMP implementation
    if (x.empty() || x.size() != weight.size()) {
        Logger::error("RMSNorm vector size mismatch or empty input.");
        out.assign(x.size(), 0.0f); // Zero out output on error
        return; 
    }
    out.resize(x.size());
    size_t n = x.size();

    // Calculate sum of squares
    double ssq = 0.0;
    #pragma omp parallel for reduction(+:ssq)
    for (size_t i = 0; i < n; ++i) {
        ssq += static_cast<double>(x[i]) * static_cast<double>(x[i]);
    }
    ssq /= n;

    // Compute normalization factor
    float norm_factor = 1.0f / std::sqrt(static_cast<float>(ssq) + eps);

    // Normalize and apply weight
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        out[i] = x[i] * norm_factor * weight[i];
    }
#endif // HAS_CUDA
}
// --- END: C++ Vector RMSNorm ---

// --- START: C++ Vector Softmax ---
// Compute softmax for a vector in-place (or return new vector)
static void softmax_vector(const std::vector<float>& x, std::vector<float>& out) {
#ifdef HAS_CUDA
    // Logger::info("Using CUDA Softmax"); // Optional log
    softmax_vector_cuda(x, out, x.size());
#else
    // Logger::info("Using CPU Softmax (OpenMP)"); // Optional log
    // Original OpenMP Implementation
    if (x.empty()) return;
    out.resize(x.size());

    // Find max element for numerical stability
    float max_val = *std::max_element(x.begin(), x.end());

    // Compute exponentials and sum
    float exp_sum = 0.0f;
    for (size_t i = 0; i < x.size(); ++i) {
        out[i] = std::exp(x[i] - max_val);
        exp_sum += out[i];
    }

    // Normalize
    float inv_sum = 1.0f / exp_sum;
    #pragma omp parallel for
    for (size_t i = 0; i < x.size(); ++i) {
        out[i] *= inv_sum;
    }
#endif // HAS_CUDA
}
// --- END: C++ Vector Softmax ---

// C++ Vector-based SiLU activation: y = x * sigmoid(x)
static void silu(const std::vector<float>& x, std::vector<float>& out) {
#ifdef HAS_CUDA
    // Logger::info("Using CUDA SiLU"); // Optional log
    silu_cuda(x, out, x.size());
#else
    // Logger::info("Using CPU SiLU (OpenMP)"); // Optional log
    // Original OpenMP implementation
    if (x.size() != out.size()) out.resize(x.size()); // Ensure output is sized correctly
    #pragma omp parallel for
    for (size_t i = 0; i < x.size(); ++i) {
        float sigmoid_x = 1.0f / (1.0f + std::exp(-x[i])); // Calculate sigmoid explicitly
        out[i] = x[i] * sigmoid_x; // Multiply
    }
#endif // HAS_CUDA
}

// Helper: log vector stats (min, max, mean, all finite)
static void log_vec_stats(const std::string& name, const std::vector<float>& v) {
    if (v.empty()) {
        Logger::info(name + ": EMPTY VECTOR");
        return;
    }
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


// Helper function to convert bfloat16 vector to float vector
static std::vector<float> bf16vec_to_float_vec(const std::vector<uint16_t>& v_bf16) {
    std::vector<float> v_f32(v_bf16.size());
    #pragma omp parallel for
    for (size_t i = 0; i < v_bf16.size(); ++i) {
        v_f32[i] = bfloat16_to_float32(v_bf16[i]);
    }
    return v_f32;
}

// --- START: C++ Vector MatVec BF16 * F32 -> F32 with Kahan Summation ---
// Performs matrix-vector multiplication: out = mat (bf16) * vec (f32)
// mat shape: [rows, cols], vec shape: [cols], out shape: [rows]
static void matvec_bf16_f32_vector(const std::vector<uint16_t>& mat_bf16,
                                   const std::vector<float>& vec_f32,
                                   std::vector<float>& out_f32,
                                   int rows, int cols) {
#ifdef HAS_CUDA
    // Logger::info("Using CUDA MatVec (BF16*F32->F32)"); // Optional log
    // Call the CUDA wrapper function
    matvec_bf16_f32_cuda(mat_bf16, vec_f32, out_f32, rows, cols);
#else
    // Logger::info("Using CPU MatVec (BF16*F32->F32, OpenMP+Kahan)"); // Optional log
    // Original OpenMP + Kahan implementation
    if (mat_bf16.size() != (size_t)rows * cols || vec_f32.size() != (size_t)cols) {
        Logger::error("matvec_bf16_f32_vector: Size mismatch. Mat: " + std::to_string(mat_bf16.size()) +
                     " (Expected " + std::to_string(rows*cols) + "), Vec: " + std::to_string(vec_f32.size()) +
                     " (Expected " + std::to_string(cols) + ")");
        out_f32.assign(rows, 0.0f); // Zero out output on error
        return;
    }
    out_f32.resize(rows);

    #pragma omp parallel for
    for (int r = 0; r < rows; ++r) {
        double sum = 0.0;        // Running sum (using double for intermediate precision)
        double c = 0.0;          // Kahan summation compensation
        size_t row_offset = r * cols;
        
        for (int c_idx = 0; c_idx < cols; ++c_idx) {
            // Get weight and input value
            float weight = bfloat16_to_float32(mat_bf16[row_offset + c_idx]);
            double term = static_cast<double>(weight) * static_cast<double>(vec_f32[c_idx]);
            
            // Kahan Summation step
            double y = term - c;
            double t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        out_f32[r] = static_cast<float>(sum); // Assign final sum (cast back to float)
    }
#endif // HAS_CUDA
}
// --- END: C++ Vector MatVec with Kahan ---

// --- START: C++ Weighted Sum (Probs * V) ---
// Calculates weighted sum: out = probs @ V
// probs shape: [seq_len], V shape: [seq_len, head_dim], out shape: [head_dim]
static void weighted_sum_probs_v(const std::vector<float>& probs, 
                                 const std::vector<float>& V, 
                                 std::vector<float>& out, 
                                 int seq_len, int head_dim) {
    if (probs.size() != seq_len || V.size() != (size_t)seq_len * head_dim) {
        Logger::error("weighted_sum_probs_v: Size mismatch. Probs: " + std::to_string(probs.size()) + 
                     " (Expected " + std::to_string(seq_len) + "), V: " + std::to_string(V.size()) + 
                     " (Expected " + std::to_string(seq_len * head_dim) + ")");
        out.assign(head_dim, 0.0f);
        return;
    }
    out.resize(head_dim);

    #pragma omp parallel for
    for (int j = 0; j < head_dim; ++j) {
        double sum = 0.0;
        double c_kahan = 0.0; // Kahan summation compensation
        for (int i = 0; i < seq_len; ++i) {
            double term = static_cast<double>(probs[i]) * static_cast<double>(V[i * head_dim + j]);
            // Kahan sum
            double y = term - c_kahan;
            double t = sum + y;
            c_kahan = (t - sum) - y;
            sum = t;
        }
        out[j] = static_cast<float>(sum);
    }
}
// --- END: C++ Weighted Sum (Probs * V) ---

// --- START: C++ Attention Scores (Q * K^T) ---
// Calculates attention scores: scores = (Q @ K^T) * scale
// Q shape: [head_dim], K shape: [seq_len, head_dim], scores shape: [seq_len]
static void calculate_attention_scores(const std::vector<float>& Q, 
                                     const std::vector<float>& K, 
                                     std::vector<float>& scores, 
                                     int seq_len, int head_dim, float scale) {
    if (Q.size() != head_dim || K.size() != (size_t)seq_len * head_dim) {
        Logger::error("calculate_attention_scores: Size mismatch. Q: " + std::to_string(Q.size()) + 
                     " (Expected " + std::to_string(head_dim) + "), K: " + std::to_string(K.size()) + 
                     " (Expected " + std::to_string(seq_len * head_dim) + ")");
        scores.assign(seq_len, 0.0f);
        return;
    }
    scores.resize(seq_len);

    #pragma omp parallel for
    for (int t = 0; t < seq_len; ++t) { // Iterate over each K vector (timestep)
        double dot_product = 0.0;
        double c_kahan = 0.0; // Kahan compensation
        size_t k_offset = t * head_dim;
        
        for (int i = 0; i < head_dim; ++i) { // Dot product calculation
            double term = static_cast<double>(Q[i]) * static_cast<double>(K[k_offset + i]);
            // Kahan sum
            double y = term - c_kahan;
            double t_sum = dot_product + y;
            c_kahan = (t_sum - dot_product) - y;
            dot_product = t_sum;
        }
        scores[t] = static_cast<float>(dot_product) * scale; // Apply scale
    }
}
// --- END: C++ Attention Scores ---
static void apply_rope_vector(std::vector<float>& x, int num_heads, int head_dim, int pos, 
                       const std::vector<std::pair<float, float>>& freqs_cis) {
    if (x.size() != num_heads * head_dim) {
        Logger::error("apply_rope_vector: Input vector x has incorrect size. Expected " + 
                     std::to_string(num_heads * head_dim) + ", got " + std::to_string(x.size()));
        return;
    }
    if (head_dim % 2 != 0) {
        Logger::error("apply_rope_vector: head_dim must be even, got " + std::to_string(head_dim));
        return;
    }

    const int dim_half = head_dim / 2;

    #pragma omp parallel for // Re-enable OMP
    for (int h = 0; h < num_heads; ++h) {
        size_t head_offset = h * head_dim;
        for (int i = 0; i < dim_half; ++i) { // Iterate up to dim_half
            // Get corresponding elements from first and second halves
            double x0 = static_cast<double>(x[head_offset + i]);          // Element from first half
            double x1 = static_cast<double>(x[head_offset + i + dim_half]); // Element from second half
            
            // Get frequencies for this dimension index 'i'
            double cos_val = static_cast<double>(freqs_cis[i].first); 
            double sin_val = static_cast<double>(freqs_cis[i].second);
            
            // Apply rotation correctly
            double rotated_x0 = x0 * cos_val - x1 * sin_val; // Rotated first half element
            double rotated_x1 = x0 * sin_val + x1 * cos_val; // Rotated second half element

            // Write rotated values back to their original positions
            x[head_offset + i] = static_cast<float>(rotated_x0); 
            x[head_offset + i + dim_half] = static_cast<float>(rotated_x1);
        }
    }
}

/* // Torch tensor apply_rope REMOVED
void apply_rope(torch::Tensor& x, int num_heads, int head_dim, int pos, 
                const std::vector<std::pair<float, float>>& freqs_cis) {
    // ... 
}
*/

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

// Update lookup_embedding to return std::vector<float>
std::vector<float> TinyLlamaModel::lookup_embedding(int token_id) {
    int hs = config_.hidden_size;
    int vs = config_.vocab_size;
    if (token_id < 0 || token_id >= vs) {
        Logger::error("Token ID out of bounds in lookup_embedding: " + std::to_string(token_id));
        // Return a zero vector of the correct shape
        return std::vector<float>(hs, 0.0f); 
    }
    
    size_t offset = (size_t)token_id * hs;
    if (offset + hs > embed_tokens.size()) {
         Logger::error("Embedding offset out of bounds in lookup_embedding for token: " + std::to_string(token_id));
         return std::vector<float>(hs, 0.0f);
    }

    // Create a sub-vector view (or copy)
    std::vector<uint16_t> token_embedding_bf16(embed_tokens.begin() + offset, embed_tokens.begin() + offset + hs);
    
    // Convert the bfloat16 vector slice to a float32 vector
    std::vector<float> embedding_vec = bf16vec_to_float_vec(token_embedding_bf16);
    
    // Logger::info("Performed embedding lookup for token: " + std::to_string(token_id));
    return embedding_vec; // Return the vector
}

// --- Forward Pass (Now takes std::vector<float>&) ---
std::vector<float> TinyLlamaModel::forward(std::vector<float>& x_vec, int pos, KVCache* cache, const std::vector<int>* attention_mask) {
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

#ifdef HAS_CUDA
    Logger::info("[CUDA] Allocating device buffer for x_vec");
    float* x_dev = nullptr;
    gpuErrchk(cudaMalloc(&x_dev, hs * sizeof(float)));
    Logger::info("[CUDA] Device buffer allocated");
    Logger::info("[CUDA] Copying initial embedding to device");
    gpuErrchk(cudaMemcpy(x_dev, x_vec.data(), hs * sizeof(float), cudaMemcpyHostToDevice));
    Logger::info("[CUDA] Initial embedding copied to device");
    // Allocate device buffer for RMSNorm output
    float* x_norm_dev = nullptr;
    gpuErrchk(cudaMalloc(&x_norm_dev, hs * sizeof(float)));
    // Copy RMSNorm weights to device once
    float* w_norm1_dev = nullptr;
    std::vector<float> w_norm1_vec = bf16vec_to_float_vec(layers[0].input_layernorm); // Use layer 0 for now
    gpuErrchk(cudaMalloc(&w_norm1_dev, hs * sizeof(float)));
    gpuErrchk(cudaMemcpy(w_norm1_dev, w_norm1_vec.data(), hs * sizeof(float), cudaMemcpyHostToDevice));
#endif

    bool log_initial = (pos == 0);
    
    if (pos >= max_seq_len) {
        Logger::error("Position index exceeds max_position_embeddings");
        return std::vector<float>(vs, 0.0f); // Return zero logits
    }
    if (!cache) {
        Logger::error("KVCache is required for token-by-token forward pass");
        return std::vector<float>(vs, 0.0f);
    }
    if (x_vec.size() != hs) {
        Logger::error("Input vector x_vec has incorrect size. Expected " + std::to_string(hs) + ", got " + std::to_string(x_vec.size()));
        return std::vector<float>(vs, 0.0f);
    }

    // Intermediate vectors needed within the loop
    std::vector<float> x_norm_vec1(hs);
    std::vector<float> q_vec(hs);
    std::vector<float> k_vec(n_kv_heads * head_dim);
    std::vector<float> v_vec(n_kv_heads * head_dim);
    std::vector<float> attn_out_vec(hs);
    std::vector<float> attn_proj_vec(hs);
    std::vector<float> x_norm_vec2(hs);
    std::vector<float> gate_vec(is);
    std::vector<float> up_vec(is);
    std::vector<float> silu_out_vec(is);
    std::vector<float> swiglu_result_vec(is);
    std::vector<float> mlp_out_vec(hs);

    // 2. Process through all transformer layers
    for (int l = 0; l < nhl; ++l) {
        bool log_target_layer = (pos == 1 && l == 0);
        const auto& lw = layers[l];

        // Store residual
        std::vector<float> x_resid1_vec = x_vec; // Copy for residual connection 1
        if (log_target_layer) log_vector_summary("TROUBLESHOOTING L0 P1 Input", x_resid1_vec);

        // a) Input RMSNorm (Using C++ Vector version)
#ifdef HAS_CUDA
        // Log input and weights before RMSNorm
        std::stringstream ss_in, ss_w;
        ss_in << "[CUDA RMSNorm] x_vec first 5: ";
        for (int i = 0; i < 5 && i < x_vec.size(); ++i) ss_in << x_vec[i] << " ";
        Logger::info(ss_in.str());
        std::vector<float> w_norm1_vec = bf16vec_to_float_vec(lw.input_layernorm);
        ss_w << "[CUDA RMSNorm] w_norm1_vec first 5: ";
        for (int i = 0; i < 5 && i < w_norm1_vec.size(); ++i) ss_w << w_norm1_vec[i] << " ";
        Logger::info(ss_w.str());
        gpuErrchk(cudaMemcpy(x_dev, x_vec.data(), hs * sizeof(float), cudaMemcpyHostToDevice));
        float* w_norm1_dev = nullptr;
        gpuErrchk(cudaMalloc(&w_norm1_dev, hs * sizeof(float)));
        gpuErrchk(cudaMemcpy(w_norm1_dev, w_norm1_vec.data(), hs * sizeof(float), cudaMemcpyHostToDevice));
        Logger::info("[CUDA] Calling device-pointer RMSNorm for input");
        rmsnorm_vector_cuda(x_dev, w_norm1_dev, x_norm_dev, hs, eps);
        gpuErrchk(cudaDeviceSynchronize());
        Logger::info("[CUDA] Copying RMSNorm output back to host");
        cudaMemcpy(x_norm_vec1.data(), x_norm_dev, hs * sizeof(float), cudaMemcpyDeviceToHost);
        // Log output after RMSNorm
        std::stringstream ss_out;
        ss_out << "[CUDA RMSNorm] x_norm_vec1 first 5: ";
        for (int i = 0; i < 5 && i < x_norm_vec1.size(); ++i) ss_out << x_norm_vec1[i] << " ";
        Logger::info(ss_out.str());
        gpuErrchk(cudaFree(w_norm1_dev));
#else
        std::vector<float> w_norm1_vec = bf16vec_to_float_vec(lw.input_layernorm); 
        rmsnorm_vector(x_vec, w_norm1_vec, x_norm_vec1, eps); // Input is x_vec, output x_norm_vec1
#endif
        if (log_target_layer) log_vector_summary("TROUBLESHOOTING L0 P1 Output C++ RMSNorm1", x_norm_vec1);

        // b) Q, K, V projections (Using C++ Vector MatVec)
        matvec_bf16_f32_vector(lw.q_proj, x_norm_vec1, q_vec, hs, hs);
        matvec_bf16_f32_vector(lw.k_proj, x_norm_vec1, k_vec, n_kv_heads * head_dim, hs);
        matvec_bf16_f32_vector(lw.v_proj, x_norm_vec1, v_vec, n_kv_heads * head_dim, hs);
        
        // Log vectors BEFORE RoPE
        if (log_target_layer) {
            log_vector_summary("TROUBLESHOOTING L0 P1 Q Proj Output (C++ Vec, Before RoPE)", q_vec);
            log_vector_summary("TROUBLESHOOTING L0 P1 K Proj Output (C++ Vec, Before RoPE)", k_vec);
            log_vector_summary("TROUBLESHOOTING L0 P1 V Proj Output (C++ Vec)", v_vec);
        }

        // c) RoPE for Q, K (Using C++ Vector version)
        size_t freqs_offset = (pos * head_dim / 2);
        std::vector<std::pair<float, float>> current_freqs_cis(precomputed_freqs_cis_.begin() + freqs_offset, 
                                                               precomputed_freqs_cis_.begin() + freqs_offset + head_dim / 2);
        apply_rope_vector(q_vec, n_heads, head_dim, pos, current_freqs_cis);
        apply_rope_vector(k_vec, n_kv_heads, head_dim, pos, current_freqs_cis);
        
        if (log_target_layer) {
            log_vector_summary("TROUBLESHOOTING L0 P1 Q AFTER C++ RoPE", q_vec);
            log_vector_summary("TROUBLESHOOTING L0 P1 K AFTER C++ RoPE", k_vec);
        }

        // Write to KV Cache (use RoPE-modified k_vec, original v_vec)
        float* k_current_ptr = k_vec.data(); 
        float* v_current_ptr = v_vec.data(); 
            for (int kvh = 0; kvh < n_kv_heads; ++kvh) {
                size_t current_k_offset = (size_t)kvh * head_dim;
                size_t current_v_offset = (size_t)kvh * head_dim;
            size_t write_offset = pos * n_kv_heads * head_dim + kvh * head_dim;
            if (write_offset + head_dim <= cache->layers[l].k.size()) {
                std::memcpy(&cache->layers[l].k[write_offset], k_current_ptr + current_k_offset, head_dim * sizeof(float));
                std::memcpy(&cache->layers[l].v[write_offset], v_current_ptr + current_v_offset, head_dim * sizeof(float));
                } else {
                Logger::error("KVCache write out of bounds: write_offset=" + std::to_string(write_offset) + ", cache size=" + std::to_string(cache->layers[l].k.size()));
            }
        }

        // d) Attention (using C++ Scores and Weighted Sum)
        std::fill(attn_out_vec.begin(), attn_out_vec.end(), 0.0f); // Zero out accumulator vector
        int current_seq_len = pos + 1;
        float scale = 1.0f / std::sqrt(head_dim);
        
            for (int h = 0; h < n_heads; ++h) {
            // Get Q head slice from RoPE-modified q_vec
            std::vector<float> q_head_rope_vec(q_vec.begin() + h * head_dim, q_vec.begin() + (h + 1) * head_dim);
            
            int kv_head_idx = h / (n_heads / n_kv_heads);
            
            // Retrieve K and V from cache
            std::vector<float> k_cache_head_vec(current_seq_len * head_dim); 
            std::vector<float> v_cache_head_vec(current_seq_len * head_dim); 
            for (int j = 0; j < current_seq_len; ++j) {
                size_t cache_pos_offset = j * n_kv_heads * head_dim + kv_head_idx * head_dim;
                if (cache_pos_offset + head_dim <= cache->layers[l].k.size()) {
                    std::memcpy(k_cache_head_vec.data() + j * head_dim, &cache->layers[l].k[cache_pos_offset], head_dim * sizeof(float));
                    std::memcpy(v_cache_head_vec.data() + j * head_dim, &cache->layers[l].v[cache_pos_offset], head_dim * sizeof(float));
                    } else {
                    std::fill(k_cache_head_vec.begin() + j * head_dim, k_cache_head_vec.begin() + (j + 1) * head_dim, 0.0f);
                    std::fill(v_cache_head_vec.begin() + j * head_dim, v_cache_head_vec.begin() + (j + 1) * head_dim, 0.0f);
                    Logger::error("Attention K/V access out of bounds: cache_pos_offset=" + std::to_string(cache_pos_offset));
                }
            }
            
            // Calculate Scores
            std::vector<float> scores_vec(current_seq_len);
            calculate_attention_scores(q_head_rope_vec, k_cache_head_vec, scores_vec, current_seq_len, head_dim, scale);

            // Apply Softmax
            std::vector<float> probs_vec(current_seq_len);
            softmax_vector(scores_vec, probs_vec);

            // Calculate Weighted Sum
            std::vector<float> head_attn_out_vec(head_dim);
            weighted_sum_probs_v(probs_vec, v_cache_head_vec, head_attn_out_vec, current_seq_len, head_dim);
            
            // Accumulate head output into attn_out_vec (Vector addition)
            size_t out_offset = h * head_dim;
            for(int i=0; i<head_dim; ++i) {
                attn_out_vec[out_offset + i] += head_attn_out_vec[i];
            }
        }

        // e) Output projection (Using C++ Vector MatVec)
        matvec_bf16_f32_vector(lw.o_proj, attn_out_vec, attn_proj_vec, hs, hs);

        // f) First residual connection (Using C++ Vector version)
        #pragma omp parallel for
        for(size_t i=0; i<hs; ++i) {
            x_vec[i] = x_resid1_vec[i] + attn_proj_vec[i]; // Update x_vec in-place
        }
        
        if (log_target_layer) {
            log_vector_summary("TROUBLESHOOTING L0 P1 State BEFORE 1st Residual", x_resid1_vec);
            log_vector_summary("TROUBLESHOOTING L0 P1 State AFTER 1st C++ Residual", x_vec);
        }

        // g) MLP block
        std::vector<float> x_resid2_vec = x_vec; // Copy for residual connection 2
        if (log_target_layer) log_vector_summary("TROUBLESHOOTING L0 P1 Input to MLP", x_resid2_vec);
        
        // Post-attention RMSNorm
        std::vector<float> w_norm2_vec = bf16vec_to_float_vec(lw.post_attention_layernorm);
        rmsnorm_vector(x_vec, w_norm2_vec, x_norm_vec2, eps); // Input x_vec, output x_norm_vec2
        log_vec_stats("Input to MLP C++ RMSNorm (L" + std::to_string(l) + " P" + std::to_string(pos) + ")", x_norm_vec2);

        // MLP MatVecs
        matvec_bf16_f32_vector(lw.gate_proj, x_norm_vec2, gate_vec, is, hs);
        matvec_bf16_f32_vector(lw.up_proj, x_norm_vec2, up_vec, is, hs);

        // SiLU
        silu(gate_vec, silu_out_vec);

        // SwiGLU
        #pragma omp parallel for
        for(size_t i = 0; i < is; ++i) {
            swiglu_result_vec[i] = silu_out_vec[i] * up_vec[i];
        }
        
        if (log_target_layer) {
            log_vector_summary("TROUBLESHOOTING L0 P1 Gate Proj Result (C++ Vec)", gate_vec);
            log_vector_summary("TROUBLESHOOTING L0 P1 Up Proj Result (C++ Vec)", up_vec);
            log_vector_summary("TROUBLESHOOTING L0 P1 SwiGLU Result (C++ Vec)", swiglu_result_vec);
        }
        
        // Down projection
        matvec_bf16_f32_vector(lw.down_proj, swiglu_result_vec, mlp_out_vec, hs, is);
        
        log_vec_stats("MLP Output (C++ Vec) (L" + std::to_string(l) + " P" + std::to_string(pos) + ")", mlp_out_vec);
        
        // h) Second residual connection
        #pragma omp parallel for
        for(size_t i=0; i<hs; ++i) {
            x_vec[i] = x_resid2_vec[i] + mlp_out_vec[i]; // Update x_vec in-place
        }

        if (log_target_layer) { 
            log_vector_summary("TROUBLESHOOTING L0 P1 State BEFORE 2nd Residual", x_resid2_vec);
            log_vector_summary("TROUBLESHOOTING L0 P1 State AFTER 2nd C++ Residual", x_vec);
            Logger::info("--- TROUBLESHOOTING L0 P1 C++ Layer End ---");
        }
    } // End layer loop

    // 3. Final RMSNorm
    std::vector<float> x_final_norm_input_vec = x_vec; // Copy for logging if needed
    std::vector<float> w_final_norm_vec = bf16vec_to_float_vec(final_norm);
    std::vector<float> x_final_norm_vec(hs);
    rmsnorm_vector(x_vec, w_final_norm_vec, x_final_norm_vec, eps); // Input x_vec, output x_final_norm_vec
    if (log_initial) { 
        log_vector_summary("TROUBLESHOOTING Input to Final RMSNorm (P0)", x_final_norm_input_vec);
        log_vector_summary("TROUBLESHOOTING Output of Final C++ RMSNorm (P0)", x_final_norm_vec); 
    }

    // 4. Output projection to logits
    std::vector<float> logits(vs); 
    matvec_bf16_f32_vector(lm_head, x_final_norm_vec, logits, vs, hs); 
    
#ifdef HAS_CUDA
    Logger::info("[CUDA] Freeing device buffers for RMSNorm");
    gpuErrchk(cudaFree(x_norm_dev));
    gpuErrchk(cudaFree(w_norm1_dev));
#endif

    if (log_initial) {
        log_vector_summary("TROUBLESHOOTING Final Logits (P0, C++ MatVec)", logits, 10);
    }

    if (log_initial) Logger::info("--- TROUBLESHOOTING L0 P0 (C++) END pos=0 ---");
    return logits;
}

// --- Get Vocab Size --- 
int TinyLlamaModel::get_vocab_size() const {
    return config_.vocab_size;
} 