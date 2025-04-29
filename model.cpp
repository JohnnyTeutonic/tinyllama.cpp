#include "model.h"
#include "logger.h"
#include "cuda_kernels.h" // Include CUDA kernel declarations (ALWAYS INCLUDE THIS)
#ifdef HAS_CUDA
// extern declaration removed as it's handled by the header now
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
    softmax_vector_cuda(x, out, x.size());
#else
    if (x.empty()) return;
    out.resize(x.size());
    size_t n = x.size();

    // 1. Find max element for numerical stability
    float max_val = x[0];
    #pragma omp parallel for reduction(max:max_val)
    for (size_t i = 1; i < n; ++i) {
        if (x[i] > max_val) max_val = x[i];
    }

    // 2. Compute exponentials and sum
    float exp_sum = 0.0f;
    #pragma omp parallel for reduction(+:exp_sum)
    for (size_t i = 0; i < n; ++i) {
        out[i] = std::exp(x[i] - max_val);
        exp_sum += out[i];
    }

    // 3. Normalize
    float inv_sum = 1.0f / exp_sum;
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
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

// KVCache initialization method definition
void KVCache::initialize(int num_layers, int max_seq_len, int num_kv_heads, int head_dim) {
    layers.resize(num_layers); // Resize the vector of layers
    seq_len = 0; // Reset sequence length

#ifdef HAS_CUDA
    // --- CUDA Path --- 
    // Store allocation parameters for destructor and indexing
    allocated_num_layers = num_layers;
    allocated_max_seq_len = max_seq_len;
    allocated_num_kv_heads = num_kv_heads;
    allocated_head_dim = head_dim;

    // Calculate the FLAT size needed PER LAYER for K and V caches
    // Layout: [max_seq_len, num_kv_heads, head_dim]
    size_t cache_elems_per_layer = static_cast<size_t>(max_seq_len) * 
                                 static_cast<size_t>(num_kv_heads) * 
                                 static_cast<size_t>(head_dim);
    size_t cache_bytes_per_layer = cache_elems_per_layer * sizeof(float);

    if (cache_elems_per_layer == 0) {
        throw std::runtime_error("KVCache (CUDA): Calculated cache size per layer is zero. Check parameters.");
    }

    Logger::info("Allocating KVCache on GPU: " + std::to_string(num_layers) + " layers, size per layer: " + 
                 std::to_string(cache_bytes_per_layer / (1024.0*1024.0)) + " MB");

    // Allocate device memory for each layer's K and V cache
    for (int l = 0; l < num_layers; ++l) {
        // Free existing memory if re-initializing (should ideally happen only via destructor)
        if (layers[l].k_dev) { Logger::info("Re-initializing KVCache layer K dev pointer without proper destruction?"); gpuErrchk(cudaFree(layers[l].k_dev)); }
        if (layers[l].v_dev) { Logger::info("Re-initializing KVCache layer V dev pointer without proper destruction?"); gpuErrchk(cudaFree(layers[l].v_dev)); }
        
        gpuErrchk(cudaMalloc(&layers[l].k_dev, cache_bytes_per_layer));
        gpuErrchk(cudaMalloc(&layers[l].v_dev, cache_bytes_per_layer));
        // Optional: Zero out the allocated memory (good practice)
        gpuErrchk(cudaMemset(layers[l].k_dev, 0, cache_bytes_per_layer));
        gpuErrchk(cudaMemset(layers[l].v_dev, 0, cache_bytes_per_layer));
    }
    Logger::info("KVCache GPU allocation complete.");

#else 
    // --- CPU Path --- 
    // Calculate the size needed for each layer's K and V cache (host vectors)
    size_t cache_size_per_layer = static_cast<size_t>(max_seq_len) * 
                                 static_cast<size_t>(num_kv_heads) * 
                                 static_cast<size_t>(head_dim);

    if (cache_size_per_layer == 0) {
        throw std::runtime_error("KVCache (CPU): Calculated cache size is zero. Check parameters.");
    }
    
    // Allocate and zero-initialize all host cache vectors
    for (int l = 0; l < num_layers; ++l) {
        layers[l].k.assign(cache_size_per_layer, 0.0f); // Use assign for resize + fill
        layers[l].v.assign(cache_size_per_layer, 0.0f);
    }
     Logger::info("KVCache (CPU) initialized with dimensions: " +
                   std::to_string(num_layers) + " layers, " +
                   std::to_string(max_seq_len) + " seq len, " +
                   std::to_string(num_kv_heads) + " KV heads, " +
                   std::to_string(head_dim) + " head dim");
#endif
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
    // Create a temporary handle for this static function context
    cublasHandle_t temp_handle;
    cublasStatus_t create_status = cublasCreate(&temp_handle);
    if (create_status != CUBLAS_STATUS_SUCCESS) {
        Logger::error("Temporary cuBLAS handle creation failed in matvec_bf16_f32_vector");
        throw std::runtime_error("Failed to create temp cuBLAS handle");
    }
    matvec_bf16_f32_cuda(temp_handle, mat_bf16, vec_f32, out_f32, rows, cols);
    // Destroy the temporary handle
    cublasStatus_t destroy_status = cublasDestroy(temp_handle);
    if (destroy_status != CUBLAS_STATUS_SUCCESS) {
         Logger::error("Temporary cuBLAS handle destruction failed in matvec_bf16_f32_vector");
         // Log but continue
    }
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

#ifdef HAS_CUDA
    // --- START CUBLAS Initialization ---
    cublasStatus_t cublas_status = cublasCreate(&cublas_handle_);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        Logger::error("cuBLAS handle creation failed with error code: " + std::to_string(cublas_status));
        // Handle error appropriately - maybe throw an exception or set an error state
        throw std::runtime_error("Failed to initialize cuBLAS");
    }
    Logger::info("cuBLAS handle created successfully.");
    // --- END CUBLAS Initialization ---

    // Allocate and copy final_norm weights to device
    std::vector<float> final_norm_f32 = bf16vec_to_float_vec(final_norm);
    gpuErrchk(cudaMalloc(&final_norm_dev, final_norm_f32.size() * sizeof(float)));
    gpuErrchk(cudaMemcpy(final_norm_dev, final_norm_f32.data(), final_norm_f32.size() * sizeof(float), cudaMemcpyHostToDevice));
    Logger::info("Copied final_norm weights to GPU.");

    // --- START: Allocate and copy persistent BF16 weights ---
    // Embedding Table
    gpuErrchk(cudaMalloc(&token_embedding_table_dev_, embed_tokens.size() * sizeof(uint16_t)));
    gpuErrchk(cudaMemcpy(token_embedding_table_dev_, embed_tokens.data(), embed_tokens.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
    Logger::info("Copied token_embedding_table (bf16) to GPU.");

    // LM Head
    gpuErrchk(cudaMalloc(&lm_head_dev_, lm_head.size() * sizeof(uint16_t)));
    gpuErrchk(cudaMemcpy(lm_head_dev_, lm_head.data(), lm_head.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
    Logger::info("Copied lm_head (bf16) to GPU.");

#endif

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

#ifdef HAS_CUDA
            // Allocate and copy layer norm weights to device
            std::vector<float> input_ln_f32 = bf16vec_to_float_vec(lw.input_layernorm);
            gpuErrchk(cudaMalloc(&lw.input_layernorm_dev, input_ln_f32.size() * sizeof(float)));
            gpuErrchk(cudaMemcpy(lw.input_layernorm_dev, input_ln_f32.data(), input_ln_f32.size() * sizeof(float), cudaMemcpyHostToDevice));

            std::vector<float> post_attn_ln_f32 = bf16vec_to_float_vec(lw.post_attention_layernorm);
            gpuErrchk(cudaMalloc(&lw.post_attention_layernorm_dev, post_attn_ln_f32.size() * sizeof(float)));
            gpuErrchk(cudaMemcpy(lw.post_attention_layernorm_dev, post_attn_ln_f32.data(), post_attn_ln_f32.size() * sizeof(float), cudaMemcpyHostToDevice));
#endif

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
    // Layer Weights (Concatenate on Host first, then copy)
    size_t layer_q_size = (size_t)hs * hs;
    size_t layer_k_size = (size_t)kv_dim * hs;
    size_t layer_v_size = (size_t)kv_dim * hs;
    size_t layer_o_size = (size_t)hs * hs;
    size_t layer_gate_size = (size_t)is * hs;
    size_t layer_up_size = (size_t)is * hs;
    size_t layer_down_size = (size_t)hs * is;

    std::vector<uint16_t> all_q_proj_host, all_k_proj_host, all_v_proj_host, all_o_proj_host;
    std::vector<uint16_t> all_gate_proj_host, all_up_proj_host, all_down_proj_host;

    all_q_proj_host.reserve(nhl * layer_q_size);
    all_k_proj_host.reserve(nhl * layer_k_size);
    all_v_proj_host.reserve(nhl * layer_v_size);
    all_o_proj_host.reserve(nhl * layer_o_size);
    all_gate_proj_host.reserve(nhl * layer_gate_size);
    all_up_proj_host.reserve(nhl * layer_up_size);
    all_down_proj_host.reserve(nhl * layer_down_size);

    for (int i = 0; i < nhl; ++i) {
        const auto& lw = layers[i];
        all_q_proj_host.insert(all_q_proj_host.end(), lw.q_proj.begin(), lw.q_proj.end());
        all_k_proj_host.insert(all_k_proj_host.end(), lw.k_proj.begin(), lw.k_proj.end());
        all_v_proj_host.insert(all_v_proj_host.end(), lw.v_proj.begin(), lw.v_proj.end());
        all_o_proj_host.insert(all_o_proj_host.end(), lw.o_proj.begin(), lw.o_proj.end());
        all_gate_proj_host.insert(all_gate_proj_host.end(), lw.gate_proj.begin(), lw.gate_proj.end());
        all_up_proj_host.insert(all_up_proj_host.end(), lw.up_proj.begin(), lw.up_proj.end());
        all_down_proj_host.insert(all_down_proj_host.end(), lw.down_proj.begin(), lw.down_proj.end());
    }
    Logger::info("Concatenated all layer weights on host.");

    // Allocate device memory for concatenated weights
    gpuErrchk(cudaMalloc(&w_q_dev_, all_q_proj_host.size() * sizeof(uint16_t)));
    gpuErrchk(cudaMalloc(&w_k_dev_, all_k_proj_host.size() * sizeof(uint16_t)));
    gpuErrchk(cudaMalloc(&w_v_dev_, all_v_proj_host.size() * sizeof(uint16_t)));
    gpuErrchk(cudaMalloc(&w_o_dev_, all_o_proj_host.size() * sizeof(uint16_t)));
    gpuErrchk(cudaMalloc(&w_gate_dev_, all_gate_proj_host.size() * sizeof(uint16_t)));
    gpuErrchk(cudaMalloc(&w_up_dev_, all_up_proj_host.size() * sizeof(uint16_t)));
    gpuErrchk(cudaMalloc(&w_down_dev_, all_down_proj_host.size() * sizeof(uint16_t)));
    Logger::info("Allocated GPU memory for concatenated layer weights.");

    // Copy concatenated weights to device
    gpuErrchk(cudaMemcpy(w_q_dev_, all_q_proj_host.data(), all_q_proj_host.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(w_k_dev_, all_k_proj_host.data(), all_k_proj_host.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(w_v_dev_, all_v_proj_host.data(), all_v_proj_host.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(w_o_dev_, all_o_proj_host.data(), all_o_proj_host.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(w_gate_dev_, all_gate_proj_host.data(), all_gate_proj_host.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(w_up_dev_, all_up_proj_host.data(), all_up_proj_host.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(w_down_dev_, all_down_proj_host.data(), all_down_proj_host.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
    Logger::info("Copied concatenated layer weights (bf16) to GPU.");
    // --- END: Allocate and copy persistent BF16 weights ---
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

#ifdef HAS_CUDA
    // Allocate persistent buffer for all RoPE frequencies on device
    if (!precomputed_freqs_cis_.empty()) { // Check if host vector is populated
        size_t total_freq_elements = precomputed_freqs_cis_.size() * 2; // 2 floats per pair (cos, sin)
        gpuErrchk(cudaMalloc(&all_freqs_cis_dev, total_freq_elements * sizeof(float)));
        Logger::info("Allocated persistent RoPE frequency buffer on GPU: " + 
                     std::to_string(total_freq_elements * sizeof(float) / 1024.0) + " KB");

        // Flatten host data and copy to device
        std::vector<float> flat_host_freqs;
        flat_host_freqs.reserve(total_freq_elements);
        for (const auto& p : precomputed_freqs_cis_) {
            flat_host_freqs.push_back(p.first);  // cos
            flat_host_freqs.push_back(p.second); // sin
        }
        gpuErrchk(cudaMemcpy(all_freqs_cis_dev, flat_host_freqs.data(), total_freq_elements * sizeof(float), cudaMemcpyHostToDevice));
        Logger::info("Copied all precomputed RoPE frequencies to persistent GPU buffer.");

    } else {
        Logger::info("Host precomputed_freqs_cis_ is empty, skipping GPU RoPE buffer allocation.");
    }
#endif
}

// --- START: TinyLlamaModel Destructor Definition ---
TinyLlamaModel::~TinyLlamaModel() {
#ifdef HAS_CUDA
    Logger::info("Freeing TinyLlamaModel CUDA resources...");
    // --- START CUBLAS Destruction ---
    if (cublas_handle_) {
        cublasStatus_t cublas_status = cublasDestroy(cublas_handle_);
        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
            Logger::error("cuBLAS handle destruction failed with error code: " + std::to_string(cublas_status));
            // Log error, but don't throw from destructor
        }
        cublas_handle_ = nullptr;
        Logger::info("cuBLAS handle destroyed.");
    }
    // --- END CUBLAS Destruction ---

    // Free final norm device pointer
    if (final_norm_dev) {
        gpuErrchk(cudaFree(final_norm_dev));
        final_norm_dev = nullptr; // Good practice to null pointers after freeing
    }
    // Free layer norm device pointers
    for (auto& layer : layers) { // Iterate through the layers vector
        if (layer.input_layernorm_dev) {
            gpuErrchk(cudaFree(layer.input_layernorm_dev));
            layer.input_layernorm_dev = nullptr;
        }
        if (layer.post_attention_layernorm_dev) {
            gpuErrchk(cudaFree(layer.post_attention_layernorm_dev));
            layer.post_attention_layernorm_dev = nullptr;
        }
    }
    // Free RoPE frequencies device pointer
    if (all_freqs_cis_dev) {
        gpuErrchk(cudaFree(all_freqs_cis_dev));
        all_freqs_cis_dev = nullptr;
    }
    // --- START: Free persistent BF16 weights ---
    if (token_embedding_table_dev_) {
        gpuErrchk(cudaFree(token_embedding_table_dev_));
        token_embedding_table_dev_ = nullptr;
    }
    if (lm_head_dev_) {
        gpuErrchk(cudaFree(lm_head_dev_));
        lm_head_dev_ = nullptr;
    }
    if (w_q_dev_) {
        gpuErrchk(cudaFree(w_q_dev_));
        w_q_dev_ = nullptr;
    }
    if (w_k_dev_) {
        gpuErrchk(cudaFree(w_k_dev_));
        w_k_dev_ = nullptr;
    }
    if (w_v_dev_) {
        gpuErrchk(cudaFree(w_v_dev_));
        w_v_dev_ = nullptr;
    }
    if (w_o_dev_) {
        gpuErrchk(cudaFree(w_o_dev_));
        w_o_dev_ = nullptr;
    }
    if (w_gate_dev_) {
        gpuErrchk(cudaFree(w_gate_dev_));
        w_gate_dev_ = nullptr;
    }
    if (w_up_dev_) {
        gpuErrchk(cudaFree(w_up_dev_));
        w_up_dev_ = nullptr;
    }
    if (w_down_dev_) {
        gpuErrchk(cudaFree(w_down_dev_));
        w_down_dev_ = nullptr;
    }
    // --- END: Free persistent BF16 weights ---

    Logger::info("Finished freeing TinyLlamaModel CUDA weight memory.");
#endif
} 
// --- END: TinyLlamaModel Destructor Definition ---
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

// --- Forward Pass (Restructured) --- 
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

    // --- Basic Input Validation ---
    bool log_initial = (pos == 0);
    if (pos >= max_seq_len) {
        Logger::error("Position index exceeds max_position_embeddings");
        return std::vector<float>(vs, 0.0f); 
    }
    if (!cache) {
        Logger::error("KVCache is required for token-by-token forward pass");
        return std::vector<float>(vs, 0.0f);
    }
    if (x_vec.size() != hs) {
        Logger::error("Input vector x_vec has incorrect size. Expected " + std::to_string(hs) + ", got " + std::to_string(x_vec.size()));
        return std::vector<float>(vs, 0.0f);
    }

#ifdef HAS_CUDA
    // --- CUDA Path Setup ---
    // Logger::info("[CUDA] Using CUDA path in forward()");
    
    // Allocate ALL necessary device buffers ONCE outside the loop
    float* x_dev = nullptr;
    float* x_norm_dev = nullptr;
    float* x_resid1_dev = nullptr;
    float* x_resid2_dev = nullptr;
    float* q_dev = nullptr;
    float* k_dev = nullptr;
    float* v_dev = nullptr;
    float* attn_out_dev = nullptr;
    float* attn_proj_dev = nullptr;
    float* gate_vec_dev = nullptr;
    float* up_vec_dev = nullptr;
    float* swiglu_vec_dev = nullptr;
    float* mlp_down_dev = nullptr;
    float* freqs_dev = nullptr; // For RoPE frequencies

    gpuErrchk(cudaMalloc(&x_dev, hs * sizeof(float)));
    gpuErrchk(cudaMalloc(&x_norm_dev, hs * sizeof(float)));
    gpuErrchk(cudaMalloc(&x_resid1_dev, hs * sizeof(float)));
    gpuErrchk(cudaMalloc(&x_resid2_dev, hs * sizeof(float)));
    gpuErrchk(cudaMalloc(&q_dev, hs * sizeof(float)));
    gpuErrchk(cudaMalloc(&k_dev, n_kv_heads * head_dim * sizeof(float)));
    gpuErrchk(cudaMalloc(&v_dev, n_kv_heads * head_dim * sizeof(float)));
    gpuErrchk(cudaMalloc(&attn_out_dev, hs * sizeof(float)));
    gpuErrchk(cudaMalloc(&attn_proj_dev, hs * sizeof(float)));
    gpuErrchk(cudaMalloc(&gate_vec_dev, is * sizeof(float)));
    gpuErrchk(cudaMalloc(&up_vec_dev, is * sizeof(float)));
    gpuErrchk(cudaMalloc(&swiglu_vec_dev, is * sizeof(float)));
    gpuErrchk(cudaMalloc(&mlp_down_dev, hs * sizeof(float)));
    // Note: freqs_dev allocated inside loop as size depends on pos

    // Copy initial embedding to device
    gpuErrchk(cudaMemcpy(x_dev, x_vec.data(), hs * sizeof(float), cudaMemcpyHostToDevice));

    // --- Layer Loop (CUDA Path) ---
    for (int l = 0; l < nhl; ++l) {
        const auto& lw = layers[l];

        // Residual 1
        gpuErrchk(cudaMemcpy(x_resid1_dev, x_dev, hs * sizeof(float), cudaMemcpyDeviceToDevice));

        // RMSNorm 1 (Input: x_dev, Output: x_norm_dev)
        // float* w_norm1_dev = nullptr; // Declare device pointer - COMMENTED OUT
        std::vector<float> w_norm1_vec = bf16vec_to_float_vec(lw.input_layernorm); // Keep conversion for now (though unused)
        // gpuErrchk(cudaMalloc(&w_norm1_dev, hs * sizeof(float))); // COMMENTED OUT
        // gpuErrchk(cudaMemcpy(w_norm1_dev, w_norm1_vec.data(), hs * sizeof(float), cudaMemcpyHostToDevice)); // COMMENTED OUT
        rmsnorm_vector_cuda(x_dev, lw.input_layernorm_dev, x_norm_dev, hs, eps); // USES PERSISTENT POINTER - UNCHANGED
        // gpuErrchk(cudaFree(w_norm1_dev)); // COMMENTED OUT
        gpuErrchk(cudaDeviceSynchronize());

        // Q, K, V projections (Input: x_norm_dev, Outputs: q_dev, k_dev, v_dev)
        matvec_bf16_f32_cuda(cublas_handle_, lw.q_proj, x_norm_dev, q_dev, hs, hs); // Pass handle
        matvec_bf16_f32_cuda(cublas_handle_, lw.k_proj, x_norm_dev, k_dev, n_kv_heads * head_dim, hs); // Pass handle
        matvec_bf16_f32_cuda(cublas_handle_, lw.v_proj, x_norm_dev, v_dev, n_kv_heads * head_dim, hs); // Pass handle
        gpuErrchk(cudaDeviceSynchronize());

        // RoPE (Applied in-place to q_dev, k_dev)
        size_t freqs_offset = (pos * head_dim / 2);
        std::vector<std::pair<float, float>> current_freqs_cis(precomputed_freqs_cis_.begin() + freqs_offset, 
                                                               precomputed_freqs_cis_.begin() + freqs_offset + head_dim / 2);
        std::vector<float> flat_freqs; // Host vector
        flat_freqs.reserve(current_freqs_cis.size() * 2);
        for (const auto& p : current_freqs_cis) {
            flat_freqs.push_back(p.first);
            flat_freqs.push_back(p.second);
        }
        gpuErrchk(cudaMalloc(&freqs_dev, flat_freqs.size() * sizeof(float))); // Allocate freqs_dev
        gpuErrchk(cudaMemcpy(freqs_dev, flat_freqs.data(), flat_freqs.size() * sizeof(float), cudaMemcpyHostToDevice));
        
        // Call device-pointer RoPE version directly using persistent buffer and pos
        rope_cuda(q_dev, n_heads, head_dim, all_freqs_cis_dev, pos); // UPDATED CALL in forward()
        rope_cuda(k_dev, n_kv_heads, head_dim, all_freqs_cis_dev, pos); // UPDATED CALL in forward()
        
        gpuErrchk(cudaFree(freqs_dev)); // Free freqs_dev for this layer
        gpuErrchk(cudaDeviceSynchronize());

        // KVCache Update (Directly on Device)
        // Call the new kernel for each K/V head
        for (int kvh = 0; kvh < n_kv_heads; ++kvh) {
            // Calculate pointer to the *start* of the data for this specific head in k_dev/v_dev
            const float* current_k_head_ptr = k_dev + kvh * head_dim;
            const float* current_v_head_ptr = v_dev + kvh * head_dim;
            
            // Call kernel to update K cache for this head
            update_kv_cache_cuda(cache->layers[l].k_dev, // Base pointer for layer K cache
                                 current_k_head_ptr,     // Pointer to current K data for this head
                                 pos, 
                                 kvh, 
                                 cache->allocated_max_seq_len,
                                 cache->allocated_num_kv_heads,
                                 cache->allocated_head_dim);
                                 
            // Call kernel to update V cache for this head
            update_kv_cache_cuda(cache->layers[l].v_dev, // Base pointer for layer V cache
                                 current_v_head_ptr,     // Pointer to current V data for this head
                                 pos, 
                                 kvh, 
                                 cache->allocated_max_seq_len,
                                 cache->allocated_num_kv_heads,
                                 cache->allocated_head_dim);
        }
        // No need to copy K/V back to host here anymore
        // delete[] k_current_ptr_host;
        // delete[] v_current_ptr_host;

        // --- Attention (Reads directly from Device Cache) --- 
        int current_seq_len = pos + 1;
        float scale = 1.0f / std::sqrt(head_dim);

        // REMOVE preparation of K_cache_dev / V_cache_dev host/device buffers
        // float* K_cache_dev = nullptr;
        // float* V_cache_dev = nullptr;
        // ... (remove allocation and H->D copy code) ...

        // Call Attention Kernel directly with layer cache base pointers
        // Input Q: q_dev (current token's Q, shape [hs = n_heads * head_dim])
        // Input K: cache->layers[l].k_dev (base pointer for layer K cache)
        // Input V: cache->layers[l].v_dev (base pointer for layer V cache)
        // Output: attn_out_dev (shape [hs])
        attention_cuda(q_dev,                      // Q vector for current token
                       cache->layers[l].k_dev,   // Base K cache pointer for this layer
                       cache->layers[l].v_dev,   // Base V cache pointer for this layer
                       attn_out_dev,             // Output attention result
                       n_heads,                  // Number of Q heads
                       current_seq_len,          // Current sequence length
                       head_dim,                 // Head dimension
                       scale,                    // Scaling factor
                       cache->allocated_max_seq_len, // Cache max sequence length
                       cache->allocated_num_kv_heads // Cache K/V head count
                       /*, stream */);          // Optional stream omitted
        gpuErrchk(cudaDeviceSynchronize());

        // Output projection (Input: attn_out_dev, Output: attn_proj_dev)
        matvec_bf16_f32_cuda(cublas_handle_, lw.o_proj, attn_out_dev, attn_proj_dev, hs, hs); // Pass handle
        gpuErrchk(cudaDeviceSynchronize());

        // First residual connection (x = x_resid1 + attn_proj) - FUSED KERNEL
        add_residual_cuda(attn_proj_dev, x_resid1_dev, x_dev, hs);
        gpuErrchk(cudaDeviceSynchronize());

        // --- MLP Block --- 
        // Residual 2 Prep (Copy x_dev to x_resid2_dev)
        // x_resid2_dev needs declaration in this scope
        float* x_resid2_dev = nullptr;
        gpuErrchk(cudaMalloc(&x_resid2_dev, hs * sizeof(float))); // Allocate it
        gpuErrchk(cudaMemcpy(x_resid2_dev, x_dev, hs * sizeof(float), cudaMemcpyDeviceToDevice)); 

        // Post-attention RMSNorm
        // float* w_norm2_dev = nullptr; // Keep alloc/free for now, just change the source ptr - COMMENTED OUT
        std::vector<float> w_norm2_vec = bf16vec_to_float_vec(lw.post_attention_layernorm);
        // gpuErrchk(cudaMalloc(&w_norm2_dev, hs * sizeof(float))); // COMMENTED OUT
        // gpuErrchk(cudaMemcpy(w_norm2_dev, w_norm2_vec.data(), hs * sizeof(float), cudaMemcpyHostToDevice)); // COMMENTED OUT
        rmsnorm_vector_cuda(x_dev, lw.post_attention_layernorm_dev, x_norm_dev, hs, eps); // CHANGED: Use persistent ptr lw.post_attention_layernorm_dev - UNCHANGED
        // gpuErrchk(cudaFree(w_norm2_dev)); // COMMENTED OUT
        gpuErrchk(cudaDeviceSynchronize()); // Keep sync for now

        // MLP MatVecs & SwiGLU
        // Declare device pointers needed for MLP within this scope
        float* gate_vec_dev = nullptr;
        float* up_vec_dev = nullptr;
        float* swiglu_vec_dev = nullptr;
        float* mlp_down_dev = nullptr;
        gpuErrchk(cudaMalloc(&gate_vec_dev, is * sizeof(float)));
        gpuErrchk(cudaMalloc(&up_vec_dev, is * sizeof(float)));
        gpuErrchk(cudaMalloc(&swiglu_vec_dev, is * sizeof(float)));
        gpuErrchk(cudaMalloc(&mlp_down_dev, hs * sizeof(float)));
        // Use x_norm_dev as input
        matvec_bf16_f32_cuda(cublas_handle_, lw.gate_proj, x_norm_dev, gate_vec_dev, is, hs); // Pass handle
        matvec_bf16_f32_cuda(cublas_handle_, lw.up_proj, x_norm_dev, up_vec_dev, is, hs); // Pass handle
        gpuErrchk(cudaDeviceSynchronize());
        swiglu_cuda(gate_vec_dev, up_vec_dev, swiglu_vec_dev, is);
        gpuErrchk(cudaDeviceSynchronize());
        matvec_bf16_f32_cuda(cublas_handle_, lw.down_proj, swiglu_vec_dev, mlp_down_dev, hs, is); // Pass handle
        gpuErrchk(cudaDeviceSynchronize());
        
        // Add residual 2 (FUSED KERNEL)
        add_residual_cuda(mlp_down_dev, x_resid2_dev, x_dev, hs);
        gpuErrchk(cudaDeviceSynchronize());
        
        // Free CUDA MLP intermediates
        gpuErrchk(cudaFree(gate_vec_dev));
        gpuErrchk(cudaFree(up_vec_dev));
        gpuErrchk(cudaFree(swiglu_vec_dev));
        gpuErrchk(cudaFree(mlp_down_dev));
        gpuErrchk(cudaFree(x_resid2_dev)); // Free residual buffer

    } // End layer loop

    // --- Final Steps (Outside Layer Loop) ---
    // Final RMSNorm 
    float* w_final_norm_dev = nullptr; // Keep alloc/free for now, just change the source ptr
    std::vector<float> w_final_norm_vec = bf16vec_to_float_vec(final_norm);
    gpuErrchk(cudaMalloc(&w_final_norm_dev, hs * sizeof(float)));
    gpuErrchk(cudaMemcpy(w_final_norm_dev, w_final_norm_vec.data(), hs * sizeof(float), cudaMemcpyHostToDevice));
    // Use x_norm_dev declared at the top of the function
    rmsnorm_vector_cuda(x_dev, w_final_norm_dev, x_norm_dev, hs, eps);
    gpuErrchk(cudaFree(w_final_norm_dev));
    gpuErrchk(cudaDeviceSynchronize());

    // Final LM Head Projection 
    float* logits_dev = nullptr;
    gpuErrchk(cudaMalloc(&logits_dev, vs * sizeof(float)));
    // Use x_norm_dev as input
    matvec_bf16_f32_cuda(cublas_handle_, lm_head, x_norm_dev, logits_dev, vs, hs); // Pass handle
    gpuErrchk(cudaDeviceSynchronize());

    // Copy final logits from device to host
    std::vector<float> logits(vs); 
    gpuErrchk(cudaMemcpy(logits.data(), logits_dev, vs * sizeof(float), cudaMemcpyDeviceToHost));

    // Free ALL CUDA Buffers allocated at the start
    gpuErrchk(cudaFree(x_dev));
    gpuErrchk(cudaFree(x_norm_dev)); 
    // (Other device pointers freed inside loop or here)
    gpuErrchk(cudaFree(logits_dev));

    return logits;

#else 
    // --- CPU Path Setup ---
    // Logger::info("[CPU] Using CPU path in forward()");

    // Intermediate vectors needed within the loop (CPU Path)
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

    // --- Layer Loop (CPU Path) ---
    for (int l = 0; l < nhl; ++l) {
        const auto& lw = layers[l];
        std::vector<float> x_resid1_vec = x_vec; // Residual 1

        // RMSNorm 1
        std::vector<float> w_norm1_vec = bf16vec_to_float_vec(lw.input_layernorm); 
        rmsnorm_vector(x_vec, w_norm1_vec, x_norm_vec1, eps); 

        // Q, K, V projections 
        matvec_bf16_f32_vector(lw.q_proj, x_norm_vec1, q_vec, hs, hs);
        matvec_bf16_f32_vector(lw.k_proj, x_norm_vec1, k_vec, n_kv_heads * head_dim, hs);
        matvec_bf16_f32_vector(lw.v_proj, x_norm_vec1, v_vec, n_kv_heads * head_dim, hs);

        // RoPE 
        size_t freqs_offset = (pos * head_dim / 2);
        std::vector<std::pair<float, float>> current_freqs_cis(precomputed_freqs_cis_.begin() + freqs_offset, 
                                                               precomputed_freqs_cis_.begin() + freqs_offset + head_dim / 2);
        apply_rope_vector(q_vec, n_heads, head_dim, pos, current_freqs_cis);
        apply_rope_vector(k_vec, n_kv_heads, head_dim, pos, current_freqs_cis);
        
        // KVCache Update
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
                 Logger::error("KVCache write out of bounds: layer=" + std::to_string(l) + ", pos=" + std::to_string(pos) + ", kv_head=" + std::to_string(kvh));
             }
         }

        // Attention 
        std::fill(attn_out_vec.begin(), attn_out_vec.end(), 0.0f); 
        int current_seq_len = pos + 1;
        float scale = 1.0f / std::sqrt(head_dim);
        for (int h = 0; h < n_heads; ++h) {
            std::vector<float> q_head_rope_vec(q_vec.begin() + h * head_dim, q_vec.begin() + (h + 1) * head_dim);
            int kv_head_idx = h / (n_heads / n_kv_heads);
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
            std::vector<float> scores_vec(current_seq_len);
            calculate_attention_scores(q_head_rope_vec, k_cache_head_vec, scores_vec, current_seq_len, head_dim, scale);
            std::vector<float> probs_vec(current_seq_len);
            softmax_vector(scores_vec, probs_vec);
            std::vector<float> head_attn_out_vec(head_dim);
            weighted_sum_probs_v(probs_vec, v_cache_head_vec, head_attn_out_vec, current_seq_len, head_dim);
            size_t out_offset = h * head_dim;
            for(int i=0; i<head_dim; ++i) {
                attn_out_vec[out_offset + i] += head_attn_out_vec[i];
            }
        }
        matvec_bf16_f32_vector(lw.o_proj, attn_out_vec, attn_proj_vec, hs, hs);
        #pragma omp parallel for
        for(size_t i=0; i<hs; ++i) {
            x_vec[i] = x_resid1_vec[i] + attn_proj_vec[i]; 
        }
        
        // --- MLP Block --- 
        std::vector<float> x_resid2_vec = x_vec; 
        std::vector<float> w_norm2_vec = bf16vec_to_float_vec(lw.post_attention_layernorm);
        rmsnorm_vector(x_vec, w_norm2_vec, x_norm_vec2, eps);
        matvec_bf16_f32_vector(cublas_handle_, lw.gate_proj, x_norm_vec2, gate_vec, is); // Pass handle
        matvec_bf16_f32_vector(cublas_handle_, lw.up_proj, x_norm_vec2, up_vec, is); // Pass handle
        silu(gate_vec, silu_out_vec);
        #pragma omp parallel for
        for(size_t i = 0; i < is; ++i) {
            swiglu_result_vec[i] = silu_out_vec[i] * up_vec[i];
        }
        matvec_bf16_f32_vector(cublas_handle_, lw.down_proj, swiglu_result_vec, mlp_out_vec, hs); // Pass handle
        #pragma omp parallel for
        for(size_t i=0; i<hs; ++i) {
            x_vec[i] = x_resid2_vec[i] + mlp_out_vec[i]; 
        }

        // Optional logging for CPU path (can be added similarly to CUDA path if needed)
        // if (log_target_layer) { ... }

    } // End layer loop

    // --- Final Steps (Outside Layer Loop) ---
    std::vector<float> w_final_norm_vec = bf16vec_to_float_vec(final_norm);
    std::vector<float> x_final_norm_vec(hs);
    rmsnorm_vector(x_vec, w_final_norm_vec, x_final_norm_vec, eps);
    if (log_initial) { 
        log_vector_summary("TROUBLESHOOTING Output of Final C++ RMSNorm (P0, CPU)", x_final_norm_vec); 
    }
    std::vector<float> logits(vs); 
    matvec_bf16_f32_vector(cublas_handle_, lm_head, x_final_norm_vec, logits, vs); // Pass handle
    if (log_initial) {
        log_vector_summary("TROUBLESHOOTING Final Logits (P0, C++ MatVec, CPU Path)", logits, 10);
    }
    return logits;

#endif // This should be the main #endif for the function body

} // End of forward function

// --- Get Vocab Size --- 
int TinyLlamaModel::get_vocab_size() const {
    return config_.vocab_size;
}

// --- Forward Pass (Device Implementation) --- 
// Restore the #ifdef HAS_CUDA guard around the entire function definition
#ifdef HAS_CUDA
std::vector<float> TinyLlamaModel::forward_device(int token_id, int pos, KVCache* cache, const std::vector<int>* attention_mask, cudaStream_t stream) {
    // Logger::info("[CUDA] forward_device called for token: " + std::to_string(token_id) + " pos: " + std::to_string(pos));
    int hs = config_.hidden_size;
    int vs = config_.vocab_size;
    int n_heads = config_.num_attention_heads;
    int n_kv_heads = config_.num_key_value_heads;
    int head_dim = hs / n_heads;
    int nhl = config_.num_hidden_layers;
    int is = config_.intermediate_size;
    float eps = config_.rms_norm_eps;
    int max_seq_len = config_.max_position_embeddings; // Added for cache size

    // --- Device Memory Allocation --- 
    float* x_dev = nullptr;
    float* x_norm_dev = nullptr;
    float* x_resid1_dev = nullptr;
    float* x_resid2_dev = nullptr;
    float* q_dev = nullptr;
    float* k_dev = nullptr;
    float* v_dev = nullptr;
    float* attn_out_dev = nullptr;
    float* attn_proj_dev = nullptr;
    float* gate_vec_dev = nullptr;
    float* up_vec_dev = nullptr;
    float* swiglu_vec_dev = nullptr;
    float* mlp_out_dev = nullptr;
    float* logits_dev = nullptr;
    float* freqs_dev = nullptr; // For RoPE frequencies

    // --- Set cuBLAS Stream --- 
    cublasStatus_t stream_status = cublasSetStream(cublas_handle_, stream);
    if (stream_status != CUBLAS_STATUS_SUCCESS) {
        Logger::error("cublasSetStream failed in forward_device");
        // Handle error, maybe throw or return empty vector
        return {}; 
    }

    // Allocate device buffers using async allocation on the specified stream
    gpuErrchk(cudaMallocAsync(&x_dev, hs * sizeof(float), stream));
    gpuErrchk(cudaMallocAsync(&x_norm_dev, hs * sizeof(float), stream));
    gpuErrchk(cudaMallocAsync(&x_resid1_dev, hs * sizeof(float), stream));
    gpuErrchk(cudaMallocAsync(&x_resid2_dev, hs * sizeof(float), stream));
    gpuErrchk(cudaMallocAsync(&q_dev, hs * sizeof(float), stream));
    gpuErrchk(cudaMallocAsync(&k_dev, n_kv_heads * head_dim * sizeof(float), stream));
    gpuErrchk(cudaMallocAsync(&v_dev, n_kv_heads * head_dim * sizeof(float), stream));
    gpuErrchk(cudaMallocAsync(&attn_out_dev, hs * sizeof(float), stream));
    // gpuErrchk(cudaMallocAsync(&attn_proj_dev, hs * sizeof(float), stream)); // Use add_residual directly writing to x_dev
    gpuErrchk(cudaMallocAsync(&gate_vec_dev, is * sizeof(float), stream));
    gpuErrchk(cudaMallocAsync(&up_vec_dev, is * sizeof(float), stream));
    gpuErrchk(cudaMallocAsync(&swiglu_vec_dev, is * sizeof(float), stream));
    // gpuErrchk(cudaMallocAsync(&mlp_out_dev, hs * sizeof(float), stream)); // Use add_residual directly writing to x_dev
    gpuErrchk(cudaMallocAsync(&logits_dev, vs * sizeof(float), stream));

    // --- Initial Embedding Lookup & Copy --- 
    lookup_embedding_bf16_f32_cuda(
        token_embedding_table_dev_, // Use persistent device pointer
        x_dev,                      // Destination: FP32 state vector on GPU
        token_id,
        hs,
        vs                          // <-- PASS vocab_size
        , stream                    // Pass stream
    );

    // --- Layer Loop --- 
    for (int l = 0; l < nhl; ++l) {
        // Layer weights device pointers (calculated offsets into persistent buffers)
        size_t layer_q_size    = (size_t)hs * hs;
        size_t layer_k_size    = (size_t)n_kv_heads * head_dim * hs; // kv_dim * hs
        size_t layer_v_size    = (size_t)n_kv_heads * head_dim * hs; // kv_dim * hs
        size_t layer_o_size    = (size_t)hs * hs;
        size_t layer_gate_size = (size_t)is * hs;
        size_t layer_up_size   = (size_t)is * hs;
        size_t layer_down_size = (size_t)hs * is;

        const uint16_t* lw_q_proj_dev    = w_q_dev_    + (size_t)l * layer_q_size;
        const uint16_t* lw_k_proj_dev    = w_k_dev_    + (size_t)l * layer_k_size;
        const uint16_t* lw_v_proj_dev    = w_v_dev_    + (size_t)l * layer_v_size;
        const uint16_t* lw_o_proj_dev    = w_o_dev_    + (size_t)l * layer_o_size;
        const uint16_t* lw_gate_proj_dev = w_gate_dev_ + (size_t)l * layer_gate_size;
        const uint16_t* lw_up_proj_dev   = w_up_dev_   + (size_t)l * layer_up_size;
        const uint16_t* lw_down_proj_dev = w_down_dev_ + (size_t)l * layer_down_size;
        const float*    lw_in_norm_dev   = layers[l].input_layernorm_dev;
        const float*    lw_post_norm_dev = layers[l].post_attention_layernorm_dev;

        // Residual 1 Prep (Copy x -> x_resid1)
        gpuErrchk(cudaMemcpyAsync(x_resid1_dev, x_dev, hs * sizeof(float), cudaMemcpyDeviceToDevice, stream));

        // RMSNorm 1 (Input: x_dev, Weight: lw_in_norm_dev, Output: x_norm_dev)
        rmsnorm_vector_cuda(x_dev, lw_in_norm_dev, x_norm_dev, hs, eps, stream); 

        // Q, K, V projections (Input: x_norm_dev, Outputs: q_dev, k_dev, v_dev)
        matvec_bf16_f32_cuda(cublas_handle_, lw_q_proj_dev, x_norm_dev, q_dev, hs, hs, stream); 
        matvec_bf16_f32_cuda(cublas_handle_, lw_k_proj_dev, x_norm_dev, k_dev, n_kv_heads * head_dim, hs, stream); 
        matvec_bf16_f32_cuda(cublas_handle_, lw_v_proj_dev, x_norm_dev, v_dev, n_kv_heads * head_dim, hs, stream); 

        // Apply RoPE to Q (K/V handled in fused cache update)
        rope_cuda(q_dev, n_heads, head_dim, all_freqs_cis_dev, pos, stream);

        // KVCache Update (Fused RoPE for K, Separate Update for V)
        for (int kvh = 0; kvh < n_kv_heads; ++kvh) {
            const float* current_k_head_ptr = k_dev + kvh * head_dim;
            const float* current_v_head_ptr = v_dev + kvh * head_dim;
            
            // Call fused kernel for K (Applies RoPE + Updates Cache)
            rope_and_update_kv_cache_cuda(
                cache->layers[l].k_dev, // Base K cache pointer
                current_k_head_ptr,     // Original K data for this head
                all_freqs_cis_dev,      // Global RoPE frequencies buffer
                pos, kvh, max_seq_len, n_kv_heads, head_dim, stream);
                
            // Call original kernel for V (No RoPE needed for V)
            update_kv_cache_cuda(
                cache->layers[l].v_dev, // Base V cache pointer
                current_v_head_ptr,     // Original V data for this head (NO RoPE)
                pos, kvh, max_seq_len, n_kv_heads, head_dim, stream);
        }

        // Attention (Input: q_dev, K/V Cache, Output: attn_out_dev)
        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        attention_cuda(q_dev, cache->layers[l].k_dev, cache->layers[l].v_dev, attn_out_dev,
                       n_heads, pos + 1, head_dim, scale, max_seq_len, n_kv_heads, stream);
        
        // Attention Output Projection (Input: attn_out_dev, Output: attn_proj_dev)
        // We need a temporary buffer here if we fuse residual add into matvec
        // For now, use add_residual_cuda separately
        float* attn_proj_dev = nullptr; // Allocate temporary buffer for o_proj result
        gpuErrchk(cudaMallocAsync(&attn_proj_dev, hs * sizeof(float), stream));
        matvec_bf16_f32_cuda(cublas_handle_, lw_o_proj_dev, attn_out_dev, attn_proj_dev, hs, hs, stream); 

        // Residual 1 Add (Input: x_resid1_dev + attn_proj_dev, Output: x_dev)
        add_residual_cuda(attn_proj_dev, x_resid1_dev, x_dev, hs, stream);
        gpuErrchk(cudaFreeAsync(attn_proj_dev, stream)); // Free the temporary buffer

        // --- MLP Block --- 
        // Residual 2 Prep (Copy x -> x_resid2)
        gpuErrchk(cudaMemcpyAsync(x_resid2_dev, x_dev, hs * sizeof(float), cudaMemcpyDeviceToDevice, stream));

        // RMSNorm 2 (Input: x_dev, Weight: lw_post_norm_dev, Output: x_norm_dev)
        rmsnorm_vector_cuda(x_dev, lw_post_norm_dev, x_norm_dev, hs, eps, stream); 

        // MLP Projections (Input: x_norm_dev -> gate/up)
        matvec_bf16_f32_cuda(cublas_handle_, lw_gate_proj_dev, x_norm_dev, gate_vec_dev, is, hs, stream); 
        matvec_bf16_f32_cuda(cublas_handle_, lw_up_proj_dev, x_norm_dev, up_vec_dev, is, hs, stream); 

        // SwiGLU (Input: gate_vec_dev, up_vec_dev, Output: swiglu_vec_dev)
        swiglu_cuda(gate_vec_dev, up_vec_dev, swiglu_vec_dev, is, stream);

        // MLP Down Projection (Input: swiglu_vec_dev, Output: x_resid1_dev - Reuse buffer)
        // Need temporary buffer for down_proj output before residual add
        float* mlp_out_dev = nullptr; // Allocate temporary buffer
        gpuErrchk(cudaMallocAsync(&mlp_out_dev, hs * sizeof(float), stream));
        matvec_bf16_f32_cuda(cublas_handle_, lw_down_proj_dev, swiglu_vec_dev, mlp_out_dev, hs, is, stream); 

        // Residual 2 Add (Input: x_resid2_dev + mlp_out_dev, Output: x_dev)
        add_residual_cuda(mlp_out_dev, x_resid2_dev, x_dev, hs, stream);
        gpuErrchk(cudaFreeAsync(mlp_out_dev, stream)); // Free the temporary buffer

    } // End layer loop

    // --- Final Steps (Outside Layer Loop) ---
    
    // Final RMSNorm (Input: x_dev, Weight: final_norm_dev, Output: x_norm_dev)
    rmsnorm_vector_cuda(x_dev, final_norm_dev, x_norm_dev, hs, eps, stream); 

    // Final LM Head Projection (Input: x_norm_dev, Output: logits_dev)
    matvec_bf16_f32_cuda(cublas_handle_, lm_head_dev_, x_norm_dev, logits_dev, vs, hs, stream); 

    // --- Synchronize Stream before Copy and Free --- 
    // Wait for all kernels on the stream to complete before copying back logits
    gpuErrchk(cudaStreamSynchronize(stream)); 

    // --- Copy Logits Device -> Host --- 
    std::vector<float> logits(vs);
    // Use synchronous copy since we just synchronized the stream
    gpuErrchk(cudaMemcpy(logits.data(), logits_dev, vs * sizeof(float), cudaMemcpyDeviceToHost));

    // --- Free Device Memory --- 
    // Use cudaFreeAsync on the same stream for potentially better performance
    gpuErrchk(cudaFreeAsync(x_dev, stream));
    gpuErrchk(cudaFreeAsync(x_norm_dev, stream));
    gpuErrchk(cudaFreeAsync(x_resid1_dev, stream));
    gpuErrchk(cudaFreeAsync(x_resid2_dev, stream));
    gpuErrchk(cudaFreeAsync(q_dev, stream));
    gpuErrchk(cudaFreeAsync(k_dev, stream));
    gpuErrchk(cudaFreeAsync(v_dev, stream));
    gpuErrchk(cudaFreeAsync(attn_out_dev, stream));
    // attn_proj_dev freed above
    gpuErrchk(cudaFreeAsync(gate_vec_dev, stream));
    gpuErrchk(cudaFreeAsync(up_vec_dev, stream));
    gpuErrchk(cudaFreeAsync(swiglu_vec_dev, stream));
    // mlp_out_dev freed above
    gpuErrchk(cudaFreeAsync(logits_dev, stream));

    // Optional: Synchronize again to ensure frees are processed if needed before function returns,
    // although technically not required if caller manages stream synchronization.
    // gpuErrchk(cudaStreamSynchronize(stream)); 

    // Logger::info("[CUDA] forward_device finished.");
    return logits;

} // End forward_device
#endif // HAS_CUDA (End of forward_device function definition)

// --- Helper function definitions (e.g., parse_model_config, etc.) ---

// ... Rest of the file (parse_model_config, etc.)
