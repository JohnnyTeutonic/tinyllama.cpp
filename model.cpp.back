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
#include "gguf_parser.h"
#include <variant> // Add include for variant index
#include "quantization.h" // Include for GGML_QK_K

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

// --- START: Private Helper: Initialize Weights ---
void TinyLlamaModel::initialize_weights(const SafeTensorsLoader* loader, const GGUFData* gguf) {
    Logger::info("Initializing model weights...");
    int hs = config_.hidden_size;
    int is = config_.intermediate_size;
    int nhl = config_.num_hidden_layers;
    int vs = config_.vocab_size;
    int n_heads = config_.num_attention_heads;
    int n_kv_heads = config_.num_key_value_heads;
    int kv_dim = (hs / n_heads) * n_kv_heads;

    layers.resize(nhl); // Ensure layers vector is sized

    if (gguf) {
        Logger::info("Mapping weights from GGUF data...");
        // Call the friend function directly since gguf is provided
        map_gguf_weights(*gguf, *this);
    } else if (loader) {
        Logger::info("Loading weights from SafeTensors data...");
        // Load weights directly into appropriate vectors from the loader
        // NOTE: This currently only loads BF16. Needs extension if safetensors can contain other types.
        try {
            embed_tokens = uint8_vector_to_uint16_vector(loader->get_tensor_bytes("model.embed_tokens.weight"), vs * hs);
        } catch (const std::exception& e) { Logger::error("Missing model.embed_tokens.weight: " + std::string(e.what())); }
        try {
            lm_head = uint8_vector_to_uint16_vector(loader->get_tensor_bytes("lm_head.weight"), vs * hs);
        } catch (const std::exception& e) { Logger::error("Missing lm_head.weight: " + std::string(e.what())); }
        try {
            final_norm = uint8_vector_to_uint16_vector(loader->get_tensor_bytes("model.norm.weight"), hs);
        } catch (const std::exception& e) { Logger::error("Missing model.norm.weight: " + std::string(e.what())); }

        for (int i = 0; i < nhl; ++i) {
            Logger::info("Loading SafeTensors weights for layer " + std::to_string(i));
            std::string prefix = "model.layers." + std::to_string(i) + ".";
            auto& lw = layers[i]; // Get mutable ref
            try { lw.q_proj = uint8_vector_to_uint16_vector(loader->get_tensor_bytes(prefix + "self_attn.q_proj.weight"), hs * hs); } 
            catch (const std::exception& e) { Logger::error("Missing " + prefix + "self_attn.q_proj.weight: " + std::string(e.what())); }
            try { lw.k_proj = uint8_vector_to_uint16_vector(loader->get_tensor_bytes(prefix + "self_attn.k_proj.weight"), kv_dim * hs); } 
            catch (const std::exception& e) { Logger::error("Missing " + prefix + "self_attn.k_proj.weight: " + std::string(e.what())); }
            try { lw.v_proj = uint8_vector_to_uint16_vector(loader->get_tensor_bytes(prefix + "self_attn.v_proj.weight"), kv_dim * hs); } 
            catch (const std::exception& e) { Logger::error("Missing " + prefix + "self_attn.v_proj.weight: " + std::string(e.what())); }
            try { lw.o_proj = uint8_vector_to_uint16_vector(loader->get_tensor_bytes(prefix + "self_attn.o_proj.weight"), hs * hs); } 
            catch (const std::exception& e) { Logger::error("Missing " + prefix + "self_attn.o_proj.weight: " + std::string(e.what())); }
            try { lw.gate_proj = uint8_vector_to_uint16_vector(loader->get_tensor_bytes(prefix + "mlp.gate_proj.weight"), is * hs); } 
            catch (const std::exception& e) { Logger::error("Missing " + prefix + "mlp.gate_proj.weight: " + std::string(e.what())); }
            try { lw.up_proj = uint8_vector_to_uint16_vector(loader->get_tensor_bytes(prefix + "mlp.up_proj.weight"), is * hs); } 
            catch (const std::exception& e) { Logger::error("Missing " + prefix + "mlp.up_proj.weight: " + std::string(e.what())); }
            try { lw.down_proj = uint8_vector_to_uint16_vector(loader->get_tensor_bytes(prefix + "mlp.down_proj.weight"), hs * is); } 
            catch (const std::exception& e) { Logger::error("Missing " + prefix + "mlp.down_proj.weight: " + std::string(e.what())); }
            try { lw.input_layernorm = uint8_vector_to_uint16_vector(loader->get_tensor_bytes(prefix + "input_layernorm.weight"), hs); } 
            catch (const std::exception& e) { Logger::error("Missing " + prefix + "input_layernorm.weight: " + std::string(e.what())); }
            try { lw.post_attention_layernorm = uint8_vector_to_uint16_vector(loader->get_tensor_bytes(prefix + "post_attention_layernorm.weight"), hs); } 
            catch (const std::exception& e) { Logger::error("Missing " + prefix + "post_attention_layernorm.weight: " + std::string(e.what())); }
            
            // If safetensors are BF16, populate the F32 fields too for consistency (or add logic later to handle FP32 safetensors)
            lw.input_layernorm_f32 = bf16vec_to_float_vec(lw.input_layernorm);
            lw.post_attention_layernorm_f32 = bf16vec_to_float_vec(lw.post_attention_layernorm);
            lw.q_proj_f32 = bf16vec_to_float_vec(lw.q_proj);
            lw.k_proj_f32 = bf16vec_to_float_vec(lw.k_proj);
            lw.v_proj_f32 = bf16vec_to_float_vec(lw.v_proj);
            lw.o_proj_f32 = bf16vec_to_float_vec(lw.o_proj);
            lw.gate_proj_f32 = bf16vec_to_float_vec(lw.gate_proj);
            lw.up_proj_f32 = bf16vec_to_float_vec(lw.up_proj);
            lw.down_proj_f32 = bf16vec_to_float_vec(lw.down_proj);
        }
         // Populate top-level F32 fields from BF16 safetensors
        embed_tokens_f32 = bf16vec_to_float_vec(embed_tokens);
        lm_head_f32 = bf16vec_to_float_vec(lm_head);
        final_norm_f32 = bf16vec_to_float_vec(final_norm);

    } else {
        throw std::runtime_error("TinyLlamaModel::initialize_weights called with neither GGUF nor SafeTensors loader.");
    }
    Logger::info("Finished initializing model weights.");
}
// --- END: Private Helper: Initialize Weights ---


// --- START: Private Helper: Initialize GPU Resources & RoPE ---
void TinyLlamaModel::initialize_gpu_and_rope() {
    Logger::info("Initializing GPU resources and RoPE...");
    int hs = config_.hidden_size;
    int is = config_.intermediate_size;
    int nhl = config_.num_hidden_layers;
    int vs = config_.vocab_size;
    int n_heads = config_.num_attention_heads;
    int n_kv_heads = config_.num_key_value_heads;
    
    // --- START: VALIDATION --- 
    if (hs <= 0) {
        throw std::runtime_error("Invalid model configuration: hidden_size must be positive. Check GGUF metadata ('llama.embedding_length').");
    }
    if (vs <= 0) {
        throw std::runtime_error("Invalid model configuration: vocab_size must be positive. Check GGUF metadata ('general.vocab_size').");
    }
    if (n_heads <= 0) {
        throw std::runtime_error("Invalid model configuration: num_attention_heads must be positive. Check GGUF metadata ('llama.head_count').");
    }
    if (n_kv_heads <= 0) {
         throw std::runtime_error("Invalid model configuration: num_key_value_heads must be positive. Check GGUF metadata ('llama.head_count_kv').");
    }
    if (hs % n_heads != 0) {
         throw std::runtime_error("Invalid model configuration: hidden_size must be divisible by num_attention_heads.");
    }
    // --- END: VALIDATION --- 

    int kv_dim = (hs / n_heads) * n_kv_heads; // Now safe from division by zero for n_heads
    int head_dim = hs / n_heads;             // Now safe from division by zero

#ifdef HAS_CUDA
    Logger::info("Initializing CUDA resources...");
    // --- START CUBLAS Initialization ---
    cublasStatus_t cublas_status = cublasCreate(&cublas_handle_);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        Logger::error("cuBLAS handle creation failed: " + std::to_string(cublas_status));
        throw std::runtime_error("Failed to initialize cuBLAS");
    }
    Logger::info("cuBLAS handle created successfully.");
    // --- END CUBLAS Initialization ---

    // Allocate and copy final_norm weights to device (FP32 version)
    // Ensure final_norm_f32 is populated before this!
    if (final_norm_f32.empty() && !final_norm.empty()) { // If F32 is empty but BF16 exists (e.g. from safetensors)
         final_norm_f32 = bf16vec_to_float_vec(final_norm);
    }
    if (!final_norm_f32.empty()) {
    gpuErrchk(cudaMalloc(&final_norm_dev, final_norm_f32.size() * sizeof(float)));
    gpuErrchk(cudaMemcpy(final_norm_dev, final_norm_f32.data(), final_norm_f32.size() * sizeof(float), cudaMemcpyHostToDevice));
        Logger::info("Copied final_norm weights (FP32) to GPU.");
    } else {
         Logger::warning("Final norm weights (FP32) empty, skipping GPU copy.");
    }


    // --- Allocate and copy Layer Norm weights to GPU ---
    for (int i = 0; i < nhl; ++i) {
        // Ensure F32 versions are populated
        if (layers[i].input_layernorm_f32.empty() && !layers[i].input_layernorm.empty()) {
             layers[i].input_layernorm_f32 = bf16vec_to_float_vec(layers[i].input_layernorm);
        }
         if (layers[i].post_attention_layernorm_f32.empty() && !layers[i].post_attention_layernorm.empty()) {
             layers[i].post_attention_layernorm_f32 = bf16vec_to_float_vec(layers[i].post_attention_layernorm);
        }

        // Copy input layernorm
        if (!layers[i].input_layernorm_f32.empty()) {
            gpuErrchk(cudaMalloc(&layers[i].input_layernorm_dev, layers[i].input_layernorm_f32.size() * sizeof(float)));
            gpuErrchk(cudaMemcpy(layers[i].input_layernorm_dev, layers[i].input_layernorm_f32.data(), layers[i].input_layernorm_f32.size() * sizeof(float), cudaMemcpyHostToDevice));
        }
        // Copy post-attention layernorm
        if (!layers[i].post_attention_layernorm_f32.empty()) {
            gpuErrchk(cudaMalloc(&layers[i].post_attention_layernorm_dev, layers[i].post_attention_layernorm_f32.size() * sizeof(float)));
            gpuErrchk(cudaMemcpy(layers[i].post_attention_layernorm_dev, layers[i].post_attention_layernorm_f32.data(), layers[i].post_attention_layernorm_f32.size() * sizeof(float), cudaMemcpyHostToDevice));
        }
    }
     Logger::info("Copied all layer norm weights (FP32) to GPU.");


    // --- START: Allocate and copy persistent BF16 weights ---
    // TODO: Add checks if these vectors are populated before copying
    if (!embed_tokens.empty()) {
        gpuErrchk(cudaMalloc(&token_embedding_table_dev_, embed_tokens.size() * sizeof(uint16_t)));
        gpuErrchk(cudaMemcpy(token_embedding_table_dev_, embed_tokens.data(), embed_tokens.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
        Logger::info("Copied token_embedding_table (bf16) to GPU.");
    }
     if (!lm_head.empty()) {
        gpuErrchk(cudaMalloc(&lm_head_dev_, lm_head.size() * sizeof(uint16_t)));
        gpuErrchk(cudaMemcpy(lm_head_dev_, lm_head.data(), lm_head.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
        Logger::info("Copied lm_head (bf16) to GPU.");
    }

    // Concatenate Layer Weights (BF16) - Only if BF16 weights exist
    bool has_bf16_layer_weights = !layers.empty() && !layers[0].q_proj.empty(); // Check first layer's q_proj
    if(has_bf16_layer_weights) {
        size_t layer_q_size = (size_t)hs * hs;
        size_t layer_k_size = (size_t)kv_dim * hs;
        size_t layer_v_size = (size_t)kv_dim * hs;
        size_t layer_o_size = (size_t)hs * hs;
        size_t layer_gate_size = (size_t)is * hs;
        size_t layer_up_size = (size_t)is * hs;
        size_t layer_down_size = (size_t)hs * is;

        std::vector<uint16_t> all_q_proj_host, all_k_proj_host, all_v_proj_host, all_o_proj_host;
        std::vector<uint16_t> all_gate_proj_host, all_up_proj_host, all_down_proj_host;

        // Reserve space
        all_q_proj_host.reserve(nhl * layer_q_size);
        all_k_proj_host.reserve(nhl * layer_k_size);
        // ... reserve for others ...

        for (int i = 0; i < nhl; ++i) {
            const auto& lw = layers[i];
             if (lw.q_proj.empty()) continue; // Skip if this layer is missing bf16 weights
            all_q_proj_host.insert(all_q_proj_host.end(), lw.q_proj.begin(), lw.q_proj.end());
            all_k_proj_host.insert(all_k_proj_host.end(), lw.k_proj.begin(), lw.k_proj.end());
            all_v_proj_host.insert(all_v_proj_host.end(), lw.v_proj.begin(), lw.v_proj.end());
            all_o_proj_host.insert(all_o_proj_host.end(), lw.o_proj.begin(), lw.o_proj.end());
            all_gate_proj_host.insert(all_gate_proj_host.end(), lw.gate_proj.begin(), lw.gate_proj.end());
            all_up_proj_host.insert(all_up_proj_host.end(), lw.up_proj.begin(), lw.up_proj.end());
            all_down_proj_host.insert(all_down_proj_host.end(), lw.down_proj.begin(), lw.down_proj.end());
        }
        Logger::info("Concatenated BF16 layer weights on host.");

        // Allocate and Copy concatenated BF16 weights
        if (!all_q_proj_host.empty()) { // Check if concatenation resulted in data
            gpuErrchk(cudaMalloc(&w_q_dev_, all_q_proj_host.size() * sizeof(uint16_t)));
            gpuErrchk(cudaMemcpy(w_q_dev_, all_q_proj_host.data(), all_q_proj_host.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
            // ... cudaMalloc/cudaMemcpy for k, v, o, gate, up, down ...
             gpuErrchk(cudaMalloc(&w_k_dev_, all_k_proj_host.size() * sizeof(uint16_t)));
             gpuErrchk(cudaMemcpy(w_k_dev_, all_k_proj_host.data(), all_k_proj_host.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
             gpuErrchk(cudaMalloc(&w_v_dev_, all_v_proj_host.size() * sizeof(uint16_t)));
             gpuErrchk(cudaMemcpy(w_v_dev_, all_v_proj_host.data(), all_v_proj_host.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
             gpuErrchk(cudaMalloc(&w_o_dev_, all_o_proj_host.size() * sizeof(uint16_t)));
             gpuErrchk(cudaMemcpy(w_o_dev_, all_o_proj_host.data(), all_o_proj_host.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
             gpuErrchk(cudaMalloc(&w_gate_dev_, all_gate_proj_host.size() * sizeof(uint16_t)));
             gpuErrchk(cudaMemcpy(w_gate_dev_, all_gate_proj_host.data(), all_gate_proj_host.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
             gpuErrchk(cudaMalloc(&w_up_dev_, all_up_proj_host.size() * sizeof(uint16_t)));
             gpuErrchk(cudaMemcpy(w_up_dev_, all_up_proj_host.data(), all_up_proj_host.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
             gpuErrchk(cudaMalloc(&w_down_dev_, all_down_proj_host.size() * sizeof(uint16_t)));
             gpuErrchk(cudaMemcpy(w_down_dev_, all_down_proj_host.data(), all_down_proj_host.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
            Logger::info("Copied concatenated layer weights (bf16) to GPU.");
                 } else {
             Logger::info("No BF16 layer weights found to concatenate/copy to GPU.");
        }
    } else {
         Logger::info("Skipping BF16 layer weight concatenation/copy (no BF16 weights found).");
    }
    // --- END: Allocate and copy persistent BF16 weights ---


    // --- START: Allocate and copy persistent FP32 weights ---
    // Ensure F32 vectors are populated before copying
     if (embed_tokens_f32.empty() && !embed_tokens.empty()) { // Convert if needed (e.g., from safetensors)
        embed_tokens_f32 = bf16vec_to_float_vec(embed_tokens);
    }
    if (lm_head_f32.empty() && !lm_head.empty()) {
         lm_head_f32 = bf16vec_to_float_vec(lm_head);
    }
    // TODO: Similar checks/conversions might be needed for layer weights if loaded as BF16

    if (!embed_tokens_f32.empty()) {
        gpuErrchk(cudaMalloc(&token_embedding_table_f32_dev_, embed_tokens_f32.size() * sizeof(float)));
        gpuErrchk(cudaMemcpy(token_embedding_table_f32_dev_, embed_tokens_f32.data(), embed_tokens_f32.size() * sizeof(float), cudaMemcpyHostToDevice));
        Logger::info("Copied token_embedding_table (fp32) to GPU.");
    }
     if (!lm_head_f32.empty()) {
        gpuErrchk(cudaMalloc(&lm_head_f32_dev_, lm_head_f32.size() * sizeof(float)));
        gpuErrchk(cudaMemcpy(lm_head_f32_dev_, lm_head_f32.data(), lm_head_f32.size() * sizeof(float), cudaMemcpyHostToDevice));
        Logger::info("Copied lm_head (fp32) to GPU.");
    }

    // Concatenate Layer Weights (FP32) - Only if FP32 weights exist
     bool has_fp32_layer_weights = !layers.empty() && !layers[0].q_proj_f32.empty(); // Check first layer's q_proj_f32
     if (has_fp32_layer_weights) {
        size_t layer_q_size = (size_t)hs * hs;
        size_t layer_k_size = (size_t)kv_dim * hs;
        // ... other sizes ...

        std::vector<float> all_q_proj_f32, all_k_proj_f32, all_v_proj_f32, all_o_proj_f32;
        std::vector<float> all_gate_proj_f32, all_up_proj_f32, all_down_proj_f32;

        // Reserve space
        all_q_proj_f32.reserve(nhl * layer_q_size);
        // ... reserve for others ...

        for (int i = 0; i < nhl; ++i) {
            const auto& lw = layers[i];
             if (lw.q_proj_f32.empty()) continue; // Skip if this layer is missing f32 weights
            all_q_proj_f32.insert(all_q_proj_f32.end(), lw.q_proj_f32.begin(), lw.q_proj_f32.end());
            all_k_proj_f32.insert(all_k_proj_f32.end(), lw.k_proj_f32.begin(), lw.k_proj_f32.end());
            all_v_proj_f32.insert(all_v_proj_f32.end(), lw.v_proj_f32.begin(), lw.v_proj_f32.end());
            all_o_proj_f32.insert(all_o_proj_f32.end(), lw.o_proj_f32.begin(), lw.o_proj_f32.end());
            all_gate_proj_f32.insert(all_gate_proj_f32.end(), lw.gate_proj_f32.begin(), lw.gate_proj_f32.end());
            all_up_proj_f32.insert(all_up_proj_f32.end(), lw.up_proj_f32.begin(), lw.up_proj_f32.end());
            all_down_proj_f32.insert(all_down_proj_f32.end(), lw.down_proj_f32.begin(), lw.down_proj_f32.end());
        }
        Logger::info("Concatenated FP32 layer weights on host.");

        // Allocate and Copy concatenated FP32 weights
         if (!all_q_proj_f32.empty()) {
            gpuErrchk(cudaMalloc(&w_q_f32_dev_, all_q_proj_f32.size() * sizeof(float)));
            gpuErrchk(cudaMemcpy(w_q_f32_dev_, all_q_proj_f32.data(), all_q_proj_f32.size() * sizeof(float), cudaMemcpyHostToDevice));
            // ... cudaMalloc/cudaMemcpy for k, v, o, gate, up, down ...
             gpuErrchk(cudaMalloc(&w_k_f32_dev_, all_k_proj_f32.size() * sizeof(float)));
             gpuErrchk(cudaMemcpy(w_k_f32_dev_, all_k_proj_f32.data(), all_k_proj_f32.size() * sizeof(float), cudaMemcpyHostToDevice));
             gpuErrchk(cudaMalloc(&w_v_f32_dev_, all_v_proj_f32.size() * sizeof(float)));
             gpuErrchk(cudaMemcpy(w_v_f32_dev_, all_v_proj_f32.data(), all_v_proj_f32.size() * sizeof(float), cudaMemcpyHostToDevice));
             gpuErrchk(cudaMalloc(&w_o_f32_dev_, all_o_proj_f32.size() * sizeof(float)));
             gpuErrchk(cudaMemcpy(w_o_f32_dev_, all_o_proj_f32.data(), all_o_proj_f32.size() * sizeof(float), cudaMemcpyHostToDevice));
             gpuErrchk(cudaMalloc(&w_gate_f32_dev_, all_gate_proj_f32.size() * sizeof(float)));
             gpuErrchk(cudaMemcpy(w_gate_f32_dev_, all_gate_proj_f32.data(), all_gate_proj_f32.size() * sizeof(float), cudaMemcpyHostToDevice));
             gpuErrchk(cudaMalloc(&w_up_f32_dev_, all_up_proj_f32.size() * sizeof(float)));
             gpuErrchk(cudaMemcpy(w_up_f32_dev_, all_up_proj_f32.data(), all_up_proj_f32.size() * sizeof(float), cudaMemcpyHostToDevice));
             gpuErrchk(cudaMalloc(&w_down_f32_dev_, all_down_proj_f32.size() * sizeof(float)));
             gpuErrchk(cudaMemcpy(w_down_f32_dev_, all_down_proj_f32.data(), all_down_proj_f32.size() * sizeof(float), cudaMemcpyHostToDevice));
            Logger::info("Copied all concatenated layer weights (fp32) to GPU.");
        } else {
             Logger::info("No FP32 layer weights found to concatenate/copy to GPU.");
        }
    } else {
         Logger::info("Skipping FP32 layer weight concatenation/copy (no FP32 weights found).");
    }
    // --- END: Allocate and copy persistent FP32 weights ---

    // --- TODO: Allocate and copy persistent Q4_K / Q6_K weights ---
    // Add similar logic here if needed for quantized types on GPU

    Logger::info("Finished initializing CUDA weights.");
#endif // HAS_CUDA


    // --- Precompute RoPE cos/sin values (CPU part) ---
    Logger::info("Precomputing RoPE frequencies...");
    int max_seq_len = config_.max_position_embeddings; // Use config_ member
    precomputed_freqs_cis_.resize((max_seq_len * head_dim) / 2);
    float theta = config_.rope_theta;
    for (int pos = 0; pos < max_seq_len; ++pos) {
    for (int i = 0; i < head_dim; i += 2) {
            float freq = std::pow(theta, -((float)i) / head_dim);
            float angle = pos * freq;
            float cos_val = std::cos(angle);
            float sin_val = std::sin(angle);
            precomputed_freqs_cis_[(pos * head_dim / 2) + (i / 2)] = {cos_val, sin_val};
            }
        }
    Logger::info("Finished precomputing RoPE cos/sin frequencies.");

#ifdef HAS_CUDA
    // --- Allocate and copy RoPE frequencies to GPU ---
    if (!precomputed_freqs_cis_.empty()) {
        size_t total_freq_elements = precomputed_freqs_cis_.size() * 2;
        gpuErrchk(cudaMalloc(&all_freqs_cis_dev, total_freq_elements * sizeof(float)));
        Logger::info("Allocated persistent RoPE frequency buffer on GPU: " + 
                     std::to_string(total_freq_elements * sizeof(float) / 1024.0) + " KB");

        std::vector<float> flat_host_freqs;
        flat_host_freqs.reserve(total_freq_elements);
        for (const auto& p : precomputed_freqs_cis_) {
            flat_host_freqs.push_back(p.first);
            flat_host_freqs.push_back(p.second);
        }
        gpuErrchk(cudaMemcpy(all_freqs_cis_dev, flat_host_freqs.data(), total_freq_elements * sizeof(float), cudaMemcpyHostToDevice));
        Logger::info("Copied all precomputed RoPE frequencies to persistent GPU buffer.");

    } else {
        Logger::warning("Host precomputed_freqs_cis_ is empty, skipping GPU RoPE buffer allocation.");
    }
    Logger::info("Finished initializing CUDA RoPE frequencies.");
#endif // HAS_CUDA

    Logger::info("Finished initializing GPU resources and RoPE.");
}
// --- END: Private Helper: Initialize GPU Resources & RoPE ---


// TinyLlamaModel constructor: from config and safetensors loader
TinyLlamaModel::TinyLlamaModel(const ModelConfig& config, const SafeTensorsLoader& loader)
    : config_(config) // Initialize config
{
    Logger::info("Constructing TinyLlamaModel from SafeTensorsLoader.");
    initialize_weights(&loader, nullptr); // Load weights from loader
    initialize_gpu_and_rope();            // Initialize GPU resources and RoPE
    Logger::info("TinyLlamaModel construction from SafeTensorsLoader complete.");
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

    // --- START: Free persistent FP32 weights ---
    if (token_embedding_table_f32_dev_) {
        gpuErrchk(cudaFree(token_embedding_table_f32_dev_));
        token_embedding_table_f32_dev_ = nullptr;
    }
    if (lm_head_f32_dev_) {
        gpuErrchk(cudaFree(lm_head_f32_dev_));
        lm_head_f32_dev_ = nullptr;
    }
    if (w_q_f32_dev_) {
        gpuErrchk(cudaFree(w_q_f32_dev_));
        w_q_f32_dev_ = nullptr;
    }
    if (w_k_f32_dev_) {
        gpuErrchk(cudaFree(w_k_f32_dev_));
        w_k_f32_dev_ = nullptr;
    }
    if (w_v_f32_dev_) {
        gpuErrchk(cudaFree(w_v_f32_dev_));
        w_v_f32_dev_ = nullptr;
    }
    if (w_o_f32_dev_) {
        gpuErrchk(cudaFree(w_o_f32_dev_));
        w_o_f32_dev_ = nullptr;
    }
    if (w_gate_f32_dev_) {
        gpuErrchk(cudaFree(w_gate_f32_dev_));
        w_gate_f32_dev_ = nullptr;
    }
    if (w_up_f32_dev_) {
        gpuErrchk(cudaFree(w_up_f32_dev_));
        w_up_f32_dev_ = nullptr;
    }
    if (w_down_f32_dev_) {
        gpuErrchk(cudaFree(w_down_f32_dev_));
        w_down_f32_dev_ = nullptr;
    }
    // --- END: Free persistent FP32 weights ---

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

        // BF16 Pointers (Keep for reference? Or remove if not needed)
        // const uint16_t* lw_q_proj_dev    = w_q_dev_    + (size_t)l * layer_q_size;
        // const uint16_t* lw_k_proj_dev    = w_k_dev_    + (size_t)l * layer_k_size;
        // const uint16_t* lw_v_proj_dev    = w_v_dev_    + (size_t)l * layer_v_size;
        // const uint16_t* lw_o_proj_dev    = w_o_dev_    + (size_t)l * layer_o_size;
        // const uint16_t* lw_gate_proj_dev = w_gate_dev_ + (size_t)l * layer_gate_size;
        // const uint16_t* lw_up_proj_dev   = w_up_dev_   + (size_t)l * layer_up_size;
        // const uint16_t* lw_down_proj_dev = w_down_dev_ + (size_t)l * layer_down_size;
        
        // FP32 Pointers
        const float* lw_q_proj_f32_dev    = w_q_f32_dev_    + (size_t)l * layer_q_size;
        const float* lw_k_proj_f32_dev    = w_k_f32_dev_    + (size_t)l * layer_k_size;
        const float* lw_v_proj_f32_dev    = w_v_f32_dev_    + (size_t)l * layer_v_size;
        const float* lw_o_proj_f32_dev    = w_o_f32_dev_    + (size_t)l * layer_o_size;
        const float* lw_gate_proj_f32_dev = w_gate_f32_dev_ + (size_t)l * layer_gate_size;
        const float* lw_up_proj_f32_dev   = w_up_f32_dev_   + (size_t)l * layer_up_size;
        const float* lw_down_proj_f32_dev = w_down_f32_dev_ + (size_t)l * layer_down_size;

        const float* lw_in_norm_dev   = layers[l].input_layernorm_dev;
        const float* lw_post_norm_dev = layers[l].post_attention_layernorm_dev;

        // Residual 1 Prep (Copy x -> x_resid1)
        gpuErrchk(cudaMemcpyAsync(x_resid1_dev, x_dev, hs * sizeof(float), cudaMemcpyDeviceToDevice, stream));

        // RMSNorm 1 (Input: x_dev, Weight: lw_in_norm_dev, Output: x_norm_dev)
        rmsnorm_vector_cuda(x_dev, lw_in_norm_dev, x_norm_dev, hs, eps, stream); 

        // Q, K, V projections (Input: x_norm_dev, Outputs: q_dev, k_dev, v_dev)
        matvec_f32_f32_cuda(cublas_handle_, lw_q_proj_f32_dev, x_norm_dev, q_dev, hs, hs, stream); 
        matvec_f32_f32_cuda(cublas_handle_, lw_k_proj_f32_dev, x_norm_dev, k_dev, n_kv_heads * head_dim, hs, stream); 
        matvec_f32_f32_cuda(cublas_handle_, lw_v_proj_f32_dev, x_norm_dev, v_dev, n_kv_heads * head_dim, hs, stream); 

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
        matvec_f32_f32_cuda(cublas_handle_, lw_o_proj_f32_dev, attn_out_dev, attn_proj_dev, hs, hs, stream); 

        // Residual 1 Add (Input: x_resid1_dev + attn_proj_dev, Output: x_dev)
        add_residual_cuda(attn_proj_dev, x_resid1_dev, x_dev, hs, stream);
        gpuErrchk(cudaFreeAsync(attn_proj_dev, stream)); // Free the temporary buffer

        // --- MLP Block --- 
        // Residual 2 Prep (Copy x -> x_resid2)
        gpuErrchk(cudaMemcpyAsync(x_resid2_dev, x_dev, hs * sizeof(float), cudaMemcpyDeviceToDevice, stream));

        // RMSNorm 2 (Input: x_dev, Weight: lw_post_norm_dev, Output: x_norm_dev)
        rmsnorm_vector_cuda(x_dev, lw_post_norm_dev, x_norm_dev, hs, eps, stream); 

        // MLP Projections (Input: x_norm_dev -> gate/up)
        matvec_f32_f32_cuda(cublas_handle_, lw_gate_proj_f32_dev, x_norm_dev, gate_vec_dev, is, hs, stream); 
        matvec_f32_f32_cuda(cublas_handle_, lw_up_proj_f32_dev, x_norm_dev, up_vec_dev, is, hs, stream); 

        // SwiGLU (Input: gate_vec_dev, up_vec_dev, Output: swiglu_vec_dev)
        swiglu_cuda(gate_vec_dev, up_vec_dev, swiglu_vec_dev, is, stream);

        // MLP Down Projection (Input: swiglu_vec_dev, Output: x_resid1_dev - Reuse buffer)
        // Need temporary buffer for down_proj output before residual add
        float* mlp_out_dev = nullptr; // Allocate temporary buffer
        gpuErrchk(cudaMallocAsync(&mlp_out_dev, hs * sizeof(float), stream));
        matvec_f32_f32_cuda(cublas_handle_, lw_down_proj_f32_dev, swiglu_vec_dev, mlp_out_dev, hs, is, stream); 
        
        // Residual 2 Add (Input: x_resid2_dev + mlp_out_dev, Output: x_dev)
        add_residual_cuda(mlp_out_dev, x_resid2_dev, x_dev, hs, stream);
        gpuErrchk(cudaFreeAsync(mlp_out_dev, stream)); // Free the temporary buffer

    } // End layer loop

    // --- Final Steps (Outside Layer Loop) --- 
    
    // Final RMSNorm (Input: x_dev, Weight: final_norm_dev, Output: x_norm_dev)
    rmsnorm_vector_cuda(x_dev, final_norm_dev, x_norm_dev, hs, eps, stream); 

    // Final LM Head Projection (Input: x_norm_dev, Output: logits_dev)
    matvec_f32_f32_cuda(cublas_handle_, lm_head_f32_dev_, x_norm_dev, logits_dev, vs, hs, stream); 

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

// Map GGUF weights into the model's vectors (BF16 only)
void map_gguf_weights(const GGUFData& gguf, TinyLlamaModel& model) {
    Logger::info("Mapping GGUF weights to model fields...");
    int nhl = model.get_config().num_hidden_layers;
    int hs = model.get_config().hidden_size;
    int is = model.get_config().intermediate_size;
    int vs = model.get_config().vocab_size;
    int n_heads = model.get_config().num_attention_heads;
    int n_kv_heads = model.get_config().num_key_value_heads;
    int kv_dim = (hs / n_heads) * n_kv_heads;

    // --- START: Log total tensor data size --- 
    size_t total_data_size = gguf.tensor_data.size();
    Logger::info("map_gguf_weights: Total tensor data size available: " + std::to_string(total_data_size) + " bytes.");
    // --- END: Log total tensor data size --- 

    // Helper lambdas for assignment
    auto assign_vec_bf16 = [&](std::vector<uint16_t>& vec, const GGUFTensorInfo& tinfo) {
        size_t n_elem = tinfo.num_elements;
        const uint16_t* src = reinterpret_cast<const uint16_t*>(gguf.tensor_data.data() + tinfo.offset);
        vec.assign(src, src + n_elem); // This one is fine (BF16 is not blocked)
    };
    auto assign_vec_f32 = [&](std::vector<float>& vec, const GGUFTensorInfo& tinfo) {
        size_t n_elem = tinfo.num_elements;
        const float* src = reinterpret_cast<const float*>(gguf.tensor_data.data() + tinfo.offset);
        vec.assign(src, src + n_elem); // This one is fine (F32 is not blocked)
    };
    auto assign_vec_q4k = [&](std::vector<block_q4_K>& vec, const GGUFTensorInfo& tinfo) {
        if (tinfo.num_elements == 0) { vec.clear(); return; } // Handle empty tensor
        if (GGML_QK_K == 0) throw std::runtime_error("GGML_QK_K is zero!"); // Avoid division by zero
        if (tinfo.num_elements % GGML_QK_K != 0) {
             throw std::runtime_error("Tensor '" + tinfo.name + "' num_elements (" + std::to_string(tinfo.num_elements)
                                           + ") not divisible by GGML_QK_K (" + std::to_string(GGML_QK_K) + ") for Q4_K assignment.");
        }
        size_t num_blocks = tinfo.num_elements / GGML_QK_K; // Calculate number of blocks
        const block_q4_K* src = reinterpret_cast<const block_q4_K*>(gguf.tensor_data.data() + tinfo.offset);
        // Check source pointer validity (optional extra check)
        uintptr_t src_end_addr = reinterpret_cast<uintptr_t>(src) + num_blocks * sizeof(block_q4_K);
        if (src_end_addr > reinterpret_cast<uintptr_t>(gguf.tensor_data.data()) + gguf.tensor_data.size()) {
             throw std::out_of_range("Calculated source range for Q4_K tensor '" + tinfo.name + "' exceeds data buffer.");
        }
        vec.assign(src, src + num_blocks); // Use num_blocks for assignment
    };
    auto assign_vec_q6k = [&](std::vector<block_q6_K>& vec, const GGUFTensorInfo& tinfo) {
        if (tinfo.num_elements == 0) { vec.clear(); return; } // Handle empty tensor
        if (GGML_QK_K == 0) throw std::runtime_error("GGML_QK_K is zero!"); // Avoid division by zero
        if (tinfo.num_elements % GGML_QK_K != 0) {
             throw std::runtime_error("Tensor '" + tinfo.name + "' num_elements (" + std::to_string(tinfo.num_elements)
                                           + ") not divisible by GGML_QK_K (" + std::to_string(GGML_QK_K) + ") for Q6_K assignment.");
        }
        size_t num_blocks = tinfo.num_elements / GGML_QK_K; // Calculate number of blocks
        const block_q6_K* src = reinterpret_cast<const block_q6_K*>(gguf.tensor_data.data() + tinfo.offset);
        // Check source pointer validity (optional extra check)
        uintptr_t src_end_addr = reinterpret_cast<uintptr_t>(src) + num_blocks * sizeof(block_q6_K);
        if (src_end_addr > reinterpret_cast<uintptr_t>(gguf.tensor_data.data()) + gguf.tensor_data.size()) {
             throw std::out_of_range("Calculated source range for Q6_K tensor '" + tinfo.name + "' exceeds data buffer.");
        }
        vec.assign(src, src + num_blocks); // Use num_blocks for assignment
    };

    for (const auto& tinfo : gguf.tensor_infos) {
        const std::string& name = tinfo.name;

        // --- START: Detailed Tensor Logging --- 
        std::stringstream ss_log;
        ss_log << "Attempting to map tensor: '" << name << "'"
               << ", Type: " << static_cast<int>(tinfo.type) // Log raw enum value
               // << ", TypeName: " << ggml_type_name(tinfo.type) // Requires ggml_type_name in scope
               << ", Offset: " << tinfo.offset
               << ", NumElem: " << tinfo.num_elements
               << ", SizeBytes: " << tinfo.size_in_bytes;
        uintptr_t src_addr = reinterpret_cast<uintptr_t>(gguf.tensor_data.data()) + tinfo.offset;
        uintptr_t src_end_addr = src_addr + tinfo.size_in_bytes;
        uintptr_t data_start_addr = reinterpret_cast<uintptr_t>(gguf.tensor_data.data());
        uintptr_t data_end_addr = data_start_addr + total_data_size;
        ss_log << ", SrcAddr: " << src_addr
               << ", ReadEndAddr: " << src_end_addr
               << ", DataBuffer: [" << data_start_addr << " - " << data_end_addr << "]";
        bool within_bounds = (src_addr >= data_start_addr) && (src_end_addr <= data_end_addr);
        ss_log << ", InBounds: " << (within_bounds ? "YES" : "NO - CRASH LIKELY!");
        Logger::info(ss_log.str());
        if (!within_bounds && tinfo.size_in_bytes > 0) { // Check size > 0 to avoid false alarm on empty tensors
             Logger::error("Tensor '" + name + "' read range [" + std::to_string(tinfo.offset) + ", " 
                         + std::to_string(tinfo.offset + tinfo.size_in_bytes) + ") potentially out of bounds for data size " + std::to_string(total_data_size));
             // Consider throwing an error here if you want to stop immediately on bounds issues
             // throw std::runtime_error("Tensor mapping out of bounds!");
        }
        // --- END: Detailed Tensor Logging --- 

        // Top-level weights
        if (name == "model.embed_tokens.weight" || name == "token_embd.weight") {
            if (tinfo.type == GGMLType::GGML_TYPE_F32) { // Use .type and correct enum name
                assign_vec_f32(model.embed_tokens_f32, tinfo);
                Logger::info("Mapped GGUF tensor '" + name + "' (FP32) to model.embed_tokens_f32");
            } else if (tinfo.type == GGMLType::GGML_TYPE_Q4_K) { // Use .type and correct enum name
                assign_vec_q4k(model.embed_tokens_q4k, tinfo);
                Logger::info("Mapped GGUF tensor '" + name + "' (Q4_K) to model.embed_tokens_q4k");
            } else if (tinfo.type == GGMLType::GGML_TYPE_Q6_K) { // Use .type and correct enum name
                assign_vec_q6k(model.embed_tokens_q6k, tinfo);
                Logger::info("Mapped GGUF tensor '" + name + "' (Q6_K) to model.embed_tokens_q6k");
            } else {
                assign_vec_bf16(model.embed_tokens, tinfo);
                Logger::info("Mapped GGUF tensor '" + name + "' (BF16/Other) to model.embed_tokens"); // Adjusted log
            }
            Logger::info("-> Successfully assigned '" + name + "'"); // Add success log
            continue;
        }
        if (name == "lm_head.weight" || name == "output.weight") {
            if (tinfo.type == GGMLType::GGML_TYPE_F32) { // Use .type and correct enum name
                assign_vec_f32(model.lm_head_f32, tinfo);
                Logger::info("Mapped GGUF tensor '" + name + "' (FP32) to model.lm_head_f32");
            } else if (tinfo.type == GGMLType::GGML_TYPE_Q4_K) { // Use .type and correct enum name
                assign_vec_q4k(model.lm_head_q4k, tinfo);
                Logger::info("Mapped GGUF tensor '" + name + "' (Q4_K) to model.lm_head_q4k");
            } else if (tinfo.type == GGMLType::GGML_TYPE_Q6_K) { // Use .type and correct enum name
                assign_vec_q6k(model.lm_head_q6k, tinfo);
                Logger::info("Mapped GGUF tensor '" + name + "' (Q6_K) to model.lm_head_q6k");
            } else {
                assign_vec_bf16(model.lm_head, tinfo);
                Logger::info("Mapped GGUF tensor '" + name + "' (BF16/Other) to model.lm_head"); // Adjusted log
            }
            Logger::info("-> Successfully assigned '" + name + "'"); // Add success log
            continue;
        }
        if (name == "model.norm.weight" || name == "norm.weight" || name == "output_norm.weight") {
            if (tinfo.type == GGMLType::GGML_TYPE_F32) { // Use .type and correct enum name
                assign_vec_f32(model.final_norm_f32, tinfo);
                Logger::info("Mapped GGUF tensor '" + name + "' (FP32) to model.final_norm_f32");
            } else {
                assign_vec_bf16(model.final_norm, tinfo);
                Logger::info("Mapped GGUF tensor '" + name + "' (BF16/Other) to model.final_norm"); // Adjusted log
            }
            Logger::info("-> Successfully assigned '" + name + "'"); // Add success log
            continue;
        }
        // Per-layer weights - Handle alternative prefixes 'blk.' or 'model.layers.'
        int layer_idx = -1;
        size_t suffix_start_pos = std::string::npos;

        if (name.rfind("model.layers.", 0) == 0) { // Check for "model.layers." prefix
            size_t idx_start = 13;
            size_t idx_end = name.find('.', idx_start);
            if (idx_end != std::string::npos) {
                try {
                    layer_idx = std::stoi(name.substr(idx_start, idx_end - idx_start));
                    suffix_start_pos = idx_end + 1;
                } catch (const std::exception& e) {
                    Logger::warning("Could not parse layer index from GGUF tensor name: " + name);
                }
            }
        } else if (name.rfind("blk.", 0) == 0) { // Check for "blk." prefix
            size_t idx_start = 4; // Length of "blk."
            size_t idx_end = name.find('.', idx_start);
            if (idx_end != std::string::npos) {
                 try {
                    layer_idx = std::stoi(name.substr(idx_start, idx_end - idx_start));
                    suffix_start_pos = idx_end + 1;
                } catch (const std::exception& e) {
                     Logger::warning("Could not parse layer index from GGUF tensor name: " + name);
                }
            }
        }

        if (layer_idx != -1 && suffix_start_pos != std::string::npos && layer_idx >= 0 && layer_idx < nhl) {
            // --- START: Log Layer Index Check ---
            if (layer_idx >= model.get_layers().size()) { // Check bounds BEFORE access
                 Logger::fatal("FATAL: Calculated layer index " + std::to_string(layer_idx) + " is out of bounds for model layers vector size " + std::to_string(model.get_layers().size()) + " for tensor '" + name + "'");
                 // Throw or exit needed here if fatal doesn't exit
                 throw std::out_of_range("Layer index out of bounds in map_gguf_weights");
            }
            // --- END: Log Layer Index Check ---
            std::string suffix = name.substr(suffix_start_pos);
            auto& lw = model.get_layers()[layer_idx];

            // Map known fields based on suffix (the suffix logic remains the same)
            // Q_PROJ
            if (suffix == "self_attn.q_proj.weight" || suffix == "attn.wq.weight") { // Added alternative suffix
                if (tinfo.type == GGMLType::GGML_TYPE_F32) { // Use .type and correct enum name
                    assign_vec_f32(lw.q_proj_f32, tinfo);
                    Logger::info("Mapped GGUF tensor '" + name + "' (FP32) to layers[" + std::to_string(layer_idx) + "].q_proj_f32");
                } else if (tinfo.type == GGMLType::GGML_TYPE_Q4_K) { // Use .type and correct enum name
                    assign_vec_q4k(lw.q_proj_q4k, tinfo);
                    Logger::info("Mapped GGUF tensor '" + name + "' (Q4_K) to layers[" + std::to_string(layer_idx) + "].q_proj_q4k");
                } else if (tinfo.type == GGMLType::GGML_TYPE_Q6_K) { // Use .type and correct enum name
                    assign_vec_q6k(lw.q_proj_q6k, tinfo);
                    Logger::info("Mapped GGUF tensor '" + name + "' (Q6_K) to layers[" + std::to_string(layer_idx) + "].q_proj_q6k");
                } else {
                    assign_vec_bf16(lw.q_proj, tinfo); // REMOVED const_cast
                    Logger::info("Mapped GGUF tensor '" + name + "' (BF16/Other) to layers[" + std::to_string(layer_idx) + "].q_proj");
                }
                Logger::info("-> Successfully assigned '" + name + "' to layer " + std::to_string(layer_idx)); // Add success log
                continue;
            }
            // K_PROJ
            if (suffix == "self_attn.k_proj.weight" || suffix == "attn.wk.weight") { // Added alternative suffix
                if (tinfo.type == GGMLType::GGML_TYPE_F32) { // Use .type and correct enum name
                    assign_vec_f32(lw.k_proj_f32, tinfo);
                    Logger::info("Mapped GGUF tensor '" + name + "' (FP32) to layers[" + std::to_string(layer_idx) + "].k_proj_f32");
                } else if (tinfo.type == GGMLType::GGML_TYPE_Q4_K) { // Use .type and correct enum name
                    assign_vec_q4k(lw.k_proj_q4k, tinfo);
                    Logger::info("Mapped GGUF tensor '" + name + "' (Q4_K) to layers[" + std::to_string(layer_idx) + "].k_proj_q4k");
                } else if (tinfo.type == GGMLType::GGML_TYPE_Q6_K) { // Use .type and correct enum name
                    assign_vec_q6k(lw.k_proj_q6k, tinfo);
                    Logger::info("Mapped GGUF tensor '" + name + "' (Q6_K) to layers[" + std::to_string(layer_idx) + "].k_proj_q6k");
                } else {
                    assign_vec_bf16(lw.k_proj, tinfo); // REMOVED const_cast
                    Logger::info("Mapped GGUF tensor '" + name + "' (BF16/Other) to layers[" + std::to_string(layer_idx) + "].k_proj");
                }
                Logger::info("-> Successfully assigned '" + name + "' to layer " + std::to_string(layer_idx)); // Add success log
                continue;
            }
            // V_PROJ
            if (suffix == "self_attn.v_proj.weight" || suffix == "attn.wv.weight") { // Added alternative suffix
                if (tinfo.type == GGMLType::GGML_TYPE_F32) { // Use .type and correct enum name
                    assign_vec_f32(lw.v_proj_f32, tinfo);
                    Logger::info("Mapped GGUF tensor '" + name + "' (FP32) to layers[" + std::to_string(layer_idx) + "].v_proj_f32");
                } else if (tinfo.type == GGMLType::GGML_TYPE_Q4_K) { // Use .type and correct enum name
                    assign_vec_q4k(lw.v_proj_q4k, tinfo);
                    Logger::info("Mapped GGUF tensor '" + name + "' (Q4_K) to layers[" + std::to_string(layer_idx) + "].v_proj_q4k");
                } else if (tinfo.type == GGMLType::GGML_TYPE_Q6_K) { // Use .type and correct enum name
                    assign_vec_q6k(lw.v_proj_q6k, tinfo);
                    Logger::info("Mapped GGUF tensor '" + name + "' (Q6_K) to layers[" + std::to_string(layer_idx) + "].v_proj_q6k");
                } else {
                    assign_vec_bf16(lw.v_proj, tinfo); // REMOVED const_cast
                    Logger::info("Mapped GGUF tensor '" + name + "' (BF16/Other) to layers[" + std::to_string(layer_idx) + "].v_proj");
                }
                Logger::info("-> Successfully assigned '" + name + "' to layer " + std::to_string(layer_idx)); // Add success log
                continue;
            }
            // O_PROJ
            if (suffix == "self_attn.o_proj.weight" || suffix == "attn.wo.weight") { // Added alternative suffix
                if (tinfo.type == GGMLType::GGML_TYPE_F32) { // Use .type and correct enum name
                    assign_vec_f32(lw.o_proj_f32, tinfo);
                    Logger::info("Mapped GGUF tensor '" + name + "' (FP32) to layers[" + std::to_string(layer_idx) + "].o_proj_f32");
                } else if (tinfo.type == GGMLType::GGML_TYPE_Q4_K) { // Use .type and correct enum name
                    assign_vec_q4k(lw.o_proj_q4k, tinfo);
                    Logger::info("Mapped GGUF tensor '" + name + "' (Q4_K) to layers[" + std::to_string(layer_idx) + "].o_proj_q4k");
                } else if (tinfo.type == GGMLType::GGML_TYPE_Q6_K) { // Use .type and correct enum name
                    assign_vec_q6k(lw.o_proj_q6k, tinfo);
                    Logger::info("Mapped GGUF tensor '" + name + "' (Q6_K) to layers[" + std::to_string(layer_idx) + "].o_proj_q6k");
                } else {
                    assign_vec_bf16(lw.o_proj, tinfo); // REMOVED const_cast
                    Logger::info("Mapped GGUF tensor '" + name + "' (BF16/Other) to layers[" + std::to_string(layer_idx) + "].o_proj");
                }
                Logger::info("-> Successfully assigned '" + name + "' to layer " + std::to_string(layer_idx)); // Add success log
                continue;
            }
            // GATE_PROJ
            if (suffix == "mlp.gate_proj.weight" || suffix == "ffn.w1.weight") { // Added alternative suffix
                if (tinfo.type == GGMLType::GGML_TYPE_F32) { // Use .type and correct enum name
                    assign_vec_f32(lw.gate_proj_f32, tinfo);
                    Logger::info("Mapped GGUF tensor '" + name + "' (FP32) to layers[" + std::to_string(layer_idx) + "].gate_proj_f32");
                } else if (tinfo.type == GGMLType::GGML_TYPE_Q4_K) { // Use .type and correct enum name
                    assign_vec_q4k(lw.gate_proj_q4k, tinfo);
                    Logger::info("Mapped GGUF tensor '" + name + "' (Q4_K) to layers[" + std::to_string(layer_idx) + "].gate_proj_q4k");
                } else if (tinfo.type == GGMLType::GGML_TYPE_Q6_K) { // Use .type and correct enum name
                    assign_vec_q6k(lw.gate_proj_q6k, tinfo);
                    Logger::info("Mapped GGUF tensor '" + name + "' (Q6_K) to layers[" + std::to_string(layer_idx) + "].gate_proj_q6k");
                } else {
                    assign_vec_bf16(lw.gate_proj, tinfo); // REMOVED const_cast
                    Logger::info("Mapped GGUF tensor '" + name + "' (BF16/Other) to layers[" + std::to_string(layer_idx) + "].gate_proj");
                }
                Logger::info("-> Successfully assigned '" + name + "' to layer " + std::to_string(layer_idx)); // Add success log
                continue;
            }
            // UP_PROJ
            if (suffix == "mlp.up_proj.weight" || suffix == "ffn.w3.weight") { // Added alternative suffix
                if (tinfo.type == GGMLType::GGML_TYPE_F32) { // Use .type and correct enum name
                    assign_vec_f32(lw.up_proj_f32, tinfo);
                    Logger::info("Mapped GGUF tensor '" + name + "' (FP32) to layers[" + std::to_string(layer_idx) + "].up_proj_f32");
                } else if (tinfo.type == GGMLType::GGML_TYPE_Q4_K) { // Use .type and correct enum name
                    assign_vec_q4k(lw.up_proj_q4k, tinfo);
                    Logger::info("Mapped GGUF tensor '" + name + "' (Q4_K) to layers[" + std::to_string(layer_idx) + "].up_proj_q4k");
                } else if (tinfo.type == GGMLType::GGML_TYPE_Q6_K) { // Use .type and correct enum name
                    assign_vec_q6k(lw.up_proj_q6k, tinfo);
                    Logger::info("Mapped GGUF tensor '" + name + "' (Q6_K) to layers[" + std::to_string(layer_idx) + "].up_proj_q6k");
                } else {
                    assign_vec_bf16(lw.up_proj, tinfo); // REMOVED const_cast
                    Logger::info("Mapped GGUF tensor '" + name + "' (BF16/Other) to layers[" + std::to_string(layer_idx) + "].up_proj");
                }
                Logger::info("-> Successfully assigned '" + name + "' to layer " + std::to_string(layer_idx)); // Add success log
                continue;
            }
            // DOWN_PROJ
            if (suffix == "mlp.down_proj.weight" || suffix == "ffn.w2.weight") { // Added alternative suffix
                if (tinfo.type == GGMLType::GGML_TYPE_F32) { // Use .type and correct enum name
                    assign_vec_f32(lw.down_proj_f32, tinfo);
                    Logger::info("Mapped GGUF tensor '" + name + "' (FP32) to layers[" + std::to_string(layer_idx) + "].down_proj_f32");
                } else if (tinfo.type == GGMLType::GGML_TYPE_Q4_K) { // Use .type and correct enum name
                    assign_vec_q4k(lw.down_proj_q4k, tinfo);
                    Logger::info("Mapped GGUF tensor '" + name + "' (Q4_K) to layers[" + std::to_string(layer_idx) + "].down_proj_q4k");
                } else if (tinfo.type == GGMLType::GGML_TYPE_Q6_K) { // Use .type and correct enum name
                    assign_vec_q6k(lw.down_proj_q6k, tinfo);
                    Logger::info("Mapped GGUF tensor '" + name + "' (Q6_K) to layers[" + std::to_string(layer_idx) + "].down_proj_q6k");
                } else {
                    assign_vec_bf16(lw.down_proj, tinfo); // REMOVED const_cast
                    Logger::info("Mapped GGUF tensor '" + name + "' (BF16/Other) to layers[" + std::to_string(layer_idx) + "].down_proj");
                }
                Logger::info("-> Successfully assigned '" + name + "' to layer " + std::to_string(layer_idx)); // Add success log
                continue;
            }
            // LAYER NORM (ATTN)
            if (suffix == "input_layernorm.weight" || suffix == "attn_norm.weight") { // Added alternative suffix
                if (tinfo.type == GGMLType::GGML_TYPE_F32) { // Use .type and correct enum name
                    assign_vec_f32(lw.input_layernorm_f32, tinfo);
                    Logger::info("Mapped GGUF tensor '" + name + "' (FP32) to layers[" + std::to_string(layer_idx) + "].input_layernorm_f32");
                } else {
                    assign_vec_bf16(lw.input_layernorm, tinfo); // REMOVED const_cast
                    Logger::info("Mapped GGUF tensor '" + name + "' (BF16/Other) to layers[" + std::to_string(layer_idx) + "].input_layernorm");
                }
                Logger::info("-> Successfully assigned '" + name + "' to layer " + std::to_string(layer_idx)); // Add success log
                continue;
            }
            // LAYER NORM (FFN)
            if (suffix == "post_attention_layernorm.weight" || suffix == "ffn_norm.weight") { // Added alternative suffix
                if (tinfo.type == GGMLType::GGML_TYPE_F32) { // Use .type and correct enum name
                    assign_vec_f32(lw.post_attention_layernorm_f32, tinfo);
                    Logger::info("Mapped GGUF tensor '" + name + "' (FP32) to layers[" + std::to_string(layer_idx) + "].post_attention_layernorm_f32");
                } else {
                    assign_vec_bf16(lw.post_attention_layernorm, tinfo); // REMOVED const_cast
                    Logger::info("Mapped GGUF tensor '" + name + "' (BF16/Other) to layers[" + std::to_string(layer_idx) + "].post_attention_layernorm");
                }
                Logger::info("-> Successfully assigned '" + name + "' to layer " + std::to_string(layer_idx)); // Add success log
                continue;
            }
        }
        Logger::warning("Unmapped GGUF tensor: '" + name + "' with type: " + std::to_string(static_cast<int>(tinfo.type))); // Added type logging
    }
    Logger::info("Finished mapping GGUF weights.");
}

TinyLlamaModel::TinyLlamaModel(const ModelConfig& config, const std::string& weights_path)
    : config_(config) // Initialize with provided config initially
{
    Logger::info("Constructing TinyLlamaModel from path: " + weights_path);
    bool is_gguf = false;
    // Check extension first
    if (weights_path.size() >= 5 && weights_path.substr(weights_path.size() - 5) == ".gguf") {
        is_gguf = true;
    } else {
        // Check file magic
        std::ifstream file(weights_path, std::ios::binary);
        if (file.is_open()) {
            uint32_t magic = 0;
            file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
            if (magic == GGUF_MAGIC) {
                is_gguf = true;
            }
        }
    }

    if (is_gguf) {
        Logger::info("Detected GGUF file. Loading metadata and mapping weights...");
        gguf_data_ = std::make_unique<GGUFData>(load_gguf_meta(weights_path));
        config_ = parse_model_config_from_gguf(*gguf_data_); // OVERWRITE config_ with GGUF metadata
        // Note: layers vector is resized inside initialize_weights
        initialize_weights(nullptr, gguf_data_.get()); // Map weights from GGUF data
        Logger::info("GGUF weights mapped.");
        initialize_gpu_and_rope(); // Initialize GPU resources and RoPE
    } else {
        Logger::info("Detected non-GGUF file. Loading with SafeTensors loader...");
        SafeTensorsLoader loader(weights_path);
        // config_ remains as initially provided
        initialize_weights(&loader, nullptr); // Load weights from safetensors
        Logger::info("SafeTensors weights loaded.");
        // REMOVED: *this = TinyLlamaModel(config, loader);
        initialize_gpu_and_rope(); // Initialize GPU resources and RoPE
    }

    // Initialize GPU resources and RoPE after weights are loaded/mapped - REMOVED FROM HERE
    // initialize_gpu_and_rope(); 

    Logger::info("TinyLlamaModel construction from path complete.");
}

ModelConfig parse_model_config_from_gguf(const GGUFData& gguf) {
    // --- START: Log all available metadata keys --- 
    Logger::info("--- GGUF Metadata Keys Found ---");
    if (gguf.metadata.empty()) {
        Logger::info("(No metadata keys found in GGUF data)");
    } else {
        std::stringstream keys_ss;
        bool first_key = true;
        for (const auto& pair : gguf.metadata) {
            if (!first_key) keys_ss << ", ";
            keys_ss << "'" << pair.first << "'";
            first_key = false;
        }
        Logger::info(keys_ss.str());
    }
    Logger::info("---------------------------------");
    // --- END: Log all available metadata keys ---

    ModelConfig cfg;
    // Helper lambda to extract a value from metadata
    auto get = [&](const std::string& key, auto default_val) -> decltype(default_val) {
        using T = decltype(default_val);
        auto it = gguf.metadata.find(key);

        // Special handling for general.file_type (often stored as uint32 enum)
        if constexpr (std::is_same_v<T, std::string>) {
            if (key == "general.file_type") {
                if (it != gguf.metadata.end()) {
                    if (std::holds_alternative<std::string>(it->second)) {
                        return std::get<std::string>(it->second); // Already a string, return it
                    } else if (std::holds_alternative<uint32_t>(it->second)) {
                        // It's a uint32_t, try to map it to a string based on GGMLType
                        uint32_t type_enum = std::get<uint32_t>(it->second);
                        switch (static_cast<GGMLType>(type_enum)) {
                            case GGML_TYPE_F32:  return std::string("float32");
                            case GGML_TYPE_F16:  return std::string("float16");
                            case GGML_TYPE_Q4_0: return std::string("q4_0");
                            case GGML_TYPE_Q4_1: return std::string("q4_1");
                            case GGML_TYPE_Q5_0: return std::string("q5_0");
                            case GGML_TYPE_Q5_1: return std::string("q5_1");
                            case GGML_TYPE_Q8_0: return std::string("q8_0");
                            case GGML_TYPE_Q8_1: return std::string("q8_1");
                            case GGML_TYPE_Q2_K: return std::string("q2_k");
                            case GGML_TYPE_Q3_K: return std::string("q3_k");
                            case GGML_TYPE_Q4_K: return std::string("q4_k");
                            case GGML_TYPE_Q5_K: return std::string("q5_k");
                            case GGML_TYPE_Q6_K: return std::string("q6_k");
                            case GGML_TYPE_Q8_K: return std::string("q8_k");
                            case GGML_TYPE_I8:   return std::string("int8");
                            case GGML_TYPE_I16:  return std::string("int16");
                            case GGML_TYPE_I32:  return std::string("int32");
                            // Add BF16 mapping if needed (not standard GGMLType enum)
                            // case SOME_BF16_ENUM_VALUE: return std::string("bfloat16");
                            default:
                                Logger::warning("GGUF metadata key 'general.file_type' had unknown uint32 value: " + std::to_string(type_enum) + ". Using default.");
                                break; // Fall through to return default_val
                        }
                    } else {
                         Logger::warning("GGUF metadata key 'general.file_type' had unexpected type index: " + std::to_string(it->second.index()) + ". Using default.");
                    }
                } else {
                     Logger::warning("GGUF metadata missing key: 'general.file_type'. Using default.");
                }
                 // If we reach here, either key was missing, type was wrong, or uint32 mapping failed
                 return default_val; // Return default string
            }
        } // End of special handling for general.file_type

        // --- START: Debug log for specific key type --- // REMOVED OLD DEBUG LOG
        // --- END: Debug log for specific key type ---

        if (it == gguf.metadata.end()) {
            // Adjust logging based on type T
            if constexpr (std::is_same_v<T, std::string>) {
                 Logger::warning("GGUF metadata missing key: '" + key + "', using default: "" + default_val + """);
            } else {
                 Logger::warning("GGUF metadata missing key: '" + key + "', using default: " + std::to_string(default_val));
            }
            return default_val;
        }

        try {
            if constexpr (std::is_same_v<T, int>) {
                if (std::holds_alternative<int32_t>(it->second)) return (int)std::get<int32_t>(it->second);
                if (std::holds_alternative<uint32_t>(it->second)) return (int)std::get<uint32_t>(it->second);
                if (std::holds_alternative<int64_t>(it->second)) return (int)std::get<int64_t>(it->second);
                if (std::holds_alternative<uint64_t>(it->second)) return (int)std::get<uint64_t>(it->second);
            } else if constexpr (std::is_same_v<T, float>) {
                if (std::holds_alternative<float>(it->second)) return std::get<float>(it->second);
                if (std::holds_alternative<double>(it->second)) return (float)std::get<double>(it->second);
            } else if constexpr (std::is_same_v<T, std::string>) {
                if (std::holds_alternative<std::string>(it->second)) return std::get<std::string>(it->second);
            }
        } catch (const std::exception& e) {
            Logger::warning("GGUF metadata key '" + key + "' type mismatch: " + e.what());
        }
        // Adjust logging based on type T
        if constexpr (std::is_same_v<T, std::string>) {
            Logger::warning("GGUF metadata key '" + key + "' has unexpected type or error, using default: \"" + default_val + "\"");
        } else {
            Logger::warning("GGUF metadata key '" + key + "' has unexpected type or error, using default: " + std::to_string(default_val));
        }
        return default_val;
    };

    // Map GGUF keys to ModelConfig fields - USE CORRECTED KEYS
    cfg.hidden_size = get("llama.embedding_length", 0);
    cfg.intermediate_size = get("llama.feed_forward_length", 0);
    cfg.num_attention_heads = get("llama.attention.head_count", 0); // Corrected key
    cfg.num_key_value_heads = get("llama.attention.head_count_kv", 0); // Corrected key
    cfg.num_hidden_layers = get("llama.block_count", 0);
    // cfg.vocab_size = get("general.vocab_size", 0); // OLD, INCORRECT WAY

    // --- START: Get vocab size from tokenizer.ggml.vocab array ---
    int determined_vocab_size = 0; // Default to 0 if not found/invalid
    auto vocab_it = gguf.metadata.find("tokenizer.ggml.tokens"); // CORRECTED KEY HERE
    if (vocab_it != gguf.metadata.end()) {
        const auto& metadata_value = vocab_it->second;
        if (std::holds_alternative<GGUFArray>(metadata_value)) {
            const GGUFArray& vocab_array = std::get<GGUFArray>(metadata_value);
            // We expect an array of strings
            if (vocab_array.type == GGUFValueType::STRING) {
                 determined_vocab_size = static_cast<int>(vocab_array.len);
                 Logger::info("Determined vocab_size from 'tokenizer.ggml.tokens' array: " + std::to_string(determined_vocab_size)); // CORRECTED KEY IN LOG
            } else {
                 Logger::fatal("GGUF metadata key 'tokenizer.ggml.tokens' is an array, but not of strings (type=" + std::to_string(static_cast<int>(vocab_array.type)) + "). Cannot determine vocab size."); // CORRECTED KEY IN LOG
            }
        } else {
             Logger::fatal("GGUF metadata key 'tokenizer.ggml.tokens' exists but is not an array (type index=" + std::to_string(metadata_value.index()) + "). Cannot determine vocab size."); // CORRECTED KEY IN LOG
        }
    } else {
         Logger::fatal("GGUF metadata key 'tokenizer.ggml.tokens' not found. Cannot determine vocab size."); // CORRECTED KEY IN LOG
         // Maybe try the old key as a last resort? Or just fail hard? For now, fail hard.
         // int fallback_vocab_size = get("general.vocab_size", 0);
         // if (fallback_vocab_size > 0) {
         //     Logger::warning("Falling back to 'general.vocab_size': " + std::to_string(fallback_vocab_size));
         //     determined_vocab_size = fallback_vocab_size;
         // }
    }
    cfg.vocab_size = determined_vocab_size;
    // --- END: Get vocab size from tokenizer.ggml.vocab array ---

    cfg.max_position_embeddings = get("llama.context_length", 0);
    cfg.rms_norm_eps = get("llama.attention.layer_norm_rms_epsilon", 1e-5f); // Corrected key
    cfg.rope_theta = get("llama.rope.freq_base", 10000.0f); // Corrected key
    cfg.hidden_act = get("llama.activation_function", std::string("silu")); // Still missing - uses default
    cfg.torch_dtype = get("general.file_type", std::string("bfloat16")); // *** This call now uses the updated get lambda ***
    cfg.bos_token_id = get("tokenizer.ggml.bos_token_id", 1); // Corrected key
    cfg.eos_token_id = get("tokenizer.ggml.eos_token_id", 2); // Corrected key
    return cfg;
}
