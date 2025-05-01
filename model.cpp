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
#include <fstream> // ADDED for std::ifstream
#include <memory>   // ADDED for std::make_unique
#include <limits>   // ADDED for std::numeric_limits
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

// --- START: Forward Declarations for Helpers ---
// Needed if definitions appear after their first use (e.g., in TinyLlamaModel::forward)

// Forward declaration for Q6K matvec helper
static void matvec_q6k_f32_vector_cpu(const std::vector<block_q6_K>& mat_q6k,
                                  const std::vector<float>& vec_f32,
                                  std::vector<float>& out_f32,
                                  int rows, int cols,
                                  bool log_first_block = false);

// Forward declaration for Q4K matvec helper
static void matvec_q4k_f32_vector_cpu(const std::vector<block_q4_K>& mat_q4k,
                                  const std::vector<float>& vec_f32,
                                  std::vector<float>& out_f32,
                                  int rows, int cols,
                                  bool log_first_block = false);

// Forward declaration for float32_to_bfloat16 helper
inline uint16_t float32_to_bfloat16(float val);

// --- ADDED: Forward declaration for F32 matvec helper ---
static void matvec_f32_f32_vector_cpu(const std::vector<float>& mat_f32,
                                  const std::vector<float>& vec_f32,
                                  std::vector<float>& out_f32,
                                  int rows, int cols);
// --- END ADDED ---

// --- END: Forward Declarations for Helpers ---

// --- START: Implementation for float32_to_bfloat16 ---
// Convert float32 to bfloat16
// Note: This is a basic implementation and might not handle all edge cases like NaN/Inf correctly depending on the compiler/platform.
// Refined version based on common practices:
inline uint16_t float32_to_bfloat16(float val) {
    uint32_t bits;
    std::memcpy(&bits, &val, sizeof(float));
    // Add half of the least significant bit of the mantissa before truncating
    // This helps with rounding.
    bits += 0x7FFF + ((bits >> 16) & 1); // Round to nearest or even
    return static_cast<uint16_t>(bits >> 16);
}
// --- END: Implementation for float32_to_bfloat16 ---

// --- START: Implementation for matvec_q6k_f32_vector_cpu ---
// Perform matrix-vector multiplication: out = mat * vec
// mat: Q6_K format (vector of blocks), vec: FP32, out: FP32
// MODIFIED: Added log_first_block parameter
static void matvec_q6k_f32_vector_cpu(const std::vector<block_q6_K>& mat_q6k,
                                  const std::vector<float>& vec_f32,
                                  std::vector<float>& out_f32,
                                  int rows, int cols,
                                  bool log_first_block) { // REMOVED: Default = false
    if (cols % GGML_QK_K != 0) {
        throw std::runtime_error("matvec_q6k_f32_vector_cpu: cols (" + std::to_string(cols) + ") must be divisible by GGML_QK_K (" + std::to_string(GGML_QK_K) + ")");
    }
    if (vec_f32.size() != cols) {
        throw std::runtime_error("matvec_q6k_f32_vector_cpu: vec_f32 size mismatch. Expected " + std::to_string(cols) + ", got " + std::to_string(vec_f32.size()));
    }
    size_t num_blocks_per_row = cols / GGML_QK_K;
    size_t total_blocks_expected = (size_t)rows * num_blocks_per_row;
    if (mat_q6k.size() != total_blocks_expected) {
         throw std::runtime_error("matvec_q6k_f32_vector_cpu: mat_q6k size mismatch. Expected " + std::to_string(total_blocks_expected) + " blocks, got " + std::to_string(mat_q6k.size()));
    }

    out_f32.resize(rows);
    float dequantized_block[GGML_QK_K]; // Temporary buffer for one dequantized block

    #pragma omp parallel for private(dequantized_block) // Ensure each thread has its own buffer
    for (int r = 0; r < rows; ++r) {
        double row_sum = 0.0;
        double kahan_c = 0.0; // Kahan summation compensation for this row

        size_t block_row_offset = r * num_blocks_per_row;

        for (size_t block_col_idx = 0; block_col_idx < num_blocks_per_row; ++block_col_idx) {
            // Dequantize the current block
            const block_q6_K* qblock = &mat_q6k[block_row_offset + block_col_idx];
            // --- MODIFIED: Enable logging for the very first block (row 0, col 0) if requested --- 
            bool enable_dequant_log = log_first_block && (r == 0 && block_col_idx == 0);
            // dequantize_q6_k(qblock, dequantized_block, GGML_QK_K, enable_dequant_log); // Original with flag
            dequantize_q6_k(qblock, dequantized_block, GGML_QK_K); // Reverted: Remove boolean flag
            // --- END MODIFICATION ---

            // Calculate dot product for this block
            size_t vec_offset = block_col_idx * GGML_QK_K;
            for (int i = 0; i < GGML_QK_K; ++i) {
                 double term = static_cast<double>(dequantized_block[i]) * static_cast<double>(vec_f32[vec_offset + i]);
                 // Kahan sum
                 double y = term - kahan_c;
                 double t = row_sum + y;
                 kahan_c = (t - row_sum) - y;
                 row_sum = t;
            }
        }
        out_f32[r] = static_cast<float>(row_sum);
    }
}
// --- END: Implementation for matvec_q6k_f32_vector_cpu ---

// --- START: Implementation for matvec_q4k_f32_vector_cpu ---
// Perform matrix-vector multiplication: out = mat * vec
// mat: Q4_K format (vector of blocks), vec: FP32, out: FP32
// MODIFIED: Added log_first_block parameter
static void matvec_q4k_f32_vector_cpu(const std::vector<block_q4_K>& mat_q4k,
                                  const std::vector<float>& vec_f32,
                                  std::vector<float>& out_f32,
                                  int rows, int cols,
                                  bool log_first_block) { // REMOVED: Default = false
    if (cols % GGML_QK_K != 0) {
        throw std::runtime_error("matvec_q4k_f32_vector_cpu: cols (" + std::to_string(cols) + ") must be divisible by GGML_QK_K (" + std::to_string(GGML_QK_K) + ")");
    }
    if (vec_f32.size() != cols) {
        throw std::runtime_error("matvec_q4k_f32_vector_cpu: vec_f32 size mismatch. Expected " + std::to_string(cols) + ", got " + std::to_string(vec_f32.size()));
    }
    size_t num_blocks_per_row = cols / GGML_QK_K;
    size_t total_blocks_expected = (size_t)rows * num_blocks_per_row;
    if (mat_q4k.size() != total_blocks_expected) {
         throw std::runtime_error("matvec_q4k_f32_vector_cpu: mat_q4k size mismatch. Expected " + std::to_string(total_blocks_expected) + " blocks, got " + std::to_string(mat_q4k.size()));
    }

    out_f32.resize(rows);
    float dequantized_block[GGML_QK_K]; // Temporary buffer for one dequantized block

    #pragma omp parallel for private(dequantized_block) // Ensure each thread has its own buffer
    for (int r = 0; r < rows; ++r) {
        double row_sum = 0.0;
        double kahan_c = 0.0; // Kahan summation compensation for this row

        size_t block_row_offset = r * num_blocks_per_row;

        for (size_t block_col_idx = 0; block_col_idx < num_blocks_per_row; ++block_col_idx) {
            // Dequantize the current block
            const block_q4_K* qblock = &mat_q4k[block_row_offset + block_col_idx];
            // --- MODIFIED: Enable logging for the very first block (row 0, col 0) if requested --- 
            bool enable_dequant_log = log_first_block && (r == 0 && block_col_idx == 0);
            dequantize_q4_k_m(qblock, dequantized_block, GGML_QK_K, enable_dequant_log);
            // --- END MODIFICATION ---

            // Calculate dot product for this block
            size_t vec_offset = block_col_idx * GGML_QK_K;
            for (int i = 0; i < GGML_QK_K; ++i) {
                 double term = static_cast<double>(dequantized_block[i]) * static_cast<double>(vec_f32[vec_offset + i]);
                 // Kahan sum
                 double y = term - kahan_c;
                 double t = row_sum + y;
                 kahan_c = (t - row_sum) - y;
                 row_sum = t;
            }
        }
        out_f32[r] = static_cast<float>(row_sum);
    }
}
// --- END: Implementation for matvec_q4k_f32_vector_cpu ---

// --- ADDED: CPU Matrix-Vector Multiplication (F32 x F32 -> F32) ---
static void matvec_f32_f32_vector_cpu(const std::vector<float>& mat_f32,
                                  const std::vector<float>& vec_f32,
                                  std::vector<float>& out_f32,
                                  int rows, int cols)
{
    // Basic validation
    if (mat_f32.empty() || vec_f32.empty()) {
        Logger::error("matvec_f32_f32_vector_cpu: Input matrix or vector is empty.");
        out_f32.assign(rows, 0.0f); // Assign zeros to output
        return;
    }
    if (mat_f32.size() != (size_t)rows * cols) {
        Logger::error("matvec_f32_f32_vector_cpu: Matrix dimensions mismatch. Expected "
                      + std::to_string((size_t)rows * cols) + ", got " + std::to_string(mat_f32.size()));
        out_f32.assign(rows, 0.0f);
        return;
    }
    if (vec_f32.size() != (size_t)cols) {
         Logger::error("matvec_f32_f32_vector_cpu: Vector dimension mismatch. Expected "
                      + std::to_string(cols) + ", got " + std::to_string(vec_f32.size()));
        out_f32.assign(rows, 0.0f);
        return;
    }

    out_f32.resize(rows); // Ensure output vector has correct size

    #pragma omp parallel for schedule(static) // Parallelize the outer loop over rows
    for(int r = 0; r < rows; ++r) {
        float sum = 0.0f;
        size_t row_offset = (size_t)r * cols;
        // Use pointer arithmetic for potential inner loop optimization
        const float* mat_row_ptr = mat_f32.data() + row_offset;
        const float* vec_ptr = vec_f32.data();

        // Simple dot product - potentially optimize with SIMD later if needed
        // Kahan summation might be overkill for F32*F32 but could be added if precision issues arise.
        for(int c = 0; c < cols; ++c) {
            sum += mat_row_ptr[c] * vec_ptr[c];
        }
        out_f32[r] = sum;
    }
}
// --- END ADDED ---

void log_vector_summary(const std::string& name, const std::vector<float>& v, int head_count) {
    if (v.empty()) {
        Logger::info(name + ": EMPTY");
        return;
    }
    std::stringstream ss;
    ss << name << ": size=" << v.size() << ", first " << std::min(v.size(), (size_t)head_count) << ": ["; // Removed (int) cast, cast head_count to size_t
    for(size_t i = 0; i < std::min(v.size(), (size_t)head_count); ++i) { // Use size_t for loop, cast head_count
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
    // +++ START ADDED LOGGING +++
    auto max_it = std::max_element(v.begin(), v.end());
    float max_val = *max_it;
    int max_idx = std::distance(v.begin(), max_it);
    Logger::debug("[ARGMAX HELPER] Max value found: " + std::to_string(max_val) + " at index: " + std::to_string(max_idx));
    // +++ END ADDED LOGGING +++
    return max_idx; // Return the calculated index
    // return std::distance(v.begin(), std::max_element(v.begin(), v.end())); // Original return
}
// --- END: Argmax Helper ---

// mat: [M, N], vec: [N] -> out: [M]
// torch::Tensor matvec(const torch::Tensor& mat, const torch::Tensor& vec) {
//     return torch::matmul(mat, vec);
// }

// --- START: C++ Vector RMSNorm CPU Implementation ---
// Renamed from rmsnorm_vector, removed internal #ifdef
static void rmsnorm_vector_cpu(const std::vector<float>& x, const std::vector<float>& weight, std::vector<float>& out, float eps) {
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
}
// --- END: C++ Vector RMSNorm CPU Implementation ---

// --- START: C++ Vector Softmax CPU Implementation ---
// Renamed from softmax_vector, removed internal #ifdef
static void softmax_vector_cpu(const std::vector<float>& x, std::vector<float>& out) {
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
}
// --- END: C++ Vector Softmax CPU Implementation ---

// C++ Vector-based SiLU activation CPU Implementation: y = x * sigmoid(x)
// Renamed from silu, removed internal #ifdef
static void silu_cpu(const std::vector<float>& x, std::vector<float>& out) {
    // Logger::info("Using CPU SiLU (OpenMP)"); // Optional log
    // Original OpenMP implementation
    if (x.size() != out.size()) out.resize(x.size()); // Ensure output is sized correctly
    #pragma omp parallel for
    for (size_t i = 0; i < x.size(); ++i) {
        float sigmoid_x = 1.0f / (1.0f + std::exp(-x[i])); // Calculate sigmoid explicitly
        out[i] = x[i] * sigmoid_x; // Multiply
    }
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
    for (int t = 0; t < num_tokens; ++t) { // Rewritten loop with braces
        for (int h = 0; h < hidden_size; ++h) {
            result[t][h] = flat[t * hidden_size + h];
        }
    }
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

    // --- REVERT: Always Allocate CPU Host Vectors --- 
    Logger::info("Allocating KVCache host vectors..."); // Simplified message
    size_t cache_size_per_layer = static_cast<size_t>(max_seq_len) * 
                                 static_cast<size_t>(num_kv_heads) * 
                                 static_cast<size_t>(head_dim);

    if (cache_size_per_layer == 0 && max_seq_len > 0) { // Allow 0 if max_seq_len is 0
        throw std::runtime_error("KVCache (CPU): Calculated cache size is zero. Check parameters.");
    }
    
    for (int l = 0; l < num_layers; ++l) {
        try {
            layers[l].k.assign(cache_size_per_layer, 0.0f); // Use assign for resize + fill
            layers[l].v.assign(cache_size_per_layer, 0.0f);
        } catch (const std::bad_alloc& e) {
            Logger::error("Failed to allocate CPU KVCache for layer " + std::to_string(l) + ": " + e.what());
            throw; // Re-throw after logging
        }
    }
    Logger::info("KVCache (CPU) vectors allocated for " + std::to_string(num_layers) + " layers.");

#ifdef HAS_CUDA
    // --- CUDA Path (Allocate GPU memory) --- 
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
    // --- CPU Path Log (No allocation here if #ifndef above was removed) ---
     Logger::info("KVCache (CPU-only build) initialized with dimensions: " + 
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

// --- START: C++ Vector MatVec BF16 * F32 -> F32 CPU Implementation ---
// Renamed from matvec_bf16_f32_vector, removed internal #ifdef
static void matvec_bf16_f32_vector_cpu(const std::vector<uint16_t>& mat_bf16,
                                   const std::vector<float>& vec_f32,
                                   std::vector<float>& out_f32,
                                   int rows, int cols) {
    // Logger::info("Using CPU MatVec (BF16*F32->F32, OpenMP+Kahan)"); // Optional log
    // Original OpenMP + Kahan implementation
    if (mat_bf16.size() != (size_t)rows * cols || vec_f32.size() != (size_t)cols) {
        Logger::error("matvec_bf16_f32_vector_cpu: Size mismatch. Mat: " + std::to_string(mat_bf16.size()) +
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
}
// --- END: C++ Vector MatVec BF16 CPU Implementation ---

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

    // --- START: Allocate Persistent Workspace Buffers ---
#ifdef HAS_CUDA
    Logger::info("Allocating persistent GPU workspace buffers...");
    size_t hs_bytes = (size_t)hs * sizeof(float);
    size_t is_bytes = (size_t)is * sizeof(float);
    size_t vs_bytes = (size_t)vs * sizeof(float);
    size_t kv_head_bytes = (size_t)n_kv_heads * head_dim * sizeof(float); // Size for K or V for current token

    gpuErrchk(cudaMalloc(&x_dev_, hs_bytes));
    gpuErrchk(cudaMalloc(&x_norm_dev_, hs_bytes));
    gpuErrchk(cudaMalloc(&x_resid1_dev_, hs_bytes));
    gpuErrchk(cudaMalloc(&x_resid2_dev_, hs_bytes));
    gpuErrchk(cudaMalloc(&q_dev_, hs_bytes));
    gpuErrchk(cudaMalloc(&k_dev_, kv_head_bytes)); // Only need space for current token K
    gpuErrchk(cudaMalloc(&v_dev_, kv_head_bytes)); // Only need space for current token V
    gpuErrchk(cudaMalloc(&attn_out_dev_, hs_bytes));
    gpuErrchk(cudaMalloc(&attn_proj_dev_, hs_bytes));
    gpuErrchk(cudaMalloc(&gate_vec_dev_, is_bytes));
    gpuErrchk(cudaMalloc(&up_vec_dev_, is_bytes));
    gpuErrchk(cudaMalloc(&swiglu_vec_dev_, is_bytes));
    gpuErrchk(cudaMalloc(&mlp_down_dev_, hs_bytes));
    gpuErrchk(cudaMalloc(&logits_dev_, vs_bytes));
    Logger::info("Finished allocating persistent GPU workspace buffers.");
#endif
    // --- END: Allocate Persistent Workspace Buffers ---

    Logger::info("Finished initializing GPU resources and RoPE.");
}
// --- END: Private Helper: Initialize GPU Resources & RoPE ---


// TinyLlamaModel constructor: from config and safetensors loader
TinyLlamaModel::TinyLlamaModel(const ModelConfig& config, const SafeTensorsLoader& loader)
    : config_(config) // Initialize config
{
    Logger::info("Constructing TinyLlamaModel from SafeTensorsLoader.");
    initialize_weights(&loader, nullptr); // Load weights from loader
#ifdef HAS_CUDA
    initialize_gpu_and_rope();            // Initialize GPU resources and RoPE
#endif
    Logger::info("TinyLlamaModel construction from SafeTensorsLoader complete.");
}

// TinyLlamaModel constructor: from config and weights path (handles GGUF/SafeTensors)
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
            // Need to include <fstream> for this check
        } else {
             Logger::warning("Could not open weights file to check magic number: " + weights_path);
        }
    }

    if (is_gguf) {
        Logger::info("Detected GGUF file. Loading metadata and mapping weights...");
        // Need to include <memory> for std::make_unique
        gguf_data_ = std::make_unique<GGUFData>(load_gguf_meta(weights_path));
        config_ = parse_model_config_from_gguf(*gguf_data_); // OVERWRITE config_ with GGUF metadata
        // Note: layers vector is resized inside initialize_weights
        initialize_weights(nullptr, gguf_data_.get()); // Map weights from GGUF data
        Logger::info("GGUF weights mapped.");
#ifdef HAS_CUDA
        initialize_gpu_and_rope(); // Initialize GPU resources and RoPE
#endif
    } else {
        Logger::info("Detected non-GGUF file (or failed GGUF check). Loading with SafeTensors loader...");
        // Need SafeTensorsLoader defined/included
        SafeTensorsLoader loader(weights_path);
        // config_ remains as initially provided (or potentially parsed from a separate config file if needed)
        initialize_weights(&loader, nullptr); // Load weights from safetensors
        Logger::info("SafeTensors weights loaded.");
#ifdef HAS_CUDA
        initialize_gpu_and_rope(); // Initialize GPU resources and RoPE
#endif
    }
    Logger::info("TinyLlamaModel construction from path complete.");
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

    // --- START: Free Persistent Workspace Buffers ---
    // Add null checks before freeing, just in case allocation failed
    if (x_dev_) { gpuErrchk(cudaFree(x_dev_)); x_dev_ = nullptr; }
    if (x_norm_dev_) { gpuErrchk(cudaFree(x_norm_dev_)); x_norm_dev_ = nullptr; }
    if (x_resid1_dev_) { gpuErrchk(cudaFree(x_resid1_dev_)); x_resid1_dev_ = nullptr; }
    if (x_resid2_dev_) { gpuErrchk(cudaFree(x_resid2_dev_)); x_resid2_dev_ = nullptr; }
    if (q_dev_) { gpuErrchk(cudaFree(q_dev_)); q_dev_ = nullptr; }
    if (k_dev_) { gpuErrchk(cudaFree(k_dev_)); k_dev_ = nullptr; }
    if (v_dev_) { gpuErrchk(cudaFree(v_dev_)); v_dev_ = nullptr; }
    if (attn_out_dev_) { gpuErrchk(cudaFree(attn_out_dev_)); attn_out_dev_ = nullptr; }
    if (attn_proj_dev_) { gpuErrchk(cudaFree(attn_proj_dev_)); attn_proj_dev_ = nullptr; }
    if (gate_vec_dev_) { gpuErrchk(cudaFree(gate_vec_dev_)); gate_vec_dev_ = nullptr; }
    if (up_vec_dev_) { gpuErrchk(cudaFree(up_vec_dev_)); up_vec_dev_ = nullptr; }
    if (swiglu_vec_dev_) { gpuErrchk(cudaFree(swiglu_vec_dev_)); swiglu_vec_dev_ = nullptr; }
    if (mlp_down_dev_) { gpuErrchk(cudaFree(mlp_down_dev_)); mlp_down_dev_ = nullptr; }
    if (logits_dev_) { gpuErrchk(cudaFree(logits_dev_)); logits_dev_ = nullptr; }
    Logger::info("Freed persistent GPU workspace buffers.");
    // --- END: Free Persistent Workspace Buffers ---


    Logger::info("Finished freeing TinyLlamaModel CUDA weight memory.");
#endif
} 
// --- END: TinyLlamaModel Destructor Definition ---

// --- Lookup Embedding (Handles Q4_K, F32, BF16) ---
// REPLACED ORIGINAL FUNCTION
std::vector<float> TinyLlamaModel::lookup_embedding(int token_id) {
    int hs = config_.hidden_size;
    int vs = config_.vocab_size;
    bool log_initial = (token_id == config_.bos_token_id); // Simple way to log for first token usually

    if (token_id < 0 || token_id >= vs) {
        Logger::error("Token ID out of bounds in lookup_embedding: " + std::to_string(token_id));
        return std::vector<float>(hs, 0.0f); // Return a zero vector
    }

    std::vector<float> embedding_vec(hs, 0.0f); // Initialize with zeros

    // 1. Prioritize Q4_K if available
    if (!embed_tokens_q4k.empty()) {
        // Logger::info("[CPU_EMBED] Using Q4_K embedding table for token: " + std::to_string(token_id)); // Optional log
        if (hs % GGML_QK_K != 0) {
             Logger::error("Hidden size (" + std::to_string(hs) + ") is not divisible by GGML_QK_K (" + std::to_string(GGML_QK_K) + ") for Q4_K embedding lookup.");
             return embedding_vec; // Return zeros
        }

        size_t blocks_per_row = hs / GGML_QK_K;
        size_t start_block_idx = (size_t)token_id * blocks_per_row;
        size_t end_block_idx = start_block_idx + blocks_per_row;

        if (end_block_idx > embed_tokens_q4k.size()) {
            Logger::error("Calculated block index out of bounds for Q4_K embedding table. Token: " + std::to_string(token_id) + ", StartBlock: " + std::to_string(start_block_idx) + ", EndBlock: " + std::to_string(end_block_idx) + ", TableSize: " + std::to_string(embed_tokens_q4k.size()));
            return embedding_vec; // Return zeros
        }

        float dequantized_block[GGML_QK_K]; // Temporary buffer for one block

        for (size_t block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
            const block_q4_K* qblock = &embed_tokens_q4k[start_block_idx + block_idx];
            // --- MODIFICATION: Enable logging for the first block of the embedding ---
            bool log_this_embedding_block = log_initial && (block_idx == 0);
            dequantize_q4_k_m(qblock, dequantized_block, GGML_QK_K, log_this_embedding_block); // Pass flag
            // --- END MODIFICATION ---
            
            // Copy dequantized data into the correct slice of the output vector
            std::memcpy(embedding_vec.data() + block_idx * GGML_QK_K,
                        dequantized_block,
                        GGML_QK_K * sizeof(float));
        }
        // --- ADDED: Logging after Q4_K embedding ---
        if (log_initial) {
            log_vector_summary("[CPU_EMBED Q4_K] Output Embedding (Token " + std::to_string(token_id) + ")", embedding_vec);
        }
        // --- END: Logging ---
        return embedding_vec; // Return the dequantized vector

    // 2. Check F32 if Q4_K is empty
    } else if (!embed_tokens_f32.empty()) {
        // Logger::info("[CPU_EMBED] Using F32 embedding table for token: " + std::to_string(token_id)); // Optional log
        size_t offset = (size_t)token_id * hs;
        if (offset + hs > embed_tokens_f32.size()) {
             Logger::error("Embedding offset out of bounds in F32 lookup for token: " + std::to_string(token_id));
             return embedding_vec; // Return zeros initialized earlier
        }
        // Directly copy the slice
        std::copy(embed_tokens_f32.begin() + offset, embed_tokens_f32.begin() + offset + hs, embedding_vec.begin());
         if (log_initial) {
             log_vector_summary("[CPU_EMBED F32] Output Embedding (Token " + std::to_string(token_id) + ")", embedding_vec);
         }
        return embedding_vec;

    // 3. Fallback to BF16 if F32 is also empty
    } else if (!embed_tokens.empty()) {
        // Logger::info("[CPU_EMBED] Using BF16 embedding table for token: " + std::to_string(token_id)); // Optional log
        size_t offset = (size_t)token_id * hs;
        if (offset + hs > embed_tokens.size()) {
             Logger::error("Embedding offset out of bounds in BF16 lookup for token: " + std::to_string(token_id));
             return embedding_vec; // Return zeros initialized earlier
        }
        // Create a sub-vector view (or copy)
        std::vector<uint16_t> token_embedding_bf16(embed_tokens.begin() + offset, embed_tokens.begin() + offset + hs);
        // Convert the bfloat16 vector slice to a float32 vector (fills embedding_vec)
        embedding_vec = bf16vec_to_float_vec(token_embedding_bf16);
         if (log_initial) {
             log_vector_summary("[CPU_EMBED BF16] Output Embedding (Token " + std::to_string(token_id) + ")", embedding_vec);
         }
         return embedding_vec;

    // 4. Error if all tables are empty
    } else {
        Logger::error("No valid embedding table found (Q4_K, F32, BF16) for token: " + std::to_string(token_id));
        // Return the zero vector initialized earlier
        return embedding_vec;
    }
}

// --- Forward Pass (CPU Implementation Only) --- 
// Renamed helper calls to use _cpu suffix
std::vector<float> TinyLlamaModel::forward(std::vector<float>& input, int n_tokens, KVCache* kv_cache, const std::vector<int>* input_ids) {
    bool log_initial = true; // Set to false to disable initial layer logging
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
    bool log_this_step = (n_tokens < 2); // Log first few positions
    if (log_this_step) {
        Logger::info("[CPU_FWD] forward called. n_tokens=" + std::to_string(n_tokens));
        log_vector_summary("[CPU_FWD] Input input (n_tokens=" + std::to_string(n_tokens) + ")", input);
    }

    if (n_tokens >= max_seq_len) {
        Logger::error("Position index exceeds max_position_embeddings");
        return std::vector<float>(vs, 0.0f); 
    }
    if (!kv_cache) {
        Logger::error("KVCache is required for token-by-token forward pass");
        return std::vector<float>(vs, 0.0f);
    }
    if (input.size() != hs) {
        Logger::error("Input vector input has incorrect size. Expected " + std::to_string(hs) + ", got " + std::to_string(input.size()));
        return std::vector<float>(vs, 0.0f);
    }

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
        bool log_this_layer = log_this_step && (l == 0); // Log only first layer of first few steps
        if (log_this_layer) {
            Logger::info("[CPU_FWD] ------ START Layer " + std::to_string(l) + " (pos=" + std::to_string(n_tokens) + ") ------");
        }

        const auto& lw = layers[l];
        std::vector<float> x_resid1_vec = input; // Residual 1

        // RMSNorm 1
        // Use F32 norm weights if available, otherwise convert BF16
        const std::vector<float>& w_norm1_vec = lw.input_layernorm_f32.empty() ? bf16vec_to_float_vec(lw.input_layernorm) : lw.input_layernorm_f32;
        rmsnorm_vector_cpu(input, w_norm1_vec, x_norm_vec1, eps);
        if (log_this_layer) log_vector_summary("[CPU_FWD L" + std::to_string(l) + "] RMSNorm1 Out (x_norm_vec1)", x_norm_vec1);

        // Q, K, V projections - Use correct type based on loaded weights
        bool log_matvec_dequant = log_this_layer; // Log dequant for first block if logging this layer
        // Q
        if (!lw.q_proj_q6k.empty()) {
            if (log_this_layer) Logger::info("[CPU_FWD L" + std::to_string(l) + "] MatVec: Q_Proj (Q6_K)");
            if (log_this_layer) log_vector_summary("  Input (x_norm_vec1)", x_norm_vec1);
            matvec_q6k_f32_vector_cpu(lw.q_proj_q6k, x_norm_vec1, q_vec, hs, hs, log_matvec_dequant);
            if (log_this_layer) log_vector_summary("  Output (q_vec)", q_vec);
        } else if (!lw.q_proj_q4k.empty()) {
             if (log_this_layer) Logger::info("[CPU_FWD L" + std::to_string(l) + "] MatVec: Q_Proj (Q4_K)");
             if (log_this_layer) log_vector_summary("  Input (x_norm_vec1)", x_norm_vec1);
             matvec_q4k_f32_vector_cpu(lw.q_proj_q4k, x_norm_vec1, q_vec, hs, hs, log_matvec_dequant);
             if (log_this_layer) log_vector_summary("  Output (q_vec)", q_vec);
        } else if (!lw.q_proj_f32.empty()) { // <<< CHECK F32 FIRST
             if (log_this_layer) Logger::info("[CPU_FWD L" + std::to_string(l) + "] MatVec: Q_Proj (F32)");
             matvec_f32_f32_vector_cpu(lw.q_proj_f32, x_norm_vec1, q_vec, hs, hs); // <<< USE F32 MATVEC
             if (log_this_layer) log_vector_summary("  Output (q_vec)", q_vec);
        } else if (!lw.q_proj.empty()) { // Fallback BF16
            if (log_this_layer) Logger::warning("[CPU_FWD L" + std::to_string(l) + "] MatVec: Q_Proj (BF16 Fallback)");
            matvec_bf16_f32_vector_cpu(lw.q_proj, x_norm_vec1, q_vec, hs, hs);
             if (log_this_layer) log_vector_summary("  Output (q_vec)", q_vec);
        } else {
            throw std::runtime_error("Layer " + std::to_string(l) + ": No valid Q projection weights found!");
        }
        // K
        if (!lw.k_proj_q6k.empty()) {
            if (log_this_layer) Logger::info("[CPU_FWD L" + std::to_string(l) + "] MatVec: K_Proj (Q6_K)");
            matvec_q6k_f32_vector_cpu(lw.k_proj_q6k, x_norm_vec1, k_vec, n_kv_heads * head_dim, hs);
        } else if (!lw.k_proj_q4k.empty()) {
            if (log_this_layer) Logger::info("[CPU_FWD L" + std::to_string(l) + "] MatVec: K_Proj (Q4_K)");
            matvec_q4k_f32_vector_cpu(lw.k_proj_q4k, x_norm_vec1, k_vec, n_kv_heads * head_dim, hs);
        } else if (!lw.k_proj_f32.empty()) { // <<< CHECK F32 FIRST
             if (log_this_layer) Logger::info("[CPU_FWD L" + std::to_string(l) + "] MatVec: K_Proj (F32)");
             matvec_f32_f32_vector_cpu(lw.k_proj_f32, x_norm_vec1, k_vec, n_kv_heads * head_dim, hs); // <<< USE F32 MATVEC
        } else if (!lw.k_proj.empty()) { // Fallback BF16
             if (log_this_layer) Logger::warning("[CPU_FWD L" + std::to_string(l) + "] MatVec: K_Proj (BF16 Fallback)");
             matvec_bf16_f32_vector_cpu(lw.k_proj, x_norm_vec1, k_vec, n_kv_heads * head_dim, hs);
        } else {
            throw std::runtime_error("Layer " + std::to_string(l) + ": No valid K projection weights found!");
        }
        // V
        // +++ START V-PROJ LOGGING +++
        if (log_this_layer) {
            log_vector_summary("  [V-Proj L0] Input (x_norm_vec1)", x_norm_vec1);
        }
        // +++ END V-PROJ LOGGING +++
        if (!lw.v_proj_q6k.empty()) {
             if (log_this_layer) Logger::info("[CPU_FWD L" + std::to_string(l) + "] MatVec: V_Proj (Q6_K)");
             matvec_q6k_f32_vector_cpu(lw.v_proj_q6k, x_norm_vec1, v_vec, n_kv_heads * head_dim, hs, log_matvec_dequant);
        } else if (!lw.v_proj_q4k.empty()) {
             if (log_this_layer) Logger::info("[CPU_FWD L" + std::to_string(l) + "] MatVec: V_Proj (Q4_K)");
             matvec_q4k_f32_vector_cpu(lw.v_proj_q4k, x_norm_vec1, v_vec, n_kv_heads * head_dim, hs, log_matvec_dequant);
        } else if (!lw.v_proj_f32.empty()) { // <<< CHECK F32 FIRST
             if (log_this_layer) Logger::info("[CPU_FWD L" + std::to_string(l) + "] MatVec: V_Proj (F32)");
             matvec_f32_f32_vector_cpu(lw.v_proj_f32, x_norm_vec1, v_vec, n_kv_heads * head_dim, hs); // <<< USE F32 MATVEC
        } else if (!lw.v_proj.empty()) { // Fallback BF16
             if (log_this_layer) Logger::warning("[CPU_FWD L" + std::to_string(l) + "] MatVec: V_Proj (BF16 Fallback)");
             matvec_bf16_f32_vector_cpu(lw.v_proj, x_norm_vec1, v_vec, n_kv_heads * head_dim, hs);
        } else {
            throw std::runtime_error("Layer " + std::to_string(l) + ": No valid V projection weights found!");
        }
        // +++ START V-PROJ LOGGING +++
        if (log_this_layer) {
            log_vector_summary("  [V-Proj L0] Output (v_vec)", v_vec);
        }
        // +++ END V-PROJ LOGGING +++

        // RoPE 
        size_t freqs_offset = (n_tokens * head_dim / 2);
        std::vector<std::pair<float, float>> current_freqs_cis(precomputed_freqs_cis_.begin() + freqs_offset, 
                                                               precomputed_freqs_cis_.begin() + freqs_offset + head_dim / 2);
        apply_rope_vector(q_vec, n_heads, head_dim, n_tokens, current_freqs_cis);
        apply_rope_vector(k_vec, n_kv_heads, head_dim, n_tokens, current_freqs_cis);
        
        // KVCache Update (CPU uses host vectors)
        float* k_current_ptr = k_vec.data(); 
        float* v_current_ptr = v_vec.data(); 
        for (int kvh = 0; kvh < n_kv_heads; ++kvh) {
             size_t current_k_offset = (size_t)kvh * head_dim;
             size_t current_v_offset = (size_t)kvh * head_dim;
             size_t write_offset = n_tokens * n_kv_heads * head_dim + kvh * head_dim;
             if (write_offset + head_dim <= kv_cache->layers[l].k.size()) {
                 std::memcpy(&kv_cache->layers[l].k[write_offset], k_current_ptr + current_k_offset, head_dim * sizeof(float));
                 std::memcpy(&kv_cache->layers[l].v[write_offset], v_current_ptr + current_v_offset, head_dim * sizeof(float));
        } else {
                 Logger::error("KVCache write out of bounds: layer=" + std::to_string(l) + ", pos=" + std::to_string(n_tokens) + ", kv_head=" + std::to_string(kvh));
             }
         }

        // Attention 
        std::fill(attn_out_vec.begin(), attn_out_vec.end(), 0.0f); 
        int current_seq_len = n_tokens + 1;
        float scale = 1.0f / std::sqrt(head_dim);
        for (int h = 0; h < n_heads; ++h) {
            std::vector<float> q_head_rope_vec(q_vec.begin() + h * head_dim, q_vec.begin() + (h + 1) * head_dim);
            int kv_head_idx = h / (n_heads / n_kv_heads);
            std::vector<float> k_cache_head_vec(current_seq_len * head_dim); 
            std::vector<float> v_cache_head_vec(current_seq_len * head_dim); 
            // +++ START ATTENTION LOGGING (Layer 0, Head 0, Pos 0/1) +++
            bool log_attn_details = log_this_layer && (h == 0); // Log only first head of first layer
            if (log_attn_details) {
                log_vector_summary("  [Attn L0H0] Q Head Rope Vec", q_head_rope_vec);
            }
            // +++ END ATTENTION LOGGING +++
            for (int j = 0; j < current_seq_len; ++j) {
                size_t cache_pos_offset = j * n_kv_heads * head_dim + kv_head_idx * head_dim;
                if (cache_pos_offset + head_dim <= kv_cache->layers[l].k.size()) {
                    std::memcpy(k_cache_head_vec.data() + j * head_dim, &kv_cache->layers[l].k[cache_pos_offset], head_dim * sizeof(float));
                    std::memcpy(v_cache_head_vec.data() + j * head_dim, &kv_cache->layers[l].v[cache_pos_offset], head_dim * sizeof(float));
        } else {
                    std::fill(k_cache_head_vec.begin() + j * head_dim, k_cache_head_vec.begin() + (j + 1) * head_dim, 0.0f);
                    std::fill(v_cache_head_vec.begin() + j * head_dim, v_cache_head_vec.begin() + (j + 1) * head_dim, 0.0f);
                    Logger::error("Attention K/V access out of bounds: cache_pos_offset=" + std::to_string(cache_pos_offset));
                }
            }
            // +++ START ATTENTION LOGGING (Layer 0, Head 0, Pos 0/1) +++
            if (log_attn_details) {
                log_vector_summary("  [Attn L0H0] K Cache Head Vec (SeqLen=" + std::to_string(current_seq_len) + ")", k_cache_head_vec);
                log_vector_summary("  [Attn L0H0] V Cache Head Vec (SeqLen=" + std::to_string(current_seq_len) + ")", v_cache_head_vec);
            }
            // +++ END ATTENTION LOGGING +++
            std::vector<float> scores_vec(current_seq_len);
            calculate_attention_scores(q_head_rope_vec, k_cache_head_vec, scores_vec, current_seq_len, head_dim, scale);
            // +++ START ATTENTION LOGGING (Layer 0, Head 0, Pos 0/1) +++
            if (log_attn_details) {
                log_vector_summary("  [Attn L0H0] Scores Vec (Before Softmax)", scores_vec);
            }
            // +++ END ATTENTION LOGGING +++
            std::vector<float> probs_vec(current_seq_len);
            softmax_vector_cpu(scores_vec, probs_vec);
            // +++ START ATTENTION LOGGING (Layer 0, Head 0, Pos 0/1) +++
            if (log_attn_details) {
                log_vector_summary("  [Attn L0H0] Probs Vec (After Softmax)", probs_vec);
            }
            // +++ END ATTENTION LOGGING +++
            std::vector<float> head_attn_out_vec(head_dim);
            weighted_sum_probs_v(probs_vec, v_cache_head_vec, head_attn_out_vec, current_seq_len, head_dim);
            // +++ START ATTENTION LOGGING (Layer 0, Head 0, Pos 0/1) +++
            if (log_attn_details) {
                log_vector_summary("  [Attn L0H0] Head Attn Out Vec", head_attn_out_vec);
            }
            // +++ END ATTENTION LOGGING +++
            size_t out_offset = h * head_dim;
            for(int i=0; i<head_dim; ++i) {
                attn_out_vec[out_offset + i] += head_attn_out_vec[i]; // Note: += here, potential issue if not zeroed?
            }
        }
        // O Projection - Use correct type based on loaded weights
        if (!lw.o_proj_q6k.empty()) {
             if (log_this_layer) Logger::info("[CPU_FWD L" + std::to_string(l) + "] MatVec: O_Proj (Q6_K)");
             matvec_q6k_f32_vector_cpu(lw.o_proj_q6k, attn_out_vec, attn_proj_vec, hs, hs);
        } else if (!lw.o_proj_q4k.empty()) {
             if (log_this_layer) Logger::info("[CPU_FWD L" + std::to_string(l) + "] MatVec: O_Proj (Q4_K)");
             matvec_q4k_f32_vector_cpu(lw.o_proj_q4k, attn_out_vec, attn_proj_vec, hs, hs);
        } else if (!lw.o_proj_f32.empty()) { // <<< CHECK F32 FIRST
             if (log_this_layer) Logger::info("[CPU_FWD L" + std::to_string(l) + "] MatVec: O_Proj (F32)");
             matvec_f32_f32_vector_cpu(lw.o_proj_f32, attn_out_vec, attn_proj_vec, hs, hs); // <<< USE F32 MATVEC
        } else if (!lw.o_proj.empty()) { // Fallback BF16
             if (log_this_layer) Logger::warning("[CPU_FWD L" + std::to_string(l) + "] MatVec: O_Proj (BF16 Fallback)");
             matvec_bf16_f32_vector_cpu(lw.o_proj, attn_out_vec, attn_proj_vec, hs, hs);
        } else {
            throw std::runtime_error("Layer " + std::to_string(l) + ": No valid O projection weights found!");
        }

        // +++ START LAYER 0 LOGGING +++
        if (l == 0 && log_initial) {
            log_vector_summary("Layer 0 Attn Proj Out (attn_proj_vec)", attn_proj_vec);
        }
        // +++ END LAYER 0 LOGGING +++

        #pragma omp parallel for
        for(size_t i=0; i<hs; ++i) {
            input[i] = x_resid1_vec[i] + attn_proj_vec[i]; 
        }

        // +++ START LAYER 0 LOGGING +++
        if (l == 0 && log_initial) {
            log_vector_summary("Layer 0 After Attn Residual (input)", input); // FIX: Use input instead of x_vec
        }
        // +++ END LAYER 0 LOGGING +++
        
        // --- MLP Block --- 
        std::vector<float> x_resid2_vec = input; // FIX: Use input instead of x_vec
        // Post-attention RMSNorm
        const std::vector<float>& w_norm2_vec = lw.post_attention_layernorm_f32.empty() ? bf16vec_to_float_vec(lw.post_attention_layernorm) : lw.post_attention_layernorm_f32;
        rmsnorm_vector_cpu(input, w_norm2_vec, x_norm_vec2, eps);
        // Gate & Up Projections - Use correct type based on loaded weights
        // Gate
        if (!lw.gate_proj_q6k.empty()) {
            if (log_this_layer) Logger::info("[CPU_FWD L" + std::to_string(l) + "] MatVec: Gate_Proj (Q6_K)");
            matvec_q6k_f32_vector_cpu(lw.gate_proj_q6k, x_norm_vec2, gate_vec, is, hs);
        } else if (!lw.gate_proj_q4k.empty()) {
            if (log_this_layer) Logger::info("[CPU_FWD L" + std::to_string(l) + "] MatVec: Gate_Proj (Q4_K)");
            matvec_q4k_f32_vector_cpu(lw.gate_proj_q4k, x_norm_vec2, gate_vec, is, hs);
        } else if (!lw.gate_proj_f32.empty()) { // <<< CHECK F32 FIRST
            if (log_this_layer) Logger::info("[CPU_FWD L" + std::to_string(l) + "] MatVec: Gate_Proj (F32)");
            matvec_f32_f32_vector_cpu(lw.gate_proj_f32, x_norm_vec2, gate_vec, is, hs); // <<< USE F32 MATVEC
        } else if (!lw.gate_proj.empty()) { // Fallback BF16
            if (log_this_layer) Logger::warning("[CPU_FWD L" + std::to_string(l) + "] MatVec: Gate_Proj (BF16 Fallback)");
            matvec_bf16_f32_vector_cpu(lw.gate_proj, x_norm_vec2, gate_vec, is, hs);
        } else {
            throw std::runtime_error("Layer " + std::to_string(l) + ": No valid Gate projection weights found!");
        }
        // Up
        if (!lw.up_proj_q6k.empty()) {
             if (log_this_layer) Logger::info("[CPU_FWD L" + std::to_string(l) + "] MatVec: Up_Proj (Q6_K)");
             matvec_q6k_f32_vector_cpu(lw.up_proj_q6k, x_norm_vec2, up_vec, is, hs);
        } else if (!lw.up_proj_q4k.empty()) {
             if (log_this_layer) Logger::info("[CPU_FWD L" + std::to_string(l) + "] MatVec: Up_Proj (Q4_K)");
             matvec_q4k_f32_vector_cpu(lw.up_proj_q4k, x_norm_vec2, up_vec, is, hs);
        } else if (!lw.up_proj_f32.empty()) { // <<< CHECK F32 FIRST
             if (log_this_layer) Logger::info("[CPU_FWD L" + std::to_string(l) + "] MatVec: Up_Proj (F32)");
             matvec_f32_f32_vector_cpu(lw.up_proj_f32, x_norm_vec2, up_vec, is, hs); // <<< USE F32 MATVEC
        } else if (!lw.up_proj.empty()) { // Fallback BF16
             if (log_this_layer) Logger::warning("[CPU_FWD L" + std::to_string(l) + "] MatVec: Up_Proj (BF16 Fallback)");
             matvec_bf16_f32_vector_cpu(lw.up_proj, x_norm_vec2, up_vec, is, hs);
        } else {
            throw std::runtime_error("Layer " + std::to_string(l) + ": No valid Up projection weights found!");
        }

        // SiLU Activation
        silu_cpu(gate_vec, silu_out_vec);
        // SwiGLU Element-wise Product
        #pragma omp parallel for
        for(size_t i = 0; i < is; ++i) {
            swiglu_result_vec[i] = silu_out_vec[i] * up_vec[i];
        }
        // Down Projection - Use correct type based on loaded weights
        if (!lw.down_proj_q6k.empty()) {
            if (log_this_layer) Logger::info("[CPU_FWD L" + std::to_string(l) + "] MatVec: Down_Proj (Q6_K)");
            matvec_q6k_f32_vector_cpu(lw.down_proj_q6k, swiglu_result_vec, mlp_out_vec, hs, is);
        } else if (!lw.down_proj_q4k.empty()) {
            if (log_this_layer) Logger::info("[CPU_FWD L" + std::to_string(l) + "] MatVec: Down_Proj (Q4_K)");
            matvec_q4k_f32_vector_cpu(lw.down_proj_q4k, swiglu_result_vec, mlp_out_vec, hs, is);
        } else if (!lw.down_proj_f32.empty()) { // <<< CHECK F32 FIRST
            if (log_this_layer) Logger::info("[CPU_FWD L" + std::to_string(l) + "] MatVec: Down_Proj (F32)");
            matvec_f32_f32_vector_cpu(lw.down_proj_f32, swiglu_result_vec, mlp_out_vec, hs, is); // <<< USE F32 MATVEC
        } else if (!lw.down_proj.empty()) { // Fallback BF16
            if (log_this_layer) Logger::warning("[CPU_FWD L" + std::to_string(l) + "] MatVec: Down_Proj (BF16 Fallback)");
            matvec_bf16_f32_vector_cpu(lw.down_proj, swiglu_result_vec, mlp_out_vec, hs, is);
        } else {
             throw std::runtime_error("Layer " + std::to_string(l) + ": No valid Down projection weights found!");
        }

        // Add residual 2
        #pragma omp parallel for
        for(size_t i=0; i<hs; ++i) {
            input[i] = x_resid2_vec[i] + mlp_out_vec[i]; 
        }

        // +++ START LAYER 0 LOGGING +++
        if (l == 0 && log_initial) {
            log_vector_summary("Layer 0 MLP Down Proj Out (mlp_out_vec)", mlp_out_vec);
        }
        // +++ END LAYER 0 LOGGING +++

        // +++ START LAYER 0 LOGGING +++
        if (l == 0 && log_initial) {
            log_vector_summary("Layer 0 End (After MLP Residual) (input)", input); // FIX: Use input instead of x_vec
        }
        // +++ END LAYER 0 LOGGING +++

        if (log_this_layer) {
             Logger::info("[CPU_FWD] ------ END Layer " + std::to_string(l) + " (pos=" + std::to_string(n_tokens) + ") ------");
        }
    } // End layer loop

    // --- Final Steps (Outside Layer Loop) ---
    // Final RMSNorm
    const std::vector<float>& w_final_norm_vec = final_norm_f32.empty() ? bf16vec_to_float_vec(final_norm) : final_norm_f32;
    std::vector<float> x_final_norm_vec(hs);
    rmsnorm_vector_cpu(input, w_final_norm_vec, x_final_norm_vec, eps);
    if (log_initial) { 
        log_vector_summary("Output of Final CPU RMSNorm (P0)", x_final_norm_vec); // Updated log msg
    }

    std::vector<float> logits(vs); 
    bool lm_head_logged = false; // Track if we logged the method

    // Conditional LM Head MatVec (CPU) - REVISED ORDER & F32 HANDLING
    if (!lm_head_q6k.empty()) {
        if (log_initial) Logger::info("[CPU_FWD] Using Q6_K LM Head");
        matvec_q6k_f32_vector_cpu(lm_head_q6k, x_final_norm_vec, logits, vs, hs, log_initial);
        lm_head_logged = true;
    } else if (!lm_head_q4k.empty()) {
        if (log_initial) Logger::info("[CPU_FWD] Using Q4_K LM Head");
        matvec_q4k_f32_vector_cpu(lm_head_q4k, x_final_norm_vec, logits, vs, hs, log_initial); // ADDED CALL (assuming function exists and works now)
        lm_head_logged = true;
    } else if (!lm_head_f32.empty()) { // Check for F32 weights BEFORE falling back to BF16
        if (log_initial) Logger::info("[CPU_FWD] Using F32 LM Head");
        // *** USE THE NEW F32x F32 MATVEC ***
        matvec_f32_f32_vector_cpu(lm_head_f32, x_final_norm_vec, logits, vs, hs);
        lm_head_logged = true;
    } else if (!lm_head.empty()) { // Fallback to original BF16 if NO OTHER TYPE is present
         // THIS PATH SHOULD IDEALLY NOT BE HIT IF F32 WEIGHTS ARE LOADED CORRECTLY
         // It implies lm_head_f32 is empty, but lm_head (bf16) is not.
         Logger::warning("Using BF16 LM head weights directly - Ensure F32 version wasn't expected.");
         if (log_initial) Logger::info("[CPU_FWD] Using BF16 LM Head (Fallback)");
         matvec_bf16_f32_vector_cpu(lm_head, x_final_norm_vec, logits, vs, hs); // Use original BF16 weights
         lm_head_logged = true;
    } else {
         Logger::fatal("No valid LM Head weights found (Q6_K, Q4_K, F32, or BF16).");
         throw std::runtime_error("Missing LM head weights");
    }

    if (log_initial && lm_head_logged) {
        log_vector_summary("Final Logits (CPU Path)", logits, 10);
    }
    if (log_this_step) {
        Logger::info("[CPU_FWD] forward complete. pos=" + std::to_string(n_tokens));
    }
    return logits;

} // End of forward function

// --- Get Vocab Size --- 
int TinyLlamaModel::get_vocab_size() const {
    return config_.vocab_size;
}

// --- Forward Pass (Device Implementation) --- 
// Restore the #ifdef HAS_CUDA guard around the entire function definition
#ifdef HAS_CUDA
std::vector<float> TinyLlamaModel::forward_device(int token_id, int pos, KVCache* kv_cache, const std::vector<int>* attention_mask, cudaStream_t stream) {
    // Logger::info("[CUDA] forward_device called for token: " + std::to_string(token_id) + " pos: " + std::to_string(pos));
    int hs = config_.hidden_size;
    int vs = config_.vocab_size; // Use member vs here
    int n_heads = config_.num_attention_heads;
    int n_kv_heads = config_.num_key_value_heads;
    // --- FIX: Calculate head_dim instead of accessing non-existent member ---
    // (Keep this fix from previous step, as it's likely correct)
    if (n_heads == 0) { // Avoid division by zero
        Logger::fatal("Number of attention heads is zero during forward_device.");
        throw std::runtime_error("Division by zero: n_heads is zero.");
    }
    int head_dim = hs / n_heads; // Calculate head_dim
    // --- END FIX ---
    int nhl = config_.num_hidden_layers;
    int is = config_.intermediate_size;
    float eps = config_.rms_norm_eps;
    int max_seq_len = config_.max_position_embeddings; // Added for cache size

    // --- START: REMOVE Local Device Memory Allocation & Pointers ---
    /*
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
    float* logits_dev = nullptr;
    float* freqs_dev = nullptr; // For RoPE frequencies
    */
    // --- END: REMOVE Local Device Memory Allocation & Pointers ---

    // --- Set cuBLAS Stream --- 
    cublasStatus_t stream_status = cublasSetStream(cublas_handle_, stream);
    if (stream_status != CUBLAS_STATUS_SUCCESS) {
        Logger::error("cublasSetStream failed in forward_device");
        // Handle error, maybe throw or return empty vector
        return {}; 
    }

    // --- START: REMOVE Local Device Memory Allocation ---
    /*
    // Allocate device buffers using async allocation on the specified stream
    gpuErrchk(cudaMallocAsync(&x_dev, hs * sizeof(float), stream));
    gpuErrchk(cudaMallocAsync(&x_norm_dev, hs * sizeof(float), stream));
    gpuErrchk(cudaMallocAsync(&x_resid1_dev, hs * sizeof(float), stream));
    gpuErrchk(cudaMallocAsync(&x_resid2_dev, hs * sizeof(float), stream));
    gpuErrchk(cudaMallocAsync(&q_dev, hs * sizeof(float), stream));
    gpuErrchk(cudaMallocAsync(&k_dev, n_kv_heads * head_dim * sizeof(float), stream));
    gpuErrchk(cudaMallocAsync(&v_dev, n_kv_heads * head_dim * sizeof(float), stream));
    gpuErrchk(cudaMallocAsync(&attn_out_dev, hs * sizeof(float), stream));
    gpuErrchk(cudaMallocAsync(&attn_proj_dev, hs * sizeof(float), stream)); // Allocate outside loop like .back
    gpuErrchk(cudaMallocAsync(&gate_vec_dev, is * sizeof(float), stream));
    gpuErrchk(cudaMallocAsync(&up_vec_dev, is * sizeof(float), stream));
    gpuErrchk(cudaMallocAsync(&swiglu_vec_dev, is * sizeof(float), stream));
    gpuErrchk(cudaMallocAsync(&mlp_down_dev, hs * sizeof(float), stream)); // Allocate outside loop like .back
    gpuErrchk(cudaMallocAsync(&logits_dev, vs * sizeof(float), stream));
    */
    // --- END: REMOVE Local Device Memory Allocation ---

    // --- Initial Embedding: Use CUDA kernel and persistent buffer ---
    const void* embed_table_dev_ptr = nullptr;
    bool is_bf16_embedding = false;
    if (token_embedding_table_f32_dev_) {
        embed_table_dev_ptr = token_embedding_table_f32_dev_;
        is_bf16_embedding = false;
    } else if (token_embedding_table_dev_) {
        embed_table_dev_ptr = token_embedding_table_dev_;
        is_bf16_embedding = true;
    } else {
        Logger::error("No embedding table found on GPU (FP32 or BF16) in forward_device.");
        return {}; // Return empty vector on error (no memory to free here now)
    }

    // Launch the kernel to perform embedding lookup directly into persistent x_dev_
    lookup_embedding_cuda(embed_table_dev_ptr, x_dev_, token_id, hs, vs, is_bf16_embedding, stream);
    // --- Initial Embedding END ---


    // --- Layer Loop --- 
    for (int l = 0; l < nhl; ++l) {
        // Use persistent member buffers (x_dev_, x_norm_dev_, q_dev_, etc.)
        const auto& lw = layers[l]; // Need layer weights reference
        // Layer weights device pointers (calculated offsets into persistent buffers)
        size_t layer_q_size    = (size_t)hs * hs;
        size_t layer_k_size    = (size_t)n_kv_heads * head_dim * hs; // kv_dim * hs
        size_t layer_v_size    = (size_t)n_kv_heads * head_dim * hs; // kv_dim * hs
        size_t layer_o_size    = (size_t)hs * hs;
        size_t layer_gate_size = (size_t)is * hs;
        size_t layer_up_size   = (size_t)is * hs;
        size_t layer_down_size = (size_t)hs * is;

        // --- MODIFIED: Define pointers for BOTH BF16 and FP32 persistent weights ---
        // BF16 pointers (used only as fallback or if FP32 doesn't exist)
        const uint16_t* lw_q_proj_bf16_dev    = w_q_dev_    ? w_q_dev_    + (size_t)l * layer_q_size : nullptr;
        const uint16_t* lw_k_proj_bf16_dev    = w_k_dev_    ? w_k_dev_    + (size_t)l * layer_k_size : nullptr;
        const uint16_t* lw_v_proj_bf16_dev    = w_v_dev_    ? w_v_dev_    + (size_t)l * layer_v_size : nullptr;
        const uint16_t* lw_o_proj_bf16_dev    = w_o_dev_    ? w_o_dev_    + (size_t)l * layer_o_size : nullptr;
        const uint16_t* lw_gate_proj_bf16_dev = w_gate_dev_ ? w_gate_dev_ + (size_t)l * layer_gate_size : nullptr;
        const uint16_t* lw_up_proj_bf16_dev   = w_up_dev_   ? w_up_dev_   + (size_t)l * layer_up_size : nullptr;
        const uint16_t* lw_down_proj_bf16_dev = w_down_dev_ ? w_down_dev_ + (size_t)l * layer_down_size : nullptr;
        // FP32 pointers (preferred)
        const float* lw_q_proj_f32_dev    = w_q_f32_dev_    ? w_q_f32_dev_    + (size_t)l * layer_q_size : nullptr;
        const float* lw_k_proj_f32_dev    = w_k_f32_dev_    ? w_k_f32_dev_    + (size_t)l * layer_k_size : nullptr;
        const float* lw_v_proj_f32_dev    = w_v_f32_dev_    ? w_v_f32_dev_    + (size_t)l * layer_v_size : nullptr;
        const float* lw_o_proj_f32_dev    = w_o_f32_dev_    ? w_o_f32_dev_    + (size_t)l * layer_o_size : nullptr;
        const float* lw_gate_proj_f32_dev = w_gate_f32_dev_ ? w_gate_f32_dev_ + (size_t)l * layer_gate_size : nullptr;
        const float* lw_up_proj_f32_dev   = w_up_f32_dev_   ? w_up_f32_dev_   + (size_t)l * layer_up_size : nullptr;
        const float* lw_down_proj_f32_dev = w_down_f32_dev_ ? w_down_f32_dev_ + (size_t)l * layer_down_size : nullptr;
        // --- END MODIFICATION ---


        const float* lw_in_norm_dev   = layers[l].input_layernorm_dev;
        const float* lw_post_norm_dev = layers[l].post_attention_layernorm_dev;

        // Residual 1 Prep (Copy x -> x_resid1)
        gpuErrchk(cudaMemcpyAsync(x_resid1_dev_, x_dev_, hs * sizeof(float), cudaMemcpyDeviceToDevice, stream));

        // RMSNorm
        rmsnorm_vector_cuda(x_dev_, layers[l].input_layernorm_dev, x_norm_dev_, hs, eps, stream);

        // QKV Projections
        // --- MODIFIED: Call matvec_f32_f32_cuda and pass FP32 weights ---
        // Prioritize using FP32 weights if they exist
        if (lw_q_proj_f32_dev && lw_k_proj_f32_dev && lw_v_proj_f32_dev) {
            // Use the function that expects FP32 weights (we'll rename it next)
            matvec_f32_f32_cuda(cublas_handle_, lw_q_proj_f32_dev, x_norm_dev_, q_dev_, hs, hs, stream);
            matvec_f32_f32_cuda(cublas_handle_, lw_k_proj_f32_dev, x_norm_dev_, k_dev_, n_kv_heads * head_dim, hs, stream);
            matvec_f32_f32_cuda(cublas_handle_, lw_v_proj_f32_dev, x_norm_dev_, v_dev_, n_kv_heads * head_dim, hs, stream);
        } else if (lw_q_proj_bf16_dev && lw_k_proj_bf16_dev && lw_v_proj_bf16_dev) { // Fallback to BF16 (if needed, though inefficient)
             Logger::warning("Layer " + std::to_string(l) + ": Using BF16 matvec path (less efficient) for QKV.");
             // Keep the old call (which does conversion internally) - RENAME LATER IF NEEDED
             matvec_bf16_f32_cuda(cublas_handle_, lw_q_proj_bf16_dev, x_norm_dev_, q_dev_, hs, hs, stream);
             matvec_bf16_f32_cuda(cublas_handle_, lw_k_proj_bf16_dev, x_norm_dev_, k_dev_, n_kv_heads * head_dim, hs, stream);
             matvec_bf16_f32_cuda(cublas_handle_, lw_v_proj_bf16_dev, x_norm_dev_, v_dev_, n_kv_heads * head_dim, hs, stream);
        } else {
             Logger::error("Layer " + std::to_string(l) + ": No valid QKV projection weights found on GPU (FP32 or BF16).");
             return {}; // Or handle error appropriately
        }
        // --- END MODIFICATION ---
  
        // RoPE
        rope_cuda(q_dev_, n_heads, head_dim, all_freqs_cis_dev, pos, stream);
        rope_cuda(k_dev_, n_kv_heads, head_dim, all_freqs_cis_dev, pos, stream);
  
        // KVCache Update
        for (int kvh = 0; kvh < n_kv_heads; ++kvh) {
            const float* current_k_head_ptr = k_dev_ + kvh * head_dim;
            const float* current_v_head_ptr = v_dev_ + kvh * head_dim;
            
            update_kv_cache_cuda(
                kv_cache->layers[l].k_dev,
                current_k_head_ptr,
                pos, kvh, 
                kv_cache->allocated_max_seq_len,
                kv_cache->allocated_num_kv_heads,
                kv_cache->allocated_head_dim,
                stream);
                                 
            update_kv_cache_cuda(
                kv_cache->layers[l].v_dev,
                current_v_head_ptr,
                pos, kvh, 
                kv_cache->allocated_max_seq_len,
                kv_cache->allocated_num_kv_heads,
                kv_cache->allocated_head_dim,
                stream);
        }

        // Attention
        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
         attention_cuda(q_dev_, kv_cache->layers[l].k_dev, kv_cache->layers[l].v_dev, attn_out_dev_,
                        n_heads, pos + 1, head_dim, scale, 
                        kv_cache->allocated_max_seq_len,
                        kv_cache->allocated_num_kv_heads,
                        stream);

        // Attention Output Projection
        // --- MODIFIED: Call matvec_f32_f32_cuda and pass FP32 weights ---
        if (lw_o_proj_f32_dev) {
             matvec_f32_f32_cuda(cublas_handle_, lw_o_proj_f32_dev, attn_out_dev_, attn_proj_dev_, hs, hs, stream);
        } else if (lw_o_proj_bf16_dev) {
             Logger::warning("Layer " + std::to_string(l) + ": Using BF16 matvec path (less efficient) for O-Proj.");
             matvec_bf16_f32_cuda(cublas_handle_, lw_o_proj_bf16_dev, attn_out_dev_, attn_proj_dev_, hs, hs, stream);
        } else {
             Logger::error("Layer " + std::to_string(l) + ": No valid O projection weights found on GPU (FP32 or BF16).");
             return {};
        }
        // --- END MODIFICATION ---

        // Residual 1 Add
        add_residual_cuda(attn_proj_dev_, x_resid1_dev_, x_dev_, hs, stream);

        // --- MLP Block --- 
        // Residual 2 Prep
        gpuErrchk(cudaMemcpyAsync(x_resid2_dev_, x_dev_, hs * sizeof(float), cudaMemcpyDeviceToDevice, stream));

        // Post-attention RMSNorm
        rmsnorm_vector_cuda(x_dev_, layers[l].post_attention_layernorm_dev, x_norm_dev_, hs, eps, stream);

        // MLP Projections
        // --- MODIFIED: Call matvec_f32_f32_cuda and pass FP32 weights ---
        if (lw_gate_proj_f32_dev && lw_up_proj_f32_dev) {
             matvec_f32_f32_cuda(cublas_handle_, lw_gate_proj_f32_dev, x_norm_dev_, gate_vec_dev_, is, hs, stream);
             matvec_f32_f32_cuda(cublas_handle_, lw_up_proj_f32_dev, x_norm_dev_, up_vec_dev_, is, hs, stream);
        } else if (lw_gate_proj_bf16_dev && lw_up_proj_bf16_dev) {
             Logger::warning("Layer " + std::to_string(l) + ": Using BF16 matvec path (less efficient) for Gate/Up Proj.");
             matvec_bf16_f32_cuda(cublas_handle_, lw_gate_proj_bf16_dev, x_norm_dev_, gate_vec_dev_, is, hs, stream);
             matvec_bf16_f32_cuda(cublas_handle_, lw_up_proj_bf16_dev, x_norm_dev_, up_vec_dev_, is, hs, stream);
        } else {
             Logger::error("Layer " + std::to_string(l) + ": No valid Gate/Up projection weights found on GPU (FP32 or BF16).");
             return {};
        }
        // --- END MODIFICATION ---

        // SwiGLU
        swiglu_cuda(gate_vec_dev_, up_vec_dev_, swiglu_vec_dev_, is, stream);

        // MLP Down Projection
        // --- MODIFIED: Call matvec_f32_f32_cuda and pass FP32 weights ---
        if (lw_down_proj_f32_dev) {
             matvec_f32_f32_cuda(cublas_handle_, lw_down_proj_f32_dev, swiglu_vec_dev_, mlp_down_dev_, hs, is, stream);
        } else if (lw_down_proj_bf16_dev) {
             Logger::warning("Layer " + std::to_string(l) + ": Using BF16 matvec path (less efficient) for Down Proj.");
             matvec_bf16_f32_cuda(cublas_handle_, lw_down_proj_bf16_dev, swiglu_vec_dev_, mlp_down_dev_, hs, is, stream);
        } else {
             Logger::error("Layer " + std::to_string(l) + ": No valid Down projection weights found on GPU (FP32 or BF16).");
             return {};
        }
        // --- END MODIFICATION ---
         
        // Residual 2 Add
        add_residual_cuda(mlp_down_dev_, x_resid2_dev_, x_dev_, hs, stream);

    } // End layer loop

    // --- Final Steps (Outside Layer Loop) --- 
     
    // Final RMSNorm
    rmsnorm_vector_cuda(x_dev_, final_norm_dev, x_norm_dev_, hs, eps, stream);

    // Final LM Head Projection
    // --- MODIFIED: Call matvec_f32_f32_cuda and pass FP32 weights ---
    if (lm_head_f32_dev_) {
         matvec_f32_f32_cuda(cublas_handle_, lm_head_f32_dev_, x_norm_dev_, logits_dev_, vs, hs, stream);
    } else if (lm_head_dev_) {
         Logger::warning("Using BF16 matvec path (less efficient) for LM Head.");
         matvec_bf16_f32_cuda(cublas_handle_, lm_head_dev_, x_norm_dev_, logits_dev_, vs, hs, stream);
    } else {
         Logger::error("No valid LM head weights found on GPU (FP32 or BF16).");
         return {};
    }
    // --- END MODIFICATION ---

    // Synchronize Stream before Copy
    gpuErrchk(cudaStreamSynchronize(stream));
 
    // Copy Logits Device -> Host
    std::vector<float> logits(vs);
     gpuErrchk(cudaMemcpy(logits.data(), logits_dev_, vs * sizeof(float), cudaMemcpyDeviceToHost));
 
    // --- START: REMOVE Local Device Memory Free --- 
    /*
    gpuErrchk(cudaFree(x_dev));
// ... rest of frees ...
    gpuErrchk(cudaFree(logits_dev));
    */
    // --- END: REMOVE Local Device Memory Free ---
 
    return logits;

} // End forward_device
#endif // HAS_CUDA (End of forward_device function definition)


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

    // --- ADDED: Helper lambda for FP16 assignment (reads uint16_t, converts to float) ---
    auto assign_vec_f16 = [&](std::vector<float>& vec_f32, const GGUFTensorInfo& tinfo) {
        size_t n_elem = tinfo.num_elements;
        if (n_elem == 0) { vec_f32.clear(); return; }
        const uint16_t* src = reinterpret_cast<const uint16_t*>(gguf.tensor_data.data() + tinfo.offset);
        // Check source pointer validity (optional extra check)
        uintptr_t src_end_addr = reinterpret_cast<uintptr_t>(src) + n_elem * sizeof(uint16_t);
        if (src_end_addr > reinterpret_cast<uintptr_t>(gguf.tensor_data.data()) + gguf.tensor_data.size()) {
             throw std::out_of_range("Calculated source range for FP16 tensor '" + tinfo.name + "' exceeds data buffer.");
        }
        vec_f32.resize(n_elem);
        #pragma omp parallel for
        for(size_t i = 0; i < n_elem; ++i) {
            vec_f32[i] = fp16_to_fp32(src[i]);
        }
    };
    // --- END ADDED HELPER ---

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
            if (tinfo.type == GGMLType::GGML_TYPE_F32) {
                assign_vec_f32(model.embed_tokens_f32, tinfo);
                Logger::info("Mapped GGUF tensor '" + name + "' (FP32) to model.embed_tokens_f32");
            } else if (tinfo.type == GGMLType::GGML_TYPE_F16) { // <<< ADDED F16 CASE >>>
                assign_vec_f16(model.embed_tokens_f32, tinfo);
                Logger::info("Mapped GGUF tensor '" + name + "' (FP16) to model.embed_tokens_f32");
            } else if (tinfo.type == GGMLType::GGML_TYPE_Q4_K) {
                // --- ADD DETAILED LOGGING FOR Q4_K EMBEDDING MAPPING ---
                size_t num_blocks = 0;
                uintptr_t src_addr = 0;
                uintptr_t src_end_addr_calc = 0;
                uintptr_t data_start_addr = reinterpret_cast<uintptr_t>(gguf.tensor_data.data());
                uintptr_t data_end_addr = data_start_addr + total_data_size;
                const block_q4_K* first_block_ptr = nullptr;
                uint16_t raw_d = 0, raw_dmin = 0;
                float converted_d = NAN, converted_dmin = NAN;

                try {
                    if (GGML_QK_K != 0 && tinfo.num_elements % GGML_QK_K == 0) {
                        num_blocks = tinfo.num_elements / GGML_QK_K;
                        src_addr = reinterpret_cast<uintptr_t>(gguf.tensor_data.data()) + tinfo.offset;
                        src_end_addr_calc = src_addr + num_blocks * sizeof(block_q4_K);
                        // Check if src_addr is valid and points within the buffer before dereferencing
                        if (num_blocks > 0 && src_addr >= data_start_addr && (src_addr + sizeof(block_q4_K)) <= data_end_addr) {
                             first_block_ptr = reinterpret_cast<const block_q4_K*>(src_addr);
                             // Safely read d and dmin only if the pointer is valid
                             raw_d = first_block_ptr->d;
                             raw_dmin = first_block_ptr->dmin;
                             converted_d = fp16_to_fp32(raw_d); // Assuming fp16_to_fp32 exists
                             converted_dmin = fp16_to_fp32(raw_dmin);
                        } else if (num_blocks > 0) {
                            // Pointer calculation is okay, but read would be out of bounds
                             Logger::warning("[MAP GGUF Q4_K EMBEDDING]: Calculated source address " + std::to_string(src_addr) + " is out of bounds for reading the first block.");
                        }
                    } else {
                         Logger::warning("[MAP GGUF Q4_K EMBEDDING]: num_elements not divisible by GGML_QK_K or GGML_QK_K is zero.");
                    }
                    std::stringstream log_map;
                    log_map << "[MAP GGUF Q4_K EMBEDDING]: Tensor='" << tinfo.name << "', Offset=" << tinfo.offset
                           << ", NumElem=" << tinfo.num_elements << ", NumBlocks=" << num_blocks
                           << ", SizeBytes(TensorInfo)=" << tinfo.size_in_bytes << ", SizeBytes(Calc)=" << (num_blocks * sizeof(block_q4_K)) << "\\n"
                           << "    DataBuffer: [" << data_start_addr << " - " << data_end_addr << "] (Size: " << total_data_size << ")\\n"
                           << "    SrcPtrAddr: " << src_addr << " (RelOffset: " << (src_addr - data_start_addr) << ")\\n"
                           << "    CalcEndPtr: " << src_end_addr_calc << "\\n"
                           << "    ReadRangeOK: " << (src_addr >= data_start_addr && src_end_addr_calc <= data_end_addr && src_addr <= src_end_addr_calc ? "YES" : "NO!") << "\\n";
                    if (first_block_ptr && src_addr >= data_start_addr && (src_addr + sizeof(block_q4_K)) <= data_end_addr) {
                         log_map << "    First Block @ " << reinterpret_cast<uintptr_t>(first_block_ptr) << ":\\n"
                                << "      Raw d=0x" << std::hex << raw_d << std::dec << " -> fp32=" << converted_d << "\\n"
                                << "      Raw dmin=0x" << std::hex << raw_dmin << std::dec << " -> fp32=" << converted_dmin;
                    } else {
                         log_map << "    First Block: Could not read (invalid ptr or num_blocks=0 or OOB).";
                    }
                    Logger::info(log_map.str());
                } catch (const std::exception& log_ex) {
                     Logger::error("Exception during Q4_K embedding mapping log generation: " + std::string(log_ex.what()));
                }
                // --- END DETAILED LOGGING ---
                assign_vec_q4k(model.embed_tokens_q4k, tinfo); // Assign the vector
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
            if (tinfo.type == GGMLType::GGML_TYPE_F32) {
                assign_vec_f32(model.lm_head_f32, tinfo);
                Logger::info("Mapped GGUF tensor '" + name + "' (FP32) to model.lm_head_f32");
            } else if (tinfo.type == GGMLType::GGML_TYPE_F16) { // <<< ADDED F16 CASE >>>
                assign_vec_f16(model.lm_head_f32, tinfo);
                Logger::info("Mapped GGUF tensor '" + name + "' (FP16) to model.lm_head_f32");
            } else if (tinfo.type == GGMLType::GGML_TYPE_Q4_K) {
                assign_vec_q4k(model.lm_head_q4k, tinfo);
                Logger::info("Mapped GGUF tensor '" + name + "' (Q4_K) to model.lm_head_q4k");
            } else if (tinfo.type == GGMLType::GGML_TYPE_Q6_K) {
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
            if (tinfo.type == GGMLType::GGML_TYPE_F32) {
                assign_vec_f32(model.final_norm_f32, tinfo);
                Logger::info("Mapped GGUF tensor '" + name + "' (FP32) to model.final_norm_f32");
            } else if (tinfo.type == GGMLType::GGML_TYPE_F16) { // <<< ADDED F16 CASE >>>
                 assign_vec_f16(model.final_norm_f32, tinfo);
                 Logger::info("Mapped GGUF tensor '" + name + "' (FP16) to model.final_norm_f32");
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

            // Map known fields based on suffix, adding F16 cases
            // Q_PROJ
            if (suffix == "self_attn.q_proj.weight" || suffix == "attn.wq.weight" || suffix == "attn_q.weight") {
                if (tinfo.type == GGMLType::GGML_TYPE_F32) {
                    assign_vec_f32(lw.q_proj_f32, tinfo);
                    Logger::info("Mapped GGUF tensor '" + name + "' (FP32) to layers[" + std::to_string(layer_idx) + "].q_proj_f32");
                 } else if (tinfo.type == GGMLType::GGML_TYPE_F16) { // <<< ADDED F16 CASE >>>
                    assign_vec_f16(lw.q_proj_f32, tinfo);
                    Logger::info("Mapped GGUF tensor '" + name + "' (FP16) to layers[" + std::to_string(layer_idx) + "].q_proj_f32");
                } else if (tinfo.type == GGMLType::GGML_TYPE_Q4_K) {
                    assign_vec_q4k(lw.q_proj_q4k, tinfo);
                    Logger::info("Mapped GGUF tensor '" + name + "' (Q4_K) to layers[" + std::to_string(layer_idx) + "].q_proj_q4k");
                } else if (tinfo.type == GGMLType::GGML_TYPE_Q6_K) {
                    assign_vec_q6k(lw.q_proj_q6k, tinfo);
                    Logger::info("Mapped GGUF tensor '" + name + "' (Q6_K) to layers[" + std::to_string(layer_idx) + "].q_proj_q6k");
                } else {
                    assign_vec_bf16(lw.q_proj, tinfo);
                    Logger::info("Mapped GGUF tensor '" + name + "' (BF16/Other) to layers[" + std::to_string(layer_idx) + "].q_proj");
                }
                Logger::info("-> Successfully assigned '" + name + "' to layer " + std::to_string(layer_idx));
                continue;
            }
            // K_PROJ
            if (suffix == "self_attn.k_proj.weight" || suffix == "attn.wk.weight" || suffix == "attn_k.weight") { 
                if (tinfo.type == GGMLType::GGML_TYPE_F32) { assign_vec_f32(lw.k_proj_f32, tinfo); Logger::info("Mapped GGUF tensor '" + name + "' (FP32) to layers[" + std::to_string(layer_idx) + "].k_proj_f32"); } 
                else if (tinfo.type == GGMLType::GGML_TYPE_F16) { assign_vec_f16(lw.k_proj_f32, tinfo); Logger::info("Mapped GGUF tensor '" + name + "' (FP16) to layers[" + std::to_string(layer_idx) + "].k_proj_f32"); } 
                else if (tinfo.type == GGMLType::GGML_TYPE_Q4_K) { assign_vec_q4k(lw.k_proj_q4k, tinfo); Logger::info("Mapped GGUF tensor '" + name + "' (Q4_K) to layers[" + std::to_string(layer_idx) + "].k_proj_q4k"); }
                else if (tinfo.type == GGMLType::GGML_TYPE_Q6_K) { assign_vec_q6k(lw.k_proj_q6k, tinfo); Logger::info("Mapped GGUF tensor '" + name + "' (Q6_K) to layers[" + std::to_string(layer_idx) + "].k_proj_q6k"); } 
                else { assign_vec_bf16(lw.k_proj, tinfo); Logger::info("Mapped GGUF tensor '" + name + "' (BF16/Other) to layers[" + std::to_string(layer_idx) + "].k_proj"); }
                 Logger::info("-> Successfully assigned '" + name + "' to layer " + std::to_string(layer_idx));
                continue;
            }
            // V_PROJ
            if (suffix == "self_attn.v_proj.weight" || suffix == "attn.wv.weight" || suffix == "attn_v.weight") { 
                if (tinfo.type == GGMLType::GGML_TYPE_F32) { assign_vec_f32(lw.v_proj_f32, tinfo); Logger::info("Mapped GGUF tensor '" + name + "' (FP32) to layers[" + std::to_string(layer_idx) + "].v_proj_f32"); } 
                else if (tinfo.type == GGMLType::GGML_TYPE_F16) { assign_vec_f16(lw.v_proj_f32, tinfo); Logger::info("Mapped GGUF tensor '" + name + "' (FP16) to layers[" + std::to_string(layer_idx) + "].v_proj_f32"); } 
                else if (tinfo.type == GGMLType::GGML_TYPE_Q4_K) { assign_vec_q4k(lw.v_proj_q4k, tinfo); Logger::info("Mapped GGUF tensor '" + name + "' (Q4_K) to layers[" + std::to_string(layer_idx) + "].v_proj_q4k"); }
                else if (tinfo.type == GGMLType::GGML_TYPE_Q6_K) { assign_vec_q6k(lw.v_proj_q6k, tinfo); Logger::info("Mapped GGUF tensor '" + name + "' (Q6_K) to layers[" + std::to_string(layer_idx) + "].v_proj_q6k"); } 
                else { assign_vec_bf16(lw.v_proj, tinfo); Logger::info("Mapped GGUF tensor '" + name + "' (BF16/Other) to layers[" + std::to_string(layer_idx) + "].v_proj"); }
                Logger::info("-> Successfully assigned '" + name + "' to layer " + std::to_string(layer_idx));
                continue;
            }
            // O_PROJ
            if (suffix == "self_attn.o_proj.weight" || suffix == "attn.wo.weight" || suffix == "attn_output.weight") { 
                if (tinfo.type == GGMLType::GGML_TYPE_F32) { assign_vec_f32(lw.o_proj_f32, tinfo); Logger::info("Mapped GGUF tensor '" + name + "' (FP32) to layers[" + std::to_string(layer_idx) + "].o_proj_f32"); } 
                else if (tinfo.type == GGMLType::GGML_TYPE_F16) { assign_vec_f16(lw.o_proj_f32, tinfo); Logger::info("Mapped GGUF tensor '" + name + "' (FP16) to layers[" + std::to_string(layer_idx) + "].o_proj_f32"); } 
                else if (tinfo.type == GGMLType::GGML_TYPE_Q4_K) { assign_vec_q4k(lw.o_proj_q4k, tinfo); Logger::info("Mapped GGUF tensor '" + name + "' (Q4_K) to layers[" + std::to_string(layer_idx) + "].o_proj_q4k"); }
                else if (tinfo.type == GGMLType::GGML_TYPE_Q6_K) { assign_vec_q6k(lw.o_proj_q6k, tinfo); Logger::info("Mapped GGUF tensor '" + name + "' (Q6_K) to layers[" + std::to_string(layer_idx) + "].o_proj_q6k"); } 
                else { assign_vec_bf16(lw.o_proj, tinfo); Logger::info("Mapped GGUF tensor '" + name + "' (BF16/Other) to layers[" + std::to_string(layer_idx) + "].o_proj"); }
                Logger::info("-> Successfully assigned '" + name + "' to layer " + std::to_string(layer_idx));
                continue;
            }
            // GATE_PROJ
            if (suffix == "mlp.gate_proj.weight" || suffix == "ffn.w1.weight" || suffix == "ffn_gate.weight") { 
                if (tinfo.type == GGMLType::GGML_TYPE_F32) { assign_vec_f32(lw.gate_proj_f32, tinfo); Logger::info("Mapped GGUF tensor '" + name + "' (FP32) to layers[" + std::to_string(layer_idx) + "].gate_proj_f32"); } 
                else if (tinfo.type == GGMLType::GGML_TYPE_F16) { assign_vec_f16(lw.gate_proj_f32, tinfo); Logger::info("Mapped GGUF tensor '" + name + "' (FP16) to layers[" + std::to_string(layer_idx) + "].gate_proj_f32"); } 
                else if (tinfo.type == GGMLType::GGML_TYPE_Q4_K) { assign_vec_q4k(lw.gate_proj_q4k, tinfo); Logger::info("Mapped GGUF tensor '" + name + "' (Q4_K) to layers[" + std::to_string(layer_idx) + "].gate_proj_q4k"); } 
                else if (tinfo.type == GGMLType::GGML_TYPE_Q6_K) { assign_vec_q6k(lw.gate_proj_q6k, tinfo); Logger::info("Mapped GGUF tensor '" + name + "' (Q6_K) to layers[" + std::to_string(layer_idx) + "].gate_proj_q6k"); } 
                else { assign_vec_bf16(lw.gate_proj, tinfo); Logger::info("Mapped GGUF tensor '" + name + "' (BF16/Other) to layers[" + std::to_string(layer_idx) + "].gate_proj"); }
                Logger::info("-> Successfully assigned '" + name + "' to layer " + std::to_string(layer_idx));
                continue;
            }
            // UP_PROJ
            if (suffix == "mlp.up_proj.weight" || suffix == "ffn.w3.weight" || suffix == "ffn_up.weight") { 
                if (tinfo.type == GGMLType::GGML_TYPE_F32) { assign_vec_f32(lw.up_proj_f32, tinfo); Logger::info("Mapped GGUF tensor '" + name + "' (FP32) to layers[" + std::to_string(layer_idx) + "].up_proj_f32"); } 
                else if (tinfo.type == GGMLType::GGML_TYPE_F16) { assign_vec_f16(lw.up_proj_f32, tinfo); Logger::info("Mapped GGUF tensor '" + name + "' (FP16) to layers[" + std::to_string(layer_idx) + "].up_proj_f32"); } 
                else if (tinfo.type == GGMLType::GGML_TYPE_Q4_K) { assign_vec_q4k(lw.up_proj_q4k, tinfo); Logger::info("Mapped GGUF tensor '" + name + "' (Q4_K) to layers[" + std::to_string(layer_idx) + "].up_proj_q4k"); } 
                else if (tinfo.type == GGMLType::GGML_TYPE_Q6_K) { assign_vec_q6k(lw.up_proj_q6k, tinfo); Logger::info("Mapped GGUF tensor '" + name + "' (Q6_K) to layers[" + std::to_string(layer_idx) + "].up_proj_q6k"); } 
                else { assign_vec_bf16(lw.up_proj, tinfo); Logger::info("Mapped GGUF tensor '" + name + "' (BF16/Other) to layers[" + std::to_string(layer_idx) + "].up_proj"); }
                Logger::info("-> Successfully assigned '" + name + "' to layer " + std::to_string(layer_idx));
                continue;
            }
            // DOWN_PROJ
            if (suffix == "mlp.down_proj.weight" || suffix == "ffn.w2.weight" || suffix == "ffn_down.weight") { 
                if (tinfo.type == GGMLType::GGML_TYPE_F32) { assign_vec_f32(lw.down_proj_f32, tinfo); Logger::info("Mapped GGUF tensor '" + name + "' (FP32) to layers[" + std::to_string(layer_idx) + "].down_proj_f32"); } 
                else if (tinfo.type == GGMLType::GGML_TYPE_F16) { assign_vec_f16(lw.down_proj_f32, tinfo); Logger::info("Mapped GGUF tensor '" + name + "' (FP16) to layers[" + std::to_string(layer_idx) + "].down_proj_f32"); } 
                else if (tinfo.type == GGMLType::GGML_TYPE_Q4_K) { assign_vec_q4k(lw.down_proj_q4k, tinfo); Logger::info("Mapped GGUF tensor '" + name + "' (Q4_K) to layers[" + std::to_string(layer_idx) + "].down_proj_q4k"); } 
                else if (tinfo.type == GGMLType::GGML_TYPE_Q6_K) { assign_vec_q6k(lw.down_proj_q6k, tinfo); Logger::info("Mapped GGUF tensor '" + name + "' (Q6_K) to layers[" + std::to_string(layer_idx) + "].down_proj_q6k"); } 
                else { assign_vec_bf16(lw.down_proj, tinfo); Logger::info("Mapped GGUF tensor '" + name + "' (BF16/Other) to layers[" + std::to_string(layer_idx) + "].down_proj"); }
                Logger::info("-> Successfully assigned '" + name + "' to layer " + std::to_string(layer_idx));
                continue;
            }
            // LAYER NORM (ATTN)
            if (suffix == "input_layernorm.weight" || suffix == "attn_norm.weight") { 
                if (tinfo.type == GGMLType::GGML_TYPE_F32) { assign_vec_f32(lw.input_layernorm_f32, tinfo); Logger::info("Mapped GGUF tensor '" + name + "' (FP32) to layers[" + std::to_string(layer_idx) + "].input_layernorm_f32"); } 
                else if (tinfo.type == GGMLType::GGML_TYPE_F16) { assign_vec_f16(lw.input_layernorm_f32, tinfo); Logger::info("Mapped GGUF tensor '" + name + "' (FP16) to layers[" + std::to_string(layer_idx) + "].input_layernorm_f32"); } 
                else { assign_vec_bf16(lw.input_layernorm, tinfo); Logger::info("Mapped GGUF tensor '" + name + "' (BF16/Other) to layers[" + std::to_string(layer_idx) + "].input_layernorm"); }
                Logger::info("-> Successfully assigned '" + name + "' to layer " + std::to_string(layer_idx));
                continue;
            }
            // LAYER NORM (FFN)
            if (suffix == "post_attention_layernorm.weight" || suffix == "ffn_norm.weight") { 
                if (tinfo.type == GGMLType::GGML_TYPE_F32) { assign_vec_f32(lw.post_attention_layernorm_f32, tinfo); Logger::info("Mapped GGUF tensor '" + name + "' (FP32) to layers[" + std::to_string(layer_idx) + "].post_attention_layernorm_f32"); } 
                else if (tinfo.type == GGMLType::GGML_TYPE_F16) { assign_vec_f16(lw.post_attention_layernorm_f32, tinfo); Logger::info("Mapped GGUF tensor '" + name + "' (FP16) to layers[" + std::to_string(layer_idx) + "].post_attention_layernorm_f32"); } 
                else { assign_vec_bf16(lw.post_attention_layernorm, tinfo); Logger::info("Mapped GGUF tensor '" + name + "' (BF16/Other) to layers[" + std::to_string(layer_idx) + "].post_attention_layernorm"); }
                Logger::info("-> Successfully assigned '" + name + "' to layer " + std::to_string(layer_idx));
                continue;
            }
        }
        Logger::warning("Unmapped GGUF tensor: '" + name + "' with type: " + std::to_string(static_cast<int>(tinfo.type)));
    }
    Logger::info("Finished mapping GGUF weights.");
}

// --- START: Definition for parse_model_config_from_gguf ---
ModelConfig parse_model_config_from_gguf(const GGUFData& gguf) {
    Logger::info("Parsing ModelConfig from GGUF metadata...");
    ModelConfig cfg;
    const auto& meta = gguf.metadata;

    // Helper to get metadata value or default
    auto get_meta_value = [&](const std::string& key, auto default_value) {
        using TargetType = decltype(default_value);
        auto it = meta.find(key);
        if (it == meta.end()) {
            // Logger::info("Metadata key '" + key + "' not found. Using default: " + std::to_string(default_value));
            return default_value;
        }

        TargetType result_value = default_value;
        bool conversion_ok = false;

        try {
            std::visit([&](const auto& stored_val) {
                using StoredType = std::decay_t<decltype(stored_val)>;

                // Direct assignment if types match
                if constexpr (std::is_same_v<StoredType, TargetType>) {
                    result_value = stored_val;
                    conversion_ok = true;
            }
                // Allow conversion from any integer/float to float/double target
                else if constexpr (std::is_floating_point_v<TargetType> && (std::is_integral_v<StoredType> || std::is_floating_point_v<StoredType>)) {
                    result_value = static_cast<TargetType>(stored_val);
                    conversion_ok = true;
                }
                // Allow conversion from uint32_t/int32_t/uint16_t/int16_t to int/size_t/uint64_t target with range check
                else if constexpr (std::is_integral_v<TargetType> &&
                                   (std::is_same_v<StoredType, uint32_t> || std::is_same_v<StoredType, int32_t> ||
                                    std::is_same_v<StoredType, uint16_t> || std::is_same_v<StoredType, int16_t>)) {
                    // Basic range check example (can be more sophisticated)
                    if (static_cast<long long>(stored_val) >= static_cast<long long>(std::numeric_limits<TargetType>::min()) &&
                        static_cast<long long>(stored_val) <= static_cast<long long>(std::numeric_limits<TargetType>::max())) {
                         result_value = static_cast<TargetType>(stored_val);
                         conversion_ok = true;
    } else {
                         Logger::warning("Metadata value for key '" + key + "' ('" + std::to_string(stored_val) + "') out of range for target type. Using default."); // FIXED typo: '' -> "')"
                     }
                }
                 // Add more specific conversions if needed (e.g., string to number)
                 else if constexpr (std::is_same_v<StoredType, std::string> && std::is_arithmetic_v<TargetType>) {
                    try {
                        std::string s = stored_val;
                        if constexpr (std::is_integral_v<TargetType>) { result_value = static_cast<TargetType>(std::stoll(s)); }
                        else if constexpr (std::is_floating_point_v<TargetType>) { result_value = static_cast<TargetType>(std::stof(s)); }
                         conversion_ok = true;
                    } catch (const std::exception& e) {
                        Logger::warning("Metadata string-to-numeric conversion failed for key '" + key + "', value: '" + stored_val + "': " + e.what());
                    }
                 }
                // else { // Optional: Log if no specific conversion path was taken but types differ
                //     if constexpr (!std::is_same_v<StoredType, TargetType>) {
                //         Logger::warning("No explicit conversion logic for key '" + key + "' from stored type to target type. Using default.");
                //     }
                // }
            }, it->second); // Access variant directly
        } catch (const std::bad_variant_access& e) {
            Logger::error("Bad variant access for key '" + key + "': " + e.what());
        } catch (const std::exception& e) {
            Logger::error("Error processing metadata value for key '" + key + "': " + e.what());
        }

        if (!conversion_ok) {
             Logger::warning("Metadata conversion failed or not applicable for key '" + key + "'. Using default value: " + std::to_string(default_value));
             // Ensure default is returned if conversion flag wasn't set
             result_value = default_value;
        }
        return result_value;
    };

     auto get_meta_string = [&](const std::string& key, const std::string& default_value) {
        auto it = meta.find(key);
        if (it != meta.end() && std::holds_alternative<std::string>(it->second)) { // FIXED: Removed .value
            return std::get<std::string>(it->second); // FIXED: Removed .value
        }
        // Logger::info("Metadata key '" + key + "' not found or not a string. Using default value: " + default_value);
        return default_value;
    };

    // --- Extract config values using helpers ---
    // Try common variations of keys found in different GGUF files
    cfg.hidden_size = get_meta_value("llama.embedding_length", 0);
    cfg.intermediate_size = get_meta_value("llama.feed_forward_length", 0);
    cfg.num_attention_heads = get_meta_value("llama.attention.head_count", 0);
    cfg.num_key_value_heads = get_meta_value("llama.attention.head_count_kv", cfg.num_attention_heads); // Default to n_heads if kv not present
    cfg.num_hidden_layers = get_meta_value("llama.block_count", 0);
    cfg.max_position_embeddings = get_meta_value("llama.context_length", 0);
    cfg.rms_norm_eps = get_meta_value("llama.attention.layer_norm_rms_epsilon", 1e-5f);
    cfg.rope_theta = get_meta_value("llama.rope.freq_base", 10000.0f);
    // cfg.hidden_act = get_meta_string("llama.activation_function", "silu"); // Activation func usually not in metadata directly? Defaulting.
    cfg.bos_token_id = get_meta_value("tokenizer.bos_token_id", 1);
    cfg.eos_token_id = get_meta_value("tokenizer.eos_token_id", 2);

    // --- Vocab Size ---
    auto tokens_it = meta.find("tokenizer.ggml.tokens");
    if (tokens_it != meta.end() && std::holds_alternative<GGUFArray>(tokens_it->second)) { // FIXED: Removed .value
        const auto& token_array_info = std::get<GGUFArray>(tokens_it->second); // FIXED: Removed .value
        cfg.vocab_size = static_cast<int>(token_array_info.len);
         Logger::info("Derived vocab_size (" + std::to_string(cfg.vocab_size) + ") from tokenizer.ggml.tokens array length.");
    } else {
        Logger::warning("Could not find 'tokenizer.ggml.tokens' array in metadata to derive vocab_size. Defaulting to 0.");
        cfg.vocab_size = 0;
    }

    // --- File Type / Quantization ---
    auto file_type_it = meta.find("general.file_type");
    if (file_type_it != meta.end()) {
         if (std::holds_alternative<uint32_t>(file_type_it->second)) { // FIXED: Removed .value
             uint32_t file_type_enum = std::get<uint32_t>(file_type_it->second); // FIXED: Removed .value
             Logger::info("Found general.file_type enum: " + std::to_string(file_type_enum));
                    } else {
              Logger::warning("Metadata 'general.file_type' exists but is not a uint32_t.");
                    }
                } else {
        Logger::warning("Metadata key 'general.file_type' not found.");
                }

    // --- torch_dtype equivalent ---
    std::string inferred_dtype = "unknown";
    // Prefer checking a non-quantized weight if possible, like norm weights
    std::string norm_weight_key = "";
    if (gguf.tensor_infos_map.count("output_norm.weight")) norm_weight_key = "output_norm.weight";
    else if (gguf.tensor_infos_map.count("norm.weight")) norm_weight_key = "norm.weight";

    if (!norm_weight_key.empty()) {
       const auto& tensor_info = gguf.tensor_infos_map.at(norm_weight_key);
       GGMLType type = tensor_info.type;
       if (type == GGMLType::GGML_TYPE_F32) inferred_dtype = "float32";
       else if (type == GGMLType::GGML_TYPE_F16) inferred_dtype = "float16";
       // REMOVED Check for non-existent GGML_TYPE_BF16
       // else if (type == GGML_TYPE_BF16) inferred_dtype = "bfloat16";
       // If norm is quantized, maybe check embedding?
       else if (inferred_dtype == "unknown" && gguf.tensor_infos_map.count("token_embd.weight")) {
            const auto& embed_info = gguf.tensor_infos_map.at("token_embd.weight");
            GGMLType embed_type = embed_info.type;
            if (embed_type == GGMLType::GGML_TYPE_F32) inferred_dtype = "float32";
            else if (embed_type == GGMLType::GGML_TYPE_F16) inferred_dtype = "float16";
            // REMOVED Check for non-existent GGML_TYPE_BF16
            // else if (embed_type == GGML_TYPE_BF16) inferred_dtype = "bfloat16";
       }
        Logger::info("Inferred model dtype '" + inferred_dtype + "' from tensor '" + norm_weight_key + "' or embedding.");
            } else {
         Logger::warning("Could not find a norm weight or embedding tensor to infer model dtype.");
        }
    cfg.torch_dtype = inferred_dtype;


    // --- Validation & Logging ---
    bool config_ok = true;
    if (cfg.hidden_size <= 0) { Logger::error("FATAL: Failed to parse 'hidden_size' (llama.embedding_length) from GGUF metadata."); config_ok = false; }
    if (cfg.intermediate_size <= 0) { Logger::error("FATAL: Failed to parse 'intermediate_size' (llama.feed_forward_length) from GGUF metadata."); config_ok = false; }
    if (cfg.num_attention_heads <= 0) { Logger::error("FATAL: Failed to parse 'num_attention_heads' (llama.attention.head_count) from GGUF metadata."); config_ok = false; }
    if (cfg.num_key_value_heads <= 0) { Logger::error("FATAL: Failed to parse 'num_key_value_heads' (llama.attention.head_count_kv) from GGUF metadata."); config_ok = false; }
    if (cfg.num_hidden_layers <= 0) { Logger::error("FATAL: Failed to parse 'num_hidden_layers' (llama.block_count) from GGUF metadata."); config_ok = false; }
    if (cfg.vocab_size <= 0) { Logger::error("FATAL: Failed to determine 'vocab_size' from GGUF metadata ('tokenizer.ggml.tokens')."); config_ok = false; }
    if (cfg.max_position_embeddings <= 0) { Logger::error("FATAL: Failed to parse 'max_position_embeddings' (llama.context_length) from GGUF metadata."); config_ok = false; }

    if (!config_ok) {
        // If Logger::error doesn't exit, we need to explicitly stop.
         throw std::runtime_error("Failed to parse required configuration fields from GGUF metadata. Check logs.");
    }

    Logger::info("Finished parsing ModelConfig from GGUF.");
    Logger::info("Parsed Config: HS=" + std::to_string(cfg.hidden_size) +
                 ", IS=" + std::to_string(cfg.intermediate_size) +
                 ", NHL=" + std::to_string(cfg.num_hidden_layers) +
                 ", NHEAD=" + std::to_string(cfg.num_attention_heads) +
                 ", NKVHEAD=" + std::to_string(cfg.num_key_value_heads) +
                 ", VS=" + std::to_string(cfg.vocab_size) +
                 ", MaxSeq=" + std::to_string(cfg.max_position_embeddings) +
                 ", DType=" + cfg.torch_dtype);

    return cfg;
}
// --- END: Definition for parse_model_config_from_gguf ---
