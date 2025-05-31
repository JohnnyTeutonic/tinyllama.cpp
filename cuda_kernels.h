#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#include "cuda_compat_windows.h"

#ifdef HAS_CUDA

#ifdef _WIN32
#ifndef CUDA_NO_HALF
#define CUDA_NO_HALF
#endif
#define __CUDA_NO_HALF_OPERATORS__
#define __CUDA_NO_HALF_CONVERSIONS__
#define __CUDA_NO_HALF2_OPERATORS__
#define __CUDA_NO_BFLOAT16_CONVERSIONS__
#define CUDA_NO_HALF_OPERATORS
#define CUDA_NO_HALF_CONVERSIONS
#define __CUDA_NO_BFLOAT16_CONVERSIONS__
#endif

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "logger.h"

/**
 * @file cuda_kernels.h
 * @brief CUDA kernel wrappers for GPU-accelerated tensor operations in TinyLlama
 *
 * This file provides C++ wrappers around CUDA kernels used in the TinyLlama
 * implementation. It includes functions for matrix operations, attention
 * mechanisms, and various tensor manipulations optimized for GPU execution.
 * 
 * Key features:
 * - RMS normalization for transformer layers
 * - Matrix-vector multiplication with FP32 and BF16 support
 * - Attention mechanism with key-value caching
 * - Rotary Position Embeddings (RoPE)
 * - SiLU activation function
 * - Softmax normalization
 * - Residual connections
 * 
 * The implementation supports both FP32 and BF16 (Brain Floating Point) formats,
 * with automatic conversion between formats when needed. All operations are
 * optimized for GPU execution using CUDA and cuBLAS.
 */

/**
 * @brief CUDA error checking and assertion utilities
 * 
 * These utilities provide robust error checking for CUDA operations.
 * The gpuAssert function and gpuErrchk macro ensure proper error handling
 * and reporting for CUDA operations.
 */

/**
 * @brief Asserts CUDA operations for error checking
 * 
 * Helper function to check CUDA operations and throw exceptions on errors.
 * Used internally by the gpuErrchk macro.
 * 
 * @param code CUDA error code to check
 * @param file Source file where the error occurred
 * @param line Line number where the error occurred
 * @param abort Whether to throw an exception on error
 * @throws std::runtime_error if a CUDA error occurs and abort is true
 */
inline void gpuAssert(cudaError_t code, const char* file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    std::string err_msg =
        "GPUassert: " + std::string(cudaGetErrorString(code)) + " " +
        std::string(file) + " " + std::to_string(line);
    Logger::error(err_msg);
    if (abort) throw std::runtime_error(err_msg);
  }
}

/**
 * @brief Macro for GPU error checking
 * @param ans CUDA operation to check
 */
#define gpuErrchk(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }

/**
 * @brief RMS Normalization Operations
 * 
 * These functions implement Root Mean Square (RMS) normalization, which is
 * a key component in transformer architectures. The implementation includes
 * both device-side and host-side versions for flexibility.
 */

/**
 * @brief Performs RMS normalization on GPU with device pointers
 * 
 * Computes root mean square normalization of input tensor on GPU.
 * 
 * @param x_dev Input tensor (device pointer)
 * @param weight_dev Normalization weights (device pointer)
 * @param out_dev Output tensor (device pointer)
 * @param n Size of tensors
 * @param eps Epsilon for numerical stability
 * @param stream CUDA stream (optional)
 */
void rmsnorm_vector_cuda(const float* x_dev, const float* weight_dev,
                         float* out_dev, int n, float eps,
                         cudaStream_t stream = 0);

/**
 * @brief Performs RMS normalization on GPU with host vectors
 * 
 * Host-side version that handles device memory allocation internally.
 * 
 * @param x_in_host Input tensor (host vector)
 * @param weight_host Weight tensor (host vector)
 * @param out_host Output tensor (host vector)
 * @param n Size of tensors
 * @param eps Epsilon for numerical stability
 */
void rmsnorm_vector_cuda(const std::vector<float>& x_in_host,
                         const std::vector<float>& weight_host,
                         std::vector<float>& out_host, int n, float eps);

// New Batch RMSNorm
void rmsnorm_batch_cuda(float* d_out, float* d_in, const float* d_weight, 
                        int num_tokens, int hidden_size, float eps, 
                        cudaStream_t stream);

__global__ void reduce_partial_sums_kernel(const float* partial_sums, float* total_sum_sq_out, int num_partial_sums);

/**
 * @brief Matrix-Vector Multiplication Operations
 * 
 * These functions implement matrix-vector multiplication using cuBLAS,
 * supporting both FP32 and BF16 formats. The implementation includes:
 * - Direct device pointer operations
 * - Host vector wrappers with automatic memory management
 * - Support for different precisions (FP32, BF16)
 */

/**
 * @brief Matrix-vector multiplication on GPU with host vectors
 * 
 * Performs matrix-vector multiplication using cuBLAS, handling device
 * memory allocation internally.
 * 
 * @param handle cuBLAS handle
 * @param mat_f32_host Matrix in row-major format (host vector)
 * @param vec_f32_host Vector to multiply (host vector)
 * @param out_f32_host Output vector (host vector)
 * @param rows Number of matrix rows
 * @param cols Number of matrix columns
 */
void matvec_f32_f32_cuda(cublasHandle_t handle,
                         const std::vector<float>& mat_f32_host,
                         const std::vector<float>& vec_f32_host,
                         std::vector<float>& out_f32_host, int rows, int cols);

/**
 * @brief Matrix-vector multiplication on GPU with device pointers
 * 
 * Device-side version for use with pre-allocated device memory.
 * 
 * @param handle cuBLAS handle
 * @param mat_f32_dev Matrix in row-major format (device pointer)
 * @param vec_f32_dev Vector to multiply (device pointer)
 * @param out_f32_dev Output vector (device pointer)
 * @param rows Number of matrix rows
 * @param cols Number of matrix columns
 * @param stream CUDA stream (optional)
 */
void matvec_f32_f32_cuda(cublasHandle_t handle, const float* mat_f32_dev,
                        const float* vec_f32_dev, float* out_f32_dev,
                        int rows, int cols, cudaStream_t stream = 0);

// New Generic GEMM for FP32 (if needed, can be adapted for BF16 later or use cublasGemmEx)
void gemm_f32_f32_cuda(cublasHandle_t handle, 
                       bool transa, bool transb, 
                       int m, int n, int k, 
                       const float* alpha, 
                       const float* A, int lda, 
                       const float* B, int ldb, 
                       const float* beta, 
                       float* C, int ldc, 
                       cudaStream_t stream);

/**
 * @brief Activation Functions
 * 
 * GPU-accelerated implementations of common activation functions:
 * - SiLU (Sigmoid Linear Unit)
 * - Softmax
 */

/**
 * @brief SiLU activation function on GPU
 * 
 * Computes x * sigmoid(x) element-wise on GPU.
 *
 * @param x_host Input tensor (host vector)
 * @param out_host Output tensor (host vector)
 * @param n Size of tensors
 */
void silu_cuda(const std::vector<float>& x_host,
               std::vector<float>& out_host, int n);

/**
 * @brief Softmax function on GPU
 * 
 * Computes softmax normalization on GPU.
 *
 * @param x_host Input tensor (host vector)
 * @param out_host Output tensor (host vector)
 * @param n Size of tensors
 */
void softmax_vector_cuda(const std::vector<float>& x_host,
                         std::vector<float>& out_host, int n);

/**
 * @brief Attention Mechanism Operations
 * 
 * These functions implement the attention mechanism used in transformer
 * architectures, including:
 * - Key-Value caching for efficient inference
 * - Rotary Position Embeddings (RoPE)
 * - Multi-head attention computation
 */

/**
 * @brief Applies rotary position embeddings on GPU
 * 
 * @param vec Vector to apply RoPE to (device pointer)
 * @param num_heads Number of attention heads
 * @param head_dim Dimension of each head
 * @param freqs_cis_dev Complex rotation frequencies (device pointer)
 * @param pos Current sequence position
 * @param use_adjacent_pairing Whether to use adjacent pairing
 * @param stream CUDA stream (optional)
 */
void rope_cuda(float* vec, int num_heads, int head_dim,
               const float* freqs_cis_dev, int pos, bool use_adjacent_pairing, cudaStream_t stream);

/**
 * @brief Computes attention scores and values on GPU
 * 
 * Performs the attention mechanism computation using the key-value cache.
 *
 * @param Q_current_dev Current query vectors (device pointer)
 * @param K_layer_cache_base Key cache base pointer (device pointer)
 * @param V_layer_cache_base Value cache base pointer (device pointer)
 * @param out_dev Output attention vectors (device pointer)
 * @param num_heads Number of attention heads
 * @param current_seq_len Current sequence length
 * @param head_dim Dimension of each head
 * @param scale Attention scale factor
 * @param cache_max_seq_len Maximum sequence length in cache
 * @param cache_num_kv_heads Number of key-value heads
 * @param stream CUDA stream (optional)
 */
void attention_cuda(const float* Q_current_dev, const float* K_layer_cache_base,
                    const float* V_layer_cache_base, float* out_dev,
                    int num_heads, int current_seq_len, int head_dim,
                    float scale, int cache_max_seq_len, int cache_num_kv_heads,
                    cudaStream_t stream = 0);

/**
 * @brief Computes attention with selective dequantization of quantized KVCache
 * 
 * Performs attention computation by selectively dequantizing only the required 
 * portions of the INT8 KVCache for each head, significantly reducing memory usage.
 *
 * @param Q_current_dev Current query vectors (device pointer)
 * @param K_quantized_cache_base Quantized INT8 K cache base pointer (device pointer)
 * @param V_quantized_cache_base Quantized INT8 V cache base pointer (device pointer) 
 * @param K_scales_cache_base K cache scales (device pointer)
 * @param V_scales_cache_base V cache scales (device pointer)
 * @param selective_k_dequant_buffer Temporary K dequantization buffer (device pointer)
 * @param selective_v_dequant_buffer Temporary V dequantization buffer (device pointer)
 * @param out_dev Output attention vectors (device pointer)
 * @param num_heads Number of attention heads
 * @param current_seq_len Current sequence length
 * @param head_dim Dimension of each head
 * @param scale Attention scale factor
 * @param cache_max_seq_len Maximum sequence length in cache
 * @param cache_num_kv_heads Number of key-value heads
 * @param stream CUDA stream (optional)
 */
void attention_cuda_selective_dequant(const float* Q_current_dev, 
                                     const int8_t* K_quantized_cache_base,
                                     const int8_t* V_quantized_cache_base,
                                     const float* K_scales_cache_base,
                                     const float* V_scales_cache_base,
                                     float* selective_k_dequant_buffer,
                                     float* selective_v_dequant_buffer,
                                     float* out_dev,
                                     int num_heads, int current_seq_len, int head_dim,
                                     float scale, int cache_max_seq_len, int cache_num_kv_heads,
                                     cudaStream_t stream = 0);

/**
 * @brief Vector Operations
 * 
 * Basic vector operations optimized for GPU execution:
 * - Element-wise addition
 * - Residual connections
 * - Embedding lookups
 */

/**
 * @brief Element-wise vector addition on GPU
 *
 * @param a_dev First input vector (device pointer)
 * @param b_dev Second input vector (device pointer)
 * @param result_dev Output vector (device pointer)
 * @param n Size of vectors
 * @param stream CUDA stream (optional)
 */
void add_vectors_cuda(const float* a_dev, const float* b_dev,
                     float* result_dev, int n, cudaStream_t stream = 0);

/**
 * @brief Adds residual connection on GPU
 * 
 * @param matvec_out_dev Matrix output (device pointer)
 * @param residual_dev Residual tensor (device pointer)
 * @param result_dev Output tensor (device pointer)
 * @param n Size of tensors
 * @param stream CUDA stream (optional)
 */
void add_residual_cuda(const float* matvec_out_dev, const float* residual_dev,
                       float* result_dev, int n, cudaStream_t stream = 0);

/**
 * @brief Updates key-value cache on GPU
 *
 * @param cache_base_ptr Cache base pointer (device pointer)
 * @param current_kv_vector Current KV vector (device pointer)
 * @param pos Current sequence position
 * @param kv_head_idx KV head index
 * @param max_seq_len Maximum sequence length
 * @param num_kv_heads Number of KV heads
 * @param head_dim Head dimension
 * @param stream CUDA stream (optional)
 */
void update_kv_cache_cuda(float* cache_base_ptr,
                         const float* current_kv_vector,
                          int pos, int kv_head_idx, int max_seq_len,
                          int num_kv_heads, int head_dim,
                          cudaStream_t stream = 0);

/**
 * @brief Applies RoPE and updates KV cache in one operation
 * 
 * @param cache_base_ptr Cache base pointer (device pointer)
 * @param kv_vector_head KV vector head (device pointer)
 * @param all_freqs_cis_base RoPE frequencies (device pointer)
 * @param pos Current sequence position
 * @param kv_head_idx KV head index
 * @param max_seq_len Maximum sequence length
 * @param num_kv_heads Number of KV heads
 * @param head_dim Head dimension
 * @param stream CUDA stream (optional)
 */
void rope_and_update_kv_cache_cuda(float* cache_base_ptr,
                                   const float* kv_vector_head,
                                  const float* all_freqs_cis_base,
                                  int pos, int kv_head_idx, int max_seq_len,
                                   int num_kv_heads, int head_dim,
                                   cudaStream_t stream = 0);

/**
 * @brief Applies SwiGLU activation on GPU
 * 
 * @param gate_dev Gate tensor (device pointer)
 * @param up_dev Up-projection tensor (device pointer)
 * @param out_dev Output tensor (device pointer)
 * @param n Size of tensors
 * @param stream CUDA stream (optional)
 */
void swiglu_cuda(const float* gate_dev, const float* up_dev,
                 float* out_dev, int n, cudaStream_t stream = 0);

/**
 * @brief Performs embedding lookup with BF16 to F32 conversion
 * 
 * @param embedding_table_dev Embedding table (device pointer)
 * @param output_vector_dev Output vector (device pointer)
 * @param token_id Token ID to look up
 * @param hidden_size Hidden dimension size
 * @param vocab_size Vocabulary size
 * @param stream CUDA stream (optional)
 */
void lookup_embedding_bf16_f32_cuda(const uint16_t* embedding_table_dev,
                                   float* output_vector_dev,
                                   int token_id, int hidden_size,
                                   int vocab_size, cudaStream_t stream = 0);

/**
 * @brief Generic embedding lookup supporting multiple formats
 * 
 * @param table_dev Embedding table (device pointer)
 * @param output_dev Output vector (device pointer)
 * @param token_id Token ID to look up
 * @param hidden_size Hidden dimension size
 * @param vocab_size Vocabulary size
 * @param is_bf16 Whether table is in BF16 format
 * @param stream CUDA stream (optional)
 */
void lookup_embedding_cuda(const void* table_dev, float* output_dev,
                           int token_id, int hidden_size, int vocab_size,
                           bool is_bf16, cudaStream_t stream);

/**
 * @brief Matrix-vector multiplication with BF16 matrix and F32 vector
 * 
 * @param handle cuBLAS handle
 * @param mat_bf16_dev BF16 matrix (device pointer)
 * @param vec_f32_dev F32 vector (device pointer)
 * @param out_f32_dev Output vector (device pointer)
 * @param rows Number of matrix rows
 * @param cols Number of matrix columns
 * @param stream CUDA stream (optional)
 */
void matvec_bf16_f32_cuda(cublasHandle_t handle,
                         const uint16_t* mat_bf16_dev,
                         const float* vec_f32_dev,
                         float* out_f32_dev,
                         int rows, int cols,
                         cudaStream_t stream = 0);

/**
 * @brief CUDA Kernel: Converts a block of BF16 values to FP32 values.
 * 
 * This kernel performs an element-wise conversion from Brain Floating Point 16 (BF16)
 * format to standard IEEE 754 single-precision floating point (FP32) format.
 * It is designed to be launched with enough threads to cover all elements in the input array.
 *
 * @param bf16_in Pointer to the input array of BF16 values (uint16_t) on the device.
 * @param fp32_out Pointer to the output array for FP32 values (float) on the device.
 * @param n_elements The total number of elements to convert.
 */
__global__ void convert_bf16_to_fp32_kernel(const uint16_t* __restrict__ bf16_in,
                                            float* __restrict__ fp32_out,
                                            size_t n_elements);

// KVCache Quantization Kernels (FP32 <-> INT8)

/**
 * @brief Quantizes an FP32 vector to INT8 with a per-tensor symmetric scale.
 *
 * @param fp32_in_dev Device pointer to the input FP32 vector.
 * @param int8_out_dev Device pointer to the output INT8 vector.
 * @param scale_out_dev Device pointer to store the single FP32 scale factor.
 * @param num_elements Number of elements in the vector.
 * @param stream CUDA stream for asynchronous execution.
 */
void quantize_fp32_to_int8_symmetric_per_tensor_cuda(
    const float* fp32_in_dev, 
    int8_t* int8_out_dev, 
    float* scale_out_dev,
    int num_elements, 
    cudaStream_t stream = 0);

/**
 * @brief Dequantizes an INT8 vector to FP32 using a per-tensor symmetric scale.
 *
 * @param int8_in_dev Device pointer to the input INT8 vector.
 * @param scale_in_dev Device pointer to the single FP32 scale factor.
 * @param fp32_out_dev Device pointer to the output FP32 vector.
 * @param num_elements Number of elements in the vector.
 * @param stream CUDA stream for asynchronous execution.
 */
void dequantize_int8_to_fp32_symmetric_per_tensor_cuda(
    const int8_t* int8_in_dev, 
    const float* scale_in_dev, 
    float* fp32_out_dev,
    int num_elements, 
    cudaStream_t stream = 0);

// New Batched SwiGLU (SiLU + element-wise multiply)
void swiglu_batch_cuda(float* d_out_batch, // Output: [num_tokens, intermediate_size]
                       const float* d_gate_act_batch,   // Input: Gate activations [num_tokens, intermediate_size]
                       const float* d_up_act_batch,     // Input: Up activations [num_tokens, intermediate_size]
                       int num_tokens,
                       int intermediate_size,
                       cudaStream_t stream);

// New Batched RoPE
void rope_batch_cuda(float* d_q_batch, float* d_k_batch,
                     const float* d_all_freqs_cis_base, // Changed name
                     int num_tokens, int num_q_heads, int num_kv_heads, int head_dim,
                     int start_pos_offset,      // Changed name
                     bool use_adjacent_pairing, // Changed name and type
                     cudaStream_t stream);

// New Batched Attention for Prefill
void attention_batch_prefill_cuda(
    const float* d_q_batch_strided,   // Input Q: [B, H_q, D_h]
    const float* d_k_batch_strided,   // Input K for current batch
    const float* d_v_batch_strided,   // Input V for current batch
    float* d_kv_cache_k_base,         // K Cache: [S_max, H_kv, D_h]
    float* d_kv_cache_v_base,         // V Cache: [S_max, H_kv, D_h]
    float* d_output_batch_strided,    // Output: [B, H_q, D_h]
    int num_tokens_in_batch,          // B
    int start_pos_in_kv_cache,        // Start position for this batch in KV cache
    int cache_max_seq_len,            // Max capacity of KV cache
    int num_q_heads,                  // H_q
    int num_kv_heads,                 // H_kv
    int head_dim,                     // D_h
    float scale,
    cudaStream_t stream,
    const int* attention_mask_cu = nullptr // Optional attention mask, changed name
);

// New Batched Add Residual
void add_residual_batch_cuda(float* d_output_batch, // Output: [num_tokens, hidden_size]
                             const float* d_input_a_batch,   // Input A: [num_tokens, hidden_size]
                             const float* d_input_b_batch,   // Input B: [num_tokens, hidden_size]
                             int num_tokens, int hidden_size,
                             cudaStream_t stream);

// New Batched KV Cache Update (FP32 example)
// This function seems to be missing a corresponding definition for separate K and V caches.
// The existing update_kv_cache_batch_cuda takes a single cache layer base.
// For now, I will comment out the _fp32 version and assume the generic one is intended to be used twice.
/*
void update_kv_cache_batch_cuda_fp32(
    float* d_kvcache_k, float* d_kvcache_v,
    const float* d_k_batch_current, const float* d_v_batch_current,
    int num_tokens_in_batch, int start_pos_in_kv_cache,
    int max_seq_len_in_cache, int num_kv_heads, int head_dim,
    cudaStream_t stream
);
*/
// Assuming the following is the intended generic function for batch KV update:
void update_kv_cache_batch_cuda(
    float* d_kv_cache_layer_base,        // Device pointer to the K or V cache for the current layer
    const float* d_keys_or_values_batch, // Device pointer to the batch of K or V vectors to be written
    int start_pos_in_kv_cache,           // The sequence position in the cache where writing for this batch should begin
    int num_tokens_in_batch,             // Number of tokens in the d_keys_or_values_batch
    int num_kv_heads,                    // Number of K/V heads
    int head_dim,                        // Dimension of each K/V head
    int cache_max_seq_len,               // Maximum sequence length capacity of the cache
    cudaStream_t stream
);



#endif // HAS_CUDA

#endif // CUDA_KERNELS_H