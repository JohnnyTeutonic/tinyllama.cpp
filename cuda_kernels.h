#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#ifdef HAS_CUDA

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "logger.h"

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
#define gpuErrchk(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }

/**
 * @brief Performs RMS Normalization on the GPU (device pointers version).
 * @param x_dev Input vector (device pointer).
 * @param weight_dev Normalization weights (device pointer).
 * @param out_dev Output vector (device pointer).
 * @param n Size of the vectors.
 * @param eps Epsilon value for numerical stability.
 * @param stream CUDA stream (optional, default 0).
 */
void rmsnorm_vector_cuda(const float* x_dev, const float* weight_dev,
                         float* out_dev, int n, float eps,
                         cudaStream_t stream = 0);

void rmsnorm_vector_cuda(const std::vector<float>& x_in_host,
                         const std::vector<float>& weight_host,
                         std::vector<float>& out_host, int n, float eps);

/**
 * @brief Host-vector version: Allocates all device buffers internally.
 */
void matvec_f32_f32_cuda(cublasHandle_t handle,
                         const std::vector<float>& mat_f32_host,
                         const std::vector<float>& vec_f32_host,
                         std::vector<float>& out_f32_host, int rows, int cols);

/**
 * @brief Device-pointer version: for use with device input/output buffers.
 */
void matvec_f32_f32_cuda(cublasHandle_t handle, const float* mat_f32_dev,
                         const float* vec_f32_dev, float* out_f32_dev, int rows,
                         int cols, cudaStream_t stream = 0);

/**
 * @brief Performs element-wise SiLU activation (x * sigmoid(x)) on the GPU.
 *
 * @param x_host Input vector (host memory).
 * @param out_host Output vector (host memory, will be resized and filled).
 * @param n Size of the vectors.
 */
void silu_cuda(const std::vector<float>& x_host, std::vector<float>& out_host,
               int n);

/**
 * @brief Computes the Softmax of a vector on the GPU.
 *
 * @param x_host Input vector (host memory).
 * @param out_host Output vector (host memory, will be resized and filled).
 * @param n Size of the vectors.
 */
void softmax_vector_cuda(const std::vector<float>& x_host,
                         std::vector<float>& out_host, int n);

void rope_cuda(float* vec, int num_heads, int head_dim,
               const float* freqs_cis_dev, int pos, cudaStream_t stream);

/**
 * @brief CUDA Attention kernel wrapper (Reads directly from flat K/V Cache)
 *
 * @param Q_current_dev Pointer to the current token's Q vector (all heads),
 * device memory.
 * @param K_layer_cache_base Pointer to the base of the K cache for the *current
 * layer*, device memory.
 * @param V_layer_cache_base Pointer to the base of the V cache for the *current
 * layer*, device memory.
 * @param out_dev Output attention vector (all heads), device memory.
 * @param num_heads Number of query heads.
 * @param current_seq_len The current sequence length (pos + 1).
 * @param head_dim Dimension of each attention head.
 * @param scale Scaling factor (1/sqrt(head_dim)).
 * @param cache_max_seq_len Max sequence length dimension of the cache.
 * @param cache_num_kv_heads Number of K/V heads dimension of the cache.
 * @param stream CUDA stream (optional, default 0).
 */
void attention_cuda(const float* Q_current_dev, const float* K_layer_cache_base,
                    const float* V_layer_cache_base, float* out_dev,
                    int num_heads, int current_seq_len, int head_dim,
                    float scale, int cache_max_seq_len, int cache_num_kv_heads,
                    cudaStream_t stream = 0);

/**
 * @brief Performs element-wise addition of two vectors on the GPU (result = a +
 * b).
 *
 * @param a_dev First input vector (device pointer).
 * @param b_dev Second input vector (device pointer).
 * @param result_dev Output vector (device pointer).
 * @param n Size of the vectors.
 * @param stream CUDA stream (optional, default 0).
 */
void add_vectors_cuda(const float* a_dev, const float* b_dev, float* result_dev,
                      int n, cudaStream_t stream = 0);

void add_residual_cuda(const float* matvec_out_dev, const float* residual_dev,
                       float* result_dev, int n, cudaStream_t stream = 0);

/**
 * @brief Updates a specific entry in the flat K/V cache on the device.
 *
 * @param cache_base_ptr Base pointer to the K or V cache for the specific
 * layer.
 * @param current_kv_vector Pointer to the K or V vector for the current token.
 * @param pos The current sequence position (timestep).
 * @param kv_head_idx The index of the K/V head to update.
 * @param max_seq_len Max sequence length dimension of the cache.
 * @param num_kv_heads Number of K/V heads dimension of the cache.
 * @param head_dim Head dimension of the cache.
 * @param stream CUDA stream (optional, default 0).
 */
void update_kv_cache_cuda(float* cache_base_ptr, const float* current_kv_vector,
                          int pos, int kv_head_idx, int max_seq_len,
                          int num_kv_heads, int head_dim,
                          cudaStream_t stream = 0);

void rope_and_update_kv_cache_cuda(float* cache_base_ptr,
                                   const float* kv_vector_head,
                                   const float* all_freqs_cis_base, int pos,
                                   int kv_head_idx, int max_seq_len,
                                   int num_kv_heads, int head_dim,
                                   cudaStream_t stream = 0);

void swiglu_cuda(const float* gate_dev, const float* up_dev, float* out_dev,
                 int n, cudaStream_t stream = 0);

void lookup_embedding_bf16_f32_cuda(const uint16_t* embedding_table_dev,
                                    float* output_vector_dev, int token_id,
                                    int hidden_size, int vocab_size,
                                    cudaStream_t stream = 0);

void matvec_f32_f32_cuda(cublasHandle_t handle, const float* mat_f32_dev,
                         const float* vec_f32_dev, float* out_f32_dev, int rows,
                         int cols, cudaStream_t stream);

void lookup_embedding_cuda(const void* table_dev, float* output_dev,
                           int token_id, int hidden_size, int vocab_size,
                           bool is_bf16, cudaStream_t stream);

void matvec_bf16_f32_cuda(cublasHandle_t handle, const uint16_t* mat_bf16_dev,
                          const float* vec_f32_dev, float* out_f32_dev,
                          int rows, int cols, cudaStream_t stream = 0);

#endif

#endif