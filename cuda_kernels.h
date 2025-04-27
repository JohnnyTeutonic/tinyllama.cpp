#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

// Only define these functions if CUDA is enabled during compilation
#ifdef HAS_CUDA

#include <vector>
#include <cstdint> // For uint16_t

// --- CUDA Error Checking Utility --- 
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include "logger.h"

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      std::string err_msg = "GPUassert: " + std::string(cudaGetErrorString(code)) + " " + std::string(file) + " " + std::to_string(line);
      Logger::error(err_msg); // Use your logger
      if (abort) throw std::runtime_error(err_msg);
   }
}
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }


// --- Kernel Wrapper Declarations --- 

/**
 * @brief Performs RMS Normalization on the GPU (device pointers version).
 * @param x_dev Input vector (device pointer).
 * @param weight_dev Normalization weights (device pointer).
 * @param out_dev Output vector (device pointer).
 * @param n Size of the vectors.
 * @param eps Epsilon value for numerical stability.
 * @param stream CUDA stream (optional, default 0).
 */
void rmsnorm_vector_cuda(const float* x_dev,
                         const float* weight_dev,
                         float* out_dev,
                         int n,
                         float eps,
                         cudaStream_t stream = 0);

// (Optional: keep the host vector overload for now)
void rmsnorm_vector_cuda(const std::vector<float>& x_in_host,
                         const std::vector<float>& weight_host,
                         std::vector<float>& out_host,
                         int n,
                         float eps);

/**
 * @brief Host-vector version: for use with std::vector<float> input/output.
 */
void matvec_bf16_f32_cuda(const std::vector<uint16_t>& mat_bf16_host,
                          const std::vector<float>& vec_f32_host,
                          std::vector<float>& out_f32_host,
                          int rows,
                          int cols);

/**
 * @brief Device-pointer version: for use with device input/output buffers.
 */
void matvec_bf16_f32_cuda(const std::vector<uint16_t>& mat_bf16_host,
                          const float* vec_f32_dev,
                          float* out_f32_dev,
                          int rows,
                          int cols,
                          cudaStream_t stream = 0);

/**
 * @brief Performs element-wise SiLU activation (x * sigmoid(x)) on the GPU.
 * 
 * @param x_host Input vector (host memory).
 * @param out_host Output vector (host memory, will be resized and filled).
 * @param n Size of the vectors.
 */
void silu_cuda(const std::vector<float>& x_host,
               std::vector<float>& out_host,
               int n);

/**
 * @brief Computes the Softmax of a vector on the GPU.
 * 
 * @param x_host Input vector (host memory).
 * @param out_host Output vector (host memory, will be resized and filled).
 * @param n Size of the vectors.
 */
void softmax_vector_cuda(const std::vector<float>& x_host, 
                         std::vector<float>& out_host,
                         int n);

void rope_cuda(std::vector<float>& x, int num_heads, int head_dim, const std::vector<float>& freqs_cis);

// --- Moved SwiGLU declaration OUTSIDE HAS_CUDA --- 
void swiglu_cuda(const std::vector<float>& gate_host,
                 const std::vector<float>& up_host,
                 std::vector<float>& out_host,
                 int n);

#endif // HAS_CUDA

#endif // CUDA_KERNELS_H 