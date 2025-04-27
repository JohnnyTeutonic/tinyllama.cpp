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
 * @brief Performs RMS Normalization on the GPU.
 * 
 * @param x_in_host Input vector (host memory).
 * @param weight_host Normalization weights (host memory).
 * @param out_host Output vector (host memory, will be resized and filled).
 * @param n Size of the vectors.
 * @param eps Epsilon value for numerical stability.
 */
void rmsnorm_vector_cuda(const std::vector<float>& x_in_host,
                         const std::vector<float>& weight_host,
                         std::vector<float>& out_host,
                         int n,
                         float eps);

/**
 * @brief Performs Matrix-Vector multiplication (BF16 * F32 -> F32) on the GPU.
 *
 * Computes out = mat * vec.
 * 
 * @param mat_bf16_host Input matrix (row-major, bfloat16, host memory).
 * @param vec_f32_host Input vector (float32, host memory).
 * @param out_f32_host Output vector (float32, host memory, will be resized and filled).
 * @param rows Number of rows in the matrix (size of output vector).
 * @param cols Number of columns in the matrix (size of input vector).
 */
void matvec_bf16_f32_cuda(const std::vector<uint16_t>& mat_bf16_host,
                          const std::vector<float>& vec_f32_host,
                          std::vector<float>& out_f32_host,
                          int rows,
                          int cols);


#endif // HAS_CUDA

#endif // CUDA_KERNELS_H 