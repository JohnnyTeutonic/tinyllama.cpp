/**
 * @file cuda_kernels.cu
 * @brief CUDA kernel implementations for TinyLlama GPU operations
 *
 * This file contains the actual CUDA kernel implementations for the GPU-accelerated
 * operations declared in cuda_kernels.h. The implementations are optimized for
 * NVIDIA GPUs and use CUDA-specific features for maximum performance.
 *
 * Key implementation details:
 * - Uses shared memory for efficient reduction operations
 * - Implements parallel reduction patterns for normalization
 * - Leverages cuBLAS for matrix operations
 * - Supports both FP32 and BF16 formats
 * - Includes error checking and memory management
 */

#include "cuda_compat_windows.h"

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

#include "cuda_kernels.h"
#include "logger.h"
#include "model_macros.h"

#ifdef HAS_CUDA

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#endif

#include <stdint.h>
#include <stdio.h>

#include <cmath>
#include <iostream>
#include <cfloat>

/**
 * @brief Converts BF16 to FP32 on the device
 * 
 * This device function converts Brain Floating Point (BF16) values to FP32.
 * It uses CUDA's native BF16 support on architectures >= 800, and falls back
 * to a manual conversion on older architectures.
 * 
 * @param bf16_raw Raw BF16 value as uint16_t
 * @return Converted FP32 value
 */
__device__ inline float bf16_to_float32_device(uint16_t bf16_raw) {
#if __CUDA_ARCH__ >= 800
  __nv_bfloat16 bf16_val;

  memcpy(&bf16_val, &bf16_raw, sizeof(uint16_t));
  return __bfloat162float(bf16_val);
#else

  unsigned int bits = ((unsigned int)bf16_raw) << 16;
  float result;
  memcpy(&result, &bits, sizeof(float));
  return result;
#endif
}

/**
 * @brief Kernel for computing sum of squares in RMS normalization
 * 
 * This kernel computes the sum of squares of input elements using parallel
 * reduction with shared memory. It's the first step in RMS normalization.
 * 
 * @param x Input tensor
 * @param partial_sums Output array for partial sums
 * @param n Size of input tensor
 */
__global__ void rmsnorm_sum_squares_kernel(const float* x, float* partial_sums,
                                           int n) {
  extern __shared__ float sdata[];

  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  sdata[tid] = (i < n) ? x[i] * x[i] : 0.0f;
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    partial_sums[blockIdx.x] = sdata[0];
  }
}

/**
 * @brief Kernel for applying RMS normalization
 * 
 * This kernel applies the normalization weights and scaling factor to the
 * input tensor. It's the second step in RMS normalization.
 * 
 * @param x Input tensor
 * @param weight Normalization weights
 * @param out Output tensor
 * @param n Size of tensors
 * @param inv_norm_factor Inverse of the normalization factor
 */
__global__ void rmsnorm_apply_kernel(const float* x, const float* weight,
                                     float* out, int n, float inv_norm_factor) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = x[i] * inv_norm_factor * weight[i];
  }
}

__global__ void reduce_partial_sums_kernel(const float* partial_sums, float* total_sum_sq_out, int num_partial_sums) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;

    float my_sum = 0.0f;
    for (unsigned int i = tid; i < num_partial_sums; i += blockDim.x) {
        my_sum += partial_sums[i];
    }
    sdata[tid] = my_sum;
    __syncthreads();

    // Standard parallel reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        total_sum_sq_out[0] = sdata[0];
    }
}

/**
 * @brief Implementation of RMS normalization with device pointers
 * 
 * This function implements the complete RMS normalization process:
 * 1. Computes sum of squares using parallel reduction
 * 2. Calculates normalization factor
 * 3. Applies normalization with weights
 * 
 * @param x_dev Input tensor (device pointer)
 * @param weight_dev Normalization weights (device pointer)
 * @param out_dev Output tensor (device pointer)
 * @param n Size of tensors
 * @param eps Epsilon for numerical stability
 * @param stream CUDA stream for asynchronous execution
 */
void rmsnorm_vector_cuda(const float* x_dev, const float* weight_dev,
                         float* out_dev, int n, float eps,
                         cudaStream_t stream) {
  const int threads_per_block = 256; // General purpose threads per block
  int num_blocks_reduce_pass1 = (n + threads_per_block - 1) / threads_per_block;
  size_t shared_mem_size_pass1 = threads_per_block * sizeof(float);

  float* partial_sums_dev = nullptr;
  gpuErrchk(cudaMalloc(&partial_sums_dev, num_blocks_reduce_pass1 * sizeof(float)));

  rmsnorm_sum_squares_kernel<<<num_blocks_reduce_pass1, threads_per_block,
                               shared_mem_size_pass1, stream>>>(x_dev,
                                                          partial_sums_dev, n);
  gpuErrchk(cudaGetLastError());

  float* total_sum_sq_dev = nullptr;
  gpuErrchk(cudaMalloc(&total_sum_sq_dev, sizeof(float)));

  const int threads_for_final_reduction = (num_blocks_reduce_pass1 <= 256) ? 256 :
                                          (num_blocks_reduce_pass1 <= 512) ? 512 : 1024; 
                                          // Cap at 1024, ensure kernel handles num_partial_sums correctly if it's less than blockDim.x
  size_t shared_mem_final_reduction = threads_for_final_reduction * sizeof(float);
  
  // Launch a single block for the final reduction of partial_sums_dev
  reduce_partial_sums_kernel<<<1, threads_for_final_reduction, 
                               shared_mem_final_reduction, stream>>>(
                                   partial_sums_dev, total_sum_sq_dev, num_blocks_reduce_pass1);
  gpuErrchk(cudaGetLastError());
  gpuErrchk(cudaFree(partial_sums_dev)); // Free intermediate partial sums

  float h_total_ssq = 0.0f;
  gpuErrchk(cudaMemcpyAsync(&h_total_ssq, total_sum_sq_dev, sizeof(float),
                       cudaMemcpyDeviceToHost, stream));
  gpuErrchk(cudaStreamSynchronize(stream)); // Ensure h_total_ssq is ready
  gpuErrchk(cudaFree(total_sum_sq_dev));

  double total_ssq_double = static_cast<double>(h_total_ssq);
  total_ssq_double /= n;
  float inv_norm_factor = 1.0f / SAFE_SQRT(static_cast<float>(total_ssq_double) + eps);

  int num_blocks_apply = (n + threads_per_block - 1) / threads_per_block;
  rmsnorm_apply_kernel<<<num_blocks_apply, threads_per_block, 0, stream>>>(
      x_dev, weight_dev, out_dev, n, inv_norm_factor);
  gpuErrchk(cudaGetLastError());
}

void rmsnorm_vector_cuda(const std::vector<float>& x_in_host,
                         const std::vector<float>& weight_host,
                         std::vector<float>& out_host, int n, float eps) {
  out_host.resize(n);
  float *x_dev = nullptr, *weight_dev = nullptr, *out_dev = nullptr;
  gpuErrchk(cudaMalloc(&x_dev, n * sizeof(float)));
  gpuErrchk(cudaMalloc(&weight_dev, n * sizeof(float)));
  gpuErrchk(cudaMalloc(&out_dev, n * sizeof(float)));
  gpuErrchk(cudaMemcpy(x_dev, x_in_host.data(), n * sizeof(float),
                       cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(weight_dev, weight_host.data(), n * sizeof(float),
                       cudaMemcpyHostToDevice));
  rmsnorm_vector_cuda(x_dev, weight_dev, out_dev, n, eps, 0);
  gpuErrchk(cudaDeviceSynchronize());
  gpuErrchk(cudaMemcpy(out_host.data(), out_dev, n * sizeof(float),
                       cudaMemcpyDeviceToHost));
  gpuErrchk(cudaFree(x_dev));
  gpuErrchk(cudaFree(weight_dev));
  gpuErrchk(cudaFree(out_dev));
}

/**
 * @brief Implementation of matrix-vector multiplication with FP32
 * 
 * This function performs matrix-vector multiplication using cuBLAS,
 * optimized for FP32 precision. It handles the cuBLAS setup and
 * error checking.
 * 
 * @param handle cuBLAS handle
 * @param mat_f32_dev Matrix in row-major format (device pointer)
 * @param vec_f32_dev Vector to multiply (device pointer)
 * @param out_f32_dev Output vector (device pointer)
 * @param rows Number of matrix rows
 * @param cols Number of matrix columns
 * @param stream CUDA stream for asynchronous execution
 */
void matvec_f32_f32_cuda(cublasHandle_t handle, const float* mat_f32_dev,
                         const float* vec_f32_dev, float* out_f32_dev, int rows,
                         int cols, cudaStream_t stream) {
  

  const float alpha = 1.0f;
  const float beta = 0.0f;
  int M_blas = rows;
  int N_blas = 1; 
  int K_blas = cols;

  cublasStatus_t status = cublasSetStream(handle, stream);
  if (status != CUBLAS_STATUS_SUCCESS) {
    Logger::error("cublasSetStream failed in matvec_f32_f32_cuda with error: " + std::to_string(status) + " (" + cublasGetStatusString(status) + ")");
    throw std::runtime_error("cublasSetStream failed");
  }
  
  status = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 
                       M_blas, N_blas, K_blas, 
                       &alpha, 
                       mat_f32_dev, K_blas,  // LDA for A when transA=T is original number of columns of A
                       vec_f32_dev, K_blas,  // LDB for B when transB=N is original number of rows of B
                       &beta, 
                       out_f32_dev, M_blas); // LDC for C is original number of rows of C

  if (status != CUBLAS_STATUS_SUCCESS) {
    Logger::error("cublasSgemm (FP32 matvec) failed with status: " + std::to_string(status) + " (" + cublasGetStatusString(status) + ")");
    Logger::error("MatVec GEMM params: M=" + std::to_string(M_blas) + " N=" + std::to_string(N_blas) + " K=" + std::to_string(K_blas) +
                  " LDA=" + std::to_string(K_blas) + " LDB=" + std::to_string(K_blas) + " LDC=" + std::to_string(M_blas) );
    throw std::runtime_error("cublasSgemm (FP32 matvec) failed");
  }
}
/**
 * @brief Implementation of matrix-vector multiplication with BF16
 * 
 * This function performs matrix-vector multiplication using cuBLAS,
 * with automatic conversion from BF16 to FP32. It's optimized for
 * models using Brain Floating Point format.
 * 
 * @param handle cuBLAS handle
 * @param mat_bf16_dev Matrix in BF16 format (device pointer)
 * @param vec_f32_dev Vector in FP32 format (device pointer)
 * @param out_f32_dev Output vector in FP32 format (device pointer)
 * @param rows Number of matrix rows
 * @param cols Number of matrix columns
 * @param stream CUDA stream for asynchronous execution
 */
void matvec_bf16_f32_cuda(cublasHandle_t handle, const uint16_t* mat_bf16_dev,
                          const float* vec_f32_dev, float* out_f32_dev,
                          int rows, int cols, cudaStream_t stream) {
  float* mat_fp32_dev = nullptr;
  size_t mat_size = (size_t)rows * cols;

  gpuErrchk(cudaMalloc(&mat_fp32_dev, mat_size * sizeof(float)));

  const int threads_per_block_convert = 256;
  const int num_blocks_convert = (mat_size + threads_per_block_convert - 1) / threads_per_block_convert;
  convert_bf16_to_fp32_kernel<<<num_blocks_convert, threads_per_block_convert, 0, stream>>>(mat_bf16_dev, mat_fp32_dev, mat_size);
  gpuErrchk(cudaGetLastError()); // Check for errors after kernel launch

  const float alpha = 1.0f;
  const float beta = 0.0f;
  cublasStatus_t status = cublasSetStream(handle, stream);
  if (status != CUBLAS_STATUS_SUCCESS) {
    Logger::error("cublasSetStream failed in matvec_bf16_f32_cuda fallback");
    gpuErrchk(cudaFree(mat_fp32_dev));
    throw std::runtime_error("cublasSetStream failed");
  }

  status = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, rows, 1, cols, &alpha,
                       mat_fp32_dev, cols, vec_f32_dev, cols, &beta,
                       out_f32_dev, rows);

  if (status != CUBLAS_STATUS_SUCCESS) {
    Logger::error("cublasSgemm (BF16 fallback) failed with status: " +
                  std::to_string(status));
    gpuErrchk(cudaFree(mat_fp32_dev));
    throw std::runtime_error("cublasSgemm (BF16 fallback) failed");
  }

  gpuErrchk(cudaFree(mat_fp32_dev));
}

void matvec_f32_f32_cuda(cublasHandle_t handle,
                         const std::vector<float>& mat_f32_host,
                         const std::vector<float>& vec_f32_host,
                         std::vector<float>& out_f32_host, int rows, int cols) {
  if (mat_f32_host.size() != (size_t)rows * cols) {
    throw std::runtime_error(
        "matvec_f32_f32_cuda (host/host/host): mat size mismatch.");
  }
  if (vec_f32_host.size() != (size_t)cols) {
    throw std::runtime_error(
        "matvec_f32_f32_cuda (host/host/host): vec size mismatch.");
  }
  out_f32_host.resize(rows);

  float* mat_f32_dev = nullptr;
  float* vec_f32_dev = nullptr;
  float* out_f32_dev = nullptr;

  gpuErrchk(cudaMalloc(&mat_f32_dev, mat_f32_host.size() * sizeof(float)));
  gpuErrchk(cudaMalloc(&vec_f32_dev, vec_f32_host.size() * sizeof(float)));
  gpuErrchk(cudaMalloc(&out_f32_dev, out_f32_host.size() * sizeof(float)));

  gpuErrchk(cudaMemcpy(mat_f32_dev, mat_f32_host.data(),
                       mat_f32_host.size() * sizeof(float),
                       cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(vec_f32_dev, vec_f32_host.data(),
                       vec_f32_host.size() * sizeof(float),
                       cudaMemcpyHostToDevice));

  matvec_f32_f32_cuda(handle, mat_f32_dev, vec_f32_dev, out_f32_dev, rows, cols,
                      0);

  gpuErrchk(cudaDeviceSynchronize());

  gpuErrchk(cudaMemcpy(out_f32_host.data(), out_f32_dev,
                       out_f32_host.size() * sizeof(float),
                       cudaMemcpyDeviceToHost));

  gpuErrchk(cudaFree(mat_f32_dev));
  gpuErrchk(cudaFree(vec_f32_dev));
  gpuErrchk(cudaFree(out_f32_dev));
}

__global__ void silu_kernel(const float* x, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float x_val = x[i];
    out[i] = x_val / (1.0f + expf(-x_val));
  }
}

void silu_cuda(const std::vector<float>& x_host, std::vector<float>& out_host,
               int n) {
  if (x_host.size() != n) {
    throw std::runtime_error("SiLU CUDA: Input vector size mismatch.");
  }
  out_host.resize(n);

  float* x_dev = nullptr;
  float* out_dev = nullptr;
  gpuErrchk(cudaMalloc(&x_dev, n * sizeof(float)));
  gpuErrchk(cudaMalloc(&out_dev, n * sizeof(float)));

  gpuErrchk(cudaMemcpy(x_dev, x_host.data(), n * sizeof(float),
                       cudaMemcpyHostToDevice));

  const int threads_per_block = 256;
  int num_blocks = (n + threads_per_block - 1) / threads_per_block;
  silu_kernel<<<num_blocks, threads_per_block>>>(x_dev, out_dev, n);
  gpuErrchk(cudaGetLastError());

  gpuErrchk(cudaMemcpy(out_host.data(), out_dev, n * sizeof(float),
                       cudaMemcpyDeviceToHost));
  gpuErrchk(cudaDeviceSynchronize());

  gpuErrchk(cudaFree(x_dev));
  gpuErrchk(cudaFree(out_dev));
}

__global__ void softmax_find_max_kernel(const float* x, float* partial_max,
                                        int n) {
  extern __shared__ float sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  sdata[tid] = (i < n) ? x[i] : -INFINITY;
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
    }
    __syncthreads();
  }

  if (tid == 0) {
    partial_max[blockIdx.x] = sdata[0];
  }
}

__global__ void softmax_exp_sum_kernel(const float* x, float* partial_sums,
                                       int n, float max_val) {
  extern __shared__ float sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  sdata[tid] = (i < n) ? expf(x[i] - max_val) : 0.0f;
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    partial_sums[blockIdx.x] = sdata[0];
  }
}

__global__ void softmax_normalize_kernel(const float* x, float* out, int n,
                                         float max_val, float inv_sum) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = expf(x[i] - max_val) * inv_sum;
  }
}

void softmax_vector_cuda(const std::vector<float>& x_host,
                         std::vector<float>& out_host, int n) {
  if (x_host.size() != n) {
    throw std::runtime_error("Softmax CUDA: Input vector size mismatch.");
  }
  if (n == 0) {
    out_host.clear();
    return;
  }
  out_host.resize(n);

  float* x_dev = nullptr;
  float* out_dev = nullptr;
  float* partial_max_dev = nullptr;
  float* partial_sum_dev = nullptr;

  const int threads_per_block = 256;
  int num_blocks = (n + threads_per_block - 1) / threads_per_block;
  size_t shared_mem_size = threads_per_block * sizeof(float);

  float* partial_max_host = new float[num_blocks];
  float* partial_sum_host = new float[num_blocks];

  gpuErrchk(cudaMalloc(&x_dev, n * sizeof(float)));
  gpuErrchk(cudaMalloc(&out_dev, n * sizeof(float)));
  gpuErrchk(cudaMalloc(&partial_max_dev, num_blocks * sizeof(float)));
  gpuErrchk(cudaMalloc(&partial_sum_dev, num_blocks * sizeof(float)));

  gpuErrchk(cudaMemcpy(x_dev, x_host.data(), n * sizeof(float),
                       cudaMemcpyHostToDevice));

  softmax_find_max_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(
      x_dev, partial_max_dev, n);
  gpuErrchk(cudaGetLastError());
  gpuErrchk(cudaMemcpy(partial_max_host, partial_max_dev,
                       num_blocks * sizeof(float), cudaMemcpyDeviceToHost));

  float max_val = -INFINITY;
  for (int i = 0; i < num_blocks; ++i) {
    if (partial_max_host[i] > max_val) {
      max_val = partial_max_host[i];
    }
  }

  softmax_exp_sum_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(
      x_dev, partial_sum_dev, n, max_val);
  gpuErrchk(cudaGetLastError());
  gpuErrchk(cudaMemcpy(partial_sum_host, partial_sum_dev,
                       num_blocks * sizeof(float), cudaMemcpyDeviceToHost));

  double exp_sum = 0.0;
  for (int i = 0; i < num_blocks; ++i) {
    exp_sum += partial_sum_host[i];
  }
  float inv_sum = 1.0f / static_cast<float>(exp_sum);

  softmax_normalize_kernel<<<num_blocks, threads_per_block>>>(x_dev, out_dev, n,
                                                              max_val, inv_sum);
  gpuErrchk(cudaGetLastError());

  gpuErrchk(cudaMemcpy(out_host.data(), out_dev, n * sizeof(float),
                       cudaMemcpyDeviceToHost));

  delete[] partial_max_host;
  delete[] partial_sum_host;
  gpuErrchk(cudaFree(x_dev));
  gpuErrchk(cudaFree(out_dev));
  gpuErrchk(cudaFree(partial_max_dev));
  gpuErrchk(cudaFree(partial_sum_dev));
}

__global__ void swiglu_kernel(const float* gate, const float* up, float* out,
                              int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float gate_val = gate[i];
    float silu_gate = gate_val / (1.0f + expf(-gate_val));
    out[i] = silu_gate * up[i];
  }
}

void swiglu_cuda(const float* gate_dev, const float* up_dev, float* out_dev,
                 int n, cudaStream_t stream) {
  const int threads_per_block = 256;
  int num_blocks = (n + threads_per_block - 1) / threads_per_block;
  swiglu_kernel<<<num_blocks, threads_per_block, 0, stream>>>(gate_dev, up_dev,
                                                              out_dev, n);
  gpuErrchk(cudaGetLastError());
}

// New Batched SwiGLU Kernel
__global__ void swiglu_batch_cuda_kernel(
    float* d_out_batch,              // Output: [num_tokens, intermediate_size]
    const float* d_gate_act_batch,   // Input: Gate activations [num_tokens, intermediate_size]
    const float* d_up_act_batch,     // Input: Up activations [num_tokens, intermediate_size]
    int num_tokens,
    int intermediate_size) {

    int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) {
        return;
    }

    // Base pointers for the current token
    const float* current_token_gate_ptr = d_gate_act_batch + (size_t)token_idx * intermediate_size;
    const float* current_token_up_ptr = d_up_act_batch + (size_t)token_idx * intermediate_size;
    float* current_token_out_ptr = d_out_batch + (size_t)token_idx * intermediate_size;

    // Threads in the block parallelize over intermediate_size
    for (int i = threadIdx.x; i < intermediate_size; i += blockDim.x) {
        float gate_val = current_token_gate_ptr[i];
        float silu_gate = gate_val / (1.0f + expf(-gate_val)); // SiLU(x) = x * sigmoid(x)
        float up_val = current_token_up_ptr[i];
        current_token_out_ptr[i] = silu_gate * up_val;
    }
}

// Host wrapper for Batched SwiGLU
void swiglu_batch_cuda(
    float* d_out_batch,              // Output: [num_tokens, intermediate_size]
    const float* d_gate_act_batch,   // Input: Gate activations [num_tokens, intermediate_size]
    const float* d_up_act_batch,     // Input: Up activations [num_tokens, intermediate_size]
    int num_tokens,
    int intermediate_size,           // This is typically config_.ffn_hidden_size or similar
    cudaStream_t stream) {


    if (num_tokens == 0 || intermediate_size == 0) {
        Logger::info("[SWIGLU_BATCH_CUDA_SKIP] num_tokens or intermediate_size is 0. Nothing to do.");
        Logger::info("[SWIGLU_BATCH_CUDA_EXIT] Skipped operation.");
        return; // Nothing to do
    }

    // Log initial input values (first token, first few elements)
    int log_count_elements = std::min(3, intermediate_size);
    if (num_tokens > 0 && intermediate_size > 0) {
        if (d_gate_act_batch) {
            std::vector<float> h_gate_sample(log_count_elements);
            // Gate is [num_tokens, intermediate_size]. Offset for T0 is 0.
            gpuErrchk(cudaMemcpyAsync(h_gate_sample.data(), d_gate_act_batch, log_count_elements * sizeof(float), cudaMemcpyDeviceToHost, stream));
        }
        if (d_up_act_batch) {
            std::vector<float> h_up_sample(log_count_elements);
            // Up is [num_tokens, intermediate_size]. Offset for T0 is 0.
            gpuErrchk(cudaMemcpyAsync(h_up_sample.data(), d_up_act_batch, log_count_elements * sizeof(float), cudaMemcpyDeviceToHost, stream));
        }
    }

    const int threads_per_block = 256;
    dim3 grid_dim(num_tokens);
    dim3 block_dim(threads_per_block);

    swiglu_batch_cuda_kernel<<<grid_dim, block_dim, 0, stream>>>(
        d_out_batch,
        d_gate_act_batch,
        d_up_act_batch,
        num_tokens,
        intermediate_size
    );
    gpuErrchk(cudaGetLastError());

    // Log output values
    if (num_tokens > 0 && intermediate_size > 0 && d_out_batch) {
        std::vector<float> h_out_sample(log_count_elements);
        // Out is [num_tokens, intermediate_size]. Offset for T0 is 0.
        gpuErrchk(cudaMemcpyAsync(h_out_sample.data(), d_out_batch, log_count_elements * sizeof(float), cudaMemcpyDeviceToHost, stream));
    }
}
__global__ void rope_kernel(float* x, int num_heads, int head_dim,
                            const float* all_freqs_cis_base, int pos, bool use_adjacent_pairing) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x; // Global thread index for pairs
  int total_pairs = num_heads * (head_dim / 2);   // Total (dim_i, dim_{i+1}) or (dim_i, dim_{i+D/2}) pairs across all heads
  if (idx >= total_pairs) return;

  int head_idx = idx / (head_dim / 2);        // The specific head this thread works on
  int dim_pair_idx = idx % (head_dim / 2);  // The index of the pair within this head (0 to head_dim/2 - 1)

  size_t freq_cos_sin_pair_idx = (size_t)pos * (head_dim / 2) + dim_pair_idx;
  size_t freq_base_offset = freq_cos_sin_pair_idx * 2; 

  int base_x0, base_x1;
  if (use_adjacent_pairing) {
    // Adjacent pairing: Rotate x[h*HD + 2*j] with x[h*HD + 2*j + 1]
    base_x0 = head_idx * head_dim + (dim_pair_idx * 2);
    base_x1 = head_idx * head_dim + (dim_pair_idx * 2 + 1);
  } else {
    // Split-half pairing: Rotate x[h*HD + j] with x[h*HD + j + head_dim/2]
    base_x0 = head_idx * head_dim + dim_pair_idx;
    base_x1 = head_idx * head_dim + dim_pair_idx + (head_dim / 2);
  }

  float x0 = x[base_x0];
  float x1 = x[base_x1];
  
  float cos_val = all_freqs_cis_base[freq_base_offset];
  float sin_val = all_freqs_cis_base[freq_base_offset + 1];

  x[base_x0] = x0 * cos_val - x1 * sin_val;
  x[base_x1] = x0 * sin_val + x1 * cos_val;
}

void rope_cuda(float* x_dev, int num_heads, int head_dim,
               const float* all_freqs_cis_dev_base, int pos, bool use_adjacent_pairing, cudaStream_t stream) {
  int total_pairs = num_heads * (head_dim / 2);
  int threads_per_block = 256;
  int num_blocks = (total_pairs + threads_per_block - 1) / threads_per_block;

  rope_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
      x_dev, num_heads, head_dim, all_freqs_cis_dev_base, pos, use_adjacent_pairing);
  gpuErrchk(cudaGetLastError());
}

// New Batched RoPE Kernel
__global__ void rope_batch_kernel_cuda(
    float* x_batch_tensor,          // Q_batch or K_batch tensor
    int num_tokens,                 // Number of tokens in this batch
    int num_heads_in_tensor,        // num_q_heads or num_kv_heads for this tensor
    int head_dim,                   // Dimension of each head
    const float* d_all_freqs_cis_base, // Precomputed cos/sin values table
    int start_pos_offset,           // Starting sequence position for the tokens in this batch
    bool use_adjacent_pairing) {

    // Grid: dim3 grid_dim(num_tokens, num_heads_in_tensor);
    // Block: dim3 block_dim(head_dim / 2);
    int pair_within_head_idx = threadIdx.x; // Thread index within the block (0 to head_dim/2 - 1)
    int head_idx = blockIdx.y;              // Block index in Y dimension maps to head index
    int token_idx = blockIdx.x;             // Block index in X dimension maps to token index within the batch

    // Boundary checks
    if (pair_within_head_idx >= (head_dim / 2) || 
        head_idx >= num_heads_in_tensor ||
        token_idx >= num_tokens) {
        return;
    }

    // Determine the actual sequence position for RoPE for the current token
    int current_token_seq_pos = start_pos_offset + token_idx;

    // Fetch cos and sin values from the precomputed table
    // d_all_freqs_cis_base is effectively [max_seq_len][head_dim/2][2]
    // Access: (pos * (head_dim/2) + pair_idx_in_head) * 2 for cos, and +1 for sin
    size_t freq_pair_offset_in_table = ((size_t)current_token_seq_pos * (head_dim / 2) + pair_within_head_idx) * 2;
    float cos_val = d_all_freqs_cis_base[freq_pair_offset_in_table];
    float sin_val = d_all_freqs_cis_base[freq_pair_offset_in_table + 1];

    // Calculate base pointer to the current token's data in x_batch_tensor
    // x_batch_tensor is [num_tokens][num_heads_in_tensor][head_dim]
    float* token_data_start = x_batch_tensor + (size_t)token_idx * num_heads_in_tensor * head_dim;
    // Pointer to the start of the current head's data for this token
    float* head_data_start = token_data_start + (size_t)head_idx * head_dim;

    int x0_idx_in_head, x1_idx_in_head;
    if (use_adjacent_pairing) {
        // Adjacent pairing: (x_0, x_1), (x_2, x_3), ... for a head
        // pair_within_head_idx = 0 maps to (x_0, x_1), pair_within_head_idx = 1 maps to (x_2, x_3)
        x0_idx_in_head = pair_within_head_idx * 2;
        x1_idx_in_head = pair_within_head_idx * 2 + 1;
    } else {
        // Split-half pairing: (x_0, x_{D/2}), (x_1, x_{D/2+1}), ... for a head
        // pair_within_head_idx = 0 maps to (x_0, x_{D/2}), pair_within_head_idx = 1 maps to (x_1, x_{D/2+1})
        x0_idx_in_head = pair_within_head_idx;
        x1_idx_in_head = pair_within_head_idx + (head_dim / 2);
    }

    float x0_val = head_data_start[x0_idx_in_head];
    float x1_val = head_data_start[x1_idx_in_head];


    head_data_start[x0_idx_in_head] = x0_val * cos_val - x1_val * sin_val;
    head_data_start[x1_idx_in_head] = x0_val * sin_val + x1_val * cos_val;
}

// Host wrapper for batched RoPE application
void rope_batch_cuda(float* d_q_batch, float* d_k_batch,
                     const float* d_all_freqs_cis_base, // Moved to 3rd position
                     int num_tokens, int num_q_heads, int num_kv_heads, int head_dim,
                     int start_pos_offset, 
                     bool use_adjacent_pairing,
                     cudaStream_t stream) {


    if (head_dim == 0) { 
        Logger::warning("[ROPE_BATCH_CUDA_SKIP] head_dim is 0. Skipping RoPE application.");
        return;
    }
    if (head_dim % 2 != 0) {
        Logger::error("rope_batch_cuda: head_dim must be even. Got " + std::to_string(head_dim));
        Logger::info("[ROPE_BATCH_CUDA_EXIT] Error: head_dim not even.");
        return; 
    }
    if (num_tokens == 0) {
        Logger::info("[ROPE_BATCH_CUDA_EXIT] num_tokens is 0. Nothing to process.");
        return; // Nothing to process
    }

    int log_count_elements = std::min(3, head_dim);

    // Log initial Q values
    if (d_q_batch && num_q_heads > 0 && head_dim > 0) {
        for (int token_to_log_idx = 0; token_to_log_idx < std::min(num_tokens, 2); ++token_to_log_idx) {
            std::vector<float> h_q_sample_before(log_count_elements);
            size_t q_offset_elements = (size_t)token_to_log_idx * num_q_heads * head_dim;
            
            gpuErrchk(cudaMemcpyAsync(h_q_sample_before.data(), 
                                      d_q_batch + q_offset_elements, 
                                      log_count_elements * sizeof(float), 
                                      cudaMemcpyDeviceToHost, stream));
        }
    }

    // Log initial K values
    if (d_k_batch && num_kv_heads > 0 && head_dim > 0) {
        for (int token_to_log_idx = 0; token_to_log_idx < std::min(num_tokens, 2); ++token_to_log_idx) {
            std::vector<float> h_k_sample_before(log_count_elements);
            size_t k_offset_elements = (size_t)token_to_log_idx * num_kv_heads * head_dim;

            gpuErrchk(cudaMemcpyAsync(h_k_sample_before.data(), 
                                      d_k_batch + k_offset_elements, 
                                      log_count_elements * sizeof(float), 
                                      cudaMemcpyDeviceToHost, stream));
        }
    }

    dim3 block_dim(head_dim / 2);

    // Apply RoPE to Q batch
    if (d_q_batch && num_q_heads > 0) {
        dim3 grid_dim_q(num_tokens, num_q_heads);
        rope_batch_kernel_cuda<<<grid_dim_q, block_dim, 0, stream>>>(
            d_q_batch,
            num_tokens,
            num_q_heads,
            head_dim,
            d_all_freqs_cis_base,
            start_pos_offset,
            use_adjacent_pairing
        );
        gpuErrchk(cudaGetLastError()); 
    }

    // Apply RoPE to K batch
    if (d_k_batch && num_kv_heads > 0) {
        dim3 grid_dim_k(num_tokens, num_kv_heads); 
        rope_batch_kernel_cuda<<<grid_dim_k, block_dim, 0, stream>>>(
            d_k_batch,
            num_tokens,
            num_kv_heads, 
            head_dim,
            d_all_freqs_cis_base,
            start_pos_offset,
            use_adjacent_pairing
        );
        gpuErrchk(cudaGetLastError()); 
    }

    // Log Q values after RoPE
    if (d_q_batch && num_q_heads > 0 && head_dim > 0) {
        for (int token_to_log_idx = 0; token_to_log_idx < std::min(num_tokens, 2); ++token_to_log_idx) {
            std::vector<float> h_q_sample_after(log_count_elements);
            size_t q_offset_elements = (size_t)token_to_log_idx * num_q_heads * head_dim;
            
            gpuErrchk(cudaMemcpyAsync(h_q_sample_after.data(), 
                                      d_q_batch + q_offset_elements, 
                                      log_count_elements * sizeof(float), 
                                      cudaMemcpyDeviceToHost, stream));
        }
    }

    // Log K values after RoPE
    if (d_k_batch && num_kv_heads > 0 && head_dim > 0) {
        for (int token_to_log_idx = 0; token_to_log_idx < std::min(num_tokens, 2); ++token_to_log_idx) {
            std::vector<float> h_k_sample_after(log_count_elements);
            size_t k_offset_elements = (size_t)token_to_log_idx * num_kv_heads * head_dim;

            gpuErrchk(cudaMemcpyAsync(h_k_sample_after.data(), 
                                      d_k_batch + k_offset_elements, 
                                      log_count_elements * sizeof(float), 
                                      cudaMemcpyDeviceToHost, stream));
        }
    }
}

__global__ void attention_kernel(const float* Q_current,
                                 const float* K_layer_cache_base,
                                 const float* V_layer_cache_base, float* out,
                                 int current_seq_len, int head_dim, float scale,
                                 int cache_num_kv_heads, int num_q_heads) {


  int q_head_idx = blockIdx.x;
  int d_idx = threadIdx.x; // dimension index for this thread

  if (q_head_idx >= num_q_heads) return; // Should not happen with correct launch

  // Determine the corresponding KV head for this Q head (for GQA/MQA)
  int heads_per_kv = num_q_heads / cache_num_kv_heads;
  int kv_head_idx = q_head_idx / heads_per_kv;

  // Shared memory for scores (one score per k_pos) and for dot product reduction
  extern __shared__ float shared_data[];
  float* scores = shared_data; // First part of shared_data: size current_seq_len
  float* dot_product_terms = &shared_data[current_seq_len]; // Second part: size head_dim (blockDim.x)

  // Pointer to the current query head's data
  const float* q_head_ptr = Q_current + q_head_idx * head_dim;

  for (int k_pos = 0; k_pos < current_seq_len; ++k_pos) {
    // Pointer to the k_pos-th key vector for the relevant kv_head
    size_t k_vec_offset = (size_t)k_pos * cache_num_kv_heads * head_dim +
                          (size_t)kv_head_idx * head_dim;
    const float* k_vec_ptr = K_layer_cache_base + k_vec_offset;

    dot_product_terms[d_idx] = q_head_ptr[d_idx] * k_vec_ptr[d_idx];
    __syncthreads(); // Ensure all terms are written before reduction

    // Reduction in shared memory (e.g., by thread 0 or tree-based)
    if (d_idx == 0) {
      float current_dot = 0.0f;
      for (int r = 0; r < head_dim; ++r) {
        current_dot += dot_product_terms[r];
      }
      scores[k_pos] = current_dot * scale;
    }
    __syncthreads(); 
  }

  float thread_max_score = -INFINITY;
  for (int i = d_idx; i < current_seq_len; i += blockDim.x) { // blockDim.x is head_dim
    if (scores[i] > thread_max_score) {
      thread_max_score = scores[i];
    }
  }
  dot_product_terms[d_idx] = thread_max_score; // Repurpose dot_product_terms shared memory
  __syncthreads();

  float block_max_score = -INFINITY;
  if (d_idx == 0) {
    for (int r = 0; r < head_dim; ++r) { // head_dim might be > current_seq_len for first few tokens
      if (dot_product_terms[r] > block_max_score) { // Check if dot_product_terms[r] was validly written
        block_max_score = dot_product_terms[r];
      }
    }
  }
  __syncthreads();
  if(d_idx == 0) dot_product_terms[0] = block_max_score; // Store it.
  __syncthreads();
  block_max_score = dot_product_terms[0]; // All threads read it.


  // Calculate exp scores and sum
  float thread_exp_sum = 0.0f;
  for (int i = d_idx; i < current_seq_len; i += blockDim.x) {
    float prob = expf(scores[i] - block_max_score);
    scores[i] = prob; // Update scores in-place with exp(val - max)
    thread_exp_sum += prob;
  }
  // Reduce thread_exp_sum across the block
  dot_product_terms[d_idx] = thread_exp_sum; // Repurpose again
  __syncthreads();

  float block_exp_sum = 0.0f;
  if (d_idx == 0) {
    for (int r = 0; r < head_dim; ++r) { // Similar concern as max_score reduction range
      block_exp_sum += dot_product_terms[r];
    }
  }
  __syncthreads();
  // Store and reload block_exp_sum for all threads
  if(d_idx == 0) dot_product_terms[0] = block_exp_sum;
  __syncthreads();
  block_exp_sum = dot_product_terms[0];


  float inv_sum = 1.0f / (block_exp_sum + 1e-9f); // Add epsilon for stability

  // Normalize scores in shared memory
  for (int i = d_idx; i < current_seq_len; i += blockDim.x) {
    scores[i] *= inv_sum;
  }
  __syncthreads(); // Ensure all scores are normalized before weighted sum

  double weighted_val_d = 0.0; // Use double for accumulation
  for (int k_pos = 0; k_pos < current_seq_len; ++k_pos) {
    size_t v_vec_offset = (size_t)k_pos * cache_num_kv_heads * head_dim +
                          (size_t)kv_head_idx * head_dim;
    const float* v_vec_ptr = V_layer_cache_base + v_vec_offset;
    
    weighted_val_d += static_cast<double>(scores[k_pos]) * static_cast<double>(v_vec_ptr[d_idx]);
  }
  
  // Write the final result for this thread's dimension d_idx
  out[q_head_idx * head_dim + d_idx] = static_cast<float>(weighted_val_d);
}

void attention_cuda(const float* Q_current_dev, const float* K_layer_cache_base,
                    const float* V_layer_cache_base, float* out_dev,
                    int num_heads, int current_seq_len, int head_dim,
                    float scale, int cache_max_seq_len, int cache_num_kv_heads,
                    cudaStream_t stream) {
  // Grid: num_heads blocks, each block handles one attention head
  // Block: head_dim threads, each thread handles one output dimension
  dim3 grid_dim(num_heads);
  dim3 block_dim(head_dim);

  attention_kernel<<<grid_dim, block_dim, 0, stream>>>(
      Q_current_dev, K_layer_cache_base, V_layer_cache_base, out_dev,
      current_seq_len, head_dim, scale, cache_num_kv_heads, num_heads);
  gpuErrchk(cudaGetLastError());
}

// Selective dequantization attention implementation
__global__ void attention_selective_dequant_kernel(
    const float* __restrict__ Q_current_dev,
    const int8_t* __restrict__ K_quantized_cache_base,
    const int8_t* __restrict__ V_quantized_cache_base,
    const float* __restrict__ K_scales_cache_base,
    const float* __restrict__ V_scales_cache_base,
    float* __restrict__ selective_k_dequant_buffer,
    float* __restrict__ selective_v_dequant_buffer,
    float* __restrict__ out_dev,
    int num_heads, int current_seq_len, int head_dim,
    float scale, int cache_max_seq_len, int cache_num_kv_heads) {
    
    int head_idx = blockIdx.x;
    int dim_idx = threadIdx.x;
    
    if (head_idx >= num_heads || dim_idx >= head_dim) return;
    
    // Map query head to corresponding KV head (for GQA/MQA)
    int kv_heads_per_q_group = num_heads / cache_num_kv_heads;
    int kv_head_idx = head_idx / kv_heads_per_q_group;
    
    // Get current query vector for this head and dimension
    float q_val = Q_current_dev[head_idx * head_dim + dim_idx];
    
    float attn_out = 0.0f;
    
    // Selective dequantization and attention computation
    for (int t = 0; t < current_seq_len; ++t) {
        // Calculate offset for this token and KV head in the quantized cache
        size_t cache_offset_base = (size_t)t * cache_num_kv_heads * head_dim + (size_t)kv_head_idx * head_dim;
        size_t scale_offset = (size_t)t * cache_num_kv_heads + kv_head_idx;
        
        // Dequantize only the specific elements we need for this head/dimension
        if (dim_idx == 0) {
            // Only thread 0 in each block dequantizes the entire head for this token
            float k_scale = K_scales_cache_base[scale_offset];
            float v_scale = V_scales_cache_base[scale_offset];
            
            for (int d = 0; d < head_dim; ++d) {
                // Dequantize K values for this token/head
                int8_t k_quant = K_quantized_cache_base[cache_offset_base + d];
                selective_k_dequant_buffer[t * head_dim + d] = k_scale * (float)k_quant;
                
                // Dequantize V values for this token/head  
                int8_t v_quant = V_quantized_cache_base[cache_offset_base + d];
                selective_v_dequant_buffer[t * head_dim + d] = v_scale * (float)v_quant;
            }
        }
        
        __syncthreads(); // Ensure dequantization is complete before using values
        
        // Compute attention score for this token
        float k_val = selective_k_dequant_buffer[t * head_dim + dim_idx];
        float score = q_val * k_val * scale;
        
        // Apply attention score to value
        float v_val = selective_v_dequant_buffer[t * head_dim + dim_idx];
        attn_out += score * v_val;
        
        __syncthreads(); // Ensure all threads are done before next iteration
    }
    
    // Write output
    out_dev[head_idx * head_dim + dim_idx] = attn_out;
}

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
                                     cudaStream_t stream) {
    
    if (num_heads == 0 || current_seq_len == 0 || head_dim == 0) {
        Logger::warning("[ATTENTION_SELECTIVE_DEQUANT] Invalid parameters: num_heads=" + 
                       std::to_string(num_heads) + ", current_seq_len=" + std::to_string(current_seq_len) + 
                       ", head_dim=" + std::to_string(head_dim));
        return;
    }
    
    // Grid: num_heads blocks, each block handles one attention head
    // Block: head_dim threads, each thread handles one output dimension  
    dim3 grid_dim(num_heads);
    dim3 block_dim(head_dim);
    
    attention_selective_dequant_kernel<<<grid_dim, block_dim, 0, stream>>>(
        Q_current_dev, K_quantized_cache_base, V_quantized_cache_base,
        K_scales_cache_base, V_scales_cache_base,
        selective_k_dequant_buffer, selective_v_dequant_buffer,
        out_dev, num_heads, current_seq_len, head_dim, scale,
        cache_max_seq_len, cache_num_kv_heads);
    
    gpuErrchk(cudaGetLastError());
}

__global__ void add_vectors_kernel(const float* a, const float* b,
                                   float* result, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    result[i] = a[i] + b[i];
  }
}

void add_vectors_cuda(const float* a_dev, const float* b_dev, float* result_dev,
                      int n, cudaStream_t stream) {
  const int threads_per_block = 256;
  const int num_blocks = (n + threads_per_block - 1) / threads_per_block;

  add_vectors_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
      a_dev, b_dev, result_dev, n);
  gpuErrchk(cudaGetLastError());
}

__global__ void add_residual_kernel(const float* matvec_out,
                                    const float* residual, float* result,
                                    int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    result[i] = matvec_out[i] + residual[i];
  }
}

void add_residual_cuda(const float* matvec_out_dev, const float* residual_dev,
                       float* result_dev, int n, cudaStream_t stream) {
  const int threads_per_block = 256;
  const int num_blocks = (n + threads_per_block - 1) / threads_per_block;

  add_residual_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
      matvec_out_dev, residual_dev, result_dev, n);
  gpuErrchk(cudaGetLastError());
}

// New Batched Add Residual Kernel
__global__ void add_residual_batch_cuda_kernel(
    float* d_output_batch,          // Output: [num_tokens, hidden_size]
    const float* d_input_a_batch,   // Input A: [num_tokens, hidden_size] (e.g., sub-layer output)
    const float* d_input_b_batch,   // Input B: [num_tokens, hidden_size] (e.g., original input to sub-layer)
    int num_tokens,
    int hidden_size) {

    int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) {
        return;
    }

    // Base pointers for the current token
    const float* current_token_input_a_ptr = d_input_a_batch + (size_t)token_idx * hidden_size;
    const float* current_token_input_b_ptr = d_input_b_batch + (size_t)token_idx * hidden_size;
    float* current_token_output_ptr = d_output_batch + (size_t)token_idx * hidden_size;

    // Threads in the block parallelize over hidden_size
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        current_token_output_ptr[i] = current_token_input_a_ptr[i] + current_token_input_b_ptr[i];
    }
}

// Host wrapper for Batched Add Residual
void add_residual_batch_cuda(
    float* d_output_batch,          // Output: [num_tokens, hidden_size]
    const float* d_input_a_batch,   // Input A: [num_tokens, hidden_size]
    const float* d_input_b_batch,   // Input B: [num_tokens, hidden_size]
    int num_tokens,
    int hidden_size,
    cudaStream_t stream) {


    if (num_tokens == 0 || hidden_size == 0) {
        Logger::info("[ADD_RESIDUAL_BATCH_CUDA_SKIP] num_tokens or hidden_size is 0. Nothing to do.");
        Logger::info("[ADD_RESIDUAL_BATCH_CUDA_EXIT] Skipped operation.");
        return; // Nothing to do
    }

    // Log initial input values (first token, first few elements)
    int log_count_elements = std::min(3, hidden_size);
    if (num_tokens > 0 && hidden_size > 0) {
        if (d_input_a_batch) {
            std::vector<float> h_input_a_sample(log_count_elements);
            // Input A is [num_tokens, hidden_size]. Offset for T0 is 0.
            gpuErrchk(cudaMemcpyAsync(h_input_a_sample.data(), d_input_a_batch, log_count_elements * sizeof(float), cudaMemcpyDeviceToHost, stream));
        }
        if (d_input_b_batch) {
            std::vector<float> h_input_b_sample(log_count_elements);
            // Input B is [num_tokens, hidden_size]. Offset for T0 is 0.
            gpuErrchk(cudaMemcpyAsync(h_input_b_sample.data(), d_input_b_batch, log_count_elements * sizeof(float), cudaMemcpyDeviceToHost, stream));
        }
    }

    const int threads_per_block = 256; 
    dim3 grid_dim(num_tokens);
    dim3 block_dim(threads_per_block);

    add_residual_batch_cuda_kernel<<<grid_dim, block_dim, 0, stream>>>(
        d_output_batch,
        d_input_a_batch,
        d_input_b_batch,
        num_tokens,
        hidden_size
    );
    gpuErrchk(cudaGetLastError());

    // Log output values
    if (num_tokens > 0 && hidden_size > 0 && d_output_batch) {
        std::vector<float> h_output_sample(log_count_elements);
        // Output is [num_tokens, hidden_size]. Offset for T0 is 0.
        gpuErrchk(cudaMemcpyAsync(h_output_sample.data(), d_output_batch, log_count_elements * sizeof(float), cudaMemcpyDeviceToHost, stream));
    }
}
__global__ void update_kv_cache_kernel(float* cache_base_ptr,
                                       const float* current_kv_vector, int pos,
                                       int kv_head_idx, int max_seq_len,
                                       int num_kv_heads, int head_dim) {
  int d = threadIdx.x;
  if (d >= head_dim) return;

  size_t cache_offset = (size_t)pos * num_kv_heads * head_dim +
                        (size_t)kv_head_idx * head_dim + d;

  size_t source_offset = d;

  size_t total_cache_size = (size_t)max_seq_len * num_kv_heads * head_dim;
  if (cache_offset >= total_cache_size) {
    return;
  }

  cache_base_ptr[cache_offset] = current_kv_vector[source_offset];
}

void update_kv_cache_cuda(float* cache_base_ptr,
                          const float* current_kv_head_vector,

                          int pos, int kv_head_idx, int max_seq_len,
                          int num_kv_heads, int head_dim, cudaStream_t stream) {
  dim3 blockDim(head_dim);
  dim3 gridDim(1);

  if (pos < 0 || pos >= max_seq_len) {
    Logger::error("update_kv_cache_cuda: pos out of bounds (" +
                  std::to_string(pos) + " >= " + std::to_string(max_seq_len) +
                  ")");
    return;
  }

  if (kv_head_idx < 0 || kv_head_idx >= num_kv_heads) {
    Logger::error("update_kv_cache_cuda: kv_head_idx out of bounds (" +
                  std::to_string(kv_head_idx) +
                  " >= " + std::to_string(num_kv_heads) + ")");
    return;
  }
  if (!current_kv_head_vector) {
    Logger::error("update_kv_cache_cuda: Input K/V vector pointer is null.");
    return;
  }

  update_kv_cache_kernel<<<gridDim, blockDim, 0, stream>>>(
      cache_base_ptr, current_kv_head_vector, pos, kv_head_idx, max_seq_len,
      num_kv_heads, head_dim);
  gpuErrchk(cudaGetLastError());
}

__global__ void rope_and_update_kv_cache_kernel(
    float* cache_base_ptr, const float* kv_vector_head,

    const float* all_freqs_cis_base, int pos, int kv_head_idx, int max_seq_len,
    int num_kv_heads, int head_dim) {
  int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int dim_half = head_dim / 2;

  if (pair_idx >= dim_half) return;

  int idx0 = pair_idx;
  int idx1 = pair_idx + dim_half;
  float kv0 = kv_vector_head[idx0];
  float kv1 = kv_vector_head[idx1];

  size_t freq_base_offset = (size_t)pos * head_dim + (size_t)pair_idx * 2;
  float cos_val = all_freqs_cis_base[freq_base_offset];
  float sin_val = all_freqs_cis_base[freq_base_offset + 1];

  float kv0_rotated = kv0 * cos_val - kv1 * sin_val;
  float kv1_rotated = kv0 * sin_val + kv1 * cos_val;

  size_t cache_offset_0 = (size_t)pos * num_kv_heads * head_dim +
                          (size_t)kv_head_idx * head_dim + idx0;
  size_t cache_offset_1 = cache_offset_0 + dim_half;

  size_t total_cache_size = (size_t)max_seq_len * num_kv_heads * head_dim;
  if (cache_offset_0 >= total_cache_size ||
      cache_offset_1 >= total_cache_size) {
    return;
  }

  cache_base_ptr[cache_offset_0] = kv0_rotated;
  cache_base_ptr[cache_offset_1] = kv1_rotated;
}

void rope_and_update_kv_cache_cuda(float* cache_base_ptr,
                                   const float* kv_vector_head,
                                   const float* all_freqs_cis_base, int pos,
                                   int kv_head_idx, int max_seq_len,
                                   int num_kv_heads, int head_dim,
                                   cudaStream_t stream) {
  if (head_dim % 2 != 0) {
    Logger::error("rope_and_update_kv_cache_cuda: head_dim must be even.");
    return;
  }
  if (pos < 0 || pos >= max_seq_len) {
    Logger::error("rope_and_update_kv_cache_cuda: pos out of bounds.");
    return;
  }
  if (kv_head_idx < 0 || kv_head_idx >= num_kv_heads) {
    Logger::error("rope_and_update_kv_cache_cuda: kv_head_idx out of bounds.");
    return;
  }
  if (!kv_vector_head || !all_freqs_cis_base || !cache_base_ptr) {
    Logger::error(
        "rope_and_update_kv_cache_cuda: Received null device pointer(s).");
    return;
  }

  int threads_per_block = 128;
  int num_blocks = (head_dim / 2 + threads_per_block - 1) / threads_per_block;

  rope_and_update_kv_cache_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
      cache_base_ptr, kv_vector_head, all_freqs_cis_base, pos, kv_head_idx,
      max_seq_len, num_kv_heads, head_dim);
  gpuErrchk(cudaGetLastError());
}

__global__ void lookup_embedding_kernel(const void* __restrict__ table_dev,
                                        float* __restrict__ output_dev,
                                        int token_id, int hidden_size,
                                        int vocab_size, bool is_bf16) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= hidden_size) {
    return;
  }

  if (token_id < 0 || token_id >= vocab_size) {
    output_dev[idx] = 0.0f;
    return;
  }
  size_t offset = (size_t)token_id * hidden_size + idx;

  if (is_bf16) {
    const uint16_t* table_bf16 = static_cast<const uint16_t*>(table_dev);

    uint16_t val_bf16 = table_bf16[offset];

    output_dev[idx] = bf16_to_float32_device(val_bf16);
  } else {
    const float* table_f32 = static_cast<const float*>(table_dev);

    output_dev[idx] = table_f32[offset];
  }
}

void lookup_embedding_cuda(const void* table_dev, float* output_dev,
                           int token_id, int hidden_size, int vocab_size,
                           bool is_bf16, cudaStream_t stream) {
  if (!table_dev || !output_dev) {
    Logger::error("lookup_embedding_cuda: Received null device pointer(s).");

    return;
  }
  if (hidden_size <= 0 || vocab_size <= 0) {
    Logger::error("lookup_embedding_cuda: Invalid hidden_size or vocab_size.");
    return;
  }

  int threads_per_block = 256;

  int blocks_per_grid =
      (hidden_size + threads_per_block - 1) / threads_per_block;

  lookup_embedding_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(
      table_dev, output_dev, token_id, hidden_size, vocab_size, is_bf16);

  gpuErrchk(cudaGetLastError());
}

__global__ void convert_bf16_to_fp32_kernel(const uint16_t* __restrict__ bf16_in,
                                            float* __restrict__ fp32_out,
                                            size_t n_elements) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n_elements) {
    fp32_out[idx] = bf16_to_float32_device(bf16_in[idx]);
  }
}

// KVCache Quantization Kernels (FP32 <-> INT8)
#ifdef HAS_CUDA

// Kernel to find the maximum absolute value in a float array (for per-tensor scaling)
__global__ void find_max_abs_kernel(const float* __restrict__ data, int n_elements, float* __restrict__ max_abs_val) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    float local_max = 0.0f;
    if (i < n_elements) {
        local_max = fabsf(data[i]);
    }
    sdata[tid] = local_max;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        float old_val = *max_abs_val;
        float new_val = sdata[0];
        while (new_val > old_val) {
            float assumed_old = old_val;
            old_val = atomicCAS((unsigned int*)max_abs_val, 
                                __float_as_int(assumed_old), 
                                __float_as_int(new_val));
            // If atomicCAS did not return what we thought was the old value,
            // it means another thread updated it. We re-read and retry if our new_val is still greater.
            if (__int_as_float(old_val) != assumed_old) {
                 // Re-check condition with the actual current max_abs_val (now in old_val after CAS)
                if (new_val <= old_val) break; // Another thread set a higher or equal max, or our new_val is no longer the max
            } else {
                // Our CAS was successful, or new_val was not greater than the initial read.
                break; 
            }
        }
    }
}

// Kernel to quantize FP32 to INT8 using a symmetric per-tensor scale
__global__ void quantize_fp32_to_int8_symmetric_kernel(
    const float* __restrict__ fp32_in,
    int8_t* __restrict__ int8_out,
    float scale,
    int n_elements
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_elements) {
        float val_fp32 = fp32_in[i];
        // Symmetric quantization: scaled_value = round(value / scale)
        // Clamp to [-127, 127] for INT8 (as -128 might not be used or handled differently by some dequant schemes)
        float scaled_val = roundf(val_fp32 / scale);
        int8_out[i] = (int8_t)fmaxf(-127.0f, fminf(127.0f, scaled_val));
    }
}

// Kernel to dequantize INT8 to FP32 using a symmetric per-tensor scale
__global__ void dequantize_int8_to_fp32_symmetric_kernel(
    const int8_t* __restrict__ int8_in,
    float scale,
    float* __restrict__ fp32_out,
    int n_elements
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_elements) {
        fp32_out[i] = (float)int8_in[i] * scale;
    }
}

void quantize_fp32_to_int8_symmetric_per_tensor_cuda(
    const float* fp32_in_dev, 
    int8_t* int8_out_dev, 
    float* scale_out_dev, // Device pointer to store the single scale value
    int num_elements, 
    cudaStream_t stream
) {
    if (num_elements == 0) return;

    const int threads_per_block = 256;
    
    // --- Step 1: Find the maximum absolute value for scaling ---
    float* d_max_abs_val_accumulator = nullptr; // Accumulator on device for max_abs from blocks
    gpuErrchk(cudaMalloc(&d_max_abs_val_accumulator, sizeof(float)));
    gpuErrchk(cudaMemsetAsync(d_max_abs_val_accumulator, 0, sizeof(float), stream)); // Initialize to 0

    // Max reduction kernel requires shared memory proportional to block size
    int num_blocks_reduce = (num_elements + threads_per_block - 1) / threads_per_block;
    size_t shared_mem_size = threads_per_block * sizeof(float);
    
    find_max_abs_kernel<<<num_blocks_reduce, threads_per_block, shared_mem_size, stream>>>(
        fp32_in_dev, num_elements, d_max_abs_val_accumulator);
    gpuErrchk(cudaGetLastError());

    // Copy the final max absolute value back to `scale_out_dev` which is the designated output for the scale.
    // The actual scale factor will be calculated from this max_abs_val.
    float h_max_abs_val = 0.0f;
    gpuErrchk(cudaMemcpyAsync(&h_max_abs_val, d_max_abs_val_accumulator, sizeof(float), cudaMemcpyDeviceToHost, stream));
    gpuErrchk(cudaStreamSynchronize(stream)); // Ensure h_max_abs_val is ready

    gpuErrchk(cudaFree(d_max_abs_val_accumulator));

    // Calculate scale: scale = max_abs / 127.0 (so that 127 maps to max_abs)
    // Add a small epsilon to prevent division by zero if all values are zero.
    float scale = (h_max_abs_val > 1e-8) ? (h_max_abs_val / 127.0f) : 1.0f; 
    gpuErrchk(cudaMemcpyAsync(scale_out_dev, &scale, sizeof(float), cudaMemcpyHostToDevice, stream));

    // --- Step 2: Quantize using the calculated scale ---
    int num_blocks_quantize = (num_elements + threads_per_block - 1) / threads_per_block;
    quantize_fp32_to_int8_symmetric_kernel<<<num_blocks_quantize, threads_per_block, 0, stream>>>(
        fp32_in_dev, int8_out_dev, scale, num_elements);
    gpuErrchk(cudaGetLastError());
}

void dequantize_int8_to_fp32_symmetric_per_tensor_cuda(
    const int8_t* int8_in_dev, 
    const float* scale_in_dev, // Device pointer to the single scale value
    float* fp32_out_dev,
    int num_elements, 
    cudaStream_t stream
) {
    if (num_elements == 0) return;

    // Copy scale from device to host to pass as kernel argument
    // (Kernels typically take scalar arguments by value)
    float h_scale;
    gpuErrchk(cudaMemcpyAsync(&h_scale, scale_in_dev, sizeof(float), cudaMemcpyDeviceToHost, stream));
    // We need to synchronize here to ensure h_scale is available before launching the kernel
    // A more optimized approach might involve a device-side broadcast of the scale if it were part of a larger struct.
    gpuErrchk(cudaStreamSynchronize(stream)); 

    const int threads_per_block = 256;
    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;
    dequantize_int8_to_fp32_symmetric_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        int8_in_dev, h_scale, fp32_out_dev, num_elements);
    gpuErrchk(cudaGetLastError());
}

// New Generic GEMM for FP32
void gemm_f32_f32_cuda(cublasHandle_t handle, 
                       bool transa_user, bool transb_user, 
                       int m_user, int n_user, int k_user, 
                       const float* alpha_user, 
                       const float* A_user, int lda_user, 
                       const float* B_user, int ldb_user, 
                       const float* beta_user, 
                       float* C_user, int ldc_user, 
                       cudaStream_t stream) {
    
    cublasStatus_t status = cublasSetStream(handle, stream);
    if (status != CUBLAS_STATUS_SUCCESS) {
        Logger::error("cublasSetStream failed in gemm_f32_f32_cuda with error: " + std::to_string(status) + " (" + cublasGetStatusString(status) + ")");
        throw std::runtime_error("cublasSetStream failed");
    }

    cublasOperation_t opA_cublas, opB_cublas;
    int M_cublas, N_cublas, K_cublas;
    const float *A_cublas_ptr, *B_cublas_ptr;
    int LDA_cublas, LDB_cublas;


    M_cublas = n_user;
    N_cublas = m_user;
    K_cublas = k_user;

    A_cublas_ptr = B_user;    // B_user data is the first matrix for cuBLAS call
    LDA_cublas = (transb_user) ? k_user : ldb_user;

    B_cublas_ptr = A_user;    // A_user data is the second matrix for cuBLAS call
    LDB_cublas = (transa_user) ? m_user : lda_user;

    // Determine cuBLAS operations for A_cublas_ptr (B_user) and B_cublas_ptr (A_user)
    // opA_cublas applies to B_user, opB_cublas applies to A_user.
    if (!transa_user && !transb_user) {
        // User wants C = A * B
        opA_cublas = CUBLAS_OP_N; // for B_user
        opB_cublas = CUBLAS_OP_N; // for A_user
    } else if (transa_user && !transb_user) {
        // User wants C = A^T * B
        opA_cublas = CUBLAS_OP_N; // for B_user
        opB_cublas = CUBLAS_OP_T; // for A_user (A^T)
    } else if (!transa_user && transb_user) {
        // User wants C = A * B^T
        opA_cublas = CUBLAS_OP_T; // for B_user (B^T)
        opB_cublas = CUBLAS_OP_N; // for A_user
    } else { // transa_user && transb_user
        // User wants C = A^T * B^T
        opA_cublas = CUBLAS_OP_T; // for B_user (B^T)
        opB_cublas = CUBLAS_OP_T; // for A_user (A^T)
    }
    

    // For swapped computation: C^T result is (n_user x m_user) in column-major
    int LDC_cublas = ldc_user; // Leading dimension of C (which is n_user for row-major C(m,n))
    
    status = cublasSgemm(handle, opA_cublas, opB_cublas, 
                           M_cublas, N_cublas, K_cublas, 
                           alpha_user, 
                           A_cublas_ptr, LDA_cublas, 
                           B_cublas_ptr, LDB_cublas, 
                           beta_user, 
                           C_user, LDC_cublas); // FIXED: Use LDC_cublas instead of ldc_user

    if (status != CUBLAS_STATUS_SUCCESS) {
        Logger::error("cublasSgemm failed in gemm_f32_f32_cuda with error: " + std::to_string(status) + " (" + cublasGetStatusString(status) + ")");
        Logger::error("GEMM params (original user view): transa_user=" + std::to_string(transa_user) + " transb_user=" + std::to_string(transb_user) + 
                       " m=" + std::to_string(m_user) + " n=" + std::to_string(n_user) + " k=" + std::to_string(k_user) + 
                       " lda=" + std::to_string(lda_user) + " ldb=" + std::to_string(ldb_user) + " ldc=" + std::to_string(ldc_user));
        Logger::error("GEMM params (internal cublas view at failure): transA=" + std::to_string(opA_cublas==CUBLAS_OP_T) + " transB=" + std::to_string(opB_cublas==CUBLAS_OP_T) + 
                       " M_c=" + std::to_string(M_cublas) + " N_c=" + std::to_string(N_cublas) + " K_c=" + std::to_string(K_cublas) + 
                       " LDA_c=" + std::to_string(LDA_cublas) + " LDB_c=" + std::to_string(LDB_cublas) + " LDC_c=" + std::to_string(LDC_cublas));
        throw std::runtime_error("cublasSgemm failed");
    }
}


__global__ void attention_batch_prefill_cuda_kernel(
    float* d_output_batch_strided,      // Output: [B, H_q, D_h] effectively, though might be [B, H_q*D_h] and reshaped later
    const float* d_q_batch_strided,     // Q input: [B, H_q, D_h]
    const float* d_k_batch_strided,     // K input for current batch: [B, H_kv, D_h] (may be nullptr)
    const float* d_v_batch_strided,     // V input for current batch: [B, H_kv, D_h] (may be nullptr)
    float* d_kv_cache_k_base,           // K Cache base: [S_max, H_kv, D_h]
    float* d_kv_cache_v_base,           // V Cache base: [S_max, H_kv, D_h]
    int num_tokens_in_batch,            // B
    int num_q_heads,                    // H_q
    int num_kv_heads,                   // H_kv
    int head_dim,                       // D_h
    int start_pos_in_kv_cache,          // S_offset
    float scale,
    int cache_max_seq_len,
    const int* attention_mask_cu) {

    // Grid: (num_tokens_in_batch, num_q_heads)
    // Block: (head_dim)
    int token_idx_in_batch = blockIdx.x; // Current token in the batch we are computing attention FOR
    int q_head_idx = blockIdx.y;         // Current Q head we are computing attention FOR
    int dim_thread_idx = threadIdx.x;    // Thread index within the head dimension

    if (token_idx_in_batch >= num_tokens_in_batch || q_head_idx >= num_q_heads || dim_thread_idx >= head_dim) {
        return;
    }

    // Determine the GQA/MQA mapping: which KV head corresponds to the current Q head
    int kv_heads_per_q_group = num_q_heads / num_kv_heads;
    int kv_head_idx_for_q = q_head_idx / kv_heads_per_q_group;

    // Pointer to the current Q vector for this token and Q head
    const float* q_current_head_ptr = d_q_batch_strided +
                                      (size_t)token_idx_in_batch * num_q_heads * head_dim +
                                      (size_t)q_head_idx * head_dim;

    // Output pointer for the current token and Q head
    float* output_current_head_ptr = d_output_batch_strided +
                                     (size_t)token_idx_in_batch * num_q_heads * head_dim +
                                     (size_t)q_head_idx * head_dim;

    // CORRECTED: KV cache already contains current batch, so total context is available in cache
    int total_context_len = start_pos_in_kv_cache + num_tokens_in_batch;
    int max_attend_pos_for_this_token = start_pos_in_kv_cache + token_idx_in_batch;

    if (total_context_len > cache_max_seq_len) {
        total_context_len = cache_max_seq_len;
    }

    extern __shared__ float sdata[];
    float* s_scores = sdata; // Size: total_context_len
    float* s_dot_product_scratch = &sdata[total_context_len]; // Size: head_dim

    // Calculate QK^T for each K vector in the context with causal masking
    for (int context_pos = 0; context_pos < total_context_len; ++context_pos) {
        // CORRECTED: All K/V comes from cache since cache already contains current batch
        const float* k_vec_ptr = d_kv_cache_k_base +
                                 (size_t)context_pos * num_kv_heads * head_dim +
                                 (size_t)kv_head_idx_for_q * head_dim;
                    

        // Initialize score for this context position 
        if (dim_thread_idx == 0) {
            // Apply causal masking: token can only attend to positions <= its own position
            if (context_pos > max_attend_pos_for_this_token) {
                s_scores[context_pos] = -FLT_MAX; // Mask out future positions
            } else {
                s_scores[context_pos] = 0.0f;
            }
        }
        __syncthreads(); // Ensure s_scores[context_pos] is initialized before use

        // Skip computation if this position is masked
        if (context_pos <= max_attend_pos_for_this_token) {
            // Parallel dot product: each thread computes one element of Q_d * K_d
            // Store into shared memory scratchpad
            float q_val = q_current_head_ptr[dim_thread_idx];
            float k_val = k_vec_ptr[dim_thread_idx];
            s_dot_product_scratch[dim_thread_idx] = q_val * k_val;
            __syncthreads(); // Ensure all partial products are in s_dot_product_scratch

            // Thread 0 sums up the partial products from scratchpad
            if (dim_thread_idx == 0) {
                double dot_prod = 0.0; // USE DOUBLE FOR ACCUMULATION
                for (int d = 0; d < head_dim; ++d) {
                    dot_prod += (double)s_dot_product_scratch[d];
                }
                s_scores[context_pos] = (float)(dot_prod * scale); // Scale and cast back to float
            }
            __syncthreads(); // Ensure s_scores[context_pos] is written by thread 0 before next iteration
        }
    }

    // --- Softmax ---
    // Parallel softmax over s_scores[0...total_context_len-1]
    // 1. Find max score in s_scores (reduction)
    float max_score = -FLT_MAX; 

    float thread_local_max_for_scores = -FLT_MAX;
    for (int i = dim_thread_idx; i < total_context_len; i += blockDim.x) {
         thread_local_max_for_scores = fmaxf(thread_local_max_for_scores, s_scores[i]);
    }
    s_dot_product_scratch[dim_thread_idx] = thread_local_max_for_scores; 
    __syncthreads(); 

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (dim_thread_idx < s) { 
            s_dot_product_scratch[dim_thread_idx] = fmaxf(s_dot_product_scratch[dim_thread_idx], s_dot_product_scratch[dim_thread_idx + s]);
        }
        __syncthreads();
    }
    if (dim_thread_idx == 0) {
        max_score = s_dot_product_scratch[0];
    }
    __syncthreads(); 
    max_score = s_dot_product_scratch[0]; 

    float thread_local_sum_exp = 0.0f;
     for (int i = dim_thread_idx; i < total_context_len; i += blockDim.x) {
         float val = expf(s_scores[i] - max_score);
         s_scores[i] = val; 
         thread_local_sum_exp += val;
    }
    s_dot_product_scratch[dim_thread_idx] = thread_local_sum_exp; 
    __syncthreads(); 
    
    float sum_exp_scores = 0.0f; 
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (dim_thread_idx < s) {
            s_dot_product_scratch[dim_thread_idx] += s_dot_product_scratch[dim_thread_idx + s];
        }
        __syncthreads();
    }
    if (dim_thread_idx == 0) {
        sum_exp_scores = s_dot_product_scratch[0];
    }
    __syncthreads(); 
    sum_exp_scores = s_dot_product_scratch[0]; 

    float inv_sum_exp_scores = 1.0f / (sum_exp_scores + 1e-9f); 
    for (int i = dim_thread_idx; i < total_context_len; i += blockDim.x) {
        s_scores[i] *= inv_sum_exp_scores; 
    }
    __syncthreads(); 

    // --- Weighted Sum of V ---
    double weighted_sum_v_d = 0.0; // USE DOUBLE FOR ACCUMULATION
    for (int context_pos = 0; context_pos < total_context_len; ++context_pos) {
        // CORRECTED: All V comes from cache since cache already contains current batch
        const float* v_vec_ptr = d_kv_cache_v_base +
                                 (size_t)context_pos * num_kv_heads * head_dim +
                                 (size_t)kv_head_idx_for_q * head_dim;
        
        weighted_sum_v_d += (double)s_scores[context_pos] * (double)v_vec_ptr[dim_thread_idx];
    }
    
    output_current_head_ptr[dim_thread_idx] = (float)weighted_sum_v_d;

}

__global__ void rmsnorm_batch_kernel(float* d_out, const float* d_in, const float* d_weight, 
                                   int num_tokens, int hidden_size, float eps) {
    int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) {
        return;
    }

    const float* current_token_input = d_in + token_idx * hidden_size;
    float* current_token_output = d_out + token_idx * hidden_size;

    // Calculate sum of squares for the current token's hidden state
    float ssq = 0.0f;
    // A single thread per token for sum of squares calculation is not efficient.
    // This should be a parallel reduction per token.
    // For simplicity in this step, doing it serially per token via one thread block.
    // A more optimized version would use a grid-stride loop or a 2D grid.
    // Or, launch one block per token, and do a parallel reduction within the block.
    if (threadIdx.x == 0) { // Let one thread in the block handle this token
        for (int i = 0; i < hidden_size; ++i) {
            ssq += current_token_input[i] * current_token_input[i];
        }
        ssq /= hidden_size;
        float inv_norm_factor = rsqrtf(ssq + eps);

        for (int i = 0; i < hidden_size; ++i) {
            current_token_output[i] = current_token_input[i] * inv_norm_factor * d_weight[i];
        }
    }
}

// Optimized Batched RMSNorm Kernel - one block per token, parallel reduction within block
__global__ void rmsnorm_batch_kernel_optimized(float* d_out, const float* d_in, const float* d_weight, 
                                             int num_tokens, int hidden_size, float eps) {
    extern __shared__ float sdata[]; // For reduction
    int token_idx = blockIdx.x;

    if (token_idx >= num_tokens) {
        return;
    }

    const float* current_token_input = d_in + token_idx * hidden_size;
    float* current_token_output = d_out + token_idx * hidden_size;

    unsigned int tid = threadIdx.x;
    unsigned int N_per_token = hidden_size; // elements in one token's hidden state

    // Load input into shared memory and calculate sum of squares (partial)
    float local_ssq_sum = 0.0f;
    for (unsigned int i = tid; i < N_per_token; i += blockDim.x) {
        float val = current_token_input[i];
        local_ssq_sum += val * val;
    }
    sdata[tid] = local_ssq_sum;
    __syncthreads();

    // Parallel reduction in shared memory for ssq
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Thread 0 gets final sum of squares for this token, calculates norm, and applies
    if (tid == 0) {
        float total_ssq = sdata[0];
        total_ssq /= hidden_size;
        float inv_norm_factor = rsqrtf(total_ssq + eps);
        // This final application loop should also be parallelized if hidden_size is large.
        // For now, thread 0 does it.
        for (unsigned int i = 0; i < N_per_token; ++i) {
             current_token_output[i] = current_token_input[i] * inv_norm_factor * d_weight[i];
        }
    }    
}

// Final Optimized Batched RMSNorm Kernel - 2D grid, one warp per row (token), parallel reduction within warp
__global__ void rmsnorm_batch_kernel_final_optimized(float* __restrict__ d_out, 
                                             const float* __restrict__ d_in, 
                                             const float* __restrict__ d_weight, 
                                             int num_tokens, int hidden_size, float eps) {
    int token_idx = blockIdx.x; // Each block processes one token

    if (token_idx >= num_tokens) {
        return;
    }

    const float* current_token_input = d_in + token_idx * hidden_size;
    float* current_token_output = d_out + token_idx * hidden_size;
    
    // MODIFIED: Use double for ssq accumulation
    double ssq_double = 0.0;
    // Parallel reduction for ssq across threads in the block for the current token
    // Each thread handles a subset of the hidden_size elements for this token.
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float val = current_token_input[i];
        // MODIFIED: Accumulate into double
        ssq_double += static_cast<double>(val) * static_cast<double>(val);
    }

    // Reduce ssq_double across threads in the block
    extern __shared__ double sdata_double[]; // MODIFIED: Shared memory of type double
    sdata_double[threadIdx.x] = ssq_double;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata_double[threadIdx.x] += sdata_double[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        // MODIFIED: ssq_double is now fully reduced in sdata_double[0]
        ssq_double = sdata_double[0]; 
    }
    __syncthreads(); 

    if (threadIdx.x == 0) {
        // MODIFIED: Store mean square (as double) in shared memory
        sdata_double[0] = sdata_double[0] / hidden_size; 
    }
    __syncthreads();
    
    // MODIFIED: All threads read the mean_square (as double)
    double mean_square_double = sdata_double[0]; 
    // MODIFIED: Cast to float before rsqrtf
    float mean_square_float = static_cast<float>(mean_square_double);
    float inv_norm_factor = rsqrtf(mean_square_float + eps);

    // Apply normalization in parallel by all threads in the block for this token
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        current_token_output[i] = current_token_input[i] * inv_norm_factor * d_weight[i];
    }
}


void rmsnorm_batch_cuda(float* d_out, float* d_in, const float* d_weight, 
                        int num_tokens, int hidden_size, float eps, 
                        cudaStream_t stream) {

    if (num_tokens == 0 || hidden_size == 0) {
        Logger::info("[RMSNORM_BATCH_CUDA_EXIT] num_tokens or hidden_size is 0. Nothing to do.");
        return; // Nothing to do
    }

    // Log initial input values (first token, first few elements)
    if (num_tokens > 0 && hidden_size > 0) {
        int log_count = std::min(3, hidden_size);
        std::vector<float> h_in_sample(log_count);
        gpuErrchk(cudaMemcpyAsync(h_in_sample.data(), d_in, log_count * sizeof(float), cudaMemcpyDeviceToHost, stream));
        std::vector<float> h_weight_sample(log_count);
        gpuErrchk(cudaMemcpyAsync(h_weight_sample.data(), d_weight, log_count * sizeof(float), cudaMemcpyDeviceToHost, stream));
    }


    // Choose a reasonable block size. For RMSNorm, threads operate along hidden_size.
    // Max threads per block is typically 1024.
    // If hidden_size is small (e.g., < 256), could use hidden_size as blockDim.x
    // If hidden_size is large, use a fixed size like 256 or 512.
    int threads_per_block_x = (hidden_size <= 256) ? hidden_size : 256;
    if (hidden_size == 0) threads_per_block_x = 1; // Avoid 0 blockDim
    
    // We launch one block per token.
    dim3 block_dim(threads_per_block_x);
    dim3 grid_dim(num_tokens);
    
    // Shared memory for reduction within each block (for one token)
    // MODIFIED: Change shared memory size to use sizeof(double)
    size_t shared_mem_size = threads_per_block_x * sizeof(double);
    if (threads_per_block_x == 0) shared_mem_size = sizeof(double); // Min for sdata_double[0]

    rmsnorm_batch_kernel_final_optimized<<<grid_dim, block_dim, shared_mem_size, stream>>>(
        d_out, d_in, d_weight, num_tokens, hidden_size, eps
    );
    gpuErrchk(cudaGetLastError()); // Check for errors immediately after kernel launch

    // Log output values (first token, first few elements)
    if (num_tokens > 0 && hidden_size > 0) {
        int log_count = std::min(3, hidden_size);
        std::vector<float> h_out_sample(log_count);
        gpuErrchk(cudaMemcpyAsync(h_out_sample.data(), d_out, log_count * sizeof(float), cudaMemcpyDeviceToHost, stream));
    }
}

// Host wrapper for Batched Attention Prefill
void attention_batch_prefill_cuda(
    const float* d_q_batch_strided,   // Input Q: [B, H_q, D_h]
    const float* d_k_batch_strided,   // Input K for current batch (NEW)
    const float* d_v_batch_strided,   // Input V for current batch (NEW)
    float* d_kv_cache_k_base,         // K Cache: [S_max, H_kv, D_h] (changed to non-const)
    float* d_kv_cache_v_base,         // V Cache: [S_max, H_kv, D_h] (changed to non-const)
    float* d_output_batch_strided,    // Output: [B, H_q, D_h]
    int num_tokens_in_batch,          // B
    int start_pos_in_kv_cache,        // Start position for this batch in KV cache (NEW)
    int cache_max_seq_len,            // Max capacity of KV cache (NEW)
    int num_q_heads,                  // H_q
    int num_kv_heads,                 // H_kv
    int head_dim,                     // D_h
    float scale,
    cudaStream_t stream,
    const int* attention_mask_cu      // Optional attention mask
    ) {


    if (num_tokens_in_batch == 0 || head_dim == 0) {
        Logger::info("[ATTN_BATCH_PREFILL_CUDA_SKIP] num_tokens_in_batch or head_dim is 0. Nothing to do.");
        Logger::info("[ATTN_BATCH_PREFILL_CUDA_EXIT] Skipped operation.");
        return; // Nothing to do
    }
    if (num_kv_heads == 0 || num_q_heads % num_kv_heads != 0) {
        Logger::error("[ATTN_BATCH_PREFILL_CUDA_ERROR] Invalid num_kv_heads (" + std::to_string(num_kv_heads) +
                      ") or num_q_heads (" + std::to_string(num_q_heads) + ") for GQA.");
        Logger::info("[ATTN_BATCH_PREFILL_CUDA_EXIT] Error in head configuration.");
        return;
    }

    // Log initial input values (first token, first head, first few elements)
    int log_count_elements = std::min(3, head_dim);
    if (num_tokens_in_batch > 0 && head_dim > 0) {
        // Log Q for Token 0 and Token 1
        if (d_q_batch_strided && num_q_heads > 0) {
            for (int token_to_log_idx = 0; token_to_log_idx < std::min(num_tokens_in_batch, 2); ++token_to_log_idx) {
                std::vector<float> h_q_sample(log_count_elements);
                // Q is [B, H_q, D_h]. 
                // Offset for Token `token_to_log_idx`, Head 0 is `token_to_log_idx * num_q_heads * head_dim`.
                // We are logging the first `log_count_elements` of the first head (H0) for that token.
                const float* q_token_ptr = d_q_batch_strided + (size_t)token_to_log_idx * num_q_heads * head_dim;
                gpuErrchk(cudaMemcpyAsync(h_q_sample.data(), q_token_ptr, log_count_elements * sizeof(float), cudaMemcpyDeviceToHost, stream));
            }
        }

        // Log K_batch for Token 0 and Token 1
        if (d_k_batch_strided && num_kv_heads > 0) {
            for (int token_to_log_idx = 0; token_to_log_idx < std::min(num_tokens_in_batch, 2); ++token_to_log_idx) {
                std::vector<float> h_k_sample(log_count_elements);
                // K_batch is [B, H_kv, D_h].
                const float* k_token_ptr = d_k_batch_strided + (size_t)token_to_log_idx * num_kv_heads * head_dim;
                gpuErrchk(cudaMemcpyAsync(h_k_sample.data(), k_token_ptr, log_count_elements * sizeof(float), cudaMemcpyDeviceToHost, stream));
            }
        }

        // Log V_batch for Token 0 and Token 1
        if (d_v_batch_strided && num_kv_heads > 0) {
            for (int token_to_log_idx = 0; token_to_log_idx < std::min(num_tokens_in_batch, 2); ++token_to_log_idx) {
                std::vector<float> h_v_sample(log_count_elements);
                // V_batch is [B, H_kv, D_h].
                const float* v_token_ptr = d_v_batch_strided + (size_t)token_to_log_idx * num_kv_heads * head_dim;
                gpuErrchk(cudaMemcpyAsync(h_v_sample.data(), v_token_ptr, log_count_elements * sizeof(float), cudaMemcpyDeviceToHost, stream));
            }
        }
        
        // Note: K/V cache sampling might be less informative here if start_pos_in_kv_cache is large,
        // as it shows past context rather than current batch contributions to cache.
        // However, logging it for completeness if cache pointers are valid.
        if (d_kv_cache_k_base && start_pos_in_kv_cache < cache_max_seq_len && num_kv_heads > 0) {
            std::vector<float> h_kc_sample(log_count_elements);
            // K Cache: [S_max, H_kv, D_h]. Pointer to S_offset, H0.
            float* cache_k_ptr = d_kv_cache_k_base + (size_t)start_pos_in_kv_cache * num_kv_heads * head_dim;
            gpuErrchk(cudaMemcpyAsync(h_kc_sample.data(), cache_k_ptr, log_count_elements * sizeof(float), cudaMemcpyDeviceToHost, stream));
        }
        if (d_kv_cache_v_base && start_pos_in_kv_cache < cache_max_seq_len && num_kv_heads > 0) {
            std::vector<float> h_vc_sample(log_count_elements);
            // V Cache: [S_max, H_kv, D_h]. Pointer to S_offset, H0.
            float* cache_v_ptr = d_kv_cache_v_base + (size_t)start_pos_in_kv_cache * num_kv_heads * head_dim;
            gpuErrchk(cudaMemcpyAsync(h_vc_sample.data(), cache_v_ptr, log_count_elements * sizeof(float), cudaMemcpyDeviceToHost, stream));
        }
    }

    dim3 grid_dim(num_tokens_in_batch, num_q_heads); // One block per (token_in_batch, q_head)
    dim3 block_dim(head_dim); // Threads in block work on one head's computation, primarily along head_dim

    int total_context_len = start_pos_in_kv_cache + num_tokens_in_batch;
    if (total_context_len > cache_max_seq_len) total_context_len = cache_max_seq_len;
    size_t shared_mem_bytes = (total_context_len + head_dim) * sizeof(float);

    if (shared_mem_bytes > 48 * 1024) { 
        Logger::warning("[ATTN_BATCH_PREFILL_CUDA_WARN] Requested shared memory (" + std::to_string(shared_mem_bytes) + " bytes) might be large.");
    }

    attention_batch_prefill_cuda_kernel<<<grid_dim, block_dim, shared_mem_bytes, stream>>>(
        d_output_batch_strided,
        d_q_batch_strided,
        d_k_batch_strided,
        d_v_batch_strided,
        d_kv_cache_k_base,
        d_kv_cache_v_base,
        num_tokens_in_batch,
        num_q_heads,
        num_kv_heads,
        head_dim,
        start_pos_in_kv_cache,
        scale, 
        cache_max_seq_len,
        attention_mask_cu
    );
    gpuErrchk(cudaGetLastError());

    // Log output values
    if (num_tokens_in_batch > 0 && num_q_heads > 0 && head_dim > 0 && d_output_batch_strided) {
        for (int token_to_log_idx = 0; token_to_log_idx < std::min(num_tokens_in_batch, 2); ++token_to_log_idx) {
            std::vector<float> h_out_sample(log_count_elements);
            // Output is [B, H_q, D_h]. 
            const float* out_token_ptr = d_output_batch_strided + (size_t)token_to_log_idx * num_q_heads * head_dim;
            gpuErrchk(cudaMemcpyAsync(h_out_sample.data(), out_token_ptr, log_count_elements * sizeof(float), cudaMemcpyDeviceToHost, stream));
        }
    }
    cudaDeviceSynchronize();

    // Log output values
    if (num_tokens_in_batch > 0 && num_q_heads > 0 && head_dim > 0 && d_output_batch_strided) {
        for (int token_to_log_idx = 0; token_to_log_idx < std::min(num_tokens_in_batch, 2); ++token_to_log_idx) {
            std::vector<float> h_out_sample(log_count_elements);
            // Output is [B, H_q, D_h]. 
            const float* out_token_ptr = d_output_batch_strided + (size_t)token_to_log_idx * num_q_heads * head_dim;
            gpuErrchk(cudaMemcpyAsync(h_out_sample.data(), out_token_ptr, log_count_elements * sizeof(float), cudaMemcpyDeviceToHost, stream));
        }
    }
}

__global__ void update_kv_cache_batch_kernel(
    float* d_kv_cache_layer_base,         // K or V cache for the layer: [max_seq_len, num_kv_heads, head_dim]
    const float* d_keys_or_values_batch,  // Batch of K or V vectors from current forward pass: [num_tokens_in_batch, num_kv_heads, head_dim]
    int start_pos_in_kv_cache,            // Starting sequence position in cache for this batch
    int num_tokens_in_batch,
    int num_kv_heads,
    int head_dim,
    int cache_max_seq_len) {

    int token_idx_in_batch = blockIdx.x;  // Identifies which token from the batch (0 to num_tokens_in_batch-1)
    int kv_head_idx = blockIdx.y;       // Identifies which KV head (0 to num_kv_heads-1)
    int dim_idx = threadIdx.x;          // Identifies which dimension within the head_dim (0 to head_dim-1)

    // Boundary checks
    if (token_idx_in_batch >= num_tokens_in_batch || 
        kv_head_idx >= num_kv_heads || 
        dim_idx >= head_dim) {
    return;
  }

    // Calculate the global sequence position in the cache where this token's KV vector will be written
    int global_seq_pos = start_pos_in_kv_cache + token_idx_in_batch;

    // Boundary check for cache capacity
    if (global_seq_pos >= cache_max_seq_len) {
        // This should ideally be prevented by checks in the host code before launching,
        // or by model logic that handles sequence length limits.
        // For a kernel, simply returning is a common way to handle out-of-bounds access.
    return;
  }

    // Calculate the offset in the source batch tensor (d_keys_or_values_batch)
    // It's laid out as [token_idx_in_batch][kv_head_idx][dim_idx]
    size_t source_offset = (size_t)token_idx_in_batch * num_kv_heads * head_dim + 
                           (size_t)kv_head_idx * head_dim + 
                           dim_idx;

    // Calculate the offset in the destination KV cache tensor (d_kv_cache_layer_base)
    // It's laid out as [global_seq_pos][kv_head_idx][dim_idx]
    size_t cache_offset = (size_t)global_seq_pos * num_kv_heads * head_dim + 
                          (size_t)kv_head_idx * head_dim + 
                          dim_idx;

    // Perform the copy
    d_kv_cache_layer_base[cache_offset] = d_keys_or_values_batch[source_offset];
}

void update_kv_cache_batch_cuda(
    float* d_kv_cache_layer_base,        // Device pointer to the K or V cache for the current layer
    const float* d_keys_or_values_batch, // Device pointer to the batch of K or V vectors to be written
    int start_pos_in_kv_cache,           // The sequence position in the cache where writing for this batch should begin
    int num_tokens_in_batch,             // Number of tokens in the d_keys_or_values_batch
    int num_kv_heads,                    // Number of K/V heads
    int head_dim,                        // Dimension of each K/V head
    int cache_max_seq_len,               // Maximum sequence length capacity of the cache
                                   cudaStream_t stream) {

    if (num_tokens_in_batch == 0 || head_dim == 0 || num_kv_heads == 0) {
        Logger::debug("[UPDATE_KV_CACHE_BATCH_CUDA_SKIP] Nothing to update (num_tokens_in_batch, head_dim, or num_kv_heads is 0).");
        Logger::info("[UPDATE_KV_CACHE_BATCH_CUDA_EXIT] Skipped update.");
    return;
  }

    if (start_pos_in_kv_cache + num_tokens_in_batch > cache_max_seq_len) {
        Logger::error("[UPDATE_KV_CACHE_BATCH_CUDA_WARN] Batch (start: " + std::to_string(start_pos_in_kv_cache) + ", count: " + std::to_string(num_tokens_in_batch) +
                      ") might exceed KV cache capacity (" + std::to_string(cache_max_seq_len) + "). Kernel will clip if not careful, or this is an error.");
    }

    // Log input K/V values (first token, first head, first few elements)
    int log_count_elements = std::min(3, head_dim);
    if (num_tokens_in_batch > 0 && num_kv_heads > 0 && head_dim > 0) {
        std::vector<float> h_kv_sample_in(log_count_elements);
        // d_keys_or_values_batch is [num_tokens_in_batch][num_kv_heads][head_dim]
        // Offset for first token, first head = 0 (assuming data is contiguous for the first head of the first token)
        gpuErrchk(cudaMemcpyAsync(h_kv_sample_in.data(), d_keys_or_values_batch, log_count_elements * sizeof(float), cudaMemcpyDeviceToHost, stream));
    }

    // Grid: One block per (token_in_batch, kv_head)
    dim3 grid_dim(num_tokens_in_batch, num_kv_heads);
    // Block: One thread per dimension in the head vector
    dim3 block_dim(head_dim);

    update_kv_cache_batch_kernel<<<grid_dim, block_dim, 0, stream>>>(
        d_kv_cache_layer_base,
        d_keys_or_values_batch,
        start_pos_in_kv_cache,
        num_tokens_in_batch,
        num_kv_heads,
        head_dim,
        cache_max_seq_len
    );
  gpuErrchk(cudaGetLastError());

    // Log values written to cache (first token written, first head, first few elements)
    if (num_tokens_in_batch > 0 && num_kv_heads > 0 && head_dim > 0 && start_pos_in_kv_cache < cache_max_seq_len) {
        std::vector<float> h_kv_sample_out(log_count_elements);
        // Cache is [cache_max_seq_len][num_kv_heads][head_dim]
        // Offset for first token written in this batch, first head:
        // (start_pos_in_kv_cache * num_kv_heads * head_dim) + (0 * head_dim)
        // Pointer to start_pos_in_kv_cache, for the first head (head 0)
        float* first_token_first_head_in_cache_ptr = d_kv_cache_layer_base + 
            (size_t)start_pos_in_kv_cache * num_kv_heads * head_dim;
            
        gpuErrchk(cudaMemcpyAsync(h_kv_sample_out.data(), first_token_first_head_in_cache_ptr, log_count_elements * sizeof(float), cudaMemcpyDeviceToHost, stream));
    }
}

#endif // HAS_CUDA