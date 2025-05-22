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
  int M_blas = rows; // For C(M,N) = op(A)(M,K) * op(B)(K,N), M is rows of op(A) and C
  int N_blas = 1;    // N is cols of op(B) and C
  int K_blas = cols; // K is cols of op(A) and rows of op(B)

  // Assuming mat_f32_dev is row-major (actual_rows=rows, actual_cols=cols)
  // To use with CUBLAS_OP_T, A becomes actual_cols x actual_rows (K_blas x M_blas)
  // op(A) will have M_blas rows and K_blas cols for cuBLAS
  // vec_f32_dev is col-vector (actual_rows=cols, actual_cols=1)
  // op(B) will have K_blas rows and N_blas cols for cuBLAS

  cublasStatus_t status = cublasSetStream(handle, stream);
  if (status != CUBLAS_STATUS_SUCCESS) {
    Logger::error("cublasSetStream failed in matvec_f32_f32_cuda with error: " + std::to_string(status) + " (" + cublasGetStatusString(status) + ")");
    throw std::runtime_error("cublasSetStream failed");
  }

  Logger::info("[MATVEC_DEBUG] cublasSgemm call with: transA=T, transB=N" 
               ", M=" + std::to_string(M_blas) + ", N=" + std::to_string(N_blas) + ", K=" + std::to_string(K_blas) +
               ", LDA=" + std::to_string(K_blas) + " (cols_orig_A for T)" +
               ", LDB=" + std::to_string(K_blas) + " (rows_orig_B for N)" +
               ", LDC=" + std::to_string(M_blas) + " (rows_orig_C for N)" +
               ", alpha=" + std::to_string(alpha) + ", beta=" + std::to_string(beta) +
               ", A_ptr=" + Logger::ptrToString(mat_f32_dev) + ", B_ptr=" + Logger::ptrToString(vec_f32_dev) + ", C_ptr=" + Logger::ptrToString(out_f32_dev) );

  // C(M,N) = A(M,K)^T * B(K,N) ( ভুল ) -> C(M,N) = op(A)(M,K) * op(B)(K,N)
  // A is mat_f32_dev (user rows x user cols). op(A)=A^T (user cols x user rows)
  // B is vec_f32_dev (user cols x 1). op(B)=B (user cols x 1)
  // Result C is out_f32_dev (user rows x 1)
  // M_cublas = user rows (vocab_size)
  // N_cublas = 1
  // K_cublas = user cols (hidden_size)
  // For A^T: A is (rows x cols)_user_row_major. A^T (col-major view for cublas) is (cols x rows)_user.
  //            lda for A (when opA=T) is cols_user.
  // For B: B is (cols x 1)_user_col_vector. ldb for B (when opB=N) is cols_user.
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
    int intermediate_size,           // This is typically config_.ffn_hidden_size / 2
    cudaStream_t stream) {

    if (num_tokens == 0 || intermediate_size == 0) {
        return; // Nothing to do
    }

    // Choose a reasonable number of threads per block.
    // For element-wise operations like this, 256 is a common choice.
    const int threads_per_block = 256;
    
    // Launch one block per token.
    // Threads within each block will iterate over the intermediate_size if intermediate_size > threads_per_block.
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

    if (head_dim == 0) { // Should not happen with valid models
        // Logger::warn(\"rope_batch_cuda: head_dim is 0. Skipping RoPE application.\");
        return;
    }
    if (head_dim % 2 != 0) {
        Logger::error("rope_batch_cuda: head_dim must be even. Got " + std::to_string(head_dim));
        // Depending on error policy, might throw std::runtime_error or return an error code
        return; 
    }
    if (num_tokens == 0) {
        return; // Nothing to process
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
        gpuErrchk(cudaGetLastError()); // Check for errors after kernel launch
    }

    // Apply RoPE to K batch
    if (d_k_batch && num_kv_heads > 0) {
        dim3 grid_dim_k(num_tokens, num_kv_heads); // Use num_kv_heads for K
        rope_batch_kernel_cuda<<<grid_dim_k, block_dim, 0, stream>>>(
            d_k_batch,
            num_tokens,
            num_kv_heads, // Use num_kv_heads for K
            head_dim,
            d_all_freqs_cis_base,
            start_pos_offset,
            use_adjacent_pairing
        );
        gpuErrchk(cudaGetLastError()); // Check for errors after kernel launch
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
                    int num_q_heads, int current_seq_len, int head_dim,
                    float scale, int cache_max_seq_len, int cache_num_kv_heads,
                    cudaStream_t stream) {
  dim3 grid(num_q_heads);  // One block per Q head
  dim3 block(head_dim);   // head_dim threads per block

  // Shared memory: scores array (size current_seq_len) + dot_product_terms (size head_dim for reduction)
  size_t shared_mem_bytes = (current_seq_len + head_dim) * sizeof(float);
  

  attention_kernel<<<grid, block, shared_mem_bytes, stream>>>(
      Q_current_dev, K_layer_cache_base, V_layer_cache_base, out_dev,
      current_seq_len, head_dim, scale, cache_num_kv_heads, num_q_heads);
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
        return; // Nothing to do
    }

    const int threads_per_block = 256; // A common choice for element-wise operations
    
    // Launch one block per token.
    // Threads within each block will iterate over hidden_size if hidden_size > threads_per_block.
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

    if (!transa_user && !transb_user) {
        // Standard C_rowmajor(m,n) = A_rowmajor(m,k) * B_rowmajor(k,n)
        // This is equivalent to C_colmajor(n,m) = B_colmajor(n,k) * A_colmajor(k,m)
        // So, we call cublasSgemm with m_blas=n_user, n_blas=m_user, k_blas=k_user
        // A_blas_ptr = B_user, LDA_blas = ldb_user (which is n_user, the number of columns in B_user)
        // B_blas_ptr = A_user, LDB_blas = lda_user (which is k_user, the number of columns in A_user)
        // C_blas_ptr = C_user, LDC_blas = ldc_user (which is n_user, the number of columns in C_user)
        // All ops are CUBLAS_OP_N because we are passing pointers to row-major matrices
        // and cuBLAS will interpret them as column-major matrices with dimensions swapped for A and B vs C.
        // More precisely, cuBLAS expects A to be M_cublas x K_cublas (col-major) and B to be K_cublas x N_cublas (col-major)
        // If B_user (k_user x n_user row-major) is A_cublas, then A_cublas is n_user x k_user col-major. So M_cublas=n_user, K_cublas=k_user, LDA_cublas=n_user (=ldb_user).
        // If A_user (m_user x k_user row-major) is B_cublas, then B_cublas is k_user x m_user col-major. So K_cublas=k_user, N_cublas=m_user, LDB_cublas=k_user (=lda_user).
        // This leads to C_cublas being n_user x m_user.

        opA_cublas = CUBLAS_OP_N; 
        opB_cublas = CUBLAS_OP_N;
        M_cublas = n_user;         
        N_cublas = m_user;          
        K_cublas = k_user;         
        A_cublas_ptr = B_user;      
        LDA_cublas = ldb_user; // For B_user (k_user x n_user RM) treated as A_cublas (n_user x k_user CM), LDA is n_user.
        B_cublas_ptr = A_user;     
        LDB_cublas = lda_user; // For A_user (m_user x k_user RM) treated as B_cublas (k_user x m_user CM), LDB is k_user.
    } else {
        // User has specified transpositions. This case needs careful review if used,
        // as lda/ldb are for row-major from user, but cuBLAS needs them for col-major view with trans.
        // For now, direct pass-through (likely incorrect if transposing row-major matrices).
        Logger::warning("[GEMM_WRAPPER] transa_user or transb_user is true. Direct pass-through of ops and dims.");
        opA_cublas = transa_user ? CUBLAS_OP_T : CUBLAS_OP_N;
        opB_cublas = transb_user ? CUBLAS_OP_T : CUBLAS_OP_N;
        M_cublas = m_user;
        N_cublas = n_user;
        K_cublas = k_user;
        A_cublas_ptr = A_user;
        LDA_cublas = lda_user;
        B_cublas_ptr = B_user;
        LDB_cublas = ldb_user;
    }

    // Logging before the call
    Logger::info("[GEMM_WRAPPER_DEBUG] cublasSgemm effective call: transA=" + std::to_string(opA_cublas == CUBLAS_OP_T) + 
                 ", transB=" + std::to_string(opB_cublas == CUBLAS_OP_T) + 
                 ", M=" + std::to_string(M_cublas) + ", N=" + std::to_string(N_cublas) + ", K=" + std::to_string(K_cublas) + 
                 ", LDA=" + std::to_string(LDA_cublas) + ", LDB=" + std::to_string(LDB_cublas) + ", LDC=" + std::to_string(ldc_user) +
                 ", alpha=" + std::to_string(*alpha_user) + ", beta=" + std::to_string(*beta_user) +
                 ", A_ptr=" + Logger::ptrToString(A_cublas_ptr) + ", B_ptr=" + Logger::ptrToString(B_cublas_ptr) + ", C_ptr=" + Logger::ptrToString(C_user) );

    status = cublasSgemm(handle, opA_cublas, opB_cublas, 
                           M_cublas, N_cublas, K_cublas, 
                           alpha_user, 
                           A_cublas_ptr, LDA_cublas, 
                           B_cublas_ptr, LDB_cublas, 
                           beta_user, 
                           C_user, ldc_user); // C_user and ldc_user are for the final C_orig(m_user, n_user) row-major matrix.
                                          // For C_colmajor(n_user, m_user), ldc should be n_user.

    if (status != CUBLAS_STATUS_SUCCESS) {
        Logger::error("cublasSgemm failed in gemm_f32_f32_cuda with error: " + std::to_string(status) + " (" + cublasGetStatusString(status) + ")");
        Logger::error("GEMM params (original user view): transa_user=" + std::to_string(transa_user) + " transb_user=" + std::to_string(transb_user) + 
                       " m=" + std::to_string(m_user) + " n=" + std::to_string(n_user) + " k=" + std::to_string(k_user) + 
                       " lda=" + std::to_string(lda_user) + " ldb=" + std::to_string(ldb_user) + " ldc=" + std::to_string(ldc_user));
        Logger::error("GEMM params (internal cublas view at failure): transA=" + std::to_string(opA_cublas==CUBLAS_OP_T) + " transB=" + std::to_string(opB_cublas==CUBLAS_OP_T) + 
                       " M_c=" + std::to_string(M_cublas) + " N_c=" + std::to_string(N_cublas) + " K_c=" + std::to_string(K_cublas) + 
                       " LDA_c=" + std::to_string(LDA_cublas) + " LDB_c=" + std::to_string(LDB_cublas) + " LDC_c(user)=" + std::to_string(ldc_user));
        throw std::runtime_error("cublasSgemm failed");
    }
}

// cuda_kernels.cu

// ... (other existing kernels and includes) ...

// Declaration (already present in your file)
__global__ void attention_batch_prefill_cuda_kernel(
    float* d_output_batch_strided,      // Output: [num_tokens_in_batch, num_q_heads, head_dim]
    const float* d_q_batch_strided,     // Q input: [num_tokens_in_batch, num_q_heads, head_dim]
    const float* d_k_batch_strided,     // K input for current batch: [num_tokens_in_batch, num_kv_heads, head_dim]
    const float* d_v_batch_strided,     // V input for current batch: [num_tokens_in_batch, num_kv_heads, head_dim]
    float* d_kv_cache_k_base,           // K Cache base: [cache_max_seq_len, num_kv_heads, head_dim]
    float* d_kv_cache_v_base,           // V Cache base: [cache_max_seq_len, num_kv_heads, head_dim]
    int num_tokens_in_batch,            // B
    int num_q_heads,                    // H_q
    int num_kv_heads,                   // H_kv
    int head_dim,                       // D_h
    int start_pos_in_kv_cache,          // S_offset (usually 0 for full prefill, but can be >0 if appending to existing prefill)
    float scale,
    int cache_max_seq_len,
    const int* attention_mask_cu       // Optional, assumes [B, 1, S_ctx, S_ctx] or similar, adapt if needed
);


// Definition for attention_batch_prefill_cuda_kernel
__global__ void attention_batch_prefill_cuda_kernel(
    float* d_output_batch_strided,      // Output: [B, H_q, D_h] effectively, though might be [B, H_q*D_h] and reshaped later
    const float* d_q_batch_strided,     // Q input: [B, H_q, D_h]
    const float* d_k_batch_strided,     // K input for current batch: [B, H_kv, D_h]
    const float* d_v_batch_strided,     // V input for current batch: [B, H_kv, D_h]
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
    // Q is [B, H_q, D_h], so offset is token_idx_in_batch * (num_q_heads * head_dim) + q_head_idx * head_dim
    const float* q_current_head_ptr = d_q_batch_strided + 
                                      (size_t)token_idx_in_batch * num_q_heads * head_dim + 
                                      (size_t)q_head_idx * head_dim;

    // Output pointer for the current token and Q head
    float* output_current_head_ptr = d_output_batch_strided + 
                                     (size_t)token_idx_in_batch * num_q_heads * head_dim +
                                     (size_t)q_head_idx * head_dim;

    // --- Shared Memory Allocation ---
    // Max context length for any token in this prefill batch can be up to (start_pos_in_kv_cache + num_tokens_in_batch)
    // However, each token attends to a *different* effective context length.
    // Token `t` in the batch (0-indexed) attends to `start_pos_in_kv_cache + t + 1` previous tokens.
    int effective_context_len_for_this_token = start_pos_in_kv_cache + token_idx_in_batch + 1;
    if (effective_context_len_for_this_token > cache_max_seq_len) {
        effective_context_len_for_this_token = cache_max_seq_len;
    }

    extern __shared__ float sdata[];
    // sdata will hold:
    // 1. Scores for the current Q head against all its context KVs: size `effective_context_len_for_this_token`
    // 2. Temporary storage for dot product reduction (if needed, though direct sum is often fine): size `head_dim`
    // Shared memory size is passed by the host wrapper. Ensure it's sufficient for max_effective_ctx_for_batch + head_dim.
    float* s_scores = sdata;
    float* s_dot_product_scratch = &sdata[effective_context_len_for_this_token]; // if needed


    // --- Attention Score Calculation (Q dot K^T) ---
    // Each Q vector (for current token_idx_in_batch, q_head_idx) must attend to:
    // 1. KVs from the cache (0 to start_pos_in_kv_cache - 1)
    // 2. KVs from the current batch up to and including itself (0 to token_idx_in_batch)

    for (int k_context_idx = 0; k_context_idx < effective_context_len_for_this_token; ++k_context_idx) {
        // k_context_idx is the absolute position in the sequence being attended to.
        const float* k_vec_ptr;

        if (k_context_idx < start_pos_in_kv_cache) {
            // Key is from the past KV cache
            // K Cache: [S_max, H_kv, D_h]
            k_vec_ptr = d_kv_cache_k_base + 
                        (size_t)k_context_idx * num_kv_heads * head_dim + 
                        (size_t)kv_head_idx_for_q * head_dim;
        } else {
            // Key is from the current prefill batch
            int k_token_idx_in_batch = k_context_idx - start_pos_in_kv_cache;
            // K Batch: [B, H_kv, D_h]
            k_vec_ptr = d_k_batch_strided +
                        (size_t)k_token_idx_in_batch * num_kv_heads * head_dim +
                        (size_t)kv_head_idx_for_q * head_dim;
        }

        // Dot product for the current (dim_thread_idx) element
        // Initialize sum for this score in shared memory by thread 0 of the first score calculation
        if (dim_thread_idx == 0) {
            s_scores[k_context_idx] = 0.0f;
        }
         __syncthreads(); // Ensure s_scores[k_context_idx] is initialized by thread 0

        // Each thread computes its partial dot product and adds atomically or via reduction
        // For simplicity, a direct atomicAdd. For higher precision, use shared memory reduction.
        float q_val = q_current_head_ptr[dim_thread_idx];
        float k_val = k_vec_ptr[dim_thread_idx];
        // Accumulate dot product in s_scores[k_context_idx]
        // This requires a reduction over `head_dim` threads for EACH `k_context_idx`
        // A better way is to have each thread calculate q_val * k_val and store it in shared memory, then reduce.
        s_dot_product_scratch[dim_thread_idx] = q_val * k_val;
        __syncthreads();

        if (dim_thread_idx == 0) {
            float dot_prod = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                dot_prod += s_dot_product_scratch[d];
            }
            s_scores[k_context_idx] = dot_prod * scale;
        }
        __syncthreads(); // Ensure score is written before next k_context_idx or softmax
    }
    // At this point, s_scores[0...effective_context_len_for_this_token-1] contains scaled QK^T for the current Q head.

    // --- Softmax ---
    // Parallel softmax over s_scores[0...effective_context_len_for_this_token-1]
    // 1. Find max score in s_scores (reduction)
    float max_score = -__FLT_MAX__;
    for (int i = dim_thread_idx; i < effective_context_len_for_this_token; i += blockDim.x) {
         // Using s_dot_product_scratch for temporary max values per thread
         if (i == dim_thread_idx) s_dot_product_scratch[dim_thread_idx] = -__FLT_MAX__; // Initialize for this thread's pass
         s_dot_product_scratch[dim_thread_idx] = fmaxf(s_dot_product_scratch[dim_thread_idx], s_scores[i]);
    }
    __syncthreads();

    // Reduce max_score in shared memory (s_dot_product_scratch)
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (dim_thread_idx < s && (dim_thread_idx + s) < blockDim.x) { // ensure valid access
            s_dot_product_scratch[dim_thread_idx] = fmaxf(s_dot_product_scratch[dim_thread_idx], s_dot_product_scratch[dim_thread_idx + s]);
        }
        __syncthreads();
    }
    if (dim_thread_idx == 0) {
        max_score = s_dot_product_scratch[0];
    }
    __syncthreads(); // Ensure all threads have max_score (broadcast from thread 0 if needed, or all read s_dot_product_scratch[0])
    max_score = s_dot_product_scratch[0]; // All threads get the block-wide max_score

    // 2. Subtract max, exponentiate, and sum
    float sum_exp_scores = 0.0f;
     for (int i = dim_thread_idx; i < effective_context_len_for_this_token; i += blockDim.x) {
         float val = expf(s_scores[i] - max_score);
         s_scores[i] = val; // Store exp(score - max)
         // Using s_dot_product_scratch for temporary sum values per thread
         if (i == dim_thread_idx) s_dot_product_scratch[dim_thread_idx] = 0.0f; // Initialize for this thread's pass
         s_dot_product_scratch[dim_thread_idx] += val;
    }
    __syncthreads();
    
    // Reduce sum_exp_scores in shared memory (s_dot_product_scratch)
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (dim_thread_idx < s && (dim_thread_idx + s) < blockDim.x) { // ensure valid access
            s_dot_product_scratch[dim_thread_idx] += s_dot_product_scratch[dim_thread_idx + s];
        }
        __syncthreads();
    }
    if (dim_thread_idx == 0) {
        sum_exp_scores = s_dot_product_scratch[0];
    }
    __syncthreads(); // Ensure all threads have sum_exp_scores
    sum_exp_scores = s_dot_product_scratch[0]; // All threads get the block-wide sum

    // 3. Normalize (divide by sum)
    float inv_sum_exp_scores = 1.0f / (sum_exp_scores + 1e-9f); // Epsilon for stability
    for (int i = dim_thread_idx; i < effective_context_len_for_this_token; i += blockDim.x) {
        s_scores[i] *= inv_sum_exp_scores;
    }
    __syncthreads(); // All scores are now softmax probabilities

    // --- Weighted Sum of V ---
    // Each thread (dim_thread_idx) computes one element of the output head vector.
    float weighted_sum_v_d = 0.0f;
    for (int k_context_idx = 0; k_context_idx < effective_context_len_for_this_token; ++k_context_idx) {
        const float* v_vec_ptr;
        if (k_context_idx < start_pos_in_kv_cache) {
            // Value is from the past KV cache
            v_vec_ptr = d_kv_cache_v_base +
                        (size_t)k_context_idx * num_kv_heads * head_dim +
                        (size_t)kv_head_idx_for_q * head_dim;
        } else {
            // Value is from the current prefill batch
            int v_token_idx_in_batch = k_context_idx - start_pos_in_kv_cache;
            v_vec_ptr = d_v_batch_strided +
                        (size_t)v_token_idx_in_batch * num_kv_heads * head_dim +
                        (size_t)kv_head_idx_for_q * head_dim;
        }
        weighted_sum_v_d += s_scores[k_context_idx] * v_vec_ptr[dim_thread_idx];
    }
    
    // Write output
    output_current_head_ptr[dim_thread_idx] = weighted_sum_v_d;
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
    
    float ssq = 0.0f;
    // Parallel reduction for ssq across threads in the block for the current token
    // Each thread handles a subset of the hidden_size elements for this token.
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float val = current_token_input[i];
        ssq += val * val;
    }

    // Reduce ssq across threads in the block
    // Using warp-level primitives for better performance if available and hidden_size aligns well
    // For a generic block-wide reduction:
    extern __shared__ float sdata[];
    sdata[threadIdx.x] = ssq;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    // ssq is now fully reduced in sdata[0] for this block (token)
    if (threadIdx.x == 0) {
        ssq = sdata[0];
    }
    __syncthreads(); // Ensure all threads see the correct ssq from sdata[0]
    // It seems ssq is only correct in thread 0. We need to broadcast or re-read.
    // Let thread 0 write the final ssq to shared memory, and all threads read it back.
    if (threadIdx.x == 0) {
        sdata[0] = sdata[0] / hidden_size; // sdata[0] now holds mean square
    }
    __syncthreads();
    
    float mean_square = sdata[0]; // All threads read the mean_square for this token
    float inv_norm_factor = rsqrtf(mean_square + eps);

    // Apply normalization in parallel by all threads in the block for this token
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        current_token_output[i] = current_token_input[i] * inv_norm_factor * d_weight[i];
    }
}


void rmsnorm_batch_cuda(float* d_out, float* d_in, const float* d_weight, 
                        int num_tokens, int hidden_size, float eps, 
                        cudaStream_t stream) {
    if (num_tokens == 0 || hidden_size == 0) {
        return; // Nothing to do
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
    size_t shared_mem_size = threads_per_block_x * sizeof(float);
    if (threads_per_block_x == 0) shared_mem_size = sizeof(float); // Min for sdata[0]

    rmsnorm_batch_kernel_final_optimized<<<grid_dim, block_dim, shared_mem_size, stream>>>(
        d_out, d_in, d_weight, num_tokens, hidden_size, eps
    );
    gpuErrchk(cudaGetLastError());
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
        return; // Nothing to do
    }
    if (num_kv_heads == 0 || num_q_heads % num_kv_heads != 0) {
        // Handle error: num_kv_heads cannot be zero and must divide num_q_heads
        // For simplicity, returning, but in a real scenario, log an error or throw.
        // Logger::error("Error: Invalid num_kv_heads (%d) or num_q_heads (%d) for GQA.\\n", num_kv_heads, num_q_heads);
        return;
    }

    dim3 grid_dim(num_tokens_in_batch, num_q_heads); // One block per (token_in_batch, q_head)
    dim3 block_dim(head_dim); // Threads in block work on one head's computation, primarily along head_dim

    // Calculate shared memory needed.
    // Max of effective_context_len is start_pos_in_kv_cache + num_tokens_in_batch. Ensure it doesn't exceed cache_max_seq_len.
    int max_effective_ctx_for_batch = start_pos_in_kv_cache + num_tokens_in_batch;
    if (max_effective_ctx_for_batch > cache_max_seq_len) max_effective_ctx_for_batch = cache_max_seq_len;
    // Shared memory: s_scores (max_effective_ctx_for_batch) + reduction scratch (head_dim)
    size_t shared_mem_bytes = (max_effective_ctx_for_batch + head_dim) * sizeof(float);

    if (shared_mem_bytes > 48 * 1024) { // Check against device limits if necessary
        // Logger::warn("Requested shared memory might be too large.");
        // Potentially reduce shared_mem_bytes or error out
    }

    attention_batch_prefill_cuda_kernel<<<grid_dim, block_dim, shared_mem_bytes, stream>>>(
        d_output_batch_strided,
        d_q_batch_strided,
        d_k_batch_strided,        // Pass new parameter
        d_v_batch_strided,        // Pass new parameter
        d_kv_cache_k_base,        // Pass non-const version
        d_kv_cache_v_base,        // Pass non-const version
        num_tokens_in_batch,
        num_q_heads,
        num_kv_heads,
        head_dim,
        start_pos_in_kv_cache,    // Pass new parameter
        scale,                    // Scale is passed before cache_max_seq_len to kernel
        cache_max_seq_len,        // Pass new parameter
        attention_mask_cu
    );
    gpuErrchk(cudaGetLastError());
}

// Placeholder for a batched KV cache update kernel (Recommended)
// Replace this comment and the one below with the actual implementation.
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
        // Logger::debug("update_kv_cache_batch_cuda: Nothing to update (num_tokens_in_batch, head_dim, or num_kv_heads is 0).");
        return;
    }
    if (start_pos_in_kv_cache + num_tokens_in_batch > cache_max_seq_len) {
        // This indicates an issue that should likely be handled before calling this kernel,
        // e.g., by truncating the input or erroring out.
        // Logger::error("update_kv_cache_batch_cuda: Batch exceeds KV cache capacity.");
        // Consider throwing an error or specific handling if this case is critical.
        // For now, the kernel handles out-of-bound writes by returning, but this might hide issues.
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
}

#endif // HAS_CUDA