

#include "cuda_kernels.h"

#ifdef HAS_CUDA

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

#include <cmath>
#include <iostream>

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

__global__ void rmsnorm_apply_kernel(const float* x, const float* weight,
                                     float* out, int n, float inv_norm_factor) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = x[i] * inv_norm_factor * weight[i];
  }
}

void rmsnorm_vector_cuda(const float* x_dev, const float* weight_dev,
                         float* out_dev, int n, float eps,
                         cudaStream_t stream) {
  const int threads_per_block = 256;
  int num_blocks_reduce = (n + threads_per_block - 1) / threads_per_block;
  size_t shared_mem_size = threads_per_block * sizeof(float);

  float* partial_sums_dev = nullptr;
  gpuErrchk(cudaMalloc(&partial_sums_dev, num_blocks_reduce * sizeof(float)));

  rmsnorm_sum_squares_kernel<<<num_blocks_reduce, threads_per_block,
                               shared_mem_size, stream>>>(x_dev,
                                                          partial_sums_dev, n);
  gpuErrchk(cudaGetLastError());

  float* partial_sums_host = new float[num_blocks_reduce];
  gpuErrchk(cudaMemcpy(partial_sums_host, partial_sums_dev,
                       num_blocks_reduce * sizeof(float),
                       cudaMemcpyDeviceToHost));

  double total_ssq = 0.0;
  for (int i = 0; i < num_blocks_reduce; ++i) {
    total_ssq += partial_sums_host[i];
  }
  total_ssq /= n;
  float inv_norm_factor = 1.0f / std::sqrt(static_cast<float>(total_ssq) + eps);
  delete[] partial_sums_host;
  gpuErrchk(cudaFree(partial_sums_dev));

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

void matvec_f32_f32_cuda(cublasHandle_t handle, const float* mat_f32_dev,
                         const float* vec_f32_dev, float* out_f32_dev, int rows,
                         int cols, cudaStream_t stream) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  int M = rows;
  int N = 1;
  int K = cols;

  cublasStatus_t status = cublasSetStream(handle, stream);
  if (status != CUBLAS_STATUS_SUCCESS) {
    Logger::error("cublasSetStream failed");
    throw std::runtime_error("cublasSetStream failed");
  }

  status = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha,
                       mat_f32_dev, K, vec_f32_dev, K, &beta, out_f32_dev, M);

  if (status != CUBLAS_STATUS_SUCCESS) {
    Logger::error("cublasSgemm (FP32) failed with status: " +
                  std::to_string(status));

    throw std::runtime_error("cublasSgemm (FP32) failed");
  }
}

void matvec_bf16_f32_cuda(cublasHandle_t handle, const uint16_t* mat_bf16_dev,
                          const float* vec_f32_dev, float* out_f32_dev,
                          int rows, int cols, cudaStream_t stream) {
  float* mat_fp32_dev = nullptr;
  size_t mat_size = (size_t)rows * cols;

  gpuErrchk(cudaMalloc(&mat_fp32_dev, mat_size * sizeof(float)));

  Logger::error(
      "matvec_bf16_f32_cuda fallback was called, but conversion kernel is "
      "removed! Zeroing output.");
  gpuErrchk(
      cudaMemsetAsync(out_f32_dev, 0, (size_t)rows * sizeof(float), stream));
  gpuErrchk(cudaFree(mat_fp32_dev));
  return;

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

void swiglu_cuda(const std::vector<float>& gate_host,
                 const std::vector<float>& up_host,
                 std::vector<float>& out_host, int n) {
  if (gate_host.size() != n || up_host.size() != n) {
    throw std::runtime_error("SwiGLU CUDA: Input vector size mismatch.");
  }
  out_host.resize(n);

  float* gate_dev = nullptr;
  float* up_dev = nullptr;
  float* out_dev = nullptr;
  gpuErrchk(cudaMalloc(&gate_dev, n * sizeof(float)));
  gpuErrchk(cudaMalloc(&up_dev, n * sizeof(float)));
  gpuErrchk(cudaMalloc(&out_dev, n * sizeof(float)));

  gpuErrchk(cudaMemcpy(gate_dev, gate_host.data(), n * sizeof(float),
                       cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(up_dev, up_host.data(), n * sizeof(float),
                       cudaMemcpyHostToDevice));

  const int threads_per_block = 256;
  int num_blocks = (n + threads_per_block - 1) / threads_per_block;
  swiglu_kernel<<<num_blocks, threads_per_block>>>(gate_dev, up_dev, out_dev,
                                                   n);
  gpuErrchk(cudaGetLastError());

  gpuErrchk(cudaMemcpy(out_host.data(), out_dev, n * sizeof(float),
                       cudaMemcpyDeviceToHost));
  gpuErrchk(cudaDeviceSynchronize());

  gpuErrchk(cudaFree(gate_dev));
  gpuErrchk(cudaFree(up_dev));
  gpuErrchk(cudaFree(out_dev));
}

void swiglu_cuda(const float* gate_dev, const float* up_dev, float* out_dev,
                 int n, cudaStream_t stream) {
  const int threads_per_block = 256;
  int num_blocks = (n + threads_per_block - 1) / threads_per_block;
  swiglu_kernel<<<num_blocks, threads_per_block, 0, stream>>>(gate_dev, up_dev,
                                                              out_dev, n);
  gpuErrchk(cudaGetLastError());
}

__global__ void rope_kernel(float* x, int num_heads, int head_dim,
                            const float* all_freqs_cis_base, int pos) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_pairs = num_heads * (head_dim / 2);
  if (idx >= total_pairs) return;

  int head = idx / (head_dim / 2);
  int pair_idx_in_head = idx % (head_dim / 2);

  size_t freq_base_offset =
      (size_t)pos * head_dim + (size_t)pair_idx_in_head * 2;

  int base_x0 = head * head_dim + pair_idx_in_head;
  int base_x1 = head * head_dim + pair_idx_in_head + head_dim / 2;

  float x0 = x[base_x0];
  float x1 = x[base_x1];
  float cos_val = all_freqs_cis_base[freq_base_offset];

  float sin_val = all_freqs_cis_base[freq_base_offset + 1];

  x[base_x0] = x0 * cos_val - x1 * sin_val;
  x[base_x1] = x0 * sin_val + x1 * cos_val;
}

void rope_cuda(float* x_dev, int num_heads, int head_dim,
               const float* all_freqs_cis_dev_base, int pos,
               cudaStream_t stream) {
  int total_pairs = num_heads * (head_dim / 2);
  int threads_per_block = 256;
  int num_blocks = (total_pairs + threads_per_block - 1) / threads_per_block;

  rope_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
      x_dev, num_heads, head_dim, all_freqs_cis_dev_base, pos);
  gpuErrchk(cudaGetLastError());
}

__global__ void attention_kernel(const float* Q_current,
                                 const float* K_layer_cache_base,
                                 const float* V_layer_cache_base, float* out,
                                 int current_seq_len, int head_dim, float scale,
                                 int cache_num_kv_heads, int num_q_heads) {
  int q_head = blockIdx.x;
  if (q_head >= num_q_heads) return;

  int heads_per_kv = num_q_heads / cache_num_kv_heads;
  int kv_head_idx = q_head / heads_per_kv;

  extern __shared__ float shared_scores[];
  float* scores = shared_scores;

  float max_score = -INFINITY;
  const float* q_head_ptr = Q_current + q_head * head_dim;

  for (int k_pos = 0; k_pos < current_seq_len; ++k_pos) {
    size_t k_cache_base_offset = (size_t)k_pos * cache_num_kv_heads * head_dim +
                                 (size_t)kv_head_idx * head_dim;
    const float* k_vec_ptr = K_layer_cache_base + k_cache_base_offset;

    float dot = 0.0f;

    for (int d = 0; d < head_dim; ++d) {
      dot += q_head_ptr[d] * k_vec_ptr[d];
    }
    dot *= scale;
    scores[k_pos] = dot;
    if (dot > max_score) max_score = dot;
  }

  float exp_sum = 0.0f;
  for (int k_pos = 0; k_pos < current_seq_len; ++k_pos) {
    float prob = expf(scores[k_pos] - max_score);
    scores[k_pos] = prob;
    exp_sum += prob;
  }
  float inv_sum = 1.0f / (exp_sum + 1e-9f);

  for (int k_pos = 0; k_pos < current_seq_len; ++k_pos) {
    scores[k_pos] *= inv_sum;
  }

  int d = threadIdx.x;
  if (d < head_dim) {
    double weighted_sum = 0.0;
    for (int k_pos = 0; k_pos < current_seq_len; ++k_pos) {
      size_t v_cache_base_offset =
          (size_t)k_pos * cache_num_kv_heads * head_dim +
          (size_t)kv_head_idx * head_dim;
      const float* v_vec_ptr = V_layer_cache_base + v_cache_base_offset;

      weighted_sum += static_cast<double>(scores[k_pos]) *
                      static_cast<double>(v_vec_ptr[d]);
    }

    out[q_head * head_dim + d] = static_cast<float>(weighted_sum);
  }
}

void attention_cuda(const float* Q_current_dev, const float* K_layer_cache_base,
                    const float* V_layer_cache_base, float* out_dev,
                    int num_q_heads, int current_seq_len, int head_dim,
                    float scale, int cache_max_seq_len, int cache_num_kv_heads,
                    cudaStream_t stream) {
  dim3 grid(num_q_heads);
  dim3 block(head_dim);

  size_t shared_mem_bytes = current_seq_len * sizeof(float);

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

#endif