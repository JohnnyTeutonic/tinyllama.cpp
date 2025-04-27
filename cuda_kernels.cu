// Kernels and wrappers for CUDA operations

#include "cuda_kernels.h"

#ifdef HAS_CUDA

#include <cuda_runtime.h>
#include <cuda_bf16.h> // For __nv_bfloat16 and conversion functions
#include <cmath>
#include <iostream> // For debug prints if needed

// <<< UTILITY KERNELS / DEVICE FUNCTIONS >>>

// Simple __device__ function to convert bfloat16 (stored as uint16_t) to float
// This assumes the uint16_t holds the raw bfloat16 bits.
__device__ inline float bf16_to_float32_device(uint16_t bf16_raw) {
    // Use CUDA's intrinsic if available (requires compute capability >= 80)
    #if __CUDA_ARCH__ >= 800
        __nv_bfloat16 bf16_val;
        // Directly copy bits. This is safe because we're dealing with raw bits.
        memcpy(&bf16_val, &bf16_raw, sizeof(uint16_t)); 
        return __bfloat162float(bf16_val);
    #else
        // Manual conversion for older architectures
        unsigned int bits = ((unsigned int)bf16_raw) << 16;
        float result;
        memcpy(&result, &bits, sizeof(float)); // Use memcpy for type-punning safety
        return result;
    #endif
}


// <<< RMS NORM KERNELS >>>

// Kernel 1: Calculate sum of squares (reduction)
// Uses shared memory for efficiency
__global__ void rmsnorm_sum_squares_kernel(const float* x, float* partial_sums, int n) {
    extern __shared__ float sdata[]; // Shared memory for reduction
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int gridSize = blockDim.x * gridDim.x;

    // Load data into shared memory
    sdata[tid] = (i < n) ? x[i] * x[i] : 0.0f;
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write block's result to global memory
    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

// Kernel 2: Apply normalization and weights
__global__ void rmsnorm_apply_kernel(const float* x, const float* weight, float* out, int n, float inv_norm_factor) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = x[i] * inv_norm_factor * weight[i];
    }
}

// --- RMSNorm C++ Wrapper (device pointer version) ---
void rmsnorm_vector_cuda(const float* x_dev,
                         const float* weight_dev,
                         float* out_dev,
                         int n,
                         float eps,
                         cudaStream_t stream) 
{
    // --- Device Memory is already allocated and filled ---
    // --- Kernel Launch Configuration ---
    const int threads_per_block = 256;
    int num_blocks_reduce = (n + threads_per_block - 1) / threads_per_block;
    size_t shared_mem_size = threads_per_block * sizeof(float);

    // Allocate partial_sums on device
    float* partial_sums_dev = nullptr;
    gpuErrchk(cudaMalloc(&partial_sums_dev, num_blocks_reduce * sizeof(float)));

    // --- Kernel 1: Sum of Squares Reduction ---
    rmsnorm_sum_squares_kernel<<<num_blocks_reduce, threads_per_block, shared_mem_size, stream>>>(x_dev, partial_sums_dev, n);
    gpuErrchk(cudaGetLastError());

    // --- Copy Partial Sums Device -> Host for Final Reduction ---
    float* partial_sums_host = new float[num_blocks_reduce];
    gpuErrchk(cudaMemcpyAsync(partial_sums_host, partial_sums_dev, num_blocks_reduce * sizeof(float), cudaMemcpyDeviceToHost, stream));
    gpuErrchk(cudaStreamSynchronize(stream));

    // --- Final Reduction on CPU ---
    double total_ssq = 0.0;
    for(int i=0; i < num_blocks_reduce; ++i) {
        total_ssq += partial_sums_host[i];
    }
    total_ssq /= n;
    float inv_norm_factor = 1.0f / std::sqrt(static_cast<float>(total_ssq) + eps);
    delete[] partial_sums_host;
    gpuErrchk(cudaFree(partial_sums_dev));

    // --- Kernel 2: Apply Normalization ---
    int num_blocks_apply = (n + threads_per_block - 1) / threads_per_block;
    rmsnorm_apply_kernel<<<num_blocks_apply, threads_per_block, 0, stream>>>(x_dev, weight_dev, out_dev, n, inv_norm_factor);
    gpuErrchk(cudaGetLastError());
}

// --- Old host-vector overload (calls device-pointer version internally) ---
void rmsnorm_vector_cuda(const std::vector<float>& x_in_host,
                         const std::vector<float>& weight_host,
                         std::vector<float>& out_host,
                         int n,
                         float eps)
{
    out_host.resize(n);
    float *x_dev = nullptr, *weight_dev = nullptr, *out_dev = nullptr;
    gpuErrchk(cudaMalloc(&x_dev, n * sizeof(float)));
    gpuErrchk(cudaMalloc(&weight_dev, n * sizeof(float)));
    gpuErrchk(cudaMalloc(&out_dev, n * sizeof(float)));
    gpuErrchk(cudaMemcpy(x_dev, x_in_host.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(weight_dev, weight_host.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    rmsnorm_vector_cuda(x_dev, weight_dev, out_dev, n, eps);
    gpuErrchk(cudaMemcpy(out_host.data(), out_dev, n * sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(x_dev));
    gpuErrchk(cudaFree(weight_dev));
    gpuErrchk(cudaFree(out_dev));
}


// <<< MATVEC (BF16 * F32 -> F32) KERNELS >>>

// Kernel: Computes out = mat * vec
// Grid: Typically launch one block per row (or group of rows)
// Block: Threads cooperate to compute dot product for their assigned row(s)
__global__ void matvec_bf16_f32_kernel(const uint16_t* mat_bf16,
                                       const float* vec_f32,
                                       float* out_f32,
                                       int rows,
                                       int cols)
{
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= rows) return;

    // Use double for higher precision accumulation in shared memory
    extern __shared__ double s_dot_product[]; // <-- Changed to double
    
    // Each thread calculates a partial sum using double
    double partial_sum = 0.0; // <-- Changed to double
    int thread_col_start = threadIdx.x;
    int stride = blockDim.x;

    for (int c = thread_col_start; c < cols; c += stride) {
        float mat_val = bf16_to_float32_device(mat_bf16[row * cols + c]);
        // Cast input vec value to double for multiplication
        partial_sum += static_cast<double>(mat_val) * static_cast<double>(vec_f32[c]); // <-- Accumulate in double
    }
    
    // Store partial sum in shared memory
    s_dot_product[threadIdx.x] = partial_sum;
    __syncthreads();

    // Reduce partial sums within the block (using double)
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_dot_product[threadIdx.x] += s_dot_product[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Write final result for the row (cast back to float)
    if (threadIdx.x == 0) {
        out_f32[row] = static_cast<float>(s_dot_product[0]); // <-- Cast final result back to float
    }
}

// --- MatVec C++ Wrapper (device pointer version) ---
void matvec_bf16_f32_cuda(const std::vector<uint16_t>& mat_bf16_host,
                          const float* vec_f32_dev,
                          float* out_f32_dev,
                          int rows,
                          int cols,
                          cudaStream_t stream) {
    // Copy weights to device
    uint16_t* mat_bf16_dev = nullptr;
    gpuErrchk(cudaMalloc(&mat_bf16_dev, rows * cols * sizeof(uint16_t)));
    gpuErrchk(cudaMemcpy(mat_bf16_dev, mat_bf16_host.data(), rows * cols * sizeof(uint16_t), cudaMemcpyHostToDevice));
    // Kernel launch config
    const int threads_per_block_x = 256;
    const int threads_per_block_y = 1;
    dim3 threads_per_block(threads_per_block_x, threads_per_block_y);
    int num_blocks_x = (rows + threads_per_block_y -1) / threads_per_block_y;
    int num_blocks_y = 1;
    dim3 num_blocks(num_blocks_x, num_blocks_y);
    size_t shared_mem_size = threads_per_block_x * sizeof(double); // <-- Adjusted for double
    // Launch kernel
    matvec_bf16_f32_kernel<<<num_blocks, threads_per_block, shared_mem_size, stream>>>(
        mat_bf16_dev, vec_f32_dev, out_f32_dev, rows, cols
    );
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaFree(mat_bf16_dev));
}

// Host-vector version: allocates/copies, calls device-pointer version, copies back
void matvec_bf16_f32_cuda(const std::vector<uint16_t>& mat_bf16_host,
                          const std::vector<float>& vec_f32_host,
                          std::vector<float>& out_f32_host,
                          int rows,
                          int cols) {
    out_f32_host.resize(rows);
    float* vec_f32_dev = nullptr;
    float* out_f32_dev = nullptr;
    gpuErrchk(cudaMalloc(&vec_f32_dev, cols * sizeof(float)));
    gpuErrchk(cudaMalloc(&out_f32_dev, rows * sizeof(float)));
    gpuErrchk(cudaMemcpy(vec_f32_dev, vec_f32_host.data(), cols * sizeof(float), cudaMemcpyHostToDevice));
    matvec_bf16_f32_cuda(mat_bf16_host, vec_f32_dev, out_f32_dev, rows, cols);
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMemcpy(out_f32_host.data(), out_f32_dev, rows * sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(vec_f32_dev));
    gpuErrchk(cudaFree(out_f32_dev));
}


// <<< SILU KERNEL >>>

__global__ void silu_kernel(const float* x, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x_val = x[i];
        out[i] = x_val / (1.0f + expf(-x_val)); // x * sigmoid(x)
    }
}

// --- SiLU C++ Wrapper ---
void silu_cuda(const std::vector<float>& x_host,
               std::vector<float>& out_host,
               int n)
{
    if (x_host.size() != n) {
        throw std::runtime_error("SiLU CUDA: Input vector size mismatch.");
    }
    out_host.resize(n);

    // --- Device Memory ---
    float* x_dev = nullptr;
    float* out_dev = nullptr;
    gpuErrchk(cudaMalloc(&x_dev, n * sizeof(float)));
    gpuErrchk(cudaMalloc(&out_dev, n * sizeof(float)));

    // --- Copy Host -> Device ---
    gpuErrchk(cudaMemcpy(x_dev, x_host.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    // --- Kernel Launch ---
    const int threads_per_block = 256;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    silu_kernel<<<num_blocks, threads_per_block>>>(x_dev, out_dev, n);
    gpuErrchk(cudaGetLastError());

    // --- Copy Device -> Host ---
    gpuErrchk(cudaMemcpy(out_host.data(), out_dev, n * sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaDeviceSynchronize());

    // --- Cleanup ---
    gpuErrchk(cudaFree(x_dev));
    gpuErrchk(cudaFree(out_dev));
}


// <<< SOFTMAX KERNELS >>>

// Kernel 1: Find max element in the vector (Reduction)
__global__ void softmax_find_max_kernel(const float* x, float* partial_max, int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load element into shared memory, using negative infinity for out-of-bounds
    sdata[tid] = (i < n) ? x[i] : -INFINITY;
    __syncthreads();

    // Max reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // Write block's maximum to global memory
    if (tid == 0) {
        partial_max[blockIdx.x] = sdata[0];
    }
}

// Kernel 2: Calculate exponentials and sum (Reduction)
__global__ void softmax_exp_sum_kernel(const float* x, float* partial_sums, int n, float max_val) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load shifted exponential into shared memory
    sdata[tid] = (i < n) ? expf(x[i] - max_val) : 0.0f;
    __syncthreads();

    // Sum reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write block's sum to global memory
    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

// Kernel 3: Normalize the output vector
__global__ void softmax_normalize_kernel(const float* x, float* out, int n, float max_val, float inv_sum) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = expf(x[i] - max_val) * inv_sum;
    }
}

// --- Softmax C++ Wrapper ---
void softmax_vector_cuda(const std::vector<float>& x_host,
                         std::vector<float>& out_host,
                         int n)
{
    if (x_host.size() != n) {
        throw std::runtime_error("Softmax CUDA: Input vector size mismatch.");
    }
    if (n == 0) {
        out_host.clear();
        return;
    }
    out_host.resize(n);

    // --- Device Memory ---
    float* x_dev = nullptr;
    float* out_dev = nullptr;
    float* partial_max_dev = nullptr;
    float* partial_sum_dev = nullptr;

    const int threads_per_block = 256; // Common block size
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    size_t shared_mem_size = threads_per_block * sizeof(float);

    // Allocate host buffers for partial results
    float* partial_max_host = new float[num_blocks];
    float* partial_sum_host = new float[num_blocks];

    gpuErrchk(cudaMalloc(&x_dev, n * sizeof(float)));
    gpuErrchk(cudaMalloc(&out_dev, n * sizeof(float)));
    gpuErrchk(cudaMalloc(&partial_max_dev, num_blocks * sizeof(float)));
    gpuErrchk(cudaMalloc(&partial_sum_dev, num_blocks * sizeof(float)));

    // --- Copy Host -> Device ---
    gpuErrchk(cudaMemcpy(x_dev, x_host.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    // --- Kernel 1: Find Max --- 
    softmax_find_max_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(x_dev, partial_max_dev, n);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaMemcpy(partial_max_host, partial_max_dev, num_blocks * sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaDeviceSynchronize()); 

    // --- Final Max Reduction on CPU ---
    float max_val = -INFINITY;
    for (int i = 0; i < num_blocks; ++i) {
        if (partial_max_host[i] > max_val) {
            max_val = partial_max_host[i];
        }
    }

    // --- Kernel 2: Calculate Exp Sum --- 
    softmax_exp_sum_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(x_dev, partial_sum_dev, n, max_val);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaMemcpy(partial_sum_host, partial_sum_dev, num_blocks * sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaDeviceSynchronize());

    // --- Final Sum Reduction on CPU ---
    double exp_sum = 0.0;
    for (int i = 0; i < num_blocks; ++i) {
        exp_sum += partial_sum_host[i];
    }
    float inv_sum = 1.0f / static_cast<float>(exp_sum);

    // --- Kernel 3: Normalize Output --- 
    softmax_normalize_kernel<<<num_blocks, threads_per_block>>>(x_dev, out_dev, n, max_val, inv_sum);
    gpuErrchk(cudaGetLastError());

    // --- Copy Result Device -> Host ---
    gpuErrchk(cudaMemcpy(out_host.data(), out_dev, n * sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaDeviceSynchronize());

    // --- Cleanup ---
    delete[] partial_max_host;
    delete[] partial_sum_host;
    gpuErrchk(cudaFree(x_dev));
    gpuErrchk(cudaFree(out_dev));
    gpuErrchk(cudaFree(partial_max_dev));
    gpuErrchk(cudaFree(partial_sum_dev));
}

// <<< SWIGLU KERNEL (Fused SiLU) >>>
__global__ void swiglu_kernel(const float* gate, const float* up, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float gate_val = gate[i];
        float silu_gate = gate_val / (1.0f + expf(-gate_val));
        out[i] = silu_gate * up[i];
    }
}

// --- SwiGLU C++ Wrapper (Fused) ---
void swiglu_cuda(const std::vector<float>& gate_host,
                 const std::vector<float>& up_host,
                 std::vector<float>& out_host,
                 int n)
{
    if (gate_host.size() != n || up_host.size() != n) {
        throw std::runtime_error("SwiGLU CUDA: Input vector size mismatch.");
    }
    out_host.resize(n);

    // --- Device Memory ---
    float* gate_dev = nullptr;
    float* up_dev = nullptr;
    float* out_dev = nullptr;
    gpuErrchk(cudaMalloc(&gate_dev, n * sizeof(float)));
    gpuErrchk(cudaMalloc(&up_dev, n * sizeof(float)));
    gpuErrchk(cudaMalloc(&out_dev, n * sizeof(float)));

    // --- Copy Host -> Device ---
    gpuErrchk(cudaMemcpy(gate_dev, gate_host.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(up_dev, up_host.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    // --- Kernel Launch ---
    const int threads_per_block = 256;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    swiglu_kernel<<<num_blocks, threads_per_block>>>(gate_dev, up_dev, out_dev, n);
    gpuErrchk(cudaGetLastError());

    // --- Copy Device -> Host ---
    gpuErrchk(cudaMemcpy(out_host.data(), out_dev, n * sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaDeviceSynchronize());

    // --- Cleanup ---
    gpuErrchk(cudaFree(gate_dev));
    gpuErrchk(cudaFree(up_dev));
    gpuErrchk(cudaFree(out_dev));
}

// <<< ROPE KERNEL >>>
__global__ void rope_kernel(float* x, int num_heads, int head_dim, const float* freqs_cis) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pairs = num_heads * (head_dim / 2);
    if (idx >= total_pairs) return;
    int head = idx / (head_dim / 2);
    int pair = idx % (head_dim / 2);
    int base = head * head_dim + pair;
    int base2 = head * head_dim + pair + head_dim / 2;
    float x0 = x[base];
    float x1 = x[base2];
    float cos_val = freqs_cis[2 * pair];
    float sin_val = freqs_cis[2 * pair + 1];
    x[base]  = x0 * cos_val - x1 * sin_val;
    x[base2] = x0 * sin_val + x1 * cos_val;
}

void rope_cuda(std::vector<float>& x, int num_heads, int head_dim, const std::vector<float>& freqs_cis) {
    if (x.size() != (size_t)(num_heads * head_dim)) {
        throw std::runtime_error("RoPE CUDA: x size mismatch.");
    }
    if (freqs_cis.size() != (size_t)head_dim) {
        throw std::runtime_error("RoPE CUDA: freqs_cis size mismatch.");
    }
    float* x_dev = nullptr;
    float* freqs_dev = nullptr;
    gpuErrchk(cudaMalloc(&x_dev, x.size() * sizeof(float)));
    gpuErrchk(cudaMalloc(&freqs_dev, freqs_cis.size() * sizeof(float)));
    gpuErrchk(cudaMemcpy(x_dev, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(freqs_dev, freqs_cis.data(), freqs_cis.size() * sizeof(float), cudaMemcpyHostToDevice));
    int total_pairs = num_heads * (head_dim / 2);
    int threads_per_block = 256;
    int num_blocks = (total_pairs + threads_per_block - 1) / threads_per_block;
    rope_kernel<<<num_blocks, threads_per_block>>>(x_dev, num_heads, head_dim, freqs_dev);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaMemcpy(x.data(), x_dev, x.size() * sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(x_dev));
    gpuErrchk(cudaFree(freqs_dev));
}

#endif // HAS_CUDA 