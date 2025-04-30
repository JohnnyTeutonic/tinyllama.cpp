// Kernels and wrappers for CUDA operations

#include "cuda_kernels.h"

#ifdef HAS_CUDA

#include <cuda_runtime.h>
#include <cuda_bf16.h> // For __nv_bfloat16 and conversion functions
#include <cublas_v2.h> // <<< INCLUDE CUBLAS HERE >>>
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

// Add a dedicated conversion kernel for BF16->FP32
__global__ void convert_bf16_to_fp32_kernel(const uint16_t* __restrict__ bf16_input,
                                           float* __restrict__ fp32_output,
                                           int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        // Convert BF16 raw bits to FP32
        uint16_t bf16_val = bf16_input[i];
        unsigned int ui = ((unsigned int)bf16_val) << 16;
        fp32_output[i] = *reinterpret_cast<float*>(&ui);
    }
}

// --- MatVec C++ Wrapper (Device/Device/Device - USING CUBLAS) ---
void matvec_bf16_f32_cuda(cublasHandle_t handle,       // <<< ADDED HANDLE
                          const uint16_t* mat_bf16_dev, 
                          const float* vec_f32_dev,
                          float* out_f32_dev,
                          int rows, // M dimension (output size)
                          int cols, // K dimension (input size)
                          cudaStream_t stream) 
{
    // SIMPLER APPROACH: Convert BF16->FP32 weights and use regular SGEMM
    // Allocate temporary FP32 matrix
    float* mat_fp32_dev = nullptr;
    size_t mat_size = (size_t)rows * cols;
    gpuErrchk(cudaMallocAsync(&mat_fp32_dev, mat_size * sizeof(float), stream));
    
    // Convert weights from BF16 to FP32
    int threads_per_block = 256;
    int num_blocks = (mat_size + threads_per_block - 1) / threads_per_block;
    convert_bf16_to_fp32_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        mat_bf16_dev, mat_fp32_dev, mat_size);
    gpuErrchk(cudaGetLastError());
    
    // Now we can use the simpler, more reliable cublasSgemm
    const float alpha = 1.0f;
    const float beta = 0.0f;
    int M = rows;
    int N = 1; // Vector treated as matrix with 1 column
    int K = cols;

    // Set cuBLAS stream
    cublasStatus_t status = cublasSetStream(handle, stream);
    if (status != CUBLAS_STATUS_SUCCESS) {
        Logger::error("cublasSetStream failed");
        throw std::runtime_error("cublasSetStream failed");
    }

    // Call regular cublasSgemm (much more reliable than GemmEx)
    // Experiment with both layouts to see which works
    // Assuming A is row-major MxK, we'll try transa=T with lda=K
    status = cublasSgemm(handle, 
                         CUBLAS_OP_T,     // Transpose A (row-major -> col-major)
                         CUBLAS_OP_N,     // No transpose B
                         M,               // Rows of C and op(A)
                         N,               // Cols of C and op(B) = 1
                         K,               // Cols of op(A) and rows of op(B)
                         &alpha,          // Scalar alpha
                         mat_fp32_dev,    // A matrix (now float)
                         K,               // Leading dimension of A = K for row-major
                         vec_f32_dev,     // B matrix (input vector)
                         K,               // Leading dimension of B = K for Kx1 col-major
                         &beta,           // Scalar beta 
                         out_f32_dev,     // C matrix (output vector)
                         M);              // Leading dimension of C = M for Mx1 col-major

    if (status != CUBLAS_STATUS_SUCCESS) {
        Logger::error("cublasSgemm failed with status: " + std::to_string(status));
        throw std::runtime_error("cublasSgemm failed");
    }

    // Free temporary memory
    gpuErrchk(cudaFreeAsync(mat_fp32_dev, stream));
}

// --- Corrected Mixed Host/Device matvec overload implementation ---
// Allocates temp device matrix, copies host matrix, calls device-pointer kernel/wrapper
void matvec_bf16_f32_cuda(cublasHandle_t handle, // <<< ADDED HANDLE
                          const std::vector<uint16_t>& mat_bf16_host, // HOST Matrix
                          const float* vec_f32_dev,                 // DEVICE Vector In
                          float* out_f32_dev,                       // DEVICE Vector Out
                          int rows,
                          int cols,
                          cudaStream_t stream)                     // Added stream
{
    if (mat_bf16_host.size() != (size_t)rows * cols) {
        throw std::runtime_error("matvec_bf16_f32_cuda (mixed): mat size mismatch.");
    }
    
    uint16_t* mat_bf16_dev = nullptr; // Need temp device matrix
    
    // Allocate temporary device buffer *only* for the matrix
    gpuErrchk(cudaMallocAsync(&mat_bf16_dev, mat_bf16_host.size() * sizeof(uint16_t), stream));
    
    // Copy host matrix data to device
    gpuErrchk(cudaMemcpyAsync(mat_bf16_dev, mat_bf16_host.data(), mat_bf16_host.size() * sizeof(uint16_t), cudaMemcpyHostToDevice, stream));
    
    // Call the *all-device-pointer* version of the wrapper (which now uses cuBLAS)
    matvec_bf16_f32_cuda(handle, mat_bf16_dev, vec_f32_dev, out_f32_dev, rows, cols, stream); // <<< PASS HANDLE
    
    // Free the temporary device matrix
    // Sync stream before freeing async alloc/memcpy memory on it.
    gpuErrchk(cudaStreamSynchronize(stream)); 
    gpuErrchk(cudaFree(mat_bf16_dev));
}

// --- START Add Host/Host/Host matvec overload implementation ---
// Allocates temp device matrix AND vectors, copies data, calls device-pointer version, copies result back
void matvec_bf16_f32_cuda(cublasHandle_t handle, // <<< ADDED HANDLE
                          const std::vector<uint16_t>& mat_bf16_host,
                          const std::vector<float>& vec_f32_host,
                          std::vector<float>& out_f32_host,
                          int rows,
                          int cols) 
{
    if (mat_bf16_host.size() != (size_t)rows * cols) {
        throw std::runtime_error("matvec_bf16_f32_cuda (host/host/host): mat size mismatch.");
    }
    if (vec_f32_host.size() != (size_t)cols) {
        throw std::runtime_error("matvec_bf16_f32_cuda (host/host/host): vec size mismatch.");
    }
    out_f32_host.resize(rows);
    
    uint16_t* mat_bf16_dev = nullptr;
    float* vec_f32_dev = nullptr;
    float* out_f32_dev = nullptr;
    
    // Allocate temporary device buffers 
    gpuErrchk(cudaMalloc(&mat_bf16_dev, mat_bf16_host.size() * sizeof(uint16_t)));
    gpuErrchk(cudaMalloc(&vec_f32_dev, vec_f32_host.size() * sizeof(float)));
    gpuErrchk(cudaMalloc(&out_f32_dev, out_f32_host.size() * sizeof(float)));
    
    // Copy host data to device
    gpuErrchk(cudaMemcpy(mat_bf16_dev, mat_bf16_host.data(), mat_bf16_host.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(vec_f32_dev, vec_f32_host.data(), vec_f32_host.size() * sizeof(float), cudaMemcpyHostToDevice));
    
    // Call the *all-device-pointer* version of the wrapper (which now uses cuBLAS)
    // Use default stream (0)
    matvec_bf16_f32_cuda(handle, mat_bf16_dev, vec_f32_dev, out_f32_dev, rows, cols, 0); // <<< PASS HANDLE
    
    // Synchronize default stream before copying back result
    gpuErrchk(cudaDeviceSynchronize());
    
    // Copy result back to host output vector
    gpuErrchk(cudaMemcpy(out_f32_host.data(), out_f32_dev, out_f32_host.size() * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Free temporary device buffers
    gpuErrchk(cudaFree(mat_bf16_dev));
    gpuErrchk(cudaFree(vec_f32_dev));
    gpuErrchk(cudaFree(out_f32_dev));
}
// --- END Add Host/Host/Host matvec overload implementation ---

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

// Device-pointer version of SwiGLU
void swiglu_cuda(const float* gate_dev, const float* up_dev, float* out_dev, int n, cudaStream_t stream) {
    const int threads_per_block = 256;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    swiglu_kernel<<<num_blocks, threads_per_block, 0, stream>>>(gate_dev, up_dev, out_dev, n);
    gpuErrchk(cudaGetLastError());
}

// <<< ROPE KERNEL (UPDATED) >>>
__global__ void rope_kernel(float* x, 
                            int num_heads, 
                            int head_dim, 
                            const float* all_freqs_cis_base, // Use base pointer
                            int pos) // Added pos
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pairs = num_heads * (head_dim / 2);
    if (idx >= total_pairs) return;
    
    int head = idx / (head_dim / 2);
    int pair_idx_in_head = idx % (head_dim / 2); // Index within the head's pairs (0 to dim/2 - 1)
    
    // Calculate the offset into the *full* frequency buffer
    // Layout: [max_seq_len, head_dim / 2, 2] flattened -> [max_seq_len * head_dim]
    // Offset for current pos and current pair index:
    size_t freq_base_offset = (size_t)pos * head_dim + (size_t)pair_idx_in_head * 2;
    
    // Indices into the input vector x
    int base_x0 = head * head_dim + pair_idx_in_head;
    int base_x1 = head * head_dim + pair_idx_in_head + head_dim / 2;
    
    // Fetch values
    float x0 = x[base_x0];
    float x1 = x[base_x1];
    float cos_val = all_freqs_cis_base[freq_base_offset];     // cos value for (pos, pair_idx_in_head)
    float sin_val = all_freqs_cis_base[freq_base_offset + 1]; // sin value for (pos, pair_idx_in_head)
    
    // Apply rotation
    x[base_x0] = x0 * cos_val - x1 * sin_val;
    x[base_x1] = x0 * sin_val + x1 * cos_val;
}

// --- DEVICE-POINTER ROPE WRAPPER (UPDATED) ---
void rope_cuda(float* x_dev, 
               int num_heads, 
               int head_dim, 
               const float* all_freqs_cis_dev_base, // Changed parameter name
               int pos, // Added pos
               cudaStream_t stream)
{
    int total_pairs = num_heads * (head_dim / 2);
    int threads_per_block = 256;
    int num_blocks = (total_pairs + threads_per_block - 1) / threads_per_block;
    
    // Call the updated kernel
    rope_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        x_dev, 
        num_heads, 
        head_dim, 
        all_freqs_cis_dev_base, // Pass base pointer
        pos // Pass pos
    );
    gpuErrchk(cudaGetLastError());
}

// <<< ATTENTION KERNEL (Reads directly from Cache) >>>
__global__ void attention_kernel(const float* Q_current, // Shape [num_heads * head_dim]
                               const float* K_layer_cache_base, // Base ptr for layer's K cache
                               const float* V_layer_cache_base, // Base ptr for layer's V cache
                               float* out,              // Output, Shape [num_heads * head_dim]
                               int current_seq_len,     // pos + 1
                               int head_dim, 
                               float scale, 
                               int cache_num_kv_heads, // Num K/V heads in cache
                               int num_q_heads)        // Num Query heads (gridDim.x)
{
    // Each block = one query head (q_head)
    int q_head = blockIdx.x;
    if (q_head >= num_q_heads) return;

    // Determine the corresponding K/V head index
    // Assuming simple repetition if num_q_heads > cache_num_kv_heads (GQA)
    int heads_per_kv = num_q_heads / cache_num_kv_heads; 
    int kv_head_idx = q_head / heads_per_kv;

    // Use shared memory for scores and probabilities for this q_head
    // Size needs to accommodate current_seq_len
    extern __shared__ float shared_scores[]; 
    float* scores = shared_scores; // [current_seq_len]
    // If not enough shared memory, this will fail kernel launch or cause errors.
    // Ensure max_seq_len isn't too large for available shared memory.

    // 1. Compute attention scores for the current q_head against all keys in its group
    float max_score = -INFINITY;
    const float* q_head_ptr = Q_current + q_head * head_dim;
    
    for (int k_pos = 0; k_pos < current_seq_len; ++k_pos) { // Iterate up to current timestep
        // Calculate offset into the FLAT K cache for the specific key vector
        // Layout: [max_seq_len, num_kv_heads, head_dim] <-- This was WRONG assumption before. Model uses [pos, kv_head, dim]
        // Correct Layout used in update kernel: [pos * num_kv_heads * head_dim + kv_head_idx * head_dim + d]
        size_t k_cache_base_offset = (size_t)k_pos * cache_num_kv_heads * head_dim + 
                                     (size_t)kv_head_idx * head_dim;
        const float* k_vec_ptr = K_layer_cache_base + k_cache_base_offset;
        
        float dot = 0.0f;
        // Calculate dot product: Q[q_head] . K[kv_head_idx, k_pos]
        for (int d = 0; d < head_dim; ++d) {
            dot += q_head_ptr[d] * k_vec_ptr[d];
        }
        dot *= scale;
        scores[k_pos] = dot;
        if (dot > max_score) max_score = dot;
    }

    // 2. Softmax (numerically stable)
    float exp_sum = 0.0f;
    for (int k_pos = 0; k_pos < current_seq_len; ++k_pos) {
        // Reuse shared memory for probabilities after scores are computed
        float prob = expf(scores[k_pos] - max_score);
        scores[k_pos] = prob; // Store prob back in shared mem
        exp_sum += prob;
    }
    float inv_sum = 1.0f / (exp_sum + 1e-9f); // Add epsilon for stability
    // Normalize probabilities in shared memory
    for (int k_pos = 0; k_pos < current_seq_len; ++k_pos) {
        scores[k_pos] *= inv_sum; // scores now holds probabilities
    }

    // 3. Weighted sum over V cache values
    // Each thread computes one dimension 'd' of the output vector for this q_head
    int d = threadIdx.x; // Thread index corresponds to element within head_dim
    if (d < head_dim) {
        double weighted_sum = 0.0; // Use double for accumulation
        for (int k_pos = 0; k_pos < current_seq_len; ++k_pos) {
            // Calculate offset into the FLAT V cache for the specific value vector
            size_t v_cache_base_offset = (size_t)k_pos * cache_num_kv_heads * head_dim + 
                                         (size_t)kv_head_idx * head_dim;
            const float* v_vec_ptr = V_layer_cache_base + v_cache_base_offset;
            
            weighted_sum += static_cast<double>(scores[k_pos]) * static_cast<double>(v_vec_ptr[d]);
        }
        // Write the final result for dimension 'd' of the output head vector
        out[q_head * head_dim + d] = static_cast<float>(weighted_sum);
    }
}

// --- ATTENTION WRAPPER (Reads directly from flat K/V Cache) ---
void attention_cuda(const float* Q_current_dev, 
                    const float* K_layer_cache_base, // Base ptr for layer's K cache
                    const float* V_layer_cache_base, // Base ptr for layer's V cache
                    float* out_dev,        
                    int num_q_heads,       // Renamed for clarity
                    int current_seq_len, 
                    int head_dim, 
                    float scale, 
                    int cache_max_seq_len, // Added - needed for kernel?
                    int cache_num_kv_heads, // Added
                    cudaStream_t stream) 
{
    // Each block = one query head
    // Each thread computes one dimension of the output vector within a head
    dim3 grid(num_q_heads);
    dim3 block(head_dim); // Threads iterate over head_dim for the weighted sum
    
    // Shared memory per block: need space for scores/probs for current_seq_len
    // This might exceed limits if current_seq_len is very large!
    size_t shared_mem_bytes = current_seq_len * sizeof(float); 
    // TODO: Add check against device shared memory limits if max_seq_len is large
    // cudaDeviceGetAttribute(&sharedMemPerBlock, cudaDevAttrMaxSharedMemoryPerBlock, devId);
    // if (shared_mem_bytes > sharedMemPerBlock) { /* error or use different kernel strategy */ }

    attention_kernel<<<grid, block, shared_mem_bytes, stream>>>(
        Q_current_dev, 
        K_layer_cache_base, 
        V_layer_cache_base, 
        out_dev, 
        current_seq_len, 
        head_dim, 
        scale, 
        cache_num_kv_heads, // Pass cache K/V head count
        num_q_heads         // Pass query head count (gridDim.x)
    );
    gpuErrchk(cudaGetLastError());
}

// <<< VECTOR ADDITION KERNEL >>>

__global__ void add_vectors_kernel(const float* a, const float* b, float* result, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        result[i] = a[i] + b[i];
    }
}

// --- Vector Addition C++ Wrapper ---
void add_vectors_cuda(const float* a_dev, 
                      const float* b_dev, 
                      float* result_dev, 
                      int n, 
                      cudaStream_t stream) 
{
    const int threads_per_block = 256;
    const int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    
    add_vectors_kernel<<<num_blocks, threads_per_block, 0, stream>>>(a_dev, b_dev, result_dev, n);
    gpuErrchk(cudaGetLastError()); // Check for errors after kernel launch
}

// <<< FUSED RESIDUAL ADDITION KERNEL >>>
// Kernel: result = matvec_out + residual
__global__ void add_residual_kernel(const float* matvec_out, const float* residual, float* result, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        result[i] = matvec_out[i] + residual[i];
    }
}

// --- Fused Residual Addition C++ Wrapper ---
void add_residual_cuda(const float* matvec_out_dev,
                       const float* residual_dev,
                       float* result_dev,
                       int n,
                       cudaStream_t stream)
{
    const int threads_per_block = 256;
    const int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    
    add_residual_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        matvec_out_dev, 
        residual_dev, 
        result_dev, 
        n
    );
    gpuErrchk(cudaGetLastError()); // Check for errors after kernel launch
}

// <<< K/V CACHE UPDATE KERNEL >>>

__global__ void update_kv_cache_kernel(float* cache_base_ptr,
                                     const float* current_kv_vector,
                                     int pos,
                                     int kv_head_idx,
                                     int max_seq_len,
                                     int num_kv_heads,
                                     int head_dim)
{
    int d = threadIdx.x; // Thread index corresponds to element within head_dim
    if (d >= head_dim) return;

    // Calculate the flat offset in the destination cache
    // Layout is [max_seq_len, num_kv_heads, head_dim]
    size_t cache_offset = (size_t)pos * num_kv_heads * head_dim + 
                          (size_t)kv_head_idx * head_dim + 
                          d;

    // Calculate the offset in the source vector (current K or V for this head)
    // Assuming current_kv_vector points to the start of the data for the target kv_head_idx
    size_t source_offset = d;

    // TODO: Add bounds checking for cache_offset against total cache size if needed, though parameters should guarantee validity if used correctly.
    // Example: size_t total_cache_size = (size_t)max_seq_len * num_kv_heads * head_dim;
    //          if (cache_offset >= total_cache_size) return; 

    // Copy the float value
    cache_base_ptr[cache_offset] = current_kv_vector[source_offset];
}

// --- K/V Cache Update C++ Wrapper ---
void update_kv_cache_cuda(float* cache_base_ptr, // Base pointer for K or V for the layer
                            const float* current_kv_vector, // Pointer to current token's K or V (just this head's data)
                            int pos,
                            int kv_head_idx,
                            int max_seq_len, 
                            int num_kv_heads, 
                            int head_dim,
                            cudaStream_t stream)
{
    // We need one thread per element in the head dimension to copy
    dim3 blockDim(head_dim); 
    dim3 gridDim(1); // Only need one block as kv_head_idx is specified

    // Calculate pointer to the start of the data for the specific head within the full current K/V vector
    // Assumes current_kv_vector passed to the *wrapper* might contain data for *all* heads.
    // We need to adjust the pointer passed to the kernel to point *only* to the relevant head's data.
    // NO - Let's simplify. Assume current_kv_vector passed to WRAPPER is ALREADY just for the target head.
    // This makes the call site in model.cpp slightly more complex but simplifies this wrapper.
    
    // Check if pos is within bounds (optional but good practice)
    if (pos < 0 || pos >= max_seq_len) {
        Logger::error("update_kv_cache_cuda: pos out of bounds (" + std::to_string(pos) + " >= " + std::to_string(max_seq_len) + ")");
        // Potentially throw or return an error code
        return; 
    }
    // Check if kv_head_idx is within bounds
    if (kv_head_idx < 0 || kv_head_idx >= num_kv_heads) {
         Logger::error("update_kv_cache_cuda: kv_head_idx out of bounds (" + std::to_string(kv_head_idx) + " >= " + std::to_string(num_kv_heads) + ")");
         return;
    }
    
    update_kv_cache_kernel<<<gridDim, blockDim, 0, stream>>>(
        cache_base_ptr,         // Pass base cache pointer for the layer
        current_kv_vector,      // Pass pointer to the *current* head's K/V data
        pos, 
        kv_head_idx, 
        max_seq_len, 
        num_kv_heads, 
        head_dim
    );
    gpuErrchk(cudaGetLastError()); // Check for errors after kernel launch
}

// <<< FUSED RoPE + K/V CACHE UPDATE KERNEL >>>
__global__ void rope_and_update_kv_cache_kernel(
    float* cache_base_ptr,          // Base pointer for K or V cache for the layer
    const float* kv_vector_head,    // Pointer to the *original* K or V data for *this specific head*
    const float* all_freqs_cis_base,// Base pointer to the global RoPE frequency buffer
    int pos,                        // Current sequence position
    int kv_head_idx,                // Index of the current K/V head
    int max_seq_len,                // Cache dimension: Max sequence length
    int num_kv_heads,               // Cache dimension: Number of K/V heads
    int head_dim                    // Cache dimension: Head dimension (must be even)
) {
    // Thread index corresponds to element pair index within the head dimension (0 to head_dim/2 - 1)
    int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int dim_half = head_dim / 2;

    if (pair_idx >= dim_half) return; // Only need threads for half the dimension

    // --- Load Original K/V values ---
    // Indices into the source kv_vector_head (assumed tightly packed for this head)
    int idx0 = pair_idx;
    int idx1 = pair_idx + dim_half;
    float kv0 = kv_vector_head[idx0];
    float kv1 = kv_vector_head[idx1];

    // --- Calculate RoPE Frequencies ---
    // Offset into the *full* frequency buffer [max_seq_len * head_dim] for the current pos and pair index
    size_t freq_base_offset = (size_t)pos * head_dim + (size_t)pair_idx * 2;
    float cos_val = all_freqs_cis_base[freq_base_offset];     // cos value for (pos, pair_idx)
    float sin_val = all_freqs_cis_base[freq_base_offset + 1]; // sin value for (pos, pair_idx)

    // --- Apply RoPE Rotation ---
    float kv0_rotated = kv0 * cos_val - kv1 * sin_val;
    float kv1_rotated = kv0 * sin_val + kv1 * cos_val;

    // --- Calculate Cache Write Offset ---
    // Layout is [max_seq_len, num_kv_heads, head_dim]
    size_t cache_offset_0 = (size_t)pos * num_kv_heads * head_dim + 
                            (size_t)kv_head_idx * head_dim + 
                            idx0; // Offset for the first element of the pair
    size_t cache_offset_1 = cache_offset_0 + dim_half; // Offset for the second element

    // --- Write RoPE'd values directly to Cache ---
    // Optional: Add bounds checking for cache_offset against total cache size
    // size_t total_cache_size = (size_t)max_seq_len * num_kv_heads * head_dim;
    // if (cache_offset_0 >= total_cache_size || cache_offset_1 >= total_cache_size) return; 
    
    cache_base_ptr[cache_offset_0] = kv0_rotated;
    cache_base_ptr[cache_offset_1] = kv1_rotated;
}

// --- Fused RoPE + K/V Cache Update C++ Wrapper ---
void rope_and_update_kv_cache_cuda(
    float* cache_base_ptr,          // Base K or V cache ptr for the layer
    const float* kv_vector_head,    // Original K or V data for *this head*
    const float* all_freqs_cis_base,// Global RoPE frequencies buffer
    int pos,
    int kv_head_idx,
    int max_seq_len, 
    int num_kv_heads, 
    int head_dim,
    cudaStream_t stream
) {
    if (head_dim % 2 != 0) {
        Logger::error("rope_and_update_kv_cache_cuda: head_dim must be even.");
        // Potentially throw or return error
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

    // Launch configuration: Need threads for half the head dimension
    int threads_per_block = 128; // Can tune this
    int num_blocks = (head_dim / 2 + threads_per_block - 1) / threads_per_block;

    rope_and_update_kv_cache_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        cache_base_ptr,
        kv_vector_head,
        all_freqs_cis_base,
        pos,
        kv_head_idx,
        max_seq_len,
        num_kv_heads,
        head_dim
    );
    gpuErrchk(cudaGetLastError()); 
}


// ============================================================================ 
// Kernel Implementation: Lookup Embedding (BF16 -> FP32)
// ============================================================================

__global__ void lookup_embedding_bf16_f32_kernel(const uint16_t* __restrict__ embedding_table,
                                               float* __restrict__ output_vector,
                                               int token_id,
                                               int hidden_size,
                                               int vocab_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate base offset for the token's embedding row
    // Assuming embedding table is tightly packed: [vocab_size, hidden_size]
    size_t base_offset = (size_t)token_id * hidden_size;

    if (idx < hidden_size && token_id >= 0 && token_id < vocab_size) {
        // Read BF16 value from the embedding table
        uint16_t bf16_val = embedding_table[base_offset + idx];

        // Convert BF16 to FP32
        // Simplest conversion: Shift left by 16 bits
        unsigned int ui = ((unsigned int)bf16_val) << 16;
        // Reinterpret the bits as a float
        output_vector[idx] = *reinterpret_cast<float*>(&ui);
    }
}

// Host wrapper for the lookup embedding kernel (MODIFIED)
void lookup_embedding_bf16_f32_cuda(const uint16_t* embedding_table_dev, // Changed: takes device pointer
                                    float* output_vector_dev,       // Renamed for clarity
                                    int token_id,
                                    int hidden_size,
                                    int vocab_size, // <-- ADDED BACK
                                    cudaStream_t stream) // Added stream 
{
    // REMOVED: Allocate temporary device memory for the embedding table
    // uint16_t* embedding_table_dev = nullptr;
    // size_t table_size_bytes = (size_t)vocab_size * hidden_size * sizeof(uint16_t);
    // gpuErrchk(cudaMalloc(&embedding_table_dev, table_size_bytes));

    // REMOVED: Copy the host table to the device
    // gpuErrchk(cudaMemcpy(embedding_table_dev, embedding_table_host.data(), table_size_bytes, cudaMemcpyHostToDevice));

    // --- Launch Kernel --- 
    dim3 threads_per_block(256);
    // Need enough blocks to cover the hidden_size dimension
    dim3 num_blocks((hidden_size + threads_per_block.x - 1) / threads_per_block.x);

    // Pass the actual vocab_size to the kernel now
    lookup_embedding_bf16_f32_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        embedding_table_dev, // Pass the device pointer parameter
        output_vector_dev,
        token_id,
        hidden_size,
        vocab_size // <-- PASS CORRECT VALUE
    );
    gpuErrchk(cudaGetLastError()); // Check for kernel launch errors
    
    // REMOVED: Free the temporary device memory for the table
    // gpuErrchk(cudaFree(embedding_table_dev));

    // No explicit sync needed here, subsequent operations will sync if necessary
    // gpuErrchk(cudaDeviceSynchronize()); 
}

// --- MatVec FP32/FP32 C++ Wrapper (Device/Device/Device - USING CUBLAS) ---
void matvec_f32_f32_cuda(cublasHandle_t handle,
                         const float* mat_f32_dev,
                         const float* vec_f32_dev,
                         float* out_f32_dev,
                         int rows, // M dimension (output size)
                         int cols, // K dimension (input size)
                         cudaStream_t stream)
{
    const float alpha = 1.0f;
    const float beta = 0.0f;
    int M = rows;
    int N = 1; // Vector treated as matrix with 1 column
    int K = cols;

    // Set cuBLAS stream
    cublasStatus_t status = cublasSetStream(handle, stream);
    if (status != CUBLAS_STATUS_SUCCESS) {
        Logger::error("cublasSetStream failed");
        throw std::runtime_error("cublasSetStream failed");
    }

    // Use the same layout as matvec_bf16_f32_cuda: row-major MxK, so transa=T, lda=K
    status = cublasSgemm(handle,
                        CUBLAS_OP_T,     // Transpose A (row-major -> col-major)
                        CUBLAS_OP_N,     // No transpose B
                        M,               // Rows of C and op(A)
                        N,               // Cols of C and op(B) = 1
                        K,               // Cols of op(A) and rows of op(B)
                        &alpha,          // Scalar alpha
                        mat_f32_dev,     // A matrix (float)
                        K,               // Leading dimension of A = K for row-major
                        vec_f32_dev,     // B matrix (input vector)
                        K,               // Leading dimension of B = K for Kx1 col-major
                        &beta,           // Scalar beta
                        out_f32_dev,     // C matrix (output vector)
                        M);              // Leading dimension of C = M for Mx1 col-major

    if (status != CUBLAS_STATUS_SUCCESS) {
        Logger::error("cublasSgemm (FP32) failed with status: " + std::to_string(status));
        throw std::runtime_error("cublasSgemm (FP32) failed");
    }
    // Debug log for first call
    static bool first_call = true;
    if (first_call) {
        std::cerr << "[CUDA] matvec_f32_f32_cuda called (rows=" << rows << ", cols=" << cols << ")\n";
        first_call = false;
    }
}

#endif // HAS_CUDA