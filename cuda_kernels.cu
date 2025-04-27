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

// --- RMSNorm C++ Wrapper --- 
void rmsnorm_vector_cuda(const std::vector<float>& x_in_host,
                         const std::vector<float>& weight_host,
                         std::vector<float>& out_host,
                         int n,
                         float eps) 
{
    if (x_in_host.size() != n || weight_host.size() != n) {
        throw std::runtime_error("RMSNorm CUDA: Input vector size mismatch.");
    }
    out_host.resize(n);

    // --- Device Memory Allocation ---
    float *x_in_dev = nullptr;
    float *weight_dev = nullptr;
    float *out_dev = nullptr;
    float *partial_sums_dev = nullptr;
    
    // Determine grid/block size for reduction
    // Needs careful tuning, using simple defaults for now
    const int threads_per_block = 256;
    int num_blocks_reduce = (n + threads_per_block - 1) / threads_per_block;
    size_t shared_mem_size = threads_per_block * sizeof(float);

    float *partial_sums_host = new float[num_blocks_reduce]; // Host buffer for final reduction

    gpuErrchk(cudaMalloc(&x_in_dev, n * sizeof(float)));
    gpuErrchk(cudaMalloc(&weight_dev, n * sizeof(float)));
    gpuErrchk(cudaMalloc(&out_dev, n * sizeof(float)));
    gpuErrchk(cudaMalloc(&partial_sums_dev, num_blocks_reduce * sizeof(float)));

    // --- Copy Data Host -> Device ---
    gpuErrchk(cudaMemcpy(x_in_dev, x_in_host.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(weight_dev, weight_host.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    // --- Launch Kernel 1: Sum of Squares Reduction ---
    rmsnorm_sum_squares_kernel<<<num_blocks_reduce, threads_per_block, shared_mem_size>>>(x_in_dev, partial_sums_dev, n);
    gpuErrchk(cudaGetLastError()); // Check for kernel launch errors

    // --- Copy Partial Sums Device -> Host for Final Reduction ---
    gpuErrchk(cudaMemcpy(partial_sums_host, partial_sums_dev, num_blocks_reduce * sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaDeviceSynchronize()); // Ensure copy is complete before CPU access

    // --- Final Reduction on CPU ---
    double total_ssq = 0.0;
    for(int i=0; i < num_blocks_reduce; ++i) {
        total_ssq += partial_sums_host[i];
    }
    total_ssq /= n; // Calculate mean sum of squares
    float inv_norm_factor = 1.0f / std::sqrt(static_cast<float>(total_ssq) + eps);

    // --- Launch Kernel 2: Apply Normalization ---
    int num_blocks_apply = (n + threads_per_block - 1) / threads_per_block;
    rmsnorm_apply_kernel<<<num_blocks_apply, threads_per_block>>>(x_in_dev, weight_dev, out_dev, n, inv_norm_factor);
    gpuErrchk(cudaGetLastError());

    // --- Copy Result Device -> Host ---
    gpuErrchk(cudaMemcpy(out_host.data(), out_dev, n * sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaDeviceSynchronize()); // Ensure final copy is done

    // --- Cleanup --- 
    delete[] partial_sums_host;
    gpuErrchk(cudaFree(x_in_dev));
    gpuErrchk(cudaFree(weight_dev));
    gpuErrchk(cudaFree(out_dev));
    gpuErrchk(cudaFree(partial_sums_dev));
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
    int row = blockIdx.x * blockDim.y + threadIdx.y; // Assign threads to rows
    if (row >= rows) return; // Boundary check

    // Use shared memory for accumulating dot product within a block/row
    // This specific kernel structure assigns one row per block for simplicity,
    // so shared memory might not be strictly necessary here, but shows the pattern.
    // A more optimized version would use shared memory to cache parts of vec_f32
    // if multiple rows are processed per block.
    extern __shared__ float s_dot_product[]; 
    
    // Each thread calculates a partial sum
    float partial_sum = 0.0f;
    int thread_col_start = threadIdx.x;
    int stride = blockDim.x;

    for (int c = thread_col_start; c < cols; c += stride) {
        float mat_val = bf16_to_float32_device(mat_bf16[row * cols + c]);
        partial_sum += mat_val * vec_f32[c];
    }
    
    // Store partial sum in shared memory
    s_dot_product[threadIdx.x] = partial_sum;
    __syncthreads();

    // Reduce partial sums within the block (assuming blockDim.x is power of 2)
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_dot_product[threadIdx.x] += s_dot_product[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Write final result for the row
    if (threadIdx.x == 0) {
        out_f32[row] = s_dot_product[0];
    }
}

// --- MatVec C++ Wrapper ---
void matvec_bf16_f32_cuda(const std::vector<uint16_t>& mat_bf16_host,
                          const std::vector<float>& vec_f32_host,
                          std::vector<float>& out_f32_host,
                          int rows,
                          int cols)
{
    if (mat_bf16_host.size() != (size_t)rows * cols || vec_f32_host.size() != cols) {
         throw std::runtime_error("MatVec CUDA: Input size mismatch.");
    }
    out_f32_host.resize(rows);

    // --- Device Memory Allocation ---
    uint16_t* mat_bf16_dev = nullptr;
    float* vec_f32_dev = nullptr;
    float* out_f32_dev = nullptr;

    gpuErrchk(cudaMalloc(&mat_bf16_dev, rows * cols * sizeof(uint16_t)));
    gpuErrchk(cudaMalloc(&vec_f32_dev, cols * sizeof(float)));
    gpuErrchk(cudaMalloc(&out_f32_dev, rows * sizeof(float)));

    // --- Copy Data Host -> Device ---
    gpuErrchk(cudaMemcpy(mat_bf16_dev, mat_bf16_host.data(), rows * cols * sizeof(uint16_t), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(vec_f32_dev, vec_f32_host.data(), cols * sizeof(float), cudaMemcpyHostToDevice));

    // --- Kernel Launch Configuration ---
    // Needs tuning! Using simple defaults.
    // Assign threads along columns for reduction, blocks along rows.
    const int threads_per_block_x = 256; // Threads cooperating on one row's dot product
    const int threads_per_block_y = 1;   // Process one row per block
    dim3 threads_per_block(threads_per_block_x, threads_per_block_y);
    
    int num_blocks_x = (rows + threads_per_block_y -1) / threads_per_block_y;
    int num_blocks_y = 1;
    dim3 num_blocks(num_blocks_x, num_blocks_y); 

    size_t shared_mem_size = threads_per_block_x * sizeof(float); // Shared memory for reduction within a block

    // --- Launch Kernel ---
    matvec_bf16_f32_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(
        mat_bf16_dev, vec_f32_dev, out_f32_dev, rows, cols
    );
    gpuErrchk(cudaGetLastError()); // Check for kernel launch errors

    // --- Copy Result Device -> Host ---
    gpuErrchk(cudaMemcpy(out_f32_host.data(), out_f32_dev, rows * sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaDeviceSynchronize()); // Ensure copy is complete

    // --- Cleanup ---
    gpuErrchk(cudaFree(mat_bf16_dev));
    gpuErrchk(cudaFree(vec_f32_dev));
    gpuErrchk(cudaFree(out_f32_dev));
}

#endif // HAS_CUDA 