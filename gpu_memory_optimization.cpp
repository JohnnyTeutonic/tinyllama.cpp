#include "model.h"
#include "logger.h"

#ifdef HAS_CUDA
#include <cuda_runtime.h>
#include "cuda_kernels.h"

void TinyLlamaModel::allocate_persistent_batch_buffers() {
    if (d_persistent_batch_input_ != nullptr) {
        Logger::info("[GPU_OPT] Persistent batch buffers already allocated");
        return;
    }
    
    int hs = config_.hidden_size;
    int is = config_.intermediate_size;
    int n_kv_heads = config_.num_key_value_heads;
    int head_dim = hs / config_.num_attention_heads;
    
    size_t input_size = MAX_BATCH_TOKENS * hs * sizeof(float);
    size_t norm_size = MAX_BATCH_TOKENS * hs * sizeof(float);
    size_t residual_size = MAX_BATCH_TOKENS * hs * 2 * sizeof(float);
    size_t q_size = MAX_BATCH_TOKENS * hs * sizeof(float);
    size_t k_size = MAX_BATCH_TOKENS * n_kv_heads * head_dim * sizeof(float);
    size_t v_size = MAX_BATCH_TOKENS * n_kv_heads * head_dim * sizeof(float);
    size_t attn_out_size = MAX_BATCH_TOKENS * hs * sizeof(float);
    size_t attn_proj_size = MAX_BATCH_TOKENS * hs * sizeof(float);
    size_t gate_size = MAX_BATCH_TOKENS * is * sizeof(float);
    size_t up_size = MAX_BATCH_TOKENS * is * sizeof(float);
    size_t swiglu_size = MAX_BATCH_TOKENS * is * sizeof(float);
    size_t mlp_down_size = MAX_BATCH_TOKENS * hs * sizeof(float);
    
    size_t total_size = input_size + norm_size + residual_size + q_size + k_size + v_size +
                       attn_out_size + attn_proj_size + gate_size + up_size + swiglu_size + mlp_down_size;
    
    Logger::info("[GPU_OPT] Allocating persistent batch buffers: " + 
                 std::to_string(total_size / (1024*1024)) + " MB total");
    
    gpuErrchk(cudaMalloc(&d_persistent_batch_input_, input_size));
    gpuErrchk(cudaMalloc(&d_persistent_batch_norm_out_, norm_size));
    gpuErrchk(cudaMalloc(&d_persistent_batch_residual_, residual_size));
    gpuErrchk(cudaMalloc(&d_persistent_q_batch_, q_size));
    gpuErrchk(cudaMalloc(&d_persistent_k_batch_, k_size));
    gpuErrchk(cudaMalloc(&d_persistent_v_batch_, v_size));
    gpuErrchk(cudaMalloc(&d_persistent_attn_output_, attn_out_size));
    gpuErrchk(cudaMalloc(&d_persistent_attn_proj_out_, attn_proj_size));
    gpuErrchk(cudaMalloc(&d_persistent_gate_proj_out_, gate_size));
    gpuErrchk(cudaMalloc(&d_persistent_up_proj_out_, up_size));
    gpuErrchk(cudaMalloc(&d_persistent_swiglu_out_, swiglu_size));
    gpuErrchk(cudaMalloc(&d_persistent_mlp_down_out_, mlp_down_size));
    
    Logger::info("[GPU_OPT] Successfully allocated persistent batch buffers");
}

void TinyLlamaModel::free_persistent_batch_buffers() {
    if (d_persistent_batch_input_) {
        Logger::info("[GPU_OPT] Freeing persistent batch buffers");
        gpuErrchk(cudaFree(d_persistent_batch_input_));
        gpuErrchk(cudaFree(d_persistent_batch_norm_out_));
        gpuErrchk(cudaFree(d_persistent_batch_residual_));
        gpuErrchk(cudaFree(d_persistent_q_batch_));
        gpuErrchk(cudaFree(d_persistent_k_batch_));
        gpuErrchk(cudaFree(d_persistent_v_batch_));
        gpuErrchk(cudaFree(d_persistent_attn_output_));
        gpuErrchk(cudaFree(d_persistent_attn_proj_out_));
        gpuErrchk(cudaFree(d_persistent_gate_proj_out_));
        gpuErrchk(cudaFree(d_persistent_up_proj_out_));
        gpuErrchk(cudaFree(d_persistent_swiglu_out_));
        gpuErrchk(cudaFree(d_persistent_mlp_down_out_));
        
        d_persistent_batch_input_ = nullptr;
        d_persistent_batch_norm_out_ = nullptr;
        d_persistent_batch_residual_ = nullptr;
        d_persistent_q_batch_ = nullptr;
        d_persistent_k_batch_ = nullptr;
        d_persistent_v_batch_ = nullptr;
        d_persistent_attn_output_ = nullptr;
        d_persistent_attn_proj_out_ = nullptr;
        d_persistent_gate_proj_out_ = nullptr;
        d_persistent_up_proj_out_ = nullptr;
        d_persistent_swiglu_out_ = nullptr;
        d_persistent_mlp_down_out_ = nullptr;
    }
}

void TinyLlamaModel::resize_persistent_batch_buffers_if_needed(int required_batch_size) {
    if (required_batch_size <= MAX_BATCH_TOKENS) {
        if (d_persistent_batch_input_ == nullptr) {
            allocate_persistent_batch_buffers();
        }
        return;
    }
    
    Logger::warning("[GPU_OPT] Required batch size " + std::to_string(required_batch_size) + 
                   " exceeds MAX_BATCH_TOKENS " + std::to_string(MAX_BATCH_TOKENS) + 
                   ". This will cause memory reallocation per forward pass.");
}

#endif // HAS_CUDA 