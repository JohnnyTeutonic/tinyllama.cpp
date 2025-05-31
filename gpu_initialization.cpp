#include "gpu_initialization.h"
#include "logger.h"
#include "utils.h"
#include "quantization.h"
#include "model_constants.h"
#include "model_macros.h"
#include "weight_management.h"
#ifdef HAS_CUDA
#include "cuda_kernels.h"
#endif
#include <algorithm>
#include <cmath>
#include <cstring>

void TinyLlamaModel::initialize_gpu_and_rope() {
  Logger::info("[INIT_GPU_ROPE_DEBUG_L1113] Absolute Start of initialize_gpu_and_rope: config_.num_cpu_offload_layers = " + std::to_string(config_.num_cpu_offload_layers) + 
              ", config_.num_hidden_layers = " + std::to_string(config_.num_hidden_layers));
  Logger::info("[GPU_ROPE_INIT_ENTRY] Entered initialize_gpu_and_rope. Requested CPU Offload Layers: " + std::to_string(config_.num_cpu_offload_layers) + ", Total Hidden Layers: " + std::to_string(config_.num_hidden_layers));
  int hs = config_.hidden_size;
  int is = config_.intermediate_size;
  int nhl = config_.num_hidden_layers;
  int vs = config_.vocab_size;
  int n_heads = config_.num_attention_heads;
  int n_kv_heads = config_.num_key_value_heads;

  int num_cpu_layers_clamped = config_.num_cpu_offload_layers;
  if (num_cpu_layers_clamped < 0) num_cpu_layers_clamped = 0;
  if (num_cpu_layers_clamped > nhl) {
      Logger::warning("Requested CPU offload layers (" + std::to_string(config_.num_cpu_offload_layers) +
                      ") exceeds total hidden layers (" + std::to_string(nhl) +
                      "). Clamping to " + std::to_string(nhl) + " layers on CPU.");
      num_cpu_layers_clamped = nhl;
  }
  int active_num_cpu_layers = num_cpu_layers_clamped; 
  int active_num_gpu_layers = nhl - active_num_cpu_layers;

  Logger::info("Effective CPU layers for this init: " + std::to_string(active_num_cpu_layers) + ", Effective GPU layers for this init: " + std::to_string(active_num_gpu_layers));

  if (hs <= 0) throw std::runtime_error("Invalid model config: hidden_size must be positive.");
  if (vs <= 0) throw std::runtime_error("Invalid model config: vocab_size must be positive.");
  if (n_heads <= 0) throw std::runtime_error("Invalid model config: num_attention_heads must be positive.");
  if (n_kv_heads <= 0) throw std::runtime_error("Invalid model config: num_key_value_heads must be positive.");
  if (hs % n_heads != 0) throw std::runtime_error("Invalid model config: hidden_size not divisible by num_attention_heads.");

  int kv_dim = (hs / n_heads) * n_kv_heads;
  int head_dim = hs / n_heads;

  Logger::info("Precomputing RoPE frequencies on CPU (always done).");
  int max_seq_len = config_.max_position_embeddings;
  precomputed_freqs_cis_.resize((max_seq_len * head_dim) / 2);
  float theta = config_.rope_theta;
  for (int pos = 0; pos < max_seq_len; ++pos) {
    for (int i_rope = 0; i_rope < head_dim; i_rope += 2) {
      float freq = std::pow(theta, -((float)i_rope) / head_dim);
      float angle = pos * freq;
      precomputed_freqs_cis_[(pos * head_dim / 2) + (i_rope / 2)] = {std::cos(angle), std::sin(angle)};
    }
  }
  Logger::info("Finished precomputing RoPE cos/sin frequencies on CPU.");

#ifdef HAS_CUDA
#define SAFE_CUDA_FREE(ptr) if(ptr) { cudaFree(ptr); ptr = nullptr; }

  if (active_num_gpu_layers == 0) {
    Logger::info("No layers assigned to GPU (active_num_gpu_layers = 0). Cleaning up existing CUDA resources and skipping GPU initialization.");
    
    SAFE_CUDA_FREE(final_norm_dev);
    for (int i = 0; i < nhl; ++i) {
        SAFE_CUDA_FREE(layers[i].input_layernorm_dev);
        SAFE_CUDA_FREE(layers[i].post_attention_layernorm_dev);
    }
    SAFE_CUDA_FREE(token_embedding_table_dev_);
    SAFE_CUDA_FREE(lm_head_dev_);
    SAFE_CUDA_FREE(w_q_dev_); SAFE_CUDA_FREE(w_k_dev_); SAFE_CUDA_FREE(w_v_dev_); SAFE_CUDA_FREE(w_o_dev_);
    SAFE_CUDA_FREE(w_gate_dev_); SAFE_CUDA_FREE(w_up_dev_); SAFE_CUDA_FREE(w_down_dev_);
    SAFE_CUDA_FREE(all_freqs_cis_dev);
    SAFE_CUDA_FREE(x_dev_); SAFE_CUDA_FREE(x_norm_dev_); SAFE_CUDA_FREE(x_resid1_dev_); SAFE_CUDA_FREE(x_resid2_dev_);
    SAFE_CUDA_FREE(q_dev_); SAFE_CUDA_FREE(k_dev_); SAFE_CUDA_FREE(v_dev_); SAFE_CUDA_FREE(attn_out_dev_);
    SAFE_CUDA_FREE(attn_proj_dev_); SAFE_CUDA_FREE(gate_vec_dev_); SAFE_CUDA_FREE(up_vec_dev_);
    SAFE_CUDA_FREE(swiglu_vec_dev_); SAFE_CUDA_FREE(mlp_down_dev_); SAFE_CUDA_FREE(logits_dev_);
    SAFE_CUDA_FREE(token_embedding_table_f32_dev_);
    SAFE_CUDA_FREE(lm_head_f32_dev_);
    SAFE_CUDA_FREE(w_q_f32_dev_); SAFE_CUDA_FREE(w_k_f32_dev_); SAFE_CUDA_FREE(w_v_f32_dev_); SAFE_CUDA_FREE(w_o_f32_dev_);
    SAFE_CUDA_FREE(w_gate_f32_dev_); SAFE_CUDA_FREE(w_up_f32_dev_); SAFE_CUDA_FREE(w_down_f32_dev_);

    if (cublas_handle_) { cublasDestroy(cublas_handle_); cublas_handle_ = nullptr; }
    return;
  }

  Logger::info("Initializing CUDA resources for " + std::to_string(active_num_gpu_layers) + " GPU layers.");
  if (!cublas_handle_) {
    cublasStatus_t cublas_status = cublasCreate(&cublas_handle_);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("Failed to initialize cuBLAS: " + std::to_string(cublas_status));
    }
    Logger::info("cuBLAS handle created successfully.");
  }

  if (final_norm_f32.empty() && !final_norm.empty()) {
      Logger::info("Converting final_norm (BF16) to FP32 for GPU.");
      final_norm_f32 = bf16vec_to_float_vec(final_norm);
  }
  if (!final_norm_f32.empty()) {
    SAFE_CUDA_FREE(final_norm_dev);
    gpuErrchk(cudaMalloc(&final_norm_dev, final_norm_f32.size() * sizeof(float)));
    gpuErrchk(cudaMemcpy(final_norm_dev, final_norm_f32.data(), final_norm_f32.size() * sizeof(float), cudaMemcpyHostToDevice));
    Logger::info("Copied final_norm weights (FP32) to GPU.");
  } else {
    Logger::warning("Final norm weights (FP32) are empty, skipping GPU copy. This might be an issue if GPU layers are expected to use it.");
  }

  for (int i = 0; i < active_num_cpu_layers; ++i) {
      if (static_cast<size_t>(i) < layers.size()) {
        SAFE_CUDA_FREE(layers[i].input_layernorm_dev);
        SAFE_CUDA_FREE(layers[i].post_attention_layernorm_dev);
      }
  }
  Logger::info("Copying layer norm weights (FP32) to GPU for layers " + std::to_string(active_num_cpu_layers) + " to " + std::to_string(nhl - 1));
  Logger::info("[INIT_DEBUG_PRE_LOOP] Active CPU layers: " + std::to_string(active_num_cpu_layers));
  if (nhl > 0 && layers.size() > 0) {
    Logger::info("[INIT_DEBUG_PRE_LOOP] layers[0].input_layernorm_f32.empty(): " + std::string(layers[0].input_layernorm_f32.empty() ? "YES" : "NO") + 
                 ", Size: " + std::to_string(layers[0].input_layernorm_f32.size()));
  }
  for (int i = active_num_cpu_layers; i < nhl; ++i) {
    if (static_cast<size_t>(i) >= layers.size()) {
        Logger::error("Layer index " + std::to_string(i) + " out of bounds for layers vector (size: " + std::to_string(layers.size()) + ")");
        continue; 
    }
    SAFE_CUDA_FREE(layers[i].input_layernorm_dev);
    SAFE_CUDA_FREE(layers[i].post_attention_layernorm_dev);

    if (layers[i].input_layernorm_f32.empty() && !layers[i].input_layernorm.empty()) {
        layers[i].input_layernorm_f32 = bf16vec_to_float_vec(layers[i].input_layernorm);
    }
    if (layers[i].post_attention_layernorm_f32.empty() && !layers[i].post_attention_layernorm.empty()) {
        layers[i].post_attention_layernorm_f32 = bf16vec_to_float_vec(layers[i].post_attention_layernorm);
    }
    
    if (!layers[i].input_layernorm_f32.empty()) {
      gpuErrchk(cudaMalloc(&layers[i].input_layernorm_dev, layers[i].input_layernorm_f32.size() * sizeof(float)));
      gpuErrchk(cudaMemcpy(layers[i].input_layernorm_dev, layers[i].input_layernorm_f32.data(), layers[i].input_layernorm_f32.size() * sizeof(float), cudaMemcpyHostToDevice));
      if (i == active_num_cpu_layers) {
          Logger::info("[INIT_DEBUG] layers[" + std::to_string(i) + "].input_layernorm_dev allocated. Pointer: " + Logger::ptrToString(layers[i].input_layernorm_dev) + 
                       ", Size used for malloc: " + std::to_string(layers[i].input_layernorm_f32.size() * sizeof(float)) + " bytes (" +
                       std::to_string(layers[i].input_layernorm_f32.size()) + " elements). Host vector empty: " + (layers[i].input_layernorm_f32.empty() ? "YES" : "NO"));
      }
    } else {
      throw std::runtime_error("GPU Layer " + std::to_string(i) + ": input_layernorm_f32 weights are empty. Cannot offload to GPU without them.");
    }
    
    if (!layers[i].post_attention_layernorm_f32.empty()) {
      gpuErrchk(cudaMalloc(&layers[i].post_attention_layernorm_dev, layers[i].post_attention_layernorm_f32.size() * sizeof(float)));
      gpuErrchk(cudaMemcpy(layers[i].post_attention_layernorm_dev, layers[i].post_attention_layernorm_f32.data(), layers[i].post_attention_layernorm_f32.size() * sizeof(float), cudaMemcpyHostToDevice));
    } else {
      throw std::runtime_error("GPU Layer " + std::to_string(i) + ": post_attention_layernorm_f32 weights are empty. Cannot offload to GPU without them.");
    }
  }
  Logger::info("Finished processing layer norm weights for GPU layers.");


  SAFE_CUDA_FREE(token_embedding_table_dev_);
  SAFE_CUDA_FREE(token_embedding_table_f32_dev_);
  ensure_embed_tokens_dequantized();
  bool token_embeddings_processed_to_gpu_bf16 = false;

  if (active_num_gpu_layers > 0) {
  if (!embed_tokens.empty()) {
    gpuErrchk(cudaMalloc(&token_embedding_table_dev_, embed_tokens.size() * sizeof(uint16_t)));
    gpuErrchk(cudaMemcpy(token_embedding_table_dev_, embed_tokens.data(), embed_tokens.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
      Logger::info("Copied token_embedding_table (bf16 direct from model.embed_tokens) to GPU.");
      token_embeddings_processed_to_gpu_bf16 = true;
    }
    else if (!embed_tokens_f32.empty()) {
      std::vector<uint16_t> bf16_data(embed_tokens_f32.size());
      #pragma omp parallel for
      for (int i = 0; i < (int)embed_tokens_f32.size(); ++i) {
        bf16_data[i] = float32_to_bfloat16(embed_tokens_f32[i]);
      }
      gpuErrchk(cudaMalloc(&token_embedding_table_dev_, bf16_data.size() * sizeof(uint16_t)));
      gpuErrchk(cudaMemcpy(token_embedding_table_dev_, bf16_data.data(), bf16_data.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
      Logger::info("Converted token_embedding_table (fp32 source -> bf16) to GPU.");
      token_embeddings_processed_to_gpu_bf16 = true;
    }
    else if (!embed_tokens_q8_0.empty()) {
      std::vector<float> temp_f32_data(embed_tokens_q8_0.size() * GGML_QK8_0);
      #pragma omp parallel for
      for (int i = 0; i < (int)embed_tokens_q8_0.size(); ++i) {
        dequantize_q8_0_block(&embed_tokens_q8_0[i], &temp_f32_data[i * GGML_QK8_0]);
      }
      std::vector<uint16_t> bf16_data(temp_f32_data.size());
      #pragma omp parallel for
      for (int i = 0; i < (int)temp_f32_data.size(); ++i) {
        bf16_data[i] = float32_to_bfloat16(temp_f32_data[i]);
      }
      gpuErrchk(cudaMalloc(&token_embedding_table_dev_, bf16_data.size() * sizeof(uint16_t)));
      gpuErrchk(cudaMemcpy(token_embedding_table_dev_, bf16_data.data(), bf16_data.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
      Logger::info("Dequantized token_embedding_table (Q8_0 -> fp32 -> bf16) to GPU.");
      token_embeddings_processed_to_gpu_bf16 = true;
    }
    else if (!embed_tokens_q4k.empty()) {
      std::vector<float> temp_f32_data(embed_tokens_q4k.size() * GGML_QK_K);
      #pragma omp parallel for
      for (int i = 0; i < (int)embed_tokens_q4k.size(); ++i) {
        dequantize_q4_k_m(&embed_tokens_q4k[i], &temp_f32_data[i * GGML_QK_K], GGML_QK_K);
      }
      std::vector<uint16_t> bf16_data(temp_f32_data.size());
      #pragma omp parallel for
      for (int i = 0; i < (int)temp_f32_data.size(); ++i) {
        bf16_data[i] = float32_to_bfloat16(temp_f32_data[i]);
      }
      gpuErrchk(cudaMalloc(&token_embedding_table_dev_, bf16_data.size() * sizeof(uint16_t)));
      gpuErrchk(cudaMemcpy(token_embedding_table_dev_, bf16_data.data(), bf16_data.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
      Logger::info("Dequantized token_embedding_table (Q4_K -> fp32 -> bf16) to GPU.");
      token_embeddings_processed_to_gpu_bf16 = true;
    }
    else if (!embed_tokens_q6k.empty()) {
      std::vector<float> temp_f32_data(embed_tokens_q6k.size() * GGML_QK_K);
      #pragma omp parallel for
      for (int i = 0; i < (int)embed_tokens_q6k.size(); ++i) {
        dequantize_q6_k(&embed_tokens_q6k[i], &temp_f32_data[i * GGML_QK_K], GGML_QK_K);
      }
      std::vector<uint16_t> bf16_data(temp_f32_data.size());
      #pragma omp parallel for
      for (int i = 0; i < (int)temp_f32_data.size(); ++i) {
        bf16_data[i] = float32_to_bfloat16(temp_f32_data[i]);
      }
      gpuErrchk(cudaMalloc(&token_embedding_table_dev_, bf16_data.size() * sizeof(uint16_t)));
      gpuErrchk(cudaMemcpy(token_embedding_table_dev_, bf16_data.data(), bf16_data.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
      Logger::info("Dequantized token_embedding_table (Q6_K -> fp32 -> bf16) to GPU.");
      token_embeddings_processed_to_gpu_bf16 = true;
    }

    if (token_embeddings_processed_to_gpu_bf16) {
        Logger::info("[INIT_DEBUG] token_embedding_table_dev_ (BF16 on GPU) processed. Pointer: " + Logger::ptrToString(token_embedding_table_dev_) + 
                     ". Flag token_embeddings_processed_to_gpu_bf16: YES");
    }
    if (!token_embeddings_processed_to_gpu_bf16 && active_num_gpu_layers > 0) {
        Logger::warning("Token embeddings were not processed to GPU as BF16, despite GPU layers being active. This might indicate missing source embedding data in the model structure or an unhandled GGUF type for embeddings.");
    }
  } else {
    Logger::info("No GPU layers active, skipping token embedding table processing for GPU.");
  }

  SAFE_CUDA_FREE(lm_head_dev_);
  SAFE_CUDA_FREE(lm_head_f32_dev_);
  ensure_lm_head_dequantized();
  bool lm_head_processed_to_gpu_bf16 = false;

  if (active_num_gpu_layers > 0) {
  if (!lm_head.empty()) {
    gpuErrchk(cudaMalloc(&lm_head_dev_, lm_head.size() * sizeof(uint16_t)));
    gpuErrchk(cudaMemcpy(lm_head_dev_, lm_head.data(), lm_head.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
      Logger::info("Copied lm_head (bf16 direct from model.lm_head) to GPU.");
      lm_head_processed_to_gpu_bf16 = true;
    }
    else if (!lm_head_f32.empty()) {
      std::vector<uint16_t> bf16_data(lm_head_f32.size());
      #pragma omp parallel for
      for (int i = 0; i < (int)lm_head_f32.size(); ++i) {
        bf16_data[i] = float32_to_bfloat16(lm_head_f32[i]);
      }
      gpuErrchk(cudaMalloc(&lm_head_dev_, bf16_data.size() * sizeof(uint16_t)));
      gpuErrchk(cudaMemcpy(lm_head_dev_, bf16_data.data(), bf16_data.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
      Logger::info("Converted lm_head (fp32 source -> bf16) to GPU.");
      lm_head_processed_to_gpu_bf16 = true;
    }
    else if (!lm_head_q8_0.empty()) {
      std::vector<float> temp_f32_data(lm_head_q8_0.size() * GGML_QK8_0);
      #pragma omp parallel for
      for (int i = 0; i < (int)lm_head_q8_0.size(); ++i) {
        dequantize_q8_0_block(&lm_head_q8_0[i], &temp_f32_data[i * GGML_QK8_0]);
      }
      std::vector<uint16_t> bf16_data(temp_f32_data.size());
      #pragma omp parallel for
      for (int i = 0; i < (int)temp_f32_data.size(); ++i) {
        bf16_data[i] = float32_to_bfloat16(temp_f32_data[i]);
      }
      gpuErrchk(cudaMalloc(&lm_head_dev_, bf16_data.size() * sizeof(uint16_t)));
      gpuErrchk(cudaMemcpy(lm_head_dev_, bf16_data.data(), bf16_data.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
      Logger::info("Dequantized lm_head (Q8_0 -> fp32 -> bf16) to GPU.");
      lm_head_processed_to_gpu_bf16 = true;
    }
    else if (!lm_head_q4k.empty()) {
      std::vector<float> temp_f32_data(lm_head_q4k.size() * GGML_QK_K);
      #pragma omp parallel for
      for (int i = 0; i < (int)lm_head_q4k.size(); ++i) {
        dequantize_q4_k_m(&lm_head_q4k[i], &temp_f32_data[i * GGML_QK_K], GGML_QK_K);
      }
      std::vector<uint16_t> bf16_data(temp_f32_data.size());
      #pragma omp parallel for
      for (int i = 0; i < (int)temp_f32_data.size(); ++i) {
        bf16_data[i] = float32_to_bfloat16(temp_f32_data[i]);
      }
      gpuErrchk(cudaMalloc(&lm_head_dev_, bf16_data.size() * sizeof(uint16_t)));
      gpuErrchk(cudaMemcpy(lm_head_dev_, bf16_data.data(), bf16_data.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
      Logger::info("Dequantized lm_head (Q4_K -> fp32 -> bf16) to GPU.");
      lm_head_processed_to_gpu_bf16 = true;
    }
    else if (!lm_head_q6k.empty()) {
      std::vector<float> temp_f32_data(lm_head_q6k.size() * GGML_QK_K);
      #pragma omp parallel for
      for (int i = 0; i < (int)lm_head_q6k.size(); ++i) {
        dequantize_q6_k(&lm_head_q6k[i], &temp_f32_data[i * GGML_QK_K], GGML_QK_K);
      }
      std::vector<uint16_t> bf16_data(temp_f32_data.size());
      #pragma omp parallel for
      for (int i = 0; i < (int)temp_f32_data.size(); ++i) {
        bf16_data[i] = float32_to_bfloat16(temp_f32_data[i]);
      }
      gpuErrchk(cudaMalloc(&lm_head_dev_, bf16_data.size() * sizeof(uint16_t)));
      gpuErrchk(cudaMemcpy(lm_head_dev_, bf16_data.data(), bf16_data.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
      Logger::info("Dequantized lm_head (Q6_K -> fp32 -> bf16) to GPU.");
      lm_head_processed_to_gpu_bf16 = true;
    }

    if (!lm_head_processed_to_gpu_bf16) {
        Logger::warning("LM head was not processed to GPU as BF16, despite GPU layers being active. This might indicate missing source LM head data in the model structure or an unhandled GGUF type for LM head.");
    }
  } else {
    Logger::info("No GPU layers active, skipping LM head processing for GPU.");
  }

    SAFE_CUDA_FREE(lm_head_f32_dev_);

  if (active_num_gpu_layers > 0) {
    if (!lm_head_f32.empty()) {
      gpuErrchk(cudaMalloc(&lm_head_f32_dev_, lm_head_f32.size() * sizeof(float)));
      gpuErrchk(cudaMemcpy(lm_head_f32_dev_, lm_head_f32.data(), lm_head_f32.size() * sizeof(float), cudaMemcpyHostToDevice));
      Logger::info("[INIT_GPU_ROPE] Copied lm_head_f32 (host FP32) to GPU for lm_head_f32_dev_. Pointer: " + Logger::ptrToString(lm_head_f32_dev_));
    } else {
      Logger::error("[INIT_GPU_ROPE] Host lm_head_f32 is EMPTY. Cannot populate lm_head_f32_dev_. This WILL CAUSE a cublasSgemm error in the final matvec. Check model loading and initialize_weights logic for lm_head_f32 population.");
      lm_head_f32_dev_ = nullptr;
    }
  } else {
    lm_head_f32_dev_ = nullptr; 
  }

  
  Logger::info("Finished processing embedding and LM head tables for GPU.");

  SAFE_CUDA_FREE(all_freqs_cis_dev);
  if (active_num_gpu_layers > 0) {
    if (!precomputed_freqs_cis_.empty()) {
      size_t total_freq_elements = precomputed_freqs_cis_.size() * 2;
      gpuErrchk(cudaMalloc(&all_freqs_cis_dev, total_freq_elements * sizeof(float)));
      std::vector<float> flat_host_freqs; flat_host_freqs.reserve(total_freq_elements);
      for (const auto& p : precomputed_freqs_cis_) { flat_host_freqs.push_back(p.first); flat_host_freqs.push_back(p.second); }
      gpuErrchk(cudaMemcpy(all_freqs_cis_dev, flat_host_freqs.data(), total_freq_elements * sizeof(float), cudaMemcpyHostToDevice));
      Logger::info("Copied all precomputed RoPE frequencies to persistent GPU buffer.");
    } else {
      Logger::warning("Host precomputed_freqs_cis_ is empty. Skipping GPU RoPE buffer allocation. This WILL cause issues if GPU layers use RoPE.");
    }
    Logger::info("Finished processing RoPE frequencies for GPU.");
  } else {
    Logger::info("No GPU layers active, skipping RoPE GPU buffer allocation.");
  }

  if (active_num_gpu_layers > 0) {
    Logger::info("Allocating/Reallocating persistent GPU workspace buffers for " + std::to_string(active_num_gpu_layers) + " GPU layers.");
    size_t hs_bytes = (size_t)hs * sizeof(float);
    size_t is_bytes = (size_t)is * sizeof(float);
    size_t vs_bytes = (size_t)vs * sizeof(float);
    size_t k_dev_size_bytes = (size_t)n_kv_heads * head_dim * sizeof(float);
    size_t v_dev_size_bytes = (size_t)n_kv_heads * head_dim * sizeof(float);

#define REALLOC_GPU_WORKSPACE(ptr, sz) SAFE_CUDA_FREE(ptr); gpuErrchk(cudaMalloc(&ptr, sz));
    REALLOC_GPU_WORKSPACE(x_dev_, hs_bytes);
    REALLOC_GPU_WORKSPACE(x_norm_dev_, hs_bytes);
    REALLOC_GPU_WORKSPACE(x_resid1_dev_, hs_bytes);
    REALLOC_GPU_WORKSPACE(x_resid2_dev_, hs_bytes);
    REALLOC_GPU_WORKSPACE(q_dev_, hs_bytes);
    REALLOC_GPU_WORKSPACE(k_dev_, k_dev_size_bytes);
    REALLOC_GPU_WORKSPACE(v_dev_, v_dev_size_bytes);
    REALLOC_GPU_WORKSPACE(attn_out_dev_, hs_bytes);
    REALLOC_GPU_WORKSPACE(attn_proj_dev_, hs_bytes); 
    REALLOC_GPU_WORKSPACE(gate_vec_dev_, is_bytes);
    REALLOC_GPU_WORKSPACE(up_vec_dev_, is_bytes);
    REALLOC_GPU_WORKSPACE(swiglu_vec_dev_, is_bytes);
    REALLOC_GPU_WORKSPACE(mlp_down_dev_, hs_bytes); 
    REALLOC_GPU_WORKSPACE(logits_dev_, vs_bytes);
    Logger::info("Finished allocating/reallocating GPU workspace buffers.");
  } else {
    Logger::info("No GPU layers active, skipping GPU workspace buffer allocation.");
    SAFE_CUDA_FREE(x_dev_); SAFE_CUDA_FREE(x_norm_dev_); SAFE_CUDA_FREE(x_resid1_dev_); SAFE_CUDA_FREE(x_resid2_dev_);
    SAFE_CUDA_FREE(q_dev_); SAFE_CUDA_FREE(k_dev_); SAFE_CUDA_FREE(v_dev_); SAFE_CUDA_FREE(attn_out_dev_); SAFE_CUDA_FREE(attn_proj_dev_);
    SAFE_CUDA_FREE(gate_vec_dev_); SAFE_CUDA_FREE(up_vec_dev_); SAFE_CUDA_FREE(swiglu_vec_dev_); SAFE_CUDA_FREE(mlp_down_dev_); SAFE_CUDA_FREE(logits_dev_);
  }

  if (active_num_gpu_layers > 0) {
        selective_dequant_buffer_size_ = static_cast<size_t>(config_.max_position_embeddings) * head_dim;
        size_t selective_buffer_bytes = selective_dequant_buffer_size_ * sizeof(float);
        
        if (selective_dequant_buffer_size_ > 0) {
            SAFE_CUDA_FREE(selective_k_dequant_buffer_dev_);
            gpuErrchk(cudaMalloc(&selective_k_dequant_buffer_dev_, selective_buffer_bytes));
            SAFE_CUDA_FREE(selective_v_dequant_buffer_dev_);
            gpuErrchk(cudaMalloc(&selective_v_dequant_buffer_dev_, selective_buffer_bytes));
            Logger::info("Allocated SELECTIVE KVCache dequantization buffers (K and V) on GPU. Size per buffer: " + 
                         std::to_string(selective_buffer_bytes / (1024.0 * 1024.0)) + " MB (vs " +
                         std::to_string((static_cast<size_t>(config_.max_position_embeddings) * n_kv_heads * head_dim * sizeof(float)) / (1024.0 * 1024.0)) + " MB for full buffers)");
        } else {
            Logger::warning("Selective KVCache dequantization buffer size is 0. Skipping allocation.");
            SAFE_CUDA_FREE(selective_k_dequant_buffer_dev_);
            SAFE_CUDA_FREE(selective_v_dequant_buffer_dev_);
        }
    
            SAFE_CUDA_FREE(dequant_k_cache_buffer_dev_); 
            SAFE_CUDA_FREE(dequant_v_cache_buffer_dev_);
  } else {
    SAFE_CUDA_FREE(dequant_k_cache_buffer_dev_);
    SAFE_CUDA_FREE(dequant_v_cache_buffer_dev_);
    SAFE_CUDA_FREE(selective_k_dequant_buffer_dev_);
    SAFE_CUDA_FREE(selective_v_dequant_buffer_dev_);
  }

  bool process_bf16_concat_weights = active_num_gpu_layers > 0 && !layers[active_num_cpu_layers].q_proj.empty();
  if (process_bf16_concat_weights) {
    size_t layer_q_size = (size_t)hs*hs, layer_k_size = (size_t)kv_dim*hs, layer_v_size = (size_t)kv_dim*hs, layer_o_size = (size_t)hs*hs;
    size_t layer_gate_size = (size_t)is*hs, layer_up_size = (size_t)is*hs, layer_down_size = (size_t)hs*is;
    
    std::vector<uint16_t> h_q, h_k, h_v, h_o, h_gate, h_up, h_down;
    h_q.reserve(active_num_gpu_layers * layer_q_size); h_k.reserve(active_num_gpu_layers * layer_k_size);
    h_v.reserve(active_num_gpu_layers * layer_v_size); h_o.reserve(active_num_gpu_layers * layer_o_size);
    h_gate.reserve(active_num_gpu_layers * layer_gate_size); h_up.reserve(active_num_gpu_layers * layer_up_size);
    h_down.reserve(active_num_gpu_layers * layer_down_size);

    Logger::info("Concatenating BF16 weights for GPU layers on host (zero-padding if missing for a layer)...");
    for (int i = 0; i < active_num_gpu_layers; ++i) {
      int model_layer_idx = active_num_cpu_layers + i;
      const auto& lw = layers[model_layer_idx];
      
      if (!lw.q_proj.empty()) {
        h_q.insert(h_q.end(), lw.q_proj.begin(), lw.q_proj.end());
      } else {
        h_q.insert(h_q.end(), layer_q_size, bfloat16::ZERO);
      }

      if (!lw.k_proj.empty()) {
        h_k.insert(h_k.end(), lw.k_proj.begin(), lw.k_proj.end());
      } else {
        h_k.insert(h_k.end(), layer_k_size, bfloat16::ZERO);
      }

      if (!lw.v_proj.empty()) {
        h_v.insert(h_v.end(), lw.v_proj.begin(), lw.v_proj.end());
      } else {
        h_v.insert(h_v.end(), layer_v_size, bfloat16::ZERO);
      }

      if (!lw.o_proj.empty()) {
        h_o.insert(h_o.end(), lw.o_proj.begin(), lw.o_proj.end());
      } else {
        h_o.insert(h_o.end(), layer_o_size, bfloat16::ZERO);
      }

      if (!lw.gate_proj.empty()) {
        h_gate.insert(h_gate.end(), lw.gate_proj.begin(), lw.gate_proj.end());
      } else {
        h_gate.insert(h_gate.end(), layer_gate_size, bfloat16::ZERO);
      }

      if (!lw.up_proj.empty()) {
        h_up.insert(h_up.end(), lw.up_proj.begin(), lw.up_proj.end());
      } else {
        h_up.insert(h_up.end(), layer_up_size, bfloat16::ZERO);
      }

      if (!lw.down_proj.empty()) {
        h_down.insert(h_down.end(), lw.down_proj.begin(), lw.down_proj.end());
      } else {
        h_down.insert(h_down.end(), layer_down_size, bfloat16::ZERO);
      }
    }

#define ALLOC_COPY_CONCAT_BF16(dev_ptr, host_vec, weight_name_str) \
    SAFE_CUDA_FREE(dev_ptr); \
    if (!host_vec.empty()) { \
        gpuErrchk(cudaMalloc(&dev_ptr, host_vec.size() * sizeof(uint16_t))); \
        gpuErrchk(cudaMemcpy(dev_ptr, host_vec.data(), host_vec.size() * sizeof(uint16_t), cudaMemcpyHostToDevice)); \
        Logger::info("Copied concatenated " weight_name_str " (BF16) to GPU for GPU layers."); \
    } else if (active_num_gpu_layers > 0) { \
        Logger::info("Host vector for concatenated " weight_name_str " (BF16) is empty. Skipping GPU copy."); \
    }

    ALLOC_COPY_CONCAT_BF16(w_q_dev_, h_q, "Q Proj"); ALLOC_COPY_CONCAT_BF16(w_k_dev_, h_k, "K Proj"); ALLOC_COPY_CONCAT_BF16(w_v_dev_, h_v, "V Proj");
    ALLOC_COPY_CONCAT_BF16(w_o_dev_, h_o, "O Proj"); ALLOC_COPY_CONCAT_BF16(w_gate_dev_, h_gate, "Gate Proj"); 
    ALLOC_COPY_CONCAT_BF16(w_up_dev_, h_up, "Up Proj"); ALLOC_COPY_CONCAT_BF16(w_down_dev_, h_down, "Down Proj");
#undef ALLOC_COPY_CONCAT_BF16

  } else {
    Logger::info("Skipping BF16 concatenated layer weight processing (first GPU layer appears not to use BF16 q_proj, or no GPU layers).");
    SAFE_CUDA_FREE(w_q_dev_); SAFE_CUDA_FREE(w_k_dev_); SAFE_CUDA_FREE(w_v_dev_); SAFE_CUDA_FREE(w_o_dev_);
    SAFE_CUDA_FREE(w_gate_dev_); SAFE_CUDA_FREE(w_up_dev_); SAFE_CUDA_FREE(w_down_dev_);
  }

  Logger::info("DEFERRING concatenated F32 weight processing for GPU layers to save memory during initialization");
  Logger::info("Concatenated F32 weights will be processed on-demand during first inference");
  
  SAFE_CUDA_FREE(w_q_f32_dev_); SAFE_CUDA_FREE(w_k_f32_dev_); SAFE_CUDA_FREE(w_v_f32_dev_); SAFE_CUDA_FREE(w_o_f32_dev_);
  SAFE_CUDA_FREE(w_gate_f32_dev_); SAFE_CUDA_FREE(w_up_f32_dev_); SAFE_CUDA_FREE(w_down_f32_dev_);

  Logger::info("Finished deferring concatenated F32 weight processing for GPU layers.");

  // Allocate persistent batch processing buffers for GPU memory optimization
  if (active_num_gpu_layers > 0) {
    allocate_persistent_batch_buffers();
  }

#undef SAFE_CUDA_FREE
#else
  if (active_num_gpu_layers > 0 && nhl > 0) {
      Logger::warning("CUDA not available, but " + std::to_string(active_num_gpu_layers) + " layer(s) were configured for GPU. Model will run entirely on CPU.");
  } else {
      Logger::info("CUDA not available or no GPU layers configured. Model will run entirely on CPU.");
  }
#endif
} 