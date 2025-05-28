#include "model.h"

#include "logger.h"
#ifdef HAS_CUDA
#include "cuda_kernels.h"
#endif
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#ifdef _WIN32
#include <windows.h>
#endif
#include <cassert>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <variant>

#include "gguf_parser.h"
#include "quantization.h"
#include "model_constants.h"
#include "model_macros.h"
#include "safetensors_loader.h"
#include "utils.h"
#include "model_config.h"

void KVCache::initialize(const ModelConfig& config, 
                         int total_num_model_layers, int num_gpu_layers_to_allocate, 
                         int max_seq_len_arg, int num_kv_heads,
                         int head_dim, int max_batch_size_arg) {
  this->total_model_layers_ = total_num_model_layers; // Store for use in other KVCache methods if needed
  this->max_seq_len_config_ = max_seq_len_arg; // Store the passed max_seq_len
  this->max_batch_size = max_batch_size_arg; // Store the max batch size
  this->current_batch_size = 0; // Initialize current batch size
  this->batch_seq_lens.clear(); // Clear batch sequence lengths
  this->batch_seq_lens.resize(max_batch_size_arg, 0); // Initialize with zeros
  layers.resize(total_num_model_layers); // CPU KVCacheLayer vector sized for all layers
  seq_len = 0;
  Logger::info("Allocating KVCache host vectors...");
  size_t cache_size_per_layer = static_cast<size_t>(max_seq_len_arg) *
                                static_cast<size_t>(max_batch_size_arg) *
                                static_cast<size_t>(num_kv_heads) *
                                static_cast<size_t>(head_dim);
  if (cache_size_per_layer == 0 && max_seq_len_arg > 0 && total_num_model_layers > 0) { // Check total_num_model_layers too
    throw std::runtime_error(
        "KVCache (CPU): Calculated cache size is zero for non-empty model. Check parameters.");
  }

  for (int l = 0; l < total_num_model_layers; ++l) { // Allocate CPU part for all model layers
    try {
      layers[l].k.assign(cache_size_per_layer, 0.0f);
      layers[l].v.assign(cache_size_per_layer, 0.0f);
    } catch (const std::bad_alloc& e) {
      Logger::error("Failed to allocate CPU KVCache for layer " +
                    std::to_string(l) + ": " + e.what());
      throw;
    }
  }
  Logger::info("KVCache (CPU) vectors allocated for " +
               std::to_string(total_num_model_layers) + " layers.");

#ifdef HAS_CUDA
  // Store the actual number of layers for which GPU memory will be allocated.
  this->allocated_num_layers = num_gpu_layers_to_allocate; 
  this->allocated_max_seq_len = max_seq_len_arg; // Use max_seq_len_arg
  this->allocated_num_kv_heads = num_kv_heads;
  this->allocated_head_dim = head_dim;

  if (num_gpu_layers_to_allocate > 0) { // Only proceed if GPU layers are requested
      if (num_gpu_layers_to_allocate > total_num_model_layers) {
          Logger::warning("KVCache::initialize: num_gpu_layers_to_allocate (" + std::to_string(num_gpu_layers_to_allocate) +
                          ") > total_num_model_layers (" + std::to_string(total_num_model_layers) + 
                          "). Clamping to total_num_model_layers.");
          this->allocated_num_layers = total_num_model_layers; // Update member
          num_gpu_layers_to_allocate = total_num_model_layers; // Use clamped value for local logic
      }

      size_t cache_elems_per_layer_gpu = static_cast<size_t>(max_seq_len_arg) * // Use max_seq_len_arg
                                 static_cast<size_t>(num_kv_heads) *
                                 static_cast<size_t>(head_dim);
      
      // Sizes for different KVCache types
      size_t fp32_cache_bytes_per_layer_gpu = cache_elems_per_layer_gpu * sizeof(float);
      size_t int8_cache_bytes_per_layer_gpu = cache_elems_per_layer_gpu * sizeof(int8_t);
      // For scales: one scale per head per token position
      size_t num_scales_per_layer_gpu = static_cast<size_t>(max_seq_len_arg) * static_cast<size_t>(num_kv_heads); // Use max_seq_len_arg
      size_t scales_bytes_per_layer_gpu = num_scales_per_layer_gpu * sizeof(float);

      if (cache_elems_per_layer_gpu == 0 && config.use_kvcache_quantization) {
        throw std::runtime_error(
            "KVCache (CUDA INT8): Calculated cache elements per layer is zero. Check parameters.");
      } else if (cache_elems_per_layer_gpu == 0) {
        throw std::runtime_error(
            "KVCache (CUDA FP32): Calculated cache elements per layer is zero. Check parameters.");
      }

      if (config.use_kvcache_quantization) {
        Logger::info("Allocating INT8 KVCache + FP32 Scales on GPU for " + std::to_string(num_gpu_layers_to_allocate) +
                 " layers. Data size per layer: " +
                     std::to_string(int8_cache_bytes_per_layer_gpu / (1024.0 * 1024.0)) +
                 " MB. Scales size per layer: " + 
                     std::to_string(scales_bytes_per_layer_gpu / (1024.0 * 1024.0)) + " MB");
      } else {
        Logger::info("Allocating FP32 KVCache on GPU for " + std::to_string(num_gpu_layers_to_allocate) +
                 " layers, size per layer: " +
                     std::to_string(fp32_cache_bytes_per_layer_gpu / (1024.0 * 1024.0)) +
                 " MB");
      }

      int gpu_layer_start_model_idx = this->total_model_layers_ - num_gpu_layers_to_allocate;
      Logger::info("KVCache GPU allocation will target model layers from index " + std::to_string(gpu_layer_start_model_idx) +
                   " to " + std::to_string(gpu_layer_start_model_idx + num_gpu_layers_to_allocate - 1));

      for (int i = 0; i < num_gpu_layers_to_allocate; ++i) { // Loop 'i' for count
        int current_model_idx_for_gpu = gpu_layer_start_model_idx + i; // Calculate actual model index

        if (current_model_idx_for_gpu < 0 || static_cast<size_t>(current_model_idx_for_gpu) >= layers.size()) {
            Logger::error("KVCache::initialize: Calculated current_model_idx_for_gpu (" + std::to_string(current_model_idx_for_gpu) + ") is out of bounds for layers vector (size " + std::to_string(layers.size()) + "). Skipping this layer.");
            continue;
        }

        if (layers[current_model_idx_for_gpu].k_dev_fp32) {
          Logger::warning(
              "KVCache::initialize: Re-initializing KVCache layer " + std::to_string(current_model_idx_for_gpu) + " K dev fp32 pointer without proper destruction?");
          gpuErrchk(cudaFree(layers[current_model_idx_for_gpu].k_dev_fp32));
          layers[current_model_idx_for_gpu].k_dev_fp32 = nullptr;
        }
        if (layers[current_model_idx_for_gpu].v_dev_fp32) {
          Logger::warning(
              "KVCache::initialize: Re-initializing KVCache layer " + std::to_string(current_model_idx_for_gpu) + " V dev fp32 pointer without proper destruction?");
          gpuErrchk(cudaFree(layers[current_model_idx_for_gpu].v_dev_fp32));
          layers[current_model_idx_for_gpu].v_dev_fp32 = nullptr;
        }
        if (layers[current_model_idx_for_gpu].k_dev_quantized) {
          Logger::warning(
              "KVCache::initialize: Re-initializing KVCache layer " + std::to_string(current_model_idx_for_gpu) + " K dev quantized pointer without proper destruction?");
          gpuErrchk(cudaFree(layers[current_model_idx_for_gpu].k_dev_quantized));
          layers[current_model_idx_for_gpu].k_dev_quantized = nullptr;
        }
        if (layers[current_model_idx_for_gpu].v_dev_quantized) {
          Logger::warning(
              "KVCache::initialize: Re-initializing KVCache layer " + std::to_string(current_model_idx_for_gpu) + " V dev quantized pointer without proper destruction?");
          gpuErrchk(cudaFree(layers[current_model_idx_for_gpu].v_dev_quantized));
          layers[current_model_idx_for_gpu].v_dev_quantized = nullptr;
        }
        if (layers[current_model_idx_for_gpu].k_dev_scales) {
          Logger::warning(
              "KVCache::initialize: Re-initializing KVCache layer " + std::to_string(current_model_idx_for_gpu) + " K dev scales pointer without proper destruction?");
          gpuErrchk(cudaFree(layers[current_model_idx_for_gpu].k_dev_scales));
          layers[current_model_idx_for_gpu].k_dev_scales = nullptr;
        }
        if (layers[current_model_idx_for_gpu].v_dev_scales) {
          Logger::warning(
              "KVCache::initialize: Re-initializing KVCache layer " + std::to_string(current_model_idx_for_gpu) + " V dev scales pointer without proper destruction?");
          gpuErrchk(cudaFree(layers[current_model_idx_for_gpu].v_dev_scales));
          layers[current_model_idx_for_gpu].v_dev_scales = nullptr;
        }
        
        if (config.use_kvcache_quantization) {
            gpuErrchk(cudaMalloc(&layers[current_model_idx_for_gpu].k_dev_quantized, int8_cache_bytes_per_layer_gpu));
            gpuErrchk(cudaMalloc(&layers[current_model_idx_for_gpu].v_dev_quantized, int8_cache_bytes_per_layer_gpu));
            gpuErrchk(cudaMalloc(&layers[current_model_idx_for_gpu].k_dev_scales, scales_bytes_per_layer_gpu));
            gpuErrchk(cudaMalloc(&layers[current_model_idx_for_gpu].v_dev_scales, scales_bytes_per_layer_gpu));

            gpuErrchk(cudaMemset(layers[current_model_idx_for_gpu].k_dev_quantized, 0, int8_cache_bytes_per_layer_gpu));
            gpuErrchk(cudaMemset(layers[current_model_idx_for_gpu].v_dev_quantized, 0, int8_cache_bytes_per_layer_gpu));
            gpuErrchk(cudaMemset(layers[current_model_idx_for_gpu].k_dev_scales, 0, scales_bytes_per_layer_gpu));
            gpuErrchk(cudaMemset(layers[current_model_idx_for_gpu].v_dev_scales, 0, scales_bytes_per_layer_gpu));
        } else {
            gpuErrchk(cudaMalloc(&layers[current_model_idx_for_gpu].k_dev_fp32, fp32_cache_bytes_per_layer_gpu));
            gpuErrchk(cudaMalloc(&layers[current_model_idx_for_gpu].v_dev_fp32, fp32_cache_bytes_per_layer_gpu));
            gpuErrchk(cudaMemset(layers[current_model_idx_for_gpu].k_dev_fp32, 0, fp32_cache_bytes_per_layer_gpu));
            gpuErrchk(cudaMemset(layers[current_model_idx_for_gpu].v_dev_fp32, 0, fp32_cache_bytes_per_layer_gpu));
        }
  }
      Logger::info("KVCache GPU allocation and zeroing complete for " + std::to_string(num_gpu_layers_to_allocate) + " layers.");
  } else {
      Logger::info("KVCache: No GPU layers requested for allocation (num_gpu_layers_to_allocate is 0). Skipping GPU KVCache allocation.");
      this->allocated_num_layers = 0; 
  }

#else
  Logger::info("KVCache (CPU-only build) initialized with dimensions for " +
               std::to_string(total_num_model_layers) + " layers, " +
               std::to_string(max_seq_len_arg) + " seq len, " // Use max_seq_len_arg
               std::to_string(num_kv_heads) + " KV heads, " +
               std::to_string(head_dim) + " head dim");
#endif
}





void TinyLlamaModel::ensure_embed_tokens_dequantized() {
    if (!this->embed_tokens_f32.empty()) return;
    
    size_t total_elements_embed = static_cast<size_t>(config_.vocab_size) * config_.hidden_size;
    if (!this->embed_tokens_q6k.empty()) {
        dequantize_vector_q6k_to_f32(this->embed_tokens_q6k, this->embed_tokens_f32, total_elements_embed, 1);
    } else if (!this->embed_tokens_q4k.empty()) {
        dequantize_vector_q4k_to_f32(this->embed_tokens_q4k, this->embed_tokens_f32, total_elements_embed, 1);
    } else if (!this->embed_tokens_q8k.empty()) {
        dequantize_q8_k(this->embed_tokens_q8k, this->embed_tokens_f32, total_elements_embed, true);
    } else if (!this->embed_tokens_q8_0.empty()) {
        dequantize_vector_q8_0_to_f32(this->embed_tokens_q8_0, this->embed_tokens_f32, total_elements_embed, 1);
    } else if (!this->embed_tokens.empty()) { 
        this->embed_tokens_f32 = bf16vec_to_float_vec(this->embed_tokens);
    }
}
void TinyLlamaModel::initialize_weights(const SafeTensorsLoader* loader,
                                        const GGUFData* gguf) {
  Logger::info("Initializing model weights...");
  int hs = config_.hidden_size;
  int is = config_.intermediate_size;
  int nhl = config_.num_hidden_layers;
  int vs = config_.vocab_size;
  layers.resize(nhl);

  if (gguf) {
    Logger::info("Processing weights from GGUF data source...");

    if (gguf && (gguf->mapped_tensor_data || !gguf->tensor_data.empty())) {
      map_gguf_weights(*gguf, *this);
      Logger::info("[INIT_WEIGHTS_GGUF] map_gguf_weights(*gguf, *this) CALLED (using function parameter).");
    } else if (gguf_data_ && (gguf_data_->mapped_tensor_data || !gguf_data_->tensor_data.empty())) {
      map_gguf_weights(*gguf_data_, *this);
      Logger::info("[INIT_WEIGHTS_GGUF] map_gguf_weights(*gguf_data_, *this) CALLED (using member gguf_data_).");
    } else {
      Logger::error("[INIT_WEIGHTS_GGUF] map_gguf_weights failed - tensor data not available. No GGUF weights mapped.");
    }

    // LAZY DEQUANTIZATION: Only dequantize what's immediately needed
    Logger::info("[INIT_WEIGHTS_GGUF] Using lazy dequantization to prevent OOM");

    // Only dequantize embed_tokens and final_norm immediately (small and always needed)
    if (this->embed_tokens_f32.empty()) {
      size_t total_elements_embed = static_cast<size_t>(config_.vocab_size) * config_.hidden_size;
      if (!this->embed_tokens_q6k.empty()) {
        dequantize_vector_q6k_to_f32(this->embed_tokens_q6k, this->embed_tokens_f32, total_elements_embed, 1);
      } else if (!this->embed_tokens_q4k.empty()) {
        dequantize_vector_q4k_to_f32(this->embed_tokens_q4k, this->embed_tokens_f32, total_elements_embed, 1);
      } else if (!this->embed_tokens_q8k.empty()) {
        dequantize_q8_k(this->embed_tokens_q8k, this->embed_tokens_f32, total_elements_embed, true);
      } else if (!this->embed_tokens_q8_0.empty()) {
        dequantize_vector_q8_0_to_f32(this->embed_tokens_q8_0, this->embed_tokens_f32, total_elements_embed, 1);
      } else if (!this->embed_tokens.empty()) { 
        this->embed_tokens_f32 = bf16vec_to_float_vec(this->embed_tokens);
      }
      if (!this->embed_tokens_f32.empty()) {
        Logger::info("[INIT_WEIGHTS_GGUF_DEQUANT] embed_tokens_f32 populated. Size: " + std::to_string(this->embed_tokens_f32.size()));
      }
    }

    if (this->final_norm_f32.empty()) {
        if (!this->final_norm.empty()) { 
            this->final_norm_f32 = bf16vec_to_float_vec(this->final_norm);
            if (!this->final_norm_f32.empty()) {
                Logger::info("[INIT_WEIGHTS_GGUF_DEQUANT] Successfully converted final_norm (BF16) to final_norm_f32. Size: " + std::to_string(this->final_norm_f32.size()));
            }
        }
    }

    // DEFER lm_head dequantization until actually needed (it's huge)
    Logger::info("[INIT_WEIGHTS_GGUF] Deferring lm_head dequantization until needed to save memory");

    // DEFER all layer weight dequantization until the layer is actually used
    Logger::info("[INIT_WEIGHTS_GGUF] Deferring all layer weight dequantization until layers are used");

    // Only populate layer norms immediately (small and needed for validation)
    for (int l = 0; l < nhl; ++l) {
        auto& lw = layers[l];
        
        if (lw.input_layernorm_f32.empty() && !lw.input_layernorm.empty()) {
            lw.input_layernorm_f32 = bf16vec_to_float_vec(lw.input_layernorm);
            if (!lw.input_layernorm_f32.empty()) Logger::info("  L" + std::to_string(l) + " input_layernorm_f32 populated from BF16. Size: " + std::to_string(lw.input_layernorm_f32.size()));
        }
        if (lw.post_attention_layernorm_f32.empty() && !lw.post_attention_layernorm.empty()) {
            lw.post_attention_layernorm_f32 = bf16vec_to_float_vec(lw.post_attention_layernorm);
            if (!lw.post_attention_layernorm_f32.empty()) Logger::info("  L" + std::to_string(l) + " post_attention_layernorm_f32 populated from BF16. Size: " + std::to_string(lw.post_attention_layernorm_f32.size()));
        }
    }

    // Validation checks for layer norms
    for (int l = 0; l < nhl; ++l) {
        const auto& lw = layers[l]; 
        if (lw.input_layernorm_f32.empty()) {
            Logger::error("[INIT_WEIGHTS_GGUF_CHECK] Layer " + std::to_string(l) + 
                          ": input_layernorm_f32 is EMPTY post-GGUF. This WILL cause GPU init errors if this layer is on GPU.");
        }
        if (lw.post_attention_layernorm_f32.empty()) {
            Logger::error("[INIT_WEIGHTS_GGUF_CHECK] Layer " + std::to_string(l) + 
                          ": post_attention_layernorm_f32 is EMPTY post-GGUF. This WILL cause GPU init errors if this layer is on GPU.");
        }
    }
    Logger::info("[INIT_WEIGHTS_GGUF] Finished per-layer NORM F32 vector checks post-GGUF.");

  } else {
    Logger::fatal("TinyLlamaModel::initialize_weights called with neither GGUF nor SafeTensors loader. Cannot initialize weights.");
    throw std::runtime_error("Model weights source (GGUF or SafeTensors) not provided to initialize_weights.");
  }

  Logger::info("Finished initializing model weights logic block.");

  if (this->final_norm_f32.empty()) {
      Logger::error("[INIT_WEIGHTS_FINAL_CHECK] final_norm_f32 is EMPTY. This WILL cause errors if final normalization is needed in F32.");
  } else {
      Logger::info("[INIT_WEIGHTS_FINAL_CHECK] final_norm_f32 is POPULATED. Size: " + std::to_string(this->final_norm_f32.size()));
  }

  if (this->embed_tokens_f32.empty()) {
    Logger::error("[INIT_WEIGHTS_FINAL_CHECK] embed_tokens_f32 is EMPTY. This WILL cause errors if token embeddings are needed in F32.");
  } else {
    Logger::info("[INIT_WEIGHTS_FINAL_CHECK] embed_tokens_f32 is POPULATED. Size: " + std::to_string(this->embed_tokens_f32.size()));
  }
}




void TinyLlamaModel::ensure_f32_concatenated_weights_loaded() {
  // OPTIMIZED: Use concatenated weights for maximum GPU performance
  Logger::info("Loading concatenated F32 weights for optimal GPU performance");
  
  int active_num_gpu_layers = config_.num_hidden_layers - config_.num_cpu_offload_layers;
  if (f32_concatenated_weights_loaded_ || active_num_gpu_layers == 0) {
    return;
  }

  Logger::info("Loading F32 concatenated weights on-demand for GPU inference");

  // Concatenated FP32 / Dequantized Layer Weights for GPU layers (w_..._f32_dev_ pointers)
  int hs = config_.hidden_size;
  int is = config_.intermediate_size;
  int kv_dim = (hs / config_.num_attention_heads) * config_.num_key_value_heads;
  
  size_t layer_q_size_f32 = (size_t)hs*hs, layer_k_size_f32 = (size_t)kv_dim*hs, layer_v_size_f32 = (size_t)kv_dim*hs;
  size_t layer_o_size_f32 = (size_t)hs*hs, layer_gate_size_f32 = (size_t)is*hs, layer_up_size_f32 = (size_t)is*hs;
  size_t layer_down_size_f32 = (size_t)hs*is;

  auto process_gpu_layer_weights_to_f32_concat = [&] (
      const std::function<const std::vector<float>&(const LayerWeights&)>& f32_accessor,
      const std::function<const std::vector<uint16_t>&(const LayerWeights&)>& bf16_accessor,
      const std::function<const std::vector<block_q8_0>&(const LayerWeights&)>& q8_accessor,
      const std::function<const std::vector<block_q4_K>&(const LayerWeights&)>& q4k_accessor,
      const std::function<const std::vector<block_q6_K>&(const LayerWeights&)>& q6k_accessor,
      float*& dev_ptr, size_t single_layer_elem_size, const std::string& weight_name) {

    std::vector<float> concatenated_f32;
    concatenated_f32.reserve(active_num_gpu_layers * single_layer_elem_size);

    for (int l_gpu_idx = 0; l_gpu_idx < active_num_gpu_layers; ++l_gpu_idx) {
      int l_model_idx = config_.num_cpu_offload_layers + l_gpu_idx;
      // Individual weight dequantization will be handled by the lambda accessors below
      const LayerWeights& lw = layers[l_model_idx];

      const auto& f32_vec = f32_accessor(lw);
      const auto& bf16_vec = bf16_accessor(lw);
      const auto& q8_vec = q8_accessor(lw);
      const auto& q4k_vec = q4k_accessor(lw);
      const auto& q6k_vec = q6k_accessor(lw);

      if (!f32_vec.empty()) {
        concatenated_f32.insert(concatenated_f32.end(), f32_vec.begin(), f32_vec.end());
      } else if (!bf16_vec.empty()) {
        for (uint16_t bf16_val : bf16_vec) {
          concatenated_f32.push_back(bfloat16_to_float32(bf16_val));
        }
      } else if (!q8_vec.empty()) {
        std::vector<float> temp_f32(q8_vec.size() * GGML_QK8_0);
        for (size_t i = 0; i < q8_vec.size(); ++i) {
          dequantize_q8_0_block(&q8_vec[i], &temp_f32[i * GGML_QK8_0]);
        }
        concatenated_f32.insert(concatenated_f32.end(), temp_f32.begin(), temp_f32.end());
      } else if (!q4k_vec.empty()) {
        std::vector<float> temp_f32(q4k_vec.size() * GGML_QK_K);
        for (size_t i = 0; i < q4k_vec.size(); ++i) {
          dequantize_q4_k_m(&q4k_vec[i], &temp_f32[i * GGML_QK_K], GGML_QK_K);
        }
        concatenated_f32.insert(concatenated_f32.end(), temp_f32.begin(), temp_f32.end());
      } else if (!q6k_vec.empty()) {
        std::vector<float> temp_f32(q6k_vec.size() * GGML_QK_K);
        for (size_t i = 0; i < q6k_vec.size(); ++i) {
          dequantize_q6_k(&q6k_vec[i], &temp_f32[i * GGML_QK_K], GGML_QK_K);
        }
        concatenated_f32.insert(concatenated_f32.end(), temp_f32.begin(), temp_f32.end());
      } else {
        throw std::runtime_error("Layer " + std::to_string(l_model_idx) + ": No " + weight_name + " weights found for GPU processing");
      }
    }

    if (!concatenated_f32.empty()) {
      if (dev_ptr) { cudaFree(dev_ptr); dev_ptr = nullptr; }
      gpuErrchk(cudaMalloc(&dev_ptr, concatenated_f32.size() * sizeof(float)));
      gpuErrchk(cudaMemcpy(dev_ptr, concatenated_f32.data(), concatenated_f32.size() * sizeof(float), cudaMemcpyHostToDevice));
      Logger::info("Loaded concatenated " + weight_name + " (F32) to GPU for " + std::to_string(active_num_gpu_layers) + " layers");
    }
  };

  process_gpu_layer_weights_to_f32_concat(
    [this](const LayerWeights& lw) -> const std::vector<float>& { 
      int layer_idx = &lw - &layers[0]; 
      const_cast<TinyLlamaModel*>(this)->ensure_q_proj_dequantized(layer_idx); 
      return lw.q_proj_f32; 
    },
    [](const LayerWeights& lw) -> const std::vector<uint16_t>& { return lw.q_proj; },
    [](const LayerWeights& lw) -> const std::vector<block_q8_0>& { return lw.q_proj_q8_0; },
    [](const LayerWeights& lw) -> const std::vector<block_q4_K>& { return lw.q_proj_q4k; },
    [](const LayerWeights& lw) -> const std::vector<block_q6_K>& { return lw.q_proj_q6k; },
    w_q_f32_dev_, layer_q_size_f32, "Q Proj");
  
  // CRITICAL: Immediately clear Q-proj CPU memory after GPU upload to prevent OOM
  if (config_.enable_memory_efficient_layers) {
    int first_gpu_layer = config_.num_cpu_offload_layers;
    int last_gpu_layer = config_.num_hidden_layers - 1;
    for (int layer_idx = first_gpu_layer; layer_idx <= last_gpu_layer; ++layer_idx) {
      layers[layer_idx].q_proj_f32.clear();
      layers[layer_idx].q_proj_f32.shrink_to_fit();
    }
    Logger::info("Cleared Q-proj CPU memory immediately after GPU upload");
  }

  process_gpu_layer_weights_to_f32_concat(
    [this](const LayerWeights& lw) -> const std::vector<float>& { 
      int layer_idx = &lw - &layers[0]; 
      const_cast<TinyLlamaModel*>(this)->ensure_k_proj_dequantized(layer_idx); 
      return lw.k_proj_f32; 
    },
    [](const LayerWeights& lw) -> const std::vector<uint16_t>& { return lw.k_proj; },
    [](const LayerWeights& lw) -> const std::vector<block_q8_0>& { return lw.k_proj_q8_0; },
    [](const LayerWeights& lw) -> const std::vector<block_q4_K>& { return lw.k_proj_q4k; },
    [](const LayerWeights& lw) -> const std::vector<block_q6_K>& { return lw.k_proj_q6k; },
    w_k_f32_dev_, layer_k_size_f32, "K Proj");
  
  if (config_.enable_memory_efficient_layers) {
    int first_gpu_layer = config_.num_cpu_offload_layers;
    int last_gpu_layer = config_.num_hidden_layers - 1;
    for (int layer_idx = first_gpu_layer; layer_idx <= last_gpu_layer; ++layer_idx) {
      layers[layer_idx].k_proj_f32.clear();
      layers[layer_idx].k_proj_f32.shrink_to_fit();
    }
    Logger::info("Cleared K-proj CPU memory immediately after GPU upload");
  }

  process_gpu_layer_weights_to_f32_concat(
    [this](const LayerWeights& lw) -> const std::vector<float>& { 
      int layer_idx = &lw - &layers[0]; 
      const_cast<TinyLlamaModel*>(this)->ensure_v_proj_dequantized(layer_idx); 
      return lw.v_proj_f32; 
    },
    [](const LayerWeights& lw) -> const std::vector<uint16_t>& { return lw.v_proj; },
    [](const LayerWeights& lw) -> const std::vector<block_q8_0>& { return lw.v_proj_q8_0; },
    [](const LayerWeights& lw) -> const std::vector<block_q4_K>& { return lw.v_proj_q4k; },
    [](const LayerWeights& lw) -> const std::vector<block_q6_K>& { return lw.v_proj_q6k; },
    w_v_f32_dev_, layer_v_size_f32, "V Proj");
  
  // Clear V-proj CPU memory immediately
  if (config_.enable_memory_efficient_layers) {
    int first_gpu_layer = config_.num_cpu_offload_layers;
    int last_gpu_layer = config_.num_hidden_layers - 1;
    for (int layer_idx = first_gpu_layer; layer_idx <= last_gpu_layer; ++layer_idx) {
      layers[layer_idx].v_proj_f32.clear();
      layers[layer_idx].v_proj_f32.shrink_to_fit();
    }
  }

  process_gpu_layer_weights_to_f32_concat(
    [this](const LayerWeights& lw) -> const std::vector<float>& { 
      int layer_idx = &lw - &layers[0]; 
      const_cast<TinyLlamaModel*>(this)->ensure_o_proj_dequantized(layer_idx); 
      return lw.o_proj_f32; 
    },
    [](const LayerWeights& lw) -> const std::vector<uint16_t>& { return lw.o_proj; },
    [](const LayerWeights& lw) -> const std::vector<block_q8_0>& { return lw.o_proj_q8_0; },
    [](const LayerWeights& lw) -> const std::vector<block_q4_K>& { return lw.o_proj_q4k; },
    [](const LayerWeights& lw) -> const std::vector<block_q6_K>& { return lw.o_proj_q6k; },
    w_o_f32_dev_, layer_o_size_f32, "O Proj");
  
  // Clear O-proj CPU memory immediately  
  if (config_.enable_memory_efficient_layers) {
    int first_gpu_layer = config_.num_cpu_offload_layers;
    int last_gpu_layer = config_.num_hidden_layers - 1;
    for (int layer_idx = first_gpu_layer; layer_idx <= last_gpu_layer; ++layer_idx) {
      layers[layer_idx].o_proj_f32.clear();
      layers[layer_idx].o_proj_f32.shrink_to_fit();
    }
  }

  process_gpu_layer_weights_to_f32_concat(
    [this](const LayerWeights& lw) -> const std::vector<float>& { 
      int layer_idx = &lw - &layers[0]; 
      const_cast<TinyLlamaModel*>(this)->ensure_gate_proj_dequantized(layer_idx); 
      return lw.gate_proj_f32; 
    },
    [](const LayerWeights& lw) -> const std::vector<uint16_t>& { return lw.gate_proj; },
    [](const LayerWeights& lw) -> const std::vector<block_q8_0>& { return lw.gate_proj_q8_0; },
    [](const LayerWeights& lw) -> const std::vector<block_q4_K>& { return lw.gate_proj_q4k; },
    [](const LayerWeights& lw) -> const std::vector<block_q6_K>& { return lw.gate_proj_q6k; },
    w_gate_f32_dev_, layer_gate_size_f32, "Gate Proj");
  
  // Clear Gate-proj CPU memory immediately
  if (config_.enable_memory_efficient_layers) {
    int first_gpu_layer = config_.num_cpu_offload_layers;
    int last_gpu_layer = config_.num_hidden_layers - 1;
    for (int layer_idx = first_gpu_layer; layer_idx <= last_gpu_layer; ++layer_idx) {
      layers[layer_idx].gate_proj_f32.clear();
      layers[layer_idx].gate_proj_f32.shrink_to_fit();
    }
  }

  process_gpu_layer_weights_to_f32_concat(
    [this](const LayerWeights& lw) -> const std::vector<float>& { 
      int layer_idx = &lw - &layers[0]; 
      const_cast<TinyLlamaModel*>(this)->ensure_up_proj_dequantized(layer_idx); 
      return lw.up_proj_f32; 
    },
    [](const LayerWeights& lw) -> const std::vector<uint16_t>& { return lw.up_proj; },
    [](const LayerWeights& lw) -> const std::vector<block_q8_0>& { return lw.up_proj_q8_0; },
    [](const LayerWeights& lw) -> const std::vector<block_q4_K>& { return lw.up_proj_q4k; },
    [](const LayerWeights& lw) -> const std::vector<block_q6_K>& { return lw.up_proj_q6k; },
    w_up_f32_dev_, layer_up_size_f32, "Up Proj");
  
  // Clear Up-proj CPU memory immediately
  if (config_.enable_memory_efficient_layers) {
    int first_gpu_layer = config_.num_cpu_offload_layers;
    int last_gpu_layer = config_.num_hidden_layers - 1;
    for (int layer_idx = first_gpu_layer; layer_idx <= last_gpu_layer; ++layer_idx) {
      layers[layer_idx].up_proj_f32.clear();
      layers[layer_idx].up_proj_f32.shrink_to_fit();
    }
  }

  process_gpu_layer_weights_to_f32_concat(
    [this](const LayerWeights& lw) -> const std::vector<float>& { 
      int layer_idx = &lw - &layers[0]; 
      const_cast<TinyLlamaModel*>(this)->ensure_down_proj_dequantized(layer_idx); 
      return lw.down_proj_f32; 
    },
    [](const LayerWeights& lw) -> const std::vector<uint16_t>& { return lw.down_proj; },
    [](const LayerWeights& lw) -> const std::vector<block_q8_0>& { return lw.down_proj_q8_0; },
    [](const LayerWeights& lw) -> const std::vector<block_q4_K>& { return lw.down_proj_q4k; },
    [](const LayerWeights& lw) -> const std::vector<block_q6_K>& { return lw.down_proj_q6k; },
    w_down_f32_dev_, layer_down_size_f32, "Down Proj");
  
  // Clear Down-proj CPU memory immediately
  if (config_.enable_memory_efficient_layers) {
    int first_gpu_layer = config_.num_cpu_offload_layers;
    int last_gpu_layer = config_.num_hidden_layers - 1;
    for (int layer_idx = first_gpu_layer; layer_idx <= last_gpu_layer; ++layer_idx) {
      layers[layer_idx].down_proj_f32.clear();
      layers[layer_idx].down_proj_f32.shrink_to_fit();
    }
  }

  f32_concatenated_weights_loaded_ = true;
  Logger::info("F32 concatenated weights loaded successfully on-demand with immediate memory cleanup");
}

void TinyLlamaModel::ensure_layer_weights_dequantized(int layer_idx) {
    Logger::warning("[DEPRECATED] ensure_layer_weights_dequantized called for layer " + std::to_string(layer_idx) + ". This function is deprecated to prevent OOM. Use individual ensure_*_dequantized functions instead.");
}

void TinyLlamaModel::ensure_q_proj_dequantized(int layer_idx) {
    if (layer_idx < 0 || layer_idx >= layers.size()) return;
    auto& lw = layers[layer_idx];
    if (!lw.q_proj_f32.empty()) return;
    
    int hs = config_.hidden_size;
    size_t q_proj_elements = static_cast<size_t>(hs) * hs;
    Logger::info("[DEQUANT_MEM] Layer " + std::to_string(layer_idx) + ": Q-proj dequantization starting, elements=" + std::to_string(q_proj_elements));
    
    if (!lw.q_proj_q6k.empty()) dequantize_vector_q6k_to_f32(lw.q_proj_q6k, lw.q_proj_f32, q_proj_elements, 0);
    else if (!lw.q_proj_q4k.empty()) dequantize_vector_q4k_to_f32(lw.q_proj_q4k, lw.q_proj_f32, q_proj_elements, 0);
    else if (!lw.q_proj_q8k.empty()) dequantize_q8_k(lw.q_proj_q8k, lw.q_proj_f32, q_proj_elements, true);
    else if (!lw.q_proj_q8_0.empty()) dequantize_vector_q8_0_to_f32(lw.q_proj_q8_0, lw.q_proj_f32, q_proj_elements, 0);
    else if (!lw.q_proj.empty()) lw.q_proj_f32 = bf16vec_to_float_vec(lw.q_proj);
    
    Logger::info("[DEQUANT_MEM] Layer " + std::to_string(layer_idx) + ": Q-proj dequantization completed, f32 size=" + std::to_string(lw.q_proj_f32.size()));
}

void TinyLlamaModel::clear_layer_dequantized_weights(int layer_idx) {
    if (layer_idx < 0 || layer_idx >= layers.size()) return;
    auto& lw = layers[layer_idx];
    
    // Clear all dequantized f32 weight vectors for this layer
    lw.q_proj_f32.clear();
    lw.q_proj_f32.shrink_to_fit();
    
    lw.k_proj_f32.clear();
    lw.k_proj_f32.shrink_to_fit();
    
    lw.v_proj_f32.clear();
    lw.v_proj_f32.shrink_to_fit();
    
    lw.o_proj_f32.clear();
    lw.o_proj_f32.shrink_to_fit();
    
    lw.gate_proj_f32.clear();
    lw.gate_proj_f32.shrink_to_fit();
    
    lw.up_proj_f32.clear();
    lw.up_proj_f32.shrink_to_fit();
    
    lw.down_proj_f32.clear();
    lw.down_proj_f32.shrink_to_fit();
    
    Logger::info("[MEMORY_EVICT] Layer " + std::to_string(layer_idx) + ": Cleared all dequantized f32 weights");
}

void TinyLlamaModel::ensure_k_proj_dequantized(int layer_idx) {
    if (layer_idx < 0 || layer_idx >= layers.size()) return;
    auto& lw = layers[layer_idx];
    if (!lw.k_proj_f32.empty()) return;
    
    int hs = config_.hidden_size;
    size_t k_proj_elements = static_cast<size_t>(config_.num_key_value_heads * (hs / config_.num_attention_heads)) * hs;
    
    if (!lw.k_proj_q6k.empty()) dequantize_vector_q6k_to_f32(lw.k_proj_q6k, lw.k_proj_f32, k_proj_elements, 0);
    else if (!lw.k_proj_q4k.empty()) dequantize_vector_q4k_to_f32(lw.k_proj_q4k, lw.k_proj_f32, k_proj_elements, 0);
    else if (!lw.k_proj_q8k.empty()) dequantize_q8_k(lw.k_proj_q8k, lw.k_proj_f32, k_proj_elements, false);
    else if (!lw.k_proj_q8_0.empty()) dequantize_vector_q8_0_to_f32(lw.k_proj_q8_0, lw.k_proj_f32, k_proj_elements, 0);
    else if (!lw.k_proj.empty()) lw.k_proj_f32 = bf16vec_to_float_vec(lw.k_proj);
}

void TinyLlamaModel::ensure_v_proj_dequantized(int layer_idx) {
    if (layer_idx < 0 || layer_idx >= layers.size()) return;
    auto& lw = layers[layer_idx];
    if (!lw.v_proj_f32.empty()) return;
    
    int hs = config_.hidden_size;
    size_t v_proj_elements = static_cast<size_t>(config_.num_key_value_heads * (hs / config_.num_attention_heads)) * hs;
    
    if (!lw.v_proj_q6k.empty()) dequantize_vector_q6k_to_f32(lw.v_proj_q6k, lw.v_proj_f32, v_proj_elements, 0);
    else if (!lw.v_proj_q4k.empty()) dequantize_vector_q4k_to_f32(lw.v_proj_q4k, lw.v_proj_f32, v_proj_elements, 0);
    else if (!lw.v_proj_q8k.empty()) dequantize_q8_k(lw.v_proj_q8k, lw.v_proj_f32, v_proj_elements, false);
    else if (!lw.v_proj_q8_0.empty()) dequantize_vector_q8_0_to_f32(lw.v_proj_q8_0, lw.v_proj_f32, v_proj_elements, 0);
    else if (!lw.v_proj.empty()) lw.v_proj_f32 = bf16vec_to_float_vec(lw.v_proj);
}

void TinyLlamaModel::ensure_o_proj_dequantized(int layer_idx) {
    if (layer_idx < 0 || layer_idx >= layers.size()) return;
    auto& lw = layers[layer_idx];
    if (!lw.o_proj_f32.empty()) return;
    
    int hs = config_.hidden_size;
    size_t o_proj_elements = static_cast<size_t>(hs) * hs;
    
    if (!lw.o_proj_q6k.empty()) dequantize_vector_q6k_to_f32(lw.o_proj_q6k, lw.o_proj_f32, o_proj_elements, 0);
    else if (!lw.o_proj_q4k.empty()) dequantize_vector_q4k_to_f32(lw.o_proj_q4k, lw.o_proj_f32, o_proj_elements, 0);
    else if (!lw.o_proj_q8k.empty()) dequantize_q8_k(lw.o_proj_q8k, lw.o_proj_f32, o_proj_elements, false);
    else if (!lw.o_proj_q8_0.empty()) dequantize_vector_q8_0_to_f32(lw.o_proj_q8_0, lw.o_proj_f32, o_proj_elements, 0);
    else if (!lw.o_proj.empty()) lw.o_proj_f32 = bf16vec_to_float_vec(lw.o_proj);
}

void TinyLlamaModel::ensure_gate_proj_dequantized(int layer_idx) {
    if (layer_idx < 0 || layer_idx >= layers.size()) return;
    auto& lw = layers[layer_idx];
    if (!lw.gate_proj_f32.empty()) return;
    
    int hs = config_.hidden_size;
    int is = config_.intermediate_size;
    size_t gate_proj_elements = static_cast<size_t>(is) * hs;
    
    if (!lw.gate_proj_q6k.empty()) dequantize_vector_q6k_to_f32(lw.gate_proj_q6k, lw.gate_proj_f32, gate_proj_elements, 0);
    else if (!lw.gate_proj_q4k.empty()) dequantize_vector_q4k_to_f32(lw.gate_proj_q4k, lw.gate_proj_f32, gate_proj_elements, 0);
    else if (!lw.gate_proj_q8k.empty()) dequantize_q8_k(lw.gate_proj_q8k, lw.gate_proj_f32, gate_proj_elements, false);
    else if (!lw.gate_proj_q8_0.empty()) dequantize_vector_q8_0_to_f32(lw.gate_proj_q8_0, lw.gate_proj_f32, gate_proj_elements, 0);
    else if (!lw.gate_proj.empty()) lw.gate_proj_f32 = bf16vec_to_float_vec(lw.gate_proj);
}

void TinyLlamaModel::ensure_up_proj_dequantized(int layer_idx) {
    if (layer_idx < 0 || layer_idx >= layers.size()) return;
    auto& lw = layers[layer_idx];
    if (!lw.up_proj_f32.empty()) return;
    
    int hs = config_.hidden_size;
    int is = config_.intermediate_size;
    size_t up_proj_elements = static_cast<size_t>(is) * hs;
    
    if (!lw.up_proj_q6k.empty()) dequantize_vector_q6k_to_f32(lw.up_proj_q6k, lw.up_proj_f32, up_proj_elements, 0);
    else if (!lw.up_proj_q4k.empty()) dequantize_vector_q4k_to_f32(lw.up_proj_q4k, lw.up_proj_f32, up_proj_elements, 0);
    else if (!lw.up_proj_q8k.empty()) dequantize_q8_k(lw.up_proj_q8k, lw.up_proj_f32, up_proj_elements, false);
    else if (!lw.up_proj_q8_0.empty()) dequantize_vector_q8_0_to_f32(lw.up_proj_q8_0, lw.up_proj_f32, up_proj_elements, 0);
    else if (!lw.up_proj.empty()) lw.up_proj_f32 = bf16vec_to_float_vec(lw.up_proj);
}

void TinyLlamaModel::ensure_down_proj_dequantized(int layer_idx) {
    if (layer_idx < 0 || layer_idx >= layers.size()) return;
    auto& lw = layers[layer_idx];
    if (!lw.down_proj_f32.empty()) return;
    
    int hs = config_.hidden_size;
    int is = config_.intermediate_size;
    size_t down_proj_elements = static_cast<size_t>(hs) * is;
    
    if (!lw.down_proj_q6k.empty()) dequantize_vector_q6k_to_f32(lw.down_proj_q6k, lw.down_proj_f32, down_proj_elements, 0);
    else if (!lw.down_proj_q4k.empty()) dequantize_vector_q4k_to_f32(lw.down_proj_q4k, lw.down_proj_f32, down_proj_elements, 0);
    else if (!lw.down_proj_q8k.empty()) dequantize_q8_k(lw.down_proj_q8k, lw.down_proj_f32, down_proj_elements, false);
    else if (!lw.down_proj_q8_0.empty()) dequantize_vector_q8_0_to_f32(lw.down_proj_q8_0, lw.down_proj_f32, down_proj_elements, 0);
    else if (!lw.down_proj.empty()) lw.down_proj_f32 = bf16vec_to_float_vec(lw.down_proj);
}

void TinyLlamaModel::ensure_layer_weights_on_gpu(int layer_idx) {
#ifdef HAS_CUDA
  if (layer_idx < 0 || layer_idx >= layers.size()) return;
  
  // Check if this layer is supposed to be on GPU
  int first_gpu_layer = config_.num_cpu_offload_layers;
  int last_gpu_layer = config_.num_hidden_layers - 1;
  if (layer_idx < first_gpu_layer || layer_idx > last_gpu_layer) return;
  
  LayerWeights& lw = layers[layer_idx];
  
  // Check if weights are already loaded on GPU for this layer
  if (lw.q_proj_f32_dev && lw.k_proj_f32_dev && lw.v_proj_f32_dev && 
      lw.o_proj_f32_dev && lw.gate_proj_f32_dev && lw.up_proj_f32_dev && lw.down_proj_f32_dev) {
    return; // Already loaded
  }
  
  // AGGRESSIVE MEMORY MANAGEMENT: Free previous layer weights to make room
  // Keep only the current layer and maybe the next one
  if (layer_idx > first_gpu_layer) {
    int prev_layer = layer_idx - 1;
    if (prev_layer >= first_gpu_layer && prev_layer < layers.size()) {
      free_layer_gpu_weights(prev_layer);
    }
  }
  
  // If still hitting memory limits, free ALL other GPU layers except current
  if (layer_idx > first_gpu_layer + 1) {
    for (int i = first_gpu_layer; i < layer_idx - 1; ++i) {
      free_layer_gpu_weights(i);
    }
  }
  
  Logger::info("JIT loading layer " + std::to_string(layer_idx) + " weights to GPU (with aggressive eviction)");
  
  // Dequantize and load each weight matrix individually
  auto load_single_weight = [&](
    const std::function<void()>& ensure_dequantized,
    const std::function<const std::vector<float>&()>& get_f32_weights,
    float*& dev_ptr,
    const std::string& weight_name
  ) {
    ensure_dequantized();
    const auto& f32_weights = get_f32_weights();
    if (!f32_weights.empty()) {
      if (dev_ptr) { cudaFree(dev_ptr); dev_ptr = nullptr; }
      
      // Try allocation with error handling
      cudaError_t malloc_result = cudaMalloc(&dev_ptr, f32_weights.size() * sizeof(float));
      if (malloc_result != cudaSuccess) {
        Logger::warning("GPU memory allocation failed for " + weight_name + " in layer " + std::to_string(layer_idx) + 
                       ". Attempting emergency cleanup...");
        
        // Emergency cleanup: free ALL other layer weights
        for (int emergency_idx = first_gpu_layer; emergency_idx <= last_gpu_layer; ++emergency_idx) {
          if (emergency_idx != layer_idx) {
            free_layer_gpu_weights(emergency_idx);
          }
        }
        
        // Try allocation again after cleanup
        malloc_result = cudaMalloc(&dev_ptr, f32_weights.size() * sizeof(float));
        if (malloc_result != cudaSuccess) {
          throw std::runtime_error("GPU OOM: Cannot allocate " + std::to_string(f32_weights.size() * sizeof(float)) + 
                                   " bytes for " + weight_name + " in layer " + std::to_string(layer_idx) + 
                                   " even after emergency cleanup. Try reducing --n-gpu-layers.");
        }
        Logger::info("Emergency cleanup successful, allocated " + weight_name);
      }
      
      gpuErrchk(cudaMemcpy(dev_ptr, f32_weights.data(), f32_weights.size() * sizeof(float), cudaMemcpyHostToDevice));
      
      // Immediately clear CPU memory to save RAM
      if (config_.enable_memory_efficient_layers) {
        const_cast<std::vector<float>&>(f32_weights).clear();
        const_cast<std::vector<float>&>(f32_weights).shrink_to_fit();
      }
    }
  };
  
  // Load Q projection
  load_single_weight(
    [this, layer_idx]() { ensure_q_proj_dequantized(layer_idx); },
    [&lw]() -> const std::vector<float>& { return lw.q_proj_f32; },
    lw.q_proj_f32_dev,
    "Q Proj"
  );
  
  // Load K projection  
  load_single_weight(
    [this, layer_idx]() { ensure_k_proj_dequantized(layer_idx); },
    [&lw]() -> const std::vector<float>& { return lw.k_proj_f32; },
    lw.k_proj_f32_dev,
    "K Proj"
  );
  
  // Load V projection
  load_single_weight(
    [this, layer_idx]() { ensure_v_proj_dequantized(layer_idx); },
    [&lw]() -> const std::vector<float>& { return lw.v_proj_f32; },
    lw.v_proj_f32_dev, 
    "V Proj"
  );
  
  // Load O projection
  load_single_weight(
    [this, layer_idx]() { ensure_o_proj_dequantized(layer_idx); },
    [&lw]() -> const std::vector<float>& { return lw.o_proj_f32; },
    lw.o_proj_f32_dev,
    "O Proj"
  );
  
  // Load Gate projection
  load_single_weight(
    [this, layer_idx]() { ensure_gate_proj_dequantized(layer_idx); },
    [&lw]() -> const std::vector<float>& { return lw.gate_proj_f32; },
    lw.gate_proj_f32_dev,
    "Gate Proj"
  );
  
  // Load Up projection
  load_single_weight(
    [this, layer_idx]() { ensure_up_proj_dequantized(layer_idx); },
    [&lw]() -> const std::vector<float>& { return lw.up_proj_f32; },
    lw.up_proj_f32_dev,
    "Up Proj"
  );
  
  // Load Down projection
  load_single_weight(
    [this, layer_idx]() { ensure_down_proj_dequantized(layer_idx); },
    [&lw]() -> const std::vector<float>& { return lw.down_proj_f32; },
    lw.down_proj_f32_dev,
    "Down Proj"
  );
#endif
}

void TinyLlamaModel::free_layer_gpu_weights(int layer_idx) {
#ifdef HAS_CUDA
  if (layer_idx < 0 || layer_idx >= layers.size()) return;
  
  LayerWeights& lw = layers[layer_idx];
  
  if (lw.q_proj_f32_dev) { cudaFree(lw.q_proj_f32_dev); lw.q_proj_f32_dev = nullptr; }
  if (lw.k_proj_f32_dev) { cudaFree(lw.k_proj_f32_dev); lw.k_proj_f32_dev = nullptr; }
  if (lw.v_proj_f32_dev) { cudaFree(lw.v_proj_f32_dev); lw.v_proj_f32_dev = nullptr; }
  if (lw.o_proj_f32_dev) { cudaFree(lw.o_proj_f32_dev); lw.o_proj_f32_dev = nullptr; }
  if (lw.gate_proj_f32_dev) { cudaFree(lw.gate_proj_f32_dev); lw.gate_proj_f32_dev = nullptr; }
  if (lw.up_proj_f32_dev) { cudaFree(lw.up_proj_f32_dev); lw.up_proj_f32_dev = nullptr; }
  if (lw.down_proj_f32_dev) { cudaFree(lw.down_proj_f32_dev); lw.down_proj_f32_dev = nullptr; }
  
  Logger::info("Freed GPU weights for layer " + std::to_string(layer_idx) + " (~200MB freed)");
#endif
}

void TinyLlamaModel::ensure_lm_head_dequantized() {
    if (!this->lm_head_f32.empty()) return;
    
    size_t total_elements_lm_head = static_cast<size_t>(config_.vocab_size) * config_.hidden_size;
    if (!this->lm_head_q6k.empty()) {
        dequantize_vector_q6k_to_f32(this->lm_head_q6k, this->lm_head_f32, total_elements_lm_head, 1);
    } else if (!this->lm_head_q4k.empty()) {
        dequantize_vector_q4k_to_f32(this->lm_head_q4k, this->lm_head_f32, total_elements_lm_head, 1);
    } else if (!this->lm_head_q8k.empty()) {
        dequantize_q8_k(this->lm_head_q8k, this->lm_head_f32, total_elements_lm_head, true);
    } else if (!this->lm_head_q8_0.empty()) {
        dequantize_vector_q8_0_to_f32(this->lm_head_q8_0, this->lm_head_f32, total_elements_lm_head, 1);
    } else if (!this->lm_head.empty()) { 
        this->lm_head_f32 = bf16vec_to_float_vec(this->lm_head);
    }
}
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
    for (int i = 0; i < nhl; ++i) { // Clear dev pointers for ALL layers
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
  

  // Final Norm (always on GPU if any GPU layers are active)
  if (final_norm_f32.empty() && !final_norm.empty()) {
      Logger::info("Converting final_norm (BF16) to FP32 for GPU.");
      final_norm_f32 = bf16vec_to_float_vec(final_norm); // Ensure FP32 version exists
  }
  if (!final_norm_f32.empty()) {
    SAFE_CUDA_FREE(final_norm_dev);
    gpuErrchk(cudaMalloc(&final_norm_dev, final_norm_f32.size() * sizeof(float)));
    gpuErrchk(cudaMemcpy(final_norm_dev, final_norm_f32.data(), final_norm_f32.size() * sizeof(float), cudaMemcpyHostToDevice));
    Logger::info("Copied final_norm weights (FP32) to GPU.");
  } else {
    Logger::warning("Final norm weights (FP32) are empty, skipping GPU copy. This might be an issue if GPU layers are expected to use it.");
  }

  // Layer-specific norms for GPU layers
  // First, clear any existing dev pointers for layers that are now designated for CPU
  for (int i = 0; i < active_num_cpu_layers; ++i) {
      if (static_cast<size_t>(i) < layers.size()) { // Boundary check
        SAFE_CUDA_FREE(layers[i].input_layernorm_dev);
        SAFE_CUDA_FREE(layers[i].post_attention_layernorm_dev);
      }
  }
  Logger::info("Copying layer norm weights (FP32) to GPU for layers " + std::to_string(active_num_cpu_layers) + " to " + std::to_string(nhl - 1));
  Logger::info("[INIT_DEBUG_PRE_LOOP] Active CPU layers: " + std::to_string(active_num_cpu_layers));
  if (nhl > 0 && layers.size() > 0) { // Check if layers exist to prevent out of bounds
    Logger::info("[INIT_DEBUG_PRE_LOOP] layers[0].input_layernorm_f32.empty(): " + std::string(layers[0].input_layernorm_f32.empty() ? "YES" : "NO") + 
                 ", Size: " + std::to_string(layers[0].input_layernorm_f32.size()));
  }
  for (int i = active_num_cpu_layers; i < nhl; ++i) { // Iterate ONLY over GPU layers
    if (static_cast<size_t>(i) >= layers.size()) { // Boundary check
        Logger::error("Layer index " + std::to_string(i) + " out of bounds for layers vector (size: " + std::to_string(layers.size()) + ")");
        continue; 
    }
    SAFE_CUDA_FREE(layers[i].input_layernorm_dev); // Free before realloc, in case of re-initialization
    SAFE_CUDA_FREE(layers[i].post_attention_layernorm_dev); // Free before realloc

    // Ensure FP32 versions of norm weights exist if original was BF16
    if (layers[i].input_layernorm_f32.empty() && !layers[i].input_layernorm.empty()) {
        layers[i].input_layernorm_f32 = bf16vec_to_float_vec(layers[i].input_layernorm);
    }
    if (layers[i].post_attention_layernorm_f32.empty() && !layers[i].post_attention_layernorm.empty()) {
        layers[i].post_attention_layernorm_f32 = bf16vec_to_float_vec(layers[i].post_attention_layernorm);
    }
    
    if (!layers[i].input_layernorm_f32.empty()) {
      gpuErrchk(cudaMalloc(&layers[i].input_layernorm_dev, layers[i].input_layernorm_f32.size() * sizeof(float)));
      gpuErrchk(cudaMemcpy(layers[i].input_layernorm_dev, layers[i].input_layernorm_f32.data(), layers[i].input_layernorm_f32.size() * sizeof(float), cudaMemcpyHostToDevice));
      if (i == active_num_cpu_layers) { // Log only for the first GPU layer processed
          Logger::info("[INIT_DEBUG] layers[" + std::to_string(i) + "].input_layernorm_dev allocated. Pointer: " + Logger::ptrToString(layers[i].input_layernorm_dev) + 
                       ", Size used for malloc: " + std::to_string(layers[i].input_layernorm_f32.size() * sizeof(float)) + " bytes (" +
                       std::to_string(layers[i].input_layernorm_f32.size()) + " elements). Host vector empty: " + (layers[i].input_layernorm_f32.empty() ? "YES" : "NO"));
      }
    } else {
      // This layer is designated for GPU. It MUST have its norm weights.
      throw std::runtime_error("GPU Layer " + std::to_string(i) + ": input_layernorm_f32 weights are empty. Cannot offload to GPU without them.");
    }
    
    if (!layers[i].post_attention_layernorm_f32.empty()) {
      gpuErrchk(cudaMalloc(&layers[i].post_attention_layernorm_dev, layers[i].post_attention_layernorm_f32.size() * sizeof(float)));
      gpuErrchk(cudaMemcpy(layers[i].post_attention_layernorm_dev, layers[i].post_attention_layernorm_f32.data(), layers[i].post_attention_layernorm_f32.size() * sizeof(float), cudaMemcpyHostToDevice));
    } else {
      // This layer is designated for GPU. It MUST have its norm weights.
      throw std::runtime_error("GPU Layer " + std::to_string(i) + ": post_attention_layernorm_f32 weights are empty. Cannot offload to GPU without them.");
    }
  }
  Logger::info("Finished processing layer norm weights for GPU layers.");


  // --- TOKEN EMBEDDING TABLE to GPU (as BF16) ---
  SAFE_CUDA_FREE(token_embedding_table_dev_);    // Target for BF16
  SAFE_CUDA_FREE(token_embedding_table_f32_dev_); // Ensure this is cleared and not used by new embedding logic
  ensure_embed_tokens_dequantized();
  bool token_embeddings_processed_to_gpu_bf16 = false;

  if (active_num_gpu_layers > 0) { // Only process if GPU layers are active
    // Path 1: Source is already BF16 (model.embed_tokens is std::vector<uint16_t>)
  if (!embed_tokens.empty()) {
    gpuErrchk(cudaMalloc(&token_embedding_table_dev_, embed_tokens.size() * sizeof(uint16_t)));
    gpuErrchk(cudaMemcpy(token_embedding_table_dev_, embed_tokens.data(), embed_tokens.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
      Logger::info("Copied token_embedding_table (bf16 direct from model.embed_tokens) to GPU.");
      token_embeddings_processed_to_gpu_bf16 = true;
    }
    // Path 2: Source is FP32 (model.embed_tokens_f32) -> convert to BF16
    else if (!embed_tokens_f32.empty()) {
      std::vector<uint16_t> bf16_data(embed_tokens_f32.size());
      #pragma omp parallel for
      for (size_t i = 0; i < embed_tokens_f32.size(); ++i) {
        bf16_data[i] = float32_to_bfloat16(embed_tokens_f32[i]);
      }
      gpuErrchk(cudaMalloc(&token_embedding_table_dev_, bf16_data.size() * sizeof(uint16_t)));
      gpuErrchk(cudaMemcpy(token_embedding_table_dev_, bf16_data.data(), bf16_data.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
      Logger::info("Converted token_embedding_table (fp32 source -> bf16) to GPU.");
      token_embeddings_processed_to_gpu_bf16 = true;
    }
    // Path 3: Source is Q8_0 (model.embed_tokens_q8_0) -> dequantize to FP32, then convert to BF16
    else if (!embed_tokens_q8_0.empty()) {
      std::vector<float> temp_f32_data(embed_tokens_q8_0.size() * GGML_QK8_0);
      #pragma omp parallel for
      for (size_t i = 0; i < embed_tokens_q8_0.size(); ++i) {
        dequantize_q8_0_block(&embed_tokens_q8_0[i], &temp_f32_data[i * GGML_QK8_0]);
      }
      std::vector<uint16_t> bf16_data(temp_f32_data.size());
      #pragma omp parallel for
      for (size_t i = 0; i < temp_f32_data.size(); ++i) {
        bf16_data[i] = float32_to_bfloat16(temp_f32_data[i]);
      }
      gpuErrchk(cudaMalloc(&token_embedding_table_dev_, bf16_data.size() * sizeof(uint16_t)));
      gpuErrchk(cudaMemcpy(token_embedding_table_dev_, bf16_data.data(), bf16_data.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
      Logger::info("Dequantized token_embedding_table (Q8_0 -> fp32 -> bf16) to GPU.");
      token_embeddings_processed_to_gpu_bf16 = true;
    }
    // Path 4: Source is Q4_K (model.embed_tokens_q4k) -> dequantize to FP32, then convert to BF16
    else if (!embed_tokens_q4k.empty()) {
      std::vector<float> temp_f32_data(embed_tokens_q4k.size() * GGML_QK_K);
      #pragma omp parallel for
      for (size_t i = 0; i < embed_tokens_q4k.size(); ++i) {
        dequantize_q4_k_m(&embed_tokens_q4k[i], &temp_f32_data[i * GGML_QK_K], GGML_QK_K);
      }
      std::vector<uint16_t> bf16_data(temp_f32_data.size());
      #pragma omp parallel for
      for (size_t i = 0; i < temp_f32_data.size(); ++i) {
        bf16_data[i] = float32_to_bfloat16(temp_f32_data[i]);
      }
      gpuErrchk(cudaMalloc(&token_embedding_table_dev_, bf16_data.size() * sizeof(uint16_t)));
      gpuErrchk(cudaMemcpy(token_embedding_table_dev_, bf16_data.data(), bf16_data.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
      Logger::info("Dequantized token_embedding_table (Q4_K -> fp32 -> bf16) to GPU.");
      token_embeddings_processed_to_gpu_bf16 = true;
    }
    // Path 5: Source is Q6_K (model.embed_tokens_q6k) -> dequantize to FP32, then convert to BF16
    else if (!embed_tokens_q6k.empty()) {
      std::vector<float> temp_f32_data(embed_tokens_q6k.size() * GGML_QK_K);
      #pragma omp parallel for
      for (size_t i = 0; i < embed_tokens_q6k.size(); ++i) {
        dequantize_q6_k(&embed_tokens_q6k[i], &temp_f32_data[i * GGML_QK_K], GGML_QK_K);
      }
      std::vector<uint16_t> bf16_data(temp_f32_data.size());
      #pragma omp parallel for
      for (size_t i = 0; i < temp_f32_data.size(); ++i) {
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
    } // This closing brace was for the "if (active_num_gpu_layers > 0)" for token embeddings.
    // The next line is the warning check.
    if (!token_embeddings_processed_to_gpu_bf16 && active_num_gpu_layers > 0) { // Added active_num_gpu_layers check here too
        Logger::warning("Token embeddings were not processed to GPU as BF16, despite GPU layers being active. This might indicate missing source embedding data in the model structure or an unhandled GGUF type for embeddings.");
    }
  } else {
    Logger::info("No GPU layers active, skipping token embedding table processing for GPU.");
  }

  // --- LM HEAD to GPU (as BF16) ---
  SAFE_CUDA_FREE(lm_head_dev_);    // Target for BF16
  SAFE_CUDA_FREE(lm_head_f32_dev_); // Ensure this is cleared and not used by new LM head logic
  ensure_lm_head_dequantized();
  bool lm_head_processed_to_gpu_bf16 = false;

  if (active_num_gpu_layers > 0) { // Only process if GPU layers are active
    // Path 1: Source is already BF16 (model.lm_head is std::vector<uint16_t>)
  if (!lm_head.empty()) {
    gpuErrchk(cudaMalloc(&lm_head_dev_, lm_head.size() * sizeof(uint16_t)));
    gpuErrchk(cudaMemcpy(lm_head_dev_, lm_head.data(), lm_head.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
      Logger::info("Copied lm_head (bf16 direct from model.lm_head) to GPU.");
      lm_head_processed_to_gpu_bf16 = true;
    }
    // Path 2: Source is FP32 (model.lm_head_f32) -> convert to BF16
    else if (!lm_head_f32.empty()) {
      std::vector<uint16_t> bf16_data(lm_head_f32.size());
      #pragma omp parallel for
      for (size_t i = 0; i < lm_head_f32.size(); ++i) {
        bf16_data[i] = float32_to_bfloat16(lm_head_f32[i]);
      }
      gpuErrchk(cudaMalloc(&lm_head_dev_, bf16_data.size() * sizeof(uint16_t)));
      gpuErrchk(cudaMemcpy(lm_head_dev_, bf16_data.data(), bf16_data.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
      Logger::info("Converted lm_head (fp32 source -> bf16) to GPU.");
      lm_head_processed_to_gpu_bf16 = true;
    }
    // Path 3: Source is Q8_0 (model.lm_head_q8_0) -> dequantize to FP32, then convert to BF16
    else if (!lm_head_q8_0.empty()) {
      std::vector<float> temp_f32_data(lm_head_q8_0.size() * GGML_QK8_0);
      #pragma omp parallel for
      for (size_t i = 0; i < lm_head_q8_0.size(); ++i) {
        dequantize_q8_0_block(&lm_head_q8_0[i], &temp_f32_data[i * GGML_QK8_0]);
      }
      std::vector<uint16_t> bf16_data(temp_f32_data.size());
      #pragma omp parallel for
      for (size_t i = 0; i < temp_f32_data.size(); ++i) {
        bf16_data[i] = float32_to_bfloat16(temp_f32_data[i]);
      }
      gpuErrchk(cudaMalloc(&lm_head_dev_, bf16_data.size() * sizeof(uint16_t)));
      gpuErrchk(cudaMemcpy(lm_head_dev_, bf16_data.data(), bf16_data.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
      Logger::info("Dequantized lm_head (Q8_0 -> fp32 -> bf16) to GPU.");
      lm_head_processed_to_gpu_bf16 = true;
    }
    // Path 4: Source is Q4_K (model.lm_head_q4k) -> dequantize to FP32, then convert to BF16
    else if (!lm_head_q4k.empty()) {
      std::vector<float> temp_f32_data(lm_head_q4k.size() * GGML_QK_K);
      #pragma omp parallel for
      for (size_t i = 0; i < lm_head_q4k.size(); ++i) {
        dequantize_q4_k_m(&lm_head_q4k[i], &temp_f32_data[i * GGML_QK_K], GGML_QK_K);
      }
      std::vector<uint16_t> bf16_data(temp_f32_data.size());
      #pragma omp parallel for
      for (size_t i = 0; i < temp_f32_data.size(); ++i) {
        bf16_data[i] = float32_to_bfloat16(temp_f32_data[i]);
      }
      gpuErrchk(cudaMalloc(&lm_head_dev_, bf16_data.size() * sizeof(uint16_t)));
      gpuErrchk(cudaMemcpy(lm_head_dev_, bf16_data.data(), bf16_data.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
      Logger::info("Dequantized lm_head (Q4_K -> fp32 -> bf16) to GPU.");
      lm_head_processed_to_gpu_bf16 = true;
    }
    // Path 5: Source is Q6_K (model.lm_head_q6k) -> dequantize to FP32, then convert to BF16
    else if (!lm_head_q6k.empty()) {
      std::vector<float> temp_f32_data(lm_head_q6k.size() * GGML_QK_K);
      #pragma omp parallel for
      for (size_t i = 0; i < lm_head_q6k.size(); ++i) {
        dequantize_q6_k(&lm_head_q6k[i], &temp_f32_data[i * GGML_QK_K], GGML_QK_K);
      }
      std::vector<uint16_t> bf16_data(temp_f32_data.size());
      #pragma omp parallel for
      for (size_t i = 0; i < temp_f32_data.size(); ++i) {
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
  // --- END LM HEAD ---

    SAFE_CUDA_FREE(lm_head_f32_dev_); // Ensure it's clear before attempting to populate

  if (active_num_gpu_layers > 0) {
    if (!lm_head_f32.empty()) { // Check if host lm_head_f32 has data
      gpuErrchk(cudaMalloc(&lm_head_f32_dev_, lm_head_f32.size() * sizeof(float)));
      gpuErrchk(cudaMemcpy(lm_head_f32_dev_, lm_head_f32.data(), lm_head_f32.size() * sizeof(float), cudaMemcpyHostToDevice));
      Logger::info("[INIT_GPU_ROPE] Copied lm_head_f32 (host FP32) to GPU for lm_head_f32_dev_. Pointer: " + Logger::ptrToString(lm_head_f32_dev_));
    } else {
      // This is a critical issue if lm_head_f32 is empty, as the matvec will fail.
      Logger::error("[INIT_GPU_ROPE] Host lm_head_f32 is EMPTY. Cannot populate lm_head_f32_dev_. This WILL CAUSE a cublasSgemm error in the final matvec. Check model loading and initialize_weights logic for lm_head_f32 population.");
      lm_head_f32_dev_ = nullptr; // Explicitly ensure it's null if source is empty
    }
  } else {
    // Ensure it's null if no GPU layers are active (though SAFE_CUDA_FREE above would handle this)
    lm_head_f32_dev_ = nullptr; 
  }

  
  Logger::info("Finished processing embedding and LM head tables for GPU.");

  // RoPE GPU Buffer - Only allocate if GPU layers are active
  SAFE_CUDA_FREE(all_freqs_cis_dev);
  if (active_num_gpu_layers > 0) {
    if (!precomputed_freqs_cis_.empty()) { // RoPE freqs are always precomputed on CPU
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

  // Workspace GPU Buffers - Only allocate if GPU layers are active
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
    REALLOC_GPU_WORKSPACE(q_dev_, hs_bytes); // q_dev for Q projection output is full hidden size
    REALLOC_GPU_WORKSPACE(k_dev_, k_dev_size_bytes); // k_dev for K projection output
    REALLOC_GPU_WORKSPACE(v_dev_, v_dev_size_bytes); // v_dev for V projection output
    REALLOC_GPU_WORKSPACE(attn_out_dev_, hs_bytes);
    REALLOC_GPU_WORKSPACE(attn_proj_dev_, hs_bytes); 
    REALLOC_GPU_WORKSPACE(gate_vec_dev_, is_bytes);
    REALLOC_GPU_WORKSPACE(up_vec_dev_, is_bytes);
    REALLOC_GPU_WORKSPACE(swiglu_vec_dev_, is_bytes);
    REALLOC_GPU_WORKSPACE(mlp_down_dev_, hs_bytes); 
    REALLOC_GPU_WORKSPACE(logits_dev_, vs_bytes); // For final logits calculation on GPU
    Logger::info("Finished allocating/reallocating GPU workspace buffers.");
  } else {
    Logger::info("No GPU layers active, skipping GPU workspace buffer allocation.");
    // Ensure all GPU workspace buffers are freed/null for CPU-only mode
    SAFE_CUDA_FREE(x_dev_); SAFE_CUDA_FREE(x_norm_dev_); SAFE_CUDA_FREE(x_resid1_dev_); SAFE_CUDA_FREE(x_resid2_dev_);
    SAFE_CUDA_FREE(q_dev_); SAFE_CUDA_FREE(k_dev_); SAFE_CUDA_FREE(v_dev_); SAFE_CUDA_FREE(attn_out_dev_); SAFE_CUDA_FREE(attn_proj_dev_);
    SAFE_CUDA_FREE(gate_vec_dev_); SAFE_CUDA_FREE(up_vec_dev_); SAFE_CUDA_FREE(swiglu_vec_dev_); SAFE_CUDA_FREE(mlp_down_dev_); SAFE_CUDA_FREE(logits_dev_);
  }

  // Allocate KVCache dequantization buffers if GPU layers are active
  if (active_num_gpu_layers > 0) {
    // Always use selective approach when quantization is enabled (memory efficient)
    // Allocate much smaller buffers (only max_seq_len * head_dim per head)
        // This is enough to dequantize one head's worth of history at a time
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
    
    // Clean up any old full-size buffers that may exist
            SAFE_CUDA_FREE(dequant_k_cache_buffer_dev_); 
            SAFE_CUDA_FREE(dequant_v_cache_buffer_dev_);
  } else { // No active GPU layers
    SAFE_CUDA_FREE(dequant_k_cache_buffer_dev_);
    SAFE_CUDA_FREE(dequant_v_cache_buffer_dev_);
    SAFE_CUDA_FREE(selective_k_dequant_buffer_dev_);
    SAFE_CUDA_FREE(selective_v_dequant_buffer_dev_);
  }

// #undef REALLOC_GPU_WORKSPACE // Not needed if we expanded it or it was already defined

  // Concatenated BF16 layer weights for GPU layers (w_..._dev_ pointers)
  // Check if the first active GPU layer has BF16 q_proj weights to decide if this path should be taken.
  bool process_bf16_concat_weights = active_num_gpu_layers > 0 && !layers[active_num_cpu_layers].q_proj.empty();
  if (process_bf16_concat_weights) {
    size_t layer_q_size = (size_t)hs*hs, layer_k_size = (size_t)kv_dim*hs, layer_v_size = (size_t)kv_dim*hs, layer_o_size = (size_t)hs*hs;
    size_t layer_gate_size = (size_t)is*hs, layer_up_size = (size_t)is*hs, layer_down_size = (size_t)hs*is;
    
    std::vector<uint16_t> h_q, h_k, h_v, h_o, h_gate, h_up, h_down; // h for host
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

  // DEFER Concatenated FP32 / Dequantized Layer Weights for GPU layers to save memory during initialization
  // This massive dequantization was causing OOM issues during model loading
  Logger::info("DEFERRING concatenated F32 weight processing for GPU layers to save memory during initialization");
  Logger::info("Concatenated F32 weights will be processed on-demand during first inference");
  
  // Just ensure the pointers are null for now
  SAFE_CUDA_FREE(w_q_f32_dev_); SAFE_CUDA_FREE(w_k_f32_dev_); SAFE_CUDA_FREE(w_v_f32_dev_); SAFE_CUDA_FREE(w_o_f32_dev_);
  SAFE_CUDA_FREE(w_gate_f32_dev_); SAFE_CUDA_FREE(w_up_f32_dev_); SAFE_CUDA_FREE(w_down_f32_dev_);

  Logger::info("Finished deferring concatenated F32 weight processing for GPU layers.");

#undef SAFE_CUDA_FREE
#else // HAS_CUDA not defined
  if (active_num_gpu_layers > 0 && nhl > 0) {
      Logger::warning("CUDA not available, but " + std::to_string(active_num_gpu_layers) + " layer(s) were configured for GPU. Model will run entirely on CPU.");
  } else {
      Logger::info("CUDA not available or no GPU layers configured. Model will run entirely on CPU.");
  }
#endif // HAS_CUDA
}
}

TinyLlamaModel::TinyLlamaModel(const ModelConfig& config,
                               const SafeTensorsLoader& loader)
    : config_(config) { // Copies the potentially faulty config first
  config_.is_gguf_file_loaded = false; // Explicitly set to false for SafeTensors path
  Logger::info("Constructing TinyLlamaModel from SafeTensorsLoader (is_gguf_file_loaded set to false).");
  initialize_weights(&loader, nullptr);
  initialize_gpu_and_rope();
  Logger::info("TinyLlamaModel construction from SafeTensorsLoader complete.");
}

TinyLlamaModel::TinyLlamaModel(const ModelConfig& initial_config,
                               const std::string& model_path)
    : model_path_(model_path) 
#ifdef HAS_CUDA
      , cublas_handle_(nullptr), token_embedding_table_dev_(nullptr), lm_head_dev_(nullptr), final_norm_dev(nullptr), w_q_dev_(nullptr), w_k_dev_(nullptr), w_v_dev_(nullptr), w_o_dev_(nullptr), w_gate_dev_(nullptr), w_up_dev_(nullptr), w_down_dev_(nullptr), all_freqs_cis_dev(nullptr), x_dev_(nullptr), x_norm_dev_(nullptr), x_resid1_dev_(nullptr), x_resid2_dev_(nullptr), q_dev_(nullptr), k_dev_(nullptr), v_dev_(nullptr), attn_out_dev_(nullptr), attn_proj_dev_(nullptr), gate_vec_dev_(nullptr), up_vec_dev_(nullptr), swiglu_vec_dev_(nullptr), mlp_down_dev_(nullptr), logits_dev_(nullptr), token_embedding_table_f32_dev_(nullptr), lm_head_f32_dev_(nullptr), w_q_f32_dev_(nullptr), w_k_f32_dev_(nullptr), w_v_f32_dev_(nullptr), w_o_f32_dev_(nullptr), w_gate_f32_dev_(nullptr), w_up_f32_dev_(nullptr), w_down_f32_dev_(nullptr)
#endif
{
  Logger::info("TinyLlamaModel constructor entered. Model path (from string): " + model_path);
  int cli_gpu_layer_request = initial_config.num_cpu_offload_layers; 
  bool cli_mmap_preference = initial_config.use_mmap_for_gguf;


  this->config_ = initial_config; // Start with initial_config, then overwrite with GGUF specifics

  if (this->model_path_.empty() && !model_path.empty()) {
      this->model_path_ = model_path;
  }

  std::unique_ptr<SafeTensorsLoader> loader = nullptr;

  if (!this->model_path_.empty() && this->model_path_.size() > 5 &&
      this->model_path_.substr(this->model_path_.size() - 5) == ".gguf") {
    Logger::info("GGUF file detected by path in Model Constructor: " + this->model_path_);
    try {
      bool force_mmap_for_gguf_load = cli_mmap_preference; 
      Logger::info("TinyLlamaModel GGUF path: Using mmap setting " + std::string(force_mmap_for_gguf_load ? "true" : "false") + 
                   " for gguf_meta/weight loading based on CLI mmap preference: " + 
                   std::string(cli_mmap_preference ? "true" : "false"));

      this->gguf_data_ = std::make_unique<GGUFData>(load_gguf_meta(this->model_path_, force_mmap_for_gguf_load));

      ModelConfig config_from_gguf = parse_model_config_from_gguf(*(this->gguf_data_));
      this->config_ = config_from_gguf; 
      this->config_.use_mmap_for_gguf = cli_mmap_preference; // Honor CLI mmap preference for session's use of mmap later
      this->config_.is_gguf_file_loaded = true;
      if (cli_gpu_layer_request < 0) {
        this->config_.num_cpu_offload_layers = 0;
        Logger::info("TinyLlamaModel GGUF Ctor CALC: CLI hint < 0 (all GPU). num_cpu_offload_layers set to 0.");
      } else if (cli_gpu_layer_request == 0) {
        this->config_.num_cpu_offload_layers = this->config_.num_hidden_layers;
        Logger::info("TinyLlamaModel GGUF Ctor CALC: CLI hint == 0 (all CPU). num_cpu_offload_layers set to num_hidden_layers (" + std::to_string(this->config_.num_cpu_offload_layers) + ").");
      } else { // CLI hint > 0, meaning cli_gpu_layer_request is the number of desired GPU layers
        if (this->config_.num_hidden_layers > 0) {
            if (cli_gpu_layer_request >= this->config_.num_hidden_layers) {
                this->config_.num_cpu_offload_layers = 0; // More GPU layers requested than available -> all on GPU
                Logger::info("TinyLlamaModel GGUF Ctor CALC: CLI GPU layer request ("+ std::to_string(cli_gpu_layer_request) +") >= total layers. num_cpu_offload_layers set to 0.");
            } else {
                this->config_.num_cpu_offload_layers = this->config_.num_hidden_layers - cli_gpu_layer_request;
                Logger::info("TinyLlamaModel GGUF Ctor CALC: Partial GPU. CLI GPU req: " + std::to_string(cli_gpu_layer_request) + ". num_cpu_offload_layers set to " + std::to_string(this->config_.num_cpu_offload_layers));
            }
        } else { // num_hidden_layers is 0 or negative, something is wrong with GGUF. Default to all CPU.
            this->config_.num_cpu_offload_layers = 0; 
            Logger::warning("TinyLlamaModel GGUF Ctor CALC: num_hidden_layers from GGUF is <= 0. Defaulting num_cpu_offload_layers to 0. CLI GPU req: " + std::to_string(cli_gpu_layer_request));
        }
      }
      Logger::info("TinyLlamaModel GGUF Ctor: POST-CALC (within GGUF block) final num_cpu_offload_layers = " + std::to_string(this->config_.num_cpu_offload_layers));
      Logger::info("[CTOR_GGUF_DEBUG_L1860] After CLI hint logic: this->config_.num_cpu_offload_layers = " + std::to_string(this->config_.num_cpu_offload_layers) +
                    ", this->config_.num_hidden_layers = " + std::to_string(this->config_.num_hidden_layers));
    } catch (const std::exception& e) {
      Logger::error("Failed to load or parse GGUF file: " + std::string(e.what()));
      throw; 
    }
  } else if (model_path.size() > 12 &&
             model_path.substr(model_path.size() - 12) == ".safetensors") {
    Logger::info("SafeTensors file detected: " + model_path);
    ModelConfig config_from_json; 
    bool json_loaded_successfully = SafeTensorsLoader::load_model_config_from_json(model_path, config_from_json);
    
    // For SafeTensors, start with JSON config, then layer CLI preferences.
    if (json_loaded_successfully) {
        Logger::info("Successfully loaded and parsed config.json for SafeTensors model.");
        this->config_ = config_from_json; // Base is from JSON
    } else {
        Logger::warning("Failed to load config.json or it was not found for SafeTensors model. Proceeding with initial_config defaults and CLI overrides.");
    }
        this->config_.is_gguf_file_loaded = false;
    this->config_.use_mmap_for_gguf = cli_mmap_preference; // This field is GGUF specific, but store CLI pref anyway.

    // Calculate num_cpu_offload_layers for SafeTensors based on cli_gpu_layer_request and config's num_hidden_layers
    if (cli_gpu_layer_request < 0) {
        this->config_.num_cpu_offload_layers = 0;
    } else if (cli_gpu_layer_request == 0) {
        this->config_.num_cpu_offload_layers = this->config_.num_hidden_layers;
    } else {
        if (this->config_.num_hidden_layers > 0) {
            if (cli_gpu_layer_request >= this->config_.num_hidden_layers) {
                this->config_.num_cpu_offload_layers = 0;
            } else {
                this->config_.num_cpu_offload_layers = this->config_.num_hidden_layers - cli_gpu_layer_request;
            }
        } else {
            this->config_.num_cpu_offload_layers = 0; // Fallback if num_hidden_layers not known
            Logger::warning("SafeTensors path: num_hidden_layers is 0 from JSON/default. Defaulting num_cpu_offload_layers to 0 despite CLI GPU request: " + std::to_string(cli_gpu_layer_request));
        }
    }
    Logger::info("SafeTensors path: Calculated num_cpu_offload_layers = " + std::to_string(this->config_.num_cpu_offload_layers));

    try {
      loader = std::make_unique<SafeTensorsLoader>(model_path);
      Logger::info("SafeTensorsLoader initialized for: " + model_path);
    } catch (const std::exception& e) {
        Logger::error("Failed to initialize SafeTensorsLoader: " + std::string(e.what()));
        throw; 
    }
  } else {
    throw std::runtime_error(
        "Unsupported model file type. Please use .gguf or .safetensors");
  }

  Logger::info("TinyLlamaModel constructor: After specific loader block. Current config_.num_cpu_offload_layers = " + std::to_string(this->config_.num_cpu_offload_layers) + 
               ", config_.num_hidden_layers = " + std::to_string(this->config_.num_hidden_layers));
  Logger::info("TinyLlamaModel constructor: Current config_.use_mmap_for_gguf = " + std::string(this->config_.use_mmap_for_gguf ? "true" : "false"));

  if (this->config_.num_cpu_offload_layers < 0) { // Should not happen if logic above is correct for -1 CLI hint
      this->config_.num_cpu_offload_layers = 0;
      Logger::warning("Clamping num_cpu_offload_layers: was < 0, set to 0.");
  }
  if (this->config_.num_hidden_layers > 0 && this->config_.num_cpu_offload_layers > this->config_.num_hidden_layers) {
      Logger::warning("Clamping num_cpu_offload_layers: Requested CPU offload layers (" + std::to_string(this->config_.num_cpu_offload_layers) +
                      ") exceeds total hidden layers (" + std::to_string(this->config_.num_hidden_layers) +
                      "). Clamping to " + std::to_string(this->config_.num_hidden_layers) + " (all CPU).");
      this->config_.num_cpu_offload_layers = this->config_.num_hidden_layers;
  }
  Logger::info("TinyLlamaModel constructor: Final clamped num_cpu_offload_layers = " + std::to_string(this->config_.num_cpu_offload_layers));
  Logger::info("[CTOR_DEBUG_L1921] End of Model Ctor (before initialize_weights/rope call): this->config_.num_cpu_offload_layers = " + std::to_string(this->config_.num_cpu_offload_layers) +
              ", this->config_.num_hidden_layers = " + std::to_string(this->config_.num_hidden_layers));
  Logger::info("Final ModelConfig (before initialize_weights/rope):");
  Logger::info("  hidden_size: " + std::to_string(config_.hidden_size));
  Logger::info("  intermediate_size: " + std::to_string(config_.intermediate_size));
  Logger::info("  num_attention_heads: " + std::to_string(config_.num_attention_heads));
  Logger::info("  num_key_value_heads: " + std::to_string(config_.num_key_value_heads));
  Logger::info("  num_hidden_layers: " + std::to_string(config_.num_hidden_layers));
  Logger::info("  vocab_size: " + std::to_string(config_.vocab_size));
  Logger::info("  max_position_embeddings: " + std::to_string(config_.max_position_embeddings));
  Logger::info("  architecture: " + config_.architecture);
  Logger::info("  is_gguf_file_loaded: " + std::string(config_.is_gguf_file_loaded ? "true" : "false"));
  Logger::info("  use_mmap_for_gguf: " + std::string(config_.use_mmap_for_gguf ? "true" : "false"));
    // --- BEGIN GGUFData Integrity Check ---
  if (this->config_.is_gguf_file_loaded && this->gguf_data_) {
    if (this->gguf_data_->tensor_infos_map.empty()) {
        Logger::error("[CTOR_GGUF_PRE_INIT_W] CRITICAL: gguf_data_->tensor_infos_map is EMPTY. Weights will not be loaded by map_gguf_weights.");
    }
  } else if (this->config_.is_gguf_file_loaded && !this->gguf_data_) {
    Logger::error("[CTOR_GGUF_PRE_INIT_W] CRITICAL: config_.is_gguf_file_loaded is TRUE, but gguf_data_ pointer IS NULL. Weights cannot be loaded.");
  } else if (!this->config_.is_gguf_file_loaded) {
    Logger::info("[CTOR_GGUF_PRE_INIT_W] Not a GGUF file load context (e.g., SafeTensors). Skipping gguf_data_ check here.");
  }
  // --- END GGUFData Integrity Check ---
  initialize_weights(loader.get(), this->gguf_data_.get()); 
  initialize_gpu_and_rope(); 

  Logger::info("TinyLlamaModel (from path string) constructed and initialized successfully.");
}
TinyLlamaModel::TinyLlamaModel(const ModelConfig& config_from_session,
                               std::unique_ptr<GGUFData> gguf_data_from_session)
    : config_(config_from_session), 
      gguf_data_(std::move(gguf_data_from_session)), 
      model_path_("loaded_from_gguf_data_memory")
#ifdef HAS_CUDA
      // Initialize all CUDA pointers to nullptr as in the other constructor
      , cublas_handle_(nullptr), token_embedding_table_dev_(nullptr), lm_head_dev_(nullptr), final_norm_dev(nullptr), w_q_dev_(nullptr), w_k_dev_(nullptr), w_v_dev_(nullptr), w_o_dev_(nullptr), w_gate_dev_(nullptr), w_up_dev_(nullptr), w_down_dev_(nullptr), all_freqs_cis_dev(nullptr), x_dev_(nullptr), x_norm_dev_(nullptr), x_resid1_dev_(nullptr), x_resid2_dev_(nullptr), q_dev_(nullptr), k_dev_(nullptr), v_dev_(nullptr), attn_out_dev_(nullptr), attn_proj_dev_(nullptr), gate_vec_dev_(nullptr), up_vec_dev_(nullptr), swiglu_vec_dev_(nullptr), mlp_down_dev_(nullptr), logits_dev_(nullptr), token_embedding_table_f32_dev_(nullptr), lm_head_f32_dev_(nullptr), w_q_f32_dev_(nullptr), w_k_f32_dev_(nullptr), w_v_f32_dev_(nullptr), w_o_f32_dev_(nullptr), w_gate_f32_dev_(nullptr), w_up_f32_dev_(nullptr), w_down_f32_dev_(nullptr)
#endif
{
    Logger::info("TinyLlamaModel constructor entered (with pre-loaded GGUFData). Model path placeholder: " + model_path_);
    this->config_.is_gguf_file_loaded = true; // Ensure this is set

    if (this->config_.num_cpu_offload_layers < 0) {
        this->config_.num_cpu_offload_layers = 0;
    }
    if (this->config_.num_hidden_layers > 0 && this->config_.num_cpu_offload_layers > this->config_.num_hidden_layers) {
        Logger::warning("Requested CPU offload layers (" + std::to_string(this->config_.num_cpu_offload_layers) +
                        ") exceeds total hidden layers (" + std::to_string(this->config_.num_hidden_layers) +
                        "). Clamping to " + std::to_string(this->config_.num_hidden_layers) + " layers on CPU (all CPU).");
        this->config_.num_cpu_offload_layers = this->config_.num_hidden_layers;
    }
    Logger::info("TinyLlamaModel (pre-loaded GGUF): Final clamped num_cpu_offload_layers = " + std::to_string(this->config_.num_cpu_offload_layers));

    initialize_weights(nullptr, gguf_data_.get()); // Pass raw GGUFData pointer
    initialize_gpu_and_rope();
    Logger::info("TinyLlamaModel (with pre-loaded GGUFData) constructed and initialized successfully.");
}

TinyLlamaModel::~TinyLlamaModel() {
#ifdef HAS_CUDA
  // Only perform GPU cleanup if GPU layers were actually used
  int active_num_gpu_layers = config_.num_hidden_layers - config_.num_cpu_offload_layers;
  if (active_num_gpu_layers > 0) {
    Logger::info("Freeing TinyLlamaModel CUDA resources...");
    if (cublas_handle_) {
      cublasStatus_t cublas_status = cublasDestroy(cublas_handle_);
      if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        Logger::error("cuBLAS handle destruction failed with error code: " +
                      std::to_string(cublas_status));
      }
      cublas_handle_ = nullptr;
      Logger::info("cuBLAS handle destroyed.");
    }
  } else {
    // CPU-only mode: just clean up cuBLAS handle if it exists
    if (cublas_handle_) {
      cublasStatus_t cublas_status = cublasDestroy(cublas_handle_);
      if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        Logger::error("cuBLAS handle destruction failed with error code: " +
                      std::to_string(cublas_status));
      }
      cublas_handle_ = nullptr;
    }
  }

  // Continue GPU cleanup only if GPU layers were active
  if (active_num_gpu_layers > 0) {
    if (final_norm_dev) {
      gpuErrchk(cudaFree(final_norm_dev));
      final_norm_dev = nullptr;
    }

    for (auto& layer : layers) {
      if (layer.input_layernorm_dev) {
        gpuErrchk(cudaFree(layer.input_layernorm_dev));
        layer.input_layernorm_dev = nullptr;
      }
      if (layer.post_attention_layernorm_dev) {
        gpuErrchk(cudaFree(layer.post_attention_layernorm_dev));
        layer.post_attention_layernorm_dev = nullptr;
      }
    }

  if (all_freqs_cis_dev) {
    gpuErrchk(cudaFree(all_freqs_cis_dev));
    all_freqs_cis_dev = nullptr;
  }
  if (token_embedding_table_dev_) {
    gpuErrchk(cudaFree(token_embedding_table_dev_));
    token_embedding_table_dev_ = nullptr;
  }
  if (lm_head_dev_) {
    gpuErrchk(cudaFree(lm_head_dev_));
    lm_head_dev_ = nullptr;
  }
  if (w_q_dev_) {
    gpuErrchk(cudaFree(w_q_dev_));
    w_q_dev_ = nullptr;
  }
  if (w_k_dev_) {
    gpuErrchk(cudaFree(w_k_dev_));
    w_k_dev_ = nullptr;
  }
  if (w_v_dev_) {
    gpuErrchk(cudaFree(w_v_dev_));
    w_v_dev_ = nullptr;
  }
  if (w_o_dev_) {
    gpuErrchk(cudaFree(w_o_dev_));
    w_o_dev_ = nullptr;
  }
  if (w_gate_dev_) {
    gpuErrchk(cudaFree(w_gate_dev_));
    w_gate_dev_ = nullptr;
  }
  if (w_up_dev_) {
    gpuErrchk(cudaFree(w_up_dev_));
    w_up_dev_ = nullptr;
  }
  if (w_down_dev_) {
    gpuErrchk(cudaFree(w_down_dev_));
    w_down_dev_ = nullptr;
  }
  if (token_embedding_table_f32_dev_) {
    gpuErrchk(cudaFree(token_embedding_table_f32_dev_));
    token_embedding_table_f32_dev_ = nullptr;
  }
  if (lm_head_f32_dev_) {
    gpuErrchk(cudaFree(lm_head_f32_dev_));
    lm_head_f32_dev_ = nullptr;
  }
  if (w_q_f32_dev_) {
    gpuErrchk(cudaFree(w_q_f32_dev_));
    w_q_f32_dev_ = nullptr;
  }
  if (w_k_f32_dev_) {
    gpuErrchk(cudaFree(w_k_f32_dev_));
    w_k_f32_dev_ = nullptr;
  }
  if (w_v_f32_dev_) {
    gpuErrchk(cudaFree(w_v_f32_dev_));
    w_v_f32_dev_ = nullptr;
  }
  if (w_o_f32_dev_) {
    gpuErrchk(cudaFree(w_o_f32_dev_));
    w_o_f32_dev_ = nullptr;
  }
  if (w_gate_f32_dev_) {
    gpuErrchk(cudaFree(w_gate_f32_dev_));
    w_gate_f32_dev_ = nullptr;
  }
  if (w_up_f32_dev_) {
    gpuErrchk(cudaFree(w_up_f32_dev_));
    w_up_f32_dev_ = nullptr;
  }
  if (w_down_f32_dev_) {
    gpuErrchk(cudaFree(w_down_f32_dev_));
    w_down_f32_dev_ = nullptr;
  }

  if (x_dev_) {
    gpuErrchk(cudaFree(x_dev_));
    x_dev_ = nullptr;
  }
  if (x_norm_dev_) {
    gpuErrchk(cudaFree(x_norm_dev_));
    x_norm_dev_ = nullptr;
  }
  if (x_resid1_dev_) {
    gpuErrchk(cudaFree(x_resid1_dev_));
    x_resid1_dev_ = nullptr;
  }
  if (x_resid2_dev_) {
    gpuErrchk(cudaFree(x_resid2_dev_));
    x_resid2_dev_ = nullptr;
  }
  if (q_dev_) {
    gpuErrchk(cudaFree(q_dev_));
    q_dev_ = nullptr;
  }
  if (k_dev_) {
    gpuErrchk(cudaFree(k_dev_));
    k_dev_ = nullptr;
  }
  if (v_dev_) {
    gpuErrchk(cudaFree(v_dev_));
    v_dev_ = nullptr;
  }
  if (attn_out_dev_) {
    gpuErrchk(cudaFree(attn_out_dev_));
    attn_out_dev_ = nullptr;
  }
  if (attn_proj_dev_) {
    gpuErrchk(cudaFree(attn_proj_dev_));
    attn_proj_dev_ = nullptr;
  }
  if (gate_vec_dev_) {
    gpuErrchk(cudaFree(gate_vec_dev_));
    gate_vec_dev_ = nullptr;
  }
  if (up_vec_dev_) {
    gpuErrchk(cudaFree(up_vec_dev_));
    up_vec_dev_ = nullptr;
  }
  if (swiglu_vec_dev_) {
    gpuErrchk(cudaFree(swiglu_vec_dev_));
    swiglu_vec_dev_ = nullptr;
  }
  if (mlp_down_dev_) {
    gpuErrchk(cudaFree(mlp_down_dev_));
    mlp_down_dev_ = nullptr;
  }
  if (logits_dev_) {
    gpuErrchk(cudaFree(logits_dev_));
    logits_dev_ = nullptr;
  }
  // Free KVCache dequantization buffers
  if (dequant_k_cache_buffer_dev_) {
    gpuErrchk(cudaFree(dequant_k_cache_buffer_dev_));
    dequant_k_cache_buffer_dev_ = nullptr;
  }
  if (dequant_v_cache_buffer_dev_) {
    gpuErrchk(cudaFree(dequant_v_cache_buffer_dev_));
    dequant_v_cache_buffer_dev_ = nullptr;
  }
  // Free selective KVCache dequantization buffers
  if (selective_k_dequant_buffer_dev_) {
    gpuErrchk(cudaFree(selective_k_dequant_buffer_dev_));
    selective_k_dequant_buffer_dev_ = nullptr;
  }
  if (selective_v_dequant_buffer_dev_) {
    gpuErrchk(cudaFree(selective_v_dequant_buffer_dev_));
    selective_v_dequant_buffer_dev_ = nullptr;
  }

    Logger::info("Freed persistent GPU workspace buffers.");
    Logger::info("Finished freeing TinyLlamaModel CUDA weight memory.");
  } else {
    Logger::info("CPU-only mode: No GPU resources to free.");
  }
#endif
}

std::vector<float> TinyLlamaModel::lookup_embedding(int token_id) {
  int hs = config_.hidden_size;
  int vs = config_.vocab_size;

  if (token_id < 0 || token_id >= vs) {
    Logger::error("Token ID out of bounds in lookup_embedding: " +
                  std::to_string(token_id));
    return std::vector<float>(hs, 0.0f);
  }

  std::vector<float> embedding_vec(hs, 0.0f);

  if (!embed_tokens_q4k.empty()) {
    if (hs % GGML_QK_K != 0) {
      Logger::error("Hidden size (" + std::to_string(hs) +
                    ") is not divisible by GGML_QK_K (" +
                    std::to_string(GGML_QK_K) + ") for Q4_K embedding lookup.");
      return embedding_vec;
    }

    size_t blocks_per_row = hs / GGML_QK_K;
    size_t start_block_idx = (size_t)token_id * blocks_per_row;
    size_t end_block_idx = start_block_idx + blocks_per_row;

    if (end_block_idx > embed_tokens_q4k.size()) {
      Logger::error(
          "Calculated block index out of bounds for Q4_K embedding table. "
          "Token: " +
          std::to_string(token_id) +
          ", StartBlock: " + std::to_string(start_block_idx) +
          ", EndBlock: " + std::to_string(end_block_idx) +
          ", TableSize: " + std::to_string(embed_tokens_q4k.size()));
      return embedding_vec;
    }

    float dequantized_block[GGML_QK_K];
    for (size_t block_n = 0; block_n < blocks_per_row; ++block_n) {
      dequantize_q4_k_m(&embed_tokens_q4k[start_block_idx + block_n],
                        dequantized_block, GGML_QK_K, false);

      size_t dest_offset = block_n * GGML_QK_K;

      size_t elements_to_copy = SAFE_MIN((size_t)GGML_QK_K, (size_t)(hs - dest_offset));
      std::memcpy(&embedding_vec[dest_offset], dequantized_block,
                  elements_to_copy * sizeof(float));
    }
    return embedding_vec;
  }

  else if (!embed_tokens_q8_0.empty()) {
    if (hs % GGML_QK8_0 != 0) {
      Logger::error("Hidden size (" + std::to_string(hs) +
                    ") is not divisible by GGML_QK8_0 (" +
                    std::to_string(GGML_QK8_0) +
                    ") for Q8_0 embedding lookup.");
      return embedding_vec;
    }
    size_t blocks_per_row = hs / GGML_QK8_0;
    size_t start_block_idx = (size_t)token_id * blocks_per_row;
    size_t end_block_idx = start_block_idx + blocks_per_row;

    if (end_block_idx > embed_tokens_q8_0.size()) {
      Logger::error(
          "Calculated block index out of bounds for Q8_0 embedding table. "
          "Token: " +
          std::to_string(token_id) +
          ", StartBlock: " + std::to_string(start_block_idx) +
          ", EndBlock: " + std::to_string(end_block_idx) +
          ", TableSize: " + std::to_string(embed_tokens_q8_0.size()));
      return embedding_vec;
    }

    float dequantized_block[GGML_QK8_0];
    
    for (size_t block_n = 0; block_n < blocks_per_row; ++block_n) {
      dequantize_q8_0_block(&embed_tokens_q8_0[start_block_idx + block_n],
                            dequantized_block);
      size_t dest_offset = block_n * GGML_QK8_0;
      size_t elements_to_copy = SAFE_MIN(static_cast<size_t>(GGML_QK8_0), static_cast<size_t>(hs - dest_offset));
      std::memcpy(&embedding_vec[dest_offset], dequantized_block,
                  elements_to_copy * sizeof(float));
      
    }
    
    // Log final embedding vector stats for first few tokens
    if (token_id < 2) {
      float sum = 0.0f, min_val = embedding_vec[0], max_val = embedding_vec[0];
      for (int i = 0; i < hs; ++i) {
        sum += embedding_vec[i];
        min_val = std::min(min_val, embedding_vec[i]);
        max_val = std::max(max_val, embedding_vec[i]);
      }
      Logger::info("[Q8_0_EMBED_FINAL] Token " + std::to_string(token_id) + 
                   " embedding stats: sum=" + std::to_string(sum) + 
                   ", mean=" + std::to_string(sum / hs) + 
                   ", min=" + std::to_string(min_val) + 
                   ", max=" + std::to_string(max_val) + 
                   ", first_4=[" + std::to_string(embedding_vec[0]) + 
                   ", " + std::to_string(embedding_vec[1]) + 
                   ", " + std::to_string(embedding_vec[2]) + 
                   ", " + std::to_string(embedding_vec[3]) + "]");
    }
    return embedding_vec;
  }

  else if (!embed_tokens_q6k.empty()) {
    if (hs % GGML_QK_K != 0) {
      Logger::error("Hidden size (" + std::to_string(hs) +
                    ") is not divisible by GGML_QK_K (" +
                    std::to_string(GGML_QK_K) + ") for Q6_K embedding lookup.");
      return embedding_vec;
    }
    size_t blocks_per_row = hs / GGML_QK_K;
    size_t start_block_idx = (size_t)token_id * blocks_per_row;
    size_t end_block_idx = start_block_idx + blocks_per_row;

    if (end_block_idx > embed_tokens_q6k.size()) {
      Logger::error(
          "Calculated block index out of bounds for Q6_K embedding table. "
          "Token: " +
          std::to_string(token_id) +
          ", StartBlock: " + std::to_string(start_block_idx) +
          ", EndBlock: " + std::to_string(end_block_idx) +
          ", TableSize: " + std::to_string(embed_tokens_q6k.size()));
      return embedding_vec;
    }

    float dequantized_block[GGML_QK_K];
    for (size_t block_n = 0; block_n < blocks_per_row; ++block_n) {
      dequantize_q6_k(&embed_tokens_q6k[start_block_idx + block_n],
                        dequantized_block, GGML_QK_K);
      size_t dest_offset = block_n * GGML_QK_K;
      size_t elements_to_copy = SAFE_MIN(static_cast<size_t>(GGML_QK_K), static_cast<size_t>(hs - dest_offset));
      std::memcpy(&embedding_vec[dest_offset], dequantized_block,
                  elements_to_copy * sizeof(float));
    }
    return embedding_vec;
  }

  else if (!embed_tokens_f32.empty()) {
    size_t offset = (size_t)token_id * hs;
    if (offset + hs > embed_tokens_f32.size()) {
      Logger::error("Embedding offset out of bounds in F32 lookup for token: " +
                    std::to_string(token_id));
      return embedding_vec;
    }

    std::copy(embed_tokens_f32.begin() + offset,
              embed_tokens_f32.begin() + offset + hs, embedding_vec.begin());
    return embedding_vec;

  } else if (!embed_tokens.empty()) {
    size_t offset = (size_t)token_id * hs;
    if (offset + hs > embed_tokens.size()) {
      Logger::error(
          "Embedding offset out of bounds in BF16 lookup for token: " +
          std::to_string(token_id));
      return embedding_vec;
    }
    std::vector<uint16_t> token_embedding_bf16(
        embed_tokens.begin() + offset, embed_tokens.begin() + offset + hs);

    embedding_vec = bf16vec_to_float_vec(token_embedding_bf16);
    return embedding_vec;

  } else {
    Logger::error(
        "No valid embedding table found (Q4_K, Q8_0, Q6_K, F32, BF16) for token: " +
        std::to_string(token_id));

    return embedding_vec;
  }
}

std::vector<float> TinyLlamaModel::forward(
    std::vector<float>& input,
                                           int n_tokens, KVCache* kv_cache,
    const std::vector<int>* attention_mask) {
  Logger::info("[CPU_FWD] Entered. Processing up to layer " + std::to_string(config_.num_cpu_offload_layers -1) + ". Input n_tokens: " + std::to_string(n_tokens));

  int hs = config_.hidden_size;
  int vs = config_.vocab_size;
  int is = config_.intermediate_size;
  int n_heads = config_.num_attention_heads;
  int n_kv_heads = config_.num_key_value_heads;
  int head_dim = hs / n_heads;
  float eps = config_.rms_norm_eps;
  int max_pos_embeddings = config_.max_position_embeddings;

  bool log_first_gen_step = (n_tokens == 0);
  bool log_this_step = log_first_gen_step || (n_tokens == 12) || (n_tokens == 13);

  // Layer processing loop - ONLY for CPU-offloaded layers
  for (int l = 0; l < config_.num_cpu_offload_layers; ++l) {
    Logger::info("[CPU_FWD_MEM] Starting layer " + std::to_string(l) + " processing");
    
    bool log_this_layer = log_this_step && (l == 0); // Log details only for layer 0 on specific steps
    if (log_this_layer) {
      Logger::info("[CPU_FWD] ------ START Layer " + std::to_string(l) +
                   " (pos=" + std::to_string(n_tokens) + ") ------");
      log_vector_summary("Layer " + std::to_string(l) + " Input (input)", input);
    }

    const auto& lw = layers[l];
    std::vector<float> x_norm_vec1(hs);
    const std::vector<float>& w_input_norm_vec =
        lw.input_layernorm_f32.empty()
            ? bf16vec_to_float_vec(lw.input_layernorm)
            : lw.input_layernorm_f32;
    rmsnorm_vector_cpu(input, w_input_norm_vec, x_norm_vec1, eps);
    
    Logger::info("[CPU_FWD_MEM] Layer " + std::to_string(l) + ": Allocating QKV vectors");
    std::vector<float> q_vec(hs), k_vec(n_kv_heads * head_dim), v_vec(n_kv_heads * head_dim);
    // Example: Q-projection (adapt for other projections and quantization types)
    bool enable_debug_logging = (l == 0); // Only log for first layer
    Logger::info("[CPU_FWD_MEM] Layer " + std::to_string(l) + ": About to ensure_q_proj_dequantized");
    ensure_q_proj_dequantized(l);
    Logger::info("[CPU_FWD_MEM] Layer " + std::to_string(l) + ": ensure_q_proj_dequantized completed");
    if (!lw.q_proj_f32.empty()) matvec_f32_f32_vector_cpu(lw.q_proj_f32, x_norm_vec1, q_vec, hs, hs);
    else if (!lw.q_proj_q8k.empty() && config_.is_gguf_file_loaded) matvec_q8k_f32_vector_cpu(lw.q_proj_q8k, x_norm_vec1, q_vec, hs, hs, enable_debug_logging);
    else if (!lw.q_proj_q8_0.empty() && config_.is_gguf_file_loaded) matvec_q8_0_f32_vector_cpu(lw.q_proj_q8_0, x_norm_vec1, q_vec, hs, hs, enable_debug_logging);
    else if (!lw.q_proj_q4k.empty() && config_.is_gguf_file_loaded) matvec_q4k_f32_vector_cpu(lw.q_proj_q4k, x_norm_vec1, q_vec, hs, hs, enable_debug_logging);
    else if (!lw.q_proj_q6k.empty() && config_.is_gguf_file_loaded) matvec_q6k_f32_vector_cpu(lw.q_proj_q6k, x_norm_vec1, q_vec, hs, hs, enable_debug_logging);
    else if (!lw.q_proj.empty()) matvec_bf16_f32_vector_cpu(lw.q_proj, x_norm_vec1, q_vec, hs, hs); // BF16 from SafeTensors
    else throw std::runtime_error("Layer " + std::to_string(l) + ": No Q proj weights (f32, q8k, q8, q4k, q6k, bf16) for CPU");
    
    // ... K, V projections ...
    Logger::info("[CPU_FWD_MEM] Layer " + std::to_string(l) + ": About to ensure_k_proj_dequantized");
    ensure_k_proj_dequantized(l);
    Logger::info("[CPU_FWD_MEM] Layer " + std::to_string(l) + ": ensure_k_proj_dequantized completed");
    if (!lw.k_proj_f32.empty()) matvec_f32_f32_vector_cpu(lw.k_proj_f32, x_norm_vec1, k_vec, n_kv_heads * head_dim, hs);
    else if (!lw.k_proj_q8k.empty() && config_.is_gguf_file_loaded) matvec_q8k_f32_vector_cpu(lw.k_proj_q8k, x_norm_vec1, k_vec, n_kv_heads * head_dim, hs, enable_debug_logging);
    else if (!lw.k_proj_q8_0.empty() && config_.is_gguf_file_loaded) matvec_q8_0_f32_vector_cpu(lw.k_proj_q8_0, x_norm_vec1, k_vec, n_kv_heads * head_dim, hs, enable_debug_logging);
    else if (!lw.k_proj_q4k.empty() && config_.is_gguf_file_loaded) matvec_q4k_f32_vector_cpu(lw.k_proj_q4k, x_norm_vec1, k_vec, n_kv_heads * head_dim, hs, enable_debug_logging);
    else if (!lw.k_proj_q6k.empty() && config_.is_gguf_file_loaded) matvec_q6k_f32_vector_cpu(lw.k_proj_q6k, x_norm_vec1, k_vec, n_kv_heads * head_dim, hs, enable_debug_logging);
    else if (!lw.k_proj.empty()) matvec_bf16_f32_vector_cpu(lw.k_proj, x_norm_vec1, k_vec, n_kv_heads * head_dim, hs);
    else throw std::runtime_error("Layer " + std::to_string(l) + ": No K proj weights (f32, q8k, q8, q4k, q6k, bf16) for CPU");

    Logger::info("[CPU_FWD_MEM] Layer " + std::to_string(l) + ": About to ensure_v_proj_dequantized");
    ensure_v_proj_dequantized(l);
    Logger::info("[CPU_FWD_MEM] Layer " + std::to_string(l) + ": ensure_v_proj_dequantized completed");
    if (!lw.v_proj_f32.empty()) matvec_f32_f32_vector_cpu(lw.v_proj_f32, x_norm_vec1, v_vec, n_kv_heads * head_dim, hs);
    else if (!lw.v_proj_q8k.empty() && config_.is_gguf_file_loaded) matvec_q8k_f32_vector_cpu(lw.v_proj_q8k, x_norm_vec1, v_vec, n_kv_heads * head_dim, hs, enable_debug_logging);
    else if (!lw.v_proj_q8_0.empty() && config_.is_gguf_file_loaded) matvec_q8_0_f32_vector_cpu(lw.v_proj_q8_0, x_norm_vec1, v_vec, n_kv_heads * head_dim, hs, enable_debug_logging);
    else if (!lw.v_proj_q4k.empty() && config_.is_gguf_file_loaded) matvec_q4k_f32_vector_cpu(lw.v_proj_q4k, x_norm_vec1, v_vec, n_kv_heads * head_dim, hs, enable_debug_logging);
    else if (!lw.v_proj_q6k.empty() && config_.is_gguf_file_loaded) matvec_q6k_f32_vector_cpu(lw.v_proj_q6k, x_norm_vec1, v_vec, n_kv_heads * head_dim, hs, enable_debug_logging);
    else if (!lw.v_proj.empty()) matvec_bf16_f32_vector_cpu(lw.v_proj, x_norm_vec1, v_vec, n_kv_heads * head_dim, hs);
    else throw std::runtime_error("Layer " + std::to_string(l) + ": No V proj weights (f32, q8k, q8, q4k, q6k, bf16) for CPU");

    apply_rope_vector(q_vec, n_heads, head_dim, n_tokens, precomputed_freqs_cis_, max_pos_embeddings, config_.is_gguf_file_loaded);
    apply_rope_vector(k_vec, n_kv_heads, head_dim, n_tokens, precomputed_freqs_cis_, max_pos_embeddings, config_.is_gguf_file_loaded);

    // KV Cache update
    if (kv_cache) {
        if (static_cast<size_t>(l) < kv_cache->layers.size()) {
            KVCacheLayer& kv_layer = kv_cache->layers[l];
            size_t layer_max_seq_len = static_cast<size_t>(kv_cache->max_seq_len_config_);            
            if (static_cast<size_t>(n_tokens) >= layer_max_seq_len && layer_max_seq_len > 0) {
                Logger::error("KV Cache access out of bounds in CPU forward. Layer " + std::to_string(l) + 
                              ", n_tokens: " + std::to_string(n_tokens) + 
                              ", configured layer_max_seq_len: " + std::to_string(layer_max_seq_len) + ". Skipping KV update.");
            } else if (layer_max_seq_len == 0 && n_tokens > 0) {
                 Logger::error("KV Cache layer_max_seq_len is 0, but n_tokens > 0. Layer " + std::to_string(l) + ". Skipping KV update.");
            } else {
                 for(int h=0; h < n_kv_heads; ++h) {
                     std::copy(k_vec.begin() + h * head_dim, k_vec.begin() + (h+1) * head_dim, kv_layer.k.begin() + n_tokens * (n_kv_heads * head_dim) + h * head_dim);
                     std::copy(v_vec.begin() + h * head_dim, v_vec.begin() + (h+1) * head_dim, kv_layer.v.begin() + n_tokens * (n_kv_heads * head_dim) + h * head_dim);
                 }
            }
        } else {
            Logger::error("KV Cache layer index " + std::to_string(l) + " out of bounds for kv_cache->layers.size() = " + std::to_string(kv_cache->layers.size()));
        }
    }
    
    std::vector<float> attn_out_vec(hs);
    std::vector<float> x_resid1_vec = input; // Store residual
    float att_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    std::fill(attn_out_vec.begin(), attn_out_vec.end(), 0.0f);
    for (int h = 0; h < n_heads; ++h) {
        std::vector<float> q_head(head_dim);
        std::copy(q_vec.begin() + h * head_dim, q_vec.begin() + (h + 1) * head_dim, q_head.begin());
        std::vector<float> current_multihead_attn_out(head_dim, 0.0f);
        int kv_cache_num_kv_heads = n_kv_heads; // from KVCache struct if available
        int kv_group = n_heads / kv_cache_num_kv_heads;
        int kv_head_idx = h / kv_group;

        if (kv_cache && static_cast<size_t>(l) < kv_cache->layers.size()) {
            const KVCacheLayer& kv_layer = kv_cache->layers[l];
            int current_seq_len = n_tokens + 1;
            std::vector<float> scores(current_seq_len);
            for (int t = 0; t < current_seq_len; ++t) {
                float score = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    score += q_head[d] * kv_layer.k[t * (n_kv_heads * head_dim) + kv_head_idx * head_dim + d];
                }
                scores[t] = score * att_scale;
            }
            softmax_vector_cpu(scores, scores); // In-place softmax
            for (int t = 0; t < current_seq_len; ++t) {
                for (int d = 0; d < head_dim; ++d) {
                    current_multihead_attn_out[d] += scores[t] * kv_layer.v[t * (n_kv_heads * head_dim) + kv_head_idx * head_dim + d];
                }
            }
        }
        std::copy(current_multihead_attn_out.begin(), current_multihead_attn_out.end(), attn_out_vec.begin() + h * head_dim);
    }
    

    std::vector<float> attn_proj_vec(hs);
    Logger::info("[CPU_FWD_MEM] Layer " + std::to_string(l) + ": About to ensure_o_proj_dequantized");
    ensure_o_proj_dequantized(l);
    Logger::info("[CPU_FWD_MEM] Layer " + std::to_string(l) + ": ensure_o_proj_dequantized completed");
    if(!lw.o_proj_f32.empty()) matvec_f32_f32_vector_cpu(lw.o_proj_f32, attn_out_vec, attn_proj_vec, hs, hs);
    else if (!lw.o_proj_q8k.empty() && config_.is_gguf_file_loaded) matvec_q8k_f32_vector_cpu(lw.o_proj_q8k, attn_out_vec, attn_proj_vec, hs, hs, enable_debug_logging);
    else if (!lw.o_proj_q8_0.empty() && config_.is_gguf_file_loaded) matvec_q8_0_f32_vector_cpu(lw.o_proj_q8_0, attn_out_vec, attn_proj_vec, hs, hs, enable_debug_logging);
    else if (!lw.o_proj_q4k.empty() && config_.is_gguf_file_loaded) matvec_q4k_f32_vector_cpu(lw.o_proj_q4k, attn_out_vec, attn_proj_vec, hs, hs, enable_debug_logging);
    else if (!lw.o_proj_q6k.empty() && config_.is_gguf_file_loaded) matvec_q6k_f32_vector_cpu(lw.o_proj_q6k, attn_out_vec, attn_proj_vec, hs, hs, enable_debug_logging);
    else if(!lw.o_proj.empty()) matvec_bf16_f32_vector_cpu(lw.o_proj, attn_out_vec, attn_proj_vec, hs, hs);
    else throw std::runtime_error("Layer " + std::to_string(l) + ": No O proj weights (f32, q8k, q8, q4k, q6k, bf16) for CPU");

    for(size_t i=0; i<input.size(); ++i) input[i] = x_resid1_vec[i] + attn_proj_vec[i]; // Update input by reference

    // MLP part
    std::vector<float> x_norm_vec2(hs);
    std::vector<float> x_resid2_vec = input; // Store residual for MLP
    const std::vector<float>& w_post_attn_norm_vec =
        lw.post_attention_layernorm_f32.empty()
            ? bf16vec_to_float_vec(lw.post_attention_layernorm)
            : lw.post_attention_layernorm_f32;
    rmsnorm_vector_cpu(input, w_post_attn_norm_vec, x_norm_vec2, eps);

    std::vector<float> gate_vec(is), up_vec(is);
    // Gate-projection
    Logger::info("[CPU_FWD_MEM] Layer " + std::to_string(l) + ": About to ensure_gate_proj_dequantized");
    ensure_gate_proj_dequantized(l);
    Logger::info("[CPU_FWD_MEM] Layer " + std::to_string(l) + ": ensure_gate_proj_dequantized completed");
    if(!lw.gate_proj_f32.empty()) matvec_f32_f32_vector_cpu(lw.gate_proj_f32, x_norm_vec2, gate_vec, is, hs);
    else if (!lw.gate_proj_q8k.empty() && config_.is_gguf_file_loaded) matvec_q8k_f32_vector_cpu(lw.gate_proj_q8k, x_norm_vec2, gate_vec, is, hs, enable_debug_logging);
    else if (!lw.gate_proj_q8_0.empty() && config_.is_gguf_file_loaded) matvec_q8_0_f32_vector_cpu(lw.gate_proj_q8_0, x_norm_vec2, gate_vec, is, hs, enable_debug_logging);
    else if (!lw.gate_proj_q4k.empty() && config_.is_gguf_file_loaded) matvec_q4k_f32_vector_cpu(lw.gate_proj_q4k, x_norm_vec2, gate_vec, is, hs, enable_debug_logging);
    else if (!lw.gate_proj_q6k.empty() && config_.is_gguf_file_loaded) matvec_q6k_f32_vector_cpu(lw.gate_proj_q6k, x_norm_vec2, gate_vec, is, hs, enable_debug_logging);
    else if(!lw.gate_proj.empty()) matvec_bf16_f32_vector_cpu(lw.gate_proj, x_norm_vec2, gate_vec, is, hs);
    else throw std::runtime_error("Layer " + std::to_string(l) + ": No Gate proj weights (f32, q8k, q8, q4k, q6k, bf16) for CPU");

    // Up-projection
    Logger::info("[CPU_FWD_MEM] Layer " + std::to_string(l) + ": About to ensure_up_proj_dequantized");
    ensure_up_proj_dequantized(l);
    Logger::info("[CPU_FWD_MEM] Layer " + std::to_string(l) + ": ensure_up_proj_dequantized completed");
    if(!lw.up_proj_f32.empty()) matvec_f32_f32_vector_cpu(lw.up_proj_f32, x_norm_vec2, up_vec, is, hs);
    else if (!lw.up_proj_q8k.empty() && config_.is_gguf_file_loaded) matvec_q8k_f32_vector_cpu(lw.up_proj_q8k, x_norm_vec2, up_vec, is, hs, enable_debug_logging);
    else if (!lw.up_proj_q8_0.empty() && config_.is_gguf_file_loaded) matvec_q8_0_f32_vector_cpu(lw.up_proj_q8_0, x_norm_vec2, up_vec, is, hs, enable_debug_logging);
    else if (!lw.up_proj_q4k.empty() && config_.is_gguf_file_loaded) matvec_q4k_f32_vector_cpu(lw.up_proj_q4k, x_norm_vec2, up_vec, is, hs, enable_debug_logging);
    else if (!lw.up_proj_q6k.empty() && config_.is_gguf_file_loaded) matvec_q6k_f32_vector_cpu(lw.up_proj_q6k, x_norm_vec2, up_vec, is, hs, enable_debug_logging);
    else if(!lw.up_proj.empty()) matvec_bf16_f32_vector_cpu(lw.up_proj, x_norm_vec2, up_vec, is, hs);
    else throw std::runtime_error("Layer " + std::to_string(l) + ": No Up proj weights (f32, q8k, q8, q4k, q6k, bf16) for CPU");

    std::vector<float> silu_out_vec(is);
    silu_cpu(gate_vec, silu_out_vec);

    std::vector<float> swiglu_result_vec(is);
    for(size_t i=0; i<is; ++i) swiglu_result_vec[i] = silu_out_vec[i] * up_vec[i];

    std::vector<float> mlp_out_vec(hs);
    // Down-projection
    Logger::info("[CPU_FWD_MEM] Layer " + std::to_string(l) + ": About to ensure_down_proj_dequantized");
    ensure_down_proj_dequantized(l);
    Logger::info("[CPU_FWD_MEM] Layer " + std::to_string(l) + ": ensure_down_proj_dequantized completed");
    if(!lw.down_proj_f32.empty()) matvec_f32_f32_vector_cpu(lw.down_proj_f32, swiglu_result_vec, mlp_out_vec, hs, is);
    else if (!lw.down_proj_q8k.empty() && config_.is_gguf_file_loaded) matvec_q8k_f32_vector_cpu(lw.down_proj_q8k, swiglu_result_vec, mlp_out_vec, hs, is, enable_debug_logging);
    else if (!lw.down_proj_q8_0.empty() && config_.is_gguf_file_loaded) matvec_q8_0_f32_vector_cpu(lw.down_proj_q8_0, swiglu_result_vec, mlp_out_vec, hs, is, enable_debug_logging);
    else if (!lw.down_proj_q4k.empty() && config_.is_gguf_file_loaded) matvec_q4k_f32_vector_cpu(lw.down_proj_q4k, swiglu_result_vec, mlp_out_vec, hs, is, enable_debug_logging);
    else if (!lw.down_proj_q6k.empty() && config_.is_gguf_file_loaded) matvec_q6k_f32_vector_cpu(lw.down_proj_q6k, swiglu_result_vec, mlp_out_vec, hs, is, enable_debug_logging);
    else if(!lw.down_proj.empty()) matvec_bf16_f32_vector_cpu(lw.down_proj, swiglu_result_vec, mlp_out_vec, hs, is);
    else throw std::runtime_error("Layer " + std::to_string(l) + ": No Down proj weights (f32, q8k, q8, q4k, q6k, bf16) for CPU");

    for(size_t i=0; i<input.size(); ++i) input[i] = x_resid2_vec[i] + mlp_out_vec[i]; // Update input by reference
    

    if (log_this_layer) {
      Logger::info("[CPU_FWD] ------ END Layer " + std::to_string(l) +
                   " (pos=" + std::to_string(n_tokens) + ") ------");
    }
    
    // Memory-efficient layer eviction: Clear dequantized weights from 2 layers ago
    // This maintains a small memory footprint while preserving performance
    // EXCEPTION: Don't clear GPU layers as they're needed for concatenated F32 weight loading
    if (config_.enable_memory_efficient_layers && l >= 2) {
        int layer_to_clear = l - 2;
        int first_gpu_layer = config_.num_cpu_offload_layers;
        if (layer_to_clear < first_gpu_layer) {
            clear_layer_dequantized_weights(layer_to_clear);
        }
    }
  }

  if (config_.num_cpu_offload_layers == config_.num_hidden_layers) {
    Logger::info("[CPU_FWD] All layers processed on CPU. Performing final RMSNorm and Logits.");
  const std::vector<float>& w_final_norm_vec =
      final_norm_f32.empty() ? bf16vec_to_float_vec(final_norm)
                             : final_norm_f32;
  std::vector<float> x_final_norm_vec(hs);
  rmsnorm_vector_cpu(input, w_final_norm_vec, x_final_norm_vec, eps);

  std::vector<float> logits(vs);
  ensure_lm_head_dequantized();
    bool enable_lm_head_debug_logging = true; // Always log LM head for debugging
    if (!lm_head_f32.empty()) matvec_f32_f32_vector_cpu(lm_head_f32, x_final_norm_vec, logits, vs, hs);
    else if (!lm_head_q8k.empty() && config_.is_gguf_file_loaded) matvec_q8k_f32_vector_cpu(lm_head_q8k, x_final_norm_vec, logits, vs, hs, enable_lm_head_debug_logging);
    else if (!lm_head_q8_0.empty() && config_.is_gguf_file_loaded) matvec_q8_0_f32_vector_cpu(lm_head_q8_0, x_final_norm_vec, logits, vs, hs, enable_lm_head_debug_logging);
    else if (!lm_head_q4k.empty() && config_.is_gguf_file_loaded) matvec_q4k_f32_vector_cpu(lm_head_q4k, x_final_norm_vec, logits, vs, hs, enable_lm_head_debug_logging);
    else if (!lm_head_q6k.empty() && config_.is_gguf_file_loaded) matvec_q6k_f32_vector_cpu(lm_head_q6k, x_final_norm_vec, logits, vs, hs, enable_lm_head_debug_logging);
    else if (!lm_head.empty()) matvec_bf16_f32_vector_cpu(lm_head, x_final_norm_vec, logits, vs, hs); // Fallback for BF16 SafeTensors
    else throw std::runtime_error("No valid LM Head weights (f32, q8k, q8, q4k, q6k, bf16) found for CPU final stage.");

  if (log_this_step || log_first_gen_step) {
        log_vector_summary("[CPU_FWD] Final Logits (all CPU, pos=" + std::to_string(n_tokens) + ")", logits, 15);
    }
    return logits; // Return final logits if all layers were CPU
  }

  Logger::info("[CPU_FWD] Finished processing " + std::to_string(config_.num_cpu_offload_layers) + " CPU layers. Output is intermediate activation.");
  return input; // Return the intermediate activations if not all layers were processed here.
}

int TinyLlamaModel::get_vocab_size() const { return config_.vocab_size; }

#ifdef HAS_CUDA
std::vector<float> TinyLlamaModel::forward_device(
    float* x_input_dev,
    int pos, KVCache* kv_cache,
    const std::vector<int>* attention_mask, cudaStream_t stream) {
  
  int hs = config_.hidden_size;
  int vs = config_.vocab_size;
  int n_heads = config_.num_attention_heads;
  int n_kv_heads = config_.num_key_value_heads;
  if (n_heads == 0) {
    Logger::fatal("Number of attention heads is zero during forward_device.");
    throw std::runtime_error("Division by zero: n_heads is zero.");
  }
  int head_dim = hs / n_heads;
  int total_model_layers = config_.num_hidden_layers;
  int num_cpu_layers = config_.num_cpu_offload_layers;
  int num_gpu_layers = total_model_layers - num_cpu_layers;

  if (num_gpu_layers <= 0) {
      Logger::warning("forward_device called with no GPU layers to process (num_gpu_layers = " + std::to_string(num_gpu_layers) + "). Returning empty.");
      return {};
  }
  if (!x_input_dev) {
      Logger::error("forward_device called with null x_input_dev. This should be model_->x_dev_.");
      return {};
  }
  if (!kv_cache) {
      Logger::error("forward_device called with null KVCache.");
      return {};
  }

  int is = config_.intermediate_size;
  float eps = config_.rms_norm_eps;
  std::vector<float> h_x_input_dev(config_.hidden_size);

  cublasStatus_t stream_status = cublasSetStream(cublas_handle_, stream);
    gpuErrchk(cudaMemcpyAsync(h_x_input_dev.data(), x_input_dev, config_.hidden_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
    gpuErrchk(cudaStreamSynchronize(stream)); // Ensure copy is done before logging
    
  if (stream_status != CUBLAS_STATUS_SUCCESS) {
    Logger::error("cublasSetStream failed in forward_device");
    return {};
  }

  float* current_x_dev = x_input_dev; // This is effectively model_->x_dev_

  for (int l_gpu_idx = 0; l_gpu_idx < num_gpu_layers; ++l_gpu_idx) {
    int l_model_idx = num_cpu_layers + l_gpu_idx; // Actual layer index in the model
    
    // Layer-specific norm weights are indexed by the model layer index (l_model_idx)
    const float* lw_in_norm_dev = layers[l_model_idx].input_layernorm_dev;
    const float* lw_post_norm_dev = layers[l_model_idx].post_attention_layernorm_dev;

    gpuErrchk(cudaMemcpyAsync(x_resid1_dev_, x_dev_, hs * sizeof(float),
                              cudaMemcpyDeviceToDevice, stream));

    if (!lw_in_norm_dev) { 
        throw std::runtime_error("[TM::fw_dev pos=" + std::to_string(pos) + " L" + std::to_string(l_model_idx) + "] Error: input_layernorm_dev is nullptr. GPU layer cannot proceed.");
    }

    rmsnorm_vector_cuda(x_dev_, lw_in_norm_dev, x_norm_dev_, hs, eps, stream);

    // Use concatenated weights for optimal performance
    ensure_f32_concatenated_weights_loaded();
    
    if (w_q_f32_dev_ && w_k_f32_dev_ && w_v_f32_dev_) {
      const float* w_q_layer_ptr = w_q_f32_dev_ + (size_t)l_gpu_idx * hs * hs;
      const float* w_k_layer_ptr = w_k_f32_dev_ + (size_t)l_gpu_idx * n_kv_heads * head_dim * hs;
      const float* w_v_layer_ptr = w_v_f32_dev_ + (size_t)l_gpu_idx * n_kv_heads * head_dim * hs;
      
      matvec_f32_f32_cuda(cublas_handle_, w_q_layer_ptr, x_norm_dev_,
                          q_dev_, hs, hs, stream);
      matvec_f32_f32_cuda(cublas_handle_, w_k_layer_ptr, x_norm_dev_,
                          k_dev_, n_kv_heads * head_dim, hs, stream);
      matvec_f32_f32_cuda(cublas_handle_, w_v_layer_ptr, x_norm_dev_,
                           v_dev_, n_kv_heads * head_dim, hs, stream);
    } else {
      Logger::error("GPU L" + std::to_string(l_model_idx) + " (gpu_idx " + std::to_string(l_gpu_idx) + "): No valid concatenated QKV weights."); return {};
    }

    
    // RoPE Application:
    rope_cuda(q_dev_, n_heads, head_dim, all_freqs_cis_dev, pos, config_.is_gguf_file_loaded, stream);
    rope_cuda(k_dev_, n_kv_heads, head_dim, all_freqs_cis_dev, pos, config_.is_gguf_file_loaded, stream);


    // K/V Cache Update Logic
    if (static_cast<size_t>(l_model_idx) < kv_cache->layers.size()) {
        KVCacheLayer& current_kv_layer = kv_cache->layers[l_model_idx];
        if (config_.use_kvcache_quantization) {
            // Quantize and store K, V for each head
            for (int kvh = 0; kvh < n_kv_heads; ++kvh) {
                const float* current_k_head_ptr_fp32 = k_dev_ + kvh * head_dim;
                const float* current_v_head_ptr_fp32 = v_dev_ + kvh * head_dim;

                size_t token_head_offset_quant = (static_cast<size_t>(pos) * n_kv_heads + kvh) * head_dim;
                int8_t* k_quant_target_ptr = current_kv_layer.k_dev_quantized + token_head_offset_quant;
                int8_t* v_quant_target_ptr = current_kv_layer.v_dev_quantized + token_head_offset_quant;

                size_t scale_offset = static_cast<size_t>(pos) * n_kv_heads + kvh;
                float* k_scale_target_ptr = current_kv_layer.k_dev_scales + scale_offset;
                float* v_scale_target_ptr = current_kv_layer.v_dev_scales + scale_offset;

                quantize_fp32_to_int8_symmetric_per_tensor_cuda(
                    current_k_head_ptr_fp32, k_quant_target_ptr, k_scale_target_ptr, head_dim, stream);
                quantize_fp32_to_int8_symmetric_per_tensor_cuda(
                    current_v_head_ptr_fp32, v_quant_target_ptr, v_scale_target_ptr, head_dim, stream);
            }
        } else {
            // Store FP32 K, V directly (original path, now using k_dev_fp32, v_dev_fp32)
            for (int kvh = 0; kvh < n_kv_heads; ++kvh) {
                const float* current_k_head_ptr = k_dev_ + kvh * head_dim;
                const float* current_v_head_ptr = v_dev_ + kvh * head_dim;

                update_kv_cache_cuda(current_kv_layer.k_dev_fp32, current_k_head_ptr, pos,
                                   kvh, kv_cache->allocated_max_seq_len,
                                   kv_cache->allocated_num_kv_heads, 
                                   kv_cache->allocated_head_dim, stream); 

                update_kv_cache_cuda(current_kv_layer.v_dev_fp32, current_v_head_ptr, pos,
                                   kvh, kv_cache->allocated_max_seq_len,
                                   kv_cache->allocated_num_kv_heads,
                                   kv_cache->allocated_head_dim, stream);
            }
        }

    } else {
        Logger::error("KVCache layer index " + std::to_string(l_model_idx) + " out of bounds for kv_cache->layers access in forward_device.");
        return {}; // Or throw
    }

    float scale = 1.0f / SAFE_SQRT(static_cast<float>(head_dim));
    
    const float* attention_k_cache_ptr_dev = nullptr;
    const float* attention_v_cache_ptr_dev = nullptr;
    KVCacheLayer& attention_kv_layer = kv_cache->layers[l_model_idx]; 

    if (config_.use_kvcache_quantization) {
        // Always use selective dequantization approach - no upfront dequantization
            // The attention function will handle dequantization on-demand
            Logger::info("[GPU L" + std::to_string(l_model_idx) + "] Using SELECTIVE KVCache dequantization");
    } else {
        attention_k_cache_ptr_dev = attention_kv_layer.k_dev_fp32;
        attention_v_cache_ptr_dev = attention_kv_layer.v_dev_fp32;
    }

    float current_attention_scale = 1.0f / sqrtf((float)head_dim);
    
    if (config_.use_kvcache_quantization && 
        selective_k_dequant_buffer_dev_ && selective_v_dequant_buffer_dev_) {
        attention_cuda_selective_dequant(
            q_dev_,
            attention_kv_layer.k_dev_quantized,
            attention_kv_layer.v_dev_quantized,
            attention_kv_layer.k_dev_scales,
            attention_kv_layer.v_dev_scales,
            selective_k_dequant_buffer_dev_,
            selective_v_dequant_buffer_dev_,
            attn_out_dev_,
            config_.num_attention_heads,
            pos + 1,
            head_dim,
            current_attention_scale,
            kv_cache->allocated_max_seq_len,
            config_.num_key_value_heads,
            stream
        );
    } else {
    attention_cuda(
        q_dev_,                            
        attention_k_cache_ptr_dev,       
        attention_v_cache_ptr_dev,       
        attn_out_dev_,                    
        config_.num_attention_heads,     
        pos + 1,                         
        head_dim,                        
        current_attention_scale,         
        kv_cache->allocated_max_seq_len, 
        config_.num_key_value_heads,     
        stream                           
    );
    }


    if (w_o_f32_dev_) {
      const float* lw_o_proj_f32_dev = w_o_f32_dev_ + (size_t)l_gpu_idx * hs * hs;
      matvec_f32_f32_cuda(cublas_handle_, lw_o_proj_f32_dev, attn_out_dev_, attn_proj_dev_, hs, hs, stream);
    } else {
      Logger::error("GPU L" + std::to_string(l_model_idx) + " (gpu_idx " + std::to_string(l_gpu_idx) + "): No valid O proj weights (FP32/BF16)."); return {};
    }

    add_residual_cuda(attn_proj_dev_, x_resid1_dev_, current_x_dev, hs, stream); 

    gpuErrchk(cudaMemcpyAsync(x_resid2_dev_, current_x_dev, hs * sizeof(float), cudaMemcpyDeviceToDevice, stream)); 

    if (!lw_post_norm_dev) { Logger::error("Missing post_attention_layernorm_dev for GPU layer model_idx=" + std::to_string(l_model_idx)); return {}; }
    rmsnorm_vector_cuda(current_x_dev, lw_post_norm_dev, x_norm_dev_, hs, eps, stream); 

    if (w_o_f32_dev_) {
      const float* w_o_layer_ptr = w_o_f32_dev_ + (size_t)l_gpu_idx * hs * hs;
      matvec_f32_f32_cuda(cublas_handle_, w_o_layer_ptr, attn_out_dev_, attn_proj_dev_, hs, hs, stream);
    } else {
      Logger::error("GPU L" + std::to_string(l_model_idx) + ": No valid O projection weights."); return {};
    }

    add_residual_cuda(attn_proj_dev_, x_resid1_dev_, current_x_dev, hs, stream);
    gpuErrchk(cudaMemcpyAsync(x_resid2_dev_, current_x_dev, hs * sizeof(float), cudaMemcpyDeviceToDevice, stream));

    if (!lw_post_norm_dev) { 
      Logger::error("Missing post_attention_layernorm_dev for GPU layer model_idx=" + std::to_string(l_model_idx)); return {}; 
    }
    rmsnorm_vector_cuda(current_x_dev, lw_post_norm_dev, x_norm_dev_, hs, eps, stream);

    if (w_gate_f32_dev_ && w_up_f32_dev_) {
      const float* w_gate_layer_ptr = w_gate_f32_dev_ + (size_t)l_gpu_idx * is * hs;
      const float* w_up_layer_ptr = w_up_f32_dev_ + (size_t)l_gpu_idx * is * hs;
      
      matvec_f32_f32_cuda(cublas_handle_, w_gate_layer_ptr, x_norm_dev_,
                          gate_vec_dev_, is, hs, stream);
      matvec_f32_f32_cuda(cublas_handle_, w_up_layer_ptr, x_norm_dev_,
                           up_vec_dev_, is, hs, stream);
    } else {
      Logger::error("GPU L" + std::to_string(l_model_idx) + ": No valid Gate/Up projection weights.");
      return {};
    }

    swiglu_cuda(gate_vec_dev_, up_vec_dev_, swiglu_vec_dev_, is, stream);

    if (w_down_f32_dev_) {
      const float* w_down_layer_ptr = w_down_f32_dev_ + (size_t)l_gpu_idx * hs * is;
      matvec_f32_f32_cuda(cublas_handle_, w_down_layer_ptr, swiglu_vec_dev_,
                          mlp_down_dev_, hs, is, stream);
    } else {
      Logger::error("GPU L" + std::to_string(l_model_idx) + ": No valid Down projection weights.");
      return {};
    }


    add_residual_cuda(mlp_down_dev_, x_resid2_dev_, current_x_dev, hs, stream); 

  } // End of layer loop

  rmsnorm_vector_cuda(x_dev_, final_norm_dev, x_norm_dev_, hs, eps, stream);
  

  ensure_lm_head_dequantized();
  if (lm_head_dev_) { 
    matvec_bf16_f32_cuda(cublas_handle_, lm_head_dev_, x_norm_dev_, logits_dev_,
                         vs, hs, stream);
  } else {
    Logger::error("LM head (lm_head_dev_ for BF16) is null. Cannot calculate logits on GPU.");
    return {};
  }

  gpuErrchk(cudaStreamSynchronize(stream)); // Ensure all ops including LM head are done before memcpy DtoH




  std::vector<float> logits(vs);
  gpuErrchk(cudaMemcpy(logits.data(), logits_dev_, vs * sizeof(float),
                       cudaMemcpyDeviceToHost));
  return logits;
}

#endif // HAS_CUDA

void map_gguf_weights(const GGUFData& gguf, TinyLlamaModel& model) {
  Logger::info("Mapping GGUF weights to model fields (ULTRA-OPTIMIZED VERSION)...");
    
  const uint8_t* actual_data_block_start = nullptr;
  
  // Determine which data source to use
  if (gguf.mapped_tensor_data != nullptr && gguf.mapped_tensor_data_size > 0) {
    const uint8_t* mmap_buffer_start = static_cast<const uint8_t*>(gguf.mapped_tensor_data);
    actual_data_block_start = mmap_buffer_start + gguf.offset_diff_for_mmap;
    Logger::info("map_gguf_weights: Using mmap mode (ZERO-COPY). Size: " +
               std::to_string(gguf.mapped_tensor_data_size) + " bytes.");
  } else if (!gguf.tensor_data.empty()) {
    actual_data_block_start = gguf.tensor_data.data();
    Logger::info("map_gguf_weights: Using non-mmap mode. Size: " +
                 std::to_string(gguf.tensor_data.size()) + " bytes.");
  } else {
    Logger::error("GGUF tensor data is not available. Cannot map weights.");
    return;
  }

  const size_t num_tensors = gguf.tensor_infos_map.size();
  Logger::info("Processing " + std::to_string(num_tensors) + " tensors with ultra-optimized parallel mapping...");

  // Pre-allocate containers to avoid reallocations during parallel processing
  std::vector<std::pair<std::string, GGUFTensorInfo>> tensor_pairs;
  tensor_pairs.reserve(num_tensors);
  for (const auto& pair : gguf.tensor_infos_map) {
    tensor_pairs.emplace_back(pair.first, pair.second);
  }

  // Reserve capacity for major model containers to reduce allocations
  const size_t typical_blocks = 4096;
  if (model.lm_head_q8_0.capacity() == 0) model.lm_head_q8_0.reserve(32768);
  if (model.embed_tokens_q8_0.capacity() == 0) model.embed_tokens_q8_0.reserve(32768);
  
  for (auto& layer : model.layers) {
    if (layer.q_proj_q8_0.capacity() == 0) layer.q_proj_q8_0.reserve(typical_blocks);
    if (layer.k_proj_q8_0.capacity() == 0) layer.k_proj_q8_0.reserve(typical_blocks);
    if (layer.v_proj_q8_0.capacity() == 0) layer.v_proj_q8_0.reserve(typical_blocks);
    if (layer.o_proj_q8_0.capacity() == 0) layer.o_proj_q8_0.reserve(typical_blocks);
    if (layer.gate_proj_q8_0.capacity() == 0) layer.gate_proj_q8_0.reserve(typical_blocks);
    if (layer.up_proj_q8_0.capacity() == 0) layer.up_proj_q8_0.reserve(typical_blocks);
    if (layer.down_proj_q8_0.capacity() == 0) layer.down_proj_q8_0.reserve(typical_blocks);
  }

  // BLAZING FAST: Sort tensors by type and process in bulk
  std::vector<size_t> global_tensor_indices;
  std::vector<std::vector<size_t>> layer_tensor_indices(model.layers.size());
  
  global_tensor_indices.reserve(10); // output.weight, token_embd.weight, output_norm.weight, etc.
  for (auto& layer_indices : layer_tensor_indices) {
    layer_indices.reserve(9); // 7 weights + 2 norms per layer
  }
  
  // ULTRA-FAST categorization without string operations
  for (size_t i = 0; i < tensor_pairs.size(); ++i) {
    const std::string& name = tensor_pairs[i].first;
    if (name[0] == 'o' || name[0] == 't') { // output.weight, token_embd.weight, output_norm.weight
      global_tensor_indices.push_back(i);
    } else if (name.size() > 4 && name[0] == 'b' && name[1] == 'l' && name[2] == 'k' && name[3] == '.') {
      // Extract layer index without substr - MUCH faster
      size_t layer_start = 4;
      size_t layer_end = name.find('.', layer_start);
      if (layer_end != std::string::npos) {
        int layer_idx = 0;
        for (size_t pos = layer_start; pos < layer_end; ++pos) {
          layer_idx = layer_idx * 10 + (name[pos] - '0');
        }
        if (layer_idx >= 0 && static_cast<size_t>(layer_idx) < model.layers.size()) {
          layer_tensor_indices[layer_idx].push_back(i);
        }
      }
    }
  }
  
  std::atomic<int> processed_count{0};
  std::atomic<int> error_count{0};

  // Process global tensors sequentially (small count, avoid overhead)
  for (size_t idx : global_tensor_indices) {
    try {
    const std::string& target_field_key = tensor_pairs[idx].first;
    const GGUFTensorInfo& info = tensor_pairs[idx].second;
    const uint8_t* tensor_data_ptr = actual_data_block_start + info.offset;

      // Global tensors with optimized type dispatch
    if (target_field_key == "output.weight") {
        switch (info.type) {
          case GGMLType::GGML_TYPE_Q6_K: {
        size_t num_blocks = info.size_in_bytes / sizeof(block_q6_K);
        model.lm_head_q6k.resize(num_blocks);
        std::memcpy(model.lm_head_q6k.data(), tensor_data_ptr, info.size_in_bytes);
            break;
          }
          case GGMLType::GGML_TYPE_Q4_K: {
        size_t num_blocks = info.size_in_bytes / sizeof(block_q4_K);
        model.lm_head_q4k.resize(num_blocks);
        std::memcpy(model.lm_head_q4k.data(), tensor_data_ptr, info.size_in_bytes);
            break;
          }
          case GGMLType::GGML_TYPE_Q8_0: {
        size_t num_blocks = info.size_in_bytes / sizeof(block_q8_0);
        model.lm_head_q8_0.resize(num_blocks);
        std::memcpy(model.lm_head_q8_0.data(), tensor_data_ptr, info.size_in_bytes);
            break;
          }
          case GGMLType::GGML_TYPE_Q8_K: {
        size_t num_blocks = info.size_in_bytes / sizeof(block_q8_K);
        model.lm_head_q8k.resize(num_blocks);
        std::memcpy(model.lm_head_q8k.data(), tensor_data_ptr, info.size_in_bytes);
            break;
          }
          case GGMLType::GGML_TYPE_F32: {
        size_t num_elements = info.size_in_bytes / sizeof(float);
        model.lm_head_f32.resize(num_elements);
        std::memcpy(model.lm_head_f32.data(), tensor_data_ptr, info.size_in_bytes);
            break;
          }
        }
        processed_count++;
      continue;
      }
      
      if (target_field_key == "token_embd.weight") {
        switch (info.type) {
          case GGMLType::GGML_TYPE_Q4_K: {
        size_t num_blocks = info.size_in_bytes / sizeof(block_q4_K);
        model.embed_tokens_q4k.resize(num_blocks);
        std::memcpy(model.embed_tokens_q4k.data(), tensor_data_ptr, info.size_in_bytes);
            break;
          }
          case GGMLType::GGML_TYPE_Q8_0: {
        size_t num_blocks = info.size_in_bytes / sizeof(block_q8_0);
        model.embed_tokens_q8_0.resize(num_blocks);
        std::memcpy(model.embed_tokens_q8_0.data(), tensor_data_ptr, info.size_in_bytes);
            break;
          }
          case GGMLType::GGML_TYPE_Q8_K: {
        size_t num_blocks = info.size_in_bytes / sizeof(block_q8_K);
        model.embed_tokens_q8k.resize(num_blocks);
        std::memcpy(model.embed_tokens_q8k.data(), tensor_data_ptr, info.size_in_bytes);
            break;
          }
          case GGMLType::GGML_TYPE_Q6_K: {
        size_t num_blocks = info.size_in_bytes / sizeof(block_q6_K);
        model.embed_tokens_q6k.resize(num_blocks);
        std::memcpy(model.embed_tokens_q6k.data(), tensor_data_ptr, info.size_in_bytes);
            break;
          }
          case GGMLType::GGML_TYPE_F32: {
        size_t num_elements = info.size_in_bytes / sizeof(float);
        model.embed_tokens_f32.resize(num_elements);
        std::memcpy(model.embed_tokens_f32.data(), tensor_data_ptr, info.size_in_bytes);
            break;
          }
        }
        processed_count++;
      continue;
      }
      
      if (target_field_key == "output_norm.weight") {
      if (info.type == GGMLType::GGML_TYPE_F32) {
        size_t num_elements = info.size_in_bytes / sizeof(float);
        model.final_norm_f32.resize(num_elements);
        std::memcpy(model.final_norm_f32.data(), tensor_data_ptr, info.size_in_bytes);
      }
        processed_count++;
      continue;
    }

    } catch (const std::exception& e) {
      error_count++;
    }
  }
  
  // ULTRA-FAST: Process layers in parallel with pre-sorted tensors
  #pragma omp parallel for schedule(static) if(model.layers.size() > 4)
  for (size_t layer_idx = 0; layer_idx < layer_tensor_indices.size(); ++layer_idx) {
    const auto& layer_indices = layer_tensor_indices[layer_idx];
    if (layer_indices.empty()) continue;
    
    LayerWeights& layer = model.layers[layer_idx];
    
    try {
      for (size_t idx : layer_indices) {
        const std::string& name = tensor_pairs[idx].first;
        const GGUFTensorInfo& info = tensor_pairs[idx].second;
        const uint8_t* tensor_data_ptr = actual_data_block_start + info.offset;
        
        // BLAZING FAST: Direct character matching (faster than hashing)
        const size_t last_dot = name.find_last_of('.');
        if (last_dot == std::string::npos) continue;
        
        const char* field = name.c_str() + name.find('.', 4) + 1;
        
                // BLAZING FAST: Direct character-based dispatch without string operations
        #define FAST_COPY_WEIGHT(target_vec, block_type) \
          target_vec.resize(info.size_in_bytes / sizeof(block_type)); \
          std::memcpy(target_vec.data(), tensor_data_ptr, info.size_in_bytes);
        
        // IMPROVED: Pattern matching based on tensor name structure
        const char* name_cstr = name.c_str();
        const size_t name_len = name.length();
        
        if (name_len > 10 && name.find("attn_") != std::string::npos) {
          if (name.find("attn_q.weight") != std::string::npos) {
            switch (info.type) {
              case GGMLType::GGML_TYPE_Q8_0: FAST_COPY_WEIGHT(layer.q_proj_q8_0, block_q8_0); break;
              case GGMLType::GGML_TYPE_Q4_K: FAST_COPY_WEIGHT(layer.q_proj_q4k, block_q4_K); break;
              case GGMLType::GGML_TYPE_Q6_K: FAST_COPY_WEIGHT(layer.q_proj_q6k, block_q6_K); break;
              case GGMLType::GGML_TYPE_Q8_K: FAST_COPY_WEIGHT(layer.q_proj_q8k, block_q8_K); break;
              case GGMLType::GGML_TYPE_BF16: FAST_COPY_WEIGHT(layer.q_proj, uint16_t); break;
            }
          } else if (name.find("attn_k.weight") != std::string::npos) {
            switch (info.type) {
              case GGMLType::GGML_TYPE_Q8_0: FAST_COPY_WEIGHT(layer.k_proj_q8_0, block_q8_0); break;
              case GGMLType::GGML_TYPE_Q4_K: FAST_COPY_WEIGHT(layer.k_proj_q4k, block_q4_K); break;
              case GGMLType::GGML_TYPE_Q6_K: FAST_COPY_WEIGHT(layer.k_proj_q6k, block_q6_K); break;
              case GGMLType::GGML_TYPE_Q8_K: FAST_COPY_WEIGHT(layer.k_proj_q8k, block_q8_K); break;
              case GGMLType::GGML_TYPE_BF16: FAST_COPY_WEIGHT(layer.k_proj, uint16_t); break;
            }
          } else if (name.find("attn_v.weight") != std::string::npos) {
            switch (info.type) {
              case GGMLType::GGML_TYPE_Q8_0: FAST_COPY_WEIGHT(layer.v_proj_q8_0, block_q8_0); break;
              case GGMLType::GGML_TYPE_Q4_K: FAST_COPY_WEIGHT(layer.v_proj_q4k, block_q4_K); break;
              case GGMLType::GGML_TYPE_Q6_K: FAST_COPY_WEIGHT(layer.v_proj_q6k, block_q6_K); break;
              case GGMLType::GGML_TYPE_Q8_K: FAST_COPY_WEIGHT(layer.v_proj_q8k, block_q8_K); break;
              case GGMLType::GGML_TYPE_BF16: FAST_COPY_WEIGHT(layer.v_proj, uint16_t); break;
            }
          } else if (name.find("attn_output.weight") != std::string::npos) {
            switch (info.type) {
              case GGMLType::GGML_TYPE_Q8_0: FAST_COPY_WEIGHT(layer.o_proj_q8_0, block_q8_0); break;
              case GGMLType::GGML_TYPE_Q4_K: FAST_COPY_WEIGHT(layer.o_proj_q4k, block_q4_K); break;
              case GGMLType::GGML_TYPE_Q6_K: FAST_COPY_WEIGHT(layer.o_proj_q6k, block_q6_K); break;
              case GGMLType::GGML_TYPE_Q8_K: FAST_COPY_WEIGHT(layer.o_proj_q8k, block_q8_K); break;
              case GGMLType::GGML_TYPE_BF16: FAST_COPY_WEIGHT(layer.o_proj, uint16_t); break;
            }
          } else if (name.find("attn_norm.weight") != std::string::npos && info.type == GGMLType::GGML_TYPE_F32) {
            FAST_COPY_WEIGHT(layer.input_layernorm_f32, float);
          }
        } else if (name_len > 10 && name.find("ffn_") != std::string::npos) {
          if (name.find("ffn_gate.weight") != std::string::npos) {
            switch (info.type) {
              case GGMLType::GGML_TYPE_Q8_0: FAST_COPY_WEIGHT(layer.gate_proj_q8_0, block_q8_0); break;
              case GGMLType::GGML_TYPE_Q4_K: FAST_COPY_WEIGHT(layer.gate_proj_q4k, block_q4_K); break;
              case GGMLType::GGML_TYPE_Q6_K: FAST_COPY_WEIGHT(layer.gate_proj_q6k, block_q6_K); break;
              case GGMLType::GGML_TYPE_Q8_K: FAST_COPY_WEIGHT(layer.gate_proj_q8k, block_q8_K); break;
              case GGMLType::GGML_TYPE_BF16: FAST_COPY_WEIGHT(layer.gate_proj, uint16_t); break;
            }
          } else if (name.find("ffn_up.weight") != std::string::npos) {
            switch (info.type) {
              case GGMLType::GGML_TYPE_Q8_0: FAST_COPY_WEIGHT(layer.up_proj_q8_0, block_q8_0); break;
              case GGMLType::GGML_TYPE_Q4_K: FAST_COPY_WEIGHT(layer.up_proj_q4k, block_q4_K); break;
              case GGMLType::GGML_TYPE_Q6_K: FAST_COPY_WEIGHT(layer.up_proj_q6k, block_q6_K); break;
              case GGMLType::GGML_TYPE_Q8_K: FAST_COPY_WEIGHT(layer.up_proj_q8k, block_q8_K); break;
              case GGMLType::GGML_TYPE_BF16: FAST_COPY_WEIGHT(layer.up_proj, uint16_t); break;
            }
          } else if (name.find("ffn_down.weight") != std::string::npos) {
            switch (info.type) {
              case GGMLType::GGML_TYPE_Q8_0: FAST_COPY_WEIGHT(layer.down_proj_q8_0, block_q8_0); break;
              case GGMLType::GGML_TYPE_Q4_K: FAST_COPY_WEIGHT(layer.down_proj_q4k, block_q4_K); break;
              case GGMLType::GGML_TYPE_Q6_K: FAST_COPY_WEIGHT(layer.down_proj_q6k, block_q6_K); break;
              case GGMLType::GGML_TYPE_Q8_K: FAST_COPY_WEIGHT(layer.down_proj_q8k, block_q8_K); break;
              case GGMLType::GGML_TYPE_BF16: FAST_COPY_WEIGHT(layer.down_proj, uint16_t); break;
            }
          } else if (name.find("ffn_norm.weight") != std::string::npos && info.type == GGMLType::GGML_TYPE_F32) {
            FAST_COPY_WEIGHT(layer.post_attention_layernorm_f32, float);
          }
        }
        
        /*
        // OLD complex dispatch - keeping for reference
        if (name_len > 10) {
          char field_char = name_cstr[name.find('.', 4) + 5];
          
          switch (field_char) {
         }
         */
         
         #undef FAST_COPY_WEIGHT
      }
      processed_count++;
    } catch (const std::exception& e) {
      error_count++;
    }
  }

  Logger::info("Finished mapping GGUF weights: " + std::to_string(processed_count.load()) + "/" + 
               std::to_string(num_tensors) + " tensors processed successfully (errors: " + 
               std::to_string(error_count.load()) + ") with ultra-optimized parallel mapping");
}


void TinyLlamaModel::initialize_rope_freqs() {
  Logger::info("[ROPE_FREQ_ENTRY] Entered initialize_rope_freqs.");

  Logger::info("[ROPE_FREQ_CHECK] num_attention_heads: " + std::to_string(config_.num_attention_heads));
  if (config_.num_attention_heads == 0) {
    Logger::error("Cannot initialize RoPE frequencies: num_attention_heads is zero.");
    return;
  }
  int head_dim = config_.hidden_size / config_.num_attention_heads;
  Logger::info("[ROPE_FREQ_CHECK] calculated head_dim: " + std::to_string(head_dim));
  if (head_dim == 0) {
    Logger::error("Cannot initialize RoPE frequencies: calculated head_dim is zero.");
    return;
  }
  Logger::info("[ROPE_FREQ_CHECK] head_dim % 2 check. head_dim: " + std::to_string(head_dim));
  if (head_dim % 2 != 0) {
    Logger::error("Cannot initialize RoPE frequencies: head_dim must be even.");
    return;
  }

  // Log parameters used for RoPE initialization
  Logger::info("[ROPE_INIT] Initializing RoPE with head_dim=" + std::to_string(head_dim) +
               ", configured max_pos_emb=" + std::to_string(config_.max_position_embeddings) +
               ", using internal rope::MAX_SEQUENCE_LENGTH=" + std::to_string(rope::MAX_SEQUENCE_LENGTH) +
               ", configured rope_theta=" + std::to_string(config_.rope_theta));


  if (precomputed_freqs_cis_.empty()) { 
    int max_seq_len = rope::MAX_SEQUENCE_LENGTH; // Or config_.max_position_embeddings if preferred
    size_t required_size = (static_cast<size_t>(max_seq_len) * head_dim) / 2;
    if (required_size == 0) {
        Logger::warning("RoPE precomputation resulted in zero size. Max seq len: " + 
                        std::to_string(max_seq_len) + ", head_dim: " + std::to_string(head_dim));
        return;
    }
    precomputed_freqs_cis_.resize(required_size);
    
    float rope_theta = config_.rope_theta > 0 ? config_.rope_theta : rope::ROPE_THETA;

    for (int pos = 0; pos < max_seq_len; ++pos) {
      for (int i = 0; i < head_dim; i += 2) {
        float freq = 1.0f / std::pow(rope_theta, float(i) / head_dim);
        float val = static_cast<float>(pos) * freq;
        float cos_val = std::cos(val);
        float sin_val = std::sin(val);
        size_t flat_idx = (static_cast<size_t>(pos) * head_dim / 2) + (i / 2);
        if (flat_idx < precomputed_freqs_cis_.size()){
            precomputed_freqs_cis_[flat_idx] = {cos_val, sin_val};
        } else {
            Logger::error("RoPE precomputation index out of bounds: " + std::to_string(flat_idx) + 
                          " vs size " + std::to_string(precomputed_freqs_cis_.size()));
            return; 
        }
      }
    }
    Logger::info("Precomputed RoPE frequencies on CPU. Size: " + std::to_string(precomputed_freqs_cis_.size()));
  } else {
      Logger::info("RoPE frequencies already precomputed.");
  }
}

static void update_kv_cache_batch_cpu(
    KVCache* kv_cache,
    int layer_idx, // The absolute model layer index
    const std::vector<float>& k_batch_for_layer, // [num_tokens, num_kv_heads * head_dim] for this layer
    const std::vector<float>& v_batch_for_layer, // [num_tokens, num_kv_heads * head_dim] for this layer
    int num_tokens_in_batch,
    int start_pos_in_sequence, // Starting position of this batch in the overall sequence
    int num_kv_heads,
    int head_dim
) {
    if (!kv_cache) {
        Logger::error("update_kv_cache_batch_cpu: KVCache is null.");
        return;
    }
    if (layer_idx < 0 || static_cast<size_t>(layer_idx) >= kv_cache->layers.size()) {
        Logger::error("update_kv_cache_batch_cpu: layer_idx " + std::to_string(layer_idx) + " is out of bounds for KVCache layers (size " + std::to_string(kv_cache->layers.size()) + ").");
        return;
    }
    Logger::info("[CPU_KV_UPDATE] Layer=" + std::to_string(layer_idx) + 
                ", start_pos=" + std::to_string(start_pos_in_sequence) + 
                ", num_tokens=" + std::to_string(num_tokens_in_batch) +
                ", k_batch_first_vals=[" + std::to_string(k_batch_for_layer[0]) + 
                "," + std::to_string(k_batch_for_layer[1]) + "," + std::to_string(k_batch_for_layer[2]) + "]");
    KVCacheLayer& layer_cache = kv_cache->layers[layer_idx];
    int kv_dim = num_kv_heads * head_dim;

    if (k_batch_for_layer.size() != static_cast<size_t>(num_tokens_in_batch * kv_dim)) {
        Logger::error("[KV_BATCH_UPDATE L" + std::to_string(layer_idx) + "] k_batch_for_layer size mismatch. Expected " +
                      std::to_string(num_tokens_in_batch * kv_dim) + ", got " + std::to_string(k_batch_for_layer.size()));
        return;
    }
    
    if (v_batch_for_layer.size() != static_cast<size_t>(num_tokens_in_batch * kv_dim)) {
        Logger::error("[KV_BATCH_UPDATE L" + std::to_string(layer_idx) + "] v_batch_for_layer size mismatch. Expected " +
                      std::to_string(num_tokens_in_batch * kv_dim) + ", got " + std::to_string(v_batch_for_layer.size()));
        return;
    }
    
    size_t expected_total_elements_in_layer_cache = static_cast<size_t>(kv_cache->max_seq_len_config_) * static_cast<size_t>(kv_cache->max_batch_size) * kv_dim;
    if (layer_cache.k.size() != expected_total_elements_in_layer_cache || layer_cache.v.size() != expected_total_elements_in_layer_cache) {
        Logger::error("[KV_BATCH_UPDATE L" + std::to_string(layer_idx) + 
                      "] Precondition failed: Layer cache not sized to max_seq_len_config. K size: " + std::to_string(layer_cache.k.size()) +
                      ", V size: " + std::to_string(layer_cache.v.size()) + 
                      ", Expected size: " + std::to_string(expected_total_elements_in_layer_cache) +
                      ". Check KVCache::initialize.");
        return; // Critical error, cannot safely proceed
    }
    for (int token_idx_in_batch = 0; token_idx_in_batch < num_tokens_in_batch; ++token_idx_in_batch) {
        size_t current_token_batch_offset = static_cast<size_t>(token_idx_in_batch) * kv_dim;

        // Calculate the global sequence position in the cache where this token's KV vector will be written
        // This matches the GPU implementation: sequential position regardless of sequence boundaries
        int global_seq_pos = start_pos_in_sequence + token_idx_in_batch;

        if (global_seq_pos >= kv_cache->max_seq_len_config_ * kv_cache->max_batch_size) {
            Logger::error("[KV_BATCH_UPDATE L" + std::to_string(layer_idx) + 
                          "] Error: global_seq_pos (" + std::to_string(global_seq_pos) +
                          ") is out of bounds for total cache size. Skipping update for this token.");
            continue; 
        }

        size_t destination_offset_in_layer_cache = static_cast<size_t>(global_seq_pos) * kv_dim;
        // Log K cache update
        size_t k_size_before = layer_cache.k.size(); // This should be the full max_seq_len size
        std::string k_vals_to_log = " vals to copy: ";
        for(int i = 0; i < std::min(3, kv_dim); ++i) { k_vals_to_log += std::to_string(k_batch_for_layer[current_token_batch_offset + i]) + " "; }
        if (kv_dim > 3) k_vals_to_log += "...";

        
        std::copy(k_batch_for_layer.begin() + current_token_batch_offset,
                  k_batch_for_layer.begin() + current_token_batch_offset + kv_dim,
                  layer_cache.k.begin() + destination_offset_in_layer_cache);
        

        // Log V cache update
        size_t v_size_before = layer_cache.v.size(); // This should be the full max_seq_len size
        std::string v_vals_to_log = " vals to copy: ";
        for(int i = 0; i < std::min(3, kv_dim); ++i) { v_vals_to_log += std::to_string(v_batch_for_layer[current_token_batch_offset + i]) + " "; }
        if (kv_dim > 3) v_vals_to_log += "...";

        std::copy(v_batch_for_layer.begin() + current_token_batch_offset,
                  v_batch_for_layer.begin() + current_token_batch_offset + kv_dim,
                  layer_cache.v.begin() + destination_offset_in_layer_cache);

    }
    
}
static void attention_batch_cpu(
    const std::vector<float>& q_batch_roped, // [num_tokens, num_q_heads * head_dim] (hs dimension)
    KVCacheLayer& current_layer_kv_cache,    // K and V for the current layer
    std::vector<float>& batch_attn_output,   // Output: [num_tokens, num_q_heads * head_dim] (hs dimension)
    int num_tokens_in_batch,
    int start_pos_in_sequence,               // The sequence position where this batch begins
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float attention_scale
) {
    size_t expected_q_size = (size_t)num_tokens_in_batch * num_q_heads * head_dim;
    if (q_batch_roped.size() != expected_q_size) {
        Logger::error("[ATTN_BATCH_CPU] q_batch_roped size mismatch. Expected: " + std::to_string(expected_q_size) +
                      ", Got: " + std::to_string(q_batch_roped.size()));
        std::fill(batch_attn_output.begin(), batch_attn_output.end(), 0.0f); // Ensure output is zeroed if error
        return;
    }
    Logger::info("[ATTENTION_BATCH_CPU_ENTRY] Called with num_tokens=" + std::to_string(num_tokens_in_batch));
    size_t expected_output_size = (size_t)num_tokens_in_batch * num_q_heads * head_dim;
    batch_attn_output.assign(expected_output_size, 0.0f);



    for (int token_idx = 0; token_idx < num_tokens_in_batch; ++token_idx) {
        size_t q_token_offset = (size_t)token_idx * num_q_heads * head_dim;
        size_t attn_out_token_offset = (size_t)token_idx * num_q_heads * head_dim;
        int current_token_absolute_pos = start_pos_in_sequence + token_idx;

        for (int h_q = 0; h_q < num_q_heads; ++h_q) {
            const float* q_head_for_token_ptr = q_batch_roped.data() + q_token_offset + (h_q * head_dim);
            int kv_group_head_idx = h_q / (num_q_heads / num_kv_heads); 
            
            bool log_details_for_this_head = (token_idx == 0 && h_q == 0); // Log details only for 1st token, 1st Q-head


            int history_len = current_token_absolute_pos + 1;
            if (history_len <= 0) { // Should not happen if current_token_absolute_pos >= 0
                 Logger::warning("[ATTN_BATCH_CPU] Token_idx " + std::to_string(token_idx) + ", Q_Head " + std::to_string(h_q) +
                                 ": history_len is " + std::to_string(history_len) + ". Skipping score calculation for this head.");
                continue;
            }
            std::vector<float> scores(history_len);

            for (int t_hist = 0; t_hist < history_len; ++t_hist) { 
                size_t k_cache_offset = ((size_t)t_hist * num_kv_heads + kv_group_head_idx) * head_dim;
                                if (token_idx == 0 && h_q == 0 && t_hist < 3) {
                  Logger::info("[CPU_ATTN_MEM] T" + std::to_string(token_idx) + "_H" + std::to_string(h_q) + 
                              " accessing K_cache[pos=" + std::to_string(t_hist) + ",kv_head=" + std::to_string(kv_group_head_idx) + 
                              "]: offset=" + std::to_string(k_cache_offset) + 
                              ", k_vals=[" + std::to_string(current_layer_kv_cache.k[k_cache_offset]) + 
                              "," + std::to_string(current_layer_kv_cache.k[k_cache_offset + 1]) + 
                              "," + std::to_string(current_layer_kv_cache.k[k_cache_offset + 2]) + "]");
              }
                if (k_cache_offset + head_dim > current_layer_kv_cache.k.size()) {
                     Logger::error("[ATTN_BATCH_CPU] K cache out of bounds. Token_idx " + std::to_string(token_idx) +
                                   " (abs_pos " + std::to_string(current_token_absolute_pos) + "), Q_Head " + std::to_string(h_q) +
                                   ", history_pos " + std::to_string(t_hist) +
                                   ". Required k_cache_offset " + std::to_string(k_cache_offset + head_dim) +
                                   " > cache_k_size " + std::to_string(current_layer_kv_cache.k.size()));
                    scores[t_hist] = -std::numeric_limits<float>::infinity(); 
                    continue;
                }

                float current_dot_product = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    current_dot_product += q_head_for_token_ptr[d] * current_layer_kv_cache.k[k_cache_offset + d];
                }
                                if (token_idx == 0 && h_q == 0 && t_hist < 2) {
                Logger::info("[CPU_ATTN_SCORE] T0_H0 pos=" + std::to_string(t_hist) + 
                            ", q_vals=[" + std::to_string(q_head_for_token_ptr[0]) + 
                            "," + std::to_string(q_head_for_token_ptr[1]) + "] " +
                            ", k_vals=[" + std::to_string(current_layer_kv_cache.k[k_cache_offset]) + 
                            "," + std::to_string(current_layer_kv_cache.k[k_cache_offset + 1]) + "]" +
                            ", dot=" + std::to_string(current_dot_product) + ", scale=" + std::to_string(attention_scale));
            }
                scores[t_hist] = current_dot_product * attention_scale;

            }

            softmax_vector_cpu(scores, scores); 
            if (token_idx == 0 && h_q == 0) {
                std::string scores_str = "";
                for (int i = 0; i < std::min(3, (int)scores.size()); i++) {
                    scores_str += std::to_string(scores[i]) + " ";
                }
                Logger::info("[CPU_SOFTMAX] T0_H0 first_3_probs=[" + scores_str + "]");
            }
            float* current_attn_out_head_ptr = batch_attn_output.data() + attn_out_token_offset + (h_q * head_dim);

            for (int t_hist = 0; t_hist < history_len; ++t_hist) {
                if (scores[t_hist] == -std::numeric_limits<float>::infinity() || scores[t_hist] == 0.0f) continue;

                size_t v_cache_offset = ((size_t)t_hist * num_kv_heads + kv_group_head_idx) * head_dim;
                if (v_cache_offset + head_dim > current_layer_kv_cache.v.size()) {
                     Logger::error("[ATTN_BATCH_CPU] V cache out of bounds. Token_idx " + std::to_string(token_idx) +
                                   " (abs_pos " + std::to_string(current_token_absolute_pos) + "), Q_Head " + std::to_string(h_q) +
                                   ", history_pos " + std::to_string(t_hist) +
                                   ". Required v_cache_offset " + std::to_string(v_cache_offset + head_dim) +
                                   " > cache_v_size " + std::to_string(current_layer_kv_cache.v.size()));
                    continue; 
                }

                for (int d = 0; d < head_dim; ++d) {
                    float val_before = (log_details_for_this_head && t_hist < 2 && d < 2) ? current_attn_out_head_ptr[d] : 0.0f;
                    current_attn_out_head_ptr[d] += scores[t_hist] * current_layer_kv_cache.v[v_cache_offset + d];
                }
            }
        } 
    } 
}

std::vector<float> TinyLlamaModel::forward_cpu_logits_batch(
    const std::vector<float>& final_batch_activations, // [num_tokens, hidden_size]
    int num_tokens_in_batch) {


    if (final_batch_activations.size() != (size_t)num_tokens_in_batch * config_.hidden_size) {
        Logger::error("[CPU_LOGITS_BATCH] final_batch_activations size mismatch. Expected: " +
                      std::to_string((size_t)num_tokens_in_batch * config_.hidden_size) + " Got: " +
                      std::to_string(final_batch_activations.size()));
        return {};
    }

    int hs = config_.hidden_size;
    int vs = config_.vocab_size;
    float eps = config_.rms_norm_eps;

    // 1. Final RMSNorm
    std::vector<float> final_batch_norm_out(num_tokens_in_batch * hs);
    const std::vector<float>& w_final_norm_vec =
        final_norm_f32.empty() ? bf16vec_to_float_vec(final_norm)
                               : final_norm_f32;
    if (w_final_norm_vec.empty()) {
         Logger::error("[CPU_LOGITS_BATCH] Final RMSNorm weights are empty (neither f32 nor bf16 available).");
         return {};
    }

    rmsnorm_batch_cpu(final_batch_activations, w_final_norm_vec, final_batch_norm_out,
                      num_tokens_in_batch, hs, eps);
    
    // 2. Batched LM Head multiplication
    std::vector<float> batch_logits_out(num_tokens_in_batch * vs);

    if (!lm_head_f32.empty()) {
        Logger::info("[CPU_LOGITS_BATCH] Using F32 LM Head weights.");
        matmul_f32_f32_batch_cpu(lm_head_f32, final_batch_norm_out, batch_logits_out,
                                 num_tokens_in_batch, vs, hs);
    } else if (!lm_head_q8_0.empty() && config_.is_gguf_file_loaded) {
        Logger::info("[CPU_LOGITS_BATCH] Using Q8_0 LM Head weights.");
        matmul_q8_0_f32_batch_cpu(lm_head_q8_0, final_batch_norm_out, batch_logits_out,
                                   num_tokens_in_batch, vs, hs);
    } else if (!lm_head_q6k.empty() && config_.is_gguf_file_loaded) {
        Logger::info("[CPU_LOGITS_BATCH] Using Q6_K LM Head weights.");
        matmul_q6k_f32_batch_cpu(lm_head_q6k, final_batch_norm_out, batch_logits_out,
                                  num_tokens_in_batch, vs, hs);
    } else if (!lm_head_q4k.empty() && config_.is_gguf_file_loaded) {
        Logger::info("[CPU_LOGITS_BATCH] Using Q4_K LM Head weights.");
        matmul_q4k_f32_batch_cpu(lm_head_q4k, final_batch_norm_out, batch_logits_out,
                                  num_tokens_in_batch, vs, hs);
    } else if (!lm_head.empty()) { // BF16 SafeTensors weights
        Logger::info("[CPU_LOGITS_BATCH] Using BF16 LM Head weights (converting to F32 for matmul).");
        std::vector<float> lm_head_f32_temp = bf16vec_to_float_vec(lm_head);
        if (lm_head_f32_temp.empty()) {
            Logger::error("[CPU_LOGITS_BATCH] Failed to convert BF16 LM Head to F32.");
            return {};
        }
        matmul_f32_f32_batch_cpu(lm_head_f32_temp, final_batch_norm_out, batch_logits_out,
                                 num_tokens_in_batch, vs, hs);
    } else {
        Logger::error("[CPU_LOGITS_BATCH] No valid LM Head weights found (F32, Q8_0, Q6_K, Q4_K, BF16).");
        return {};
    }
    
    return batch_logits_out;
}

static void update_kv_cache_batch_cpu_sequence_aware(
    KVCache* kv_cache,
    int layer_idx,
    const std::vector<float>& k_batch_for_layer,
    const std::vector<float>& v_batch_for_layer,
    int num_tokens_in_batch,
    const std::vector<int>& sequence_indices,
    const std::vector<int>& position_in_sequence,
    int num_kv_heads,
    int head_dim
) {
    if (!kv_cache) {
        Logger::error("update_kv_cache_batch_cpu_sequence_aware: KVCache is null.");
        return;
    }
    if (layer_idx < 0 || static_cast<size_t>(layer_idx) >= kv_cache->layers.size()) {
        Logger::error("update_kv_cache_batch_cpu_sequence_aware: layer_idx " + std::to_string(layer_idx) + 
                      " is out of bounds for KVCache layers (size " + std::to_string(kv_cache->layers.size()) + ").");
        return;
    }
    
    KVCacheLayer& layer_cache = kv_cache->layers[layer_idx];
    int kv_dim = num_kv_heads * head_dim;

    for (int token_idx = 0; token_idx < num_tokens_in_batch; ++token_idx) {
        size_t current_token_batch_offset = static_cast<size_t>(token_idx) * kv_dim;
        
        int seq_idx = sequence_indices[token_idx];
        int pos_in_seq = position_in_sequence[token_idx];
        
        // Calculate sequence-specific base offset
        int sequence_base_offset = seq_idx * kv_cache->max_seq_len_config_;
        int actual_cache_position = sequence_base_offset + pos_in_seq;
        if (actual_cache_position >= kv_cache->max_seq_len_config_ * kv_cache->max_batch_size) {
            Logger::error("[KV_BATCH_UPDATE_SEQ_AWARE L" + std::to_string(layer_idx) + 
                          "] Error: actual_cache_position (" + std::to_string(actual_cache_position) +
                          ") is out of bounds for total cache size. Skipping update for this token.");
            continue;
        }
        
        size_t destination_offset_in_layer_cache = static_cast<size_t>(actual_cache_position) * kv_dim;
        
        std::copy(k_batch_for_layer.begin() + current_token_batch_offset,
                  k_batch_for_layer.begin() + current_token_batch_offset + kv_dim,
                  layer_cache.k.begin() + destination_offset_in_layer_cache);
                  
        std::copy(v_batch_for_layer.begin() + current_token_batch_offset,
                  v_batch_for_layer.begin() + current_token_batch_offset + kv_dim,
                  layer_cache.v.begin() + destination_offset_in_layer_cache);
    }
}

static void attention_batch_cpu_sequence_aware(
    const std::vector<float>& q_batch_roped,
    KVCacheLayer& current_layer_kv_cache,
    std::vector<float>& batch_attn_output,
    int num_tokens_in_batch,
    const std::vector<int>& sequence_indices,
    const std::vector<int>& position_in_sequence,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float attention_scale,
    int max_seq_len_per_sequence
) {
    size_t expected_q_size = (size_t)num_tokens_in_batch * num_q_heads * head_dim;
    if (q_batch_roped.size() != expected_q_size) {
        Logger::error("[ATTN_BATCH_CPU_SEQ_AWARE] q_batch_roped size mismatch. Expected: " + std::to_string(expected_q_size) +
                      ", Got: " + std::to_string(q_batch_roped.size()));
        std::fill(batch_attn_output.begin(), batch_attn_output.end(), 0.0f);
        return;
    }
    
    batch_attn_output.assign((size_t)num_tokens_in_batch * num_q_heads * head_dim, 0.0f);

    for (int token_idx = 0; token_idx < num_tokens_in_batch; ++token_idx) {
        size_t q_token_offset = (size_t)token_idx * num_q_heads * head_dim;
        size_t attn_out_token_offset = (size_t)token_idx * num_q_heads * head_dim;
        
        int seq_idx = sequence_indices[token_idx];
        int pos_in_seq = position_in_sequence[token_idx];
        int sequence_base_offset = seq_idx * max_seq_len_per_sequence;

        for (int h_q = 0; h_q < num_q_heads; ++h_q) {
            const float* q_head_for_token_ptr = q_batch_roped.data() + q_token_offset + (h_q * head_dim);
            int kv_group_head_idx = h_q / (num_q_heads / num_kv_heads);
            
            int history_len = pos_in_seq + 1;
            std::vector<float> scores(history_len);

            for (int t_hist = 0; t_hist < history_len; ++t_hist) {
                size_t k_cache_offset = ((size_t)(sequence_base_offset + t_hist) * num_kv_heads + kv_group_head_idx) * head_dim;
                
                if (k_cache_offset + head_dim > current_layer_kv_cache.k.size()) {
                    scores[t_hist] = -std::numeric_limits<float>::infinity();
                    continue;
                }

                float current_dot_product = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    current_dot_product += q_head_for_token_ptr[d] * current_layer_kv_cache.k[k_cache_offset + d];
                }
                scores[t_hist] = current_dot_product * attention_scale;
            }

            softmax_vector_cpu(scores, scores);
            
            float* current_attn_out_head_ptr = batch_attn_output.data() + attn_out_token_offset + (h_q * head_dim);

            for (int t_hist = 0; t_hist < history_len; ++t_hist) {
                if (scores[t_hist] == -std::numeric_limits<float>::infinity() || scores[t_hist] == 0.0f) continue;

                size_t v_cache_offset = ((size_t)(sequence_base_offset + t_hist) * num_kv_heads + kv_group_head_idx) * head_dim;
                if (v_cache_offset + head_dim > current_layer_kv_cache.v.size()) {
                    continue;
                }

                for (int d = 0; d < head_dim; ++d) {
                    current_attn_out_head_ptr[d] += scores[t_hist] * current_layer_kv_cache.v[v_cache_offset + d];
                }
            }
        }
    }
}

std::vector<std::vector<float>> TinyLlamaModel::forward_cpu_batch_generation(
    const std::vector<float>& batch_input_activations, // [num_tokens, hidden_size]
    const std::vector<int>& token_positions, // Position of each token in its respective sequence
    const std::vector<int>& original_sequence_indices, // Original sequence index for each token
    int num_tokens_in_batch,
    KVCache* kv_cache) {

    Logger::info("[CPU_BATCH_GEN] Entry: num_tokens=" + std::to_string(num_tokens_in_batch));
    std::string pos_str = "token_positions=[";
    for (int i = 0; i < std::min(num_tokens_in_batch, 3); ++i) {
        pos_str += std::to_string(token_positions[i]) + " ";
    }
    pos_str += "]";
    std::string seq_str = "original_sequence_indices=[";
    for (int i = 0; i < std::min(num_tokens_in_batch, 3); ++i) {
        seq_str += std::to_string(original_sequence_indices[i]) + " ";
    }
    seq_str += "]";
    Logger::info("[CPU_BATCH_GEN] " + pos_str + ", " + seq_str);   
    if (batch_input_activations.size() != (size_t)num_tokens_in_batch * config_.hidden_size) {
        Logger::error("[CPU_BATCH_GENERATION] batch_input_activations size mismatch. Expected: " +
                      std::to_string((size_t)num_tokens_in_batch * config_.hidden_size) + " Got: " +
                      std::to_string(batch_input_activations.size()));
        return {};
    }

    if (token_positions.size() != static_cast<size_t>(num_tokens_in_batch)) {
        Logger::error("[CPU_BATCH_GENERATION] token_positions size mismatch. Expected: " + 
                      std::to_string(num_tokens_in_batch) + " Got: " + std::to_string(token_positions.size()));
        return {};
    }

    int hs = config_.hidden_size;
    int is = config_.intermediate_size;
    int n_heads = config_.num_attention_heads;
    int n_kv_heads = config_.num_key_value_heads;
    if (n_heads == 0) {
        Logger::error("[CPU_BATCH_GENERATION] Error: num_attention_heads is zero.");
        return {};
    }
    int head_dim = hs / n_heads;
    float eps = config_.rms_norm_eps;
    int max_pos_embeddings = config_.max_position_embeddings;
    bool use_rope_adjacent_pairing = config_.is_gguf_file_loaded;
    float attention_scale = 1.0f / SAFE_SQRT(static_cast<float>(head_dim));
    int vs = config_.vocab_size;
    int kv_group = n_heads / n_kv_heads;  // Pre-calculate GQA grouping

    std::vector<float> current_batch_activations = batch_input_activations;

    // Note: Unlike prefill batch processing, generation uses variable positions for each token
    // We fall back to individual token processing for position-dependent operations
    for (int l = 0; l < config_.num_cpu_offload_layers; ++l) {
        const auto& lw = layers[l];
        
        // Batch RMSNorm for attention
        std::vector<float> batch_x_norm1(current_batch_activations.size());
        const std::vector<float>& w_input_norm_vec =
            lw.input_layernorm_f32.empty()
                ? bf16vec_to_float_vec(lw.input_layernorm)
                : lw.input_layernorm_f32;
        rmsnorm_batch_cpu(current_batch_activations, w_input_norm_vec, batch_x_norm1, num_tokens_in_batch, hs, eps);

        std::vector<float> residual_batch_component_attn = current_batch_activations; 

        // Batch Q, K, V projections
        std::vector<float> q_batch((size_t)num_tokens_in_batch * hs);
        std::vector<float> k_batch((size_t)num_tokens_in_batch * n_kv_heads * head_dim);
        std::vector<float> v_batch((size_t)num_tokens_in_batch * n_kv_heads * head_dim);

        // Q Projection (batched)
        if (!lw.q_proj_f32.empty()) {
            matmul_f32_f32_batch_cpu(lw.q_proj_f32, batch_x_norm1, q_batch, num_tokens_in_batch, hs, hs);
        } else if (!lw.q_proj_q8_0.empty()) {
            matmul_q8_0_f32_batch_cpu(lw.q_proj_q8_0, batch_x_norm1, q_batch, num_tokens_in_batch, hs, hs);
        } else if (!lw.q_proj_q6k.empty()) {
            matmul_q6k_f32_batch_cpu(lw.q_proj_q6k, batch_x_norm1, q_batch, num_tokens_in_batch, hs, hs);
        } else if (!lw.q_proj_q4k.empty()) {
            matmul_q4k_f32_batch_cpu(lw.q_proj_q4k, batch_x_norm1, q_batch, num_tokens_in_batch, hs, hs);
        } else {
            Logger::error("[CPU_BATCH_GENERATION] Layer " + std::to_string(l) + ": No Q proj weights found for CPU (batched)"); 
            return {};
        }
        
        // K Projection (batched)
        if (!lw.k_proj_f32.empty()) {
            matmul_f32_f32_batch_cpu(lw.k_proj_f32, batch_x_norm1, k_batch, num_tokens_in_batch, n_kv_heads * head_dim, hs);
        } else if (!lw.k_proj_q8_0.empty()) {
            matmul_q8_0_f32_batch_cpu(lw.k_proj_q8_0, batch_x_norm1, k_batch, num_tokens_in_batch, n_kv_heads * head_dim, hs);
        } else if (!lw.k_proj_q6k.empty()) {
            matmul_q6k_f32_batch_cpu(lw.k_proj_q6k, batch_x_norm1, k_batch, num_tokens_in_batch, n_kv_heads * head_dim, hs);
        } else if (!lw.k_proj_q4k.empty()) {
            matmul_q4k_f32_batch_cpu(lw.k_proj_q4k, batch_x_norm1, k_batch, num_tokens_in_batch, n_kv_heads * head_dim, hs);
        } else {
            Logger::error("[CPU_BATCH_GENERATION] Layer " + std::to_string(l) + ": No K proj weights found for CPU (batched)"); 
            return {};
        }

        // V Projection (batched)
        if (!lw.v_proj_f32.empty()) {
            matmul_f32_f32_batch_cpu(lw.v_proj_f32, batch_x_norm1, v_batch, num_tokens_in_batch, n_kv_heads * head_dim, hs);
        } else if (!lw.v_proj_q8_0.empty()) {
            matmul_q8_0_f32_batch_cpu(lw.v_proj_q8_0, batch_x_norm1, v_batch, num_tokens_in_batch, n_kv_heads * head_dim, hs);
        } else if (!lw.v_proj_q6k.empty()) {
            matmul_q6k_f32_batch_cpu(lw.v_proj_q6k, batch_x_norm1, v_batch, num_tokens_in_batch, n_kv_heads * head_dim, hs);
        } else if (!lw.v_proj_q4k.empty()) {
            matmul_q4k_f32_batch_cpu(lw.v_proj_q4k, batch_x_norm1, v_batch, num_tokens_in_batch, n_kv_heads * head_dim, hs);
        } else {
            Logger::error("[CPU_BATCH_GENERATION] Layer " + std::to_string(l) + ": No V proj weights found for CPU (batched)"); 
            return {};
        }

        // Optimized RoPE, KV cache update, and attention with OpenMP and SIMD
        std::vector<float> batch_attn_output((size_t)num_tokens_in_batch * hs);
        
        // Ensure weights are dequantized before parallel processing
        ensure_q_proj_dequantized(l);
        ensure_k_proj_dequantized(l);
        ensure_v_proj_dequantized(l);
        ensure_o_proj_dequantized(l);
        ensure_gate_proj_dequantized(l);
        ensure_up_proj_dequantized(l);
        ensure_down_proj_dequantized(l);
        
        #pragma omp parallel if(num_tokens_in_batch > 1)
        {
            // Thread-local buffers to avoid allocations in loop
            std::vector<float> q_token(hs);
            std::vector<float> k_token(n_kv_heads * head_dim);
            std::vector<float> v_token(n_kv_heads * head_dim);
            std::vector<float> scores_buffer;
            
            #pragma omp for
            for (int token_idx = 0; token_idx < num_tokens_in_batch; ++token_idx) {
                int pos = token_positions[token_idx];
                
                // Extract Q, K, V for this token (reuse thread-local buffers)
                std::copy(q_batch.begin() + (size_t)token_idx * hs, 
                         q_batch.begin() + (size_t)(token_idx + 1) * hs, 
                         q_token.begin());
                std::copy(k_batch.begin() + (size_t)token_idx * n_kv_heads * head_dim, 
                         k_batch.begin() + (size_t)(token_idx + 1) * n_kv_heads * head_dim, 
                         k_token.begin());
                std::copy(v_batch.begin() + (size_t)token_idx * n_kv_heads * head_dim, 
                         v_batch.begin() + (size_t)(token_idx + 1) * n_kv_heads * head_dim, 
                         v_token.begin());
                
                // Apply RoPE individually
                apply_rope_vector(q_token, n_heads, head_dim, pos, precomputed_freqs_cis_, max_pos_embeddings, use_rope_adjacent_pairing);
                apply_rope_vector(k_token, n_kv_heads, head_dim, pos, precomputed_freqs_cis_, max_pos_embeddings, use_rope_adjacent_pairing);
                
                // Update KV cache at specific position - Sequence-Major Layout
                if (kv_cache && static_cast<size_t>(l) < kv_cache->layers.size()) {
                    auto& layer_cache = kv_cache->layers[l];
                    // Use sequence-major layout to match prefill behavior
                    int seq_idx = original_sequence_indices[token_idx];
                    int sequence_base_offset = seq_idx * kv_cache->max_seq_len_config_;
                    int kv_offset = (sequence_base_offset + pos) * n_kv_heads * head_dim;
                    #pragma omp critical
                    {
                        if (kv_offset + n_kv_heads * head_dim <= static_cast<int>(layer_cache.k.size())) {
                            std::copy(k_token.begin(), k_token.end(), layer_cache.k.begin() + kv_offset);
                            std::copy(v_token.begin(), v_token.end(), layer_cache.v.begin() + kv_offset);
                        }
                    }
                }
                
                // Copy back RoPE'd Q values to batch
                std::copy(q_token.begin(), q_token.end(), q_batch.begin() + (size_t)token_idx * hs);
                
                // SIMD-optimized attention computation for this token
                int seq_idx = original_sequence_indices[token_idx];
                int history_len = (seq_idx < kv_cache->current_batch_size) ? kv_cache->batch_seq_lens[seq_idx] : pos + 1;
                scores_buffer.resize(history_len);
                
                const float* q_token_ptr = q_batch.data() + (size_t)token_idx * hs;
                float* attn_output_ptr = batch_attn_output.data() + (size_t)token_idx * hs;
        
        if (kv_cache && static_cast<size_t>(l) < kv_cache->layers.size()) {
                    const auto& layer_cache = kv_cache->layers[l];
                    
                    // Process all heads for this token efficiently with SIMD
                    for (int h = 0; h < n_heads; ++h) {
                        int kv_head_idx = h / kv_group;  // Use pre-calculated kv_group
                        const float* q_head_ptr = q_token_ptr + h * head_dim;
                        float* head_output_ptr = attn_output_ptr + h * head_dim;
                        
                        // SIMD-optimized attention score computation
                        for (int t = 0; t < history_len; ++t) {
                            // sequence-major layout: each sequence has contiguous region                            
                            int seq_idx = original_sequence_indices[token_idx];
                            int sequence_base_offset = seq_idx * kv_cache->max_seq_len_config_;
                            const float* k_ptr = layer_cache.k.data() + (sequence_base_offset + t) * n_kv_heads * head_dim + kv_head_idx * head_dim;
                            
                            // Use SIMD dot product for Q·K computation
#if defined(__AVX2__) || defined(__SSE2__) || defined(__ARM_NEON)
                            float score = simd_dot_product(q_head_ptr, k_ptr, head_dim);
#else
                            float score = 0.0f;
                            for (int d = 0; d < head_dim; ++d) {
                                score += q_head_ptr[d] * k_ptr[d];
                            }
#endif
                            scores_buffer[t] = score * attention_scale;
                        }
                
                        // Softmax
                        softmax_vector_cpu(scores_buffer, scores_buffer);
                        
                        // SIMD-optimized weighted sum with V
                        std::fill(head_output_ptr, head_output_ptr + head_dim, 0.0f);
                        for (int t = 0; t < history_len; ++t) {
                            // Sequence-major layout: each sequence has contiguous region
                            int seq_idx = original_sequence_indices[token_idx];
                            int sequence_base_offset = seq_idx * kv_cache->max_seq_len_config_;                    
                            const float* v_ptr = layer_cache.v.data() + (sequence_base_offset + t) * n_kv_heads * head_dim + kv_head_idx * head_dim;
                            float score = scores_buffer[t];
                            
                            // Use SIMD scaled vector addition for score * V accumulation
#if defined(__AVX2__) || defined(__SSE2__) || defined(__ARM_NEON)
                            simd_scaled_add(head_output_ptr, v_ptr, score, head_dim);
#else
                            for (int d = 0; d < head_dim; ++d) {
                                head_output_ptr[d] += score * v_ptr[d];
                            }
#endif
                        }
                    }
                } else {
                    std::fill(attn_output_ptr, attn_output_ptr + hs, 0.0f);
                }
            }
        }
        
        // O-Projection (batched)
        std::vector<float> batch_attn_proj_out((size_t)num_tokens_in_batch * hs);
        if(!lw.o_proj_f32.empty()) {
              matmul_f32_f32_batch_cpu(lw.o_proj_f32, batch_attn_output, batch_attn_proj_out, num_tokens_in_batch, hs, hs);
        } else if (!lw.o_proj_q8_0.empty()) {
            matmul_q8_0_f32_batch_cpu(lw.o_proj_q8_0, batch_attn_output, batch_attn_proj_out, num_tokens_in_batch, hs, hs);
        } else if (!lw.o_proj_q6k.empty()) {
            matmul_q6k_f32_batch_cpu(lw.o_proj_q6k, batch_attn_output, batch_attn_proj_out, num_tokens_in_batch, hs, hs);
        } else if (!lw.o_proj_q4k.empty()) {
            matmul_q4k_f32_batch_cpu(lw.o_proj_q4k, batch_attn_output, batch_attn_proj_out, num_tokens_in_batch, hs, hs);
        } else { 
            Logger::error("[CPU_BATCH_GENERATION] Layer " + std::to_string(l) + ": No O proj weights found for CPU"); 
            return {};
        }

        // First Residual Connection (batched)
        for(size_t i=0; i < current_batch_activations.size(); ++i) {
            current_batch_activations[i] = residual_batch_component_attn[i] + batch_attn_proj_out[i];
        }

        // MLP processing (batched where possible)
        std::vector<float> residual_batch_component_mlp = current_batch_activations;
        std::vector<float> batch_x_norm2(current_batch_activations.size());
        
        const std::vector<float>& w_post_attn_norm_vec =
            lw.post_attention_layernorm_f32.empty()
                ? bf16vec_to_float_vec(lw.post_attention_layernorm)
                : lw.post_attention_layernorm_f32;
        rmsnorm_batch_cpu(current_batch_activations, w_post_attn_norm_vec, batch_x_norm2, num_tokens_in_batch, hs, eps);
        
        std::vector<float> batch_gate_proj_out((size_t)num_tokens_in_batch * is);
        std::vector<float> batch_up_proj_out((size_t)num_tokens_in_batch * is);

        // Gate and Up projections (batched)
        if (!lw.gate_proj_f32.empty()) {
            matmul_f32_f32_batch_cpu(lw.gate_proj_f32, batch_x_norm2, batch_gate_proj_out, num_tokens_in_batch, is, hs);
        } else if (!lw.gate_proj_q8_0.empty()) {
            matmul_q8_0_f32_batch_cpu(lw.gate_proj_q8_0, batch_x_norm2, batch_gate_proj_out, num_tokens_in_batch, is, hs);
        } else if (!lw.gate_proj_q6k.empty()) {
            matmul_q6k_f32_batch_cpu(lw.gate_proj_q6k, batch_x_norm2, batch_gate_proj_out, num_tokens_in_batch, is, hs);
        } else if (!lw.gate_proj_q4k.empty()) {
            matmul_q4k_f32_batch_cpu(lw.gate_proj_q4k, batch_x_norm2, batch_gate_proj_out, num_tokens_in_batch, is, hs);
        } else { 
            Logger::error("[CPU_BATCH_GENERATION] Layer " + std::to_string(l) + ": No gate_proj weights found for CPU"); 
            return {};
        }

        if (!lw.up_proj_f32.empty()) {
            matmul_f32_f32_batch_cpu(lw.up_proj_f32, batch_x_norm2, batch_up_proj_out, num_tokens_in_batch, is, hs);
        } else if (!lw.up_proj_q8_0.empty()) {
            matmul_q8_0_f32_batch_cpu(lw.up_proj_q8_0, batch_x_norm2, batch_up_proj_out, num_tokens_in_batch, is, hs);
        } else if (!lw.up_proj_q6k.empty()) {
            matmul_q6k_f32_batch_cpu(lw.up_proj_q6k, batch_x_norm2, batch_up_proj_out, num_tokens_in_batch, is, hs);
        } else if (!lw.up_proj_q4k.empty()) {
            matmul_q4k_f32_batch_cpu(lw.up_proj_q4k, batch_x_norm2, batch_up_proj_out, num_tokens_in_batch, is, hs);
        } else { 
            Logger::error("[CPU_BATCH_GENERATION] Layer " + std::to_string(l) + ": No up_proj weights found for CPU"); 
            return {};
        }

        // SwiGLU (batched)
        std::vector<float> batch_swiglu_out((size_t)num_tokens_in_batch * is);
        for (size_t i = 0; i < batch_gate_proj_out.size(); ++i) {
            float gate_val = batch_gate_proj_out[i];
            float silu_gate_val = gate_val / (1.0f + std::exp(-gate_val));
            batch_swiglu_out[i] = silu_gate_val * batch_up_proj_out[i];
        }
        
        // Down Projection (batched)
        std::vector<float> batch_mlp_down_proj_out((size_t)num_tokens_in_batch * hs);
        if (!lw.down_proj_f32.empty()) {
            matmul_f32_f32_batch_cpu(lw.down_proj_f32, batch_swiglu_out, batch_mlp_down_proj_out, num_tokens_in_batch, hs, is);
        } else if (!lw.down_proj_q8_0.empty()) {
            matmul_q8_0_f32_batch_cpu(lw.down_proj_q8_0, batch_swiglu_out, batch_mlp_down_proj_out, num_tokens_in_batch, hs, is);
        } else if (!lw.down_proj_q6k.empty()) {
            matmul_q6k_f32_batch_cpu(lw.down_proj_q6k, batch_swiglu_out, batch_mlp_down_proj_out, num_tokens_in_batch, hs, is);
        } else if (!lw.down_proj_q4k.empty()) {
            matmul_q4k_f32_batch_cpu(lw.down_proj_q4k, batch_swiglu_out, batch_mlp_down_proj_out, num_tokens_in_batch, hs, is);
        } else { 
            Logger::error("[CPU_BATCH_GENERATION] Layer " + std::to_string(l) + ": No down_proj weights found for CPU"); 
            return {};
        }

        // Second Residual Connection (batched)
        for(size_t i = 0; i < current_batch_activations.size(); ++i) {
            current_batch_activations[i] = residual_batch_component_mlp[i] + batch_mlp_down_proj_out[i];
        }
    }

    // Update KV cache sequence length
    if (kv_cache && num_tokens_in_batch > 0) {
        // For batch mode, track positions per sequence
        if (kv_cache->current_batch_size > 0) {
            // Update batch_seq_lens based on the highest position seen for each sequence
            std::vector<int> max_positions_per_seq(kv_cache->current_batch_size, -1);
            
            for (int i = 0; i < num_tokens_in_batch; ++i) {
                int seq_idx = original_sequence_indices[i];
                int pos = token_positions[i];
                
                if (seq_idx >= 0 && seq_idx < kv_cache->current_batch_size) {
                    max_positions_per_seq[seq_idx] = std::max(max_positions_per_seq[seq_idx], pos);
                }
            }
            
            // Update batch_seq_lens to reflect new positions (pos + 1 since pos is 0-indexed)
            for (int seq_idx = 0; seq_idx < kv_cache->current_batch_size; ++seq_idx) {
                if (max_positions_per_seq[seq_idx] >= 0) {
                    kv_cache->batch_seq_lens[seq_idx] = max_positions_per_seq[seq_idx] + 1;
                    Logger::info("[CPU_BATCH_GEN] KV Length Max Update: seq_idx=" + std::to_string(seq_idx) + 
                    ", old_batch_seq_len=" + std::to_string(kv_cache->batch_seq_lens[seq_idx]) + 
                    ", new_batch_seq_len=" + std::to_string(max_positions_per_seq[seq_idx] + 1));
                }
            }
            // For single-sequence compatibility, update seq_len to the max
            kv_cache->seq_len = *std::max_element(kv_cache->batch_seq_lens.begin(), 
                                                  kv_cache->batch_seq_lens.begin() + kv_cache->current_batch_size);
    } else {
            // Fallback for single sequence mode
            int max_pos = *std::max_element(token_positions.begin(), token_positions.end());
            kv_cache->seq_len = std::max(kv_cache->seq_len, max_pos + 1);
        }
    }
    
    // Final normalization and logits calculation for ALL tokens
    std::vector<float> batch_logits = forward_cpu_logits_batch(current_batch_activations, num_tokens_in_batch);
    
    // Convert flat logits to per-token vectors
    std::vector<std::vector<float>> all_logits(num_tokens_in_batch, std::vector<float>(vs));
    for (int token_idx = 0; token_idx < num_tokens_in_batch; ++token_idx) {
        std::copy(batch_logits.begin() + (size_t)token_idx * vs,
                 batch_logits.begin() + (size_t)(token_idx + 1) * vs,
                 all_logits[token_idx].begin());
    }
    return all_logits;
}

#ifdef HAS_CUDA
std::vector<float> TinyLlamaModel::forward_device_batch_prefill(
    float* d_batch_input_embeddings, // This is now assumed to be activations *after* CPU layers if any
    int num_tokens_in_batch,
    int current_model_pos, // This should be the starting position *within the KV cache* for the batch
    KVCache* kv_cache,
    cudaStream_t stream) {

    Logger::info("[FWD_DEV_BATCH_PREFILL_ENTRY] num_tokens_in_batch: " + std::to_string(num_tokens_in_batch) +
                 ", current_model_pos: " + std::to_string(current_model_pos) +
                 ", d_batch_input_embeddings: " + Logger::ptrToString(d_batch_input_embeddings));

    const int hidden_size = config_.hidden_size;
    const int head_dim = config_.hidden_size / config_.num_attention_heads;
    const int ffn_intermediate_dim = config_.intermediate_size;
    const int n_kv_dim = config_.num_key_value_heads * head_dim;
    const int vocab_size = config_.vocab_size;

    Logger::debug("[FWD_DEV_BATCH_PREFILL_PARAMS] hidden_size: " + std::to_string(hidden_size) +
                  ", head_dim: " + std::to_string(head_dim) +
                  ", ffn_intermediate_dim: " + std::to_string(ffn_intermediate_dim) +
                  ", n_kv_dim: " + std::to_string(n_kv_dim) +
                  ", vocab_size: " + std::to_string(vocab_size) +
                  ", num_attention_heads: " + std::to_string(config_.num_attention_heads) +
                  ", num_key_value_heads: " + std::to_string(config_.num_key_value_heads));

    float* d_batch_x_ptr = d_batch_input_embeddings; // Input to the first GPU layer
    float* d_batch_x_norm_out_attn;
    float* d_batch_q_proj_out;
    float* d_batch_k_proj_out;
    float* d_batch_v_proj_out;
    float* d_batch_attn_heads_concat_out;
    float* d_batch_attn_final_proj_out;
    float* d_batch_residual_attn_in; 
    float* d_batch_residual_ffn_in;  
    float* d_batch_x_norm_out_ffn;
    float* d_batch_ffn_gate_proj_out;
    float* d_batch_ffn_up_proj_out;
    float* d_batch_ffn_swiglu_out;
    float* d_batch_ffn_down_proj_out;
    float* d_batch_layer_output = nullptr; 

    size_t batch_hidden_size_elems = (size_t)num_tokens_in_batch * hidden_size;
    size_t batch_kv_proj_size_elems = (size_t)num_tokens_in_batch * n_kv_dim;
    size_t batch_ffn_intermediate_elems = (size_t)num_tokens_in_batch * ffn_intermediate_dim;

    size_t batch_hidden_size_bytes = batch_hidden_size_elems * sizeof(float);
    size_t batch_kv_proj_size_bytes = batch_kv_proj_size_elems * sizeof(float);
    size_t batch_ffn_intermediate_bytes = batch_ffn_intermediate_elems * sizeof(float);

    gpuErrchk(cudaMalloc(&d_batch_x_norm_out_attn, batch_hidden_size_bytes));
    gpuErrchk(cudaMalloc(&d_batch_q_proj_out, batch_hidden_size_bytes));
    gpuErrchk(cudaMalloc(&d_batch_k_proj_out, batch_kv_proj_size_bytes));
    gpuErrchk(cudaMalloc(&d_batch_v_proj_out, batch_kv_proj_size_bytes));
    gpuErrchk(cudaMalloc(&d_batch_attn_heads_concat_out, batch_hidden_size_bytes));
    gpuErrchk(cudaMalloc(&d_batch_attn_final_proj_out, batch_hidden_size_bytes));
    gpuErrchk(cudaMalloc(&d_batch_residual_attn_in, batch_hidden_size_bytes)); 
    gpuErrchk(cudaMalloc(&d_batch_residual_ffn_in, batch_hidden_size_bytes)); 
    gpuErrchk(cudaMalloc(&d_batch_x_norm_out_ffn, batch_hidden_size_bytes));
    gpuErrchk(cudaMalloc(&d_batch_ffn_gate_proj_out, batch_ffn_intermediate_bytes));
    gpuErrchk(cudaMalloc(&d_batch_ffn_up_proj_out, batch_ffn_intermediate_bytes));
    gpuErrchk(cudaMalloc(&d_batch_ffn_swiglu_out, batch_ffn_intermediate_bytes));
    gpuErrchk(cudaMalloc(&d_batch_ffn_down_proj_out, batch_hidden_size_bytes));

    if (config_.num_cpu_offload_layers < config_.num_hidden_layers) {
         gpuErrchk(cudaMalloc(&d_batch_layer_output, batch_hidden_size_bytes));
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasStatus_t stream_status = cublasSetStream(cublas_handle_, stream);
    if (stream_status != CUBLAS_STATUS_SUCCESS) {
        Logger::fatal("cublasSetStream failed in forward_device_batch_prefill");
        throw std::runtime_error("cublasSetStream failed");
    }

    Logger::info("[FWD_DEV_BATCH_PREFILL_MAIN_LOOP_ENTRY] num_cpu_offload_layers: " + std::to_string(config_.num_cpu_offload_layers) + 
                 ", total_hidden_layers: " + std::to_string(config_.num_hidden_layers));

    for (int l_model_idx = config_.num_cpu_offload_layers; l_model_idx < config_.num_hidden_layers; ++l_model_idx) {
        int l_gpu_idx = l_model_idx - config_.num_cpu_offload_layers;
        Logger::info("[FWD_DEV_BATCH_PREFILL_LAYER_START] Processing Layer: model_idx=" + std::to_string(l_model_idx) + ", gpu_idx=" + std::to_string(l_gpu_idx) + 
                     ". Current d_batch_x_ptr: " + Logger::ptrToString(d_batch_x_ptr));
        // Weight dequantization handled by ensure_f32_concatenated_weights_loaded() below
        gpuErrchk(cudaMemcpyAsync(d_batch_residual_attn_in, d_batch_x_ptr, batch_hidden_size_bytes, cudaMemcpyDeviceToDevice, stream));

        rmsnorm_batch_cuda(d_batch_x_norm_out_attn, d_batch_x_ptr, 
                           layers[l_model_idx].input_layernorm_dev,
                           num_tokens_in_batch, hidden_size, config_.rms_norm_eps, stream);

        ensure_f32_concatenated_weights_loaded();
        
        const float* w_q_layer_ptr = w_q_f32_dev_ + (size_t)l_gpu_idx * hidden_size * hidden_size;
        gemm_f32_f32_cuda(cublas_handle_, false, true, num_tokens_in_batch, hidden_size, hidden_size, &alpha,
                          d_batch_x_norm_out_attn, hidden_size, w_q_layer_ptr, hidden_size, &beta,
                          d_batch_q_proj_out, hidden_size, stream);
        Logger::info("[GPU_Q_PROJ] Layer=" + std::to_string(l_model_idx) + 
            ", input_ptr=" + Logger::ptrToString(d_batch_x_norm_out_attn) +
            ", weight_ptr=" + Logger::ptrToString(w_q_layer_ptr) +
            ", output_ptr=" + Logger::ptrToString(d_batch_q_proj_out));
        // Q_PROJ_OUT LOGGING (Unchanged)
        if (l_model_idx == config_.num_cpu_offload_layers && num_tokens_in_batch > 0 && hidden_size > 0) { // Existing logging condition
            int log_elements_common = std::min(static_cast<int>(head_dim), 3);
            if (log_elements_common <= 0 && hidden_size > 0) log_elements_common = std::min(static_cast<int>(hidden_size), 3);
            if (log_elements_common > 0) {
                std::vector<float> h_sample_t0(log_elements_common);
                gpuErrchk(cudaMemcpyAsync(h_sample_t0.data(), d_batch_q_proj_out, log_elements_common * sizeof(float), cudaMemcpyDeviceToHost, stream));
                if (num_tokens_in_batch <= 1) gpuErrchk(cudaStreamSynchronize(stream)); 
                std::string str_t0 = ""; for(float val : h_sample_t0) { str_t0 += std::to_string(val) + " "; }
                Logger::debug("[FWD_DEV_BATCH_PREFILL_LAYER_L" + std::to_string(l_model_idx) + "] Q_PROJ_OUT (T0, H0, first " + std::to_string(log_elements_common) + "): " + str_t0);
                if (num_tokens_in_batch > 1) {
                    std::vector<float> h_sample_t1(log_elements_common);
                    gpuErrchk(cudaMemcpyAsync(h_sample_t1.data(), d_batch_q_proj_out + hidden_size, log_elements_common * sizeof(float), cudaMemcpyDeviceToHost, stream));
                    gpuErrchk(cudaStreamSynchronize(stream));
                    std::string str_t1 = ""; for(float val : h_sample_t1) { str_t1 += std::to_string(val) + " "; }
                    Logger::debug("[FWD_DEV_BATCH_PREFILL_LAYER_L" + std::to_string(l_model_idx) + "] Q_PROJ_OUT (T1, H0, first " + std::to_string(log_elements_common) + "): " + str_t1);
                }
            }
        }

        const float* w_k_layer_ptr = w_k_f32_dev_ + (size_t)l_gpu_idx * n_kv_dim * hidden_size;
        gemm_f32_f32_cuda(cublas_handle_, false, true, num_tokens_in_batch, n_kv_dim, hidden_size, &alpha,
                          d_batch_x_norm_out_attn, hidden_size, w_k_layer_ptr, n_kv_dim, &beta,
                          d_batch_k_proj_out, n_kv_dim, stream);

        const float* w_v_layer_ptr = w_v_f32_dev_ + (size_t)l_gpu_idx * n_kv_dim * hidden_size;
        gemm_f32_f32_cuda(cublas_handle_, false, true, num_tokens_in_batch, n_kv_dim, hidden_size, &alpha,
                          d_batch_x_norm_out_attn, hidden_size, w_v_layer_ptr, n_kv_dim, &beta,
                          d_batch_v_proj_out, n_kv_dim, stream);

        // PRE-ROPE LOGGING (Unchanged)
        if (l_model_idx == config_.num_cpu_offload_layers && num_tokens_in_batch > 0 && head_dim > 0) { // Existing logging condition
            int log_elements_rope = std::min(3, head_dim);
            for (int token_to_log_idx_rope = 0; token_to_log_idx_rope < std::min(num_tokens_in_batch, 2); ++token_to_log_idx_rope) {
                if (d_batch_q_proj_out && config_.num_attention_heads > 0) {
                    std::vector<float> h_q_pre_rope(log_elements_rope);
                    size_t q_log_offset = (size_t)token_to_log_idx_rope * config_.num_attention_heads * head_dim;
                    gpuErrchk(cudaMemcpyAsync(h_q_pre_rope.data(), d_batch_q_proj_out + q_log_offset, log_elements_rope * sizeof(float), cudaMemcpyDeviceToHost, stream));
                    gpuErrchk(cudaStreamSynchronize(stream)); 
                    std::string str_q_pre_rope = ""; for(float val : h_q_pre_rope) { str_q_pre_rope += std::to_string(val) + " "; }
                }
                if (d_batch_k_proj_out && config_.num_key_value_heads > 0) {
                    std::vector<float> h_k_pre_rope(log_elements_rope);
                    size_t k_log_offset = (size_t)token_to_log_idx_rope * config_.num_key_value_heads * head_dim;
                    gpuErrchk(cudaMemcpyAsync(h_k_pre_rope.data(), d_batch_k_proj_out + k_log_offset, log_elements_rope * sizeof(float), cudaMemcpyDeviceToHost, stream));
                    gpuErrchk(cudaStreamSynchronize(stream)); 
                    std::string str_k_pre_rope = ""; for(float val : h_k_pre_rope) { str_k_pre_rope += std::to_string(val) + " "; }
                }
            }
        }
        
        rope_batch_cuda(d_batch_q_proj_out, d_batch_k_proj_out, all_freqs_cis_dev, num_tokens_in_batch,
                        config_.num_attention_heads, config_.num_key_value_heads, head_dim,
                        current_model_pos, config_.is_gguf_file_loaded, stream);
        gpuErrchk(cudaStreamSynchronize(stream));

        if (l_model_idx == config_.num_cpu_offload_layers && num_tokens_in_batch > 0 && head_dim > 0) { // Existing logging condition
            int log_elements_rope = std::min(3, head_dim);
            for (int token_to_log_idx_rope = 0; token_to_log_idx_rope < std::min(num_tokens_in_batch, 2); ++token_to_log_idx_rope) {
                if (d_batch_q_proj_out && config_.num_attention_heads > 0) {
                    std::vector<float> h_q_post_rope(log_elements_rope);
                    size_t q_log_offset = (size_t)token_to_log_idx_rope * config_.num_attention_heads * head_dim;
                    gpuErrchk(cudaMemcpy(h_q_post_rope.data(), d_batch_q_proj_out + q_log_offset, log_elements_rope * sizeof(float), cudaMemcpyDeviceToHost));
                    std::string str_q_post_rope = ""; for(float val : h_q_post_rope) { str_q_post_rope += std::to_string(val) + " "; }
                }
                if (d_batch_k_proj_out && config_.num_key_value_heads > 0) {
                    std::vector<float> h_k_post_rope(log_elements_rope);
                    size_t k_log_offset = (size_t)token_to_log_idx_rope * config_.num_key_value_heads * head_dim;
                    gpuErrchk(cudaMemcpy(h_k_post_rope.data(), d_batch_k_proj_out + k_log_offset, log_elements_rope * sizeof(float), cudaMemcpyDeviceToHost));
                    std::string str_k_post_rope = ""; for(float val : h_k_post_rope) { str_k_post_rope += std::to_string(val) + " "; }
                }
            }
        }

        if (l_model_idx == config_.num_cpu_offload_layers && num_tokens_in_batch > 0 && head_dim > 0 && config_.num_key_value_heads > 0) { // Existing logging
            int log_elements = std::min(3, head_dim);
            if (d_batch_k_proj_out) {
                std::vector<float> h_k_log_token0(log_elements);
                gpuErrchk(cudaMemcpyAsync(h_k_log_token0.data(), d_batch_k_proj_out, log_elements * sizeof(float), cudaMemcpyDeviceToHost, stream));
                std::vector<float> h_k_log_token1(log_elements);
                if (num_tokens_in_batch > 1) { gpuErrchk(cudaMemcpyAsync(h_k_log_token1.data(), d_batch_k_proj_out + n_kv_dim, log_elements * sizeof(float), cudaMemcpyDeviceToHost, stream));}
                gpuErrchk(cudaStreamSynchronize(stream));
            }
            if (d_batch_v_proj_out) {
                std::vector<float> h_v_log_token0(log_elements);
                gpuErrchk(cudaMemcpyAsync(h_v_log_token0.data(), d_batch_v_proj_out, log_elements * sizeof(float), cudaMemcpyDeviceToHost, stream));
                std::vector<float> h_v_log_token1(log_elements);
                if (num_tokens_in_batch > 1) { gpuErrchk(cudaMemcpyAsync(h_v_log_token1.data(), d_batch_v_proj_out + n_kv_dim, log_elements * sizeof(float), cudaMemcpyDeviceToHost, stream));}
                gpuErrchk(cudaStreamSynchronize(stream));
            }
        }

        float* d_layer_k_cache_ptr = kv_cache->layers[l_model_idx].k_dev_fp32; 
        float* d_layer_v_cache_ptr = kv_cache->layers[l_model_idx].v_dev_fp32;
        update_kv_cache_batch_cuda(d_layer_k_cache_ptr, d_batch_k_proj_out, current_model_pos, num_tokens_in_batch,
                                   config_.num_key_value_heads, head_dim, kv_cache->max_seq_len_config_, stream);
        update_kv_cache_batch_cuda(d_layer_v_cache_ptr, d_batch_v_proj_out, current_model_pos, num_tokens_in_batch,
                                   config_.num_key_value_heads, head_dim, kv_cache->max_seq_len_config_, stream);
        gpuErrchk(cudaStreamSynchronize(stream)); 
        
        float current_attention_scale = 1.0f / sqrtf((float)head_dim);
        attention_batch_prefill_cuda(d_batch_q_proj_out, nullptr, nullptr, 
                                     d_layer_k_cache_ptr, d_layer_v_cache_ptr, 
                                     d_batch_attn_heads_concat_out, num_tokens_in_batch, current_model_pos, 
                                     kv_cache->max_seq_len_config_, config_.num_attention_heads, 
                                     config_.num_key_value_heads, head_dim, current_attention_scale, stream, nullptr);        
        const float* w_o_layer_ptr = w_o_f32_dev_ + (size_t)l_gpu_idx * hidden_size * hidden_size;
        gemm_f32_f32_cuda(cublas_handle_, false, true, num_tokens_in_batch, hidden_size, hidden_size, &alpha,
                          d_batch_attn_heads_concat_out, hidden_size, w_o_layer_ptr, hidden_size, &beta,
                          d_batch_attn_final_proj_out, hidden_size, stream);

        // O_PROJ_OUT LOGGING (Unchanged) - Logging Point 1
        if (l_model_idx == config_.num_cpu_offload_layers && num_tokens_in_batch > 1 && hidden_size > 0) { /* ... */ }


        add_residual_batch_cuda(d_batch_residual_ffn_in, d_batch_attn_final_proj_out, d_batch_residual_attn_in,
                                num_tokens_in_batch, hidden_size, stream);
        // POST_RESIDUAL_ATTN LOGGING (Unchanged) - Logging Point 2
        if (l_model_idx == config_.num_cpu_offload_layers && num_tokens_in_batch > 1 && hidden_size > 0) { /* ... */ }


        rmsnorm_batch_cuda(d_batch_x_norm_out_ffn, d_batch_residual_ffn_in, 
                           layers[l_model_idx].post_attention_layernorm_dev,
                           num_tokens_in_batch, hidden_size, config_.rms_norm_eps, stream);
        // RMSNORM_FFN_OUT LOGGING (Unchanged) - Logging Point 3
        if (l_model_idx == config_.num_cpu_offload_layers && num_tokens_in_batch > 1 && hidden_size > 0) { /* ... */ }


        const float* w1_layer_ptr = w_gate_f32_dev_ + (size_t)l_gpu_idx * hidden_size * ffn_intermediate_dim;
        gemm_f32_f32_cuda(cublas_handle_, false, true, num_tokens_in_batch, ffn_intermediate_dim, hidden_size, &alpha,
                          d_batch_x_norm_out_ffn, hidden_size, w1_layer_ptr, ffn_intermediate_dim, &beta,
                          d_batch_ffn_gate_proj_out, ffn_intermediate_dim, stream);
        // FFN_GATE_PROJ_OUT LOGGING (Unchanged) - Logging Point 4
        if (l_model_idx == config_.num_cpu_offload_layers && num_tokens_in_batch > 1 && ffn_intermediate_dim > 0) { /* ... */ }


        const float* w3_layer_ptr = w_up_f32_dev_ + (size_t)l_gpu_idx * hidden_size * ffn_intermediate_dim;
        gemm_f32_f32_cuda(cublas_handle_, false, true, num_tokens_in_batch, ffn_intermediate_dim, hidden_size, &alpha,
                          d_batch_x_norm_out_ffn, hidden_size, w3_layer_ptr, ffn_intermediate_dim, &beta,
                          d_batch_ffn_up_proj_out, ffn_intermediate_dim, stream);
        // FFN_UP_PROJ_OUT LOGGING (Unchanged) - Logging Point 5
        if (l_model_idx == config_.num_cpu_offload_layers && num_tokens_in_batch > 1 && ffn_intermediate_dim > 0) { /* ... */ }


        swiglu_batch_cuda(d_batch_ffn_swiglu_out, d_batch_ffn_gate_proj_out, d_batch_ffn_up_proj_out,
                          num_tokens_in_batch, ffn_intermediate_dim, stream);
        // FFN_SWIGLU_OUT LOGGING (Unchanged) - Logging Point 6
         if (l_model_idx == config_.num_cpu_offload_layers && num_tokens_in_batch > 1 && ffn_intermediate_dim > 0) { /* ... */ }


        const float* w2_layer_ptr = w_down_f32_dev_ + (size_t)l_gpu_idx * ffn_intermediate_dim * hidden_size;
        gemm_f32_f32_cuda(cublas_handle_, false, true, num_tokens_in_batch, hidden_size, ffn_intermediate_dim, &alpha,
                          d_batch_ffn_swiglu_out, ffn_intermediate_dim, w2_layer_ptr, hidden_size, &beta,
                          d_batch_ffn_down_proj_out, hidden_size, stream);
        // FFN_DOWN_PROJ_OUT LOGGING (Unchanged) - Logging Point 7
        if (l_model_idx == config_.num_cpu_offload_layers && num_tokens_in_batch > 1 && hidden_size > 0) { /* ... */ }
        

        add_residual_batch_cuda(d_batch_layer_output, d_batch_ffn_down_proj_out, d_batch_residual_ffn_in,
                                num_tokens_in_batch, hidden_size, stream);
        // POST_RESIDUAL_FFN LOGGING (Unchanged) - Logging Point 8
        if (l_model_idx == config_.num_cpu_offload_layers && num_tokens_in_batch > 1 && hidden_size > 0) { /* ... */ }

        
        d_batch_x_ptr = d_batch_layer_output; 
        Logger::info("[FWD_DEV_BATCH_PREFILL_LAYER_END] Layer " + std::to_string(l_model_idx) + " finished. Next d_batch_x_ptr: " + Logger::ptrToString(d_batch_x_ptr));
    }

  // =============== BEGIN DEBUG LOG: Hidden state of LAST TOKEN before final RMSNorm ===============
  if (num_tokens_in_batch > 0) {
      std::vector<float> h_last_token_hidden_state(config_.hidden_size);
      // current_d_batch_x_ptr should hold the final hidden states for all tokens in the batch after all layers.
      // The layout is [token0_hs, token1_hs, ..., tokenN-1_hs].
      // Offset for the last token (num_tokens_in_batch - 1) is (num_tokens_in_batch - 1) * hidden_size.
      size_t offset_last_token_hidden_state = (size_t)(num_tokens_in_batch - 1) * config_.hidden_size;
      
      gpuErrchk(cudaMemcpyAsync(h_last_token_hidden_state.data(),
                                d_batch_x_ptr + offset_last_token_hidden_state,
                                config_.hidden_size * sizeof(float),
                                cudaMemcpyDeviceToHost, stream));
      gpuErrchk(cudaStreamSynchronize(stream)); 
      Logger::log_vector_stats("[FWD_DEV_BATCH_PREFILL_LAST_TOKEN_HIDDEN_STATE_PRE_FINAL_RMSNORM]", h_last_token_hidden_state, 20);
  }
  // =============== END DEBUG LOG ===============


    rmsnorm_batch_cuda(d_batch_x_norm_out_attn, d_batch_x_ptr, 
                       final_norm_dev,
                       num_tokens_in_batch, hidden_size, config_.rms_norm_eps, stream);
    
    // FINAL_NORM_OUT LOGGING (Unchanged)
    if (config_.num_cpu_offload_layers < config_.num_hidden_layers && num_tokens_in_batch > 0 && hidden_size > 0) { /* ... */ }


    float* d_logits_last_token;
    gpuErrchk(cudaMalloc(&d_logits_last_token, (size_t)vocab_size * sizeof(float)));
    
    // Only calculate logits for the last token in the batch for prefill output
    float* d_last_token_activations_for_logits = d_batch_x_norm_out_attn + (size_t)(num_tokens_in_batch - 1) * hidden_size;

    matvec_f32_f32_cuda(cublas_handle_, lm_head_f32_dev_, d_last_token_activations_for_logits,
                        d_logits_last_token, vocab_size, hidden_size, stream);

    std::vector<float> h_logits(vocab_size);
    gpuErrchk(cudaMemcpyAsync(h_logits.data(), d_logits_last_token, (size_t)vocab_size * sizeof(float),
                           cudaMemcpyDeviceToHost, stream));
    gpuErrchk(cudaStreamSynchronize(stream)); 

    Logger::log_vector_stats("[FWD_DEV_BATCH_PREFILL_FINAL_LOGITS]", h_logits, 20);

    // =============== BEGIN KVCache DUMP AFTER BATCH PREFILL ===============
    if (config_.num_hidden_layers > config_.num_cpu_offload_layers && kv_cache != nullptr && num_tokens_in_batch > 0) {
        int first_gpu_layer_model_idx = config_.num_cpu_offload_layers;
        if (static_cast<size_t>(first_gpu_layer_model_idx) < kv_cache->layers.size()) {
            const KVCacheLayer& cache_layer_to_log = kv_cache->layers[first_gpu_layer_model_idx];
            const float* d_k_cache_ptr = cache_layer_to_log.k_dev_fp32;
            const float* d_v_cache_ptr = cache_layer_to_log.v_dev_fp32;
            const int num_kv_h = config_.num_key_value_heads;
            // head_dim is already defined in this function's scope
            const int local_n_kv_dim_for_log = num_kv_h * head_dim; 
            const int log_elems_kv = std::min(3, head_dim);

            if (d_k_cache_ptr && d_v_cache_ptr && log_elems_kv > 0 && local_n_kv_dim_for_log > 0) {
                for (int tk_idx = 0; tk_idx < num_tokens_in_batch; ++tk_idx) {
                    int cache_pos_for_token = current_model_pos + tk_idx; 
                    if (cache_pos_for_token >= kv_cache->max_seq_len_config_) {
                        Logger::warning("[KVDUMP_POST_BATCH_PREFILL] L" + std::to_string(first_gpu_layer_model_idx) + 
                                        " Token " + std::to_string(tk_idx) + " (CachePos " + std::to_string(cache_pos_for_token) + 
                                        ") would be out of bounds (" + std::to_string(kv_cache->max_seq_len_config_) + "). Skipping.");
                        continue;
                    }
                    for (int kvh_idx = 0; kvh_idx < num_kv_h; ++kvh_idx) {
                        size_t offset_in_cache = (size_t)cache_pos_for_token * local_n_kv_dim_for_log + (size_t)kvh_idx * head_dim;
                        
                        std::vector<float> h_k_dump(log_elems_kv);
                        gpuErrchk(cudaMemcpy(h_k_dump.data(), d_k_cache_ptr + offset_in_cache, log_elems_kv * sizeof(float), cudaMemcpyDeviceToHost));
                        std::string str_k_dump = ""; for(float val : h_k_dump) { str_k_dump += std::to_string(val) + " "; }
                        std::vector<float> h_v_dump(log_elems_kv);
                        gpuErrchk(cudaMemcpy(h_v_dump.data(), d_v_cache_ptr + offset_in_cache, log_elems_kv * sizeof(float), cudaMemcpyDeviceToHost));
                        std::string str_v_dump = ""; for(float val : h_v_dump) { str_v_dump += std::to_string(val) + " "; }
                    }
                }
            } else {
                Logger::warning("[KVDUMP_POST_BATCH_PREFILL] L" + std::to_string(first_gpu_layer_model_idx) + 
                                " cannot log K/V cache: null pointers, log_elems_kv <= 0, or local_n_kv_dim_for_log <=0.");
            }
        } else {
            Logger::warning("[KVDUMP_POST_BATCH_PREFILL] First GPU layer index " + std::to_string(first_gpu_layer_model_idx) + 
                             " out of bounds for kv_cache->layers (size " + std::to_string(kv_cache->layers.size()) + ")");
        }
    }
    // =============== END KVCache DUMP AFTER BATCH PREFILL ===============

    gpuErrchk(cudaFree(d_batch_x_norm_out_attn));
    gpuErrchk(cudaFree(d_batch_q_proj_out));
    gpuErrchk(cudaFree(d_batch_k_proj_out));
    gpuErrchk(cudaFree(d_batch_v_proj_out));
    gpuErrchk(cudaFree(d_batch_attn_heads_concat_out));
    gpuErrchk(cudaFree(d_batch_attn_final_proj_out));
    gpuErrchk(cudaFree(d_batch_residual_attn_in));
    gpuErrchk(cudaFree(d_batch_residual_ffn_in));
    gpuErrchk(cudaFree(d_batch_x_norm_out_ffn));
    gpuErrchk(cudaFree(d_batch_ffn_gate_proj_out));
    gpuErrchk(cudaFree(d_batch_ffn_up_proj_out));
    gpuErrchk(cudaFree(d_batch_ffn_swiglu_out));
    gpuErrchk(cudaFree(d_batch_ffn_down_proj_out));
    if (d_batch_layer_output) { 
        gpuErrchk(cudaFree(d_batch_layer_output));
    }
    gpuErrchk(cudaFree(d_logits_last_token));
    Logger::info("[FWD_DEV_BATCH_PREFILL_EXIT] Function finished.");
    return h_logits;
}

std::vector<std::vector<float>> TinyLlamaModel::forward_device_batch_generation(
    float* d_batch_input_embeddings,    // Device pointer to [num_tokens_in_batch, config_.hidden_size]
    const std::vector<int>& token_positions, // Position of each token in its respective sequence
    const std::vector<int>& original_sequence_indices, // Original sequence index for each token
    int num_tokens_in_batch,
    KVCache* kv_cache,
    cudaStream_t stream) {
    Logger::info("[FWD_DEV_BATCH_GENERATION_ENTRY] num_tokens_in_batch: " + std::to_string(num_tokens_in_batch) +
                 ", d_batch_input_embeddings: " + Logger::ptrToString(d_batch_input_embeddings));

    if (token_positions.size() != static_cast<size_t>(num_tokens_in_batch)) {
        Logger::error("[FWD_DEV_BATCH_GENERATION] token_positions size mismatch. Expected: " + 
                      std::to_string(num_tokens_in_batch) + " Got: " + std::to_string(token_positions.size()));
        return {};
    }
    if (original_sequence_indices.size() != static_cast<size_t>(num_tokens_in_batch)) {
        Logger::error("[CPU_BATCH_GENERATION] original_sequence_indices size mismatch. Expected: " + 
                      std::to_string(num_tokens_in_batch) + " Got: " + std::to_string(original_sequence_indices.size()));
        return {};
    }
    const int hidden_size = config_.hidden_size;
    const int head_dim = config_.hidden_size / config_.num_attention_heads;
    const int ffn_intermediate_dim = config_.intermediate_size;
    const int n_kv_dim = config_.num_key_value_heads * head_dim;
    const int vocab_size = config_.vocab_size;
    
    const size_t batch_hidden_size_bytes = (size_t)num_tokens_in_batch * hidden_size * sizeof(float);
    const size_t batch_intermediate_size_bytes = (size_t)num_tokens_in_batch * ffn_intermediate_dim * sizeof(float);
    const size_t batch_q_size_bytes = (size_t)num_tokens_in_batch * hidden_size * sizeof(float);
    const size_t batch_kv_size_bytes = (size_t)num_tokens_in_batch * n_kv_dim * sizeof(float);

    // Allocate GPU memory for batch processing
    float* d_batch_x_norm_out_attn;
    float* d_batch_q_proj_out;
    float* d_batch_k_proj_out;
    float* d_batch_v_proj_out;
    float* d_batch_attn_heads_concat_out;
    float* d_batch_attn_final_proj_out;
    float* d_batch_residual_attn_in; 
    float* d_batch_residual_ffn_in;  
    float* d_batch_x_norm_out_ffn;
    float* d_batch_ffn_gate_proj_out;
    float* d_batch_ffn_up_proj_out;
    float* d_batch_ffn_swiglu_out;
    float* d_batch_ffn_down_proj_out;
    float* d_batch_layer_output = nullptr;

    gpuErrchk(cudaMalloc(&d_batch_x_norm_out_attn, batch_hidden_size_bytes));
    gpuErrchk(cudaMalloc(&d_batch_q_proj_out, batch_q_size_bytes));
    gpuErrchk(cudaMalloc(&d_batch_k_proj_out, batch_kv_size_bytes));
    gpuErrchk(cudaMalloc(&d_batch_v_proj_out, batch_kv_size_bytes));
    gpuErrchk(cudaMalloc(&d_batch_attn_heads_concat_out, batch_hidden_size_bytes));
    gpuErrchk(cudaMalloc(&d_batch_attn_final_proj_out, batch_hidden_size_bytes));
    gpuErrchk(cudaMalloc(&d_batch_residual_attn_in, batch_hidden_size_bytes));
    gpuErrchk(cudaMalloc(&d_batch_residual_ffn_in, batch_hidden_size_bytes));
    gpuErrchk(cudaMalloc(&d_batch_x_norm_out_ffn, batch_hidden_size_bytes));
    gpuErrchk(cudaMalloc(&d_batch_ffn_gate_proj_out, batch_intermediate_size_bytes));
    gpuErrchk(cudaMalloc(&d_batch_ffn_up_proj_out, batch_intermediate_size_bytes));
    gpuErrchk(cudaMalloc(&d_batch_ffn_swiglu_out, batch_intermediate_size_bytes));
    gpuErrchk(cudaMalloc(&d_batch_ffn_down_proj_out, batch_hidden_size_bytes));
    gpuErrchk(cudaMalloc(&d_batch_layer_output, batch_hidden_size_bytes));

    const float alpha = 1.0f, beta = 0.0f;

    cublasStatus_t stream_status = cublasSetStream(cublas_handle_, stream);
    if (stream_status != CUBLAS_STATUS_SUCCESS) {
        Logger::fatal("cublasSetStream failed in forward_device_batch_generation");
        // Free allocated memory before returning
        gpuErrchk(cudaFree(d_batch_x_norm_out_attn));
        gpuErrchk(cudaFree(d_batch_q_proj_out));
        gpuErrchk(cudaFree(d_batch_k_proj_out));
        gpuErrchk(cudaFree(d_batch_v_proj_out));
        gpuErrchk(cudaFree(d_batch_attn_heads_concat_out));
        gpuErrchk(cudaFree(d_batch_attn_final_proj_out));
        gpuErrchk(cudaFree(d_batch_residual_attn_in));
        gpuErrchk(cudaFree(d_batch_residual_ffn_in));
        gpuErrchk(cudaFree(d_batch_x_norm_out_ffn));
        gpuErrchk(cudaFree(d_batch_ffn_gate_proj_out));
        gpuErrchk(cudaFree(d_batch_ffn_up_proj_out));
        gpuErrchk(cudaFree(d_batch_ffn_swiglu_out));
        gpuErrchk(cudaFree(d_batch_ffn_down_proj_out));
        gpuErrchk(cudaFree(d_batch_layer_output));
        throw std::runtime_error("cublasSetStream failed");
    }

    float* d_batch_x_ptr = d_batch_input_embeddings; // Input to the first GPU layer

    Logger::info("[FWD_DEV_BATCH_GENERATION_MAIN_LOOP_ENTRY] num_cpu_offload_layers: " + std::to_string(config_.num_cpu_offload_layers) + 
                 ", total_hidden_layers: " + std::to_string(config_.num_hidden_layers));

    for (int l_model_idx = config_.num_cpu_offload_layers; l_model_idx < config_.num_hidden_layers; ++l_model_idx) {
        int l_gpu_idx = l_model_idx - config_.num_cpu_offload_layers;
        Logger::info("[FWD_DEV_BATCH_GENERATION_LAYER_START] Processing Layer: model_idx=" + std::to_string(l_model_idx) + ", gpu_idx=" + std::to_string(l_gpu_idx) + 
                     ". Current d_batch_x_ptr: " + Logger::ptrToString(d_batch_x_ptr));
        
        gpuErrchk(cudaMemcpyAsync(d_batch_residual_attn_in, d_batch_x_ptr, batch_hidden_size_bytes, cudaMemcpyDeviceToDevice, stream));

        rmsnorm_batch_cuda(d_batch_x_norm_out_attn, d_batch_x_ptr, 
                           layers[l_model_idx].input_layernorm_dev,
                           num_tokens_in_batch, hidden_size, config_.rms_norm_eps, stream);
        // Weight dequantization handled by ensure_f32_concatenated_weights_loaded() below
        ensure_f32_concatenated_weights_loaded();
        
        const float* w_q_layer_ptr = w_q_f32_dev_ + (size_t)l_gpu_idx * hidden_size * hidden_size;
        gemm_f32_f32_cuda(cublas_handle_, false, true, num_tokens_in_batch, hidden_size, hidden_size, &alpha,
                          d_batch_x_norm_out_attn, hidden_size, w_q_layer_ptr, hidden_size, &beta,
                          d_batch_q_proj_out, hidden_size, stream);

        const float* w_k_layer_ptr = w_k_f32_dev_ + (size_t)l_gpu_idx * n_kv_dim * hidden_size;
        gemm_f32_f32_cuda(cublas_handle_, false, true, num_tokens_in_batch, n_kv_dim, hidden_size, &alpha,
                          d_batch_x_norm_out_attn, hidden_size, w_k_layer_ptr, n_kv_dim, &beta,
                          d_batch_k_proj_out, n_kv_dim, stream);

        const float* w_v_layer_ptr = w_v_f32_dev_ + (size_t)l_gpu_idx * n_kv_dim * hidden_size;
        gemm_f32_f32_cuda(cublas_handle_, false, true, num_tokens_in_batch, n_kv_dim, hidden_size, &alpha,
                          d_batch_x_norm_out_attn, hidden_size, w_v_layer_ptr, n_kv_dim, &beta,
                          d_batch_v_proj_out, n_kv_dim, stream);

        // For generation mode, we need to apply RoPE with individual positions for each token
        // We'll use the single-token RoPE function in a loop for now
        // TODO: Create a batch RoPE function that handles variable positions
        for (int token_idx = 0; token_idx < num_tokens_in_batch; ++token_idx) {
            int current_pos = token_positions[token_idx];
            
            float* q_token_ptr = d_batch_q_proj_out + (size_t)token_idx * config_.num_attention_heads * head_dim;
            float* k_token_ptr = d_batch_k_proj_out + (size_t)token_idx * config_.num_key_value_heads * head_dim;
            
            rope_cuda(q_token_ptr, config_.num_attention_heads, head_dim, all_freqs_cis_dev, 
                     current_pos, config_.is_gguf_file_loaded, stream);
            rope_cuda(k_token_ptr, config_.num_key_value_heads, head_dim, all_freqs_cis_dev, 
                     current_pos, config_.is_gguf_file_loaded, stream);
        }
// Update KV cache for each token at its specific position with sequence-aware offsets
float* d_layer_k_cache_ptr = kv_cache->layers[l_model_idx].k_dev_fp32; 
float* d_layer_v_cache_ptr = kv_cache->layers[l_model_idx].v_dev_fp32;

for (int token_idx = 0; token_idx < num_tokens_in_batch; ++token_idx) {
    int current_pos = token_positions[token_idx];
    int sequence_idx = original_sequence_indices[token_idx];
    
    // Calculate sequence-specific offset in the cache
    int sequence_cache_offset = sequence_idx * kv_cache->max_seq_len_config_;
    int actual_cache_pos = sequence_cache_offset + current_pos;
    
    const float* k_token_ptr = d_batch_k_proj_out + (size_t)token_idx * config_.num_key_value_heads * head_dim;
    const float* v_token_ptr = d_batch_v_proj_out + (size_t)token_idx * config_.num_key_value_heads * head_dim;
    
    // Update individual positions in the KV cache with sequence-specific offsets
    for (int kvh = 0; kvh < config_.num_key_value_heads; ++kvh) {
        const float* current_k_head_ptr = k_token_ptr + kvh * head_dim;
        const float* current_v_head_ptr = v_token_ptr + kvh * head_dim;

        update_kv_cache_cuda(d_layer_k_cache_ptr, current_k_head_ptr, actual_cache_pos,
                             kvh, kv_cache->allocated_max_seq_len * kv_cache->max_batch_size,
                             kv_cache->allocated_num_kv_heads, 
                             kv_cache->allocated_head_dim, stream); 

        update_kv_cache_cuda(d_layer_v_cache_ptr, current_v_head_ptr, actual_cache_pos,
                             kvh, kv_cache->allocated_max_seq_len * kv_cache->max_batch_size,
                             kv_cache->allocated_num_kv_heads,
                             kv_cache->allocated_head_dim, stream);
    }
}
        // For attention, we need a generation-specific function that handles variable positions
        // For now, we'll use the single-token attention function in a loop
        // TODO: Create a batch attention function for generation
        for (int token_idx = 0; token_idx < num_tokens_in_batch; ++token_idx) {
            int current_pos = token_positions[token_idx];
            
            float* q_token_ptr = d_batch_q_proj_out + (size_t)token_idx * config_.num_attention_heads * head_dim;
            float* attn_output_token_ptr = d_batch_attn_heads_concat_out + (size_t)token_idx * config_.num_attention_heads * head_dim;
            
            float scale = 1.0f / SAFE_SQRT(static_cast<float>(head_dim));
            
            if (config_.use_kvcache_quantization && 
                selective_k_dequant_buffer_dev_ && selective_v_dequant_buffer_dev_) {
                KVCacheLayer& batch_kv_layer = kv_cache->layers[l_model_idx];
                attention_cuda_selective_dequant(
                    q_token_ptr,
                    batch_kv_layer.k_dev_quantized,
                    batch_kv_layer.v_dev_quantized,
                    batch_kv_layer.k_dev_scales,
                    batch_kv_layer.v_dev_scales,
                    selective_k_dequant_buffer_dev_,
                    selective_v_dequant_buffer_dev_,
                    attn_output_token_ptr,
                    config_.num_attention_heads,
                    current_pos + 1,
                    head_dim,
                    scale,
                    kv_cache->allocated_max_seq_len,
                    kv_cache->allocated_num_kv_heads,
                    stream
                );
            } else {
            attention_cuda(q_token_ptr, d_layer_k_cache_ptr, d_layer_v_cache_ptr,
                          attn_output_token_ptr, config_.num_attention_heads, current_pos + 1, head_dim,
                          scale, kv_cache->allocated_max_seq_len, kv_cache->allocated_num_kv_heads, stream);
            }
        }

        const float* w_o_layer_ptr = w_o_f32_dev_ + (size_t)l_gpu_idx * hidden_size * hidden_size;
        gemm_f32_f32_cuda(cublas_handle_, false, true, num_tokens_in_batch, hidden_size, hidden_size, &alpha,
                          d_batch_attn_heads_concat_out, hidden_size, w_o_layer_ptr, hidden_size, &beta,
                          d_batch_attn_final_proj_out, hidden_size, stream);

        add_residual_batch_cuda(d_batch_residual_ffn_in, d_batch_attn_final_proj_out, d_batch_residual_attn_in,
                                num_tokens_in_batch, hidden_size, stream);

        rmsnorm_batch_cuda(d_batch_x_norm_out_ffn, d_batch_residual_ffn_in, 
                           layers[l_model_idx].post_attention_layernorm_dev,
                           num_tokens_in_batch, hidden_size, config_.rms_norm_eps, stream);

        const float* w1_layer_ptr = w_gate_f32_dev_ + (size_t)l_gpu_idx * hidden_size * ffn_intermediate_dim;
        gemm_f32_f32_cuda(cublas_handle_, false, true, num_tokens_in_batch, ffn_intermediate_dim, hidden_size, &alpha,
                          d_batch_x_norm_out_ffn, hidden_size, w1_layer_ptr, ffn_intermediate_dim, &beta,
                          d_batch_ffn_gate_proj_out, ffn_intermediate_dim, stream);

        const float* w3_layer_ptr = w_up_f32_dev_ + (size_t)l_gpu_idx * hidden_size * ffn_intermediate_dim;
        gemm_f32_f32_cuda(cublas_handle_, false, true, num_tokens_in_batch, ffn_intermediate_dim, hidden_size, &alpha,
                          d_batch_x_norm_out_ffn, hidden_size, w3_layer_ptr, ffn_intermediate_dim, &beta,
                          d_batch_ffn_up_proj_out, ffn_intermediate_dim, stream);

        swiglu_batch_cuda(d_batch_ffn_swiglu_out, d_batch_ffn_gate_proj_out, d_batch_ffn_up_proj_out,
                          num_tokens_in_batch, ffn_intermediate_dim, stream);

        const float* w2_layer_ptr = w_down_f32_dev_ + (size_t)l_gpu_idx * ffn_intermediate_dim * hidden_size;
        gemm_f32_f32_cuda(cublas_handle_, false, true, num_tokens_in_batch, hidden_size, ffn_intermediate_dim, &alpha,
                          d_batch_ffn_swiglu_out, ffn_intermediate_dim, w2_layer_ptr, hidden_size, &beta,
                          d_batch_ffn_down_proj_out, hidden_size, stream);

        add_residual_batch_cuda(d_batch_layer_output, d_batch_ffn_down_proj_out, d_batch_residual_ffn_in,
                                num_tokens_in_batch, hidden_size, stream);
        
        d_batch_x_ptr = d_batch_layer_output; 
        Logger::info("[FWD_DEV_BATCH_GENERATION_LAYER_END] Layer " + std::to_string(l_model_idx) + " finished. Next d_batch_x_ptr: " + Logger::ptrToString(d_batch_x_ptr));
    }

    rmsnorm_batch_cuda(d_batch_x_norm_out_attn, d_batch_x_ptr, 
                       final_norm_dev,
                       num_tokens_in_batch, hidden_size, config_.rms_norm_eps, stream);
    
    // Calculate logits for ALL tokens in the batch (not just the last one)
    float* d_logits_batch;
    gpuErrchk(cudaMalloc(&d_logits_batch, (size_t)num_tokens_in_batch * vocab_size * sizeof(float)));
    
    // Use GEMM instead of individual MatVec calls for efficiency
    gemm_f32_f32_cuda(cublas_handle_, false, true, num_tokens_in_batch, vocab_size, hidden_size, &alpha,
                      d_batch_x_norm_out_attn, hidden_size, lm_head_f32_dev_, vocab_size, &beta,
                      d_logits_batch, vocab_size, stream);

    // Copy logits back to host for all tokens
    std::vector<std::vector<float>> all_logits(num_tokens_in_batch, std::vector<float>(vocab_size));
    for (int token_idx = 0; token_idx < num_tokens_in_batch; ++token_idx) {
        gpuErrchk(cudaMemcpyAsync(all_logits[token_idx].data(), 
                                  d_logits_batch + (size_t)token_idx * vocab_size, 
                                  vocab_size * sizeof(float),
                                  cudaMemcpyDeviceToHost, stream));
    }
    gpuErrchk(cudaStreamSynchronize(stream)); 

    Logger::info("[FWD_DEV_BATCH_GENERATION_FINAL_LOGITS] Calculated logits for " + std::to_string(num_tokens_in_batch) + " tokens");

    // Free allocated memory
    gpuErrchk(cudaFree(d_batch_x_norm_out_attn));
    gpuErrchk(cudaFree(d_batch_q_proj_out));
    gpuErrchk(cudaFree(d_batch_k_proj_out));
    gpuErrchk(cudaFree(d_batch_v_proj_out));
    gpuErrchk(cudaFree(d_batch_attn_heads_concat_out));
    gpuErrchk(cudaFree(d_batch_attn_final_proj_out));
    gpuErrchk(cudaFree(d_batch_residual_attn_in));
    gpuErrchk(cudaFree(d_batch_residual_ffn_in));
    gpuErrchk(cudaFree(d_batch_x_norm_out_ffn));
    gpuErrchk(cudaFree(d_batch_ffn_gate_proj_out));
    gpuErrchk(cudaFree(d_batch_ffn_up_proj_out));
    gpuErrchk(cudaFree(d_batch_ffn_swiglu_out));
    gpuErrchk(cudaFree(d_batch_ffn_down_proj_out));
    if (d_batch_layer_output) { 
        gpuErrchk(cudaFree(d_batch_layer_output));
    }
    gpuErrchk(cudaFree(d_logits_batch));
    
    Logger::info("[FWD_DEV_BATCH_GENERATION_EXIT] Function finished.");
    return all_logits;
}
#endif // HAS_CUDA

std::vector<float> TinyLlamaModel::forward_cpu_batch(
    const std::vector<float>& batch_input_activations, // [num_tokens, hidden_size]
    int num_tokens_in_batch,
    int num_cpu_layers_to_process,
    int start_pos_in_sequence, // Starting position of this batch in the overall sequence (for KVCache)
    KVCache* kv_cache,
    const std::vector<int>& prompt_lengths) {

    if (batch_input_activations.size() != (size_t)num_tokens_in_batch * config_.hidden_size) {
        Logger::error("[CPU_BATCH_FWD] Input size mismatch. Expected: " +
                      std::to_string((size_t)num_tokens_in_batch * config_.hidden_size) + " Got: " +
                      std::to_string(batch_input_activations.size()));
        return {};
    }

    int hs = config_.hidden_size;
    int is = config_.intermediate_size;
    int n_heads = config_.num_attention_heads;
    int n_kv_heads = config_.num_key_value_heads;
    if (n_heads == 0) {
        Logger::error("[CPU_BATCH_FWD] Error: num_attention_heads is zero.");
        return {};
    }
    int head_dim = hs / n_heads;
    float eps = config_.rms_norm_eps;
    int max_pos_embeddings = config_.max_position_embeddings;
    bool use_rope_adjacent_pairing = config_.is_gguf_file_loaded;
    float attention_scale = 1.0f / SAFE_SQRT(static_cast<float>(head_dim));

    std::vector<float> current_batch_activations = batch_input_activations;

    // Calculate sequence boundaries and positions
    std::vector<int> sequence_indices(num_tokens_in_batch);
    std::vector<int> position_in_sequence(num_tokens_in_batch);
    
    if (!prompt_lengths.empty()) {
        // Multi-sequence batch mode
        int token_offset = 0;
        for (size_t seq_idx = 0; seq_idx < prompt_lengths.size(); ++seq_idx) {
            for (int pos = 0; pos < prompt_lengths[seq_idx]; ++pos) {
                if (token_offset >= num_tokens_in_batch) {
                    Logger::error("[CPU_BATCH_FWD] Token offset exceeded num_tokens_in_batch");
                    return {};
                }
                sequence_indices[token_offset] = seq_idx;
                position_in_sequence[token_offset] = pos;
                token_offset++;
            }
        }
    } else {
        // Single sequence mode (backward compatibility)
        for (int token_idx = 0; token_idx < num_tokens_in_batch; ++token_idx) {
            sequence_indices[token_idx] = 0;
            position_in_sequence[token_idx] = start_pos_in_sequence + token_idx;
        }
    }

    for (int l = 0; l < num_cpu_layers_to_process; ++l) {
        // Ensure individual weights are dequantized for memory efficiency
        ensure_q_proj_dequantized(l);
        ensure_k_proj_dequantized(l);
        ensure_v_proj_dequantized(l);
        ensure_o_proj_dequantized(l);
        ensure_gate_proj_dequantized(l);
        ensure_up_proj_dequantized(l);
        ensure_down_proj_dequantized(l);
        
        const auto& lw = layers[l];
        
        std::vector<float> batch_x_norm1(current_batch_activations.size());
        const std::vector<float>& w_input_norm_vec =
            lw.input_layernorm_f32.empty()
                ? bf16vec_to_float_vec(lw.input_layernorm)
                : lw.input_layernorm_f32;
        rmsnorm_batch_cpu(current_batch_activations, w_input_norm_vec, batch_x_norm1, num_tokens_in_batch, hs, eps);

        std::vector<float> residual_batch_component_attn = current_batch_activations;

        // Batch Q, K, V projections
        std::vector<float> q_batch((size_t)num_tokens_in_batch * hs);
        std::vector<float> k_batch((size_t)num_tokens_in_batch * n_kv_heads * head_dim);
        std::vector<float> v_batch((size_t)num_tokens_in_batch * n_kv_heads * head_dim);

        // Q Projection
        if (!lw.q_proj_f32.empty()) {
            matmul_f32_f32_batch_cpu(lw.q_proj_f32, batch_x_norm1, q_batch, num_tokens_in_batch, hs, hs);
        } else if (!lw.q_proj_q8_0.empty()) {
            matmul_q8_0_f32_batch_cpu(lw.q_proj_q8_0, batch_x_norm1, q_batch, num_tokens_in_batch, hs, hs);
        } else if (!lw.q_proj_q6k.empty()) {
            matmul_q6k_f32_batch_cpu(lw.q_proj_q6k, batch_x_norm1, q_batch, num_tokens_in_batch, hs, hs);
        } else if (!lw.q_proj_q4k.empty()) {
            matmul_q4k_f32_batch_cpu(lw.q_proj_q4k, batch_x_norm1, q_batch, num_tokens_in_batch, hs, hs);
        } else {
            Logger::error("[CPU_BATCH_FWD] Layer " + std::to_string(l) + ": No Q proj weights found for CPU");
            return {};
        }

        // K Projection
        if (!lw.k_proj_f32.empty()) {
            matmul_f32_f32_batch_cpu(lw.k_proj_f32, batch_x_norm1, k_batch, num_tokens_in_batch, n_kv_heads * head_dim, hs);
        } else if (!lw.k_proj_q8_0.empty()) {
            matmul_q8_0_f32_batch_cpu(lw.k_proj_q8_0, batch_x_norm1, k_batch, num_tokens_in_batch, n_kv_heads * head_dim, hs);
        } else if (!lw.k_proj_q6k.empty()) {
            matmul_q6k_f32_batch_cpu(lw.k_proj_q6k, batch_x_norm1, k_batch, num_tokens_in_batch, n_kv_heads * head_dim, hs);
        } else if (!lw.k_proj_q4k.empty()) {
            matmul_q4k_f32_batch_cpu(lw.k_proj_q4k, batch_x_norm1, k_batch, num_tokens_in_batch, n_kv_heads * head_dim, hs);
        } else {
            Logger::error("[CPU_BATCH_FWD] Layer " + std::to_string(l) + ": No K proj weights found for CPU");
            return {};
        }

        // V Projection
        if (!lw.v_proj_f32.empty()) {
            matmul_f32_f32_batch_cpu(lw.v_proj_f32, batch_x_norm1, v_batch, num_tokens_in_batch, n_kv_heads * head_dim, hs);
        } else if (!lw.v_proj_q8_0.empty()) {
            matmul_q8_0_f32_batch_cpu(lw.v_proj_q8_0, batch_x_norm1, v_batch, num_tokens_in_batch, n_kv_heads * head_dim, hs);
        } else if (!lw.v_proj_q6k.empty()) {
            matmul_q6k_f32_batch_cpu(lw.v_proj_q6k, batch_x_norm1, v_batch, num_tokens_in_batch, n_kv_heads * head_dim, hs);
        } else if (!lw.v_proj_q4k.empty()) {
            matmul_q4k_f32_batch_cpu(lw.v_proj_q4k, batch_x_norm1, v_batch, num_tokens_in_batch, n_kv_heads * head_dim, hs);
        } else {
            Logger::error("[CPU_BATCH_FWD] Layer " + std::to_string(l) + ": No V proj weights found for CPU");
            return {};
        }

        // FIXED: Sequence-aware RoPE Application
        if (!prompt_lengths.empty()) {
            // Multi-sequence mode: use sequence-aware RoPE
            for (int t = 0; t < num_tokens_in_batch; ++t) {
                int current_token_pos = position_in_sequence[t];
                int seq_idx = sequence_indices[t];

                if (current_token_pos < 0 || current_token_pos >= max_pos_embeddings) {
                    Logger::warning("[CPU_BATCH_FWD] Token " + std::to_string(t) + " (seq=" + std::to_string(seq_idx) + 
                                    ", pos=" + std::to_string(current_token_pos) + ") is out of range. Skipping RoPE.");
                    continue;
                }

                // Extract Q and K for this token
                std::vector<float> q_token(hs);
                std::vector<float> k_token(n_kv_heads * head_dim);
                
                std::copy(q_batch.begin() + (size_t)t * hs, 
                         q_batch.begin() + (size_t)(t + 1) * hs, 
                         q_token.begin());
                std::copy(k_batch.begin() + (size_t)t * n_kv_heads * head_dim, 
                         k_batch.begin() + (size_t)(t + 1) * n_kv_heads * head_dim, 
                         k_token.begin());

                // Apply RoPE with correct position
                apply_rope_vector(q_token, n_heads, head_dim, current_token_pos, precomputed_freqs_cis_, max_pos_embeddings, use_rope_adjacent_pairing);
                apply_rope_vector(k_token, n_kv_heads, head_dim, current_token_pos, precomputed_freqs_cis_, max_pos_embeddings, use_rope_adjacent_pairing);

                // Copy back to batch
                std::copy(q_token.begin(), q_token.end(), q_batch.begin() + (size_t)t * hs);
                std::copy(k_token.begin(), k_token.end(), k_batch.begin() + (size_t)t * n_kv_heads * head_dim);
            }
        } else {
            // Single sequence mode: use original batch RoPE
            apply_rope_batch_cpu(q_batch, k_batch, num_tokens_in_batch, n_heads, n_kv_heads, head_dim, 
                                  start_pos_in_sequence, precomputed_freqs_cis_, max_pos_embeddings, use_rope_adjacent_pairing);
        }

        // Batched KV Cache Update
        if (kv_cache) {
            if (!prompt_lengths.empty()) {
                // Multi-sequence mode: use sequence-aware update
                update_kv_cache_batch_cpu_sequence_aware(kv_cache, l, k_batch, v_batch, num_tokens_in_batch,
                                                         sequence_indices, position_in_sequence, n_kv_heads, head_dim);
            } else {
                // Single sequence mode: use original update
            update_kv_cache_batch_cpu(kv_cache, l, k_batch, v_batch, num_tokens_in_batch, 
                                      start_pos_in_sequence, n_kv_heads, head_dim);        
            }
        }
        
        // Batched Attention
        std::vector<float> batch_attn_output((size_t)num_tokens_in_batch * hs);
        
        if (kv_cache && static_cast<size_t>(l) < kv_cache->layers.size()) {
            if (!prompt_lengths.empty()) {
                // Multi-sequence mode: use sequence-aware attention
                attention_batch_cpu_sequence_aware(q_batch, kv_cache->layers[l], batch_attn_output,
                                                  num_tokens_in_batch, sequence_indices, position_in_sequence,
                                                  n_heads, n_kv_heads, head_dim, attention_scale,
                                                  kv_cache->max_seq_len_config_);
            } else {
                // Single sequence mode: use original attention
                attention_batch_cpu(q_batch, kv_cache->layers[l], batch_attn_output,
                                   num_tokens_in_batch, start_pos_in_sequence,
                                   n_heads, n_kv_heads, head_dim, attention_scale);
            }
        } else if (kv_cache) { 
            Logger::error("[CPU_BATCH_FWD] Layer " + std::to_string(l) + 
                          " is out of bounds for KV Cache access during attention. KVCache layers size: " + 
                          std::to_string(kv_cache->layers.size()) + 
                          ". Filling attention output with zeros.");
            std::fill(batch_attn_output.begin(), batch_attn_output.end(), 0.0f); 
        } else {
            Logger::error("[CPU_BATCH_FWD] KV Cache is null, cannot perform attention for layer " + std::to_string(l) +
                          ". Filling attention output with zeros.");
            std::fill(batch_attn_output.begin(), batch_attn_output.end(), 0.0f); 
        }

        // O-Projection (Batched)
        std::vector<float> batch_attn_proj_out((size_t)num_tokens_in_batch * hs);
        if(!lw.o_proj_f32.empty()) {
              matmul_f32_f32_batch_cpu(lw.o_proj_f32, batch_attn_output, batch_attn_proj_out, num_tokens_in_batch, hs, hs);
        } else if (!lw.o_proj_q8_0.empty()) {
            matmul_q8_0_f32_batch_cpu(lw.o_proj_q8_0, batch_attn_output, batch_attn_proj_out, num_tokens_in_batch, hs, hs);
        } else if (!lw.o_proj_q6k.empty()) {
            matmul_q6k_f32_batch_cpu(lw.o_proj_q6k, batch_attn_output, batch_attn_proj_out, num_tokens_in_batch, hs, hs);
        } else if (!lw.o_proj_q4k.empty()) {
            matmul_q4k_f32_batch_cpu(lw.o_proj_q4k, batch_attn_output, batch_attn_proj_out, num_tokens_in_batch, hs, hs);
        } else { 
            Logger::error("[CPU_BATCH_FWD] Layer " + std::to_string(l) + ": No O proj weights found for CPU"); 
            return {};
        }

        // First Residual Connection (Batched)
        for(size_t i=0; i < current_batch_activations.size(); ++i) {
            current_batch_activations[i] = residual_batch_component_attn[i] + batch_attn_proj_out[i];
        }

        // --- Batched MLP Part ---
        std::vector<float> residual_batch_component_mlp = current_batch_activations;
        std::vector<float> batch_x_norm2(current_batch_activations.size());
        
        const std::vector<float>& w_post_attn_norm_vec =
            lw.post_attention_layernorm_f32.empty()
                ? bf16vec_to_float_vec(lw.post_attention_layernorm)
                : lw.post_attention_layernorm_f32;
        // Batched RMSNorm for MLP
        rmsnorm_batch_cpu(current_batch_activations, w_post_attn_norm_vec, batch_x_norm2, num_tokens_in_batch, hs, eps);
        
        // Batched Gate and Up Projections
        std::vector<float> batch_gate_proj_out((size_t)num_tokens_in_batch * is);
        std::vector<float> batch_up_proj_out((size_t)num_tokens_in_batch * is);

        // Gate Projection
        if (!lw.gate_proj_f32.empty()) {
            matmul_f32_f32_batch_cpu(lw.gate_proj_f32, batch_x_norm2, batch_gate_proj_out, num_tokens_in_batch, is, hs);
        } else if (!lw.gate_proj_q8_0.empty()) {
            matmul_q8_0_f32_batch_cpu(lw.gate_proj_q8_0, batch_x_norm2, batch_gate_proj_out, num_tokens_in_batch, is, hs);
        } else if (!lw.gate_proj_q6k.empty()) {
            matmul_q6k_f32_batch_cpu(lw.gate_proj_q6k, batch_x_norm2, batch_gate_proj_out, num_tokens_in_batch, is, hs);
        } else if (!lw.gate_proj_q4k.empty()) {
            matmul_q4k_f32_batch_cpu(lw.gate_proj_q4k, batch_x_norm2, batch_gate_proj_out, num_tokens_in_batch, is, hs);
        } else { 
            Logger::error("[CPU_BATCH_FWD] Layer " + std::to_string(l) + ": No gate_proj weights found for CPU"); 
            return {};
        }

        // Up Projection
        if (!lw.up_proj_f32.empty()) {
            matmul_f32_f32_batch_cpu(lw.up_proj_f32, batch_x_norm2, batch_up_proj_out, num_tokens_in_batch, is, hs);
        } else if (!lw.up_proj_q8_0.empty()) {
            matmul_q8_0_f32_batch_cpu(lw.up_proj_q8_0, batch_x_norm2, batch_up_proj_out, num_tokens_in_batch, is, hs);
        } else if (!lw.up_proj_q6k.empty()) {
            matmul_q6k_f32_batch_cpu(lw.up_proj_q6k, batch_x_norm2, batch_up_proj_out, num_tokens_in_batch, is, hs);
        } else if (!lw.up_proj_q4k.empty()) {
            matmul_q4k_f32_batch_cpu(lw.up_proj_q4k, batch_x_norm2, batch_up_proj_out, num_tokens_in_batch, is, hs);
        } else { 
            Logger::error("[CPU_BATCH_FWD] Layer " + std::to_string(l) + ": No up_proj weights found for CPU"); 
            return {};
        }
        // Batched SwiGLU: SiLU(gate_proj_out) * up_proj_out
        std::vector<float> batch_swiglu_out((size_t)num_tokens_in_batch * is);
        for (size_t i = 0; i < batch_gate_proj_out.size(); ++i) {
            float gate_val = batch_gate_proj_out[i];
            float silu_gate_val = gate_val / (1.0f + std::exp(-gate_val));
            batch_swiglu_out[i] = silu_gate_val * batch_up_proj_out[i];
        }
        
        // Down Projection
        std::vector<float> batch_mlp_down_proj_out((size_t)num_tokens_in_batch * hs);
        if (!lw.down_proj_f32.empty()) {
            matmul_f32_f32_batch_cpu(lw.down_proj_f32, batch_swiglu_out, batch_mlp_down_proj_out, num_tokens_in_batch, hs, is);
        } else if (!lw.down_proj_q8_0.empty()) {
            matmul_q8_0_f32_batch_cpu(lw.down_proj_q8_0, batch_swiglu_out, batch_mlp_down_proj_out, num_tokens_in_batch, hs, is);
        } else if (!lw.down_proj_q6k.empty()) {
            matmul_q6k_f32_batch_cpu(lw.down_proj_q6k, batch_swiglu_out, batch_mlp_down_proj_out, num_tokens_in_batch, hs, is);
        } else if (!lw.down_proj_q4k.empty()) {
            matmul_q4k_f32_batch_cpu(lw.down_proj_q4k, batch_swiglu_out, batch_mlp_down_proj_out, num_tokens_in_batch, hs, is);
        } else { 
            Logger::error("[CPU_BATCH_FWD] Layer " + std::to_string(l) + ": No down_proj weights found for CPU"); 
            return {};
        }
        // Second Residual Connection (Batched)
        for(size_t i = 0; i < current_batch_activations.size(); ++i) {
            current_batch_activations[i] = residual_batch_component_mlp[i] + batch_mlp_down_proj_out[i];
        }
    } // End layer loop

    if (kv_cache && num_tokens_in_batch > 0) {
        kv_cache->seq_len = start_pos_in_sequence + num_tokens_in_batch;
    }

    return current_batch_activations;
}