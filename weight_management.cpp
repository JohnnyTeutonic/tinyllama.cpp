#include "weight_management.h"
#include "ggml_types.h"
#include "logger.h"
#include "quantization.h"
#include "utils.h"

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
    if (layer_idx < 0 || layer_idx >= layers.size()) {
        Logger::warning("clear_layer_dequantized_weights: Invalid layer index " + std::to_string(layer_idx));
        return;
    }
    
    LayerWeights& lw = layers[layer_idx];
    
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
    
    Logger::info("Cleared dequantized weights for layer " + std::to_string(layer_idx));
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

