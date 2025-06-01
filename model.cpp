#include "model.h"

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

#include "cpu_attention.h"
#include "cpu_batch_processor.h"
#include "gguf_parser.h"
#include "gpu_initialization.h"
#include "kv_cache.h"
#include "logger.h"
#include "model_config.h"
#include "model_constants.h"
#include "model_macros.h"
#include "model_utils.h"
#include "quantization.h"
#include "safetensors_loader.h"
#include "utils.h"
#include "weight_management.h"
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
  this->config_ = initial_config;
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
      this->config_.use_mmap_for_gguf = cli_mmap_preference;
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

  // Free persistent batch processing buffers
  free_persistent_batch_buffers();

    Logger::info("Freed persistent GPU workspace buffers.");
    Logger::info("Finished freeing TinyLlamaModel CUDA weight memory.");
  } else {
    Logger::info("CPU-only mode: No GPU resources to free.");
  }
#endif
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
    bool enable_debug_logging = (l == 0);
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
    gpuErrchk(cudaStreamSynchronize(stream));
    
  if (stream_status != CUBLAS_STATUS_SUCCESS) {
    Logger::error("cublasSetStream failed in forward_device");
    return {};
  }
  float* current_x_dev = x_input_dev;
  for (int l_gpu_idx = 0; l_gpu_idx < num_gpu_layers; ++l_gpu_idx) {
    int l_model_idx = num_cpu_layers + l_gpu_idx;
    
    // Layer-specific norm weights are indexed by the model layer index (l_model_idx)
    const float* lw_in_norm_dev = layers[l_model_idx].input_layernorm_dev;
    const float* lw_post_norm_dev = layers[l_model_idx].post_attention_layernorm_dev;

    gpuErrchk(cudaMemcpyAsync(x_resid1_dev_, x_dev_, hs * sizeof(float),
                              cudaMemcpyDeviceToDevice, stream));

    if (!lw_in_norm_dev) { 
        throw std::runtime_error("[TM::fw_dev pos=" + std::to_string(pos) + " L" + std::to_string(l_model_idx) + "] Error: input_layernorm_dev is nullptr. GPU layer cannot proceed.");
    }
    
    // Use optimized kernels if enabled, fallback to standard if needed
    if (config_.use_optimized_cuda_kernels) {
        rmsnorm_vector_cuda_optimized(x_dev_, lw_in_norm_dev, x_norm_dev_, hs, eps, stream);
    } else {
        rmsnorm_vector_cuda(x_dev_, lw_in_norm_dev, x_norm_dev_, hs, eps, stream);
    }
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
    rope_cuda(q_dev_, n_heads, head_dim, all_freqs_cis_dev, pos, config_.is_gguf_file_loaded, stream);
    rope_cuda(k_dev_, n_kv_heads, head_dim, all_freqs_cis_dev, pos, config_.is_gguf_file_loaded, stream);

    // K/V Cache Update Logic
    if (static_cast<size_t>(l_model_idx) < kv_cache->layers.size()) {
        KVCacheLayer& current_kv_layer = kv_cache->layers[l_model_idx];
        if (config_.use_kvcache_quantization) {
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
        return {};
    }

    float scale = 1.0f / SAFE_SQRT(static_cast<float>(head_dim));
    const float* attention_k_cache_ptr_dev = nullptr;
    const float* attention_v_cache_ptr_dev = nullptr;
    KVCacheLayer& attention_kv_layer = kv_cache->layers[l_model_idx]; 

    if (config_.use_kvcache_quantization) {
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
    // Use optimized kernels if enabled, fallback to standard if needed
    if (config_.use_optimized_cuda_kernels) {
        attention_cuda_optimized(
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
    
    // Use optimized kernels if enabled, fallback to standard if needed
    if (config_.use_optimized_cuda_kernels) {
        rmsnorm_vector_cuda_optimized(current_x_dev, lw_post_norm_dev, x_norm_dev_, hs, eps, stream);
    } else {
        rmsnorm_vector_cuda(current_x_dev, lw_post_norm_dev, x_norm_dev_, hs, eps, stream);
    }

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
    
    // Use optimized kernels if enabled, fallback to standard if needed
    if (config_.use_optimized_cuda_kernels) {
        rmsnorm_vector_cuda_optimized(current_x_dev, lw_post_norm_dev, x_norm_dev_, hs, eps, stream);
    } else {
        rmsnorm_vector_cuda(current_x_dev, lw_post_norm_dev, x_norm_dev_, hs, eps, stream);
    }

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

  }
  // Use optimized kernels if enabled, fallback to standard if needed
  if (config_.use_optimized_cuda_kernels) {
      rmsnorm_vector_cuda_optimized(x_dev_, final_norm_dev, x_norm_dev_, hs, eps, stream);
  } else {
      rmsnorm_vector_cuda(x_dev_, final_norm_dev, x_norm_dev_, hs, eps, stream);
  }
  ensure_lm_head_dequantized();
  if (lm_head_dev_) { 
    matvec_bf16_f32_cuda(cublas_handle_, lm_head_dev_, x_norm_dev_, logits_dev_,
                         vs, hs, stream);
  } else {
    Logger::error("LM head (lm_head_dev_ for BF16) is null. Cannot calculate logits on GPU.");
    return {};
  }

  gpuErrchk(cudaStreamSynchronize(stream));
  std::vector<float> logits(vs);
  gpuErrchk(cudaMemcpy(logits.data(), logits_dev_, vs * sizeof(float),
                       cudaMemcpyDeviceToHost));
  return logits;
}

#endif // HAS_CUDA

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
    resize_persistent_batch_buffers_if_needed(num_tokens_in_batch);
    
    // Assign persistent buffers instead of allocating per forward pass
    d_batch_x_norm_out_attn = d_persistent_batch_norm_out_;
    d_batch_q_proj_out = d_persistent_q_batch_;
    d_batch_k_proj_out = d_persistent_k_batch_;
    d_batch_v_proj_out = d_persistent_v_batch_;
    d_batch_attn_heads_concat_out = d_persistent_attn_output_;
    d_batch_attn_final_proj_out = d_persistent_attn_proj_out_;
    d_batch_residual_attn_in = d_persistent_batch_residual_;
    d_batch_residual_ffn_in = d_persistent_batch_residual_ + num_tokens_in_batch * hidden_size; // Offset for second residual
    d_batch_x_norm_out_ffn = d_persistent_batch_norm_out_;  // Can reuse norm buffer
    d_batch_ffn_gate_proj_out = d_persistent_gate_proj_out_;
    d_batch_ffn_up_proj_out = d_persistent_up_proj_out_;
    d_batch_ffn_swiglu_out = d_persistent_swiglu_out_;
    d_batch_ffn_down_proj_out = d_persistent_mlp_down_out_;

    if (config_.num_cpu_offload_layers < config_.num_hidden_layers) {
         d_batch_layer_output = d_persistent_batch_input_; // Can reuse input buffer for layer output
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

        if (l_model_idx == config_.num_cpu_offload_layers && num_tokens_in_batch > 1 && hidden_size > 0) { /* ... */ }
        add_residual_batch_cuda(d_batch_residual_ffn_in, d_batch_attn_final_proj_out, d_batch_residual_attn_in,
                                num_tokens_in_batch, hidden_size, stream);
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

  if (num_tokens_in_batch > 0) {
      std::vector<float> h_last_token_hidden_state(config_.hidden_size);
      size_t offset_last_token_hidden_state = (size_t)(num_tokens_in_batch - 1) * config_.hidden_size;
      
      gpuErrchk(cudaMemcpyAsync(h_last_token_hidden_state.data(),
                                d_batch_x_ptr + offset_last_token_hidden_state,
                                config_.hidden_size * sizeof(float),
                                cudaMemcpyDeviceToHost, stream));
      gpuErrchk(cudaStreamSynchronize(stream)); 
      Logger::log_vector_stats("[FWD_DEV_BATCH_PREFILL_LAST_TOKEN_HIDDEN_STATE_PRE_FINAL_RMSNORM]", h_last_token_hidden_state, 20);
  }
    rmsnorm_batch_cuda(d_batch_x_norm_out_attn, d_batch_x_ptr, 
                       final_norm_dev,
                       num_tokens_in_batch, hidden_size, config_.rms_norm_eps, stream);
    
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

    if (config_.num_hidden_layers > config_.num_cpu_offload_layers && kv_cache != nullptr && num_tokens_in_batch > 0) {
        int first_gpu_layer_model_idx = config_.num_cpu_offload_layers;
        if (static_cast<size_t>(first_gpu_layer_model_idx) < kv_cache->layers.size()) {
            const KVCacheLayer& cache_layer_to_log = kv_cache->layers[first_gpu_layer_model_idx];
            const float* d_k_cache_ptr = cache_layer_to_log.k_dev_fp32;
            const float* d_v_cache_ptr = cache_layer_to_log.v_dev_fp32;
            const int num_kv_h = config_.num_key_value_heads;
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

    resize_persistent_batch_buffers_if_needed(num_tokens_in_batch);

    d_batch_x_norm_out_attn = d_persistent_batch_norm_out_;
    d_batch_q_proj_out = d_persistent_q_batch_;
    d_batch_k_proj_out = d_persistent_k_batch_;
    d_batch_v_proj_out = d_persistent_v_batch_;
    d_batch_attn_heads_concat_out = d_persistent_attn_output_;
    d_batch_attn_final_proj_out = d_persistent_attn_proj_out_;
    d_batch_residual_attn_in = d_persistent_batch_residual_;
    d_batch_residual_ffn_in = d_persistent_batch_residual_ + num_tokens_in_batch * hidden_size;
    d_batch_x_norm_out_ffn = d_persistent_batch_norm_out_;
    d_batch_ffn_gate_proj_out = d_persistent_gate_proj_out_;
    d_batch_ffn_up_proj_out = d_persistent_up_proj_out_;
    d_batch_ffn_swiglu_out = d_persistent_swiglu_out_;
    d_batch_ffn_down_proj_out = d_persistent_mlp_down_out_;
    d_batch_layer_output = d_persistent_batch_input_;

    const float alpha = 1.0f, beta = 0.0f;

    cublasStatus_t stream_status = cublasSetStream(cublas_handle_, stream);
    if (stream_status != CUBLAS_STATUS_SUCCESS) {
        Logger::fatal("cublasSetStream failed in forward_device_batch_generation");
        throw std::runtime_error("cublasSetStream failed");
    }

    float* d_batch_x_ptr = d_batch_input_embeddings;

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
            // Use optimized kernels if enabled, fallback to standard if needed
            if (config_.use_optimized_cuda_kernels) {
                attention_cuda_optimized(q_token_ptr, d_layer_k_cache_ptr, d_layer_v_cache_ptr,
                              attn_output_token_ptr, config_.num_attention_heads, current_pos + 1, head_dim,
                              scale, kv_cache->allocated_max_seq_len, kv_cache->allocated_num_kv_heads, stream);
            } else {
                attention_cuda(q_token_ptr, d_layer_k_cache_ptr, d_layer_v_cache_ptr,
                             attn_output_token_ptr, config_.num_attention_heads, current_pos + 1, head_dim,
                             scale, kv_cache->allocated_max_seq_len, kv_cache->allocated_num_kv_heads, stream);
            }
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

    if (!cpu_batch_processor_) {
        cpu_batch_processor_ = std::make_unique<CPUBatchProcessor>(this);
    }
    
    return cpu_batch_processor_->forward_cpu_batch(
        batch_input_activations,
        num_tokens_in_batch,
        num_cpu_layers_to_process,
        start_pos_in_sequence,
        kv_cache,
        prompt_lengths
    );
}
