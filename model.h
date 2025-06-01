#ifndef MODEL_H
#define MODEL_H

#include <cstdint>
#include <functional>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <vector>

#include "safetensors_loader.h"
#ifdef HAS_CUDA
// Use safe headers only for Windows CUDA 12.1+ workaround, normal headers everywhere else
#if defined(WINDOWS_CUDA_12_1_WORKAROUND) && defined(_WIN32)
#include "cuda_safe_headers.h"
#else
// Normal CUDA header inclusion for non-problematic platforms (Ubuntu, etc.)
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#endif

#include "cuda_kernels.h"
#endif
#include <memory>

#include "quantization.h"

/**
 * @brief Enumeration of tensor names used in the TinyLlama model
 * 
 * This enum class defines the different types of tensors used in the transformer
 * architecture, including attention projections, feed-forward layers, and embeddings.
 */
enum class TensorName {
    Q_PROJ,      /**< Query projection matrix */
    K_PROJ,      /**< Key projection matrix */
    V_PROJ,      /**< Value projection matrix */
    O_PROJ,      /**< Output projection matrix */
    GATE_PROJ,   /**< Gate projection for SwiGLU activation */
    UP_PROJ,     /**< Upward projection in feed-forward network */
    DOWN_PROJ,   /**< Downward projection in feed-forward network */
    TOKEN_EMBD,  /**< Token embedding matrix */
    LM_HEAD,     /**< Language model head for final token prediction */
    UNKNOWN      /**< Unknown tensor type */
};

static std::string tensor_name_to_string(TensorName tn) {
  switch (tn) {
    case TensorName::Q_PROJ:
      return "Q_PROJ";
    case TensorName::K_PROJ:
      return "K_PROJ";
    case TensorName::V_PROJ:
      return "V_PROJ";
    case TensorName::O_PROJ:
      return "O_PROJ";
    case TensorName::GATE_PROJ:
      return "GATE_PROJ";
    case TensorName::UP_PROJ:
      return "UP_PROJ";
    case TensorName::DOWN_PROJ:
      return "DOWN_PROJ";
    case TensorName::TOKEN_EMBD:
      return "TOKEN_EMBD";
    case TensorName::LM_HEAD:
      return "LM_HEAD";
    default:
      return "UNKNOWN";
  }
}

/**
 * @brief Model configuration structure holding architecture and hyperparameters.
 *
 * Contains all key parameters needed to construct and run a transformer model, including
 * hidden size, number of layers, attention heads, vocabulary size, special token IDs, etc.
 */
struct ModelConfig {
    int hidden_size;              /**< Size of the hidden layers */
    int intermediate_size;        /**< Size of the intermediate (feed-forward) layers */
    int num_attention_heads;      /**< Number of attention heads */
    int num_key_value_heads;      /**< Number of key/value heads for grouped-query attention */
    int num_hidden_layers;        /**< Number of transformer layers */
    int vocab_size;              /**< Size of the vocabulary */
    int max_position_embeddings; /**< Maximum sequence length supported */
    float rms_norm_eps;         /**< Epsilon for RMSNorm operation */
    float rope_theta;           /**< Base for rotary position embeddings */
    std::string hidden_act;     /**< Activation function in hidden layers */
    std::string torch_dtype;    /**< Data type used in the original PyTorch model */
    int bos_token_id;          /**< Beginning of sequence token ID */
    int eos_token_id;          /**< End of sequence token ID */
    int unk_token_id = -1;     /**< Unknown token ID, default to -1 if not specified */
    int pad_token_id = -1;      /**< Padding token ID, default to -1 if not specified */
    std::string architecture;   /**< Model architecture identifier */
    std::string model_name;    /**< Name of the model */
    std::string chat_template_type; /**< Type of chat template used */
    std::string pre_tokenizer_type; /**< Type of pre-tokenizer */
    std::string chat_template_string; /**< Template string for chat formatting */
    bool is_gguf_file_loaded;   /**< Flag indicating if model was loaded from GGUF format */
    bool use_mmap_for_gguf = true; // Whether to use mmap for GGUF files, defaults to true
    bool use_kvcache_quantization = false; /**< Whether to use INT8 quantization for KVCache on GPU */
    int num_cpu_offload_layers = 0; /**< Number of layers to offload to CPU */
    
    // Memory management: Enable layer-wise weight eviction to prevent OOM
    bool enable_memory_efficient_layers = true; /**< Enable automatic layer weight eviction during forward pass */

    bool enable_prefill_chunking = true;
    bool use_optimized_cuda_kernels = true; // Re-enabled: fixed performance issues with simpler implementations

    enum class TokenizerFamily {
        UNKNOWN,
        LLAMA_SENTENCEPIECE, // For Llama 2 and similar SentencePiece BPE
        LLAMA3_TIKTOKEN      // For Llama 3's Tiktoken-based BPE
    };
    TokenizerFamily tokenizer_family = TokenizerFamily::UNKNOWN;
};

struct GGUFData;
struct ModelConfig;
ModelConfig parse_model_config_from_gguf(const GGUFData& gguf);

/**
 * @brief Key-Value cache for a single transformer layer
 * 
 * Stores the key and value tensors for attention mechanism, with optional
 * CUDA support for GPU acceleration.
 */
struct KVCacheLayer {
    std::vector<float> k;     // Key cache (CPU)
    std::vector<float> v;     // Value cache (CPU)
#ifdef HAS_CUDA
    float* k_dev_fp32 = nullptr;      // Original FP32 Key cache (GPU device pointer)
    float* v_dev_fp32 = nullptr;      // Original FP32 Value cache (GPU device pointer)

    int8_t* k_dev_quantized = nullptr; // Quantized INT8 Key cache (GPU device pointer)
    int8_t* v_dev_quantized = nullptr; // Quantized INT8 Value cache (GPU device pointer)
    float* k_dev_scales = nullptr;    // Scales for K cache (GPU device pointer)
    float* v_dev_scales = nullptr;    // Scales for V cache (GPU device pointer)
#endif
};

/**
 * @brief Complete Key-Value cache for all transformer layers
 * 
 * Manages the KV cache across all layers of the transformer model,
 * including memory management for both CPU and GPU implementations.
 * Supports both single-sequence and multi-sequence batch processing.
 */
struct KVCache {
    std::vector<KVCacheLayer> layers; /**< KV cache for each layer */
    
    // Single-sequence mode (legacy compatibility)
    int seq_len = 0;                  /**< Current sequence length (single-sequence mode) */
    
    // Multi-sequence mode (new batch functionality)
    std::vector<int> batch_seq_lens;  /**< Sequence lengths for each sequence in batch */
    int max_batch_size = 1;           /**< Maximum number of sequences that can be cached */
    int current_batch_size = 0;       /**< Current number of active sequences */
    
    int total_model_layers_ = 0;       /**< Total number of layers in the model */
    int max_seq_len_config_ = 0;       /**< Store the original max_seq_len */

    /**
     * @brief Initializes the KV cache with given dimensions
     * @param config The model configuration, used to determine if KVCache quantization is enabled.
     * @param total_num_model_layers Total number of layers in the model (for sizing CPU cache vectors)
     * @param num_gpu_layers_to_allocate Number of layers for which to allocate GPU device memory. Can be 0.
     * @param max_seq_len Maximum sequence length to cache
     * @param num_kv_heads Number of key/value heads
     * @param head_dim Dimension of each attention head
     * @param max_batch_size_arg Maximum number of sequences for batch processing (default: 1 for single-sequence)
     */
    void initialize(const ModelConfig& config,
                    int total_num_model_layers, int num_gpu_layers_to_allocate,
                    int max_seq_len_arg, int num_kv_heads, int head_dim,
                    int max_batch_size_arg = 1);

    void clear_data() {
        // Single-sequence mode (legacy compatibility)
        seq_len = 0;
        
        // Multi-sequence mode 
        current_batch_size = 0;
        batch_seq_lens.clear();
        
        // For batch processing, we MUST clear the actual KV data to prevent cross-sequence contamination
        for (auto& layer : layers) {
            std::fill(layer.k.begin(), layer.k.end(), 0.0f);
            std::fill(layer.v.begin(), layer.v.end(), 0.0f);
        }
        
        // Logger::debug("[KVCache] clear_data() called. seq_len reset to 0. K/V vectors cleared for batch processing.");
    }
    
    /**
     * @brief Initialize batch mode with specified number of sequences
     * @param batch_size Number of sequences to process in batch
     */
    void initialize_batch(int batch_size) {
        if (batch_size > max_batch_size) {
            Logger::warning("Requested batch size " + std::to_string(batch_size) + 
                           " exceeds max batch size " + std::to_string(max_batch_size) + 
                           ". Using max batch size.");
            batch_size = max_batch_size;
        }
        current_batch_size = batch_size;
        batch_seq_lens.resize(batch_size, 0);
    }

    void destroy_gpu_resources(); // Implementation moved to kv_cache.cpp

#ifdef HAS_CUDA
    int allocated_num_layers = 0;     /**< Number of GPU layers for which device memory was actually allocated */
    int allocated_max_seq_len = 0;    /**< Maximum sequence length allocated */
    int allocated_num_kv_heads = 0;   /**< Number of key/value heads allocated */
    int allocated_head_dim = 0;       /**< Dimension of each head allocated */

    ~KVCache() {
        destroy_gpu_resources();
    }
#else
    ~KVCache() {
        destroy_gpu_resources();
    }
#endif
};

using ForwardDiagCallback = std::function<void(
    int layer, const std::string& name, const std::vector<float>& v)>;

/**
 * @brief Structure holding all weights for a single transformer layer.
 *
 * Contains projections for attention and MLP, as well as normalization weights, in various formats.
 */
struct LayerWeights {
  std::vector<uint16_t> input_layernorm;
  std::vector<uint16_t> post_attention_layernorm;

  std::vector<uint16_t> q_proj;
  std::vector<uint16_t> k_proj;
  std::vector<uint16_t> v_proj;
  std::vector<uint16_t> o_proj;

  std::vector<uint16_t> gate_proj;
  std::vector<uint16_t> up_proj;
  std::vector<uint16_t> down_proj;

  std::vector<float> input_layernorm_f32;
  std::vector<float> post_attention_layernorm_f32;
  std::vector<float> q_proj_f32, k_proj_f32, v_proj_f32, o_proj_f32;
  std::vector<float> gate_proj_f32, up_proj_f32, down_proj_f32;
  std::vector<block_q4_K> q_proj_q4k, k_proj_q4k, v_proj_q4k, o_proj_q4k;
  std::vector<block_q4_K> gate_proj_q4k, up_proj_q4k, down_proj_q4k;
  std::vector<block_q6_K> q_proj_q6k, k_proj_q6k, v_proj_q6k, o_proj_q6k;
  std::vector<block_q6_K> gate_proj_q6k, up_proj_q6k, down_proj_q6k;
  std::vector<block_q8_0> q_proj_q8_0, k_proj_q8_0, v_proj_q8_0, o_proj_q8_0;
  std::vector<block_q8_0> gate_proj_q8_0, up_proj_q8_0, down_proj_q8_0;
  std::vector<block_q8_K> q_proj_q8k, k_proj_q8k, v_proj_q8k, o_proj_q8k;
  std::vector<block_q8_K> gate_proj_q8k, up_proj_q8k, down_proj_q8k;

#ifdef HAS_CUDA

  float* input_layernorm_dev = nullptr;
  float* post_attention_layernorm_dev = nullptr;
  
  // Individual layer device pointers for JIT weight loading
  float* q_proj_f32_dev = nullptr;
  float* k_proj_f32_dev = nullptr;
  float* v_proj_f32_dev = nullptr;
  float* o_proj_f32_dev = nullptr;
  float* gate_proj_f32_dev = nullptr;
  float* up_proj_f32_dev = nullptr;
  float* down_proj_f32_dev = nullptr;
#endif
};

/**
 * @brief Main transformer model class for TinyLlama.
 *
 * Handles weight loading, forward pass, and GPU/CPU offloading logic. Supports both GGUF and SafeTensors formats.
 */
class TinyLlamaModel {
 public:
  /**
   * @brief Construct a TinyLlamaModel from a SafeTensorsLoader.
   * @param config Model configuration.
   * @param loader SafeTensorsLoader instance.
   */
  TinyLlamaModel(const ModelConfig& config, const SafeTensorsLoader& loader);

  /**
   * @brief Construct a TinyLlamaModel from a model path (GGUF or SafeTensors).
   * @param initial_config Initial model configuration (may be overridden by file metadata).
   * @param model_path Path to the model file or directory.
   */
  TinyLlamaModel(const ModelConfig& initial_config, const std::string& model_path);

  /**
   * @brief Construct a TinyLlamaModel from pre-loaded GGUFData.
   * @param config_from_session Model configuration.
   * @param gguf_data_from_session Unique pointer to GGUFData.
   */
  TinyLlamaModel(const ModelConfig& config_from_session,
                 std::unique_ptr<GGUFData> gguf_data_from_session);

  /**
   * @brief Destructor. Cleans up all allocated resources.
   */
  ~TinyLlamaModel();

  /**
   * @brief Run the forward pass for the model on CPU layers.
   * @param input Input vector (modified in-place).
   * @param n_tokens Current token position.
   * @param kv_cache Pointer to the key-value cache.
   * @param attention_mask Optional attention mask.
   * @return Output logits or intermediate activations.
   */
  std::vector<float> forward(
      std::vector<float>& input,
      int n_tokens, KVCache* kv_cache,
      const std::vector<int>* attention_mask);

  void ensure_q_proj_dequantized(int layer_idx);
  void ensure_k_proj_dequantized(int layer_idx);
  void ensure_v_proj_dequantized(int layer_idx);
  void ensure_o_proj_dequantized(int layer_idx);
  void ensure_gate_proj_dequantized(int layer_idx);
  void ensure_up_proj_dequantized(int layer_idx);
  void ensure_down_proj_dequantized(int layer_idx);
  void ensure_lm_head_dequantized();
  void ensure_embed_tokens_dequantized();
  void ensure_f32_concatenated_weights_loaded();
  void ensure_layer_weights_on_gpu(int layer_idx);
  void free_layer_gpu_weights(int layer_idx);
  void clear_layer_dequantized_weights(int layer_idx);
  void initialize_gpu_and_rope();
#ifdef HAS_CUDA
  /**
   * @brief Performs forward pass on GPU for the layers designated to run on GPU.
   * @param x_input_dev Device pointer to the input activations (e.g., model_->x_dev_ prepared by caller).
   * @param pos Position in the sequence.
   * @param cache KV cache.
   * @param attention_mask Optional attention mask.
   * @param stream CUDA stream.
   * @return Output logits if this stage includes the final layer, otherwise intermediate activations (though current impl. always returns logits from this func).
   */
  std::vector<float> forward_device(
    float* x_input_dev,
    int pos, 
    KVCache* cache,
    const std::vector<int>* attention_mask = nullptr,
    cudaStream_t stream = 0);
  
  float* get_x_dev() { return x_dev_; }

  void forward_device(int token_id, int pos, KVCache* kv_cache,
                      cudaStream_t stream = 0);
  void forward_device_token(int token_id, int pos, KVCache* kv_cache, cudaStream_t stream = 0);

  std::vector<float> forward_device_batch_prefill(
      float* d_batch_input_hidden_states, // Device pointer to [num_tokens_in_batch, config_.hidden_size]
      int num_tokens_in_batch,
      int start_pos_in_kv_cache,         // Typically 0 for prefill
      KVCache* kv_cache,
      cudaStream_t stream
  );

  std::vector<std::vector<float>> forward_device_batch_generation(
      float* d_batch_input_hidden_states, // Device pointer to [num_tokens_in_batch, config_.hidden_size]
      const std::vector<int>& token_positions, // Position of each token in its respective sequence
      const std::vector<int>& original_sequence_indices, // Original sequence index for each token
      int num_tokens_in_batch,
      KVCache* kv_cache,
      cudaStream_t stream
  );

  // Memory management for layer-wise weight eviction

  // GPU workspace buffers
  
  // Persistent batch processing buffers to eliminate per-forward-pass allocations
  static constexpr int MAX_BATCH_TOKENS = 2048;  // Maximum tokens we can process in one batch
  
  // Persistent GPU buffers for batch processing (allocated once, reused)
  float* d_persistent_batch_input_ = nullptr;           // [MAX_BATCH_TOKENS, hidden_size]
  float* d_persistent_batch_norm_out_ = nullptr;        // [MAX_BATCH_TOKENS, hidden_size]
  float* d_persistent_batch_residual_ = nullptr;        // [MAX_BATCH_TOKENS, hidden_size]
  float* d_persistent_q_batch_ = nullptr;               // [MAX_BATCH_TOKENS, hidden_size]
  float* d_persistent_k_batch_ = nullptr;               // [MAX_BATCH_TOKENS, n_kv_heads * head_dim]
  float* d_persistent_v_batch_ = nullptr;               // [MAX_BATCH_TOKENS, n_kv_heads * head_dim]
  float* d_persistent_attn_output_ = nullptr;           // [MAX_BATCH_TOKENS, hidden_size]
  float* d_persistent_attn_proj_out_ = nullptr;         // [MAX_BATCH_TOKENS, hidden_size]
  float* d_persistent_gate_proj_out_ = nullptr;         // [MAX_BATCH_TOKENS, intermediate_size]
  float* d_persistent_up_proj_out_ = nullptr;           // [MAX_BATCH_TOKENS, intermediate_size]
  float* d_persistent_swiglu_out_ = nullptr;            // [MAX_BATCH_TOKENS, intermediate_size]
  float* d_persistent_mlp_down_out_ = nullptr;          // [MAX_BATCH_TOKENS, hidden_size]
  
  // Buffer management functions
  void allocate_persistent_batch_buffers();
  void free_persistent_batch_buffers();
  void resize_persistent_batch_buffers_if_needed(int required_batch_size);

#endif // HAS_CUDA

  const ModelConfig& get_config() const { return config_; }

  const std::vector<uint16_t>& get_lm_head() const { return lm_head; }

  const std::vector<uint16_t>& get_embed_tokens() const { return embed_tokens; }

  std::vector<LayerWeights>& get_layers() { return layers; }

  /**
   * @brief Lookup the embedding vector for a given token ID.
   * @param token_id The token ID to lookup.
   * @return The embedding vector as a std::vector<float>.
   */
  std::vector<float> lookup_embedding(int token_id);

  /**
   * @brief Get the vocabulary size for the model.
   * @return Vocabulary size.
   */
  int get_vocab_size() const;

  const GGUFData* get_gguf_data() const {
    return gguf_data_ ? gguf_data_.get() : nullptr;
  }

  GGUFData* get_gguf_data_ptr() { return gguf_data_.get(); }

  void initialize_rope_freqs();

  std::vector<float> forward_cpu_batch(
      const std::vector<float>& batch_input_activations,
      int num_tokens_in_batch,
      int num_cpu_layers_to_process,
      int start_pos_in_sequence,
      KVCache* kv_cache,
      const std::vector<int>& prompt_lengths = {}
  );

  std::vector<float> forward_cpu_logits_batch(
      const std::vector<float>& final_batch_activations,
      int num_tokens_in_batch
  );

  std::vector<std::vector<float>> forward_cpu_batch_generation(
      const std::vector<float>& batch_input_activations,
      const std::vector<int>& token_positions,
      const std::vector<int>& original_sequence_indices,
      int num_tokens_in_batch,
      KVCache* kv_cache
  );

  friend void map_gguf_weights(const GGUFData& gguf, TinyLlamaModel& model);
  friend class CPUBatchProcessor;

 private:
  ModelConfig config_;

  std::vector<uint16_t> embed_tokens;
  std::vector<uint16_t> lm_head;
  std::vector<uint16_t> final_norm;
  std::vector<float> embed_tokens_f32, lm_head_f32, final_norm_f32;
  std::vector<block_q4_K> embed_tokens_q4k, lm_head_q4k, final_norm_q4k;
  std::vector<block_q6_K> embed_tokens_q6k, lm_head_q6k, final_norm_q6k;
  std::vector<block_q8_0> embed_tokens_q8_0, lm_head_q8_0;
  std::vector<block_q8_K> embed_tokens_q8k, lm_head_q8k;
  std::vector<LayerWeights> layers;

#ifdef HAS_CUDA
  float* final_norm_dev = nullptr;
  float* all_freqs_cis_dev = nullptr;
  uint16_t* token_embedding_table_dev_ = nullptr;
  uint16_t* w_q_dev_ = nullptr;
  uint16_t* w_k_dev_ = nullptr;
  uint16_t* w_v_dev_ = nullptr;
  uint16_t* w_o_dev_ = nullptr;
  uint16_t* w_gate_dev_ = nullptr;
  uint16_t* w_up_dev_ = nullptr;
  uint16_t* w_down_dev_ = nullptr;
  uint16_t* lm_head_dev_ = nullptr;
  float* token_embedding_table_f32_dev_ = nullptr;
  float* w_q_f32_dev_ = nullptr;
  float* w_k_f32_dev_ = nullptr;
  float* w_v_f32_dev_ = nullptr;
  float* w_o_f32_dev_ = nullptr;
  float* w_gate_f32_dev_ = nullptr;
  float* w_up_f32_dev_ = nullptr;
  float* w_down_f32_dev_ = nullptr;
  float* lm_head_f32_dev_ = nullptr;
  cublasHandle_t cublas_handle_ = nullptr;

  float* x_dev_ = nullptr;
  float* x_norm_dev_ = nullptr;
  float* x_resid1_dev_ = nullptr;
  float* x_resid2_dev_ = nullptr;
  float* q_dev_ = nullptr;
  float* k_dev_ = nullptr;
  float* v_dev_ = nullptr;
  float* attn_out_dev_ = nullptr;
  float* attn_proj_dev_ = nullptr;
  float* gate_vec_dev_ = nullptr;
  float* up_vec_dev_ = nullptr;
  float* swiglu_vec_dev_ = nullptr;
  float* mlp_down_dev_ = nullptr;
  float* logits_dev_ = nullptr;

  // Temporary buffers for KVCache dequantization
  float* dequant_k_cache_buffer_dev_ = nullptr;  // For KVCache dequantization (full cache size)
  float* dequant_v_cache_buffer_dev_ = nullptr;  // For KVCache dequantization (full cache size)
  
  // Selective KVCache dequantization buffers (much smaller - only per head per token)
  float* selective_k_dequant_buffer_dev_ = nullptr;  // Small buffer for selective K dequantization
  float* selective_v_dequant_buffer_dev_ = nullptr;  // Small buffer for selective V dequantization
  size_t selective_dequant_buffer_size_ = 0;         // Size of selective buffers in elements

  // GPU workspace buffers
#endif

  std::vector<std::pair<float, float>> precomputed_freqs_cis_;

  std::unique_ptr<GGUFData> gguf_data_;
  std::string model_path_;
  bool f32_concatenated_weights_loaded_ = false;

  std::unique_ptr<class CPUBatchProcessor> cpu_batch_processor_;

  void initialize_weights(const SafeTensorsLoader* loader,
                          const GGUFData* gguf);

};

ModelConfig parse_model_config(const nlohmann::json& json);

int argmax(const std::vector<float>& v);

float bfloat16_to_float32(uint16_t b16);

void rmsnorm(const std::vector<float>& x, const std::vector<uint16_t>& weight,
             float eps, std::vector<float>& out);

void matvec_bf16_f32(const std::vector<uint16_t>& mat,
                     const std::vector<float>& vec, std::vector<float>& out,
                     int M, int N);

void softmax(std::vector<float>& x);

struct KVCache;

float bfloat16_to_float32(uint16_t b16);
std::vector<uint16_t> uint8_vector_to_uint16_vector(
    const std::vector<uint8_t>& bytes, size_t numel);

void log_vector_summary(const std::string& name, const std::vector<float>& v,
                        int head_count = 5);

void log_vector_summary_batch(const std::string& name, const std::vector<float>& batch_vector,
                              int num_tokens_in_batch, int single_token_vector_size,
                              int head_count = 5);

#endif