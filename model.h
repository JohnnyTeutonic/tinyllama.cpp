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
#include <cublas_v2.h>
#include <cuda_runtime.h>

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
 * @brief Configuration parameters for the TinyLlama model
 * 
 * This structure holds all the hyperparameters and configuration settings
 * needed to initialize and run the TinyLlama model.
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
    std::string architecture;   /**< Model architecture identifier */
    std::string model_name;    /**< Name of the model */
    std::string chat_template_type; /**< Type of chat template used */
    std::string pre_tokenizer_type; /**< Type of pre-tokenizer */
    std::string chat_template_string; /**< Template string for chat formatting */
    bool is_gguf_file_loaded;   /**< Flag indicating if model was loaded from GGUF format */
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
    std::vector<float> k;     /**< Key cache */
    std::vector<float> v;     /**< Value cache */

#ifdef HAS_CUDA
    float* k_dev = nullptr;  /**< Device pointer for key cache */
    float* v_dev = nullptr;  /**< Device pointer for value cache */
#endif
};

/**
 * @brief Complete Key-Value cache for all transformer layers
 * 
 * Manages the KV cache across all layers of the transformer model,
 * including memory management for both CPU and GPU implementations.
 */
struct KVCache {
    std::vector<KVCacheLayer> layers; /**< KV cache for each layer */
    int seq_len = 0;                  /**< Current sequence length */

#ifdef HAS_CUDA
    int allocated_num_layers = 0;     /**< Number of layers allocated on GPU */
    int allocated_max_seq_len = 0;    /**< Maximum sequence length allocated */
    int allocated_num_kv_heads = 0;   /**< Number of key/value heads allocated */
    int allocated_head_dim = 0;       /**< Dimension of each head allocated */

    ~KVCache() {
      if (allocated_num_layers > 0) {
        Logger::info("Freeing KVCache CUDA memory...");
        for (int l = 0; l < allocated_num_layers; ++l) {
          if (layers[l].k_dev) {
            gpuErrchk(cudaFree(layers[l].k_dev));
            layers[l].k_dev = nullptr;
          }
          if (layers[l].v_dev) {
            gpuErrchk(cudaFree(layers[l].v_dev));
            layers[l].v_dev = nullptr;
          }
        }
        Logger::info("KVCache CUDA memory freed.");
      }
    }
#endif

    /**
     * @brief Initializes the KV cache with given dimensions
     * @param num_layers Number of transformer layers
     * @param max_seq_len Maximum sequence length to cache
     * @param num_kv_heads Number of key/value heads
     * @param head_dim Dimension of each attention head
     */
    void initialize(int num_layers, int max_seq_len, int num_kv_heads, int head_dim);
};

using ForwardDiagCallback = std::function<void(
    int layer, const std::string& name, const std::vector<float>& v)>;

/**
 * @brief Weights for a single transformer layer
 * 
 * Contains all the weight matrices and normalization parameters for
 * a transformer layer, supporting multiple quantization formats.
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

#ifdef HAS_CUDA

  float* input_layernorm_dev = nullptr;
  float* post_attention_layernorm_dev = nullptr;
#endif
};

/**
 * @brief Main TinyLlama model class
 * 
 * Implements the TinyLlama language model, providing methods for
 * initialization, inference, and weight management.
 */
class TinyLlamaModel {
 public:
  /**
   * @brief Constructs a TinyLlama model from SafeTensors format
   * @param config Model configuration
   * @param loader SafeTensors loader containing weights
   */
  TinyLlamaModel(const ModelConfig& config, const SafeTensorsLoader& loader);
  TinyLlamaModel(const ModelConfig& config, const std::string& weights_path);
  ~TinyLlamaModel();

  /**
   * @brief Performs forward pass on CPU
   * @param x_vec Input tensor
   * @param pos Position in the sequence
   * @param cache Optional KV cache
   * @param attention_mask Optional attention mask
   * @return Output logits
   */
  std::vector<float> forward(std::vector<float>& x_vec, int pos,
                             KVCache* cache = nullptr,
                             const std::vector<int>* attention_mask = nullptr);

#ifdef HAS_CUDA
  /**
   * @brief Performs forward pass on GPU
   * @param token_id Input token ID
   * @param pos Position in the sequence
   * @param cache KV cache
   * @param attention_mask Optional attention mask
   * @param stream CUDA stream
   * @return Output logits
   */
  std::vector<float> forward_device(
      int token_id, int pos, KVCache* cache,
      const std::vector<int>* attention_mask = nullptr,
      cudaStream_t stream = 0);
#endif

  const ModelConfig& get_config() const { return config_; }

  const std::vector<uint16_t>& get_lm_head() const { return lm_head; }

  const std::vector<uint16_t>& get_embed_tokens() const { return embed_tokens; }

  std::vector<LayerWeights>& get_layers() { return layers; }

  std::vector<float> lookup_embedding(int token_id);

  int get_vocab_size() const;

  const GGUFData* get_gguf_data() const {
    return gguf_data_ ? gguf_data_.get() : nullptr;
  }

  friend void map_gguf_weights(const GGUFData& gguf, TinyLlamaModel& model);

  void initialize_rope_freqs();

 private:
  ModelConfig config_;

  std::vector<uint16_t> embed_tokens;
  std::vector<uint16_t> lm_head;
  std::vector<uint16_t> final_norm;
  std::vector<float> embed_tokens_f32, lm_head_f32, final_norm_f32;
  std::vector<block_q4_K> embed_tokens_q4k, lm_head_q4k, final_norm_q4k;
  std::vector<block_q6_K> embed_tokens_q6k, lm_head_q6k, final_norm_q6k;
  std::vector<block_q8_0> embed_tokens_q8_0, lm_head_q8_0;
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
#endif

  std::vector<std::pair<float, float>> precomputed_freqs_cis_;

  std::unique_ptr<GGUFData> gguf_data_;
  std::string model_path_;

  void initialize_weights(const SafeTensorsLoader* loader,
                          const GGUFData* gguf);
  void initialize_gpu_and_rope();

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

#endif