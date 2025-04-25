#ifndef MODEL_H
#define MODEL_H

#include <string>
#include <vector>
#include <unordered_map>
#include <nlohmann/json.hpp>
#include "safetensors_loader.h"
#include <functional>
#include <cstdint> // Include for uint16_t

struct ModelConfig {
    int hidden_size;
    int intermediate_size;
    int num_attention_heads;
    int num_key_value_heads;
    int num_hidden_layers;
    int vocab_size;
    int max_position_embeddings;
    float rms_norm_eps;
    float rope_theta;
    std::string hidden_act;
    std::string torch_dtype;
    int bos_token_id;
    int eos_token_id;
};

// KVCache for autoregressive inference
struct KVCache {
    struct LayerKV {
        std::vector<float> k; // [num_kv_heads, seq_len, head_dim] (flattened)
        std::vector<float> v; // [num_kv_heads, seq_len, head_dim] (flattened)
    };
    std::vector<LayerKV> layers; // num_hidden_layers
    int seq_len = 0;
};

// Forward diagnostic callback type
using ForwardDiagCallback = std::function<void(int layer, const std::string& name, const std::vector<float>& v)>;

// LayerWeights struct for transformer layers
struct LayerWeights {
    std::vector<uint16_t> input_layernorm;         // [hidden_size]
    std::vector<uint16_t> post_attention_layernorm;// [hidden_size]
    // Attention
    std::vector<uint16_t> q_proj; // [hidden_size, hidden_size]
    std::vector<uint16_t> k_proj; // [hidden_size, kv_dim]
    std::vector<uint16_t> v_proj; // [hidden_size, kv_dim]
    std::vector<uint16_t> o_proj; // [hidden_size, hidden_size]
    // MLP
    std::vector<uint16_t> gate_proj; // [intermediate_size, hidden_size]
    std::vector<uint16_t> up_proj;   // [intermediate_size, hidden_size]
    std::vector<uint16_t> down_proj; // [hidden_size, intermediate_size]
};

class TinyLlamaModel {
public:
    // Construct from config and safetensors loader
    TinyLlamaModel(const ModelConfig& config, const SafeTensorsLoader& loader);

    // Forward pass: input_id, token_idx, kv_cache, returns logits
    // Optionally accepts a diagnostic callback for per-layer logging
    std::vector<float> forward(int input_id, int token_idx, KVCache* cache, ForwardDiagCallback diag_cb = nullptr);

    // Get model config
    const ModelConfig& get_config() const { return config_; }

    // Getter for lm_head weights
    const std::vector<uint16_t>& get_lm_head() const { return lm_head; }

    // Getter for embed_tokens weights
    const std::vector<uint16_t>& get_embed_tokens() const { return embed_tokens; }

    // Getter for layers (for debugging)
    const std::vector<LayerWeights>& get_layers() const { return layers; }

private:
    ModelConfig config_;

    // Weight storage (changed to uint16_t for bfloat16)
    std::vector<uint16_t> embed_tokens; // [vocab_size, hidden_size]
    std::vector<uint16_t> lm_head;      // [vocab_size, hidden_size]
    std::vector<uint16_t> final_norm;   // [hidden_size]

    std::vector<LayerWeights> layers; // num_hidden_layers
};

// Utility: parse ModelConfig from nlohmann::json
ModelConfig parse_model_config(const nlohmann::json& json);

// Helper function declaration
int argmax(const std::vector<float>& v);

// Convert a single bfloat16 (uint16_t) to float32
float bfloat16_to_float32(uint16_t b16);

// rmsnorm declaration
void rmsnorm(const std::vector<float>& x, const std::vector<uint16_t>& weight, float eps, std::vector<float>& out);

// matvec_bf16_f32 declaration
void matvec_bf16_f32(const std::vector<uint16_t>& mat, const std::vector<float>& vec, std::vector<float>& out, int M, int N);

// apply_rope declaration
void apply_rope(std::vector<float>& x, int num_heads, int head_dim, int pos, float rope_theta);

// softmax declaration
void softmax(std::vector<float>& x);

#endif // MODEL_H 