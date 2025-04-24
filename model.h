#ifndef MODEL_H
#define MODEL_H

#include <string>
#include <vector>
#include <unordered_map>
#include <nlohmann/json.hpp>
#include "safetensors_loader.h"

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

class TinyLlamaModel {
public:
    // Construct from config and safetensors loader
    TinyLlamaModel(const ModelConfig& config, const SafeTensorsLoader& loader);

    // Forward pass: input_ids (single token for now), kv_cache (to be defined), returns logits
    std::vector<float> forward(int input_id /*, kv_cache */);

    // Get model config
    const ModelConfig& get_config() const { return config_; }

private:
    ModelConfig config_;

    // Weight storage
    std::vector<float> embed_tokens; // [vocab_size, hidden_size]
    std::vector<float> lm_head;      // [vocab_size, hidden_size]
    std::vector<float> final_norm;   // [hidden_size]

    struct LayerWeights {
        std::vector<float> input_layernorm;         // [hidden_size]
        std::vector<float> post_attention_layernorm;// [hidden_size]
        // Attention
        std::vector<float> q_proj; // [hidden_size, hidden_size]
        std::vector<float> k_proj; // [hidden_size, hidden_size/num_attention_heads*num_key_value_heads]
        std::vector<float> v_proj; // [hidden_size, hidden_size/num_attention_heads*num_key_value_heads]
        std::vector<float> o_proj; // [hidden_size, hidden_size]
        // MLP
        std::vector<float> gate_proj; // [intermediate_size, hidden_size]
        std::vector<float> up_proj;   // [intermediate_size, hidden_size]
        std::vector<float> down_proj; // [hidden_size, intermediate_size]
    };
    std::vector<LayerWeights> layers; // num_hidden_layers

    // Helper: convert bfloat16 bytes to float32 vector
    static std::vector<float> bfloat16_to_float32(const std::vector<uint8_t>& b16, size_t numel);
};

// Utility: parse ModelConfig from nlohmann::json
ModelConfig parse_model_config(const nlohmann::json& json);

#endif // MODEL_H 