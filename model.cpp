#include "model.h"
#include "logger.h"
#include <stdexcept>
#include <cstring>

// Helper: convert bfloat16 bytes to float32 vector
std::vector<float> TinyLlamaModel::bfloat16_to_float32(const std::vector<uint8_t>& b16, size_t numel) {
    std::vector<float> out(numel);
    for (size_t i = 0; i < numel; ++i) {
        uint16_t b = (b16[2 * i + 1] << 8) | b16[2 * i];
        uint32_t f = ((uint32_t)b) << 16;
        std::memcpy(&out[i], &f, sizeof(float));
    }
    return out;
}

// Parse ModelConfig from nlohmann::json
ModelConfig parse_model_config(const nlohmann::json& json) {
    ModelConfig cfg;
    cfg.hidden_size = json.value("hidden_size", 0);
    cfg.intermediate_size = json.value("intermediate_size", 0);
    cfg.num_attention_heads = json.value("num_attention_heads", 0);
    cfg.num_key_value_heads = json.value("num_key_value_heads", 0);
    cfg.num_hidden_layers = json.value("num_hidden_layers", 0);
    cfg.vocab_size = json.value("vocab_size", 0);
    cfg.max_position_embeddings = json.value("max_position_embeddings", 0);
    cfg.rms_norm_eps = json.value("rms_norm_eps", 1e-5f);
    cfg.rope_theta = json.value("rope_theta", 10000.0f);
    cfg.hidden_act = json.value("hidden_act", "silu");
    cfg.torch_dtype = json.value("torch_dtype", "bfloat16");
    cfg.bos_token_id = json.value("bos_token_id", 1);
    cfg.eos_token_id = json.value("eos_token_id", 2);
    return cfg;
}

// TinyLlamaModel constructor: load all weights from safetensors
TinyLlamaModel::TinyLlamaModel(const ModelConfig& config, const SafeTensorsLoader& loader)
    : config_(config)
{
    int hs = config.hidden_size;
    int is = config.intermediate_size;
    int nhl = config.num_hidden_layers;
    int vs = config.vocab_size;

    // Embedding
    try {
        auto bytes = loader.get_tensor_bytes("model.embed_tokens.weight");
        embed_tokens = bfloat16_to_float32(bytes, vs * hs);
    } catch (const std::exception& e) {
        Logger::error("Missing model.embed_tokens.weight: " + std::string(e.what()));
    }
    // LM head
    try {
        auto bytes = loader.get_tensor_bytes("lm_head.weight");
        lm_head = bfloat16_to_float32(bytes, vs * hs);
    } catch (const std::exception& e) {
        Logger::error("Missing lm_head.weight: " + std::string(e.what()));
    }
    // Final norm
    try {
        auto bytes = loader.get_tensor_bytes("model.norm.weight");
        final_norm = bfloat16_to_float32(bytes, hs);
    } catch (const std::exception& e) {
        Logger::error("Missing model.norm.weight: " + std::string(e.what()));
    }

    // Per-layer weights
    layers.resize(nhl);
    for (int i = 0; i < nhl; ++i) {
        auto& lw = layers[i];
        std::string prefix = "model.layers." + std::to_string(i) + ".";
        try {
            lw.input_layernorm = bfloat16_to_float32(loader.get_tensor_bytes(prefix + "input_layernorm.weight"), hs);
            lw.post_attention_layernorm = bfloat16_to_float32(loader.get_tensor_bytes(prefix + "post_attention_layernorm.weight"), hs);
            lw.q_proj = bfloat16_to_float32(loader.get_tensor_bytes(prefix + "self_attn.q_proj.weight"), hs * hs);
            lw.k_proj = bfloat16_to_float32(loader.get_tensor_bytes(prefix + "self_attn.k_proj.weight"), hs * (hs / config.num_attention_heads * config.num_key_value_heads));
            lw.v_proj = bfloat16_to_float32(loader.get_tensor_bytes(prefix + "self_attn.v_proj.weight"), hs * (hs / config.num_attention_heads * config.num_key_value_heads));
            lw.o_proj = bfloat16_to_float32(loader.get_tensor_bytes(prefix + "self_attn.o_proj.weight"), hs * hs);
            lw.gate_proj = bfloat16_to_float32(loader.get_tensor_bytes(prefix + "mlp.gate_proj.weight"), is * hs);
            lw.up_proj = bfloat16_to_float32(loader.get_tensor_bytes(prefix + "mlp.up_proj.weight"), is * hs);
            lw.down_proj = bfloat16_to_float32(loader.get_tensor_bytes(prefix + "mlp.down_proj.weight"), hs * is);
        } catch (const std::exception& e) {
            Logger::error("Missing or malformed weights in layer " + std::to_string(i) + ": " + e.what());
        }
    }
    Logger::info("All model weights loaded.");
}

// Forward pass stub: returns empty logits
std::vector<float> TinyLlamaModel::forward(int input_id /*, kv_cache */) {
    // TODO: Implement full forward pass
    return {};
} 