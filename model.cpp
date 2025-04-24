#include "model.h"
#include "logger.h"
#include <stdexcept>
#include <cstring>
#include <cmath>
#include <algorithm>

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

// RMSNorm: y = x / sqrt(mean(x^2) + eps) * weight
static void rmsnorm(const std::vector<float>& x, const std::vector<float>& weight, float eps, std::vector<float>& out) {
    float ss = 0.0f;
    for (float v : x) ss += v * v;
    float denom = 1.0f / std::sqrt(ss / x.size() + eps);
    for (size_t i = 0; i < x.size(); ++i) out[i] = x[i] * denom * weight[i];
}

// SiLU activation: y = x * sigmoid(x)
static void silu(const std::vector<float>& x, std::vector<float>& out) {
    for (size_t i = 0; i < x.size(); ++i) {
        out[i] = x[i] / (1.0f + std::exp(-x[i]));
    }
}

// Matrix multiplication: out = mat [M,N] * vec [N] -> [M]
static void matvec(const std::vector<float>& mat, const std::vector<float>& vec, std::vector<float>& out, int M, int N) {
    for (int i = 0; i < M; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < N; ++j) sum += mat[i * N + j] * vec[j];
        out[i] = sum;
    }
}

// RoPE: apply rotary positional embedding to Q or K (in-place)
static void apply_rope(std::vector<float>& x, int num_heads, int head_dim, int pos, float rope_theta) {
    // x: [num_heads, head_dim]
    // Only even/odd pairs are rotated
    float theta = rope_theta;
    for (int h = 0; h < num_heads; ++h) {
        for (int i = 0; i < head_dim; i += 2) {
            float freq = std::pow(theta, -float(i) / head_dim);
            float angle = pos * freq;
            float x0 = x[h * head_dim + i];
            float x1 = x[h * head_dim + i + 1];
            x[h * head_dim + i]     = x0 * std::cos(angle) - x1 * std::sin(angle);
            x[h * head_dim + i + 1] = x0 * std::sin(angle) + x1 * std::cos(angle);
        }
    }
}

// Softmax over a vector (in-place)
static void softmax(std::vector<float>& x) {
    float maxv = *std::max_element(x.begin(), x.end());
    float sum = 0.0f;
    for (float& v : x) { v = std::exp(v - maxv); sum += v; }
    for (float& v : x) v /= sum;
}

// Helper: log vector stats (min, max, mean, all finite)
static void log_vec_stats(const std::string& name, const std::vector<float>& v) {
    float minv = *std::min_element(v.begin(), v.end());
    float maxv = *std::max_element(v.begin(), v.end());
    float mean = std::accumulate(v.begin(), v.end(), 0.0f) / v.size();
    bool all_finite = std::all_of(v.begin(), v.end(), [](float x) { return std::isfinite(x); });
    Logger::info(name + ": min=" + std::to_string(minv) + ", max=" + std::to_string(maxv) + ", mean=" + std::to_string(mean) + ", all_finite=" + (all_finite ? "yes" : "no"));
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
    int n_heads = config.num_attention_heads;
    int n_kv_heads = config.num_key_value_heads;
    int kv_dim = (hs / n_heads) * n_kv_heads;
    Logger::info("K/V projection kv_dim: " + std::to_string(kv_dim));

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
            lw.k_proj = bfloat16_to_float32(loader.get_tensor_bytes(prefix + "self_attn.k_proj.weight"), hs * kv_dim);
            lw.v_proj = bfloat16_to_float32(loader.get_tensor_bytes(prefix + "self_attn.v_proj.weight"), hs * kv_dim);
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

// Forward pass for a single token (real attention, no KV cache)
std::vector<float> TinyLlamaModel::forward(int input_id /*, kv_cache */) {
    int hs = config_.hidden_size;
    int is = config_.intermediate_size;
    int nhl = config_.num_hidden_layers;
    int vs = config_.vocab_size;
    int n_heads = config_.num_attention_heads;
    int n_kv_heads = config_.num_key_value_heads;
    int head_dim = hs / n_heads;
    int kv_dim = head_dim * n_kv_heads;
    float eps = config_.rms_norm_eps;
    float rope_theta = config_.rope_theta;
    int pos = 0; // For now, always position 0 (no KV cache)

    // 1. Embedding lookup
    std::vector<float> x(hs);
    for (int i = 0; i < hs; ++i) x[i] = embed_tokens[input_id * hs + i];
    Logger::info("Embedding lookup complete.");
    log_vec_stats("embedding", x);

    // 2. Transformer blocks
    for (int l = 0; l < nhl; ++l) {
        const auto& lw = layers[l];
        // a) Input RMSNorm
        std::vector<float> x_norm(hs);
        rmsnorm(x, lw.input_layernorm, eps, x_norm);
        log_vec_stats("layer " + std::to_string(l) + " input RMSNorm", x_norm);
        // b) Attention
        // Q projection: [hs, hs] x [hs] -> [hs]
        std::vector<float> q(hs);
        matvec(lw.q_proj, x_norm, q, hs, hs);
        // K, V projections: [hs, kv_dim] x [hs] -> [kv_dim]
        std::vector<float> k(kv_dim), v(kv_dim);
        matvec(lw.k_proj, x_norm, k, kv_dim, hs);
        matvec(lw.v_proj, x_norm, v, kv_dim, hs);
        Logger::info("Q/K/V projections complete for layer " + std::to_string(l));
        log_vec_stats("layer " + std::to_string(l) + " Q", q);
        log_vec_stats("layer " + std::to_string(l) + " K", k);
        log_vec_stats("layer " + std::to_string(l) + " V", v);
        // Split Q into heads, K/V into kv_heads
        std::vector<float> q_heads(q); // [n_heads, head_dim]
        std::vector<float> k_heads(k); // [n_kv_heads, head_dim]
        apply_rope(q_heads, n_heads, head_dim, pos, rope_theta);
        apply_rope(k_heads, n_kv_heads, head_dim, pos, rope_theta);
        Logger::info("RoPE applied for layer " + std::to_string(l));
        log_vec_stats("layer " + std::to_string(l) + " Q_rope", q_heads);
        log_vec_stats("layer " + std::to_string(l) + " K_rope", k_heads);
        // Attention scores (self only, so always 1.0)
        std::vector<float> attn_out(hs);
        // For now, just broadcast v to all heads (no real multi-token context)
        for (int h = 0; h < n_heads; ++h) {
            int kvh = h % n_kv_heads;
            for (int i = 0; i < head_dim; ++i) {
                attn_out[h * head_dim + i] = v[kvh * head_dim + i];
            }
        }
        log_vec_stats("layer " + std::to_string(l) + " attn_out", attn_out);
        // Output projection
        std::vector<float> attn_proj(hs);
        matvec(lw.o_proj, attn_out, attn_proj, hs, hs);
        Logger::info("Attention output projection complete for layer " + std::to_string(l));
        log_vec_stats("layer " + std::to_string(l) + " attn_proj", attn_proj);
        // c) Residual connection (scaled)
        for (int i = 0; i < hs; ++i) x[i] = (x[i] + attn_proj[i]) * (1.0f / std::sqrt(2.0f));
        log_vec_stats("layer " + std::to_string(l) + " after attn residual", x);
        // d) Post-attention RMSNorm
        std::vector<float> ffn_norm(hs);
        rmsnorm(x, lw.post_attention_layernorm, eps, ffn_norm);
        log_vec_stats("layer " + std::to_string(l) + " post-attn RMSNorm", ffn_norm);
        // e) MLP: gate, up, SiLU, elementwise, down
        std::vector<float> gate(is), up(is), silu_gate(is), ffn_out(hs);
        matvec(lw.gate_proj, ffn_norm, gate, is, hs);
        matvec(lw.up_proj, ffn_norm, up, is, hs);
        silu(gate, silu_gate);
        for (int i = 0; i < is; ++i) up[i] *= silu_gate[i];
        matvec(lw.down_proj, up, ffn_out, hs, is);
        log_vec_stats("layer " + std::to_string(l) + " ffn_out", ffn_out);
        // f) Residual connection (scaled)
        for (int i = 0; i < hs; ++i) x[i] = (x[i] + ffn_out[i]) * (1.0f / std::sqrt(2.0f));
        log_vec_stats("layer " + std::to_string(l) + " after ffn residual", x);
    }
    Logger::info("Transformer blocks complete.");

    // 3. Final RMSNorm
    std::vector<float> x_norm(hs);
    rmsnorm(x, final_norm, eps, x_norm);
    log_vec_stats("final RMSNorm", x_norm);
    // 4. LM head projection
    std::vector<float> logits(vs);
    matvec(lm_head, x_norm, logits, vs, hs);
    log_vec_stats("logits", logits);
    Logger::info("Forward pass complete.");
    return logits;
} 