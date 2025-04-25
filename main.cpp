#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <memory>
#include <stdexcept>
#include <nlohmann/json.hpp>
#include <sentencepiece_processor.h>
#include <map>
#include "safetensors_loader.h"
#include "tokenizer.h"
#include "logger.h"
#include "prompt.h"
#include "model.h"
#include <limits>
#include <random>
#include <functional>
#include <cstdio> // For std::remove
#include <sstream>
#include <numeric>

// TODO: Implement safetensors loader
// TODO: Implement TinyLlama model and inference

// Utility: Read file into string
std::string read_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Failed to open file: " + path);
    return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

// Sampling function: top-k, top-p, temperature
int sample_top_k_top_p_temperature(const std::vector<float>& logits, float temperature, int top_k, float top_p, std::mt19937& rng) {
    // 1. Apply temperature
    std::vector<float> scaled_logits(logits.size());
    float max_logit = *std::max_element(logits.begin(), logits.end());
    for (size_t i = 0; i < logits.size(); ++i) {
        scaled_logits[i] = (logits[i] - max_logit) / temperature;
    }
    // 2. Compute softmax probabilities
    std::vector<float> probs(logits.size());
    float sum = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::exp(scaled_logits[i]);
        sum += probs[i];
    }
    for (float& p : probs) p /= sum;
    // 3. Top-k filter
    std::vector<int> indices(logits.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + top_k, indices.end(),
        [&](int a, int b) { return probs[a] > probs[b]; });
    float topk_sum = 0.0f;
    for (int i = 0; i < top_k; ++i) topk_sum += probs[indices[i]];
    // 4. Top-p filter
    std::vector<std::pair<float, int>> prob_idx;
    for (int i = 0; i < top_k; ++i) prob_idx.emplace_back(probs[indices[i]], indices[i]);
    std::sort(prob_idx.begin(), prob_idx.end(), std::greater<>());
    float cumulative = 0.0f;
    std::vector<std::pair<float, int>> filtered;
    for (const auto& pi : prob_idx) {
        cumulative += pi.first;
        filtered.push_back(pi);
        if (cumulative >= top_p) break;
    }
    // 5. Normalize filtered probabilities
    float filtered_sum = 0.0f;
    for (const auto& pi : filtered) filtered_sum += pi.first;
    std::vector<float> norm_probs(filtered.size());
    for (size_t i = 0; i < filtered.size(); ++i) norm_probs[i] = filtered[i].first / filtered_sum;
    // 6. Sample
    std::discrete_distribution<int> dist(norm_probs.begin(), norm_probs.end());
    int idx = dist(rng);
    return filtered[idx].second;
}

// Forward diagnostic callback type
using ForwardDiagCallback = std::function<void(int layer, const std::string& name, const std::vector<float>& v)>;

int main(int argc, char** argv) {
    // Accept data directory as argument (default: "data")
    std::string data_dir = "data";
    if (argc > 1) {
        data_dir = argv[1];
    }
    std::string config_path = data_dir + "/config.json";
    std::string tokenizer_path = data_dir + "/tokenizer.model";
    std::string safetensors_path = data_dir + "/model.safetensors";
    std::string tokenizer_config_path = data_dir + "/tokenizer_config.json"; // Path to tokenizer config

    Logger::info("Using data directory: " + data_dir);
    Logger::info("Loading config: " + config_path);

    // 1. Load config.json
    nlohmann::json config;
    try {
        std::string config_str = read_file(config_path);
        config = nlohmann::json::parse(config_str);
    } catch (const std::exception& e) {
        Logger::error(std::string("Error loading config.json: ") + e.what());
        return 1;
    }

    // Load tokenizer_config.json for special token strings
    nlohmann::json tokenizer_config;
    std::string bos_token_string = "<s>"; // Default BOS token
    try {
        std::string tok_config_str = read_file(tokenizer_config_path);
        tokenizer_config = nlohmann::json::parse(tok_config_str);
        if (tokenizer_config.contains("bos_token") && tokenizer_config["bos_token"].is_string()) {
            bos_token_string = tokenizer_config["bos_token"].get<std::string>();
            Logger::info("Using BOS token string from config: " + bos_token_string);
        } else {
            Logger::info("Could not find string 'bos_token' in tokenizer_config.json, using default: " + bos_token_string);
        }
    } catch (const std::exception& e) {
        Logger::info(std::string("Error loading tokenizer_config.json: ") + e.what() + ". Using default BOS token: " + bos_token_string);
    }

    // 2. Load tokenizer.model (SentencePiece)
    sentencepiece::SentencePieceProcessor sp;
    auto sp_status = sp.Load(tokenizer_path);
    if (!sp_status.ok()) {
        Logger::error("Failed to load SentencePiece model: " + sp_status.ToString());
        return 1;
    }
    // 3. Load model.safetensors
    try {
        SafeTensorsLoader st_loader(safetensors_path);
        auto names = st_loader.tensor_names();
        Logger::info("Loaded " + std::to_string(names.size()) + " tensors from model.safetensors:");
        for (const auto& n : names) {
            const auto& info = st_loader.get_tensor_info(n);
            std::string shape_str = "[";
            for (size_t i = 0; i < info.shape.size(); ++i) {
                shape_str += std::to_string(info.shape[i]);
                if (i + 1 < info.shape.size()) shape_str += ", ";
            }
            shape_str += "]";
            Logger::info("  " + n + " | dtype: " + info.dtype + ", shape: " + shape_str);
        }

        // Model weight loading example
        try {
            ModelConfig mcfg = parse_model_config(config);
            TinyLlamaModel model(mcfg, st_loader);
            Logger::info("TinyLlamaModel weights loaded successfully.");

            // Print first 10 values of lm_head (converted to float32)
            const auto& lm_head = model.get_lm_head();
            std::stringstream ss_lm;
            ss_lm << "lm_head first 10 values: ";
            for (int i = 0; i < 10 && i < lm_head.size(); ++i) ss_lm << bfloat16_to_float32(lm_head[i]) << " ";
            Logger::info(ss_lm.str());

            // Declare and Initialize KVCache
            KVCache cache;
            try {
                int nhl = mcfg.num_hidden_layers;
                int n_kv_heads = mcfg.num_key_value_heads;
                int max_seq_len = mcfg.max_position_embeddings;
                int head_dim = mcfg.hidden_size / mcfg.num_attention_heads;
                size_t cache_size_per_layer = (size_t)n_kv_heads * max_seq_len * head_dim;
                if (cache_size_per_layer == 0) {
                    throw std::runtime_error("Calculated cache size is zero. Check model config.");
                }
                cache.layers.resize(nhl);
                for (int l = 0; l < nhl; ++l) {
                    cache.layers[l].k.resize(cache_size_per_layer, 0.0f);
                    cache.layers[l].v.resize(cache_size_per_layer, 0.0f);
                }
                Logger::info("KVCache initialized. Layers: " + std::to_string(nhl) + ", Size per layer: " + std::to_string(cache_size_per_layer));
            } catch (const std::exception& e) {
                Logger::error(std::string("Failed to initialize KVCache: ") + e.what());
                return 1;
            }

            Tokenizer tokenizer(data_dir);
            int eos_id = tokenizer.get_special_token_id("eos");

            // Use only the PyTorch prompt template
            std::string prompt = "Q: What is the capital of France?\nA:";
            std::string full_prompt = bos_token_string + prompt;
            Logger::info("Full Prompt (with BOS string): " + full_prompt);

            // Tokenize the full prompt string
            std::vector<int> prompt_ids = tokenizer.tokenize(full_prompt);
            Logger::info("Tokenized IDs: Count=" + std::to_string(prompt_ids.size()));
            std::stringstream ss_token_ids;
            ss_token_ids << "Prompt token IDs: [ ";
            for (int id : prompt_ids) ss_token_ids << id << " ";
            ss_token_ids << "]";
            Logger::info(ss_token_ids.str());

            // Feed prompt tokens to model to fill KVCache
            cache.seq_len = 0; // Reset sequence length for each new prompt
            for (size_t i = 0; i < prompt_ids.size(); ++i) {
                // For the first token, print embedding stats
                if (i == 0) {
                    const auto& embed_tokens = model.get_embed_tokens();
                    std::vector<float> embedding_vec(model.get_config().hidden_size);
                    for (int j = 0; j < model.get_config().hidden_size; ++j) {
                        embedding_vec[j] = bfloat16_to_float32(embed_tokens[prompt_ids[0] * model.get_config().hidden_size + j]);
                    }
                    float minv = *std::min_element(embedding_vec.begin(), embedding_vec.end());
                    float maxv = *std::max_element(embedding_vec.begin(), embedding_vec.end());
                    float mean = std::accumulate(embedding_vec.begin(), embedding_vec.end(), 0.0f) / embedding_vec.size();
                    std::stringstream ss_emb;
                    ss_emb << "Embedding stats for first token: min=" << minv << ", max=" << maxv << ", mean=" << mean;
                    Logger::info(ss_emb.str());

                    // Print stats for first RMSNorm (input_layernorm) output for the first token
                    std::vector<float> rmsnorm_out(model.get_config().hidden_size);
                    rmsnorm(embedding_vec, model.get_embed_tokens().size() >= model.get_config().hidden_size ? model.get_embed_tokens() : model.get_lm_head(), model.get_config().rms_norm_eps, rmsnorm_out); // Use the correct weights below
                    // Actually, use the first layer's input_layernorm weights
                    const auto& layers = model.get_layers();
                    if (!layers.empty()) {
                        rmsnorm(embedding_vec, layers[0].input_layernorm, model.get_config().rms_norm_eps, rmsnorm_out);
                        float minv_rms = *std::min_element(rmsnorm_out.begin(), rmsnorm_out.end());
                        float maxv_rms = *std::max_element(rmsnorm_out.begin(), rmsnorm_out.end());
                        float mean_rms = std::accumulate(rmsnorm_out.begin(), rmsnorm_out.end(), 0.0f) / rmsnorm_out.size();
                        std::stringstream ss_rms;
                        ss_rms << "First RMSNorm output stats: min=" << minv_rms << ", max=" << maxv_rms << ", mean=" << mean_rms;
                        Logger::info(ss_rms.str());

                        // Print stats for first Q projection (q_proj) output for the first token
                        const auto& q_proj = layers[0].q_proj;
                        int hs = model.get_config().hidden_size;
                        std::vector<float> q_proj_out(hs, 0.0f);
                        // matvec_bf16_f32: out = mat [M,N] * vec [N] -> [M]
                        matvec_bf16_f32(q_proj, rmsnorm_out, q_proj_out, hs, hs);
                        float minv_q = *std::min_element(q_proj_out.begin(), q_proj_out.end());
                        float maxv_q = *std::max_element(q_proj_out.begin(), q_proj_out.end());
                        float mean_q = std::accumulate(q_proj_out.begin(), q_proj_out.end(), 0.0f) / q_proj_out.size();
                        std::stringstream ss_q;
                        ss_q << "First Q projection output stats: min=" << minv_q << ", max=" << maxv_q << ", mean=" << mean_q;
                        Logger::info(ss_q.str());

                        // Q before RoPE
                        std::stringstream ss_qpre;
                        ss_qpre << "Q before RoPE shape: [" << q_proj_out.size() << "] num_heads=" << model.get_config().num_attention_heads << " head_dim=" << hs / model.get_config().num_attention_heads << " pos=0 ";
                        ss_qpre << " first 5: ";
                        for (int j = 0; j < 5 && j < q_proj_out.size(); ++j) ss_qpre << q_proj_out[j] << " ";
                        Logger::info(ss_qpre.str());
                        // Q after RoPE
                        std::vector<float> q_rope = q_proj_out;
                        apply_rope(q_rope, model.get_config().num_attention_heads, hs / model.get_config().num_attention_heads, 0, model.get_config().rope_theta);
                        std::stringstream ss_qpost;
                        ss_qpost << "Q after RoPE shape: [" << q_rope.size() << "] first 5: ";
                        for (int j = 0; j < 5 && j < q_rope.size(); ++j) ss_qpost << q_rope[j] << " ";
                        Logger::info(ss_qpost.str());

                        // K projection
                        const auto& k_proj = layers[0].k_proj;
                        int kv_dim = (hs / model.get_config().num_attention_heads) * model.get_config().num_key_value_heads;
                        std::vector<float> k_proj_out(kv_dim, 0.0f);
                        matvec_bf16_f32(k_proj, rmsnorm_out, k_proj_out, kv_dim, hs);
                        float minv_k = *std::min_element(k_proj_out.begin(), k_proj_out.end());
                        float maxv_k = *std::max_element(k_proj_out.begin(), k_proj_out.end());
                        float mean_k = std::accumulate(k_proj_out.begin(), k_proj_out.end(), 0.0f) / k_proj_out.size();
                        std::stringstream ss_k;
                        ss_k << "First K projection output stats: min=" << minv_k << ", max=" << maxv_k << ", mean=" << mean_k;
                        Logger::info(ss_k.str());

                        // V projection
                        const auto& v_proj = layers[0].v_proj;
                        std::vector<float> v_proj_out(kv_dim, 0.0f);
                        matvec_bf16_f32(v_proj, rmsnorm_out, v_proj_out, kv_dim, hs);
                        float minv_v = *std::min_element(v_proj_out.begin(), v_proj_out.end());
                        float maxv_v = *std::max_element(v_proj_out.begin(), v_proj_out.end());
                        float mean_v = std::accumulate(v_proj_out.begin(), v_proj_out.end(), 0.0f) / v_proj_out.size();
                        std::stringstream ss_v;
                        ss_v << "First V projection output stats: min=" << minv_v << ", max=" << maxv_v << ", mean=" << mean_v;
                        Logger::info(ss_v.str());

                        // Q after RoPE (already logged above)
                        // K after RoPE (no-op for t=0, but keep for symmetry)
                        std::vector<float> k_rope = k_proj_out;
                        // Attention score (dot Q_rope, K_rope) using only first kv_dim of Q
                        float attn_score = 0.0f;
                        for (int i = 0; i < kv_dim; ++i) attn_score += q_rope[i] * k_rope[i];
                        Logger::info("First attention score (dot Q_rope, K_rope): " + std::to_string(attn_score));

                        // Attention probability (softmax)
                        std::vector<float> attn_scores = {attn_score};
                        softmax(attn_scores);
                        Logger::info("First attention probability (after softmax): " + std::to_string(attn_scores[0]));

                        // Attention output (context vector, weighted sum of V) using only first kv_dim
                        std::vector<float> attn_out(kv_dim, 0.0f);
                        for (int i = 0; i < kv_dim; ++i) attn_out[i] = attn_scores[0] * v_proj_out[i];
                        float minv_attn = *std::min_element(attn_out.begin(), attn_out.end());
                        float maxv_attn = *std::max_element(attn_out.begin(), attn_out.end());
                        float mean_attn = std::accumulate(attn_out.begin(), attn_out.end(), 0.0f) / attn_out.size();
                        std::stringstream ss_attn;
                        ss_attn << "First attention output stats: min=" << minv_attn << ", max=" << maxv_attn << ", mean=" << mean_attn;
                        Logger::info(ss_attn.str());
                    }
                }
                model.forward(prompt_ids[i], i, &cache);
            }

            // Generation loop
            std::vector<int> generated_ids;
            int max_new_tokens = 10; // Restrict to 10 tokens for efficiency
            int last_token = prompt_ids.empty() ? tokenizer.get_special_token_id("bos") : prompt_ids.back();

            for (int t = 0; t < max_new_tokens; ++t) {
                int current_pos = cache.seq_len;
                std::vector<float> logits = model.forward(last_token, current_pos, &cache);
                if (t == 0) {
                    std::stringstream ss_logits;
                    ss_logits << "First generated token logits (first 10): ";
                    for (int i = 0; i < 10 && i < logits.size(); ++i) ss_logits << logits[i] << " ";
                    Logger::info(ss_logits.str());
                }
                int next_token = argmax(logits);
                if (next_token == eos_id) break;
                generated_ids.push_back(next_token);
                last_token = next_token;
            }

            // Log the raw generated IDs before detokenization
            std::stringstream ss_ids;
            ss_ids << "Generated Token IDs: [ ";
            for (int id : generated_ids) {
                ss_ids << id << " ";
            }
            ss_ids << "]";
            Logger::info(ss_ids.str());

            // Detokenize and print answer
            std::string answer = tokenizer.detokenize(generated_ids);
            Logger::info("Generated answer: " + answer);
        } catch (const std::exception& e) {
            Logger::error(std::string("Model weight loading error: ") + e.what());
            return 1;
        }
    } catch (const std::exception& e) {
        Logger::error(std::string("Error loading model.safetensors: ") + e.what());
        return 1;
    }
    Logger::info("Pipeline execution finished.\n");
    return 0;
} 