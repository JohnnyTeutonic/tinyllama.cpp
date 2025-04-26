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

// Helper: log vector stats (min, max, mean, all finite)
void log_vec_stats(const std::string& name, const std::vector<float>& v) {
    if (v.empty()) {
        Logger::info(name + ": EMPTY VECTOR");
        return;
    }
    float minv = *std::min_element(v.begin(), v.end());
    float maxv = *std::max_element(v.begin(), v.end());
    float mean = std::accumulate(v.begin(), v.end(), 0.0f) / v.size();
    bool all_finite = std::all_of(v.begin(), v.end(), [](float x) { return std::isfinite(x); });
    Logger::info(name + ": min=" + std::to_string(minv) + ", max=" + std::to_string(maxv) + ", mean=" + std::to_string(mean) + ", all_finite=" + (all_finite ? "yes" : "no"));
}

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

            // --- Declare persistent state vector ---
            std::vector<float> current_x(mcfg.hidden_size);
            // --------------------------------------

            // --- Feed prompt tokens & Generate ---
            cache.seq_len = 0; // Reset sequence length 
            int next_token_id = -1; // Initialize next token
            int max_new_tokens = 50; // Example limit
            int num_prompt_tokens = prompt_ids.size();
            std::vector<int> generated_ids = prompt_ids;
            std::vector<int> generated_only_ids; 
            
            Logger::info("Starting combined prompt processing and generation loop...");
            for (int pos = 0; pos < num_prompt_tokens + max_new_tokens; ++pos) {
                if (pos >= mcfg.max_position_embeddings) {
                    Logger::info("Reached max sequence length.");
                    break;
                }

                // 1. Determine Input Token ID
                int input_token_id = (pos < num_prompt_tokens) ? prompt_ids[pos] : next_token_id;

                // 2. Lookup Embedding for the CURRENT token
                std::vector<float> embedding = model.lookup_embedding(input_token_id); 
                
                // 3. Initialize or Use Previous State
                if (pos == 0) {
                    // Initialize the state with the first token's embedding
                    current_x = embedding; 
                    Logger::info("Initialized current_x with embedding for pos 0.");
                } else {
                    // For subsequent steps, 'current_x' already holds the output state 
                    // from the previous iteration's forward pass. 
                    // Pass it directly to the forward call.
                    if (pos == 1) { // Logging for the step we are debugging
                        log_vec_stats("main.cpp: current_x BEFORE forward call (pos=1)", current_x);
                        std::stringstream ss_x_main_before;
                        ss_x_main_before << "main.cpp: current_x BEFORE forward call (pos=1) first 5: [";
                        for(int i=0; i < 5 && i < current_x.size(); ++i) ss_x_main_before << (i > 0 ? " " : "") << current_x[i];
                        ss_x_main_before << "]";
                        Logger::info(ss_x_main_before.str());
                    }
                }

                // 4. Call the Modified Forward Pass
                // Pass 'current_x' by reference. It will be updated in place.
                // 'pos' represents the current position being processed.
                Logger::info("Calling forward for pos: " + std::to_string(pos));
                std::vector<float> logits = model.forward(current_x, pos, &cache); 
                
                // --- INCREMENT CACHE LENGTH *AFTER* FORWARD CALL --- 
                cache.seq_len = pos + 1; // Update cache sequence length after processing 'pos'
                // ----------------------------------------------------

                // 5. Log State After Forward (Optional but helpful)
                if (pos == 1) { // Log specifically after pos=1 processing
                    log_vec_stats("main.cpp: current_x AFTER forward call (pos=1)", current_x);
                    std::stringstream ss_x_main_after;
                    ss_x_main_after << "main.cpp: current_x AFTER forward call (pos=1) first 5: [";
                    for(int i=0; i < 5 && i < current_x.size(); ++i) ss_x_main_after << (i > 0 ? " " : "") << current_x[i];
                    ss_x_main_after << "]";
                    Logger::info(ss_x_main_after.str());
                }
                
                // 6. Sample Next Token (Only during Generation Phase)
                if (pos >= num_prompt_tokens - 1) {
                    if (logits.empty()) {
                        Logger::error("model.forward returned empty logits at pos " + std::to_string(pos));
                        break;
                    }
                    
                    // Simple argmax sampling for now
                    next_token_id = argmax(logits); 
                    Logger::info("Predicted next token ID: " + std::to_string(next_token_id) + " for pos " + std::to_string(pos));

                    // Store generated token if generating
                    if (pos >= num_prompt_tokens) { // Make sure we only store *newly* generated tokens
                        generated_only_ids.push_back(next_token_id);
                    }
                    generated_ids.push_back(next_token_id); // Keep track of the full sequence

                    // Check for EOS
                    if (next_token_id == eos_id) {
                        Logger::info("EOS token generated at pos " + std::to_string(pos) + ". Stopping generation.");
                        break; 
                    }
                } else {
                    // During prompt processing, we don't sample, just continue
                    // next_token_id is only needed *after* the last prompt token
                     Logger::info("Processed prompt token at pos " + std::to_string(pos) + ". Continuing.");
                }
            } // End generation loop

            // Decode and print the generated part
            std::string generated_text = tokenizer.detokenize(generated_only_ids);
            Logger::info("Generated Token IDs: " + [&](){
                std::stringstream ss;
                ss << "[ ";
                for(int id : generated_only_ids) ss << id << " ";
                ss << "]";
                return ss.str();
            }());
            Logger::info("Generated answer: " + generated_text);
            std::cout << "\nGenerated Answer:\n" << generated_text << std::endl;
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