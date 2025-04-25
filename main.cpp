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

            // Declare KVCache once for both logging and generation
            KVCache cache;

            // --- START: Dedicated Forward Pass for Token ID 1 Logging ---
            Logger::info("--- Deleting previous layer_0_cpp_outputs.log (if exists) ---");
            std::remove("layer_0_cpp_outputs.log"); // Delete log file

            Logger::info("--- Running dedicated forward pass for token ID 1 for logging ---");
            ForwardDiagCallback diag_cb_for_token1 = [](int layer, const std::string& name, const std::vector<float>& v) {
                 // Minimal logging or reuse main diag_cb logic
                 // Logger::log_vector_stats("Token1 Layer " + std::to_string(layer) + " " + name, v, 5);
            };
            model.forward(1, 0, &cache, diag_cb_for_token1);
            Logger::info("--- Dedicated forward pass for token ID 1 complete ---");
            // --- END: Dedicated Forward Pass ---

            // === End-to-end text generation with sampling ===
            // Create the tokenizer needed for this section
            Tokenizer tokenizer(data_dir); // Instantiate tokenizer here

            std::vector<std::string> questions = {
                "What is the capital of France?",
                "Who wrote Hamlet?"
            };
            for (const auto& question : questions) {
                std::string prompt = format_prompt(question);
                Logger::info("Prompt: " + prompt);

                // Tokenize prompt and prepend BOS token
                std::vector<int> prompt_ids = tokenizer.tokenize(prompt);
                int bos_id = tokenizer.get_special_token_id("bos");
                int eos_id = tokenizer.get_special_token_id("eos");
                prompt_ids.insert(prompt_ids.begin(), bos_id);

                // Feed prompt tokens to model to fill KVCache
                cache.seq_len = 0;
                for (size_t i = 0; i < prompt_ids.size(); ++i) {
                    model.forward(prompt_ids[i], i, &cache);
                    cache.seq_len++;
                }

                // Generation loop
                std::vector<int> generated_ids;
                int max_new_tokens = 64;
                int last_token = prompt_ids.back();
                std::mt19937 rng(42); // Fixed seed for reproducibility
                for (int t = 0; t < max_new_tokens; ++t) {
                    std::vector<float> logits = model.forward(last_token, prompt_ids.size() + t, &cache);
                    int next_token = sample_top_k_top_p_temperature(logits, 0.8f, 40, 0.95f, rng);
                    if (next_token == eos_id) break;
                    generated_ids.push_back(next_token);
                    last_token = next_token;
                    cache.seq_len++;
                }

                // Detokenize and print answer
                std::string answer = tokenizer.detokenize(generated_ids);
                Logger::info("Generated answer: " + answer);
            }
        } catch (const std::exception& e) {
            Logger::error(std::string("Model weight loading error: ") + e.what());
            return 1;
        }
    } catch (const std::exception& e) {
        Logger::error(std::string("Error loading model.safetensors: ") + e.what());
        return 1;
    }
    Logger::info("Pipeline finalised.\n");
    return 0;
} 