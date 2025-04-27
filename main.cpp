#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <memory>
#include <stdexcept>
#include <nlohmann/json.hpp>
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
#include <iomanip> // Include for std::setw, std::fixed, std::setprecision
#include "vocab_loader.h"
#include <torch/torch.h>

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
    std::string tokenizer_path = data_dir + "/tokenizer.json";
    std::string vocab_path = tokenizer_path; // Use tokenizer.json for vocab
    std::string safetensors_path = data_dir + "/model.safetensors";
    // std::string tokenizer_config_path = data_dir + "/tokenizer_config.json"; // No longer needed

    Logger::info("Using data directory: " + data_dir);
    Logger::info("Loading config: " + config_path);

    // 1. Load model config.json
    nlohmann::json config;
    try {
        std::string config_str = read_file(config_path);
        config = nlohmann::json::parse(config_str);
    } catch (const std::exception& e) {
        Logger::error(std::string("Error loading config.json: ") + e.what());
        return 1;
    }

    // --- REMOVED tokenizer_config.json loading --- 

    // 2. Load tokenizer (using mlc-ai/tokenizers-cpp wrapper)
    std::unique_ptr<Tokenizer> tokenizer_ptr;
    try {
        tokenizer_ptr = std::make_unique<Tokenizer>(tokenizer_path, vocab_path);
        Logger::info("Loaded tokenizer: " + tokenizer_path);
    } catch (const std::exception& e) {
        Logger::error(std::string("Error loading tokenizer: ") + e.what());
        return 1;
    }
    const Tokenizer& tokenizer = *tokenizer_ptr;

    // 3. Load model.safetensors
    try {
        SafeTensorsLoader st_loader(safetensors_path);
        auto names = st_loader.tensor_names();
        Logger::info("Loaded " + std::to_string(names.size()) + " tensors from model.safetensors:");

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
                
                // Use the new initialization method
                cache.initialize(nhl, max_seq_len, n_kv_heads, head_dim);
                
                Logger::info("KVCache successfully initialized.");
            } catch (const std::exception& e) {
                Logger::error(std::string("Failed to initialize KVCache: ") + e.what());
                return 1;
            }

            // --- Special Token Handling --- 
            // Use tokenizer's accessors to get special token IDs
            int eos_id = tokenizer.eos_token_id();
            int bos_id = tokenizer.bos_token_id();
            int unk_id = tokenizer.unk_token_id();
            int pad_id = tokenizer.pad_token_id();
            
            Logger::info("Special token IDs from tokenizer: EOS=" + std::to_string(eos_id) + 
                         ", BOS=" + std::to_string(bos_id) + 
                         ", UNK=" + std::to_string(unk_id) + 
                         ", PAD=" + std::to_string(pad_id));
            
            // Check if EOS token is valid
            if (eos_id < 0) {
                Logger::info("Warning: Invalid EOS token ID. Using default value of 2.");
                eos_id = 2;  // Fallback to common default
            }

            // Use only the PyTorch prompt template
            std::string prompt = "Q: What is the capital of France?\nA:";
            // Option 1: Use the low-level API as before
            std::vector<std::string> prompt_tokens = tokenizer.tokenize(prompt);
            std::vector<int> prompt_ids = tokenizer.tokens_to_ids(prompt_tokens);
            
            // Option 2 (alternative): Use the encode method which can add special tokens if configured
            // std::vector<int> prompt_ids = tokenizer.encode(prompt, true); // true to add BOS/EOS if configured
            
            Logger::info("Tokenizing prompt...");

            Logger::info("Tokenized IDs: Count=" + std::to_string(prompt_ids.size()));
            std::stringstream ss_token_ids;
            ss_token_ids << "Prompt token IDs: [ ";
            for (int id : prompt_ids) ss_token_ids << id << " ";
            ss_token_ids << "]";
            Logger::info(ss_token_ids.str());

            // --- Feed prompt tokens & Generate ---
            int next_token_id = -1; // Initialize next token
            int max_new_tokens = 30; // Set max generated tokens to 30
            int num_prompt_tokens = prompt_ids.size();
            std::vector<int> generated_ids = prompt_ids;
            std::vector<int> generated_only_ids;
            torch::Tensor current_x_tensor; // Use Tensor for state
            cache.seq_len = 0; // Reset cache length

            Logger::info("Starting token-by-token processing loop with KVCache...");
            std::cout << "Generating tokens..." << std::endl; // Initial message
            
            int total_steps = num_prompt_tokens + max_new_tokens -1; // Prompt processing + generation
            int generated_count = 0;
            std::mt19937 rng(1234); // Random number generator for sampling

            for (int pos = 0; pos < total_steps; ++pos) {
                Logger::info("--- main loop: START pos=" + std::to_string(pos) + " ---");
                if (pos >= mcfg.max_position_embeddings) {
                    Logger::info("Reached max sequence length.");
                    break;
                }

                // --- START: Progress Bar --- 
                if (pos >= num_prompt_tokens - 1) { // Show progress only during generation
                    float progress = (float)(generated_count + 1) / max_new_tokens;
                    int barWidth = 40;
                    std::cout << "[";
                    int bar_pos = barWidth * progress;
                    for (int k = 0; k < barWidth; ++k) {
                        if (k < bar_pos) std::cout << "=";
                        else if (k == bar_pos) std::cout << ">";
                        else std::cout << " ";
                    }
                    std::cout << "] " << std::fixed << std::setprecision(1) << progress * 100.0 << "% (" << (generated_count + 1) << "/" << max_new_tokens << ")\r";
                    std::cout.flush();
                }
                // --- END: Progress Bar ---

                // 1. Determine Input Token ID & Prepare State x
                int input_token_id = (pos < num_prompt_tokens) ? prompt_ids[pos] : next_token_id;
                Logger::info("main loop: input_token_id=" + std::to_string(input_token_id));
                // Lookup embedding (returns tensor)
                current_x_tensor = model.lookup_embedding(input_token_id);
                // Log tensor stats directly if possible, or convert for old logger
                log_vector_summary("main loop: current_x after lookup", std::vector<float>(static_cast<float*>(current_x_tensor.data_ptr()), static_cast<float*>(current_x_tensor.data_ptr()) + mcfg.hidden_size));

                // 2. Call the Forward Pass (token-by-token)
                // Pass tensor by reference. It will be updated in place by some layers?
                // Or does forward return the new state? CHECK MODEL.CPP SIGNATURE
                // --> forward modifies x_tensor in place via RMSNorm, RoPE, residuals
                // --> forward *returns* logits as vector<float>
                Logger::info("main loop: Calling model.forward for pos=" + std::to_string(pos));
                std::vector<float> logits = model.forward(current_x_tensor, pos, &cache, nullptr); // Pass tensor state
                Logger::info("main loop: Returned from model.forward for pos=" + std::to_string(pos));
                
                // --- Crucial: Increment cache sequence length *AFTER* forward call for position `pos`
                cache.seq_len = pos + 1; 
                Logger::info("main loop: Updated cache.seq_len=" + std::to_string(cache.seq_len));
                
                // --- RUNTIME CHECKS ---
                if (logits.empty()) {
                    Logger::error("model.forward returned empty logits at pos " + std::to_string(pos));
                    break;
                }
                // Check tensor state after forward pass
                if (current_x_tensor.numel() != mcfg.hidden_size) { // Check tensor size
                    Logger::error("current_x_tensor size mismatch after forward at pos " + std::to_string(pos) + ". Expected " + std::to_string(mcfg.hidden_size) + ", got " + std::to_string(current_x_tensor.numel()));
                    break;
                }
                
                log_vector_summary("main loop: logits after forward", logits); // Logits are still vector
                log_vector_summary("main loop: current_x_tensor after forward (state)", std::vector<float>(static_cast<float*>(current_x_tensor.data_ptr()), static_cast<float*>(current_x_tensor.data_ptr()) + mcfg.hidden_size));
                
                // 3. Sample Next Token (Only during Generation Phase)
                if (pos >= num_prompt_tokens - 1) {
                    if (logits.empty()) {
                        Logger::error("model.forward returned empty logits at pos " + std::to_string(pos));
                        break;
                    }
                    
                    // Sample from the returned logits (which are only for the current position)
                    next_token_id = sample_top_k_top_p_temperature(logits, /*temp=*/0.7f, /*top_k=*/50, /*top_p=*/0.9f, rng);
                    
                    // --- ADD LOGGING HERE ---
                    Logger::info("C++ Generation: pos=" + std::to_string(pos) + ", sampled_token_id=" + std::to_string(next_token_id));
                    // --- END LOGGING ---

                    // Store generated token
                    generated_only_ids.push_back(next_token_id);
                    generated_ids.push_back(next_token_id); // Keep track of the full sequence for context
                    generated_count++;

                    // Print the token as it's generated
                    std::string next_token_str = tokenizer.decode(std::vector<int>{next_token_id}, true); // Skip special tokens
                    Logger::info("Step " + std::to_string(generated_count+1) + ": Predicted token ID: " + std::to_string(next_token_id) + " ('" + next_token_str + "')");
                    std::cout << next_token_str << std::flush; // Print token immediately, with flush

                    // --- Use eos_id from tokenizer --- 
                    if (next_token_id == eos_id || generated_count >= max_new_tokens) {
                        if (next_token_id == eos_id) Logger::info("EOS token (" + std::to_string(eos_id) + ") generated. Stopping generation.");
                        else Logger::info("Max new tokens reached. Stopping generation.");
                        break; 
                    }
                } else {
                     // During prompt processing, we don't sample, just process the next prompt token.
                     // The state `current_x_tensor` is updated in-place by model.forward().
                     Logger::info("Processed prompt token at pos " + std::to_string(pos) + ".");
                }
                Logger::info("--- main loop: END pos=" + std::to_string(pos) + " ---");
            } // End generation loop
            std::cout << std::endl; // Move to the next line after progress bar finishes

            // Decode and print the generated part using the improved decode method
            std::string generated_text = tokenizer.decode(generated_only_ids, true); // Skip special tokens
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