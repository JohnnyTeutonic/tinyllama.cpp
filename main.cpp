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
#include "model.h"
#include <limits>
#include <random>
#include <functional>
#include <cstdio> // For std::remove
#include <sstream>
#include <numeric>
#include <iomanip> // Include for std::setw, std::fixed, std::setprecision
#include "vocab_loader.h"

// --- START: Helper Functions (Sampling & Softmax) ---

// Softmax function (copied from model.cpp helpers)
static void softmax_vector_cpu(const std::vector<float>& x, std::vector<float>& out) {
    if (x.empty()) return;
    out.resize(x.size());

    // Find max value (for numerical stability)
    float max_val = -std::numeric_limits<float>::infinity();
    for (float val : x) {
        if (val > max_val) {
            max_val = val;
        }
    }

    // Exp and sum
    float sum = 0.0f;
    for (size_t i = 0; i < x.size(); ++i) {
        out[i] = std::exp(x[i] - max_val);
        sum += out[i];
    }

    // Normalize
    if (sum == 0.0f) { // Handle case where all exp(x_i - max_val) are zero
         // Assign uniform probability (or handle as error/special case)
         float uniform_prob = 1.0f / out.size();
         for (size_t i = 0; i < out.size(); ++i) {
             out[i] = uniform_prob;
         }
         // Optionally log a warning
         // Logger::warning("Softmax sum is zero, assigning uniform probability.");
    } else {
        float inv_sum = 1.0f / sum;
        for (size_t i = 0; i < out.size(); ++i) {
            out[i] *= inv_sum;
        }
    }
}

// Sample from a multinomial distribution given probabilities
static int sample_multinomial(const std::vector<float>& probabilities) {
    if (probabilities.empty()) {
        throw std::runtime_error("Cannot sample from empty probability vector.");
    }

    // Set up random number generation
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float sample = dist(gen);

    // Calculate cumulative probabilities and find the sample index
    float cumulative_prob = 0.0f;
    for (size_t i = 0; i < probabilities.size(); ++i) {
        cumulative_prob += probabilities[i];
        if (sample < cumulative_prob) {
            return static_cast<int>(i);
        }
    }

    // Should not happen if probabilities sum to 1, but return last index as fallback
    Logger::warning("Multinomial sampling failed to find index due to sample value >= cumulative sum. Returning last index.");
     return static_cast<int>(probabilities.size() - 1);
}

// --- END: Helper Functions ---

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
            Logger::info("[main.cpp] Config - rope_theta: " + std::to_string(mcfg.rope_theta) + ", rms_norm_eps: " + std::to_string(mcfg.rms_norm_eps));
            Logger::info("[main.cpp] Full Config: hidden_size=" + std::to_string(mcfg.hidden_size) +
                         ", intermediate_size=" + std::to_string(mcfg.intermediate_size) +
                         ", num_attention_heads=" + std::to_string(mcfg.num_attention_heads) +
                         ", num_key_value_heads=" + std::to_string(mcfg.num_key_value_heads) +
                         ", num_hidden_layers=" + std::to_string(mcfg.num_hidden_layers) +
                         ", vocab_size=" + std::to_string(mcfg.vocab_size) +
                         ", max_position_embeddings=" + std::to_string(mcfg.max_position_embeddings) +
                         ", rms_norm_eps=" + std::to_string(mcfg.rms_norm_eps) +
                         ", rope_theta=" + std::to_string(mcfg.rope_theta) +
                         ", bos_token_id=" + std::to_string(mcfg.bos_token_id) +
                         ", eos_token_id=" + std::to_string(mcfg.eos_token_id)
                         );
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
            std::string raw_prompt_query = "What color is the sky?"; // Standardized spelling
            std::string prompt_to_tokenize = "Q: " + raw_prompt_query + "\nA:";
            Logger::info("[main.cpp] Raw Prompt Query: '" + raw_prompt_query + "'");
            Logger::info("[main.cpp] Prompt to Tokenize: '" + prompt_to_tokenize + "'");

            // Option 1: Use the low-level API as before
            std::vector<std::string> prompt_tokens = tokenizer.tokenize(prompt_to_tokenize);
            std::vector<int> prompt_ids = tokenizer.tokens_to_ids(prompt_tokens);
            
            Logger::info("Tokenizing prompt...");
            // +++ START TOKEN LOGGING +++
            std::stringstream ptk_ss;
            ptk_ss << "Prompt token IDs (SafeTensors): [";
            for(size_t i=0; i < prompt_ids.size(); ++i) {
                ptk_ss << prompt_ids[i] << (i + 1 < prompt_ids.size() ? ", " : "");
            }
            ptk_ss << "]";
            Logger::info(ptk_ss.str());
            // +++ END TOKEN LOGGING +++

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
            std::vector<float> current_x_vec; // Use std::vector for state
            cache.seq_len = 0; // Reset cache length
            
            Logger::info("Starting token-by-token processing loop with KVCache...");
            std::cout << "Generating tokens..." << std::endl; // Initial message
            
            int total_steps = num_prompt_tokens + max_new_tokens -1; // Prompt processing + generation
            int generated_count = 0;
            std::mt19937 rng(1234); // Random number generator for sampling

            for (int pos = 0; pos < total_steps; ++pos) {
                // +++ START POS 0 LOGGING +++
                bool log_this_step = (pos == 0);
                // +++ END POS 0 LOGGING +++
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
                
                // Lookup embedding (returns vector)
                current_x_vec = model.lookup_embedding(input_token_id);
                
                // Log the vector state
                // +++ START POS 0 LOGGING +++
                if (log_this_step) {
                    log_vector_summary("Prompt Input Embedding (pos=0)", current_x_vec, 10);
                }
                // +++ END POS 0 LOGGING +++
                // log_vector_summary("main loop: current_x after lookup (C++ Vec)", current_x_vec); 

                // --- Select Forward Pass based on CUDA availability ---
                std::vector<float> logits;
#ifdef HAS_CUDA
                // Use CUDA-accelerated forward pass
                Logger::info("main loop: Calling model.forward_device for pos=" + std::to_string(pos));
                logits = model.forward_device(input_token_id, pos, &cache, nullptr); // Use device pipeline
                Logger::info("main loop: Returned from model.forward_device for pos=" + std::to_string(pos));
                // +++ START POS 0 LOGGING +++
                if (log_this_step) {
                    Logger::info("Completed model.forward_device() for prompt token at pos=0.");
                }
                // +++ END POS 0 LOGGING +++
#else
                // Use CPU forward pass
                Logger::info("main loop: Calling model.forward (CPU) for pos=" + std::to_string(pos));
                logits = model.forward(current_x_vec, pos, &cache, nullptr); // Use original CPU pipeline
                Logger::info("main loop: Returned from model.forward (CPU) for pos=" + std::to_string(pos));
                // +++ START POS 0 LOGGING +++
                if (log_this_step) {
                    Logger::info("Completed model.forward() (CPU) for prompt token at pos=0.");
                }
                // +++ END POS 0 LOGGING +++
#endif // HAS_CUDA
                // --- End Forward Pass Selection ---

                // Check if logits are empty (could happen if forward fails)
                if (logits.empty()) {
#ifdef HAS_CUDA
                     Logger::error("model.forward_device returned empty logits at pos " + std::to_string(pos));
#else
                     Logger::error("model.forward returned empty logits at pos " + std::to_string(pos));
#endif
                     break; // Stop generation if forward pass failed
                }
                
                // --- Crucial: Increment cache sequence length *AFTER* forward call for position `pos`
                cache.seq_len = pos + 1; 
                Logger::info("main loop: Updated cache.seq_len=" + std::to_string(cache.seq_len));
                
                // --- RUNTIME CHECKS --- // Updated for vector state
                if (current_x_vec.size() != mcfg.hidden_size) { // Check vector size
                    Logger::error("current_x_vec size mismatch after forward at pos " + std::to_string(pos) + ". Expected " + std::to_string(mcfg.hidden_size) + ", got " + std::to_string(current_x_vec.size()));
                    break;
                }
                
                log_vector_summary("main loop: logits after forward", logits); // Logits are still vector
                log_vector_summary("main loop: current_x_vec after forward (state)", current_x_vec); // Log vector state
                
                // 3. Sample Next Token (Only during Generation Phase)
                if (pos >= num_prompt_tokens - 1) {
                    if (logits.empty()) {
                        Logger::error("model.forward (or device) returned empty logits at pos " + std::to_string(pos));
                        break;
                    }

                    // <<< START TOP-K LOGGING >>>
                    if (generated_count < 10) { // Log for first 10 generated tokens
                        std::vector<std::pair<float, int>> logits_pairs;
                        for (size_t i = 0; i < logits.size(); ++i) {
                            logits_pairs.push_back({logits[i], static_cast<int>(i)});
                        }
                        std::partial_sort(logits_pairs.begin(), logits_pairs.begin() + 5, logits_pairs.end(), std::greater<std::pair<float, int>>());
                        std::stringstream top_k_ss;
                        top_k_ss << "[main.cpp] Gen Step " << generated_count << " (Pos " << pos << ") Top-5 logits: ";
                        for (int k_idx = 0; k_idx < 5 && k_idx < logits_pairs.size(); ++k_idx) {
                            std::string token_str = "<INVALID_TOPK>";
                            if (logits_pairs[k_idx].second >= 0 && static_cast<size_t>(logits_pairs[k_idx].second) < tokenizer.vocab_size()) {
                                 std::vector<std::string> t_vec = tokenizer.ids_to_tokens({logits_pairs[k_idx].second});
                                 if(!t_vec.empty()) token_str = t_vec[0];
                            }
                            top_k_ss << "(" << logits_pairs[k_idx].second << ": '" << token_str << "', " << logits_pairs[k_idx].first << ") ";
                        }
                        Logger::info(top_k_ss.str());
                    }
                    // <<< END TOP-K LOGGING >>>
                    
                    next_token_id = argmax(logits);
                    Logger::info("[main.cpp] Gen Step " + std::to_string(generated_count) + " (Pos " + std::to_string(pos) + ") PREDICTED ID: " + std::to_string(next_token_id));

                    generated_only_ids.push_back(next_token_id);
                    generated_ids.push_back(next_token_id); 
                    
                    std::string next_token_str = "<INVALID>";
                    if (next_token_id >= 0 && static_cast<size_t>(next_token_id) < tokenizer.vocab_size()) {
                         std::vector<std::string> token_vec_str = tokenizer.ids_to_tokens({next_token_id});
                         if (!token_vec_str.empty()) { next_token_str = token_vec_str[0]; }
                    }
                    Logger::info("[main.cpp] Gen Step " + std::to_string(generated_count) + " (Pos " + std::to_string(pos) + ") DECODED: '" + next_token_str + "'");
                    std::cout << next_token_str << std::flush; // Print token immediately

                    generated_count++;

                    if (next_token_id == eos_id) {
                        Logger::info("[main.cpp] Gen Step " + std::to_string(generated_count -1) + ": EOS token (ID: " + std::to_string(eos_id) + ") generated. Stopping generation.");
                        break; 
                    }
                    if (generated_count >= max_new_tokens) {
                        Logger::info("[main.cpp] Gen Step " + std::to_string(generated_count -1) + ": Max new tokens (" + std::to_string(max_new_tokens) + ") reached. Stopping generation.");
                        break;
                    }
                } else {
                     // The state `current_x_tensor` is updated in-place by model.forward_device().
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