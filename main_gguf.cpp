#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <random>     // ADDED for random number generation
#include <numeric>    // ADDED for std::partial_sum
#include <limits>     // ADDED for std::numeric_limits
#include <algorithm>  // ADDED for std::transform, std::max_element
#include <cmath>      // ADDED for std::exp, std::log
#include <iomanip>    // ADDED for std::setw, std::setfill
#include "model.h"
#include "logger.h"
#include "gguf_parser.h"
#include "tokenizer.h"

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

// --- ADDED: Function to run generation experiment ---
void run_generation_experiment(TinyLlamaModel& model,
                               Tokenizer& tokenizer,
                               const std::vector<int>& prompt_ids, // Renamed from prompt_tokens
                               int max_new_tokens,
                               float temperature,
                               const ModelConfig& config,
                               const std::string& attempt_label)
{
    Logger::info("\n==============================================");
    Logger::info("Starting Generation Experiment: " + attempt_label);
    Logger::info("==============================================");

    try {
        // Log the provided tokenization result
        std::stringstream ptk_ss;
        ptk_ss << "Prompt token IDs (" << attempt_label << "): [";
        for(size_t i=0; i < prompt_ids.size(); ++i) {
            ptk_ss << prompt_ids[i] << (i + 1 < prompt_ids.size() ? ", " : "");
        }
        ptk_ss << "]";
        Logger::info(ptk_ss.str());
        Logger::info("Prompt token count: " + std::to_string(prompt_ids.size()));
        if (!prompt_ids.empty()) {
            std::vector<std::string> decoded_prompt_strs = tokenizer.ids_to_tokens(prompt_ids);
            std::stringstream dpt_ss;
            dpt_ss << "Decoded prompt tokens: [";
            for(size_t i=0; i < decoded_prompt_strs.size(); ++i) {
                dpt_ss << "'" << decoded_prompt_strs[i] << "'" << (i + 1 < decoded_prompt_strs.size() ? ", " : "");
            }
            dpt_ss << "]";
            Logger::debug(dpt_ss.str());
        } else {
             Logger::debug("Decoded prompt tokens: [] (Prompt was empty)");
        }

        // --- Generation Setup ---
        int bos_token_id = config.bos_token_id; // Still needed? main.cpp doesn't seem to use it explicitly here
        int eos_token_id = config.eos_token_id;
        std::vector<int> generated_ids = prompt_ids; // Start with prompt
        std::vector<int> generated_only_ids; // To store only newly generated tokens

        Logger::info("Using EOS token ID: " + std::to_string(eos_token_id));
        Logger::info("Max new tokens: " + std::to_string(max_new_tokens));

        // Initialize KVCache (needs to be fresh for each run)
        KVCache kv_cache;
        kv_cache.initialize(config.num_hidden_layers, config.max_position_embeddings,
                            config.num_key_value_heads, config.hidden_size / config.num_attention_heads);
        // kv_cache.seq_len = 0; // Implicitly 0 after initialization

        Logger::info("KVCache initialized for this run.");

        // --- Unified Generation Loop (mirrors main.cpp) ---
        int next_token_id = -1; // Initialize next token for generation phase
        int num_prompt_tokens = prompt_ids.size();
        int total_steps = num_prompt_tokens + max_new_tokens - 1; // Max steps allowed
        int generated_count = 0;

        Logger::info("Starting unified generation loop...");

        for (int pos = 0; pos < total_steps; ++pos) {
             Logger::info("--- Unified Loop: START pos=" + std::to_string(pos) + " ---");
             if (pos >= config.max_position_embeddings) {
                 Logger::warning("Reached max sequence length (" + std::to_string(config.max_position_embeddings) + "). Stopping.");
                 break;
             }

             // 1. Determine Input Token ID
             int input_token_id = (pos < num_prompt_tokens) ? prompt_ids[pos] : next_token_id;
             Logger::info("Loop pos=" + std::to_string(pos) + ", input_token_id=" + std::to_string(input_token_id));

             // 2. Get Input Embedding (Moved inside #else for CPU path)
             // std::vector<float> input_embedding = model.lookup_embedding(input_token_id);
             // if (input_embedding.empty() || input_embedding.size() != config.hidden_size) {
             //     throw std::runtime_error("Failed to get valid embedding for token ID " + std::to_string(input_token_id) + " at pos " + std::to_string(pos));
             // }
             // if (pos == 0) { // Log first embedding
             //     log_vector_summary("Input Embedding (pos=0)", input_embedding, 10);
             // }

             // 3. Perform Forward Pass
             std::vector<float> logits;
#ifdef HAS_CUDA
            Logger::info("Loop pos=" + std::to_string(pos) + ", calling model.forward_device with token_id=" + std::to_string(input_token_id));
            logits = model.forward_device(input_token_id, pos, &kv_cache, nullptr);
            if (pos == 0) { // Log first logits from CUDA path
                 log_vector_summary("Output Logits from forward_device (pos=0)", logits, 10);
            }
#else
            // Get Input Embedding for CPU Path
            std::vector<float> input_embedding = model.lookup_embedding(input_token_id);
            if (input_embedding.empty() || input_embedding.size() != config.hidden_size) {
                 throw std::runtime_error("Failed to get valid embedding for token ID " + std::to_string(input_token_id) + " at pos " + std::to_string(pos));
            }
            if (pos == 0) { // Log first embedding for CPU path
                  log_vector_summary("Input Embedding to CPU forward (pos=0)", input_embedding, 10);
            }
            Logger::info("Loop pos=" + std::to_string(pos) + ", calling model.forward (CPU) with embedding");
            logits = model.forward(input_embedding, pos, &kv_cache, nullptr);
            if (pos == 0) { // Log first logits from CPU path
                  log_vector_summary("Output Logits from CPU forward (pos=0)", logits, 10);
            }
#endif // HAS_CUDA

            if (logits.empty() || logits.size() != config.vocab_size) {
#ifdef HAS_CUDA
                 throw std::runtime_error("model.forward_device() returned invalid logits at pos " + std::to_string(pos) + ". Size: " + std::to_string(logits.size()) + ", Expected: " + std::to_string(config.vocab_size));
#else
                 throw std::runtime_error("model.forward() (CPU) returned invalid logits at pos " + std::to_string(pos) + ". Size: " + std::to_string(logits.size()) + ", Expected: " + std::to_string(config.vocab_size));
#endif
             }
             // if (pos == 0) { // Log first logits (original combined log point)
             //      log_vector_summary("Output Logits (pos=0)", logits, 10);
             //  }

             // --- IMPORTANT: Increment cache sequence length *AFTER* forward call for position `pos` ---
             kv_cache.seq_len = pos + 1;
             // Logger::info("Loop pos=" + std::to_string(pos) + ", Updated kv_cache.seq_len = " + std::to_string(kv_cache.seq_len));


             // 4. Sample Next Token (Only during Generation Phase)
             if (pos >= num_prompt_tokens - 1) {
                  // Log Top-K before sampling (only for first few generation steps)
                  if (generated_count < 4) {
                        std::vector<std::pair<int, float>> idx_logit_pairs;
                        idx_logit_pairs.reserve(logits.size());
                        for (int i = 0; i < (int)logits.size(); ++i) {
                            idx_logit_pairs.emplace_back(i, logits[i]);
                        }
                        std::partial_sort(idx_logit_pairs.begin(), idx_logit_pairs.begin() + 5, idx_logit_pairs.end(),
                            [](const auto& a, const auto& b) { return a.second > b.second; });
                        
                        std::vector<int> top_ids; top_ids.reserve(5);
                        std::vector<float> top_logits; top_logits.reserve(5);
                        for (int k = 0; k < 5 && k < idx_logit_pairs.size(); ++k) {
                            top_ids.push_back(idx_logit_pairs[k].first);
                            top_logits.push_back(idx_logit_pairs[k].second);
                        }
                        std::vector<std::string> top_tokens = tokenizer.ids_to_tokens(top_ids);
                        
                        std::stringstream topk_ss;
                        topk_ss << "[GEN] Step " << generated_count << " (Pos " << pos << ") Top-5 logits: ";
                        for (int k = 0; k < top_ids.size(); ++k) {
                            topk_ss << "(" << top_ids[k] << ": '" << top_tokens[k] << "', " << top_logits[k] << ") ";
                        }
                        Logger::info(topk_ss.str());
                  }

                  // Use Greedy Sampling (argmax)
                  next_token_id = argmax(logits);

             // Decode for logging
             std::string next_token_str = "<INVALID>";
             if (next_token_id >= 0 && static_cast<size_t>(next_token_id) < tokenizer.vocab_size()) {
                 std::vector<std::string> token_vec_str = tokenizer.ids_to_tokens({next_token_id});
                 if (!token_vec_str.empty()) { next_token_str = token_vec_str[0]; }
             }

                  generated_only_ids.push_back(next_token_id); // Store only generated tokens
                  generated_ids.push_back(next_token_id); // Store full sequence

                  Logger::info("Gen Step " + std::to_string(generated_count) + " (Pos " + std::to_string(pos + 1) + "): Generated token ID: " + std::to_string(next_token_id) + " ('" + next_token_str + "')");
                  generated_count++;

                  // Check stopping conditions
             if (next_token_id == eos_token_id) {
                      Logger::info("EOS token generated. Stopping generation.");
                      break;
                  }
                  if (generated_count >= max_new_tokens) {
                      Logger::info("Max new tokens reached. Stopping generation.");
                 break;
                  }
             } else {
                  // Processing prompt token, no sampling needed. Logits are discarded.
                  // Need to set `next_token_id` for the next iteration if this was the last prompt token
                  if (pos == num_prompt_tokens - 2) { 
                      // We just processed the second-to-last prompt token.
                      // The *next* iteration (pos = num_prompt_tokens - 1) will process the *last* prompt token.
                      // The logits from *that* iteration will determine the first generated token.
                      // So, nothing special to do here regarding next_token_id.
                  }
                  Logger::info("Processed prompt token at pos=" + std::to_string(pos));
             }
             Logger::info("--- Unified Loop: END pos=" + std::to_string(pos) + " ---");
        } // End unified generation loop

        Logger::info("Finished generation loop for " + attempt_label + ".");
        
        // Log and Decode Final Output
        std::stringstream final_seq;
        final_seq << "Final generated sequence IDs (" << attempt_label << "): [";
        for(size_t i=0; i < generated_ids.size(); ++i) { // Log the full sequence including prompt
            final_seq << generated_ids[i] << (i + 1 < generated_ids.size() ? ", " : "");
        }
        final_seq << "]";
        Logger::info(final_seq.str());

        Logger::info("Decoding generated sequence (" + attempt_label + ")...");
        // Decode only the *newly generated* tokens, excluding the prompt.
        // (generated_only_ids already contains only the new tokens)
        std::string decoded_output = tokenizer.decode(generated_only_ids, true); // Skip special tokens like EOS
        Logger::info("--- Decoded Output (" + attempt_label + ") ---");
        Logger::info("[" + attempt_label + "] " + decoded_output); // Added log output
        Logger::info("----------------------");

    } catch (const std::exception& e) {
        Logger::error("*** Error during generation experiment (" + attempt_label + ") ***");
        Logger::error(e.what());
    }
    Logger::info("==============================================\n");
}
// --- END: Generation Experiment Function ---

// --- Main Function ---
int main(int argc, char **argv) {
    // Logger::init("gguf_loader.log", Logger::Level::DEBUG); // Correct log file name

    std::string default_prompt_content = "What is 2+2?";
    std::string default_system_message = "You are a helpful assistant.";

    // Configuration for Models
    struct ModelExperimentConfig {
        std::string model_name;
        std::string gguf_path;
    };
    std::vector<ModelExperimentConfig> models_to_test = {
        {"TinyLlama", "data/tiny_llama_q8_requantised.gguf"} // Updated to quantized model
    };

    // Configuration for Special Token Sources - REMOVED (using external JSON)

    // Configuration for Pre-tokenization Methods - REMOVED (using tokenizer's default)

    // Configuration for Prompt Variations
    struct PromptAttempt {
        std::string name;
        std::string prompt;
        bool add_bos = true; 
        bool add_eos = false; 
    };
    std::vector<PromptAttempt> prompt_attempts = {
        // --- ORIGINAL PROMPTS ---
        {"Open Ended 1", "Question: How are you?\nAnswer:", false, false},
        {"Open Ended 2", "Question: What kind of music do you like?\nAnswer:", false, false},
        {"Knowledge 1 (Sky)", "Question: What colour is the sky?\nAnswer:", false, false},
        {"Knowledge 2 (France)", "Question: What is the capital of France?\nAnswer:", false, false},
        {"Knowledge 3 (Hamlet)", "Question: Who wrote Hamlet?\nAnswer:", false, false},
        {"Knowledge 4 (Planet)", "Question: What is the largest planet in our solar system?\nAnswer:", false, false},
        // --- Q/A FORMAT PROMPTS ---
        {"Open Ended 1 (Q/A)", "Q: How are you?\nA:", false, false},
        {"Open Ended 2 (Q/A)", "Q: What kind of music do you like?\nA:", false, false},
        {"Knowledge 1 (Sky, Q/A)", "Q: What colour is the sky?\nA:", false, false},
        {"Knowledge 2 (France, Q/A)", "Q: What is the capital of France?\nA:", false, false},
        {"Knowledge 3 (Hamlet, Q/A)", "Q: Who wrote Hamlet?\nA:", false, false},
        {"Knowledge 4 (Planet, Q/A)", "Q: What is the largest planet in our solar system?\nA:", false, false},
        // --- NO-NEWLINE ORIGINAL PROMPTS ---
        {"Open Ended 1 (no \\n)", "Question: How are you? Answer:", false, false},
        {"Open Ended 2 (no \\n)", "Question: What kind of music do you like? Answer:", false, false},
        {"Knowledge 1 (Sky, no \\n)", "Question: What colour is the sky? Answer:", false, false},
        {"Knowledge 2 (France, no \\n)", "Question: What is the capital of France? Answer:", false, false},
        {"Knowledge 3 (Hamlet, no \\n)", "Question: Who wrote Hamlet? Answer:", false, false},
        {"Knowledge 4 (Planet, no \\n)", "Question: What is the largest planet in our solar system? Answer:", false, false},
        // --- NO-NEWLINE Q/A FORMAT PROMPTS ---
        {"Open Ended 1 (Q/A, no \\n)", "Q: How are you? A:", false, false},
        {"Open Ended 2 (Q/A, no \\n)", "Q: What kind of music do you like? A:", false, false},
        {"Knowledge 1 (Sky, Q/A, no \\n)", "Q: What colour is the sky? A:", false, false},
        {"Knowledge 2 (France, Q/A, no \\n)", "Q: What is the capital of France? A:", false, false},
        {"Knowledge 3 (Hamlet, Q/A, no \\n)", "Q: Who wrote Hamlet? A:", false, false},
        {"Knowledge 4 (Planet, Q/A, no \\n)", "Q: What is the largest planet in our solar system? A:", false, false}
        // --- END NO-NEWLINE Q/A FORMAT PROMPTS ---
    };

    int max_new_tokens = 14;
    float temperature = 0.0f; // Keep greedy sampling

    // --- Restore Experiment Loops --- 
    for (const auto& model_config_item : models_to_test) {
        Logger::info("\n############################################");
        Logger::info("Testing Model: " + model_config_item.model_name);
        Logger::info("############################################");

        // --- REMOVED: Explicit GGUF parsing and config creation ---
        // GGUFData gguf_data; // Load only metadata initially
        // ModelConfig config; // Define config here to be accessible in the outer scope of try
        try {
            // --- Instantiate Tokenizer ---
            std::string tokenizer_path = "data/tiny-llama-pure/tokenizer.json"; // Using presumed main.cpp path
            Tokenizer tokenizer(tokenizer_path, tokenizer_path); // model_path = vocab_path for merges in tokenizer.json
            Logger::info("INITIALIZED TOKENIZER FROM EXTERNAL JSON: " + tokenizer_path);

            // --- GGUF Based Tokenizer (Commented out for now) ---
            // Tokenizer tokenizer(gguf_data); // USE GGUF constructor

            // Construct Model using the parsed config and path
            TinyLlamaModel model(ModelConfig{}, model_config_item.gguf_path);
            Logger::info("Model constructed successfully.");

            // --- Loop through prompt attempts (Removed const from attempt) ---
            for (auto& attempt : prompt_attempts) {
                std::string experiment_label = "[Model:" + model_config_item.model_name +
                                               "_Tokenizer:JSON_Pmt:" + std::to_string(&attempt - &prompt_attempts[0] + 1) + // Calculate index
                                               "(" + attempt.name + ")]";
                Logger::info("------- Prompt Attempt: " + std::to_string(&attempt - &prompt_attempts[0] + 1) + " (" + attempt.name + ") -------");

                try {
                    // Modify the prompt string (now allowed)
                    std::string current_prompt = attempt.prompt; // Work with a copy if original shouldn't change globally?
                                                              // Or keep modifying 'attempt.prompt' if that's intended for this loop structure.
                    if (attempt.add_bos) {
                        // Note: Modifying attempt.prompt directly here
                        attempt.prompt = "<s>" + attempt.prompt; // Assuming <s> is the BOS token string 
                    }
                    if (attempt.add_eos) {
                        // Note: Modifying attempt.prompt directly here
                        attempt.prompt += "</s>"; // Assuming </s> is the EOS token string
                    }
                    Logger::info("Using Processed Prompt: '" + attempt.prompt + "'");
                    
                    // --- MODIFIED: Tokenize using tokenize() + tokens_to_ids() ---
                    std::vector<std::string> prompt_tokens_str = tokenizer.tokenize(attempt.prompt);
                    std::vector<int> prompt_ids = tokenizer.tokens_to_ids(prompt_tokens_str);
                    Logger::info("Tokenized prompt using tokenize() + tokens_to_ids(). Count: " + std::to_string(prompt_ids.size()));
                    // --- END MODIFICATION ---

                    // Use model.get_config() for config
                    run_generation_experiment(model, tokenizer, prompt_ids, max_new_tokens, temperature, model.get_config(), experiment_label);

                    // Optional: Restore original prompt if modifications should not persist across model tests?
                    // attempt.prompt = current_prompt; // Uncomment if needed

                } catch (const std::exception& e) {
                    Logger::error("Error during inner experiment " + experiment_label + ": " + e.what());
                }

            } // End prompt attempt loop

        } catch (const std::exception& e) {
            // Catch errors during model construction or config retrieval
             Logger::error("Failed to construct model or run experiments for " + model_config_item.model_name + ": " + e.what());
             continue; // Skip to the next model if construction fails
        }
    } // End model loop

    Logger::info("\n===== All experiments completed. =====\n");
    return 0;
} 