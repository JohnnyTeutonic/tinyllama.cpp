#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <random>     // ADDED for random number generation
#include <numeric>    // ADDED for std::partial_sum
#include <limits>     // ADDED for std::numeric_limits
#include <algorithm>  // ADDED for std::transform, std::max_element
#include <cmath>      // ADDED for std::exp, std::log
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
                               const ModelConfig& config,
                               const std::string& prompt,
                               bool use_regex_pretokenize,
                               int max_new_tokens,
                               float temperature)
{
    std::string mode_str = use_regex_pretokenize ? "REGEX" : "WHITESPACE";
    Logger::info("\n==============================================");
    Logger::info("Starting Generation Experiment with Pre-Tokenization: " + mode_str);
    Logger::info("==============================================");

    try {
        // 1. Format and Tokenize Prompt (using the specified method)
        // Logger::info("User prompt (raw): " + prompt); // Reverted: Log raw prompt if needed
        // --- RESTORE CHAT TEMPLATE --- 
        std::string formatted_prompt = tokenizer.apply_chat_template(prompt); // Use the template again
        Logger::info("Formatted prompt: " + formatted_prompt); // Log the result of the template
        // --- END RESTORE ---
        
        // --- REVERT TO encode (handles template output) ---
        // std::vector<std::string> prompt_tokens_str = tokenizer.tokenize(prompt); // Tokenize the raw prompt
        // std::vector<int> prompt_tokens = tokenizer.tokens_to_ids(prompt_tokens_str); // Convert strings to IDs
        std::vector<int> prompt_tokens = tokenizer.encode(formatted_prompt, true, false); // Encode the *formatted* prompt, add BOS, use whitespace pre-tok
        // --- END REVERT ---
        
        // Log tokenization result for this mode
        std::stringstream ptk_ss;
        ptk_ss << "Prompt token IDs (" << mode_str << "): [";
        for(size_t i=0; i < prompt_tokens.size(); ++i) {
            ptk_ss << prompt_tokens[i] << (i + 1 < prompt_tokens.size() ? ", " : "");
        }
        ptk_ss << "]";
        Logger::info(ptk_ss.str());
        // +++ ADDED: Log full token count +++
        Logger::info("Full prompt token count: " + std::to_string(prompt_tokens.size()));
        // +++ ADDED: Log decoded prompt tokens for verification +++
        if (!prompt_tokens.empty()) {
            std::vector<std::string> decoded_prompt_strs = tokenizer.ids_to_tokens(prompt_tokens);
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
        // --- END ADDED ---

        // --- Generation Setup ---
        int bos_token_id = config.bos_token_id; // Assuming bos is added by encode
        int eos_token_id = config.eos_token_id;
        // Start generation sequence with the prompt tokens.
        std::vector<int> generated_tokens = prompt_tokens; 

        Logger::info("Using BOS token ID: " + std::to_string(bos_token_id)); // Log the expected BOS
        Logger::info("Using EOS token ID: " + std::to_string(eos_token_id));
        Logger::info("Max new tokens: " + std::to_string(max_new_tokens));

        // Initialize KVCache (needs to be fresh for each run)
        KVCache kv_cache;
        kv_cache.initialize(config.num_hidden_layers, config.max_position_embeddings,
                            config.num_key_value_heads, config.hidden_size / config.num_attention_heads);
        Logger::info("KVCache initialized for this run.");

        Logger::info("Using temperature: " + std::to_string(temperature));
        std::vector<float> probabilities(config.vocab_size);

        // --- ADDED: Prompt Processing Loop ---
        Logger::info("Processing prompt tokens to fill KVCache...");
        if (!prompt_tokens.empty()) {
            // --- MODIFICATION: Skip BOS if present ---
            size_t start_index = 0;
            if (prompt_tokens[0] == bos_token_id) {
                Logger::info("Detected BOS token ID " + std::to_string(bos_token_id) + " at the start of prompt tokens. Skipping explicit processing.");
                start_index = 1; // Start from the second token
            } else {
                 Logger::info("BOS token not detected at the start, processing all prompt tokens.");
            }
            // --- END MODIFICATION ---

            for (size_t i = start_index; i < prompt_tokens.size(); ++i) {
                int current_token_id = prompt_tokens[i];
                // --- MODIFICATION: Adjust position if BOS was skipped ---
                int current_pos = static_cast<int>(i - start_index); // Position relative to the *actual* start of content
                // --- END MODIFICATION ---

                // Log only for the very first *processed* token (pos=0 relative to content)
                bool log_this_prompt_step = (current_pos == 0);

                std::vector<float> input_embedding = model.lookup_embedding(current_token_id);
                if (input_embedding.empty() || input_embedding.size() != config.hidden_size) {
                    throw std::runtime_error("Prompt Processing: Failed to get valid embedding for token ID " + std::to_string(current_token_id) + " at effective pos " + std::to_string(current_pos));
                }
                if (log_this_prompt_step) {
                    log_vector_summary("Prompt Input Embedding (Effective pos=0)", input_embedding, 10);
                }

                // Call forward to fill KVCache, ignore logits
                // NOTE: Pass the input_embedding vector by value to avoid modification if forward modifies it
                std::vector<float> ignored_logits = model.forward(input_embedding, current_pos, &kv_cache, nullptr);
                if (log_this_prompt_step) {
                    // Optionally log something after the forward call for pos=0
                    Logger::info("Completed model.forward() for prompt token at effective pos=0.");
                }
            }
             // Log the number of *effectively* processed tokens
             Logger::info("Finished processing " + std::to_string(prompt_tokens.size() - start_index) + " effective prompt tokens.");
        } else {
            Logger::info("Prompt is empty, skipping KVCache prefill.");
        }
        // --- END: Prompt Processing Loop ---

        // --- Generation Loop (Starts predicting token AFTER the last prompt token) ---
        // The position for the *first generated token* will be the *effective* length of the prompt (excluding BOS).
        int effective_prompt_len = prompt_tokens.empty() ? 0 : (prompt_tokens.size() - (prompt_tokens[0] == bos_token_id ? 1 : 0));
        int first_gen_pos = effective_prompt_len; 
        // The position passed to forward() in the first iteration should be the position of the last *processed* prompt token.
        int current_loop_pos = first_gen_pos - 1; 

        // Get the ID of the *last* token from the prompt (or BOS if prompt was empty)
        // int next_token_id_input = prompt_tokens.empty() ? bos_token_id : prompt_tokens.back(); // Original: Use last token
        // --- MODIFICATION: Use second-to-last token if prompt isn't tiny --- 
        int next_token_id_input;
        if (prompt_tokens.size() <= 1) { // Handle empty or single-token (BOS) prompt
             next_token_id_input = bos_token_id;
        } else {
             next_token_id_input = prompt_tokens[prompt_tokens.size() - 2]; // Use the token BEFORE the final space
             Logger::info("Using second-to-last prompt token (" + std::to_string(next_token_id_input) + ") as input for first prediction instead of last token (" + std::to_string(prompt_tokens.back()) + ")");
        }
        // --- END MODIFICATION ---
        Logger::info("Starting generation loop. Input token for first prediction: " + std::to_string(next_token_id_input) + ", Starting pos for forward(): " + std::to_string(current_loop_pos) );
        
        for (int step = 0; step < max_new_tokens; ++step) {
            int current_token_id = next_token_id_input; // Use the input token for this step

            // +++ START ADDED LOGGING +++
            Logger::info("[GGUF GEN LOOP] step=" + std::to_string(step) + ", current_token_id=" + std::to_string(current_token_id) + ", current_loop_pos=" + std::to_string(current_loop_pos));
            // +++ END ADDED LOGGING +++

            std::vector<float> input_embedding = model.lookup_embedding(current_token_id);
            if (input_embedding.empty() || input_embedding.size() != config.hidden_size) {
                throw std::runtime_error("Failed to get valid embedding for token ID " + std::to_string(current_token_id) + " at step " + std::to_string(step));
            }
             if (step == 0) { // Only log the first step's embedding
                 log_vector_summary("Input Embedding (step=0, pos=" + std::to_string(current_loop_pos) + ")", input_embedding, 10);
             }

            // Pass the correct position to the forward method
            std::vector<float> logits = model.forward(input_embedding, current_loop_pos, &kv_cache, nullptr);
            if (logits.empty() || logits.size() != config.vocab_size) {
                throw std::runtime_error("model.forward() returned invalid logits at step " + std::to_string(step) + ". Size: " + std::to_string(logits.size()) + ", Expected: " + std::to_string(config.vocab_size));
            }
             if (step == 0) { // Only log the first step's logits
                 log_vector_summary("Output Logits (step=0, pos=" + std::to_string(current_loop_pos) + ")", logits, 10);
             }

             // --- MODIFIED: Use Greedy Sampling (argmax) --- 
             // Temperature Scaling - REMOVED
             // if (temperature > 0.0f) {
             //    for(float& logit : logits) { logit /= temperature; }
             // }
             // Softmax - REMOVED
             // softmax_vector_cpu(logits, probabilities);
             // Multinomial Sampling - REMOVED
             // int next_token_id = sample_multinomial(probabilities);
             // Greedy Sampling (argmax) - ADDED
             int next_token_id = argmax(logits);
             // +++ START ADDED LOGGING +++
             Logger::debug("[GGUF GEN LOOP] argmax returned token ID: " + std::to_string(next_token_id));
             // +++ END ADDED LOGGING +++
             // --- END MODIFICATION ---

             // Decode for logging
             std::string next_token_str = "<INVALID>";
             if (next_token_id >= 0 && static_cast<size_t>(next_token_id) < tokenizer.vocab_size()) {
                 std::vector<std::string> token_vec_str = tokenizer.ids_to_tokens({next_token_id});
                 if (!token_vec_str.empty()) { next_token_str = token_vec_str[0]; }
             }

             // Prepare for the *next* iteration
             next_token_id_input = next_token_id; // The generated token becomes the next input
             current_loop_pos++; // Increment position *after* using it for the current step

             generated_tokens.push_back(next_token_id);
             Logger::info("Gen Step " + std::to_string(step) + " (Pos " + std::to_string(current_loop_pos + 1) + "): Generated token ID: " + std::to_string(next_token_id) + " ('" + next_token_str + "')");

             if (next_token_id == eos_token_id) {
                 Logger::info("EOS token generated. Stopping generation for this run.");
                 break;
             }
        }

        Logger::info("Finished generation loop for " + mode_str + " mode.");
        
        // Log and Decode Final Output
        std::stringstream final_seq;
        final_seq << "Final generated sequence IDs (" << mode_str << "): [";
        for(size_t i=0; i < generated_tokens.size(); ++i) {
            final_seq << generated_tokens[i] << (i + 1 < generated_tokens.size() ? ", " : "");
        }
        final_seq << "]";
        Logger::info(final_seq.str());

        Logger::info("Decoding generated sequence (" + mode_str + ")...");
        // Decode only the *newly generated* tokens, excluding the prompt.
        std::vector<int> tokens_to_decode;
        if (generated_tokens.size() > prompt_tokens.size()) {
            tokens_to_decode.assign(generated_tokens.begin() + prompt_tokens.size(), generated_tokens.end());
        }
        // Don't need to check for BOS here as we are skipping the prompt where BOS would be.
        
        std::string decoded_output = tokenizer.decode(tokens_to_decode, true); // Skip special tokens like EOS
        Logger::info("--- Decoded Output (" + mode_str + ") ---");
        std::cout << "[" << mode_str << "] " << decoded_output << std::endl;
        Logger::info("----------------------");

    } catch (const std::exception& e) {
        Logger::error("*** Error during generation experiment (" + mode_str + ") ***");
        Logger::error(e.what());
    }
    Logger::info("==============================================\n");
}
// --- END: Generation Experiment Function ---

int main(int argc, char **argv) {
    Logger::info("GGUF Loader - Starting...");

    // TODO: Add file path argument parsing (e.g., from argv)
    std::string filename = "data/TinyLlama-1.1B-Chat-v1.0.FP16.gguf"; // Hardcoded for now

    try {
        // --- Load Model ---
        Logger::info("Loading GGUF model and tokenizer...");
        ModelConfig config; // Config will be loaded from GGUF
        TinyLlamaModel model(config, filename); // Constructor handles loading
        config = model.get_config(); // Update config with values read from GGUF
        std::stringstream ss;
        ss << "--- Model Summary ---\n";
        ss << "Hidden size: " << config.hidden_size << "\n";
        ss << "Intermediate size: " << config.intermediate_size << "\n";
        ss << "Num attention heads: " << config.num_attention_heads << "\n";
        ss << "Num key/value heads: " << config.num_key_value_heads << "\n";
        ss << "Num hidden layers: " << config.num_hidden_layers << "\n";
        ss << "Vocab size: " << config.vocab_size << "\n";
        ss << "Max position embeddings: " << config.max_position_embeddings << "\n";
        ss << "---------------------\n";
        ss << "Model loaded successfully from: " << filename;
        Logger::info(ss.str());

        // For debugging: log GGUFData tensor info and metadata from the model
        if (const GGUFData* gguf_data = model.get_gguf_data()) {
            Logger::info("--- GGUF Tensor Info ---");
            for (const auto& t : gguf_data->tensor_infos) {
                std::stringstream ts;
                ts << "Tensor: '" << t.name << "', Type=" << t.type
                   << ", Shape=[";
                for (size_t i = 0; i < t.shape.size(); ++i) ts << t.shape[i] << (i + 1 < t.shape.size() ? ", " : "");
                ts << "], Offset=" << t.offset << ", Size=" << t.size_in_bytes << " bytes";
                Logger::info(ts.str());
            }
            Logger::info("------------------------");
            // Optionally log some metadata
            Logger::info("--- GGUF Metadata ---");
            for (const auto& kv : gguf_data->metadata) {
                std::stringstream ms;
                ms << kv.first << ": ";
                // Print the value as string if possible
                if (std::holds_alternative<std::string>(kv.second)) {
                    ms << std::get<std::string>(kv.second);
                } else if (std::holds_alternative<int32_t>(kv.second)) {
                    ms << std::get<int32_t>(kv.second);
                } else if (std::holds_alternative<uint32_t>(kv.second)) {
                    ms << std::get<uint32_t>(kv.second);
                } else if (std::holds_alternative<int64_t>(kv.second)) {
                    ms << std::get<int64_t>(kv.second);
                } else if (std::holds_alternative<uint64_t>(kv.second)) {
                    ms << std::get<uint64_t>(kv.second);
                } else if (std::holds_alternative<float>(kv.second)) {
                    ms << std::get<float>(kv.second);
                } else if (std::holds_alternative<double>(kv.second)) {
                    ms << std::get<double>(kv.second);
                } else if (std::holds_alternative<bool>(kv.second)) {
                    ms << (std::get<bool>(kv.second) ? "true" : "false");
                } else {
                    ms << "(unhandled type)";
                }
                Logger::info(ms.str());
            }
            Logger::info("---------------------");
        }

        // --- START: CPU Generation Loop -> Now Experiment --- 
        Logger::info("Initializing Tokenizer and running experiments...");
        try {
             // 1. Get GGUF Data from model (check if it exists)
             // const GGUFData* gguf_data = model.get_gguf_data(); // NO LONGER NEEDED FOR TOKENIZER
             // if (!gguf_data) {
             //    throw std::runtime_error("Failed to get GGUF data from model for tokenizer initialization.");
             // }

            // 2. Initialize Tokenizer using the external JSON file
            std::string tokenizer_json_path = "data/tokenizer.json";
            // Tokenizer tokenizer("", tokenizer_json_path); // <<< OLD WAY - Incorrectly ignores merges
            Tokenizer tokenizer(tokenizer_json_path, tokenizer_json_path); // <<< NEW WAY - Pass JSON path for both model and vocab
            Logger::info("Tokenizer initialized from JSON: " + tokenizer_json_path + ". Vocab size: " + std::to_string(tokenizer.vocab_size()));

            // 3. Define Prompt and parameters
            std::string prompt = "What is the capital of France?";
            int max_new_tokens = 50; 
            float temperature = 0.8f;
            
            // --- Run experiment with REGEX pre-tokenization ---
            // run_generation_experiment(model, tokenizer, config, prompt, true, max_new_tokens, temperature); // <<<< COMMENT THIS OUT
            
            // --- Run experiment with WHITESPACE pre-tokenization ---
            run_generation_experiment(model, tokenizer, config, prompt, false, max_new_tokens, temperature);

        } catch (const std::exception& e) {
             Logger::error("*** Error during Tokenizer init or Experiment setup ***");
             Logger::error(e.what());
        }
        // --- END: CPU Generation Experiment ---

    } catch (const std::exception& e) {
        Logger::error(std::string("*** Error loading model ***\n") + e.what());
        return 1;
    }

    Logger::info("GGUF Loader - Finished successfully.");
    return 0;
} 