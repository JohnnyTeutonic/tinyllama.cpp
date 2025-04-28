#include "api.h"

// Include all necessary implementation headers
#include "model.h"
#include "tokenizer.h"
#include "safetensors_loader.h"
#include "logger.h"
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <filesystem> // For path manipulation (C++17)
#include <random>     // For std::mt19937
#include <vector>
#include <string>
#include <memory>
#include <fstream>    // For std::ifstream
#include <map>        // For json library
#include <cstdint>    // For json library
#include <iostream>   // For sampling debugging (optional)
#include <numeric>    // For std::iota, std::accumulate
#include <algorithm>  // For std::sort, std::partial_sort, std::max_element
#include <cmath>      // For std::exp, std::sqrt
#include <functional> // For std::function, std::greater

namespace tinyllama {

// Utility: Read file into string (copied from main.cpp)
static std::string read_file_api(const std::string& path) {
    // Use std::filesystem::path for better path handling
    std::filesystem::path fs_path(path);
    std::ifstream file(fs_path, std::ios::binary);
    if (!file) throw std::runtime_error("Failed to open file: " + path);
    return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

// --- START: Argmax Helper --- (Copied from model.cpp)
// Find the index of the maximum element in a vector
// Made static as it's only used within this file for now.
static int argmax(const std::vector<float>& v) {
    if (v.empty()) {
        Logger::error("Cannot perform argmax on empty vector"); 
        return -1; // Return an invalid index
    }
    // Need <algorithm> for std::max_element and <iterator> for std::distance (already included indirectly)
    return std::distance(v.begin(), std::max_element(v.begin(), v.end()));
}
// --- END: Argmax Helper ---

// Sampling function (copied and adapted from main.cpp)
static int sample_top_k_top_p_temperature(
    const std::vector<float>& logits, 
    float temperature, 
    int top_k, 
    float top_p, 
    std::mt19937& rng) 
{
    if (logits.empty()) {
        throw std::runtime_error("Cannot sample from empty logits.");
    }

    int vocab_size = logits.size();

    // Clamp top_k to vocab_size
    top_k = std::min(top_k, vocab_size);
    if (top_k <= 0) top_k = vocab_size; // If top_k is non-positive, consider all tokens

    // Apply temperature
    std::vector<float> scaled_logits(vocab_size);
    float max_logit = -std::numeric_limits<float>::infinity();
    for(float logit : logits) max_logit = std::max(max_logit, logit);
    
    for (int i = 0; i < vocab_size; ++i) {
        scaled_logits[i] = (logits[i] - max_logit) / std::max(temperature, 1e-6f); // Avoid division by zero
    }

    // Compute softmax probabilities
    std::vector<double> probs_double(vocab_size);
    double sum_exp = 0.0;
    for (int i = 0; i < vocab_size; ++i) {
        probs_double[i] = std::exp(static_cast<double>(scaled_logits[i]));
        sum_exp += probs_double[i];
    }
    // Normalize probabilities
    for (int i = 0; i < vocab_size; ++i) {
        probs_double[i] /= sum_exp;
    }

    // Convert to float probabilities and pair with indices
    std::vector<std::pair<float, int>> prob_idx(vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
        prob_idx[i] = {static_cast<float>(probs_double[i]), i};
    }

    // Sort by probability (descending)
    std::sort(prob_idx.begin(), prob_idx.end(), std::greater<std::pair<float, int>>());

    // Apply Top-K filtering
    if (top_k < vocab_size) {
        prob_idx.resize(top_k);
    }

    // Apply Top-P (Nucleus) filtering
    float cumulative_prob = 0.0f;
    int last_idx = 0; // Index of the last element to keep
    for (int i = 0; i < prob_idx.size(); ++i) {
        cumulative_prob += prob_idx[i].first;
        last_idx = i;
        if (cumulative_prob >= top_p) {
            break; // Stop when cumulative probability threshold is reached
        }
    }
    prob_idx.resize(last_idx + 1); // Keep only the nucleus

    // Normalize the final probabilities
    float final_sum = 0.0f;
    for (const auto& pi : prob_idx) {
        final_sum += pi.first;
    }
    std::vector<float> final_probs(prob_idx.size());
    for (size_t i = 0; i < prob_idx.size(); ++i) {
        final_probs[i] = prob_idx[i].first / std::max(final_sum, 1e-6f);
    }

    // Sample from the final distribution
    std::discrete_distribution<int> dist(final_probs.begin(), final_probs.end());
    int sampled_idx_in_filtered = dist(rng);

    // Return the original token ID
    return prob_idx[sampled_idx_in_filtered].second;
}

// Define the implementation struct (PImpl)
struct TinyLlamaSession::SessionImpl {
    ModelConfig config;
    std::unique_ptr<Tokenizer> tokenizer;
    std::unique_ptr<TinyLlamaModel> model;
    std::unique_ptr<KVCache> kv_cache;
    std::mt19937 rng; // Random number generator for sampling

    SessionImpl() : rng(std::random_device{}()) {} // Seed RNG
};

// --- TinyLlamaSession Constructor --- 
TinyLlamaSession::TinyLlamaSession(const std::string& model_dir) 
    : pimpl_(std::make_unique<SessionImpl>()) 
{
    Logger::info("TinyLlamaSession: Initializing...");
    std::filesystem::path dir_path(model_dir);

    if (!std::filesystem::exists(dir_path) || !std::filesystem::is_directory(dir_path)) {
        throw std::runtime_error("Model directory does not exist or is not a directory: " + model_dir);
    }

    // Construct paths
    std::string config_path = (dir_path / "config.json").string();
    std::string tokenizer_path = (dir_path / "tokenizer.json").string();
    std::string safetensors_path = (dir_path / "model.safetensors").string();

    // 1. Load Config
    Logger::info("Loading config: " + config_path);
    nlohmann::json config_json;
    try {
        std::string config_str = read_file_api(config_path);
        config_json = nlohmann::json::parse(config_str);
        pimpl_->config = parse_model_config(config_json); // Use helper from model.h/cpp
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Error loading/parsing config.json: ") + e.what());
    }

    // 2. Load Tokenizer
    Logger::info("Loading tokenizer: " + tokenizer_path);
    try {
        pimpl_->tokenizer = std::make_unique<Tokenizer>(tokenizer_path, tokenizer_path); // Use tokenizer path also for vocab path
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Error loading tokenizer: ") + e.what());
    }

    // 3. Load Model Weights
    Logger::info("Loading model weights: " + safetensors_path);
    try {
        SafeTensorsLoader st_loader(safetensors_path);
        pimpl_->model = std::make_unique<TinyLlamaModel>(pimpl_->config, st_loader);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Error loading model weights: ") + e.what());
    }

    // 4. Initialize KVCache
    Logger::info("Initializing KVCache...");
    try {
        pimpl_->kv_cache = std::make_unique<KVCache>();
        int nhl = pimpl_->config.num_hidden_layers;
        int n_kv_heads = pimpl_->config.num_key_value_heads;
        int max_seq_len = pimpl_->config.max_position_embeddings;
        int head_dim = pimpl_->config.hidden_size / pimpl_->config.num_attention_heads;
        pimpl_->kv_cache->initialize(nhl, max_seq_len, n_kv_heads, head_dim);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to initialize KVCache: ") + e.what());
    }

    Logger::info("TinyLlamaSession: Initialization complete.");
}

// --- TinyLlamaSession Destructor --- 
TinyLlamaSession::~TinyLlamaSession() {
    // unique_ptr handles deletion of model, tokenizer, cache (including CUDA freeing via KVCache destructor)
    Logger::info("TinyLlamaSession: Destroyed.");
}

// --- TinyLlamaSession Generate Method --- 
std::string TinyLlamaSession::generate(
    const std::string& prompt, 
    int max_new_tokens, 
    float temperature, // Parameter remains but won't be used if using argmax
    int top_k,         // Parameter remains but won't be used if using argmax
    float top_p)       // Parameter remains but won't be used if using argmax
{
    Logger::info("Generate called. Max new tokens: " + std::to_string(max_new_tokens));
    if (!pimpl_ || !pimpl_->model || !pimpl_->tokenizer || !pimpl_->kv_cache) {
        throw std::runtime_error("TinyLlamaSession is not properly initialized.");
    }

    // Get references for convenience
    TinyLlamaModel& model = *pimpl_->model;
    const Tokenizer& tokenizer = *pimpl_->tokenizer;
    KVCache& cache = *pimpl_->kv_cache;
    const ModelConfig& mcfg = pimpl_->config;
    std::mt19937& rng = pimpl_->rng;

    // Reset KVCache sequence length for new generation
    cache.seq_len = 0;

    // Tokenize prompt
    std::vector<int> prompt_ids = tokenizer.encode(prompt, false); // Don't add special tokens here
    Logger::info("Prompt tokenized. Num tokens: " + std::to_string(prompt_ids.size()));

    // Get EOS token ID
    int eos_id = tokenizer.eos_token_id();
    if (eos_id < 0) eos_id = 2; // Fallback

    // --- Generation Loop --- 
    int next_token_id = -1;
    int num_prompt_tokens = prompt_ids.size();
    std::vector<int> generated_only_ids;
    int generated_count = 0;
    int total_steps = num_prompt_tokens + max_new_tokens - 1; // Max steps allowed

    for (int pos = 0; pos < total_steps; ++pos) {
        Logger::info("--- Generate loop: START pos=" + std::to_string(pos) + " ---");
        if (pos >= mcfg.max_position_embeddings) {
            Logger::info("Reached max sequence length (" + std::to_string(mcfg.max_position_embeddings) + "). Stopping.");
            break;
        }

        // Determine input token
        int input_token_id = (pos < num_prompt_tokens) ? prompt_ids[pos] : next_token_id;
        Logger::info("Input token ID: " + std::to_string(input_token_id));

        // Select forward pass
        std::vector<float> logits;
#ifdef HAS_CUDA
        Logger::info("Calling forward_device...");
        logits = model.forward_device(input_token_id, pos, &cache, nullptr);
        Logger::info("Returned from forward_device.");
#else
        Logger::info("Calling forward (CPU)...");
        std::vector<float> current_x = model.lookup_embedding(input_token_id);
        logits = model.forward(current_x, pos, &cache, nullptr);
        Logger::info("Returned from forward (CPU).");
#endif

        // Check for errors from forward pass
        if (logits.empty()) {
            Logger::error("Forward pass returned empty logits at pos " + std::to_string(pos) + ". Stopping generation.");
            break; 
        }
        
        // --- IMPORTANT: Increment cache sequence length *AFTER* the forward call for position `pos` --- 
        cache.seq_len = pos + 1; 

        // Sample next token (only during generation phase)
        if (pos >= num_prompt_tokens - 1) {
            // Use greedy sampling (argmax)
            next_token_id = argmax(logits);
            // next_token_id = sample_top_k_top_p_temperature(logits, temperature, top_k, top_p, rng);
            Logger::info("Greedy sampled token ID: " + std::to_string(next_token_id));

            generated_only_ids.push_back(next_token_id);
            generated_count++;

            // Check for EOS
            if (next_token_id == eos_id || generated_count >= max_new_tokens) {
                if (next_token_id == eos_id) Logger::info("EOS token generated.");
                else Logger::info("Max new tokens reached.");
                Logger::info("Stopping generation.");
                break; 
            }
        } else {
            // Processing prompt, no sampling needed
             Logger::info("Processed prompt token at pos " + std::to_string(pos));
        }
        Logger::info("--- Generate loop: END pos=" + std::to_string(pos) + " ---");
    } // End generation loop

    // Decode the generated IDs
    std::string generated_text = tokenizer.decode(generated_only_ids, true); // Skip special tokens
    Logger::info("Decoding complete. Generated text length: " + std::to_string(generated_text.length()));
    return generated_text;
}

} // namespace tinyllama 