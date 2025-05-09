#include "api.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "logger.h"
#include "model.h"
#include "safetensors_loader.h"
#include "tokenizer.h"

namespace tinyllama {

static std::string read_file_api(const std::string& path) {
  std::filesystem::path fs_path(path);
  std::ifstream file(fs_path, std::ios::binary);
  if (!file) throw std::runtime_error("Failed to open file: " + path);
  return std::string((std::istreambuf_iterator<char>(file)),
                     std::istreambuf_iterator<char>());
}

static int argmax(const std::vector<float>& v) {
  if (v.empty()) {
    Logger::error("Cannot perform argmax on empty vector");
    return -1;
  }

  return std::distance(v.begin(), std::max_element(v.begin(), v.end()));
}

static int sample_top_k_top_p_temperature(const std::vector<float>& logits,
                                          float temperature, int top_k,
                                          float top_p, std::mt19937& rng) {
  if (logits.empty()) {
    throw std::runtime_error("Cannot sample from empty logits.");
  }

  int vocab_size = logits.size();

  top_k = std::min(top_k, vocab_size);
  if (top_k <= 0) top_k = vocab_size;

  std::vector<float> scaled_logits(vocab_size);
  float max_logit = -std::numeric_limits<float>::infinity();
  for (float logit : logits) max_logit = std::max(max_logit, logit);

  for (int i = 0; i < vocab_size; ++i) {
    scaled_logits[i] = (logits[i] - max_logit) / std::max(temperature, 1e-6f);
  }

  std::vector<double> probs_double(vocab_size);
  double sum_exp = 0.0;
  for (int i = 0; i < vocab_size; ++i) {
    probs_double[i] = std::exp(static_cast<double>(scaled_logits[i]));
    sum_exp += probs_double[i];
  }

  for (int i = 0; i < vocab_size; ++i) {
    probs_double[i] /= sum_exp;
  }

  std::vector<std::pair<float, int>> prob_idx(vocab_size);
  for (int i = 0; i < vocab_size; ++i) {
    prob_idx[i] = {static_cast<float>(probs_double[i]), i};
  }

  std::sort(prob_idx.begin(), prob_idx.end(),
            std::greater<std::pair<float, int>>());

  if (top_k < vocab_size) {
    prob_idx.resize(top_k);
  }

  float cumulative_prob = 0.0f;
  int last_idx = 0;
  for (int i = 0; i < prob_idx.size(); ++i) {
    cumulative_prob += prob_idx[i].first;
    last_idx = i;
    if (cumulative_prob >= top_p) {
      break;
    }
  }
  prob_idx.resize(last_idx + 1);

  float final_sum = 0.0f;
  for (const auto& pi : prob_idx) {
    final_sum += pi.first;
  }
  std::vector<float> final_probs(prob_idx.size());
  for (size_t i = 0; i < prob_idx.size(); ++i) {
    final_probs[i] = prob_idx[i].first / std::max(final_sum, 1e-6f);
  }

  std::discrete_distribution<int> dist(final_probs.begin(), final_probs.end());
  int sampled_idx_in_filtered = dist(rng);

  return prob_idx[sampled_idx_in_filtered].second;
}

TinyLlamaSession::TinyLlamaSession(const std::string& model_path) {
  Logger::info("TinyLlamaSession: Initializing with path: " + model_path);

  std::filesystem::path path_obj(model_path);
  std::filesystem::path base_dir;

  std::string tokenizer_path_str;
  if (std::filesystem::is_directory(path_obj)) {
    base_dir = path_obj;
    tokenizer_path_str = (base_dir / "tokenizer.json").string();
  } else if (std::filesystem::is_regular_file(path_obj) &&
             path_obj.extension() == ".gguf") {
    base_dir = path_obj.parent_path();
    tokenizer_path_str = (base_dir / "tokenizer.json").string();
  } else {
    throw std::runtime_error(
        "Invalid model_path: Must be a directory or a .gguf file. Path: " +
        model_path);
  }
  Logger::info("Base directory: " + base_dir.string());
  Logger::info("Attempting tokenizer: " + tokenizer_path_str);

  try {
    tokenizer_ =
        std::make_unique<Tokenizer>(tokenizer_path_str, tokenizer_path_str);
    Logger::info("Tokenizer loaded successfully.");
  } catch (const std::exception& e) {
    throw std::runtime_error("Failed to load tokenizer from " +
                             tokenizer_path_str + ": " + e.what());
  }

  if (std::filesystem::is_directory(path_obj)) {
    Logger::info("Loading SafeTensors model from directory: " + model_path);
    std::string config_path_str = (base_dir / "config.json").string();
    std::string safetensors_path_str =
        (base_dir / "model.safetensors").string();

    if (!std::filesystem::exists(config_path_str)) {
      throw std::runtime_error("config.json not found in directory: " +
                               model_path);
    }
    if (!std::filesystem::exists(safetensors_path_str)) {
      throw std::runtime_error("model.safetensors not found in directory: " +
                               model_path);
    }

    try {
      Logger::info("Loading config: " + config_path_str);
      nlohmann::json config_json =
          nlohmann::json::parse(read_file_api(config_path_str));
      ModelConfig loaded_config = parse_model_config(config_json);

      Logger::info("Loading model weights: " + safetensors_path_str);
      SafeTensorsLoader st_loader(safetensors_path_str);

      model_ = std::make_unique<TinyLlamaModel>(loaded_config, st_loader);
      Logger::info("SafeTensors Model loaded successfully.");

    } catch (const std::exception& e) {
      throw std::runtime_error(
          "Failed during SafeTensors model loading from directory " +
          model_path + ": " + e.what());
    }

  } else {
    Logger::info("Loading GGUF model from file: " + model_path);
    try {
      model_ = std::make_unique<TinyLlamaModel>(ModelConfig{}, model_path);
      Logger::info("GGUF Model loaded successfully.");
    } catch (const std::exception& e) {
      throw std::runtime_error("Failed to load GGUF model from " + model_path +
                               ": " + e.what());
    }
  }

  config_ = model_->get_config();

  eos_token_id_ = config_.eos_token_id;
  int head_dim = config_.hidden_size / config_.num_attention_heads;

  Logger::info("Initializing KVCache with max_pos_emb: " +
               std::to_string(config_.max_position_embeddings));
  kv_cache_.initialize(config_.num_hidden_layers,
                       config_.max_position_embeddings,
                       config_.num_key_value_heads, head_dim);
  Logger::info("TinyLlamaSession initialization complete.");
}

TinyLlamaSession::~TinyLlamaSession() {
  Logger::info("TinyLlamaSession: Destroyed.");
}

std::string TinyLlamaSession::generate(const std::string& prompt_input,
                                       int steps, float temperature,
                                       const std::string& system_prompt,
                                       bool apply_q_a_format) {
  Logger::info("Generate called. Initial Prompt: \"" + prompt_input +
               "\", Steps: " + std::to_string(steps) +
               ", Apply Q&A Format: " + (apply_q_a_format ? "true" : "false"));

  std::string final_prompt_for_tokenization;
  if (apply_q_a_format) {
    final_prompt_for_tokenization = "Q: " + prompt_input + "\nA:";
    Logger::info("Applied Q:A: format. Prompt for tokenization: \"" +
                 final_prompt_for_tokenization + "\"");
  } else {
    final_prompt_for_tokenization = prompt_input;
    Logger::info("Using provided prompt as-is for tokenization: \"" +
                 final_prompt_for_tokenization + "\"");
  }

  std::vector<std::string> token_strs =
      tokenizer_->tokenize(final_prompt_for_tokenization);
  std::vector<int> token_ids = tokenizer_->tokens_to_ids(token_strs);

  if (token_ids.empty()) {
    Logger::warning("Tokenization resulted in empty ID list for prompt: " +
                    final_prompt_for_tokenization);
    return "";
  }

  int num_prompt_tokens = token_ids.size();
  int total_steps = num_prompt_tokens + steps - 1;
  int generated_count = 0;
  int next_token_id = -1;

  kv_cache_.seq_len = 0;

  for (int pos = 0; pos < total_steps; ++pos) {
    if (pos >= config_.max_position_embeddings) {
      Logger::warning("Reached max sequence length (" +
                      std::to_string(config_.max_position_embeddings) +
                      "). Stopping.");
      break;
    }

    int input_token_id =
        (pos < num_prompt_tokens) ? token_ids[pos] : next_token_id;
    std::vector<float> logits;

#ifdef HAS_CUDA
    logits = model_->forward_device(input_token_id, pos, &kv_cache_, nullptr);
#else

    std::vector<float> input_embedding =
        model_->lookup_embedding(input_token_id);
    if (input_embedding.empty()) {
      Logger::error("Failed to get embedding for token " +
                    std::to_string(input_token_id));
      break;
    }
    logits = model_->forward(input_embedding, pos, &kv_cache_, nullptr);
#endif

    kv_cache_.seq_len = pos + 1;

    if (logits.empty()) {
#ifdef HAS_CUDA
      Logger::error("Model forward_device returned empty logits at pos " +
                    std::to_string(pos));
#else
      Logger::error("Model forward (CPU) returned empty logits at pos " +
                    std::to_string(pos));
#endif
      break;
    }

    if (pos >= num_prompt_tokens - 1) {
      next_token_id = argmax(logits);

      if (next_token_id == eos_token_id_) {
        Logger::info("EOS token generated. Stopping.");
        break;
      }

      token_ids.push_back(next_token_id);
      generated_count++;

      if (generated_count >= steps) {
        Logger::info("Max steps reached. Stopping.");
        break;
      }
    }
  }

  std::vector<int> generated_only_ids(token_ids.begin() + num_prompt_tokens,
                                      token_ids.end());
  std::string result = tokenizer_->decode(generated_only_ids, true);
  Logger::info("Generated response: " + result);
  return result;
}

}  // namespace tinyllama