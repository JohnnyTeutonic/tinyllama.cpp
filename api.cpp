#include "api.h"
#include "gguf_parser.h"
#include "model_macros.h"

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

  // If temperature is very low, fall back to greedy sampling
  if (temperature < 0.05f) {
    return std::distance(logits.begin(), std::max_element(logits.begin(), logits.end()));
  }

  int vocab_size = logits.size();

  top_k = (std::min)(top_k, vocab_size);
  if (top_k <= 0) top_k = vocab_size;

  std::vector<float> scaled_logits(vocab_size);
  float max_logit = -std::numeric_limits<float>::infinity();
  for (float logit : logits) max_logit = (std::max)(max_logit, logit);

  // Scale logits to avoid numerical instability
  const float scale = 1.0f / temperature;
  for (int i = 0; i < vocab_size; ++i) {
    scaled_logits[i] = (logits[i] - max_logit) * scale;
  }

  std::vector<double> probs_double(vocab_size);
  double sum_exp = 0.0;
  for (int i = 0; i < vocab_size; ++i) {
    probs_double[i] = std::exp(static_cast<double>(scaled_logits[i]));
    sum_exp += probs_double[i];
  }

  // Normalize probabilities
  if (sum_exp > 0.0) {
    for (int i = 0; i < vocab_size; ++i) {
      probs_double[i] /= sum_exp;
    }
  } else {
    // If all probabilities are zero, fall back to uniform distribution
    for (int i = 0; i < vocab_size; ++i) {
      probs_double[i] = 1.0 / vocab_size;
    }
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

  // Renormalize probabilities after top-k and top-p filtering
  std::vector<float> final_probs(prob_idx.size());
  if (final_sum > 0.0f) {
    for (size_t i = 0; i < prob_idx.size(); ++i) {
      final_probs[i] = prob_idx[i].first / final_sum;
    }
  } else {
    // If all probabilities are zero after filtering, use uniform distribution
    float uniform_prob = 1.0f / prob_idx.size();
    std::fill(final_probs.begin(), final_probs.end(), uniform_prob);
  }

  std::discrete_distribution<int> dist(final_probs.begin(), final_probs.end());
  int sampled_idx_in_filtered = dist(rng);

  return prob_idx[sampled_idx_in_filtered].second;
}

TinyLlamaSession::TinyLlamaSession(const std::string& model_path,
                                   const std::string& tokenizer_path,
                                   int threads,
                                   int num_gpu_layers_from_cli,
                                   bool cli_use_mmap)
    : tokenizer_(nullptr), model_(nullptr), kv_cache_() {
  Logger::info(std::string("TinyLlamaSession constructor entered. Model path: ") +
               model_path + ", Tokenizer path (CLI hint): " + tokenizer_path +
               ", Threads: " + std::to_string(threads) +
               ", Num GPU Layers from CLI: " + std::to_string(num_gpu_layers_from_cli) +
               ", Use mmap from CLI: " + (cli_use_mmap ? "true" : "false"));

  ModelConfig initial_model_constructor_config; 
  initial_model_constructor_config.num_cpu_offload_layers = 0; 
  initial_model_constructor_config.use_mmap_for_gguf = cli_use_mmap; 

  int total_hidden_layers_from_file = 0;
  std::filesystem::path path_obj(model_path);
  std::filesystem::path base_dir = std::filesystem::is_directory(path_obj) ? path_obj : path_obj.parent_path();
  std::string default_tokenizer_json_path = (base_dir / "tokenizer.json").string();

  Logger::info(std::string("Determined base_dir: ") + base_dir.string());
  Logger::info(std::string("Default external tokenizer.json path: ") + default_tokenizer_json_path);

  // --- 1. Load Model and its definitive ModelConfig --- 
  if (std::filesystem::is_directory(path_obj)) { // SafeTensors Path
    Logger::info(std::string("Loading SafeTensors model from directory: ") + model_path);
    std::string config_json_path_str = (base_dir / "config.json").string();
    std::string safetensors_path_str = (base_dir / "model.safetensors").string();

    if (!std::filesystem::exists(config_json_path_str)) {
      throw std::runtime_error(std::string("config.json not found in directory: ") + model_path);
    }
    if (!std::filesystem::exists(safetensors_path_str)) {
      throw std::runtime_error(std::string("model.safetensors not found in directory: ") + model_path);
    }

    try {
      nlohmann::json config_json = nlohmann::json::parse(read_file_api(config_json_path_str));
      initial_model_constructor_config = parse_model_config(config_json);
      total_hidden_layers_from_file = initial_model_constructor_config.num_hidden_layers;
      initial_model_constructor_config.use_mmap_for_gguf = cli_use_mmap; 
      Logger::info(std::string("SafeTensors config.json: num_hidden_layers = ") + std::to_string(total_hidden_layers_from_file));

      int actual_cpu_offload_layers = 0;
      if (num_gpu_layers_from_cli == -1) actual_cpu_offload_layers = 0;
      else if (num_gpu_layers_from_cli == 0) actual_cpu_offload_layers = total_hidden_layers_from_file;
      else actual_cpu_offload_layers = total_hidden_layers_from_file - num_gpu_layers_from_cli;
      
      actual_cpu_offload_layers = std::max(0, std::min(actual_cpu_offload_layers, total_hidden_layers_from_file));
      initial_model_constructor_config.num_cpu_offload_layers = actual_cpu_offload_layers;
      Logger::info(std::string("SafeTensors path: initial_model_constructor_config num_cpu_offload_layers = ") + std::to_string(initial_model_constructor_config.num_cpu_offload_layers));

      SafeTensorsLoader st_loader(safetensors_path_str);
      model_ = std::make_unique<TinyLlamaModel>(initial_model_constructor_config, st_loader);
    } catch (const std::exception& e) {
      throw std::runtime_error(std::string("Failed during SafeTensors model loading: ") + e.what());
    }
  } else if (std::filesystem::is_regular_file(path_obj) && path_obj.extension() == ".gguf") { // GGUF Path
    Logger::info(std::string("Loading GGUF model from file: ") + model_path);
    try {
      Logger::info("[API GGUF Path] Before load_gguf_meta");
      auto temp_gguf_data_uptr = std::make_unique<GGUFData>(load_gguf_meta(model_path, cli_use_mmap));
      Logger::info("[API GGUF Path] After load_gguf_meta, before parse_model_config_from_gguf");
      
      ModelConfig config_parsed_from_gguf = parse_model_config_from_gguf(*temp_gguf_data_uptr);
      Logger::info("[API GGUF Path] After parse_model_config_from_gguf");
      
      total_hidden_layers_from_file = config_parsed_from_gguf.num_hidden_layers;
      Logger::info(std::string("GGUF metadata peek (from API session): num_hidden_layers = ") + std::to_string(total_hidden_layers_from_file));
      
      // Prepare the final config for the model constructor
      ModelConfig final_model_constructor_config = config_parsed_from_gguf; // Start with GGUF parsed config
      final_model_constructor_config.use_mmap_for_gguf = cli_use_mmap; // Apply CLI mmap preference
      
      int actual_cpu_offload_layers = 0;
      if (num_gpu_layers_from_cli == -1) actual_cpu_offload_layers = 0; // All GPU (or all CPU if no GPU support)
      else if (num_gpu_layers_from_cli == 0) actual_cpu_offload_layers = total_hidden_layers_from_file; // All CPU
      else actual_cpu_offload_layers = total_hidden_layers_from_file - num_gpu_layers_from_cli;
      
      actual_cpu_offload_layers = std::max(0, std::min(actual_cpu_offload_layers, total_hidden_layers_from_file));
      final_model_constructor_config.num_cpu_offload_layers = actual_cpu_offload_layers;
      Logger::info(std::string("GGUF path (API session): final_model_constructor_config num_cpu_offload_layers = ") + std::to_string(final_model_constructor_config.num_cpu_offload_layers));

      Logger::info("[API GGUF Path] Before TinyLlamaModel constructor call (with GGUFData uptr)");
      model_ = std::make_unique<TinyLlamaModel>(final_model_constructor_config, std::move(temp_gguf_data_uptr));
      Logger::info("[API GGUF Path] After TinyLlamaModel constructor call (with GGUFData uptr)");
    } catch (const std::exception& e) {
      Logger::error("[API GGUF Path] Exception caught: " + std::string(e.what()));
      throw std::runtime_error(std::string("Failed to load GGUF model from ") + model_path + ": " + e.what());
    }
  } else {
     throw std::runtime_error(std::string("Invalid model_path: Must be a directory (for SafeTensors) or a .gguf file. Path: ") + model_path);
  }

  if (!model_) {
    throw std::runtime_error("Model pointer is null after loading attempt.");
  }
  config_ = model_->get_config(); 
  std::string family_str_log = "UNKNOWN";
  if (config_.tokenizer_family == ModelConfig::TokenizerFamily::LLAMA3_TIKTOKEN) family_str_log = "LLAMA3_TIKTOKEN";
  else if (config_.tokenizer_family == ModelConfig::TokenizerFamily::LLAMA_SENTENCEPIECE) family_str_log = "LLAMA_SENTENCEPIECE";
  Logger::info(std::string("Model loaded. Definitive ModelConfig obtained. Architecture: ") + config_.architecture +
               ", Tokenizer Family: " + family_str_log +
               ", Vocab Size: " + std::to_string(config_.vocab_size));

  // --- 2. Instantiate Tokenizer using the definitive config_ ---
  Logger::info("[Tokenizer Init Check] config_.is_gguf_file_loaded = " + std::string(config_.is_gguf_file_loaded ? "true" : "false"));
  try {
    if (config_.is_gguf_file_loaded) {
      const GGUFData* gguf_data_ptr = model_->get_gguf_data();
      if (!gguf_data_ptr) {
        throw std::runtime_error("GGUF model loaded but GGUFData pointer is null from model.");
      }
      Logger::info("Attempting to initialize tokenizer from GGUF data directly.");
      tokenizer_ = std::make_unique<Tokenizer>(*gguf_data_ptr, config_);
      Logger::info("Tokenizer initialized from GGUF data.");
    } else { // SafeTensors path - tokenizer is always external from tokenizer.json
      Logger::info(std::string("SafeTensors model: Initializing tokenizer from external file: ") + default_tokenizer_json_path);
      if (!std::filesystem::exists(default_tokenizer_json_path)) {
        throw std::runtime_error(std::string("tokenizer.json not found at: ") + default_tokenizer_json_path);
      }
      tokenizer_ = std::make_unique<Tokenizer>(default_tokenizer_json_path, default_tokenizer_json_path, config_); 
      Logger::info("Tokenizer initialized from external files for SafeTensors model.");
    }
  } catch (const std::exception& e) {
    throw std::runtime_error(std::string("Failed to initialize Tokenizer: ") + e.what());
  }
  
  if (!tokenizer_) {
      throw std::runtime_error("Tokenizer pointer is null after instantiation attempt.");
  }

  eos_token_id_ = config_.eos_token_id;
  int head_dim = config_.hidden_size / config_.num_attention_heads;
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
                                      int top_k, float top_p,
                                      const std::string& system_prompt,
                                      bool apply_q_a_format) {
  Logger::info("Generate called. Initial Prompt: \"" + prompt_input +
               "\", Steps: " + std::to_string(steps) +
               ", Temperature: " + std::to_string(temperature) +
               ", Top-K: " + std::to_string(top_k) +
               ", Top-P: " + std::to_string(top_p) +
               ", System Prompt: \"" + system_prompt +
               "\", Apply Chat/Q&A Format: " + (apply_q_a_format ? "true" : "false"));

  std::string final_prompt_for_tokenization;
  if (apply_q_a_format) {
    // If GGUF is loaded, always use simple Q:A format as it works best.
    // For SafeTensors, prefer chat template if available.
    if (config_.is_gguf_file_loaded) {
        Logger::info("GGUF model detected. Using simple Q:A format for prompt.");
        final_prompt_for_tokenization = "Q: " + prompt_input + "\nA:";
        Logger::info("Applied Q:A format (GGUF). Prompt for tokenization: \"" +
                     final_prompt_for_tokenization + "\"");
    } else if (tokenizer_ && !config_.chat_template_string.empty()) {
        // Non-GGUF (SafeTensors usually) - try chat template
        try {
            Logger::info("Applying chat template (SafeTensors). System: '" + system_prompt + "', User: '" + prompt_input + "'");
            final_prompt_for_tokenization = tokenizer_->apply_chat_template(prompt_input, system_prompt, config_);
            Logger::info("Formatted prompt using chat template: \"" + final_prompt_for_tokenization + "\"");
        } catch (const std::exception& e) {
            Logger::error("Error applying chat template (SafeTensors): " + std::string(e.what()) + ". Falling back to simple Q:A format.");
            final_prompt_for_tokenization = "Q: " + prompt_input + "\nA:"; 
            Logger::info("Applied Q:A: fallback format (SafeTensors). Prompt for tokenization: \"" +
                         final_prompt_for_tokenization + "\"");
        }
    } else {
        // Fallback for SafeTensors if no chat template string or no tokenizer
        Logger::warning("Chat template string empty or tokenizer not available (SafeTensors). Using simple Q:A format.");
        final_prompt_for_tokenization = "Q: " + prompt_input + "\nA:";
        Logger::info("Applied Q:A: format (SafeTensors, no template). Prompt for tokenization: \"" +
                     final_prompt_for_tokenization + "\"");
    }
  } else {
    final_prompt_for_tokenization = prompt_input;
    Logger::info("Using provided prompt as-is for tokenization: \"" +
                 final_prompt_for_tokenization + "\"");
  }

  // New way: Directly encode the prompt to IDs. 
  // Assuming encode should not add BOS/EOS here as prompt handling is complex and might already include them.
  // Or, if encode is meant to be the primary entry, it should handle BOS/EOS internally based on some logic.
  // For now, matching the previous behavior of just getting IDs for the final_prompt_for_tokenization.
  std::vector<int> token_ids = tokenizer_->encode(final_prompt_for_tokenization, false, false);

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

  std::vector<float> current_data_host; // To hold embedding or output of CPU layers

  for (int pos = 0; pos < total_steps; ++pos) {
    if (pos >= config_.max_position_embeddings) {
      Logger::warning("Reached max sequence length (" +
                      std::to_string(config_.max_position_embeddings) +
                      "). Stopping.");
      break;
    }

    int input_token_id =
        (pos < num_prompt_tokens) ? token_ids[pos] : next_token_id;
    std::vector<float> logits; // Declare logits here

    current_data_host = model_->lookup_embedding(input_token_id);
    if (current_data_host.empty()) {
        Logger::error("Failed to get embedding for token " + std::to_string(input_token_id) + " at pos " + std::to_string(pos));
        break;
    }

    int num_total_layers = config_.num_hidden_layers;
    int num_cpu = config_.num_cpu_offload_layers;
    int num_gpu = num_total_layers - num_cpu;

    if (num_cpu > 0) { // Process CPU layers if any
        Logger::debug("[Generate] Processing " + std::to_string(num_cpu) + " CPU layers for pos " + std::to_string(pos));
        current_data_host = model_->forward(current_data_host, pos, &kv_cache_, nullptr);
        // current_data_host now holds output of last CPU layer, or final logits if all layers are CPU (num_gpu == 0)
    }

#ifdef HAS_CUDA
    if (num_gpu > 0) { // Process GPU layers if any
        Logger::debug("[Generate] Processing " + std::to_string(num_gpu) + " GPU layers for pos " + std::to_string(pos));
        // current_data_host contains either embedding (if num_cpu == 0) or output of CPU layers.
        // Copy this host data to device model_->x_dev_ to start GPU pipeline.
        float* x_dev_ptr = model_->get_x_dev();
        if (!x_dev_ptr) { 
            Logger::fatal("model_->x_dev_ is null before GPU forward pass!"); 
            throw std::runtime_error("Critical error: x_dev_ is null in GPU pipeline.");
        }
        if (current_data_host.size() * sizeof(float) == 0) {
             Logger::fatal("current_data_host is empty before cudaMemcpy to x_dev_!"); 
            throw std::runtime_error("Critical error: current_data_host is empty for GPU pipeline.");
        }
        gpuErrchk(cudaMemcpy(x_dev_ptr, current_data_host.data(), current_data_host.size() * sizeof(float), cudaMemcpyHostToDevice));
        
        logits = model_->forward_device(x_dev_ptr, pos, &kv_cache_, nullptr);
    } else if (num_cpu > 0 && num_gpu == 0) { // All layers were CPU, and model_->forward() returned logits
        logits = current_data_host;
    } else if (num_total_layers == 0) { // No layers at all
        Logger::warning("No layers in the model (HAS_CUDA path).");
        logits.assign(config_.vocab_size > 0 ? config_.vocab_size : 1, 0.0f); // Assign dummy if vocab_size is valid
    } else if (num_cpu == 0 && num_gpu == 0 && num_total_layers > 0) { // Should not happen if logic is correct
         Logger::error("Inconsistent state: HAS_CUDA, num_cpu=0, num_gpu=0, but num_total_layers > 0. Defaulting to CPU path for safety.");
         // Fallback to full CPU path just in case, using original embedding as input to forward.
         // This assumes model_->forward() is safe to call even if it was configured for GPU layers that are now skipped.
         current_data_host = model_->lookup_embedding(input_token_id); // Re-fetch original embedding for safety
         logits = model_->forward(current_data_host, pos, &kv_cache_, nullptr);
    }
#else // No CUDA compiled - All layers (if any) run on CPU
    if (num_total_layers > 0) {
        // current_data_host already contains embedding if num_cpu_layers was 0, or output of CPU layers if num_cpu_layers > 0.
        // If num_cpu_layers was > 0, model_->forward has already been called. We call it here if it hasn't (pure CPU path from start)
        if (num_cpu == 0) { // Only call model->forward if it wasn't called for CPU layers above
             Logger::debug("[Generate] NO_CUDA: Processing all " + std::to_string(num_total_layers) + " layers on CPU for pos " + std::to_string(pos));
             logits = model_->forward(current_data_host, pos, &kv_cache_, nullptr);
        } else { // num_cpu > 0, means model->forward was already called and current_data_host has final logits
            logits = current_data_host;
        }
    } else {
        Logger::warning("No layers in the model (NO_CUDA path).");
        logits.assign(config_.vocab_size > 0 ? config_.vocab_size : 1, 0.0f); // Assign dummy
    }
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
      next_token_id = sample_top_k_top_p_temperature(logits, temperature, top_k, top_p, rng_);
      Logger::debug("[Generate Loop] Pos: " + std::to_string(pos) + ", Prompt Tokens: " + std::to_string(num_prompt_tokens) + ", Generated Token ID: " + std::to_string(next_token_id));

      if (next_token_id == eos_token_id_) {
        Logger::info("EOS token generated (ID: " + std::to_string(next_token_id) + "). Stopping.");
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

  // Log all generated IDs before decoding
  std::string generated_ids_str = "[Generated IDs Pre-Decode] ";
  for(int gen_id : generated_only_ids) {
    generated_ids_str += std::to_string(gen_id) + " ";
  }
  Logger::debug(generated_ids_str);

  std::string result = tokenizer_->decode(generated_only_ids, true);
  Logger::info("Generated response: " + result);
  return result;
}

}  // namespace tinyllama