#include "api.h"
#include "gguf_parser.h"
#include "model_macros.h"

#include <algorithm>
#include <chrono>
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
#include <iomanip>
#include <sstream>

#include "logger.h"
#include "model.h"
#include "safetensors_loader.h"
#include "tokenizer.h"

namespace tinyllama {


static void log_vector_summary_detailed(const std::string& name,
                                        const std::vector<float>& v,
                                        int current_pos, int current_layer,
                                        int N) {
  if (v.empty()) {
    Logger::info(name + " (pos=" + std::to_string(current_pos) + ", layer=" +
                 std::to_string(current_layer) + "): EMPTY VECTOR");
    return;
  }
  std::stringstream ss;
  ss << name << " (pos=" << std::to_string(current_pos)
     << ", layer=" << std::to_string(current_layer) << "): size=" << v.size();
  ss << ", first " << N << ": [";
  for (int i = 0; i < N && i < v.size(); ++i) {
    ss << std::fixed << std::setprecision(4) << v[i]
       << (i == N - 1 || i == v.size() - 1 ? "" : ", ");
  }
  ss << "]";
  float min_val = v[0], max_val = v[0], sum = 0.0f;
  bool all_finite = true;
  for (float val : v) {
    if (val < min_val) min_val = val;
    if (val > max_val) max_val = val;
    sum += val;
    if (!std::isfinite(val)) all_finite = false;
  }
  ss << ", min=" << std::fixed << std::setprecision(4) << min_val;
  ss << ", max=" << std::fixed << std::setprecision(4) << max_val;
  ss << ", mean=" << std::fixed << std::setprecision(4) << (sum / v.size());
  ss << ", finite=" << (all_finite ? "yes" : "no");
  Logger::info(ss.str());
}

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


TinyLlamaSession::TinyLlamaSession(const std::string& model_path_arg,
                                   const std::string& tokenizer_path_arg,
                                   int threads,
                                   int num_gpu_layers_from_cli,
                                   bool cli_use_mmap,
                                   bool use_kv_quant)
    : threads_(threads), rng_(std::random_device{}()) {
  Logger::info("TinyLlamaSession constructor entered. Model path: " + model_path_arg +
               ", Tokenizer path: " + tokenizer_path_arg +
               ", Threads: " + std::to_string(threads) +
               ", Num GPU Layers (CLI): " + std::to_string(num_gpu_layers_from_cli) +
               ", Use MMAP (CLI): " + (cli_use_mmap ? "true" : "false") +
               ", Use KV Quant (CLI): " + (use_kv_quant ? "true" : "false"));

  std::string effective_model_file_path = model_path_arg; 
  std::string path_for_config_json = model_path_arg;

  ModelConfig initial_model_config_for_model_ctor;
  initial_model_config_for_model_ctor.use_mmap_for_gguf = cli_use_mmap;
  initial_model_config_for_model_ctor.use_kvcache_quantization = use_kv_quant;
  if (num_gpu_layers_from_cli < 0) { 
      initial_model_config_for_model_ctor.num_cpu_offload_layers = 0;
  } else {
      initial_model_config_for_model_ctor.num_cpu_offload_layers = num_gpu_layers_from_cli;
  }

  std::filesystem::path fs_model_path(model_path_arg);
  bool is_dir = std::filesystem::is_directory(fs_model_path);

  if (is_dir) {
    Logger::info("Model path is a directory. Assuming SafeTensors model directory: " + model_path_arg);
    effective_model_file_path = (fs_model_path / "model.safetensors").string();
    std::string config_json_path_in_dir = (fs_model_path / "config.json").string(); 

    Logger::info("Derived SafeTensors model file path: " + effective_model_file_path);
    Logger::info("Path for loading config.json: " + config_json_path_in_dir);

    // Directly populate initial_model_config_for_model_ctor
    // load_model_config_from_json returns bool and populates the passed ModelConfig&
    bool st_config_loaded = SafeTensorsLoader::load_model_config_from_json(config_json_path_in_dir, initial_model_config_for_model_ctor);
    
    if (st_config_loaded) {
        Logger::info("Successfully loaded config.json directly into initial_model_config_for_model_ctor.");
        // Log tokenizer_family IMMEDIATELY after loading from config.json
        std::string family_after_json_load = "UNKNOWN_POST_JSON_LOAD";
        if (initial_model_config_for_model_ctor.tokenizer_family == ModelConfig::TokenizerFamily::LLAMA_SENTENCEPIECE) family_after_json_load = "LLAMA_SENTENCEPIECE";
        else if (initial_model_config_for_model_ctor.tokenizer_family == ModelConfig::TokenizerFamily::LLAMA3_TIKTOKEN) family_after_json_load = "LLAMA3_TIKTOKEN";
        Logger::info("[API_CPP_POST_JSON_LOAD_DIR_CASE] Tokenizer family in initial_model_config_for_model_ctor: " + family_after_json_load);
        // tokenizer_family should now be set in initial_model_config_for_model_ctor
        // num_hidden_layers is also now set from config.json
    } else {
        Logger::warning("Failed to load config.json for SafeTensors. initial_model_config_for_model_ctor will have defaults/CLI overrides for some fields, tokenizer_family likely UNKNOWN.");
        // If config.json fails, num_hidden_layers might be 0 or default, which will affect cpu offload calculation.
        // It's crucial config.json loads for correct layer counts.
    }
    
    // Apply CLI overrides for mmap and GPU layers. GPU layer logic depends on total_hidden_layers from config.
    initial_model_config_for_model_ctor.use_mmap_for_gguf = cli_use_mmap; 

    int total_layers_from_config = initial_model_config_for_model_ctor.num_hidden_layers;
    if (total_layers_from_config <= 0 && st_config_loaded) {
        Logger::warning("config.json loaded but num_hidden_layers is <= 0. GPU offload logic might be incorrect.");
    } else if (total_layers_from_config <= 0 && !st_config_loaded) {
        Logger::warning("config.json NOT loaded and num_hidden_layers is <= 0 (default). GPU offload may not work as expected. Model load will likely fail.");
        // If config.json didn't load, total_layers_from_config is likely 0 from default ModelConfig.
        // The TinyLlamaModel constructor will ultimately use its own parsed config, but this intermediate step needs care.
    }

    if (num_gpu_layers_from_cli < 0) { // -1 signifies all layers on GPU
        initial_model_config_for_model_ctor.num_cpu_offload_layers = 0;
    } else if (num_gpu_layers_from_cli == 0) { // 0 signifies all layers on CPU
        initial_model_config_for_model_ctor.num_cpu_offload_layers = total_layers_from_config; // All layers offloaded to CPU
    } else { // N > 0 signifies N layers on GPU
        if (total_layers_from_config > 0) {
            initial_model_config_for_model_ctor.num_cpu_offload_layers = total_layers_from_config - num_gpu_layers_from_cli;
        } else {
            // Cannot determine actual GPU layer count if total_layers_from_config is unknown.
            // Pass the CLI hint, TinyLlamaModel ctor will deal with it against its own parsed config.
            initial_model_config_for_model_ctor.num_cpu_offload_layers = num_gpu_layers_from_cli; 
            Logger::warning("Total hidden layers unknown from config.json before model load; passing num_gpu_layers_from_cli as num_cpu_offload_layers hint.");
        }
    }
    // Clamp num_cpu_offload_layers
    if (total_layers_from_config > 0) {
         initial_model_config_for_model_ctor.num_cpu_offload_layers = std::max(0, std::min(initial_model_config_for_model_ctor.num_cpu_offload_layers, total_layers_from_config));
    }

    initial_model_config_for_model_ctor.is_gguf_file_loaded = false;

    SafeTensorsLoader st_loader(effective_model_file_path); 
    model_ = std::make_unique<TinyLlamaModel>(initial_model_config_for_model_ctor, st_loader);
    config_ = model_->get_config(); 
    config_.is_gguf_file_loaded = false; // Ensure this is set for session's copy too

    config_.use_kvcache_quantization = use_kv_quant; // Re-apply CLI/constructor preference

    Logger::info("TinyLlamaSession: Finalizing ModelConfig for KVCache initialization. use_kvcache_quantization set to: " + 
                  std::string(config_.use_kvcache_quantization ? "true" : "false"));

    kv_cache_.initialize(config_, config_.num_hidden_layers, 
                         config_.num_hidden_layers - config_.num_cpu_offload_layers, 
                         config_.max_position_embeddings, 
                         config_.num_key_value_heads, 
                         config_.hidden_size / config_.num_attention_heads);

  } else { // Not a directory, assume it's a file path and check extension
    std::string extension = fs_model_path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

    if (extension == ".gguf") {
      Logger::info("GGUF model type detected by extension for Session constructor: " + model_path_arg);
      model_ = std::make_unique<TinyLlamaModel>(initial_model_config_for_model_ctor, model_path_arg);
      config_ = model_->get_config(); 
    } else if (extension == ".safetensors") {
      Logger::info("SafeTensors model type detected by extension for Session constructor (file case): " + model_path_arg);
      effective_model_file_path = model_path_arg;
      
      bool st_config_loaded = SafeTensorsLoader::load_model_config_from_json(effective_model_file_path, initial_model_config_for_model_ctor);
      if (st_config_loaded) {
          Logger::info("Successfully loaded config.json for SafeTensors in Session ctor (file case).");
          // Log tokenizer_family IMMEDIATELY after loading from config.json (file case)
          std::string family_after_json_load_file_case = "UNKNOWN_POST_JSON_LOAD_FILE_CASE";
          if (initial_model_config_for_model_ctor.tokenizer_family == ModelConfig::TokenizerFamily::LLAMA_SENTENCEPIECE) family_after_json_load_file_case = "LLAMA_SENTENCEPIECE";
          else if (initial_model_config_for_model_ctor.tokenizer_family == ModelConfig::TokenizerFamily::LLAMA3_TIKTOKEN) family_after_json_load_file_case = "LLAMA3_TIKTOKEN";
          Logger::info("[API_CPP_POST_JSON_LOAD_FILE_CASE] Tokenizer family in initial_model_config_for_model_ctor: " + family_after_json_load_file_case);
          // tokenizer_family and num_hidden_layers are now set in initial_model_config_for_model_ctor
      } else {
          Logger::warning("Failed to load config.json for SafeTensors in Session ctor (file case). Model will use defaults or CLI overrides.");
      }
      
      initial_model_config_for_model_ctor.use_mmap_for_gguf = cli_use_mmap;
      // Correctly calculate num_cpu_offload_layers based on total_layers_from_config
      int total_layers_from_config_file_case = initial_model_config_for_model_ctor.num_hidden_layers;
      if (num_gpu_layers_from_cli < 0) { 
          initial_model_config_for_model_ctor.num_cpu_offload_layers = 0;
      } else if (num_gpu_layers_from_cli == 0) { 
          initial_model_config_for_model_ctor.num_cpu_offload_layers = total_layers_from_config_file_case;
      } else { 
          if (total_layers_from_config_file_case > 0) {
            initial_model_config_for_model_ctor.num_cpu_offload_layers = total_layers_from_config_file_case - num_gpu_layers_from_cli;
          } else {
            initial_model_config_for_model_ctor.num_cpu_offload_layers = num_gpu_layers_from_cli; 
            Logger::warning("Total hidden layers unknown from config.json (file case); passing num_gpu_layers_from_cli as num_cpu_offload_layers hint.");
          }
      }
      if (total_layers_from_config_file_case > 0) {
        initial_model_config_for_model_ctor.num_cpu_offload_layers = std::max(0, std::min(initial_model_config_for_model_ctor.num_cpu_offload_layers, total_layers_from_config_file_case));
      }

      initial_model_config_for_model_ctor.is_gguf_file_loaded = false;

      SafeTensorsLoader st_loader(effective_model_file_path);
      model_ = std::make_unique<TinyLlamaModel>(initial_model_config_for_model_ctor, st_loader);
      config_ = model_->get_config(); 
      config_.is_gguf_file_loaded = false; // Ensure this is set for session's copy too

      config_.use_kvcache_quantization = use_kv_quant; // Re-apply CLI/constructor preference

      Logger::info("TinyLlamaSession: Finalizing ModelConfig for KVCache initialization. use_kvcache_quantization set to: " + 
                    std::string(config_.use_kvcache_quantization ? "true" : "false"));

      // Initialize KVCache with potentially updated config_ (from model load) 
      // and the now-set use_kvcache_quantization flag.
      kv_cache_.initialize(config_, config_.num_hidden_layers, 
                           config_.num_hidden_layers - config_.num_cpu_offload_layers, 
                           config_.max_position_embeddings, 
                           config_.num_key_value_heads, 
                           config_.hidden_size / config_.num_attention_heads);
    } else {
      throw std::runtime_error("Unsupported model file type or extension in Session constructor: " + model_path_arg +
                               ". Please provide a directory for SafeTensors, a .gguf file, or a .safetensors file.");
    }
  }

  if (!model_) {
      throw std::runtime_error("Model pointer is null after instantiation attempt in Session constructor.");
  }

  try {
    if (config_.is_gguf_file_loaded) {
      const GGUFData* gguf_data = model_->get_gguf_data();
      if (!gguf_data) {
          throw std::runtime_error("GGUF model loaded but GGUFData is null in Session constructor.");
      }
      tokenizer_ = std::make_unique<Tokenizer>(*gguf_data, config_);
      Logger::info("Tokenizer initialized from GGUF metadata.");
    } else { // SafeTensors (either from directory or direct .safetensors file)
      std::filesystem::path p_tokenizer_arg(tokenizer_path_arg);
      std::string tokenizer_dir = p_tokenizer_arg.parent_path().string();
      if (tokenizer_dir.empty()) { 
          tokenizer_dir = "."; 
      }
      
      std::string vocab_json_path = (std::filesystem::path(tokenizer_dir) / "tokenizer.json").string();
      // The model_path for the tokenizer constructor should be the actual sentencepiece model file, e.g. data/tokenizer.model
      // tokenizer_path_arg already holds this (e.g. data/tokenizer.model)
      std::string sp_model_path = tokenizer_path_arg; 

      Logger::info("Initializing Tokenizer for SafeTensors. Vocab JSON path: " + vocab_json_path + ", SP Model path: " + sp_model_path);
      // Log the tokenizer_family from the config_ that will be passed to the Tokenizer
      std::string family_to_log = "UNKNOWN_IN_API_CPP";
      if (config_.tokenizer_family == ModelConfig::TokenizerFamily::LLAMA_SENTENCEPIECE) family_to_log = "LLAMA_SENTENCEPIECE";
      else if (config_.tokenizer_family == ModelConfig::TokenizerFamily::LLAMA3_TIKTOKEN) family_to_log = "LLAMA3_TIKTOKEN";
      Logger::info("[API_CPP_TOKENIZER_INIT] Tokenizer family from session config for SafeTensors: " + family_to_log);

      tokenizer_ = std::make_unique<Tokenizer>(vocab_json_path, sp_model_path, config_); 
      Logger::info("Tokenizer initialized from external files for SafeTensors model.");
    }
  } catch (const std::exception& e) {
    throw std::runtime_error(std::string("Failed to initialize Tokenizer: ") + e.what());
  }
  
  if (!tokenizer_) {
      throw std::runtime_error("Tokenizer pointer is null after instantiation attempt.");
  }

  eos_token_id_ = config_.eos_token_id; // Use the session's config_ which is now model's config
  
  const ModelConfig& final_model_config = model_->get_config(); // Explicitly use model's config
  int total_model_layers = final_model_config.num_hidden_layers;
  
  int effective_cpu_offload_layers = final_model_config.num_cpu_offload_layers;
  int gpu_layers_for_kvcache = total_model_layers - effective_cpu_offload_layers;
  if (gpu_layers_for_kvcache < 0) gpu_layers_for_kvcache = 0; // Sanity check, should not happen if model ctor is correct
  if (gpu_layers_for_kvcache > total_model_layers) gpu_layers_for_kvcache = total_model_layers; // Sanity

  Logger::info("[Session KVCache Init] Total Layers: " + std::to_string(total_model_layers) +
               ", Effective CPU Offload by Model: " + std::to_string(effective_cpu_offload_layers) +
               ", GPU Layers for KVCache: " + std::to_string(gpu_layers_for_kvcache));

  if (total_model_layers <= 0) {
      throw std::runtime_error("Model config has zero or negative num_hidden_layers before KVCache init.");
  }
  if (final_model_config.num_attention_heads <= 0) {
      throw std::runtime_error("Model config has zero or negative num_attention_heads before KVCache init.");
  }

  int head_dim = final_model_config.hidden_size / final_model_config.num_attention_heads;
  
  kv_cache_.initialize(final_model_config,          // Total layers for CPU part of KVCache
                       total_model_layers, 
                       gpu_layers_for_kvcache,    // Actual GPU layers for device memory
                       final_model_config.max_position_embeddings,
                       final_model_config.num_key_value_heads, 
                       head_dim);
  Logger::info("TinyLlamaSession initialization complete (after KVCache init).");
}

TinyLlamaSession::~TinyLlamaSession() {
  Logger::info("TinyLlamaSession: Destroyed.");
}

std::string TinyLlamaSession::generate(const std::string& user_prompt, int steps,
                                     float temperature,
                                     int top_k, float top_p,
                                     const std::string& system_prompt_arg, // Renamed for clarity inside function
                                     bool apply_q_a_format_cli_hint) { // Renamed for clarity
  auto t_start = std::chrono::high_resolution_clock::now(); // Start timing

  generated_text_for_api_return_.clear(); // Clear for new generation
  generated_stream_.str("");             // Clear for new generation
  generated_stream_.clear();          // Clear error flags

  Logger::info("[Generate API] User prompt: \"" + user_prompt + "\", System prompt: \"" + system_prompt_arg + "\", Steps: " + std::to_string(steps));

  if (!model_ || !tokenizer_) {
    throw std::runtime_error("Model or tokenizer not loaded.");
  }

  std::string final_prompt_for_encoding;
  bool used_chat_template = false;

  // Log conditions for chat template application
  if (tokenizer_) {
    bool gguf_template_empty = tokenizer_->get_gguf_chat_template().empty();
    Logger::info("[Generate API] GGUF chat template from tokenizer is empty: " + std::string(gguf_template_empty ? "true" : "false"));
    if (!gguf_template_empty) {
        Logger::info("[Generate API] GGUF Template Content (first 100 chars): " + tokenizer_->get_gguf_chat_template().substr(0, 100));
    }
  } else {
    Logger::warning("[Generate API] Tokenizer is null before checking chat template!");
  }
  std::string family_log_str = "UNKNOWN";
  if (config_.tokenizer_family == ModelConfig::TokenizerFamily::LLAMA_SENTENCEPIECE) family_log_str = "LLAMA_SENTENCEPIECE";
  else if (config_.tokenizer_family == ModelConfig::TokenizerFamily::LLAMA3_TIKTOKEN) family_log_str = "LLAMA3_TIKTOKEN";
  Logger::info("[Generate API] Configured tokenizer_family: " + family_log_str);

  // New Priority Logic:
  // Priority 1 (NEW): Legacy Q/A formatting if CLI hint is true
  if (apply_q_a_format_cli_hint) {
    Logger::info("[Generate API] Using legacy Q/A formatting (CLI Hint is true - Priority 1).");
    std::string temp_prompt = user_prompt;
    if (!system_prompt_arg.empty()) {
        temp_prompt = system_prompt_arg + "\\n\\nQ: " + user_prompt + "\\nA:";
    } else {
        temp_prompt = "Q: " + user_prompt + "\\nA:";
    }
    final_prompt_for_encoding = temp_prompt;
    used_chat_template = false; // Q/A is not a 'chat template' in the GGUF sense
  }
  // Priority 2 (WAS 1): GGUF Chat Template from Tokenizer (only if Q/A hint is false)
  else if (tokenizer_ && !tokenizer_->get_gguf_chat_template().empty()) {
    std::string gguf_template_content = tokenizer_->get_gguf_chat_template();
    bool is_llama_sentencepiece_family = (config_.tokenizer_family == ModelConfig::TokenizerFamily::LLAMA_SENTENCEPIECE);
    bool looks_like_jinja = (gguf_template_content.find("{%") != std::string::npos);

    if (is_llama_sentencepiece_family && looks_like_jinja) {
        Logger::info("[Generate API] Detected LLAMA_SENTENCEPIECE model with a Jinja-like GGUF template. Forcing Q/A format to avoid C++ Jinja processing issues (Priority 2 Override).");
        std::string temp_prompt = user_prompt;
        if (!system_prompt_arg.empty()) {
            temp_prompt = system_prompt_arg + "\\\\n\\\\nQ: " + user_prompt + "\\\\nA:";
        } else {
            temp_prompt = "Q: " + user_prompt + "\\\\nA:";
        }
        final_prompt_for_encoding = temp_prompt;
        used_chat_template = false;
    } else {
        Logger::info("[Generate API] Using GGUF chat template from tokenizer (Q/A Hint false - Priority 2).");
        final_prompt_for_encoding = tokenizer_->apply_chat_template(user_prompt, system_prompt_arg, config_);
        used_chat_template = true;
    }
  } 
  // Priority 3 (WAS 2): Llama 3 family specific template (only if Q/A hint false and no GGUF template)
  else if (config_.tokenizer_family == ModelConfig::TokenizerFamily::LLAMA3_TIKTOKEN) {
    Logger::info("[Generate API] Llama 3 tokenizer family detected, using apply_chat_template (Q/A Hint false, No GGUF template - Priority 3).");
    final_prompt_for_encoding = tokenizer_->apply_chat_template(user_prompt, system_prompt_arg, config_);
    used_chat_template = true;
  }
  // Priority 4 (WAS 3/4 depending on apply_q_a_format_cli_hint): Raw prompt (if all above are false)
  else {
    Logger::info("[Generate API] No applicable template/hint. Using user prompt as is (prepending system prompt if available - Priority 4).");
    if (!system_prompt_arg.empty()) {
        final_prompt_for_encoding = system_prompt_arg + "\\n\\n" + user_prompt;
    } else {
        final_prompt_for_encoding = user_prompt;
    }
    used_chat_template = false;
  }
  
  Logger::debug("[Generate API] Final prompt for encoding (first 100 chars): \\\"" + final_prompt_for_encoding.substr(0, 100) + "\\\"");

  std::vector<int> tokens = tokenizer_->encode(final_prompt_for_encoding, true, false, Tokenizer::PreTokenizeMethod::DEFAULT); // add_bos=true, add_eos=false (EOS handled by loop)

  if (tokens.empty()) {
    Logger::warning("Tokenization resulted in empty ID list for prompt: " +
                    final_prompt_for_encoding);
    return "";
  }

  int num_prompt_tokens = tokens.size();
  Logger::info("[Generate API] Number of prompt tokens: " + std::to_string(num_prompt_tokens));

  int total_steps = num_prompt_tokens + steps -1; // Max total tokens including prompt
  int generated_count = 0;
  int next_token_id = -1;

  std::vector<float> logits; // Declare logits here, to be populated by prefill or loop
  std::vector<int> generated_token_ids; // Track generated tokens separately
  
  kv_cache_.clear_data(); // Clear K/V vector data for all layers
  kv_cache_.seq_len = 0;  // Reset KVCache logical sequence length for new sequence

  std::vector<float> current_data_host; // To hold embedding or output of CPU layers
  int start_pos_for_loop = 0;

bool prefill_enabled = (num_prompt_tokens > 1);

if (prefill_enabled) {
      Logger::info("[Generate API] Prefill enabled. num_prompt_tokens: " + std::to_string(num_prompt_tokens) + 
                   ", num_cpu_offload_layers: " + std::to_string(config_.num_cpu_offload_layers) +
                   ", total_hidden_layers: " + std::to_string(config_.num_hidden_layers));
      
      std::vector<float> batch_initial_embeddings(num_prompt_tokens * config_.hidden_size);
      for (int i = 0; i < num_prompt_tokens; ++i) {
          std::vector<float> token_embedding = model_->lookup_embedding(tokens[i]);
          if (token_embedding.empty()) {
              Logger::error("Prefill: Embedding lookup returned empty vector for token ID: " + std::to_string(tokens[i]) + " at prompt pos " + std::to_string(i));
              return ""; // Or handle error appropriately
          }
          std::copy(token_embedding.begin(), token_embedding.end(), batch_initial_embeddings.begin() + i * config_.hidden_size);
      }

      // If there are CPU layers to process for prefill
      std::vector<float> cpu_processed_embeddings;
      if (config_.num_cpu_offload_layers > 0) {
          Logger::info("[Generate API] Prefill: Processing " + std::to_string(config_.num_cpu_offload_layers) + " CPU layers for the batch.");
          cpu_processed_embeddings = model_->forward_cpu_batch(batch_initial_embeddings, num_prompt_tokens, config_.num_cpu_offload_layers, 0, &kv_cache_);
          if (cpu_processed_embeddings.empty()) {
               Logger::error("Prefill: forward_cpu_batch returned empty or failed.");
               return "";
          }
      } else {
          cpu_processed_embeddings = batch_initial_embeddings; // No CPU layers, pass embeddings directly
      }

      // If all layers are CPU layers (i.e., num_gpu_layers == 0)
      if (config_.num_cpu_offload_layers == config_.num_hidden_layers) {
          Logger::info("[Generate API] Prefill: All layers are on CPU. Getting logits from final CPU layer output.");
          std::vector<float> batch_logits = model_->forward_cpu_logits_batch(cpu_processed_embeddings, num_prompt_tokens);
          if (batch_logits.empty() || batch_logits.size() % config_.vocab_size != 0) {
              Logger::error("Prefill: forward_cpu_logits_batch returned invalid logits.");
              return "";
          }
          // Extract logits for the last token of the prompt
          logits.assign(batch_logits.begin() + (num_prompt_tokens - 1) * config_.vocab_size,
                        batch_logits.begin() + num_prompt_tokens * config_.vocab_size);
      } else { // GPU layers exist and need to be processed
          Logger::info("[Generate API] Prefill: Processing GPU layers for the batch.");
          // Copy the (potentially CPU-processed) embeddings to the device
          float* d_temp_batch_embeddings = nullptr;
          size_t batch_embeddings_size_bytes = cpu_processed_embeddings.size() * sizeof(float);

          if (batch_embeddings_size_bytes == 0) {
              Logger::error("Prefill: cpu_processed_embeddings is empty, cannot proceed with GPU batch prefill.");
              return ""; // Or handle error appropriately
          }

          gpuErrchk(cudaMalloc(&d_temp_batch_embeddings, batch_embeddings_size_bytes));
          if (!d_temp_batch_embeddings) {
              Logger::error("Prefill: cudaMalloc failed for d_temp_batch_embeddings.");
              return ""; // Or handle error appropriately
          }
          
          gpuErrchk(cudaMemcpy(d_temp_batch_embeddings, cpu_processed_embeddings.data(), 
                               batch_embeddings_size_bytes, cudaMemcpyHostToDevice));

          logits = model_->forward_device_batch_prefill(d_temp_batch_embeddings, num_prompt_tokens, start_pos_for_loop, &kv_cache_, 0);
            // DEBUG: Check if logits are coherent after GPU batch prefill
  if (logits.size() >= config_.vocab_size) {
      std::vector<float> logits_sample(std::min(10, (int)config_.vocab_size));
      std::copy(logits.begin(), logits.begin() + logits_sample.size(), logits_sample.begin());
            
      // Check for NaN/Inf
      bool has_nan_inf = false;
      for (float val : logits) {
          if (std::isnan(val) || std::isinf(val)) {
              has_nan_inf = true;
              break;
          }
      }
    if (has_nan_inf) {
        std::cout << "ERROR: GPU batch prefill produced NaN/Inf logits!" << std::endl;
    }
    
}

          if (d_temp_batch_embeddings) {
              gpuErrchk(cudaFree(d_temp_batch_embeddings));
          }
      }
      
      if (logits.empty()) {
          Logger::error("Prefill: Logits are empty after prefill processing.");
          return ""; // Critical error
      }
      next_token_id = sample_top_k_top_p_temperature(logits, temperature, top_k, top_p, rng_);
      // DON'T stream yet - token hasn't been processed through model!
      generated_token_ids.push_back(next_token_id); // Track generated token
      generated_count++;
      start_pos_for_loop = num_prompt_tokens; // Next token will be at num_prompt_tokens
      kv_cache_.seq_len = num_prompt_tokens; // KVCache is now filled up to num_prompt_tokens

      Logger::info("[Generate API] Prefill completed. next_token_id: " + std::to_string(next_token_id) + 
                   ", Decoded: \"" + tokenizer_->decode({next_token_id}, false) + "\"" + 
                   ", start_pos_for_loop set to: " + std::to_string(start_pos_for_loop));
  }

  for (int pos = start_pos_for_loop; pos < total_steps; ++pos) {
    if (pos >= config_.max_position_embeddings) {
      Logger::warning("Reached max sequence length (" +
                      std::to_string(config_.max_position_embeddings) +
                      "). Stopping.");
      break;
    }

    int input_token_id;

    if (pos == num_prompt_tokens && start_pos_for_loop == num_prompt_tokens) {
        // This is the first token *after* a successful prefill.
        // `next_token_id` was already sampled using prefill's logits from the *last prompt token*.
        // This `next_token_id` is the actual input for the current position `pos`.
        input_token_id = next_token_id; 
        Logger::debug("[Generate Loop] First token post-prefill. Using prefill's next_token_id: " + std::to_string(input_token_id) + " for pos " + std::to_string(pos));
    } else {
        // Standard iterative logic:
        // If prefill didn't run (start_pos_for_loop == 0):
        //   For pos < num_prompt_tokens: use prompt token.
        //   For pos >= num_prompt_tokens: use previously sampled next_token_id.
        // If prefill did run (start_pos_for_loop == num_prompt_tokens):
        //   This 'else' block is for pos > num_prompt_tokens, so use previously sampled next_token_id.
        input_token_id = (pos < num_prompt_tokens && start_pos_for_loop == 0) ? tokens[pos] : next_token_id;
        if (start_pos_for_loop == 0 && pos < num_prompt_tokens) {
             Logger::debug("[Generate Loop] No prefill, prompt token. Using tokens[" + std::to_string(pos) + "]: " + std::to_string(input_token_id) + " for pos " + std::to_string(pos));
        } else {
             Logger::debug("[Generate Loop] Standard generation. Using previously sampled next_token_id: " + std::to_string(input_token_id) + " for pos " + std::to_string(pos));
        }
    }
    
    current_data_host = model_->lookup_embedding(input_token_id);
    if (pos == 14 || pos == 15 || pos == 16) {
        log_vector_summary_detailed("[API_CPP GenLoop] current_data_host after lookup_embedding for input_token_id=" + std::to_string(input_token_id),
                                    current_data_host, pos, -100, 8);
    }

    if (current_data_host.empty()) {
        Logger::error("Embedding lookup returned empty vector for token ID: " + std::to_string(input_token_id) + " at pos " + std::to_string(pos));
        break; 
    }

    // Determine if the rest of the pass for this token should go to GPU
    bool use_gpu_path = (config_.num_hidden_layers > config_.num_cpu_offload_layers);

    // Single forward pass logic for the current token
    if (use_gpu_path) {
      // Copy host embeddings to device scratchpad
      gpuErrchk(cudaMemcpy(model_->get_x_dev(), current_data_host.data(), current_data_host.size() * sizeof(float), cudaMemcpyHostToDevice));
      // Call forward_device with device pointer, pos, and KVCache
      logits = model_->forward_device(model_->get_x_dev(), pos, &kv_cache_, nullptr);
    } else {
      // Call forward with host embeddings vector, pos, and KVCache
      logits = model_->forward(current_data_host, pos, &kv_cache_, nullptr);
    }

    // The following block for prompt tokens (pos < num_prompt_tokens -1) is only relevant if prefill is NOT enabled.
    // If prefill IS enabled, start_pos_for_loop will be num_prompt_tokens, so this block is skipped.
    // The 'else if (prefill_enabled && pos == num_prompt_tokens -1)' is also effectively handled by prefill block now.
    // The main logic for sampling is in the 'else' block below.
    
    if (pos < num_prompt_tokens -1 && !prefill_enabled) { 
      // This is a prompt token (but not the last one if no prefill)
      // For GGUF/CPU-only, KVCache is updated inside model_->forward().
      // For GPU layers, KVCache is updated inside model_->forward_device().
      // No explicit sampling needed here, we just process the prompt token.
      Logger::debug("[Generate Loop] Prompt token (no prefill, not last). pos: " + std::to_string(pos) + ". KV Cache updated by forward call.");
      // If this was the *last* prompt token and prefill was off, we'd fall through to sample.
      // But since it's `pos < num_prompt_tokens - 1`, we just continue to the next prompt token.
      // `next_token_id` will be tokens[pos+1] in the next iteration via the `input_token_id` logic.
      // No, this is wrong. If it's a prompt token, `next_token_id` isn't used yet.
      // The `input_token_id` for the next iteration will correctly pick `tokens[pos+1]`.
      // We don't sample, we just ensure KV cache is populated.
    } else if (pos == num_prompt_tokens -1 && !prefill_enabled) { // if it's the last prompt token (and no prefill)
      next_token_id = sample_top_k_top_p_temperature(logits, temperature, top_k, top_p, rng_);
      generated_token_ids.push_back(next_token_id); // Track generated token
      generated_count++;
      { // Scope for log_str
        std::string decoded_str = tokenizer_->decode({next_token_id});
        std::string log_str = "[Generate API Loop EndPromptNoPrefill] Sampled token ID: " + std::to_string(next_token_id) + ", Decoded: \"" + decoded_str + "\"";
        Logger::info(log_str);
      }
    } else { // otherwise, it's a generated token (or first after prefill)
      next_token_id = sample_top_k_top_p_temperature(logits, temperature, top_k, top_p, rng_);
      // DON'T stream yet - token hasn't been processed through model!
      generated_token_ids.push_back(next_token_id); // Track generated token
      generated_count++;
    }

    if (next_token_id == eos_token_id_ && pos >= num_prompt_tokens) { // EOS only if we are generating
      Logger::info("EOS token (" + std::to_string(eos_token_id_) +
                     ") sampled at pos " + std::to_string(pos) + ". Stopping.");
      break;
    }

    // Stream the token that was just processed (input_token_id), not the one just sampled (next_token_id)
    generated_stream_ << tokenizer_->decode({input_token_id}, false);
    generated_text_for_api_return_ += tokenizer_->decode({input_token_id}, false);

    if (generated_count >= steps) { // steps is max new tokens
      Logger::info("Reached max generation steps (" + std::to_string(steps) + "). Stopping.");
      break;
    }

    // KVCache seq_len update for single token pass (already handled by batch prefill for its tokens)
    // This needs to happen *after* the forward pass for the current 'pos' has updated the cache for 'pos'.
    // So, after processing 'pos', the cache now contains information up to and including 'pos'.
    // The length of the sequence in the cache is pos + 1.
    if (!prefill_enabled || pos >= num_prompt_tokens) { // only update if not prefill, or if we are past prompt token processing in prefill case
        kv_cache_.seq_len = pos + 1; 
        // Logger::debug("[Generate Loop] KVCache seq_len updated to: " + std::to_string(kv_cache_.seq_len) + " after processing pos " + std::to_string(pos));
    }
  }

  // Log all generated IDs before decoding
  std::string generated_ids_str = "[Generated IDs Pre-Decode] ";
  for(int gen_id : generated_token_ids) {
    generated_ids_str += std::to_string(gen_id) + " ";
  }
  Logger::debug(generated_ids_str);

  std::string result = tokenizer_->decode(generated_token_ids, true);
  Logger::info("Generated response: " + result);

  auto t_end = std::chrono::high_resolution_clock::now(); // End timing
  double time_taken_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
  
  // Create a string with the desired precision for the time
  std::ostringstream time_ss;
  time_ss << std::fixed << std::setprecision(4) << time_taken_ms;
  Logger::info("[INFO] Total generation processing time: " + time_ss.str() + " ms");

  return result;
}

}  // namespace tinyllama