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

  Logger::info("[Generate API] User prompt: \"" + user_prompt + "\", System prompt: \"" + system_prompt_arg + "\", Steps: " + std::to_string(steps));

  if (!model_ || !tokenizer_) {
    throw std::runtime_error("Model or tokenizer not loaded.");
  }

  std::string final_prompt_for_encoding;
  bool used_chat_template = false;

  // Priority 1: GGUF Chat Template from Tokenizer
  if (tokenizer_ && !tokenizer_->get_gguf_chat_template().empty()) {
    Logger::info("[Generate API] Using GGUF chat template from tokenizer.");
    final_prompt_for_encoding = tokenizer_->apply_chat_template(user_prompt, system_prompt_arg, config_);
    used_chat_template = true;
  } 
  // Priority 2: Llama 3 family specific template (handled by apply_chat_template's fallback)
  else if (config_.tokenizer_family == ModelConfig::TokenizerFamily::LLAMA3_TIKTOKEN) {
    Logger::info("[Generate API] Llama 3 tokenizer family detected, using apply_chat_template.");
    final_prompt_for_encoding = tokenizer_->apply_chat_template(user_prompt, system_prompt_arg, config_);
    used_chat_template = true;
  }
  // Priority 3: Legacy Q/A formatting if CLI hint is true AND no advanced template was used
  else if (apply_q_a_format_cli_hint) {
    Logger::info("[Generate API] No GGUF/Llama3 template, applying legacy Q/A formatting due to CLI hint.");
    std::string temp_prompt = user_prompt;
    if (!system_prompt_arg.empty()) {
        temp_prompt = system_prompt_arg + "\n\nQ: " + user_prompt + "\nA:"; // Basic combination
    } else {
        temp_prompt = "Q: " + user_prompt + "\nA:";
    }
    final_prompt_for_encoding = temp_prompt;
  }
  else {
    Logger::info("[Generate API] No GGUF/Llama3 template and Q/A formatting hint is false. Using user prompt as is (prepending system prompt if available).");
    if (!system_prompt_arg.empty()) {
        final_prompt_for_encoding = system_prompt_arg + "\n\n" + user_prompt; // Basic prepend
    } else {
        final_prompt_for_encoding = user_prompt;
    }
  }
  
  Logger::debug("[Generate API] Final prompt for encoding (first 100 chars): \"" + final_prompt_for_encoding.substr(0, 100) + "\"");

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
  kv_cache_.seq_len = 0; // Reset KVCache for new sequence

  std::vector<float> current_data_host; // To hold embedding or output of CPU layers
  int start_pos_for_loop = 0;


#ifdef HAS_CUDA
  int num_total_layers_model = config_.num_hidden_layers;
  int num_cpu_layers_model = config_.num_cpu_offload_layers;
  int num_gpu_layers_model = num_total_layers_model - num_cpu_layers_model;

  if (num_gpu_layers_model > 0 && num_prompt_tokens > 0) {
      Logger::info("[Generate API] Attempting GPU batch prefill for " + std::to_string(num_prompt_tokens) + " prompt tokens.");
      
      std::vector<float> batch_intermediate_activations_host; // Will hold initial embeddings or output of CPU layers
      batch_intermediate_activations_host.reserve(num_prompt_tokens * config_.hidden_size);

      // 1. Initial Embedding Lookup
      for (int i = 0; i < num_prompt_tokens; ++i) {
          std::vector<float> current_token_embedding = model_->lookup_embedding(tokens[i]);
          if (current_token_embedding.empty()) {
              Logger::error("Failed to get embedding for prompt token ID " + std::to_string(tokens[i]) + " at index " + std::to_string(i));
              return ""; 
          }
          batch_intermediate_activations_host.insert(batch_intermediate_activations_host.end(), current_token_embedding.begin(), current_token_embedding.end());
      }

      // 2. CPU Layer Batch Processing (New Step)
      if (num_cpu_layers_model > 0) {
          Logger::info("[Generate API] Processing " + std::to_string(num_cpu_layers_model) + " CPU layers in batch for prefill.");
          // This function needs to be implemented in TinyLlamaModel.
          // It should update kv_cache_ for the CPU layers it processes.
          // The '0' for start_pos assumes this batch processing for CPU layers starts from sequence position 0 for this batch.
          batch_intermediate_activations_host = model_->forward_cpu_batch(
              batch_intermediate_activations_host, 
              num_prompt_tokens, 
              num_cpu_layers_model, 
              0, // start_pos for this batch segment in CPU processing
              &kv_cache_
          ); 
          if (batch_intermediate_activations_host.empty()) {
              Logger::error("CPU batch processing for prefill returned empty activations.");
              return "";
          }
          Logger::info("[Generate API] CPU layers processed for prefill. Activations size: " + std::to_string(batch_intermediate_activations_host.size()));
          // After CPU layers, kv_cache_.seq_len would have been updated by forward_cpu_batch up to num_prompt_tokens
          // if it processed the entire prompt.
      }

      // 3. Data Transfer to GPU (activations from CPU layers, or initial embeddings if no CPU layers)
      float* d_batch_gpu_input_activations = nullptr;
      if (!batch_intermediate_activations_host.empty()) {
          gpuErrchk(cudaMalloc(&d_batch_gpu_input_activations, batch_intermediate_activations_host.size() * sizeof(float)));
          gpuErrchk(cudaMemcpy(d_batch_gpu_input_activations, batch_intermediate_activations_host.data(), batch_intermediate_activations_host.size() * sizeof(float), cudaMemcpyHostToDevice));
          
          cudaError_t err_sync_after_memcpy = cudaDeviceSynchronize(); // Existing debug log location
          if (err_sync_after_memcpy != cudaSuccess) {
              Logger::error("[Generate API DEBUG] CUDA error after d_batch_gpu_input_activations setup (sync): " + std::string(cudaGetErrorString(err_sync_after_memcpy)));
          }
          cudaError_t err_last_after_memcpy = cudaGetLastError();
          if (err_last_after_memcpy != cudaSuccess) {
              Logger::error("[Generate API DEBUG] CUDA error after d_batch_gpu_input_activations setup (last): " + std::string(cudaGetErrorString(err_last_after_memcpy)));
          } else {
              Logger::info("[Generate API DEBUG] d_batch_gpu_input_activations setup for prefill SYNC OK. Pointer: " + Logger::ptrToString(d_batch_gpu_input_activations));
          }
      } else {
          Logger::error("Batch intermediate activations host vector is empty before GPU prefill stage.");
          return "";
      }

      // kv_cache_.seq_len for the GPU part should start from 0 if CPU layers didn't touch it,
      // or num_prompt_tokens if CPU layers fully processed the prompt.
      // The current_model_pos for forward_device_batch_prefill is the starting *offset* for RoPE and KV cache updates
      // for the tokens being processed *by the GPU layers*.
      // If CPU layers handled tokens 0..N-1, GPU layers handle the same tokens 0..N-1 but for *their* respective layers.
      // So, current_model_pos for RoPE in GPU layers should still be based on the token's actual position in sequence.
      // The KVCache update inside forward_device_batch_prefill uses current_model_pos as the offset.
      // This seems correct: current_model_pos = 0, as this function processes the whole prompt batch from its perspective.
      // The KVCache itself tracks total length.
      
      // Important: If CPU layers modified kv_cache_.seq_len, it should be consistent here.
      // Let's assume forward_cpu_batch sets kv_cache_.seq_len correctly if it runs.
      // If no CPU layers ran, kv_cache_.seq_len is still 0 here.
      // The `forward_device_batch_prefill` will update KV cache from `current_model_pos`.
      
      logits = model_->forward_device_batch_prefill(
          d_batch_gpu_input_activations,
          num_prompt_tokens,
          0, // current_model_pos for this batch segment for GPU layers
          &kv_cache_,
          0 // default stream
      );
      
      if (d_batch_gpu_input_activations) {
          gpuErrchk(cudaFree(d_batch_gpu_input_activations));
      }

      if (logits.empty()) {
          Logger::error("forward_device_batch_prefill returned empty logits.");
          return "";
      }
      
      kv_cache_.seq_len = num_prompt_tokens; // KVCache is now filled up to num_prompt_tokens
      start_pos_for_loop = num_prompt_tokens; // Main loop starts after the prompt
      
      // Logits for the first token *after* the prompt are now available.
      // next_token_id will be sampled from these logits before the loop begins if prefill was successful.
      // This sampling is to prepare `next_token_id` for the very first iteration of the adjusted loop.
      next_token_id = sample_top_k_top_p_temperature(logits, temperature, top_k, top_p, rng_);
      Logger::info("[Generate API] Prefill successful. Next token ID (for pos=" + std::to_string(num_prompt_tokens) + "): " + std::to_string(next_token_id));

  } else { // This 'else' means (num_gpu_layers_model == 0 || num_prompt_tokens == 0) when HAS_CUDA is defined
      Logger::info("[Generate API] Using CPU for prefill (either no GPU layers, or no prompt tokens for GPU prefill). Prompt tokens: " + std::to_string(num_prompt_tokens));
      if (num_prompt_tokens > 0) {
          std::vector<float> prompt_embeddings_flat;
          prompt_embeddings_flat.reserve(num_prompt_tokens * config_.hidden_size);
          for (int i = 0; i < num_prompt_tokens; ++i) {
              std::vector<float> current_token_embedding = model_->lookup_embedding(tokens[i]);
              if (current_token_embedding.empty()) {
                  Logger::error("Failed to get embedding for prompt token ID " + std::to_string(tokens[i]) + " at index " + std::to_string(i));
                  return ""; 
              }
              prompt_embeddings_flat.insert(prompt_embeddings_flat.end(), current_token_embedding.begin(), current_token_embedding.end());
          }

          // All layers are processed by CPU in this path
          std::vector<float> cpu_prefill_output_activations = model_->forward_cpu_batch(
              prompt_embeddings_flat,
              num_prompt_tokens,
              config_.num_hidden_layers, // Process ALL layers on CPU
              0, // start_pos for this batch segment in CPU processing
              &kv_cache_
          );

          if (cpu_prefill_output_activations.empty()) {
              Logger::error("CPU batch prefill (all layers) returned empty activations.");
              return "";
          }
          Logger::info("[Generate API] CPU batch prefill (all layers) processed. Activations size: " + std::to_string(cpu_prefill_output_activations.size()));
          
          // Get logits for all tokens in the prefill batch
          std::vector<float> all_logits_from_prefill = model_->forward_cpu_logits_batch(
              cpu_prefill_output_activations,
              num_prompt_tokens
          );

          if (all_logits_from_prefill.empty() || all_logits_from_prefill.size() != (size_t)num_prompt_tokens * config_.vocab_size) {
              Logger::error("CPU batch prefill logit calculation failed or returned unexpected size.");
              return "";
          }

          // Extract logits for the LAST token of the prompt to predict the next token
          logits.resize(config_.vocab_size);
          size_t last_token_logits_offset = (size_t)(num_prompt_tokens - 1) * config_.vocab_size;
          std::copy(all_logits_from_prefill.begin() + last_token_logits_offset,
                    all_logits_from_prefill.begin() + last_token_logits_offset + config_.vocab_size,
                    logits.begin());

          kv_cache_.seq_len = num_prompt_tokens; // KVCache is now filled up to num_prompt_tokens
          start_pos_for_loop = num_prompt_tokens; // Main loop starts after the prompt
          next_token_id = sample_top_k_top_p_temperature(logits, temperature, top_k, top_p, rng_);
          Logger::info("[Generate API] CPU Prefill successful. Next token ID (for pos=" + std::to_string(num_prompt_tokens) + "): " + std::to_string(next_token_id));
      } else {
          start_pos_for_loop = 0; // No prompt tokens, standard loop from beginning
      }
  }
#else // No HAS_CUDA defined - Pure CPU path from the start
  Logger::info("[Generate API] No CUDA. Using CPU for prefill. Prompt tokens: " + std::to_string(num_prompt_tokens));
  if (num_prompt_tokens > 0) {
      std::vector<float> prompt_embeddings_flat;
      prompt_embeddings_flat.reserve(num_prompt_tokens * config_.hidden_size);
      for (int i = 0; i < num_prompt_tokens; ++i) {
          std::vector<float> current_token_embedding = model_->lookup_embedding(tokens[i]);
          if (current_token_embedding.empty()) {
              Logger::error("Failed to get embedding for prompt token ID " + std::to_string(tokens[i]) + " at index " + std::to_string(i));
              return ""; 
          }
          prompt_embeddings_flat.insert(prompt_embeddings_flat.end(), current_token_embedding.begin(), current_token_embedding.end());
      }

      std::vector<float> cpu_prefill_output_activations = model_->forward_cpu_batch(
          prompt_embeddings_flat,
          num_prompt_tokens,
          config_.num_hidden_layers, // Process ALL layers on CPU
          0, // start_pos
          &kv_cache_
      );

      if (cpu_prefill_output_activations.empty()) {
          Logger::error("CPU batch prefill (all layers, no CUDA path) returned empty activations.");
          return "";
      }
      Logger::info("[Generate API] CPU batch prefill (all layers, no CUDA path) processed. Activations size: " + std::to_string(cpu_prefill_output_activations.size()));
      
      std::vector<float> all_logits_from_prefill = model_->forward_cpu_logits_batch(
          cpu_prefill_output_activations,
          num_prompt_tokens // Corrected: was num_tokens_in_batch
      );

      if (all_logits_from_prefill.empty() || all_logits_from_prefill.size() != (size_t)num_prompt_tokens * config_.vocab_size) {
          Logger::error("CPU batch prefill logit calculation failed or returned unexpected size (no CUDA path).");
          return "";
      }

      logits.resize(config_.vocab_size);
      size_t last_token_logits_offset = (size_t)(num_prompt_tokens - 1) * config_.vocab_size;
      std::copy(all_logits_from_prefill.begin() + last_token_logits_offset,
                all_logits_from_prefill.begin() + last_token_logits_offset + config_.vocab_size,
                logits.begin());

      kv_cache_.seq_len = num_prompt_tokens;
      start_pos_for_loop = num_prompt_tokens;
      next_token_id = sample_top_k_top_p_temperature(logits, temperature, top_k, top_p, rng_);
      Logger::info("[Generate API] CPU Prefill successful (no CUDA path). Next token ID (for pos=" + std::to_string(num_prompt_tokens) + "): " + std::to_string(next_token_id));
  } else {
      start_pos_for_loop = 0; // No prompt tokens, standard loop from beginning
  }
#endif


  for (int pos = start_pos_for_loop; pos < total_steps; ++pos) {
    if (pos >= config_.max_position_embeddings) {
      Logger::warning("Reached max sequence length (" +
                      std::to_string(config_.max_position_embeddings) +
                      "). Stopping.");
      break;
    }

    int input_token_id;

    // If we are at the first position *after* a successful prefill,
    // `next_token_id` was already sampled using prefill's logits.
    // `tokens` does not yet contain this `next_token_id`.
    // The loop's job starts by setting this as the current `input_token_id`
    // and then proceeding to generate the *next* one.
    if (pos == num_prompt_tokens && start_pos_for_loop == num_prompt_tokens) {
        // `next_token_id` is from prefill's logits (sampled above if prefill happened)
        input_token_id = next_token_id; 
        // It's crucial that `next_token_id` was validly sampled if we reach here.
        // The prefill block should ensure `next_token_id` is set if `logits` were successfully obtained.
    } else {
        // Standard iterative logic:
        // For pos < num_prompt_tokens (only if prefill didn't run, i.e. start_pos_for_loop = 0): use prompt token.
        // For pos >= num_prompt_tokens (if prefill didn't run OR for subsequent steps after prefill): use previously sampled next_token_id.
        input_token_id = (pos < num_prompt_tokens) ? tokens[pos] : next_token_id;
    }
    
    // Only calculate logits if they weren't provided by prefill for this specific position.
    // This means if (pos == num_prompt_tokens && start_pos_for_loop == num_prompt_tokens),
    // we use the `logits` from prefill (already sampled into next_token_id for this iteration's input).
    // For all other cases (iterative prompt processing or subsequent generation steps), calculate new logits.
    if (!(pos == num_prompt_tokens && start_pos_for_loop == num_prompt_tokens)) {
        current_data_host = model_->lookup_embedding(input_token_id);
        if (current_data_host.empty()) {
            Logger::error("Failed to get embedding for token " + std::to_string(input_token_id) + " at pos " + std::to_string(pos));
            break;
        }

        int num_total_layers = config_.num_hidden_layers;
        int num_cpu = config_.num_cpu_offload_layers;
        int num_gpu = num_total_layers - num_cpu;

        if (num_cpu > 0) {
            Logger::debug("[Generate] Processing " + std::to_string(num_cpu) + " CPU layers for pos " + std::to_string(pos));
            current_data_host = model_->forward(current_data_host, pos, &kv_cache_, nullptr);
        }

#ifdef HAS_CUDA
        if (num_gpu > 0) {
            Logger::debug("[Generate] Processing " + std::to_string(num_gpu) + " GPU layers for pos " + std::to_string(pos));
            float* x_dev_ptr = model_->get_x_dev();
            if (!x_dev_ptr) { 
                Logger::fatal("model_->x_dev_ is null before GPU forward pass!"); 
                throw std::runtime_error("Critical error: x_dev_ is null in GPU pipeline.");
            }
            if (current_data_host.empty()) { // Check if current_data_host has valid data
                Logger::fatal("current_data_host is empty before cudaMemcpy to x_dev_ at pos " + std::to_string(pos)); 
                throw std::runtime_error("Critical error: current_data_host is empty for GPU pipeline at pos " + std::to_string(pos));
            }
            gpuErrchk(cudaMemcpy(x_dev_ptr, current_data_host.data(), current_data_host.size() * sizeof(float), cudaMemcpyHostToDevice));
            
            logits = model_->forward_device(x_dev_ptr, pos, &kv_cache_, nullptr);
        } else if (num_cpu > 0 && num_gpu == 0) {
            logits = current_data_host;
        } else if (num_total_layers == 0) {
            Logger::warning("No layers in the model (HAS_CUDA path).");
            logits.assign(config_.vocab_size > 0 ? config_.vocab_size : 1, 0.0f);
        } else if (num_cpu == 0 && num_gpu == 0 && num_total_layers > 0) {
             Logger::error("Inconsistent state: HAS_CUDA, num_cpu=0, num_gpu=0, but num_total_layers > 0. Defaulting to CPU path for safety.");
             current_data_host = model_->lookup_embedding(input_token_id);
             logits = model_->forward(current_data_host, pos, &kv_cache_, nullptr);
        }
#else // No CUDA
        if (num_total_layers > 0) {
            // If prefill didn't run, num_cpu could be 0 if all layers were meant for GPU (but CUDA not avail).
            // Or num_cpu > 0 if some were designated for CPU.
            // model->forward needs to be called if current_data_host is just an embedding.
             if (num_cpu == 0) { // Only call model->forward if it wasn't called for CPU layers above
                Logger::debug("[Generate] NO_CUDA: Processing all " + std::to_string(num_total_layers) + " layers on CPU for pos " + std::to_string(pos));
                logits = model_->forward(current_data_host, pos, &kv_cache_, nullptr);
            } else { // num_cpu > 0, means model->forward was already called and current_data_host has final logits
                logits = current_data_host;
            }
        } else {
            Logger::warning("No layers in the model (NO_CUDA path).");
            logits.assign(config_.vocab_size > 0 ? config_.vocab_size : 1, 0.0f);
        }
#endif
        // Update KVCache seq_len *after* the forward pass for this position
        kv_cache_.seq_len = pos + 1;
    }
    // If (pos == num_prompt_tokens && start_pos_for_loop == num_prompt_tokens),
    // logits are from prefill. kv_cache_.seq_len was already set to num_prompt_tokens.
    // For the *next* iteration (pos = num_prompt_tokens + 1), kv_cache_.seq_len will be updated
    // after its forward pass. This seems correct.

    if (logits.empty()) {
      Logger::error("Logits vector is empty after forward pass at pos " + std::to_string(pos));
      break;
    }

    // Generation part: sample next token if current 'pos' is for a token to be generated,
    // or if it's the last token of the prompt (and prefill didn't run for the prompt).
    // If prefill ran, `start_pos_for_loop` is `num_prompt_tokens`.
    // The first `pos` value in the loop would be `num_prompt_tokens`.
    // So `pos >= num_prompt_tokens -1` becomes `num_prompt_tokens >= num_prompt_tokens -1` which is true.
    // If prefill didn't run, `start_pos_for_loop` is `0`.
    // The condition `pos >= num_prompt_tokens -1` will ensure we start sampling
    // after processing the last prompt token.
    if (pos >= num_prompt_tokens -1) { // This condition correctly handles both prefill and non-prefill cases
      next_token_id = sample_top_k_top_p_temperature(logits, temperature, top_k, top_p, rng_);
      
      // If this is the first generated token (i.e., pos == num_prompt_tokens if prefill ran,
      // or pos == num_prompt_tokens-1 if prefill didn't run but we just processed last prompt token),
      // `input_token_id` for this iteration was the one *being processed*.
      // `next_token_id` is the one *to be added* to `tokens`.

      std::string decoded_next_token = tokenizer_->decode({next_token_id}, true); 
      Logger::info("[Generate] Pos=" + std::to_string(pos) + 
                   ", InputID=" + std::to_string(input_token_id) + // This is the token that *led* to these logits
                   ", Sampled NextID: " + std::to_string(next_token_id) + 
                   ", Decoded Next: '" + decoded_next_token + "'");

      if (next_token_id == eos_token_id_) {
        Logger::info("EOS token generated (ID: " + std::to_string(next_token_id) + "). Stopping.");
        break;
      }

      tokens.push_back(next_token_id); // Add the *newly sampled* token
      generated_count++;

      if (generated_count >= steps) { // `steps` is max *new* tokens
        Logger::info("Max steps (" + std::to_string(steps) + ") reached. Stopping.");
        break;
      }
    }
  }

  std::vector<int> generated_only_ids;
  if (num_prompt_tokens < tokens.size()) { // Ensure there are generated tokens
    generated_only_ids.assign(tokens.begin() + num_prompt_tokens, tokens.end());
  }

  // Log all generated IDs before decoding
  std::string generated_ids_str = "[Generated IDs Pre-Decode] ";
  for(int gen_id : generated_only_ids) {
    generated_ids_str += std::to_string(gen_id) + " ";
  }
  Logger::debug(generated_ids_str);

  std::string result = tokenizer_->decode(generated_only_ids, true);
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