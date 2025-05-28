#include "model_config.h"

#include "logger.h"
#include "gguf_parser.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <variant>

ModelConfig parse_model_config(const nlohmann::json& json) {
  ModelConfig cfg;
  cfg.hidden_size = json.value("hidden_size", 0);
  cfg.intermediate_size = json.value("intermediate_size", 0);
  cfg.num_attention_heads = json.value("num_attention_heads", 0);
  cfg.num_key_value_heads = json.value("num_key_value_heads", 0);
  cfg.num_hidden_layers = json.value("num_hidden_layers", 0);
  cfg.vocab_size = json.value("vocab_size", 0);
  cfg.max_position_embeddings = json.value("max_position_embeddings", 0);
  cfg.rms_norm_eps = json.value("rms_norm_eps", 1e-5f);
  cfg.rope_theta = json.value("rope_theta", 10000.0f);
  cfg.hidden_act = json.value("hidden_act", "silu");
  cfg.torch_dtype = json.value("torch_dtype", "bfloat16");
  cfg.bos_token_id = json.value("bos_token_id", 1);
  cfg.eos_token_id = json.value("eos_token_id", 2);
  cfg.unk_token_id = json.value("unk_token_id", -1);
  cfg.pad_token_id = json.value("pad_token_id", -1); 

  // Infer Architecture if available
  if (json.contains("architectures") && json["architectures"].is_array() && !json["architectures"].empty()) {
      // Take the first architecture string if multiple are listed
      cfg.architecture = json["architectures"][0].get<std::string>();
  } else {
      cfg.architecture = "unknown"; 
  }
  cfg.model_name = json.value("model_type", cfg.architecture); // Use model_type or fallback to architecture

  
  Logger::info("[parse_json_config] Inferring tokenizer family for SafeTensors. Arch: '" + cfg.architecture + "', Vocab: " + std::to_string(cfg.vocab_size));
  bool is_llama3_vocab_size_json = (cfg.vocab_size == 128256);
  bool is_llama3_arch_hint_json = (cfg.architecture.find("LlamaForCausalLM") != std::string::npos && // Llama 3 often uses this
                              cfg.architecture.find("Llama2") == std::string::npos); // Exclude Llama 2 explicitly if needed

  if (is_llama3_vocab_size_json && is_llama3_arch_hint_json) {
      cfg.tokenizer_family = ModelConfig::TokenizerFamily::LLAMA3_TIKTOKEN;
      Logger::info("[parse_json_config] Result: Identified LLAMA3_TIKTOKEN (vocab size + arch hint).");
       if (cfg.rope_theta == 10000.0f) { 
            float llama3_rope_candidate = json.value("rope_theta", 500000.0f); // Check rope_theta in config.json
            if (llama3_rope_candidate > 10000.0f) {
                cfg.rope_theta = llama3_rope_candidate;
                Logger::info("[parse_json_config] Adjusted rope_theta to " + std::to_string(cfg.rope_theta) + " for Llama 3 model (was 10000.0).");
            }
       }
  } else if (cfg.vocab_size == 32000 || cfg.architecture.find("Llama") != std::string::npos) { // Common for Llama 1/2/TinyLlama
      cfg.tokenizer_family = ModelConfig::TokenizerFamily::LLAMA_SENTENCEPIECE;
      Logger::info("[parse_json_config] Result: Identified LLAMA_SENTENCEPIECE (vocab size or arch hint).");
  } else {
      cfg.tokenizer_family = ModelConfig::TokenizerFamily::UNKNOWN;
      Logger::warning("[parse_json_config] Result: UNKNOWN tokenizer family.");
  }
  

  return cfg;
}

ModelConfig parse_model_config_from_gguf(const GGUFData& gguf) {
  ModelConfig config;
  Logger::info("[parse_gguf_config] Entered function.");

  auto get_meta_string = [&](const std::string& key,
                             const std::string& default_val) -> std::string {
    auto it = gguf.metadata.find(key);
    if (it != gguf.metadata.end() &&
        std::holds_alternative<std::string>(it->second)) {
      return std::get<std::string>(it->second);
    }
    return default_val;
  };

  auto get_meta_value = [&](const std::string& key, auto default_value) {
    using TargetType = typename std::decay<decltype(default_value)>::type;
    auto it = gguf.metadata.find(key);
    if (it != gguf.metadata.end()) {
      return std::visit(
          [&](const auto& val) -> TargetType {
            using T = std::decay_t<decltype(val)>;

            if constexpr (std::is_integral_v<TargetType>) {
              if constexpr (std::is_integral_v<T> && !std::is_same_v<T, bool>) {
                if constexpr (std::is_unsigned_v<T> &&
                              std::is_signed_v<TargetType>) {
                  if (val > static_cast<std::make_unsigned_t<TargetType>>(
                                std::numeric_limits<TargetType>::max())) {
                    Logger::warning("Metadata key '" + key + "' value " +
                                    std::to_string(val) +
                                    " overflows TargetType. Using default.");
                    return default_value;
                  }
                }

                else if constexpr (std::is_signed_v<T> &&
                                   std::is_signed_v<TargetType> &&
                                   sizeof(T) > sizeof(TargetType)) {
                  if (val > static_cast<T>(
                                std::numeric_limits<TargetType>::max()) ||
                      val < static_cast<T>(
                                std::numeric_limits<TargetType>::lowest())) {
                    Logger::warning("Metadata key '" + key + "' value " +
                                    std::to_string(val) +
                                    " overflows TargetType. Using default.");
                    return default_value;
                  }
                }
                return static_cast<TargetType>(val);
              }
            } else if constexpr (std::is_floating_point_v<TargetType>) {
              if constexpr (std::is_floating_point_v<T>) {
                return static_cast<TargetType>(val);
              }
            } else if constexpr (std::is_same_v<TargetType, bool>) {
              if constexpr (std::is_same_v<T, bool>) {
                return val;
              }
            } else if constexpr (std::is_same_v<TargetType, std::string>) {
              if constexpr (std::is_same_v<T, std::string>) {
                return val;
              }
            }
            Logger::warning("Metadata key '" + key +
                            "' has stored type incompatible with requested "
                            "TargetType. Using default.");
            return default_value;
          },
          it->second);
    } else {
      return default_value;
    }
  };

  config.vocab_size = get_meta_value("tokenizer.ggml.vocab_size",
                                     get_meta_value("llama.vocab_size", 32000));
  config.hidden_size = get_meta_value("llama.embedding_length", 4096);
  config.intermediate_size = get_meta_value("llama.feed_forward_length", 11008);
  config.num_attention_heads = get_meta_value("llama.attention.head_count", 32);
  config.num_hidden_layers = get_meta_value("llama.block_count", 32);
  config.num_key_value_heads = get_meta_value("llama.attention.head_count_kv",
                                              config.num_attention_heads);
  config.max_position_embeddings = get_meta_value("llama.context_length", 4096);
  if (config.max_position_embeddings == 0 ||
      config.max_position_embeddings > 8192) {
    Logger::warning("max_position_embeddings from GGUF is " +
                    std::to_string(config.max_position_embeddings) +
                    ", overriding to sensible default (2048)");
    config.max_position_embeddings = 2048;
  }
  config.rms_norm_eps =
      get_meta_value("llama.attention.layer_norm_rms_epsilon", 1e-5f);
  config.rope_theta = get_meta_value("llama.rope.freq_base", 10000.0f);
  config.hidden_act = "silu";
  config.bos_token_id = get_meta_value("tokenizer.ggml.bos_token_id", -1);
  config.eos_token_id = get_meta_value("tokenizer.ggml.eos_token_id", -1);
  config.unk_token_id = get_meta_value("tokenizer.ggml.unk_token_id", -1);
  config.pad_token_id = get_meta_value("tokenizer.ggml.padding_token_id", -1);

  config.architecture = get_meta_string("general.architecture", "unknown");
  config.model_name = get_meta_string("general.name", "unknown");
  bool has_pre_key = gguf.metadata.count("tokenizer.ggml.pre");
  bool has_merges = !gguf.tokenizer_merges.empty();

  Logger::info("[parse_gguf_config] Architecture: " + config.architecture +
               ", Vocab Size: " + std::to_string(config.vocab_size) +
               ", Has Merges: " + (has_merges ? "Yes" : "No"));

  
  Logger::info("[parse_gguf_config] Identifying tokenizer family...");
  bool is_llama3_arch_hint = (config.architecture.find("llama3") != std::string::npos ||
                         config.architecture.find("Llama-3") != std::string::npos ||
                         config.architecture.find("Meta-Llama-3") != std::string::npos);
  bool is_llama3_vocab_size = (config.vocab_size == 128256);
  std::string ggml_tokenizer_model = get_meta_string("tokenizer.ggml.model", "");
  bool is_tiktoken_style_tokenizer_model = (ggml_tokenizer_model == "gpt2");

  Logger::info("[parse_gguf_config] L3 Hints: arch_hint=" + std::string(is_llama3_arch_hint ? "Y":"N") +
                 ", vocab_size_match=" + std::string(is_llama3_vocab_size ? "Y":"N") +
                 ", has_merges=" + std::string(has_merges ? "Y":"N") +
                 ", ggml_tokenizer_model_key='" + ggml_tokenizer_model + "' (is_tiktoken_style: " + std::string(is_tiktoken_style_tokenizer_model ? "Y":"N") + ")" );

  if (has_merges && is_llama3_vocab_size && is_tiktoken_style_tokenizer_model) {
    config.tokenizer_family = ModelConfig::TokenizerFamily::LLAMA3_TIKTOKEN;
    Logger::info("[parse_gguf_config] Result: Identified LLAMA3_TIKTOKEN (merges + vocab_size + ggml_tokenizer_model='gpt2'). Architecture string was: '" + config.architecture + "'");
    if (!is_llama3_arch_hint && config.architecture == "llama") {
         Logger::info("[parse_gguf_config] Note: Classified as Llama 3 based on tokenizer/vocab, but arch string was 'llama'.");
    }
    if (config.rope_theta == 10000.0f) { 
         float llama3_rope_candidate = get_meta_value("llama.rope.freq_base", 500000.0f); 
         if (llama3_rope_candidate > 10000.0f) {
             config.rope_theta = llama3_rope_candidate;
             Logger::info("[parse_gguf_config] Adjusted rope_theta to " + std::to_string(config.rope_theta) + " for Llama 3 model (was 10000.0).");
         }
    }
  } else if (config.architecture == "llama" || config.architecture.find("Llama-2") != std::string::npos || config.architecture.find("TinyLlama") != std::string::npos) {
    config.tokenizer_family = ModelConfig::TokenizerFamily::LLAMA_SENTENCEPIECE;
     Logger::info("[parse_gguf_config] Result: Identified LLAMA_SENTENCEPIECE based on architecture: '" + config.architecture + "'");
  } else {
    config.tokenizer_family = ModelConfig::TokenizerFamily::UNKNOWN;
     Logger::info("[parse_gguf_config] Result: UNKNOWN tokenizer family for architecture: '" + config.architecture + "'");
  }

  // Existing chat_template_type and pre_tokenizer_type logic based on architecture and pre_key
  if (config.model_name.find("TinyLlama") != std::string::npos ||
      (config.architecture == "llama" && has_pre_key)) {
    config.chat_template_type = "tinyllama";
  } else if (config.architecture == "llama" && !has_pre_key) {
    config.chat_template_type = "llama2";
  } else {
    config.chat_template_type = "unknown";
    Logger::warning("Could not determine chat template type for arch='" +
                    config.architecture + "', name='" + config.model_name +
                    "'.");
  }

  if (has_pre_key) {
    config.pre_tokenizer_type =
        get_meta_string("tokenizer.ggml.pre", "unknown");
  } else if (config.architecture == "llama") {
    config.pre_tokenizer_type = "llama";
  } else {
    config.pre_tokenizer_type = "unknown";
  }
  Logger::info("Determined config: architecture='" + config.architecture +
               "', model_name='" + config.model_name + "', chat_template='" +
               config.chat_template_type + "', pre_tokenizer='" +
               config.pre_tokenizer_type + "'");

  if (config.model_name == "llama" && config.pre_tokenizer_type != "llama") {
    config.chat_template_type = "llama2";
    Logger::info(
        "Inferred chat_template_type='llama2' based on model_type and "
        "missing/different pre_tokenizer_type.");
  }

  auto template_it = gguf.metadata.find("tokenizer.chat_template");
  if (template_it != gguf.metadata.end() &&
      std::holds_alternative<std::string>(template_it->second)) {
    config.chat_template_string = std::get<std::string>(template_it->second);
    Logger::info("Found tokenizer.chat_template in metadata.");

  } else {
    Logger::info(
        "tokenizer.chat_template not found or not a string in metadata. Will "
        "use fallback logic.");
    config.chat_template_string = "";
  }
  if (config.chat_template_type == "unknown") {
    if (config.model_name == "llama" && config.pre_tokenizer_type != "llama") {
      config.chat_template_type = "llama2";
      Logger::info(
          "Inferred chat_template_type='llama2' based on model name and "
          "missing/different pre_tokenizer_type.");
    } else if (config.tokenizer_family == ModelConfig::TokenizerFamily::LLAMA3_TIKTOKEN) {
        Logger::info("Llama 3 model identified. Chat template will primarily rely on 'tokenizer.chat_template' from GGUF if present.");
        // Set a generic type for now, actual application will use the string.
        if (gguf.metadata.count("tokenizer.chat_template")) {
            config.chat_template_type = "llama3_gguf_direct"; 
        } else {
            config.chat_template_type = "llama3_fallback"; // Or some other indicator
            Logger::warning("Llama 3 model detected, but 'tokenizer.chat_template' not found in GGUF metadata.");
        }
    }
  }

  Logger::info(std::string("[parse_gguf_config] Finished parsing. Returning config. Family: ") + 
                (config.tokenizer_family == ModelConfig::TokenizerFamily::LLAMA3_TIKTOKEN ? "L3_TIKTOKEN" : 
                 (config.tokenizer_family == ModelConfig::TokenizerFamily::LLAMA_SENTENCEPIECE ? "L2_SPM" : "UNKNOWN")));
  return config;
} 