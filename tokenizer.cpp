#include "tokenizer.h"

#include <algorithm>  
#include <cctype>
#include <fstream>
#include <iostream>  
#include <map>
#include <nlohmann/json.hpp>
#include <queue>
#include <regex>  
#include <sstream>
#include <stdexcept>  
#include <unordered_set>

#include "logger.h"

using json = nlohmann::json;




std::string capitalize_first_letter(const std::string& s) {
  if (s.empty()) return s;

  std::string result = s;
  size_t first_letter_pos = 0;
  const std::string sp_space = "\xE2\x96\x81";  

  
  if (result.rfind(sp_space, 0) == 0) {
    if (result.length() > sp_space.length()) {
      first_letter_pos =
          sp_space.length();  
    } else {
      return result;  
    }
  }
  

  if (first_letter_pos < result.length()) {
    
    result[first_letter_pos] =
        std::toupper(static_cast<unsigned char>(result[first_letter_pos]));
  }

  return result;
}


Tokenizer::Tokenizer(const std::string& model_path,
                     const std::string& vocab_path)
    : unk_token_("<unk>"),
      bos_token_("<s>"),
      eos_token_("</s>"),
      pad_token_("<pad>") {
  try {
    Logger::info("Loading tokenizer and vocab from: " + vocab_path);

    
    load_vocab_from_json(vocab_path, token_to_id_, id_to_token_);

    
    if (token_to_id_.find(unk_token_) != token_to_id_.end()) {
      unk_token_id_ = token_to_id_[unk_token_];
    } else {
      Logger::info("No UNK token found in vocabulary");
      unk_token_id_ = 0;
    }

    if (token_to_id_.find(bos_token_) != token_to_id_.end()) {
      bos_token_id_ = token_to_id_[bos_token_];
    } else {
      Logger::info("No BOS token found in vocabulary");
      bos_token_id_ = -1;  
    }

    if (token_to_id_.find(eos_token_) != token_to_id_.end()) {
      eos_token_id_ = token_to_id_[eos_token_];
    } else {
      Logger::info("No EOS token found in vocabulary");
      eos_token_id_ = -1;  
    }

    
    if (model_path.size() > 0) {
      if (model_path.size() > 6 &&
          model_path.substr(model_path.size() - 6) == ".model") {
        Logger::info("Loading SentencePiece model: " + model_path);
        load_sentencepiece_model(model_path);
      } else if (model_path.size() > 5 &&
                 model_path.substr(model_path.size() - 5) == ".json") {
        Logger::info("Loading BPE merges from JSON: " + model_path);
        load_bpe_merges_from_json(model_path);
      } else {
        Logger::info("Unsupported model format: " + model_path +
                     " - falling back to space tokenization");
      }
    } else {
      Logger::info(
          "No model path provided - falling back to space tokenization");
    }
  } catch (const std::exception& e) {
    std::cerr << "Failed to load tokenizer or vocab from " << vocab_path << ": "
              << e.what() << std::endl;
    Logger::error(std::string("Failed to load tokenizer or vocab from \"") +
                  vocab_path + "\": " + e.what());
    throw;
  }

  if (id_to_token_.empty()) {
    throw std::runtime_error(
        "Failed to initialize tokenizer vocabulary from: " + vocab_path);
  }

  Logger::info("Successfully initialized tokenizer with " +
               std::to_string(id_to_token_.size()) + " tokens");

  
  const std::vector<std::pair<std::string, int>> known_chat_tokens = {
      {"<|system|>", 32000}, {"<|user|>", 32001}, {"<|assistant|>", 32002}};
  int manually_injected_count = 0;
  size_t vocab_size = id_to_token_.size();
  for (const auto& pair : known_chat_tokens) {
    const std::string& tok = pair.first;
    int id = pair.second;
    
    if (added_tokens_.find(tok) == added_tokens_.end() &&
        static_cast<size_t>(id) >= vocab_size) {
      added_tokens_[tok] = id;
      id_to_added_token_[id] = tok;
      manually_injected_count++;
      Logger::info("[MANUAL INJECT] Added missing chat token: '" + tok +
                   "' with assumed ID: " + std::to_string(id));
    } else if (added_tokens_.find(tok) != added_tokens_.end()) {
      Logger::debug("[MANUAL INJECT] Chat token '" + tok +
                    "' already loaded from JSON. Skipping injection.");
    } else {  
      Logger::warning("[MANUAL INJECT] Cannot add chat token '" + tok +
                      "', assumed ID " + std::to_string(id) +
                      " clashes with loaded vocab size (" +
                      std::to_string(vocab_size) + ").");
    }
  }
  if (manually_injected_count > 0) {
    Logger::info("Manually injected " +
                 std::to_string(manually_injected_count) +
                 " missing chat tokens.");
  }
}

Tokenizer::Tokenizer(const GGUFData& gguf_data) : initialized_from_gguf_(true) {
  Logger::info("Initializing Tokenizer from GGUFData...");

  if (gguf_data.tokenizer_tokens.empty()) {
    throw std::runtime_error(
        "GGUF data does not contain 'tokenizer.ggml.tokens'");
  }
  if (gguf_data.tokenizer_scores.empty()) {
    Logger::warning(
        "GGUF data does not contain 'tokenizer.ggml.scores'. BPE merging from "
        "scores will not work.");
  }
  if (gguf_data.tokenizer_tokens.size() != gguf_data.tokenizer_scores.size() &&
      !gguf_data.tokenizer_scores.empty()) {  
    Logger::warning(
        "GGUF token and score array sizes mismatch: tokens=" +
        std::to_string(gguf_data.tokenizer_tokens.size()) + ", scores=" +
        std::to_string(gguf_data.tokenizer_scores.size()));  
  }

  id_to_token_ = gguf_data.tokenizer_tokens;
  token_scores_ = gguf_data.tokenizer_scores;
  token_types_.resize(gguf_data.tokenizer_token_types.size());
  std::transform(gguf_data.tokenizer_token_types.begin(),
                 gguf_data.tokenizer_token_types.end(), token_types_.begin(),
                 [](unsigned int u) { return static_cast<int32_t>(u); });

  
  token_to_id_.reserve(id_to_token_.size());
  for (size_t i = 0; i < id_to_token_.size(); ++i) {
    token_to_id_[id_to_token_[i]] = static_cast<int>(i);
  }
  Logger::info("Loaded " + std::to_string(id_to_token_.size()) +
               " tokens from GGUF.");
  if (!token_scores_.empty()) {
    Logger::info("Loaded " + std::to_string(token_scores_.size()) +
                 " token scores from GGUF.");
  }
  if (!token_types_.empty()) {
    Logger::info("Loaded " + std::to_string(token_types_.size()) +
                 " token types from GGUF.");
  } else {
    Logger::info("Token types array not found or empty in GGUF.");
  }

  
  byte_char_to_id_.clear();
  std::regex byte_token_regex(R"(<0x([0-9A-Fa-f]{2})>)");
  for (size_t i = 0; i < id_to_token_.size(); ++i) {
    const std::string& tok = id_to_token_[i];
    std::smatch match;
    if (std::regex_match(tok, match, byte_token_regex)) {
      int byte_val = std::stoi(match[1].str(), nullptr, 16);
      char byte_char = static_cast<char>(byte_val);
      byte_char_to_id_[byte_char] = static_cast<int>(i);
      Logger::debug("Mapped byte token '" + tok + "' to char 0x" +
                    match[1].str() + " (ID: " + std::to_string(i) + ")");
    }
  }

  
  auto get_meta_value = [&](const std::string& key, auto default_value) {
    using TargetType = typename std::decay<decltype(default_value)>::type;
    auto it = gguf_data.metadata.find(key);
    if (it != gguf_data.metadata.end()) {
      return std::visit(
          [&](const auto& val) -> TargetType {
            using T = std::decay_t<decltype(val)>;
            if constexpr (std::is_arithmetic_v<TargetType> &&
                          std::is_arithmetic_v<T>) {
              if constexpr (std::is_integral_v<TargetType> &&
                            std::is_floating_point_v<T>) {
                if (val > static_cast<T>(
                              std::numeric_limits<TargetType>::max()) ||
                    val < static_cast<T>(
                              std::numeric_limits<TargetType>::lowest())) {
                  Logger::warning(
                      "Potential overflow casting float to int for GGUF key '" +
                      key + "'. Using default.");
                  return default_value;
                }
              } else if constexpr (std::is_floating_point_v<TargetType> &&
                                   std::is_integral_v<T>) {
              }
              return static_cast<TargetType>(val);
            } else if constexpr (std::is_same_v<TargetType, bool> &&
                                 std::is_same_v<T, bool>) {
              return val;
            } else if constexpr (std::is_same_v<TargetType, std::string> &&
                                 std::is_same_v<T, std::string>) {
              return val;
            }
            Logger::warning(
                "GGUF metadata key '" + key +
                "' type mismatch or unhandled conversion. Using default.");
            return default_value;
          },
          it->second);
    }
    return default_value;
  };

  auto get_meta_string = [&](const std::string& key,
                             const std::string& default_val) -> std::string {
    auto it = gguf_data.metadata.find(key);
    if (it != gguf_data.metadata.end() &&
        std::holds_alternative<std::string>(it->second)) {
      return std::get<std::string>(it->second);
    }
    return default_val;
  };

  
  bos_token_id_ = get_meta_value("tokenizer.ggml.bos_token_id", -1);
  eos_token_id_ = get_meta_value("tokenizer.ggml.eos_token_id", -1);
  unk_token_id_ = get_meta_value("tokenizer.ggml.unknown_token_id",
                                 -1);  
  pad_token_id_ = get_meta_value("tokenizer.ggml.padding_token_id",
                                 -1);  

  
  if (bos_token_id_ >= 0 && bos_token_id_ < id_to_token_.size())
    bos_token_ = id_to_token_[bos_token_id_];
  else
    bos_token_ = "";
  if (eos_token_id_ >= 0 && eos_token_id_ < id_to_token_.size())
    eos_token_ = id_to_token_[eos_token_id_];
  else
    eos_token_ = "";
  if (unk_token_id_ >= 0 && unk_token_id_ < id_to_token_.size())
    unk_token_ = id_to_token_[unk_token_id_];
  else
    unk_token_ = "<unk>";  
  if (pad_token_id_ >= 0 && pad_token_id_ < id_to_token_.size())
    pad_token_ = id_to_token_[pad_token_id_];
  else
    pad_token_ = "";  

  Logger::info(
      "Loaded Special Tokens from GGUF: BOS=" + std::to_string(bos_token_id_) +
      " ('" + bos_token_ + "'), EOS=" + std::to_string(eos_token_id_) + " ('" +
      eos_token_ + "'), UNK=" + std::to_string(unk_token_id_) + " ('" +
      unk_token_ + "'), PAD=" + std::to_string(pad_token_id_) + " ('" +
      pad_token_ + "')");

  
  pre_tok_type_ = get_meta_string("tokenizer.ggml.pre", "unknown");
  
  if (pre_tok_type_ == "unknown") {
    std::string arch = get_meta_string("general.architecture", "unknown");
    if (arch == "llama") {
      pre_tok_type_ =
          "llama";  
      Logger::info("Inferred pre_tok_type_ = 'llama' based on architecture.");
    }
  }
  Logger::info("[FINAL] pre_tok_type_ is set to '" + pre_tok_type_ +
               "'. No downstream code should override this value.");

  
  chat_template_special_tokens.clear();
  std::string chat_template = get_meta_string("tokenizer.chat_template", "");
  if (!chat_template.empty()) {
    Logger::info("Parsing chat_template for special tokens...");
    
    std::regex special_token_regex(R"(<\|[^>]+\|>)");
    auto tokens_begin = std::sregex_iterator(
        chat_template.begin(), chat_template.end(), special_token_regex);
    auto tokens_end = std::sregex_iterator();
    for (auto it = tokens_begin; it != tokens_end; ++it) {
      std::string found = it->str();
      chat_template_special_tokens.insert(found);
      Logger::info("Found chat template special token: " + found);
    }
    
    std::vector<std::string> std_specials = {"<s>", "</s>", "<unk>"};
    for (const auto& s : std_specials) {
      if (chat_template.find(s) != std::string::npos) {
        chat_template_special_tokens.insert(s);
        Logger::info("Found chat template standard special token: " + s);
      }
    }
  } else {
    Logger::info("No chat_template found in GGUF metadata.");
  }

  
  int chat_template_added_count = 0;
  for (const auto& special_token : chat_template_special_tokens) {
    auto it = token_to_id_.find(special_token);
    if (it != token_to_id_.end()) {
      int token_id = it->second;
      
      if (added_tokens_.find(special_token) == added_tokens_.end()) {
        added_tokens_[special_token] = token_id;
        id_to_added_token_[token_id] = special_token;
        chat_template_added_count++;
        Logger::info("Added chat template special token to added_tokens_: '" +
                     special_token + "' (ID: " + std::to_string(token_id) +
                     ")");
      }
    } else {
      Logger::warning("Chat template special token not found in vocab: '" +
                      special_token + "'");
    }
  }
  if (chat_template_added_count > 0) {
    Logger::info("Total chat template special tokens added to added_tokens_: " +
                 std::to_string(chat_template_added_count));
  }

  
  
  added_tokens_.clear();  
                          
  id_to_added_token_.clear();

  if (!token_types_.empty() && token_types_.size() == id_to_token_.size()) {
    int added_count = 0;
    for (size_t i = 0; i < token_types_.size(); ++i) {
      bool is_special_type = (token_types_[i] == 3 || token_types_[i] == 4);
      int token_id = static_cast<int>(i);
      bool is_known_special_id =
          (token_id == bos_token_id_ || token_id == eos_token_id_ ||
           token_id == unk_token_id_ || token_id == pad_token_id_);

      
      
      
      if (is_special_type || is_known_special_id) {
        const std::string& token_str = id_to_token_[i];
        
        if (added_tokens_.find(token_str) == added_tokens_.end()) {
          added_tokens_[token_str] = token_id;
          id_to_added_token_[token_id] = token_str;
          added_count++;
        }
      }
    }
    Logger::info(
        "Identified " + std::to_string(added_tokens_.size()) +
        " added/special tokens from GGUF token_types array and known IDs.");
  } else {
    Logger::warning(
        "Cannot identify added tokens from GGUF types (array missing or size "
        "mismatch). Manually adding BOS/EOS/UNK/PAD if found.");
    
    
    if (bos_token_id_ != -1 && !bos_token_.empty()) {
      added_tokens_[bos_token_] = bos_token_id_;
      id_to_added_token_[bos_token_id_] = bos_token_;
    }
    if (eos_token_id_ != -1 && !eos_token_.empty()) {
      added_tokens_[eos_token_] = eos_token_id_;
      id_to_added_token_[eos_token_id_] = eos_token_;
    }
    if (unk_token_id_ != -1 && !unk_token_.empty()) {
      added_tokens_[unk_token_] = unk_token_id_;
      id_to_added_token_[unk_token_id_] = unk_token_;
    }
    if (pad_token_id_ != -1 && !pad_token_.empty()) {
      added_tokens_[pad_token_] = pad_token_id_;
      id_to_added_token_[pad_token_id_] = pad_token_;
    }
    Logger::info("Manually added BOS/EOS/UNK/PAD tokens results in " +
                 std::to_string(added_tokens_.size()) + " added tokens.");
  }

  Logger::info("Tokenizer successfully initialized from GGUFData.");
}

std::vector<std::string> Tokenizer::regex_tokenize(
    const std::string& text) const {  
  std::vector<std::string> tokens;
  
  
  
  try {
    
    std::regex pattern(" ?[^\\s]+");

    auto words_begin = std::sregex_iterator(text.begin(), text.end(), pattern);
    auto words_end = std::sregex_iterator();

    for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
      std::smatch match = *i;
      if (!match.str().empty()) {  
        tokens.push_back(match.str());
      }
    }
  } catch (const std::regex_error& e) {
    Logger::error("Regex error in regex_tokenize: " + std::string(e.what()) +
                  " for text: '" + text + "'");
    return tokens;
  }
  return tokens;
}


std::vector<std::string> Tokenizer::tokenize(const std::string& text) const {
  
  if (!bpe_merges_.empty()) {
    return bpe_tokenize(text);
  }

  
  if (sentencepiece_model_loaded_) {
    return sentencepiece_tokenize(text);
  }

  
  return space_tokenize(text);
}


std::vector<std::string> Tokenizer::space_tokenize(
    const std::string& text) const {
  std::vector<std::string> tokens;
  std::istringstream iss(text);
  std::string token;

  while (iss >> token) {
    tokens.push_back(token);
  }

  return tokens;
}


std::vector<std::string> Tokenizer::bpe_tokenize(
    const std::string& text) const {
  std::vector<std::string> all_tokens;

  
  bool using_space_prefix = false;
  for (const auto& token : id_to_token_) {
    if (!token.empty() && token[0] == '\xC4' && token.size() > 1 &&
        token[1] == '\xA0') {
      
      using_space_prefix = true;
      break;
    }
    
    if (!token.empty() && token[0] == '\xE2' && token.size() > 2 &&
        token[1] == '\x96' && token[2] == '\x81') {
      
      using_space_prefix = true;
      break;
    }
  }

  
  std::string processed_text = text;
  if (using_space_prefix && !text.empty() && text[0] != ' ') {
    processed_text = " " + text;
  }

  
  std::vector<std::string> words;
  std::string current_word;
  bool in_whitespace = true;

  for (size_t i = 0; i < processed_text.size(); ++i) {
    char c = processed_text[i];
    if (std::isspace(static_cast<unsigned char>(c))) {
      if (!in_whitespace) {
        
        if (!current_word.empty()) {
          words.push_back(current_word);
          current_word.clear();
        }
        in_whitespace = true;
      }
      
      if (using_space_prefix && in_whitespace) {
        
        if (!current_word.empty()) {
          words.push_back(current_word);
        }
        current_word = " ";
        in_whitespace = false;
      }
    } else {
      current_word.push_back(c);
      in_whitespace = false;
    }
  }

  
  if (!current_word.empty()) {
    words.push_back(current_word);
  }

  
  for (auto& word : words) {
    bool trace_this_word = (word == " Studying");  
    if (trace_this_word) {
      Logger::debug("[BPE TRACE] START Processing word: '" + word + "'");
    }
    
    if (using_space_prefix && word.size() > 0 && word[0] == ' ') {
      
      if (word.size() == 1) {
        word = "\xE2\x96\x81";
      } else {
        word = "\xE2\x96\x81" + word.substr(1);
      }
      
      
      trace_this_word =
          (word == "\xE2\x96\x81Studying");  
      if (trace_this_word) {  
        Logger::debug("[BPE TRACE] Word after space replacement: '" + word +
                      "'");
      }
    }

    
    std::vector<std::string> chars;
    for (size_t i = 0; i < word.size();) {
      
      int bytes = 1;
      if ((word[i] & 0xE0) == 0xC0)
        bytes = 2;
      else if ((word[i] & 0xF0) == 0xE0)
        bytes = 3;
      else if ((word[i] & 0xF8) == 0xF0)
        bytes = 4;

      
      if (i + bytes <= word.size()) {
        chars.push_back(word.substr(i, bytes));
      } else {
        
        chars.push_back(word.substr(i));
      }
      i += bytes;
    }

    if (trace_this_word) {
      std::stringstream ss_initial_chars;
      ss_initial_chars << "[BPE TRACE] Initial chars: [";
      for (size_t k = 0; k < chars.size(); ++k)
        ss_initial_chars << "'" << chars[k] << "'"
                         << (k + 1 < chars.size() ? "," : "");
      ss_initial_chars << "]";
      Logger::debug(ss_initial_chars.str());
    }

    if (chars.empty()) continue;

    
    bool changes = true;
    int merge_iteration = 0;
    while (changes && chars.size() > 1) {
      merge_iteration++;
      if (trace_this_word) {
        Logger::debug("[BPE TRACE] --- Merge Iteration " +
                      std::to_string(merge_iteration) + " ---");
        std::stringstream ss_chars_before;
        ss_chars_before << "[BPE TRACE] Chars before iteration: [";
        for (size_t k = 0; k < chars.size(); ++k)
          ss_chars_before << "'" << chars[k] << "'"
                          << (k + 1 < chars.size() ? "," : "");
        ss_chars_before << "]";
        Logger::debug(ss_chars_before.str());
      }
      changes = false;
      int best_score = std::numeric_limits<int>::max();
      int best_i = -1;

      for (size_t i = 0; i < chars.size() - 1; ++i) {
        std::string pair = chars[i] + chars[i + 1];
        auto it = bpe_merges_.find(pair);
        if (trace_this_word) {
          Logger::debug("[BPE TRACE] Checking pair[" + std::to_string(i) +
                        "]: '" + chars[i] + "' + '" + chars[i + 1] + "' -> '" +
                        pair + "'. Found in merges: " +
                        (it != bpe_merges_.end()
                             ? "Yes (Score: " + std::to_string(it->second) + ")"
                             : "No"));
        }
        if (it != bpe_merges_.end() && it->second < best_score) {
          best_score = it->second;
          best_i = i;
        }
      }

      if (trace_this_word) {
        Logger::debug("[BPE TRACE] Best merge found at index: " +
                      std::to_string(best_i) + " with score: " +
                      (best_i >= 0 ? std::to_string(best_score) : "N/A"));
      }

      if (best_i >= 0) {
        std::string merged = chars[best_i] + chars[best_i + 1];
        if (trace_this_word) {
          Logger::debug("[BPE TRACE] Applying merge: '" + chars[best_i] +
                        "' + '" + chars[best_i + 1] + "' -> '" + merged +
                        "' at index " + std::to_string(best_i));
        }
        chars[best_i] = merged;
        chars.erase(chars.begin() + best_i + 1);
        changes = true;
        if (trace_this_word) {
          std::stringstream ss_chars_after;
          ss_chars_after << "[BPE TRACE] Chars after merge: [";
          for (size_t k = 0; k < chars.size(); ++k)
            ss_chars_after << "'" << chars[k] << "'"
                           << (k + 1 < chars.size() ? "," : "");
          ss_chars_after << "]";
          Logger::debug(ss_chars_after.str());
        }
      }
    }
    if (trace_this_word) {
      Logger::debug("[BPE TRACE] END Processing word: '" + word + "'");
    }

    
    all_tokens.insert(all_tokens.end(), chars.begin(), chars.end());
  }

  return all_tokens;
}


std::vector<std::string> Tokenizer::sentencepiece_tokenize(
    const std::string& text) const {
  
  
  Logger::info(
      "SentencePiece tokenization not fully implemented - falling back to "
      "space tokenization");
  return space_tokenize(text);
}


std::string Tokenizer::detokenize(
    const std::vector<std::string>& tokens) const {
  std::string result;

  
  bool using_space_prefix = false;
  const std::string gpt2_space_prefix = "\xC4\xA0";           
  const std::string tinyllama_space_prefix = "\xE2\x96\x81";  

  for (const auto& token : tokens) {
    if (!token.empty() &&
        ((token.size() >= 2 && token.substr(0, 2) == gpt2_space_prefix) ||
         (token.size() >= 3 && token.substr(0, 3) == tinyllama_space_prefix))) {
      using_space_prefix = true;
      break;
    }
  }

  for (size_t i = 0; i < tokens.size(); ++i) {
    std::string token = tokens[i];

    
    if (using_space_prefix) {
      
      if (!token.empty()) {
        
        if (token.size() >= 2 && token.substr(0, 2) == gpt2_space_prefix) {
          if (token.size() > 2) {
            result += ' ' + token.substr(2);
          } else {
            result += ' ';
          }
        }
        
        else if (token.size() >= 3 &&
                 token.substr(0, 3) == tinyllama_space_prefix) {
          if (token.size() > 3) {
            result += ' ' + token.substr(3);
          } else {
            result += ' ';
          }
        } else {
          
          result += token;
        }
      }
    } else {
      
      
      if (token.size() >= 4 && token.substr(token.size() - 4) == "</w>") {
        result += token.substr(0, token.size() - 4);
        result += " ";
        continue;
      }

      
      if (i > 0) {
        result += " ";
      }

      result += token;
    }
  }

  
  if (!result.empty() && result[0] == ' ') {
    result = result.substr(1);
  }

  
  std::string clean_result;
  bool prev_space = false;
  for (char c : result) {
    if (c == ' ') {
      if (!prev_space) {
        clean_result += c;
      }
      prev_space = true;
    } else {
      clean_result += c;
      prev_space = false;
    }
  }

  return clean_result;
}


std::vector<int> Tokenizer::tokens_to_ids(
    const std::vector<std::string>& tokens) const {
  std::vector<int> ids;
  ids.reserve(tokens.size());

  for (const auto& token : tokens) {
    auto added_it = added_tokens_.find(token);
    if (added_it != added_tokens_.end()) {
      ids.push_back(added_it->second);
      Logger::debug("[TOK_TO_ID] Found added token: '" + token +
                    "' -> ID: " + std::to_string(added_it->second));
    } else {
      auto base_it = token_to_id_.find(token);
      if (base_it != token_to_id_.end()) {
        ids.push_back(base_it->second);
        Logger::debug("[TOK_TO_ID] Found base token: '" + token +
                      "' -> ID: " + std::to_string(base_it->second));
      } else {
        
        std::string capitalized_token = capitalize_first_letter(token);
        if (capitalized_token !=
            token) {  
          auto capitalized_it = token_to_id_.find(capitalized_token);
          if (capitalized_it != token_to_id_.end()) {
            
            ids.push_back(capitalized_it->second);
            Logger::debug(
                "[TOK_TO_ID] FALLBACK: Found capitalized base token: '" +
                token + "' -> '" + capitalized_token +
                "' -> ID: " + std::to_string(capitalized_it->second));
            continue;  
          }
        }

        
        if (token.length() == 1) {
          char c = token[0];
          auto byte_it = byte_char_to_id_.find(c);
          if (byte_it != byte_char_to_id_.end()) {
            ids.push_back(byte_it->second);
            Logger::debug("[TOK_TO_ID] FALLBACK: Mapped single-byte token '" +
                          std::string(1, c) + "' to byte token ID " +
                          std::to_string(byte_it->second));
            continue;  
          }
        }
        
        Logger::debug("[TOK_TO_ID] UNKNOWN: Token '" + token +
                      "' not found in added, base, capitalized fallback, or "
                      "byte tokens. Using UNK ID: " +
                      std::to_string(unk_token_id_));
        ids.push_back(unk_token_id_);
      }
    }
  }

  return ids;
}


std::vector<std::string> Tokenizer::ids_to_tokens(
    const std::vector<int>& ids) const {
  std::vector<std::string> tokens;
  tokens.reserve(ids.size());

  for (int id : ids) {
    auto added_it = id_to_added_token_.find(id);
    if (added_it != id_to_added_token_.end()) {
      
      tokens.push_back(added_it->second);
      
      
    } else if (id >= 0 && static_cast<size_t>(id) < id_to_token_.size()) {
      
      if (!id_to_token_[id].empty()) {  
        tokens.push_back(id_to_token_[id]);
        
        
      } else {
        
        
        tokens.push_back(unk_token_);
        Logger::warning(
            "ID " + std::to_string(id) +
            " found in base vocab range but has empty string. Using UNK.");
      }
    } else {
      
      tokens.push_back(unk_token_);
    }
  }

  return tokens;
}


std::vector<int> Tokenizer::encode(const std::string& text, bool add_bos,
                                   bool add_eos,
                                   PreTokenizeMethod pre_tok_override) const {
  std::vector<int> final_ids;
  Logger::debug("[ENCODE] Encoding text: '" + text +
                "' (add_bos=" + std::to_string(add_bos) +
                ", add_eos=" + std::to_string(add_eos) + ")");

  
  if (!initialized_from_gguf_) {
    Logger::debug(
        "[ENCODE] Using simplified merge-based tokenizer path (calling "
        "bpe_tokenize directly).");

    std::vector<std::string> bpe_pieces =
        bpe_tokenize(text);  
    Logger::debug("[ENCODE] bpe_tokenize returned " +
                  std::to_string(bpe_pieces.size()) + " pieces.");

    final_ids = tokens_to_ids(bpe_pieces);

    
    if (add_bos && bos_token_id_ != -1) {
      final_ids.insert(final_ids.begin(), bos_token_id_);
      Logger::debug("[ENCODE] Prepended BOS token: " +
                    std::to_string(bos_token_id_));
    }
    if (add_eos && eos_token_id_ != -1) {
      final_ids.push_back(eos_token_id_);
      Logger::debug("[ENCODE] Appended EOS token: " +
                    std::to_string(eos_token_id_));
    }

    Logger::debug("[ENCODE] Final IDs (Simplified Merge Path): " +
                  std::to_string(final_ids.size()) + " tokens.");
    return final_ids;
  } else {
    Logger::debug("[ENCODE] Using GGUF score-based tokenizer path.");
    
    if (add_bos && bos_token_id_ != -1) {
      final_ids.push_back(bos_token_id_);
      Logger::debug("[ENCODE] Added BOS token: " +
                    std::to_string(bos_token_id_));
    }

    
    std::vector<std::pair<std::string, bool>>
        segments;  
    std::string current_segment;
    std::string text_to_process = text;

    
    PreTokenizeMethod method_to_use;
    if (pre_tok_override == PreTokenizeMethod::DEFAULT) {
      if (pre_tok_type_ == "default") {
        method_to_use = PreTokenizeMethod::DEFAULT;
        Logger::debug(
            "[ENCODE] Using DEFAULT pre-tokenization path (split by special "
            "tokens, BPE for non-specials).");
      } else if (pre_tok_type_ == "llama") {
        method_to_use = PreTokenizeMethod::LLAMA_REGEX;
      } else {
        method_to_use = PreTokenizeMethod::WHITESPACE;
      }
    } else {
      method_to_use = pre_tok_override;
    }
    std::string method_str;
    if (method_to_use == PreTokenizeMethod::LLAMA_REGEX)
      method_str = "LLAMA_REGEX";
    else if (method_to_use == PreTokenizeMethod::WHITESPACE)
      method_str = "WHITESPACE";
    else
      method_str = "DEFAULT";
    Logger::debug("[ENCODE] Effective pre-tokenization method: " + method_str);

    if (method_to_use == PreTokenizeMethod::DEFAULT) {
      
      
      
      std::unordered_set<std::string> all_special_tokens;
      for (const auto& pair : added_tokens_) {
        all_special_tokens.insert(pair.first);
      }
      
      for (const auto& s : chat_template_special_tokens) {
        all_special_tokens.insert(s);
      }
      std::string special_pattern_str = "(";
      bool first_special = true;
      for (const auto& tok : all_special_tokens) {
        if (!tok.empty()) {
          if (!first_special) special_pattern_str += "|";
          std::string escaped_token = std::regex_replace(
              tok, std::regex("[\\^\\$\\.\\*\\+\\?\\(\\)\\[\\]\\{\\}\\|]"),
              "\\$&");
          special_pattern_str += escaped_token;
          first_special = false;
        }
      }
      special_pattern_str += ")";

      std::vector<std::pair<std::string, bool>> segments;
      if (all_special_tokens.empty()) {
        segments.push_back({text, false});
        Logger::debug(
            "[ENCODE] No special tokens found, treating whole text as one "
            "segment.");
      } else {
        std::regex special_regex(special_pattern_str);
        auto words_begin =
            std::sregex_iterator(text.begin(), text.end(), special_regex);
        auto words_end = std::sregex_iterator();
        long last_pos = 0;
        for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
          std::smatch match = *i;
          long current_pos = match.position();
          std::string match_str = match.str();
          if (current_pos > last_pos) {
            segments.push_back(
                {text.substr(last_pos, current_pos - last_pos), false});
          }
          segments.push_back({match_str, true});
          last_pos = current_pos + match.length();
        }
        if (last_pos < text.length()) {
          segments.push_back({text.substr(last_pos), false});
        }
        Logger::debug("[ENCODE] Split text into " +
                      std::to_string(segments.size()) +
                      " segments by special tokens.");
      }
      for (const auto& segment_pair : segments) {
        const std::string& segment_to_process = segment_pair.first;
        bool is_special = segment_pair.second;
        if (segment_to_process.empty()) continue;
        if (is_special) {
          auto it = added_tokens_.find(segment_to_process);
          if (it != added_tokens_.end()) {
            final_ids.push_back(it->second);
            Logger::debug("[ENCODE] Added special token ID: " +
                          std::to_string(it->second) + " for '" +
                          segment_to_process + "'");
          } else {
            Logger::debug("[ENCODE] Skipping template-only special token: '" +
                          segment_to_process + "'");
          }
        } else {
          std::vector<std::string> bpe_pieces;
          if (initialized_from_gguf_) {
            if (!token_scores_.empty()) {
              Logger::debug(
                  "[ENCODE] Passing segment to bpe_tokenize_from_scores: '" +
                  segment_to_process + "'");
              bpe_pieces = bpe_tokenize_from_scores(segment_to_process);
            } else {
              Logger::warning(
                  "[ENCODE] GGUF Initialized but no scores. Tokenizing segment "
                  "'" +
                  segment_to_process + "' with space fallback.");
              std::vector<std::string> space_pieces;
              std::string word;
              std::istringstream iss(segment_to_process);
              while (iss >> word) {
                space_pieces.push_back(word);
              }
              bpe_pieces = space_pieces;
            }
          } else {
            Logger::debug(
                "[ENCODE] Passing segment to bpe_tokenize (merge-based): '" +
                segment_to_process + "' with method: " + method_str);
            bpe_pieces = bpe_tokenize(
                segment_to_process);  
          }
          std::vector<int> segment_ids = tokens_to_ids(bpe_pieces);
          final_ids.insert(final_ids.end(), segment_ids.begin(),
                           segment_ids.end());
          std::stringstream ss_bpe;
          ss_bpe << "[ENCODE] BPE Results for segment '" << segment_to_process
                 << "': Pieces=[ ";
          for (const auto& p : bpe_pieces) ss_bpe << "'" << p << "' ";
          ss_bpe << "], IDs=[ ";
          for (int id : segment_ids) ss_bpe << id << " ";
          ss_bpe << "]";
          Logger::debug(ss_bpe.str());
        }
      }
    } else {
    }

    
    if (add_eos && eos_token_id_ != -1) {
      final_ids.push_back(eos_token_id_);
      Logger::debug("[ENCODE] Added EOS token: " +
                    std::to_string(eos_token_id_));
    }

    Logger::debug("[ENCODE] Final IDs: " + std::to_string(final_ids.size()) +
                  " tokens.");
    return final_ids;
  }
}


std::string Tokenizer::decode(const std::vector<int>& ids,
                              bool skip_special_tokens) const {
  std::stringstream ss;
  bool first_token = true;  
  for (int id : ids) {
    if (id >= 0 && static_cast<size_t>(id) < id_to_token_.size()) {
      std::string token = id_to_token_[id];
      if (skip_special_tokens) {
        if (id == bos_token_id_ || id == eos_token_id_ || id == pad_token_id_ ||
            id == unk_token_id_) {
          continue;  
        }
      }
      
      
      const std::string gpt2_space_prefix = "\xC4\xA0";
      const std::string tinyllama_space_prefix = "\xE2\x96\x81";
      if (!token.empty()) {
        if ((token.size() >= gpt2_space_prefix.size() &&
             token.substr(0, gpt2_space_prefix.size()) == gpt2_space_prefix)) {
          if (!first_token) ss << " ";  
          ss << token.substr(gpt2_space_prefix.size());
        } else if ((token.size() >= tinyllama_space_prefix.size() &&
                    token.substr(0, tinyllama_space_prefix.size()) ==
                        tinyllama_space_prefix)) {
          if (!first_token) ss << " ";  
          ss << token.substr(tinyllama_space_prefix.size());
        } else {
          ss << token;  
        }
      }
      first_token = false;  
    } else {
      ss << "[INVALID_ID:" << id << "]";
      first_token = false;
    }
  }
  return ss.str();
}


std::string Tokenizer::apply_chat_template(const std::string& user_prompt,
                                           const std::string& system_message,
                                           const ModelConfig& config)
    const  
{
  
  auto find_added_token_str = [&](const std::string& content,
                                  const std::string& fallback) -> std::string {
    for (const auto& pair : added_tokens_) {
      if (pair.first == content) return pair.first;
    }
    
    
    if (!added_tokens_.empty()) {
      Logger::warning("Could not find added token '" + content +
                      "' in added_tokens_ map. Using fallback: '" + fallback +
                      "'");
    }  
       
    return fallback;  
  };

  
  std::string sys_tok = find_added_token_str("<|system|>", "<|system|>");
  std::string user_tok = find_added_token_str("<|user|>", "<|user|>");
  std::string assist_tok =
      find_added_token_str("<|assistant|>", "<|assistant|>");
  
  std::string eos_tok_str = eos_token_;
  if (eos_token_id_ >= 0 &&
      static_cast<size_t>(eos_token_id_) < id_to_token_.size()) {
    eos_tok_str = id_to_token_[eos_token_id_];
  } else {
    Logger::warning(
        "apply_chat_template: EOS token ID not found in vocab, using default "
        "'</s>'");
  }

  Logger::info(
      "Applying MANUALLY IMPLEMENTED TinyLlama chat template structure (NO "
      "NEWLINES).");
  std::stringstream ss;
  
  if (!system_message.empty()) {
    ss << sys_tok << system_message << eos_tok_str;  
  }
  
  ss << user_tok << user_prompt << eos_tok_str;  
  ss << assist_tok;                              
  return ss.str();
}


void Tokenizer::load_vocab_from_json(
    const std::string& vocab_path,
    std::unordered_map<std::string, int>& token_to_id,
    std::vector<std::string>& id_to_token) {
  
  token_to_id.clear();
  id_to_token.clear();

  try {
    
    std::ifstream file(vocab_path);
    if (!file.is_open()) {
      throw std::runtime_error("Failed to open vocabulary file: " + vocab_path);
    }

    json vocab_json;
    file >> vocab_json;

    
    if (vocab_json.contains("model") && vocab_json["model"].contains("vocab") &&
        vocab_json["model"]["vocab"].is_object()) {
      
      Logger::info("Detected HuggingFace tokenizer.json format");
      const auto& vocab = vocab_json["model"]["vocab"];

      
      size_t vocab_size = vocab.size();

      
      if (vocab_json.contains("added_tokens") &&
          vocab_json["added_tokens"].is_array()) {
        const auto& added_tokens = vocab_json["added_tokens"];
        for (const auto& token_obj : added_tokens) {
          if (token_obj.contains("content") && token_obj.contains("id")) {
            std::string token = token_obj["content"];
            int id = token_obj["id"];

            
            if (token == "<unk>")
              unk_token_ = token;
            else if (token == "<s>")
              bos_token_ = token;
            else if (token == "</s>")
              eos_token_ = token;
            else if (token == "<pad>")
              pad_token_ = token;

            
            added_tokens_[token] = id;

            if (id >= 0) {  
              if (static_cast<size_t>(id) >= id_to_token.size()) {
                id_to_token.resize(id + 1);
              }
              id_to_token[id] = token;
            }
            Logger::info("Processed added token: " + token + " with ID " +
                         std::to_string(id));  
          }
        }
      }

      
      if (id_to_token.size() < vocab_size) {
        id_to_token.resize(vocab_size);
      }

      
      for (auto it = vocab.begin(); it != vocab.end(); ++it) {
        std::string token = it.key();
        int id = it.value().get<int>();

        token_to_id[token] = id;

        
        if (static_cast<size_t>(id) >= id_to_token.size()) {
          id_to_token.resize(id + 1);
        }

        id_to_token[id] = token;
      }
    }
    
    else if (vocab_json.is_object()) {
      Logger::info("Detected plain vocabulary format");

      
      size_t vocab_size = vocab_json.size();
      id_to_token.resize(vocab_size);

      
      for (auto it = vocab_json.begin(); it != vocab_json.end(); ++it) {
        std::string token = it.key();
        int id = it.value().get<int>();

        
        if (token == "<unk>")
          unk_token_ = token;
        else if (token == "<s>")
          bos_token_ = token;
        else if (token == "</s>")
          eos_token_ = token;
        else if (token == "<pad>")
          pad_token_ = token;

        
        token_to_id[token] = id;

        
        if (static_cast<size_t>(id) >= id_to_token.size()) {
          id_to_token.resize(id + 1);
        }

        id_to_token[id] = token;
      }
    } else {
      throw std::runtime_error("Vocabulary JSON has an unsupported format");
    }

    
    Logger::info("Special tokens: UNK=" + unk_token_ + ", BOS=" + bos_token_ +
                 ", EOS=" + eos_token_ + ", PAD=" + pad_token_);

    
    for (size_t i = 0; i < id_to_token.size(); ++i) {
      if (id_to_token[i].empty()) {
        Logger::info("Token ID " + std::to_string(i) +
                     " is missing in vocabulary");
        id_to_token[i] = "<missing>";
      }
    }

    Logger::info("Loaded vocabulary with " +
                 std::to_string(token_to_id.size()) + " tokens");

  } catch (const std::exception& e) {
    throw std::runtime_error("Error loading vocabulary: " +
                             std::string(e.what()));
  }
}


void Tokenizer::load_bpe_merges_from_json(const std::string& model_path) {
  try {
    
    std::ifstream file(model_path);
    if (!file.is_open()) {
      throw std::runtime_error("Failed to open BPE model file: " + model_path);
    }

    json model_json;
    file >> model_json;

    
    bpe_merges_.clear();

    
    if (model_json.contains("model") && model_json["model"].contains("type") &&
        model_json["model"]["type"] == "BPE") {
      Logger::info("Detected HuggingFace tokenizer.json format");

      
      if (model_json["model"].contains("merges") &&
          model_json["model"]["merges"].is_array()) {
        
        const auto& merges = model_json["model"]["merges"];

        for (size_t i = 0; i < merges.size(); ++i) {
          std::string merge_entry = merges[i];
          size_t space_pos = merge_entry.find(' ');

          if (space_pos != std::string::npos) {
            std::string first = merge_entry.substr(0, space_pos);
            std::string second = merge_entry.substr(space_pos + 1);
            std::string pair = first + second;

            bpe_merges_[pair] = i;  
          }
        }
      }
      
      
      
      else {
        
        
        
        Logger::info(
            "No explicit merges array found, extracting from vocabulary "
            "patterns");

        int merge_index = 0;
        
        
        if (model_json["model"].contains("vocab") &&
            model_json["model"]["vocab"].is_object()) {
          const auto& vocab = model_json["model"]["vocab"];

          for (auto it = vocab.begin(); it != vocab.end(); ++it) {
            const std::string& token = it.key();

            
            size_t space_pos = token.find(' ');
            if (space_pos != std::string::npos) {
              std::string first = token.substr(0, space_pos);
              std::string second = token.substr(space_pos + 1);
              std::string pair = first + second;

              
              
              int priority = merge_index++;
              bpe_merges_[pair] = priority;
            }
          }
        }
      }
    }
    
    else if (model_json.contains("merges") && model_json["merges"].is_array()) {
      
      Logger::info("Detected classic BPE merges format (fallback).");
      const auto& merges = model_json["merges"];

      for (size_t i = 0; i < merges.size(); ++i) {
        std::string merge_entry = merges[i];
        size_t space_pos = merge_entry.find(' ');

        if (space_pos != std::string::npos) {
          std::string first = merge_entry.substr(0, space_pos);
          std::string second = merge_entry.substr(space_pos + 1);
          std::string pair = first + second;

          bpe_merges_[pair] = i;  
        }
      }
    } else {
      throw std::runtime_error(
          "Unsupported tokenizer model format: no merges found");
    }

    if (bpe_merges_.empty()) {
      
      Logger::warning("No BPE merges found or loaded from the model file.");
    } else {
      Logger::info("Loaded " + std::to_string(bpe_merges_.size()) +
                   " BPE merges");
    }

  } catch (const std::exception& e) {
    std::string error_msg = "Error loading BPE merges from \"" + model_path +
                            "\": " + std::string(e.what());
    Logger::error(error_msg);
    throw std::runtime_error(error_msg);  
  }
}


void Tokenizer::load_sentencepiece_model(const std::string& model_path) {
  
  Logger::info("SentencePiece model loading not implemented yet");
  sentencepiece_model_loaded_ = false;
}

int Tokenizer::vocab_size() const { return id_to_token_.size(); }

bool Tokenizer::is_added_token(int id) const {
  return id_to_added_token_.count(id) > 0;
}

struct BPEMerge {
  float score;  
  int index;    
                
  
  bool operator<(const BPEMerge& other) const { return score < other.score; }
};

std::vector<std::string> Tokenizer::bpe_tokenize_from_scores(
    const std::string& text) const {
  std::vector<std::string> all_tokens;
  std::vector<std::string> initial_units;
  std::regex llama_regex(
      R"('s|'t|'re|'ve|'m|'ll|'d| ?[[:alpha:]]+| ?[[:digit:]]+| ?[^\s[:alpha:][:digit:]]+|\s+(?!\S)|\s+)");
  std::smatch match;
  std::string text_to_search = text;
  while (std::regex_search(text_to_search, match, llama_regex)) {
    initial_units.push_back(match.str(0));
    text_to_search = match.suffix().str();
  }
  if (!text_to_search.empty()) {  
    initial_units.push_back(text_to_search);
  }
  
  std::vector<std::string> filtered_units;
  int spaces_filtered = 0;
  for (const std::string& unit : initial_units) {
    if (unit == " ") {
      spaces_filtered++;
    } else {
      filtered_units.push_back(unit);
    }
  }
  if (spaces_filtered > 0) {
    Logger::debug("[BPE_SCORES] Filtered out " +
                  std::to_string(spaces_filtered) + " standalone space units.");
  }

  
  for (const std::string& unit_raw : filtered_units) {  
    if (unit_raw.empty()) continue;

    
    std::string unit = unit_raw;
    const std::string sp_space = "\xE2\x96\x81";  
    bool using_space_prefix = true;  
    if (using_space_prefix && unit.length() > 0 && unit[0] == ' ') {
      unit.replace(0, 1, sp_space);
      
      
    } else if (using_space_prefix && unit == "\n") {
      Logger::debug("[BPE_TOKENIZE] Passing newline unit through: '" +
                    unit_raw + "'");
    }

    
    std::vector<std::string> chars;
    for (size_t i = 0; i < unit.size();) {
      
      int bytes = 1;
      if ((unit[i] & 0xE0) == 0xC0)
        bytes = 2;
      else if ((unit[i] & 0xF0) == 0xE0)
        bytes = 3;
      else if ((unit[i] & 0xF8) == 0xF0)
        bytes = 4;

      
      if (i + bytes <= unit.size()) {
        chars.push_back(unit.substr(i, bytes));
      } else {
        
        chars.push_back(unit.substr(i));
      }
      i += bytes;
    }

    if (chars.empty()) continue;

    
    bool changes = true;
    while (changes && chars.size() > 1) {
      changes = false;
      int best_score = std::numeric_limits<int>::max();
      int best_i = -1;

      for (size_t i = 0; i < chars.size() - 1; ++i) {
        std::string pair = chars[i] + chars[i + 1];
        auto it = bpe_merges_.find(pair);
        if (it != bpe_merges_.end() && it->second < best_score) {
          best_score = it->second;
          best_i = i;
        }
      }

      if (best_i >= 0) {
        std::string merged = chars[best_i] + chars[best_i + 1];
        chars[best_i] = merged;
        chars.erase(chars.begin() + best_i + 1);
        changes = true;
      }
    }

    
    all_tokens.insert(all_tokens.end(), chars.begin(), chars.end());
  }

  return all_tokens;
}
