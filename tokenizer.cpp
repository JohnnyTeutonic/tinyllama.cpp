#include "tokenizer.h"

#include <algorithm>  
#include <cctype>
#include <fstream>
#include <iomanip>
#include <iostream>  
#include <map>
#include <nlohmann/json.hpp>
#include <queue>
#include <boost/regex.hpp> 
#include <boost/xpressive/xpressive.hpp>
#include <sstream>
#include <stdexcept>  
#include <unordered_set>
#include <vector>
#include <string>
#include <limits>
#include <utility>   // For std::pair
#include <functional> // For std::less
#include <filesystem>

#include "logger.h"

// Define BPE_SPACE_CHAR at file scope for broader accessibility
const std::string BPE_SPACE_CHAR = "\xC4\xA0"; // GPT-2 BPE space character (Ġ)

using json = nlohmann::json;

// Forward declaration for helper function defined later in an anonymous namespace
namespace {
    size_t unicode_char_len(char src);
} // end anonymous namespace

// Helper function to check if a string represents a number.
bool is_numeric(const std::string& s) {
    if (s.empty()) {
        return false; // An empty string is not considered numeric
    }
    for (char c : s) {
        if (!std::isdigit(static_cast<unsigned char>(c))) {
            return false; // Found a non-digit character
        }
    }
    return true; // All characters are digits
}


// Finds the rank of a potential BPE merge.
// Returns the rank (lower is better) if the merge exists, otherwise -1.
int Tokenizer::find_bpe_rank(const std::string & token_left, const std::string & token_right) const {
    auto it = bpe_merges_.find(token_left + token_right); // Ensure this uses the correct combined form if prefixes are involved
    if (it != bpe_merges_.end()) {
        return it->second; // Return the rank
    }
    return -1; // Merge not found
}

std::vector<std::string> Tokenizer::bpe_tokenize_from_scores(
    const std::string& text) const {
  std::vector<std::string> all_tokens;
  std::vector<std::string> initial_units; // Pre-tokenized parts (words, symbols, spaces)

  // Llama-like regex for pre-tokenization
  boost::regex llama_regex(
      // This pattern is common for SentencePiece-like splitting by words, numbers, symbols, and whitespace.
      R"([\r\n]+|[[:space:]]+|[^\r\n[:space:][:alnum:]]+|[[:alnum:]]+)"); 
  boost::smatch match;
  std::string text_to_search = text;

  // Pre-tokenize the text using the regex
  while (boost::regex_search(text_to_search, match, llama_regex)) {
    if (!match.str(0).empty()) { // Ensure no empty strings are added
        initial_units.push_back(match.str(0));
    }
    text_to_search = match.suffix().str();
  }
  if (!text_to_search.empty()) {  // Add any trailing part not matched
    initial_units.push_back(text_to_search);
  }
  
  Logger::debug("[BPE_SCORES] Regex pre-tokenization resulted in " + std::to_string(initial_units.size()) + " initial units.");

  const std::string sp_space_prefix = "\xE2\x96\x81";  // SentencePiece space U+2581
  bool next_word_needs_prefix = true;

  for (const std::string& unit_raw : initial_units) {  
    if (unit_raw.empty()) continue;

    // Check if the unit is purely whitespace
    bool unit_is_whitespace = true;
    for (char c : unit_raw) {
        if (!std::isspace(static_cast<unsigned char>(c))) {
            unit_is_whitespace = false;
            break;
        }
    }

    if (unit_is_whitespace) {
        // Whitespace signals that the *next* non-whitespace unit needs the prefix.
        next_word_needs_prefix = true;
        Logger::debug("[BPE_SCORES] Unit '" + unit_raw + "' is whitespace. Setting prefix flag for next word.");
        continue; // Skip to the next unit
    }

    
    std::string unit_to_bpe = unit_raw;
    if (next_word_needs_prefix) {
        unit_to_bpe = sp_space_prefix + unit_to_bpe;
        Logger::debug("[BPE_SCORES] Prefixed unit: '" + unit_raw + "' -> '" + unit_to_bpe + "'");
        next_word_needs_prefix = false; // Reset flag after applying prefix
    } else {
         Logger::debug("[BPE_SCORES] Processing unit without prefix: '" + unit_to_bpe + "'");
    }
    
    if (unit_raw == "\n") {
        Logger::debug("[BPE_SCORES] Raw unit is newline. It will be split into chars. Current unit_to_bpe: '" + unit_to_bpe + "'");
        // If a newline is a standalone token, it should be found. If it's part of merges, it will be handled.
    }

    std::vector<std::string> chars; // Characters/sub-units of the current unit_to_bpe
    // Split unit_to_bpe into UTF-8 characters
    for (size_t i = 0; i < unit_to_bpe.size();) {
      int bytes = unicode_char_len(unit_to_bpe[i]);
      
      if (i + bytes <= unit_to_bpe.size()) {
        chars.push_back(unit_to_bpe.substr(i, bytes));
      } else {
        Logger::warning("[BPE_SCORES] Invalid UTF-8 sequence or length error for: '" + unit_to_bpe.substr(i) + "'");
        chars.push_back(unit_to_bpe.substr(i));
        break; 
      }
      i += bytes;
    }

    if (chars.empty()) {
        Logger::warning("[BPE_SCORES] Unit '" + unit_to_bpe + "' (original: '" + unit_raw + "') produced no chars for BPE.");
        continue;
    }
    
    // Perform BPE merges based on scores (ranks in bpe_merges_)
    bool changes = true;
    while (changes && chars.size() > 1) {
      changes = false;
      int best_rank = std::numeric_limits<int>::max(); // For rank-based merges, lower is better
      int best_i = -1;

      for (size_t i = 0; i < chars.size() - 1; ++i) {
        std::string pair = chars[i] + chars[i + 1];
        auto it = bpe_merges_.find(pair); 
        if (it != bpe_merges_.end() && it->second < best_rank) { // Using rank from bpe_merges_
          best_rank = it->second;
          best_i = i;
        }
      }

      if (best_i >= 0) { // If a merge was found
        std::string merged = chars[best_i] + chars[best_i + 1];
        chars[best_i] = merged;
        chars.erase(chars.begin() + best_i + 1);
        changes = true;
      }
    }
    
    all_tokens.insert(all_tokens.end(), chars.begin(), chars.end());
  }

  Logger::debug("[BPE_SCORES] Final token count after BPE: " + std::to_string(all_tokens.size()));
  return all_tokens;
}
std::vector<int> Tokenizer::tokens_to_ids(
    const std::vector<std::string>& tokens) const {
  std::vector<int> ids;
  ids.reserve(tokens.size());

  for (const auto& token : tokens) {
    
    if (token == "\n") {
        Logger::debug("[TOK_TO_ID_NL_DEBUG] Processing token: '\n' (actual newline char). Length: " + std::to_string(token.length()));
        bool found_in_added = false;
        for (const auto& pair : added_tokens_) {
            if (pair.first == "\n") {
                Logger::debug("[TOK_TO_ID_NL_DEBUG] Found '\n' key in added_tokens_ map. ID: " + std::to_string(pair.second));
                found_in_added = true;
                break;
            }
        }
        if (!found_in_added) {
            Logger::debug("[TOK_TO_ID_NL_DEBUG] '\n' key NOT found in added_tokens_ map by direct string compare.");
            // Log all keys in added_tokens_ if newline is not found, to see what IS there
            std::string keys_in_map = "Keys in added_tokens_: ";
            for (const auto& pair : added_tokens_) {
                std::string key_escaped;
                for (char c_key : pair.first) {
                    if (c_key == '\n') key_escaped += "<NL>";
                    else if (c_key == '\r') key_escaped += "<CR>";
                    else if (c_key == '\t') key_escaped += "<TAB>";
                    else if (std::isprint(static_cast<unsigned char>(c_key))) key_escaped += c_key;
                    else { std::stringstream ss_hex; ss_hex << "<0x" << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(static_cast<unsigned char>(c_key)) << ">"; key_escaped += ss_hex.str(); }
                }
                keys_in_map += "['" + key_escaped + "' (len:" + std::to_string(pair.first.length()) + ")] ";
            }
            Logger::debug(keys_in_map);
        }
    }
    

    auto added_it = added_tokens_.find(token);
    if (added_it != added_tokens_.end()) { // Check added tokens first
      ids.push_back(added_it->second);
      Logger::debug("[TOK_TO_ID] Found added token: '" + token +
                    "' -> ID: " + std::to_string(added_it->second));
    } else { // Not an added token, check base vocabulary
      auto base_it = token_to_id_.find(token);
      if (base_it != token_to_id_.end()) {
        ids.push_back(base_it->second);
        Logger::debug("[TOK_TO_ID] Found base token: '" + token +
                      "' -> ID: " + std::to_string(base_it->second));
      } else { // Not in base vocab, try capitalized version
        std::string capitalized_token = capitalize_first_letter(token);
        if (capitalized_token != token) { // If capitalization changed something
          auto capitalized_it = token_to_id_.find(capitalized_token);
          if (capitalized_it != token_to_id_.end()) {
            ids.push_back(capitalized_it->second);
            Logger::debug(
                "[TOK_TO_ID] FALLBACK: Found capitalized base token: '" +
                token + "' -> '" + capitalized_token +
                "' -> ID: " + std::to_string(capitalized_it->second));
            continue;  // Skip further fallbacks for this token
          }
        }
        
        // Fallback for single-byte tokens if not found yet
        if (token.length() == 1) {
          char c = token[0];
          auto byte_it = byte_char_to_id_.find(c);
          if (byte_it != byte_char_to_id_.end()) {
            ids.push_back(byte_it->second);
            Logger::debug("[TOK_TO_ID] FALLBACK: Mapped single-byte token '" +
                          std::string(1, c) + "' to byte token ID " +
                          std::to_string(byte_it->second));
            continue;  // Skip further fallbacks
          }
        }
        
        // If all fallbacks fail, use UNK token
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
    auto added_it = id_to_added_token_.find(id); // Check added tokens first
    if (added_it != id_to_added_token_.end()) {
      tokens.push_back(added_it->second);
    } else if (id >= 0 && static_cast<size_t>(id) < id_to_token_.size()) { // Check base vocabulary
      if (!id_to_token_[id].empty()) {  // Ensure token string is not empty
        tokens.push_back(id_to_token_[id]);
      } else {
        tokens.push_back(unk_token_); // Fallback to UNK string
        Logger::warning(
            "ID " + std::to_string(id) +
            " found in base vocab range but has empty string. Using UNK token string: '" + unk_token_ + "'.");
      }
    } else { // ID is out of bounds or negative (and not an added token)
      tokens.push_back(unk_token_); // Fallback to UNK string
    }
  }

  return tokens;
}


Tokenizer::Tokenizer(const std::string& vocab_path, 
                     const std::string& model_path, 
                     const ModelConfig& config)
    : tokenizer_family_(config.tokenizer_family),
      unk_token_("<unk>"),
      bos_token_("<s>"),
      eos_token_("</s>"),
      pad_token_("<pad>") {
  Logger::info("[Tokenizer Constructor JSON] vocab_path: '" + vocab_path + "', model_path: '" + model_path + "'"); // Diagnostic log
  try {
    std::filesystem::path vocab_json_path_abs(vocab_path);
    if (!std::filesystem::exists(vocab_json_path_abs)) {
        throw std::runtime_error("Tokenizer vocab_path (tokenizer.json) does not exist: " + vocab_json_path_abs.string());
    }

    Logger::info(std::string("Loading tokenizer and vocab from: ") + vocab_json_path_abs.string());
    std::string family_str = "UNKNOWN";
    if (tokenizer_family_ == ModelConfig::TokenizerFamily::LLAMA_SENTENCEPIECE) family_str = "LLAMA_SENTENCEPIECE";
    else if (tokenizer_family_ == ModelConfig::TokenizerFamily::LLAMA3_TIKTOKEN) family_str = "LLAMA3_TIKTOKEN";
    Logger::info(std::string("Tokenizer family based on config: ") + family_str);

    load_vocab_from_json(vocab_json_path_abs.string(), token_to_id_, id_to_token_);

    if (tokenizer_family_ == ModelConfig::TokenizerFamily::LLAMA_SENTENCEPIECE) {
        Logger::info("LLAMA_SENTENCEPIECE family detected for JSON constructor, attempting to load BPE merges from: " + vocab_json_path_abs.string());
        load_bpe_merges_from_json(vocab_json_path_abs.string()); 
    }

    unk_token_id_ = (token_to_id_.count(unk_token_)) ? token_to_id_[unk_token_] : config.bos_token_id; // Fallback to BOS if UNK not in vocab
    bos_token_id_ = (token_to_id_.count(bos_token_)) ? token_to_id_[bos_token_] : config.bos_token_id;
    eos_token_id_ = (token_to_id_.count(eos_token_)) ? token_to_id_[eos_token_] : config.eos_token_id;
    pad_token_id_ = (token_to_id_.count(pad_token_)) ? token_to_id_[pad_token_] : -1;

    if (bos_token_id_ >= 0 && static_cast<size_t>(bos_token_id_) < id_to_token_.size() && !token_to_id_.count(bos_token_)) bos_token_ = id_to_token_[bos_token_id_];
    if (eos_token_id_ >= 0 && static_cast<size_t>(eos_token_id_) < id_to_token_.size() && !token_to_id_.count(eos_token_)) eos_token_ = id_to_token_[eos_token_id_];
    if (unk_token_id_ >= 0 && static_cast<size_t>(unk_token_id_) < id_to_token_.size() && !token_to_id_.count(unk_token_)) unk_token_ = id_to_token_[unk_token_id_];
    if (pad_token_id_ >= 0 && static_cast<size_t>(pad_token_id_) < id_to_token_.size()) pad_token_ = id_to_token_[pad_token_id_];

    Logger::info("Final Special Tokens (JSON constructor path): BOS=" + std::to_string(bos_token_id_) +
                     " ('" + bos_token_ + "'), EOS=" + std::to_string(eos_token_id_) + " ('" +
                     eos_token_ + "'), UNK=" + std::to_string(unk_token_id_) + " ('" +
                     unk_token_ + "'), PAD=" + std::to_string(pad_token_id_) + " ('" +
                     pad_token_ + "')"); // Removed extra backslashes from PAD log

    std::string init_log_message = "Tokenizer successfully initialized from JSON/Config. Detected type based on config: ";
    init_log_message += (tokenizer_family_ == ModelConfig::TokenizerFamily::LLAMA3_TIKTOKEN ? "LLAMA3_TIKTOKEN (assumed BPE)" :
                        (tokenizer_family_ == ModelConfig::TokenizerFamily::LLAMA_SENTENCEPIECE ? "LLAMA_SENTENCEPIECE (assumed BPE/SPM)" : "UNKNOWN"));
    Logger::info(init_log_message);

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

  Logger::info("Loaded " + std::to_string(id_to_token_.size()) +
               " tokens from vocabulary file: " + vocab_path);

  if (id_to_token_.size() > 0) {
    std::string first_few_tokens_log = "First few (up to 10 or vocab size) tokens from " + vocab_path + ": ";
    for (size_t i = 0; i < std::min((size_t)10, id_to_token_.size()); ++i) {
        first_few_tokens_log += "ID[" + std::to_string(i) + "]=";
        std::string escaped_token;
        for (char c_tok : id_to_token_[i]) {
            if (c_tok == '\\') {
                escaped_token += "\\\\";
            } else if (c_tok == '\'') {
                escaped_token += "\\'";
            } else if (std::isprint(static_cast<unsigned char>(c_tok))) {
                escaped_token += c_tok;
            } else {
                std::stringstream ss_hex;
                ss_hex << "<0x" << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(static_cast<unsigned char>(c_tok)) << ">";
                escaped_token += ss_hex.str();
            }
        }
         first_few_tokens_log += "'" + escaped_token + "' "; // Enclose in single quotes
    }
    Logger::info(first_few_tokens_log);
  }

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

static std::unordered_map<std::string, int> generate_bpe_merges_from_vocab_scores(
    const std::vector<std::string>& id_to_token,
    const std::vector<float>& token_scores) {
    
    std::unordered_map<std::string, int> generated_merges;
    
    if (token_scores.empty() || id_to_token.empty()) {
        Logger::warning("Cannot generate BPE merges: empty scores or vocabulary");
        return generated_merges;
    }
    
    Logger::info("Generating BPE merges from vocabulary and scores for older Llama models...");
    
    // Create a list of tokens with their scores, sorted by score (higher score = higher priority)
    std::vector<std::pair<float, std::string>> scored_tokens;
    for (size_t id = 0; id < id_to_token.size(); ++id) {
        if (id < token_scores.size()) {
            const std::string& token = id_to_token[id];
            // Skip special tokens and single characters
            if (token.length() > 1 && 
                token.find("<") == std::string::npos && 
                token.find(">") == std::string::npos &&
                token != "▁") {  // Skip SentencePiece space token
                scored_tokens.emplace_back(token_scores[id], token);
            }
        }
    }
    
    // Sort by score (descending - higher scores first)
    std::sort(scored_tokens.begin(), scored_tokens.end(), 
              [](const auto& a, const auto& b) { return a.first > b.first; });
    
    Logger::info("Found " + std::to_string(scored_tokens.size()) + " candidate tokens for merge generation");
    
    // Generate merges by finding tokens that can be decomposed into pairs
    int merge_rank = 0;
    std::unordered_set<std::string> processed_tokens;
    
    for (const auto& [score, token] : scored_tokens) {
        if (processed_tokens.count(token)) continue;
        
        // Try to find the best split point for this token
        std::string best_left, best_right;
        float best_combined_score = -std::numeric_limits<float>::infinity();
        
        // Try all possible split points
        for (size_t split = 1; split < token.length(); ++split) {
            std::string left = token.substr(0, split);
            std::string right = token.substr(split);
            
            // Check if both parts exist in vocabulary
            auto left_it = std::find(id_to_token.begin(), id_to_token.end(), left);
            auto right_it = std::find(id_to_token.begin(), id_to_token.end(), right);
            
            if (left_it != id_to_token.end() && right_it != id_to_token.end()) {
                // Both parts exist, calculate combined score
                size_t left_id = std::distance(id_to_token.begin(), left_it);
                size_t right_id = std::distance(id_to_token.begin(), right_it);
                float left_score = (left_id < token_scores.size()) ? 
                                 token_scores[left_id] : 0.0f;
                float right_score = (right_id < token_scores.size()) ? 
                                  token_scores[right_id] : 0.0f;
                float combined_score = left_score + right_score;
                
                if (combined_score > best_combined_score) {
                    best_combined_score = combined_score;
                    best_left = left;
                    best_right = right;
                }
            }
        }
        
        // If we found a valid decomposition, add it as a merge rule
        if (!best_left.empty() && !best_right.empty()) {
            std::string merge_key = best_left + best_right;
            if (generated_merges.find(merge_key) == generated_merges.end()) {
                generated_merges[merge_key] = merge_rank++;
                Logger::debug("Generated merge: '" + best_left + "' + '" + best_right + "' -> '" + token + "' (rank " + std::to_string(merge_rank-1) + ")");
            }
        }
        
        processed_tokens.insert(token);
        
        // Limit the number of merges to prevent excessive computation
        if (merge_rank >= 50000) {
            Logger::info("Reached maximum merge limit (50000), stopping generation");
            break;
        }
    }
    
    Logger::info("Generated " + std::to_string(generated_merges.size()) + " BPE merge rules from vocabulary and scores");
    return generated_merges;
}

Tokenizer::Tokenizer(const GGUFData& gguf_data, const ModelConfig& config)
    : tokenizer_family_(config.tokenizer_family),
      initialized_from_gguf_(true) {
  Logger::info("Initializing Tokenizer from GGUFData...");
  std::string family_str_gguf = "UNKNOWN";
  if (tokenizer_family_ == ModelConfig::TokenizerFamily::LLAMA_SENTENCEPIECE) family_str_gguf = "LLAMA_SENTENCEPIECE";
  else if (tokenizer_family_ == ModelConfig::TokenizerFamily::LLAMA3_TIKTOKEN) family_str_gguf = "LLAMA3_TIKTOKEN";
  Logger::info(std::string("Tokenizer family from ModelConfig: ") + family_str_gguf);

  // Attempt to load chat template from GGUF metadata
  try {
    auto it = gguf_data.metadata.find("tokenizer.chat_template");
    if (it != gguf_data.metadata.end()) {
      if (std::holds_alternative<std::string>(it->second)) {
        gguf_chat_template_ = std::get<std::string>(it->second);
        if (!gguf_chat_template_.empty()) {
            Logger::info("[Tokenizer GGUF Init] Found and loaded 'tokenizer.chat_template' from GGUF metadata.");
            // Further log the template content if it's not too long, or a snippet
            size_t log_len = std::min(gguf_chat_template_.length(), (size_t)70); // Log up to 70 chars
            std::string template_snippet = gguf_chat_template_.substr(0, log_len);
            if (gguf_chat_template_.length() > log_len) template_snippet += "...";
             // Replace newlines with printable \n for one-line logging
            std::string loggable_snippet;
            for (char ch : template_snippet) {
                if (ch == '\n') loggable_snippet += "\\n";
                else if (ch == '\r') loggable_snippet += "\\r";
                else if (ch == '\t') loggable_snippet += "\\t";
                else if (std::isprint(static_cast<unsigned char>(ch))) loggable_snippet += ch;
                else loggable_snippet += "."; // Replace non-printable with a dot
            }
            Logger::debug("[Tokenizer GGUF Init] Chat template snippet: " + loggable_snippet);
        } else {
            Logger::info("[Tokenizer GGUF Init] 'tokenizer.chat_template' found in GGUF metadata but is empty.");
        }
      } else {
        Logger::warning("[Tokenizer GGUF Init] 'tokenizer.chat_template' found in GGUF metadata but is not a string type.");
      }
    } else {
      Logger::info("[Tokenizer GGUF Init] 'tokenizer.chat_template' not found in GGUF metadata.");
    }
  } catch (const std::exception& e) {
    Logger::error("[Tokenizer GGUF Init] Exception while trying to access 'tokenizer.chat_template': " + std::string(e.what()));
  }

  if (gguf_data.tokenizer_tokens.empty()) {
      throw std::runtime_error(
        "GGUF data does not contain 'tokenizer.ggml.tokens'");
  }

  // Common vocabulary loading
  id_to_token_ = gguf_data.tokenizer_tokens;
  token_to_id_.clear(); // Ensure map is clear before populating
  token_to_id_.reserve(id_to_token_.size());
  for (size_t i = 0; i < id_to_token_.size(); ++i) {
    token_to_id_[id_to_token_[i]] = static_cast<int>(i);
    
    if (static_cast<int>(i) == 1734) {
        const std::string& token_at_1734 = id_to_token_[i];
        std::string escaped_token_1734;
        for (char c : token_at_1734) {
            if (c == '\n') escaped_token_1734 += "\\n";
            else if (c == '\r') escaped_token_1734 += "\\r";
            else if (c == '\t') escaped_token_1734 += "\\t";
            else if (c == '\\') escaped_token_1734 += "\\\\";
            else if (std::isprint(static_cast<unsigned char>(c))) escaped_token_1734 += c;
            else {
                std::stringstream ss_hex;
                ss_hex << "<0x" << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(static_cast<unsigned char>(c)) << ">";
                escaped_token_1734 += ss_hex.str();
            }
        }
        Logger::info("[GGUF_VOCAB_SCAN] Token string at ID 1734 is: '" + escaped_token_1734 + "' (length: " + std::to_string(token_at_1734.length()) + ")");
    }
    
  }
  Logger::info("Loaded " + std::to_string(id_to_token_.size()) +
               " tokens from GGUF tokenizer_tokens.");

  // Log first few tokens for inspection
  if (id_to_token_.size() > 0) {
    std::string first_few_tokens_log = "First few (up to 10 or vocab size) GGUF tokens: ";
    for (size_t i = 0; i < std::min((size_t)10, id_to_token_.size()); ++i) {
        first_few_tokens_log += "ID[" + std::to_string(i) + "]='";
        // Safely print token, escaping non-printables for logging
        for (char c_tok : id_to_token_[i]) {
            if (std::isprint(static_cast<unsigned char>(c_tok))) {
                first_few_tokens_log += c_tok;
  } else {
                std::stringstream ss_hex;
                ss_hex << "<0x" << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(static_cast<unsigned char>(c_tok)) << ">";
                first_few_tokens_log += ss_hex.str();
            }
        }
        first_few_tokens_log += "' ";
    }
    Logger::info(first_few_tokens_log);
  }

  // Conditional loading based on family
  if (tokenizer_family_ == ModelConfig::TokenizerFamily::LLAMA3_TIKTOKEN) {
    type_ = Type::TIKTOKEN_BPE; 
    Logger::info("Configuring for LLAMA3_TIKTOKEN (gpt2-style BPE).");

    if (gguf_data.tokenizer_merges.empty()) {
        Logger::warning("Llama 3 Tiktoken family specified, but GGUF data does not contain 'tokenizer.ggml.merges'. Tiktoken BPE may not function correctly without explicit merges.");
    } else {
        bpe_merges_.clear(); 
        int rank = 0;
        // Removed sample_merges vector and related logging logic
        for (const std::string& merge_str : gguf_data.tokenizer_merges) {
            std::string part1, part2;
            size_t space_pos = merge_str.find(' ');
            if (space_pos != std::string::npos && space_pos > 0 && space_pos < merge_str.length() - 1) {
                part1 = merge_str.substr(0, space_pos);
                part2 = merge_str.substr(space_pos + 1);
                std::string merged = part1 + part2;
                bpe_merges_[merged] = rank++; // Simplified rank assignment
            } else {
                Logger::warning("Skipping malformed Tiktoken merge rule from GGUF: '" + merge_str + "'");
            }
        }
        
        Logger::info("Processed " + std::to_string(bpe_merges_.size()) +
                 " Tiktoken merges from GGUF tokenizer_merges into bpe_merges_ map with ranks.");
    }
    // Scores are usually not the primary driver for Tiktoken BPE but load if present.
    if (!gguf_data.tokenizer_scores.empty()) {
        Logger::info("Llama 3 GGUF contains " + std::to_string(gguf_data.tokenizer_scores.size()) + " scores. Loaded.");
        token_scores_ = gguf_data.tokenizer_scores; 
    }

    // DEBUGGING: Log vocab/merges for neoplasm
    Logger::debug("[DEBUG_VOCAB] LLAMA3_TIKTOKEN bpe_merges_ size: " + std::to_string(bpe_merges_.size()));
    std::string target_token_neoplasm = BPE_SPACE_CHAR + "neoplasm"; // "Ġneoplasm"
    std::string target_sub_ne = BPE_SPACE_CHAR + "ne";          // "Ġne"
    std::string target_sub_o = BPE_SPACE_CHAR + "o";            // "Ġo"
    std::string target_sub_oplasm = "oplasm";
    std::string target_sub_goplasm = BPE_SPACE_CHAR + "oplasm"; // "Ġoplasm"

    auto check_and_log_vocab = [&](const std::string& token_to_check) {
        if (token_to_id_.count(token_to_check)) {
            Logger::debug("[DEBUG_VOCAB] Found '" + token_to_check + "' in vocab with ID: " + std::to_string(token_to_id_.at(token_to_check)));
        } else {
            Logger::debug("[DEBUG_VOCAB] Token '" + token_to_check + "' NOT FOUND in vocab.");
        }
    };

    auto check_and_log_merge = [&](const std::string& p1, const std::string& p2) {
        auto merge_it = bpe_merges_.find(p1 + p2);
        if (merge_it != bpe_merges_.end()) {
            Logger::debug("[DEBUG_VOCAB] Found merge for '" + p1 + "' + '" + p2 + "' ('" + (p1+p2) + "') with rank: " + std::to_string(merge_it->second));
        } else {
            Logger::debug("[DEBUG_VOCAB] Merge for '" + p1 + "' + '" + p2 + "' ('" + (p1+p2) + "') NOT FOUND.");
        }
    };

  } else if (tokenizer_family_ == ModelConfig::TokenizerFamily::LLAMA_SENTENCEPIECE) {
    type_ = Type::SENTENCEPIECE_BPE; 
    Logger::info("Configuring for LLAMA_SENTENCEPIECE.");
    if (!gguf_data.tokenizer_scores.empty()) {
        token_scores_ = gguf_data.tokenizer_scores;
        Logger::info("Loaded " + std::to_string(token_scores_.size()) + " token scores from GGUF for SentencePiece style.");
        if (id_to_token_.size() != token_scores_.size()) {  
          Logger::warning("GGUF (SentencePiece path) token and score array sizes mismatch: tokens=" +
                          std::to_string(id_to_token_.size()) + ", scores=" + std::to_string(token_scores_.size()));
      }
    } else {
        Logger::warning("SentencePiece family: No scores found. BPE merging will likely not work if no other SP model data is available.");
    }
    
    
    if (!gguf_data.tokenizer_merges.empty()) {
        Logger::info("SentencePiece family path: Found 'tokenizer.ggml.merges' in GGUF. Loading them into bpe_merges_ map.");
        bpe_merges_.clear(); 
        int rank = 0;
        for (const std::string& merge_str : gguf_data.tokenizer_merges) {
            std::string part1, part2;
            size_t space_pos = merge_str.find(' ');
            if (space_pos != std::string::npos && space_pos > 0 && space_pos < merge_str.length() - 1) {
                part1 = merge_str.substr(0, space_pos);
                part2 = merge_str.substr(space_pos + 1);
                bpe_merges_[part1 + part2] = rank++; 
            } else {
                Logger::warning("Skipping malformed SentencePiece merge rule from GGUF: '" + merge_str + "'");
            }
        }
        Logger::info("Processed " + std::to_string(bpe_merges_.size()) +
                     " merges from GGUF tokenizer_merges into bpe_merges_ map (SentencePiece path).");
    } else {
        Logger::warning("SentencePiece family path: No 'tokenizer.ggml.merges' found in GGUF. Attempting to generate merges from vocabulary and scores...");
        
        // Generate BPE merges from vocabulary and scores (llama.cpp approach)
        auto generated_merges = generate_bpe_merges_from_vocab_scores(id_to_token_, token_scores_);
        if (!generated_merges.empty()) {
            bpe_merges_ = std::move(generated_merges);
            Logger::info("Successfully generated " + std::to_string(bpe_merges_.size()) + " BPE merges from vocabulary and scores for SentencePiece tokenizer");
        } else {
            Logger::warning("Failed to generate BPE merges. Tokenization may be suboptimal for this model.");
        }
    }
    

  } else { // UNKNOWN tokenizer family
    type_ = Type::UNKNOWN;
    Logger::warning("Tokenizer family is UNKNOWN. Tokenizer may not function as expected. Will attempt to load basic vocab and scores if present.");
    if (!gguf_data.tokenizer_scores.empty()) {
        token_scores_ = gguf_data.tokenizer_scores;
        Logger::info("Loaded " + std::to_string(token_scores_.size()) + " token scores from GGUF for UNKNOWN family as a fallback.");
    }
  }

  if (!gguf_data.tokenizer_token_types.empty() && gguf_data.tokenizer_token_types.size() == id_to_token_.size()){
    token_types_.resize(gguf_data.tokenizer_token_types.size()); 
    std::transform(gguf_data.tokenizer_token_types.begin(),
                   gguf_data.tokenizer_token_types.end(), token_types_.begin(),
                   [](unsigned int u) { return static_cast<int32_t>(u); });
    Logger::info("Loaded and transformed " + std::to_string(token_types_.size()) + " token types from GGUF.");

    // Populate byte_char_to_id_ and added_tokens_ using token_types_
    byte_char_to_id_.clear();
    added_tokens_.clear();
    id_to_added_token_.clear();
    int byte_tokens_from_type = 0;
    int special_tokens_from_type = 0;

    for (size_t i = 0; i < token_types_.size(); ++i) {
        int32_t tt = token_types_[i];
        const std::string& token_str = id_to_token_[i];
      int token_id = static_cast<int>(i);
        bool processed_as_byte = false; // Flag to track if token was handled as byte

        if (tt == 6) { // LLAMA_TOKEN_TYPE_BYTE
            bool added_byte = false;
            if (token_str.length() == 1) {
                byte_char_to_id_[token_str[0]] = token_id;
                added_byte = true;
            } else if (token_str.rfind("<0x", 0) == 0 && token_str.back() == '>' && token_str.length() == 6) {
                 try {
                    int byte_val = std::stoi(token_str.substr(3, 2), nullptr, 16);
                    byte_char_to_id_[static_cast<char>(byte_val)] = token_id;
                    added_byte = true;
                 } catch (const std::exception& e) {
                    Logger::warning("Could not parse byte value from type-BYTE (6) token string: '" + token_str + "'");
                 }
            } else {
                // Log if a token is marked as BYTE but doesn't match expected formats
                Logger::warning("Token type is BYTE (6) but does not match single char or <0xNN> format: '" + token_str + "' ID: " + std::to_string(token_id));
            }

            if(added_byte) {
                 byte_tokens_from_type++;
                 processed_as_byte = true; 
            }
        }
        if (!processed_as_byte && (tt == 2 || tt == 3 || tt == 4 || tt == 5)) {
        if (added_tokens_.find(token_str) == added_tokens_.end()) {
            added_tokens_[token_str] = token_id;
            id_to_added_token_[token_id] = token_str;
                special_tokens_from_type++;
        }
      }
    }
    // Log message now reflects bytes identified from type 6 tokens
    Logger::info("From GGUF token_types (BYTE=6): Identified " + std::to_string(byte_tokens_from_type) + " byte tokens (for byte_char_to_id_). " +
                 "Identified " + std::to_string(special_tokens_from_type) + " other special/added tokens (types 2,3,4,5).");

    
    // If token types were processed but yielded no byte tokens for Tiktoken, try the fallback vocab scan.
    if (tokenizer_family_ == ModelConfig::TokenizerFamily::LLAMA3_TIKTOKEN && byte_tokens_from_type == 0) {
        Logger::warning("No byte tokens identified via token_types metadata for Tiktoken. Attempting fallback scan of vocabulary.");
        // Manually populate byte_char_to_id_ by checking vocab for <0xNN> and literal byte strings
        byte_char_to_id_.clear(); // Clear again in case some non-byte type 3 were added incorrectly before
        int bytes_found_in_vocab_fallback = 0;
        for (int i = 0; i < 256; ++i) {
            std::stringstream ss_hex_repr;
            ss_hex_repr << "<0x" << std::hex << std::setw(2) << std::setfill('0') << i << ">";
            std::string byte_token_str_repr = ss_hex_repr.str();
            std::string literal_byte_char_str(1, static_cast<char>(i));
            bool is_space_char = (static_cast<char>(i) == ' ');

            
            if (is_space_char) {
                Logger::debug("[BYTE_FALLBACK_DEBUG] Checking for SPACE (byte 32). Looking for '<0x20>' and ' '.");
            }
            

            auto it = token_to_id_.find(byte_token_str_repr);
            if (it != token_to_id_.end()) {
                
                if (is_space_char) {
                    Logger::debug("[BYTE_FALLBACK_DEBUG] Found '<0x20>' token with ID: " + std::to_string(it->second) + ". Adding to map.");
                }
                
                byte_char_to_id_[static_cast<char>(i)] = it->second;
                bytes_found_in_vocab_fallback++;
            } else {
                // Also check for literal single-byte characters if they are printable
                if (std::isprint(static_cast<unsigned char>(i))) { 
                     auto lit_it = token_to_id_.find(literal_byte_char_str);
                     if (lit_it != token_to_id_.end()) {
                        
                        if (is_space_char) {
                           Logger::debug("[BYTE_FALLBACK_DEBUG] Did not find '<0x20>', but found literal ' ' token with ID: " + std::to_string(lit_it->second));
                        }
                        
                        
                        // Ensure this token ID hasn't already been mapped (e.g., by a <0xNN> entry)
                        bool id_already_mapped = false;
                        for(const auto& pair : byte_char_to_id_) { if (pair.second == lit_it->second) { id_already_mapped = true; break; } }
                        if (!id_already_mapped) {
                            
                            if (is_space_char) {
                                Logger::debug("[BYTE_FALLBACK_DEBUG] ID " + std::to_string(lit_it->second) + " for ' ' not already mapped. Adding to map.");
                            }
                            
                            byte_char_to_id_[static_cast<char>(i)] = lit_it->second;
                            bytes_found_in_vocab_fallback++;
                            // Don't need a continue here, just prevents double-counting if somehow both exist
                        } else {
                            
                            if (is_space_char) {
                                Logger::debug("[BYTE_FALLBACK_DEBUG] ID " + std::to_string(lit_it->second) + " for ' ' was already mapped (likely by <0x20>). Skipping literal add.");
                            }
                            
                        }
                     } else {
                         
                         if (is_space_char) {
                             Logger::debug("[BYTE_FALLBACK_DEBUG] Did not find '<0x20>' OR literal ' ' token in vocab.");
                         }
                         
                     }
                } else {
                     
                     if (is_space_char) {
                         Logger::debug("[BYTE_FALLBACK_DEBUG] Did not find '<0x20>' token, and space is not printable, so didn't check for literal ' '.");
                     }
                     
                }
            }
        }
        Logger::info("Fallback byte_char_to_id_ map population: Found representations for " + std::to_string(bytes_found_in_vocab_fallback) +
                     " byte values in GGUF vocab (using <0xNN> or literal). Intended for Tiktoken BPE.");
        byte_tokens_from_type = bytes_found_in_vocab_fallback; 
    }
    

  } else {
    Logger::warning("GGUF tokenizer_token_types array missing or size mismatch. Byte token and special token identification will be limited.");
    if (tokenizer_family_ == ModelConfig::TokenizerFamily::LLAMA3_TIKTOKEN) {
        byte_char_to_id_.clear();
        int bytes_found_in_vocab_fallback = 0;
        for (int i = 0; i < 256; ++i) {
            std::stringstream ss_hex_repr;
            ss_hex_repr << "<0x" << std::hex << std::setw(2) << std::setfill('0') << i << ">";
            std::string byte_token_str_repr = ss_hex_repr.str();
            std::string literal_byte_char_str(1, static_cast<char>(i));
            bool is_space_char = (static_cast<char>(i) == ' ');

            
            if (is_space_char) {
                Logger::debug("[BYTE_FALLBACK_DEBUG] Checking for SPACE (byte 32). Looking for '<0x20>' and ' '.");
            }
            

            auto it = token_to_id_.find(byte_token_str_repr);
            if (it != token_to_id_.end()) {
                
                if (is_space_char) {
                    Logger::debug("[BYTE_FALLBACK_DEBUG] Found '<0x20>' token with ID: " + std::to_string(it->second) + ". Adding to map.");
                }
                
                byte_char_to_id_[static_cast<char>(i)] = it->second;
                bytes_found_in_vocab_fallback++;
            } else {
                // Also check for literal single-byte characters if they are printable
                if (std::isprint(static_cast<unsigned char>(i))) { 
                     auto lit_it = token_to_id_.find(literal_byte_char_str);
                     if (lit_it != token_to_id_.end()) {
                        
                        if (is_space_char) {
                           Logger::debug("[BYTE_FALLBACK_DEBUG] Did not find '<0x20>', but found literal ' ' token with ID: " + std::to_string(lit_it->second));
                        }
                        
                        
                        // Ensure this token ID hasn't already been mapped (e.g., by a <0xNN> entry)
                        bool id_already_mapped = false;
                        for(const auto& pair : byte_char_to_id_) { if (pair.second == lit_it->second) { id_already_mapped = true; break; } }
                        if (!id_already_mapped) {
                            
                            if (is_space_char) {
                                Logger::debug("[BYTE_FALLBACK_DEBUG] ID " + std::to_string(lit_it->second) + " for ' ' not already mapped. Adding to map.");
                            }
                            
                            byte_char_to_id_[static_cast<char>(i)] = lit_it->second;
                            bytes_found_in_vocab_fallback++;
                            continue;
                        } else {
                            
                            if (is_space_char) {
                                Logger::debug("[BYTE_FALLBACK_DEBUG] ID " + std::to_string(lit_it->second) + " for ' ' was already mapped (likely by <0x20>). Skipping literal add.");
                            }
                            
                        }
                     } else {
                         
                         if (is_space_char) {
                             Logger::debug("[BYTE_FALLBACK_DEBUG] Did not find '<0x20>' OR literal ' ' token in vocab.");
                         }
                         
                     }
                } else {
                     
                     if (is_space_char) {
                         Logger::debug("[BYTE_FALLBACK_DEBUG] Did not find '<0x20>' token, and space is not printable, so didn't check for literal ' '.");
                     }
                     
                }
            }
        }
        Logger::info("Fallback byte_char_to_id_ map population: Found representations for " + std::to_string(bytes_found_in_vocab_fallback) +
                     " byte values in GGUF vocab (using <0xNN> or literal). Intended for Tiktoken BPE.");
    }
  }
  
  
  if (byte_char_to_id_.find(' ') == byte_char_to_id_.end()) {
    Logger::info("[GENERAL_BYTE_FALLBACK] Space ' ' not found in byte_char_to_id_. Attempting to populate from vocab.");
    int general_fallback_bytes_added = 0;
    for (int i = 0; i < 256; ++i) {
        char current_char = static_cast<char>(i);
        // Only add if not already present from a more primary source (like token_types)
        if (byte_char_to_id_.count(current_char)) {
            continue;
        }

        std::stringstream ss_hex_repr;
        ss_hex_repr << "<0x" << std::hex << std::setw(2) << std::setfill('0') << i << ">";
        std::string byte_token_str_repr = ss_hex_repr.str();
        std::string literal_byte_char_str(1, current_char);

        auto it_hex = token_to_id_.find(byte_token_str_repr);
        if (it_hex != token_to_id_.end()) {
            byte_char_to_id_[current_char] = it_hex->second;
            general_fallback_bytes_added++;
            if (current_char == ' ') Logger::debug("[GENERAL_BYTE_FALLBACK] Found space as '" + byte_token_str_repr + "' -> ID: " + std::to_string(it_hex->second));
        } else {
            auto it_lit = token_to_id_.find(literal_byte_char_str);
            if (it_lit != token_to_id_.end()) {
                byte_char_to_id_[current_char] = it_lit->second;
                general_fallback_bytes_added++;
                if (current_char == ' ') Logger::debug("[GENERAL_BYTE_FALLBACK] Found space as literal '" + literal_byte_char_str + "' -> ID: " + std::to_string(it_lit->second));
            }
        }
    }
    Logger::info("[GENERAL_BYTE_FALLBACK] Added " + std::to_string(general_fallback_bytes_added) +
                 " new entries to byte_char_to_id_ map. Final size: " + std::to_string(byte_char_to_id_.size()));
    if (byte_char_to_id_.find(' ') == byte_char_to_id_.end()) {
        Logger::warning("[GENERAL_BYTE_FALLBACK] Space ' ' still not found in byte_char_to_id_ after fallback scan!");
    }

    
    if (byte_char_to_id_.find(' ') == byte_char_to_id_.end()) { // Check again if space wasn't found by hex/literal
        const std::string sp_space_token = "\xE2\x96\x81"; // U+2581
        auto it_sp_space = token_to_id_.find(sp_space_token);
        if (it_sp_space != token_to_id_.end()) {
            byte_char_to_id_[' '] = it_sp_space->second; // Map standard space char to the ID of the SP space token
            Logger::info("[GENERAL_BYTE_FALLBACK] SUCCESS: Found SentencePiece space token '" + sp_space_token + 
                         "' (ID: " + std::to_string(it_sp_space->second) + "). Mapped standard space ' ' to this ID.");
        } else {
             // This is the final warning if space still not found
             Logger::warning("[GENERAL_BYTE_FALLBACK] Space ' ' still not found in byte_char_to_id_ after fallback scan AND specific SP space check!");
        }
    }
  }
  
  
bos_token_id_ = config.bos_token_id;
  eos_token_id_ = config.eos_token_id;
  unk_token_id_ = config.unk_token_id;
  pad_token_id_ = config.pad_token_id;

  
  // Ensure UNK token ID is valid (non-negative). Default to 0 if invalid.
  if (unk_token_id_ < 0) {
      Logger::warning("[Tokenizer GGUF Init] UNK token ID from config was invalid (" + std::to_string(unk_token_id_) + "). Forcing to 0.");
      unk_token_id_ = 0; 
  }
  

  auto setup_special_token = [&](const std::string& name, int& id_field, std::string& str_field, const std::string& default_str_val) {
    if (id_field >= 0 && static_cast<size_t>(id_field) < id_to_token_.size()) {
        str_field = id_to_token_[id_field];
    } else {
        str_field = default_str_val; // Use default string if ID is invalid or -1
        if (id_field != -1) { // Log warning only if ID was supposed to be valid but wasn't found
             Logger::warning(name + " token ID " + std::to_string(id_field) + 
                           " from config is out of vocab bounds or invalid. Using default string: '" + default_str_val + "'.");
        }
        // Attempt to find the default string in the vocab to set its ID, if ID was bad
        auto it = token_to_id_.find(default_str_val);
        if (it != token_to_id_.end()) {
            if (id_field == -1 || (id_field >=0 && static_cast<size_t>(id_field) >= id_to_token_.size()) ) { // If original ID was invalid/none
                 id_field = it->second;
                 Logger::info("Set " + name + " token ID to " + std::to_string(id_field) + " based on default string '" + default_str_val + "'.");
            }
        } else if (id_field != -1) {
             Logger::warning("Default string '" + default_str_val + "' for " + name + " token also not found in vocab.");
        }
    }
  };

  setup_special_token("BOS", bos_token_id_, bos_token_, "<s>");
  setup_special_token("EOS", eos_token_id_, eos_token_, "</s>");
  setup_special_token("UNK", unk_token_id_, unk_token_, "<unk>");
  // For PAD, if config.pad_token_id is -1, it means no pad token. String should be empty.
  // If it's a valid ID, str_field will be set. If it's an invalid positive ID, str_field becomes <pad> by default.
  if (config.pad_token_id == -1) {
      pad_token_ = ""; // Explicitly empty if ID is -1
      // bos_token_id_ etc are already set directly from config so no change needed for id_field here for pad_token_id_ == -1
  } else {
      setup_special_token("PAD", pad_token_id_, pad_token_, "<pad>");
    }

  Logger::info("Final Special Tokens (GGUF constructor): BOS ID=" + std::to_string(bos_token_id_) +
                 " ('" + bos_token_ + "'), EOS ID=" + std::to_string(eos_token_id_) + " ('" + eos_token_ + 
                 "'), UNK ID=" + std::to_string(unk_token_id_) + " ('" + unk_token_ +
                 "'), PAD ID=" + std::to_string(pad_token_id_) + " ('" + pad_token_ + "\\\\')");

  Logger::info(std::string("Tokenizer successfully initialized from GGUFData. Final type: ") + 
    (type_ == Type::TIKTOKEN_BPE ? "TIKTOKEN_BPE" : 
     (type_ == Type::SENTENCEPIECE_BPE ? "SENTENCEPIECE_BPE" : "UNKNOWN")));
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



// Function to determine the length of a UTF-8 character based on its first byte.
// Similar to the lookup method used in llama.cpp.
// Keep this function local to this translation unit.
namespace {
    inline size_t unicode_char_len(char src) {
        const size_t lookup[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4 };
        uint8_t highbits = static_cast<uint8_t>(src) >> 4;
        // Bounds check for safety, although highbits should always be 0-15
        return (highbits < 16) ? lookup[highbits] : 1; // Default to 1 for invalid highbits
    }
} // end anonymous namespace

std::vector<int> Tokenizer::encode(const std::string& text, bool add_bos,
                                   bool add_eos,
                                   PreTokenizeMethod pre_tok_override) const {
  std::vector<int> final_ids; // Initialize the vector to store final token IDs.
  std::string family_str_enc = "UNKNOWN"; // String representation for logging.

  // Determine the tokenizer family string for logging purposes.
  if (tokenizer_family_ == ModelConfig::TokenizerFamily::LLAMA_SENTENCEPIECE) family_str_enc = "LLAMA_SENTENCEPIECE";
  else if (tokenizer_family_ == ModelConfig::TokenizerFamily::LLAMA3_TIKTOKEN) family_str_enc = "LLAMA3_TIKTOKEN";

  // Log the start of the encoding process with relevant parameters.
  std::stringstream log_ss_main;
  log_ss_main << "[ENCODE] Encoding text: '" << text << "'"
              << " (add_bos=" << add_bos
              << ", add_eos=" << add_eos
              << ", family=" << family_str_enc
              << ", pre_tok_override=" << static_cast<int>(pre_tok_override)
              << ")";
  Logger::debug(log_ss_main.str());

  
  if (tokenizer_family_ == ModelConfig::TokenizerFamily::LLAMA3_TIKTOKEN) {
    Logger::debug("[ENCODE] Using LLAMA3_TIKTOKEN (bpe_tokenize_to_ids) path.");
    
    if (add_bos && this->bos_token_id_ != -1) {
        // Check if the text already starts with the BOS token string
        if (this->bos_token_.empty() || text.rfind(this->bos_token_, 0) != 0) {
            final_ids.push_back(this->bos_token_id_);
            Logger::debug("[ENCODE Llama 3 Path] Added BOS token: " + std::to_string(this->bos_token_id_) +
                          " (text did not already start with it).");
        } else {
            Logger::debug("[ENCODE Llama 3 Path] BOS token flag was true, but text already started with BOS string. Skipping explicit BOS ID addition.");
        }
    }
    
    std::vector<int> token_ids = this->bpe_tokenize_to_ids(text, false, false, false); 
    final_ids.insert(final_ids.end(), token_ids.begin(), token_ids.end());
    
    if (add_eos && this->eos_token_id_ != -1) {
        final_ids.push_back(this->eos_token_id_);
        Logger::debug("[ENCODE Llama 3 Path] Added EOS token: " + std::to_string(this->eos_token_id_));
    }

  
  } else if (tokenizer_family_ == ModelConfig::TokenizerFamily::LLAMA_SENTENCEPIECE) {
    Logger::debug("[ENCODE] Using LLAMA_SENTENCEPIECE (old SentencePiece/BPE logic) path.");
    
    if (!this->initialized_from_gguf_) { 
        Logger::debug(
            "[ENCODE SPM Path] Using simplified merge-based tokenizer path (calling "
            "bpe_tokenize directly).");

        std::vector<std::string> bpe_pieces = this->bpe_tokenize(text); 
        Logger::debug("[ENCODE SPM Path] bpe_tokenize returned " +
                      std::to_string(bpe_pieces.size()) + " pieces.");

        final_ids = this->tokens_to_ids(bpe_pieces); 

        if (add_bos && this->bos_token_id_ != -1) {
            final_ids.insert(final_ids.begin(), this->bos_token_id_);
            Logger::debug("[ENCODE SPM Path] Prepended BOS token: " +
                        std::to_string(this->bos_token_id_));
        }
        if (add_eos && this->eos_token_id_ != -1) {
            final_ids.push_back(this->eos_token_id_);
            Logger::debug("[ENCODE SPM Path] Appended EOS token: " +
                        std::to_string(this->eos_token_id_));
        }
        Logger::debug("[ENCODE SPM Path] Final IDs (Simplified Merge Path): " +
                      std::to_string(final_ids.size()) + " tokens.");
    } else { 
        Logger::debug("[ENCODE SPM Path] Using GGUF score-based tokenizer path.");

        if (add_bos && this->bos_token_id_ != -1) {
            final_ids.push_back(this->bos_token_id_);
            Logger::debug("[ENCODE SPM GGUF Path] Added BOS token: " +
                        std::to_string(this->bos_token_id_));
        }

        std::vector<std::pair<std::string, bool>> segments; 
        std::string text_to_process = text;
        PreTokenizeMethod method_to_use;

        if (pre_tok_override == PreTokenizeMethod::DEFAULT) {
            if (this->pre_tok_type_ == "default") { 
                method_to_use = PreTokenizeMethod::DEFAULT;
                 Logger::debug("[ENCODE SPM GGUF Path] Using DEFAULT pre-tokenization (split by special, BPE for non-specials).");
            } else if (this->pre_tok_type_ == "llama") {
                method_to_use = PreTokenizeMethod::LLAMA_REGEX;
                 Logger::debug("[ENCODE SPM GGUF Path] Using LLAMA_REGEX pre-tokenization.");
            } else {
                Logger::warning("[ENCODE SPM GGUF Path] pre_tok_type_ is '" + this->pre_tok_type_ + "' or unset. Defaulting to WHITESPACE pre-tokenization for GGUF/SPM path.");
                method_to_use = PreTokenizeMethod::DEFAULT; // Fallback to DEFAULT, whitespace logic handled below
            }
        } else {
            method_to_use = pre_tok_override;
        }

        std::string method_str_log;
        if (method_to_use == PreTokenizeMethod::LLAMA_REGEX) method_str_log = "LLAMA_REGEX";
        else method_str_log = "DEFAULT (Special Token Split or WHITESPACE Fallback)";
        Logger::debug("[ENCODE SPM GGUF Path] Effective pre-tokenization method: " + method_str_log);

        if (method_to_use == PreTokenizeMethod::DEFAULT && this->pre_tok_type_ == "default") {
            std::unordered_set<std::string> all_special_tokens_set;
            for (const auto& pair : this->added_tokens_) {
                if (!pair.first.empty()) all_special_tokens_set.insert(pair.first);
            }
            if (!this->bos_token_.empty()) all_special_tokens_set.insert(this->bos_token_);
            if (!this->eos_token_.empty()) all_special_tokens_set.insert(this->eos_token_);
            if (!this->unk_token_.empty()) all_special_tokens_set.insert(this->unk_token_); 

      std::string special_pattern_str = "(";
      bool first_special = true;
            for (const std::string& st : all_special_tokens_set) {
          if (!first_special) special_pattern_str += "|";
                std::string escaped_st;
                for (char c : st) {
                    if (strchr(".^$*+?()[{\\|", c)) escaped_st += '\\';
                    escaped_st += c;
                }
                special_pattern_str += escaped_st;
          first_special = false;
      }
      special_pattern_str += ")";

            if (all_special_tokens_set.empty()) { 
                Logger::debug("[ENCODE SPM GGUF Path] No special tokens defined for DEFAULT pre-tok. Treating whole text as one segment.");
                segments.push_back({text_to_process, false});
      } else {
                Logger::debug("[ENCODE SPM GGUF Path] Splitting by special tokens regex: " + special_pattern_str);
                try {
                    boost::regex special_regex(special_pattern_str); 
                    boost::sregex_iterator it(text_to_process.begin(), text_to_process.end(), special_regex);
                    boost::sregex_iterator end;
                    size_t last_pos = 0;
                    while (it != end) {
                        boost::smatch match = *it;
                        if (match.position() > last_pos) {
                            segments.push_back({text_to_process.substr(last_pos, match.position() - last_pos), false});
                        }
                        segments.push_back({match.str(), true}); 
                        last_pos = match.position() + match.length();
                        ++it;
                    }
                    if (last_pos < text_to_process.length()) {
                        segments.push_back({text_to_process.substr(last_pos), false});
                    }
                } catch (const boost::regex_error& e) { 
                    Logger::error("[ENCODE SPM GGUF Path] Regex error splitting by special tokens: " + std::string(e.what()) + ". Treating as single segment.");
                    segments.clear();
                    segments.push_back({text_to_process, false});
                }
            }
        } else if (method_to_use == PreTokenizeMethod::LLAMA_REGEX) {
             boost::regex llama_segment_regex( R"('s|'t|'re|'ve|'m|'ll|'d| ?[[:alpha:]]+| ?[[:digit:]]+| ?[^\\s[:alpha:][:digit:]]+|\\s+(?!\\S)|\\s+)");
             Logger::debug("[ENCODE SPM GGUF Path] Using LLAMA_REGEX for pre-tokenization.");
             try {
                boost::sregex_iterator it(text_to_process.begin(), text_to_process.end(), llama_segment_regex);
                boost::sregex_iterator end;
                size_t last_pos = 0;
                while(it != end) {
                    boost::smatch match = *it;
                    if (match.position() > last_pos) { 
                        segments.push_back({text_to_process.substr(last_pos, match.position() - last_pos), false});
                    }
                    segments.push_back({match.str(), false}); 
                    last_pos = match.position() + match.length();
                    ++it;
                }
                if (last_pos < text_to_process.length()) { 
                     segments.push_back({text_to_process.substr(last_pos), false});
                }
             } catch (const boost::regex_error& e) { 
                  Logger::error("[ENCODE SPM GGUF Path] Regex error during LLAMA_REGEX splitting: " + std::string(e.what()) + ". Treating as single segment.");
                  segments.clear();
                  segments.push_back({text_to_process, false});
             }
        } else { // WHITESPACE or fallback (method_to_use is DEFAULT here if pre_tok_type_ was not "default")
            Logger::debug("[ENCODE SPM GGUF Path] Using WHITESPACE pre-tokenization (or fallback).");
            std::string current_ws_segment;
            for (char c : text_to_process) {
                if (std::isspace(static_cast<unsigned char>(c))) {
                    if (!current_ws_segment.empty()) {
                        segments.push_back({current_ws_segment, false});
                        current_ws_segment.clear();
                    }
                    segments.push_back({{c}, false}); 
                } else {
                    current_ws_segment += c;
                }
            }
            if (!current_ws_segment.empty()) {
                segments.push_back({current_ws_segment, false});
            }
        }

        Logger::debug("[ENCODE SPM GGUF Path] Pre-tokenization resulted in " + std::to_string(segments.size()) + " segments.");

        std::vector<int> segment_ids;
        for (const auto& seg_pair : segments) {
            const std::string& segment_str = seg_pair.first;
            bool is_special = seg_pair.second;

            if (segment_str.empty()) continue;

        if (is_special) {
                auto it = this->token_to_id_.find(segment_str);
                if (it != this->token_to_id_.end()) {
                    segment_ids.push_back(it->second);
                    Logger::debug("[ENCODE SPM GGUF Path] Found special segment: '" + segment_str + "' -> ID: " + std::to_string(it->second));
                    } else {
                    Logger::warning("[ENCODE SPM GGUF Path] Special segment '" + segment_str +
                                  "' not in vocab. Using UNK ID: " + std::to_string(this->unk_token_id_));
                    segment_ids.push_back(this->unk_token_id_);
          }
        } else {
                std::vector<std::string> pieces = this->bpe_tokenize_from_scores(segment_str);
                std::vector<int> piece_ids = this->tokens_to_ids(pieces);
                segment_ids.insert(segment_ids.end(), piece_ids.begin(), piece_ids.end());
                Logger::debug("[ENCODE SPM GGUF Path] BPE for non-special segment '" + segment_str + "' -> " + std::to_string(piece_ids.size()) + " IDs.");
            }
        }
            final_ids.insert(final_ids.end(), segment_ids.begin(), segment_ids.end());

        if (add_eos && this->eos_token_id_ != -1) {
            final_ids.push_back(this->eos_token_id_);
            Logger::debug("[ENCODE SPM GGUF Path] Appended EOS token: " +
                        std::to_string(this->eos_token_id_));
        }
        Logger::debug("[ENCODE SPM GGUF Path] Final IDs (GGUF Score Path): " + std::to_string(final_ids.size()) + " tokens.");
    } 
  } // This closes the LLAMA_SENTENCEPIECE block
  else { // Unknown Tokenizer Family Path
    Logger::error("[ENCODE] Unknown or unsupported tokenizer family: " + family_str_enc + ". Cannot encode text.");
    if (add_bos && this->bos_token_id_ != -1) {
       final_ids.push_back(this->bos_token_id_); 
       Logger::debug("[ENCODE Unknown Path] Added BOS token: " + std::to_string(this->bos_token_id_));
    }
    if (add_eos && this->eos_token_id_ != -1) {
        final_ids.push_back(this->eos_token_id_);
        Logger::debug("[ENCODE Unknown Path] Added EOS token: " + std::to_string(this->eos_token_id_));
    }
  }

  Logger::debug("[ENCODE] Final IDs count (end of function): " + std::to_string(final_ids.size()));
  if (final_ids.empty() && !text.empty()) {
      Logger::warning("[ENCODE] Tokenization resulted in empty ID list for non-empty text: '" + text + "'");
  }
  
        return final_ids;
}

std::string Tokenizer::decode(const std::vector<int>& ids,
                              bool skip_special_tokens) const {
    // Dispatch based on tokenizer family
    if (tokenizer_family_ == ModelConfig::TokenizerFamily::LLAMA_SENTENCEPIECE) {
        // Use the dedicated SentencePiece decoding logic
        return decode_sentencepiece(ids, skip_special_tokens);
    }

    // Default to Llama 3 / Tiktoken BPE decoding logic
    Logger::debug("[decode] Decoding using Llama 3 / Tiktoken logic.");
    std::stringstream ss;
    bool first_token = true;  

    for (int id : ids) {
        // Handle potential invalid IDs first
        if (id < 0 || static_cast<size_t>(id) >= id_to_token_.size()) {
             if (!skip_special_tokens) { // Only show invalid ID if not skipping specials
                 ss << "[INVALID_ID:" << id << "]";
                 Logger::debug("[decode] Invalid token ID: " + std::to_string(id));
                 first_token = false; // Considered as outputted content
             }
             continue;
        }

        // Handle special tokens skip
        if (skip_special_tokens) {
            if (id == bos_token_id_ || id == eos_token_id_ || id == pad_token_id_ || id == unk_token_id_) {
                 Logger::debug("[decode] Skipping special token ID: " + std::to_string(id) + 
                             " (BOS/EOS/PAD/UNK)");
                 continue;
            }
            if (id_to_added_token_.count(id)) {
                Logger::debug("[decode] Skipping added token ID: " + std::to_string(id));
                continue;  
            }
        }
        
        std::string token = id_to_token_[id];
        std::string token_debug = token;
        // Make non-printable characters visible in logs
        for (size_t i = 0; i < token_debug.length(); i++) {
            if (!std::isprint(static_cast<unsigned char>(token_debug[i]))) {
                char hex[5];
                snprintf(hex, sizeof(hex), "\\x%02x", static_cast<unsigned char>(token_debug[i]));
                token_debug.replace(i, 1, hex);
                i += 3; // Skip the added hex chars
            }
        }
        Logger::debug("[decode] Processing token ID " + std::to_string(id) + 
                     ": '" + token_debug + "'");

        if (token.empty()) {
             if (!skip_special_tokens && unk_token_id_ != -1) { 
                 token = unk_token_; 
                 Logger::debug("[decode] Empty token replaced with UNK token");
        } else {
                 Logger::debug("[decode] Empty token skipped");
                 continue; 
             }
        }
        
        if (token.size() >= BPE_SPACE_CHAR.size() &&
            token.substr(0, BPE_SPACE_CHAR.size()) == BPE_SPACE_CHAR) {
            if (!first_token) { 
                ss << " "; 
                Logger::debug("[decode] Added space before token with BPE_SPACE_CHAR prefix");
            }
            ss << token.substr(BPE_SPACE_CHAR.size()); 
            Logger::debug("[decode] Added token content after BPE_SPACE_CHAR: '" + 
                         token.substr(BPE_SPACE_CHAR.size()) + "'");
            first_token = false;
        } else { // Token does NOT start with Ġ
            ss << token; // Append the token itself
            Logger::debug("[decode] Added non-BPE_SPACE_CHAR token: '" + token + "'");
            first_token = false;  // Still set first_token to false as content has been added
        }
    }
    std::string final_text = ss.str();
    Logger::debug("[decode] Final decoded text: '" + final_text + "'");
    return final_text;
}

std::string Tokenizer::decode_sentencepiece(const std::vector<int>& ids,
                                            bool skip_special_tokens) const {
    Logger::debug("[decode_sentencepiece] Decoding using SentencePiece logic.");
   
    std::stringstream ss;
    bool first_token = true; // Flag to handle potential leading space correctly
    const std::string sp_space_prefix = "\xE2\x96\x81"; // Actual UTF-8 sequence for U+2581
    const std::string gpt2_space_prefix = "\xC4\xA0"; // Actual UTF-8 sequence for U+0120 (Ġ) - check just in case

    for (int id : ids) {
        std::string token_str; // Holds the string representation of the current token
        bool is_special_or_invalid = false;

        // Handle special token skipping FIRST
        if (skip_special_tokens) {
            if (id == bos_token_id_ || id == eos_token_id_ || id == pad_token_id_ || id == unk_token_id_) {
                is_special_or_invalid = true;
                continue; // Skip this token entirely
            }
            if (id_to_added_token_.count(id)) {
                is_special_or_invalid = true;
                continue; // Skip added special tokens
            }
        }

        // Get the token string (handling invalid IDs)
        if (id >= 0 && static_cast<size_t>(id) < id_to_token_.size()) {
            token_str = id_to_token_[id];
        } else {
            auto added_it = id_to_added_token_.find(id);
            if (added_it != id_to_added_token_.end()) {
                token_str = added_it->second; // It's an added token (might be special)
            } else { // Truly invalid ID
                // Don't output if skipping specials/invalid
                if (!skip_special_tokens) {
                    token_str = "[INVALID_ID:" + std::to_string(id) + "]";
                } else {
                    token_str = ""; // Effectively skip
                }
                is_special_or_invalid = true; // Treat invalid ID as special for spacing
            }
        }

        // NEW: Check for <0xNN> format and convert if necessary
        if (token_str.length() == 6 && token_str.rfind("<0x", 0) == 0 && token_str[5] == '>') {
            try {
                std::string hex_val_str = token_str.substr(3, 2);
                int byte_val = std::stoi(hex_val_str, nullptr, 16);
                token_str = std::string(1, static_cast<char>(byte_val));
                Logger::debug("[decode_sentencepiece] Converted '<0x" + hex_val_str + ">' to char: " + std::to_string(byte_val));
            } catch (const std::exception& e) {
                Logger::warning("[decode_sentencepiece] Failed to parse hex from token: '" + token_str + "'. Error: " + e.what());
                // Keep original token_str if parsing fails
            }
        }

        if (token_str.empty() && !is_special_or_invalid) {
            if (unk_token_id_ != -1) {
                 // Check if UNK should be skipped
                if (!skip_special_tokens || unk_token_id_ != id) { // Check if UNK is the ID causing emptiness if not skipping
                   token_str = unk_token_;
                } else {
                     // Skipping specials, and UNK is considered special
                     is_special_or_invalid = true;
                     continue; // Skip this empty token
                }
            } else {
                 if (!skip_special_tokens){
                      token_str = "[EMPTY_TOKEN_FOR_ID:" + std::to_string(id) + "]";
                 } else {
                     is_special_or_invalid = true;
                     continue; // Skip
                 }
            }
            if (!is_special_or_invalid) { // Only log if we actually output something
                 Logger::warning("[decode_sentencepiece] Encountered empty token string for valid ID " + std::to_string(id) +
                               ". Using: '" + token_str + "'");
            }
        }

        // Handle cases where the token IS the space prefix itself
        if (token_str == sp_space_prefix || token_str == gpt2_space_prefix) {
            if (first_token) {
                // If it's the first token and it's just a space prefix, ignore it and wait for actual content.
                // first_token remains true.
                Logger::debug("[decode_sentencepiece] Ignored leading standalone space prefix token.");
                continue;
            }
            // If not the first token, and it's a standalone space prefix, ensure one space is added.
            std::string current_output_check = ss.str();
            if (current_output_check.empty() || current_output_check.back() != ' ') {
                ss << " ";
                Logger::debug("[decode_sentencepiece] Added space for standalone prefix token mid-sequence.");
            }
            first_token = false; // A space was effectively output.
            continue; // Move to the next token.
        }

        // Process the token string: handle prefixes and append to result
        if (!token_str.empty()) { 
            bool starts_with_sp_prefix = (token_str.rfind(sp_space_prefix, 0) == 0);
            // Check for GPT2 prefix only if SP prefix wasn't found
            bool starts_with_gpt2_prefix = (!starts_with_sp_prefix && token_str.rfind(gpt2_space_prefix, 0) == 0);

            if (starts_with_sp_prefix) {
                std::string current_output = ss.str();
                if (!first_token && (current_output.empty() || current_output.back() != ' ')) {
                    ss << " ";
                }
                std::string content = token_str.substr(sp_space_prefix.length());
                // RE-ADD: Trim any leading literal spaces from the content itself
                size_t first_non_space = content.find_first_not_of(' ');
                if (std::string::npos != first_non_space) {
                    content = content.substr(first_non_space);
                }
                ss << content;
                first_token = false; // We have outputted something
            }
            else if (starts_with_gpt2_prefix) { // Handle Ġ prefix if found
                std::string current_output = ss.str();
                if (!first_token && (current_output.empty() || current_output.back() != ' ')) {
                    ss << " ";
                }
                std::string content = token_str.substr(gpt2_space_prefix.length());
                // RE-ADD: Trim any leading literal spaces from the content itself
                size_t first_non_space = content.find_first_not_of(' ');
                if (std::string::npos != first_non_space) {
                    content = content.substr(first_non_space);
                }
                ss << content;
                first_token = false;
            }
            else { // Token does not start with a known space prefix
                ss << token_str;
                first_token = false; // Mark that we've outputted content
            }
        }
    } // End for loop over IDs

    return ss.str();
}

// Helper function to replace all occurrences of a substring
static std::string replace_all(std::string str, const std::string& from, const std::string& to) {
    size_t start_pos = 0;
    while((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length(); // Handles cases where 'to' is a substring of 'from'
    }
    return str;
}

std::string Tokenizer::apply_chat_template(const std::string& user_prompt,
                                           const std::string& system_message,
                                           const ModelConfig& config) const { 
  // Check if the GGUF template seems like a Jinja2 template
  bool is_jinja_template = (!gguf_chat_template_.empty() && 
                            (gguf_chat_template_.find("{%") != std::string::npos || 
                             gguf_chat_template_.find("{{") != std::string::npos));

  // Log the determined template type and GGUF template content for debugging
  if (!gguf_chat_template_.empty()) {
      Logger::debug("[apply_chat_template] GGUF chat template content (first 100 chars): " + gguf_chat_template_.substr(0, 100));
      if (is_jinja_template) {
          Logger::info("[apply_chat_template] GGUF chat template detected as Jinja2-like.");
      } else {
          Logger::info("[apply_chat_template] GGUF chat template detected as simple placeholder template.");
      }
  }

  if (!gguf_chat_template_.empty() && !is_jinja_template) {
    Logger::info("[apply_chat_template] Using simple GGUF chat template (non-Jinja).");
    std::string processed_template = gguf_chat_template_;
    
    std::string bos_s = this->bos_token_id_ != -1 ? this->bos_token_ : "";
    std::string eos_s = this->eos_token_id_ != -1 ? this->eos_token_ : "";

    processed_template = replace_all(processed_template, "{{bos_token}}", bos_s);
    processed_template = replace_all(processed_template, "{{eos_token}}", eos_s);
    processed_template = replace_all(processed_template, "{{user_prompt}}", user_prompt);
    if (!system_message.empty()) {
        processed_template = replace_all(processed_template, "{{system_message}}", system_message);
    } else {
        processed_template = replace_all(processed_template, "{{system_message}}", "");
    }

    std::string snippet_to_log = processed_template.substr(0, std::min((size_t)100, processed_template.length()));
    Logger::debug(std::string("[apply_chat_template] Processed simple GGUF template. Snippet: ") + snippet_to_log);
    return processed_template;
  } else {
    if (is_jinja_template) {
        Logger::warning("[apply_chat_template] GGUF chat template appears to be Jinja2, which is not fully supported by this C++ implementation. Falling back to hardcoded Llama 3 Instruct template. The model's intended GGUF chat template will be ignored.");
    } else { // Empty GGUF template
        Logger::info("[apply_chat_template] GGUF chat template not found or empty. Falling back to hardcoded Llama 3 Instruct template.");
    }

    // Fallback to a hardcoded Llama 3 Instruct style template
    auto find_added_token_str_fallback = [&](const std::string& content,
                                   const std::string& fallback_value) -> std::string {
        if (this->added_tokens_.count(content)) return content;
        if (this->token_to_id_.count(content)) return content;
        if ((!this->added_tokens_.empty() || !this->token_to_id_.empty()) && content.rfind("<",0) == 0 && content.rfind("|",0) != std::string::npos && content.rfind(">",0) == content.length()-1) {
            Logger::warning("[apply_chat_template_fallback] Could not find special token string '" + content +
                           "' in added_tokens_ or vocab. Using default/fallback string: '" + fallback_value + "'");
        }
        return fallback_value;
    };

    // Use member versions of bos_token_, etc. which are set up during constructor
    std::string bos_s_fallback = this->bos_token_id_ != -1 ? this->bos_token_ : "<s>"; 
    // For Llama3 specific tokens, ensure they are correctly fetched or have sensible defaults
    std::string start_header_s_fallback = find_added_token_str_fallback("<|start_header_id|>", "<|start_header_id|>");
    std::string end_header_s_fallback = find_added_token_str_fallback("<|end_header_id|>", "<|end_header_id|>");
    std::string eot_s_fallback = find_added_token_str_fallback("<|eot_id|>", "<|eot_id|>");
    // For role names, they are typically just strings, not special tokens themselves
    std::string system_role_name = "system";
    std::string user_role_name = "user";
    std::string assistant_role_name = "assistant";

    std::stringstream ss;
    ss << bos_s_fallback;
    if (!system_message.empty()) {
        ss << start_header_s_fallback << system_role_name << end_header_s_fallback << "\n\n" << system_message << eot_s_fallback;
    }
    ss << start_header_s_fallback << user_role_name << end_header_s_fallback << "\n\n" << user_prompt << eot_s_fallback;
    ss << start_header_s_fallback << assistant_role_name << end_header_s_fallback << "\n\n";

    Logger::info("[apply_chat_template] Applied hardcoded Llama 3 Instruct-like chat template as fallback. Prompt snippet: " + ss.str().substr(0,100));
    return ss.str();
  }
}

void Tokenizer::load_vocab_from_json(
    const std::string& vocab_path,
    std::unordered_map<std::string, int>& token_to_id_map,
    std::vector<std::string>& id_to_token_vec) {
  token_to_id_map.clear();
  id_to_token_vec.clear();

  try {
    std::ifstream file(vocab_path);
    if (!file.is_open()) {
      throw std::runtime_error("Failed to open vocabulary file: " + vocab_path);
    }

    json vocab_json;
    file >> vocab_json;
    
    // Try to determine format (HuggingFace tokenizer.json vs. plain vocab)
    if (vocab_json.contains("model") && vocab_json["model"].is_object() && 
        vocab_json["model"].contains("vocab") && vocab_json["model"]["vocab"].is_object()) {
      Logger::info("load_vocab_from_json: Detected HuggingFace tokenizer.json format.");
      const auto& vocab = vocab_json["model"]["vocab"];
      size_t max_id = 0;

      // First pass to determine max_id to size id_to_token_vec appropriately
      for (auto it = vocab.begin(); it != vocab.end(); ++it) {
        int id = it.value().get<int>();
        if (id < 0) {
            Logger::warning("load_vocab_from_json: Skipping token with negative ID: " + it.key());
            continue;
        }
        if (static_cast<size_t>(id) > max_id) {
          max_id = static_cast<size_t>(id);
        }
      }
      id_to_token_vec.resize(max_id + 1, "<unk>"); // Initialize with unk_token_ or a placeholder

      // Second pass to populate maps
      for (auto it = vocab.begin(); it != vocab.end(); ++it) {
        std::string token = it.key();
        int id = it.value().get<int>();
        if (id < 0) continue; // Already warned

        token_to_id_map[token] = id;
        if (static_cast<size_t>(id) < id_to_token_vec.size()) {
             id_to_token_vec[id] = token;
        } else {
            // This should ideally not happen if resize was correct
            Logger::warning("load_vocab_from_json: ID out of bounds during vocab population: " + std::to_string(id));
        }
      }

      // Ensure `added_tokens_` (member) is populated here.
      if (vocab_json.contains("added_tokens") &&
          vocab_json["added_tokens"].is_array()) {
        const auto& added_tokens_json = vocab_json["added_tokens"];
        Logger::info("load_vocab_from_json: Processing " + std::to_string(added_tokens_json.size()) + " added_tokens.");
        for (const auto& token_obj : added_tokens_json) {
          if (token_obj.contains("content") && token_obj.contains("id")) {
            std::string token_content = token_obj["content"];
            int token_id = token_obj["id"];

            if (token_id < 0) {
                Logger::warning("load_vocab_from_json: Skipping added_token with negative ID: " + token_content);
                continue;
            }

            // Update maps for added tokens
            token_to_id_map[token_content] = token_id; // Also add to the main map for direct lookup
            this->added_tokens_[token_content] = token_id; // Populate member variable
            this->id_to_added_token_[token_id] = token_content; // Populate member variable

            if (static_cast<size_t>(token_id) >= id_to_token_vec.size()) {
              id_to_token_vec.resize(token_id + 1, "<unk>"); // Ensure vector is large enough
            }
            id_to_token_vec[token_id] = token_content; // Ensure id_to_token_vec also has added tokens

            if (token_content == this->unk_token_) this->unk_token_id_ = token_id;
            else if (token_content == this->bos_token_) this->bos_token_id_ = token_id;
            else if (token_content == this->eos_token_) this->eos_token_id_ = token_id;
            else if (token_content == this->pad_token_) this->pad_token_id_ = token_id;
            
            Logger::debug("load_vocab_from_json: Processed added_token: '" + token_content + "' with ID " +
                         std::to_string(token_id));
          }
        }
      }

    } else if (vocab_json.is_object()) {
      Logger::info("load_vocab_from_json: Detected plain vocabulary format (direct map).");
      size_t max_id = 0;
      for (auto it = vocab_json.begin(); it != vocab_json.end(); ++it) {
        int id = it.value().get<int>();
        if (id < 0) continue;
        if (static_cast<size_t>(id) > max_id) {
          max_id = static_cast<size_t>(id);
        }
      }
      id_to_token_vec.resize(max_id + 1, "<unk>");
      
      for (auto it = vocab_json.begin(); it != vocab_json.end(); ++it) {
        std::string token = it.key();
        int id = it.value().get<int>();
        if (id < 0) {
            Logger::warning("load_vocab_from_json: Skipping token with negative ID: " + token);
            continue;
        }
        token_to_id_map[token] = id;
        if (static_cast<size_t>(id) < id_to_token_vec.size()) {
            id_to_token_vec[id] = token;
        }

        if (token == this->unk_token_) this->unk_token_id_ = id;
        else if (token == this->bos_token_) this->bos_token_id_ = id;
        else if (token == this->eos_token_) this->eos_token_id_ = id;
        else if (token == this->pad_token_) this->pad_token_id_ = id;
      }
    } else {
      throw std::runtime_error("load_vocab_from_json: Vocabulary JSON has an unsupported format.");
    }

    for (size_t i = 0; i < id_to_token_vec.size(); ++i) {
      if (id_to_token_vec[i].empty() || id_to_token_vec[i] == "<unk>") { 
        auto added_it = this->id_to_added_token_.find(static_cast<int>(i));
        if (added_it != this->id_to_added_token_.end()) {
            id_to_token_vec[i] = added_it->second;
        } else if (id_to_token_vec[i].empty()) { 
             if (id_to_token_vec[i].empty()) id_to_token_vec[i] = "<missing_id_" + std::to_string(i) + ">";
        }
      }
    }

    Logger::info("load_vocab_from_json: Loaded vocabulary with " +
                 std::to_string(token_to_id_map.size()) + " unique token strings and " +
                 std::to_string(id_to_token_vec.size()) + " ID entries.");
    Logger::debug("load_vocab_from_json: Special tokens after JSON load: UNK_ID=" + std::to_string(unk_token_id_) +
                 " ('" + unk_token_ + "'), BOS_ID=" + std::to_string(bos_token_id_) +
                 " ('" + bos_token_ + "'), EOS_ID=" + std::to_string(eos_token_id_) +
                 " ('" + eos_token_ + "'), PAD_ID=" + std::to_string(pad_token_id_) +
                 " ('" + pad_token_ + "')");

  } catch (const json::exception& e) {
    throw std::runtime_error("Error parsing vocabulary JSON from " + vocab_path + ": " + e.what());
  } catch (const std::exception& e) {
    throw std::runtime_error("Error loading vocabulary from " + vocab_path + ": " + std::string(e.what()));
  }
}

void Tokenizer::load_sentencepiece_model(const std::string& model_path) {
  Logger::warning("load_sentencepiece_model: Loading from SentencePiece model file ('" + model_path + "') is currently not implemented.");
  sentencepiece_model_loaded_ = false;
}

void Tokenizer::load_bpe_merges_from_json(const std::string& tokenizer_json_path) {
  try {
    std::ifstream file(tokenizer_json_path);
    if (!file.is_open()) {
      throw std::runtime_error("load_bpe_merges_from_json: Failed to open BPE merges file: " + tokenizer_json_path);
    }

    json model_json;
    file >> model_json;
    
    bpe_merges_.clear(); // Ensure merges map is empty before loading

    // Check for HuggingFace tokenizer.json structure first
    // Merges are typically under model.merges
    if (model_json.contains("model") && model_json["model"].is_object()) {
        const auto& model_section = model_json["model"];
        if (model_section.contains("merges") && model_section["merges"].is_array()) {
            Logger::info("load_bpe_merges_from_json: Detected HuggingFace tokenizer.json format with BPE merges from: " + tokenizer_json_path);
            const auto& merges = model_section["merges"];
            int rank = 0; // Use index as rank for merges from HF JSON
            for (const auto& merge_entry_json : merges) {
                if (merge_entry_json.is_string()) {
                     std::string merge_entry = merge_entry_json.get<std::string>();
          size_t space_pos = merge_entry.find(' ');

                     // Expecting format "part1 part2"
                     if (space_pos != std::string::npos && space_pos > 0 && space_pos < merge_entry.length() - 1) {
            std::string first = merge_entry.substr(0, space_pos);
            std::string second = merge_entry.substr(space_pos + 1);
                        // Combine without the space to form the key for the map
                        std::string pair_key = first + second;
                        bpe_merges_[pair_key] = rank++;
                     } else {
                         Logger::warning("load_bpe_merges_from_json: Skipping malformed merge rule: '" + merge_entry + "' from " + tokenizer_json_path);
                     }
                } else {
                     Logger::warning("load_bpe_merges_from_json: Merge entry is not a string, skipping. File: " + tokenizer_json_path);
                }
            }
        } else {
            // Handle case where tokenizer.json doesn't have expected BPE structure
            Logger::warning("load_bpe_merges_from_json: HuggingFace format detected, but no 'model.merges' array found in model section of: " + tokenizer_json_path);
        }
    }
    // Fallback: Check for a simple top-level "merges" array (less common format)
    else if (model_json.contains("merges") && model_json["merges"].is_array()) {
      Logger::info("load_bpe_merges_from_json: Detected simple top-level 'merges' array format in: " + tokenizer_json_path);
      const auto& merges = model_json["merges"];
      int rank = 0;
       for (const auto& merge_entry_json : merges) {
           if (merge_entry_json.is_string()) {
               std::string merge_entry = merge_entry_json.get<std::string>();
        size_t space_pos = merge_entry.find(' ');
               if (space_pos != std::string::npos && space_pos > 0 && space_pos < merge_entry.length() - 1) {
          std::string first = merge_entry.substr(0, space_pos);
          std::string second = merge_entry.substr(space_pos + 1);
                  std::string pair_key = first + second;
                  bpe_merges_[pair_key] = rank++;
               } else {
                   Logger::warning("load_bpe_merges_from_json: Skipping malformed merge rule from top-level array: '" + merge_entry + "' from " + tokenizer_json_path);
               }
           } else {
               Logger::warning("load_bpe_merges_from_json: Merge entry in top-level array is not a string, skipping. File: " + tokenizer_json_path);
        }
      }
    } else {
      // If neither format is found
      throw std::runtime_error(
          "load_bpe_merges_from_json: Unsupported BPE model format: no 'model.merges' or top-level 'merges' array found in '" + tokenizer_json_path + "'");
    }

    if (bpe_merges_.empty()) {
      Logger::warning("load_bpe_merges_from_json: No BPE merges were loaded from the file: " + tokenizer_json_path);
    } else {
      Logger::info("load_bpe_merges_from_json: Loaded " + std::to_string(bpe_merges_.size()) +
                   " BPE merges with ranks from " + tokenizer_json_path);
    }

  } catch (const json::exception& e) {
    throw std::runtime_error("Error parsing BPE merges JSON from " + tokenizer_json_path + ": " + e.what());
  } catch (const std::exception& e) {
    throw std::runtime_error("An unexpected error occurred while loading BPE merges from " + tokenizer_json_path + ": " + std::string(e.what()));
  }
}

std::string Tokenizer::capitalize_first_letter(std::string s) const { // Added Tokenizer:: scope and const
  if (s.empty()) return s;


  size_t first_letter_pos = 0;
  const std::string sp_space = "\xE2\x96\x81";  // SentencePiece space U+2581

  // Check if the string starts with the SentencePiece space
  // Using s instead of result
  if (s.rfind(sp_space, 0) == 0) {
    // If it does, the actual first letter is after the space prefix
    if (s.length() > sp_space.length()) {
      first_letter_pos = sp_space.length();
    } else {
      // String is just the space prefix, nothing to capitalize
      return s;
    }
  }

  // Capitalize the character at the determined position
  // Create result string here to modify
  std::string result = s;
  if (first_letter_pos < result.length()) {
    result[first_letter_pos] =
        std::toupper(static_cast<unsigned char>(result[first_letter_pos]));
  }

  return result;
}

std::vector<std::string> Tokenizer::bpe_tokenize(const std::string& text) const {
    Logger::debug("[Original bpe_tokenize for SentencePiece] Entered. bpe_merges_ size: " + std::to_string(bpe_merges_.size()));
    std::vector<std::string> all_final_tokens;
    const std::string sp_space_prefix = "\xE2\x96\x81"; // SentencePiece space U+2581

    std::vector<std::string> pieces;
    std::string current_piece;
    bool last_char_was_space = true;

    for (char c : text) {
        if (std::isspace(static_cast<unsigned char>(c))) {
            if (!current_piece.empty()) {
                pieces.push_back(current_piece);
                current_piece.clear();
            }
            pieces.push_back(std::string(1, c));
            last_char_was_space = true;
        } else {
            current_piece += c;
            last_char_was_space = false;
        }
    }
    if (!current_piece.empty()) {
        pieces.push_back(current_piece);
    }

    Logger::debug("[Original bpe_tokenize for SentencePiece] Split text into " + std::to_string(pieces.size()) + " pieces (words/spaces).");

    bool next_word_needs_prefix = true;

    for (const std::string& piece : pieces) {
        if (piece.empty()) continue;

        bool piece_is_whitespace = std::all_of(piece.begin(), piece.end(), 
            [](char c) { return std::isspace(static_cast<unsigned char>(c)); });

        if (piece_is_whitespace) {
            next_word_needs_prefix = true;
            Logger::debug("[Original bpe_tokenize for SentencePiece] Piece '" + piece + "' is whitespace. Setting prefix flag.");
            continue;
        }

        std::string word_to_process = piece;
        if (next_word_needs_prefix) {
            word_to_process = sp_space_prefix + word_to_process;
            Logger::debug("[Original bpe_tokenize for SentencePiece] Prefixed word: '" + piece + "' -> '" + word_to_process + "'");
            next_word_needs_prefix = false;
        } else {
             Logger::debug("[Original bpe_tokenize for SentencePiece] Processing word without prefix: '" + word_to_process + "'");
        }
        
        std::vector<std::string> chars;
        for (size_t i = 0; i < word_to_process.size();) {
            size_t bytes = unicode_char_len(word_to_process[i]);
            if (i + bytes <= word_to_process.size()) {
                chars.push_back(word_to_process.substr(i, bytes));
            } else {
                Logger::warning("[Original bpe_tokenize for SentencePiece] Invalid UTF-8 near: '" + word_to_process.substr(i) + "'");
                chars.push_back(word_to_process.substr(i, 1)); 
                bytes = 1;
            }
            i += bytes;
        }

        if (chars.empty()) {
            Logger::warning("[Original bpe_tokenize for SentencePiece] Word '" + word_to_process + "' produced no chars.");
            continue;
        }

        bool changes = true;
        while (changes && chars.size() > 1) {
            changes = false;
            int best_rank = std::numeric_limits<int>::max();
            int best_i = -1;

            for (size_t i = 0; i < chars.size() - 1; ++i) {
                std::string pair = chars[i] + chars[i + 1];
                auto it = bpe_merges_.find(pair);
                if (it != bpe_merges_.end() && it->second < best_rank) {
                    best_rank = it->second;
                    best_i = i;
                }
            }

            if (best_i >= 0) {
                std::string merged = chars[best_i] + chars[best_i + 1];
                chars[best_i] = merged;
                chars.erase(chars.begin() + best_i + 1);
                changes = true;
                Logger::debug("[Original bpe_tokenize for SentencePiece] Applied merge: '" + merged + "' with rank " + 
                            std::to_string(best_rank));
            }
        }
        all_final_tokens.insert(all_final_tokens.end(), chars.begin(), chars.end());
    }

    Logger::debug("[Original bpe_tokenize for SentencePiece] Final token count: " + std::to_string(all_final_tokens.size()));
    return all_final_tokens;
}

const std::string& Tokenizer::get_gguf_chat_template() const {
  return gguf_chat_template_;
}

// Helper to sort token map by key length (descending) for longest match
static std::vector<std::pair<std::string, int>> sort_tokens_by_length_desc(const std::unordered_map<std::string, int>& tokens_map) {
    std::vector<std::pair<std::string, int>> sorted_tokens;
    for (const auto& pair : tokens_map) {
        sorted_tokens.push_back(pair);
    }
    std::sort(sorted_tokens.begin(), sorted_tokens.end(),
              [](const auto& a, const auto& b) {
                  return a.first.length() > b.first.length();
              });
    return sorted_tokens;
}

// The bpe_tokenize_to_ids is now specifically for Tiktoken-like BPE (Llama 3)
// It assumes that if this function is called, the tokenizer_family_ is LLAMA3_TIKTOKEN.
std::vector<int> Tokenizer::bpe_tokenize_to_ids(const std::string& text,
                                                bool add_bos_token_param, 
                                                bool add_eos_token_param, 
                                                bool ignore_merges_param) const { 
  Logger::debug(std::string("[bpe_tokenize_to_ids] Starting Tiktoken BPE tokenization for text length: ") + std::to_string(text.length()) +
                ", add_bos=" + std::to_string(add_bos_token_param) +
                ", add_eos=" + std::to_string(add_eos_token_param) +
                ", ignore_merges=" + std::to_string(ignore_merges_param) );

  std::vector<int> output_ids;

  if (add_bos_token_param) {
    if (bos_token_id_ == -1) {
      Logger::warning("[bpe_tokenize_to_ids] BOS token requested but bos_token_id_ is -1.");
    } else {
      output_ids.push_back(bos_token_id_);
      Logger::debug(std::string("[bpe_tokenize_to_ids] Added BOS token: ") + std::to_string(bos_token_id_));
    }
  }

  const auto sorted_special_tokens = sort_tokens_by_length_desc(this->added_tokens_); 

  // TikToken regex pattern string
  const std::string tiktoken_pattern_str =
      R"(<\|[^|]+\||[[:alnum:]]+|\.(?![<|])|[^\s<|]+|\s+)"; // Updated regex
  
  // Compile with boost::xpressive::sregex and icase flag
  const boost::xpressive::sregex tiktoken_pattern_ = boost::xpressive::sregex::compile(
      tiktoken_pattern_str,
      boost::xpressive::regex_constants::icase
  );

  size_t current_idx = 0;
  while (current_idx < text.length()) {
    bool special_match_found = false;
    if (!sorted_special_tokens.empty()) {
        for (const auto& special_pair : sorted_special_tokens) {
            const std::string& special_text = special_pair.first;
            int special_id = special_pair.second;
            if (text.compare(current_idx, special_text.length(), special_text) == 0) {
                output_ids.push_back(special_id);
                Logger::debug("[bpe_tokenize_to_ids] Matched special token: '" + special_text + "' -> ID: " + std::to_string(special_id));
                current_idx += special_text.length();
                special_match_found = true;
                break;
            }
        }
    }

    if (special_match_found) {
      continue;
    }

    if (current_idx >= text.length()) break;

    std::string remaining_text_view_str = text.substr(current_idx);
    boost::xpressive::smatch word_match;

    if (!boost::xpressive::regex_search(remaining_text_view_str, word_match, tiktoken_pattern_, boost::xpressive::regex_constants::match_continuous)) {
      Logger::debug(std::string("[bpe_tokenize_to_ids] No more regex-matchable words at pos ") + std::to_string(current_idx) + ". Remainder: '" + remaining_text_view_str + "'");
      if (!remaining_text_view_str.empty()) {
          Logger::warning(std::string("[bpe_tokenize_to_ids] Regex could not process remainder. Processing byte-by-byte: '") + remaining_text_view_str + "'");
          for (char c : remaining_text_view_str) {
              std::string byte_str(1, c);
              auto it = token_to_id_.find(byte_str);
              if (it != token_to_id_.end()) {
                  output_ids.push_back(it->second);
              } else {
                  if (byte_char_to_id_.count(c)) {
                       output_ids.push_back(byte_char_to_id_.at(c));
                  } else if (unk_token_id_ != -1) {
                       output_ids.push_back(unk_token_id_);
                       Logger::warning(std::string("[bpe_tokenize_to_ids] Unrecognized byte '") + byte_str + std::string("' replaced with UNK."));
                  } else {
                       Logger::error(std::string("[bpe_tokenize_to_ids] Unrecognized byte '") + byte_str + std::string("' and no UNK token defined. Skipping."));
                  }
              }
          }
      }
      current_idx = text.length();
      break;
    }
    
    std::string original_word = word_match.str(0);
    
    if (original_word.empty()){ 
        Logger::warning("[bpe_tokenize_to_ids] Regex search succeeded but matched an empty string. Advancing one char from pos " + std::to_string(current_idx));
        size_t advance_len = unicode_char_len(text[current_idx]);
        if (advance_len == 0) advance_len = 1;
        
        std::string problematic_char_str = text.substr(current_idx, advance_len);
        auto it_char = token_to_id_.find(problematic_char_str);
        if (it_char != token_to_id_.end()) {
            output_ids.push_back(it_char->second);
        } else if (advance_len == 1 && byte_char_to_id_.count(problematic_char_str[0])) {
            output_ids.push_back(byte_char_to_id_.at(problematic_char_str[0]));
        } else if (unk_token_id_ != -1) {
            output_ids.push_back(unk_token_id_);
            Logger::debug("[bpe_tokenize_to_ids] Added UNK for unmatchable leading char after empty regex match: '" + problematic_char_str + "'");
        }
        current_idx += advance_len;
        continue;
    }
    
    // Check if the entire original_word (matched by regex) is a known token (especially for <|...|> cases)
    auto direct_match_it = token_to_id_.find(original_word);
    if (direct_match_it != token_to_id_.end()) {
        output_ids.push_back(direct_match_it->second);
        Logger::debug("[bpe_tokenize_to_ids] Regex-matched word '" + original_word + "' is a direct token ID: " + std::to_string(direct_match_it->second));
        current_idx += original_word.length();
        continue;
    }

    Logger::debug(std::string("[bpe_tokenize_to_ids] Processing regex-derived word for BPE: '") + original_word + "'");
    
    // Convert leading space of original_word to BPE_SPACE_CHAR (Ġ) for Tiktoken-style BPE
    // This is crucial if the vocabulary expects space-prefixed tokens like "Ġword".
    std::string word_to_process = original_word;
    if (!word_to_process.empty() && word_to_process[0] == ' ') {
        if (word_to_process.length() > 1) {
            word_to_process = BPE_SPACE_CHAR + word_to_process.substr(1);
        } else { // Word is just a single space
            word_to_process = BPE_SPACE_CHAR;
        }
        Logger::debug(std::string("[bpe_tokenize_to_ids] Converted leading space. Word for BPE: '") + word_to_process + "'");
    }

    if (ignore_merges_param) { // If ignore_merges is true, try direct lookup first
        auto it_direct = token_to_id_.find(word_to_process);
        if (it_direct != token_to_id_.end()) {
            output_ids.push_back(it_direct->second);
            Logger::debug(std::string("[bpe_tokenize_to_ids] Found word directly (ignore_merges): '") + word_to_process + "' -> ID: " + std::to_string(it_direct->second));
            current_idx += original_word.length();
            continue;
        }
        Logger::debug(std::string("[bpe_tokenize_to_ids] ignore_merges=true, but word \'") + word_to_process + "\' not in vocab directly. Proceeding with BPE char split (unusual for tiktoken special words).");

    }
    
    std::vector<llm_symbol> symbols;
    symbols.reserve(word_to_process.length());
    size_t offset = 0;
    while (offset < word_to_process.length()) {
      size_t char_len = unicode_char_len(word_to_process[offset]);
      if (offset + char_len > word_to_process.length()) {
        Logger::error("[bpe_tokenize_to_ids] Invalid UTF-8 sequence in word: '" + word_to_process + "' at offset " + std::to_string(offset));
        symbols.clear(); 
        break;
      }
      // For Tiktoken, the llm_symbol needs `text_offset` relative to `word_to_process.data()` and `n` (length).
      // The `prev` and `next` are for the linked list structure during BPE.
      symbols.emplace_back(llm_symbol{-1, -1, word_to_process.data() + offset, char_len}); 
      offset += char_len;
    }

    if (symbols.empty() && !word_to_process.empty()) {
        Logger::warning("[bpe_tokenize_to_ids] Word '" + word_to_process + "' resulted in no symbols. Skipping this word's BPE.");
        if (unk_token_id_ != -1 && !original_word.empty()){
            output_ids.push_back(unk_token_id_);
        }
        current_idx += original_word.length();
        continue;
    }
    if (symbols.empty() && word_to_process.empty()){
        current_idx += original_word.length(); 
        continue;
    }

    for (size_t i = 0; i < symbols.size(); ++i) {
      symbols[i].prev = (i > 0) ? (i - 1) : -1;
      symbols[i].next = (i < symbols.size() - 1) ? (i + 1) : -1;
    }
    
    // Use std::priority_queue for merges
    std::priority_queue<std::pair<int, int>,
                        std::vector<std::pair<int, int>>,
                        std::greater<std::pair<int, int>>> merge_queue;

    for (size_t i = 0; i + 1 < symbols.size(); ++i) {
      // add_bigram_to_queue expects data pointer, symbols vector, index of first symbol in pair, and queue
      add_bigram_to_queue_refactored(word_to_process.data(), symbols, i, merge_queue);
    }

    while (!merge_queue.empty()) {
      auto top = merge_queue.top();
      merge_queue.pop();

      int rank = top.first; 
      int p1_idx = top.second; 

      if (symbols[p1_idx].n == 0) continue; 
      int p2_idx = symbols[p1_idx].next;
      if (p2_idx == -1 || symbols[p2_idx].n == 0) continue;


      symbols[p1_idx].n += symbols[p2_idx].n; 
      symbols[p2_idx].n = 0; 
      symbols[p1_idx].next = symbols[p2_idx].next;
      if (symbols[p1_idx].next != -1) {
          symbols[symbols[p1_idx].next].prev = p1_idx;
      }
      

      // Add new bigrams
      if (symbols[p1_idx].prev != -1) {
        add_bigram_to_queue_refactored(word_to_process.data(), symbols, symbols[p1_idx].prev, merge_queue);
      }
      if (symbols[p1_idx].next != -1) {
        add_bigram_to_queue_refactored(word_to_process.data(), symbols, p1_idx, merge_queue);
      }
    }

    std::vector<int> final_word_ids;
    if (!symbols.empty()) {
        for (int i = 0; i != -1; i = symbols[i].next) {
            const llm_symbol & symbol = symbols[i];
            if (symbol.n == 0) continue; 

            std::string s(symbol.text, symbol.n);
            std::string lookup_s = s;
            
            const auto token_it = token_to_id_.find(lookup_s);

            if (token_it != token_to_id_.end()) {
                final_word_ids.push_back(token_it->second);
            } else {
                Logger::warning(std::string("[bpe_tokenize_to_ids] Symbol not found in vocab: '") + lookup_s + "'. Attempting byte-level tokenization.");
                for (char c_char : lookup_s) {
                    auto byte_map_it = byte_char_to_id_.find(c_char);
                    if (byte_map_it != byte_char_to_id_.end()){
                        final_word_ids.push_back(byte_map_it->second);
                    } else { 
                        if (unk_token_id_ != -1) {
                            final_word_ids.push_back(unk_token_id_);
                        } else {
                            Logger::error(std::string("[bpe_tokenize_to_ids] Unhandled char '") + std::string(1, c_char) + "' and no UNK token ID.");
                        }
                    }
                }
            }
        }
    } else if (!word_to_process.empty()) {
        Logger::warning(std::string("[bpe_tokenize_to_ids] Word '") + word_to_process + std::string("' yielded no final symbols. UNK if available."));
        if (unk_token_id_ != -1){ final_word_ids.push_back(unk_token_id_); }
    }

    if (final_word_ids.empty() && !original_word.empty()) {
        Logger::warning(std::string("[bpe_tokenize_to_ids] Word '") + original_word + "' resulted in no tokens. Adding UNK.");
        if (unk_token_id_ != -1) { output_ids.push_back(unk_token_id_); }
    } else {
        output_ids.insert(output_ids.end(), final_word_ids.begin(), final_word_ids.end());
    }
    current_idx += original_word.length(); 
  }

  if (add_eos_token_param) {
    if (eos_token_id_ == -1) {
        Logger::warning("[bpe_tokenize_to_ids] EOS token requested but eos_token_id_ is -1.");
    } else {
        output_ids.push_back(eos_token_id_); // Corrected to output_ids
        Logger::debug(std::string("[bpe_tokenize_to_ids] Added EOS token: ") + std::to_string(eos_token_id_));
    }
  }
  Logger::debug("[bpe_tokenize_to_ids] Finished Tiktoken BPE tokenization. Total IDs: " + std::to_string(output_ids.size()));
  return output_ids;
}

// Helper function to add a potential bigram to the priority queue (Refactored for new llm_symbol structure)
// Assumes llm_symbol stores text_offset and n (length) relative to a base data pointer.
void Tokenizer::add_bigram_to_queue_refactored(const char* text_data_base,
                                             const std::vector<llm_symbol>& symbols,
                                             llm_symbol::index first_symbol_idx,
                                             std::priority_queue<std::pair<int, int>,
                                                                 std::vector<std::pair<int, int>>,
                                                                 std::greater<std::pair<int, int>>>& work_queue) const {
    if (first_symbol_idx < 0 || static_cast<size_t>(first_symbol_idx) >= symbols.size()) {
        Logger::error(std::string("[ADD_BIGRAM_REFACTORED] Invalid first_symbol_idx: ") + std::to_string(first_symbol_idx));
        return;
    }

    const llm_symbol& s1 = symbols[first_symbol_idx];
    llm_symbol::index s2_idx = s1.next;

    if (s2_idx < 0 || static_cast<size_t>(s2_idx) >= symbols.size() || s2_idx <= first_symbol_idx) {
        return;
    }
    const llm_symbol& s2 = symbols[s2_idx];

    if (s1.n == 0 || s2.n == 0) {
        return;
    }

    std::string token_left_str(s1.text, s1.n);
    std::string token_right_str(s2.text, s2.n);
    
    std::vector<std::string> merge_attempts;
    
    // First priority: If we see Ġ, try it with a following space
    if (token_left_str == BPE_SPACE_CHAR) {
        merge_attempts.push_back(BPE_SPACE_CHAR + " " + token_right_str);
        Logger::debug("[ADD_BIGRAM] Attempting Ġ+space merge: '" + (BPE_SPACE_CHAR + " " + token_right_str) + "'");
    }
    
    // Second priority: Standard merge without space
    merge_attempts.push_back(token_left_str + token_right_str);
    Logger::debug("[ADD_BIGRAM] Attempting standard merge: '" + (token_left_str + token_right_str) + "'");
    
    // Third priority: If left token starts with Ġ but isn't just Ġ, try with space
    if (token_left_str.rfind(BPE_SPACE_CHAR, 0) == 0 && token_left_str != BPE_SPACE_CHAR) {
        merge_attempts.push_back(token_left_str + " " + token_right_str);
        Logger::debug("[ADD_BIGRAM] Attempting Ġword+space merge: '" + (token_left_str + " " + token_right_str) + "'");
    }
    
    // Fourth priority: Special case for character splits with space
    if (token_left_str.length() == 2 && token_right_str.length() == 1) {
        std::string attempt = token_left_str.substr(0, 1) + " " + token_right_str;
        merge_attempts.push_back(attempt);
        Logger::debug("[ADD_BIGRAM] Attempting char split merge: '" + attempt + "'");
    }

    int best_rank = std::numeric_limits<int>::max();
    bool found_merge = false;
    std::string matched_merge;

    for (const auto& merge_attempt : merge_attempts) {
        auto it = bpe_merges_.find(merge_attempt);
        if (it != bpe_merges_.end() && it->second < best_rank) {
            best_rank = it->second;
            found_merge = true;
            matched_merge = merge_attempt;
        }
    }

    if (found_merge) {
        work_queue.push({best_rank, first_symbol_idx});
        Logger::debug("[ADD_BIGRAM] Found merge: '" + matched_merge + "' with rank " + std::to_string(best_rank));
    } else {
        Logger::debug("[ADD_BIGRAM] No valid merges found for attempts with left='" + token_left_str + 
                     "' right='" + token_right_str + "'");
    }
}


