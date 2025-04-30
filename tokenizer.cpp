#include "tokenizer.h"
#include <stdexcept> // For runtime_error
#include <iostream>  // For cerr
#include <fstream>   // For file reading
#include <sstream>   // For stringstream
#include <algorithm> // For replace, min, max
#include <regex>     // For regex pattern matching
#include <unordered_set>
#include "logger.h"  // Include logger
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// Constructor: Load tokenizer from file path and vocab
Tokenizer::Tokenizer(const std::string& model_path, const std::string& vocab_path) 
    : unk_token_("<unk>"), 
      bos_token_("<s>"),
      eos_token_("</s>"),
      pad_token_("<pad>")
{
    try {
        Logger::info("Loading tokenizer and vocab from: " + vocab_path);
        
        // First load the vocabulary from JSON
        load_vocab_from_json(vocab_path, token_to_id_, id_to_token_);
        
        // Look up special token IDs
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
            bos_token_id_ = -1;  // Not used
        }
        
        if (token_to_id_.find(eos_token_) != token_to_id_.end()) {
            eos_token_id_ = token_to_id_[eos_token_];
        } else {
            Logger::info("No EOS token found in vocabulary");
            eos_token_id_ = -1;  // Not used
        }
        
        // Load BPE merges from model file if provided
        if (model_path.size() > 0) {
            if (model_path.size() > 6 && model_path.substr(model_path.size() - 6) == ".model") {
                Logger::info("Loading SentencePiece model: " + model_path);
                load_sentencepiece_model(model_path);
            } else if (model_path.size() > 5 && model_path.substr(model_path.size() - 5) == ".json") {
                Logger::info("Loading BPE merges from JSON: " + model_path);
                load_bpe_merges_from_json(model_path);
            } else {
                Logger::info("Unsupported model format: " + model_path + " - falling back to space tokenization");
            }
        } else {
            Logger::info("No model path provided - falling back to space tokenization");
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to load tokenizer or vocab from " << vocab_path << ": " << e.what() << std::endl;
        Logger::error(std::string("Failed to load tokenizer or vocab from \"") + vocab_path + "\": " + e.what());
        throw;
    }
    
    if (id_to_token_.empty()) {
        throw std::runtime_error("Failed to initialize tokenizer vocabulary from: " + vocab_path);
    }
    
    Logger::info("Successfully initialized tokenizer with " + std::to_string(id_to_token_.size()) + " tokens");
}

// Tokenize text using BPE if available, falling back to space tokenization
std::vector<std::string> Tokenizer::tokenize(const std::string& text) const {
    // If we have BPE merges, use BPE tokenization
    if (!bpe_merges_.empty()) {
        return bpe_tokenize(text);
    }
    
    // If we have a SentencePiece model, use that
    if (sentencepiece_model_loaded_) {
        return sentencepiece_tokenize(text);
    }
    
    // Fallback to simple space tokenization
    return space_tokenize(text);
}

// Simple space tokenization (fallback)
std::vector<std::string> Tokenizer::space_tokenize(const std::string& text) const {
    std::vector<std::string> tokens;
    std::istringstream iss(text);
    std::string token;
    
    while (iss >> token) {
        tokens.push_back(token);
    }
    
    return tokens;
}

// BPE tokenization 
std::vector<std::string> Tokenizer::bpe_tokenize(const std::string& text) const {
    std::vector<std::string> all_tokens;
    
    // Heuristic to detect if space prefix (Ġ or   ) is used in the loaded vocab
    bool using_space_prefix = false;
    const std::string gpt2_space_prefix = "\xC4\xA0"; // Ġ (U+0120)
    const std::string tinyllama_space_prefix = "\xE2\x96\x81"; //   (U+2581)
    
    // --- Use id_to_token_ like .bak --- 
    for (const auto& token : id_to_token_) { // Iterate through the loaded vocab vector
         if (!token.empty()) {
            if (token.size() >= gpt2_space_prefix.size() && token.substr(0, gpt2_space_prefix.size()) == gpt2_space_prefix) {
                 using_space_prefix = true;
                 break;
            } 
             if (token.size() >= tinyllama_space_prefix.size() && token.substr(0, tinyllama_space_prefix.size()) == tinyllama_space_prefix) {
                 using_space_prefix = true;
                 break;
             }
         }
    }
    if (using_space_prefix) {
         // Logger::info("bpe_tokenize detected space prefix"); // Simplified log
    } else {
        // Logger::info("bpe_tokenize did not detect standard space prefix.");
    }

    // Preprocess text: Add leading space if using prefix and text doesn't start with space
    std::string processed_text = text;
    if (using_space_prefix && !processed_text.empty() && processed_text[0] != ' ') {
        processed_text = " " + processed_text;
        // Logger::info("Added leading space for prefix handling.");
    }

    // --- START: Reverted Pre-tokenization logic from .bak --- 
    std::vector<std::string> words;
    std::string current_word;
    bool in_whitespace = true;
    
    for (size_t i = 0; i < processed_text.size(); ++i) {
        char c = processed_text[i];
        if (std::isspace(static_cast<unsigned char>(c))) {
            if (!in_whitespace) {
                // End of a word
                if (!current_word.empty()) {
                    words.push_back(current_word);
                    current_word.clear();
                }
                in_whitespace = true;
            }
            // Add the whitespace to the current word if we're keeping track of it
            if (using_space_prefix && in_whitespace) {
                // We mark the beginning of a new word after whitespace
                if (!current_word.empty()) {
                    words.push_back(current_word);
                }
                current_word = " "; // Start new word with space marker
                in_whitespace = false;
            }
        } else {
            // If switching from whitespace to non-whitespace, and using prefix, start word with space marker
            if (in_whitespace && using_space_prefix) {
                 if (!current_word.empty()) { // If there was already a space marker
                     words.push_back(current_word); // Add previous space marker word
                 } 
                 current_word = " "; // Start new word with space marker
            }
            current_word.push_back(c);
            in_whitespace = false;
        }
    }
    // Add the last word if there is one
    if (!current_word.empty()) {
        words.push_back(current_word);
    }
     // --- END: Reverted Pre-tokenization logic from .bak --- 

    // Process each word with BPE
    for (auto& word : words) { // Use 'word' like in .bak
        // Special handling for TinyLlama-style prefix
        if (using_space_prefix && word.size() > 0 && word[0] == ' ') {
            // Replace the space at the beginning with the special token
            if (word.size() == 1) {
                // This is just a space - use the UTF-8 sequence for   (U+2581)
                word = "\xE2\x96\x81"; 
            } else {
                word = "\xE2\x96\x81" + word.substr(1); // Or use detected_space_prefix if needed
            }
        } // Removed the extra prefix handling for regex output
        
        // Apply space prefix if needed
        // Note: The regex might capture leading spaces, adjust prefix logic if needed.
        // Let's refine prefix handling here based on regex output.
        // Logger::info("Processing segment: '" + current_segment + "'");

        // Split word into initial pieces (UTF-8 characters)
        std::vector<std::string> chars; // Use 'chars' like .bak
        for (size_t i = 0; i < word.size(); ) {
            int bytes = 1;
            unsigned char c = static_cast<unsigned char>(word[i]);
            if ((c & 0xF8) == 0xF0) bytes = 4;
            else if ((c & 0xF0) == 0xE0) bytes = 3;
            else if ((c & 0xE0) == 0xC0) bytes = 2;
            
            if (i + bytes > word.size()) { // Handle potential trailing broken char
                 bytes = word.size() - i;
            }
            chars.push_back(word.substr(i, bytes));
            i += bytes;
        }
        // Logger::info("Initial pieces: " + std::to_string(pieces.size()));

        if (chars.empty()) continue;

        // Apply BPE merges iteratively
        bool changes = true;
        while (changes && chars.size() > 1) { // Match .bak loop condition
            changes = false; // Match .bak logic
            int best_rank = std::numeric_limits<int>::max();
            int best_idx = -1;

            for (size_t i = 0; i < chars.size() - 1; ++i) {
                std::string merge_key = chars[i] + chars[i+1];
                auto it = bpe_merges_.find(merge_key);
                if (it != bpe_merges_.end()) {
                    int current_rank = it->second;
                    if (current_rank < best_rank) {
                        best_rank = current_rank;
                        best_idx = static_cast<int>(i);
                    }
                }
            }

            if (best_idx != -1) {
                 // Perform the merge
                 // Logger::info("Merging '" + chars[best_idx] + "' and '" + chars[best_idx + 1] + "' with rank " + std::to_string(best_rank));
                 std::string merged = chars[best_idx] + chars[best_idx + 1]; // Match .bak
                 chars[best_idx] = merged;
                 chars.erase(chars.begin() + best_idx + 1);
                 changes = true; // Match .bak
            } else {
                // break; // .bak doesn't explicitly break here, relies on changes flag
            }
        }
        // Add the final pieces for this word to the result
        all_tokens.insert(all_tokens.end(), chars.begin(), chars.end());
    }
    // Logger::info("Final token count: " + std::to_string(final_tokens.size()));
    return all_tokens;
}

// SentencePiece tokenization (placeholder - NOT USED FOR GGUF LLAMA)
std::vector<std::string> Tokenizer::sentencepiece_tokenize(const std::string& text) const {
    // Fallback to space tokenization until SentencePiece implementation is complete
    Logger::info("SentencePiece tokenization not fully implemented - falling back to space tokenization");
    return space_tokenize(text);
}

// Detokenize tokens to text
std::string Tokenizer::detokenize(const std::vector<std::string>& tokens) const {
    std::string result;
    
    // Check if the tokenizer uses special space prefix characters
    bool using_space_prefix = false;
    const std::string gpt2_space_prefix = "\xC4\xA0"; // Ġ (U+0120)
    const std::string tinyllama_space_prefix = "\xE2\x96\x81"; // ▁ (U+2581)
    
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
        
        // Different handling based on tokenizer type
        if (using_space_prefix) {
            // Handle different space prefix styles
            if (!token.empty()) {
                // Check for GPT-2/OpenAI style prefix (Ġ)
                if (token.size() >= 2 && token.substr(0, 2) == gpt2_space_prefix) {
                    if (token.size() > 2) {
                        result += ' ' + token.substr(2);
                    } else {
                        result += ' ';
                    }
                } 
                // Check for TinyLlama style prefix (▁)
                else if (token.size() >= 3 && token.substr(0, 3) == tinyllama_space_prefix) {
                    if (token.size() > 3) {
                        result += ' ' + token.substr(3);
                    } else {
                        result += ' ';
                    }
                }
                else {
                    // No prefix - this is a continuation subtoken
                    result += token;
                }
            }
        } 
        else {
            // Standard BPE style
            // Handle end-of-word marker for BPE
            if (token.size() >= 4 && token.substr(token.size() - 4) == "</w>") {
                result += token.substr(0, token.size() - 4);
                result += " ";
                continue;
            }
            
            // Add space between tokens that aren't marked with special characters
            if (i > 0) {
                result += " ";
            }
            
            result += token;
        }
    }
    
    // Cleanup: remove leading space if any
    if (!result.empty() && result[0] == ' ') {
        result = result.substr(1);
    }
    
    // Cleanup: merge multiple spaces
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

// Convert tokens to IDs
std::vector<int> Tokenizer::tokens_to_ids(const std::vector<std::string>& tokens) const {
    std::vector<int> ids;
    ids.reserve(tokens.size());
    
    for (const auto& token : tokens) {
        auto it = token_to_id_.find(token);
        if (it != token_to_id_.end()) {
            ids.push_back(it->second);
        } else {
            ids.push_back(unk_token_id_);  // Use defined UNK token ID
        }
    }
    
    return ids;
}

// Convert IDs to tokens
std::vector<std::string> Tokenizer::ids_to_tokens(const std::vector<int>& ids) const {
    std::vector<std::string> tokens;
    tokens.reserve(ids.size());
    
    for (int id : ids) {
        if (id >= 0 && static_cast<size_t>(id) < id_to_token_.size() && !id_to_token_[id].empty()) {
            tokens.push_back(id_to_token_[id]);
        } else {
            tokens.push_back(unk_token_);
        }
    }
    
    return tokens;
}

// Encode text to token IDs (using BPE for Llama GGUF)
std::vector<int> Tokenizer::encode(const std::string& text, 
                                 bool add_special_tokens, 
                                 bool use_regex_pretokenize /* = false */) const // SET DEFAULT TO FALSE (in comment for definition)
{
    // Logger::info("Tokenizer::encode called. Text: '" + text + "', add_special: " + 
    //              (add_special_tokens ? "true" : "false") + ", use_regex: " + (use_regex_pretokenize ? "true" : "false"));
    // For GGUF Llama, we assume BPE is the method.
    std::vector<std::string> tokens = bpe_tokenize(text); // Call the reverted BPE tokenizer (no flag)
    // Logger::info("Tokenized into " + std::to_string(tokens.size()) + " string pieces.");
    
    // Convert string tokens to IDs
    std::vector<int> ids = tokens_to_ids(tokens);
    // Logger::info("Converted to " + std::to_string(ids.size()) + " IDs (before special tokens).");

    // Prepend BOS if requested and available
    if (add_special_tokens && bos_token_id_ >= 0) {
        ids.insert(ids.begin(), bos_token_id_);
        // Logger::info("Prepended BOS token ID: " + std::to_string(bos_token_id_));
    }
    // NOTE: EOS is typically added *after* generation, not during prompt encoding.
    return ids;
}

// Decode token IDs to text
std::string Tokenizer::decode(const std::vector<int>& ids, bool skip_special_tokens) const {
    // Logger::warning("GGUFTokenizer::decode NYI"); // Old warning
    // This implementation should mostly work if id_to_token_ is loaded correctly.
    std::stringstream ss;
    bool first_token = true; // To handle potential leading space prefixes
    for (int id : ids) {
        if (id >= 0 && static_cast<size_t>(id) < id_to_token_.size()) {
             std::string token = id_to_token_[id];
             if (skip_special_tokens) {
                  if (id == bos_token_id_ || id == eos_token_id_ || 
                     id == pad_token_id_ || id == unk_token_id_) {
                      continue; // Skip special tokens
                  }
             }
             // --- Basic Space Prefix Handling during Decode ---
             // Check for prefixes like ' ' (U+2581) or 'Ġ' (U+0120)
             const std::string gpt2_space_prefix = "\xC4\xA0"; 
             const std::string tinyllama_space_prefix = "\xE2\x96\x81";
             if (!token.empty()) {
                 if ((token.size() >= gpt2_space_prefix.size() && token.substr(0, gpt2_space_prefix.size()) == gpt2_space_prefix)) {
                    if (!first_token) ss << " "; // Add space only if not the first token
                    ss << token.substr(gpt2_space_prefix.size());
                 } else if ((token.size() >= tinyllama_space_prefix.size() && token.substr(0, tinyllama_space_prefix.size()) == tinyllama_space_prefix)) {
                     if (!first_token) ss << " "; // Add space only if not the first token
                     ss << token.substr(tinyllama_space_prefix.size());
                 } else {
                     ss << token; // Append token directly if no prefix
                 }
             } // else: skip empty tokens?
             first_token = false; // Not the first token anymore
        } else {
            ss << "[INVALID_ID:" << id << "]";
            first_token = false;
        }
    }
    return ss.str(); 
}

// Apply Chat Template (Basic Implementation)
std::string Tokenizer::apply_chat_template(const std::string& user_prompt) const {
     // TODO: Implement proper Jinja-like template processing based on chat_template_ member
     // For now, implement a very basic Llama-style template assuming user role only
      Logger::info("Applying BASIC chat template (Llama user/assistant style). GGUF template ignored for now.");

     // Get EOS token string, default to "" if not found/valid
     std::string eos_str = "";
     if (eos_token_id_ >= 0 && static_cast<size_t>(eos_token_id_) < id_to_token_.size()) {
          eos_str = id_to_token_[eos_token_id_];
     }

     // Basic template: <|user|>\nPROMPT + EOS<|assistant|>\n
     std::string formatted = "<|user|>\n" + user_prompt + eos_str + "\n<|assistant|>\n";
     // Note: BOS is typically added by the `encode` method itself based on the flag.
     
     return formatted;
}

// Load vocabulary from JSON file
void Tokenizer::load_vocab_from_json(const std::string& vocab_path, 
                                    std::unordered_map<std::string, int>& token_to_id,
                                    std::vector<std::string>& id_to_token) {
    // Clear existing mappings
    token_to_id.clear();
    id_to_token.clear();
    
    try {
        // Open and parse the JSON file
        std::ifstream file(vocab_path);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open vocabulary file: " + vocab_path);
        }
        
        json vocab_json;
        file >> vocab_json;
        
        // Determine the format - HuggingFace tokenizer.json or plain vocab
        if (vocab_json.contains("model") && vocab_json["model"].contains("vocab") && vocab_json["model"]["vocab"].is_object()) {
            // HuggingFace tokenizer.json format
            Logger::info("Detected HuggingFace tokenizer.json format");
            const auto& vocab = vocab_json["model"]["vocab"];
            
            // Process the vocabulary
            size_t vocab_size = vocab.size();
            
            // Check for special tokens in the added_tokens section
            if (vocab_json.contains("added_tokens") && vocab_json["added_tokens"].is_array()) {
                const auto& added_tokens = vocab_json["added_tokens"];
                for (const auto& token_obj : added_tokens) {
                    if (token_obj.contains("content") && token_obj.contains("id")) {
                        std::string token = token_obj["content"];
                        int id = token_obj["id"];
                        
                        // Map special tokens to their IDs
                        if (token == "<unk>") unk_token_ = token;
                        else if (token == "<s>") bos_token_ = token;
                        else if (token == "</s>") eos_token_ = token;
                        else if (token == "<pad>") pad_token_ = token;
                        
                        Logger::info("Found special token: " + token + " with ID " + std::to_string(id));
                    }
                }
            }
            
            // Ensure id_to_token has the right size
            if (id_to_token.size() < vocab_size) {
                id_to_token.resize(vocab_size);
            }
            
            // Map tokens to IDs
            for (auto it = vocab.begin(); it != vocab.end(); ++it) {
                std::string token = it.key();
                int id = it.value().get<int>();
                
                token_to_id[token] = id;
                
                // Ensure the id_to_token vector has enough capacity
                if (static_cast<size_t>(id) >= id_to_token.size()) {
                    id_to_token.resize(id + 1);
                }
                
                id_to_token[id] = token;
            }
        } 
        // Plain vocab.json with direct token to ID mapping
        else if (vocab_json.is_object()) {
            Logger::info("Detected plain vocabulary format");
            
            // Determine the size of the vocabulary for pre-allocation
            size_t vocab_size = vocab_json.size();
            id_to_token.resize(vocab_size);
            
            // Process each token and its ID
            for (auto it = vocab_json.begin(); it != vocab_json.end(); ++it) {
                std::string token = it.key();
                int id = it.value().get<int>();
                
                // Map special tokens
                if (token == "<unk>") unk_token_ = token;
                else if (token == "<s>") bos_token_ = token;
                else if (token == "</s>") eos_token_ = token;
                else if (token == "<pad>") pad_token_ = token;
                
                // Store the mappings
                token_to_id[token] = id;
                
                // Ensure the id_to_token vector has enough capacity
                if (static_cast<size_t>(id) >= id_to_token.size()) {
                    id_to_token.resize(id + 1);
                }
                
                id_to_token[id] = token;
            }
        } 
        else {
            throw std::runtime_error("Vocabulary JSON has an unsupported format");
        }
        
        // Verify special tokens were found
        Logger::info("Special tokens: UNK=" + unk_token_ + 
                    ", BOS=" + bos_token_ + 
                    ", EOS=" + eos_token_ + 
                    ", PAD=" + pad_token_);
        
        // Check for any missing indices in id_to_token
        for (size_t i = 0; i < id_to_token.size(); ++i) {
            if (id_to_token[i].empty()) {
                Logger::info("Token ID " + std::to_string(i) + " is missing in vocabulary");
                id_to_token[i] = "<missing>";
            }
        }
        
        Logger::info("Loaded vocabulary with " + std::to_string(token_to_id.size()) + " tokens");
        
    } catch (const std::exception& e) {
        throw std::runtime_error("Error loading vocabulary: " + std::string(e.what()));
    }
}

// Load BPE merges from JSON file
void Tokenizer::load_bpe_merges_from_json(const std::string& model_path) {
    try {
        // Open and parse the model JSON file
        std::ifstream file(model_path);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open BPE model file: " + model_path);
        }
        
        json model_json;
        file >> model_json;
        
        // Clear existing merges
        bpe_merges_.clear();
        
        // Check what kind of tokenizer.json format we have
        if (model_json.contains("model") && model_json["model"].contains("type") && 
            model_json["model"]["type"] == "BPE") {
            
            Logger::info("Detected HuggingFace tokenizer.json format");
            
            // Handle HuggingFace format - look for merges array with patterns
            if (model_json["model"].contains("merges") && model_json["model"]["merges"].is_array()) {
                // Explicit merges array
                const auto& merges = model_json["model"]["merges"];
                
                for (size_t i = 0; i < merges.size(); ++i) {
                    std::string merge_entry = merges[i];
                    size_t space_pos = merge_entry.find(' ');
                    
                    if (space_pos != std::string::npos) {
                        std::string first = merge_entry.substr(0, space_pos);
                        std::string second = merge_entry.substr(space_pos + 1);
                        std::string pair = first + second;
                        
                        bpe_merges_[pair] = i;  // Use the index as priority
                    }
                }
            }
            // In TinyLlama's tokenizer.json, there's no explicit merges array
            // The last part of the vocabulary has entries with spaces that appear to be the merges
            else {
                // Try to identify merges from the vocab entries
                // This is a heuristic - we look for vocab entries that contain a space
                // which indicates they are likely merge pairs
                Logger::info("No explicit merges array found, extracting from vocabulary patterns");
                
                int merge_index = 0;
                // Look for patterns in the vocab that have spaces - these are likely merge pairs
                if (model_json["model"].contains("vocab") && model_json["model"]["vocab"].is_object()) {
                    const auto& vocab = model_json["model"]["vocab"];
                    
                    for (auto it = vocab.begin(); it != vocab.end(); ++it) {
                        const std::string& token = it.key();
                        
                        // If the token contains a space, it's likely a merge pattern
                        size_t space_pos = token.find(' ');
                        if (space_pos != std::string::npos) {
                            std::string first = token.substr(0, space_pos);
                            std::string second = token.substr(space_pos + 1);
                            std::string pair = first + second;
                            
                            // Some models use priority based on vocab ID, others use incremental priority
                            int priority = merge_index++;
                            bpe_merges_[pair] = priority;
                            
                            if (bpe_merges_.size() <= 5 || bpe_merges_.size() % 1000 == 0) {
                                Logger::info("Added merge: '" + first + "' + '" + second + "' -> '" + pair + "' with priority " + std::to_string(priority));
                            }
                        }
                    }
                }
            }
        } 
        // Fallback for other formats
        else if (model_json.contains("merges") && model_json["merges"].is_array()) {
            // Classic BPE merges format
            const auto& merges = model_json["merges"];
            
            for (size_t i = 0; i < merges.size(); ++i) {
                std::string merge_entry = merges[i];
                size_t space_pos = merge_entry.find(' ');
                
                if (space_pos != std::string::npos) {
                    std::string first = merge_entry.substr(0, space_pos);
                    std::string second = merge_entry.substr(space_pos + 1);
                    std::string pair = first + second;
                    
                    bpe_merges_[pair] = i;  // Use the index as priority
                }
            }
        } 
        else {
            throw std::runtime_error("Unsupported tokenizer model format: no merges found");
        }
        
        if (bpe_merges_.empty()) {
            Logger::info("No BPE merges found in the model file");
        } else {
            Logger::info("Loaded " + std::to_string(bpe_merges_.size()) + " BPE merges");
        }
        
    } catch (const std::exception& e) {
        throw std::runtime_error("Error loading BPE merges: " + std::string(e.what()));
    }
}

// Placeholder for loading SentencePiece model
void Tokenizer::load_sentencepiece_model(const std::string& model_path) {
    // We'll replace this with actual SentencePiece model loading in the future
    Logger::info("SentencePiece model loading not implemented yet");
    sentencepiece_model_loaded_ = false;
}

// --- START: NEW Tokenizer Constructor from GGUF Data ---
Tokenizer::Tokenizer(const GGUFData& gguf_data) {
    Logger::info("Initializing Tokenizer from GGUFData...");

    // Check for necessary data
    if (gguf_data.tokenizer_tokens.empty()) {
        throw std::runtime_error("Tokenizer Error: GGUF data missing tokenizer.ggml.tokens.");
    }

    // 1. Copy Vocabulary and Scores (assuming they exist)
    id_to_token_ = gguf_data.tokenizer_tokens;
    // scores_ = gguf_data.tokenizer_scores; // Not used by this BPE implementation directly
    // token_types_ = gguf_data.tokenizer_token_types; // Not used by this BPE implementation directly
    Logger::info("Loaded " + std::to_string(id_to_token_.size()) + " tokens from GGUF.");

    // 2. Build token_to_id map
    token_to_id_.clear();
    for (size_t i = 0; i < id_to_token_.size(); ++i) {
        token_to_id_[id_to_token_[i]] = static_cast<int>(i);
    }
    Logger::info("Built token_to_id map.");

    // 3. Load Special Token IDs from Metadata
     auto get_meta_int = [&](const std::string& key, int default_val) -> int {
        auto it = gguf_data.metadata.find(key);
        if (it != gguf_data.metadata.end()) {
            try {
                 return std::visit([&](const auto& val) -> int {
                    using T = std::decay_t<decltype(val)>;
                     if constexpr (std::is_integral_v<T> && !std::is_same_v<T, bool>) {
                         // --- FIXED RANGE CHECK --- 
                         // Check if the unsigned value exceeds the maximum positive value of int.
                         // Any value <= INT_MAX is safe to cast to int (including 0 and positive values).
                         // Negative values are impossible for unsigned types being read here (e.g., uint32_t).
                         if (val <= static_cast<typename std::make_unsigned<T>::type>(std::numeric_limits<int>::max())) {
                            return static_cast<int>(val);
                         } else {
                             Logger::warning("Metadata integer value for key '" + key + "' ('" + std::to_string(val) + "') exceeds INT_MAX. Using default.");
                         }
                         // --- END FIXED RANGE CHECK ---
                     } else {
                         Logger::warning("Metadata value for key '" + key + "' is not an integer type. Using default.");
                     }
                     return default_val;
                 }, it->second);
            } catch (const std::bad_variant_access& e) {
                Logger::error("Bad variant access for key '" + key + "': " + e.what() + ". Using default.");
            } catch (...) {
                Logger::error("Unknown error accessing metadata key '" + key + "'. Using default.");
            }
        }
        // Logger::info("Metadata key '" + key + "' not found. Using default: " + std::to_string(default_val));
        return default_val;
    };

    bos_token_id_ = get_meta_int("tokenizer.ggml.bos_token_id", -1);
    eos_token_id_ = get_meta_int("tokenizer.ggml.eos_token_id", -1);
    unk_token_id_ = get_meta_int("tokenizer.ggml.unknown_token_id", -1);
    pad_token_id_ = get_meta_int("tokenizer.ggml.padding_token_id", -1);

    // Set string representations if IDs are valid
    unk_token_ = (unk_token_id_ >= 0 && unk_token_id_ < id_to_token_.size()) ? id_to_token_[unk_token_id_] : "<unk>";
    bos_token_ = (bos_token_id_ >= 0 && bos_token_id_ < id_to_token_.size()) ? id_to_token_[bos_token_id_] : "<s>";
    eos_token_ = (eos_token_id_ >= 0 && eos_token_id_ < id_to_token_.size()) ? id_to_token_[eos_token_id_] : "</s>";
    pad_token_ = (pad_token_id_ >= 0 && pad_token_id_ < id_to_token_.size()) ? id_to_token_[pad_token_id_] : "<pad>";

    Logger::info("Special token IDs: BOS=" + std::to_string(bos_token_id_) + " ('" + bos_token_ + "')" +
                 ", EOS=" + std::to_string(eos_token_id_) + " ('" + eos_token_ + "')" +
                 ", UNK=" + std::to_string(unk_token_id_) + " ('" + unk_token_ + "')" +
                 ", PAD=" + std::to_string(pad_token_id_) + " ('" + pad_token_ + "')");

    // 4. Load Model Type
    std::string model_type = "unknown";
    auto it_model = gguf_data.metadata.find("tokenizer.ggml.model");
    if (it_model != gguf_data.metadata.end() && std::holds_alternative<std::string>(it_model->second)) {
        model_type = std::get<std::string>(it_model->second);
    }
    Logger::info("Tokenizer model type from GGUF: " + model_type);

    // 5. Process Merges (Only for BPE-based models like llama)
    bpe_merges_.clear();
    if (model_type == "llama") {
        Logger::info("Processing BPE merges from GGUF...");
        const auto& merges_vec = gguf_data.tokenizer_merges;
        if (merges_vec.empty()) {
            Logger::warning("Tokenizer: LLaMA model type specified but no merges found in GGUF (tokenizer.ggml.merges).");
        } else {
            bpe_merges_.clear(); // Ensure map is clear before loading
            int merge_rank = 0;
            for (const std::string& merge_str : merges_vec) {
                size_t space_pos = merge_str.find(' ');
                if (space_pos == std::string::npos) {
                    Logger::warning("Invalid merge format: '" + merge_str + "'. Skipping.");
                    continue;
                }
                std::string token1_str = merge_str.substr(0, space_pos);
                std::string token2_str = merge_str.substr(space_pos + 1);
                
                // --- FIXED: Use concatenated string as key, rank as value ---
                std::string merge_key = token1_str + token2_str;
                bpe_merges_[merge_key] = merge_rank++;
            }
             Logger::info("Processed " + std::to_string(bpe_merges_.size()) + " BPE merges from GGUF.");
        }
    } else {
         Logger::warning("GGUF Tokenizer model type '" + model_type + "' not explicitly handled for merges (assuming no BPE).");
    }

    // Mark SentencePiece as not loaded since we are using GGUF BPE
    sentencepiece_model_loaded_ = false;

    Logger::info("Tokenizer initialization from GGUFData complete.");
}
// --- END: NEW Tokenizer Constructor from GGUF Data ---

// --- ADDED: GGUF Vocab Size Implementation ---
int Tokenizer::vocab_size() const {
    return id_to_token_.size();
}

// --- END: GGUF Vocab Size Implementation ---
