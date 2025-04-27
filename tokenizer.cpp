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
    
    // Check if we're using a TinyLlama-style tokenizer
    bool using_space_prefix = false;
    for (const auto& token : id_to_token_) {
        if (!token.empty() && token[0] == '\xC4' && token.size() > 1 && token[1] == '\xA0') {
            // UTF-8 representation of Ġ (U+0120)
            using_space_prefix = true;
            break;
        }
        // TinyLlama uses a special unicode character '▁' (U+2581)
        if (!token.empty() && token[0] == '\xE2' && token.size() > 2 && token[1] == '\x96' && token[2] == '\x81') {
            // UTF-8 representation of ▁ (U+2581)
            using_space_prefix = true;
            break;
        }
    }
    
    // Add a leading space if needed
    std::string processed_text = text;
    if (using_space_prefix && !text.empty() && text[0] != ' ') {
        processed_text = " " + text;
    }
    
    // Split the text by whitespace to get words
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
                current_word = " ";
                in_whitespace = false;
            }
        } else {
            current_word.push_back(c);
            in_whitespace = false;
        }
    }
    
    // Add the last word if there is one
    if (!current_word.empty()) {
        words.push_back(current_word);
    }
    
    // Process each word with BPE
    for (auto& word : words) {
        // Special handling for TinyLlama-style prefix
        if (using_space_prefix && word.size() > 0 && word[0] == ' ') {
            // Replace the space at the beginning with the special token
            if (word.size() == 1) {
                // This is just a space - use the UTF-8 sequence for ▁ (U+2581)
                word = "\xE2\x96\x81";
            } else {
                word = "\xE2\x96\x81" + word.substr(1);
            }
        }
        
        // Split word into individual UTF-8 characters
        std::vector<std::string> chars;
        for (size_t i = 0; i < word.size(); ) {
            // Get unicode code point and advance i accordingly
            int bytes = 1;
            if ((word[i] & 0xE0) == 0xC0) bytes = 2;  // 2-byte sequence
            else if ((word[i] & 0xF0) == 0xE0) bytes = 3;  // 3-byte sequence
            else if ((word[i] & 0xF8) == 0xF0) bytes = 4;  // 4-byte sequence
            
            // Get the character
            if (i + bytes <= word.size()) {
                chars.push_back(word.substr(i, bytes));
            } else {
                // Handle truncated UTF-8 sequence
                chars.push_back(word.substr(i));
            }
            i += bytes;
        }
        
        if (chars.empty()) continue;
        
        // Apply BPE merges
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
        
        // Collect the final tokens
        all_tokens.insert(all_tokens.end(), chars.begin(), chars.end());
    }
    
    return all_tokens;
}

// SentencePiece tokenization (placeholder)
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

// Encode text to token IDs
std::vector<int> Tokenizer::encode(const std::string& text, bool add_special_tokens) const {
    std::vector<std::string> tokens = tokenize(text);
    
    if (add_special_tokens) {
        if (bos_token_id_ >= 0) {
            tokens.insert(tokens.begin(), bos_token_);
        }
        if (eos_token_id_ >= 0) {
            tokens.push_back(eos_token_);
        }
    }
    
    return tokens_to_ids(tokens);
}

// Decode token IDs to text
std::string Tokenizer::decode(const std::vector<int>& ids, bool skip_special_tokens) const {
    std::vector<std::string> tokens = ids_to_tokens(ids);
    
    if (skip_special_tokens) {
        // Remove special tokens
        std::unordered_set<std::string> special_tokens = {bos_token_, eos_token_, pad_token_, unk_token_};
        
        tokens.erase(
            std::remove_if(tokens.begin(), tokens.end(), 
                [&special_tokens](const std::string& token) {
                    return special_tokens.find(token) != special_tokens.end();
                }),
            tokens.end()
        );
    }
    
    return detokenize(tokens);
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
