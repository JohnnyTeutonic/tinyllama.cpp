#pragma once

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

/**
 * A simple tokenizer class that doesn't rely on OpenNMT or any external libraries.
 * This is a basic implementation that just handles tokenization by spaces and loads
 * a vocabulary from a JSON file.
 */
class Tokenizer {
public:
    /**
     * Constructor: Load tokenizer from vocab file
     * @param model_path Ignored for now (placeholder for future SentencePiece support)
     * @param vocab_path Path to the JSON file containing token vocabulary
     */
    Tokenizer(const std::string& model_path, const std::string& vocab_path);
    
    /**
     * Tokenize text into token strings
     * @param text The input text to tokenize
     * @return Vector of token strings
     */
    std::vector<std::string> tokenize(const std::string& text) const;
    
    /**
     * Convert tokens to token IDs
     * @param tokens Vector of token strings
     * @return Vector of token IDs
     */
    std::vector<int> tokens_to_ids(const std::vector<std::string>& tokens) const;
    
    /**
     * Convert token IDs to tokens
     * @param ids Vector of token IDs
     * @return Vector of token strings
     */
    std::vector<std::string> ids_to_tokens(const std::vector<int>& ids) const;
    
    /**
     * Detokenize tokens into text
     * @param tokens Vector of token strings
     * @return The detokenized text
     */
    std::string detokenize(const std::vector<std::string>& tokens) const;
    
    // Encode/decode methods with option to add special tokens
    std::vector<int> encode(const std::string& text, bool add_special_tokens = true) const;
    std::string decode(const std::vector<int>& ids, bool skip_special_tokens = true) const;
    
    // Special token accessors
    int bos_token_id() const { return bos_token_id_; }
    int eos_token_id() const { return eos_token_id_; }
    int pad_token_id() const { return pad_token_id_; }
    int unk_token_id() const { return unk_token_id_; }

private:
    // Tokenization implementations
    std::vector<std::string> space_tokenize(const std::string& text) const;
    std::vector<std::string> bpe_tokenize(const std::string& text) const;
    std::vector<std::string> sentencepiece_tokenize(const std::string& text) const;
    
    // Loading functions
    void load_vocab_from_json(const std::string& vocab_path, 
                             std::unordered_map<std::string, int>& token_to_id,
                             std::vector<std::string>& id_to_token);
    void load_bpe_merges_from_json(const std::string& model_path);
    void load_sentencepiece_model(const std::string& model_path);
    
    // Vocabulary mapping
    std::unordered_map<std::string, int> token_to_id_;
    std::vector<std::string> id_to_token_;
    
    // BPE merges (pair -> rank)
    std::unordered_map<std::string, int> bpe_merges_;
    
    // Special token handling
    std::string unk_token_;
    std::string bos_token_;
    std::string eos_token_;
    std::string pad_token_;
    
    int unk_token_id_ = 0;
    int bos_token_id_ = -1;
    int eos_token_id_ = -1;
    int pad_token_id_ = -1;
    
    // SentencePiece model state
    bool sentencepiece_model_loaded_ = false;
}; 