#pragma once

#include <limits>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>  
#include <vector>

#include "gguf_structs.h"  
#include "logger.h"        
#include "model.h"

/**
 * A simple tokenizer class that doesn't rely on OpenNMT or any external
 * libraries. This is a basic implementation that just handles tokenization by
 * spaces and loads a vocabulary from a JSON file.
 */
class Tokenizer {
 public:
  enum PreTokenizeMethod {
    DEFAULT,     
    WHITESPACE,  
    LLAMA_REGEX  
  };

  /**
   * Constructor: Load tokenizer from vocab file
   * @param model_path Ignored for now (placeholder for future SentencePiece
   * support)
   * @param vocab_path Path to the JSON file containing token vocabulary
   */
  Tokenizer(const std::string& model_path, const std::string& vocab_path);

  
  explicit Tokenizer(const GGUFData& gguf_data);

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

  
  std::vector<int> encode(
      const std::string& text, bool add_bos = true,
      bool add_eos = false,  
      PreTokenizeMethod pre_tok_override = PreTokenizeMethod::DEFAULT) const;
  std::string decode(const std::vector<int>& ids,
                     bool skip_special_tokens = true) const;

  
  std::string apply_chat_template(const std::string& user_prompt,
                                  const std::string& system_message,
                                  const ModelConfig& config) const;

  
  int vocab_size() const;

  bool is_added_token(int id) const;

  
  int bos_token_id() const { return bos_token_id_; }
  int eos_token_id() const { return eos_token_id_; }
  int pad_token_id() const { return pad_token_id_; }
  int unk_token_id() const { return unk_token_id_; }

  std::vector<std::string> space_tokenize(const std::string& text) const;
  std::vector<std::string> bpe_tokenize(
      const std::string& text) const;  
  std::vector<std::string> regex_tokenize(const std::string& text) const;

 private:
  
  std::vector<std::string> bpe_tokenize_from_scores(
      const std::string& text) const;  
  std::vector<std::string> sentencepiece_tokenize(
      const std::string& text) const;

  
  void load_vocab_from_json(const std::string& vocab_path,
                            std::unordered_map<std::string, int>& token_to_id,
                            std::vector<std::string>& id_to_token);
  void load_bpe_merges_from_json(const std::string& model_path);
  void load_sentencepiece_model(const std::string& model_path);

  
  std::unordered_map<std::string, int> token_to_id_;
  std::vector<std::string> id_to_token_;

  
  std::unordered_map<std::string, int> bpe_merges_;

  std::vector<float> token_scores_;   
  std::vector<int32_t> token_types_;  
                                      
  bool initialized_from_gguf_ =
      false;  

  std::unordered_map<std::string, int>
      added_tokens_;  

  
  std::string unk_token_;
  std::string bos_token_;
  std::string eos_token_;
  std::string pad_token_;

  int unk_token_id_ = 0;
  int bos_token_id_ = -1;
  int eos_token_id_ = -1;
  int pad_token_id_ = -1;

  
  bool sentencepiece_model_loaded_ = false;

  std::string pre_tok_type_ = "unknown";
  std::unordered_map<int, std::string> id_to_added_token_;

  std::unordered_set<std::string> chat_template_special_tokens;

  std::unordered_map<char, int> byte_char_to_id_;
};