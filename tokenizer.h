#pragma once

#include <limits>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <queue>
#include <functional>

#include "gguf_structs.h"
#include "logger.h"
#include "model.h"



// Helper struct to represent segments during BPE tokenization
struct llm_symbol {
  using index = int;
  index prev;              // index of the previous symbol in the linked list
  index next;              // index of the next symbol in the linked list
  const char * text;       // pointer to the start of the symbol's text in the original string
  size_t n;                // length of the symbol's text
};

// Helper struct representing a potential byte pair merge
struct llm_bigram_bpe {
    // Comparator for the priority queue: higher rank first, then lower left index
    struct comparator {
        bool operator()(const llm_bigram_bpe & l, const llm_bigram_bpe & r) const {
            // Prioritize lower rank (higher priority in BPE merges)
            // If ranks are equal, prioritize the one starting earlier (lower left index)
            return l.rank > r.rank || (l.rank == r.rank && l.left > r.left);
        }
    };

    using queue_storage = std::vector<llm_bigram_bpe>;
    // Define a min-priority queue based on the comparator
    using queue = std::priority_queue<llm_bigram_bpe, queue_storage, comparator>; 

    llm_symbol::index left;  // index of the left symbol in the pair
    llm_symbol::index right; // index of the right symbol in the pair
    std::string text;        // the merged text of the pair (for checking against merges map)
    int rank;                // rank of the merge (lower is better)
    size_t size;             // size of the merged text (for validation)
};



/**
 * @brief A lightweight tokenizer implementation for text processing
 * 
 * This tokenizer class provides basic text tokenization functionality without external
 * dependencies. It supports multiple tokenization methods including space-based,
 * BPE (Byte-Pair Encoding), and regex-based tokenization. The tokenizer can be
 * initialized either from a vocabulary file or from GGUF format data.
 */
class Tokenizer {
 public:
  /**
   * @brief Enumeration of available pre-tokenization methods
   */
  enum PreTokenizeMethod {
    DEFAULT,     /**< Use the default tokenization method specified during initialization */
    LLAMA_REGEX  /**< LLaMA-style regex-based tokenization */
  };

  enum class Type {
    UNKNOWN,
    SENTENCEPIECE_BPE,
    TIKTOKEN_BPE
  };

  /**
   * @brief Constructs a tokenizer from vocabulary and model files (for Llama 2 style JSON)
   * @param vocab_path Path to the JSON vocabulary file
   * @param model_path Path to the JSON model file containing BPE merges (optional)
   * @param config The model configuration (used to get special token IDs if not in vocab file)
   */
  Tokenizer(const std::string& vocab_path, const std::string& model_path, const ModelConfig& config);

  /**
   * @brief Constructs a tokenizer from GGUF format data
   * @param gguf_data The GGUF data containing tokenizer information
   * @param config The model configuration (contains tokenizer_family, special token IDs etc.)
   */
  explicit Tokenizer(const GGUFData& gguf_data, const ModelConfig& config);

  /**
   * @brief Tokenizes input text into token strings
   * @param text The input text to tokenize
   * @return Vector of token strings
   */
  std::vector<std::string> tokenize(const std::string& text) const;


  /**
   * @brief Converts token IDs back to token strings
   * @param ids Vector of token IDs to convert
   * @return Vector of token strings
   */
  std::vector<std::string> ids_to_tokens(const std::vector<int>& ids) const;

  /**
   * @brief Combines tokens back into text
   * @param tokens Vector of token strings to combine
   * @return The reconstructed text
   */
  std::string detokenize(const std::vector<std::string>& tokens) const;

  /**
   * @brief Encodes text into token IDs with optional special tokens
   * @param text Text to encode
   * @param add_bos Whether to add beginning-of-sequence token
   * @param add_eos Whether to add end-of-sequence token
   * @param pre_tok_override Override the default pre-tokenization method
   * @return Vector of token IDs
   */
  std::vector<int> encode(
      const std::string& text, bool add_bos = true, bool add_eos = false,
      PreTokenizeMethod pre_tok_override = PreTokenizeMethod::DEFAULT) const;

  /**
   * @brief Decodes token IDs back to text
   * @param ids Vector of token IDs to decode
   * @param skip_special_tokens Whether to skip special tokens in output
   * @return The decoded text
   */
  std::string decode(const std::vector<int>& ids,
                     bool skip_special_tokens = true) const;

  /**
   * @brief Applies chat template formatting to the input prompt
   * @param user_prompt The user's input text
   * @param system_message The system message to prepend
   * @param config Model configuration containing template information
   * @return Formatted chat text
   */
  std::string apply_chat_template(const std::string& user_prompt,
                                  const std::string& system_message,
                                  const ModelConfig& config) const;

  /**
   * @brief Returns the size of the vocabulary
   * @return Number of tokens in vocabulary
   */
  int vocab_size() const;

  /**
   * @brief Checks if a token ID represents an added token
   * @param id Token ID to check
   * @return True if token was added, false otherwise
   */
  bool is_added_token(int id) const;

  /**
   * @brief Gets the beginning-of-sequence token ID
   * @return BOS token ID
   */
  int bos_token_id() const { return bos_token_id_; }

  /**
   * @brief Gets the end-of-sequence token ID
   * @return EOS token ID
   */
  int eos_token_id() const { return eos_token_id_; }

  /**
   * @brief Gets the padding token ID
   * @return PAD token ID
   */
  int pad_token_id() const { return pad_token_id_; }

  /**
   * @brief Gets the unknown token ID
   * @return UNK token ID
   */
  int unk_token_id() const { return unk_token_id_; }

  const std::string& get_gguf_chat_template() const;

 private:
  /**
   * @brief Loads vocabulary from JSON file
   */
  void load_vocab_from_json(const std::string& vocab_path,
                            std::unordered_map<std::string, int>& token_to_id,
                            std::vector<std::string>& id_to_token);

  /**
   * @brief Loads BPE merge rules from JSON file
   */
  void load_bpe_merges_from_json(const std::string& model_path);

  /**
   * @brief Loads a SentencePiece model
   */
  void load_sentencepiece_model(const std::string& model_path);


  // Token mappings
  std::unordered_map<std::string, int> token_to_id_;    /**< Maps tokens to their IDs */
  std::vector<std::string> id_to_token_;                /**< Maps IDs to their tokens */
  std::unordered_map<std::string, int> bpe_merges_;     /**< BPE merge rules (rank/order based for SentencePiece/Tiktoken) */
  std::vector<std::string> tiktoken_merges_list_;       /**< Tiktoken BPE merge rules, loaded as ordered list from GGUF */
  std::vector<float> token_scores_;                      /**< Token scores for BPE (primarily for Llama2 GGUF) */
  std::vector<int32_t> token_types_;                    /**< Token type information from GGUF */
  ModelConfig::TokenizerFamily tokenizer_family_ = ModelConfig::TokenizerFamily::UNKNOWN;
  bool initialized_from_gguf_ = false;                   /**< Initialization source flag */
  std::unordered_map<std::string, int> added_tokens_;   /**< Additional tokens */

  // Special tokens
  std::string unk_token_;    /**< Unknown token string */
  std::string bos_token_;    /**< Beginning of sequence token string */
  std::string eos_token_;    /**< End of sequence token string */
  std::string pad_token_;    /**< Padding token string */

  // Special token IDs
  int unk_token_id_ = 0;     /**< Unknown token ID */
  int bos_token_id_ = -1;    /**< Beginning of sequence token ID */
  int eos_token_id_ = -1;    /**< End of sequence token ID */
  int pad_token_id_ = -1;    /**< Padding token ID */

  bool sentencepiece_model_loaded_ = false;              /**< SentencePiece model status */
  std::string pre_tok_type_ = "unknown";                /**< Pre-tokenization type */
  std::unordered_map<int, std::string> id_to_added_token_; /**< Maps IDs to added tokens */
  std::unordered_set<std::string> chat_template_special_tokens; /**< Special tokens for chat */
  std::unordered_map<char, int> byte_char_to_id_;       /**< Byte-level character mapping */

  Type type_ = Type::UNKNOWN;
  std::string gguf_chat_template_;                    /**< Chat template string from GGUF metadata */

  // SentencePiece specific helper methods (reinstated)
  std::vector<std::string> bpe_tokenize(const std::string& text) const;
  std::vector<std::string> bpe_tokenize_from_scores(const std::string& text) const;
  std::vector<int> tokens_to_ids(const std::vector<std::string>& tokens) const;
  std::string decode_sentencepiece(const std::vector<int>& ids, bool skip_special_tokens) const;
  std::string capitalize_first_letter(std::string s) const;


  // New BPE tokenization method for TikToken (Llama 3) path
  std::vector<int> bpe_tokenize_to_ids(const std::string& text,
                                       bool add_bos_token_param,
                                       bool add_eos_token_param,
                                       bool ignore_merges_param) const;

  // Helper for the new BPE tokenization path
  void add_bigram_to_queue_refactored(const char* text_data_base,
                                      const std::vector<llm_symbol>& symbols,
                                      llm_symbol::index first_symbol_idx,
                                      std::priority_queue<std::pair<int, int>,
                                                          std::vector<std::pair<int, int>>,
                                                          std::greater<std::pair<int, int>>>& work_queue) const;

  int find_bpe_rank(const std::string & token_left, const std::string & token_right) const;
  
};
