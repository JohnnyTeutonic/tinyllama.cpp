#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <string>
#include <vector>
#include <unordered_map>
#include <nlohmann/json.hpp>
#include <sentencepiece_processor.h>

class Tokenizer {
public:
    // Construct from data directory (loads model and config)
    Tokenizer(const std::string& data_dir);

    // Tokenize a string to vector of IDs
    std::vector<int> tokenize(const std::string& text) const;

    // Detokenize a vector of IDs to string
    std::string detokenize(const std::vector<int>& ids) const;

    // Get special token ID by name ("bos", "eos", "pad", "unk")
    int get_special_token_id(const std::string& name) const;

    // Get the chat template string (may be empty if not present)
    const std::string& chat_template() const;

private:
    sentencepiece::SentencePieceProcessor sp_;
    std::unordered_map<std::string, int> special_token_ids_;
    std::string chat_template_;
};

#endif // TOKENIZER_H 