#include "tokenizer.h"
#include <fstream>
#include <stdexcept>

Tokenizer::Tokenizer(const std::string& data_dir) {
    // Load SentencePiece model
    std::string spm_path = data_dir + "/tokenizer.model";
    auto status = sp_.Load(spm_path);
    if (!status.ok()) {
        throw std::runtime_error("Failed to load SentencePiece model: " + spm_path + ", error: " + status.ToString());
    }

    // Load tokenizer_config.json
    std::string config_path = data_dir + "/tokenizer_config.json";
    std::ifstream config_file(config_path);
    if (!config_file) throw std::runtime_error("Failed to open " + config_path);
    nlohmann::json config;
    config_file >> config;

    // Parse special tokens
    if (config.contains("bos_token")) {
        special_token_ids_["bos"] = sp_.PieceToId(config["bos_token"].get<std::string>());
    }
    if (config.contains("eos_token")) {
        special_token_ids_["eos"] = sp_.PieceToId(config["eos_token"].get<std::string>());
    }
    if (config.contains("pad_token")) {
        special_token_ids_["pad"] = sp_.PieceToId(config["pad_token"].get<std::string>());
    }
    if (config.contains("unk_token")) {
        special_token_ids_["unk"] = sp_.PieceToId(config["unk_token"].get<std::string>());
    }

    // Parse chat template
    if (config.contains("chat_template")) {
        chat_template_ = config["chat_template"].get<std::string>();
    } else {
        chat_template_.clear();
    }
}

std::vector<int> Tokenizer::tokenize(const std::string& text) const {
    std::vector<int> ids;
    sp_.Encode(text, &ids);
    return ids;
}

std::string Tokenizer::detokenize(const std::vector<int>& ids) const {
    std::string text;
    sp_.Decode(ids, &text);
    return text;
}

int Tokenizer::get_special_token_id(const std::string& name) const {
    auto it = special_token_ids_.find(name);
    if (it == special_token_ids_.end()) throw std::runtime_error("Special token not found: " + name);
    return it->second;
}

const std::string& Tokenizer::chat_template() const {
    return chat_template_;
} 