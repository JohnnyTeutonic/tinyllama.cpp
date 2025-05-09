#include "vocab_loader.h"

#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>

using json = nlohmann::json;

void load_vocab_from_json(const std::string& json_path,
                          std::unordered_map<std::string, int>& token_to_id,
                          std::vector<std::string>& id_to_token) {
  std::ifstream in(json_path);
  if (!in) throw std::runtime_error("Failed to open vocab file: " + json_path);
  json j;
  in >> j;
  if (!j.contains("model") || !j["model"].contains("vocab"))
    throw std::runtime_error("tokenizer.json missing model.vocab");
  const auto& vocab = j["model"]["vocab"];
  token_to_id.clear();
  id_to_token.clear();

  int max_id = -1;
  for (auto it = vocab.begin(); it != vocab.end(); ++it) {
    int id = it.value();
    if (id > max_id) max_id = id;
  }
  id_to_token.resize(max_id + 1);
  for (auto it = vocab.begin(); it != vocab.end(); ++it) {
    const std::string& token = it.key();
    int id = it.value();
    token_to_id[token] = id;
    id_to_token[id] = token;
  }
}