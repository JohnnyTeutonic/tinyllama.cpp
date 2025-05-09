#ifndef VOCAB_LOADER_H
#define VOCAB_LOADER_H

#include <string>
#include <unordered_map>
#include <vector>



void load_vocab_from_json(const std::string& json_path,
                          std::unordered_map<std::string, int>& token_to_id,
                          std::vector<std::string>& id_to_token);

#endif  