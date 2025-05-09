#ifndef VOCAB_LOADER_H
#define VOCAB_LOADER_H

#include <string>
#include <unordered_map>
#include <vector>

/**
 * @file vocab_loader.h
 * @brief Vocabulary loading utilities for the tokenizer
 *
 * This file provides functionality for loading vocabulary files in JSON format,
 * which are used by the tokenizer to map between tokens and their corresponding IDs.
 */

/**
 * @brief Loads a vocabulary from a JSON file
 * 
 * Reads a vocabulary file in JSON format and populates bidirectional mappings
 * between tokens and their IDs. The JSON file should contain a dictionary where
 * keys are tokens and values are their corresponding integer IDs.
 * 
 * @param json_path Path to the JSON vocabulary file
 * @param token_to_id Map to store token to ID mappings
 * @param id_to_token Vector to store ID to token mappings
 * @throws std::runtime_error if file cannot be read or has invalid format
 */
void load_vocab_from_json(const std::string& json_path,
                          std::unordered_map<std::string, int>& token_to_id,
                          std::vector<std::string>& id_to_token);

#endif