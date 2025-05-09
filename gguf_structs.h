#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <variant>
#include <vector>

#include "ggml_types.h"

struct GGUFArray {
  GGUFValueType type;
  uint64_t len;
};

struct GGUFHeader {
  uint32_t magic;
  uint32_t version;
  uint64_t tensor_count;
  uint64_t metadata_kv_count;
};

using GGUFMetadataValue =
    std::variant<uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, float,
                 bool, std::string, uint64_t, int64_t, double, GGUFArray>;

struct GGUFTensorInfo {
  std::string name;
  std::vector<uint64_t> shape;
  GGMLType type;
  uint64_t offset;
  size_t num_elements;
  size_t size_in_bytes;
};

struct GGUFData {
  GGUFHeader header;
  std::map<std::string, GGUFMetadataValue> metadata;
  std::vector<GGUFTensorInfo> tensor_infos;
  std::map<std::string, GGUFTensorInfo> tensor_infos_map;
  std::vector<std::string> tokenizer_tokens;
  std::vector<float> tokenizer_scores;
  std::vector<uint32_t> tokenizer_token_types;
  std::vector<std::string> tokenizer_merges;

  std::vector<uint8_t> tensor_data;
  uint64_t data_alignment = 32;
};
