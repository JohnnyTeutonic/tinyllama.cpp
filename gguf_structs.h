#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <variant>
#include <vector>

#include "ggml_types.h"  // For GGMLType and GGUFValueType

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

// Variant to hold different metadata value types
using GGUFMetadataValue =
    std::variant<uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, float,
                 bool, std::string, uint64_t, int64_t, double, GGUFArray>;

struct GGUFTensorInfo {
  std::string name;
  std::vector<uint64_t> shape;  // Dimensions (e.g., {channels, height, width})
  GGMLType type;                // The quantization type from ggml_types.h
  uint64_t offset;       // Offset in bytes from the start of the data section
  size_t num_elements;   // Calculated number of elements
  size_t size_in_bytes;  // Calculated size in bytes
};

struct GGUFData {
  GGUFHeader header;
  std::map<std::string, GGUFMetadataValue> metadata;
  std::vector<GGUFTensorInfo> tensor_infos;
  std::map<std::string, GGUFTensorInfo> tensor_infos_map;
  std::vector<std::string> tokenizer_tokens;
  std::vector<float> tokenizer_scores;          // Assuming scores are float
  std::vector<uint32_t> tokenizer_token_types;  // Assuming types are uint32
  std::vector<std::string> tokenizer_merges;

  // Store tensor data aligned
  std::vector<uint8_t>
      tensor_data;               // Raw tensor data bytes, potentially aligned
  uint64_t data_alignment = 32;  // Store alignment used for tensor_data
};
