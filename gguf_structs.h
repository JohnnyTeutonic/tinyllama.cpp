#pragma once

#include <vector>
#include <string>
#include <variant>
#include <cstdint>
#include "ggml_types.h" // For GGMLType and GGUFValueType

struct GGUFHeader {
    uint32_t magic;
    uint32_t version;
    uint64_t tensor_count;
    uint64_t metadata_kv_count;
};

// Variant to hold different metadata value types
// Start with basic types, add arrays later
using GGUFMetadataValue = std::variant<
    uint8_t,
    int8_t,
    uint16_t,
    int16_t,
    uint32_t,
    int32_t,
    float,
    bool,
    std::string,
    uint64_t,
    int64_t,
    double
    // TODO: std::vector<GGUFMetadataValue> // For arrays 
>;

struct GGUFTensorInfo {
    std::string name;
    std::vector<uint64_t> shape; // Dimensions (e.g., {channels, height, width})
    GGMLType type;               // The quantization type from ggml_types.h
    uint64_t offset;             // Offset in bytes from the start of the data section
    size_t num_elements;       // Calculated number of elements
    size_t size_in_bytes;      // Calculated size in bytes
}; 