#pragma once

#include <map>
#include <vector>
#include <string>
#include <variant>
#include <vector>
#include <cstdint>
#include "ggml_types.h" // For GGMLType and GGUFValueType
#include <map>

// --- START: Add GGUFArray struct ---
// Represents an array stored in GGUF metadata
struct GGUFArray {
    GGUFValueType type; // Type of elements in the array
    uint64_t len;       // Number of elements
    // We might store the actual data elsewhere or load on demand.
    // For now, let's assume the parser gives us len and type.
    // The actual data might be stored in a generic vector<uint8_t>
    // or parsed into a specific vector type if needed later.
    // std::vector<GGUFMetadataValue> data; // Example if data were stored directly
};
// --- END: Add GGUFArray struct ---

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
    double,
    GGUFArray // Add the GGUFArray type to the variant
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

struct GGUFData {
    GGUFHeader header; // ADDED: Store the parsed header info
    std::map<std::string, GGUFMetadataValue> metadata;
    std::vector<GGUFTensorInfo> tensor_infos;
    std::map<std::string, GGUFTensorInfo> tensor_infos_map; // ADDED: Map for quick tensor lookup by name

    // --- ADDED: Tokenizer Data ---
    std::vector<std::string> tokenizer_tokens;
    std::vector<float>       tokenizer_scores; // Assuming scores are float
    std::vector<uint32_t>    tokenizer_token_types; // Assuming types are uint32
    std::vector<std::string> tokenizer_merges;
    // --- END: Tokenizer Data ---

    // Store tensor data aligned
    std::vector<uint8_t> tensor_data; // Raw tensor data bytes, potentially aligned
    uint64_t data_alignment = 32; // Store alignment used for tensor_data
};
