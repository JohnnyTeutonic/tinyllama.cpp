#pragma once

#include <cstddef>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "gguf_structs.h"

// GGUF magic number, spells "GGUF" in ASCII
constexpr uint32_t GGUF_MAGIC = 0x46554747;

// Constants for GGUF parsing
constexpr uint64_t GGUF_DEFAULT_ALIGNMENT = 32;
constexpr uint32_t GGUF_MAX_TENSOR_DIMS = 4;
constexpr uint64_t GGUF_STRING_MAX_LENGTH = 1ull << 30;  // 1GB sanity limit for strings

// Constants for numeric stability and error handling
constexpr float GGUF_EPSILON = 1e-10f;  // Minimum value for numeric stability
constexpr float GGUF_SMALL_VAL = 1e-6f; // Small value for near-zero comparisons

// Block size constants for quantization
constexpr size_t GGML_QK_K = 256;                // Block size for K-quantized formats (Q2_K, Q3_K, Q4_K, Q6_K)
constexpr size_t GGML_QK8_0 = 32;                // Block size for Q8_0 format
constexpr size_t GGML_Q4K_BLOCK_SIZE = 32;       // Block size for Q4_K
constexpr size_t GGML_Q6K_BLOCK_SIZE = 256;      // Block size for Q6_K

// Constants for tensor processing
constexpr float TENSOR_SCALE_MAX = 1000.0f;  // Maximum allowed scale value
constexpr float TENSOR_SCALE_MIN = -1000.0f; // Minimum allowed scale value

// Quantization scaling constants
constexpr float Q4K_SCALE_FACTOR = 15.0f;    // Scale factor for Q4_K quantization
constexpr float Q6K_SCALE_FACTOR = 31.0f;    // Scale factor for Q6_K quantization
constexpr float Q8K_SCALE_FACTOR = 127.0f;   // Scale factor for Q8_K quantization

// Quantization offset constants
constexpr int8_t Q4K_OFFSET = 8;    // Offset for Q4_K quantization
constexpr int8_t Q6K_OFFSET = 32;   // Offset for Q6_K quantization

template <typename T>
void read_raw(std::ifstream& file, T& dest);

std::string read_gguf_string(std::ifstream& file);

GGUFData load_gguf_meta(const std::string& filename);
