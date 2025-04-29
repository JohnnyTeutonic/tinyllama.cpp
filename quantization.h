#pragma once

#include <cstddef> // For size_t, std::byte
#include <cstdint> // For integer types
#include <vector>
#include <limits> // Added for numeric_limits
#include "ggml_types.h" // Include the new header for GGMLType

// Forward declarations for specific block types
struct block_q2_K;
struct block_q3_K;
struct block_q4_K;
struct block_q6_K;

// Constants
constexpr int GGML_QK_K = 256;   // Default K-Quants block size

// --- ADDED: FP16/FP32 Conversion Utilities Declarations ---
float fp16_to_fp32(uint16_t h);
uint16_t fp32_to_fp16(float f);
// --- END ADDED ---

// Dequantization scales for Q4_K from ggml.c - keep as is
// ... existing code ...

// Represents a block of 256 4-bit quantized values using K-Quants
// Corrected based on ggml.c
struct block_q4_K {
    uint16_t d;             // scale (represented as uint16_t) - 2 bytes
    uint16_t dmin;            // min (represented as uint16_t) - 2 bytes
    uint8_t scales[12];     // scales + upper 2 bits of mins (12 bytes)
    uint8_t qs[GGML_QK_K / 2];     // quants + lower 4 bits of mins (128 bytes)
};
static_assert(sizeof(block_q4_K) == 2 + 2 + 12 + 128, "Size mismatch for standard block_q4_K"); // Should be 144 bytes

// Represents a block of 256 6-bit quantized values using K-Quants
// Based on ggml.c: block_q6_K
struct block_q6_K {
    uint8_t ql[GGML_QK_K / 2];      // lower 4 bits of quants (128 bytes)
    uint8_t qh[GGML_QK_K / 4];      // upper 2 bits of quants (64 bytes)
    int8_t scales[GGML_QK_K / 16]; // scales (16 bytes)
    uint16_t d;                     // scale (represented as uint16_t) (2 bytes)
};
static_assert(sizeof(block_q6_K) == 128 + 64 + 16 + 2, "Size mismatch for block_q6_K"); // Should be 210 bytes

// Represents a block of 256 2-bit quantized values using K-Quants
struct block_q2_K {
    uint16_t d;                  // scale (2 bytes)
    uint16_t dmin;               // min value (2 bytes)
    uint8_t scales[GGML_QK_K / 16]; // scales (16 bytes)
    uint8_t qs[GGML_QK_K / 4];   // 2-bit quants, 4 values per byte (64 bytes)
};
static_assert(sizeof(block_q2_K) == 2 + 2 + 16 + 64, "Size mismatch for block_q2_K"); // Should be 84 bytes

// Represents a block of 256 3-bit quantized values using K-Quants
// Based on ggml.c: block_q3_K
struct block_q3_K {
    uint8_t hmask[GGML_QK_K / 8]; // Weight high bit mask (32 bytes)
    uint8_t qs[GGML_QK_K / 4];   // Low 2 bits of weights (64 bytes)
    uint8_t scales[12];          // Scales (12 bytes)
    uint16_t d;                  // Super-block scale (2 bytes)
    uint16_t dmin;               // Super-block min value (2 bytes) - Added dmin based on ggml.c
};
static_assert(sizeof(block_q3_K) == 32 + 64 + 12 + 2 + 2, "Size mismatch for block_q3_K"); // Should be 112 bytes

// --- Quantization Type Information ---

// Helper to get string name for GGMLType
const char* ggml_type_name(GGMLType type);

size_t ggml_type_size(GGMLType type); // Returns size in bytes of the *elements* (e.g., float=4)
size_t ggml_type_block_size(GGMLType type); // Returns number of elements per block for block types (e.g., QK_K for _K types)

// --- Dequantization Functions ---

// Dequantize Q2_K data
// Matching the implementation signature for other dequant functions
// Added log_details flag for specific debugging
void dequantize_q2_k(const void* q_data, float* f_data, int num_weights_in_block, bool log_details_for_this_block = false);

// Placeholder/Main dequantizer for Q4_K (Might call q4_k_m or other variations)
// Matching the implementation signature which uses 'num_weights_in_block'
// Added log_details flag for specific debugging
void dequantize_q4_k_m(const void* q_data, float* f_data, int num_weights_in_block, bool log_details_for_this_block = false);

// Dequantize Q6_K data
// Matching the implementation signature
void dequantize_q6_k(const void* q_data, float* f_data, int num_weights_in_block);

// Dequantize Q3_K data (Placeholder)
void dequantize_q3_k(const void* q_data, float* f_data, int num_weights_in_block);

// Handle I8 data (simple cast for now)
// Matching the implementation signature which uses size_t
void handle_i8_tensor(const void* i8_data, float* f_data, size_t num_elements);

// --- Quantization Functions ---

// Basic Q4_K quantization function for round-trip testing
// Takes FP32 data and quantizes it to Q4_K format
// Note: This is designed for testing and works with both block_q4_K and block_q4_K_test
// structs from the test file
void quantize_q4_k_m(const float* f_data, void* q_data, int num_elements);

// Basic Q6_K quantization function for round-trip testing
void quantize_q6_k(const float* f_data, void* q_data, int num_elements);