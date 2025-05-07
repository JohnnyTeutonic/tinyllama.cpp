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
float fp16_to_fp32(uint16_t h, bool is_gguf_scale_field = false);
uint16_t fp32_to_fp16(float f);
// --- END ADDED ---

// Dequantization scales for Q4_K from ggml.c - keep as is
// ... existing code ...

#pragma pack(push, 1)
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

// Represents a block of 256 8-bit quantized values using K-Quants
// Based on ggml.c: block_q8_K
struct block_q8_K {
    uint16_t d;                     // Super-block scale factor (FP16)
    int8_t qs[GGML_QK_K];           // Quantized values (8-bit integers)
    int16_t bsums[GGML_QK_K / 16];  // Block sums, used for dot product calculation (16 sums)
};
// static_assert removed temporarily due to apply issues.
// Expected size: sizeof(uint16_t) + (sizeof(int8_t) * GGML_QK_K) + (sizeof(int16_t) * (GGML_QK_K / 16)) = 2 + 256 + 32 = 290

// --- ADDED: Q8_0 Block Definition ---
#define GGML_QK8_0 32 // Standard block size for Q8_0 (make sure this matches ggml.c if used)
struct block_q8_0 {
    uint16_t d;           // Scale (typically FP16)
    int8_t  qs[GGML_QK8_0];  // Quantized values
};
static_assert(sizeof(block_q8_0) == sizeof(uint16_t) + GGML_QK8_0, "Size mismatch for block_q8_0");
// --- END ADDED ---

#pragma pack(pop)

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

// Dequantize Q4_K data - Modified to match llama.cpp logic
void dequantize_q4_k_m(const block_q4_K* qblock,
                     float* __restrict__ output_f32,
                     int num_elements, 
                     bool log_this_block = false);

// Dequantize Q6_K data
// Matching the implementation signature
void dequantize_q6_k(const block_q6_K* qblock,
                   float* __restrict__ output_f32,
                   int num_elements, 
                   bool log_this_block = false);

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

// --- ADDED: Q8_K Quantization Function Declaration ---
std::vector<block_q8_K> quantize_fp32_to_q8_K(const std::vector<float>& f_data);

// --- ADDED: Q6_K * Q8_K Dot Product Function Declaration ---
float vec_dot_q6_k_q8_k_cpu(
    int n, // Number of elements
    const std::vector<block_q6_K>& x, // Q6_K vector (matrix row)
    const std::vector<block_q8_K>& y,  // Q8_K vector (quantized activation)
    bool log_this_call // ADDED: Logging flag
);

// --- ADDED: Q6_K * Q8_K Matrix-Vector Product Function Declaration ---
void matvec_q6k_q8k_cpu(
    const std::vector<block_q6_K>& mat_q6k, // Q6_K matrix weights
    const std::vector<block_q8_K>& vec_q8k, // Q8_K input vector (quantized activation)
    std::vector<float>& out_f32,           // Output FP32 vector
    int rows,                              // Matrix rows (output vector size)
    int cols,                              // Matrix cols (input vector size)
    bool log_calls // ADDED: Logging flag
);

// --- ADDED: Q4_K * Q8_K Dot Product Function Declaration ---
float vec_dot_q4_k_q8_k_cpu(
    int n,
    const std::vector<block_q4_K>& x_vec,
    const std::vector<block_q8_K>& y_vec,
    bool log_this_call // Optional logging flag (currently unused)
);

// --- ADDED: Q4_K * Q8_K Matrix-Vector Product Function Declaration ---
void matvec_q4k_q8k_cpu(
    const std::vector<block_q4_K>& mat_q4k,
    const std::vector<block_q8_K>& vec_q8k,
    std::vector<float>& out_f32,
    int rows,
    int cols,
    bool log_calls // Optional logging flag (currently unused)
);

// --- ADDED: Declaration for dequantize_q8_0_block ---
void dequantize_q8_0_block(const block_q8_0* qblock, float* output); // Dequantizes GGML_QK8_0 elements
// --- END ADDED ---

// --- Utility functions (if any) ---