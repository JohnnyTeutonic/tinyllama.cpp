#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "ggml_types.h"
#include "gguf_parser.h"  // Include for GGML_QK_K

/**
 * @file quantization.h
 * @brief Weight quantization structures and functions for model compression
 *
 * This file contains the definitions for various quantization formats and
 * functions to convert between them. The quantization methods include Q2_K,
 * Q3_K, Q4_K, Q6_K, and Q8_0 formats, each offering different compression
 * ratios and precision tradeoffs.
 */

// Forward declarations
struct block_q2_K;
struct block_q3_K;
struct block_q4_K;
struct block_q6_K;

/**
 * @brief Converts a 16-bit floating point number to 32-bit float
 * @param h The 16-bit float value to convert
 * @param is_gguf_scale_field Whether this value is from a GGUF scale field
 * @return The converted 32-bit float value
 */
float fp16_to_fp32(uint16_t h, bool is_gguf_scale_field = false);

/**
 * @brief Converts a 32-bit float to 16-bit floating point
 * @param f The 32-bit float value to convert
 * @return The converted 16-bit float value
 */
uint16_t fp32_to_fp16(float f);

#pragma pack(push, 1)

/**
 * @brief 4-bit K-quantized block structure
 * 
 * Stores weights quantized to 4 bits with block-wise scaling.
 * Each block contains 256 quantized values.
 */
struct block_q4_K {
    uint16_t d;          /**< Block scale */
    uint16_t dmin;       /**< Block minimum value */
    uint8_t scales[12];  /**< Sub-block scales */
    uint8_t qs[GGML_QK_K / 2]; /**< Quantized values */
};
static_assert(sizeof(block_q4_K) == 2 + 2 + 12 + 128, "Size mismatch for standard block_q4_K");

/**
 * @brief 6-bit K-quantized block structure
 * 
 * Stores weights quantized to 6 bits with block-wise scaling.
 * Provides better precision than Q4_K at the cost of more storage.
 */
struct block_q6_K {
    uint8_t ql[GGML_QK_K / 2];     /**< Lower 4 bits of quantized values */
    uint8_t qh[GGML_QK_K / 4];     /**< Upper 2 bits of quantized values */
    int8_t scales[GGML_QK_K / 16]; /**< Sub-block scales */
    uint16_t d;                    /**< Block scale */
};
static_assert(sizeof(block_q6_K) == 128 + 64 + 16 + 2, "Size mismatch for block_q6_K");

/**
 * @brief 2-bit K-quantized block structure
 * 
 * Stores weights quantized to 2 bits with block-wise scaling.
 * Provides maximum compression at the cost of precision.
 */
struct block_q2_K {
    uint16_t d;                    /**< Block scale */
    uint16_t dmin;                 /**< Block minimum value */
    uint8_t scales[GGML_QK_K / 16]; /**< Sub-block scales */
    uint8_t qs[GGML_QK_K / 4];     /**< Quantized values */
};
static_assert(sizeof(block_q2_K) == 2 + 2 + 16 + 64, "Size mismatch for block_q2_K");

/**
 * @brief 3-bit K-quantized block structure
 * 
 * Stores weights quantized to 3 bits with block-wise scaling.
 * Balances compression and precision between Q2_K and Q4_K.
 */
struct block_q3_K {
    uint8_t hmask[GGML_QK_K / 8]; /**< High bit masks */
    uint8_t qs[GGML_QK_K / 4];    /**< Quantized values */
    uint8_t scales[12];           /**< Sub-block scales */
    uint16_t d;                   /**< Block scale */
    uint16_t dmin;                /**< Block minimum value */
};
static_assert(sizeof(block_q3_K) == 32 + 64 + 12 + 2 + 2, "Size mismatch for block_q3_K");

/**
 * @brief 8-bit K-quantized block structure with block sums
 */
struct block_q8_K {
    uint16_t d;                      /**< Block scale */
    int8_t qs[GGML_QK_K];           /**< Quantized values */
    int16_t bsums[GGML_QK_K / 16];  /**< Block sums for fast dot product */
};

/**
 * @brief Simple 8-bit quantized block structure
 */
struct block_q8_0 {
    uint16_t d;              /**< Block scale */
    int8_t qs[GGML_QK8_0];  /**< Quantized values */
};
static_assert(sizeof(block_q8_0) == sizeof(uint16_t) + GGML_QK8_0, "Size mismatch for block_q8_0");

#pragma pack(pop)

/**
 * @brief Gets the string name of a GGML type
 * @param type The GGML type
 * @return String representation of the type
 */
const char* ggml_type_name(GGMLType type);

/**
 * @brief Gets the size in bytes of a GGML type
 * @param type The GGML type
 * @return Size in bytes
 */
size_t ggml_type_size(GGMLType type);

/**
 * @brief Gets the block size for a GGML type
 * @param type The GGML type
 * @return Block size in elements
 */
size_t ggml_type_block_size(GGMLType type);

/**
 * @brief Dequantizes a Q2_K quantized block to float32
 * @param q_data Pointer to quantized data
 * @param f_data Output float array
 * @param num_weights_in_block Number of weights to dequantize
 * @param log_details_for_this_block Whether to log dequantization details
 */
void dequantize_q2_k(const void* q_data, float* f_data,
                     int num_weights_in_block,
                     bool log_details_for_this_block = false);

/**
 * @brief Dequantizes a Q4_K quantized block to float32
 * @param qblock Pointer to Q4_K block
 * @param output_f32 Output float array
 * @param num_elements Number of elements to dequantize
 * @param log_this_block Whether to log dequantization details
 */
void dequantize_q4_k_m(const block_q4_K* qblock, float* __restrict__ output_f32,
                       int num_elements, bool log_this_block = false);

/**
 * @brief Dequantizes a Q6_K quantized block to float32
 * @param qblock Pointer to Q6_K block
 * @param output_f32 Output float array
 * @param num_elements Number of elements to dequantize
 * @param log_this_block Whether to log dequantization details
 */
void dequantize_q6_k(const block_q6_K* qblock, float* __restrict__ output_f32,
                     int num_elements, bool log_this_block = false);

/**
 * @brief Dequantizes a Q3_K quantized block to float32
 * @param q_data Pointer to quantized data
 * @param f_data Output float array
 * @param num_weights_in_block Number of weights to dequantize
 */
void dequantize_q3_k(const void* q_data, float* f_data,
                     int num_weights_in_block);

/**
 * @brief Handles conversion of int8 tensor data to float32
 * @param i8_data Input int8 data
 * @param f_data Output float array
 * @param num_elements Number of elements to convert
 */
void handle_i8_tensor(const void* i8_data, float* f_data, size_t num_elements);

/**
 * @brief Quantizes float32 data to Q4_K format
 * @param f_data Input float array
 * @param q_data Output quantized data
 * @param num_elements Number of elements to quantize
 */
void quantize_q4_k_m(const float* f_data, void* q_data, int num_elements);

/**
 * @brief Quantizes float32 data to Q6_K format
 * @param f_data Input float array
 * @param q_data Output quantized data
 * @param num_elements Number of elements to quantize
 */
void quantize_q6_k(const float* f_data, void* q_data, int num_elements);

/**
 * @brief Quantizes float32 data to Q8_K format
 * @param f_data Input float vector
 * @return Vector of Q8_K blocks
 */
std::vector<block_q8_K> quantize_fp32_to_q8_K(const std::vector<float>& f_data);

/**
 * @brief Computes dot product between Q6_K and Q8_K vectors on CPU
 * @param n Number of blocks
 * @param x Q6_K vector
 * @param y Q8_K vector
 * @param log_this_call Whether to log computation details
 * @return Dot product result
 */
float vec_dot_q6_k_q8_k_cpu(int n, const std::vector<block_q6_K>& x,
                            const std::vector<block_q8_K>& y,
                            bool log_this_call);

/**
 * @brief Computes matrix-vector product between Q6_K matrix and Q8_K vector on CPU
 * @param mat_q6k Q6_K matrix
 * @param vec_q8k Q8_K vector
 * @param out_f32 Output float vector
 * @param rows Number of matrix rows
 * @param cols Number of matrix columns
 * @param log_calls Whether to log computation details
 */
void matvec_q6k_q8k_cpu(const std::vector<block_q6_K>& mat_q6k,
                        const std::vector<block_q8_K>& vec_q8k,
                        std::vector<float>& out_f32, int rows, int cols,
                        bool log_calls);

/**
 * @brief Computes dot product between Q4_K and Q8_K vectors on CPU
 * @param n Number of blocks
 * @param x_vec Q4_K vector
 * @param y_vec Q8_K vector
 * @param log_this_call Whether to log computation details
 * @return Dot product result
 */
float vec_dot_q4_k_q8_k_cpu(int n, const std::vector<block_q4_K>& x_vec,
                            const std::vector<block_q8_K>& y_vec,
                            bool log_this_call);

/**
 * @brief Computes matrix-vector product between Q4_K matrix and Q8_K vector on CPU
 * @param mat_q4k Q4_K matrix
 * @param vec_q8k Q8_K vector
 * @param out_f32 Output float vector
 * @param rows Number of matrix rows
 * @param cols Number of matrix columns
 * @param log_calls Whether to log computation details
 */
void matvec_q4k_q8k_cpu(const std::vector<block_q4_K>& mat_q4k,
                        const std::vector<block_q8_K>& vec_q8k,
                        std::vector<float>& out_f32, int rows, int cols,
                        bool log_calls);

/**
 * @brief Dequantizes a Q8_0 block to float32
 * @param qblock Pointer to Q8_0 block
 * @param output Output float array
 */
void dequantize_q8_0_block(const block_q8_0* qblock, float* output);
