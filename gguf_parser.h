#pragma once

#include <cstddef>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "gguf_structs.h"

/**
 * @file gguf_parser.h
 * @brief Parser for GGUF (GPT-Generated Unified Format) files
 *
 * This file contains constants and functions for parsing GGUF format files,
 * which are used to store language models and their associated metadata.
 * The format supports various quantization methods and tensor storage options.
 */

/**
 * @brief GGUF magic number that identifies the file format
 * Spells "GGUF" in ASCII (0x47475546)
 */
constexpr uint32_t GGUF_MAGIC = 0x46554747;

/**
 * @brief Constants for GGUF file parsing and validation
 */
constexpr uint64_t GGUF_DEFAULT_ALIGNMENT = 32;          /**< Default alignment for tensor data */
constexpr uint32_t GGUF_MAX_TENSOR_DIMS = 4;            /**< Maximum number of dimensions for tensors */
constexpr uint64_t GGUF_STRING_MAX_LENGTH = 1ull << 30; /**< Maximum length for strings (1GB) */

/**
 * @brief Constants for numeric stability in calculations
 */
constexpr float GGUF_EPSILON = 1e-10f;   /**< Minimum value for numeric stability */
constexpr float GGUF_SMALL_VAL = 1e-6f;  /**< Small value for near-zero comparisons */

/**
 * @brief Block size constants for different quantization formats
 */
constexpr size_t GGML_QK_K = 256;          /**< Block size for K-quantized formats */
constexpr size_t GGML_QK8_0 = 32;          /**< Block size for Q8_0 format */
constexpr size_t GGML_Q4K_BLOCK_SIZE = 32; /**< Block size for Q4_K format */
constexpr size_t GGML_Q6K_BLOCK_SIZE = 256;/**< Block size for Q6_K format */

/**
 * @brief Constants for tensor value validation
 */
constexpr float TENSOR_SCALE_MAX = 1000.0f;  /**< Maximum allowed scale value */
constexpr float TENSOR_SCALE_MIN = -1000.0f; /**< Minimum allowed scale value */

/**
 * @brief Scale factors for different quantization methods
 */
constexpr float Q4K_SCALE_FACTOR = 15.0f;    /**< Scale factor for Q4_K quantization */
constexpr float Q6K_SCALE_FACTOR = 31.0f;    /**< Scale factor for Q6_K quantization */
constexpr float Q8K_SCALE_FACTOR = 127.0f;   /**< Scale factor for Q8_K quantization */

/**
 * @brief Offset values for quantization methods
 */
constexpr int8_t Q4K_OFFSET = 8;    /**< Offset for Q4_K quantization */
constexpr int8_t Q6K_OFFSET = 32;   /**< Offset for Q6_K quantization */

/**
 * @brief Reads raw binary data from a file stream
 * @tparam T The type of data to read
 * @param file Input file stream
 * @param dest Destination variable to store the read data
 */
template <typename T>
void read_raw(std::ifstream& file, T& dest);

/**
 * @brief Reads a string from a GGUF format file
 * @param file Input file stream positioned at the start of a string
 * @return The string read from the file
 * @throws std::runtime_error if string length exceeds GGUF_STRING_MAX_LENGTH
 */
std::string read_gguf_string(std::ifstream& file);

/**
 * @brief Loads metadata from a GGUF file
 * @param filename Path to the GGUF file
 * @return GGUFData structure containing the file's metadata
 * @throws std::runtime_error if file is invalid or corrupted
 */
GGUFData load_gguf_meta(const std::string& filename);
