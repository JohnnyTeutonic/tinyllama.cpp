#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <variant>
#include <vector>

#include "ggml_types.h"

/**
 * @file gguf_structs.h
 * @brief Data structures for GGUF (GPT-Generated Unified Format) file format
 *
 * This file defines the structures used to represent and manipulate data stored
 * in GGUF format files. GGUF is a format designed for storing large language
 * models and their associated metadata efficiently.
 */

/**
 * @brief Represents an array in GGUF metadata
 */
struct GGUFArray {
  GGUFValueType type;  /**< Type of elements in the array */
  uint64_t len;        /**< Number of elements in the array */
};

/**
 * @brief Header structure for GGUF files
 */
struct GGUFHeader {
  uint32_t magic;              /**< Magic number identifying GGUF format */
  uint32_t version;            /**< Version of the GGUF format */
  uint64_t tensor_count;       /**< Number of tensors in the file */
  uint64_t metadata_kv_count;  /**< Number of metadata key-value pairs */
};

/**
 * @brief Type for storing metadata values of various types
 * 
 * Uses std::variant to support multiple value types that can be stored
 * in GGUF metadata, including numeric types, strings, and arrays.
 */
using GGUFMetadataValue =
    std::variant<uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, float,
                 bool, std::string, uint64_t, int64_t, double, GGUFArray>;

/**
 * @brief Information about a tensor stored in a GGUF file
 */
struct GGUFTensorInfo {
  std::string name;              /**< Name of the tensor */
  std::vector<uint64_t> shape;   /**< Shape of the tensor as dimensions */
  GGMLType type;                 /**< Data type of the tensor */
  uint64_t offset;               /**< Offset of tensor data in file */
  size_t num_elements;           /**< Total number of elements in tensor */
  size_t size_in_bytes;          /**< Total size of tensor data in bytes */
};

/**
 * @brief Complete representation of a GGUF file's contents
 * 
 * This structure contains all the data from a GGUF file, including
 * header information, metadata, tensor information, and the actual
 * tensor data. It also includes tokenizer-specific data that may
 * be present in the file.
 */
struct GGUFData {
  GGUFHeader header;                                    /**< File header */
  std::map<std::string, GGUFMetadataValue> metadata;   /**< Metadata key-value pairs */
  std::vector<GGUFTensorInfo> tensor_infos;            /**< List of tensor information */
  std::map<std::string, GGUFTensorInfo> tensor_infos_map; /**< Map of tensor names to information */
  
  // Tokenizer-specific data
  std::vector<std::string> tokenizer_tokens;           /**< Vocabulary tokens */
  std::vector<float> tokenizer_scores;                 /**< Token scores for BPE */
  std::vector<uint32_t> tokenizer_token_types;         /**< Token type information */
  std::vector<std::string> tokenizer_merges;           /**< BPE merge rules */

  std::vector<uint8_t> tensor_data;                    /**< Raw tensor data */
  uint64_t data_alignment = 32;                        /**< Alignment requirement for tensor data */
};
