#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <variant>
#include <vector>

// mmap related includes
#include <sys/mman.h>   // For mmap, munmap
#include <sys/stat.h>   // For fstat, stat
#include <fcntl.h>      // For O_RDONLY
#include <unistd.h>     // For close, fstat, read, lseek

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

  // Memory-mapped tensor data instead of std::vector
  int file_descriptor = -1;                          /**< File descriptor for the GGUF file */
  void* mapped_tensor_data = nullptr;                /**< Pointer to memory-mapped tensor data block */
  size_t mapped_tensor_data_size = 0;               /**< Size of the mapped tensor data block in bytes */
  uint64_t data_alignment = 32;                        /**< Alignment requirement for tensor data */
  size_t offset_diff_for_mmap = 0;                   /**< Difference between aligned mmap offset and actual data start */

  // Default constructor
  GGUFData() : file_descriptor(-1), mapped_tensor_data(nullptr), mapped_tensor_data_size(0), data_alignment(32), offset_diff_for_mmap(0) {}

  // Destructor to clean up memory map and file descriptor
  ~GGUFData() {
    if (mapped_tensor_data != nullptr && mapped_tensor_data != MAP_FAILED) {
      munmap(mapped_tensor_data, mapped_tensor_data_size);
      mapped_tensor_data = nullptr; // Avoid double-free on accidental copy
    }
    if (file_descriptor != -1) {
      close(file_descriptor);
      file_descriptor = -1; // Avoid double-close on accidental copy
    }
  }

  // Prevent accidental copying which could lead to double free/close issues
  // If copies are needed, a proper copy constructor/assignment operator
  // that handles mmap duplication or remapping would be necessary.
  GGUFData(const GGUFData&) = delete;
  GGUFData& operator=(const GGUFData&) = delete;

  // Allow move semantics
  GGUFData(GGUFData&& other) noexcept 
    : header(other.header)
    , metadata(std::move(other.metadata))
    , tensor_infos(std::move(other.tensor_infos))
    , tensor_infos_map(std::move(other.tensor_infos_map))
    , tokenizer_tokens(std::move(other.tokenizer_tokens))
    , tokenizer_scores(std::move(other.tokenizer_scores))
    , tokenizer_token_types(std::move(other.tokenizer_token_types))
    , tokenizer_merges(std::move(other.tokenizer_merges))
    , file_descriptor(other.file_descriptor)
    , mapped_tensor_data(other.mapped_tensor_data)
    , mapped_tensor_data_size(other.mapped_tensor_data_size)
    , data_alignment(other.data_alignment)
    , offset_diff_for_mmap(other.offset_diff_for_mmap)
  {
    // Leave other in a valid but safe state (resources transferred)
    other.file_descriptor = -1;
    other.mapped_tensor_data = nullptr;
    other.mapped_tensor_data_size = 0;
    other.offset_diff_for_mmap = 0;
  }

  GGUFData& operator=(GGUFData&& other) noexcept {
    if (this != &other) {
      // Clean up existing resources first
      if (mapped_tensor_data != nullptr && mapped_tensor_data != MAP_FAILED) {
        munmap(mapped_tensor_data, mapped_tensor_data_size);
      }
      if (file_descriptor != -1) {
        close(file_descriptor);
      }

      header = other.header;
      metadata = std::move(other.metadata);
      tensor_infos = std::move(other.tensor_infos);
      tensor_infos_map = std::move(other.tensor_infos_map);
      tokenizer_tokens = std::move(other.tokenizer_tokens);
      tokenizer_scores = std::move(other.tokenizer_scores);
      tokenizer_token_types = std::move(other.tokenizer_token_types);
      tokenizer_merges = std::move(other.tokenizer_merges);
      file_descriptor = other.file_descriptor;
      mapped_tensor_data = other.mapped_tensor_data;
      mapped_tensor_data_size = other.mapped_tensor_data_size;
      data_alignment = other.data_alignment;
      offset_diff_for_mmap = other.offset_diff_for_mmap;

      // Leave other in a valid but safe state
      other.file_descriptor = -1;
      other.mapped_tensor_data = nullptr;
      other.mapped_tensor_data_size = 0;
      other.offset_diff_for_mmap = 0;
    }
    return *this;
  }
};
