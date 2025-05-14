#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <variant>
#include <vector>

// mmap related includes
#ifndef _WIN32
#include <sys/mman.h>   // For mmap, munmap, MAP_FAILED, posix_madvise
#include <sys/stat.h>   // For fstat, stat
#include <fcntl.h>      // For O_RDONLY
#include <unistd.h>     // For close, fstat, read, lseek, sysconf, _SC_PAGE_SIZE
#else
#define WIN32_LEAN_AND_MEAN
#include <windows.h>    // For CreateFile, CreateFileMapping, MapViewOfFile, etc.
                        // Also for GetSystemInfo, SYSTEM_INFO, PrefetchVirtualMemory (if used)
#endif

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

  // Memory-mapped tensor data related fields
#ifndef _WIN32
  int file_descriptor = -1;                          /**< File descriptor for POSIX mmap */
  static const void* MMapFailure;                     /**< POSIX mmap failure indicator - DECLARED here, DEFINED in .cpp */
#else
  HANDLE h_file = INVALID_HANDLE_VALUE;             /**< File handle for Windows */
  HANDLE h_map_file = NULL;                         /**< File mapping object handle for Windows */
  static constexpr void* const MMapFailure = NULL;  /**< Windows MapViewOfFile failure indicator */
#endif
  void* mapped_tensor_data = nullptr;                /**< Pointer to memory-mapped tensor data block */
  size_t mapped_tensor_data_size = 0;               /**< Size of the mapped tensor data block in bytes */
  uint64_t data_alignment = 32;                        /**< Alignment requirement for tensor data */
  size_t offset_diff_for_mmap = 0;                   /**< Difference between aligned mmap offset and actual data start */

  // Default constructor
#ifndef _WIN32
  GGUFData() : file_descriptor(-1), mapped_tensor_data(nullptr), mapped_tensor_data_size(0), data_alignment(32), offset_diff_for_mmap(0) {}
#else
  GGUFData() : h_file(INVALID_HANDLE_VALUE), h_map_file(NULL), mapped_tensor_data(nullptr), mapped_tensor_data_size(0), data_alignment(32), offset_diff_for_mmap(0) {}
#endif

  // Destructor to clean up memory map and file descriptor/handles
  ~GGUFData() {
#ifndef _WIN32
    if (mapped_tensor_data != nullptr && mapped_tensor_data != MMapFailure) { // MMapFailure will expand to MAP_FAILED
      munmap(mapped_tensor_data, mapped_tensor_data_size);
    }
    if (file_descriptor != -1) {
      close(file_descriptor);
    }
    file_descriptor = -1; 
#else // _WIN32
    if (mapped_tensor_data != nullptr) { // On Windows, MapViewOfFile returns NULL on failure
      UnmapViewOfFile(mapped_tensor_data);
    }
    if (h_map_file != NULL) {
      CloseHandle(h_map_file);
    }
    if (h_file != INVALID_HANDLE_VALUE) {
      CloseHandle(h_file);
    }
    h_file = INVALID_HANDLE_VALUE;
    h_map_file = NULL;
#endif
    mapped_tensor_data = nullptr; // Common for both
    mapped_tensor_data_size = 0;  // Common for both
    offset_diff_for_mmap = 0;     // Common for both
  }

  // Prevent accidental copying
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
    // Platform-specific handles
#ifndef _WIN32
    , file_descriptor(other.file_descriptor)
#else
    , h_file(other.h_file)
    , h_map_file(other.h_map_file)
#endif
    , mapped_tensor_data(other.mapped_tensor_data)
    , mapped_tensor_data_size(other.mapped_tensor_data_size)
    , data_alignment(other.data_alignment)
    , offset_diff_for_mmap(other.offset_diff_for_mmap)
  {
    // Leave other in a valid but safe state (resources transferred)
#ifndef _WIN32
    other.file_descriptor = -1;
#else
    other.h_file = INVALID_HANDLE_VALUE;
    other.h_map_file = NULL;
#endif
    other.mapped_tensor_data = nullptr;
    other.mapped_tensor_data_size = 0;
    other.offset_diff_for_mmap = 0;
  }

  GGUFData& operator=(GGUFData&& other) noexcept {
    if (this != &other) {
      // Clean up existing resources first (using this object's current platform state)
#ifndef _WIN32
      if (mapped_tensor_data != nullptr && mapped_tensor_data != MMapFailure) { // MMapFailure will expand to MAP_FAILED
        munmap(mapped_tensor_data, mapped_tensor_data_size);
      }
      if (file_descriptor != -1) {
        close(file_descriptor);
      }
#else // _WIN32
      if (mapped_tensor_data != nullptr) {
        UnmapViewOfFile(mapped_tensor_data);
      }
      if (h_map_file != NULL) {
        CloseHandle(h_map_file);
      }
      if (h_file != INVALID_HANDLE_VALUE) {
        CloseHandle(h_file);
      }
#endif

      // Move data members
      header = other.header;
      metadata = std::move(other.metadata);
      tensor_infos = std::move(other.tensor_infos);
      tensor_infos_map = std::move(other.tensor_infos_map);
      tokenizer_tokens = std::move(other.tokenizer_tokens);
      tokenizer_scores = std::move(other.tokenizer_scores);
      tokenizer_token_types = std::move(other.tokenizer_token_types);
      tokenizer_merges = std::move(other.tokenizer_merges);
      
      // Move platform-specific handles and mmap data
#ifndef _WIN32
      file_descriptor = other.file_descriptor;
#else
      h_file = other.h_file;
      h_map_file = other.h_map_file;
#endif
      mapped_tensor_data = other.mapped_tensor_data;
      mapped_tensor_data_size = other.mapped_tensor_data_size;
      data_alignment = other.data_alignment;
      offset_diff_for_mmap = other.offset_diff_for_mmap;

      // Leave other in a valid but safe state
#ifndef _WIN32
      other.file_descriptor = -1;
#else
      other.h_file = INVALID_HANDLE_VALUE;
      other.h_map_file = NULL;
#endif
      other.mapped_tensor_data = nullptr;
      other.mapped_tensor_data_size = 0;
      other.offset_diff_for_mmap = 0;
    }
    return *this;
  }
};
