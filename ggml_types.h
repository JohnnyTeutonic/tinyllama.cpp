#pragma once

#include <cstdint>

/**
 * @file ggml_types.h
 * @brief Type definitions for GGML (Georgi Gerganov Machine Learning) library
 *
 * This file defines the various data types used in the GGML library for
 * tensor operations and quantization. It includes both standard floating-point
 * types and various quantized formats for efficient model storage and computation.
 */

/**
 * @brief Enumeration of GGML tensor data types
 * 
 * Defines the various data types that can be used for tensors in GGML,
 * including standard floating point types (F32, F16, BF16) and various
 * quantized formats (Q2_K through Q8_K) for model compression.
 */
enum GGMLType {
  GGML_TYPE_F32 = 0,    /**< 32-bit floating point */
  GGML_TYPE_F16 = 1,    /**< 16-bit floating point */
  GGML_TYPE_Q4_0 = 2,   /**< 4-bit quantization (version 0) */
  GGML_TYPE_Q4_1 = 3,   /**< 4-bit quantization (version 1) */

  GGML_TYPE_Q5_0 = 6,   /**< 5-bit quantization (version 0) */
  GGML_TYPE_Q5_1 = 7,   /**< 5-bit quantization (version 1) */
  GGML_TYPE_Q8_0 = 8,   /**< 8-bit quantization (version 0) */
  GGML_TYPE_Q8_1 = 9,   /**< 8-bit quantization (version 1) */
  GGML_TYPE_Q2_K = 10,  /**< 2-bit quantization with K-means */
  GGML_TYPE_Q3_K = 11,  /**< 3-bit quantization with K-means */
  GGML_TYPE_Q4_K = 12,  /**< 4-bit quantization with K-means */
  GGML_TYPE_Q5_K = 13,  /**< 5-bit quantization with K-means */
  GGML_TYPE_Q6_K = 14,  /**< 6-bit quantization with K-means */
  GGML_TYPE_Q8_K = 15,  /**< 8-bit quantization with K-means */
  GGML_TYPE_I8 = 16,    /**< 8-bit signed integer */
  GGML_TYPE_I16 = 17,   /**< 16-bit signed integer */
  GGML_TYPE_I32 = 18,   /**< 32-bit signed integer */
  GGML_TYPE_BF16 = 30,  /**< Brain floating point (16-bit) */
  GGML_TYPE_COUNT,      /**< Number of defined types */
};

/**
 * @brief Enumeration of value types used in GGUF metadata
 * 
 * Defines the possible data types that can be stored in GGUF metadata
 * key-value pairs. This includes basic numeric types, strings, arrays,
 * and boolean values.
 */
enum class GGUFValueType : uint32_t {
  UINT8 = 0,     /**< 8-bit unsigned integer */
  INT8 = 1,      /**< 8-bit signed integer */
  UINT16 = 2,    /**< 16-bit unsigned integer */
  INT16 = 3,     /**< 16-bit signed integer */
  UINT32 = 4,    /**< 32-bit unsigned integer */
  INT32 = 5,     /**< 32-bit signed integer */
  FLOAT32 = 6,   /**< 32-bit floating point */
  BOOL = 7,      /**< Boolean value */
  STRING = 8,    /**< String value */
  ARRAY = 9,     /**< Array of values */
  UINT64 = 10,   /**< 64-bit unsigned integer */
  INT64 = 11,    /**< 64-bit signed integer */
  FLOAT64 = 12,  /**< 64-bit floating point */
};