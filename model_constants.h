#pragma once

#include <cstdint>

/**
 * @file model_constants.h
 * @brief Constants used throughout the TinyLlama model implementation
 *
 * This file defines various constants used in different components of the model,
 * including BFloat16 handling, attention mechanisms, rotary position embeddings,
 * and numeric stability thresholds.
 */

/**
 * @namespace bfloat16
 * @brief Constants for BFloat16 number format handling
 * 
 * BFloat16 is a 16-bit floating point format that uses 1 sign bit,
 * 8 exponent bits, and 7 mantissa bits.
 */
namespace bfloat16 {
    constexpr uint16_t EXPONENT_MASK = 0x7F80;  /**< Mask for extracting exponent bits */
    constexpr uint16_t MANTISSA_MASK = 0x007F;  /**< Mask for extracting mantissa bits */
    constexpr uint16_t SIGN_BIT = 0x8000;       /**< Mask for extracting sign bit */
    constexpr uint16_t ZERO = 0x0000;           /**< Representation of positive zero */
    constexpr uint16_t NEG_ZERO = 0x8000;       /**< Representation of negative zero */
    constexpr int SHIFT_BITS = 16;              /**< Number of bits to shift for conversion */
}

/**
 * @namespace attention
 * @brief Constants for attention mechanism calculations
 * 
 * These constants are used in the scaled dot-product attention
 * mechanism of the transformer architecture.
 */
namespace attention {
    constexpr float ATTENTION_SCALE_BASE = std::sqrt(1.0f / 8.0f);  /**< Base scaling factor for attention scores (1/sqrt(head_dim) for head_dim=8) */
    constexpr float MIN_SCALE = 1e-4f;                              /**< Minimum allowed attention scale to prevent underflow */
    constexpr float MAX_SCALE = 1e4f;                               /**< Maximum allowed attention scale to prevent overflow */
}

/**
 * @namespace rope
 * @brief Constants for Rotary Position Embedding (RoPE)
 * 
 * RoPE is used to encode positional information in the
 * transformer's attention mechanism.
 */
namespace rope {
    constexpr float ROPE_THETA = 10000.0f;          /**< Base value for frequency computation in RoPE */
    constexpr int MAX_SEQUENCE_LENGTH = 2048;       /**< Maximum supported sequence length */
}

/**
 * @namespace numeric
 * @brief Constants for ensuring numeric stability
 * 
 * These constants define thresholds and epsilon values used
 * throughout the model to prevent numerical issues.
 */
namespace numeric {
    constexpr float MIN_NORM_EPS = 1e-5f;           /**< Minimum epsilon for normalization operations */
    constexpr float DEFAULT_EPS = 1e-6f;            /**< Default epsilon for general numerical stability */
    constexpr float MAX_LOGIT_THRESHOLD = 100.0f;   /**< Maximum absolute value for logits to prevent overflow */
} 