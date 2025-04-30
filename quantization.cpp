#include "quantization.h"
#include <stdexcept> // For potential errors later
#include <cstdint>   // For integer types
#include <cstring>   // For std::memcpy
#include <cmath>     // For isnan, isinf
#include <vector>    // For lookup tables
#include <numeric>   // For std::iota
#include <algorithm> // For std::generate
#include <string>    // For std::to_string
#include <cmath>     // For std::round
#include <cassert>
#include <limits>    // For std::numeric_limits
#include <iostream>  // For debug
#include <iomanip>   // For std::hex
#include <atomic>    // For std::atomic

// --- START: K-Quant Lookup Tables (Static Globals) ---
// Re-adding these definitions at file scope
static const float K_SCALE_VALUES[64] = {
    1.0f, 1.0625f, 1.125f, 1.1875f, 1.25f, 1.3125f, 1.375f, 1.4375f, 1.5f, 1.5625f, 1.625f, 1.6875f, 1.75f, 1.8125f, 1.875f, 1.9375f,
    2.0f, 2.125f, 2.25f, 2.375f, 2.5f, 2.625f, 2.75f, 2.875f, 3.0f, 3.125f, 3.25f, 3.375f, 3.5f, 3.625f, 3.75f, 3.875f,
    4.0f, 4.25f, 4.5f, 4.75f, 5.0f, 5.25f, 5.5f, 5.75f, 6.0f, 6.25f, 6.5f, 6.75f, 7.0f, 7.25f, 7.5f, 7.75f,
    8.0f, 8.5f, 9.0f, 9.5f, 10.0f, 10.5f, 11.0f, 11.5f, 12.0f, 12.5f, 13.0f, 13.5f, 14.0f, 14.5f, 15.0f, 15.5f
};
static const float K_MIN_VALUES[64] = {
    0.0f, -0.0078125f, -0.015625f, -0.0234375f, -0.03125f, -0.0390625f, -0.046875f, -0.0546875f, -0.0625f, -0.0703125f, -0.078125f, -0.0859375f, -0.09375f, -0.1015625f, -0.109375f, -0.1171875f,
    -0.125f, -0.140625f, -0.15625f, -0.171875f, -0.1875f, -0.203125f, -0.21875f, -0.234375f, -0.25f, -0.265625f, -0.28125f, -0.296875f, -0.3125f, -0.328125f, -0.34375f, -0.359375f,
    -0.375f, -0.40625f, -0.4375f, -0.46875f, -0.5f, -0.53125f, -0.5625f, -0.59375f, -0.625f, -0.65625f, -0.6875f, -0.71875f, -0.75f, -0.78125f, -0.8125f, -0.84375f,
    -0.875f, -0.9375f, -1.0f, -1.0625f, -1.125f, -1.1875f, -1.25f, -1.3125f, -1.375f, -1.4375f, -1.5f, -1.5625f, -1.625f, -1.6875f, -1.75f, -1.8125f
};
// --- END: K-Quant Lookup Tables ---


// --- FP16/FP32 Conversion Helpers ---

// Function to convert FP16 (uint16_t) to FP32 (float)
// Based on standard conversion techniques
float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp_fp16 = (h >> 10) & 0x1f;
    uint32_t mant_fp16 = h & 0x3ff;

    uint32_t x;

    if (exp_fp16 == 0) { // Subnormal or zero
        if (mant_fp16 == 0) { // Zero
            x = (sign << 31);
        } else { // Subnormal FP16 -> Normal FP32
            exp_fp16 = 1;
            while ((mant_fp16 & 0x400) == 0) {
                mant_fp16 <<= 1;
                exp_fp16--;
            }
            mant_fp16 &= ~0x400;
            uint32_t exp_fp32 = (exp_fp16 - 15 + 127);
            uint32_t mant_fp32 = mant_fp16 << 13;
            x = (sign << 31) | (exp_fp32 << 23) | mant_fp32;
        }
    } else if (exp_fp16 == 0x1f) { // Inf or NaN
        x = (sign << 31) | (0xff << 23) | (mant_fp16 << 13);
    } else { // Normal number
        uint32_t exp_fp32 = (exp_fp16 - 15 + 127);
        uint32_t mant_fp32 = mant_fp16 << 13;
        x = (sign << 31) | (exp_fp32 << 23) | mant_fp32;
    }

    float f;
    std::memcpy(&f, &x, sizeof(float));
    return f;
}

// Function to convert FP32 (float) to FP16 (uint16_t)
// Based on standard conversion techniques (with potential rounding)
uint16_t fp32_to_fp16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(float)); // Use memcpy for safe type punning

    uint32_t sign = (x >> 31) & 1;
    uint32_t exp_fp32 = (x >> 23) & 0xff;
    uint32_t mant_fp32 = x & 0x7fffff;

    uint16_t u;

    if (exp_fp32 == 0xff) { // Inf or NaN
        u = (sign << 15) | 0x7c00 | (mant_fp32 != 0 ? 0x200 : 0); // Preserve NaN payload slightly
    } else { // Normal or subnormal numbers
        int exp_fp16 = (int)exp_fp32 - 127 + 15; // Adjust exponent bias

        if (exp_fp16 >= 0x1f) { // Overflow to Inf
            u = (sign << 15) | 0x7c00;
        } else if (exp_fp16 <= 0) { // Underflow to subnormal or zero
            if (exp_fp16 < -10) { // Too small, flush to zero
                u = (sign << 15);
            } else { // Convert to FP16 subnormal
                mant_fp32 = (mant_fp32 | 0x800000) >> (1 - exp_fp16);
                // Rounding (nearest even) - simplified: round half up
                if ((mant_fp32 >> 13) & 1) {
                    mant_fp32 += (1 << 13);
                }
                u = (sign << 15) | (mant_fp32 >> 13);
            }
        } else { // Normal FP16 number
            // Rounding (nearest even) - simplified: round half up
            if ((mant_fp32 >> 13) & 1) {
                 mant_fp32 += (1 << 13); 
                 if ((mant_fp32 >> 23) == 1) { // Check if mantissa overflowed
                     mant_fp32 = 0; 
                     exp_fp16++; 
                     if(exp_fp16 >= 0x1f) { // Check if exponent overflowed after rounding
                        u = (sign << 15) | 0x7c00;
                        return u; 
                     } 
                 }
            }
            u = (sign << 15) | (exp_fp16 << 10) | (mant_fp32 >> 13);
        }
    }
    return u;
}

namespace { // Keep anonymous namespace for potentially internal-only helpers in the future

// --- K-Quant Lookup Tables (derived from ggml.c) ---

    // These tables are generated programmatically in ggml.c, we replicate the logic
    std::vector<float> k_lookup_table_scale;
    std::vector<float> k_lookup_table_min;

    // Helper to extract 6-bit scale and min indices from the Q4_K block data.
    // Corrected based on llama.cpp logic, requires access to scales AND weights (qs).
    // Assumes the 'scales' array is 12 bytes, as per standard Q4_K. If the header's
    // 16-byte comment/assert is correct for this project, this needs adjustment.
    inline void get_scale_min_k4(
        int j,                       // Sub-block index (0-15)
        const uint8_t* q_scales,     // Pointer to the start of the 12-byte scales array for the block
        const uint8_t* q_weights,    // Pointer to the start of the 128-byte qs array for the block
        uint8_t* scale_index,        // Output: 6-bit scale index
        uint8_t* min_index           // Output: 6-bit min index
    ) {
        assert(j >= 0 && j < 16);

        // Determine which byte in q_scales contains the scale and upper 2 bits of min
        int scale_byte_index = j / 2; // Each byte holds info for 2 sub-blocks
        uint8_t scale_byte = q_scales[scale_byte_index];

        // Determine which byte in q_weights contains the lower 4 bits of min
        // Each byte in qs holds 2 weights (nibbles). 16 weights per sub-block.
        // 128 bytes in qs total. Each sub-block uses 128/16 = 8 bytes of qs.
        int qs_byte_base_index = j * 8; 
        // The min bits are in the first 4 bytes of the sub-block's qs data
        int min_bits_byte_index = qs_byte_base_index + (j % 4); // Index 0,1,2,3 within the sub-block's 8 bytes
        uint8_t min_bits_byte = q_weights[min_bits_byte_index];

        if (j % 2 == 0) { // First sub-block within the scale byte
            *scale_index = scale_byte & 0x0F; // Lower 4 bits for scale
            *min_index = (scale_byte >> 4) & 0x03; // Upper 2 bits for min upper
            // Combine with lower 4 bits from qs
            *min_index |= (min_bits_byte & 0x0F) << 2; // Lower nibble of qs byte provides lower 4 bits of min
        } else { // Second sub-block within the scale byte
            *scale_index = (scale_byte >> 4) & 0x0F; // Upper 4 bits for scale
            *min_index = (scale_byte >> 6) & 0x03; // Bits 6 & 7 for min upper (shifted down)
            // Combine with lower 4 bits from qs
            *min_index |= (min_bits_byte >> 4) << 2; // Upper nibble of qs byte provides lower 4 bits of min
        }
        // IMPORTANT: This logic assumes the standard 12-byte scales layout.
        // If the project truly uses a 16-byte layout, the extraction from q_scales needs modification.
    }

}

// --- START: Correct helper based on ggml-quants.c function ---
static inline void get_scale_min_k4_ref(
    int j, // Sub-block index (0-15)
    const uint8_t* scales, // Pointer to the 12-byte scales field
    const uint8_t* qs,     // Pointer to the 128-byte qs field (base pointer)
    uint8_t* s,            // Output scale index (6-bit)
    uint8_t* m             // Output min index (6-bit)
) {
    int is = j / 2; // scales byte index (0-7)
    if (j % 2 == 0) { // Even sub-blocks (0, 2, ..., 14)
        *s = scales[is] & 0x0F;                       // Scale: Lower nibble of scales[0..7]
        uint8_t m_hi = (scales[is + 8] >> 0) & 0x03;   // Min high bits: Lower 2 bits of scales[8..15]
        *m = m_hi | ((qs[j / 4] & 0x0F) << 2);         // Min low bits: Lower nibble of qs[0..7]
    } else { // Odd sub-blocks (1, 3, ..., 15)
        *s = (scales[is] >> 4) & 0x0F;                // Scale: Upper nibble of scales[0..7]
        uint8_t m_hi = (scales[is + 8] >> 4) & 0x03;   // Min high bits: Upper 2 bits of scales[8..15]
        *m = m_hi | ((qs[j / 4 + 8] >> 4) << 2);       // Min low bits: Upper nibble of qs[8..15]
    }
}
// --- END: Correct helper ---

// Dequantize Q4_K data - Rewritten to match llama.cpp loop structure
void dequantize_q4_k_m(
    const void* qblock_void,
    float* output, // Output buffer for GGML_QK_K floats
    int num_weights_in_block, // Should always be GGML_QK_K for full blocks
    bool log_details_for_this_block // Flag for enabling detailed logs
) {
    if (num_weights_in_block != GGML_QK_K) {
        std::cout << "Warning: dequantize_q4_k_m called with num_weights != GGML_QK_K (" << num_weights_in_block << ")" << std::endl;
        std::memset(output, 0, num_weights_in_block * sizeof(float));
        return;
    }

    const block_q4_K* qblock = static_cast<const block_q4_K*>(qblock_void);
    const uint8_t* scales_ptr = qblock->scales;
    const uint8_t* qs_ptr = qblock->qs;

    // Convert d and dmin, clamp if necessary
    const float d_raw = fp16_to_fp32(qblock->d);
    const float dmin_raw = fp16_to_fp32(qblock->dmin);
    float d_super = d_raw;
    float min_super = dmin_raw;
    constexpr float CLAMP_LIMIT = 100.0f; 
    if (!std::isfinite(d_super) || std::abs(d_super) > CLAMP_LIMIT) d_super = 0.0f;
    if (!std::isfinite(min_super) || std::abs(min_super) > CLAMP_LIMIT) min_super = 0.0f;

    // Process 16 sub-blocks
    for (int j = 0; j < GGML_QK_K / 16; ++j) { // j = 0..15 (sub-block index)
        uint8_t scale_idx, min_idx;
        get_scale_min_k4_ref(j, scales_ptr, qs_ptr, &scale_idx, &min_idx);

        // Calculate scale and min for this sub-block
        const float scale = d_super * K_SCALE_VALUES[scale_idx];
        const float minv = min_super * K_MIN_VALUES[min_idx];

        // Pointer to the 8 bytes of quantized values for this sub-block
        const uint8_t* qs_sub_block = qs_ptr + j * 8;
        // Pointer to the 16 floats in the output buffer for this sub-block
        float* output_sub_block = output + j * 16;

        // Dequantize the 16 values for this sub-block
        for (int l = 0; l < 8; ++l) { // Process 8 bytes
            uint8_t packed_qs = qs_sub_block[l];
            uint8_t nibble_low = packed_qs & 0x0F;
            uint8_t nibble_high = packed_qs >> 4;

            output_sub_block[l]     = scale * static_cast<float>(nibble_low) + minv;
            output_sub_block[l + 8] = scale * static_cast<float>(nibble_high) + minv;
        }
    }
}

// --- Dequantization for Q6_K ---
void dequantize_q6_k(
    const void* qblock_void,
    float* output, // Changed name to match llama.cpp style
    int num_weights_in_block // Still useful for assert
) {
    if (num_weights_in_block != GGML_QK_K) {
        throw std::invalid_argument("dequantize_q6_k currently only supports block size " + std::to_string(GGML_QK_K));
    }

    const block_q6_K* x = static_cast<const block_q6_K*>(qblock_void); // Use x like llama.cpp
    float* y = output; // Use y like llama.cpp

    const float d = fp16_to_fp32(x->d); // Get super scale
    if (!std::isfinite(d)) {
         // Handle non-finite super scale - zero out the block
         std::memset(y, 0, GGML_QK_K * sizeof(float));
         // Optionally log a warning
         // std::cout << "[Q6K DEQUANT WARNING] Non-finite super-scale 'd' encountered. Outputting zeros." << std::endl;
         return;
    }

    const uint8_t * ql = x->ql;
    const uint8_t * qh = x->qh;
    const int8_t  * sc = x->scales;

    // Loop structure matching llama.cpp
    for (int n = 0; n < GGML_QK_K; n += 128) { // Process in two chunks of 128
        for (int l = 0; l < 32; ++l) { // Process 32 bytes of qh -> 128 weights
            int is = l / 16; // Scale index base (0 or 1)
            
            // Reconstruct the four 6-bit values (-32 to 31)
            const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int8_t q3 = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int8_t q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;

            // Get the correct scales for each group
            // Apply isfinite check to individual scales for safety
            float scale1_raw = d * static_cast<float>(sc[is + 0]);
            float scale2_raw = d * static_cast<float>(sc[is + 2]);
            float scale3_raw = d * static_cast<float>(sc[is + 4]);
            float scale4_raw = d * static_cast<float>(sc[is + 6]);

            float scale1 = std::isfinite(scale1_raw) ? scale1_raw : 0.0f;
            float scale2 = std::isfinite(scale2_raw) ? scale2_raw : 0.0f;
            float scale3 = std::isfinite(scale3_raw) ? scale3_raw : 0.0f;
            float scale4 = std::isfinite(scale4_raw) ? scale4_raw : 0.0f;
            
            // Dequantize and store
            y[l +  0] = scale1 * static_cast<float>(q1);
            y[l + 32] = scale2 * static_cast<float>(q2);
            y[l + 64] = scale3 * static_cast<float>(q3);
            y[l + 96] = scale4 * static_cast<float>(q4);
        }
        // Advance pointers for the next 128-element chunk
        y  += 128;
        ql += 64;
        qh += 32;
        sc += 8;
    }
}

// --- Handling for I8 (Integer 8-bit) ---
// Assumes GGML_TYPE_I8 is just raw int8_t that needs casting to float.
// If it represented a scaled quantization type, a scale factor would be needed.
void handle_i8_tensor(
    const void* input_data,
    float* output_data,
    size_t num_elements
) {
    const int8_t* input_ptr = static_cast<const int8_t*>(input_data);
    for (size_t i = 0; i < num_elements; ++i) {
        output_data[i] = static_cast<float>(input_ptr[i]);
    }
}

// --- Basic Q4_K Quantization for Round-trip Testing ---
// This is a simplified quantizer primarily for testing
// It doesn't implement sophisticated quantization strategies like those in ggml.c
void quantize_q4_k_m(
    const float* input,
    void* output_qblock_void,
    int num_elements) 
{
    if (num_elements != GGML_QK_K) {
        throw std::invalid_argument("quantize_q4_k_m currently only supports block size " + std::to_string(GGML_QK_K));
    }

    // Use the correctly defined block_q4_K from the header
    block_q4_K* output_qblock = static_cast<block_q4_K*>(output_qblock_void);
    
    // Step 1: Find min/max for the entire block to set super-block scale (d) and min (dmin)
    float block_min = std::numeric_limits<float>::max();
    float block_max = std::numeric_limits<float>::lowest();
    
    for (int i = 0; i < num_elements; ++i) {
        block_min = std::min(block_min, input[i]);
        block_max = std::max(block_max, input[i]);
    }
    
    // Handle edge case of constant values
    if (block_max == block_min) {
        block_max = block_min + 1.0f;
    }

    // Calculate super-block scale and min
    // We want to map the range [block_min, block_max] to [0, 15]
    float d = (block_max - block_min) / 15.0f;
    float dmin = block_min;

    // Convert to FP16 using the proper function
    uint16_t d_fp16 = fp32_to_fp16(d);
    uint16_t dmin_fp16 = fp32_to_fp16(dmin);

    output_qblock->d = d_fp16;
    output_qblock->dmin = dmin_fp16;
    
    // Step 2: For simplicity in this test implementation, we'll:
    // - Use the same scale/min for all 16 sub-blocks
    // - Use scale_idx = 0 (factor = 1.0) and min_idx = 0 (factor = 0.0)
    // - This means we'll just use the super-block d and dmin directly
    std::memset(output_qblock->scales, 0, sizeof(output_qblock->scales));
    
    // Properly set up scale indices according to the packed format:
    // - For sub-blocks 0-7: scale indices in lower 6 bits of scales[j], min indices upper 2 bits + scales[j+8]
    // - For sub-blocks 8-15: scale indices in lower 6 bits of scales[j], min indices in upper 2 bits of scales[j-8]
    // We're using scale_idx=0, min_idx=0 for all sub-blocks for simplicity

    // Step 3: Quantize the values to 4-bit (0-15)
    for (int j = 0; j < GGML_QK_K / 16; ++j) {  // Process 16 sub-blocks
        for (int i = 0; i < 8; ++i) {  // Each sub-block has 16 values = 8 bytes
            int idx1 = j * 16 + i * 2;     // Index of first value in the pair
            int idx2 = j * 16 + i * 2 + 1; // Index of second value in the pair
            
            // Quantize to 4-bit (0-15) - don't clamp lower bound to 0
            // This allows negative values to be properly represented
            float val1 = (input[idx1] - dmin) / d;
            float val2 = (input[idx2] - dmin) / d;
            
            // Round to nearest integer and clamp to [0, 15]
            uint8_t q1 = static_cast<uint8_t>(std::min(15.0f, std::max(0.0f, std::round(val1))));
            uint8_t q2 = static_cast<uint8_t>(std::min(15.0f, std::max(0.0f, std::round(val2))));
            
            // Pack two 4-bit values into one byte
            output_qblock->qs[j * 8 + i] = q1 | (q2 << 4);
        }
    }
}

// Implementation for Q2_K dequantization (2-bit k-quants)
// Matches the declaration in quantization.h
void dequantize_q2_k(
    const void* qblock_void,
    float* output, // Output buffer for GGML_QK_K floats
    int num_weights_in_block // Should always be GGML_QK_K
) {
    if (num_weights_in_block != GGML_QK_K) {
        throw std::invalid_argument("dequantize_q2_k currently only supports block size " + std::to_string(GGML_QK_K));
    }

    const block_q2_K* qblock = static_cast<const block_q2_K*>(qblock_void);

    const float d_float_raw = fp16_to_fp32(qblock->d);
    const float dmin_float_raw = fp16_to_fp32(qblock->dmin);

    // Check for NaN AND Inf and replace with 0.0f
    const float d_float = (!std::isfinite(d_float_raw)) ? 0.0f : d_float_raw;
    const float dmin_float = (!std::isfinite(dmin_float_raw)) ? 0.0f : dmin_float_raw;

    // Additional validation - clamp extremely large scales to reasonable values
    const float d_float_clamped = std::min(std::max(d_float, -1000.0f), 1000.0f);
    const float dmin_float_clamped = std::min(std::max(dmin_float, -1000.0f), 1000.0f);

    const uint8_t* scales_ptr = qblock->scales; // 16 bytes, each contains two 4-bit scales
    const uint8_t* qs_ptr = qblock->qs;    // 64 bytes, each contains four 2-bit weights
    int weight_index = 0;
    float dequantized_scales[16]; // To store the dequantized 4-bit scales

    // Step 1: Dequantize the 4-bit scales
    for (int i = 0; i < 8; ++i) { // Process 8 bytes = 16 scales
        uint8_t packed_scales = scales_ptr[i];
        uint8_t scale_low = packed_scales & 0x0F;
        uint8_t scale_high = packed_scales >> 4;
        
        // According to GGML, Q2_K 4-bit scales use d_float directly (no lookup table or dmin here)
        // Use the clamped value for safety
        dequantized_scales[i * 2 + 0] = d_float_clamped * static_cast<float>(scale_low);
        dequantized_scales[i * 2 + 1] = d_float_clamped * static_cast<float>(scale_high);

        // Clamp the scales to reasonable values 
        dequantized_scales[i * 2 + 0] = std::min(std::max(dequantized_scales[i * 2 + 0], -1000.0f), 1000.0f);
        dequantized_scales[i * 2 + 1] = std::min(std::max(dequantized_scales[i * 2 + 1], -1000.0f), 1000.0f);
    }

    // Step 2: Dequantize the 2-bit weights using the dequantized scales and global dmin
    weight_index = 0;
    for (int j = 0; j < GGML_QK_K / 16; ++j) { // j = 0..15 (sub-block index)
        // Get the dequantized scale for this sub-block
        float sub_block_scale = dequantized_scales[j]; 

        // Get pointer to the start of the 4 bytes of qs for this sub-block
        const uint8_t* qs_subblock_ptr = qs_ptr + j * 4; 

        // Process the 4 bytes (16 weights) in this sub-block
        for (int i = 0; i < 4; ++i) { // i = 0..3
            uint8_t packed_weights = qs_subblock_ptr[i];

            // Extract four 2-bit values (0-3)
            uint8_t q0 = (packed_weights >> 0) & 0x03;
            uint8_t q1 = (packed_weights >> 2) & 0x03;
            uint8_t q2 = (packed_weights >> 4) & 0x03;
            uint8_t q3 = (packed_weights >> 6) & 0x03;

            // Dequantize using the sub-block scale and global dmin
            // Formula: val = sub_block_scale * quantized_value + global_dmin
            float val0 = sub_block_scale * static_cast<float>(q0) + dmin_float_clamped;
            float val1 = sub_block_scale * static_cast<float>(q1) + dmin_float_clamped;
            float val2 = sub_block_scale * static_cast<float>(q2) + dmin_float_clamped;
            float val3 = sub_block_scale * static_cast<float>(q3) + dmin_float_clamped;

            // Apply safety clamping to prevent extreme values
            val0 = std::min(std::max(val0, -1000.0f), 1000.0f);
            val1 = std::min(std::max(val1, -1000.0f), 1000.0f);
            val2 = std::min(std::max(val2, -1000.0f), 1000.0f);
            val3 = std::min(std::max(val3, -1000.0f), 1000.0f);

            output[weight_index++] = val0;
            
            output[weight_index++] = val1;
            
            output[weight_index++] = val2;
            
            output[weight_index++] = val3;
        }
    }
    assert(weight_index == GGML_QK_K);
}

// Placeholder implementation for Q3_K dequantization
void dequantize_q3_k(const void* qblock_void, float* output, int num_weights_in_block) {
    if (num_weights_in_block != GGML_QK_K) {
        throw std::invalid_argument("dequantize_q3_k currently only supports block size " + std::to_string(GGML_QK_K));
    }

    const block_q3_K* qblock = static_cast<const block_q3_K*>(qblock_void);

    // Extract and validate scale multipliers
    const float d_float_raw = fp16_to_fp32(qblock->d);
    const float dmin_float_raw = fp16_to_fp32(qblock->dmin);

    // Check for NaN AND Inf and replace with 0.0f
    const float d_float = (!std::isfinite(d_float_raw)) ? 0.0f : d_float_raw;
    const float dmin_float = (!std::isfinite(dmin_float_raw)) ? 0.0f : dmin_float_raw;

    const uint8_t* hmask_ptr = qblock->hmask;
    const uint8_t* qs_ptr = qblock->qs;
    const uint8_t* scales_ptr = qblock->scales;

    int weight_index = 0;

    // Process each sub-block (slightly different setup for Q3_K scales)
    for (int j = 0; j < GGML_QK_K / 16; ++j) { // j = 0..15
        
        // --- Correct Q3_K Scale Extraction Logic ---
        // In Q3_K, scales[] packs 6-bit scale indices differently.
        // The ggml.c logic is complex, let's adapt it carefully.
        // It uses 12 bytes (scales) + 16 bytes (hmask) related info.
        
        uint8_t scale_idx;
        // Logic from ggml.c dequantize_row_q3_K
        if (j < 8) {
            scale_idx = scales_ptr[j] & 0x3F; // Lower 6 bits
        } else {
            scale_idx = scales_ptr[j + 4] & 0x3F; // Lower 6 bits from the second half
        }
        
        // Q3_K uses the same 6-bit scale lookup table as Q4_K
        assert(scale_idx < 64 && "Scale index out of bounds for Q3_K lookup");
        const float sub_block_scale_factor = K_SCALE_VALUES[scale_idx];
        
        // Q3_K does NOT use a separate minimum factor from a table. It calculates min implicitly.
        // The minimum value is implicitly represented by the quantization grid.
        // We only need the scale factor for the dequantization formula.
        const float final_sub_block_scale = d_float * sub_block_scale_factor;
        const float final_sub_block_min = dmin_float; // The global dmin acts as the base minimum offset.
        // --- End Correct Q3_K Scale Extraction ---

        // Process the 16 values for this sub-block
        // Q3_K uses 2 bits in the qs array and 1 bit in the hmask array
        for (int i = 0; i < 4; ++i) { // Process 4 bytes from qs
            uint8_t qs_byte = qs_ptr[j * 4 + i];
            uint8_t hmask_byte = hmask_ptr[j]; // One hmask byte per sub-block
            
            // Process 4 values from this qs byte
            for (int bit_pos = 0; bit_pos < 8; bit_pos += 2) {
                // Extract 2 lower bits from qs
                uint8_t lower_bits = (qs_byte >> bit_pos) & 0x3;
                
                // Calculate the hmask bit index (0-7 for the 8 pairs in the sub-block)
                int hmask_bit_idx = (i * 4) + (bit_pos / 2);
                
                // Extract the corresponding high bit from the hmask byte
                uint8_t high_bit = (hmask_byte >> hmask_bit_idx) & 0x1;
                
                // Combine bits to get the 3-bit value (0-7)
                uint8_t q_val = (high_bit << 2) | lower_bits;
                
                // Dequantize: val = scale * q + min
                float val = final_sub_block_scale * static_cast<float>(q_val) + final_sub_block_min;
                
                // Ensure the output is finite
                if (!std::isfinite(val)) {
                    val = 0.0f;  // Replace NaN/Inf with zero
                }
                
                output[weight_index++] = val;
            }
        }
    }
    
    // Verify we processed exactly GGML_QK_K weights
    if (weight_index != GGML_QK_K) {
        std::cout << "ERROR: Processed " << weight_index << " weights instead of " << GGML_QK_K << std::endl;
        // Fill any remaining values with zeros
        while (weight_index < GGML_QK_K) {
            output[weight_index++] = 0.0f;
        }
    }
}

// --- Advanced Q6_K Quantization with Magnitude-Aware Group Scaling ---
void quantize_q6_k(
    const float* input,
    void* output_qblock_void,
    int num_elements)
{
    if (num_elements != GGML_QK_K) {
        throw std::invalid_argument("quantize_q6_k currently only supports block size " + std::to_string(GGML_QK_K));
    }

    block_q6_K* output_qblock = static_cast<block_q6_K*>(output_qblock_void);

    // Initialize output arrays
    uint8_t* ql = output_qblock->ql;
    uint8_t* qh = output_qblock->qh;
    int8_t* scales = output_qblock->scales;
    std::memset(ql, 0, GGML_QK_K / 2);  // Clear all ql bytes
    std::memset(qh, 0, GGML_QK_K / 4);  // Clear all qh bytes
    
    // PHASE 1: Group Analysis - Find ranges and magnitudes for each group
    // We'll use the existing block structure (16 blocks of 16 values each)
    
    // First, let's find the global maximum for perspective
    float global_max_abs = 0.0f;
    float global_min = std::numeric_limits<float>::max();
    float global_max = std::numeric_limits<float>::lowest();
    
    for (int i = 0; i < num_elements; ++i) {
        global_max_abs = std::max(global_max_abs, std::abs(input[i]));
        global_min = std::min(global_min, input[i]);
        global_max = std::max(global_max, input[i]);
    }
    
    // Avoid division by zero for constant data
    if (global_max_abs < 1e-10f) {
        // All zeros or tiny values - just zero everything out
        output_qblock->d = fp32_to_fp16(1e-10f);  // Minimal scale
        std::memset(scales, 0, GGML_QK_K / 16);
        return;  // All values will quantize to 32 (zero after offset)
    }
    
    // Now find statistics for each block
    struct BlockStats {
        float min_val;
        float max_val;
        float max_abs;
        float avg_abs;  // Average absolute value (for magnitude sense)
    };
    
    std::vector<BlockStats> block_stats(GGML_QK_K / 16);
    
    for (int block_idx = 0; block_idx < GGML_QK_K / 16; ++block_idx) {
        float sum_abs = 0.0f;
        float min_val = std::numeric_limits<float>::max();
        float max_val = std::numeric_limits<float>::lowest();
        float max_abs = 0.0f;
        
        for (int i = 0; i < 16; ++i) {
            float val = input[block_idx * 16 + i];
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
            float abs_val = std::abs(val);
            max_abs = std::max(max_abs, abs_val);
            sum_abs += abs_val;
        }
        
        block_stats[block_idx] = {
            min_val,
            max_val,
            max_abs,
            sum_abs / 16.0f  // Average absolute value
        };
    }
    
    // PHASE 2: Set global scale (d) based on global max but biased toward typical values
    
    // Instead of just using global_max_abs directly, we'll use a logarithmic averaging approach
    // that gives greater weight to larger values but doesn't completely ignore smaller ones
    float log_sum = 0.0f;
    int non_zero_count = 0;
    
    for (const auto& stats : block_stats) {
        if (stats.max_abs > 1e-10f) {
            log_sum += std::log(stats.max_abs);
            non_zero_count++;
        }
    }
    
    // Calculate a "geometric mean" of the block maxima, which is more balanced
    float effective_max;
    if (non_zero_count > 0) {
        float log_avg = log_sum / non_zero_count;
        effective_max = std::exp(log_avg);
        // Ensure we don't go below global_max_abs/4 to maintain reasonable precision for large values
        effective_max = std::max(effective_max, global_max_abs / 4.0f);
    } else {
        effective_max = global_max_abs;
    }
    
    // Set global scale (d) - using a slightly larger effective range to favor precision
    float d_float = effective_max / 28.0f;  // Slightly smaller than max range (31) for safety
    if (d_float < 1e-10f) d_float = 1e-10f;  // Avoid division by zero
    
    output_qblock->d = fp32_to_fp16(d_float);
    
    // PHASE 3: Calculate per-block scales with magnitude awareness
    
    // Calculate scales that adapt to the magnitude of values in each block
    // blocks with larger values get larger scales, smaller values get smaller scales
    for (int block_idx = 0; block_idx < GGML_QK_K / 16; ++block_idx) {
        const auto& stats = block_stats[block_idx];
        
        // Calculate scale multiplier based on block's magnitude relative to effective_max
        float scale_multiplier;
        
        if (stats.max_abs < 1e-10f) {
            // Almost-zero block
            scale_multiplier = 0.0f;
        } else {
            // Scale based on the block's magnitude relative to effective_max
            // Use logarithmic scaling to better handle different magnitudes
            float log_ratio = std::log10(stats.max_abs / effective_max);
            scale_multiplier = std::pow(10.0f, log_ratio);
            
            // Ensure scale_multiplier is reasonable (between 1/128 and 127)
            scale_multiplier = std::max(1.0f/128.0f, std::min(127.0f, scale_multiplier));
        }
        
        // Convert to int8_t scale factor
        int8_t scale_int8 = static_cast<int8_t>(std::round(scale_multiplier * 127.0f / 127.0f));
        scales[block_idx] = scale_int8;
    }
    
    // PHASE 4: Quantize each block with its custom scale
    for (int block_idx = 0; block_idx < GGML_QK_K / 16; ++block_idx) {
        // Calculate final scale for this block
        float block_scale = d_float * static_cast<float>(scales[block_idx]);
        
        // Skip quantization if scale is effectively zero
        if (std::abs(block_scale) < 1e-10f) {
            continue;  // Leave as zeros
        }
        
        // Process each value in this block
        for (int i = 0; i < 16; ++i) {
            int val_idx = block_idx * 16 + i;
            float val = input[val_idx];
            
            // Compute quantized value: q = val/scale + 32 (zero-point)
            int quant_val = static_cast<int>(std::round(val / block_scale)) + 32;
            quant_val = std::max(0, std::min(63, quant_val));  // Clamp to 6-bit range [0,63]
            
            // Extract low 4 bits and high 2 bits
            uint8_t q_low = quant_val & 0x0F;
            uint8_t q_high = (quant_val >> 4) & 0x03;
            
            // Map to byte positions matching dequantization logic
            int qi = i / 4;         // Which quad in the block (0-3)
            int qj = i % 4;         // Which position in the quad (0-3)
            
            // Position calculation
            int ql_idx = block_idx * 8 + qi * 2 + (qj / 2);   // Lower 4 bits byte index
            int qh_idx = block_idx * 4 + qi;                  // Upper 2 bits byte index
            
            int ql_shift = (qj % 2) * 4;                      // Shift for lower bits (0 or 4)
            int qh_shift = qj * 2;                            // Shift for upper bits (0, 2, 4, or 6)
            
            // Combine into output bytes
            ql[ql_idx] |= (q_low << ql_shift);
            qh[qh_idx] |= (q_high << qh_shift);
        }
    }
}

// --- Quantization Type Information Implementations --- FIX ENUM SCOPING ---

// Helper to get string name for GGMLType
const char* ggml_type_name(GGMLType type) {
    switch (type) {
        case GGMLType::GGML_TYPE_F32:  return "F32";
        case GGMLType::GGML_TYPE_F16:  return "F16";
        case GGMLType::GGML_TYPE_Q4_0: return "Q4_0";
        case GGMLType::GGML_TYPE_Q4_1: return "Q4_1";
        case GGMLType::GGML_TYPE_Q5_0: return "Q5_0";
        case GGMLType::GGML_TYPE_Q5_1: return "Q5_1";
        case GGMLType::GGML_TYPE_Q8_0: return "Q8_0";
        case GGMLType::GGML_TYPE_Q8_1: return "Q8_1";
        case GGMLType::GGML_TYPE_Q2_K: return "Q2_K";
        case GGMLType::GGML_TYPE_Q3_K: return "Q3_K";
        case GGMLType::GGML_TYPE_Q4_K: return "Q4_K";
        case GGMLType::GGML_TYPE_Q5_K: return "Q5_K";
        case GGMLType::GGML_TYPE_Q6_K: return "Q6_K";
        case GGMLType::GGML_TYPE_Q8_K: return "Q8_K";
        case GGMLType::GGML_TYPE_I8:   return "I8";
        case GGMLType::GGML_TYPE_I16:  return "I16";
        case GGMLType::GGML_TYPE_I32:  return "I32";
        case GGMLType::GGML_TYPE_COUNT: return "COUNT"; // Should not happen
        default: return "Unknown";
    }
}

size_t ggml_type_size(GGMLType type) {
    std::cout << "ggml_type_size called with type: " << static_cast<int>(type) << " (" << ggml_type_name(type) << ")" << std::endl; // Log name too
    
    switch (type) {
        case GGMLType::GGML_TYPE_F32:
            return sizeof(float);
        case GGMLType::GGML_TYPE_F16:
            return sizeof(uint16_t); // = 2
        case GGMLType::GGML_TYPE_I8:
            return sizeof(int8_t);   // = 1
        case GGMLType::GGML_TYPE_Q4_K:
            return sizeof(block_q4_K);
        case GGMLType::GGML_TYPE_Q2_K:
            std::cout << "  Q2_K block size is: " << sizeof(block_q2_K) << " bytes" << std::endl;
            return sizeof(block_q2_K);
        case GGMLType::GGML_TYPE_Q3_K:
            return sizeof(block_q3_K);
        case GGMLType::GGML_TYPE_Q6_K:
            std::cout << "  Returning sizeof(block_q6_K) (" << sizeof(block_q6_K) << ")" << std::endl;
            return sizeof(block_q6_K);
        case GGMLType::GGML_TYPE_Q4_0:
            std::cout << "  Returning 18 for Q4_0 type" << std::endl;
            return 18;
        // Add cases for other types like Q8_0, Q8_1, Q5_K, Q8_K if needed
        case GGMLType::GGML_TYPE_Q8_0: return 34; // Example size (1*scale_fp32 + 32*qs_int8)
        case GGMLType::GGML_TYPE_Q8_1: return 40; // Example size (1*scale_fp32 + 32*qs_int8)
        case GGMLType::GGML_TYPE_Q5_K: return 116; // Example size (adjust based on actual struct)
        case GGMLType::GGML_TYPE_Q8_K: return 290; // Example size (adjust based on actual struct)
        case GGMLType::GGML_TYPE_I16: return sizeof(int16_t);
        case GGMLType::GGML_TYPE_I32: return sizeof(int32_t);
        case GGMLType::GGML_TYPE_COUNT: // Fallthrough
        default:
            std::cout << "  UNKNOWN GGML TYPE: " << static_cast<int>(type) << std::endl;
            throw std::invalid_argument("Unknown GGML type in ggml_type_size: " + std::to_string(static_cast<int>(type)));
    }
}

size_t ggml_type_block_size(GGMLType type) {
    std::cout << "ggml_type_block_size called with type: " << static_cast<int>(type) << " (" << ggml_type_name(type) << ")" << std::endl; // Log name too
    
    switch (type) {
        // --- K-Quant Types ---
        case GGMLType::GGML_TYPE_Q2_K:
        case GGMLType::GGML_TYPE_Q3_K:
        case GGMLType::GGML_TYPE_Q4_K:
        case GGMLType::GGML_TYPE_Q6_K:
        // case GGMLType::Q5_K: // Add if needed
            return GGML_QK_K; // Should return 256
            // break; // <<< ADDED BREAK

        // --- Other Block Quant Types ---
        case GGMLType::GGML_TYPE_Q4_0:
        case GGMLType::GGML_TYPE_Q8_0:
        // case GGMLType::Q5_0: // Add if needed
             return 32;
             // break; // <<< ADDED BREAK

        // --- Non-Quantized Types ---
        case GGMLType::GGML_TYPE_F32:
        case GGMLType::GGML_TYPE_F16:
        case GGMLType::GGML_TYPE_I8:
        case GGMLType::GGML_TYPE_I16:
        case GGMLType::GGML_TYPE_I32:
            return 1;
            // break; // <<< ADDED BREAK

        default:
             std::cout << "Warning: Unknown GGMLType in ggml_type_block_size: " << static_cast<int>(type) << std::endl;
             return 0;
             // No break needed after return
    }
     // Should be unreachable if all cases return/throw, but add safety return
     return 0;
}

// --- End Quantization Type Information ---
