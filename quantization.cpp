#include "quantization.h"
#include "logger.h"
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
#include <sstream>   // For std::stringstream

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

// --- Global log counter for vec_dot_q4_k_q8_k_cpu logging ---
static std::atomic<int> g_vec_dot_q4_k_q8_k_log_count{0};

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

}

// --- START: Correct helper (Based on llama.cpp k_quants.c) ---
static inline void get_scale_min_indices_q4_K(
    int j,                   // Sub-block index (0-15)
    const uint8_t* scales,   // Pointer to the 12-byte scales field in block_q4_K
    uint8_t* scale_index,    // Output: 4-bit scale index (0-15)
    uint8_t* min_index       // Output: 4-bit min index (0-15)
    ) {
        assert(j >= 0 && j < 16);

    // llama.cpp logic:
    // Scales are in bytes 0-7, Mins are in bytes 8-11
    // Each byte contains two 4-bit indices.

    // Scale Index Extraction:
    // Byte index cycles through 0..7. Nibble selected by j / 8 (0 or 1).
    *scale_index = scales[j % 8] >> (4 * (j / 8));
    *scale_index &= 0x0F; // Ensure only 4 bits are kept

    // Min Index Extraction:
    // Byte index cycles through 8..11 (j % 4 + 8). Nibble selected by j / 4 (0, 1, 2, or 3).
    *min_index = scales[j % 4 + 8] >> (4 * (j / 4));
    *min_index &= 0x0F; // Ensure only 4 bits are kept
}
// --- END: Correct helper ---

/* --- OLD INCORRECT HELPER --- 
static inline void get_scale_min_indices_q4_K(
    int j,                   // Sub-block index (0-15)
    const uint8_t* scales,   // Pointer to the 12-byte scales field in block_q4_K
    uint8_t* scale_index,    // Output: 4-bit scale index (0-15)
    uint8_t* min_index       // Output: 4-bit min index (0-15)
) {
    assert(j >= 0 && j < 16);
    int scale_byte_index = j / 2; // Index into scales bytes 0-5 // <<< WRONG LOGIC
    int min_byte_index = j / 2 + 6; // Index into scales bytes 6-11 // <<< WRONG LOGIC

    if (j % 2 == 0) { // Even sub-block uses lower nibble
        *scale_index = scales[scale_byte_index] & 0x0F;
        *min_index   = scales[min_byte_index] & 0x0F;
    } else {          // Odd sub-block uses upper nibble
        *scale_index = scales[scale_byte_index] >> 4;
        *min_index   = scales[min_byte_index] >> 4;
    }
}
*/


// Dequantize a Q4_K block (256 weights)
// Input: qblock - pointer to the block_q4_K struct
// Output: output - buffer to write 256 dequantized float values

// Dequantize Q4_K data - Rewritten to match llama.cpp loop structure
void dequantize_q4_k_m(
    const block_q4_K* qblock,
    float* output, // Output buffer for GGML_QK_K floats
    int num_weights_in_block, // Should always be GGML_QK_K for full blocks
    bool log_this_block // Flag for enabling detailed logs
) {
    if (num_weights_in_block != GGML_QK_K) {
        std::cout << "Warning: dequantize_q4_k_m called with num_weights != GGML_QK_K (" << num_weights_in_block << ")" << std::endl;
        std::memset(output, 0, num_weights_in_block * sizeof(float));
        return;
    }

    const uint8_t* scales_ptr = qblock->scales;
    const uint8_t* qs_ptr = qblock->qs;

    // Convert d and dmin, clamp if necessary
    const float d_raw = fp16_to_fp32(qblock->d);
    const float dmin_raw = fp16_to_fp32(qblock->dmin);
    float d_super = d_raw;
    float min_super = dmin_raw;
    // --- REMOVED Clamping --- 
    // Optional: Add only finiteness check if needed, but clamping is wrong.
    // if (!std::isfinite(d_super)) { /* Handle NaN/inf, e.g., log warning or set to 0 */ d_super = 0.0f; }
    // if (!std::isfinite(min_super)) { /* Handle NaN/inf */ min_super = 0.0f; }
    // --- END REMOVED --- 

    // --- ADDED: Logging ---
    static std::atomic<int> log_count_q4 = 0;
    if (log_this_block && log_count_q4 < 5) { // Log first 5 blocks only
        // log_count_q4++; // Moved increment to end to ensure #1 is logged
        std::stringstream ss_log; // Use stringstream for logger
        ss_log << "[DEQUANT_Q4K] Block #" << (log_count_q4.load() + 1) << ":";
        Logger::debug(ss_log.str()); ss_log.str(""); // Log and clear

        ss_log << "  Super Scale (d_super): " << d_super << " (Raw FP16: 0x" << std::hex << qblock->d << std::dec << ")";
        Logger::debug(ss_log.str()); ss_log.str("");

        ss_log << "  Super Min (min_super): " << min_super << " (Raw FP16: 0x" << std::hex << qblock->dmin << std::dec << ")";
        Logger::debug(ss_log.str()); ss_log.str("");

        ss_log << "  Scales Field (12 bytes): ";
        for(int i=0; i<12; ++i) ss_log << "0x" << std::hex << (int)scales_ptr[i] << " ";
        ss_log << std::dec;
        Logger::debug(ss_log.str()); ss_log.str("");
        
        // --- Log details for the FIRST sub-block (j=0) --- 
        uint8_t first_scale_idx, first_min_idx;
        get_scale_min_indices_q4_K(0, scales_ptr, &first_scale_idx, &first_min_idx);
        const float first_scale = d_super * K_SCALE_VALUES[first_scale_idx];
        const float first_minv = min_super * K_MIN_VALUES[first_min_idx];
        ss_log << "  Sub-block 0: scale_idx=" << (int)first_scale_idx << ", min_idx=" << (int)first_min_idx 
                  << ", scale=" << first_scale << ", minv=" << first_minv;
        Logger::debug(ss_log.str()); ss_log.str("");

        ss_log << "  Sub-block 0 QS (8 bytes): ";
        const uint8_t* first_qs_sub_block = qs_ptr; // j=0
        for(int i=0; i<8; ++i) ss_log << "0x" << std::hex << (int)first_qs_sub_block[i] << " ";
        ss_log << std::dec;
        Logger::debug(ss_log.str()); ss_log.str("");
        // --- End sub-block 0 details ---
    }
    // --- END ADDED: Logging ---

    // Process 16 sub-blocks
    for (int j = 0; j < GGML_QK_K / 16; ++j) { // j = 0..15 (sub-block index)
        uint8_t scale_idx, min_idx;
        get_scale_min_indices_q4_K(j, scales_ptr, &scale_idx, &min_idx);

        // Ensure indices are within bounds for the lookup tables (0-15 ideally, but tables are 64)
        assert(scale_idx < 64 && "Scale index out of bounds");
        assert(min_idx < 64 && "Min index out of bounds");

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

            // FIXED: Adjust for zero-point (8) in the dequantization formula
            // This correctly maps q=0 to minv-8*scale, q=8 to minv, and q=15 to minv+7*scale
            // Original incorrect formula: output = scale * q + minv
            output_sub_block[l]     = scale * (static_cast<float>(nibble_low) - 8.0f) + minv;
            output_sub_block[l + 8] = scale * (static_cast<float>(nibble_high) - 8.0f) + minv;
        }
    }

    // --- ADDED: Logging Output ---
    if (log_this_block && log_count_q4 < 5) {
        std::stringstream ss_out_log; // Use stringstream
        ss_out_log << "  Output FP32 (first 16/256 from Sub-block 0): ";
        for(int i=0; i<16; ++i) ss_out_log << output[i] << " ";
        ss_out_log << "...";
        Logger::debug(ss_out_log.str()); // Log output
        log_count_q4++; // Increment counter AFTER logging is complete
    }
    // --- END ADDED: Logging ---
}

// --- Dequantization for Q6_K ---
void dequantize_q6_k(
    const block_q6_K* qblock,
    float* output, // Changed name to match llama.cpp style
    int num_weights_in_block, // Still useful for assert
    bool log_this_block // ADDED: Log flag
) {
    if (num_weights_in_block != GGML_QK_K) {
         std::cout << "Warning: dequantize_q6_k called with num_weights != GGML_QK_K (" << num_weights_in_block << ")" << std::endl;
         std::memset(output, 0, num_weights_in_block * sizeof(float));
         return;
    }

    const float d = fp16_to_fp32(qblock->d); // Super scale

    const uint8_t* __restrict ql = qblock->ql;
    const uint8_t* __restrict qh = qblock->qh;
    const int8_t* __restrict scales = qblock->scales; // 16 x int8 scales

    // --- ADDED: Logging ---
    static std::atomic<int> log_count_q6 = 0;
    if (log_this_block && log_count_q6 < 5) { // Log first 5 blocks only
        // log_count_q6++; // Moved increment
        std::stringstream ss_log; // Use stringstream for logger
        ss_log << "[DEQUANT_Q6K] Block #" << (log_count_q6.load() + 1) << ":";
        Logger::debug(ss_log.str()); ss_log.str(""); // Log and clear

        ss_log << "  Super Scale (d): " << d << " (Raw FP16: 0x" << std::hex << qblock->d << std::dec << ")";
        Logger::debug(ss_log.str()); ss_log.str("");

        ss_log << "  Block Scales (16 bytes, int8): ";
        for(int i=0; i<16; ++i) ss_log << (int)scales[i] << " ";
        Logger::debug(ss_log.str()); ss_log.str("");
        
        // --- Log details for the FIRST sub-block (i=0) ---
        const float first_sub_block_scale_factor = static_cast<float>(scales[0]);
        ss_log << "  Sub-block 0: block_scale_factor=" << first_sub_block_scale_factor;
        Logger::debug(ss_log.str()); ss_log.str("");

        ss_log << "  Sub-block 0 QL (first 16 bytes): "; // QL for the whole block relevant for sub-block 0
        const uint8_t* first_ql_sub_block = ql; // i=0
        for(int k=0; k<16; ++k) ss_log << "0x" << std::hex << (int)first_ql_sub_block[k] << " ";
        ss_log << std::dec;
        Logger::debug(ss_log.str()); ss_log.str("");

        ss_log << "  Sub-block 0 QH (first 4 bytes): "; // QH for the whole block relevant for sub-block 0
        const uint8_t* first_qh_sub_block = qh; // i=0
        for(int k=0; k<4; ++k) ss_log << "0x" << std::hex << (int)first_qh_sub_block[k] << " ";
        ss_log << std::dec;
        Logger::debug(ss_log.str()); ss_log.str("");
        // --- End sub-block 0 details ---
    }
    // --- END ADDED: Logging ---


    // Process 16 sub-blocks (16 values each)
    for (int i = 0; i < GGML_QK_K / 16; ++i) { // i = 0..15 (sub-block index)
        const float scale_factor = static_cast<float>(scales[i]); // Get the int8 scale factor for this sub-block
        const float sub_block_scale = d * scale_factor; // Actual scale for the sub-block
        float* __restrict y = output + i * 16;          // Output pointer for this sub-block

        // Pointer to the start of the 16 bytes of ql for this sub-block
        const uint8_t* ql_sub = ql + i * 16;
        // Pointer to the start of the 4 bytes of qh for this sub-block
        const uint8_t* qh_sub = qh + i * 4;

        // Dequantize 16 values for this sub-block
        for (int l = 0; l < 16; ++l) {
            // Lower 4 bits from ql
            uint8_t q_low = ql_sub[l] & 0x0F;

            // Higher 2 bits from qh (packed - 4 sets of 2 bits per byte)
            int byte_idx_qh = l / 4;        // Which byte in qh (0..3 for this sub-block)
            int shift_qh = (l % 4) * 2;   // Which 2 bits (0, 2, 4, 6)
            uint8_t q_high = (qh_sub[byte_idx_qh] >> shift_qh) & 0x03;

            // Combine to form 6-bit quantized value [0, 63]
            int quantized_val = q_low | (q_high << 4);

            // Dequantize: Multiply by super-scale 'd' and block-scale-factor 'scale_factor'.
            // IMPORTANT: The dequantization formula for Q6_K according to ggml.c is:
            // val = d * scale_factor * (quantized_val - 32)
            // We need to subtract the zero-point 32 AFTER getting the 6-bit value.
            float val = sub_block_scale * (static_cast<float>(quantized_val) - 32.0f);
            y[l] = val;
        }
    }
    // --- ADDED: Logging Output ---
    if (log_this_block && log_count_q6 < 5) {
        std::stringstream ss_out_log; // Use stringstream
        ss_out_log << "  Output FP32 (first 16/256 from Sub-block 0): ";
        for(int i=0; i<16; ++i) ss_out_log << output[i] << " ";
        ss_out_log << "...";
        Logger::debug(ss_out_log.str()); // Log output
        log_count_q6++; // Increment counter AFTER logging complete
    }
    // --- END ADDED: Logging ---
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
    
    // Initialize scales and qs to zero
    std::memset(output_qblock->scales, 0, sizeof(output_qblock->scales));
    std::memset(output_qblock->qs, 0, sizeof(output_qblock->qs));

    // Step 1: Find min/max for the entire block to set super-block scale (d) and min (dmin)
    float block_min_val = std::numeric_limits<float>::max();
    float block_max_val = std::numeric_limits<float>::lowest();
    for (int i = 0; i < num_elements; ++i) {
        block_min_val = std::min(block_min_val, input[i]);
        block_max_val = std::max(block_max_val, input[i]);
    }

    // Handle edge case of constant values for super-block scale
    if (block_max_val == block_min_val) {
        block_max_val = block_min_val + 1e-6f; // Add a tiny offset
    }
    if (block_max_val < 1e-10f && block_max_val > -1e-10f) {
        // Handle case where the block is effectively zero
        block_max_val = 1e-6f;
        block_min_val = 0.0f;
    }

    // Calculate super-block scale and min
    // Super-scale should ideally map the *absolute* range to the scale table factors (1.0 to 15.5)
    // Super-min should ideally map the *actual min* to the min table factors (0.0 to -1.8125)
    const float d_super_scale_candidate = (block_max_val - block_min_val) / 15.0f; // A potential scale measure
    const float d_super = d_super_scale_candidate > 1e-10f ? d_super_scale_candidate : 1e-10f;
    const float min_super = block_min_val; // Super-min is the block minimum

    // Convert to FP16 using the proper function
    output_qblock->d = fp32_to_fp16(d_super);
    output_qblock->dmin = fp32_to_fp16(min_super);

    // Step 2 & 3: Process each of the 16 sub-blocks
    for (int j = 0; j < GGML_QK_K / 16; ++j) { // j = 0..15
        const float* sub_block_input = input + j * 16;

        // a) Find local min/max for the sub-block
        float sub_min_val = sub_block_input[0];
        float sub_max_val = sub_block_input[0];
        for (int i = 1; i < 16; ++i) {
            sub_min_val = std::min(sub_min_val, sub_block_input[i]);
            sub_max_val = std::max(sub_max_val, sub_block_input[i]);
        }

        // b) Calculate ideal local scale and min
        float ideal_scale = 0.0f;
        if (sub_max_val > sub_min_val + 1e-10f) { // Avoid division by zero if sub-block is constant
            ideal_scale = (sub_max_val - sub_min_val) / 15.0f;
        }
        float ideal_min = sub_min_val;

        // c) Find best scale_idx (0-15)
        uint8_t best_scale_idx = 0;
        float min_scale_err = std::numeric_limits<float>::max();
        if (d_super > 1e-10f) { // Only search if super-scale is non-zero
            for (uint8_t k = 0; k < 16; ++k) {
                float candidate_scale = d_super * K_SCALE_VALUES[k];
                float err = std::abs(candidate_scale - ideal_scale);
                if (err < min_scale_err) {
                    min_scale_err = err;
                    best_scale_idx = k;
                }
            }
        }

        // d) Find best min_idx (0-15)
        uint8_t best_min_idx = 0;
        float min_min_err = std::numeric_limits<float>::max();
        // Note: K_MIN_VALUES are <= 0. min_super can be positive or negative.
        // We want min_super * K_MIN_VALUES[l] to be close to ideal_min.
        for (uint8_t l = 0; l < 16; ++l) {
            float candidate_min = min_super * K_MIN_VALUES[l];
            float err = std::abs(candidate_min - ideal_min);
            if (err < min_min_err) {
                min_min_err = err;
                best_min_idx = l;
            }
        }

        // e) Pack the chosen indices into output_qblock->scales
        // Inverse of get_scale_min_indices_q4_K
        int scale_byte_idx = j % 8;
        int scale_shift = 4 * (j / 8); // 0 for j=0..7, 4 for j=8..15
        output_qblock->scales[scale_byte_idx] |= (best_scale_idx << scale_shift);

        int min_byte_idx = (j % 4) + 8;
        int min_shift = 4 * (j / 4); // 0, 4, 8, 12
        output_qblock->scales[min_byte_idx] |= (best_min_idx << min_shift);

        // f) Quantize the 16 values in the sub-block using the chosen scales/mins
        float actual_scale = d_super * K_SCALE_VALUES[best_scale_idx];
        float actual_min = min_super * K_MIN_VALUES[best_min_idx];
        float inv_actual_scale = (actual_scale > 1e-10f || actual_scale < -1e-10f) ? (1.0f / actual_scale) : 0.0f;

        uint8_t packed_qs[8]; // Temporary storage for packed nibbles for this sub-block
        std::memset(packed_qs, 0, sizeof(packed_qs));

        for (int i = 0; i < 16; ++i) {
            float val = sub_block_input[i];

            // Quantize: q = round((val - actual_min) / actual_scale)
            int quant_val = 0;
            if (inv_actual_scale != 0.0f) { // Avoid division by zero
                quant_val = static_cast<int>(std::round((val - actual_min) * inv_actual_scale));
            }
            quant_val = std::max(0, std::min(15, quant_val)); // Clamp to [0, 15]

            // Pack into nibbles
            int byte_idx_qs = i / 2; // 0..7
            int shift_qs = (i % 2) * 4; // 0 or 4
            packed_qs[byte_idx_qs] |= (static_cast<uint8_t>(quant_val) << shift_qs);
        }

        // g) Copy the packed nibbles for this sub-block into the correct location in output_qblock->qs
        // Original packing logic had high/low swapped, corrected here:
        // Low nibble is index i, High nibble is index i+8
        uint8_t* qs_target = output_qblock->qs + j * 8;
        for(int i=0; i<8; ++i) {
            uint8_t low_nibble_val = packed_qs[i] & 0x0F;
            uint8_t high_nibble_val = (packed_qs[i] >> 4) & 0x0F;
            qs_target[i] = low_nibble_val | (high_nibble_val << 4); // Pack low first, then high shifted
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
// --- REPLACED with Simplified Q6_K Quantization ---
void quantize_q6_k(
    const float* input,
    void* output_qblock_void,
    int num_elements)
{
    if (num_elements != GGML_QK_K) {
        throw std::invalid_argument("quantize_q6_k currently only supports block size " + std::to_string(GGML_QK_K));
    }

    block_q6_K* output_qblock = static_cast<block_q6_K*>(output_qblock_void);

    uint8_t* ql = output_qblock->ql;
    uint8_t* qh = output_qblock->qh;
    int8_t* scales = output_qblock->scales;
    std::memset(ql, 0, GGML_QK_K / 2);
    std::memset(qh, 0, GGML_QK_K / 4);

    // Step 1: Find max absolute value for the block
    float amax = 0.0f;
    for (int i = 0; i < num_elements; ++i) {
        amax = std::max(amax, std::abs(input[i]));
    }

    // Step 2: Calculate global scale 'd'
    const float d_float = (amax > 1e-10f) ? (amax / 31.0f) : 1e-10f;
    output_qblock->d = fp32_to_fp16(d_float);

    // Step 3: Compute per-subblock scale and quantize
    for (int sub = 0; sub < GGML_QK_K / 16; ++sub) {
        const float* sub_in = input + sub * 16;
        // Find max abs value in subblock
        float sub_amax = 0.0f;
        for (int i = 0; i < 16; ++i) {
            sub_amax = std::max(sub_amax, std::abs(sub_in[i]));
        }
        // Compute subblock scale (int8_t)
        int8_t scale = (d_float > 0.0f) ? std::round(sub_amax / d_float) : 1;
        if (scale == 0) scale = 1;
        scales[sub] = scale;
        // Quantize and pack
        for (int i = 0; i < 16; ++i) {
            float val = sub_in[i];
            int q = static_cast<int>(std::round(val / (d_float * scale))) + 32;
            q = std::max(0, std::min(63, q));
            // Pack 6 bits: low 4 to ql, high 2 to qh
            int idx = sub * 16 + i;
            int ql_idx = idx / 2;
            int ql_shift = (idx % 2) * 4;
            ql[ql_idx] |= (q & 0x0F) << ql_shift;
            int qh_idx = idx / 4;
            int qh_shift = (idx % 4) * 2;
            qh[qh_idx] |= ((q >> 4) & 0x03) << qh_shift;
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
        case GGMLType::GGML_TYPE_BF16: return "BF16";
        case GGMLType::GGML_TYPE_COUNT: return "COUNT"; // Should not happen
        default: return "Unknown";
    }
}

size_t ggml_type_size(GGMLType type) {
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
            return sizeof(block_q2_K);
        case GGMLType::GGML_TYPE_Q3_K:
            return sizeof(block_q3_K);
        case GGMLType::GGML_TYPE_Q6_K:
            return sizeof(block_q6_K);
        case GGMLType::GGML_TYPE_Q4_0:
            return 18;
        // Add cases for other types like Q8_0, Q8_1, Q5_K, Q8_K if needed
        case GGMLType::GGML_TYPE_Q8_0: return 34; // Example size (1*scale_fp32 + 32*qs_int8)
        case GGMLType::GGML_TYPE_Q8_1: return 40; // Example size (1*scale_fp32 + 32*qs_int8)
        case GGMLType::GGML_TYPE_Q5_K: return 116; // Example size (adjust based on actual struct)
        case GGMLType::GGML_TYPE_Q8_K: return 290; // Example size (adjust based on actual struct)
        case GGMLType::GGML_TYPE_I16: return sizeof(int16_t);
        case GGMLType::GGML_TYPE_I32: return sizeof(int32_t);
        case GGMLType::GGML_TYPE_BF16: return sizeof(uint16_t); // <<< ADDED BF16 (size 2) >>>
        case GGMLType::GGML_TYPE_COUNT: // Fallthrough
        default:
            std::cout << "  UNKNOWN GGML TYPE: " << static_cast<int>(type) << std::endl;
            throw std::invalid_argument("Unknown GGML type in ggml_type_size: " + std::to_string(static_cast<int>(type)));
    }
}

size_t ggml_type_block_size(GGMLType type) {
    switch (type) {
        // --- K-Quant Types ---
        case GGMLType::GGML_TYPE_Q2_K:
        case GGMLType::GGML_TYPE_Q3_K:
        case GGMLType::GGML_TYPE_Q4_K:
        case GGMLType::GGML_TYPE_Q6_K:
        // case GGMLType::Q5_K: // Add if needed
            return GGML_QK_K; // Should return 256

        case GGMLType::GGML_TYPE_Q4_0:
        case GGMLType::GGML_TYPE_Q8_0:
        // case GGMLType::Q5_0: // Add if needed
             return 32;

        // --- Non-Quantized Types ---
        case GGMLType::GGML_TYPE_F32:
        case GGMLType::GGML_TYPE_F16:
        case GGMLType::GGML_TYPE_I8:
        case GGMLType::GGML_TYPE_I16:
        case GGMLType::GGML_TYPE_I32:
        case GGMLType::GGML_TYPE_BF16: // <<< ADDED BF16 >>>
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

// --- ADDED: Implementation for quantize_fp32_to_q8_K ---
std::vector<block_q8_K> quantize_fp32_to_q8_K(const std::vector<float>& f_data) {
    if (f_data.size() % GGML_QK_K != 0) {
        throw std::runtime_error("Input vector size must be a multiple of GGML_QK_K (" + std::to_string(GGML_QK_K) + ")");
    }

    size_t num_blocks = f_data.size() / GGML_QK_K;
    std::vector<block_q8_K> q_data(num_blocks);
    const float* x = f_data.data();
    block_q8_K* y = q_data.data();

    for (size_t i = 0; i < num_blocks; ++i) {
        // 1. Find max absolute value in the block
        float amax = 0.0f;
        for (int j = 0; j < GGML_QK_K; ++j) {
            amax = std::max(amax, std::abs(x[j]));
        }

        // 2. Calculate scale and inverse scale
        const float d = amax / 127.0f;
        const float id = (d != 0.f) ? 1.0f / d : 0.0f;
        y[i].d = fp32_to_fp16(d); // Store scale as FP16

        // 3. Quantize and calculate block sums
        int16_t block_sum[16] = {0}; // Accumulator for sums of 16 elements
        for (int j = 0; j < GGML_QK_K; ++j) {
            const float val_scaled = x[j] * id;
            // Round to nearest integer, clamp to [-128, 127]
            int8_t q_val = static_cast<int8_t>(std::max(-128.0f, std::min(127.0f, std::round(val_scaled))));
            y[i].qs[j] = q_val;
            block_sum[j / 16] += q_val; // Add to the sum for the corresponding 16-element block
        }
        
        // 4. Store block sums
        std::memcpy(y[i].bsums, block_sum, sizeof(block_sum));

        x += GGML_QK_K; // Move to the next block in the input
    }

    return q_data;
}

// --- ADDED: Implementation for vec_dot_q6_k_q8_k_cpu ---
// Translated from C implementation ggml_vec_dot_q6_K_q8_K (CPU path)
float vec_dot_q6_k_q8_k_cpu(
    int n, // Number of elements
    const std::vector<block_q6_K>& x_vec, // Q6_K vector (matrix row)
    const std::vector<block_q8_K>& y_vec,  // Q8_K vector (quantized activation)
    bool log_this_call // ADDED: Flag to enable logging
) {
    if (n % GGML_QK_K != 0) {
         throw std::runtime_error("vec_dot_q6_k_q8_k: n must be multiple of QK_K");
    }
    size_t nb = n / GGML_QK_K;
    if (x_vec.size() != nb || y_vec.size() != nb) {
         throw std::runtime_error("vec_dot_q6_k_q8_k: vector block count mismatch");
    }

    const block_q6_K* x = x_vec.data();
    const block_q8_K* y = y_vec.data();

    int8_t  aux8[GGML_QK_K];
    int16_t aux16[8];
    float   sums [8];
    int32_t aux32[8];
    std::memset(sums, 0, 8 * sizeof(float));

    float sumf = 0.0f;

    static std::atomic<int> log_count_dot = 0;
    bool should_log_this_block = log_this_call && log_count_dot < 5;

    for (size_t i = 0; i < nb; ++i) {
        const uint8_t * ql = x[i].ql;
        const uint8_t * qh = x[i].qh;
        const int8_t  * q8 = y[i].qs;
        std::memset(aux32, 0, 8 * sizeof(int32_t));

        int8_t * a = aux8;
        for (int j = 0; j < GGML_QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                a[l +  0] = static_cast<int8_t>(((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32);
                a[l + 32] = static_cast<int8_t>(((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32);
                a[l + 64] = static_cast<int8_t>(((ql[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32);
                a[l + 96] = static_cast<int8_t>(((ql[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32);
            }
            a  += 128;
            ql += 64;
            qh += 32;
        }

        a = aux8;
        int is = 0;
        for (int j = 0; j < GGML_QK_K / 16; ++j) {
            int scale = x[i].scales[is++];
            for (int l = 0; l < 8; ++l) aux16[l] = static_cast<int16_t>(q8[l]) * static_cast<int16_t>(a[l]);
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = static_cast<int16_t>(q8[l]) * static_cast<int16_t>(a[l]);
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
        }

        // --- Compensation term: sum(q8_bsums * q6_scales) ---
        int32_t sumi_mins = 0;
        for (int j = 0; j < GGML_QK_K / 16; ++j) {
            sumi_mins += static_cast<int32_t>(y[i].bsums[j]) * static_cast<int32_t>(x[i].scales[j]);
        }

        const float d_q6 = fp16_to_fp32(x[i].d);
        const float d_q8 = fp16_to_fp32(y[i].d);
        const float d = d_q6 * d_q8;

        float block_contribution = 0.0f;
        for (int l = 0; l < 8; ++l) {
            float term = d * (aux32[l] - 32 * sumi_mins / 8); // Distribute compensation across aux32
            sums[l] += term;
            block_contribution += term;
        }

        // --- Logging for first block ---
        if (i == 0 && should_log_this_block) {
            std::stringstream ss_log;
            ss_log << "[DOT_Q6K_Q8K] Call #" << (log_count_dot.load() + 1) << ", Block #0:";
            Logger::debug(ss_log.str()); ss_log.str("");
            ss_log << "  Q6_K Scale d_q6: " << d_q6 << " (Raw FP16: 0x" << std::hex << x[i].d << std::dec << ")";
            Logger::debug(ss_log.str()); ss_log.str("");
            ss_log << "  Q8_K Scale d_q8: " << d_q8;
            Logger::debug(ss_log.str()); ss_log.str("");
            ss_log << "  Combined Scale d: " << d;
            Logger::debug(ss_log.str()); ss_log.str("");
            ss_log << "  Q6_K Sub-scales (int8): ";
            for(int k=0; k<16; ++k) ss_log << (int)x[i].scales[k] << " ";
            Logger::debug(ss_log.str()); ss_log.str("");
            ss_log << "  Int32 Sums (aux32, before compensation): ";
            for(int l=0; l<8; ++l) ss_log << aux32[l] << " ";
            Logger::debug(ss_log.str()); ss_log.str("");
            ss_log << "  Compensation term (sumi_mins): " << sumi_mins << ", -32 * sumi_mins: " << (-32 * sumi_mins);
            Logger::debug(ss_log.str()); ss_log.str("");
            ss_log << "  Block #0 Contribution to Sums (after compensation): " << block_contribution;
            Logger::debug(ss_log.str());
        }
    }

    for (int l = 0; l < 8; ++l) {
        sumf += sums[l];
    }

    if (should_log_this_block) {
        log_count_dot++;
    }
    return sumf;
}

void matvec_q6k_q8k_cpu(
    const std::vector<block_q6_K>& mat_q6k,
    const std::vector<block_q8_K>& vec_q8k,
    std::vector<float>& out_f32,
    int rows,
    int cols,
    bool log_calls // ADDED: Flag to pass down
) {
    if (cols % GGML_QK_K != 0) {
        throw std::runtime_error("matvec_q6k_q8k_cpu: cols must be divisible by GGML_QK_K");
    }
    size_t blocks_per_row = cols / GGML_QK_K;
    if (mat_q6k.size() != (size_t)rows * blocks_per_row) {
        throw std::runtime_error("matvec_q6k_q8k_cpu: mat_q6k size mismatch");
    }
    if (vec_q8k.size() != blocks_per_row) {
        throw std::runtime_error("matvec_q6k_q8k_cpu: vec_q8k size mismatch");
    }
    out_f32.resize(rows);
    for (int r = 0; r < rows; ++r) {
        // Each row is blocks_per_row consecutive blocks
        const std::vector<block_q6_K> row_q6k(
            mat_q6k.begin() + r * blocks_per_row,
            mat_q6k.begin() + (r + 1) * blocks_per_row
        );
        // Pass the logging flag down to the dot product function
        out_f32[r] = vec_dot_q6_k_q8_k_cpu(cols, row_q6k, vec_q8k, log_calls);
    }
}
// --- END ADDED --- 

// --- ADDED: Implementation for vec_dot_q4_k_q8_k_cpu ---
// Based on the generic C path in ggml-quants.c
float vec_dot_q4_k_q8_k_cpu(
    int n,
    const std::vector<block_q4_K>& x_vec,
    const std::vector<block_q8_K>& y_vec,
    bool log_this_call // Optional logging flag (currently unused)
) {
    // Global log limit: only allow logging for the first 5 calls, ever
    int log_count_now = g_vec_dot_q4_k_q8_k_log_count.fetch_add(1);
    if (log_count_now >= 5) log_this_call = false;

    if (n % GGML_QK_K != 0) {
        throw std::runtime_error("vec_dot_q4_k_q8_k: n must be multiple of QK_K");
    }
    size_t nb = n / GGML_QK_K;
    if (x_vec.size() != nb || y_vec.size() != nb) {
        throw std::runtime_error("vec_dot_q4_k_q8_k: vector block count mismatch");
    }

    const block_q4_K* x = x_vec.data();
    const block_q8_K* y = y_vec.data();

    float sumf = 0.0f;
    for (size_t i = 0; i < nb; ++i) {
        // Unpack Q4_K values
        int8_t q4_vals[GGML_QK_K];
        const uint8_t* q4 = x[i].qs;
        for (int j = 0; j < GGML_QK_K / 2; ++j) {
            q4_vals[2 * j + 0] = static_cast<int8_t>(q4[j] & 0xF);
            q4_vals[2 * j + 1] = static_cast<int8_t>(q4[j] >> 4);
        }
        // Q8_K values
        const int8_t* q8 = y[i].qs;

        // For each sub-block (16 per block)
        for (int sub = 0; sub < 16; ++sub) {
            uint8_t scale_idx, min_idx;
            get_scale_min_indices_q4_K(sub, x[i].scales, &scale_idx, &min_idx);
            float scale = fp16_to_fp32(x[i].d) * K_SCALE_VALUES[scale_idx];
            float minv = fp16_to_fp32(x[i].dmin) * K_MIN_VALUES[min_idx];
            for (int k = 0; k < 16; ++k) {
                int idx = sub * 16 + k;
                float q4_val = static_cast<float>(q4_vals[idx]) - 8.0f;
                float q8_val = static_cast<float>(q8[idx]);
                sumf += (scale * q4_val + minv) * q8_val;
            }
        }
        // Optionally log the first block
        if (i == 0 && log_this_call) {
            std::stringstream ss;
            ss << "[Q4K_Q8K] Block #0: d: " << fp16_to_fp32(x[i].d) << ", dmin: " << fp16_to_fp32(x[i].dmin);
            Logger::debug(ss.str()); ss.str("");
            ss << "[Q4K_Q8K] Block #0: Q8_K input (first 16): ";
            for (int k = 0; k < 16; ++k) ss << (int)q8[k] << " ";
            Logger::debug(ss.str()); ss.str("");
            ss << "[Q4K_Q8K] Block #0: Q4_K unpacked (first 16): ";
            for (int k = 0; k < 16; ++k) ss << (int)q4_vals[k] << " ";
            Logger::debug(ss.str()); ss.str("");
        }
    }
    return sumf;
}

// --- ADDED: Implementation for matvec_q4k_q8k_cpu ---
void matvec_q4k_q8k_cpu(
    const std::vector<block_q4_K>& mat_q4k,
    const std::vector<block_q8_K>& vec_q8k,
    std::vector<float>& out_f32,
    int rows,
    int cols,
    bool log_calls // Optional logging flag (currently unused)
) {
    if (cols % GGML_QK_K != 0) {
        throw std::runtime_error("matvec_q4k_q8k_cpu: cols must be divisible by GGML_QK_K");
    }
    size_t blocks_per_row = cols / GGML_QK_K;
    if (mat_q4k.size() != (size_t)rows * blocks_per_row) {
        throw std::runtime_error("matvec_q4k_q8k_cpu: mat_q4k size mismatch");
    }
    if (vec_q8k.size() != blocks_per_row) {
        throw std::runtime_error("matvec_q4k_q8k_cpu: vec_q8k size mismatch");
    }
    out_f32.resize(rows);

    #pragma omp parallel for
    for (int r = 0; r < rows; ++r) {
        // Create temporary vectors for the row and pass to the dot function
        const std::vector<block_q4_K> row_q4k(
            mat_q4k.begin() + r * blocks_per_row,
            mat_q4k.begin() + (r + 1) * blocks_per_row
        );
        // log_calls parameter is available but currently unused in the dot function
        out_f32[r] = vec_dot_q4_k_q8_k_cpu(cols, row_q4k, vec_q8k, log_calls);
    }
}

// --- END ADDED --- 

// --- ADDED: Dequantize Q8_K blocks to FP32 ---
void dequantize_q8_k(const std::vector<block_q8_K>& q8k_vec, std::vector<float>& out_f32, int n) {
    if (q8k_vec.empty() || n == 0) return;
    if (out_f32.size() != (size_t)n) out_f32.resize(n);
    int num_blocks = n / GGML_QK_K;
    for (int b = 0; b < num_blocks; ++b) {
        float d = fp16_to_fp32(q8k_vec[b].d);
        for (int i = 0; i < GGML_QK_K; ++i) {
            out_f32[b * GGML_QK_K + i] = d * q8k_vec[b].qs[i];
        }
    }
}
