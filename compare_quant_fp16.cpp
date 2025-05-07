#include "gguf_parser.h"
#include "quantization.h"
#include "logger.h"
#include <cstring>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <iomanip> // For std::fixed, std::setprecision

// --- START: K-Quant Lookup Tables (Copied from quantization.cpp) ---
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

// --- START: Helper for scale/min indices (Copied from quantization.cpp) ---
static inline void get_scale_min_indices_q4_K_for_compare( // Renamed to avoid conflict if quantization.h is included directly
    int j,                   // Sub-block index (0-15)
    const uint8_t* scales,   // Pointer to the 12-byte scales field in block_q4_K
    uint8_t* scale_index,    // Output: 4-bit scale index (0-15)
    uint8_t* min_index       // Output: 4-bit min index (0-15)
    ) {
    assert(j >= 0 && j < 16);
    // llama.cpp logic for GGML Q4_K (scales are packed indices):
    // Scales are in bytes 0-7, Mins are in bytes 8-11
    // Each byte contains two 4-bit indices.
    *scale_index = scales[j % 8] >> (4 * (j / 8));
    *scale_index &= 0x0F; 
    *min_index = scales[j % 4 + 8] >> (4 * (j / 4));
    *min_index &= 0x0F;
}
// --- END: Helper for scale/min indices ---

// --- START: Legacy GGML-style Q4_K Dequantization ---
void dequantize_q4_k_m_legacy_ggml(
    const block_q4_K* qblock,
    float* output,
    int num_weights_in_block) {
    if (num_weights_in_block != GGML_QK_K) {
        std::cerr << "Warning: dequantize_q4_k_m_legacy_ggml with num_weights != GGML_QK_K" << std::endl;
        std::memset(output, 0, num_weights_in_block * sizeof(float));
        return;
    }

    const float d_super = fp16_to_fp32(qblock->d, false); // GGML d is not a GGUF scale field
    const float min_super = fp16_to_fp32(qblock->dmin, false); // GGML dmin is not a GGUF scale field
    const uint8_t* qs_ptr = qblock->qs;

    for (int sub_block_idx = 0; sub_block_idx < GGML_QK_K / 16; ++sub_block_idx) { // 16 sub-blocks
        uint8_t scale_idx, min_idx;
        get_scale_min_indices_q4_K_for_compare(sub_block_idx, qblock->scales, &scale_idx, &min_idx);
        
        // Ensure indices are within bounds for the lookup tables
        if (scale_idx >= 64 || min_idx >= 64) {
             // This should not happen with correct GGML Q4_K data where scales field contains packed 4-bit indices (0-15 for each part).
             // If K_SCALE_VALUES/K_MIN_VALUES are intended to be indexed by 0-15, then the table size or logic might be mismatched.
             // For GGML Q4_K, K_SCALE_VALUES/K_MIN_VALUES are indeed indexed by 0-15 (or rather, the 4-bit values directly).
             // The tables provided (K_SCALE_VALUES[64], K_MIN_VALUES[64]) are for the newer interpretation where scale_idx can be up to 63.
             // The old Q4_K block_q4_K stores 4-bit indices, so they are 0-15.
             // Let\'s assume the K_SCALE_VALUES and K_MIN_VALUES are to be indexed by these 0-15 values.
             // The provided tables seem to be for the full 6-bit range.
             // For strict old Q4_K, these indices (0-15) would be directly used if tables were smaller (size 16).
             // Given the current tables of size 64, we use the direct 4-bit indices.
             // This is a common confusion point: GGML Q4_K uses 4-bit indices (0-15) to select from 16-element scale/min tables.
             // The provided global K_SCALE_VALUES and K_MIN_VALUES are for a broader K-quant scheme (e.g. Q6_K uses 6-bit indices).
             // For true old Q4_K emulation, we should use the 4-bit index directly.
             // So, K_SCALE_VALUES[scale_idx] where scale_idx is 0-15.

            // For this legacy function, we use the indices directly assuming they are 0-15
            // and the tables K_SCALE_VALUES and K_MIN_VALUES are appropriately sized (or we use a 16-element subset).
            // Since the provided tables are K_SCALE_VALUES[64], it means they expect up to 6-bit indices.
            // The helper get_scale_min_indices_q4_K_for_compare correctly extracts 4-bit indices (0-15).
            // So, K_SCALE_VALUES[scale_idx] is correct.

            // The critical part for legacy GGML Q4_K is that d and dmin are super-scales,
            // and qblock->scales contains *indices* into 16-element tables.
            // The K_SCALE_VALUES and K_MIN_VALUES provided are the 64-element tables used by newer K-quants.
            // Let\'s proceed assuming scale_idx and min_idx are 0-15 and correctly index these larger tables.
        }


        const float sub_block_scale = d_super * K_SCALE_VALUES[scale_idx];
        const float sub_block_min = min_super * K_MIN_VALUES[min_idx];

        const size_t qs_offset = sub_block_idx * 8; // 8 bytes of nibbles per sub-block (16 nibbles)
        for (int k = 0; k < 8; ++k) { // Iterate over 8 bytes (16 nibbles)
            const uint8_t q_byte = qs_ptr[qs_offset + k];
            
            const int8_t q_low  = static_cast<int8_t>((q_byte & 0x0F) - 8); // Nibble to signed int8_t
            output[sub_block_idx * 16 + k] = sub_block_scale * q_low + sub_block_min;
            
            const int8_t q_high = static_cast<int8_t>((q_byte >> 4) - 8);   // Nibble to signed int8_t
            output[sub_block_idx * 16 + k + 8] = sub_block_scale * q_high + sub_block_min;
        }
    }
}
// --- END: Legacy GGML-style Q4_K Dequantization ---

// --- START: Strategy B Q4_K Dequantization (No per-sub-block scale from qblock->scales) ---
void dequantize_q4_k_m_strategy_b(
    const block_q4_K* qblock,
    float* output,
    int num_weights_in_block) {
    if (num_weights_in_block != GGML_QK_K) {
        std::cerr << "Warning: dequantize_q4_k_m_strategy_b with num_weights != GGML_QK_K" << std::endl;
        std::memset(output, 0, num_weights_in_block * sizeof(float));
        return;
    }

    const float d = fp16_to_fp32(qblock->d, true); // GGUF scale field
    const float dmin = fp16_to_fp32(qblock->dmin, true); // GGUF min field
    // const uint8_t* scales_u8 = qblock->scales; // Not used in this strategy
    const uint8_t* qs_ptr = qblock->qs;

    for (int j = 0; j < GGML_QK_K / 16; ++j) { // j is sub-block index 0..15
        // In this strategy, we ignore qblock->scales[j] as a multiplier for d.
        // The final_sub_block_scale is just d.
        const float final_sub_block_scale = d;
        const size_t qs_offset = j * 8;
        for (int k = 0; k < 8; ++k) { // k is byte index within the sub-block's qs
            const uint8_t q_byte = qs_ptr[qs_offset + k];
            const int8_t q_low  = ((q_byte & 0x0F) - 8);
            output[j * 16 + k] = final_sub_block_scale * q_low + dmin;
            const int8_t q_high = ((q_byte >> 4) - 8);
            output[j * 16 + k + 8] = final_sub_block_scale * q_high + dmin;
        }
    }
}
// --- END: Strategy B Q4_K Dequantization ---

// --- START: Strategy C Q4_K Dequantization (No dmin - based on dequantize_q4_k_no_dmin) ---
void dequantize_q4_k_m_strategy_no_dmin(
    const block_q4_K* qblock,
    float* output,
    int num_weights_in_block) {
    if (num_weights_in_block != GGML_QK_K) {
        std::cerr << "Warning: dequantize_q4_k_m_strategy_no_dmin with num_weights != GGML_QK_K" << std::endl;
        std::memset(output, 0, num_weights_in_block * sizeof(float));
        return;
    }

    const float d = fp16_to_fp32(qblock->d, true); // GGUF scale field
    // const float dmin = fp16_to_fp32(qblock->dmin, true); // dmin is ignored
    const uint8_t* scales_u8 = qblock->scales;
    const uint8_t* qs_ptr = qblock->qs;

    for (int j = 0; j < GGML_QK_K / 16; ++j) { // j is sub-block index 0..15
        const float sub_scale_factor = scales_u8[j]; 
        const float final_sub_block_scale = d * sub_scale_factor;
        const size_t qs_offset = j * 8;
        for (int k = 0; k < 8; ++k) { // k is byte index within the sub-block's qs
            const uint8_t q_byte = qs_ptr[qs_offset + k];
            const int8_t q_low  = ((q_byte & 0x0F) - 8);
            output[j * 16 + k] = final_sub_block_scale * q_low; // No dmin added
            const int8_t q_high = ((q_byte >> 4) - 8);
            output[j * 16 + k + 8] = final_sub_block_scale * q_high; // No dmin added
        }
    }
}
// --- END: Strategy C Q4_K Dequantization (No dmin) ---

// --- START: Strategy D Q4_K Dequantization (Only dmin - based on dequantize_q4_k_only_dmin) ---
void dequantize_q4_k_m_strategy_only_dmin(
    const block_q4_K* qblock,
    float* output,
    int num_weights_in_block) {
    if (num_weights_in_block != GGML_QK_K) {
        std::cerr << "Warning: dequantize_q4_k_m_strategy_only_dmin with num_weights != GGML_QK_K" << std::endl;
        std::memset(output, 0, num_weights_in_block * sizeof(float));
        return;
    }
    const float dmin = fp16_to_fp32(qblock->dmin, true); // GGUF min field
    for (int i = 0; i < num_weights_in_block; ++i) {
        output[i] = dmin;
    }
}
// --- END: Strategy D Q4_K Dequantization (Only dmin) ---

// --- START: Strategy E Q4_K Dequantization (Normalized scales_u8[j] by 16.0f) ---
void dequantize_q4_k_m_strategy_e(
    const block_q4_K* qblock,
    float* output,
    int num_weights_in_block) {
    if (num_weights_in_block != GGML_QK_K) {
        std::cerr << "Warning: dequantize_q4_k_m_strategy_e with num_weights != GGML_QK_K" << std::endl;
        std::memset(output, 0, num_weights_in_block * sizeof(float));
        return;
    }

    const float d = fp16_to_fp32(qblock->d, true); // GGUF scale field
    const float dmin = fp16_to_fp32(qblock->dmin, true); // GGUF min field
    const uint8_t* scales_u8 = qblock->scales;
    const uint8_t* qs_ptr = qblock->qs;

    for (int j = 0; j < GGML_QK_K / 16; ++j) { // j is sub-block index 0..15
        const float sub_scale_factor_normalized = static_cast<float>(scales_u8[j]) / 16.0f; 
        const float final_sub_block_scale = d * sub_scale_factor_normalized;
        const size_t qs_offset = j * 8;
        for (int k = 0; k < 8; ++k) { // k is byte index within the sub-block's qs
            const uint8_t q_byte = qs_ptr[qs_offset + k];
            const int8_t q_low  = ((q_byte & 0x0F) - 8);
            output[j * 16 + k] = final_sub_block_scale * q_low + dmin;
            const int8_t q_high = ((q_byte >> 4) - 8);
            output[j * 16 + k + 8] = final_sub_block_scale * q_high + dmin;
        }
    }
}
// --- END: Strategy E Q4_K Dequantization ---

// --- START: Strategy F Q4_K Dequantization (Normalized scales_u8[j] by 8.0f) ---
void dequantize_q4_k_m_strategy_f(
    const block_q4_K* qblock,
    float* output,
    int num_weights_in_block) {
    if (num_weights_in_block != GGML_QK_K) {
        std::cerr << "Warning: dequantize_q4_k_m_strategy_f with num_weights != GGML_QK_K" << std::endl;
        std::memset(output, 0, num_weights_in_block * sizeof(float));
        return;
    }

    const float d = fp16_to_fp32(qblock->d, true); // GGUF scale field
    const float dmin = fp16_to_fp32(qblock->dmin, true); // GGUF min field
    const uint8_t* scales_u8 = qblock->scales;
    const uint8_t* qs_ptr = qblock->qs;

    for (int j = 0; j < GGML_QK_K / 16; ++j) { // j is sub-block index 0..15
        const float sub_scale_factor_normalized = static_cast<float>(scales_u8[j]) / 8.0f; 
        const float final_sub_block_scale = d * sub_scale_factor_normalized;
        const size_t qs_offset = j * 8;
        for (int k = 0; k < 8; ++k) { // k is byte index within the sub-block's qs
            const uint8_t q_byte = qs_ptr[qs_offset + k];
            const int8_t q_low  = ((q_byte & 0x0F) - 8);
            output[j * 16 + k] = final_sub_block_scale * q_low + dmin;
            const int8_t q_high = ((q_byte >> 4) - 8);
            output[j * 16 + k + 8] = final_sub_block_scale * q_high + dmin;
        }
    }
}
// --- END: Strategy F Q4_K Dequantization ---

// --- START: Strategy G Q4_K Dequantization (Normalized scales_u8[j] by 4.0f) ---
void dequantize_q4_k_m_strategy_g(
    const block_q4_K* qblock,
    float* output,
    int num_weights_in_block) {
    if (num_weights_in_block != GGML_QK_K) {
        std::cerr << "Warning: dequantize_q4_k_m_strategy_g with num_weights != GGML_QK_K" << std::endl;
        std::memset(output, 0, num_weights_in_block * sizeof(float));
        return;
    }

    const float d = fp16_to_fp32(qblock->d, true); // GGUF scale field
    const float dmin = fp16_to_fp32(qblock->dmin, true); // GGUF min field
    const uint8_t* scales_u8 = qblock->scales;
    const uint8_t* qs_ptr = qblock->qs;

    for (int j = 0; j < GGML_QK_K / 16; ++j) { // j is sub-block index 0..15
        const float sub_scale_factor_normalized = static_cast<float>(scales_u8[j]) / 4.0f; 
        const float final_sub_block_scale = d * sub_scale_factor_normalized;
        const size_t qs_offset = j * 8;
        for (int k = 0; k < 8; ++k) { // k is byte index within the sub-block's qs
            const uint8_t q_byte = qs_ptr[qs_offset + k];
            const int8_t q_low  = ((q_byte & 0x0F) - 8);
            output[j * 16 + k] = final_sub_block_scale * q_low + dmin;
            const int8_t q_high = ((q_byte >> 4) - 8);
            output[j * 16 + k + 8] = final_sub_block_scale * q_high + dmin;
        }
    }
}
// --- END: Strategy G Q4_K Dequantization ---

// Helper to compute and print stats, and first N values
void print_vector_details(const std::string& label, const std::vector<float>& v, int n_print = 16) {
    if (v.empty()) {
        Logger::info(label + ": EMPTY");
        return;
    }
    float minv = v[0], maxv = v[0];
    double sum = 0.0;
    for (float val : v) {
        if (val < minv) minv = val;
        if (val > maxv) maxv = val;
        sum += val;
    }
    float mean = sum / v.size();

    std::stringstream ss;
    ss << label << ": size=" << v.size() << ", min=" << std::fixed << std::setprecision(6) << minv
       << ", max=" << maxv << ", mean=" << mean;
    Logger::info(ss.str());
    ss.str(""); // Clear stringstream

    ss << "  First " << std::min((int)v.size(), n_print) << " values: ";
    for (int i = 0; i < std::min((int)v.size(), n_print); ++i) {
        ss << (i > 0 ? ", " : "") << v[i];
    }
    Logger::info(ss.str());
}

int main() {
    // Logger::set_level(LogLevel::INFO); // Set log level to INFO or DEBUG - COMMENTED OUT
    std::string q4k_m_file = "data/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";
    std::string fp16_file  = "data/TinyLlama-1.1B-Chat-v1.0.FP16.gguf";

    GGUFData gguf_q4k_m, gguf_fp16;
    try {
        Logger::info("Loading Q4_K_M GGUF: " + q4k_m_file);
        gguf_q4k_m = load_gguf_meta(q4k_m_file);
        Logger::info("Loading FP16 GGUF: " + fp16_file);
        gguf_fp16 = load_gguf_meta(fp16_file);
    } catch (const std::exception& e) {
        Logger::fatal(std::string("Failed to load GGUF files: ") + e.what());
        return 1;
    }

    Logger::info("\n--- Comparison for blk.0.ffn_gate.weight (Q4_K vs FP16) ---");
    std::string tensor_name_gate = "blk.0.ffn_gate.weight";
    auto it_q4k_gate = gguf_q4k_m.tensor_infos_map.find(tensor_name_gate);
    auto it_fp16_gate = gguf_fp16.tensor_infos_map.find(tensor_name_gate);

    if (it_q4k_gate != gguf_q4k_m.tensor_infos_map.end() && it_fp16_gate != gguf_fp16.tensor_infos_map.end()) {
        const GGUFTensorInfo& tinfo_q4k_gate = it_q4k_gate->second;
        const GGUFTensorInfo& tinfo_fp16_gate = it_fp16_gate->second;

        if (tinfo_q4k_gate.type != GGMLType::GGML_TYPE_Q4_K) {
            Logger::error("Tensor " + tensor_name_gate + " in " + q4k_m_file + " is not Q4_K, but " + ggml_type_name(tinfo_q4k_gate.type));
        } else {
            const uint8_t* data_q4k_block_ptr = gguf_q4k_m.tensor_data.data() + tinfo_q4k_gate.offset;
            const block_q4_K* q4k_block_gate = reinterpret_cast<const block_q4_K*>(data_q4k_block_ptr);
            
            std::vector<float> dequant_q4k_gate_gguf(GGML_QK_K);
            // Use the dequantize_q4_k_m from quantization.h (which should be the GGUF-style one)
            // For clarity, let\'s imagine we have a renamed version or ensure quantization.cpp\'s version is the GGUF one.
            // The `dequantize_q4_k_m` in the global namespace (from quantization.h) is the one from quantization.cpp
            dequantize_q4_k_m(q4k_block_gate, dequant_q4k_gate_gguf.data(), GGML_QK_K, false);
            print_vector_details("  Dequantized Q4_K (GGUF-style, from quantization.cpp)", dequant_q4k_gate_gguf);

            std::vector<float> dequant_q4k_gate_legacy_ggml(GGML_QK_K);
            dequantize_q4_k_m_legacy_ggml(q4k_block_gate, dequant_q4k_gate_legacy_ggml.data(), GGML_QK_K);
            print_vector_details("  Dequantized Q4_K (Legacy GGML-style)", dequant_q4k_gate_legacy_ggml);

            std::vector<float> dequant_q4k_gate_strategy_b(GGML_QK_K);
            dequantize_q4_k_m_strategy_b(q4k_block_gate, dequant_q4k_gate_strategy_b.data(), GGML_QK_K);
            print_vector_details("  Dequantized Q4_K (Strategy B - No per-sub-block scale from qblock->scales)", dequant_q4k_gate_strategy_b);

            std::vector<float> dequant_q4k_gate_strategy_no_dmin(GGML_QK_K);
            dequantize_q4_k_m_strategy_no_dmin(q4k_block_gate, dequant_q4k_gate_strategy_no_dmin.data(), GGML_QK_K);
            print_vector_details("  Dequantized Q4_K (Strategy C - No dmin)", dequant_q4k_gate_strategy_no_dmin);

            std::vector<float> dequant_q4k_gate_strategy_only_dmin(GGML_QK_K);
            dequantize_q4_k_m_strategy_only_dmin(q4k_block_gate, dequant_q4k_gate_strategy_only_dmin.data(), GGML_QK_K);
            print_vector_details("  Dequantized Q4_K (Strategy D - Only dmin)", dequant_q4k_gate_strategy_only_dmin);

            std::vector<float> dequant_q4k_gate_strategy_e(GGML_QK_K);
            dequantize_q4_k_m_strategy_e(q4k_block_gate, dequant_q4k_gate_strategy_e.data(), GGML_QK_K);
            print_vector_details("  Dequantized Q4_K (Strategy E - Normalized scales_u8[j] by 16.0f)", dequant_q4k_gate_strategy_e);

            std::vector<float> dequant_q4k_gate_strategy_f(GGML_QK_K);
            dequantize_q4_k_m_strategy_f(q4k_block_gate, dequant_q4k_gate_strategy_f.data(), GGML_QK_K);
            print_vector_details("  Dequantized Q4_K (Strategy F - Normalized scales_u8[j] by 8.0f)", dequant_q4k_gate_strategy_f);

            std::vector<float> dequant_q4k_gate_strategy_g(GGML_QK_K);
            dequantize_q4_k_m_strategy_g(q4k_block_gate, dequant_q4k_gate_strategy_g.data(), GGML_QK_K);
            print_vector_details("  Dequantized Q4_K (Strategy G - Normalized scales_u8[j] by 4.0f)", dequant_q4k_gate_strategy_g);
        }

        if (tinfo_fp16_gate.type != GGMLType::GGML_TYPE_F16) {
            Logger::error("Tensor " + tensor_name_gate + " in " + fp16_file + " is not F16, but " + ggml_type_name(tinfo_fp16_gate.type));
        } else {
            const uint8_t* data_fp16_ptr = gguf_fp16.tensor_data.data() + tinfo_fp16_gate.offset;
            const uint16_t* fp16_weights_gate = reinterpret_cast<const uint16_t*>(data_fp16_ptr);
            std::vector<float> fp16_as_fp32_gate(GGML_QK_K);
            for(int i=0; i<GGML_QK_K; ++i) {
                fp16_as_fp32_gate[i] = fp16_to_fp32(fp16_weights_gate[i], false); // is_gguf_scale_field = false
            }
            print_vector_details("  FP16 converted to FP32 (from FP16 GGUF)", fp16_as_fp32_gate);
        }
    } else {
        Logger::warning("Could not find " + tensor_name_gate + " in one or both GGUF files.");
    }

    Logger::info("\n--- Comparison for blk.0.ffn_down.weight (Q6_K vs FP16) ---");
    std::string tensor_name_down = "blk.0.ffn_down.weight";
    auto it_q6k_down = gguf_q4k_m.tensor_infos_map.find(tensor_name_down);
    auto it_fp16_down = gguf_fp16.tensor_infos_map.find(tensor_name_down);

    if (it_q6k_down != gguf_q4k_m.tensor_infos_map.end() && it_fp16_down != gguf_fp16.tensor_infos_map.end()) {
        const GGUFTensorInfo& tinfo_q6k_down = it_q6k_down->second;
        const GGUFTensorInfo& tinfo_fp16_down = it_fp16_down->second;

        if (tinfo_q6k_down.type != GGMLType::GGML_TYPE_Q6_K) {
            Logger::error("Tensor " + tensor_name_down + " in " + q4k_m_file + " is not Q6_K, but " + ggml_type_name(tinfo_q6k_down.type));
        } else {
            const uint8_t* data_q6k_block_ptr = gguf_q4k_m.tensor_data.data() + tinfo_q6k_down.offset;
            const block_q6_K* q6k_block_down = reinterpret_cast<const block_q6_K*>(data_q6k_block_ptr);
            std::vector<float> dequant_q6k_down(GGML_QK_K);
            dequantize_q6_k(q6k_block_down, dequant_q6k_down.data(), GGML_QK_K, false);
            print_vector_details("  Dequantized Q6_K (from Q4_K_M GGUF)", dequant_q6k_down);
        }

        if (tinfo_fp16_down.type != GGMLType::GGML_TYPE_F16) {
            Logger::error("Tensor " + tensor_name_down + " in " + fp16_file + " is not F16, but " + ggml_type_name(tinfo_fp16_down.type));
        } else {
            const uint8_t* data_fp16_ptr = gguf_fp16.tensor_data.data() + tinfo_fp16_down.offset;
            const uint16_t* fp16_weights_down = reinterpret_cast<const uint16_t*>(data_fp16_ptr);
            std::vector<float> fp16_as_fp32_down(GGML_QK_K);
            for(int i=0; i<GGML_QK_K; ++i) {
                fp16_as_fp32_down[i] = fp16_to_fp32(fp16_weights_down[i], false); // is_gguf_scale_field = false
            }
            print_vector_details("  FP16 converted to FP32 (from FP16 GGUF)", fp16_as_fp32_down);
        }
    } else {
        Logger::warning("Could not find " + tensor_name_down + " in one or both GGUF files.");
    }

    Logger::info("\nComparison script finished.");
    return 0;
} 