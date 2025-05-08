#include "quantization.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <cstring>

// Helper to compute RMSE
float compute_rmse(const std::vector<float>& a, const std::vector<float>& b) {
    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum / a.size());
}

// Helper to load a binary file into a struct or vector
template<typename T>
bool load_block(const std::string& filename, T& block) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) return false;
    in.read(reinterpret_cast<char*>(&block), sizeof(T));
    return in.good();
}

bool load_f32_vec(const std::string& filename, std::vector<float>& vec) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) return false;
    in.read(reinterpret_cast<char*>(vec.data()), vec.size() * sizeof(float));
    return in.good();
}

// --- START: Correct Q4_K Dequantization Logic ---
void dequantize_q4_k_correct(
    const block_q4_K* qblock,
    float* output,
    int num_weights_in_block
) {
    if (num_weights_in_block != GGML_QK_K) {
        std::cerr << "Warning: dequantize_q4_k_correct called with num_weights != GGML_QK_K (" << num_weights_in_block << ")" << std::endl;
        std::memset(output, 0, num_weights_in_block * sizeof(float));
        return;
    }

    const float d = fp16_to_fp32(qblock->d);
    const float dmin = fp16_to_fp32(qblock->dmin);
    const uint8_t* scales_u8 = qblock->scales;
    const uint8_t* qs_ptr = qblock->qs;

    for (int j = 0; j < GGML_QK_K / 16; ++j) {
        const float sub_scale = d * scales_u8[j];
        const size_t qs_offset = j * 8;
        // Detailed logging for first subblock
        if (j == 0) {
            std::cout << "sub_scale[0]: " << sub_scale << std::endl;
            for (int k = 0; k < 8; ++k) {
                const uint8_t q_byte = qs_ptr[qs_offset + k];
                const int8_t q_low  = ((q_byte & 0x0F) - 8);
                const int8_t q_high = ((q_byte >> 4) - 8);
                std::cout << "k=" << k
                          << " q_byte=0x" << std::hex << (int)q_byte << std::dec
                          << " q_low=" << (int)q_low
                          << " q_high=" << (int)q_high
                          << " deq_low=" << (sub_scale * q_low + dmin)
                          << " deq_high=" << (sub_scale * q_high + dmin)
                          << std::endl;
            }
        }
        for (int k = 0; k < 8; ++k) {
            const uint8_t q_byte = qs_ptr[qs_offset + k];
            const int8_t q_low  = ((q_byte & 0x0F) - 8);
            output[j * 16 + k] = sub_scale * q_low + dmin;
            const int8_t q_high = ((q_byte >> 4) - 8);
            output[j * 16 + k + 8] = sub_scale * q_high + dmin;
        }
    }
}
// --- END: Correct Q4_K Dequantization Logic ---

// --- START: Alternative Q4_K Dequantization Strategies ---
// Strategy 5: Ignore dmin (zero-point)
void dequantize_q4_k_no_dmin(
    const block_q4_K* qblock,
    float* output,
    int num_weights_in_block
) {
    const float d = fp16_to_fp32(qblock->d);
    const uint8_t* scales_u8 = qblock->scales;
    const uint8_t* qs_ptr = qblock->qs;
    for (int j = 0; j < GGML_QK_K / 16; ++j) {
        const float sub_scale = d * scales_u8[j];
        const size_t qs_offset = j * 8;
        for (int k = 0; k < 8; ++k) {
            const uint8_t q_byte = qs_ptr[qs_offset + k];
            const int8_t q_low  = ((q_byte & 0x0F) - 8);
            output[j * 16 + k] = sub_scale * q_low;
            const int8_t q_high = ((q_byte >> 4) - 8);
            output[j * 16 + k + 8] = sub_scale * q_high;
        }
    }
}
// Strategy 6: Use only dmin (ignore scale)
void dequantize_q4_k_only_dmin(
    const block_q4_K* qblock,
    float* output,
    int num_weights_in_block
) {
    const float dmin = fp16_to_fp32(qblock->dmin);
    for (int i = 0; i < num_weights_in_block; ++i) {
        output[i] = dmin;
    }
}
// Strategy 7: Use only scale (ignore dmin)
void dequantize_q4_k_only_scale(
    const block_q4_K* qblock,
    float* output,
    int num_weights_in_block
) {
    const float d = fp16_to_fp32(qblock->d);
    const uint8_t* scales_u8 = qblock->scales;
    const uint8_t* qs_ptr = qblock->qs;
    for (int j = 0; j < GGML_QK_K / 16; ++j) {
        const float sub_scale = d * scales_u8[j];
        const size_t qs_offset = j * 8;
        for (int k = 0; k < 8; ++k) {
            const uint8_t q_byte = qs_ptr[qs_offset + k];
            const int8_t q_low  = ((q_byte & 0x0F) - 8);
            output[j * 16 + k] = sub_scale * q_low;
            const int8_t q_high = ((q_byte >> 4) - 8);
            output[j * 16 + k + 8] = sub_scale * q_high;
        }
    }
}
// --- END: Alternative Q4_K Dequantization Strategies ---

// --- START: FP16 Conversion (Required by dequantize_q4_k_correct) ---
// Function to convert FP16 (uint16_t) to FP32 (float)
// (Include a basic implementation if not already globally available)
// Placeholder - Assuming fp16_to_fp32 is available via quantization.h or similar
/* REMOVED REDUNDANT DEFINITION
#ifndef FP16_TO_FP32_DEFINED // Avoid redefinition if already included
#define FP16_TO_FP32_DEFINED
float fp16_to_fp32(uint16_t h) {
    // ... (implementation as in quantization.cpp) ...
    // Using a simplified version for brevity in this edit:
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp_fp16 = (h >> 10) & 0x1f;
    uint32_t mant_fp16 = h & 0x3ff;
    uint32_t x;
    if (exp_fp16 == 0) {
        if (mant_fp16 == 0) x = (sign << 31);
        else {
            exp_fp16 = 1;
            while ((mant_fp16 & 0x400) == 0) { mant_fp16 <<= 1; exp_fp16--; }
            mant_fp16 &= ~0x400;
            uint32_t exp_fp32 = (exp_fp16 - 15 + 127);
            uint32_t mant_fp32 = mant_fp16 << 13;
            x = (sign << 31) | (exp_fp32 << 23) | mant_fp32;
        }
    } else if (exp_fp16 == 0x1f) {
        x = (sign << 31) | (0xff << 23) | (mant_fp16 << 13);
    } else {
        uint32_t exp_fp32 = (exp_fp16 - 15 + 127);
        uint32_t mant_fp32 = mant_fp16 << 13;
        x = (sign << 31) | (exp_fp32 << 23) | mant_fp32;
    }
    float f;
    std::memcpy(&f, &x, sizeof(float));
    return f;
}
#endif
*/
// --- END: FP16 Conversion ---

int main() {
    constexpr int N = GGML_QK_K;
    std::vector<float> ref_data(N);
    if (!load_f32_vec("data/tinyllama_ref_blk.0.attn_k.weight_f32.bin", ref_data)) {
        std::cerr << "Failed to load FP32 reference block from file." << std::endl;
        return 1;
    }

    // --- Q4_K Dequantization from file ---
    block_q4_K q4k_block;
    if (!load_block("data/tinyllama_ref_blk.0.attn_k.weight_q4k_block0.bin", q4k_block)) {
        std::cerr << "Failed to load Q4_K block from file." << std::endl;
        return 1;
    }
    // Log raw Q4_K block contents
    std::cout << "Q4_K block raw values:\n";
    std::cout << "d: " << q4k_block.d << " (fp16: " << fp16_to_fp32(q4k_block.d) << ")\n";
    std::cout << "dmin: " << q4k_block.dmin << " (fp16: " << fp16_to_fp32(q4k_block.dmin) << ")\n";
    std::cout << "scales: ";
    for (int i = 0; i < 16; ++i) std::cout << (int)q4k_block.scales[i] << " ";
    std::cout << "\nqs: ";
    for (int i = 0; i < 16; ++i) std::cout << std::hex << (int)q4k_block.qs[i] << " ";
    std::cout << std::dec << "\n";
    std::vector<float> deq_q4k(N, 0.0f);
    dequantize_q4_k_m(&q4k_block, deq_q4k.data(), N);

    // --- Strategy 4: Correct Q4_K Dequantization ---
    std::vector<float> deq_q4k_strategy4(N, 0.0f);
    dequantize_q4_k_correct(&q4k_block, deq_q4k_strategy4.data(), N);
    // --- Strategy 5: No dmin ---
    std::vector<float> deq_q4k_no_dmin(N, 0.0f);
    dequantize_q4_k_no_dmin(&q4k_block, deq_q4k_no_dmin.data(), N);
    // --- Strategy 6: Only dmin ---
    std::vector<float> deq_q4k_only_dmin(N, 0.0f);
    dequantize_q4_k_only_dmin(&q4k_block, deq_q4k_only_dmin.data(), N);
    // --- Strategy 7: Only scale ---
    std::vector<float> deq_q4k_only_scale(N, 0.0f);
    dequantize_q4_k_only_scale(&q4k_block, deq_q4k_only_scale.data(), N);

    // --- Q6_K Dequantization from file ---
    block_q6_K q6k_block;
    if (!load_block("data/tinyllama_ref_blk.0.attn_k.weight_q6k_block0.bin", q6k_block)) {
        std::cerr << "Failed to load Q6_K block from file." << std::endl;
        return 1;
    }
    std::vector<float> deq_q6k(N, 0.0f);
    dequantize_q6_k(&q6k_block, deq_q6k.data(), N);

    // --- Print first 16 values for comparison ---
    std::cout << "First 16 values (index: FP32 | Q4_K (Buggy) | Q4_K (Correct) | Q4_K (No dmin) | Q4_K (Only dmin) | Q4_K (Only scale) | Q6_K):\n";
    for (int i = 0; i < 16; ++i) {
        std::cout << i << ": " << ref_data[i] << " | " << deq_q4k[i] << " | " << deq_q4k_strategy4[i] << " | " << deq_q4k_no_dmin[i] << " | " << deq_q4k_only_dmin[i] << " | " << deq_q4k_only_scale[i] << " | " << deq_q6k[i] << std::endl;
    }
    std::cout << "\nQ4_K (Buggy) RMSE vs FP32:  " << compute_rmse(ref_data, deq_q4k) << std::endl;
    std::cout << "Q4_K (Correct) RMSE vs FP32: " << compute_rmse(ref_data, deq_q4k_strategy4) << std::endl;
    std::cout << "Q4_K (No dmin) RMSE vs FP32: " << compute_rmse(ref_data, deq_q4k_no_dmin) << std::endl;
    std::cout << "Q4_K (Only dmin) RMSE vs FP32: " << compute_rmse(ref_data, deq_q4k_only_dmin) << std::endl;
    std::cout << "Q4_K (Only scale) RMSE vs FP32: " << compute_rmse(ref_data, deq_q4k_only_scale) << std::endl;
    std::cout << "Q6_K RMSE vs FP32:          " << compute_rmse(ref_data, deq_q6k) << std::endl;

    return 0;
} 