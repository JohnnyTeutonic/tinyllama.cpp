#include "quantization.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <cstring>

float compute_rmse(const std::vector<float>& a, const std::vector<float>& b) {
    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum / a.size());
}

bool load_f32_vec(const std::string& filename, std::vector<float>& vec) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) return false;
    in.read(reinterpret_cast<char*>(vec.data()), vec.size() * sizeof(float));
    return in.good();
}

template<typename T>
bool load_block(const std::string& filename, T& block) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) return false;
    in.read(reinterpret_cast<char*>(&block), sizeof(T));
    return in.good();
}

// V1: Current kernel
float matvec_q4k_q8k_v1(const block_q4_K& q4k_block, const std::vector<block_q8_K>& vec_q8k) {
    std::vector<block_q4_K> mat_q4k(1, q4k_block);
    std::vector<float> out(1);
    matvec_q4k_q8k_cpu(mat_q4k, vec_q8k, out, 1, GGML_QK_K, false);
    return out[0];
}

// V2: Dequantize both Q4_K and Q8_K to FP32, then dot
float matvec_q4k_q8k_v2(const block_q4_K& q4k_block, const std::vector<block_q8_K>& vec_q8k) {
    std::vector<float> deq_q4k(GGML_QK_K);
    std::vector<float> deq_q8k(GGML_QK_K);
    dequantize_q4_k_m(&q4k_block, deq_q4k.data(), GGML_QK_K);
    // Dequantize Q8_K
    const block_q8_K& blk = vec_q8k[0];
    float d = fp16_to_fp32(blk.d);
    for (int i = 0; i < GGML_QK_K; ++i) {
        deq_q8k[i] = d * blk.qs[i];
    }
    float dot = 0.0f;
    for (int i = 0; i < GGML_QK_K; ++i) {
        dot += deq_q4k[i] * deq_q8k[i];
    }
    return dot;
}

// V3: Dequantize Q4_K, use Q8_K quantized values directly
float matvec_q4k_q8k_v3(const block_q4_K& q4k_block, const std::vector<block_q8_K>& vec_q8k) {
    std::vector<float> deq_q4k(GGML_QK_K);
    dequantize_q4_k_m(&q4k_block, deq_q4k.data(), GGML_QK_K);
    const block_q8_K& blk = vec_q8k[0];
    float dot = 0.0f;
    for (int i = 0; i < GGML_QK_K; ++i) {
        dot += deq_q4k[i] * blk.qs[i];
    }
    return dot;
}

// Repeat for Q6_K
float matvec_q6k_q8k_v1(const block_q6_K& q6k_block, const std::vector<block_q8_K>& vec_q8k) {
    std::vector<block_q6_K> mat_q6k(1, q6k_block);
    std::vector<float> out(1);
    matvec_q6k_q8k_cpu(mat_q6k, vec_q8k, out, 1, GGML_QK_K, false);
    return out[0];
}

float matvec_q6k_q8k_v2(const block_q6_K& q6k_block, const std::vector<block_q8_K>& vec_q8k) {
    std::vector<float> deq_q6k(GGML_QK_K);
    std::vector<float> deq_q8k(GGML_QK_K);
    dequantize_q6_k(&q6k_block, deq_q6k.data(), GGML_QK_K);
    const block_q8_K& blk = vec_q8k[0];
    float d = fp16_to_fp32(blk.d);
    for (int i = 0; i < GGML_QK_K; ++i) {
        deq_q8k[i] = d * blk.qs[i];
    }
    float dot = 0.0f;
    for (int i = 0; i < GGML_QK_K; ++i) {
        dot += deq_q6k[i] * deq_q8k[i];
    }
    return dot;
}

float matvec_q6k_q8k_v3(const block_q6_K& q6k_block, const std::vector<block_q8_K>& vec_q8k) {
    std::vector<float> deq_q6k(GGML_QK_K);
    dequantize_q6_k(&q6k_block, deq_q6k.data(), GGML_QK_K);
    const block_q8_K& blk = vec_q8k[0];
    float dot = 0.0f;
    for (int i = 0; i < GGML_QK_K; ++i) {
        dot += deq_q6k[i] * blk.qs[i];
    }
    return dot;
}

int main() {
    constexpr int N = GGML_QK_K;
    block_q4_K q4k_block;
    if (!load_block("data/tinyllama_ref_blk.0.attn_k.weight_q4k_block0.bin", q4k_block)) {
        std::cerr << "Failed to load Q4_K block from file." << std::endl;
        return 1;
    }
    std::vector<float> vec_f32(N);
    if (!load_f32_vec("data/tinyllama_ref_blk.0.attn_k.weight_f32.bin", vec_f32)) {
        std::cerr << "Failed to load FP32 vector from file." << std::endl;
        return 1;
    }
    std::vector<block_q8_K> vec_q8k = quantize_fp32_to_q8_K(vec_f32);

    // Reference FP32 dot
    std::vector<float> deq_q4k(N);
    dequantize_q4_k_m(&q4k_block, deq_q4k.data(), N);
    float ref_dot = 0.0f;
    for (int i = 0; i < N; ++i) ref_dot += deq_q4k[i] * vec_f32[i];

    std::cout << "\nQ4_K*Q8_K matvec variations:\n";
    std::cout << "testing version 1\n";
    float v1 = matvec_q4k_q8k_v1(q4k_block, vec_q8k);
    std::cout << "Result:   " << v1 << "\nAbs diff: " << std::abs(v1 - ref_dot) << std::endl;
    std::cout << "testing version 2\n";
    float v2 = matvec_q4k_q8k_v2(q4k_block, vec_q8k);
    std::cout << "Result:   " << v2 << "\nAbs diff: " << std::abs(v2 - ref_dot) << std::endl;
    std::cout << "testing version 3\n";
    float v3 = matvec_q4k_q8k_v3(q4k_block, vec_q8k);
    std::cout << "Result:   " << v3 << "\nAbs diff: " << std::abs(v3 - ref_dot) << std::endl;

    // Repeat for Q6_K
    block_q6_K q6k_block;
    if (!load_block("data/tinyllama_ref_blk.0.attn_k.weight_q6k_block0.bin", q6k_block)) {
        std::cerr << "Failed to load Q6_K block from file." << std::endl;
        return 1;
    }
    std::vector<float> deq_q6k(N);
    dequantize_q6_k(&q6k_block, deq_q6k.data(), N);
    float ref_dot_q6 = 0.0f;
    for (int i = 0; i < N; ++i) ref_dot_q6 += deq_q6k[i] * vec_f32[i];

    std::cout << "\nQ6_K*Q8_K matvec variations:\n";
    std::cout << "testing version 1\n";
    float q6v1 = matvec_q6k_q8k_v1(q6k_block, vec_q8k);
    std::cout << "Result:   " << q6v1 << "\nAbs diff: " << std::abs(q6v1 - ref_dot_q6) << std::endl;
    std::cout << "testing version 2\n";
    float q6v2 = matvec_q6k_q8k_v2(q6k_block, vec_q8k);
    std::cout << "Result:   " << q6v2 << "\nAbs diff: " << std::abs(q6v2 - ref_dot_q6) << std::endl;
    std::cout << "testing version 3\n";
    float q6v3 = matvec_q6k_q8k_v3(q6k_block, vec_q8k);
    std::cout << "Result:   " << q6v3 << "\nAbs diff: " << std::abs(q6v3 - ref_dot_q6) << std::endl;

    return 0;
}
