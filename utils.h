#pragma once

#include <vector>
#include <string>
#include <cstdint>

#include "quantization.h"
#include "model_constants.h"

// SIMD optimized functions
float simd_dot_product(const float* a, const float* b, int n);
void simd_scaled_add(float* dst, const float* src, float scale, int n);

// BFloat16 conversion functions
uint16_t float32_to_bfloat16(float val);
float bfloat16_to_float32(uint16_t bf16);
std::vector<float> bfloat16_vector_to_float32(const std::vector<uint16_t>& bf16_vec);

// Vector utility functions
std::vector<uint16_t> uint8_vector_to_uint16_vector(const std::vector<uint8_t>& bytes, size_t numel);
int argmax(const std::vector<float>& v);

// Matrix-vector multiplication functions (CPU)
void matvec_q6k_f32_vector_cpu(const std::vector<block_q6_K>& mat_q6k,
                               const std::vector<float>& vec_f32,
                               std::vector<float>& out_f32, int rows,
                               int cols, bool log_first_block = false);

void matvec_q4k_f32_vector_cpu(const std::vector<block_q4_K>& mat_q4k,
                               const std::vector<float>& vec_f32,
                               std::vector<float>& out_f32, int rows,
                               int cols, bool log_first_block = false);

void matvec_q8_0_f32_vector_cpu(const std::vector<block_q8_0>& mat_q8_0,
                                const std::vector<float>& vec_f32,
                                std::vector<float>& out_f32, int rows,
                                int cols, bool log_first_block = false);

void matvec_q8k_f32_vector_cpu(const std::vector<block_q8_K>& mat_q8k,
                               const std::vector<float>& vec_f32,
                               std::vector<float>& out_f32, int rows,
                               int cols, bool log_first_block = false);

void matvec_f32_f32_vector_cpu(const std::vector<float>& mat_f32,
                               const std::vector<float>& vec_f32,
                               std::vector<float>& out_f32, int rows,
                               int cols);

// Batch matrix multiplication functions (CPU)
void matmul_q4k_f32_batch_cpu(const std::vector<block_q4_K>& mat_q4k,
                               const std::vector<float>& batch_input_activations,
                               std::vector<float>& batch_output_activations,
                               int num_tokens, int output_dim, int input_dim);

void matmul_q6k_f32_batch_cpu(const std::vector<block_q6_K>& mat_q6k,
                               const std::vector<float>& batch_input_activations,
                               std::vector<float>& batch_output_activations,
                               int num_tokens, int output_dim, int input_dim);

void matmul_q8_0_f32_batch_cpu(const std::vector<block_q8_0>& mat_q8_0,
                                const std::vector<float>& batch_input_activations,
                                std::vector<float>& batch_output_activations,
                                int num_tokens, int output_dim, int input_dim);

void matmul_q8k_f32_batch_cpu(const std::vector<block_q8_K>& mat_q8k,
                               const std::vector<float>& batch_input_activations,
                               std::vector<float>& batch_output_activations,
                               int num_tokens, int output_dim, int input_dim);

// Neural network operations (CPU) - these are implemented as static functions in model.cpp

// RoPE (Rotary Position Embedding) functions
void apply_rope_vector(std::vector<float>& x, int num_heads, int head_dim,
                       int current_token_pos,
                       const std::vector<std::pair<float, float>>& all_freqs_cis,
                       int max_pos_embeddings, bool use_adjacent_pairing);

// Other functions like apply_rope_batch_cpu, weighted_sum_probs_v, etc. are static in model.cpp

// Logging and debugging functions
void log_vector_summary(const std::string& name, const std::vector<float>& v, int head_count);
void log_vector_summary_with_tail(const std::string& name, const std::vector<float>& v,
                                   int head_count, int tail_count);


// Helper conversion functions
std::vector<float> bf16vec_to_float_vec(const std::vector<uint16_t>& v_bf16);

// Quantization utility
void dequantize_q8_k(const std::vector<block_q8_K>& q8k_vec,
                     std::vector<float>& out_f32, int n, bool log_this_block); 