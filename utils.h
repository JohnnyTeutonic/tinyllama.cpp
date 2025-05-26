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

void apply_rope_batch_cpu(std::vector<float>& q_batch, std::vector<float>& k_batch,
                          int num_tokens, int num_q_heads, int num_kv_heads,
                          int head_dim, int start_pos_in_sequence,
                          const std::vector<std::pair<float, float>>& all_freqs_cis,
                          int max_pos_embeddings, bool use_adjacent_pairing);

// Neural network operations
void rmsnorm_batch_cpu(const std::vector<float>& x_batch,
                       const std::vector<float>& weight,
                       std::vector<float>& out_batch,
                       int num_tokens, int hidden_size,
                       float eps = numeric::DEFAULT_EPS);

void rmsnorm_vector_cpu(const std::vector<float>& x,
                        const std::vector<float>& weight,
                        std::vector<float>& out,
                        float eps = numeric::DEFAULT_EPS);

void softmax_vector_cpu(const std::vector<float>& x, std::vector<float>& out);
void silu_cpu(const std::vector<float>& x, std::vector<float>& out);

// Batch matrix multiplication
void matmul_f32_f32_batch_cpu(const std::vector<float>& mat_weights,
                               const std::vector<float>& batch_input_activations,
                               std::vector<float>& batch_output_activations,
                               int num_tokens, int output_dim, int input_dim);

// BFloat16 matrix-vector operations
void matvec_bf16_f32_vector_cpu(const std::vector<uint16_t>& mat_bf16,
                                const std::vector<float>& vec_f32,
                                std::vector<float>& out_f32, int rows, int cols);

// Attention computation functions
void weighted_sum_probs_v(const std::vector<float>& probs,
                          const std::vector<float>& V,
                          std::vector<float>& out, int seq_len, int head_dim);

void calculate_attention_scores(const std::vector<float>& Q,
                                const std::vector<float>& K,
                                std::vector<float>& scores, int seq_len,
                                int head_dim, float scale);

// Logging and debugging functions
void log_vector_summary(const std::string& name, const std::vector<float>& v, int head_count);
void log_vector_summary_with_tail(const std::string& name, const std::vector<float>& v,
                                   int head_count, int tail_count);
void log_vector_summary_detailed(const std::string& name, const std::vector<float>& v,
                                  int current_pos, int current_layer, int N = 5);
void log_vec_stats(const std::string& name, const std::vector<float>& v);
void log_raw_float_pointer(const std::string& name, const float* ptr, size_t count = 5);

// File I/O utility functions
bool write_vector_to_file(const std::string& filename, const std::vector<float>& vec);
std::vector<std::vector<float>> load_rmsnorm_bin(const std::string& filename,
                                                  int num_tokens, int hidden_size);

// Helper conversion functions
std::vector<float> bf16vec_to_float_vec(const std::vector<uint16_t>& v_bf16);

// Quantization utility
void dequantize_q8_k(const std::vector<block_q8_K>& q8k_vec,
                     std::vector<float>& out_f32, int n, bool log_this_block); 