#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "ggml_types.h"

struct block_q2_K;
struct block_q3_K;
struct block_q4_K;
struct block_q6_K;

constexpr int GGML_QK_K = 256;

float fp16_to_fp32(uint16_t h, bool is_gguf_scale_field = false);
uint16_t fp32_to_fp16(float f);

#pragma pack(push, 1)

struct block_q4_K {
  uint16_t d;
  uint16_t dmin;
  uint8_t scales[12];
  uint8_t qs[GGML_QK_K / 2];
};
static_assert(sizeof(block_q4_K) == 2 + 2 + 12 + 128,
              "Size mismatch for standard block_q4_K");

struct block_q6_K {
  uint8_t ql[GGML_QK_K / 2];
  uint8_t qh[GGML_QK_K / 4];
  int8_t scales[GGML_QK_K / 16];
  uint16_t d;
};
static_assert(sizeof(block_q6_K) == 128 + 64 + 16 + 2,
              "Size mismatch for block_q6_K");

struct block_q2_K {
  uint16_t d;
  uint16_t dmin;
  uint8_t scales[GGML_QK_K / 16];
  uint8_t qs[GGML_QK_K / 4];
};
static_assert(sizeof(block_q2_K) == 2 + 2 + 16 + 64,
              "Size mismatch for block_q2_K");

struct block_q3_K {
  uint8_t hmask[GGML_QK_K / 8];
  uint8_t qs[GGML_QK_K / 4];
  uint8_t scales[12];
  uint16_t d;
  uint16_t dmin;
};
static_assert(sizeof(block_q3_K) == 32 + 64 + 12 + 2 + 2,
              "Size mismatch for block_q3_K");

struct block_q8_K {
  uint16_t d;
  int8_t qs[GGML_QK_K];
  int16_t bsums[GGML_QK_K / 16];
};

#define GGML_QK8_0 32
struct block_q8_0 {
  uint16_t d;
  int8_t qs[GGML_QK8_0];
};
static_assert(sizeof(block_q8_0) == sizeof(uint16_t) + GGML_QK8_0,
              "Size mismatch for block_q8_0");

#pragma pack(pop)

const char* ggml_type_name(GGMLType type);

size_t ggml_type_size(GGMLType type);
size_t ggml_type_block_size(GGMLType type);

void dequantize_q2_k(const void* q_data, float* f_data,
                     int num_weights_in_block,
                     bool log_details_for_this_block = false);

void dequantize_q4_k_m(const block_q4_K* qblock, float* __restrict__ output_f32,
                       int num_elements, bool log_this_block = false);

void dequantize_q6_k(const block_q6_K* qblock, float* __restrict__ output_f32,
                     int num_elements, bool log_this_block = false);

void dequantize_q3_k(const void* q_data, float* f_data,
                     int num_weights_in_block);

void handle_i8_tensor(const void* i8_data, float* f_data, size_t num_elements);

void quantize_q4_k_m(const float* f_data, void* q_data, int num_elements);

void quantize_q6_k(const float* f_data, void* q_data, int num_elements);

std::vector<block_q8_K> quantize_fp32_to_q8_K(const std::vector<float>& f_data);

float vec_dot_q6_k_q8_k_cpu(int n, const std::vector<block_q6_K>& x,
                            const std::vector<block_q8_K>& y,
                            bool log_this_call);

void matvec_q6k_q8k_cpu(const std::vector<block_q6_K>& mat_q6k,
                        const std::vector<block_q8_K>& vec_q8k,
                        std::vector<float>& out_f32, int rows, int cols,
                        bool log_calls);

float vec_dot_q4_k_q8_k_cpu(int n, const std::vector<block_q4_K>& x_vec,
                            const std::vector<block_q8_K>& y_vec,
                            bool log_this_call);

void matvec_q4k_q8k_cpu(const std::vector<block_q4_K>& mat_q4k,
                        const std::vector<block_q8_K>& vec_q8k,
                        std::vector<float>& out_f32, int rows, int cols,
                        bool log_calls);

void dequantize_q8_0_block(const block_q8_0* qblock, float* output);
