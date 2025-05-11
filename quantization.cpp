#include "quantization.h"
#include "model_macros.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "logger.h"
#include "gguf_parser.h"

constexpr float K_SCALE_VALUES[64] = {
    1.0f,  1.0625f, 1.125f, 1.1875f, 1.25f, 1.3125f, 1.375f, 1.4375f,
    1.5f,  1.5625f, 1.625f, 1.6875f, 1.75f, 1.8125f, 1.875f, 1.9375f,
    2.0f,  2.125f,  2.25f,  2.375f,  2.5f,  2.625f,  2.75f,  2.875f,
    3.0f,  3.125f,  3.25f,  3.375f,  3.5f,  3.625f,  3.75f,  3.875f,
    4.0f,  4.25f,   4.5f,   4.75f,   5.0f,  5.25f,   5.5f,   5.75f,
    6.0f,  6.25f,   6.5f,   6.75f,   7.0f,  7.25f,   7.5f,   7.75f,
    8.0f,  8.5f,    9.0f,   9.5f,    10.0f, 10.5f,   11.0f,  11.5f,
    12.0f, 12.5f,   13.0f,  13.5f,   14.0f, 14.5f,   15.0f,  15.5f};

constexpr float K_MIN_VALUES[64] = {
    0.0f,       -0.0078125f, -0.015625f, -0.0234375f, -0.03125f,  -0.0390625f,
    -0.046875f, -0.0546875f, -0.0625f,   -0.0703125f, -0.078125f, -0.0859375f,
    -0.09375f,  -0.1015625f, -0.109375f, -0.1171875f, -0.125f,    -0.140625f,
    -0.15625f,  -0.171875f,  -0.1875f,   -0.203125f,  -0.21875f,  -0.234375f,
    -0.25f,     -0.265625f,  -0.28125f,  -0.296875f,  -0.3125f,   -0.328125f,
    -0.34375f,  -0.359375f,  -0.375f,    -0.40625f,   -0.4375f,   -0.46875f,
    -0.5f,      -0.53125f,   -0.5625f,   -0.59375f,   -0.625f,    -0.65625f,
    -0.6875f,   -0.71875f,   -0.75f,     -0.78125f,   -0.8125f,   -0.84375f,
    -0.875f,    -0.9375f,    -1.0f,      -1.0625f,    -1.125f,    -1.1875f,
    -1.25f,     -1.3125f,    -1.375f,    -1.4375f,    -1.5f,      -1.5625f,
    -1.625f,    -1.6875f,    -1.75f,     -1.8125f};

static std::atomic<int> g_vec_dot_q4_k_q8_k_log_count{0};

float fp16_to_fp32(uint16_t h, bool is_gguf_scale_field) {
  uint16_t h_to_convert = h;
  bool original_sign_bit_was_set = (h & 0x8000);

  if (is_gguf_scale_field && original_sign_bit_was_set) {
    h_to_convert = h & 0x7FFF;
  }

  uint32_t sign = (h_to_convert >> 15) & 1;
  uint32_t exp_fp16 = (h_to_convert >> 10) & 0x1f;
  uint32_t mant_fp16 = h_to_convert & 0x3ff;

  uint32_t x;

  if (exp_fp16 == 0) {
    if (mant_fp16 == 0) {
      x = (sign << 31);

    } else {
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
  } else if (exp_fp16 == 0x1f) {
    x = (sign << 31) | (0xff << 23) | (mant_fp16 << 13);
  } else {
    uint32_t exp_fp32 = (exp_fp16 - 15 + 127);
    uint32_t mant_fp32 = mant_fp16 << 13;
    x = (sign << 31) | (exp_fp32 << 23) | mant_fp32;
  }

  float f;
  std::memcpy(&f, &x, sizeof(float));

  if (is_gguf_scale_field && f < 0.0f && !(std::isnan(f) || std::isinf(f))) {
    f = std::abs(f);
  }

  return f;
}

uint16_t fp32_to_fp16(float f) {
  uint32_t x;
  std::memcpy(&x, &f, sizeof(float));

  uint32_t sign = (x >> 31) & 1;
  uint32_t exp_fp32 = (x >> 23) & 0xff;
  uint32_t mant_fp32 = x & 0x7fffff;

  uint16_t u;

  if (exp_fp32 == 0xff) {
    u = (sign << 15) | 0x7c00 | (mant_fp32 != 0 ? 0x200 : 0);
  } else {
    int exp_fp16 = (int)exp_fp32 - 127 + 15;

    if (exp_fp16 >= 0x1f) {
      u = (sign << 15) | 0x7c00;
    } else if (exp_fp16 <= 0) {
      if (exp_fp16 < -10) {
        u = (sign << 15);
      } else {
        mant_fp32 = (mant_fp32 | 0x800000) >> (1 - exp_fp16);

        if ((mant_fp32 >> 13) & 1) {
          mant_fp32 += (1 << 13);
        }
        u = (sign << 15) | (mant_fp32 >> 13);
      }
    } else {
      if ((mant_fp32 >> 13) & 1) {
        mant_fp32 += (1 << 13);
        if ((mant_fp32 >> 23) == 1) {
          mant_fp32 = 0;
          exp_fp16++;
          if (exp_fp16 >= 0x1f) {
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

namespace {

std::vector<float> k_lookup_table_scale;
std::vector<float> k_lookup_table_min;

}  // namespace

static inline void get_scale_min_indices_q4_K(int j, const uint8_t* scales,
                                              uint8_t* scale_index,
                                              uint8_t* min_index) {
  assert(j >= 0 && j < 16);

  *scale_index = scales[j % 8] >> (4 * (j / 8));
  *scale_index &= 0x0F;

  *min_index = scales[j % 4 + 8] >> (4 * (j / 4));
  *min_index &= 0x0F;
}

/* --- OLD INCORRECT HELPER ---
static inline void get_scale_min_indices_q4_K(
    int j,
    const uint8_t* scales,
block_q4_K uint8_t* scale_index,
min_index
    int scale_byte_index = j / 2;
LOGIC int min_byte_index = j / 2 + 6;
WRONG LOGIC

    if (j % 2 == 0) {
        *scale_index = scales[scale_byte_index] & 0x0F;
        *min_index   = scales[min_byte_index] & 0x0F;
    } else {
        *scale_index = scales[scale_byte_index] >> 4;
        *min_index   = scales[min_byte_index] >> 4;
    }
}
*/

void dequantize_q4_k_m(const block_q4_K* qblock, float* output,
                       int num_weights_in_block, bool log_this_block) {
  if (num_weights_in_block != GGML_QK_K) {
    std::cout
        << "Warning: dequantize_q4_k_m called with num_weights != GGML_QK_K ("
        << num_weights_in_block << ")" << std::endl;
    std::memset(output, 0, num_weights_in_block * sizeof(float));
    return;
  }

  const float d = fp16_to_fp32(qblock->d, true);
  const float dmin = fp16_to_fp32(qblock->dmin, true);
  const uint8_t* scales_u8 = qblock->scales;
  const uint8_t* qs_ptr = qblock->qs;

  for (int j = 0; j < GGML_QK_K / 16; ++j) {
    const float sub_scale_factor_normalized =
        static_cast<float>(scales_u8[j]) / 4.0f;
    const float final_sub_block_scale = d * sub_scale_factor_normalized;
    const size_t qs_offset = j * 8;
    for (int k = 0; k < 8; ++k) {
      const uint8_t q_byte = qs_ptr[qs_offset + k];
      const int8_t q_low = ((q_byte & 0x0F) - 8);
      output[j * 16 + k] = final_sub_block_scale * q_low + dmin;
      const int8_t q_high = ((q_byte >> 4) - 8);
      output[j * 16 + k + 8] = final_sub_block_scale * q_high + dmin;
    }
  }
}

void dequantize_q6_k(const block_q6_K* qblock, float* output,
                     int num_weights_in_block, bool log_this_block) {
  if (num_weights_in_block != GGML_QK_K) {
    std::cout
        << "Warning: dequantize_q6_k called with num_weights != GGML_QK_K ("
        << num_weights_in_block << ")" << std::endl;
    std::memset(output, 0, num_weights_in_block * sizeof(float));
    return;
  }

  const float d = fp16_to_fp32(qblock->d, false);

  // Pointers to the start of the whole block's data
  const uint8_t * p_ql = qblock->ql;
  const uint8_t * p_qh = qblock->qh;
  const int8_t  * p_sc = qblock->scales;
  float * p_y  = output;

  // Process the 256 elements of the block.
  // The llama.cpp code structure processes it in two 128-element chunks.
  for (int half_idx = 0; half_idx < 2; ++half_idx) { // Process first 128, then next 128
      // Set up pointers for the current 128-element half
      const uint8_t * ql = p_ql + (half_idx * 64); // Each half uses 64 bytes of ql
      const uint8_t * qh = p_qh + (half_idx * 32); // Each half uses 32 bytes of qh
      const int8_t  * sc = p_sc + (half_idx * 8);  // Each half uses 8 scales for these 128 elements
      float         * y  = p_y  + (half_idx * 128); // Output pointer for this half

      // Inner loop processes 32 sets of 4 values = 128 floats
      for (int l = 0; l < 32; ++l) {
          int is = l / 16; // Scale sub-group index within this half's 8 scales (0 for l=0..15, 1 for l=16..31)

          // Extract the four 6-bit quantized values, already offset by -32
          const int8_t q1 = (int8_t)(((ql[l +  0] & 0x0F) | (((qh[l] >> 0) & 0x03) << 4))) - 32;
          const int8_t q2 = (int8_t)(((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 0x03) << 4))) - 32;
          const int8_t q3 = (int8_t)(((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 0x03) << 4))) - 32;
          const int8_t q4 = (int8_t)(((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 0x03) << 4))) - 32;

          // Apply scales and dequantize
          // sc points to the 8 scales for the current 128-element half.
          // is = 0 means l is 0-15, so we use sc[0], sc[2], sc[4], sc[6]
          // is = 1 means l is 16-31, so we use sc[1], sc[3], sc[5], sc[7]
          // This is effectively sc[is + 0], sc[is + 2], sc[is + 4], sc[is + 6] relative to the start of the 8 scales for this half
          // when considering the two sets of scales used across the 32 iterations of l.

          y[l +  0] = d * sc[is + 0] * q1;
          y[l + 32] = d * sc[is + 2] * q2;
          y[l + 64] = d * sc[is + 4] * q3;
          y[l + 96] = d * sc[is + 6] * q4;
      }
  }
}

void handle_i8_tensor(const void* input_data, float* output_data,
                      size_t num_elements) {
  const int8_t* input_ptr = static_cast<const int8_t*>(input_data);
  for (size_t i = 0; i < num_elements; ++i) {
    output_data[i] = static_cast<float>(input_ptr[i]);
  }
}

void quantize_q4_k_m(const float* input, void* output_qblock_void,
                     int num_elements) {
  if (num_elements != GGML_QK_K) {
    throw std::invalid_argument(
        "quantize_q4_k_m currently only supports block size " +
        std::to_string(GGML_QK_K));
  }

  block_q4_K* output_qblock = static_cast<block_q4_K*>(output_qblock_void);

  std::memset(output_qblock->scales, 0, sizeof(output_qblock->scales));
  std::memset(output_qblock->qs, 0, sizeof(output_qblock->qs));

  float block_min_val = std::numeric_limits<float>::max();
  float block_max_val = std::numeric_limits<float>::lowest();
  for (int i = 0; i < num_elements; ++i) {
    block_min_val = SAFE_MIN(block_min_val, input[i]);
    block_max_val = SAFE_MAX(block_max_val, input[i]);
  }

  if (block_max_val == block_min_val) {
    block_max_val = block_min_val + GGUF_SMALL_VAL;
  }
  if (block_max_val < GGUF_EPSILON && block_max_val > -GGUF_EPSILON) {
    block_max_val = GGUF_SMALL_VAL;
    block_min_val = 0.0f;
  }

  const float d_super_scale_candidate = (block_max_val - block_min_val) / Q4K_SCALE_FACTOR;
  const float d_super =
      d_super_scale_candidate > GGUF_EPSILON ? d_super_scale_candidate : GGUF_EPSILON;
  const float min_super = block_min_val;

  output_qblock->d = fp32_to_fp16(d_super);
  output_qblock->dmin = fp32_to_fp16(min_super);

  for (int j = 0; j < GGML_QK_K / 16; ++j) {
    const float* sub_block_input = input + j * 16;

    float sub_min_val = sub_block_input[0];
    float sub_max_val = sub_block_input[0];
    for (int i = 1; i < 16; ++i) {
      sub_min_val = SAFE_MIN(sub_min_val, sub_block_input[i]);
      sub_max_val = SAFE_MAX(sub_max_val, sub_block_input[i]);
    }

    float ideal_scale = 0.0f;
    if (sub_max_val > sub_min_val + GGUF_EPSILON) {
      ideal_scale = (sub_max_val - sub_min_val) / Q4K_SCALE_FACTOR;
    }
    float ideal_min = sub_min_val;

    uint8_t best_scale_idx = 0;
    float min_scale_err = std::numeric_limits<float>::max();
    if (d_super > GGUF_EPSILON) {
      for (uint8_t k = 0; k < 16; ++k) {
        float candidate_scale = d_super * K_SCALE_VALUES[k];
        float err = std::abs(candidate_scale - ideal_scale);
        if (err < min_scale_err) {
          min_scale_err = err;
          best_scale_idx = k;
        }
      }
    }

    uint8_t best_min_idx = 0;
    float min_min_err = std::numeric_limits<float>::max();

    for (uint8_t l = 0; l < 16; ++l) {
      float candidate_min = min_super * K_MIN_VALUES[l];
      float err = std::abs(candidate_min - ideal_min);
      if (err < min_min_err) {
        min_min_err = err;
        best_min_idx = l;
      }
    }

    int scale_byte_idx = j % 8;
    int scale_shift = 4 * (j / 8);
    output_qblock->scales[scale_byte_idx] |= (best_scale_idx << scale_shift);

    int min_byte_idx = (j % 4) + 8;
    int min_shift = 4 * (j / 4);
    output_qblock->scales[min_byte_idx] |= (best_min_idx << min_shift);

    float actual_scale = d_super * K_SCALE_VALUES[best_scale_idx];
    float actual_min = min_super * K_MIN_VALUES[best_min_idx];
    float inv_actual_scale = (actual_scale > GGUF_EPSILON || actual_scale < -GGUF_EPSILON)
                                 ? 1.0f / actual_scale
                                 : 0.0f;

    uint8_t packed_qs[8];

    std::memset(packed_qs, 0, sizeof(packed_qs));

    for (int i = 0; i < 16; ++i) {
      float val = sub_block_input[i];

      int quant_val = 0;
      if (inv_actual_scale != 0.0f) {
        quant_val =
            static_cast<int>(std::round((val - actual_min) * inv_actual_scale)) + Q4K_OFFSET;
      }
      quant_val = SAFE_MAX(0, SAFE_MIN(15, quant_val));

      int byte_idx_qs = i / 2;
      int shift_qs = (i % 2) * 4;
      packed_qs[byte_idx_qs] |= (static_cast<uint8_t>(quant_val) << shift_qs);
    }

    uint8_t* qs_target = output_qblock->qs + j * 8;
    for (int i = 0; i < 8; ++i) {
      uint8_t low_nibble_val = packed_qs[i] & 0x0F;
      uint8_t high_nibble_val = (packed_qs[i] >> 4) & 0x0F;
      qs_target[i] = low_nibble_val | (high_nibble_val << 4);
    }
  }
}

void dequantize_q2_k(const void* qblock_void, float* output,
                     int num_weights_in_block) {
  if (num_weights_in_block != GGML_QK_K) {
    throw std::invalid_argument(
        "dequantize_q2_k currently only supports block size " +
        std::to_string(GGML_QK_K));
  }

  const block_q2_K* qblock = static_cast<const block_q2_K*>(qblock_void);

  const float d_float_raw = fp16_to_fp32(qblock->d);
  const float dmin_float_raw = fp16_to_fp32(qblock->dmin);

  const float d_float = (!std::isfinite(d_float_raw)) ? 0.0f : d_float_raw;
  const float dmin_float =
      (!std::isfinite(dmin_float_raw)) ? 0.0f : dmin_float_raw;

  const float d_float_clamped = SAFE_MIN(SAFE_MAX(d_float, TENSOR_SCALE_MIN), TENSOR_SCALE_MAX);
  const float dmin_float_clamped = SAFE_MIN(SAFE_MAX(dmin_float, TENSOR_SCALE_MIN), TENSOR_SCALE_MAX);

  const uint8_t* scales_ptr = qblock->scales;
  const uint8_t* qs_ptr = qblock->qs;
  int weight_index = 0;
  float dequantized_scales[16];

  for (int i = 0; i < 8; ++i) {
    uint8_t packed_scales = scales_ptr[i];
    uint8_t scale_low = packed_scales & 0x0F;
    uint8_t scale_high = packed_scales >> 4;

    dequantized_scales[i * 2 + 0] =
        d_float_clamped * static_cast<float>(scale_low);
    dequantized_scales[i * 2 + 1] =
        d_float_clamped * static_cast<float>(scale_high);

    dequantized_scales[i * 2 + 0] =
        SAFE_MIN(SAFE_MAX(dequantized_scales[i * 2 + 0], TENSOR_SCALE_MIN), TENSOR_SCALE_MAX);
    dequantized_scales[i * 2 + 1] =
        SAFE_MIN(SAFE_MAX(dequantized_scales[i * 2 + 1], TENSOR_SCALE_MIN), TENSOR_SCALE_MAX);
  }

  weight_index = 0;
  for (int j = 0; j < GGML_QK_K / 16; ++j) {
    float sub_block_scale = dequantized_scales[j];

    const uint8_t* qs_subblock_ptr = qs_ptr + j * 4;

    for (int i = 0; i < 4; ++i) {
      uint8_t packed_weights = qs_subblock_ptr[i];

      uint8_t q0 = (packed_weights >> 0) & 0x03;
      uint8_t q1 = (packed_weights >> 2) & 0x03;
      uint8_t q2 = (packed_weights >> 4) & 0x03;
      uint8_t q3 = (packed_weights >> 6) & 0x03;

      float val0 =
          sub_block_scale * static_cast<float>(q0) + dmin_float_clamped;
      float val1 =
          sub_block_scale * static_cast<float>(q1) + dmin_float_clamped;
      float val2 =
          sub_block_scale * static_cast<float>(q2) + dmin_float_clamped;
      float val3 =
          sub_block_scale * static_cast<float>(q3) + dmin_float_clamped;

      val0 = SAFE_MIN(SAFE_MAX(val0, TENSOR_SCALE_MIN), TENSOR_SCALE_MAX);
      val1 = SAFE_MIN(SAFE_MAX(val1, TENSOR_SCALE_MIN), TENSOR_SCALE_MAX);
      val2 = SAFE_MIN(SAFE_MAX(val2, TENSOR_SCALE_MIN), TENSOR_SCALE_MAX);
      val3 = SAFE_MIN(SAFE_MAX(val3, TENSOR_SCALE_MIN), TENSOR_SCALE_MAX);

      output[weight_index++] = val0;

      output[weight_index++] = val1;

      output[weight_index++] = val2;

      output[weight_index++] = val3;
    }
  }
  assert(weight_index == GGML_QK_K);
}

void dequantize_q3_k(const void* qblock_void, float* output,
                     int num_weights_in_block) {
  if (num_weights_in_block != GGML_QK_K) {
    throw std::invalid_argument(
        "dequantize_q3_k currently only supports block size " +
        std::to_string(GGML_QK_K));
  }

  const block_q3_K* qblock = static_cast<const block_q3_K*>(qblock_void);

  const float d_float_raw = fp16_to_fp32(qblock->d);
  const float dmin_float_raw = fp16_to_fp32(qblock->dmin);

  const float d_float = (!std::isfinite(d_float_raw)) ? 0.0f : d_float_raw;
  const float dmin_float =
      (!std::isfinite(dmin_float_raw)) ? 0.0f : dmin_float_raw;

  const uint8_t* hmask_ptr = qblock->hmask;
  const uint8_t* qs_ptr = qblock->qs;
  const uint8_t* scales_ptr = qblock->scales;

  int weight_index = 0;

  for (int j = 0; j < GGML_QK_K / 16; ++j) {
    uint8_t scale_idx;

    if (j < 8) {
      scale_idx = scales_ptr[j] & 0x3F;
    } else {
      scale_idx = scales_ptr[j + 4] & 0x3F;
    }

    assert(scale_idx < 64 && "Scale index out of bounds for Q3_K lookup");
    const float sub_block_scale_factor = K_SCALE_VALUES[scale_idx];

    const float final_sub_block_scale = d_float * sub_block_scale_factor;
    const float final_sub_block_min = dmin_float;

    for (int i = 0; i < 4; ++i) {
      uint8_t qs_byte = qs_ptr[j * 4 + i];
      uint8_t hmask_byte = hmask_ptr[j];

      for (int bit_pos = 0; bit_pos < 8; bit_pos += 2) {
        uint8_t lower_bits = (qs_byte >> bit_pos) & 0x3;

        int hmask_bit_idx = (i * 4) + (bit_pos / 2);

        uint8_t high_bit = (hmask_byte >> hmask_bit_idx) & 0x1;

        uint8_t q_val = (high_bit << 2) | lower_bits;

        float val = final_sub_block_scale * static_cast<float>(q_val) +
                    final_sub_block_min;

        if (!std::isfinite(val)) {
          val = 0.0f;
        }

        output[weight_index++] = val;
      }
    }
  }

  if (weight_index != GGML_QK_K) {
    std::cout << "ERROR: Processed " << weight_index << " weights instead of "
              << GGML_QK_K << std::endl;

    while (weight_index < GGML_QK_K) {
      output[weight_index++] = 0.0f;
    }
  }
}

void quantize_q6_k(const float* input, void* output_qblock_void,
                   int num_elements) {
  if (num_elements != GGML_QK_K) {
    throw std::invalid_argument(
        "quantize_q6_k currently only supports block size " +
        std::to_string(GGML_QK_K));
  }

  block_q6_K* output_qblock = static_cast<block_q6_K*>(output_qblock_void);

  uint8_t* ql = output_qblock->ql;
  uint8_t* qh = output_qblock->qh;
  int8_t* scales = output_qblock->scales;
  std::memset(ql, 0, GGML_QK_K / 2);
  std::memset(qh, 0, GGML_QK_K / 4);

  float amax = 0.0f;
  for (int i = 0; i < num_elements; ++i) {
    amax = SAFE_MAX(amax, std::abs(input[i]));
  }

  const float d_float = (amax > GGUF_EPSILON) ? (amax / Q6K_SCALE_FACTOR) : GGUF_EPSILON;
  output_qblock->d = fp32_to_fp16(d_float);

  for (int sub = 0; sub < GGML_QK_K / 16; ++sub) {
    const float* sub_in = input + sub * 16;

    float sub_amax = 0.0f;
    for (int i = 0; i < 16; ++i) {
      sub_amax = SAFE_MAX(sub_amax, std::abs(sub_in[i]));
    }

    int8_t scale = (d_float > 0.0f) ? std::round(sub_amax / d_float) : 1;
    if (scale == 0) scale = 1;
    scales[sub] = scale;

    for (int i = 0; i < 16; ++i) {
      float val = sub_in[i];
      int q = static_cast<int>(std::round(val / (d_float * scale))) + Q6K_OFFSET;
      q = SAFE_MAX(0, SAFE_MIN(63, q));

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

const char* ggml_type_name(GGMLType type) {
  switch (type) {
    case GGMLType::GGML_TYPE_F32:
      return "F32";
    case GGMLType::GGML_TYPE_F16:
      return "F16";
    case GGMLType::GGML_TYPE_Q4_0:
      return "Q4_0";
    case GGMLType::GGML_TYPE_Q4_1:
      return "Q4_1";
    case GGMLType::GGML_TYPE_Q5_0:
      return "Q5_0";
    case GGMLType::GGML_TYPE_Q5_1:
      return "Q5_1";
    case GGMLType::GGML_TYPE_Q8_0:
      return "Q8_0";
    case GGMLType::GGML_TYPE_Q8_1:
      return "Q8_1";
    case GGMLType::GGML_TYPE_Q2_K:
      return "Q2_K";
    case GGMLType::GGML_TYPE_Q3_K:
      return "Q3_K";
    case GGMLType::GGML_TYPE_Q4_K:
      return "Q4_K";
    case GGMLType::GGML_TYPE_Q5_K:
      return "Q5_K";
    case GGMLType::GGML_TYPE_Q6_K:
      return "Q6_K";
    case GGMLType::GGML_TYPE_Q8_K:
      return "Q8_K";
    case GGMLType::GGML_TYPE_I8:
      return "I8";
    case GGMLType::GGML_TYPE_I16:
      return "I16";
    case GGMLType::GGML_TYPE_I32:
      return "I32";
    case GGMLType::GGML_TYPE_BF16:
      return "BF16";
    case GGMLType::GGML_TYPE_COUNT:
      return "COUNT";
    default:
      return "Unknown";
  }
}

size_t ggml_type_size(GGMLType type) {
  switch (type) {
    case GGMLType::GGML_TYPE_F32:
      return sizeof(float);
    case GGMLType::GGML_TYPE_F16:
      return sizeof(uint16_t);
    case GGMLType::GGML_TYPE_I8:
      return sizeof(int8_t);
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

    case GGMLType::GGML_TYPE_Q8_0:
      return 34;
    case GGMLType::GGML_TYPE_Q8_1:
      return 40;
    case GGMLType::GGML_TYPE_Q5_K:
      return 116;
    case GGMLType::GGML_TYPE_Q8_K:
      return 290;
    case GGMLType::GGML_TYPE_I16:
      return sizeof(int16_t);
    case GGMLType::GGML_TYPE_I32:
      return sizeof(int32_t);
    case GGMLType::GGML_TYPE_BF16:
      return sizeof(uint16_t);
    case GGMLType::GGML_TYPE_COUNT:
    default:
      std::cout << "  UNKNOWN GGML TYPE: " << static_cast<int>(type)
                << std::endl;
      throw std::invalid_argument("Unknown GGML type in ggml_type_size: " +
                                  std::to_string(static_cast<int>(type)));
  }
}

size_t ggml_type_block_size(GGMLType type) {
  switch (type) {
    case GGMLType::GGML_TYPE_Q2_K:
    case GGMLType::GGML_TYPE_Q3_K:
    case GGMLType::GGML_TYPE_Q4_K:
    case GGMLType::GGML_TYPE_Q6_K:

      return GGML_QK_K;

    case GGMLType::GGML_TYPE_Q4_0:
    case GGMLType::GGML_TYPE_Q8_0:

      return 32;

    case GGMLType::GGML_TYPE_F32:
    case GGMLType::GGML_TYPE_F16:
    case GGMLType::GGML_TYPE_I8:
    case GGMLType::GGML_TYPE_I16:
    case GGMLType::GGML_TYPE_I32:
    case GGMLType::GGML_TYPE_BF16:
      return 1;

    default:
      std::cout << "Warning: Unknown GGMLType in ggml_type_block_size: "
                << static_cast<int>(type) << std::endl;
      return 0;
  }

  return 0;
}

std::vector<block_q8_K> quantize_fp32_to_q8_K(
    const std::vector<float>& f_data) {
  if (f_data.size() % GGML_QK_K != 0) {
    throw std::runtime_error(
        "Input vector size must be a multiple of GGML_QK_K (" +
        std::to_string(GGML_QK_K) + ")");
  }

  size_t num_blocks = f_data.size() / GGML_QK_K;
  std::vector<block_q8_K> q_data(num_blocks);
  const float* x = f_data.data();
  block_q8_K* y = q_data.data();

  static std::atomic<int> log_count_q8k_quant_scales = 0;

  for (size_t i = 0; i < num_blocks; ++i) {
    float amax = 0.0f;
    for (int j = 0; j < GGML_QK_K; ++j) {
      amax = SAFE_MAX(amax, std::abs(x[j]));
    }

    const float d_fp32 = amax / Q8K_SCALE_FACTOR;
    const float id = (d_fp32 != 0.f) ? 1.0f / d_fp32 : 0.0f;
    y[i].d = fp32_to_fp16(d_fp32);

    if (log_count_q8k_quant_scales < 10) {
      std::stringstream q8k_scale_log_ss;
      q8k_scale_log_ss << "[Q8K_QUANT_SCALES] Block #" << i
                       << " Input amax=" << amax << " -> d_fp32=" << d_fp32
                       << " -> Stored d_fp16=0x" << std::hex << y[i].d
                       << std::dec;
      Logger::debug(q8k_scale_log_ss.str());
      log_count_q8k_quant_scales++;
    }

    int16_t block_sum[16] = {0};
    for (int j = 0; j < GGML_QK_K; ++j) {
      const float val_scaled = x[j] * id;

      int8_t q_val = static_cast<int8_t>(
          SAFE_MAX(-128.0f, SAFE_MIN(127.0f, std::round(val_scaled))));
      y[i].qs[j] = q_val;
      block_sum[j / 16] += q_val;
    }

    std::memcpy(y[i].bsums, block_sum, sizeof(block_sum));

    x += GGML_QK_K;
  }

  return q_data;
}

float vec_dot_q6_k_q8_k_cpu(int n, const std::vector<block_q6_K>& x_vec,
                            const std::vector<block_q8_K>& y_vec,
                            bool log_this_call) {
  if (n % GGML_QK_K != 0) {
    throw std::runtime_error("vec_dot_q6_k_q8_k: n must be multiple of QK_K");
  }
  size_t nb = n / GGML_QK_K;
  if (x_vec.size() != nb || y_vec.size() != nb) {
    throw std::runtime_error("vec_dot_q6_k_q8_k: vector block count mismatch");
  }

  const block_q6_K* x = x_vec.data();
  const block_q8_K* y = y_vec.data();

  int8_t aux8[GGML_QK_K];
  int16_t aux16[8];
  float sums[8];
  int32_t aux32[8];
  std::memset(sums, 0, 8 * sizeof(float));

  float sumf = 0.0f;

  static std::atomic<int> log_count_dot = 0;
  bool should_log_this_block = log_this_call && log_count_dot < 5;

  for (size_t i = 0; i < nb; ++i) {
    const uint8_t* ql = x[i].ql;
    const uint8_t* qh = x[i].qh;
    const int8_t* q8 = y[i].qs;
    std::memset(aux32, 0, 8 * sizeof(int32_t));

    int8_t* a = aux8;
    for (int j = 0; j < GGML_QK_K; j += 128) {
      for (int l = 0; l < 32; ++l) {
        a[l + 0] = static_cast<int8_t>(
            ((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32);
        a[l + 32] = static_cast<int8_t>(
            ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32);
        a[l + 64] = static_cast<int8_t>(
            ((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32);
        a[l + 96] = static_cast<int8_t>(
            ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
      }
      a += 128;
      ql += 64;
      qh += 32;
    }

    a = aux8;
    int is = 0;
    for (int j = 0; j < GGML_QK_K / 16; ++j) {
      int scale = x[i].scales[is++];
      for (int l = 0; l < 8; ++l)
        aux16[l] = static_cast<int16_t>(q8[l]) * static_cast<int16_t>(a[l]);
      for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
      q8 += 8;
      a += 8;
      for (int l = 0; l < 8; ++l)
        aux16[l] = static_cast<int16_t>(q8[l]) * static_cast<int16_t>(a[l]);
      for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
      q8 += 8;
      a += 8;
    }

    int32_t sumi_mins = 0;
    for (int j = 0; j < GGML_QK_K / 16; ++j) {
      sumi_mins += static_cast<int32_t>(y[i].bsums[j]) *
                   static_cast<int32_t>(x[i].scales[j]);
    }

    const float d_q6 = fp16_to_fp32(x[i].d);
    const float d_q8 = fp16_to_fp32(y[i].d);
    const float d = d_q6 * d_q8;

    float block_contribution = 0.0f;
    for (int l = 0; l < 8; ++l) {
      float term = d * (aux32[l] - 32 * sumi_mins / 8);
      sums[l] += term;
      block_contribution += term;
    }

    if (i == 0 && should_log_this_block) {
      std::stringstream ss_log;
      ss_log << "[DOT_Q6K_Q8K] Call #" << (log_count_dot.load() + 1)
             << ", Block #0:";
      Logger::debug(ss_log.str());
      ss_log.str("");
      ss_log << "  Q6_K Scale d_q6: " << d_q6 << " (Raw FP16: 0x" << std::hex
             << x[i].d << std::dec << ")";
      Logger::debug(ss_log.str());
      ss_log.str("");
      ss_log << "  Q8_K Scale d_q8: " << d_q8;
      Logger::debug(ss_log.str());
      ss_log.str("");
      ss_log << "  Combined Scale d: " << d;
      Logger::debug(ss_log.str());
      ss_log.str("");
      ss_log << "  Q6_K Sub-scales (int8): ";
      for (int k = 0; k < 16; ++k) ss_log << (int)x[i].scales[k] << " ";
      Logger::debug(ss_log.str());
      ss_log.str("");
      ss_log << "  Int32 Sums (aux32, before compensation): ";
      for (int l = 0; l < 8; ++l) ss_log << aux32[l] << " ";
      Logger::debug(ss_log.str());
      ss_log.str("");
      ss_log << "  Compensation term (sumi_mins): " << sumi_mins
             << ", -32 * sumi_mins: " << (-32 * sumi_mins);
      Logger::debug(ss_log.str());
      ss_log.str("");
      ss_log << "  Block #0 Contribution to Sums (after compensation): "
             << block_contribution;
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

void matvec_q6k_q8k_cpu(const std::vector<block_q6_K>& mat_q6k,
                        const std::vector<block_q8_K>& vec_q8k,
                        std::vector<float>& out_f32, int rows, int cols,
                        bool log_calls) {
  if (cols % GGML_QK_K != 0) {
    throw std::runtime_error(
        "matvec_q6k_q8k_cpu: cols must be divisible by GGML_QK_K");
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
    const std::vector<block_q6_K> row_q6k(
        mat_q6k.begin() + r * blocks_per_row,
        mat_q6k.begin() + (r + 1) * blocks_per_row);

    out_f32[r] = vec_dot_q6_k_q8_k_cpu(cols, row_q6k, vec_q8k, log_calls);
  }
}

float vec_dot_q4_k_q8_k_cpu(int n, const std::vector<block_q4_K>& x_vec,
                            const std::vector<block_q8_K>& y_vec,
                            bool log_this_call) {
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
    int8_t q4_vals[GGML_QK_K];
    const uint8_t* q4 = x[i].qs;
    for (int j = 0; j < GGML_QK_K / 2; ++j) {
      q4_vals[2 * j + 0] = static_cast<int8_t>(q4[j] & 0xF);
      q4_vals[2 * j + 1] = static_cast<int8_t>(q4[j] >> 4);
    }

    const int8_t* q8 = y[i].qs;

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

    if (i == 0 && log_this_call) {
      std::stringstream ss;
      ss << "[Q4K_Q8K] Block #0: d: " << fp16_to_fp32(x[i].d)
         << ", dmin: " << fp16_to_fp32(x[i].dmin);
      Logger::debug(ss.str());
      ss.str("");
      ss << "[Q4K_Q8K] Block #0: Q8_K input (first 16): ";
      for (int k = 0; k < 16; ++k) ss << (int)q8[k] << " ";
      Logger::debug(ss.str());
      ss.str("");
      ss << "[Q4K_Q8K] Block #0: Q4_K unpacked (first 16): ";
      for (int k = 0; k < 16; ++k) ss << (int)q4_vals[k] << " ";
      Logger::debug(ss.str());
      ss.str("");
    }
  }
  return sumf;
}

void matvec_q4k_q8k_cpu(const std::vector<block_q4_K>& mat_q4k,
                        const std::vector<block_q8_K>& vec_q8k,
                        std::vector<float>& out_f32, int rows, int cols,
                        bool log_calls) {
  if (cols % GGML_QK_K != 0) {
    throw std::runtime_error(
        "matvec_q4k_q8k_cpu: cols must be divisible by GGML_QK_K");
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
    const std::vector<block_q4_K> row_q4k(
        mat_q4k.begin() + r * blocks_per_row,
        mat_q4k.begin() + (r + 1) * blocks_per_row);

    out_f32[r] = vec_dot_q4_k_q8_k_cpu(cols, row_q4k, vec_q8k, log_calls);
  }
}

void dequantize_q8_k(const std::vector<block_q8_K>& q_data,
                     std::vector<float>& x, int n, bool log_this_block) {
  if (n % GGML_QK_K != 0) {
    std::cerr
        << "Error: n must be a multiple of GGML_QK_K for Q8_K dequantization."
        << std::endl;
    return;
  }
  size_t num_blocks = n / GGML_QK_K;
  if (q_data.size() < num_blocks) {
    std::cerr << "Error: Not enough Q8_K blocks provided for dequantization."
              << std::endl;
    return;
  }

  static std::atomic<int> log_count_q8k_dequant_scales = 0;

  for (size_t i = 0; i < num_blocks; ++i) {
    const block_q8_K* qblock = &q_data[i];
    float* x_block = &x[i * GGML_QK_K];

    const float d = fp16_to_fp32(qblock->d, true);

    if (log_this_block && log_count_q8k_dequant_scales < 10) {
      std::stringstream scale_log_ss;
      scale_log_ss << "[Q8K_DEQUANT_SCALES] Block #"
                   << (log_count_q8k_dequant_scales.load()) << " Raw_d_fp16=0x"
                   << std::hex << qblock->d << std::dec << " -> d=" << d;
      Logger::debug(scale_log_ss.str());
      log_count_q8k_dequant_scales++;
    }

    for (int j = 0; j < GGML_QK_K; ++j) {
      x_block[j] = d * static_cast<float>(qblock->qs[j]);
    }
  }
}

void dequantize_q8_0_block(const block_q8_0* qblock, float* output) {
  const float d_fp32 = fp16_to_fp32(qblock->d, true);
  for (int i = 0; i < GGML_QK8_0; ++i) {
    output[i] = d_fp32 * static_cast<float>(qblock->qs[i]);
  }
}
