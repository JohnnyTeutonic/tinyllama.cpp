#include "utils.h"

#include "logger.h"
#include "quantization.h"
#include "model_constants.h"
#include "model_macros.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <numeric>

// SIMD intrinsics headers for optimized computation
#if defined(__AVX2__)
#include <immintrin.h>
#define SIMD_WIDTH 8
#elif defined(__SSE2__)
#include <emmintrin.h>
#define SIMD_WIDTH 4
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#define SIMD_WIDTH 4
#endif

// SIMD optimized dot product functions
float simd_dot_product(const float* a, const float* b, int n) {
#if defined(__AVX2__)
    __m256 sum = _mm256_setzero_ps();
    int i = 0;
    for (; i <= n - 8; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        sum = _mm256_fmadd_ps(va, vb, sum);
    }
    float result[8];
    _mm256_storeu_ps(result, sum);
    float final_sum = result[0] + result[1] + result[2] + result[3] + 
                      result[4] + result[5] + result[6] + result[7];
    for (; i < n; ++i) {
        final_sum += a[i] * b[i];
    }
    return final_sum;
#elif defined(__SSE2__)
    __m128 sum = _mm_setzero_ps();
    int i = 0;
    for (; i <= n - 4; i += 4) {
        __m128 va = _mm_loadu_ps(&a[i]);
        __m128 vb = _mm_loadu_ps(&b[i]);
        sum = _mm_add_ps(sum, _mm_mul_ps(va, vb));
    }
    float result[4];
    _mm_storeu_ps(result, sum);
    float final_sum = result[0] + result[1] + result[2] + result[3];
    for (; i < n; ++i) {
        final_sum += a[i] * b[i];
    }
    return final_sum;
#elif defined(__ARM_NEON)
    float32x4_t sum = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i <= n - 4; i += 4) {
        float32x4_t va = vld1q_f32(&a[i]);
        float32x4_t vb = vld1q_f32(&b[i]);
        sum = vmlaq_f32(sum, va, vb);
    }
    float result[4];
    vst1q_f32(result, sum);
    float final_sum = result[0] + result[1] + result[2] + result[3];
    for (; i < n; ++i) {
        final_sum += a[i] * b[i];
    }
    return final_sum;
#else
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
#endif
}

// SIMD optimized scaled vector addition
void simd_scaled_add(float* dst, const float* src, float scale, int n) {
#if defined(__AVX2__)
    __m256 vscale = _mm256_set1_ps(scale);
    int i = 0;
    for (; i <= n - 8; i += 8) {
        __m256 vdst = _mm256_loadu_ps(&dst[i]);
        __m256 vsrc = _mm256_loadu_ps(&src[i]);
        __m256 result = _mm256_fmadd_ps(vsrc, vscale, vdst);
        _mm256_storeu_ps(&dst[i], result);
    }
    for (; i < n; ++i) {
        dst[i] += src[i] * scale;
    }
#elif defined(__SSE2__)
    __m128 vscale = _mm_set1_ps(scale);
    int i = 0;
    for (; i <= n - 4; i += 4) {
        __m128 vdst = _mm_loadu_ps(&dst[i]);
        __m128 vsrc = _mm_loadu_ps(&src[i]);
        __m128 result = _mm_add_ps(vdst, _mm_mul_ps(vsrc, vscale));
        _mm_storeu_ps(&dst[i], result);
    }
    for (; i < n; ++i) {
        dst[i] += src[i] * scale;
    }
#elif defined(__ARM_NEON)
    float32x4_t vscale = vdupq_n_f32(scale);
    int i = 0;
    for (; i <= n - 4; i += 4) {
        float32x4_t vdst = vld1q_f32(&dst[i]);
        float32x4_t vsrc = vld1q_f32(&src[i]);
        float32x4_t result = vmlaq_f32(vdst, vsrc, vscale);
        vst1q_f32(&dst[i], result);
    }
    for (; i < n; ++i) {
        dst[i] += src[i] * scale;
    }
#else
    for (int i = 0; i < n; ++i) {
        dst[i] += src[i] * scale;
    }
#endif
}

uint16_t float32_to_bfloat16(float val) {
  uint32_t bits;
  std::memcpy(&bits, &val, sizeof(float));

  bits += 0x7FFF + ((bits >> 16) & 1);
  return static_cast<uint16_t>(bits >> 16);
}

float bfloat16_to_float32(uint16_t bf16) {
  if (bf16 == bfloat16::ZERO) return 0.0f;
  if (bf16 == bfloat16::NEG_ZERO) return -0.0f;

  bool is_nan = ((bf16 & bfloat16::EXPONENT_MASK) == bfloat16::EXPONENT_MASK) && 
                ((bf16 & bfloat16::MANTISSA_MASK) != 0);
  if (is_nan) return std::numeric_limits<float>::quiet_NaN();

  if ((bf16 & bfloat16::EXPONENT_MASK) == bfloat16::EXPONENT_MASK && 
      (bf16 & bfloat16::MANTISSA_MASK) == 0) {
    return (bf16 & bfloat16::SIGN_BIT) ? -std::numeric_limits<float>::infinity()
                                      : std::numeric_limits<float>::infinity();
  }

  uint32_t bits = static_cast<uint32_t>(bf16) << bfloat16::SHIFT_BITS;
  float result;
  std::memcpy(&result, &bits, sizeof(float));

  return result;
}

std::vector<float> bfloat16_vector_to_float32(const std::vector<uint16_t>& bf16_vec) {
  std::vector<float> f32_vec(bf16_vec.size());

#pragma omp parallel for
  for (int64_t i = 0; i < static_cast<int64_t>(bf16_vec.size()); ++i) {
    f32_vec[i] = bfloat16_to_float32(bf16_vec[i]);
  }

  return f32_vec;
}

std::vector<uint16_t> uint8_vector_to_uint16_vector(const std::vector<uint8_t>& bytes, size_t numel) {
  if (bytes.size() != numel * 2) {
    throw std::runtime_error("Byte vector size mismatch for uint16_t conversion");
  }
  std::vector<uint16_t> out(numel);
  std::memcpy(out.data(), bytes.data(), bytes.size());
  return out;
}

int argmax(const std::vector<float>& v) {
  if (v.empty()) {
    Logger::error("Cannot perform argmax on empty vector");
    return -1;
  }
  auto max_it = std::max_element(v.begin(), v.end());
  float max_val = *max_it;
  int max_idx = std::distance(v.begin(), max_it);
  Logger::debug("[ARGMAX HELPER] Max value found: " + std::to_string(max_val) +
                " at index: " + std::to_string(max_idx));
  return max_idx;
}

std::vector<float> bf16vec_to_float_vec(const std::vector<uint16_t>& v_bf16) {
  std::vector<float> v_f32(v_bf16.size());
#pragma omp parallel for
  for (int64_t i = 0; i < static_cast<int64_t>(v_bf16.size()); ++i) {
    v_f32[i] = bfloat16_to_float32(v_bf16[i]);
  }
  return v_f32;
}

void log_vector_summary(const std::string& name, const std::vector<float>& v, int head_count) {
  if (v.empty()) {
    Logger::info(name + ": EMPTY");
    return;
  }
  std::stringstream ss;
  size_t actual_head_count = SAFE_MIN(static_cast<size_t>(head_count), v.size());

  ss << name << ": size=" << v.size();

  if (actual_head_count > 0) {
    ss << ", first " << actual_head_count << ": [";
    for (size_t i = 0; i < actual_head_count; ++i) {
      ss << (i > 0 ? " " : "") << std::fixed << std::setprecision(4) << v[i];
    }
    ss << "]";
  }
  float minv = *std::min_element(v.begin(), v.end());
  float maxv = *std::max_element(v.begin(), v.end());
  double sum = std::accumulate(v.begin(), v.end(), 0.0);
  float mean = sum / v.size();
  bool all_finite = std::all_of(v.begin(), v.end(), [](float x) { return std::isfinite(x); });
  ss << ", min=" << minv << ", max=" << maxv << ", mean=" << mean
     << ", finite=" << (all_finite ? "yes" : "NO");
  Logger::info(ss.str());
}

void log_vector_summary_with_tail(const std::string& name, const std::vector<float>& v, 
                                   int head_count, int tail_count) {
  if (v.empty()) {
    Logger::info(name + ": EMPTY");
    return;
  }
  std::stringstream ss;

  size_t actual_head_count = SAFE_MIN(static_cast<size_t>(head_count), v.size());
  size_t actual_tail_count = SAFE_MIN(static_cast<size_t>(tail_count), v.size());
  size_t total_shown = actual_head_count + actual_tail_count;
  bool overlap = total_shown > v.size();
  if (overlap) {
    actual_tail_count = v.size() - actual_head_count;
    if (actual_tail_count > SAFE_MIN(static_cast<size_t>(tail_count), v.size())) {
      actual_tail_count = SAFE_MIN(static_cast<size_t>(tail_count), v.size());
    }
    if (tail_count > 0 && actual_head_count == v.size()) {
      actual_tail_count = 0;
    }
  }
  size_t tail_start_index = v.size() - actual_tail_count;

  ss << name << ": size=" << v.size();

  if (actual_head_count > 0) {
    ss << ", first " << actual_head_count << ": [";
    for (size_t i = 0; i < actual_head_count; ++i) {
      ss << (i > 0 ? " " : "") << std::fixed << std::setprecision(4) << v[i];
    }
    ss << "]";
  }

  if (actual_tail_count > 0 && tail_start_index >= actual_head_count) {
    ss << ", last " << actual_tail_count << ": [";
    for (size_t i = 0; i < actual_tail_count; ++i) {
      ss << (i > 0 ? " " : "") << std::fixed << std::setprecision(4)
         << v[tail_start_index + i];
    }
    ss << "]";
  } else if (overlap && tail_count > 0 && actual_head_count < v.size()) {
    ss << " (... tail overlaps head ...)";
  }

  float minv = *std::min_element(v.begin(), v.end());
  float maxv = *std::max_element(v.begin(), v.end());
  double sum = std::accumulate(v.begin(), v.end(), 0.0);
  float mean = sum / v.size();
  bool all_finite = std::all_of(v.begin(), v.end(), [](float x) { return std::isfinite(x); });
  ss << ", min=" << minv << ", max=" << maxv << ", mean=" << mean
     << ", finite=" << (all_finite ? "yes" : "NO");
  Logger::info(ss.str());
}





// Matrix-vector multiplication functions
void matvec_q8_0_f32_vector_cpu(const std::vector<block_q8_0>& mat_q8_0,
                                const std::vector<float>& vec_f32,
                                std::vector<float>& out_f32, int rows,
                                int cols, bool log_first_block) {
  if (cols % GGML_QK8_0 != 0) {
    throw std::runtime_error(
        "matvec_q8_0_f32_vector_cpu: cols (" + std::to_string(cols) +
        ") must be divisible by GGML_QK8_0 (" + std::to_string(GGML_QK8_0) + ")");
  }
  if (vec_f32.size() != static_cast<size_t>(cols)) {
    throw std::runtime_error(
        "matvec_q8_0_f32_vector_cpu: vec_f32 size mismatch. Expected " +
        std::to_string(cols) + ", got " + std::to_string(vec_f32.size()));
  }
  size_t num_blocks_per_row = cols / GGML_QK8_0;
  size_t total_blocks_expected = static_cast<size_t>(rows) * num_blocks_per_row;
  if (mat_q8_0.size() != total_blocks_expected) {
    throw std::runtime_error(
        "matvec_q8_0_f32_vector_cpu: mat_q8_0 size mismatch. Expected " +
        std::to_string(total_blocks_expected) + " blocks, got " +
        std::to_string(mat_q8_0.size()));
  }

  out_f32.resize(rows);
  float dequantized_block[GGML_QK8_0];
  
  if (log_first_block) {
    Logger::info("[MATVEC_Q8_0_DEBUG] Matrix shape: " + std::to_string(rows) + "x" + std::to_string(cols) + 
                 ", blocks_per_row=" + std::to_string(num_blocks_per_row) + 
                 ", total_blocks=" + std::to_string(mat_q8_0.size()) +
                 ", input_vec_size=" + std::to_string(vec_f32.size()));
  }

#pragma omp parallel for private(dequantized_block)
  for (int64_t r = 0; r < static_cast<int64_t>(rows); ++r) {
    double row_sum = 0.0;
    double kahan_c = 0.0;

    size_t block_row_offset = static_cast<size_t>(r) * num_blocks_per_row;

    for (size_t block_col_idx = 0; block_col_idx < num_blocks_per_row; ++block_col_idx) {
      const block_q8_0* qblock = &mat_q8_0[block_row_offset + block_col_idx];
      dequantize_q8_0_block(qblock, dequantized_block);

      size_t vec_offset = block_col_idx * GGML_QK8_0;
      
      if (log_first_block && r < 2 && block_col_idx < 2) {
        Logger::info("[MATVEC_Q8_0_INNER] Row " + std::to_string(r) + 
                     " Block " + std::to_string(block_col_idx) + 
                     " scale=" + std::to_string(fp16_to_fp32(qblock->d, true)) +
                     " first_4_quant=[" + std::to_string(qblock->qs[0]) + 
                     ", " + std::to_string(qblock->qs[1]) + 
                     ", " + std::to_string(qblock->qs[2]) + 
                     ", " + std::to_string(qblock->qs[3]) + "]" +
                     " first_4_dequant=[" + std::to_string(dequantized_block[0]) + 
                     ", " + std::to_string(dequantized_block[1]) + 
                     ", " + std::to_string(dequantized_block[2]) + 
                     ", " + std::to_string(dequantized_block[3]) + "]" +
                     " first_4_input=[" + std::to_string(vec_f32[vec_offset]) + 
                     ", " + std::to_string(vec_f32[vec_offset + 1]) + 
                     ", " + std::to_string(vec_f32[vec_offset + 2]) + 
                     ", " + std::to_string(vec_f32[vec_offset + 3]) + "]");
      }
      
      for (int i = 0; i < GGML_QK8_0; ++i) {
        double term = static_cast<double>(dequantized_block[i]) *
                      static_cast<double>(vec_f32[vec_offset + i]);

        double y = term - kahan_c;
        double t = row_sum + y;
        kahan_c = (t - row_sum) - y;
        row_sum = t;
      }
    }
    out_f32[r] = static_cast<float>(row_sum);
    
    if (log_first_block && r < 3) {
      Logger::info("[MATVEC_Q8_0_OUTPUT] Row " + std::to_string(r) + 
                   " final_sum=" + std::to_string(row_sum) + 
                   " output=" + std::to_string(out_f32[r]));
    }
  }
}

void matvec_f32_f32_vector_cpu(const std::vector<float>& mat_f32,
                              const std::vector<float>& vec_f32,
                              std::vector<float>& out_f32, int rows,
                              int cols) {
  if (mat_f32.empty() || vec_f32.empty()) {
    Logger::error(
        "matvec_f32_f32_vector_cpu: Input matrix or vector is empty.");
    out_f32.assign(rows, 0.0f);
    return;
  }
  if (mat_f32.size() != (size_t)rows * cols) {
    Logger::error(
        "matvec_f32_f32_vector_cpu: Matrix dimensions mismatch. Expected " +
        std::to_string((size_t)rows * cols) + ", got " +
        std::to_string(mat_f32.size()));
    out_f32.assign(rows, 0.0f);
    return;
  }
  if (vec_f32.size() != (size_t)cols) {
    Logger::error(
        "matvec_f32_f32_vector_cpu: Vector dimension mismatch. Expected " +
        std::to_string(cols) + ", got " + std::to_string(vec_f32.size()));
    out_f32.assign(rows, 0.0f);
    return;
  }

  out_f32.resize(rows);

#pragma omp parallel for schedule(static)
  for (int64_t r = 0; r < static_cast<int64_t>(rows); ++r) {
    float sum = 0.0f;
    size_t row_offset = static_cast<size_t>(r) * cols;

    const float* mat_row_ptr = mat_f32.data() + row_offset;
    const float* vec_ptr = vec_f32.data();

    double k_sum = 0.0;
    double k_c = 0.0;

    for (int c = 0; c < cols; ++c) {
      double term = static_cast<double>(mat_row_ptr[c]) * static_cast<double>(vec_ptr[c]);
      double y = term - k_c;
      double t = k_sum + y;
      k_c = (t - k_sum) - y;
      k_sum = t;
    }
    out_f32[r] = static_cast<float>(k_sum);
  }
}

void matvec_q8k_f32_vector_cpu(const std::vector<block_q8_K>& mat_q8k,
                              const std::vector<float>& vec_f32,
                              std::vector<float>& out_f32, int rows,
                              int cols, bool log_first_block) {
  if (cols % GGML_QK_K != 0) {
    throw std::runtime_error("matvec_q8k_f32_vector_cpu: cols must be divisible by GGML_QK_K");
  }
  
  size_t num_blocks_per_row = cols / GGML_QK_K;
  size_t total_blocks_expected = (size_t)rows * num_blocks_per_row;
  if (mat_q8k.size() != total_blocks_expected) {
    throw std::runtime_error("matvec_q8k_f32_vector_cpu: mat_q8k size mismatch");
  }
  if (vec_f32.size() != (size_t)cols) {
    throw std::runtime_error("matvec_q8k_f32_vector_cpu: vec_f32 size mismatch");
  }
  
  out_f32.resize(rows);
  
  std::vector<float> mat_f32;
  dequantize_q8_k(mat_q8k, mat_f32, rows * cols, log_first_block);
  
  matvec_f32_f32_vector_cpu(mat_f32, vec_f32, out_f32, rows, cols);
  
  if (log_first_block && rows > 0) {
    Logger::info("[Q8K_MATVEC_DEBUG] First output: " + std::to_string(out_f32[0]));
  }
}

void apply_rope_vector(
    std::vector<float>& x,
    int num_heads,
    int head_dim,
    int current_token_pos,
    const std::vector<std::pair<float, float>>& all_freqs_cis,
    int max_pos_embeddings,
    bool use_adjacent_pairing
) {
  if (current_token_pos < 0 || current_token_pos >= max_pos_embeddings) {
    return;
  }
  if (head_dim % 2 != 0) { 
    Logger::error("RoPE apply_rope_vector: head_dim must be even. head_dim: " + std::to_string(head_dim));
    return;
  }

  const int dim_half = head_dim / 2;
  size_t pos_offset = static_cast<size_t>(current_token_pos) * static_cast<size_t>(dim_half);

  for (int h = 0; h < num_heads; ++h) {
    size_t head_offset = static_cast<size_t>(h) * head_dim;

    for (int i = 0; i < dim_half; ++i) {
      size_t freq_idx = pos_offset + static_cast<size_t>(i);
      
      if (freq_idx >= all_freqs_cis.size()) {
            Logger::warning("RoPE apply_rope_vector: freq_idx out of bounds. pos: " + 
                            std::to_string(current_token_pos) + ", head_dim/2: " + std::to_string(dim_half) +
                            ", i: " + std::to_string(i) + ", calculated freq_idx: " + std::to_string(freq_idx) +
                            ", all_freqs_cis.size(): " + std::to_string(all_freqs_cis.size()));
            continue;
          }

      float cos_theta = all_freqs_cis[freq_idx].first;
      float sin_theta = all_freqs_cis[freq_idx].second;
      
      float x0_val, x1_val;
      size_t x0_idx, x1_idx;

      if (use_adjacent_pairing) {
        x0_idx = head_offset + (2 * i);
        x1_idx = head_offset + (2 * i + 1);
      } else {
        x0_idx = head_offset + i;
        x1_idx = head_offset + i + dim_half;
      }

      if (x0_idx >= x.size() || x1_idx >= x.size()) {
          Logger::warning("RoPE apply_rope_vector: x index out of bounds. x.size(): " + std::to_string(x.size()) +
                          ", x0_idx: " + std::to_string(x0_idx) + ", x1_idx: " + std::to_string(x1_idx));
          continue;
      }

      x0_val = x[x0_idx];
      x1_val = x[x1_idx];
      
      x[x0_idx] = x0_val * cos_theta - x1_val * sin_theta;
      x[x1_idx] = x0_val * sin_theta + x1_val * cos_theta;
    }
  }
}

void matmul_q4k_f32_batch_cpu(
    const std::vector<block_q4_K>& mat_q4k,
    const std::vector<float>& batch_input_activations,
    std::vector<float>& batch_output_activations,
    int num_tokens,
    int output_dim,
    int input_dim
) {
    if (mat_q4k.empty() || batch_input_activations.empty()) {
        Logger::error("[MATMUL_Q4K_BATCH_CPU] Input matrix or batch_input_activations is empty.");
        batch_output_activations.assign((size_t)num_tokens * output_dim, 0.0f);
        return;
    }
    if (batch_input_activations.size() != (size_t)num_tokens * input_dim) {
        Logger::error("[MATMUL_Q4K_BATCH_CPU] batch_input_activations size mismatch. Expected " +
                      std::to_string((size_t)num_tokens * input_dim) + ", got " +
                      std::to_string(batch_input_activations.size()));
        batch_output_activations.assign((size_t)num_tokens * output_dim, 0.0f);
        return;
    }

    batch_output_activations.resize((size_t)num_tokens * output_dim);

#pragma omp parallel for
    for (int token_idx = 0; token_idx < num_tokens; ++token_idx) {
        std::vector<float> current_token_input(input_dim);
        const float* input_slice_start = batch_input_activations.data() + (size_t)token_idx * input_dim;
        std::copy(input_slice_start, input_slice_start + input_dim, current_token_input.begin());

        std::vector<float> current_token_output(output_dim);
        matvec_q4k_f32_vector_cpu(mat_q4k, current_token_input, current_token_output, output_dim, input_dim, false); 
        
        float* output_slice_start = batch_output_activations.data() + (size_t)token_idx * output_dim;
        std::copy(current_token_output.begin(), current_token_output.end(), output_slice_start);
    }
}

void matmul_q6k_f32_batch_cpu(
    const std::vector<block_q6_K>& mat_q6k,
    const std::vector<float>& batch_input_activations,
    std::vector<float>& batch_output_activations,
    int num_tokens,
    int output_dim,
    int input_dim
) {
    if (mat_q6k.empty() || batch_input_activations.empty()) {
        Logger::error("[MATMUL_Q6K_BATCH_CPU] Input matrix or batch_input_activations is empty.");
        batch_output_activations.assign((size_t)num_tokens * output_dim, 0.0f);
        return;
    }

    if (batch_input_activations.size() != (size_t)num_tokens * input_dim) {
        Logger::error("[MATMUL_Q6K_BATCH_CPU] batch_input_activations size mismatch. Expected " +
                      std::to_string((size_t)num_tokens * input_dim) + ", got " +
                      std::to_string(batch_input_activations.size()));
        batch_output_activations.assign((size_t)num_tokens * output_dim, 0.0f);
        return;
    }

    batch_output_activations.resize((size_t)num_tokens * output_dim);

#pragma omp parallel for
    for (int token_idx = 0; token_idx < num_tokens; ++token_idx) {
        std::vector<float> current_token_input(input_dim);
        const float* input_slice_start = batch_input_activations.data() + (size_t)token_idx * input_dim;
        std::copy(input_slice_start, input_slice_start + input_dim, current_token_input.begin());

        std::vector<float> current_token_output(output_dim);
        matvec_q6k_f32_vector_cpu(mat_q6k, current_token_input, current_token_output, output_dim, input_dim, false); 
        
        float* output_slice_start = batch_output_activations.data() + (size_t)token_idx * output_dim;
        std::copy(current_token_output.begin(), current_token_output.end(), output_slice_start);
    }
}

void matmul_q8_0_f32_batch_cpu(
    const std::vector<block_q8_0>& mat_q8_0,
    const std::vector<float>& batch_input_activations,
    std::vector<float>& batch_output_activations,
    int num_tokens,
    int output_dim,
    int input_dim
) {
    if (mat_q8_0.empty() || batch_input_activations.empty()) {
        Logger::error("[MATMUL_Q8_0_BATCH_CPU] Input matrix or batch_input_activations is empty.");
        batch_output_activations.assign((size_t)num_tokens * output_dim, 0.0f);
        return;
    }

    if (batch_input_activations.size() != (size_t)num_tokens * input_dim) {
        Logger::error("[MATMUL_Q8_0_BATCH_CPU] batch_input_activations size mismatch. Expected " +
                      std::to_string((size_t)num_tokens * input_dim) + ", got " +
                      std::to_string(batch_input_activations.size()));
        batch_output_activations.assign((size_t)num_tokens * output_dim, 0.0f);
        return;
    }

    batch_output_activations.resize((size_t)num_tokens * output_dim);

#pragma omp parallel for
    for (int token_idx = 0; token_idx < num_tokens; ++token_idx) {
        std::vector<float> current_token_input(input_dim);
        const float* input_slice_start = batch_input_activations.data() + (size_t)token_idx * input_dim;
        std::copy(input_slice_start, input_slice_start + input_dim, current_token_input.begin());

        std::vector<float> current_token_output(output_dim);
        matvec_q8_0_f32_vector_cpu(mat_q8_0, current_token_input, current_token_output, output_dim, input_dim, false);
        
        float* output_slice_start = batch_output_activations.data() + (size_t)token_idx * output_dim;
        std::copy(current_token_output.begin(), current_token_output.end(), output_slice_start);
    }
}

void matmul_q8k_f32_batch_cpu(
    const std::vector<block_q8_K>& mat_q8k,
    const std::vector<float>& batch_input_activations,
    std::vector<float>& batch_output_activations,
    int num_tokens,
    int output_dim,
    int input_dim
) {
    if (input_dim % GGML_QK_K != 0) {
        throw std::runtime_error("matmul_q8k_f32_batch_cpu: input_dim (" + std::to_string(input_dim) + 
                                 ") must be divisible by GGML_QK_K (" + std::to_string(GGML_QK_K) + ")");
    }
    
    size_t expected_input_size = (size_t)num_tokens * input_dim;
    if (batch_input_activations.size() != expected_input_size) {
        throw std::runtime_error("matmul_q8k_f32_batch_cpu: batch_input_activations size mismatch. Expected " +
                                 std::to_string(expected_input_size) + ", got " + std::to_string(batch_input_activations.size()));
    }
    
    size_t num_blocks_per_row = input_dim / GGML_QK_K;
    size_t total_blocks_expected = (size_t)output_dim * num_blocks_per_row;
    if (mat_q8k.size() != total_blocks_expected) {
        throw std::runtime_error("matmul_q8k_f32_batch_cpu: mat_q8k size mismatch. Expected " +
                                 std::to_string(total_blocks_expected) + " blocks, got " + std::to_string(mat_q8k.size()));
    }
    
    batch_output_activations.resize((size_t)num_tokens * output_dim);
    
    for (int t = 0; t < num_tokens; ++t) {
        std::vector<float> current_token_input(input_dim);
        for (int i = 0; i < input_dim; ++i) {
            current_token_input[i] = batch_input_activations[t * input_dim + i];
        }
        
        std::vector<float> current_token_output(output_dim);
        matvec_q8k_f32_vector_cpu(mat_q8k, current_token_input, current_token_output, output_dim, input_dim, false);
        
        for (int i = 0; i < output_dim; ++i) {
            batch_output_activations[t * output_dim + i] = current_token_output[i];
        }
    }
}

void matvec_q6k_f32_vector_cpu(const std::vector<block_q6_K>& mat_q6k,
                               const std::vector<float>& vec_f32,
                               std::vector<float>& out_f32, int rows,
                               int cols, bool log_first_block) {
  if (cols % GGML_QK_K != 0) {
    throw std::runtime_error(
        "matvec_q6k_f32_vector_cpu: cols (" + std::to_string(cols) +
        ") must be divisible by GGML_QK_K (" + std::to_string(GGML_QK_K) + ")");
  }
  if (vec_f32.size() != cols) {
    throw std::runtime_error(
        "matvec_q6k_f32_vector_cpu: vec_f32 size mismatch. Expected " +
        std::to_string(cols) + ", got " + std::to_string(vec_f32.size()));
  }
  size_t num_blocks_per_row = cols / GGML_QK_K;
  size_t total_blocks_expected = (size_t)rows * num_blocks_per_row;
  if (mat_q6k.size() != total_blocks_expected) {
    throw std::runtime_error(
        "matvec_q6k_f32_vector_cpu: mat_q6k size mismatch. Expected " +
        std::to_string(total_blocks_expected) + " blocks, got " +
        std::to_string(mat_q6k.size()));
  }

  out_f32.resize(rows);
  float dequantized_block[GGML_QK_K];

#pragma omp parallel for private(dequantized_block)
  for (int64_t r = 0; r < static_cast<int64_t>(rows); ++r) {
    double row_sum = 0.0;
    double kahan_c = 0.0;

    size_t block_row_offset = r * num_blocks_per_row;

    for (size_t block_col_idx = 0; block_col_idx < num_blocks_per_row; ++block_col_idx) {
      const block_q6_K* qblock = &mat_q6k[block_row_offset + block_col_idx];
      bool enable_dequant_log = log_first_block && (r == 0 && block_col_idx == 0);
      dequantize_q6_k(qblock, dequantized_block, GGML_QK_K);

      size_t vec_offset = block_col_idx * GGML_QK_K;
      for (int i = 0; i < GGML_QK_K; ++i) {
        double term = static_cast<double>(dequantized_block[i]) *
                      static_cast<double>(vec_f32[vec_offset + i]);

        double y = term - kahan_c;
        double t = row_sum + y;
        kahan_c = (t - row_sum) - y;
        row_sum = t;
      }
    }
    out_f32[r] = static_cast<float>(row_sum);
  }
}

void matvec_q4k_f32_vector_cpu(const std::vector<block_q4_K>& mat_q4k,
                               const std::vector<float>& vec_f32,
                               std::vector<float>& out_f32, int rows,
                               int cols, bool log_first_block) {
  if (cols % GGML_QK_K != 0) {
    throw std::runtime_error(
        "matvec_q4k_f32_vector_cpu: cols (" + std::to_string(cols) +
        ") must be divisible by GGML_QK_K (" + std::to_string(GGML_QK_K) + ")");
  }
  if (vec_f32.size() != cols) {
    throw std::runtime_error(
        "matvec_q4k_f32_vector_cpu: vec_f32 size mismatch. Expected " +
        std::to_string(cols) + ", got " + std::to_string(vec_f32.size()));
  }
  size_t num_blocks_per_row = cols / GGML_QK_K;
  size_t total_blocks_expected = (size_t)rows * num_blocks_per_row;
  if (mat_q4k.size() != total_blocks_expected) {
    throw std::runtime_error(
        "matvec_q4k_f32_vector_cpu: mat_q4k size mismatch. Expected " +
        std::to_string(total_blocks_expected) + " blocks, got " +
        std::to_string(mat_q4k.size()));
  }

  out_f32.resize(rows);
  float dequantized_block[GGML_QK_K];

#pragma omp parallel for private(dequantized_block)
  for (int64_t r = 0; r < static_cast<int64_t>(rows); ++r) {
    double row_sum = 0.0;
    double kahan_c = 0.0;

    size_t block_row_offset = r * num_blocks_per_row;

    for (size_t block_col_idx = 0; block_col_idx < num_blocks_per_row; ++block_col_idx) {
      const block_q4_K* qblock = &mat_q4k[block_row_offset + block_col_idx];
      bool enable_dequant_log = log_first_block && (r == 0 && block_col_idx == 0);
      dequantize_q4_k_m(qblock, dequantized_block, GGML_QK_K, enable_dequant_log);

      size_t vec_offset = block_col_idx * GGML_QK_K;
      for (int i = 0; i < GGML_QK_K; ++i) {
        double term = static_cast<double>(dequantized_block[i]) *
                      static_cast<double>(vec_f32[vec_offset + i]);

        double y = term - kahan_c;
        double t = row_sum + y;
        kahan_c = (t - row_sum) - y;
        row_sum = t;
      }
    }
    out_f32[r] = static_cast<float>(row_sum);
  }
}

 