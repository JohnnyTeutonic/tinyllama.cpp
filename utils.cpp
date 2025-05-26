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
  

#pragma omp parallel for private(dequantized_block)
  for (int64_t r = 0; r < static_cast<int64_t>(rows); ++r) {
    double row_sum = 0.0;
    double kahan_c = 0.0;

    size_t block_row_offset = static_cast<size_t>(r) * num_blocks_per_row;

    for (size_t block_col_idx = 0; block_col_idx < num_blocks_per_row; ++block_col_idx) {
      const block_q8_0* qblock = &mat_q8_0[block_row_offset + block_col_idx];
      dequantize_q8_0_block(qblock, dequantized_block);

      size_t vec_offset = block_col_idx * GGML_QK8_0;
      
      
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
      double t_sum = k_sum + y;
      k_c = (t_sum - k_sum) - y;
      k_sum = t_sum;
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

void apply_rope_batch_cpu(
    std::vector<float>& q_batch,
    std::vector<float>& k_batch,
    int num_tokens,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int start_pos_in_sequence,
    const std::vector<std::pair<float, float>>& all_freqs_cis,
    int max_pos_embeddings,
    bool use_adjacent_pairing
) {
    if (q_batch.size() != (size_t)num_tokens * num_q_heads * head_dim) {
        Logger::error("apply_rope_batch_cpu: q_batch size mismatch. Expected " +
                      std::to_string((size_t)num_tokens * num_q_heads * head_dim) + ", got " + std::to_string(q_batch.size()));
        return;
    }
    if (k_batch.size() != (size_t)num_tokens * num_kv_heads * head_dim) {
        Logger::error("apply_rope_batch_cpu: k_batch size mismatch. Expected " +
                      std::to_string((size_t)num_tokens * num_kv_heads * head_dim) + ", got " + std::to_string(k_batch.size()));
        return;
    }
    if (head_dim % 2 != 0) {
        Logger::error("apply_rope_batch_cpu: head_dim must be even for RoPE.");
        return; 
    }

    for (int t = 0; t < num_tokens; ++t) {
        int current_token_pos = start_pos_in_sequence + t;

        if (current_token_pos < 0 || current_token_pos >= max_pos_embeddings) {
            Logger::warning("[ROPE_BATCH_CPU] Token " + std::to_string(t) + " (actual_pos: " + std::to_string(current_token_pos) +
                            ") is out of range [0, " + std::to_string(max_pos_embeddings -1) + "]. Skipping RoPE for this token.");
            continue;
        }

        for (int h = 0; h < num_q_heads; ++h) {
            size_t head_start_offset_in_batch = ((size_t)t * num_q_heads + h) * head_dim;

            for (int i = 0; i < head_dim / 2; ++i) { 
                size_t freq_idx = (size_t)current_token_pos * (head_dim / 2) + i;
                
                if (freq_idx >= all_freqs_cis.size()) {
                    Logger::warning("[ROPE_BATCH_CPU] Q - Token " + std::to_string(t) + ", Head " + std::to_string(h) +
                                    ", DimPair " + std::to_string(i) + ": freq_idx (" + std::to_string(freq_idx) +
                                    ") out of bounds for all_freqs_cis.size (" + std::to_string(all_freqs_cis.size()) + "). Skipping pair.");
                    continue; 
                }

                float freq_cis_real = all_freqs_cis[freq_idx].first;
                float freq_cis_imag = all_freqs_cis[freq_idx].second;
                
                float val0, val1;
                size_t idx0, idx1;

                if (use_adjacent_pairing) {
                    idx0 = head_start_offset_in_batch + 2 * i;
                    idx1 = head_start_offset_in_batch + 2 * i + 1;
                } else {
                    idx0 = head_start_offset_in_batch + i;
                    idx1 = head_start_offset_in_batch + i + head_dim / 2;
                }
                
                if (idx0 >= q_batch.size() || idx1 >= q_batch.size()) {
                    Logger::warning("[ROPE_BATCH_CPU] Q - Token " + std::to_string(t) + ", Head " + std::to_string(h) +
                                    ", DimPair " + std::to_string(i) + ": q_batch index out of bounds. q_batch.size(): " + std::to_string(q_batch.size()) +
                                    ", idx0: " + std::to_string(idx0) + ", idx1: " + std::to_string(idx1) + ". Skipping pair.");
                    continue;
                }
                
                val0 = q_batch[idx0];
                val1 = q_batch[idx1];
                
                q_batch[idx0] = val0 * freq_cis_real - val1 * freq_cis_imag;
                q_batch[idx1] = val0 * freq_cis_imag + val1 * freq_cis_real;
            }
        }

        for (int h = 0; h < num_kv_heads; ++h) {
            size_t head_start_offset_in_batch = ((size_t)t * num_kv_heads + h) * head_dim;

            for (int i = 0; i < head_dim / 2; ++i) { 
                size_t freq_idx = (size_t)current_token_pos * (head_dim / 2) + i;

                if (freq_idx >= all_freqs_cis.size()) {
                     Logger::warning("[ROPE_BATCH_CPU] K - Token " + std::to_string(t) + ", Head " + std::to_string(h) +
                                    ", DimPair " + std::to_string(i) + ": freq_idx (" + std::to_string(freq_idx) +
                                    ") out of bounds for all_freqs_cis.size (" + std::to_string(all_freqs_cis.size()) + "). Skipping pair.");
                    continue;
                }

                float freq_cis_real = all_freqs_cis[freq_idx].first;
                float freq_cis_imag = all_freqs_cis[freq_idx].second;

                float val0, val1;
                size_t idx0, idx1;

                if (use_adjacent_pairing) {
                    idx0 = head_start_offset_in_batch + 2 * i;
                    idx1 = head_start_offset_in_batch + 2 * i + 1;
                } else {
                    idx0 = head_start_offset_in_batch + i;
                    idx1 = head_start_offset_in_batch + i + head_dim / 2;
                }

                if (idx0 >= k_batch.size() || idx1 >= k_batch.size()) {
                     Logger::warning("[ROPE_BATCH_CPU] K - Token " + std::to_string(t) + ", Head " + std::to_string(h) +
                                    ", DimPair " + std::to_string(i) + ": k_batch index out of bounds. k_batch.size(): " + std::to_string(k_batch.size()) +
                                    ", idx0: " + std::to_string(idx0) + ", idx1: " + std::to_string(idx1) + ". Skipping pair.");
                    continue;
                }

                val0 = k_batch[idx0];
                val1 = k_batch[idx1];

                k_batch[idx0] = val0 * freq_cis_real - val1 * freq_cis_imag;
                k_batch[idx1] = val0 * freq_cis_imag + val1 * freq_cis_real;
            }
        }
    }
}

void rmsnorm_batch_cpu(const std::vector<float>& x_batch,
                       const std::vector<float>& weight,
                       std::vector<float>& out_batch,
                       int num_tokens,
                       int hidden_size,
                       float eps) {
  if (x_batch.empty() || x_batch.size() != (size_t)num_tokens * hidden_size || weight.size() != (size_t)hidden_size) {
    Logger::error("[RMSNORM_BATCH_CPU] RMSNorm batch size mismatch or empty input. x_batch.size(): " + std::to_string(x_batch.size()) +
                  ", expected x_batch: " + std::to_string((size_t)num_tokens * hidden_size) +
                  ", weight.size(): " + std::to_string(weight.size()) +
                  ", expected weight: " + std::to_string((size_t)hidden_size));
    out_batch.assign((size_t)num_tokens * hidden_size, 0.0f);
    return;
  }
  out_batch.resize((size_t)num_tokens * hidden_size);

#pragma omp parallel for
  for (int t = 0; t < num_tokens; ++t) {
    double ssq = 0.0;
    size_t token_offset = (size_t)t * hidden_size;

    for (int i = 0; i < hidden_size; ++i) {
      ssq += static_cast<double>(x_batch[token_offset + i]) * static_cast<double>(x_batch[token_offset + i]);
    }
    
    double ssq_mean = ssq / hidden_size;
    float norm_factor_input_sqrt = static_cast<float>(ssq_mean);
    float norm_factor = 1.0f / SAFE_SQRT(norm_factor_input_sqrt + eps); 

    for (int i = 0; i < hidden_size; ++i) {
      out_batch[token_offset + i] = x_batch[token_offset + i] * norm_factor * weight[i];
    }
  }
}

void rmsnorm_vector_cpu(const std::vector<float>& x,
                        const std::vector<float>& weight,
                        std::vector<float>& out, float eps) {
  if (x.empty() || x.size() != weight.size()) {
    Logger::error("RMSNorm vector size mismatch or empty input.");
    out.assign(x.size(), 0.0f);
    return;
  }
  out.resize(x.size());
  size_t n = x.size();

  double ssq = 0.0;
#pragma omp parallel for reduction(+ : ssq)
  for (int64_t i = 0; i < static_cast<int64_t>(n); ++i) {
    ssq += static_cast<double>(x[i]) * static_cast<double>(x[i]);
  }
  ssq /= n;

  float norm_factor = 1.0f / SAFE_SQRT(static_cast<float>(ssq) + 
                   SAFE_MAX(eps, numeric::MIN_NORM_EPS));

#pragma omp parallel for
  for (int64_t i = 0; i < static_cast<int64_t>(n); ++i) {
    out[i] = x[i] * norm_factor * weight[i];
  }
}

void softmax_vector_cpu(const std::vector<float>& x,
                        std::vector<float>& out) {
  if (x.empty()) return;
  out.resize(x.size());
  size_t n = x.size();

  float max_val = x[0];
  for (size_t i = 1; i < n; ++i) {
    if (x[i] > max_val) max_val = x[i];
  }

  float exp_sum = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    out[i] = std::exp(x[i] - max_val);
    exp_sum += out[i];
  }

  float inv_sum = 1.0f / (exp_sum + 1e-9f);

#pragma omp parallel for
  for (int64_t i = 0; i < static_cast<int64_t>(n); ++i) {
    out[i] *= inv_sum;
  }
}

void silu_cpu(const std::vector<float>& x, std::vector<float>& out) {
  if (x.size() != out.size()) out.resize(x.size());
#pragma omp parallel for
  for (int64_t i = 0; i < static_cast<int64_t>(x.size()); ++i) {
    float sigmoid_x = 1.0f / (1.0f + std::exp(-x[i]));
    out[i] = x[i] * sigmoid_x;
  }
}

void matmul_f32_f32_batch_cpu(
    const std::vector<float>& mat_weights,
    const std::vector<float>& batch_input_activations,
    std::vector<float>& batch_output_activations,
    int num_tokens,
    int output_dim,
    int input_dim
) {
  if (mat_weights.empty() || batch_input_activations.empty()) {
    Logger::error("[MATMUL_F32_BATCH_CPU] Input matrix or batch_input_activations is empty.");
    batch_output_activations.assign((size_t)num_tokens * output_dim, 0.0f);
    return;
  }
  if (mat_weights.size() != (size_t)output_dim * input_dim) {
    Logger::error("[MATMUL_F32_BATCH_CPU] Matrix dimensions mismatch. Expected " +
                  std::to_string((size_t)output_dim * input_dim) + ", got " +
                  std::to_string(mat_weights.size()));
    batch_output_activations.assign((size_t)num_tokens * output_dim, 0.0f);
    return;
  }
  if (batch_input_activations.size() != (size_t)num_tokens * input_dim) {
    Logger::error(
        "[MATMUL_F32_BATCH_CPU] Batch input activations dimension mismatch. Expected " +
        std::to_string((size_t)num_tokens * input_dim) + ", got " +
        std::to_string(batch_input_activations.size()));
    batch_output_activations.assign((size_t)num_tokens * output_dim, 0.0f);
    return;
  }

  batch_output_activations.resize((size_t)num_tokens * output_dim);

#pragma omp parallel for schedule(static)
  for (int t = 0; t < num_tokens; ++t) {
    size_t input_token_offset = (size_t)t * input_dim;
    size_t output_token_offset = (size_t)t * output_dim;

    for (int o = 0; o < output_dim; ++o) {
      double k_sum = 0.0;
      double k_c = 0.0;
      size_t weight_row_offset = (size_t)o * input_dim;

      for (int i = 0; i < input_dim; ++i) {
        double term = static_cast<double>(mat_weights[weight_row_offset + i]) *
                      static_cast<double>(batch_input_activations[input_token_offset + i]);
        double y = term - k_c;
        double t_sum = k_sum + y;
        k_c = (t_sum - k_sum) - y;
        k_sum = t_sum;
      }
      batch_output_activations[output_token_offset + o] = static_cast<float>(k_sum);
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

void matvec_bf16_f32_vector_cpu(const std::vector<uint16_t>& mat_bf16,
                                const std::vector<float>& vec_f32,
                                std::vector<float>& out_f32, int rows, int cols) {
  if (mat_bf16.size() != (size_t)rows * cols ||
      vec_f32.size() != (size_t)cols) {
    Logger::error("matvec_bf16_f32_vector_cpu: Size mismatch. Mat: " +
                  std::to_string(mat_bf16.size()) + " (Expected " +
                  std::to_string(rows * cols) +
                  "), Vec: " + std::to_string(vec_f32.size()) + " (Expected " +
                  std::to_string(cols) + ")");
    out_f32.assign(rows, 0.0f);
    return;
  }
  out_f32.resize(rows);

#pragma omp parallel for
  for (int64_t r = 0; r < static_cast<int64_t>(rows); ++r) {
    double sum = 0.0;
    double c = 0.0;
    size_t row_offset = r * cols;

    for (int c_idx = 0; c_idx < cols; ++c_idx) {
      float weight = bfloat16_to_float32(mat_bf16[row_offset + c_idx]);
      double term =
          static_cast<double>(weight) * static_cast<double>(vec_f32[c_idx]);

      double y = term - c;
      double t = sum + y;
      c = (t - sum) - y;
      sum = t;
    }
    out_f32[r] = static_cast<float>(sum);
  }
}

void weighted_sum_probs_v(const std::vector<float>& probs,
                          const std::vector<float>& V,
                          std::vector<float>& out, int seq_len, int head_dim) {
  if (probs.size() != seq_len || V.size() != (size_t)seq_len * head_dim) {
    Logger::error("weighted_sum_probs_v: Size mismatch. Probs: " +
                  std::to_string(probs.size()) + " (Expected " +
                  std::to_string(seq_len) +
                  "), V: " + std::to_string(V.size()) + " (Expected " +
                  std::to_string(seq_len * head_dim) + ")");
    out.assign(head_dim, 0.0f);
    return;
  }
  out.resize(head_dim);

#pragma omp parallel for
  for (int64_t j = 0; j < static_cast<int64_t>(head_dim); ++j) {
    double sum = 0.0;
    double c_kahan = 0.0;
    for (int i = 0; i < seq_len; ++i) {
      double term = static_cast<double>(probs[i]) *
                    static_cast<double>(V[i * head_dim + j]);

      double y = term - c_kahan;
      double t = sum + y;
      c_kahan = (t - sum) - y;
      sum = t;
    }
    out[j] = static_cast<float>(sum);
  }
}

void calculate_attention_scores(const std::vector<float>& Q,
                                const std::vector<float>& K,
                                std::vector<float>& scores, int seq_len,
                                int head_dim, float scale) {
  if (Q.empty() || K.empty()) return;
  scores.resize(seq_len);

  scale = std::clamp(scale, attention::MIN_SCALE, attention::MAX_SCALE);
  float effective_scale = scale * attention::ATTENTION_SCALE_BASE;

#pragma omp parallel for collapse(1)
  for (int64_t i = 0; i < static_cast<int64_t>(seq_len); ++i) {
    double dot_product = 0.0;
    double c_kahan = 0.0;
    size_t k_offset = static_cast<size_t>(i) * head_dim;

    for (int j = 0; j < head_dim; ++j) {
      double term = static_cast<double>(Q[j]) * static_cast<double>(K[k_offset + j]);
      double y = term - c_kahan;
      double t_sum = dot_product + y;
      c_kahan = (t_sum - dot_product) - y;
      dot_product = t_sum;
    }
    
    scores[i] = static_cast<float>(dot_product * effective_scale);
  }
}

void log_vec_stats(const std::string& name, const std::vector<float>& v) {
  if (v.empty()) {
    Logger::info(name + ": EMPTY VECTOR");
    return;
  }
  float minv = *std::min_element(v.begin(), v.end());
  float maxv = *std::max_element(v.begin(), v.end());
  float mean = std::accumulate(v.begin(), v.end(), 0.0f) / v.size();
  bool all_finite =
      std::all_of(v.begin(), v.end(), [](float x) { return std::isfinite(x); });
  Logger::info(name + ": min=" + std::to_string(minv) + ", max=" +
               std::to_string(maxv) + ", mean=" + std::to_string(mean) +
               ", all_finite=" + (all_finite ? "yes" : "no"));
}

bool write_vector_to_file(const std::string& filename, const std::vector<float>& vec) {
  std::string vec_writer_vals;
  int N_log_writer = (std::min)(10, (int)vec.size());
  for (int i = 0; i < N_log_writer; ++i)
    vec_writer_vals += (i ? " " : "") + std::to_string(vec[i]);
  Logger::info("write_vector_to_file Enter: Address of vec.data() on entry: " +
               std::to_string(reinterpret_cast<uintptr_t>(vec.data())));

  std::ofstream outfile(filename, std::ios::binary);
  if (!outfile) {
    Logger::error("Failed to open file for writing: " + filename);
    return false;
  }
  outfile.write(reinterpret_cast<const char*>(vec.data()),
                vec.size() * sizeof(float));
  if (!outfile) {
    Logger::error("Failed to write data to file: " + filename);
    return false;
  }
  Logger::info("Successfully wrote vector to " + filename);
  return true;
}

std::vector<std::vector<float>> load_rmsnorm_bin(const std::string& filename, int num_tokens, int hidden_size) {
  std::ifstream infile(filename, std::ios::binary);
  if (!infile) throw std::runtime_error("Failed to open " + filename);
  std::vector<float> flat(num_tokens * hidden_size);
  infile.read(reinterpret_cast<char*>(flat.data()),
              flat.size() * sizeof(float));
  if (!infile)
    throw std::runtime_error("Failed to read all data from " + filename);
  std::vector<std::vector<float>> result(num_tokens,
                                         std::vector<float>(hidden_size));
  for (int t = 0; t < num_tokens; ++t) {
    for (int h = 0; h < hidden_size; ++h) {
      result[t][h] = flat[t * hidden_size + h];
    }
  }
  return result;
}

void log_raw_float_pointer(const std::string& name, const float* ptr, size_t count) {
  if (!ptr) {
    Logger::info(name + ": NULL POINTER");
    return;
  }
  std::stringstream ss;
  ss << name << ": [";
  for (size_t i = 0; i < count; ++i) {
    if (i > 0) ss << ", ";
    ss << std::fixed << std::setprecision(6) << ptr[i];
  }
  ss << "]";
  Logger::info(ss.str());
}

void log_vector_summary_detailed(const std::string& name,
                                  const std::vector<float>& v,
                                  int current_pos, int current_layer, int N) {
  if (v.empty()) {
    Logger::info(name + ": EMPTY");
    return;
  }
  
  std::stringstream ss;
  ss << "[POS=" << current_pos << " LAYER=" << current_layer << "] " << name;
  ss << ": size=" << v.size();
  
  size_t actual_N = SAFE_MIN(static_cast<size_t>(N), v.size());
  if (actual_N > 0) {
    ss << ", first " << actual_N << ": [";
    for (size_t i = 0; i < actual_N; ++i) {
      ss << (i > 0 ? " " : "") << std::fixed << std::setprecision(6) << v[i];
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

 