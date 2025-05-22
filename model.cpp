#include "model.h"

#include "logger.h"
#ifdef HAS_CUDA
#include "cuda_kernels.h"
#endif
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#ifdef _WIN32
#include <windows.h>
#endif
#include <cassert>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <variant>

#include "gguf_parser.h"
#include "quantization.h"
#include "model_constants.h"
#include "model_macros.h"
#include "safetensors_loader.h"

/**
 * @brief Converts a float32 value to bfloat16 representation.
 * @param val The float32 value to convert.
 * @return The bfloat16 representation as uint16_t.
 */
inline uint16_t float32_to_bfloat16(float val);

/**
 * @brief Matrix-vector multiplication for Q6_K quantized weights and float32 vector.
 * @param mat_q6k The quantized matrix.
 * @param vec_f32 The input float32 vector.
 * @param out_f32 The output float32 vector.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @param log_first_block Whether to log the first block for debugging.
 */
static void matvec_q6k_f32_vector_cpu(const std::vector<block_q6_K>& mat_q6k,
                                      const std::vector<float>& vec_f32,
                                      std::vector<float>& out_f32, int rows,
                                      int cols, bool log_first_block = false);

/**
 * @brief Matrix-vector multiplication for Q4_K quantized weights and float32 vector.
 * @param mat_q4k The quantized matrix.
 * @param vec_f32 The input float32 vector.
 * @param out_f32 The output float32 vector.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @param log_first_block Whether to log the first block for debugging.
 */
static void matvec_q4k_f32_vector_cpu(const std::vector<block_q4_K>& mat_q4k,
                                      const std::vector<float>& vec_f32,
                                      std::vector<float>& out_f32, int rows,
                                      int cols, bool log_first_block = false);

/**
 * @brief Matrix-vector multiplication for Q8_0 quantized weights and float32 vector.
 * @param mat_q8_0 The quantized matrix.
 * @param vec_f32 The input float32 vector.
 * @param out_f32 The output float32 vector.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @param log_first_block Whether to log the first block for debugging.
 */
static void matvec_q8_0_f32_vector_cpu(const std::vector<block_q8_0>& mat_q8_0,
                                      const std::vector<float>& vec_f32,
                                      std::vector<float>& out_f32, int rows,
                                      int cols, bool log_first_block = false);

/**
 * @brief Matrix-vector multiplication for float32 matrix and float32 vector.
 * @param mat_f32 The float32 matrix.
 * @param vec_f32 The input float32 vector.
 * @param out_f32 The output float32 vector.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 */
static void matvec_f32_f32_vector_cpu(const std::vector<float>& mat_f32,
                                      const std::vector<float>& vec_f32,
                                      std::vector<float>& out_f32, int rows,
                                      int cols);

/**
 * @brief Dequantizes a vector of Q8_K blocks to float32.
 * @param q8k_vec The quantized vector.
 * @param out_f32 The output float32 vector.
 * @param n Number of elements.
 * @param log_this_block Whether to log this block for debugging.
 */
void dequantize_q8_k(const std::vector<block_q8_K>& q8k_vec,
                     std::vector<float>& out_f32, int n, bool log_this_block);

/**
 * @brief Logs a detailed summary of a float vector for debugging.
 * @param name Name for the log entry.
 * @param v The vector to log.
 * @param current_pos Current position (e.g., token position).
 * @param current_layer Current layer index.
 * @param N Number of elements to log.
 */
static void log_vector_summary_detailed(const std::string& name,
                                        const std::vector<float>& v,
                                        int current_pos, int current_layer,
                                        int N = 5);

static std::vector<float> bf16vec_to_float_vec(
    const std::vector<uint16_t>& v_bf16) {
  std::vector<float> v_f32(v_bf16.size());
#pragma omp parallel for
  for (int64_t i = 0; i < static_cast<int64_t>(v_bf16.size()); ++i) {
    v_f32[i] = bfloat16_to_float32(v_bf16[i]);
  }
  return v_f32;
}

inline uint16_t float32_to_bfloat16(float val) {
  uint32_t bits;
  std::memcpy(&bits, &val, sizeof(float));

  bits += 0x7FFF + ((bits >> 16) & 1);
  return static_cast<uint16_t>(bits >> 16);
}
static void matvec_q6k_f32_vector_cpu(const std::vector<block_q6_K>& mat_q6k,
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
static void matvec_q4k_f32_vector_cpu(const std::vector<block_q4_K>& mat_q4k,
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
static void matvec_q8_0_f32_vector_cpu(const std::vector<block_q8_0>& mat_q8_0,
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
static void matvec_f32_f32_vector_cpu(const std::vector<float>& mat_f32,
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

    // Kahan summation variables
    double k_sum = 0.0;
    double k_c = 0.0;

    for (int c = 0; c < cols; ++c) {
      // sum += mat_row_ptr[c] * vec_ptr[c]; // Old direct summation
      double term = static_cast<double>(mat_row_ptr[c]) * static_cast<double>(vec_ptr[c]);
      double y = term - k_c;
      double t = k_sum + y;
      k_c = (t - k_sum) - y;
      k_sum = t;
    }
    out_f32[r] = static_cast<float>(k_sum);
  }
}

static void apply_rope_vector(
    std::vector<float>& x, // The Q or K vector for all heads, concatenated
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

    for (int i = 0; i < dim_half; ++i) { // i iterates 0 to (head_dim/2 - 1)
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
        // Adjacent pairing: (x_{2i}, x_{2i+1})
        x0_idx = head_offset + (2 * i);
        x1_idx = head_offset + (2 * i + 1);
      } else {
        // Split-half pairing: (x_i, x_{i + dim_half})
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

static void matmul_q4k_f32_batch_cpu(
    const std::vector<block_q4_K>& mat_q4k,             // Quantized matrix weights
    const std::vector<float>& batch_input_activations,  // Batched input: [num_tokens, input_dim]
    std::vector<float>& batch_output_activations,       // Batched output: [num_tokens, output_dim]
    int num_tokens,
    int output_dim, // rows of the matrix
    int input_dim   // cols of the matrix (must match vec_f32 size for single token)
) {
    if (mat_q4k.empty() || batch_input_activations.empty()) {
        Logger::error("matmul_q4k_f32_batch_cpu: Input matrix or batch_input_activations is empty.");
        batch_output_activations.assign((size_t)num_tokens * output_dim, 0.0f);
        return;
    }
    if (batch_input_activations.size() != (size_t)num_tokens * input_dim) {
        Logger::error("matmul_q4k_f32_batch_cpu: batch_input_activations size mismatch. Expected " + 
                      std::to_string((size_t)num_tokens * input_dim) + ", got " + 
                      std::to_string(batch_input_activations.size()));
        batch_output_activations.assign((size_t)num_tokens * output_dim, 0.0f);
        return;
    }
    // Further validation of mat_q4k size against output_dim and input_dim 
    // is handled by matvec_q4k_f32_vector_cpu.

    batch_output_activations.resize((size_t)num_tokens * output_dim);

#pragma omp parallel for
    for (int token_idx = 0; token_idx < num_tokens; ++token_idx) {
        // Extract the input vector for the current token
        std::vector<float> current_token_input(input_dim);
        const float* input_slice_start = batch_input_activations.data() + (size_t)token_idx * input_dim;
        std::copy(input_slice_start, input_slice_start + input_dim, current_token_input.begin());

        // Prepare output vector for the current token
        std::vector<float> current_token_output(output_dim);

        // Perform matrix-vector multiplication for the current token
        matvec_q4k_f32_vector_cpu(mat_q4k, current_token_input, current_token_output, output_dim, input_dim, false);
        
        // Copy the result into the batched output vector
        float* output_slice_start = batch_output_activations.data() + (size_t)token_idx * output_dim;
        std::copy(current_token_output.begin(), current_token_output.end(), output_slice_start);
    }
}

static void matmul_q6k_f32_batch_cpu(
    const std::vector<block_q6_K>& mat_q6k,             // Quantized matrix weights
    const std::vector<float>& batch_input_activations,  // Batched input: [num_tokens, input_dim]
    std::vector<float>& batch_output_activations,       // Batched output: [num_tokens, output_dim]
    int num_tokens,
    int output_dim, // rows of the matrix
    int input_dim   // cols of the matrix
) {
    if (mat_q6k.empty() || batch_input_activations.empty()) {
        Logger::error("matmul_q6k_f32_batch_cpu: Input matrix or batch_input_activations is empty.");
        batch_output_activations.assign((size_t)num_tokens * output_dim, 0.0f);
        return;
    }
    if (batch_input_activations.size() != (size_t)num_tokens * input_dim) {
        Logger::error("matmul_q6k_f32_batch_cpu: batch_input_activations size mismatch. Expected " +
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

static void matmul_q8_0_f32_batch_cpu(
    const std::vector<block_q8_0>& mat_q8_0,            // Quantized matrix weights
    const std::vector<float>& batch_input_activations,  // Batched input: [num_tokens, input_dim]
    std::vector<float>& batch_output_activations,       // Batched output: [num_tokens, output_dim]
    int num_tokens,
    int output_dim, // rows of the matrix
    int input_dim   // cols of the matrix
) {
    if (mat_q8_0.empty() || batch_input_activations.empty()) {
        Logger::error("matmul_q8_0_f32_batch_cpu: Input matrix or batch_input_activations is empty.");
        batch_output_activations.assign((size_t)num_tokens * output_dim, 0.0f);
        return;
    }
    if (batch_input_activations.size() != (size_t)num_tokens * input_dim) {
        Logger::error("matmul_q8_0_f32_batch_cpu: batch_input_activations size mismatch. Expected " +
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
void log_vector_summary(const std::string& name, const std::vector<float>& v,
                        int head_count) {
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
  bool all_finite =
      std::all_of(v.begin(), v.end(), [](float x) { return std::isfinite(x); });
  ss << ", min=" << minv << ", max=" << maxv << ", mean=" << mean
     << ", finite=" << (all_finite ? "yes" : "NO");
  Logger::info(ss.str());
}

void log_vector_summary_with_tail(const std::string& name,
                                  const std::vector<float>& v, int head_count,
                                  int tail_count) {
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
  bool all_finite =
      std::all_of(v.begin(), v.end(), [](float x) { return std::isfinite(x); });
  ss << ", min=" << minv << ", max=" << maxv << ", mean=" << mean
     << ", finite=" << (all_finite ? "yes" : "NO");
  Logger::info(ss.str());
}

/**
 * @brief Converts a bfloat16 value to float32.
 * @param bf16 The bfloat16 value.
 * @return The float32 representation.
 */
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

/**
 * @brief Converts a vector of bfloat16 values to float32.
 * @param bf16_vec The input vector of bfloat16 values.
 * @return The output vector of float32 values.
 */
std::vector<float> bfloat16_vector_to_float32(
    const std::vector<uint16_t>& bf16_vec) {
  std::vector<float> f32_vec(bf16_vec.size());

#pragma omp parallel for
  for (int64_t i = 0; i < static_cast<int64_t>(bf16_vec.size()); ++i) {
    f32_vec[i] = bfloat16_to_float32(bf16_vec[i]);
  }

  return f32_vec;
}

/**
 * @brief Converts a vector of uint8_t bytes to a vector of uint16_t values.
 * @param bytes The input byte vector.
 * @param numel The number of uint16_t elements expected.
 * @return The output vector of uint16_t values.
 * @throws std::runtime_error if the byte vector size does not match numel * 2.
 */
std::vector<uint16_t> uint8_vector_to_uint16_vector(
    const std::vector<uint8_t>& bytes, size_t numel) {
  if (bytes.size() != numel * 2) {
    throw std::runtime_error(
        "Byte vector size mismatch for uint16_t conversion");
  }
  std::vector<uint16_t> out(numel);

  std::memcpy(out.data(), bytes.data(), bytes.size());
  return out;
}

/**
 * @brief Returns the index of the maximum value in a float vector.
 * @param v The input vector.
 * @return The index of the maximum value, or -1 if the vector is empty.
 */
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

// --- Corrected static helper function for RoPE to be appended to the end of model.cpp ---
// (Replace the previous apply_rope_batch_cpu with this one)

static void apply_rope_batch_cpu(
    std::vector<float>& q_batch, // [num_tokens, num_q_heads * head_dim]
    std::vector<float>& k_batch, // [num_tokens, num_kv_heads * head_dim]
    int num_tokens,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int start_pos_in_sequence,
    const std::vector<std::pair<float, float>>& all_freqs_cis, // Contains pairs of (cos(m*theta_i), sin(m*theta_i))
    int max_pos_embeddings,
    bool use_adjacent_pairing // True for GGUF/Llama-style, False for some older Safetensors RoPE
) {
    if (q_batch.size() != (size_t)num_tokens * num_q_heads * head_dim) {
        Logger::error("apply_rope_batch_cpu: q_batch size mismatch.");
        return;
    }
    if (k_batch.size() != (size_t)num_tokens * num_kv_heads * head_dim) {
        Logger::error("apply_rope_batch_cpu: k_batch size mismatch.");
        return;
    }
    if (head_dim % 2 != 0) {
        Logger::error("apply_rope_batch_cpu: head_dim must be even for RoPE.");
        return; // RoPE operates on pairs of dimensions
    }

#pragma omp parallel for
    for (int t = 0; t < num_tokens; ++t) {
        int current_token_pos = start_pos_in_sequence + t;
        if (current_token_pos >= max_pos_embeddings) {
            Logger::warning("apply_rope_batch_cpu: current_token_pos " + std::to_string(current_token_pos) +
                            " exceeds max_pos_embeddings " + std::to_string(max_pos_embeddings) + ". Skipping RoPE for this token.");
            continue;
        }

        // Apply RoPE to Q for this token
        for (int h = 0; h < num_q_heads; ++h) {
            size_t head_start_offset_in_batch = ((size_t)t * num_q_heads + h) * head_dim;
            for (int i = 0; i < head_dim / 2; ++i) { // Iterate through pairs of dimensions
                float freq_cis_real = all_freqs_cis[current_token_pos * (head_dim / 2) + i].first;
                float freq_cis_imag = all_freqs_cis[current_token_pos * (head_dim / 2) + i].second;
                
                float val0, val1;
                size_t idx0, idx1;

                if (use_adjacent_pairing) { // e.g., (x0, x1), (x2, x3), ...
                    idx0 = head_start_offset_in_batch + 2 * i;
                    idx1 = head_start_offset_in_batch + 2 * i + 1;
                } else { // Half-dimension pairing e.g., (x0, x_{D/2}), (x1, x_{D/2+1}), ...
                    idx0 = head_start_offset_in_batch + i;
                    idx1 = head_start_offset_in_batch + i + head_dim / 2;
                }
                
                val0 = q_batch[idx0];
                val1 = q_batch[idx1];
                
                q_batch[idx0] = val0 * freq_cis_real - val1 * freq_cis_imag;
                q_batch[idx1] = val0 * freq_cis_imag + val1 * freq_cis_real;
            }
        }

        // Apply RoPE to K for this token
        for (int h = 0; h < num_kv_heads; ++h) {
            size_t head_start_offset_in_batch = ((size_t)t * num_kv_heads + h) * head_dim;
            for (int i = 0; i < head_dim / 2; ++i) { // Iterate through pairs of dimensions
                float freq_cis_real = all_freqs_cis[current_token_pos * (head_dim / 2) + i].first;
                float freq_cis_imag = all_freqs_cis[current_token_pos * (head_dim / 2) + i].second;

                float val0, val1;
                size_t idx0, idx1;

                if (use_adjacent_pairing) {
                    idx0 = head_start_offset_in_batch + 2 * i;
                    idx1 = head_start_offset_in_batch + 2 * i + 1;
                } else {
                    idx0 = head_start_offset_in_batch + i;
                    idx1 = head_start_offset_in_batch + i + head_dim / 2;
                }

                val0 = k_batch[idx0];
                val1 = k_batch[idx1];

                k_batch[idx0] = val0 * freq_cis_real - val1 * freq_cis_imag;
                k_batch[idx1] = val0 * freq_cis_imag + val1 * freq_cis_real;
            }
        }
    }
}
// --- End of corrected RoPE function ---
static void matmul_f32_f32_batch_cpu(
    const std::vector<float>& mat_weights,      // [output_dim, input_dim]
    const std::vector<float>& batch_input_activations, // [num_tokens, input_dim]
    std::vector<float>& batch_output_activations, // [num_tokens, output_dim]
    int num_tokens,
    int output_dim, // e.g., hs for Q, n_kv_heads * head_dim for K/V
    int input_dim   // e.g., hs
) {
  if (mat_weights.empty() || batch_input_activations.empty()) {
    Logger::error("matmul_f32_f32_batch_cpu: Input matrix or batch_input_activations is empty.");
    batch_output_activations.assign((size_t)num_tokens * output_dim, 0.0f);
    return;
  }
  if (mat_weights.size() != (size_t)output_dim * input_dim) {
    Logger::error("matmul_f32_f32_batch_cpu: Matrix dimensions mismatch. Expected " +
                  std::to_string((size_t)output_dim * input_dim) + ", got " +
                  std::to_string(mat_weights.size()));
    batch_output_activations.assign((size_t)num_tokens * output_dim, 0.0f);
    return;
  }
  if (batch_input_activations.size() != (size_t)num_tokens * input_dim) {
    Logger::error(
        "matmul_f32_f32_batch_cpu: Batch input activations dimension mismatch. Expected " +
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
        double t_sum = k_sum + y; // Renamed t to t_sum to avoid conflict if <cmath> t is in scope
        k_c = (t_sum - k_sum) - y;
        k_sum = t_sum;
      }
      batch_output_activations[output_token_offset + o] = static_cast<float>(k_sum);
    }
  }
}

static void rmsnorm_batch_cpu(const std::vector<float>& x_batch, // [num_tokens, hidden_size]
                              const std::vector<float>& weight,    // [hidden_size]
                              std::vector<float>& out_batch, // [num_tokens, hidden_size]
                              int num_tokens,
                              int hidden_size,
                              float eps = numeric::DEFAULT_EPS) {
  if (x_batch.empty() || x_batch.size() != (size_t)num_tokens * hidden_size || weight.size() != (size_t)hidden_size) {
    Logger::error("RMSNorm batch size mismatch or empty input.");
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
    ssq /= hidden_size;
    float norm_factor = 1.0f / SAFE_SQRT(static_cast<float>(ssq) + 
                                         SAFE_MAX(eps, numeric::MIN_NORM_EPS));
    for (int i = 0; i < hidden_size; ++i) {
      out_batch[token_offset + i] = x_batch[token_offset + i] * norm_factor * weight[i];
    }
  }
}

static void rmsnorm_vector_cpu(const std::vector<float>& x,
                               const std::vector<float>& weight,
                               std::vector<float>& out, float eps = numeric::DEFAULT_EPS) {
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
static void softmax_vector_cpu(const std::vector<float>& x,
                               std::vector<float>& out) {
  if (x.empty()) return;
  out.resize(x.size());
  size_t n = x.size();

  // Find max element for numerical stability (serial)
  float max_val = x[0];
  for (size_t i = 1; i < n; ++i) {
    if (x[i] > max_val) max_val = x[i];
  }

  // Compute exponentials and sum (serial)
  float exp_sum = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    out[i] = std::exp(x[i] - max_val);
    exp_sum += out[i];
  }

  // Normalize
  float inv_sum = 1.0f / (exp_sum + 1e-9f); // Add epsilon for stability

#pragma omp parallel for // Final normalization can be parallel
  for (int64_t i = 0; i < static_cast<int64_t>(n); ++i) {
    out[i] *= inv_sum;
  }
}
static void silu_cpu(const std::vector<float>& x, std::vector<float>& out) {
  if (x.size() != out.size()) out.resize(x.size());
#pragma omp parallel for
  for (int64_t i = 0; i < static_cast<int64_t>(x.size()); ++i) {
    float sigmoid_x = 1.0f / (1.0f + std::exp(-x[i]));
    out[i] = x[i] * sigmoid_x;
  }
}

static void log_vec_stats(const std::string& name,
                          const std::vector<float>& v) {
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

static bool write_vector_to_file(const std::string& filename,
                                 const std::vector<float>& vec) {
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

static std::vector<std::vector<float>> load_rmsnorm_bin(
    const std::string& filename, int num_tokens, int hidden_size) {
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

ModelConfig parse_model_config(const nlohmann::json& json) {
  ModelConfig cfg;
  cfg.hidden_size = json.value("hidden_size", 0);
  cfg.intermediate_size = json.value("intermediate_size", 0);
  cfg.num_attention_heads = json.value("num_attention_heads", 0);
  cfg.num_key_value_heads = json.value("num_key_value_heads", 0);
  cfg.num_hidden_layers = json.value("num_hidden_layers", 0);
  cfg.vocab_size = json.value("vocab_size", 0);
  cfg.max_position_embeddings = json.value("max_position_embeddings", 0);
  cfg.rms_norm_eps = json.value("rms_norm_eps", 1e-5f);
  cfg.rope_theta = json.value("rope_theta", 10000.0f);
  cfg.hidden_act = json.value("hidden_act", "silu");
  cfg.torch_dtype = json.value("torch_dtype", "bfloat16");
  cfg.bos_token_id = json.value("bos_token_id", 1);
  cfg.eos_token_id = json.value("eos_token_id", 2);
  cfg.unk_token_id = json.value("unk_token_id", -1);
  cfg.pad_token_id = json.value("pad_token_id", -1); 

  // Infer Architecture if available
  if (json.contains("architectures") && json["architectures"].is_array() && !json["architectures"].empty()) {
      // Take the first architecture string if multiple are listed
      cfg.architecture = json["architectures"][0].get<std::string>();
  } else {
      cfg.architecture = "unknown"; 
  }
  cfg.model_name = json.value("model_type", cfg.architecture); // Use model_type or fallback to architecture

  
  Logger::info("[parse_json_config] Inferring tokenizer family for SafeTensors. Arch: '" + cfg.architecture + "', Vocab: " + std::to_string(cfg.vocab_size));
  bool is_llama3_vocab_size_json = (cfg.vocab_size == 128256);
  bool is_llama3_arch_hint_json = (cfg.architecture.find("LlamaForCausalLM") != std::string::npos && // Llama 3 often uses this
                              cfg.architecture.find("Llama2") == std::string::npos); // Exclude Llama 2 explicitly if needed

  if (is_llama3_vocab_size_json && is_llama3_arch_hint_json) {
      cfg.tokenizer_family = ModelConfig::TokenizerFamily::LLAMA3_TIKTOKEN;
      Logger::info("[parse_json_config] Result: Identified LLAMA3_TIKTOKEN (vocab size + arch hint).");
       if (cfg.rope_theta == 10000.0f) { 
            float llama3_rope_candidate = json.value("rope_theta", 500000.0f); // Check rope_theta in config.json
            if (llama3_rope_candidate > 10000.0f) {
                cfg.rope_theta = llama3_rope_candidate;
                Logger::info("[parse_json_config] Adjusted rope_theta to " + std::to_string(cfg.rope_theta) + " for Llama 3 model (was 10000.0).");
            }
       }
  } else if (cfg.vocab_size == 32000 || cfg.architecture.find("Llama") != std::string::npos) { // Common for Llama 1/2/TinyLlama
      cfg.tokenizer_family = ModelConfig::TokenizerFamily::LLAMA_SENTENCEPIECE;
      Logger::info("[parse_json_config] Result: Identified LLAMA_SENTENCEPIECE (vocab size or arch hint).");
  } else {
      cfg.tokenizer_family = ModelConfig::TokenizerFamily::UNKNOWN;
      Logger::warning("[parse_json_config] Result: UNKNOWN tokenizer family.");
  }
  

  return cfg;
}

static void log_raw_float_pointer(const std::string& name, const float* ptr,
                                  size_t count = 5) {
  if (!ptr) {
    Logger::info(name + ": NULL Pointer");
    return;
  }
  std::stringstream ss;
  ss << name << " first " << count << ": [";
  for (size_t i = 0; i < count; ++i) {
    ss << (i > 0 ? " " : "") << std::fixed << std::setprecision(6) << ptr[i];
  }
  ss << "]";
  Logger::info(ss.str());
}

void KVCache::initialize(const ModelConfig& config, 
                         int total_num_model_layers, int num_gpu_layers_to_allocate, 
                         int max_seq_len, int num_kv_heads,
                         int head_dim) {
  this->total_model_layers_ = total_num_model_layers; // Store for use in other KVCache methods if needed
  layers.resize(total_num_model_layers); // CPU KVCacheLayer vector sized for all layers
  seq_len = 0;
  Logger::info("Allocating KVCache host vectors...");
  size_t cache_size_per_layer = static_cast<size_t>(max_seq_len) *
                                static_cast<size_t>(num_kv_heads) *
                                static_cast<size_t>(head_dim);

  if (cache_size_per_layer == 0 && max_seq_len > 0 && total_num_model_layers > 0) { // Check total_num_model_layers too
    throw std::runtime_error(
        "KVCache (CPU): Calculated cache size is zero for non-empty model. Check parameters.");
  }

  for (int l = 0; l < total_num_model_layers; ++l) { // Allocate CPU part for all model layers
    try {
      layers[l].k.assign(cache_size_per_layer, 0.0f);
      layers[l].v.assign(cache_size_per_layer, 0.0f);
    } catch (const std::bad_alloc& e) {
      Logger::error("Failed to allocate CPU KVCache for layer " +
                    std::to_string(l) + ": " + e.what());
      throw;
    }
  }
  Logger::info("KVCache (CPU) vectors allocated for " +
               std::to_string(total_num_model_layers) + " layers.");

#ifdef HAS_CUDA
  // Store the actual number of layers for which GPU memory will be allocated.
  this->allocated_num_layers = num_gpu_layers_to_allocate; 
  this->allocated_max_seq_len = max_seq_len;
  this->allocated_num_kv_heads = num_kv_heads;
  this->allocated_head_dim = head_dim;

  if (num_gpu_layers_to_allocate > 0) { // Only proceed if GPU layers are requested
      if (num_gpu_layers_to_allocate > total_num_model_layers) {
          Logger::warning("KVCache::initialize: num_gpu_layers_to_allocate (" + std::to_string(num_gpu_layers_to_allocate) +
                          ") > total_num_model_layers (" + std::to_string(total_num_model_layers) + 
                          "). Clamping to total_num_model_layers.");
          this->allocated_num_layers = total_num_model_layers; // Update member
          num_gpu_layers_to_allocate = total_num_model_layers; // Use clamped value for local logic
      }

      size_t cache_elems_per_layer_gpu = static_cast<size_t>(max_seq_len) *
                                 static_cast<size_t>(num_kv_heads) *
                                 static_cast<size_t>(head_dim);
      
      // Sizes for different KVCache types
      size_t fp32_cache_bytes_per_layer_gpu = cache_elems_per_layer_gpu * sizeof(float);
      size_t int8_cache_bytes_per_layer_gpu = cache_elems_per_layer_gpu * sizeof(int8_t);
      // For scales: one scale per head per token position
      size_t num_scales_per_layer_gpu = static_cast<size_t>(max_seq_len) * static_cast<size_t>(num_kv_heads);
      size_t scales_bytes_per_layer_gpu = num_scales_per_layer_gpu * sizeof(float);

      if (cache_elems_per_layer_gpu == 0 && config.use_kvcache_quantization) {
        throw std::runtime_error(
            "KVCache (CUDA INT8): Calculated cache elements per layer is zero. Check parameters.");
      } else if (cache_elems_per_layer_gpu == 0) {
        throw std::runtime_error(
            "KVCache (CUDA FP32): Calculated cache elements per layer is zero. Check parameters.");
      }

      if (config.use_kvcache_quantization) {
        Logger::info("Allocating INT8 KVCache + FP32 Scales on GPU for " + std::to_string(num_gpu_layers_to_allocate) +
                 " layers. Data size per layer: " +
                     std::to_string(int8_cache_bytes_per_layer_gpu / (1024.0 * 1024.0)) +
                 " MB. Scales size per layer: " + 
                     std::to_string(scales_bytes_per_layer_gpu / (1024.0 * 1024.0)) + " MB");
      } else {
        Logger::info("Allocating FP32 KVCache on GPU for " + std::to_string(num_gpu_layers_to_allocate) +
                 " layers, size per layer: " +
                     std::to_string(fp32_cache_bytes_per_layer_gpu / (1024.0 * 1024.0)) +
                 " MB");
      }

      int gpu_layer_start_model_idx = this->total_model_layers_ - num_gpu_layers_to_allocate;
      Logger::info("KVCache GPU allocation will target model layers from index " + std::to_string(gpu_layer_start_model_idx) +
                   " to " + std::to_string(gpu_layer_start_model_idx + num_gpu_layers_to_allocate - 1));

      for (int i = 0; i < num_gpu_layers_to_allocate; ++i) { // Loop 'i' for count
        int current_model_idx_for_gpu = gpu_layer_start_model_idx + i; // Calculate actual model index

        if (current_model_idx_for_gpu < 0 || static_cast<size_t>(current_model_idx_for_gpu) >= layers.size()) {
            Logger::error("KVCache::initialize: Calculated current_model_idx_for_gpu (" + std::to_string(current_model_idx_for_gpu) + ") is out of bounds for layers vector (size " + std::to_string(layers.size()) + "). Skipping this layer.");
            continue;
        }

        if (layers[current_model_idx_for_gpu].k_dev_fp32) {
          Logger::warning(
              "KVCache::initialize: Re-initializing KVCache layer " + std::to_string(current_model_idx_for_gpu) + " K dev fp32 pointer without proper destruction?");
          gpuErrchk(cudaFree(layers[current_model_idx_for_gpu].k_dev_fp32));
          layers[current_model_idx_for_gpu].k_dev_fp32 = nullptr;
        }
        if (layers[current_model_idx_for_gpu].v_dev_fp32) {
          Logger::warning(
              "KVCache::initialize: Re-initializing KVCache layer " + std::to_string(current_model_idx_for_gpu) + " V dev fp32 pointer without proper destruction?");
          gpuErrchk(cudaFree(layers[current_model_idx_for_gpu].v_dev_fp32));
          layers[current_model_idx_for_gpu].v_dev_fp32 = nullptr;
        }
        if (layers[current_model_idx_for_gpu].k_dev_quantized) {
          Logger::warning(
              "KVCache::initialize: Re-initializing KVCache layer " + std::to_string(current_model_idx_for_gpu) + " K dev quantized pointer without proper destruction?");
          gpuErrchk(cudaFree(layers[current_model_idx_for_gpu].k_dev_quantized));
          layers[current_model_idx_for_gpu].k_dev_quantized = nullptr;
        }
        if (layers[current_model_idx_for_gpu].v_dev_quantized) {
          Logger::warning(
              "KVCache::initialize: Re-initializing KVCache layer " + std::to_string(current_model_idx_for_gpu) + " V dev quantized pointer without proper destruction?");
          gpuErrchk(cudaFree(layers[current_model_idx_for_gpu].v_dev_quantized));
          layers[current_model_idx_for_gpu].v_dev_quantized = nullptr;
        }
        if (layers[current_model_idx_for_gpu].k_dev_scales) {
          Logger::warning(
              "KVCache::initialize: Re-initializing KVCache layer " + std::to_string(current_model_idx_for_gpu) + " K dev scales pointer without proper destruction?");
          gpuErrchk(cudaFree(layers[current_model_idx_for_gpu].k_dev_scales));
          layers[current_model_idx_for_gpu].k_dev_scales = nullptr;
        }
        if (layers[current_model_idx_for_gpu].v_dev_scales) {
          Logger::warning(
              "KVCache::initialize: Re-initializing KVCache layer " + std::to_string(current_model_idx_for_gpu) + " V dev scales pointer without proper destruction?");
          gpuErrchk(cudaFree(layers[current_model_idx_for_gpu].v_dev_scales));
          layers[current_model_idx_for_gpu].v_dev_scales = nullptr;
        }
        
        if (config.use_kvcache_quantization) {
            gpuErrchk(cudaMalloc(&layers[current_model_idx_for_gpu].k_dev_quantized, int8_cache_bytes_per_layer_gpu));
            gpuErrchk(cudaMalloc(&layers[current_model_idx_for_gpu].v_dev_quantized, int8_cache_bytes_per_layer_gpu));
            gpuErrchk(cudaMalloc(&layers[current_model_idx_for_gpu].k_dev_scales, scales_bytes_per_layer_gpu));
            gpuErrchk(cudaMalloc(&layers[current_model_idx_for_gpu].v_dev_scales, scales_bytes_per_layer_gpu));

            gpuErrchk(cudaMemset(layers[current_model_idx_for_gpu].k_dev_quantized, 0, int8_cache_bytes_per_layer_gpu));
            gpuErrchk(cudaMemset(layers[current_model_idx_for_gpu].v_dev_quantized, 0, int8_cache_bytes_per_layer_gpu));
            gpuErrchk(cudaMemset(layers[current_model_idx_for_gpu].k_dev_scales, 0, scales_bytes_per_layer_gpu));
            gpuErrchk(cudaMemset(layers[current_model_idx_for_gpu].v_dev_scales, 0, scales_bytes_per_layer_gpu));
        } else {
            gpuErrchk(cudaMalloc(&layers[current_model_idx_for_gpu].k_dev_fp32, fp32_cache_bytes_per_layer_gpu));
            gpuErrchk(cudaMalloc(&layers[current_model_idx_for_gpu].v_dev_fp32, fp32_cache_bytes_per_layer_gpu));
            gpuErrchk(cudaMemset(layers[current_model_idx_for_gpu].k_dev_fp32, 0, fp32_cache_bytes_per_layer_gpu));
            gpuErrchk(cudaMemset(layers[current_model_idx_for_gpu].v_dev_fp32, 0, fp32_cache_bytes_per_layer_gpu));
        }
  }
      Logger::info("KVCache GPU allocation and zeroing complete for " + std::to_string(num_gpu_layers_to_allocate) + " layers.");
  } else {
      Logger::info("KVCache: No GPU layers requested for allocation (num_gpu_layers_to_allocate is 0). Skipping GPU KVCache allocation.");
      this->allocated_num_layers = 0; 
  }

#else
  Logger::info("KVCache (CPU-only build) initialized with dimensions for " +
               std::to_string(total_num_model_layers) + " layers, " +
               std::to_string(max_seq_len) + " seq len, " +
               std::to_string(num_kv_heads) + " KV heads, " +
               std::to_string(head_dim) + " head dim");
#endif
}


static void matvec_bf16_f32_vector_cpu(const std::vector<uint16_t>& mat_bf16,
                                       const std::vector<float>& vec_f32,
                                       std::vector<float>& out_f32, int rows,
                                       int cols) {
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

static void weighted_sum_probs_v(const std::vector<float>& probs,
                                 const std::vector<float>& V,
                                 std::vector<float>& out, int seq_len,
                                 int head_dim) {
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

static void calculate_attention_scores(const std::vector<float>& Q,
                                       const std::vector<float>& K,
                                       std::vector<float>& scores, int seq_len,
                                       int head_dim, float scale) {
  if (Q.empty() || K.empty()) return;
  scores.resize(seq_len);

  scale = std::clamp(scale, attention::MIN_SCALE, attention::MAX_SCALE);
  float effective_scale = scale * attention::ATTENTION_SCALE_BASE;

#pragma omp parallel for collapse(1)
  for (int64_t i = 0; i < static_cast<int64_t>(seq_len); ++i) {
    double dot_product = 0.0; // Use double for accumulation
    double c_kahan = 0.0;     // Kahan summation compensation
    size_t k_offset = static_cast<size_t>(i) * head_dim;

    for (int j = 0; j < head_dim; ++j) {
      // Kahan Summation for dot product
      double term = static_cast<double>(Q[j]) * static_cast<double>(K[k_offset + j]);
      double y = term - c_kahan;
      double t_sum = dot_product + y;
      c_kahan = (t_sum - dot_product) - y;
      dot_product = t_sum;
    }
    
    scores[i] = static_cast<float>(dot_product * effective_scale); // Apply effective_scale
  }
}


void TinyLlamaModel::initialize_weights(const SafeTensorsLoader* loader,
                                        const GGUFData* gguf) {
  Logger::info("Initializing model weights...");
  int hs = config_.hidden_size;
  int is = config_.intermediate_size;
  int nhl = config_.num_hidden_layers;
  int vs = config_.vocab_size;
  layers.resize(nhl);

  if (gguf) {
    Logger::info("Mapping weights from GGUF data...");
    map_gguf_weights(*gguf, *this);
  } else if (loader) {
    Logger::info("Loading weights from SafeTensors data using parallel loader...");

    std::map<std::string, std::vector<uint8_t>> all_tensors_bytes_map;
    try {
      all_tensors_bytes_map = loader->load_all_tensors_parallel();
      Logger::info("All SafeTensors tensors loaded in parallel. Total tensors: " +
                   std::to_string(all_tensors_bytes_map.size()));
    } catch (const std::exception& e) {
      Logger::error("Failed to load all tensors in parallel: " +
                    std::string(e.what()));
      throw;
    }

    auto process_safetensor = [&](const std::string& name, std::vector<float>& target_f32_vector, const std::vector<size_t>& expected_shape_elements) {
      auto it_bytes = all_tensors_bytes_map.find(name);
      if (it_bytes == all_tensors_bytes_map.end()) {
        bool is_norm_weight = (name.find("input_layernorm.weight") != std::string::npos) ||
                              (name.find("post_attention_layernorm.weight") != std::string::npos) ||
                              (name == "model.norm.weight");
        if (is_norm_weight) {
          throw std::runtime_error("Essential normalization weight tensor \'" + name + "\' not found in SafeTensors model. This layer cannot be correctly processed.");
        }
        Logger::error("Tensor '" + name + "' not found in preloaded map.");
        size_t total_expected_elements = 1;
        for(size_t dim : expected_shape_elements) total_expected_elements *= dim;
        target_f32_vector.assign(total_expected_elements, 0.0f); // Fill with zeros for non-essential missing tensors
        return;
      }
      const std::vector<uint8_t>& tensor_data_bytes = it_bytes->second;
      const SafeTensorsLoader::TensorInfo& tensor_info = loader->get_tensor_info(name);

      size_t expected_num_elements = 1;
      for(size_t dim : tensor_info.shape) expected_num_elements *= dim; // Use shape from metadata

      if (tensor_info.dtype == "BF16" || tensor_info.dtype == "F16" || tensor_info.dtype == "F32") {
        if (tensor_data_bytes.size() != expected_num_elements * sizeof(float)) {
          Logger::error("Size mismatch for tensor '" + name + "' (original dtype " + tensor_info.dtype + ", expected FP32 bytes). Expected " +
                        std::to_string(expected_num_elements * sizeof(float)) + " bytes, got " +
                        std::to_string(tensor_data_bytes.size()) + " bytes.");
          target_f32_vector.assign(expected_num_elements, 0.0f); // Fill with zeros
          return;
        }
        target_f32_vector.resize(expected_num_elements);
        memcpy(target_f32_vector.data(), tensor_data_bytes.data(), tensor_data_bytes.size());
        Logger::info("Loaded tensor '" + name + "' (original dtype: " + tensor_info.dtype + ") as FP32 into target vector.");
      } else {
        Logger::error("Unsupported dtype '" + tensor_info.dtype + "' for tensor '" + name + "' in SafeTensors CPU path after loader conversion attempts.");
        target_f32_vector.assign(expected_num_elements, 0.0f); // Fill with zeros
      }
    };

    process_safetensor("model.embed_tokens.weight", embed_tokens_f32, {(size_t)vs, (size_t)hs});
    process_safetensor("lm_head.weight", lm_head_f32, {(size_t)vs, (size_t)hs});
    process_safetensor("model.norm.weight", final_norm_f32, {(size_t)hs});

    for (int i = 0; i < nhl; ++i) {
      std::string prefix = "model.layers." + std::to_string(i) + ".";
      auto& lw = layers[i];

      process_safetensor(prefix + "self_attn.q_proj.weight", lw.q_proj_f32, {(size_t)hs, (size_t)hs});
      process_safetensor(prefix + "self_attn.k_proj.weight", lw.k_proj_f32, {(size_t)config_.num_key_value_heads * (hs / config_.num_attention_heads), (size_t)hs});
      process_safetensor(prefix + "self_attn.v_proj.weight", lw.v_proj_f32, {(size_t)config_.num_key_value_heads * (hs / config_.num_attention_heads), (size_t)hs});
      process_safetensor(prefix + "self_attn.o_proj.weight", lw.o_proj_f32, {(size_t)hs, (size_t)hs});
      
      process_safetensor(prefix + "mlp.gate_proj.weight", lw.gate_proj_f32, {(size_t)is, (size_t)hs});
      process_safetensor(prefix + "mlp.up_proj.weight", lw.up_proj_f32, {(size_t)is, (size_t)hs});
      process_safetensor(prefix + "mlp.down_proj.weight", lw.down_proj_f32, {(size_t)hs, (size_t)is});

      process_safetensor(prefix + "input_layernorm.weight", lw.input_layernorm_f32, {(size_t)hs});
      process_safetensor(prefix + "post_attention_layernorm.weight", lw.post_attention_layernorm_f32, {(size_t)hs});
      
    }
    
    if (!embed_tokens_f32.empty() && config_.hidden_size > 0) {
        Logger::info("[DIAGNOSTIC] Inspecting embed_tokens_f32 (SafeTensors Path, New Logic)");
    }

  } else {
    throw std::runtime_error(
        "TinyLlamaModel::initialize_weights called with neither GGUF nor "
        "SafeTensors loader.");
  }
  Logger::info("Finished initializing model weights.");
}

void TinyLlamaModel::initialize_gpu_and_rope() {
  Logger::info("[INIT_GPU_ROPE_DEBUG_L1113] Absolute Start of initialize_gpu_and_rope: config_.num_cpu_offload_layers = " + std::to_string(config_.num_cpu_offload_layers) + 
              ", config_.num_hidden_layers = " + std::to_string(config_.num_hidden_layers));
  Logger::info("[GPU_ROPE_INIT_ENTRY] Entered initialize_gpu_and_rope. Requested CPU Offload Layers: " + std::to_string(config_.num_cpu_offload_layers) + ", Total Hidden Layers: " + std::to_string(config_.num_hidden_layers));
  int hs = config_.hidden_size;
  int is = config_.intermediate_size;
  int nhl = config_.num_hidden_layers;
  int vs = config_.vocab_size;
  int n_heads = config_.num_attention_heads;
  int n_kv_heads = config_.num_key_value_heads;

  int num_cpu_layers_clamped = config_.num_cpu_offload_layers;
  if (num_cpu_layers_clamped < 0) num_cpu_layers_clamped = 0;
  if (num_cpu_layers_clamped > nhl) {
      Logger::warning("Requested CPU offload layers (" + std::to_string(config_.num_cpu_offload_layers) +
                      ") exceeds total hidden layers (" + std::to_string(nhl) +
                      "). Clamping to " + std::to_string(nhl) + " layers on CPU.");
      num_cpu_layers_clamped = nhl;
  }
  int active_num_cpu_layers = num_cpu_layers_clamped; 
  int active_num_gpu_layers = nhl - active_num_cpu_layers;

  Logger::info("Effective CPU layers for this init: " + std::to_string(active_num_cpu_layers) + ", Effective GPU layers for this init: " + std::to_string(active_num_gpu_layers));

  if (hs <= 0) throw std::runtime_error("Invalid model config: hidden_size must be positive.");
  if (vs <= 0) throw std::runtime_error("Invalid model config: vocab_size must be positive.");
  if (n_heads <= 0) throw std::runtime_error("Invalid model config: num_attention_heads must be positive.");
  if (n_kv_heads <= 0) throw std::runtime_error("Invalid model config: num_key_value_heads must be positive.");
  if (hs % n_heads != 0) throw std::runtime_error("Invalid model config: hidden_size not divisible by num_attention_heads.");

  int kv_dim = (hs / n_heads) * n_kv_heads;
  int head_dim = hs / n_heads;

  Logger::info("Precomputing RoPE frequencies on CPU (always done).");
  int max_seq_len = config_.max_position_embeddings;
  precomputed_freqs_cis_.resize((max_seq_len * head_dim) / 2);
  float theta = config_.rope_theta;
  for (int pos = 0; pos < max_seq_len; ++pos) {
    for (int i_rope = 0; i_rope < head_dim; i_rope += 2) {
      float freq = std::pow(theta, -((float)i_rope) / head_dim);
      float angle = pos * freq;
      precomputed_freqs_cis_[(pos * head_dim / 2) + (i_rope / 2)] = {std::cos(angle), std::sin(angle)};
    }
  }
  Logger::info("Finished precomputing RoPE cos/sin frequencies on CPU.");

#ifdef HAS_CUDA
#define SAFE_CUDA_FREE(ptr) if(ptr) { cudaFree(ptr); ptr = nullptr; }

  if (active_num_gpu_layers == 0) {
    Logger::info("No layers assigned to GPU (active_num_gpu_layers = 0). Cleaning up existing CUDA resources and skipping GPU initialization.");
    
    SAFE_CUDA_FREE(final_norm_dev);
    for (int i = 0; i < nhl; ++i) { // Clear dev pointers for ALL layers
        SAFE_CUDA_FREE(layers[i].input_layernorm_dev);
        SAFE_CUDA_FREE(layers[i].post_attention_layernorm_dev);
    }
    SAFE_CUDA_FREE(token_embedding_table_dev_);
    SAFE_CUDA_FREE(lm_head_dev_);
    SAFE_CUDA_FREE(w_q_dev_); SAFE_CUDA_FREE(w_k_dev_); SAFE_CUDA_FREE(w_v_dev_); SAFE_CUDA_FREE(w_o_dev_);
    SAFE_CUDA_FREE(w_gate_dev_); SAFE_CUDA_FREE(w_up_dev_); SAFE_CUDA_FREE(w_down_dev_);
    SAFE_CUDA_FREE(all_freqs_cis_dev);
    SAFE_CUDA_FREE(x_dev_); SAFE_CUDA_FREE(x_norm_dev_); SAFE_CUDA_FREE(x_resid1_dev_); SAFE_CUDA_FREE(x_resid2_dev_);
    SAFE_CUDA_FREE(q_dev_); SAFE_CUDA_FREE(k_dev_); SAFE_CUDA_FREE(v_dev_); SAFE_CUDA_FREE(attn_out_dev_);
    SAFE_CUDA_FREE(attn_proj_dev_); SAFE_CUDA_FREE(gate_vec_dev_); SAFE_CUDA_FREE(up_vec_dev_);
    SAFE_CUDA_FREE(swiglu_vec_dev_); SAFE_CUDA_FREE(mlp_down_dev_); SAFE_CUDA_FREE(logits_dev_);
    SAFE_CUDA_FREE(token_embedding_table_f32_dev_);
    SAFE_CUDA_FREE(lm_head_f32_dev_);
    SAFE_CUDA_FREE(w_q_f32_dev_); SAFE_CUDA_FREE(w_k_f32_dev_); SAFE_CUDA_FREE(w_v_f32_dev_); SAFE_CUDA_FREE(w_o_f32_dev_);
    SAFE_CUDA_FREE(w_gate_f32_dev_); SAFE_CUDA_FREE(w_up_f32_dev_); SAFE_CUDA_FREE(w_down_f32_dev_);

    if (cublas_handle_) { cublasDestroy(cublas_handle_); cublas_handle_ = nullptr; }
    return;
  }

  Logger::info("Initializing CUDA resources for " + std::to_string(active_num_gpu_layers) + " GPU layers.");
  if (!cublas_handle_) {
  cublasStatus_t cublas_status = cublasCreate(&cublas_handle_);
  if (cublas_status != CUBLAS_STATUS_SUCCESS) {
      throw std::runtime_error("Failed to initialize cuBLAS: " + std::to_string(cublas_status));
  }
  Logger::info("cuBLAS handle created successfully.");
  
  // Temporarily comment out to diagnose LAUNCH_FAILED
  /*
  if (config_.is_gguf_file_loaded) {
    cublas_status = cublasSetMathMode(cublas_handle_, CUBLAS_PEDANTIC_MATH);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        Logger::warning("Failed to set cuBLAS math mode to PEDANTIC: " + std::to_string(cublas_status) + ". Performance/precision may vary.");
    } else {
        Logger::info("cuBLAS math mode set to CUBLAS_PEDANTIC_MATH for GGUF model to enhance precision.");
    }
  } else {
    Logger::info("Skipping explicit cuBLAS math mode setting for non-GGUF model (using cuBLAS default).");
  }
  */
  Logger::info("Skipping cublasSetMathMode(CUBLAS_PEDANTIC_MATH) for this run as a diagnostic.");

  // Final Norm (always on GPU if any GPU layers are active)
  if (final_norm_f32.empty() && !final_norm.empty()) {
      Logger::info("Converting final_norm (BF16) to FP32 for GPU.");
      final_norm_f32 = bf16vec_to_float_vec(final_norm); // Ensure FP32 version exists
  }
  if (!final_norm_f32.empty()) {
    SAFE_CUDA_FREE(final_norm_dev);
    gpuErrchk(cudaMalloc(&final_norm_dev, final_norm_f32.size() * sizeof(float)));
    gpuErrchk(cudaMemcpy(final_norm_dev, final_norm_f32.data(), final_norm_f32.size() * sizeof(float), cudaMemcpyHostToDevice));
    Logger::info("Copied final_norm weights (FP32) to GPU.");
  } else {
    Logger::warning("Final norm weights (FP32) are empty, skipping GPU copy. This might be an issue if GPU layers are expected to use it.");
  }

  // Layer-specific norms for GPU layers
  // First, clear any existing dev pointers for layers that are now designated for CPU
  for (int i = 0; i < active_num_cpu_layers; ++i) {
      if (static_cast<size_t>(i) < layers.size()) { // Boundary check
        SAFE_CUDA_FREE(layers[i].input_layernorm_dev);
        SAFE_CUDA_FREE(layers[i].post_attention_layernorm_dev);
      }
  }
  Logger::info("Copying layer norm weights (FP32) to GPU for layers " + std::to_string(active_num_cpu_layers) + " to " + std::to_string(nhl - 1));
  Logger::info("[INIT_DEBUG_PRE_LOOP] Active CPU layers: " + std::to_string(active_num_cpu_layers));
  if (nhl > 0 && layers.size() > 0) { // Check if layers exist to prevent out of bounds
    Logger::info("[INIT_DEBUG_PRE_LOOP] layers[0].input_layernorm_f32.empty(): " + std::string(layers[0].input_layernorm_f32.empty() ? "YES" : "NO") + 
                 ", Size: " + std::to_string(layers[0].input_layernorm_f32.size()));
  }
  for (int i = active_num_cpu_layers; i < nhl; ++i) { // Iterate ONLY over GPU layers
    if (static_cast<size_t>(i) >= layers.size()) { // Boundary check
        Logger::error("Layer index " + std::to_string(i) + " out of bounds for layers vector (size: " + std::to_string(layers.size()) + ")");
        continue; 
    }
    SAFE_CUDA_FREE(layers[i].input_layernorm_dev); // Free before realloc, in case of re-initialization
    SAFE_CUDA_FREE(layers[i].post_attention_layernorm_dev); // Free before realloc

    // Ensure FP32 versions of norm weights exist if original was BF16
    if (layers[i].input_layernorm_f32.empty() && !layers[i].input_layernorm.empty()) {
        layers[i].input_layernorm_f32 = bf16vec_to_float_vec(layers[i].input_layernorm);
    }
    if (layers[i].post_attention_layernorm_f32.empty() && !layers[i].post_attention_layernorm.empty()) {
        layers[i].post_attention_layernorm_f32 = bf16vec_to_float_vec(layers[i].post_attention_layernorm);
    }
    
    if (!layers[i].input_layernorm_f32.empty()) {
      gpuErrchk(cudaMalloc(&layers[i].input_layernorm_dev, layers[i].input_layernorm_f32.size() * sizeof(float)));
      gpuErrchk(cudaMemcpy(layers[i].input_layernorm_dev, layers[i].input_layernorm_f32.data(), layers[i].input_layernorm_f32.size() * sizeof(float), cudaMemcpyHostToDevice));
      if (i == active_num_cpu_layers) { // Log only for the first GPU layer processed
          Logger::info("[INIT_DEBUG] layers[" + std::to_string(i) + "].input_layernorm_dev allocated. Pointer: " + Logger::ptrToString(layers[i].input_layernorm_dev) + 
                       ", Size used for malloc: " + std::to_string(layers[i].input_layernorm_f32.size() * sizeof(float)) + " bytes (" +
                       std::to_string(layers[i].input_layernorm_f32.size()) + " elements). Host vector empty: " + (layers[i].input_layernorm_f32.empty() ? "YES" : "NO"));
      }
    } else {
      // This layer is designated for GPU. It MUST have its norm weights.
      throw std::runtime_error("GPU Layer " + std::to_string(i) + ": input_layernorm_f32 weights are empty. Cannot offload to GPU without them.");
    }
    
    if (!layers[i].post_attention_layernorm_f32.empty()) {
      gpuErrchk(cudaMalloc(&layers[i].post_attention_layernorm_dev, layers[i].post_attention_layernorm_f32.size() * sizeof(float)));
      gpuErrchk(cudaMemcpy(layers[i].post_attention_layernorm_dev, layers[i].post_attention_layernorm_f32.data(), layers[i].post_attention_layernorm_f32.size() * sizeof(float), cudaMemcpyHostToDevice));
    } else {
      // This layer is designated for GPU. It MUST have its norm weights.
      throw std::runtime_error("GPU Layer " + std::to_string(i) + ": post_attention_layernorm_f32 weights are empty. Cannot offload to GPU without them.");
    }
  }
  Logger::info("Finished processing layer norm weights for GPU layers.");


  // --- TOKEN EMBEDDING TABLE to GPU (as BF16) ---
  SAFE_CUDA_FREE(token_embedding_table_dev_);    // Target for BF16
  SAFE_CUDA_FREE(token_embedding_table_f32_dev_); // Ensure this is cleared and not used by new embedding logic

  bool token_embeddings_processed_to_gpu_bf16 = false;

  if (active_num_gpu_layers > 0) { // Only process if GPU layers are active
    // Path 1: Source is already BF16 (model.embed_tokens is std::vector<uint16_t>)
  if (!embed_tokens.empty()) {
    gpuErrchk(cudaMalloc(&token_embedding_table_dev_, embed_tokens.size() * sizeof(uint16_t)));
    gpuErrchk(cudaMemcpy(token_embedding_table_dev_, embed_tokens.data(), embed_tokens.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
      Logger::info("Copied token_embedding_table (bf16 direct from model.embed_tokens) to GPU.");
      token_embeddings_processed_to_gpu_bf16 = true;
    }
    // Path 2: Source is FP32 (model.embed_tokens_f32) -> convert to BF16
    else if (!embed_tokens_f32.empty()) {
      std::vector<uint16_t> bf16_data(embed_tokens_f32.size());
      #pragma omp parallel for
      for (size_t i = 0; i < embed_tokens_f32.size(); ++i) {
        bf16_data[i] = float32_to_bfloat16(embed_tokens_f32[i]);
      }
      gpuErrchk(cudaMalloc(&token_embedding_table_dev_, bf16_data.size() * sizeof(uint16_t)));
      gpuErrchk(cudaMemcpy(token_embedding_table_dev_, bf16_data.data(), bf16_data.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
      Logger::info("Converted token_embedding_table (fp32 source -> bf16) to GPU.");
      token_embeddings_processed_to_gpu_bf16 = true;
    }
    // Path 3: Source is Q8_0 (model.embed_tokens_q8_0) -> dequantize to FP32, then convert to BF16
    else if (!embed_tokens_q8_0.empty()) {
      std::vector<float> temp_f32_data(embed_tokens_q8_0.size() * GGML_QK8_0);
      #pragma omp parallel for
      for (size_t i = 0; i < embed_tokens_q8_0.size(); ++i) {
        dequantize_q8_0_block(&embed_tokens_q8_0[i], &temp_f32_data[i * GGML_QK8_0]);
      }
      std::vector<uint16_t> bf16_data(temp_f32_data.size());
      #pragma omp parallel for
      for (size_t i = 0; i < temp_f32_data.size(); ++i) {
        bf16_data[i] = float32_to_bfloat16(temp_f32_data[i]);
      }
      gpuErrchk(cudaMalloc(&token_embedding_table_dev_, bf16_data.size() * sizeof(uint16_t)));
      gpuErrchk(cudaMemcpy(token_embedding_table_dev_, bf16_data.data(), bf16_data.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
      Logger::info("Dequantized token_embedding_table (Q8_0 -> fp32 -> bf16) to GPU.");
      token_embeddings_processed_to_gpu_bf16 = true;
    }
    // Path 4: Source is Q4_K (model.embed_tokens_q4k) -> dequantize to FP32, then convert to BF16
    else if (!embed_tokens_q4k.empty()) {
      std::vector<float> temp_f32_data(embed_tokens_q4k.size() * GGML_QK_K);
      #pragma omp parallel for
      for (size_t i = 0; i < embed_tokens_q4k.size(); ++i) {
        dequantize_q4_k_m(&embed_tokens_q4k[i], &temp_f32_data[i * GGML_QK_K], GGML_QK_K);
      }
      std::vector<uint16_t> bf16_data(temp_f32_data.size());
      #pragma omp parallel for
      for (size_t i = 0; i < temp_f32_data.size(); ++i) {
        bf16_data[i] = float32_to_bfloat16(temp_f32_data[i]);
      }
      gpuErrchk(cudaMalloc(&token_embedding_table_dev_, bf16_data.size() * sizeof(uint16_t)));
      gpuErrchk(cudaMemcpy(token_embedding_table_dev_, bf16_data.data(), bf16_data.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
      Logger::info("Dequantized token_embedding_table (Q4_K -> fp32 -> bf16) to GPU.");
      token_embeddings_processed_to_gpu_bf16 = true;
    }
    // Path 5: Source is Q6_K (model.embed_tokens_q6k) -> dequantize to FP32, then convert to BF16
    else if (!embed_tokens_q6k.empty()) {
      std::vector<float> temp_f32_data(embed_tokens_q6k.size() * GGML_QK_K);
      #pragma omp parallel for
      for (size_t i = 0; i < embed_tokens_q6k.size(); ++i) {
        dequantize_q6_k(&embed_tokens_q6k[i], &temp_f32_data[i * GGML_QK_K], GGML_QK_K);
      }
      std::vector<uint16_t> bf16_data(temp_f32_data.size());
      #pragma omp parallel for
      for (size_t i = 0; i < temp_f32_data.size(); ++i) {
        bf16_data[i] = float32_to_bfloat16(temp_f32_data[i]);
      }
      gpuErrchk(cudaMalloc(&token_embedding_table_dev_, bf16_data.size() * sizeof(uint16_t)));
      gpuErrchk(cudaMemcpy(token_embedding_table_dev_, bf16_data.data(), bf16_data.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
      Logger::info("Dequantized token_embedding_table (Q6_K -> fp32 -> bf16) to GPU.");
      token_embeddings_processed_to_gpu_bf16 = true;
    }

    if (token_embeddings_processed_to_gpu_bf16) {
        Logger::info("[INIT_DEBUG] token_embedding_table_dev_ (BF16 on GPU) processed. Pointer: " + Logger::ptrToString(token_embedding_table_dev_) + 
                     ". Flag token_embeddings_processed_to_gpu_bf16: YES");
    } // This closing brace was for the "if (active_num_gpu_layers > 0)" for token embeddings.
    // The next line is the warning check.
    if (!token_embeddings_processed_to_gpu_bf16 && active_num_gpu_layers > 0) { // Added active_num_gpu_layers check here too
        Logger::warning("Token embeddings were not processed to GPU as BF16, despite GPU layers being active. This might indicate missing source embedding data in the model structure or an unhandled GGUF type for embeddings.");
    }
  } else {
    Logger::info("No GPU layers active, skipping token embedding table processing for GPU.");
  }

  // --- LM HEAD to GPU (as BF16) ---
  SAFE_CUDA_FREE(lm_head_dev_);    // Target for BF16
  SAFE_CUDA_FREE(lm_head_f32_dev_); // Ensure this is cleared and not used by new LM head logic

  bool lm_head_processed_to_gpu_bf16 = false;

  if (active_num_gpu_layers > 0) { // Only process if GPU layers are active
    // Path 1: Source is already BF16 (model.lm_head is std::vector<uint16_t>)
  if (!lm_head.empty()) {
    gpuErrchk(cudaMalloc(&lm_head_dev_, lm_head.size() * sizeof(uint16_t)));
    gpuErrchk(cudaMemcpy(lm_head_dev_, lm_head.data(), lm_head.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
      Logger::info("Copied lm_head (bf16 direct from model.lm_head) to GPU.");
      lm_head_processed_to_gpu_bf16 = true;
    }
    // Path 2: Source is FP32 (model.lm_head_f32) -> convert to BF16
    else if (!lm_head_f32.empty()) {
      std::vector<uint16_t> bf16_data(lm_head_f32.size());
      #pragma omp parallel for
      for (size_t i = 0; i < lm_head_f32.size(); ++i) {
        bf16_data[i] = float32_to_bfloat16(lm_head_f32[i]);
      }
      gpuErrchk(cudaMalloc(&lm_head_dev_, bf16_data.size() * sizeof(uint16_t)));
      gpuErrchk(cudaMemcpy(lm_head_dev_, bf16_data.data(), bf16_data.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
      Logger::info("Converted lm_head (fp32 source -> bf16) to GPU.");
      lm_head_processed_to_gpu_bf16 = true;
    }
    // Path 3: Source is Q8_0 (model.lm_head_q8_0) -> dequantize to FP32, then convert to BF16
    else if (!lm_head_q8_0.empty()) {
      std::vector<float> temp_f32_data(lm_head_q8_0.size() * GGML_QK8_0);
      #pragma omp parallel for
      for (size_t i = 0; i < lm_head_q8_0.size(); ++i) {
        dequantize_q8_0_block(&lm_head_q8_0[i], &temp_f32_data[i * GGML_QK8_0]);
      }
      std::vector<uint16_t> bf16_data(temp_f32_data.size());
      #pragma omp parallel for
      for (size_t i = 0; i < temp_f32_data.size(); ++i) {
        bf16_data[i] = float32_to_bfloat16(temp_f32_data[i]);
      }
      gpuErrchk(cudaMalloc(&lm_head_dev_, bf16_data.size() * sizeof(uint16_t)));
      gpuErrchk(cudaMemcpy(lm_head_dev_, bf16_data.data(), bf16_data.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
      Logger::info("Dequantized lm_head (Q8_0 -> fp32 -> bf16) to GPU.");
      lm_head_processed_to_gpu_bf16 = true;
    }
    // Path 4: Source is Q4_K (model.lm_head_q4k) -> dequantize to FP32, then convert to BF16
    else if (!lm_head_q4k.empty()) {
      std::vector<float> temp_f32_data(lm_head_q4k.size() * GGML_QK_K);
      #pragma omp parallel for
      for (size_t i = 0; i < lm_head_q4k.size(); ++i) {
        dequantize_q4_k_m(&lm_head_q4k[i], &temp_f32_data[i * GGML_QK_K], GGML_QK_K);
      }
      std::vector<uint16_t> bf16_data(temp_f32_data.size());
      #pragma omp parallel for
      for (size_t i = 0; i < temp_f32_data.size(); ++i) {
        bf16_data[i] = float32_to_bfloat16(temp_f32_data[i]);
      }
      gpuErrchk(cudaMalloc(&lm_head_dev_, bf16_data.size() * sizeof(uint16_t)));
      gpuErrchk(cudaMemcpy(lm_head_dev_, bf16_data.data(), bf16_data.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
      Logger::info("Dequantized lm_head (Q4_K -> fp32 -> bf16) to GPU.");
      lm_head_processed_to_gpu_bf16 = true;
    }
    // Path 5: Source is Q6_K (model.lm_head_q6k) -> dequantize to FP32, then convert to BF16
    else if (!lm_head_q6k.empty()) {
      std::vector<float> temp_f32_data(lm_head_q6k.size() * GGML_QK_K);
      #pragma omp parallel for
      for (size_t i = 0; i < lm_head_q6k.size(); ++i) {
        dequantize_q6_k(&lm_head_q6k[i], &temp_f32_data[i * GGML_QK_K], GGML_QK_K);
      }
      std::vector<uint16_t> bf16_data(temp_f32_data.size());
      #pragma omp parallel for
      for (size_t i = 0; i < temp_f32_data.size(); ++i) {
        bf16_data[i] = float32_to_bfloat16(temp_f32_data[i]);
      }
      gpuErrchk(cudaMalloc(&lm_head_dev_, bf16_data.size() * sizeof(uint16_t)));
      gpuErrchk(cudaMemcpy(lm_head_dev_, bf16_data.data(), bf16_data.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
      Logger::info("Dequantized lm_head (Q6_K -> fp32 -> bf16) to GPU.");
      lm_head_processed_to_gpu_bf16 = true;
    }

    if (!lm_head_processed_to_gpu_bf16) {
        Logger::warning("LM head was not processed to GPU as BF16, despite GPU layers being active. This might indicate missing source LM head data in the model structure or an unhandled GGUF type for LM head.");
    }
  } else {
    Logger::info("No GPU layers active, skipping LM head processing for GPU.");
  }
  // --- END LM HEAD ---
  
  Logger::info("Finished processing embedding and LM head tables for GPU.");

  // RoPE GPU Buffer
  SAFE_CUDA_FREE(all_freqs_cis_dev);
  if (!precomputed_freqs_cis_.empty()) { // RoPE freqs are always precomputed on CPU
    size_t total_freq_elements = precomputed_freqs_cis_.size() * 2;
    gpuErrchk(cudaMalloc(&all_freqs_cis_dev, total_freq_elements * sizeof(float)));
    std::vector<float> flat_host_freqs; flat_host_freqs.reserve(total_freq_elements);
    for (const auto& p : precomputed_freqs_cis_) { flat_host_freqs.push_back(p.first); flat_host_freqs.push_back(p.second); }
    gpuErrchk(cudaMemcpy(all_freqs_cis_dev, flat_host_freqs.data(), total_freq_elements * sizeof(float), cudaMemcpyHostToDevice));
    Logger::info("Copied all precomputed RoPE frequencies to persistent GPU buffer.");
  } else {
    Logger::warning("Host precomputed_freqs_cis_ is empty. Skipping GPU RoPE buffer allocation. This WILL cause issues if GPU layers use RoPE.");
  }
  Logger::info("Finished processing RoPE frequencies for GPU.");

  // Workspace GPU Buffers
  Logger::info("Allocating/Reallocating persistent GPU workspace buffers.");
  size_t hs_bytes = (size_t)hs * sizeof(float);
  size_t is_bytes = (size_t)is * sizeof(float);
  size_t vs_bytes = (size_t)vs * sizeof(float);
  size_t k_dev_size_bytes = (size_t)n_kv_heads * head_dim * sizeof(float);
  size_t v_dev_size_bytes = (size_t)n_kv_heads * head_dim * sizeof(float);

#define REALLOC_GPU_WORKSPACE(ptr, sz) SAFE_CUDA_FREE(ptr); gpuErrchk(cudaMalloc(&ptr, sz));
  REALLOC_GPU_WORKSPACE(x_dev_, hs_bytes);
  REALLOC_GPU_WORKSPACE(x_norm_dev_, hs_bytes);
  REALLOC_GPU_WORKSPACE(x_resid1_dev_, hs_bytes);
  REALLOC_GPU_WORKSPACE(x_resid2_dev_, hs_bytes);
  REALLOC_GPU_WORKSPACE(q_dev_, hs_bytes); // q_dev for Q projection output is full hidden size
  REALLOC_GPU_WORKSPACE(k_dev_, k_dev_size_bytes); // k_dev for K projection output
  REALLOC_GPU_WORKSPACE(v_dev_, v_dev_size_bytes); // v_dev for V projection output
  REALLOC_GPU_WORKSPACE(attn_out_dev_, hs_bytes);
  REALLOC_GPU_WORKSPACE(attn_proj_dev_, hs_bytes); 
  REALLOC_GPU_WORKSPACE(gate_vec_dev_, is_bytes);
  REALLOC_GPU_WORKSPACE(up_vec_dev_, is_bytes);
  REALLOC_GPU_WORKSPACE(swiglu_vec_dev_, is_bytes);
  REALLOC_GPU_WORKSPACE(mlp_down_dev_, hs_bytes); 
  REALLOC_GPU_WORKSPACE(logits_dev_, vs_bytes); // For final logits calculation on GPU

  // Allocate KVCache dequantization buffers if GPU layers are active
  if (active_num_gpu_layers > 0) {
    size_t kv_cache_dequant_buffer_elems = static_cast<size_t>(config_.max_position_embeddings) * n_kv_heads * head_dim;
    size_t kv_cache_dequant_buffer_bytes = kv_cache_dequant_buffer_elems * sizeof(float);
    if (kv_cache_dequant_buffer_elems > 0) {
        SAFE_CUDA_FREE(dequant_k_cache_buffer_dev_); 
        gpuErrchk(cudaMalloc(&dequant_k_cache_buffer_dev_, kv_cache_dequant_buffer_bytes));
        SAFE_CUDA_FREE(dequant_v_cache_buffer_dev_);
        gpuErrchk(cudaMalloc(&dequant_v_cache_buffer_dev_, kv_cache_dequant_buffer_bytes));
        Logger::info("Allocated KVCache dequantization buffers (K and V) on GPU. Size per buffer: " + 
                     std::to_string(kv_cache_dequant_buffer_bytes / (1024.0 * 1024.0)) + " MB.");
    } else {
        Logger::warning("KVCache dequantization buffer size is 0. Skipping allocation. max_pos_emb=" + std::to_string(config_.max_position_embeddings) +
                        ", n_kv_heads=" + std::to_string(n_kv_heads) + ", head_dim=" + std::to_string(head_dim));
        SAFE_CUDA_FREE(dequant_k_cache_buffer_dev_); // Ensure they are null if size is 0
        SAFE_CUDA_FREE(dequant_v_cache_buffer_dev_);
    }
  } else { // No active GPU layers
    SAFE_CUDA_FREE(dequant_k_cache_buffer_dev_);
    SAFE_CUDA_FREE(dequant_v_cache_buffer_dev_);
  }

// #undef REALLOC_GPU_WORKSPACE // Not needed if we expanded it or it was already defined
  Logger::info("Finished allocating/reallocating GPU workspace buffers.");

  // Concatenated BF16 layer weights for GPU layers (w_..._dev_ pointers)
  // Check if the first active GPU layer has BF16 q_proj weights to decide if this path should be taken.
  bool process_bf16_concat_weights = active_num_gpu_layers > 0 && !layers[active_num_cpu_layers].q_proj.empty();
  if (process_bf16_concat_weights) {
    size_t layer_q_size = (size_t)hs*hs, layer_k_size = (size_t)kv_dim*hs, layer_v_size = (size_t)kv_dim*hs, layer_o_size = (size_t)hs*hs;
    size_t layer_gate_size = (size_t)is*hs, layer_up_size = (size_t)is*hs, layer_down_size = (size_t)hs*is;
    
    std::vector<uint16_t> h_q, h_k, h_v, h_o, h_gate, h_up, h_down; // h for host
    h_q.reserve(active_num_gpu_layers * layer_q_size); h_k.reserve(active_num_gpu_layers * layer_k_size);
    h_v.reserve(active_num_gpu_layers * layer_v_size); h_o.reserve(active_num_gpu_layers * layer_o_size);
    h_gate.reserve(active_num_gpu_layers * layer_gate_size); h_up.reserve(active_num_gpu_layers * layer_up_size);
    h_down.reserve(active_num_gpu_layers * layer_down_size);

    Logger::info("Concatenating BF16 weights for GPU layers on host (zero-padding if missing for a layer)...");
    for (int i = 0; i < active_num_gpu_layers; ++i) {
      int model_layer_idx = active_num_cpu_layers + i;
      const auto& lw = layers[model_layer_idx];
      
      if (!lw.q_proj.empty()) {
        h_q.insert(h_q.end(), lw.q_proj.begin(), lw.q_proj.end());
      } else {
        h_q.insert(h_q.end(), layer_q_size, bfloat16::ZERO);
      }

      if (!lw.k_proj.empty()) {
        h_k.insert(h_k.end(), lw.k_proj.begin(), lw.k_proj.end());
      } else {
        h_k.insert(h_k.end(), layer_k_size, bfloat16::ZERO);
      }

      if (!lw.v_proj.empty()) {
        h_v.insert(h_v.end(), lw.v_proj.begin(), lw.v_proj.end());
      } else {
        h_v.insert(h_v.end(), layer_v_size, bfloat16::ZERO);
      }

      if (!lw.o_proj.empty()) {
        h_o.insert(h_o.end(), lw.o_proj.begin(), lw.o_proj.end());
      } else {
        h_o.insert(h_o.end(), layer_o_size, bfloat16::ZERO);
      }

      if (!lw.gate_proj.empty()) {
        h_gate.insert(h_gate.end(), lw.gate_proj.begin(), lw.gate_proj.end());
      } else {
        h_gate.insert(h_gate.end(), layer_gate_size, bfloat16::ZERO);
      }

      if (!lw.up_proj.empty()) {
        h_up.insert(h_up.end(), lw.up_proj.begin(), lw.up_proj.end());
      } else {
        h_up.insert(h_up.end(), layer_up_size, bfloat16::ZERO);
      }

      if (!lw.down_proj.empty()) {
        h_down.insert(h_down.end(), lw.down_proj.begin(), lw.down_proj.end());
      } else {
        h_down.insert(h_down.end(), layer_down_size, bfloat16::ZERO);
      }
    }

#define ALLOC_COPY_CONCAT_BF16(dev_ptr, host_vec, weight_name_str) \
    SAFE_CUDA_FREE(dev_ptr); \
    if (!host_vec.empty()) { \
        gpuErrchk(cudaMalloc(&dev_ptr, host_vec.size() * sizeof(uint16_t))); \
        gpuErrchk(cudaMemcpy(dev_ptr, host_vec.data(), host_vec.size() * sizeof(uint16_t), cudaMemcpyHostToDevice)); \
        Logger::info("Copied concatenated " weight_name_str " (BF16) to GPU for GPU layers."); \
    } else if (active_num_gpu_layers > 0) { \
        Logger::info("Host vector for concatenated " weight_name_str " (BF16) is empty. Skipping GPU copy."); \
    }

    ALLOC_COPY_CONCAT_BF16(w_q_dev_, h_q, "Q Proj"); ALLOC_COPY_CONCAT_BF16(w_k_dev_, h_k, "K Proj"); ALLOC_COPY_CONCAT_BF16(w_v_dev_, h_v, "V Proj");
    ALLOC_COPY_CONCAT_BF16(w_o_dev_, h_o, "O Proj"); ALLOC_COPY_CONCAT_BF16(w_gate_dev_, h_gate, "Gate Proj"); 
    ALLOC_COPY_CONCAT_BF16(w_up_dev_, h_up, "Up Proj"); ALLOC_COPY_CONCAT_BF16(w_down_dev_, h_down, "Down Proj");
#undef ALLOC_COPY_CONCAT_BF16

  } else if (active_num_gpu_layers > 0) {
    Logger::info("Skipping BF16 concatenated layer weight processing (first GPU layer appears not to use BF16 q_proj, or no GPU layers).");
    SAFE_CUDA_FREE(w_q_dev_); SAFE_CUDA_FREE(w_k_dev_); SAFE_CUDA_FREE(w_v_dev_); SAFE_CUDA_FREE(w_o_dev_);
    SAFE_CUDA_FREE(w_gate_dev_); SAFE_CUDA_FREE(w_up_dev_); SAFE_CUDA_FREE(w_down_dev_);
  }

  // Concatenated FP32 / Dequantized Layer Weights for GPU layers (w_..._f32_dev_ pointers)
  size_t layer_q_size_f32 = (size_t)hs*hs, layer_k_size_f32 = (size_t)kv_dim*hs, layer_v_size_f32 = (size_t)kv_dim*hs;
  size_t layer_o_size_f32 = (size_t)hs*hs, layer_gate_size_f32 = (size_t)is*hs, layer_up_size_f32 = (size_t)is*hs;
  size_t layer_down_size_f32 = (size_t)hs*is;

  auto process_gpu_layer_weights_to_f32_concat = [&] (
      const std::function<const std::vector<float>&(const LayerWeights&)>& f32_accessor,
      const std::function<const std::vector<uint16_t>&(const LayerWeights&)>& bf16_accessor,
      const std::function<const std::vector<block_q8_0>&(const LayerWeights&)>& q8_accessor,
      const std::function<const std::vector<block_q4_K>&(const LayerWeights&)>& q4k_accessor,
      const std::function<const std::vector<block_q6_K>&(const LayerWeights&)>& q6k_accessor,
      float*& dev_ptr, size_t single_layer_elem_size, const std::string& name_base) {
    
    SAFE_CUDA_FREE(dev_ptr); // Free before attempting to populate
    if (active_num_gpu_layers == 0) return;

    std::vector<float> concat_host_f32;
    concat_host_f32.reserve(active_num_gpu_layers * single_layer_elem_size);
    std::string source_type_str = "Unknown"; // For logging the type of the first processed layer or "Mixed"
    bool first_layer_type_logged = false;

    bool all_layers_consistent = true; // True if all GPU layers provide processable data for this weight

    for (int i = 0; i < active_num_gpu_layers; ++i) {
      int model_layer_idx = active_num_cpu_layers + i;
      const auto& lw = layers[model_layer_idx];
      std::vector<float> temp_layer_f32;
      bool current_layer_processed = false;
      std::string current_layer_source_type_for_log = "None";

      if (config_.is_gguf_file_loaded) {
        // GGUF: Prioritize dequantizing from Q_K types if available
          const auto& q8_blocks = q8_accessor(lw);
          if (!q8_blocks.empty()) {
            temp_layer_f32.resize(q8_blocks.size() * GGML_QK8_0);
            for (size_t bi = 0; bi < q8_blocks.size(); ++bi) {
              dequantize_q8_0_block(&q8_blocks[bi], &temp_layer_f32[bi * GGML_QK8_0]);
            }
            current_layer_processed = true;
          current_layer_source_type_for_log = "GGUF_Q8_0->F32";
          } else {
            const auto& q4k_blocks = q4k_accessor(lw);
            if (!q4k_blocks.empty()) {
              temp_layer_f32.resize(q4k_blocks.size() * GGML_QK_K);
              for (size_t bi = 0; bi < q4k_blocks.size(); ++bi) {
                dequantize_q4_k_m(&q4k_blocks[bi], &temp_layer_f32[bi * GGML_QK_K], GGML_QK_K);
              }
              current_layer_processed = true;
            current_layer_source_type_for_log = "GGUF_Q4_K->F32";
            } else {
              const auto& q6k_blocks = q6k_accessor(lw);
              if (!q6k_blocks.empty()) {
                temp_layer_f32.resize(q6k_blocks.size() * GGML_QK_K);
                for (size_t bi = 0; bi < q6k_blocks.size(); ++bi) {
                  dequantize_q6_k(&q6k_blocks[bi], &temp_layer_f32[bi * GGML_QK_K], GGML_QK_K);
                }
                current_layer_processed = true;
              current_layer_source_type_for_log = "GGUF_Q6_K->F32";
              }
            }
          }
        // Fallback to F32/BF16 from GGUF only if Q_K types were not found
        if (!current_layer_processed) {
          const auto& f32_data = f32_accessor(lw);
          if (!f32_data.empty()) {
            temp_layer_f32 = f32_data;
            current_layer_processed = true;
            current_layer_source_type_for_log = "GGUF_F32"; // From GGUF F32/F16/BF16
          } 
        }
      } else { // Not a GGUF file (e.g., SafeTensors)
        const auto& f32_data = f32_accessor(lw);
        if (!f32_data.empty()) {
          temp_layer_f32 = f32_data;
          current_layer_processed = true;
          current_layer_source_type_for_log = "ST_F32";
        } else {
          const auto& bf16_data = bf16_accessor(lw);
          if (!bf16_data.empty()) {
            temp_layer_f32 = bf16vec_to_float_vec(bf16_data);
            current_layer_processed = true;
            current_layer_source_type_for_log = "ST_BF16->F32";
          }
        }
      }
      
      if (current_layer_processed) {
        if (!first_layer_type_logged) {
          source_type_str = current_layer_source_type_for_log;
          first_layer_type_logged = true;
        } else if (source_type_str != current_layer_source_type_for_log && source_type_str != "Mixed") {
          Logger::info("Weight " + name_base + " has mixed types across GPU layers. Layer " +
                       std::to_string(model_layer_idx) + " is " + current_layer_source_type_for_log +
                       " while previous determined type was " + source_type_str + ". Final buffer is F32.");
          source_type_str = "Mixed"; // Indicate mixed types encountered
        }
        concat_host_f32.insert(concat_host_f32.end(), temp_layer_f32.begin(), temp_layer_f32.end());
      } else {
        all_layers_consistent = false; 
        Logger::error("Layer " + std::to_string(model_layer_idx) +
                      " for weight " + name_base + " has no F32, BF16, Q8_0, Q4_K, or Q6_K data. Cannot form concatenated GPU tensor.");
        break; 
      }
    }

    if (all_layers_consistent && !concat_host_f32.empty()) {
        if (concat_host_f32.size() != active_num_gpu_layers * single_layer_elem_size) {
            Logger::error("Concatenated host buffer for " + name_base + " has incorrect size. Expected: " +
                          std::to_string(active_num_gpu_layers * single_layer_elem_size) +
                          ", Got: " + std::to_string(concat_host_f32.size()) +
                          ". This indicates an issue with dequantization or data processing for one of the layers. Source type(s): " + source_type_str);
            SAFE_CUDA_FREE(dev_ptr); // Do not upload incorrect buffer
        } else {
            gpuErrchk(cudaMalloc(&dev_ptr, concat_host_f32.size() * sizeof(float)));
            gpuErrchk(cudaMemcpy(dev_ptr, concat_host_f32.data(), concat_host_f32.size() * sizeof(float), cudaMemcpyHostToDevice));
            Logger::info("Uploaded concatenated " + name_base + " (source(s): " + source_type_str + ", final: F32) for " + std::to_string(active_num_gpu_layers) + " GPU layers. Total elements: " + std::to_string(concat_host_f32.size()));
        }
    } else {
        Logger::warning("Could not consistently populate or host buffer empty for concatenated " + name_base + " for GPU layers. Source type(s) identified: " + source_type_str + ". All_consistent: " + (all_layers_consistent?"Y":"N") + ". Host_empty: " + (concat_host_f32.empty()?"Y":"N") + ". GPU tensor will not be created.");
        SAFE_CUDA_FREE(dev_ptr); // Ensure it's freed if allocation failed or data inconsistent
    }
  };
  
  process_gpu_layer_weights_to_f32_concat(
    [](const LayerWeights& lw) -> const std::vector<float>& { return lw.q_proj_f32; }, 
    [](const LayerWeights& lw) -> const std::vector<uint16_t>& { return lw.q_proj; }, 
    [](const LayerWeights& lw) -> const std::vector<block_q8_0>& { return lw.q_proj_q8_0; }, 
    [](const LayerWeights& lw) -> const std::vector<block_q4_K>& { return lw.q_proj_q4k; },
    [](const LayerWeights& lw) -> const std::vector<block_q6_K>& { return lw.q_proj_q6k; },
    w_q_f32_dev_, layer_q_size_f32, "W_Q");
  process_gpu_layer_weights_to_f32_concat(
    [](const LayerWeights& lw) -> const std::vector<float>& { return lw.k_proj_f32; }, 
    [](const LayerWeights& lw) -> const std::vector<uint16_t>& { return lw.k_proj; }, 
    [](const LayerWeights& lw) -> const std::vector<block_q8_0>& { return lw.k_proj_q8_0; }, 
    [](const LayerWeights& lw) -> const std::vector<block_q4_K>& { return lw.k_proj_q4k; },
    [](const LayerWeights& lw) -> const std::vector<block_q6_K>& { return lw.k_proj_q6k; },
    w_k_f32_dev_, layer_k_size_f32, "W_K");
  process_gpu_layer_weights_to_f32_concat(
    [](const LayerWeights& lw) -> const std::vector<float>& { return lw.v_proj_f32; }, 
    [](const LayerWeights& lw) -> const std::vector<uint16_t>& { return lw.v_proj; }, 
    [](const LayerWeights& lw) -> const std::vector<block_q8_0>& { return lw.v_proj_q8_0; }, 
    [](const LayerWeights& lw) -> const std::vector<block_q4_K>& { return lw.v_proj_q4k; },
    [](const LayerWeights& lw) -> const std::vector<block_q6_K>& { return lw.v_proj_q6k; },
    w_v_f32_dev_, layer_v_size_f32, "W_V");
  process_gpu_layer_weights_to_f32_concat(
    [](const LayerWeights& lw) -> const std::vector<float>& { return lw.o_proj_f32; }, 
    [](const LayerWeights& lw) -> const std::vector<uint16_t>& { return lw.o_proj; }, 
    [](const LayerWeights& lw) -> const std::vector<block_q8_0>& { return lw.o_proj_q8_0; }, 
    [](const LayerWeights& lw) -> const std::vector<block_q4_K>& { return lw.o_proj_q4k; },
    [](const LayerWeights& lw) -> const std::vector<block_q6_K>& { return lw.o_proj_q6k; },
    w_o_f32_dev_, layer_o_size_f32, "W_O");
  process_gpu_layer_weights_to_f32_concat(
    [](const LayerWeights& lw) -> const std::vector<float>& { return lw.gate_proj_f32; }, 
    [](const LayerWeights& lw) -> const std::vector<uint16_t>& { return lw.gate_proj; }, 
    [](const LayerWeights& lw) -> const std::vector<block_q8_0>& { return lw.gate_proj_q8_0; }, 
    [](const LayerWeights& lw) -> const std::vector<block_q4_K>& { return lw.gate_proj_q4k; },
    [](const LayerWeights& lw) -> const std::vector<block_q6_K>& { return lw.gate_proj_q6k; },
    w_gate_f32_dev_, layer_gate_size_f32, "W_GATE");
  process_gpu_layer_weights_to_f32_concat(
    [](const LayerWeights& lw) -> const std::vector<float>& { return lw.up_proj_f32; }, 
    [](const LayerWeights& lw) -> const std::vector<uint16_t>& { return lw.up_proj; }, 
    [](const LayerWeights& lw) -> const std::vector<block_q8_0>& { return lw.up_proj_q8_0; }, 
    [](const LayerWeights& lw) -> const std::vector<block_q4_K>& { return lw.up_proj_q4k; },
    [](const LayerWeights& lw) -> const std::vector<block_q6_K>& { return lw.up_proj_q6k; },
    w_up_f32_dev_, layer_up_size_f32, "W_UP");
  process_gpu_layer_weights_to_f32_concat(
    [](const LayerWeights& lw) -> const std::vector<float>& { return lw.down_proj_f32; }, 
    [](const LayerWeights& lw) -> const std::vector<uint16_t>& { return lw.down_proj; }, 
    [](const LayerWeights& lw) -> const std::vector<block_q8_0>& { return lw.down_proj_q8_0; }, 
    [](const LayerWeights& lw) -> const std::vector<block_q4_K>& { return lw.down_proj_q4k; },
    [](const LayerWeights& lw) -> const std::vector<block_q6_K>& { return lw.down_proj_q6k; },
    w_down_f32_dev_, layer_down_size_f32, "W_DOWN");

  Logger::info("Finished processing ALL weights for GPU layers (if any).");

#undef SAFE_CUDA_FREE
#else // HAS_CUDA not defined
  if (active_num_gpu_layers > 0 && nhl > 0) {
      Logger::warning("CUDA not available, but " + std::to_string(active_num_gpu_layers) + " layer(s) were configured for GPU. Model will run entirely on CPU.");
  } else {
      Logger::info("CUDA not available or no GPU layers configured. Model will run entirely on CPU.");
  }
#endif // HAS_CUDA
}
}

TinyLlamaModel::TinyLlamaModel(const ModelConfig& config,
                               const SafeTensorsLoader& loader)
    : config_(config) { // Copies the potentially faulty config first
  config_.is_gguf_file_loaded = false; // Explicitly set to false for SafeTensors path
  Logger::info("Constructing TinyLlamaModel from SafeTensorsLoader (is_gguf_file_loaded set to false).");
  initialize_weights(&loader, nullptr);
  initialize_gpu_and_rope();
  Logger::info("TinyLlamaModel construction from SafeTensorsLoader complete.");
}

TinyLlamaModel::TinyLlamaModel(const ModelConfig& initial_config,
                               const std::string& model_path)
    : model_path_(model_path) 
#ifdef HAS_CUDA
      , cublas_handle_(nullptr), token_embedding_table_dev_(nullptr), lm_head_dev_(nullptr), final_norm_dev(nullptr), w_q_dev_(nullptr), w_k_dev_(nullptr), w_v_dev_(nullptr), w_o_dev_(nullptr), w_gate_dev_(nullptr), w_up_dev_(nullptr), w_down_dev_(nullptr), all_freqs_cis_dev(nullptr), x_dev_(nullptr), x_norm_dev_(nullptr), x_resid1_dev_(nullptr), x_resid2_dev_(nullptr), q_dev_(nullptr), k_dev_(nullptr), v_dev_(nullptr), attn_out_dev_(nullptr), attn_proj_dev_(nullptr), gate_vec_dev_(nullptr), up_vec_dev_(nullptr), swiglu_vec_dev_(nullptr), mlp_down_dev_(nullptr), logits_dev_(nullptr), token_embedding_table_f32_dev_(nullptr), lm_head_f32_dev_(nullptr), w_q_f32_dev_(nullptr), w_k_f32_dev_(nullptr), w_v_f32_dev_(nullptr), w_o_f32_dev_(nullptr), w_gate_f32_dev_(nullptr), w_up_f32_dev_(nullptr), w_down_f32_dev_(nullptr)
#endif
{
  Logger::info("TinyLlamaModel constructor entered. Model path (from string): " + model_path);
  int cli_gpu_layer_request = initial_config.num_cpu_offload_layers; 
  bool cli_mmap_preference = initial_config.use_mmap_for_gguf;

  Logger::info("TinyLlamaModel constructor (from string): initial_config CLI hint for num_gpu_layers_requested = " + std::to_string(cli_gpu_layer_request));
  Logger::info("TinyLlamaModel constructor (from string): initial_config CLI hint for use_mmap = " + std::string(cli_mmap_preference ? "true" : "false"));

  this->config_ = initial_config; // Start with initial_config, then overwrite with GGUF specifics

  if (this->model_path_.empty() && !model_path.empty()) {
      this->model_path_ = model_path;
  }

  std::unique_ptr<SafeTensorsLoader> loader = nullptr;

  if (!this->model_path_.empty() && this->model_path_.size() > 5 &&
      this->model_path_.substr(this->model_path_.size() - 5) == ".gguf") {
    Logger::info("GGUF file detected by path in Model Constructor: " + this->model_path_);
    try {
      bool force_mmap_for_gguf_load = true; 
      Logger::info("TinyLlamaModel GGUF path: Forcing mmap to " + std::string(force_mmap_for_gguf_load ? "true" : "false") + 
                   " for gguf_meta/weight loading. Initial CLI mmap preference was: " + 
                   std::string(cli_mmap_preference ? "true" : "false"));

      this->gguf_data_ = std::make_unique<GGUFData>(load_gguf_meta(this->model_path_, force_mmap_for_gguf_load));
      
      ModelConfig config_from_gguf = parse_model_config_from_gguf(*(this->gguf_data_));
      
      this->config_ = config_from_gguf;
      Logger::info("[CTOR_GGUF_DEBUG_L1827] After parse_model_config_from_gguf: config_from_gguf.num_hidden_layers = " + std::to_string(config_from_gguf.num_hidden_layers) + 
                    ", config_from_gguf.num_cpu_offload_layers (raw from GGUF meta) = " + std::to_string(config_from_gguf.num_cpu_offload_layers));
      Logger::info("[CTOR_GGUF_DEBUG_L1828] Copied to this->config_ (before CLI hint processing): this->config_.num_hidden_layers = " + std::to_string(this->config_.num_hidden_layers) + 
                    ", this->config_.num_cpu_offload_layers = " + std::to_string(this->config_.num_cpu_offload_layers));

      this->config_.use_mmap_for_gguf = cli_mmap_preference; // Honor CLI mmap preference for session's use of mmap later
      this->config_.is_gguf_file_loaded = true;

      Logger::info("Successfully parsed GGUF metadata. Model name: " + this->config_.model_name);
      Logger::info("TinyLlamaModel GGUF Ctor: After GGUF parse: config_.num_hidden_layers = " + 
                   std::to_string(this->config_.num_hidden_layers) + 
                   ", cli_gpu_layer_request (from initial_config) = " + std::to_string(cli_gpu_layer_request));

      Logger::info("TinyLlamaModel GGUF Ctor: PRE-CALC this->config_.num_cpu_offload_layers (before GGUF specific logic) = " + std::to_string(this->config_.num_cpu_offload_layers) + 
                   " (This value is from GGUF metadata or default, about to be overridden by CLI hint)");
      Logger::info("[CTOR_GGUF_DEBUG_L1839] Before CLI hint logic: this->config_.num_cpu_offload_layers = " + std::to_string(this->config_.num_cpu_offload_layers) +
                  ", this->config_.num_hidden_layers = " + std::to_string(this->config_.num_hidden_layers) + ", cli_gpu_layer_request = " + std::to_string(cli_gpu_layer_request));    
      if (cli_gpu_layer_request < 0) {
        this->config_.num_cpu_offload_layers = 0;
        Logger::info("TinyLlamaModel GGUF Ctor CALC: CLI hint < 0 (all GPU). num_cpu_offload_layers set to 0.");
      } else if (cli_gpu_layer_request == 0) {
        this->config_.num_cpu_offload_layers = this->config_.num_hidden_layers;
        Logger::info("TinyLlamaModel GGUF Ctor CALC: CLI hint == 0 (all CPU). num_cpu_offload_layers set to num_hidden_layers (" + std::to_string(this->config_.num_cpu_offload_layers) + ").");
      } else { // CLI hint > 0, meaning cli_gpu_layer_request is the number of desired GPU layers
        if (this->config_.num_hidden_layers > 0) {
            if (cli_gpu_layer_request >= this->config_.num_hidden_layers) {
                this->config_.num_cpu_offload_layers = 0; // More GPU layers requested than available -> all on GPU
                Logger::info("TinyLlamaModel GGUF Ctor CALC: CLI GPU layer request ("+ std::to_string(cli_gpu_layer_request) +") >= total layers. num_cpu_offload_layers set to 0.");
            } else {
                this->config_.num_cpu_offload_layers = this->config_.num_hidden_layers - cli_gpu_layer_request;
                Logger::info("TinyLlamaModel GGUF Ctor CALC: Partial GPU. CLI GPU req: " + std::to_string(cli_gpu_layer_request) + ". num_cpu_offload_layers set to " + std::to_string(this->config_.num_cpu_offload_layers));
            }
        } else { // num_hidden_layers is 0 or negative, something is wrong with GGUF. Default to all CPU.
            this->config_.num_cpu_offload_layers = 0; 
            Logger::warning("TinyLlamaModel GGUF Ctor CALC: num_hidden_layers from GGUF is <= 0. Defaulting num_cpu_offload_layers to 0. CLI GPU req: " + std::to_string(cli_gpu_layer_request));
        }
      }
      Logger::info("TinyLlamaModel GGUF Ctor: POST-CALC (within GGUF block) final num_cpu_offload_layers = " + std::to_string(this->config_.num_cpu_offload_layers));
      Logger::info("[CTOR_GGUF_DEBUG_L1860] After CLI hint logic: this->config_.num_cpu_offload_layers = " + std::to_string(this->config_.num_cpu_offload_layers) +
                    ", this->config_.num_hidden_layers = " + std::to_string(this->config_.num_hidden_layers));
    } catch (const std::exception& e) {
      Logger::error("Failed to load or parse GGUF file: " + std::string(e.what()));
      throw; 
    }
  } else if (model_path.size() > 12 &&
             model_path.substr(model_path.size() - 12) == ".safetensors") {
    Logger::info("SafeTensors file detected: " + model_path);
    ModelConfig config_from_json; 
    bool json_loaded_successfully = SafeTensorsLoader::load_model_config_from_json(model_path, config_from_json);
    
    // For SafeTensors, start with JSON config, then layer CLI preferences.
    if (json_loaded_successfully) {
        Logger::info("Successfully loaded and parsed config.json for SafeTensors model.");
        this->config_ = config_from_json; // Base is from JSON
    } else {
        Logger::warning("Failed to load config.json or it was not found for SafeTensors model. Proceeding with initial_config defaults and CLI overrides.");
    }
        this->config_.is_gguf_file_loaded = false;
    this->config_.use_mmap_for_gguf = cli_mmap_preference; // This field is GGUF specific, but store CLI pref anyway.

    // Calculate num_cpu_offload_layers for SafeTensors based on cli_gpu_layer_request and config's num_hidden_layers
    if (cli_gpu_layer_request < 0) {
        this->config_.num_cpu_offload_layers = 0;
    } else if (cli_gpu_layer_request == 0) {
        this->config_.num_cpu_offload_layers = this->config_.num_hidden_layers;
    } else {
        if (this->config_.num_hidden_layers > 0) {
            if (cli_gpu_layer_request >= this->config_.num_hidden_layers) {
                this->config_.num_cpu_offload_layers = 0;
            } else {
                this->config_.num_cpu_offload_layers = this->config_.num_hidden_layers - cli_gpu_layer_request;
            }
        } else {
            this->config_.num_cpu_offload_layers = 0; // Fallback if num_hidden_layers not known
            Logger::warning("SafeTensors path: num_hidden_layers is 0 from JSON/default. Defaulting num_cpu_offload_layers to 0 despite CLI GPU request: " + std::to_string(cli_gpu_layer_request));
        }
    }
    Logger::info("SafeTensors path: Calculated num_cpu_offload_layers = " + std::to_string(this->config_.num_cpu_offload_layers));

    try {
      loader = std::make_unique<SafeTensorsLoader>(model_path);
      Logger::info("SafeTensorsLoader initialized for: " + model_path);
    } catch (const std::exception& e) {
        Logger::error("Failed to initialize SafeTensorsLoader: " + std::string(e.what()));
        throw; 
    }
  } else {
    throw std::runtime_error(
        "Unsupported model file type. Please use .gguf or .safetensors");
  }

  Logger::info("TinyLlamaModel constructor: After specific loader block. Current config_.num_cpu_offload_layers = " + std::to_string(this->config_.num_cpu_offload_layers) + 
               ", config_.num_hidden_layers = " + std::to_string(this->config_.num_hidden_layers));
  Logger::info("TinyLlamaModel constructor: Current config_.use_mmap_for_gguf = " + std::string(this->config_.use_mmap_for_gguf ? "true" : "false"));

  if (this->config_.num_cpu_offload_layers < 0) { // Should not happen if logic above is correct for -1 CLI hint
      this->config_.num_cpu_offload_layers = 0;
      Logger::warning("Clamping num_cpu_offload_layers: was < 0, set to 0.");
  }
  if (this->config_.num_hidden_layers > 0 && this->config_.num_cpu_offload_layers > this->config_.num_hidden_layers) {
      Logger::warning("Clamping num_cpu_offload_layers: Requested CPU offload layers (" + std::to_string(this->config_.num_cpu_offload_layers) +
                      ") exceeds total hidden layers (" + std::to_string(this->config_.num_hidden_layers) +
                      "). Clamping to " + std::to_string(this->config_.num_hidden_layers) + " (all CPU).");
      this->config_.num_cpu_offload_layers = this->config_.num_hidden_layers;
  }
  Logger::info("TinyLlamaModel constructor: Final clamped num_cpu_offload_layers = " + std::to_string(this->config_.num_cpu_offload_layers));
  Logger::info("[CTOR_DEBUG_L1921] End of Model Ctor (before initialize_weights/rope call): this->config_.num_cpu_offload_layers = " + std::to_string(this->config_.num_cpu_offload_layers) +
              ", this->config_.num_hidden_layers = " + std::to_string(this->config_.num_hidden_layers));
  Logger::info("Final ModelConfig (before initialize_weights/rope):");
  Logger::info("  hidden_size: " + std::to_string(config_.hidden_size));
  Logger::info("  intermediate_size: " + std::to_string(config_.intermediate_size));
  Logger::info("  num_attention_heads: " + std::to_string(config_.num_attention_heads));
  Logger::info("  num_key_value_heads: " + std::to_string(config_.num_key_value_heads));
  Logger::info("  num_hidden_layers: " + std::to_string(config_.num_hidden_layers));
  Logger::info("  vocab_size: " + std::to_string(config_.vocab_size));
  Logger::info("  max_position_embeddings: " + std::to_string(config_.max_position_embeddings));
  Logger::info("  architecture: " + config_.architecture);
  Logger::info("  is_gguf_file_loaded: " + std::string(config_.is_gguf_file_loaded ? "true" : "false"));
  Logger::info("  use_mmap_for_gguf: " + std::string(config_.use_mmap_for_gguf ? "true" : "false"));

  initialize_weights(loader.get(), this->gguf_data_.get()); 
  initialize_gpu_and_rope(); 

  Logger::info("TinyLlamaModel (from path string) constructed and initialized successfully.");
}
TinyLlamaModel::TinyLlamaModel(const ModelConfig& config_from_session,
                               std::unique_ptr<GGUFData> gguf_data_from_session)
    : config_(config_from_session), 
      gguf_data_(std::move(gguf_data_from_session)), 
      model_path_("loaded_from_gguf_data_memory")
#ifdef HAS_CUDA
      // Initialize all CUDA pointers to nullptr as in the other constructor
      , cublas_handle_(nullptr), token_embedding_table_dev_(nullptr), lm_head_dev_(nullptr), final_norm_dev(nullptr), w_q_dev_(nullptr), w_k_dev_(nullptr), w_v_dev_(nullptr), w_o_dev_(nullptr), w_gate_dev_(nullptr), w_up_dev_(nullptr), w_down_dev_(nullptr), all_freqs_cis_dev(nullptr), x_dev_(nullptr), x_norm_dev_(nullptr), x_resid1_dev_(nullptr), x_resid2_dev_(nullptr), q_dev_(nullptr), k_dev_(nullptr), v_dev_(nullptr), attn_out_dev_(nullptr), attn_proj_dev_(nullptr), gate_vec_dev_(nullptr), up_vec_dev_(nullptr), swiglu_vec_dev_(nullptr), mlp_down_dev_(nullptr), logits_dev_(nullptr), token_embedding_table_f32_dev_(nullptr), lm_head_f32_dev_(nullptr), w_q_f32_dev_(nullptr), w_k_f32_dev_(nullptr), w_v_f32_dev_(nullptr), w_o_f32_dev_(nullptr), w_gate_f32_dev_(nullptr), w_up_f32_dev_(nullptr), w_down_f32_dev_(nullptr)
#endif
{
    Logger::info("TinyLlamaModel constructor entered (with pre-loaded GGUFData). Model path placeholder: " + model_path_);
    this->config_.is_gguf_file_loaded = true; // Ensure this is set

    if (this->config_.num_cpu_offload_layers < 0) {
        this->config_.num_cpu_offload_layers = 0;
    }
    if (this->config_.num_hidden_layers > 0 && this->config_.num_cpu_offload_layers > this->config_.num_hidden_layers) {
        Logger::warning("Requested CPU offload layers (" + std::to_string(this->config_.num_cpu_offload_layers) +
                        ") exceeds total hidden layers (" + std::to_string(this->config_.num_hidden_layers) +
                        "). Clamping to " + std::to_string(this->config_.num_hidden_layers) + " layers on CPU (all CPU).");
        this->config_.num_cpu_offload_layers = this->config_.num_hidden_layers;
    }
    Logger::info("TinyLlamaModel (pre-loaded GGUF): Final clamped num_cpu_offload_layers = " + std::to_string(this->config_.num_cpu_offload_layers));

    initialize_weights(nullptr, gguf_data_.get()); // Pass raw GGUFData pointer
    initialize_gpu_and_rope();
    Logger::info("TinyLlamaModel (with pre-loaded GGUFData) constructed and initialized successfully.");
}

TinyLlamaModel::~TinyLlamaModel() {
#ifdef HAS_CUDA
  Logger::info("Freeing TinyLlamaModel CUDA resources...");
  if (cublas_handle_) {
    cublasStatus_t cublas_status = cublasDestroy(cublas_handle_);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
      Logger::error("cuBLAS handle destruction failed with error code: " +
                    std::to_string(cublas_status));
    }
    cublas_handle_ = nullptr;
    Logger::info("cuBLAS handle destroyed.");
  }

  if (final_norm_dev) {
    gpuErrchk(cudaFree(final_norm_dev));
    final_norm_dev = nullptr;
  }

  for (auto& layer : layers) {
    if (layer.input_layernorm_dev) {
      gpuErrchk(cudaFree(layer.input_layernorm_dev));
      layer.input_layernorm_dev = nullptr;
    }
    if (layer.post_attention_layernorm_dev) {
      gpuErrchk(cudaFree(layer.post_attention_layernorm_dev));
      layer.post_attention_layernorm_dev = nullptr;
    }
  }

  if (all_freqs_cis_dev) {
    gpuErrchk(cudaFree(all_freqs_cis_dev));
    all_freqs_cis_dev = nullptr;
  }
  if (token_embedding_table_dev_) {
    gpuErrchk(cudaFree(token_embedding_table_dev_));
    token_embedding_table_dev_ = nullptr;
  }
  if (lm_head_dev_) {
    gpuErrchk(cudaFree(lm_head_dev_));
    lm_head_dev_ = nullptr;
  }
  if (w_q_dev_) {
    gpuErrchk(cudaFree(w_q_dev_));
    w_q_dev_ = nullptr;
  }
  if (w_k_dev_) {
    gpuErrchk(cudaFree(w_k_dev_));
    w_k_dev_ = nullptr;
  }
  if (w_v_dev_) {
    gpuErrchk(cudaFree(w_v_dev_));
    w_v_dev_ = nullptr;
  }
  if (w_o_dev_) {
    gpuErrchk(cudaFree(w_o_dev_));
    w_o_dev_ = nullptr;
  }
  if (w_gate_dev_) {
    gpuErrchk(cudaFree(w_gate_dev_));
    w_gate_dev_ = nullptr;
  }
  if (w_up_dev_) {
    gpuErrchk(cudaFree(w_up_dev_));
    w_up_dev_ = nullptr;
  }
  if (w_down_dev_) {
    gpuErrchk(cudaFree(w_down_dev_));
    w_down_dev_ = nullptr;
  }
  if (token_embedding_table_f32_dev_) {
    gpuErrchk(cudaFree(token_embedding_table_f32_dev_));
    token_embedding_table_f32_dev_ = nullptr;
  }
  if (lm_head_f32_dev_) {
    gpuErrchk(cudaFree(lm_head_f32_dev_));
    lm_head_f32_dev_ = nullptr;
  }
  if (w_q_f32_dev_) {
    gpuErrchk(cudaFree(w_q_f32_dev_));
    w_q_f32_dev_ = nullptr;
  }
  if (w_k_f32_dev_) {
    gpuErrchk(cudaFree(w_k_f32_dev_));
    w_k_f32_dev_ = nullptr;
  }
  if (w_v_f32_dev_) {
    gpuErrchk(cudaFree(w_v_f32_dev_));
    w_v_f32_dev_ = nullptr;
  }
  if (w_o_f32_dev_) {
    gpuErrchk(cudaFree(w_o_f32_dev_));
    w_o_f32_dev_ = nullptr;
  }
  if (w_gate_f32_dev_) {
    gpuErrchk(cudaFree(w_gate_f32_dev_));
    w_gate_f32_dev_ = nullptr;
  }
  if (w_up_f32_dev_) {
    gpuErrchk(cudaFree(w_up_f32_dev_));
    w_up_f32_dev_ = nullptr;
  }
  if (w_down_f32_dev_) {
    gpuErrchk(cudaFree(w_down_f32_dev_));
    w_down_f32_dev_ = nullptr;
  }

  if (x_dev_) {
    gpuErrchk(cudaFree(x_dev_));
    x_dev_ = nullptr;
  }
  if (x_norm_dev_) {
    gpuErrchk(cudaFree(x_norm_dev_));
    x_norm_dev_ = nullptr;
  }
  if (x_resid1_dev_) {
    gpuErrchk(cudaFree(x_resid1_dev_));
    x_resid1_dev_ = nullptr;
  }
  if (x_resid2_dev_) {
    gpuErrchk(cudaFree(x_resid2_dev_));
    x_resid2_dev_ = nullptr;
  }
  if (q_dev_) {
    gpuErrchk(cudaFree(q_dev_));
    q_dev_ = nullptr;
  }
  if (k_dev_) {
    gpuErrchk(cudaFree(k_dev_));
    k_dev_ = nullptr;
  }
  if (v_dev_) {
    gpuErrchk(cudaFree(v_dev_));
    v_dev_ = nullptr;
  }
  if (attn_out_dev_) {
    gpuErrchk(cudaFree(attn_out_dev_));
    attn_out_dev_ = nullptr;
  }
  if (attn_proj_dev_) {
    gpuErrchk(cudaFree(attn_proj_dev_));
    attn_proj_dev_ = nullptr;
  }
  if (gate_vec_dev_) {
    gpuErrchk(cudaFree(gate_vec_dev_));
    gate_vec_dev_ = nullptr;
  }
  if (up_vec_dev_) {
    gpuErrchk(cudaFree(up_vec_dev_));
    up_vec_dev_ = nullptr;
  }
  if (swiglu_vec_dev_) {
    gpuErrchk(cudaFree(swiglu_vec_dev_));
    swiglu_vec_dev_ = nullptr;
  }
  if (mlp_down_dev_) {
    gpuErrchk(cudaFree(mlp_down_dev_));
    mlp_down_dev_ = nullptr;
  }
  if (logits_dev_) {
    gpuErrchk(cudaFree(logits_dev_));
    logits_dev_ = nullptr;
  }
  // Free KVCache dequantization buffers
  if (dequant_k_cache_buffer_dev_) {
    gpuErrchk(cudaFree(dequant_k_cache_buffer_dev_));
    dequant_k_cache_buffer_dev_ = nullptr;
  }
  if (dequant_v_cache_buffer_dev_) {
    gpuErrchk(cudaFree(dequant_v_cache_buffer_dev_));
    dequant_v_cache_buffer_dev_ = nullptr;
  }
  Logger::info("Freed persistent GPU workspace buffers.");

  Logger::info("Finished freeing TinyLlamaModel CUDA weight memory.");
#endif
}

std::vector<float> TinyLlamaModel::lookup_embedding(int token_id) {
  int hs = config_.hidden_size;
  int vs = config_.vocab_size;

  bool log_this_token_lookup = (token_id == 660 || token_id == 29901 || token_id == 1724 || token_id == 310 || token_id == 278);

  if (log_this_token_lookup) {
    Logger::info("[CPU_EMBED_DETAIL] Lookup for token_id: " + std::to_string(token_id));
  }

  if (token_id < 0 || token_id >= vs) {
    Logger::error("Token ID out of bounds in lookup_embedding: " +
                  std::to_string(token_id));
    return std::vector<float>(hs, 0.0f);
  }

  std::vector<float> embedding_vec(hs, 0.0f);

  if (!embed_tokens_q4k.empty()) {
    if (hs % GGML_QK_K != 0) {
      Logger::error("Hidden size (" + std::to_string(hs) +
                    ") is not divisible by GGML_QK_K (" +
                    std::to_string(GGML_QK_K) + ") for Q4_K embedding lookup.");
      return embedding_vec;
    }

    size_t blocks_per_row = hs / GGML_QK_K;
    size_t start_block_idx = (size_t)token_id * blocks_per_row;
    size_t end_block_idx = start_block_idx + blocks_per_row;

    if (end_block_idx > embed_tokens_q4k.size()) {
      Logger::error(
          "Calculated block index out of bounds for Q4_K embedding table. "
          "Token: " +
          std::to_string(token_id) +
          ", StartBlock: " + std::to_string(start_block_idx) +
          ", EndBlock: " + std::to_string(end_block_idx) +
          ", TableSize: " + std::to_string(embed_tokens_q4k.size()));
      return embedding_vec;
    }

    float dequantized_block[GGML_QK_K];
    for (size_t block_n = 0; block_n < blocks_per_row; ++block_n) {
      dequantize_q4_k_m(&embed_tokens_q4k[start_block_idx + block_n],
                        dequantized_block, GGML_QK_K, false);

      size_t dest_offset = block_n * GGML_QK_K;

      size_t elements_to_copy = SAFE_MIN((size_t)GGML_QK_K, (size_t)(hs - dest_offset));
      std::memcpy(&embedding_vec[dest_offset], dequantized_block,
                  elements_to_copy * sizeof(float));
    }
    if (log_this_token_lookup) {
      log_vector_summary("[CPU_EMBED_DETAIL Q4_K] Output Embedding (Token " +
                             std::to_string(token_id) + ")",
                         embedding_vec);
    }
    return embedding_vec;
  }

  else if (!embed_tokens_q8_0.empty()) {
    if (hs % GGML_QK8_0 != 0) {
      Logger::error("Hidden size (" + std::to_string(hs) +
                    ") is not divisible by GGML_QK8_0 (" +
                    std::to_string(GGML_QK8_0) +
                    ") for Q8_0 embedding lookup.");
      return embedding_vec;
    }
    size_t blocks_per_row = hs / GGML_QK8_0;
    size_t start_block_idx = (size_t)token_id * blocks_per_row;
    size_t end_block_idx = start_block_idx + blocks_per_row;

    if (end_block_idx > embed_tokens_q8_0.size()) {
      Logger::error(
          "Calculated block index out of bounds for Q8_0 embedding table. "
          "Token: " +
          std::to_string(token_id) +
          ", StartBlock: " + std::to_string(start_block_idx) +
          ", EndBlock: " + std::to_string(end_block_idx) +
          ", TableSize: " + std::to_string(embed_tokens_q8_0.size()));
      return embedding_vec;
    }

    float dequantized_block[GGML_QK8_0];
    for (size_t block_n = 0; block_n < blocks_per_row; ++block_n) {
      dequantize_q8_0_block(&embed_tokens_q8_0[start_block_idx + block_n],
                            dequantized_block);
      size_t dest_offset = block_n * GGML_QK8_0;
      size_t elements_to_copy = SAFE_MIN(static_cast<size_t>(GGML_QK8_0), static_cast<size_t>(hs - dest_offset));
      std::memcpy(&embedding_vec[dest_offset], dequantized_block,
                  elements_to_copy * sizeof(float));
    }
    if (log_this_token_lookup) {
      log_vector_summary("[CPU_EMBED_DETAIL Q8_0] Output Embedding (Token " +
                             std::to_string(token_id) + ")",
                         embedding_vec);
    }
    return embedding_vec;
  }

  else if (!embed_tokens_q6k.empty()) {
    if (hs % GGML_QK_K != 0) {
      Logger::error("Hidden size (" + std::to_string(hs) +
                    ") is not divisible by GGML_QK_K (" +
                    std::to_string(GGML_QK_K) + ") for Q6_K embedding lookup.");
      return embedding_vec;
    }
    size_t blocks_per_row = hs / GGML_QK_K;
    size_t start_block_idx = (size_t)token_id * blocks_per_row;
    size_t end_block_idx = start_block_idx + blocks_per_row;

    if (end_block_idx > embed_tokens_q6k.size()) {
      Logger::error(
          "Calculated block index out of bounds for Q6_K embedding table. "
          "Token: " +
          std::to_string(token_id) +
          ", StartBlock: " + std::to_string(start_block_idx) +
          ", EndBlock: " + std::to_string(end_block_idx) +
          ", TableSize: " + std::to_string(embed_tokens_q6k.size()));
      return embedding_vec;
    }

    float dequantized_block[GGML_QK_K];
    for (size_t block_n = 0; block_n < blocks_per_row; ++block_n) {
      dequantize_q6_k(&embed_tokens_q6k[start_block_idx + block_n],
                        dequantized_block, GGML_QK_K);
      size_t dest_offset = block_n * GGML_QK_K;
      size_t elements_to_copy = SAFE_MIN(static_cast<size_t>(GGML_QK_K), static_cast<size_t>(hs - dest_offset));
      std::memcpy(&embedding_vec[dest_offset], dequantized_block,
                  elements_to_copy * sizeof(float));
    }
    if (log_this_token_lookup) {
      log_vector_summary("[CPU_EMBED_DETAIL Q6_K] Output Embedding (Token " +
                             std::to_string(token_id) + ")",
                         embedding_vec);
    }
    return embedding_vec;
  }

  else if (!embed_tokens_f32.empty()) {
    size_t offset = (size_t)token_id * hs;
    if (offset + hs > embed_tokens_f32.size()) {
      Logger::error("Embedding offset out of bounds in F32 lookup for token: " +
                    std::to_string(token_id));
      return embedding_vec;
    }

    std::copy(embed_tokens_f32.begin() + offset,
              embed_tokens_f32.begin() + offset + hs, embedding_vec.begin());
    if (log_this_token_lookup) {
      log_vector_summary("[CPU_EMBED_DETAIL F32] Output Embedding (Token " +
                             std::to_string(token_id) + ")",
                         embedding_vec);
    }
    return embedding_vec;

  } else if (!embed_tokens.empty()) {
    size_t offset = (size_t)token_id * hs;
    if (offset + hs > embed_tokens.size()) {
      Logger::error(
          "Embedding offset out of bounds in BF16 lookup for token: " +
          std::to_string(token_id));
      return embedding_vec;
    }
    std::vector<uint16_t> token_embedding_bf16(
        embed_tokens.begin() + offset, embed_tokens.begin() + offset + hs);

    embedding_vec = bf16vec_to_float_vec(token_embedding_bf16);
    if (log_this_token_lookup) {
      log_vector_summary("[CPU_EMBED_DETAIL BF16] Output Embedding (Token " +
                             std::to_string(token_id) + ")",
                         embedding_vec);
    }
    return embedding_vec;

  } else {
    Logger::error(
        "No valid embedding table found (Q4_K, F32, BF16) for token: " +
        std::to_string(token_id));

    return embedding_vec;
  }
}

std::vector<float> TinyLlamaModel::forward(
    std::vector<float>& input,
                                           int n_tokens, KVCache* kv_cache,
    const std::vector<int>* attention_mask) {
  Logger::info("[CPU_FWD] Entered. Processing up to layer " + std::to_string(config_.num_cpu_offload_layers -1) + ". Input n_tokens: " + std::to_string(n_tokens));

  int hs = config_.hidden_size;
  int vs = config_.vocab_size;
  int is = config_.intermediate_size;
  int n_heads = config_.num_attention_heads;
  int n_kv_heads = config_.num_key_value_heads;
  int head_dim = hs / n_heads;
  float eps = config_.rms_norm_eps;
  int max_pos_embeddings = config_.max_position_embeddings;

  bool log_first_gen_step = (n_tokens == 0);
  bool log_this_step = log_first_gen_step || (n_tokens == 12) || (n_tokens == 13);

  // Layer processing loop - ONLY for CPU-offloaded layers
  for (int l = 0; l < config_.num_cpu_offload_layers; ++l) {
    bool log_this_layer = log_this_step && (l == 0); // Log details only for layer 0 on specific steps
    if (log_this_layer) {
      Logger::info("[CPU_FWD] ------ START Layer " + std::to_string(l) +
                   " (pos=" + std::to_string(n_tokens) + ") ------");
      log_vector_summary("Layer " + std::to_string(l) + " Input (input)", input);
    }

    const auto& lw = layers[l];
    std::vector<float> x_norm_vec1(hs);
    const std::vector<float>& w_input_norm_vec =
        lw.input_layernorm_f32.empty()
            ? bf16vec_to_float_vec(lw.input_layernorm)
            : lw.input_layernorm_f32;
    rmsnorm_vector_cpu(input, w_input_norm_vec, x_norm_vec1, eps);
    
    std::vector<float> q_vec(hs), k_vec(n_kv_heads * head_dim), v_vec(n_kv_heads * head_dim);
    // Example: Q-projection (adapt for other projections and quantization types)
    if (!lw.q_proj_f32.empty()) matvec_f32_f32_vector_cpu(lw.q_proj_f32, x_norm_vec1, q_vec, hs, hs);
    else if (!lw.q_proj_q8_0.empty() && config_.is_gguf_file_loaded) matvec_q8_0_f32_vector_cpu(lw.q_proj_q8_0, x_norm_vec1, q_vec, hs, hs);
    else if (!lw.q_proj_q4k.empty() && config_.is_gguf_file_loaded) matvec_q4k_f32_vector_cpu(lw.q_proj_q4k, x_norm_vec1, q_vec, hs, hs);
    else if (!lw.q_proj_q6k.empty() && config_.is_gguf_file_loaded) matvec_q6k_f32_vector_cpu(lw.q_proj_q6k, x_norm_vec1, q_vec, hs, hs);
    else if (!lw.q_proj.empty()) matvec_bf16_f32_vector_cpu(lw.q_proj, x_norm_vec1, q_vec, hs, hs); // BF16 from SafeTensors
    else throw std::runtime_error("Layer " + std::to_string(l) + ": No Q proj weights (f32, q8, q4k, q6k, bf16) for CPU");
    
    // ... K, V projections ...
    if (!lw.k_proj_f32.empty()) matvec_f32_f32_vector_cpu(lw.k_proj_f32, x_norm_vec1, k_vec, n_kv_heads * head_dim, hs);
    else if (!lw.k_proj_q8_0.empty() && config_.is_gguf_file_loaded) matvec_q8_0_f32_vector_cpu(lw.k_proj_q8_0, x_norm_vec1, k_vec, n_kv_heads * head_dim, hs);
    else if (!lw.k_proj_q4k.empty() && config_.is_gguf_file_loaded) matvec_q4k_f32_vector_cpu(lw.k_proj_q4k, x_norm_vec1, k_vec, n_kv_heads * head_dim, hs);
    else if (!lw.k_proj_q6k.empty() && config_.is_gguf_file_loaded) matvec_q6k_f32_vector_cpu(lw.k_proj_q6k, x_norm_vec1, k_vec, n_kv_heads * head_dim, hs);
    else if (!lw.k_proj.empty()) matvec_bf16_f32_vector_cpu(lw.k_proj, x_norm_vec1, k_vec, n_kv_heads * head_dim, hs);
    else throw std::runtime_error("Layer " + std::to_string(l) + ": No K proj weights (f32, q8, q4k, q6k, bf16) for CPU");

    if (!lw.v_proj_f32.empty()) matvec_f32_f32_vector_cpu(lw.v_proj_f32, x_norm_vec1, v_vec, n_kv_heads * head_dim, hs);
    else if (!lw.v_proj_q8_0.empty() && config_.is_gguf_file_loaded) matvec_q8_0_f32_vector_cpu(lw.v_proj_q8_0, x_norm_vec1, v_vec, n_kv_heads * head_dim, hs);
    else if (!lw.v_proj_q4k.empty() && config_.is_gguf_file_loaded) matvec_q4k_f32_vector_cpu(lw.v_proj_q4k, x_norm_vec1, v_vec, n_kv_heads * head_dim, hs);
    else if (!lw.v_proj_q6k.empty() && config_.is_gguf_file_loaded) matvec_q6k_f32_vector_cpu(lw.v_proj_q6k, x_norm_vec1, v_vec, n_kv_heads * head_dim, hs);
    else if (!lw.v_proj.empty()) matvec_bf16_f32_vector_cpu(lw.v_proj, x_norm_vec1, v_vec, n_kv_heads * head_dim, hs);
    else throw std::runtime_error("Layer " + std::to_string(l) + ": No V proj weights (f32, q8, q4k, q6k, bf16) for CPU");

    apply_rope_vector(q_vec, n_heads, head_dim, n_tokens, precomputed_freqs_cis_, max_pos_embeddings, config_.is_gguf_file_loaded);
    apply_rope_vector(k_vec, n_kv_heads, head_dim, n_tokens, precomputed_freqs_cis_, max_pos_embeddings, config_.is_gguf_file_loaded);

    // KV Cache update
    if (kv_cache) {
        if (static_cast<size_t>(l) < kv_cache->layers.size()) {
            KVCacheLayer& kv_layer = kv_cache->layers[l];
            size_t layer_max_seq_len = 0;
            if (n_kv_heads > 0 && head_dim > 0) { // Prevent division by zero if params are bad
                layer_max_seq_len = kv_layer.k.size() / (n_kv_heads * head_dim);
            }
            
            if (static_cast<size_t>(n_tokens) >= layer_max_seq_len && layer_max_seq_len > 0) {
                Logger::error("KV Cache access out of bounds in CPU forward. Layer " + std::to_string(l) + 
                              ", n_tokens: " + std::to_string(n_tokens) + 
                              ", calculated layer_max_seq_len: " + std::to_string(layer_max_seq_len) + ". Skipping KV update.");
            } else if (layer_max_seq_len == 0 && n_tokens > 0) {
                 Logger::error("KV Cache layer_max_seq_len is 0, but n_tokens > 0. Layer " + std::to_string(l) + ". Skipping KV update.");
            } else {
                 for(int h=0; h < n_kv_heads; ++h) {
                     std::copy(k_vec.begin() + h * head_dim, k_vec.begin() + (h+1) * head_dim, kv_layer.k.begin() + n_tokens * (n_kv_heads * head_dim) + h * head_dim);
                     std::copy(v_vec.begin() + h * head_dim, v_vec.begin() + (h+1) * head_dim, kv_layer.v.begin() + n_tokens * (n_kv_heads * head_dim) + h * head_dim);
                 }
            }
        } else {
            Logger::error("KV Cache layer index " + std::to_string(l) + " out of bounds for kv_cache->layers.size() = " + std::to_string(kv_cache->layers.size()));
        }
    }
    
    std::vector<float> attn_out_vec(hs);
    std::vector<float> x_resid1_vec = input; // Store residual
    float att_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    std::fill(attn_out_vec.begin(), attn_out_vec.end(), 0.0f);
    for (int h = 0; h < n_heads; ++h) {
        std::vector<float> q_head(head_dim);
        std::copy(q_vec.begin() + h * head_dim, q_vec.begin() + (h + 1) * head_dim, q_head.begin());
        std::vector<float> current_multihead_attn_out(head_dim, 0.0f);
        int kv_cache_num_kv_heads = n_kv_heads; // from KVCache struct if available
        int kv_group = n_heads / kv_cache_num_kv_heads;
        int kv_head_idx = h / kv_group;

        if (kv_cache && static_cast<size_t>(l) < kv_cache->layers.size()) {
            const KVCacheLayer& kv_layer = kv_cache->layers[l];
            int current_seq_len = n_tokens + 1;
            std::vector<float> scores(current_seq_len);
            for (int t = 0; t < current_seq_len; ++t) {
                float score = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    score += q_head[d] * kv_layer.k[t * (n_kv_heads * head_dim) + kv_head_idx * head_dim + d];
                }
                scores[t] = score * att_scale;
            }
            softmax_vector_cpu(scores, scores); // In-place softmax
            for (int t = 0; t < current_seq_len; ++t) {
                for (int d = 0; d < head_dim; ++d) {
                    current_multihead_attn_out[d] += scores[t] * kv_layer.v[t * (n_kv_heads * head_dim) + kv_head_idx * head_dim + d];
                }
            }
        }
        std::copy(current_multihead_attn_out.begin(), current_multihead_attn_out.end(), attn_out_vec.begin() + h * head_dim);
    }
    

    std::vector<float> attn_proj_vec(hs);
    if(!lw.o_proj_f32.empty()) matvec_f32_f32_vector_cpu(lw.o_proj_f32, attn_out_vec, attn_proj_vec, hs, hs);
    else if (!lw.o_proj_q8_0.empty() && config_.is_gguf_file_loaded) matvec_q8_0_f32_vector_cpu(lw.o_proj_q8_0, attn_out_vec, attn_proj_vec, hs, hs);
    else if (!lw.o_proj_q4k.empty() && config_.is_gguf_file_loaded) matvec_q4k_f32_vector_cpu(lw.o_proj_q4k, attn_out_vec, attn_proj_vec, hs, hs);
    else if (!lw.o_proj_q6k.empty() && config_.is_gguf_file_loaded) matvec_q6k_f32_vector_cpu(lw.o_proj_q6k, attn_out_vec, attn_proj_vec, hs, hs);
    else if(!lw.o_proj.empty()) matvec_bf16_f32_vector_cpu(lw.o_proj, attn_out_vec, attn_proj_vec, hs, hs);
    else throw std::runtime_error("Layer " + std::to_string(l) + ": No O proj weights (f32, q8, q4k, q6k, bf16) for CPU");

    for(size_t i=0; i<input.size(); ++i) input[i] = x_resid1_vec[i] + attn_proj_vec[i]; // Update input by reference

    // MLP part
    std::vector<float> x_norm_vec2(hs);
    std::vector<float> x_resid2_vec = input; // Store residual for MLP
    const std::vector<float>& w_post_attn_norm_vec =
        lw.post_attention_layernorm_f32.empty()
            ? bf16vec_to_float_vec(lw.post_attention_layernorm)
            : lw.post_attention_layernorm_f32;
    rmsnorm_vector_cpu(input, w_post_attn_norm_vec, x_norm_vec2, eps);

    std::vector<float> gate_vec(is), up_vec(is);
    // Gate-projection
    if(!lw.gate_proj_f32.empty()) matvec_f32_f32_vector_cpu(lw.gate_proj_f32, x_norm_vec2, gate_vec, is, hs);
    else if (!lw.gate_proj_q8_0.empty() && config_.is_gguf_file_loaded) matvec_q8_0_f32_vector_cpu(lw.gate_proj_q8_0, x_norm_vec2, gate_vec, is, hs);
    else if (!lw.gate_proj_q4k.empty() && config_.is_gguf_file_loaded) matvec_q4k_f32_vector_cpu(lw.gate_proj_q4k, x_norm_vec2, gate_vec, is, hs);
    else if (!lw.gate_proj_q6k.empty() && config_.is_gguf_file_loaded) matvec_q6k_f32_vector_cpu(lw.gate_proj_q6k, x_norm_vec2, gate_vec, is, hs);
    else if(!lw.gate_proj.empty()) matvec_bf16_f32_vector_cpu(lw.gate_proj, x_norm_vec2, gate_vec, is, hs);
    else throw std::runtime_error("Layer " + std::to_string(l) + ": No Gate proj weights (f32, q8, q4k, q6k, bf16) for CPU");

    // Up-projection
    if(!lw.up_proj_f32.empty()) matvec_f32_f32_vector_cpu(lw.up_proj_f32, x_norm_vec2, up_vec, is, hs);
    else if (!lw.up_proj_q8_0.empty() && config_.is_gguf_file_loaded) matvec_q8_0_f32_vector_cpu(lw.up_proj_q8_0, x_norm_vec2, up_vec, is, hs);
    else if (!lw.up_proj_q4k.empty() && config_.is_gguf_file_loaded) matvec_q4k_f32_vector_cpu(lw.up_proj_q4k, x_norm_vec2, up_vec, is, hs);
    else if (!lw.up_proj_q6k.empty() && config_.is_gguf_file_loaded) matvec_q6k_f32_vector_cpu(lw.up_proj_q6k, x_norm_vec2, up_vec, is, hs);
    else if(!lw.up_proj.empty()) matvec_bf16_f32_vector_cpu(lw.up_proj, x_norm_vec2, up_vec, is, hs);
    else throw std::runtime_error("Layer " + std::to_string(l) + ": No Up proj weights (f32, q8, q4k, q6k, bf16) for CPU");

    std::vector<float> silu_out_vec(is);
    silu_cpu(gate_vec, silu_out_vec);

    std::vector<float> swiglu_result_vec(is);
    for(size_t i=0; i<is; ++i) swiglu_result_vec[i] = silu_out_vec[i] * up_vec[i];

    std::vector<float> mlp_out_vec(hs);
    // Down-projection
    if(!lw.down_proj_f32.empty()) matvec_f32_f32_vector_cpu(lw.down_proj_f32, swiglu_result_vec, mlp_out_vec, hs, is);
    else if (!lw.down_proj_q8_0.empty() && config_.is_gguf_file_loaded) matvec_q8_0_f32_vector_cpu(lw.down_proj_q8_0, swiglu_result_vec, mlp_out_vec, hs, is);
    else if (!lw.down_proj_q4k.empty() && config_.is_gguf_file_loaded) matvec_q4k_f32_vector_cpu(lw.down_proj_q4k, swiglu_result_vec, mlp_out_vec, hs, is);
    else if (!lw.down_proj_q6k.empty() && config_.is_gguf_file_loaded) matvec_q6k_f32_vector_cpu(lw.down_proj_q6k, swiglu_result_vec, mlp_out_vec, hs, is);
    else if(!lw.down_proj.empty()) matvec_bf16_f32_vector_cpu(lw.down_proj, swiglu_result_vec, mlp_out_vec, hs, is);
    else throw std::runtime_error("Layer " + std::to_string(l) + ": No Down proj weights (f32, q8, q4k, q6k, bf16) for CPU");

    for(size_t i=0; i<input.size(); ++i) input[i] = x_resid2_vec[i] + mlp_out_vec[i]; // Update input by reference
    

    if (log_this_layer) {
      Logger::info("[CPU_FWD] ------ END Layer " + std::to_string(l) +
                   " (pos=" + std::to_string(n_tokens) + ") ------");
    }
  }

  if (config_.num_cpu_offload_layers == config_.num_hidden_layers) {
    Logger::info("[CPU_FWD] All layers processed on CPU. Performing final RMSNorm and Logits.");
  const std::vector<float>& w_final_norm_vec =
      final_norm_f32.empty() ? bf16vec_to_float_vec(final_norm)
                             : final_norm_f32;
  std::vector<float> x_final_norm_vec(hs);
  rmsnorm_vector_cpu(input, w_final_norm_vec, x_final_norm_vec, eps);

  std::vector<float> logits(vs);
    if (!lm_head_f32.empty()) matvec_f32_f32_vector_cpu(lm_head_f32, x_final_norm_vec, logits, vs, hs);
    else if (!lm_head_q8_0.empty() && config_.is_gguf_file_loaded) matvec_q8_0_f32_vector_cpu(lm_head_q8_0, x_final_norm_vec, logits, vs, hs);
    else if (!lm_head_q4k.empty() && config_.is_gguf_file_loaded) matvec_q4k_f32_vector_cpu(lm_head_q4k, x_final_norm_vec, logits, vs, hs);
    else if (!lm_head_q6k.empty() && config_.is_gguf_file_loaded) matvec_q6k_f32_vector_cpu(lm_head_q6k, x_final_norm_vec, logits, vs, hs);
    else if (!lm_head.empty()) matvec_bf16_f32_vector_cpu(lm_head, x_final_norm_vec, logits, vs, hs); // Fallback for BF16 SafeTensors
    else throw std::runtime_error("No valid LM Head weights (f32, q8, q4k, q6k, bf16) found for CPU final stage.");

  if (log_this_step || log_first_gen_step) {
        log_vector_summary("[CPU_FWD] Final Logits (all CPU, pos=" + std::to_string(n_tokens) + ")", logits, 15);
    }
    return logits; // Return final logits if all layers were CPU
  }

  Logger::info("[CPU_FWD] Finished processing " + std::to_string(config_.num_cpu_offload_layers) + " CPU layers. Output is intermediate activation.");
  return input; // Return the intermediate activations if not all layers were processed here.
}

int TinyLlamaModel::get_vocab_size() const { return config_.vocab_size; }

#ifdef HAS_CUDA
std::vector<float> TinyLlamaModel::forward_device(
    float* x_input_dev,
    int pos, KVCache* kv_cache,
    const std::vector<int>* attention_mask, cudaStream_t stream) {
  
  int hs = config_.hidden_size;
  int vs = config_.vocab_size;
  int n_heads = config_.num_attention_heads;
  int n_kv_heads = config_.num_key_value_heads;
  if (n_heads == 0) {
    Logger::fatal("Number of attention heads is zero during forward_device.");
    throw std::runtime_error("Division by zero: n_heads is zero.");
  }
  int head_dim = hs / n_heads;
  int total_model_layers = config_.num_hidden_layers;
  int num_cpu_layers = config_.num_cpu_offload_layers;
  int num_gpu_layers = total_model_layers - num_cpu_layers;

  if (num_gpu_layers <= 0) {
      Logger::warning("forward_device called with no GPU layers to process (num_gpu_layers = " + std::to_string(num_gpu_layers) + "). Returning empty.");
      return {};
  }
  if (!x_input_dev) {
      Logger::error("forward_device called with null x_input_dev. This should be model_->x_dev_.");
      return {};
  }
  if (!kv_cache) {
      Logger::error("forward_device called with null KVCache.");
      return {};
  }

  int is = config_.intermediate_size;
  float eps = config_.rms_norm_eps;
  bool log_this_pos = (pos == 0 || pos == 1 || pos == 13);

  if (log_this_pos) {
    Logger::info("[TM::fw_dev pos=" + std::to_string(pos) +
                 "] Entered. Processing " + std::to_string(num_gpu_layers) + " GPU layers, starting from model layer " + 
                 std::to_string(num_cpu_layers) + ". Input is x_input_dev (model_->x_dev_).");
  }

  cublasStatus_t stream_status = cublasSetStream(cublas_handle_, stream);
  if (stream_status != CUBLAS_STATUS_SUCCESS) {
    Logger::error("cublasSetStream failed in forward_device");
    return {};
  }

  float* current_x_dev = x_input_dev; // This is effectively model_->x_dev_

  for (int l_gpu_idx = 0; l_gpu_idx < num_gpu_layers; ++l_gpu_idx) {
    int l_model_idx = num_cpu_layers + l_gpu_idx; // Actual layer index in the model

    if (log_this_pos) {
      Logger::info("[TM::fw_dev pos=" + std::to_string(pos) + "] GPU Layer Loop: gpu_idx=" + std::to_string(l_gpu_idx) +
                   ", model_idx=" + std::to_string(l_model_idx) + ". Operating on model_->x_dev_.");
    }

    // Get layer sizes (kv_dim calculation was in initialize_gpu_and_rope, ensure it's accessible or recalculated)
    int kv_dim = (config_.hidden_size / config_.num_attention_heads) * config_.num_key_value_heads;
    size_t layer_q_size = (size_t)hs * hs;
    size_t layer_k_size = (size_t)kv_dim * hs; 
    size_t layer_v_size = (size_t)kv_dim * hs;
    size_t layer_o_size = (size_t)hs * hs;
    size_t layer_gate_size = (size_t)is * hs;
    size_t layer_up_size = (size_t)is * hs;
    size_t layer_down_size = (size_t)hs * is;

    // Concatenated weights are indexed by l_gpu_idx because they only contain GPU layer weights
    const uint16_t* lw_q_proj_bf16_dev = w_q_dev_ ? w_q_dev_ + (size_t)l_gpu_idx * layer_q_size : nullptr;
    const uint16_t* lw_k_proj_bf16_dev = w_k_dev_ ? w_k_dev_ + (size_t)l_gpu_idx * layer_k_size : nullptr;
    const uint16_t* lw_v_proj_bf16_dev = w_v_dev_ ? w_v_dev_ + (size_t)l_gpu_idx * layer_v_size : nullptr;
    const uint16_t* lw_o_proj_bf16_dev = w_o_dev_ ? w_o_dev_ + (size_t)l_gpu_idx * layer_o_size : nullptr;
    const uint16_t* lw_gate_proj_bf16_dev = w_gate_dev_ ? w_gate_dev_ + (size_t)l_gpu_idx * layer_gate_size : nullptr;
    const uint16_t* lw_up_proj_bf16_dev = w_up_dev_ ? w_up_dev_ + (size_t)l_gpu_idx * layer_up_size : nullptr;
    const uint16_t* lw_down_proj_bf16_dev = w_down_dev_ ? w_down_dev_ + (size_t)l_gpu_idx * layer_down_size : nullptr;

    const float* lw_q_proj_f32_dev = w_q_f32_dev_ ? w_q_f32_dev_ + (size_t)l_gpu_idx * layer_q_size : nullptr;
    const float* lw_k_proj_f32_dev = w_k_f32_dev_ ? w_k_f32_dev_ + (size_t)l_gpu_idx * layer_k_size : nullptr;
    const float* lw_v_proj_f32_dev = w_v_f32_dev_ ? w_v_f32_dev_ + (size_t)l_gpu_idx * layer_v_size : nullptr;
    const float* lw_o_proj_f32_dev = w_o_f32_dev_ ? w_o_f32_dev_ + (size_t)l_gpu_idx * layer_o_size : nullptr;
    const float* lw_gate_proj_f32_dev = w_gate_f32_dev_ ? w_gate_f32_dev_ + (size_t)l_gpu_idx * layer_gate_size : nullptr;
    const float* lw_up_proj_f32_dev = w_up_f32_dev_ ? w_up_f32_dev_ + (size_t)l_gpu_idx * layer_up_size : nullptr;
    const float* lw_down_proj_f32_dev = w_down_f32_dev_ ? w_down_f32_dev_ + (size_t)l_gpu_idx * layer_down_size : nullptr;
    
    // Layer-specific norm weights are indexed by the model layer index (l_model_idx)
    const float* lw_in_norm_dev = layers[l_model_idx].input_layernorm_dev;
    const float* lw_post_norm_dev = layers[l_model_idx].post_attention_layernorm_dev;

    gpuErrchk(cudaMemcpyAsync(x_resid1_dev_, x_dev_, hs * sizeof(float),
                              cudaMemcpyDeviceToDevice, stream));

    if (!lw_in_norm_dev) { 
        throw std::runtime_error("[TM::fw_dev pos=" + std::to_string(pos) + " L" + std::to_string(l_model_idx) + "] Error: input_layernorm_dev is nullptr. GPU layer cannot proceed.");
    }
    rmsnorm_vector_cuda(x_dev_, lw_in_norm_dev, x_norm_dev_, hs,
                        eps, stream);

    if (lw_q_proj_f32_dev && lw_k_proj_f32_dev && lw_v_proj_f32_dev) {
      matvec_f32_f32_cuda(cublas_handle_, lw_q_proj_f32_dev, x_norm_dev_,
                          q_dev_, hs, hs, stream);
      matvec_f32_f32_cuda(cublas_handle_, lw_k_proj_f32_dev, x_norm_dev_,
                          k_dev_, n_kv_heads * head_dim, hs, stream);
      matvec_f32_f32_cuda(cublas_handle_, lw_v_proj_f32_dev, x_norm_dev_,
                          v_dev_, n_kv_heads * head_dim, hs, stream);
    } else if (lw_q_proj_bf16_dev && lw_k_proj_bf16_dev && lw_v_proj_bf16_dev) {
      Logger::warning("GPU L" + std::to_string(l_model_idx) + " (gpu_idx " + std::to_string(l_gpu_idx) + "): Using BF16 matvec for QKV.");
      matvec_bf16_f32_cuda(cublas_handle_, lw_q_proj_bf16_dev, x_norm_dev_,
                           q_dev_, hs, hs, stream);
      matvec_bf16_f32_cuda(cublas_handle_, lw_k_proj_bf16_dev, x_norm_dev_,
                           k_dev_, n_kv_heads * head_dim, hs, stream);
      matvec_bf16_f32_cuda(cublas_handle_, lw_v_proj_bf16_dev, x_norm_dev_,
                           v_dev_, n_kv_heads * head_dim, hs, stream);
    } else {
      Logger::error("GPU L" + std::to_string(l_model_idx) + " (gpu_idx " + std::to_string(l_gpu_idx) + "): No valid QKV proj weights (FP32/BF16)."); return {};
    }

    if (log_this_pos) {
      std::vector<float> temp_q_host(hs);
      gpuErrchk(cudaMemcpy(temp_q_host.data(), q_dev_, hs * sizeof(float),
                           cudaMemcpyDeviceToHost));
      log_vector_summary_detailed("[TM::fw_dev pos=" + std::to_string(pos) +
                                      " L" + std::to_string(l_model_idx) +
                                      "] q_dev_ after QKV Proj",
                                  temp_q_host, pos, l_model_idx, 8);
    }
    if (log_this_pos) {
      std::vector<float> temp_q_host_rope(hs);
      gpuErrchk(cudaMemcpy(temp_q_host_rope.data(), q_dev_, hs * sizeof(float),
                           cudaMemcpyDeviceToHost));
      log_vector_summary_detailed("[TM::fw_dev pos=" + std::to_string(pos) +
                                      " L" + std::to_string(l_model_idx) +
                                      "] q_dev_ after RoPE",
                                  temp_q_host_rope, pos, l_model_idx, 8);
    }

    // RoPE Application:
    rope_cuda(q_dev_, n_heads, head_dim, all_freqs_cis_dev, pos, config_.is_gguf_file_loaded, stream);
    rope_cuda(k_dev_, n_kv_heads, head_dim, all_freqs_cis_dev, pos, config_.is_gguf_file_loaded, stream);

    // K/V Cache Update Logic
    if (static_cast<size_t>(l_model_idx) < kv_cache->layers.size()) {
        KVCacheLayer& current_kv_layer = kv_cache->layers[l_model_idx];
        if (config_.use_kvcache_quantization) {
            // Quantize and store K, V for each head
            for (int kvh = 0; kvh < n_kv_heads; ++kvh) {
                const float* current_k_head_ptr_fp32 = k_dev_ + kvh * head_dim;
                const float* current_v_head_ptr_fp32 = v_dev_ + kvh * head_dim;

                size_t token_head_offset_quant = (static_cast<size_t>(pos) * n_kv_heads + kvh) * head_dim;
                int8_t* k_quant_target_ptr = current_kv_layer.k_dev_quantized + token_head_offset_quant;
                int8_t* v_quant_target_ptr = current_kv_layer.v_dev_quantized + token_head_offset_quant;

                size_t scale_offset = static_cast<size_t>(pos) * n_kv_heads + kvh;
                float* k_scale_target_ptr = current_kv_layer.k_dev_scales + scale_offset;
                float* v_scale_target_ptr = current_kv_layer.v_dev_scales + scale_offset;

                quantize_fp32_to_int8_symmetric_per_tensor_cuda(
                    current_k_head_ptr_fp32, k_quant_target_ptr, k_scale_target_ptr, head_dim, stream);
                quantize_fp32_to_int8_symmetric_per_tensor_cuda(
                    current_v_head_ptr_fp32, v_quant_target_ptr, v_scale_target_ptr, head_dim, stream);
            }
        } else {
            // Store FP32 K, V directly (original path, now using k_dev_fp32, v_dev_fp32)
            for (int kvh = 0; kvh < n_kv_heads; ++kvh) {
                const float* current_k_head_ptr = k_dev_ + kvh * head_dim;
                const float* current_v_head_ptr = v_dev_ + kvh * head_dim;

                // Assuming update_kv_cache_cuda is adapted or overloaded for fp32 target pointers
                update_kv_cache_cuda(current_kv_layer.k_dev_fp32, current_k_head_ptr, pos,
                                   kvh, kv_cache->allocated_max_seq_len,
                                   kv_cache->allocated_num_kv_heads, // This should be n_kv_heads for indexing
                                   kv_cache->allocated_head_dim, stream); 

                update_kv_cache_cuda(current_kv_layer.v_dev_fp32, current_v_head_ptr, pos,
                                   kvh, kv_cache->allocated_max_seq_len,
                                   kv_cache->allocated_num_kv_heads, // This should be n_kv_heads for indexing
                                   kv_cache->allocated_head_dim, stream);
            }
        }
    } else {
        Logger::error("KVCache layer index " + std::to_string(l_model_idx) + " out of bounds for kv_cache->layers access in forward_device.");
        return {}; // Or throw
    }

    float scale = 1.0f / SAFE_SQRT(static_cast<float>(head_dim));
    
    // Prepare K/V Pointers for Attention Kernel
    const float* attention_k_cache_ptr_dev = nullptr;
    const float* attention_v_cache_ptr_dev = nullptr;
    KVCacheLayer& attention_kv_layer = kv_cache->layers[l_model_idx]; // Re-fetch for clarity, same as current_kv_layer

    if (config_.use_kvcache_quantization) {
        for (int t = 0; t <= pos; ++t) { // Iterate through sequence length up to current token
            for (int kvh = 0; kvh < n_kv_heads; ++kvh) { // Iterate through heads
                size_t token_head_offset_quant = (static_cast<size_t>(t) * n_kv_heads + kvh) * head_dim;
                const int8_t* k_quant_source_ptr = attention_kv_layer.k_dev_quantized + token_head_offset_quant;
                const int8_t* v_quant_source_ptr = attention_kv_layer.v_dev_quantized + token_head_offset_quant;

                size_t scale_offset = static_cast<size_t>(t) * n_kv_heads + kvh;
                const float* k_scale_source_ptr = attention_kv_layer.k_dev_scales + scale_offset;
                const float* v_scale_source_ptr = attention_kv_layer.v_dev_scales + scale_offset;

                // Output to the corresponding position in the temporary dequantized buffer
                float* k_dequant_target_ptr = dequant_k_cache_buffer_dev_ + token_head_offset_quant;
                float* v_dequant_target_ptr = dequant_v_cache_buffer_dev_ + token_head_offset_quant;

                dequantize_int8_to_fp32_symmetric_per_tensor_cuda(
                    k_quant_source_ptr, k_scale_source_ptr, k_dequant_target_ptr, head_dim, stream);
                dequantize_int8_to_fp32_symmetric_per_tensor_cuda(
                    v_quant_source_ptr, v_scale_source_ptr, v_dequant_target_ptr, head_dim, stream);
            }
        }
        attention_k_cache_ptr_dev = dequant_k_cache_buffer_dev_;
        attention_v_cache_ptr_dev = dequant_v_cache_buffer_dev_;
    } else {
        attention_k_cache_ptr_dev = attention_kv_layer.k_dev_fp32;
        attention_v_cache_ptr_dev = attention_kv_layer.v_dev_fp32;
    }

    float current_attention_scale = 1.0f / sqrtf((float)head_dim);
    attention_cuda(
        q_dev_,                             // Param 1: Q_current_dev (const float*)
        attention_k_cache_ptr_dev,        // Param 2: K_layer_cache_base (const float*)
        attention_v_cache_ptr_dev,        // Param 3: V_layer_cache_base (const float*)
        attn_out_dev_,                     // Param 4: out_dev (float*)
        config_.num_attention_heads,      // Param 5: num_heads (int)
        pos + 1,                          // Param 6: current_seq_len (int)
        head_dim,                         // Param 7: head_dim (int)
        current_attention_scale,          // Param 8: scale (float)
        kv_cache->allocated_max_seq_len,  // Param 9: cache_max_seq_len (int)
        config_.num_key_value_heads,      // Param 10: cache_num_kv_heads (int)
        stream                            // Param 11: stream (cudaStream_t)
    );

    if (log_this_pos) {
      std::vector<float> temp_attn_out_host(hs);
      gpuErrchk(cudaMemcpy(temp_attn_out_host.data(), attn_out_dev_,
                           hs * sizeof(float), cudaMemcpyDeviceToHost));
      log_vector_summary_detailed("[TM::fw_dev pos=" + std::to_string(pos) +
                                      " L" + std::to_string(l_model_idx) +
                                      "] attn_out_dev_ after Attention",
                                  temp_attn_out_host, pos, l_model_idx, 8);
    }

    if (lw_o_proj_f32_dev) {
      matvec_f32_f32_cuda(cublas_handle_, lw_o_proj_f32_dev, attn_out_dev_, attn_proj_dev_, hs, hs, stream);
    } else if (lw_o_proj_bf16_dev) {
      Logger::warning("GPU L" + std::to_string(l_model_idx) + " (gpu_idx " + std::to_string(l_gpu_idx) + "): Using BF16 matvec for O-Proj.");
      matvec_bf16_f32_cuda(cublas_handle_, lw_o_proj_bf16_dev, attn_out_dev_, attn_proj_dev_, hs, hs, stream);
    } else {
      Logger::error("GPU L" + std::to_string(l_model_idx) + " (gpu_idx " + std::to_string(l_gpu_idx) + "): No valid O proj weights (FP32/BF16)."); return {};
    }
    if (log_this_pos) {
      std::vector<float> temp_attn_proj_host(hs);
      gpuErrchk(cudaMemcpy(temp_attn_proj_host.data(), attn_proj_dev_,
                           hs * sizeof(float), cudaMemcpyDeviceToHost));
      log_vector_summary_detailed("[TM::fw_dev pos=" + std::to_string(pos) +
                                      " L" + std::to_string(l_model_idx) +
                                      "] attn_proj_dev_ after O-Proj",
                                  temp_attn_proj_host, pos, l_model_idx, 8);
    }

    add_residual_cuda(attn_proj_dev_, x_resid1_dev_, current_x_dev, hs, stream); // Output to current_x_dev (model_->x_dev_)

    gpuErrchk(cudaMemcpyAsync(x_resid2_dev_, current_x_dev, hs * sizeof(float), cudaMemcpyDeviceToDevice, stream)); // Save input to MLP

    if (!lw_post_norm_dev) { Logger::error("Missing post_attention_layernorm_dev for GPU layer model_idx=" + std::to_string(l_model_idx)); return {}; }
    rmsnorm_vector_cuda(current_x_dev, lw_post_norm_dev, x_norm_dev_, hs, eps, stream); // Input current_x_dev, output to x_norm_dev_
    if (log_this_pos) {
      std::vector<float> temp_x_host(hs);
      gpuErrchk(cudaMemcpy(temp_x_host.data(), x_norm_dev_, hs * sizeof(float),
                           cudaMemcpyDeviceToHost));
      log_vector_summary_detailed("[TM::fw_dev pos=" + std::to_string(pos) +
                                      " L" + std::to_string(l_model_idx) +
                                      "] x_norm_dev_ after Input RMSNorm",
                                  temp_x_host, pos, l_model_idx, 8);
    }

    if (lw_gate_proj_f32_dev && lw_up_proj_f32_dev) {
      matvec_f32_f32_cuda(cublas_handle_, lw_gate_proj_f32_dev, x_norm_dev_,
                          gate_vec_dev_, is, hs, stream);
      matvec_f32_f32_cuda(cublas_handle_, lw_up_proj_f32_dev, x_norm_dev_,
                          up_vec_dev_, is, hs, stream);
    } else if (lw_gate_proj_bf16_dev && lw_up_proj_bf16_dev) {
      Logger::warning("GPU L" + std::to_string(l_model_idx) + " (gpu_idx " + std::to_string(l_gpu_idx) + "): Using BF16 matvec for Gate/Up Proj.");
      matvec_bf16_f32_cuda(cublas_handle_, lw_gate_proj_bf16_dev, x_norm_dev_,
                           gate_vec_dev_, is, hs, stream);
      matvec_bf16_f32_cuda(cublas_handle_, lw_up_proj_bf16_dev, x_norm_dev_,
                           up_vec_dev_, is, hs, stream);
    } else {
      Logger::error("GPU L" + std::to_string(l_model_idx) + " (gpu_idx " + std::to_string(l_gpu_idx) + "): No valid Gate/Up projection weights found on GPU (FP32 or BF16).");
      return {};
    }
    if (log_this_pos) {
      std::vector<float> temp_gate_host(is);
      gpuErrchk(cudaMemcpy(temp_gate_host.data(), gate_vec_dev_,
                           is * sizeof(float), cudaMemcpyDeviceToHost));
      log_vector_summary_detailed("[TM::fw_dev pos=" + std::to_string(pos) +
                                      " L" + std::to_string(l_model_idx) +
                                      "] gate_vec_dev_ after Proj",
                                  temp_gate_host, pos, l_model_idx, 8);
    }

    swiglu_cuda(gate_vec_dev_, up_vec_dev_, swiglu_vec_dev_, is, stream);
    if (log_this_pos) {
      std::vector<float> temp_swiglu_host(is);
      gpuErrchk(cudaMemcpy(temp_swiglu_host.data(), swiglu_vec_dev_,
                           is * sizeof(float), cudaMemcpyDeviceToHost));
      log_vector_summary_detailed("[TM::fw_dev pos=" + std::to_string(pos) +
                                      " L" + std::to_string(l_model_idx) +
                                      "] swiglu_vec_dev_ after SwiGLU",
                                  temp_swiglu_host, pos, l_model_idx, 8);
    }

    if (lw_down_proj_f32_dev) {
      matvec_f32_f32_cuda(cublas_handle_, lw_down_proj_f32_dev, swiglu_vec_dev_,
                          mlp_down_dev_, hs, is, stream);
    } else if (lw_down_proj_bf16_dev) {
      Logger::warning("GPU L" + std::to_string(l_model_idx) + " (gpu_idx " + std::to_string(l_gpu_idx) + "): Using BF16 matvec for Down Proj.");
      matvec_bf16_f32_cuda(cublas_handle_, lw_down_proj_bf16_dev,
                           swiglu_vec_dev_, mlp_down_dev_, hs, is, stream);
    } else {
      Logger::error("GPU L" + std::to_string(l_model_idx) + " (gpu_idx " + std::to_string(l_gpu_idx) + "): No valid Down projection weights found on GPU (FP32 or BF16).");
      return {};
    }

    if (log_this_pos) {
      std::vector<float> temp_mlp_down_host(hs);
      gpuErrchk(cudaMemcpy(temp_mlp_down_host.data(), mlp_down_dev_,
                           hs * sizeof(float), cudaMemcpyDeviceToHost));
      log_vector_summary_detailed("[TM::fw_dev pos=" + std::to_string(pos) +
                                      " L" + std::to_string(l_model_idx) + // Corrected from l to l_model_idx
                                      "] mlp_down_dev_ after Down Proj",
                                  temp_mlp_down_host, pos, l_model_idx, 8); // Corrected from l to l_model_idx
    }

    add_residual_cuda(mlp_down_dev_, x_resid2_dev_, current_x_dev, hs, stream); // Output to current_x_dev (model_->x_dev_)

    if (log_this_pos && (l_model_idx == num_cpu_layers || l_model_idx == (total_model_layers - 1))) { 
      std::vector<float> x_host_output(hs);
      gpuErrchk(cudaMemcpy(x_host_output.data(), current_x_dev, hs * sizeof(float), cudaMemcpyDeviceToHost));
      log_vector_summary_detailed("[CUDA] Output of Model Layer " + std::to_string(l_model_idx) + " (GPU_idx " + std::to_string(l_gpu_idx) + ", pos=" + std::to_string(pos) + ")", x_host_output, pos, l_model_idx, 8); // Corrected from l to l_model_idx
    }
  }

  if (log_this_pos)
    Logger::info("[TM::fw_dev pos=" + std::to_string(pos) +
                 "] Processing final RMSNorm.");

  rmsnorm_vector_cuda(x_dev_, final_norm_dev, x_norm_dev_, hs, eps, stream);
  if (log_this_pos)
    Logger::info("[TM::fw_dev pos=" + std::to_string(pos) +
                 "] Processing LM Head.");

  if (lm_head_dev_) { 
    matvec_bf16_f32_cuda(cublas_handle_, lm_head_dev_, x_norm_dev_, logits_dev_,
                         vs, hs, stream);
  } else {
    Logger::error("LM head (lm_head_dev_ for BF16) is null. Cannot calculate logits on GPU.");
    return {};
  }

  gpuErrchk(cudaStreamSynchronize(stream));

  std::vector<float> logits(vs);
  gpuErrchk(cudaMemcpy(logits.data(), logits_dev_, vs * sizeof(float),
                       cudaMemcpyDeviceToHost));
  if (log_this_pos)
    Logger::info("[TM::fw_dev pos=" + std::to_string(pos) + "] Exiting.");
  return logits;
}

#endif // HAS_CUDA

void map_gguf_weights(const GGUFData& gguf, TinyLlamaModel& model) {
  Logger::info("Mapping GGUF weights to model fields...");
  if (gguf.mapped_tensor_data == nullptr || gguf.mapped_tensor_data_size == 0) {
    Logger::warning("GGUF mapped_tensor_data is null or size is 0. Cannot map weights.");
    return;
  }
  // This is the start of the mmapped region in memory.
  const uint8_t* mmap_buffer_start = static_cast<const uint8_t*>(gguf.mapped_tensor_data);
  // This is the start of the *actual* GGUF tensor data within that mmapped region,
  // accounting for any alignment padding mmap had to do.
  const uint8_t* actual_data_block_start = mmap_buffer_start + gguf.offset_diff_for_mmap;
  // The end of the valid *actual* data block within the mmap.
  const uint8_t* actual_data_block_end = actual_data_block_start + (gguf.mapped_tensor_data_size - gguf.offset_diff_for_mmap);

  Logger::info("map_gguf_weights: Total mmapped region size: " +
               std::to_string(gguf.mapped_tensor_data_size) + " bytes. " +
               "Offset diff for mmap: " + std::to_string(gguf.offset_diff_for_mmap));

  for (const auto& pair : gguf.tensor_infos_map) {
    std::stringstream ss_map;
    const std::string& target_field = pair.first;
    const GGUFTensorInfo& info = pair.second;
    // info.offset is relative to the start of the *actual* GGUF tensor data block.
    const uint8_t* tensor_data_ptr = actual_data_block_start + info.offset;
    const uint8_t* tensor_data_end = tensor_data_ptr + info.size_in_bytes;

    ss_map << "Attempting to map tensor: '" << info.name
           << "', Type: " << info.type << ", GGUFOffset: " << info.offset
           << ", NumElem: " << info.num_elements
           << ", SizeBytes: " << info.size_in_bytes
           << ", SrcAddrInMmap: " << static_cast<const void*>(tensor_data_ptr)
           << ", ReadEndAddrInMmap: " << static_cast<const void*>(tensor_data_end)
           << ", MmapRegion: [" << static_cast<const void*>(mmap_buffer_start)
           << " - " << static_cast<const void*>(mmap_buffer_start + gguf.mapped_tensor_data_size) << "]"
           << ", ActualDataBlock: [" << static_cast<const void*>(actual_data_block_start)
           << " - " << static_cast<const void*>(actual_data_block_end) << "]";

    if (tensor_data_ptr < actual_data_block_start || 
        tensor_data_end > actual_data_block_end) {
      ss_map << ", InBounds: NO";
      Logger::error(ss_map.str());
      Logger::error("Tensor data out of mmapped actual data block bounds for: " + info.name);
      continue;
    } else {
      ss_map << ", InBounds: YES";
      Logger::info(ss_map.str());
    }
    if (info.type == GGMLType::GGML_TYPE_F32) {
      size_t num_elements = info.size_in_bytes / sizeof(float);
      std::vector<float> dest_f32(num_elements);
      std::memcpy(dest_f32.data(), tensor_data_ptr, info.size_in_bytes);

      if (target_field == "token_embd.weight")
        model.embed_tokens_f32 = std::move(dest_f32);
      else if (target_field == "output.weight")
        model.lm_head_f32 = std::move(dest_f32);
      else if (target_field == "output_norm.weight")
        model.final_norm_f32 = std::move(dest_f32);
      else if (target_field.find("blk.") == 0) {
        size_t start = 4;
        size_t end = target_field.find('.', start);
        int layer_idx = std::stoi(target_field.substr(start, end - start));
        std::string sub_field = target_field.substr(end + 1);
        if (layer_idx >= 0 && layer_idx < model.layers.size()) {
          if (sub_field == "attn_q.weight")
            model.layers[layer_idx].q_proj_f32 = std::move(dest_f32);
          else if (sub_field == "attn_k.weight")
            model.layers[layer_idx].k_proj_f32 = std::move(dest_f32);
          else if (sub_field == "attn_v.weight")
            model.layers[layer_idx].v_proj_f32 = std::move(dest_f32);
          else if (sub_field == "attn_output.weight")
            model.layers[layer_idx].o_proj_f32 = std::move(dest_f32);
          else if (sub_field == "ffn_gate.weight")
            model.layers[layer_idx].gate_proj_f32 = std::move(dest_f32);
          else if (sub_field == "ffn_up.weight")
            model.layers[layer_idx].up_proj_f32 = std::move(dest_f32);
          else if (sub_field == "ffn_down.weight")
            model.layers[layer_idx].down_proj_f32 = std::move(dest_f32);
          else if (sub_field == "attn_norm.weight")
            model.layers[layer_idx].input_layernorm_f32 = std::move(dest_f32);
          else if (sub_field == "ffn_norm.weight")
            model.layers[layer_idx].post_attention_layernorm_f32 =
                std::move(dest_f32);
          else {
            Logger::warning("Unsupported layer sub-field (FP32): " + sub_field);
            continue;
          }
          Logger::info("Mapped GGUF tensor '" + info.name +
                       "' (FP32) to layer " + std::to_string(layer_idx) +
                       " field " + sub_field);
        } else {
          Logger::warning("Invalid layer index (FP32) " +
                          std::to_string(layer_idx));
          continue;
        }
      } else {
        Logger::warning("Unhandled target field (FP32): " + target_field);
      }
    } else if (info.type == GGMLType::GGML_TYPE_F16) {
      size_t num_elements = info.size_in_bytes / sizeof(uint16_t);
      std::vector<float> dest_f32(num_elements);
      const uint16_t* src_f16 =
          reinterpret_cast<const uint16_t*>(tensor_data_ptr);
      for (size_t i = 0; i < num_elements; ++i) {
        dest_f32[i] = fp16_to_fp32(src_f16[i]);
      }

      if (target_field == "token_embd.weight")
        model.embed_tokens_f32 = std::move(dest_f32);
      else if (target_field == "output.weight")
        model.lm_head_f32 = std::move(dest_f32);
      else if (target_field == "output_norm.weight")
        model.final_norm_f32 = std::move(dest_f32);
      else if (target_field.find("blk.") == 0) {
        size_t start = 4;
        size_t end = target_field.find('.', start);
        int layer_idx = std::stoi(target_field.substr(start, end - start));
        std::string sub_field = target_field.substr(end + 1);
        if (layer_idx >= 0 && layer_idx < model.layers.size()) {
          if (sub_field == "attn_q.weight")
            model.layers[layer_idx].q_proj_f32 = std::move(dest_f32);
          else if (sub_field == "attn_k.weight")
            model.layers[layer_idx].k_proj_f32 = std::move(dest_f32);
          else if (sub_field == "attn_v.weight")
            model.layers[layer_idx].v_proj_f32 = std::move(dest_f32);
          else if (sub_field == "attn_output.weight")
            model.layers[layer_idx].o_proj_f32 = std::move(dest_f32);
          else if (sub_field == "ffn_gate.weight")
            model.layers[layer_idx].gate_proj_f32 = std::move(dest_f32);
          else if (sub_field == "ffn_up.weight")
            model.layers[layer_idx].up_proj_f32 = std::move(dest_f32);
          else if (sub_field == "ffn_down.weight")
            model.layers[layer_idx].down_proj_f32 = std::move(dest_f32);

          else if (sub_field == "attn_norm.weight")
            model.layers[layer_idx].input_layernorm_f32 = std::move(dest_f32);
          else if (sub_field == "ffn_norm.weight")
            model.layers[layer_idx].post_attention_layernorm_f32 =
                std::move(dest_f32);
          else {
            Logger::warning("Unsupported layer sub-field (FP16): " + sub_field);
            continue;
          }
          Logger::info("Mapped GGUF tensor '" + info.name +
                       "' (FP16) to layer " + std::to_string(layer_idx) +
                       " field " + sub_field);
        } else {
          Logger::warning("Invalid layer index (FP16) " +
                          std::to_string(layer_idx));
          continue;
        }
      } else {
        Logger::warning("Unhandled target field (FP16): " + target_field);
      }
    } else if (info.type == GGMLType::GGML_TYPE_BF16) {
      size_t num_elements = info.size_in_bytes / sizeof(uint16_t);
      std::vector<float> dest_f32(num_elements);
      const uint16_t* src_bf16 =
          reinterpret_cast<const uint16_t*>(tensor_data_ptr);

#pragma omp parallel for
      for (int64_t i = 0; i < static_cast<int64_t>(num_elements); ++i) {
        dest_f32[i] = bfloat16_to_float32(src_bf16[i]);
      }

      if (target_field == "token_embd.weight") {
        model.embed_tokens_f32 = std::move(dest_f32);
        Logger::info("Mapped GGUF tensor '" + info.name +
                     "' (BF16) to model.embed_tokens_f32");
      } else if (target_field == "output.weight") {
        model.lm_head_f32 = std::move(dest_f32);
        Logger::info("Mapped GGUF tensor '" + info.name +
                     "' (BF16) to model.lm_head_f32");
      } else if (target_field == "output_norm.weight") {
        model.final_norm_f32 = std::move(dest_f32);
        Logger::info("Mapped GGUF tensor '" + info.name +
                     "' (BF16) to model.final_norm_f32");
      } else if (target_field.find("blk.") == 0) {
        size_t start = 4;
        size_t end = target_field.find('.', start);
        int layer_idx = std::stoi(target_field.substr(start, end - start));
        std::string sub_field = target_field.substr(end + 1);

        if (layer_idx >= 0 && layer_idx < model.layers.size()) {
          if (sub_field == "attn_q.weight")
            model.layers[layer_idx].q_proj_f32 = std::move(dest_f32);
          else if (sub_field == "attn_k.weight")
            model.layers[layer_idx].k_proj_f32 = std::move(dest_f32);
          else if (sub_field == "attn_v.weight")
            model.layers[layer_idx].v_proj_f32 = std::move(dest_f32);
          else if (sub_field == "attn_output.weight")
            model.layers[layer_idx].o_proj_f32 = std::move(dest_f32);
          else if (sub_field == "ffn_gate.weight")
            model.layers[layer_idx].gate_proj_f32 = std::move(dest_f32);
          else if (sub_field == "ffn_up.weight")
            model.layers[layer_idx].up_proj_f32 = std::move(dest_f32);
          else if (sub_field == "ffn_down.weight")
            model.layers[layer_idx].down_proj_f32 = std::move(dest_f32);
          else if (sub_field == "attn_norm.weight")
            model.layers[layer_idx].input_layernorm_f32 = std::move(dest_f32);
          else if (sub_field == "ffn_norm.weight")
            model.layers[layer_idx].post_attention_layernorm_f32 =
                std::move(dest_f32);
          else {
            Logger::warning("Unsupported layer sub-field (BF16): " + sub_field +
                            " for tensor '" + info.name + "'");
            continue;
          }
          Logger::info("Mapped GGUF tensor '" + info.name +
                       "' (BF16) to layer " + std::to_string(layer_idx) +
                       " field " + sub_field);
        } else {
          Logger::warning("Invalid layer index (BF16) " +
                          std::to_string(layer_idx) +
                          " parsed from tensor name '" + info.name + "'");
          continue;
        }
      } else {
        Logger::warning("Unhandled target field (BF16): " + target_field +
                        " for tensor '" + info.name + "'");
      }
    } else if (info.type == GGMLType::GGML_TYPE_Q4_K) {
      size_t expected_bytes =
          ggml_type_size(info.type) *
          (info.num_elements / ggml_type_block_size(info.type));
      if (info.size_in_bytes != expected_bytes) {
        Logger::warning("Size mismatch for Q4_K tensor '" + info.name +
                        "'. Expected " + std::to_string(expected_bytes) +
                        ", got " + std::to_string(info.size_in_bytes));
      }

      size_t num_blocks = info.size_in_bytes / sizeof(block_q4_K);
      std::vector<block_q4_K> dest_q4k(num_blocks);
      std::memcpy(dest_q4k.data(), tensor_data_ptr, info.size_in_bytes);

      if (target_field == "token_embd.weight")
        model.embed_tokens_q4k = std::move(dest_q4k);
      else if (target_field == "output.weight")
        model.lm_head_q4k = std::move(dest_q4k);
      else if (target_field.find("blk.") == 0) {
        size_t start = 4;
        size_t end = target_field.find('.', start);
        int layer_idx = std::stoi(target_field.substr(start, end - start));
        std::string sub_field = target_field.substr(end + 1);
        if (layer_idx >= 0 && layer_idx < model.layers.size()) {
          if (sub_field == "attn_q.weight")
            model.layers[layer_idx].q_proj_q4k = std::move(dest_q4k);
          else if (sub_field == "attn_k.weight")
            model.layers[layer_idx].k_proj_q4k = std::move(dest_q4k);
          else if (sub_field == "attn_v.weight")
            model.layers[layer_idx].v_proj_q4k = std::move(dest_q4k);
          else if (sub_field == "attn_output.weight")
            model.layers[layer_idx].o_proj_q4k = std::move(dest_q4k);
          else if (sub_field == "ffn_gate.weight")
            model.layers[layer_idx].gate_proj_q4k = std::move(dest_q4k);
          else if (sub_field == "ffn_up.weight")
            model.layers[layer_idx].up_proj_q4k = std::move(dest_q4k);
          else if (sub_field == "ffn_down.weight")
            model.layers[layer_idx].down_proj_q4k = std::move(dest_q4k);
          else {
            Logger::warning("Unsupported layer sub-field (Q4_K): " + sub_field);
            continue;
          }
          Logger::info("Mapped GGUF tensor '" + info.name +
                       "' (Q4_K) to layer " + std::to_string(layer_idx) +
                       " field " + sub_field);
        } else {
          Logger::warning("Invalid layer index (Q4_K) " +
                          std::to_string(layer_idx));
          continue;
        }
      } else {
        Logger::warning("Unhandled target field (Q4_K): " + target_field);
      }
    } else if (info.type == GGMLType::GGML_TYPE_Q6_K) {
      size_t expected_bytes =
          ggml_type_size(info.type) *
          (info.num_elements / ggml_type_block_size(info.type));
      if (info.size_in_bytes != expected_bytes) {
        Logger::warning("Size mismatch for Q6_K tensor '" + info.name +
                        "'. Expected " + std::to_string(expected_bytes) +
                        ", got " + std::to_string(info.size_in_bytes));
      }

      size_t num_blocks = info.size_in_bytes / sizeof(block_q6_K);
      std::vector<block_q6_K> dest_q6k(num_blocks);
      std::memcpy(dest_q6k.data(), tensor_data_ptr, info.size_in_bytes);

      if (target_field == "token_embd.weight")
        model.embed_tokens_q6k = std::move(dest_q6k);
      else if (target_field == "output.weight")
        model.lm_head_q6k = std::move(dest_q6k);
      else if (target_field.find("blk.") == 0) {
        size_t start = 4;
        size_t end = target_field.find('.', start);
        int layer_idx = std::stoi(target_field.substr(start, end - start));
        std::string sub_field = target_field.substr(end + 1);
        if (layer_idx >= 0 && layer_idx < model.layers.size()) {
          if (sub_field == "attn_q.weight")
            model.layers[layer_idx].q_proj_q6k = std::move(dest_q6k);
          else if (sub_field == "attn_k.weight")
            model.layers[layer_idx].k_proj_q6k = std::move(dest_q6k);
          else if (sub_field == "attn_v.weight")
            model.layers[layer_idx].v_proj_q6k = std::move(dest_q6k);
          else if (sub_field == "attn_output.weight")
            model.layers[layer_idx].o_proj_q6k = std::move(dest_q6k);
          else if (sub_field == "ffn_gate.weight")
            model.layers[layer_idx].gate_proj_q6k = std::move(dest_q6k);
          else if (sub_field == "ffn_up.weight")
            model.layers[layer_idx].up_proj_q6k = std::move(dest_q6k);
          else if (sub_field == "ffn_down.weight")
            model.layers[layer_idx].down_proj_q6k = std::move(dest_q6k);
          else {
            Logger::warning("Unsupported layer sub-field (Q6_K): " + sub_field);
            continue;
          }
          Logger::info("Mapped GGUF tensor '" + info.name +
                       "' (Q6_K) to layer " + std::to_string(layer_idx) +
                       " field " + sub_field);
        } else {
          Logger::warning("Invalid layer index (Q6_K) " +
                          std::to_string(layer_idx));
          continue;
        }
      } else {
        Logger::warning("Unhandled target field (Q6_K): " + target_field);
      }
    } else if (info.type == GGMLType::GGML_TYPE_Q8_0) {
      auto assign_vec_q8_0 = [&](std::vector<block_q8_0>& vec,
                                 const GGUFTensorInfo& info_local) {
        if (info_local.num_elements == 0) {
          Logger::warning("Tensor '" + info_local.name +
                          "' (Q8_0) has 0 elements. Skipping assignment.");
          return;
        }
        if (info_local.num_elements % GGML_QK8_0 != 0) {
          Logger::error(
              "Tensor '" + info_local.name + "' (Q8_0) num_elements " +
              std::to_string(info_local.num_elements) +
              " is not divisible by GGML_QK8_0 (" + std::to_string(GGML_QK8_0) +
              "). Cannot map as blocks.");
          vec.clear();
          return;
        }
        size_t num_blocks = info_local.num_elements / GGML_QK8_0;
        // info_local.offset is relative to the start of the actual data block within the mmap
        const block_q8_0* src = reinterpret_cast<const block_q8_0*>(
            static_cast<const uint8_t*>(gguf.mapped_tensor_data) + gguf.offset_diff_for_mmap + info_local.offset);

        const uint8_t* mmap_start_ptr = static_cast<const uint8_t*>(gguf.mapped_tensor_data);
        const uint8_t* mmap_end_ptr = mmap_start_ptr + gguf.mapped_tensor_data_size;

        if (reinterpret_cast<const uint8_t*>(src) < mmap_start_ptr ||
            reinterpret_cast<const uint8_t*>(src + num_blocks) > mmap_end_ptr) {
          Logger::error(
              "Tensor '" + info_local.name +
              "' (Q8_0) data is out of mmapped bounds. Calculated Src: " + std::to_string(reinterpret_cast<uintptr_t>(src)) +
              ", MmapStart: " + std::to_string(reinterpret_cast<uintptr_t>(mmap_start_ptr)) + 
              ", CalculatedEnd: " + std::to_string(reinterpret_cast<uintptr_t>(src + num_blocks)) + 
              ", MmapEnd: " + std::to_string(reinterpret_cast<uintptr_t>(mmap_end_ptr)) +
              ". GGUF Offset: " + std::to_string(info_local.offset) + ", NumBlocks: " +
              std::to_string(num_blocks) + ", ExpectedBlockSize: " +
              std::to_string(num_blocks * sizeof(block_q8_0)) +
              ", MmapTotalSize: " + std::to_string(gguf.mapped_tensor_data_size) +
              ", MmapOffsetDiff: " + std::to_string(gguf.offset_diff_for_mmap));
          vec.clear();
          return;
        }
        vec.assign(src, src + num_blocks);
      };

      if (target_field == "token_embd.weight") {
        assign_vec_q8_0(model.embed_tokens_q8_0, info);
        Logger::info("Mapped GGUF tensor '" + info.name +
                     "' (Q8_0) to model.embed_tokens_q8_0");
      } else if (target_field == "output.weight") {
        assign_vec_q8_0(model.lm_head_q8_0, info);
        Logger::info("Mapped GGUF tensor '" + info.name +
                     "' (Q8_0) to model.lm_head_q8_0");
      } else if (target_field.find("blk.") == 0) {
        size_t start = 4;
        size_t end = target_field.find('.', start);
        int layer_idx = -1;
        std::string sub_field;
        if (end != std::string::npos) {
          layer_idx = std::stoi(target_field.substr(start, end - start));
          sub_field = target_field.substr(end + 1);
        }
        if (layer_idx >= 0 && layer_idx < model.layers.size()) {
          auto& lw = model.layers[layer_idx];
          if (sub_field == "attn_q.weight")
            assign_vec_q8_0(lw.q_proj_q8_0, info);
          else if (sub_field == "attn_k.weight")
            assign_vec_q8_0(lw.k_proj_q8_0, info);
          else if (sub_field == "attn_v.weight")
            assign_vec_q8_0(lw.v_proj_q8_0, info);
          else if (sub_field == "attn_output.weight")
            assign_vec_q8_0(lw.o_proj_q8_0, info);
          else if (sub_field == "ffn_gate.weight")
            assign_vec_q8_0(lw.gate_proj_q8_0, info);
          else if (sub_field == "ffn_up.weight")
            assign_vec_q8_0(lw.up_proj_q8_0, info);
          else if (sub_field == "ffn_down.weight")
            assign_vec_q8_0(lw.down_proj_q8_0, info);
          else {
            Logger::warning("Unsupported layer sub-field (Q8_0): '" +
                            sub_field + "' for tensor '" + info.name + "'");
            return;
          }
          Logger::info("Mapped GGUF tensor '" + info.name +
                       "' (Q8_0) to layer " + std::to_string(layer_idx) +
                       " field " + sub_field);
        } else {
          Logger::warning("Invalid layer index (Q8_0) " +
                          std::to_string(layer_idx) +
                          " parsed from tensor name '" + info.name + "'");
        }
      } else {
        Logger::warning("Unhandled target field (Q8_0): '" + target_field +
                        "' for tensor '" + info.name + "'");
      }
    } else {
      Logger::warning("Tensor '" + info.name + "' has unhandled GGUF type: " +
                      ggml_type_name(info.type) + " (" +
                      std::to_string(static_cast<int>(info.type)) + ")");
    }
  }

  Logger::info("Finished mapping GGUF weights.");
}

ModelConfig parse_model_config_from_gguf(const GGUFData& gguf) {
  ModelConfig config;
  Logger::info("[parse_gguf_config] Entered function.");

  auto get_meta_string = [&](const std::string& key,
                             const std::string& default_val) -> std::string {
    auto it = gguf.metadata.find(key);
    if (it != gguf.metadata.end() &&
        std::holds_alternative<std::string>(it->second)) {
      return std::get<std::string>(it->second);
    }
    return default_val;
  };

  auto get_meta_value = [&](const std::string& key, auto default_value) {
    using TargetType = typename std::decay<decltype(default_value)>::type;
    auto it = gguf.metadata.find(key);
    if (it != gguf.metadata.end()) {
      return std::visit(
          [&](const auto& val) -> TargetType {
            using T = std::decay_t<decltype(val)>;

            if constexpr (std::is_integral_v<TargetType>) {
              if constexpr (std::is_integral_v<T> && !std::is_same_v<T, bool>) {
                if constexpr (std::is_unsigned_v<T> &&
                              std::is_signed_v<TargetType>) {
                  if (val > static_cast<std::make_unsigned_t<TargetType>>(
                                std::numeric_limits<TargetType>::max())) {
                    Logger::warning("Metadata key '" + key + "' value " +
                                    std::to_string(val) +
                                    " overflows TargetType. Using default.");
                    return default_value;
                  }
                }

                else if constexpr (std::is_signed_v<T> &&
                                   std::is_signed_v<TargetType> &&
                                   sizeof(T) > sizeof(TargetType)) {
                  if (val > static_cast<T>(
                                std::numeric_limits<TargetType>::max()) ||
                      val < static_cast<T>(
                                std::numeric_limits<TargetType>::lowest())) {
                    Logger::warning("Metadata key '" + key + "' value " +
                                    std::to_string(val) +
                                    " overflows TargetType. Using default.");
                    return default_value;
                  }
                }
                return static_cast<TargetType>(val);
              }
            } else if constexpr (std::is_floating_point_v<TargetType>) {
              if constexpr (std::is_floating_point_v<T>) {
                return static_cast<TargetType>(val);
              }
            } else if constexpr (std::is_same_v<TargetType, bool>) {
              if constexpr (std::is_same_v<T, bool>) {
                return val;
              }
            } else if constexpr (std::is_same_v<TargetType, std::string>) {
              if constexpr (std::is_same_v<T, std::string>) {
                return val;
              }
            }
            Logger::warning("Metadata key '" + key +
                            "' has stored type incompatible with requested "
                            "TargetType. Using default.");
            return default_value;
          },
          it->second);
    } else {
      return default_value;
    }
  };

  config.vocab_size = get_meta_value("tokenizer.ggml.vocab_size",
                                     get_meta_value("llama.vocab_size", 32000));
  config.hidden_size = get_meta_value("llama.embedding_length", 4096);
  config.intermediate_size = get_meta_value("llama.feed_forward_length", 11008);
  config.num_attention_heads = get_meta_value("llama.attention.head_count", 32);
  config.num_hidden_layers = get_meta_value("llama.block_count", 32);
  config.num_key_value_heads = get_meta_value("llama.attention.head_count_kv",
                                              config.num_attention_heads);
  config.max_position_embeddings = get_meta_value("llama.context_length", 4096);
  if (config.max_position_embeddings == 0 ||
      config.max_position_embeddings > 8192) {
    Logger::warning("max_position_embeddings from GGUF is " +
                    std::to_string(config.max_position_embeddings) +
                    ", overriding to sensible default (2048)");
    config.max_position_embeddings = 2048;
  }
  config.rms_norm_eps =
      get_meta_value("llama.attention.layer_norm_rms_epsilon", 1e-5f);
  config.rope_theta = get_meta_value("llama.rope.freq_base", 10000.0f);
  config.hidden_act = "silu";
  config.bos_token_id = get_meta_value("tokenizer.ggml.bos_token_id", -1);
  config.eos_token_id = get_meta_value("tokenizer.ggml.eos_token_id", -1);
  config.unk_token_id = get_meta_value("tokenizer.ggml.unk_token_id", -1);
  config.pad_token_id = get_meta_value("tokenizer.ggml.padding_token_id", -1);

  config.architecture = get_meta_string("general.architecture", "unknown");
  config.model_name = get_meta_string("general.name", "unknown");
  bool has_pre_key = gguf.metadata.count("tokenizer.ggml.pre");
  bool has_merges = !gguf.tokenizer_merges.empty();

  Logger::info("[parse_gguf_config] Architecture: " + config.architecture +
               ", Vocab Size: " + std::to_string(config.vocab_size) +
               ", Has Merges: " + (has_merges ? "Yes" : "No"));

  
  Logger::info("[parse_gguf_config] Identifying tokenizer family...");
  bool is_llama3_arch_hint = (config.architecture.find("llama3") != std::string::npos ||
                         config.architecture.find("Llama-3") != std::string::npos ||
                         config.architecture.find("Meta-Llama-3") != std::string::npos);
  bool is_llama3_vocab_size = (config.vocab_size == 128256);
  std::string ggml_tokenizer_model = get_meta_string("tokenizer.ggml.model", "");
  bool is_tiktoken_style_tokenizer_model = (ggml_tokenizer_model == "gpt2");

  Logger::info("[parse_gguf_config] L3 Hints: arch_hint=" + std::string(is_llama3_arch_hint ? "Y":"N") +
                 ", vocab_size_match=" + std::string(is_llama3_vocab_size ? "Y":"N") +
                 ", has_merges=" + std::string(has_merges ? "Y":"N") +
                 ", ggml_tokenizer_model_key='" + ggml_tokenizer_model + "' (is_tiktoken_style: " + std::string(is_tiktoken_style_tokenizer_model ? "Y":"N") + ")" );

  if (has_merges && is_llama3_vocab_size && is_tiktoken_style_tokenizer_model) {
    config.tokenizer_family = ModelConfig::TokenizerFamily::LLAMA3_TIKTOKEN;
    Logger::info("[parse_gguf_config] Result: Identified LLAMA3_TIKTOKEN (merges + vocab_size + ggml_tokenizer_model='gpt2'). Architecture string was: '" + config.architecture + "'");
    if (!is_llama3_arch_hint && config.architecture == "llama") {
         Logger::info("[parse_gguf_config] Note: Classified as Llama 3 based on tokenizer/vocab, but arch string was 'llama'.");
    }
    if (config.rope_theta == 10000.0f) { 
         float llama3_rope_candidate = get_meta_value("llama.rope.freq_base", 500000.0f); 
         if (llama3_rope_candidate > 10000.0f) {
             config.rope_theta = llama3_rope_candidate;
             Logger::info("[parse_gguf_config] Adjusted rope_theta to " + std::to_string(config.rope_theta) + " for Llama 3 model (was 10000.0).");
         }
    }
  } else if (config.architecture == "llama" || config.architecture.find("Llama-2") != std::string::npos || config.architecture.find("TinyLlama") != std::string::npos) {
    config.tokenizer_family = ModelConfig::TokenizerFamily::LLAMA_SENTENCEPIECE;
     Logger::info("[parse_gguf_config] Result: Identified LLAMA_SENTENCEPIECE based on architecture: '" + config.architecture + "'");
  } else {
    config.tokenizer_family = ModelConfig::TokenizerFamily::UNKNOWN;
     Logger::info("[parse_gguf_config] Result: UNKNOWN tokenizer family for architecture: '" + config.architecture + "'");
  }

  // Existing chat_template_type and pre_tokenizer_type logic based on architecture and pre_key
  if (config.model_name.find("TinyLlama") != std::string::npos ||
      (config.architecture == "llama" && has_pre_key)) {
    config.chat_template_type = "tinyllama";
  } else if (config.architecture == "llama" && !has_pre_key) {
    config.chat_template_type = "llama2";
  } else {
    config.chat_template_type = "unknown";
    Logger::warning("Could not determine chat template type for arch='" +
                    config.architecture + "', name='" + config.model_name +
                    "'.");
  }

  if (has_pre_key) {
    config.pre_tokenizer_type =
        get_meta_string("tokenizer.ggml.pre", "unknown");
  } else if (config.architecture == "llama") {
    config.pre_tokenizer_type = "llama";
  } else {
    config.pre_tokenizer_type = "unknown";
  }
  Logger::info("Determined config: architecture='" + config.architecture +
               "', model_name='" + config.model_name + "', chat_template='" +
               config.chat_template_type + "', pre_tokenizer='" +
               config.pre_tokenizer_type + "'");

  if (config.model_name == "llama" && config.pre_tokenizer_type != "llama") {
    config.chat_template_type = "llama2";
    Logger::info(
        "Inferred chat_template_type='llama2' based on model_type and "
        "missing/different pre_tokenizer_type.");
  }

  auto template_it = gguf.metadata.find("tokenizer.chat_template");
  if (template_it != gguf.metadata.end() &&
      std::holds_alternative<std::string>(template_it->second)) {
    config.chat_template_string = std::get<std::string>(template_it->second);
    Logger::info("Found tokenizer.chat_template in metadata.");

  } else {
    Logger::info(
        "tokenizer.chat_template not found or not a string in metadata. Will "
        "use fallback logic.");
    config.chat_template_string = "";
  }
  if (config.chat_template_type == "unknown") {
    if (config.model_name == "llama" && config.pre_tokenizer_type != "llama") {
      config.chat_template_type = "llama2";
      Logger::info(
          "Inferred chat_template_type='llama2' based on model name and "
          "missing/different pre_tokenizer_type.");
    } else if (config.tokenizer_family == ModelConfig::TokenizerFamily::LLAMA3_TIKTOKEN) {
        Logger::info("Llama 3 model identified. Chat template will primarily rely on 'tokenizer.chat_template' from GGUF if present.");
        // Set a generic type for now, actual application will use the string.
        if (gguf.metadata.count("tokenizer.chat_template")) {
            config.chat_template_type = "llama3_gguf_direct"; 
        } else {
            config.chat_template_type = "llama3_fallback"; // Or some other indicator
            Logger::warning("Llama 3 model detected, but 'tokenizer.chat_template' not found in GGUF metadata.");
        }
    }
  }

  Logger::info(std::string("[parse_gguf_config] Finished parsing. Returning config. Family: ") + 
                (config.tokenizer_family == ModelConfig::TokenizerFamily::LLAMA3_TIKTOKEN ? "L3_TIKTOKEN" : 
                 (config.tokenizer_family == ModelConfig::TokenizerFamily::LLAMA_SENTENCEPIECE ? "L2_SPM" : "UNKNOWN")));
  return config;
}

static void log_vector_summary_detailed(const std::string& name,
                                        const std::vector<float>& v,
                                        int current_pos, int current_layer,
                                        int N) {
  if (v.empty()) {
    Logger::info(name + " (pos=" + std::to_string(current_pos) + ", layer=" +
                 std::to_string(current_layer) + "): EMPTY VECTOR");
    return;
  }
  std::stringstream ss;
  ss << name << " (pos=" << std::to_string(current_pos)
     << ", layer=" << std::to_string(current_layer) << "): size=" << v.size();
  ss << ", first " << N << ": [";
  for (int i = 0; i < N && i < v.size(); ++i) {
    ss << std::fixed << std::setprecision(4) << v[i]
       << (i == N - 1 || i == v.size() - 1 ? "" : ", ");
  }
  ss << "]";
  float min_val = v[0], max_val = v[0], sum = 0.0f;
  bool all_finite = true;
  for (float val : v) {
    if (val < min_val) min_val = val;
    if (val > max_val) max_val = val;
    sum += val;
    if (!std::isfinite(val)) all_finite = false;
  }
  ss << ", min=" << std::fixed << std::setprecision(4) << min_val;
  ss << ", max=" << std::fixed << std::setprecision(4) << max_val;
  ss << ", mean=" << std::fixed << std::setprecision(4) << (sum / v.size());
  ss << ", finite=" << (all_finite ? "yes" : "no");
  Logger::info(ss.str());
}

void TinyLlamaModel::initialize_rope_freqs() {
  Logger::info("[ROPE_FREQ_ENTRY] Entered initialize_rope_freqs.");

  Logger::info("[ROPE_FREQ_CHECK] num_attention_heads: " + std::to_string(config_.num_attention_heads));
  if (config_.num_attention_heads == 0) {
    Logger::error("Cannot initialize RoPE frequencies: num_attention_heads is zero.");
    return;
  }
  int head_dim = config_.hidden_size / config_.num_attention_heads;
  Logger::info("[ROPE_FREQ_CHECK] calculated head_dim: " + std::to_string(head_dim));
  if (head_dim == 0) {
    Logger::error("Cannot initialize RoPE frequencies: calculated head_dim is zero.");
    return;
  }
  Logger::info("[ROPE_FREQ_CHECK] head_dim % 2 check. head_dim: " + std::to_string(head_dim));
  if (head_dim % 2 != 0) {
    Logger::error("Cannot initialize RoPE frequencies: head_dim must be even.");
    return;
  }

  // Log parameters used for RoPE initialization
  Logger::info("[ROPE_INIT] Initializing RoPE with head_dim=" + std::to_string(head_dim) +
               ", configured max_pos_emb=" + std::to_string(config_.max_position_embeddings) +
               ", using internal rope::MAX_SEQUENCE_LENGTH=" + std::to_string(rope::MAX_SEQUENCE_LENGTH) +
               ", configured rope_theta=" + std::to_string(config_.rope_theta));


  if (precomputed_freqs_cis_.empty()) { 
    int max_seq_len = rope::MAX_SEQUENCE_LENGTH; // Or config_.max_position_embeddings if preferred
    size_t required_size = (static_cast<size_t>(max_seq_len) * head_dim) / 2;
    if (required_size == 0) {
        Logger::warning("RoPE precomputation resulted in zero size. Max seq len: " + 
                        std::to_string(max_seq_len) + ", head_dim: " + std::to_string(head_dim));
        return;
    }
    precomputed_freqs_cis_.resize(required_size);
    
    float rope_theta = config_.rope_theta > 0 ? config_.rope_theta : rope::ROPE_THETA;

    for (int pos = 0; pos < max_seq_len; ++pos) {
      for (int i = 0; i < head_dim; i += 2) {
        float freq = 1.0f / std::pow(rope_theta, float(i) / head_dim);
        float val = static_cast<float>(pos) * freq;
        float cos_val = std::cos(val);
        float sin_val = std::sin(val);
        size_t flat_idx = (static_cast<size_t>(pos) * head_dim / 2) + (i / 2);
        if (flat_idx < precomputed_freqs_cis_.size()){
            precomputed_freqs_cis_[flat_idx] = {cos_val, sin_val};
        } else {
            Logger::error("RoPE precomputation index out of bounds: " + std::to_string(flat_idx) + 
                          " vs size " + std::to_string(precomputed_freqs_cis_.size()));
            return; 
        }
      }
    }
    Logger::info("Precomputed RoPE frequencies on CPU. Size: " + std::to_string(precomputed_freqs_cis_.size()));
  } else {
      Logger::info("RoPE frequencies already precomputed.");
  }
}

#ifdef HAS_CUDA
std::vector<float> TinyLlamaModel::forward_device_batch_prefill(
    float* d_batch_input_embeddings, // This is now assumed to be activations *after* CPU layers if any
    int num_tokens_in_batch,
    int current_model_pos, // This should be the starting position *within the KV cache* for the batch
    KVCache* kv_cache,
    cudaStream_t stream) {

    const int hidden_size = config_.hidden_size;
    const int head_dim = config_.hidden_size / config_.num_attention_heads;
    const int ffn_intermediate_dim = config_.intermediate_size;
    const int n_kv_dim = config_.num_key_value_heads * head_dim;

    float* d_batch_x_ptr = d_batch_input_embeddings; // Input to the first GPU layer
    float* d_batch_x_norm_out_attn;
    float* d_batch_q_proj_out;
    float* d_batch_k_proj_out;
    float* d_batch_v_proj_out;
    float* d_batch_attn_heads_concat_out;
    float* d_batch_attn_final_proj_out;
    float* d_batch_residual_attn_out; // Buffer for residual connection before MLP
    float* d_batch_x_norm_out_ffn;
    float* d_batch_ffn_gate_proj_out;
    float* d_batch_ffn_up_proj_out;
    float* d_batch_ffn_swiglu_out;
    float* d_batch_ffn_down_proj_out;
    float* d_batch_layer_output = nullptr; // Output of each GPU layer, becomes input to next

    size_t batch_hidden_size_bytes = (size_t)num_tokens_in_batch * hidden_size * sizeof(float);
    size_t batch_kv_proj_size_bytes = (size_t)num_tokens_in_batch * n_kv_dim * sizeof(float);
    size_t batch_ffn_intermediate_bytes = (size_t)num_tokens_in_batch * ffn_intermediate_dim * sizeof(float);

    gpuErrchk(cudaMalloc(&d_batch_x_norm_out_attn, batch_hidden_size_bytes));
    gpuErrchk(cudaMalloc(&d_batch_q_proj_out, batch_hidden_size_bytes));
    gpuErrchk(cudaMalloc(&d_batch_k_proj_out, batch_kv_proj_size_bytes));
    gpuErrchk(cudaMalloc(&d_batch_v_proj_out, batch_kv_proj_size_bytes));
    gpuErrchk(cudaMalloc(&d_batch_attn_heads_concat_out, batch_hidden_size_bytes));
    gpuErrchk(cudaMalloc(&d_batch_attn_final_proj_out, batch_hidden_size_bytes));
    gpuErrchk(cudaMalloc(&d_batch_residual_attn_out, batch_hidden_size_bytes)); // Allocate residual buffer
    gpuErrchk(cudaMalloc(&d_batch_x_norm_out_ffn, batch_hidden_size_bytes));
    gpuErrchk(cudaMalloc(&d_batch_ffn_gate_proj_out, batch_ffn_intermediate_bytes));
    gpuErrchk(cudaMalloc(&d_batch_ffn_up_proj_out, batch_ffn_intermediate_bytes));
    gpuErrchk(cudaMalloc(&d_batch_ffn_swiglu_out, batch_ffn_intermediate_bytes));
    gpuErrchk(cudaMalloc(&d_batch_ffn_down_proj_out, batch_hidden_size_bytes));

    // Only allocate d_batch_layer_output if there are GPU layers to process
    // This will also serve as the input to the next layer (d_batch_x_ptr)
    if (config_.num_cpu_offload_layers < config_.num_hidden_layers) {
         gpuErrchk(cudaMalloc(&d_batch_layer_output, batch_hidden_size_bytes));
    }


    const float alpha = 1.0f;
    const float beta = 0.0f;

    Logger::info("[PREFILL_DEBUG] Entering prefill. num_cpu_offload_layers: " + std::to_string(config_.num_cpu_offload_layers) + 
                 ", total_hidden_layers: " + std::to_string(config_.num_hidden_layers));

    // Iterate ONLY through layers designated for GPU
    for (int l_model_idx = config_.num_cpu_offload_layers; l_model_idx < config_.num_hidden_layers; ++l_model_idx) {
        int l_gpu_idx = l_model_idx - config_.num_cpu_offload_layers; // Index for concatenated GPU weights

        Logger::info("[PREFILL_DEBUG] Processing Layer: model_idx=" + std::to_string(l_model_idx) + ", gpu_idx=" + std::to_string(l_gpu_idx) + 
                     ". d_batch_x_ptr: " + Logger::ptrToString(d_batch_x_ptr) +
                     ", layers[l_model_idx].input_layernorm_dev: " + Logger::ptrToString(layers[l_model_idx].input_layernorm_dev));
        
        // Store the input to the attention block for the first residual connection
        // This is d_batch_x_ptr which is either initial_embeddings or output of previous layer's MLP
        gpuErrchk(cudaMemcpyAsync(d_batch_residual_attn_out, d_batch_x_ptr, batch_hidden_size_bytes, cudaMemcpyDeviceToDevice, stream));


        rmsnorm_batch_cuda(d_batch_x_norm_out_attn, d_batch_x_ptr, 
                           layers[l_model_idx].input_layernorm_dev,
                           num_tokens_in_batch, hidden_size, config_.rms_norm_eps, stream);

        const float* w_q_layer_ptr = w_q_f32_dev_ + (size_t)l_gpu_idx * hidden_size * hidden_size;
        gemm_f32_f32_cuda(cublas_handle_, false, false, num_tokens_in_batch, hidden_size, hidden_size, &alpha,
                          d_batch_x_norm_out_attn, hidden_size, w_q_layer_ptr, hidden_size, &beta,
                          d_batch_q_proj_out, hidden_size, stream);

        const float* w_k_layer_ptr = w_k_f32_dev_ + (size_t)l_gpu_idx * n_kv_dim * hidden_size;
        gemm_f32_f32_cuda(cublas_handle_, false, false, num_tokens_in_batch, n_kv_dim, hidden_size, &alpha,
                          d_batch_x_norm_out_attn, hidden_size, w_k_layer_ptr, n_kv_dim, &beta,
                          d_batch_k_proj_out, n_kv_dim, stream);

        const float* w_v_layer_ptr = w_v_f32_dev_ + (size_t)l_gpu_idx * n_kv_dim * hidden_size;
        gemm_f32_f32_cuda(cublas_handle_, false, false, num_tokens_in_batch, n_kv_dim, hidden_size, &alpha,
                          d_batch_x_norm_out_attn, hidden_size, w_v_layer_ptr, n_kv_dim, &beta,
                          d_batch_v_proj_out, n_kv_dim, stream);

        rope_batch_cuda(
            d_batch_q_proj_out,
            d_batch_k_proj_out,
            all_freqs_cis_dev,
            num_tokens_in_batch,
            config_.num_attention_heads,
            config_.num_key_value_heads,
            head_dim,
            current_model_pos, // This is the starting position in KV cache for the batch
            config_.is_gguf_file_loaded,
            stream
        );

        float* d_layer_k_cache_ptr = kv_cache->layers[l_model_idx].k_dev_fp32; 
        float* d_layer_v_cache_ptr = kv_cache->layers[l_model_idx].v_dev_fp32;

        update_kv_cache_batch_cuda(
            d_layer_k_cache_ptr,
            d_batch_k_proj_out,
            current_model_pos,
            num_tokens_in_batch,
            config_.num_key_value_heads,
            head_dim,
            config_.max_position_embeddings,
            stream
        );

        update_kv_cache_batch_cuda(
            d_layer_v_cache_ptr,
            d_batch_v_proj_out,
            current_model_pos,
            num_tokens_in_batch,
            config_.num_key_value_heads,
            head_dim,
            config_.max_position_embeddings,
            stream
        );

        float current_attention_scale = 1.0f / sqrtf((float)head_dim);
        attention_batch_prefill_cuda(
            d_batch_q_proj_out,
            d_batch_k_proj_out, // Note: For prefill, K cache is used by kernel, so this is current K
            d_batch_v_proj_out, // Note: For prefill, V cache is used by kernel, so this is current V
            d_layer_k_cache_ptr, // Full K cache for the layer
            d_layer_v_cache_ptr, // Full V cache for the layer
            d_batch_attn_heads_concat_out,
            num_tokens_in_batch,
            current_model_pos,
            config_.max_position_embeddings,
            config_.num_attention_heads,
            config_.num_key_value_heads,
            head_dim,
            current_attention_scale,
            stream,
            nullptr
        );

        const float* w_o_layer_ptr = w_o_f32_dev_ + (size_t)l_gpu_idx * hidden_size * hidden_size;
        gemm_f32_f32_cuda(cublas_handle_, false, false, num_tokens_in_batch, hidden_size, hidden_size, &alpha,
                          d_batch_attn_heads_concat_out, hidden_size, 
                          w_o_layer_ptr, hidden_size, &beta,
                          d_batch_attn_final_proj_out, hidden_size, stream);
        
        // First residual: Input to attention + Attention Output
        // d_batch_residual_attn_out was d_batch_x_ptr (input to layer's first RMSNorm)
        add_residual_batch_cuda(d_batch_x_ptr, d_batch_attn_final_proj_out, d_batch_residual_attn_out, // Store result in d_batch_residual_attn_out
                                num_tokens_in_batch, hidden_size, stream);

        // Second RMSNorm, input is the result of the first residual connection
        rmsnorm_batch_cuda(d_batch_x_norm_out_ffn, d_batch_residual_attn_out, 
                           layers[l_model_idx].post_attention_layernorm_dev,
                           num_tokens_in_batch, hidden_size, config_.rms_norm_eps, stream);

        const float* w1_layer_ptr = w_gate_f32_dev_ + (size_t)l_gpu_idx * hidden_size * ffn_intermediate_dim;
        gemm_f32_f32_cuda(cublas_handle_, false, false, num_tokens_in_batch, ffn_intermediate_dim, hidden_size, &alpha,
                          d_batch_x_norm_out_ffn, hidden_size, w1_layer_ptr, ffn_intermediate_dim, &beta,
                          d_batch_ffn_gate_proj_out, ffn_intermediate_dim, stream);

        const float* w3_layer_ptr = w_up_f32_dev_ + (size_t)l_gpu_idx * hidden_size * ffn_intermediate_dim;
        gemm_f32_f32_cuda(cublas_handle_, false, false, num_tokens_in_batch, ffn_intermediate_dim, hidden_size, &alpha,
                          d_batch_x_norm_out_ffn, hidden_size, w3_layer_ptr, ffn_intermediate_dim, &beta,
                          d_batch_ffn_up_proj_out, ffn_intermediate_dim, stream);

        swiglu_batch_cuda(d_batch_ffn_swiglu_out, d_batch_ffn_gate_proj_out, d_batch_ffn_up_proj_out,
                          num_tokens_in_batch, ffn_intermediate_dim, stream);

        const float* w2_layer_ptr = w_down_f32_dev_ + (size_t)l_gpu_idx * ffn_intermediate_dim * hidden_size;
        gemm_f32_f32_cuda(cublas_handle_, false, false, num_tokens_in_batch, hidden_size, ffn_intermediate_dim, &alpha,
                          d_batch_ffn_swiglu_out, ffn_intermediate_dim, w2_layer_ptr, hidden_size, &beta,
                          d_batch_ffn_down_proj_out, hidden_size, stream);
        
        // Second residual: Input to MLP (which is output of first residual) + MLP Output
        // Result stored in d_batch_layer_output, which becomes d_batch_x_ptr for the next iteration.
        add_residual_batch_cuda(d_batch_layer_output, d_batch_ffn_down_proj_out, d_batch_residual_attn_out,
                                num_tokens_in_batch, hidden_size, stream);
        
        d_batch_x_ptr = d_batch_layer_output; 
    }

    // After the loop, d_batch_x_ptr points to the output of the last processed GPU layer,
    // or it's still d_batch_input_embeddings if no GPU layers ran.
    // This d_batch_x_ptr is the correct input for the final RMSNorm.
    rmsnorm_batch_cuda(d_batch_x_norm_out_attn, d_batch_x_ptr,
                       final_norm_dev,
                       num_tokens_in_batch, hidden_size, config_.rms_norm_eps, stream);

    float* d_last_token_activations = d_batch_x_norm_out_attn + (size_t)(num_tokens_in_batch - 1) * hidden_size;
    
    float* d_logits_last_token;
    gpuErrchk(cudaMalloc(&d_logits_last_token, config_.vocab_size * sizeof(float)));

    matvec_f32_f32_cuda(cublas_handle_, lm_head_f32_dev_, d_last_token_activations,
                        d_logits_last_token, config_.vocab_size, hidden_size, stream);

    std::vector<float> h_logits(config_.vocab_size);
    gpuErrchk(cudaMemcpyAsync(h_logits.data(), d_logits_last_token, config_.vocab_size * sizeof(float),
                           cudaMemcpyDeviceToHost, stream));
    gpuErrchk(cudaStreamSynchronize(stream));

    // kv_cache_.seq_len is updated in api.cpp after this function returns successfully.

    gpuErrchk(cudaFree(d_batch_x_norm_out_attn));
    gpuErrchk(cudaFree(d_batch_q_proj_out));
    gpuErrchk(cudaFree(d_batch_k_proj_out));
    gpuErrchk(cudaFree(d_batch_v_proj_out));
    gpuErrchk(cudaFree(d_batch_attn_heads_concat_out));
    gpuErrchk(cudaFree(d_batch_attn_final_proj_out));
    gpuErrchk(cudaFree(d_batch_residual_attn_out));
    gpuErrchk(cudaFree(d_batch_x_norm_out_ffn));
    gpuErrchk(cudaFree(d_batch_ffn_gate_proj_out));
    gpuErrchk(cudaFree(d_batch_ffn_up_proj_out));
    gpuErrchk(cudaFree(d_batch_ffn_swiglu_out));
    gpuErrchk(cudaFree(d_batch_ffn_down_proj_out));
    if (d_batch_layer_output) { // Only free if it was allocated
        gpuErrchk(cudaFree(d_batch_layer_output));
    }
    gpuErrchk(cudaFree(d_logits_last_token));
    
    return h_logits;
}
#endif // HAS_CUDA

static void update_kv_cache_batch_cpu(
    KVCache* kv_cache,
    int layer_idx,
    const std::vector<float>& k_batch_for_layer, // [num_tokens, num_kv_heads * head_dim] for this layer
    const std::vector<float>& v_batch_for_layer, // [num_tokens, num_kv_heads * head_dim] for this layer
    int num_tokens_in_batch,
    int start_pos_in_sequence, // Starting position of this batch in the overall sequence
    int num_kv_heads,
    int head_dim
);


static void update_kv_cache_batch_cpu(
    KVCache* kv_cache,
    int layer_idx, // The absolute model layer index
    const std::vector<float>& k_batch_for_layer, // [num_tokens, num_kv_heads * head_dim] for this layer
    const std::vector<float>& v_batch_for_layer, // [num_tokens, num_kv_heads * head_dim] for this layer
    int num_tokens_in_batch,
    int start_pos_in_sequence, // Starting position of this batch in the overall sequence
    int num_kv_heads,
    int head_dim
) {
    if (!kv_cache) {
        Logger::error("update_kv_cache_batch_cpu: KVCache is null.");
        return;
    }
    if (layer_idx < 0 || static_cast<size_t>(layer_idx) >= kv_cache->layers.size()) {
        Logger::error("update_kv_cache_batch_cpu: layer_idx " + std::to_string(layer_idx) + " is out of bounds for KVCache layers (size " + std::to_string(kv_cache->layers.size()) + ").");
        return;
    }

    KVCacheLayer& layer_cache = kv_cache->layers[layer_idx];
    int kv_dim = num_kv_heads * head_dim;

    if (k_batch_for_layer.size() != static_cast<size_t>(num_tokens_in_batch * kv_dim)) {
        Logger::error("update_kv_cache_batch_cpu: k_batch_for_layer size mismatch. Expected " +
                      std::to_string(num_tokens_in_batch * kv_dim) + ", got " + std::to_string(k_batch_for_layer.size()));
        return;
    }
    if (v_batch_for_layer.size() != static_cast<size_t>(num_tokens_in_batch * kv_dim)) {
        Logger::error("update_kv_cache_batch_cpu: v_batch_for_layer size mismatch. Expected " +
                      std::to_string(num_tokens_in_batch * kv_dim) + ", got " + std::to_string(v_batch_for_layer.size()));
        return;
    }

    for (int token_idx_in_batch = 0; token_idx_in_batch < num_tokens_in_batch; ++token_idx_in_batch) {
        // Calculate the offset for the current token's K/V data within the k_batch_for_layer/v_batch_for_layer
        size_t batch_offset = static_cast<size_t>(token_idx_in_batch) * kv_dim;

        // Append the K values for the current token to the layer's K cache
        layer_cache.k.insert(layer_cache.k.end(),
                             k_batch_for_layer.begin() + batch_offset,
                             k_batch_for_layer.begin() + batch_offset + kv_dim);

        // Append the V values for the current token to the layer's V cache
        layer_cache.v.insert(layer_cache.v.end(),
                             v_batch_for_layer.begin() + batch_offset,
                             v_batch_for_layer.begin() + batch_offset + kv_dim);
    }
    // The KVCache's main seq_len will be updated after all layers and all tokens in the batch are processed,
    // not within this per-layer update function.
}

static void attention_batch_cpu(
    const std::vector<float>& q_batch_roped, // [num_tokens, num_q_heads * head_dim] (hs dimension)
    KVCacheLayer& current_layer_kv_cache,    // K and V for the current layer
    std::vector<float>& batch_attn_output,   // Output: [num_tokens, num_q_heads * head_dim] (hs dimension)
    int num_tokens_in_batch,
    int start_pos_in_sequence,               // The sequence position where this batch begins
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float attention_scale
) {
    if (q_batch_roped.size() != (size_t)num_tokens_in_batch * num_q_heads * head_dim) {
        Logger::error("attention_batch_cpu: q_batch_roped size mismatch.");
        std::fill(batch_attn_output.begin(), batch_attn_output.end(), 0.0f);
        return;
    }
    // Ensure output vector is correctly sized and initialized
    batch_attn_output.assign((size_t)num_tokens_in_batch * num_q_heads * head_dim, 0.0f);

    // Note: num_q_heads * head_dim == hidden_size (hs)

    // Consider OMP parallel for token_idx if KVCacheLayer access is thread-safe
    // or if each thread works on a private copy/reduction variables for scores.
    // For now, processing tokens sequentially for clarity and correctness.
    for (int token_idx = 0; token_idx < num_tokens_in_batch; ++token_idx) {
        size_t q_token_offset = (size_t)token_idx * num_q_heads * head_dim;
        size_t attn_out_token_offset = (size_t)token_idx * num_q_heads * head_dim;
        int current_token_absolute_pos = start_pos_in_sequence + token_idx;

        for (int h_q = 0; h_q < num_q_heads; ++h_q) {
            const float* q_head_for_token_ptr = q_batch_roped.data() + q_token_offset + (h_q * head_dim);
            
            int kv_group_head_idx = h_q / (num_q_heads / num_kv_heads); // Target KV head for this Q head

            // history_len is the number of K/V pairs to attend to for this token,
            // which is up to and including its own position.
            int history_len = current_token_absolute_pos + 1;
            std::vector<float> scores(history_len);

            for (int t_hist = 0; t_hist < history_len; ++t_hist) { // t_hist is an absolute position in the sequence
                // Offset to find the start of the relevant KV head's data for historical token t_hist
                // K cache stores K values as [tok0_kv_head0_dim0,..., tok0_kv_head(N-1)_dim(M-1), 
                //                           tok1_kv_head0_dim0, ..., tok1_kv_head(N-1)_dim(M-1), ...]
                // So, for token t_hist, and kv_group_head_idx, the data starts at:
                // (t_hist * num_kv_heads * head_dim) + (kv_group_head_idx * head_dim)
                size_t k_cache_offset = ((size_t)t_hist * num_kv_heads + kv_group_head_idx) * head_dim;
                
                if (k_cache_offset + head_dim > current_layer_kv_cache.k.size()) {
                     Logger::error("attention_batch_cpu: K cache out of bounds. Attending token_idx " + std::to_string(token_idx) +
                                   " (abs_pos " + std::to_string(current_token_absolute_pos) + ") at history_pos " + std::to_string(t_hist) +
                                   ". Required k_cache_offset " + std::to_string(k_cache_offset + head_dim) +
                                   " > cache_k_size " + std::to_string(current_layer_kv_cache.k.size()));
                    scores[t_hist] = -std::numeric_limits<float>::infinity(); 
                    continue;
                }

                float current_dot_product = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    current_dot_product += q_head_for_token_ptr[d] * current_layer_kv_cache.k[k_cache_offset + d];
                }
                scores[t_hist] = current_dot_product * attention_scale;
            }

            softmax_vector_cpu(scores, scores); // In-place softmax

            // Weighted sum with V
            float* current_attn_out_head_ptr = batch_attn_output.data() + attn_out_token_offset + (h_q * head_dim);
            // Output for this head is already zeroed by batch_attn_output.assign()

            for (int t_hist = 0; t_hist < history_len; ++t_hist) {
                if (scores[t_hist] == -std::numeric_limits<float>::infinity() || scores[t_hist] == 0.0f) continue;

                size_t v_cache_offset = ((size_t)t_hist * num_kv_heads + kv_group_head_idx) * head_dim;
                if (v_cache_offset + head_dim > current_layer_kv_cache.v.size()) {
                     Logger::error("attention_batch_cpu: V cache out of bounds. Attending token_idx " + std::to_string(token_idx) +
                                   " (abs_pos " + std::to_string(current_token_absolute_pos) + ") at history_pos " + std::to_string(t_hist) +
                                   ". Required v_cache_offset " + std::to_string(v_cache_offset + head_dim) +
                                   " > cache_v_size " + std::to_string(current_layer_kv_cache.v.size()));
                    continue; 
                }
                for (int d = 0; d < head_dim; ++d) {
                    current_attn_out_head_ptr[d] += scores[t_hist] * current_layer_kv_cache.v[v_cache_offset + d];
                }
            }
        } // End Q heads loop (h_q)
    } // End tokens_in_batch loop (token_idx)
}
std::vector<float> TinyLlamaModel::forward_cpu_batch(
    const std::vector<float>& batch_input_activations, // Batched: [num_tokens, hidden_size]
    int num_tokens_in_batch,
    int num_cpu_layers_to_process,
    int start_pos_in_sequence, // Starting position of this batch in the overall sequence (for KVCache)
    KVCache* kv_cache) {

    Logger::info("[CPU_BATCH_FWD] Entered. num_tokens_in_batch: " + std::to_string(num_tokens_in_batch) +
                  ", num_cpu_layers_to_process: " + std::to_string(num_cpu_layers_to_process) +
                  ", start_pos_in_sequence: " + std::to_string(start_pos_in_sequence));

    if (num_cpu_layers_to_process <= 0) {
        Logger::info("[CPU_BATCH_FWD] No CPU layers to process. Returning input activations.");
        return batch_input_activations;
    }
    if (batch_input_activations.size() != (size_t)num_tokens_in_batch * config_.hidden_size) {
        Logger::error("[CPU_BATCH_FWD] batch_input_activations size mismatch. Expected: " +
                      std::to_string((size_t)num_tokens_in_batch * config_.hidden_size) + " Got: " +
                      std::to_string(batch_input_activations.size()));
        return {};
    }

    int hs = config_.hidden_size;
    int is = config_.intermediate_size;
    int n_heads = config_.num_attention_heads;
    int n_kv_heads = config_.num_key_value_heads;
    if (n_heads == 0) {
        Logger::error("[CPU_BATCH_FWD] Error: num_attention_heads is zero.");
        return {};
    }
    int head_dim = hs / n_heads;
    float eps = config_.rms_norm_eps;
    int max_pos_embeddings = config_.max_position_embeddings;
    bool use_rope_adjacent_pairing = config_.is_gguf_file_loaded; // Determine RoPE style
    float attention_scale = 1.0f / SAFE_SQRT(static_cast<float>(head_dim));

    std::vector<float> current_batch_activations = batch_input_activations;

    for (int l = 0; l < num_cpu_layers_to_process; ++l) {
        const auto& lw = layers[l];
        
        std::vector<float> batch_x_norm1(current_batch_activations.size());
        const std::vector<float>& w_input_norm_vec =
            lw.input_layernorm_f32.empty()
                ? bf16vec_to_float_vec(lw.input_layernorm) // Assumes bf16vec_to_float_vec is available
                : lw.input_layernorm_f32;
        rmsnorm_batch_cpu(current_batch_activations, w_input_norm_vec, batch_x_norm1, num_tokens_in_batch, hs, eps);

        std::vector<float> residual_batch_component_attn = current_batch_activations; 

        std::vector<float> q_batch((size_t)num_tokens_in_batch * hs);
        std::vector<float> k_batch((size_t)num_tokens_in_batch * n_kv_heads * head_dim);
        std::vector<float> v_batch((size_t)num_tokens_in_batch * n_kv_heads * head_dim);

        // Q Projection
        if (!lw.q_proj_f32.empty()) {
            matmul_f32_f32_batch_cpu(lw.q_proj_f32, batch_x_norm1, q_batch, num_tokens_in_batch, hs, hs);
        } else if (!lw.q_proj_q8_0.empty()) {
            matmul_q8_0_f32_batch_cpu(lw.q_proj_q8_0, batch_x_norm1, q_batch, num_tokens_in_batch, hs, hs);
        } else if (!lw.q_proj_q6k.empty()) {
            matmul_q6k_f32_batch_cpu(lw.q_proj_q6k, batch_x_norm1, q_batch, num_tokens_in_batch, hs, hs);
        } else if (!lw.q_proj_q4k.empty()) {
            matmul_q4k_f32_batch_cpu(lw.q_proj_q4k, batch_x_norm1, q_batch, num_tokens_in_batch, hs, hs);
        } else {
            Logger::error("[CPU_BATCH_FWD] Layer " + std::to_string(l) + ": No Q proj weights found for CPU (batched)"); 
            return {};
        }
        
        // K Projection
        if (!lw.k_proj_f32.empty()) {
            matmul_f32_f32_batch_cpu(lw.k_proj_f32, batch_x_norm1, k_batch, num_tokens_in_batch, n_kv_heads * head_dim, hs);
        } else if (!lw.k_proj_q8_0.empty()) {
            matmul_q8_0_f32_batch_cpu(lw.k_proj_q8_0, batch_x_norm1, k_batch, num_tokens_in_batch, n_kv_heads * head_dim, hs);
        } else if (!lw.k_proj_q6k.empty()) {
            matmul_q6k_f32_batch_cpu(lw.k_proj_q6k, batch_x_norm1, k_batch, num_tokens_in_batch, n_kv_heads * head_dim, hs);
        } else if (!lw.k_proj_q4k.empty()) {
            matmul_q4k_f32_batch_cpu(lw.k_proj_q4k, batch_x_norm1, k_batch, num_tokens_in_batch, n_kv_heads * head_dim, hs);
        } else {
            Logger::error("[CPU_BATCH_FWD] Layer " + std::to_string(l) + ": No K proj weights found for CPU (batched)"); 
            return {};
        }

        // V Projection
        if (!lw.v_proj_f32.empty()) {
            matmul_f32_f32_batch_cpu(lw.v_proj_f32, batch_x_norm1, v_batch, num_tokens_in_batch, n_kv_heads * head_dim, hs);
        } else if (!lw.v_proj_q8_0.empty()) {
            matmul_q8_0_f32_batch_cpu(lw.v_proj_q8_0, batch_x_norm1, v_batch, num_tokens_in_batch, n_kv_heads * head_dim, hs);
        } else if (!lw.v_proj_q6k.empty()) {
            matmul_q6k_f32_batch_cpu(lw.v_proj_q6k, batch_x_norm1, v_batch, num_tokens_in_batch, n_kv_heads * head_dim, hs);
        } else if (!lw.v_proj_q4k.empty()) {
            matmul_q4k_f32_batch_cpu(lw.v_proj_q4k, batch_x_norm1, v_batch, num_tokens_in_batch, n_kv_heads * head_dim, hs);
        } else {
            Logger::error("[CPU_BATCH_FWD] Layer " + std::to_string(l) + ": No V proj weights found for CPU (batched)"); 
            return {};
        }

        // Batched RoPE Application
        apply_rope_batch_cpu(q_batch, k_batch, num_tokens_in_batch, n_heads, n_kv_heads, head_dim, 
                              start_pos_in_sequence, precomputed_freqs_cis_, max_pos_embeddings, use_rope_adjacent_pairing);

        // Batched KV Cache Update
        if (kv_cache) {
            update_kv_cache_batch_cpu(kv_cache, l, k_batch, v_batch, num_tokens_in_batch, 
                                      start_pos_in_sequence, n_kv_heads, head_dim);        
        }
        
        // --- Batched Attention ---
        // q_batch now holds the RoPEd Q values from matmul and apply_rope_batch_cpu.
        // k_batch and v_batch (after RoPE for K) have been used to update kv_cache->layers[l].
        std::vector<float> batch_attn_output((size_t)num_tokens_in_batch * hs); // hs is num_q_heads * head_dim
        
        if (kv_cache && static_cast<size_t>(l) < kv_cache->layers.size()) {
             // Ensure attention_scale is defined earlier in forward_cpu_batch, after head_dim is known.
             // float attention_scale = 1.0f / SAFE_SQRT(static_cast<float>(head_dim));
             attention_batch_cpu(q_batch,          // Input: RoPE'd Q values for the batch
                                 kv_cache->layers[l], // Input: KV Cache for the current layer (already updated)
                                 batch_attn_output,   // Output: Where attention results for the batch will be stored
                                 num_tokens_in_batch, 
                                 start_pos_in_sequence,
                                 n_heads,          // This is config_.num_attention_heads
                                 n_kv_heads,       // This is config_.num_key_value_heads
                                 head_dim, 
                                 attention_scale); // This must be defined earlier in forward_cpu_batch
        } else if (kv_cache) { 
            // This case means kv_cache exists, but the current layer 'l' is out of bounds.
            Logger::error("[CPU_BATCH_FWD] Layer " + std::to_string(l) + 
                          " is out of bounds for KV Cache access during attention. KVCache layers size: " + 
                          std::to_string(kv_cache->layers.size()) + 
                          ". Filling attention output with zeros.");
            std::fill(batch_attn_output.begin(), batch_attn_output.end(), 0.0f); 
        } else { // kv_cache is null
            Logger::error("[CPU_BATCH_FWD] KV Cache is null, cannot perform attention for layer " + std::to_string(l) +
                          ". Filling attention output with zeros.");
            std::fill(batch_attn_output.begin(), batch_attn_output.end(), 0.0f); 
        }

        // O-Projection (Batched)
        std::vector<float> batch_attn_proj_out((size_t)num_tokens_in_batch * hs);
        if(!lw.o_proj_f32.empty()) {
              matmul_f32_f32_batch_cpu(lw.o_proj_f32, batch_attn_output, batch_attn_proj_out, num_tokens_in_batch, hs, hs);
        } else if (!lw.o_proj_q8_0.empty()) {
            matmul_q8_0_f32_batch_cpu(lw.o_proj_q8_0, batch_attn_output, batch_attn_proj_out, num_tokens_in_batch, hs, hs);
        } else if (!lw.o_proj_q6k.empty()) {
            matmul_q6k_f32_batch_cpu(lw.o_proj_q6k, batch_attn_output, batch_attn_proj_out, num_tokens_in_batch, hs, hs);
        } else if (!lw.o_proj_q4k.empty()) {
            matmul_q4k_f32_batch_cpu(lw.o_proj_q4k, batch_attn_output, batch_attn_proj_out, num_tokens_in_batch, hs, hs);
        } else { 
            Logger::error("[CPU_BATCH_FWD] Layer " + std::to_string(l) + ": No O proj weights found for CPU"); 
            return {};
        }

        // First Residual Connection (Batched)
        // current_batch_activations = residual_batch_component_attn + batch_attn_proj_out;
        for(size_t i=0; i < current_batch_activations.size(); ++i) {
            current_batch_activations[i] = residual_batch_component_attn[i] + batch_attn_proj_out[i];
        }

// ... (previous parts of the layer loop, like Attention and first residual, remain the same) ...

        // --- Batched MLP Part ---
        std::vector<float> residual_batch_component_mlp = current_batch_activations; // Store for MLP residual
        std::vector<float> batch_x_norm2(current_batch_activations.size());
        
        const std::vector<float>& w_post_attn_norm_vec =
            lw.post_attention_layernorm_f32.empty()
                ? bf16vec_to_float_vec(lw.post_attention_layernorm)
                : lw.post_attention_layernorm_f32;
        // Batched RMSNorm for MLP
        rmsnorm_batch_cpu(current_batch_activations, w_post_attn_norm_vec, batch_x_norm2, num_tokens_in_batch, hs, eps);
        
        // Batched Gate and Up Projections
        std::vector<float> batch_gate_proj_out((size_t)num_tokens_in_batch * is);
        std::vector<float> batch_up_proj_out((size_t)num_tokens_in_batch * is);

        // Gate Projection
        if (!lw.gate_proj_f32.empty()) {
            matmul_f32_f32_batch_cpu(lw.gate_proj_f32, batch_x_norm2, batch_gate_proj_out, num_tokens_in_batch, is, hs);
        } else if (!lw.gate_proj_q8_0.empty()) {
            matmul_q8_0_f32_batch_cpu(lw.gate_proj_q8_0, batch_x_norm2, batch_gate_proj_out, num_tokens_in_batch, is, hs);
        } else if (!lw.gate_proj_q6k.empty()) {
            matmul_q6k_f32_batch_cpu(lw.gate_proj_q6k, batch_x_norm2, batch_gate_proj_out, num_tokens_in_batch, is, hs);
        } else if (!lw.gate_proj_q4k.empty()) {
            matmul_q4k_f32_batch_cpu(lw.gate_proj_q4k, batch_x_norm2, batch_gate_proj_out, num_tokens_in_batch, is, hs);
        } else { 
            Logger::error("[CPU_BATCH_FWD] Layer " + std::to_string(l) + ": No gate_proj weights found for CPU"); 
            return {};
        }

        // Up Projection
        if (!lw.up_proj_f32.empty()) {
            matmul_f32_f32_batch_cpu(lw.up_proj_f32, batch_x_norm2, batch_up_proj_out, num_tokens_in_batch, is, hs);
        } else if (!lw.up_proj_q8_0.empty()) {
            matmul_q8_0_f32_batch_cpu(lw.up_proj_q8_0, batch_x_norm2, batch_up_proj_out, num_tokens_in_batch, is, hs);
        } else if (!lw.up_proj_q6k.empty()) {
            matmul_q6k_f32_batch_cpu(lw.up_proj_q6k, batch_x_norm2, batch_up_proj_out, num_tokens_in_batch, is, hs);
        } else if (!lw.up_proj_q4k.empty()) {
            matmul_q4k_f32_batch_cpu(lw.up_proj_q4k, batch_x_norm2, batch_up_proj_out, num_tokens_in_batch, is, hs);
        } else { 
            Logger::error("[CPU_BATCH_FWD] Layer " + std::to_string(l) + ": No up_proj weights found for CPU"); 
            return {};
        }
        // Batched SwiGLU: SiLU(gate_proj_out) * up_proj_out
        // batch_gate_proj_out and batch_up_proj_out are already [num_tokens_in_batch * is]
        std::vector<float> batch_swiglu_out((size_t)num_tokens_in_batch * is);
        // This loop processes all elements for all tokens in a flat manner.
        // It can be parallelized with OpenMP if desired.
        // #pragma omp parallel for
        for (size_t i = 0; i < batch_gate_proj_out.size(); ++i) {
            float gate_val = batch_gate_proj_out[i];
            // SiLU = x / (1 + exp(-x))
            float silu_gate_val = gate_val / (1.0f + std::exp(-gate_val));
            batch_swiglu_out[i] = silu_gate_val * batch_up_proj_out[i];
        }
        
        // Down Projection
        std::vector<float> batch_mlp_down_proj_out((size_t)num_tokens_in_batch * hs);
        if (!lw.down_proj_f32.empty()) {
            matmul_f32_f32_batch_cpu(lw.down_proj_f32, batch_swiglu_out, batch_mlp_down_proj_out, num_tokens_in_batch, hs, is);
        } else if (!lw.down_proj_q8_0.empty()) {
            matmul_q8_0_f32_batch_cpu(lw.down_proj_q8_0, batch_swiglu_out, batch_mlp_down_proj_out, num_tokens_in_batch, hs, is);
        } else if (!lw.down_proj_q6k.empty()) {
            matmul_q6k_f32_batch_cpu(lw.down_proj_q6k, batch_swiglu_out, batch_mlp_down_proj_out, num_tokens_in_batch, hs, is);
        } else if (!lw.down_proj_q4k.empty()) {
            matmul_q4k_f32_batch_cpu(lw.down_proj_q4k, batch_swiglu_out, batch_mlp_down_proj_out, num_tokens_in_batch, hs, is);
        } else { 
            Logger::error("[CPU_BATCH_FWD] Layer " + std::to_string(l) + ": No down_proj weights found for CPU"); 
            return {};

        // Second Residual Connection (Batched)
        // current_batch_activations = residual_batch_component_mlp + batch_mlp_down_proj_out;
        // This loop processes all elements for all tokens in a flat manner.
        // #pragma omp parallel for
        for(size_t i = 0; i < current_batch_activations.size(); ++i) { // size is num_tokens_in_batch * hs
            current_batch_activations[i] = residual_batch_component_mlp[i] + batch_mlp_down_proj_out[i];
        }
    } // End layer loop (for int l = 0; l < num_cpu_layers_to_process; ++l)

    if (kv_cache && num_tokens_in_batch > 0) {
        // The KVCache seq_len is implicitly updated by update_kv_cache_batch_cpu for each token,
        // but we need a final update to the main KVCache object if it tracks a global sequence length
        // beyond individual layer states. Assuming KVCache.seq_len should reflect the end of the processed batch.
        kv_cache->seq_len = start_pos_in_sequence + num_tokens_in_batch;
        Logger::info("[CPU_BATCH_FWD] KVCache main seq_len updated to: " + std::to_string(kv_cache->seq_len));
    }

    Logger::info("[CPU_BATCH_FWD] Exiting. Output activations size: " + std::to_string(current_batch_activations.size()));
    return current_batch_activations;
}
}