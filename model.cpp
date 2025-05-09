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
#include "model_constants.h"  // Add this include
#include "model_macros.h"

static void matvec_q6k_f32_vector_cpu(const std::vector<block_q6_K>& mat_q6k,
                                      const std::vector<float>& vec_f32,
                                      std::vector<float>& out_f32, int rows,
                                      int cols, bool log_first_block = false);

static void matvec_q4k_f32_vector_cpu(const std::vector<block_q4_K>& mat_q4k,
                                      const std::vector<float>& vec_f32,
                                      std::vector<float>& out_f32, int rows,
                                      int cols, bool log_first_block = false);

inline uint16_t float32_to_bfloat16(float val);

static void matvec_f32_f32_vector_cpu(const std::vector<float>& mat_f32,
                                      const std::vector<float>& vec_f32,
                                      std::vector<float>& out_f32, int rows,
                                      int cols);

void dequantize_q8_k(const std::vector<block_q8_K>& q8k_vec,
                     std::vector<float>& out_f32, int n, bool log_this_block);
static void log_vector_summary_detailed(const std::string& name,
                                        const std::vector<float>& v,
                                        int current_pos, int current_layer,
                                        int N = 5);

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

    for (int c = 0; c < cols; ++c) {
      sum += mat_row_ptr[c] * vec_ptr[c];
    }
    out_f32[r] = sum;
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

std::vector<float> bfloat16_vector_to_float32(
    const std::vector<uint16_t>& bf16_vec) {
  std::vector<float> f32_vec(bf16_vec.size());

#pragma omp parallel for
  for (int64_t i = 0; i < static_cast<int64_t>(bf16_vec.size()); ++i) {
    f32_vec[i] = bfloat16_to_float32(bf16_vec[i]);
  }

  return f32_vec;
}

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

  float max_val = x[0];
#pragma omp parallel for reduction(max : max_val)
  for (int64_t i = 1; i < static_cast<int64_t>(n); ++i) {
    if (x[i] > max_val) max_val = x[i];
  }

  float exp_sum = 0.0f;
#pragma omp parallel for reduction(+ : exp_sum)
  for (int64_t i = 0; i < static_cast<int64_t>(n); ++i) {
    out[i] = std::exp(x[i] - max_val);
    exp_sum += out[i];
  }

  float inv_sum = 1.0f / exp_sum;
#pragma omp parallel for
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

void KVCache::initialize(int num_layers, int max_seq_len, int num_kv_heads,
                         int head_dim) {
  layers.resize(num_layers);
  seq_len = 0;
  Logger::info("Allocating KVCache host vectors...");
  size_t cache_size_per_layer = static_cast<size_t>(max_seq_len) *
                                static_cast<size_t>(num_kv_heads) *
                                static_cast<size_t>(head_dim);

  if (cache_size_per_layer == 0 && max_seq_len > 0) {
    throw std::runtime_error(
        "KVCache (CPU): Calculated cache size is zero. Check parameters.");
  }

  for (int l = 0; l < num_layers; ++l) {
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
               std::to_string(num_layers) + " layers.");

#ifdef HAS_CUDA

  allocated_num_layers = num_layers;
  allocated_max_seq_len = max_seq_len;
  allocated_num_kv_heads = num_kv_heads;
  allocated_head_dim = head_dim;

  size_t cache_elems_per_layer = static_cast<size_t>(max_seq_len) *
                                 static_cast<size_t>(num_kv_heads) *
                                 static_cast<size_t>(head_dim);
  size_t cache_bytes_per_layer = cache_elems_per_layer * sizeof(float);

  if (cache_elems_per_layer == 0) {
    throw std::runtime_error(
        "KVCache (CUDA): Calculated cache size per layer is zero. Check "
        "parameters.");
  }

  Logger::info("Allocating KVCache on GPU: " + std::to_string(num_layers) +
               " layers, size per layer: " +
               std::to_string(cache_bytes_per_layer / (1024.0 * 1024.0)) +
               " MB");

  for (int l = 0; l < num_layers; ++l) {
    if (layers[l].k_dev) {
      Logger::info(
          "Re-initializing KVCache layer K dev pointer without proper "
          "destruction?");
      gpuErrchk(cudaFree(layers[l].k_dev));
    }
    if (layers[l].v_dev) {
      Logger::info(
          "Re-initializing KVCache layer V dev pointer without proper "
          "destruction?");
      gpuErrchk(cudaFree(layers[l].v_dev));
    }

    gpuErrchk(cudaMalloc(&layers[l].k_dev, cache_bytes_per_layer));
    gpuErrchk(cudaMalloc(&layers[l].v_dev, cache_bytes_per_layer));

    gpuErrchk(cudaMemset(layers[l].k_dev, 0, cache_bytes_per_layer));
    gpuErrchk(cudaMemset(layers[l].v_dev, 0, cache_bytes_per_layer));
  }
  Logger::info("KVCache GPU allocation complete.");

#else

  Logger::info("KVCache (CPU-only build) initialized with dimensions: " +
               std::to_string(num_layers) + " layers, " +
               std::to_string(max_seq_len) + " seq len, " +
               std::to_string(num_kv_heads) + " KV heads, " +
               std::to_string(head_dim) + " head dim");
#endif
}

static std::vector<float> bf16vec_to_float_vec(
    const std::vector<uint16_t>& v_bf16) {
  std::vector<float> v_f32(v_bf16.size());
#pragma omp parallel for
  for (int64_t i = 0; i < static_cast<int64_t>(v_bf16.size()); ++i) {
    v_f32[i] = bfloat16_to_float32(v_bf16[i]);
  }
  return v_f32;
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
    float score = 0.0f;
    for (int j = 0; j < head_dim; ++j) {
      score += Q[j] * K[i * head_dim + j];
    }
    scores[i] = score * effective_scale;
  }
}

static void apply_rope_vector(
    std::vector<float>& x, int num_heads, int head_dim, int pos,
    const std::vector<std::pair<float, float>>& freqs_cis) {
  if (pos >= rope::MAX_SEQUENCE_LENGTH) {
    Logger::warning("Position " + std::to_string(pos) + 
                " exceeds maximum sequence length of " + 
                std::to_string(rope::MAX_SEQUENCE_LENGTH));
    pos = rope::MAX_SEQUENCE_LENGTH - 1;
  }

  const int dim_half = head_dim / 2;

#pragma omp parallel for
  for (int64_t h = 0; h < static_cast<int64_t>(num_heads); ++h) {
    size_t head_offset = h * head_dim;
    for (int i = 0; i < dim_half; ++i) {
      double x0 = static_cast<double>(x[head_offset + i]);
      double x1 = static_cast<double>(x[head_offset + i + dim_half]);

      double cos_val = static_cast<double>(freqs_cis[i].first);
      double sin_val = static_cast<double>(freqs_cis[i].second);

      double rotated_x0 = x0 * cos_val - x1 * sin_val;
      double rotated_x1 = x0 * sin_val + x1 * cos_val;

      x[head_offset + i] = static_cast<float>(rotated_x0);
      x[head_offset + i + dim_half] = static_cast<float>(rotated_x1);
    }
  }
}

/*
void apply_rope(torch::Tensor& x, int num_heads, int head_dim, int pos,
                const std::vector<std::pair<float, float>>& freqs_cis) {

}
*/

void TinyLlamaModel::initialize_weights(const SafeTensorsLoader* loader,
                                        const GGUFData* gguf) {
  Logger::info("Initializing model weights...");
  int hs = config_.hidden_size;
  int is = config_.intermediate_size;
  int nhl = config_.num_hidden_layers;
  int vs = config_.vocab_size;
  int n_heads = config_.num_attention_heads;
  int n_kv_heads = config_.num_key_value_heads;
  int kv_dim = (hs / n_heads) * n_kv_heads;

  layers.resize(nhl);

  if (gguf) {
    Logger::info("Mapping weights from GGUF data...");

    map_gguf_weights(*gguf, *this);
  } else if (loader) {
    Logger::info(
        "Loading weights from SafeTensors data using parallel loader...");

    std::map<std::string, std::vector<uint8_t>> all_tensors;
    try {
      all_tensors = loader->load_all_tensors_parallel();
      Logger::info(
          "All SafeTensors tensors loaded in parallel. Total tensors: " +
          std::to_string(all_tensors.size()));
    } catch (const std::exception& e) {
      Logger::error("Failed to load all tensors in parallel: " +
                    std::string(e.what()));
      throw;
    }

    auto get_tensor_data =
        [&](const std::string& name) -> const std::vector<uint8_t>& {
      auto it = all_tensors.find(name);
      if (it == all_tensors.end()) {
        throw std::runtime_error("Tensor not found in preloaded map: " + name);
      }
      return it->second;
    };

    try {
      embed_tokens = uint8_vector_to_uint16_vector(
          get_tensor_data("model.embed_tokens.weight"), vs * hs);
    } catch (const std::exception& e) {
      Logger::error("Missing model.embed_tokens.weight: " +
                    std::string(e.what()));
    }
    try {
      lm_head = uint8_vector_to_uint16_vector(
          loader->get_tensor_bytes("lm_head.weight"), vs * hs);
    } catch (const std::exception& e) {
      Logger::error("Missing lm_head.weight: " + std::string(e.what()));
    }
    try {
      final_norm = uint8_vector_to_uint16_vector(
          loader->get_tensor_bytes("model.norm.weight"), hs);
    } catch (const std::exception& e) {
      Logger::error("Missing model.norm.weight: " + std::string(e.what()));
    }

    for (int i = 0; i < nhl; ++i) {
      std::string prefix = "model.layers." + std::to_string(i) + ".";
      auto& lw = layers[i];
      try {
        lw.q_proj = uint8_vector_to_uint16_vector(
            loader->get_tensor_bytes(prefix + "self_attn.q_proj.weight"),
            hs * hs);
      } catch (const std::exception& e) {
        Logger::error("Missing " + prefix +
                      "self_attn.q_proj.weight: " + std::string(e.what()));
      }
      try {
        lw.k_proj = uint8_vector_to_uint16_vector(
            loader->get_tensor_bytes(prefix + "self_attn.k_proj.weight"),
            kv_dim * hs);
      } catch (const std::exception& e) {
        Logger::error("Missing " + prefix +
                      "self_attn.k_proj.weight: " + std::string(e.what()));
      }
      try {
        lw.v_proj = uint8_vector_to_uint16_vector(
            loader->get_tensor_bytes(prefix + "self_attn.v_proj.weight"),
            kv_dim * hs);
      } catch (const std::exception& e) {
        Logger::error("Missing " + prefix +
                      "self_attn.v_proj.weight: " + std::string(e.what()));
      }
      try {
        lw.o_proj = uint8_vector_to_uint16_vector(
            loader->get_tensor_bytes(prefix + "self_attn.o_proj.weight"),
            hs * hs);
      } catch (const std::exception& e) {
        Logger::error("Missing " + prefix +
                      "self_attn.o_proj.weight: " + std::string(e.what()));
      }
      try {
        lw.gate_proj = uint8_vector_to_uint16_vector(
            loader->get_tensor_bytes(prefix + "mlp.gate_proj.weight"), is * hs);
      } catch (const std::exception& e) {
        Logger::error("Missing " + prefix +
                      "mlp.gate_proj.weight: " + std::string(e.what()));
      }
      try {
        lw.up_proj = uint8_vector_to_uint16_vector(
            loader->get_tensor_bytes(prefix + "mlp.up_proj.weight"), is * hs);
      } catch (const std::exception& e) {
        Logger::error("Missing " + prefix +
                      "mlp.up_proj.weight: " + std::string(e.what()));
      }
      try {
        lw.down_proj = uint8_vector_to_uint16_vector(
            loader->get_tensor_bytes(prefix + "mlp.down_proj.weight"), hs * is);
      } catch (const std::exception& e) {
        Logger::error("Missing " + prefix +
                      "mlp.down_proj.weight: " + std::string(e.what()));
      }
      try {
        lw.input_layernorm = uint8_vector_to_uint16_vector(
            loader->get_tensor_bytes(prefix + "input_layernorm.weight"), hs);
      } catch (const std::exception& e) {
        Logger::error("Missing " + prefix +
                      "input_layernorm.weight: " + std::string(e.what()));
      }
      try {
        lw.post_attention_layernorm = uint8_vector_to_uint16_vector(
            loader->get_tensor_bytes(prefix +
                                     "post_attention_layernorm.weight"),
            hs);
      } catch (const std::exception& e) {
        Logger::error(
            "Missing " + prefix +
            "post_attention_layernorm.weight: " + std::string(e.what()));
      }

      lw.input_layernorm_f32 = bf16vec_to_float_vec(lw.input_layernorm);
      lw.post_attention_layernorm_f32 =
          bf16vec_to_float_vec(lw.post_attention_layernorm);
      lw.q_proj_f32 = bf16vec_to_float_vec(lw.q_proj);
      lw.k_proj_f32 = bf16vec_to_float_vec(lw.k_proj);
      lw.v_proj_f32 = bf16vec_to_float_vec(lw.v_proj);
      lw.o_proj_f32 = bf16vec_to_float_vec(lw.o_proj);
      lw.gate_proj_f32 = bf16vec_to_float_vec(lw.gate_proj);
      lw.up_proj_f32 = bf16vec_to_float_vec(lw.up_proj);
      lw.down_proj_f32 = bf16vec_to_float_vec(lw.down_proj);
    }

    embed_tokens_f32 = bf16vec_to_float_vec(embed_tokens);
    lm_head_f32 = bf16vec_to_float_vec(lm_head);
    final_norm_f32 = bf16vec_to_float_vec(final_norm);

  } else {
    throw std::runtime_error(
        "TinyLlamaModel::initialize_weights called with neither GGUF nor "
        "SafeTensors loader.");
  }
  Logger::info("Finished initializing model weights.");
}

void TinyLlamaModel::initialize_gpu_and_rope() {
  Logger::info("Initializing GPU resources and RoPE...");
  int hs = config_.hidden_size;
  int is = config_.intermediate_size;
  int nhl = config_.num_hidden_layers;
  int vs = config_.vocab_size;
  int n_heads = config_.num_attention_heads;
  int n_kv_heads = config_.num_key_value_heads;

  if (hs <= 0)
    throw std::runtime_error(
        "Invalid model configuration: hidden_size must be positive. Check GGUF "
        "metadata ('llama.embedding_length').");
  if (vs <= 0)
    throw std::runtime_error(
        "Invalid model configuration: vocab_size must be positive. Check GGUF "
        "metadata ('general.vocab_size').");
  if (n_heads <= 0)
    throw std::runtime_error(
        "Invalid model configuration: num_attention_heads must be positive. "
        "Check GGUF metadata ('llama.head_count').");
  if (n_kv_heads <= 0)
    throw std::runtime_error(
        "Invalid model configuration: num_key_value_heads must be positive. "
        "Check GGUF metadata ('llama.head_count_kv').");
  if (hs % n_heads != 0)
    throw std::runtime_error(
        "Invalid model configuration: hidden_size must be divisible by "
        "num_attention_heads.");

  int kv_dim = (hs / n_heads) * n_kv_heads;
  int head_dim = hs / n_heads;

#ifdef HAS_CUDA
  Logger::info("Initializing CUDA resources...");
  cublasStatus_t cublas_status = cublasCreate(&cublas_handle_);
  if (cublas_status != CUBLAS_STATUS_SUCCESS) {
    Logger::error("cuBLAS handle creation failed: " +
                  std::to_string(cublas_status));
    throw std::runtime_error("Failed to initialize cuBLAS");
  }
  Logger::info("cuBLAS handle created successfully.");

  if (final_norm_f32.empty() && !final_norm.empty())
    final_norm_f32 = bf16vec_to_float_vec(final_norm);
  if (!final_norm_f32.empty()) {
    gpuErrchk(
        cudaMalloc(&final_norm_dev, final_norm_f32.size() * sizeof(float)));
    gpuErrchk(cudaMemcpy(final_norm_dev, final_norm_f32.data(),
                         final_norm_f32.size() * sizeof(float),
                         cudaMemcpyHostToDevice));
    Logger::info("Copied final_norm weights (FP32) to GPU.");
  } else {
    Logger::warning("Final norm weights (FP32) empty, skipping GPU copy.");
  }

  for (int i = 0; i < nhl; ++i) {
    if (layers[i].input_layernorm_f32.empty() &&
        !layers[i].input_layernorm.empty())
      layers[i].input_layernorm_f32 =
          bf16vec_to_float_vec(layers[i].input_layernorm);
    if (layers[i].post_attention_layernorm_f32.empty() &&
        !layers[i].post_attention_layernorm.empty())
      layers[i].post_attention_layernorm_f32 =
          bf16vec_to_float_vec(layers[i].post_attention_layernorm);
    if (!layers[i].input_layernorm_f32.empty()) {
      gpuErrchk(
          cudaMalloc(&layers[i].input_layernorm_dev,
                     layers[i].input_layernorm_f32.size() * sizeof(float)));
      gpuErrchk(cudaMemcpy(layers[i].input_layernorm_dev,
                           layers[i].input_layernorm_f32.data(),
                           layers[i].input_layernorm_f32.size() * sizeof(float),
                           cudaMemcpyHostToDevice));
    }
    if (!layers[i].post_attention_layernorm_f32.empty()) {
      gpuErrchk(cudaMalloc(
          &layers[i].post_attention_layernorm_dev,
          layers[i].post_attention_layernorm_f32.size() * sizeof(float)));
      gpuErrchk(cudaMemcpy(
          layers[i].post_attention_layernorm_dev,
          layers[i].post_attention_layernorm_f32.data(),
          layers[i].post_attention_layernorm_f32.size() * sizeof(float),
          cudaMemcpyHostToDevice));
    }
  }
  Logger::info("Copied all layer norm weights (FP32) to GPU.");

  if (!embed_tokens.empty()) {
    gpuErrchk(cudaMalloc(&token_embedding_table_dev_,
                         embed_tokens.size() * sizeof(uint16_t)));
    gpuErrchk(cudaMemcpy(token_embedding_table_dev_, embed_tokens.data(),
                         embed_tokens.size() * sizeof(uint16_t),
                         cudaMemcpyHostToDevice));
    Logger::info("Copied token_embedding_table (bf16) to GPU.");
  }
  if (!lm_head.empty()) {
    gpuErrchk(cudaMalloc(&lm_head_dev_, lm_head.size() * sizeof(uint16_t)));
    gpuErrchk(cudaMemcpy(lm_head_dev_, lm_head.data(),
                         lm_head.size() * sizeof(uint16_t),
                         cudaMemcpyHostToDevice));
    Logger::info("Copied lm_head (bf16) to GPU.");
  }

  bool has_bf16_layer_weights = !layers.empty() && !layers[0].q_proj.empty();
  if (has_bf16_layer_weights) {
    size_t layer_q_size = (size_t)hs * hs;
    size_t layer_k_size = (size_t)kv_dim * hs;
    size_t layer_v_size = (size_t)kv_dim * hs;
    size_t layer_o_size = (size_t)hs * hs;
    size_t layer_gate_size = (size_t)is * hs;
    size_t layer_up_size = (size_t)is * hs;
    size_t layer_down_size = (size_t)hs * is;

    std::vector<uint16_t> all_q_proj_host, all_k_proj_host, all_v_proj_host,
        all_o_proj_host;
    std::vector<uint16_t> all_gate_proj_host, all_up_proj_host,
        all_down_proj_host;

    all_q_proj_host.reserve(nhl * layer_q_size);
    all_k_proj_host.reserve(nhl * layer_k_size);

    for (int i = 0; i < nhl; ++i) {
      const auto& lw = layers[i];
      if (lw.q_proj.empty()) continue;
      all_q_proj_host.insert(all_q_proj_host.end(), lw.q_proj.begin(),
                             lw.q_proj.end());
      all_k_proj_host.insert(all_k_proj_host.end(), lw.k_proj.begin(),
                             lw.k_proj.end());
      all_v_proj_host.insert(all_v_proj_host.end(), lw.v_proj.begin(),
                             lw.v_proj.end());
      all_o_proj_host.insert(all_o_proj_host.end(), lw.o_proj.begin(),
                             lw.o_proj.end());
      all_gate_proj_host.insert(all_gate_proj_host.end(), lw.gate_proj.begin(),
                                lw.gate_proj.end());
      all_up_proj_host.insert(all_up_proj_host.end(), lw.up_proj.begin(),
                              lw.up_proj.end());
      all_down_proj_host.insert(all_down_proj_host.end(), lw.down_proj.begin(),
                                lw.down_proj.end());
    }
    Logger::info("Concatenated BF16 layer weights on host.");

    if (!all_q_proj_host.empty()) {
      gpuErrchk(
          cudaMalloc(&w_q_dev_, all_q_proj_host.size() * sizeof(uint16_t)));
      gpuErrchk(cudaMemcpy(w_q_dev_, all_q_proj_host.data(),
                           all_q_proj_host.size() * sizeof(uint16_t),
                           cudaMemcpyHostToDevice));
      gpuErrchk(
          cudaMalloc(&w_k_dev_, all_k_proj_host.size() * sizeof(uint16_t)));
      gpuErrchk(cudaMemcpy(w_k_dev_, all_k_proj_host.data(),
                           all_k_proj_host.size() * sizeof(uint16_t),
                           cudaMemcpyHostToDevice));
      gpuErrchk(
          cudaMalloc(&w_v_dev_, all_v_proj_host.size() * sizeof(uint16_t)));
      gpuErrchk(cudaMemcpy(w_v_dev_, all_v_proj_host.data(),
                           all_v_proj_host.size() * sizeof(uint16_t),
                           cudaMemcpyHostToDevice));
      gpuErrchk(
          cudaMalloc(&w_o_dev_, all_o_proj_host.size() * sizeof(uint16_t)));
      gpuErrchk(cudaMemcpy(w_o_dev_, all_o_proj_host.data(),
                           all_o_proj_host.size() * sizeof(uint16_t),
                           cudaMemcpyHostToDevice));
      gpuErrchk(cudaMalloc(&w_gate_dev_,
                           all_gate_proj_host.size() * sizeof(uint16_t)));
      gpuErrchk(cudaMemcpy(w_gate_dev_, all_gate_proj_host.data(),
                           all_gate_proj_host.size() * sizeof(uint16_t),
                           cudaMemcpyHostToDevice));
      gpuErrchk(
          cudaMalloc(&w_up_dev_, all_up_proj_host.size() * sizeof(uint16_t)));
      gpuErrchk(cudaMemcpy(w_up_dev_, all_up_proj_host.data(),
                           all_up_proj_host.size() * sizeof(uint16_t),
                           cudaMemcpyHostToDevice));
      gpuErrchk(cudaMalloc(&w_down_dev_,
                           all_down_proj_host.size() * sizeof(uint16_t)));
      gpuErrchk(cudaMemcpy(w_down_dev_, all_down_proj_host.data(),
                           all_down_proj_host.size() * sizeof(uint16_t),
                           cudaMemcpyHostToDevice));
      Logger::info("Copied concatenated layer weights (bf16) to GPU.");
    } else {
      Logger::info("No BF16 layer weights found to concatenate/copy to GPU.");
    }
  } else {
    Logger::info(
        "Skipping BF16 layer weight concatenation/copy (no BF16 weights "
        "found).");
  }

  Logger::info("Finished initializing CUDA weights.");

  Logger::info("Precomputing RoPE frequencies...");
#endif

  int max_seq_len = config_.max_position_embeddings;
  precomputed_freqs_cis_.resize((max_seq_len * head_dim) / 2);
  float theta = config_.rope_theta;
  for (int pos = 0; pos < max_seq_len; ++pos) {
    for (int i = 0; i < head_dim; i += 2) {
      float freq = std::pow(theta, -((float)i) / head_dim);
      float angle = pos * freq;
      float cos_val = std::cos(angle);
      float sin_val = std::sin(angle);
      precomputed_freqs_cis_[(pos * head_dim / 2) + (i / 2)] = {cos_val,
                                                                sin_val};
    }
  }
  Logger::info("Finished precomputing RoPE cos/sin frequencies.");

  Logger::info("RoPE Params Check: max_seq_len=" +
               std::to_string(config_.max_position_embeddings) +
               ", head_dim=" + std::to_string(head_dim));

#ifdef HAS_CUDA
  if (!precomputed_freqs_cis_.empty()) {
    size_t total_freq_elements = precomputed_freqs_cis_.size() * 2;
    gpuErrchk(
        cudaMalloc(&all_freqs_cis_dev, total_freq_elements * sizeof(float)));
    Logger::info("Allocated persistent RoPE frequency buffer on GPU: " +
                 std::to_string(total_freq_elements * sizeof(float) / 1024.0) +
                 " KB");
    std::vector<float> flat_host_freqs;
    flat_host_freqs.reserve(total_freq_elements);
    for (const auto& p : precomputed_freqs_cis_) {
      flat_host_freqs.push_back(p.first);
      flat_host_freqs.push_back(p.second);
    }
    gpuErrchk(cudaMemcpy(all_freqs_cis_dev, flat_host_freqs.data(),
                         total_freq_elements * sizeof(float),
                         cudaMemcpyHostToDevice));
    Logger::info(
        "Copied all precomputed RoPE frequencies to persistent GPU buffer.");
  } else {
    Logger::warning(
        "Host precomputed_freqs_cis_ is empty, skipping GPU RoPE buffer "
        "allocation.");
  }
  Logger::info("Finished initializing CUDA RoPE frequencies.");

  Logger::info("Allocating persistent GPU workspace buffers...");
  size_t hs_bytes = (size_t)hs * sizeof(float);
  size_t is_bytes = (size_t)is * sizeof(float);
  size_t vs_bytes = (size_t)vs * sizeof(float);
  size_t kv_head_bytes = (size_t)n_kv_heads * head_dim * sizeof(float);

  gpuErrchk(cudaMalloc(&x_dev_, hs_bytes));
  gpuErrchk(cudaMalloc(&x_norm_dev_, hs_bytes));
  gpuErrchk(cudaMalloc(&x_resid1_dev_, hs_bytes));
  gpuErrchk(cudaMalloc(&x_resid2_dev_, hs_bytes));
  gpuErrchk(cudaMalloc(&q_dev_, hs_bytes));
  gpuErrchk(cudaMalloc(&k_dev_, kv_head_bytes));
  gpuErrchk(cudaMalloc(&v_dev_, kv_head_bytes));
  gpuErrchk(cudaMalloc(&attn_out_dev_, hs_bytes));
  gpuErrchk(cudaMalloc(&attn_proj_dev_, hs_bytes));
  gpuErrchk(cudaMalloc(&gate_vec_dev_, is_bytes));
  gpuErrchk(cudaMalloc(&up_vec_dev_, is_bytes));
  gpuErrchk(cudaMalloc(&swiglu_vec_dev_, is_bytes));
  gpuErrchk(cudaMalloc(&mlp_down_dev_, hs_bytes));
  gpuErrchk(cudaMalloc(&logits_dev_, vs_bytes));
  Logger::info("Finished allocating persistent GPU workspace buffers.");
  Logger::info("Finished initializing GPU resources and RoPE.");

  if (!token_embedding_table_f32_dev_) {
    if (!embed_tokens_f32.empty()) {
      gpuErrchk(cudaMalloc(&token_embedding_table_f32_dev_,
                           embed_tokens_f32.size() * sizeof(float)));
      gpuErrchk(cudaMemcpy(
          token_embedding_table_f32_dev_, embed_tokens_f32.data(),
          embed_tokens_f32.size() * sizeof(float), cudaMemcpyHostToDevice));
      Logger::info("Copied token_embedding_table (fp32) to GPU.");
    } else if (!embed_tokens.empty()) {
      std::vector<float> embed_tokens_f32_tmp =
          bf16vec_to_float_vec(embed_tokens);
      gpuErrchk(cudaMalloc(&token_embedding_table_f32_dev_,
                           embed_tokens_f32_tmp.size() * sizeof(float)));
      gpuErrchk(cudaMemcpy(
          token_embedding_table_f32_dev_, embed_tokens_f32_tmp.data(),
          embed_tokens_f32_tmp.size() * sizeof(float), cudaMemcpyHostToDevice));
      Logger::info(
          "Converted and copied token_embedding_table (bf16->fp32) to GPU.");
      if (!token_embedding_table_dev_) {
        gpuErrchk(cudaMalloc(&token_embedding_table_dev_,
                             embed_tokens.size() * sizeof(uint16_t)));
        gpuErrchk(cudaMemcpy(token_embedding_table_dev_, embed_tokens.data(),
                             embed_tokens.size() * sizeof(uint16_t),
                             cudaMemcpyHostToDevice));
        Logger::info("Copied token_embedding_table (bf16) to GPU.");
      }
    }
  }

  if (!lm_head_f32_dev_) {
    if (!lm_head_f32.empty()) {
      gpuErrchk(
          cudaMalloc(&lm_head_f32_dev_, lm_head_f32.size() * sizeof(float)));
      gpuErrchk(cudaMemcpy(lm_head_f32_dev_, lm_head_f32.data(),
                           lm_head_f32.size() * sizeof(float),
                           cudaMemcpyHostToDevice));
      Logger::info("Copied lm_head (fp32) to GPU.");
    } else if (!lm_head.empty()) {
      std::vector<float> lm_head_f32_tmp = bf16vec_to_float_vec(lm_head);
      gpuErrchk(cudaMalloc(&lm_head_f32_dev_,
                           lm_head_f32_tmp.size() * sizeof(float)));
      gpuErrchk(cudaMemcpy(lm_head_f32_dev_, lm_head_f32_tmp.data(),
                           lm_head_f32_tmp.size() * sizeof(float),
                           cudaMemcpyHostToDevice));
      Logger::info("Converted and copied lm_head (bf16->fp32) to GPU.");
      if (!lm_head_dev_) {
        gpuErrchk(cudaMalloc(&lm_head_dev_, lm_head.size() * sizeof(uint16_t)));
        gpuErrchk(cudaMemcpy(lm_head_dev_, lm_head.data(),
                             lm_head.size() * sizeof(uint16_t),
                             cudaMemcpyHostToDevice));
        Logger::info("Copied lm_head (bf16) to GPU.");
      }
    }
  }

  if (!token_embedding_table_f32_dev_ && !embed_tokens_q8_0.empty()) {
    std::vector<float> embed_tokens_q8_0_f32(embed_tokens_q8_0.size() *
                                             GGML_QK8_0);
    for (size_t i = 0; i < embed_tokens_q8_0.size(); ++i) {
      dequantize_q8_0_block(&embed_tokens_q8_0[i],
                            &embed_tokens_q8_0_f32[i * GGML_QK8_0]);
    }
    gpuErrchk(cudaMalloc(&token_embedding_table_f32_dev_,
                         embed_tokens_q8_0_f32.size() * sizeof(float)));
    gpuErrchk(cudaMemcpy(
        token_embedding_table_f32_dev_, embed_tokens_q8_0_f32.data(),
        embed_tokens_q8_0_f32.size() * sizeof(float), cudaMemcpyHostToDevice));
    Logger::info(
        "Dequantized and copied token_embedding_table (Q8_0->fp32) to GPU.");
  }
  if (!token_embedding_table_f32_dev_ && !embed_tokens_q4k.empty()) {
    std::vector<float> embed_tokens_q4k_f32(embed_tokens_q4k.size() *
                                            GGML_QK_K);
    for (size_t i = 0; i < embed_tokens_q4k.size(); ++i) {
      dequantize_q4_k_m(&embed_tokens_q4k[i],
                        &embed_tokens_q4k_f32[i * GGML_QK_K], GGML_QK_K);
    }
    gpuErrchk(cudaMalloc(&token_embedding_table_f32_dev_,
                         embed_tokens_q4k_f32.size() * sizeof(float)));
    gpuErrchk(cudaMemcpy(
        token_embedding_table_f32_dev_, embed_tokens_q4k_f32.data(),
        embed_tokens_q4k_f32.size() * sizeof(float), cudaMemcpyHostToDevice));
    Logger::info(
        "Dequantized and copied token_embedding_table (Q4_K->fp32) to GPU.");
  }
  if (!lm_head_f32_dev_ && !lm_head_q8_0.empty()) {
    std::vector<float> lm_head_q8_0_f32(lm_head_q8_0.size() * GGML_QK8_0);
    for (size_t i = 0; i < lm_head_q8_0.size(); ++i) {
      dequantize_q8_0_block(&lm_head_q8_0[i],
                            &lm_head_q8_0_f32[i * GGML_QK8_0]);
    }
    gpuErrchk(
        cudaMalloc(&lm_head_f32_dev_, lm_head_q8_0_f32.size() * sizeof(float)));
    gpuErrchk(cudaMemcpy(lm_head_f32_dev_, lm_head_q8_0_f32.data(),
                         lm_head_q8_0_f32.size() * sizeof(float),
                         cudaMemcpyHostToDevice));
    Logger::info("Dequantized and copied lm_head (Q8_0->fp32) to GPU.");
  }
  if (!lm_head_f32_dev_ && !lm_head_q4k.empty()) {
    std::vector<float> lm_head_q4k_f32(lm_head_q4k.size() * GGML_QK_K);
    for (size_t i = 0; i < lm_head_q4k.size(); ++i) {
      dequantize_q4_k_m(&lm_head_q4k[i], &lm_head_q4k_f32[i * GGML_QK_K],
                        GGML_QK_K);
    }
    gpuErrchk(
        cudaMalloc(&lm_head_f32_dev_, lm_head_q4k_f32.size() * sizeof(float)));
    gpuErrchk(cudaMemcpy(lm_head_f32_dev_, lm_head_q4k_f32.data(),
                         lm_head_q4k_f32.size() * sizeof(float),
                         cudaMemcpyHostToDevice));
    Logger::info("Dequantized and copied lm_head (Q4_K->fp32) to GPU.");
  }

  size_t layer_q_size = (size_t)hs * hs;
  size_t layer_k_size = (size_t)kv_dim * hs;
  size_t layer_v_size = (size_t)kv_dim * hs;
  size_t layer_o_size = (size_t)hs * hs;
  size_t layer_gate_size = (size_t)is * hs;
  size_t layer_up_size = (size_t)is * hs;
  size_t layer_down_size = (size_t)hs * is;
  int num_layers = nhl;

  auto upload_layer_f32 = [&](const std::vector<std::vector<float>>& src,
                              float*& dev_ptr, size_t elem_size,
                              const std::string& weight_name) {
    std::vector<float> concat;
    for (const auto& v : src) concat.insert(concat.end(), v.begin(), v.end());
    if (!concat.empty()) {
      Logger::info("[UPLOAD_F32_LAMBDA] Processing: " + weight_name);
      gpuErrchk(cudaMalloc(&dev_ptr, concat.size() * sizeof(float)));
      gpuErrchk(cudaMemcpy(dev_ptr, concat.data(),
                           concat.size() * sizeof(float),
                           cudaMemcpyHostToDevice));
    }
  };
  auto upload_layer_bf16 = [&](const std::vector<std::vector<uint16_t>>& src,
                               float*& dev_ptr, size_t elem_size,
                               const std::string& weight_name) {
    std::vector<float> concat;
    for (const auto& v : src) {
      std::vector<float> tmp = bf16vec_to_float_vec(v);
      concat.insert(concat.end(), tmp.begin(), tmp.end());
    }
    if (!concat.empty()) {
      gpuErrchk(cudaMalloc(&dev_ptr, concat.size() * sizeof(float)));
      gpuErrchk(cudaMemcpy(dev_ptr, concat.data(),
                           concat.size() * sizeof(float),
                           cudaMemcpyHostToDevice));
    }
  };

  if (!w_q_f32_dev_) {
    std::vector<std::vector<float>> src_f32(num_layers);
    std::vector<std::vector<uint16_t>> src_bf16(num_layers);
    for (int i = 0; i < num_layers; ++i) {
      src_f32[i] = layers[i].q_proj_f32;
      src_bf16[i] = layers[i].q_proj;
    }
    if (!src_f32[0].empty())
      upload_layer_f32(src_f32, w_q_f32_dev_, layer_q_size, "W_Q_F32");
    else if (!src_bf16[0].empty())
      upload_layer_bf16(src_bf16, w_q_f32_dev_, layer_q_size, "W_Q_BF16");
  }

  if (!w_k_f32_dev_) {
    std::vector<std::vector<float>> src_f32(num_layers);
    std::vector<std::vector<uint16_t>> src_bf16(num_layers);
    for (int i = 0; i < num_layers; ++i) {
      src_f32[i] = layers[i].k_proj_f32;
      src_bf16[i] = layers[i].k_proj;
    }
    if (!src_f32[0].empty())
      upload_layer_f32(src_f32, w_k_f32_dev_, layer_k_size, "W_K_F32");
    else if (!src_bf16[0].empty())
      upload_layer_bf16(src_bf16, w_k_f32_dev_, layer_k_size, "W_K_BF16");
  }

  if (!w_v_f32_dev_) {
    std::vector<std::vector<float>> src_f32(num_layers);
    std::vector<std::vector<uint16_t>> src_bf16(num_layers);
    for (int i = 0; i < num_layers; ++i) {
      src_f32[i] = layers[i].v_proj_f32;
      src_bf16[i] = layers[i].v_proj;
    }
    if (!src_f32[0].empty())
      upload_layer_f32(src_f32, w_v_f32_dev_, layer_v_size, "W_V_F32");
    else if (!src_bf16[0].empty())
      upload_layer_bf16(src_bf16, w_v_f32_dev_, layer_v_size, "W_V_BF16");
  }

  if (!w_o_f32_dev_) {
    std::vector<std::vector<float>> src_f32(num_layers);
    std::vector<std::vector<uint16_t>> src_bf16(num_layers);
    for (int i = 0; i < num_layers; ++i) {
      src_f32[i] = layers[i].o_proj_f32;
      src_bf16[i] = layers[i].o_proj;
    }
    if (!src_f32[0].empty())
      upload_layer_f32(src_f32, w_o_f32_dev_, layer_o_size, "W_O_F32");
    else if (!src_bf16[0].empty())
      upload_layer_bf16(src_bf16, w_o_f32_dev_, layer_o_size, "W_O_BF16");
  }

  if (!w_gate_f32_dev_) {
    std::vector<std::vector<float>> src_f32(num_layers);
    std::vector<std::vector<uint16_t>> src_bf16(num_layers);
    for (int i = 0; i < num_layers; ++i) {
      src_f32[i] = layers[i].gate_proj_f32;
      src_bf16[i] = layers[i].gate_proj;
    }
    if (!src_f32[0].empty())
      upload_layer_f32(src_f32, w_gate_f32_dev_, layer_gate_size, "W_GATE_F32");
    else if (!src_bf16[0].empty())
      upload_layer_bf16(src_bf16, w_gate_f32_dev_, layer_gate_size,
                        "W_GATE_BF16");
  }

  if (!w_up_f32_dev_) {
    std::vector<std::vector<float>> src_f32(num_layers);
    std::vector<std::vector<uint16_t>> src_bf16(num_layers);
    for (int i = 0; i < num_layers; ++i) {
      src_f32[i] = layers[i].up_proj_f32;
      src_bf16[i] = layers[i].up_proj;
    }
    if (!src_f32[0].empty())
      upload_layer_f32(src_f32, w_up_f32_dev_, layer_up_size, "W_UP_F32");
    else if (!src_bf16[0].empty())
      upload_layer_bf16(src_bf16, w_up_f32_dev_, layer_up_size, "W_UP_BF16");
  }

  if (!w_down_f32_dev_) {
    std::vector<std::vector<float>> src_f32(num_layers);
    std::vector<std::vector<uint16_t>> src_bf16(num_layers);
    for (int i = 0; i < num_layers; ++i) {
      src_f32[i] = layers[i].down_proj_f32;
      src_bf16[i] = layers[i].down_proj;
    }
    if (!src_f32[0].empty())
      upload_layer_f32(src_f32, w_down_f32_dev_, layer_down_size, "W_DOWN_F32");
    else if (!src_bf16[0].empty())
      upload_layer_bf16(src_bf16, w_down_f32_dev_, layer_down_size,
                        "W_DOWN_BF16");
  }

  if (!w_q_f32_dev_) {
    bool all_have_q8_0 = true;
    for (int i = 0; i < num_layers; ++i) {
      if (layers[i].q_proj_q8_0.empty()) {
        all_have_q8_0 = false;
        break;
      }
    }
    if (all_have_q8_0) {
      std::vector<block_q8_0> concat;
      for (int i = 0; i < num_layers; ++i) {
        concat.insert(concat.end(), layers[i].q_proj_q8_0.begin(),
                      layers[i].q_proj_q8_0.end());
      }
      std::vector<float> deq(concat.size() * GGML_QK8_0);
      for (size_t i = 0; i < concat.size(); ++i) {
        dequantize_q8_0_block(&concat[i], &deq[i * GGML_QK8_0]);
      }
      gpuErrchk(cudaMalloc(&w_q_f32_dev_, deq.size() * sizeof(float)));
      gpuErrchk(cudaMemcpy(w_q_f32_dev_, deq.data(), deq.size() * sizeof(float),
                           cudaMemcpyHostToDevice));
      Logger::info("Dequantized and copied Q projection (Q8_0->fp32) to GPU.");
    }
  }

  if (!w_k_f32_dev_) {
    bool all_have_q8_0 = true;
    for (int i = 0; i < num_layers; ++i) {
      if (layers[i].k_proj_q8_0.empty()) {
        all_have_q8_0 = false;
        break;
      }
    }
    if (all_have_q8_0) {
      std::vector<block_q8_0> concat;
      for (int i = 0; i < num_layers; ++i) {
        concat.insert(concat.end(), layers[i].k_proj_q8_0.begin(),
                      layers[i].k_proj_q8_0.end());
      }
      std::vector<float> deq(concat.size() * GGML_QK8_0);
      for (size_t i = 0; i < concat.size(); ++i) {
        dequantize_q8_0_block(&concat[i], &deq[i * GGML_QK8_0]);
      }
      gpuErrchk(cudaMalloc(&w_k_f32_dev_, deq.size() * sizeof(float)));
      gpuErrchk(cudaMemcpy(w_k_f32_dev_, deq.data(), deq.size() * sizeof(float),
                           cudaMemcpyHostToDevice));
      Logger::info("Dequantized and copied K projection (Q8_0->fp32) to GPU.");
    }
  }

  if (!w_v_f32_dev_) {
    bool all_have_q8_0 = true;
    for (int i = 0; i < num_layers; ++i) {
      if (layers[i].v_proj_q8_0.empty()) {
        all_have_q8_0 = false;
        break;
      }
    }
    if (all_have_q8_0) {
      std::vector<block_q8_0> concat;
      for (int i = 0; i < num_layers; ++i) {
        concat.insert(concat.end(), layers[i].v_proj_q8_0.begin(),
                      layers[i].v_proj_q8_0.end());
      }
      std::vector<float> deq(concat.size() * GGML_QK8_0);
      for (size_t i = 0; i < concat.size(); ++i) {
        dequantize_q8_0_block(&concat[i], &deq[i * GGML_QK8_0]);
      }
      gpuErrchk(cudaMalloc(&w_v_f32_dev_, deq.size() * sizeof(float)));
      gpuErrchk(cudaMemcpy(w_v_f32_dev_, deq.data(), deq.size() * sizeof(float),
                           cudaMemcpyHostToDevice));
      Logger::info("Dequantized and copied V projection (Q8_0->fp32) to GPU.");
    }
  }

  if (!w_o_f32_dev_) {
    bool all_have_q8_0 = true;
    for (int i = 0; i < num_layers; ++i) {
      if (layers[i].o_proj_q8_0.empty()) {
        all_have_q8_0 = false;
        break;
      }
    }
    if (all_have_q8_0) {
      std::vector<block_q8_0> concat;
      for (int i = 0; i < num_layers; ++i) {
        concat.insert(concat.end(), layers[i].o_proj_q8_0.begin(),
                      layers[i].o_proj_q8_0.end());
      }
      std::vector<float> deq(concat.size() * GGML_QK8_0);
      for (size_t i = 0; i < concat.size(); ++i) {
        dequantize_q8_0_block(&concat[i], &deq[i * GGML_QK8_0]);
      }
      gpuErrchk(cudaMalloc(&w_o_f32_dev_, deq.size() * sizeof(float)));
      gpuErrchk(cudaMemcpy(w_o_f32_dev_, deq.data(), deq.size() * sizeof(float),
                           cudaMemcpyHostToDevice));
      Logger::info("Dequantized and copied O projection (Q8_0->fp32) to GPU.");
    }
  }

  if (!w_gate_f32_dev_) {
    bool all_have_q8_0 = true;
    for (int i = 0; i < num_layers; ++i) {
      if (layers[i].gate_proj_q8_0.empty()) {
        all_have_q8_0 = false;
        break;
      }
    }
    if (all_have_q8_0) {
      std::vector<block_q8_0> concat;
      for (int i = 0; i < num_layers; ++i) {
        concat.insert(concat.end(), layers[i].gate_proj_q8_0.begin(),
                      layers[i].gate_proj_q8_0.end());
      }
      std::vector<float> deq(concat.size() * GGML_QK8_0);
      for (size_t i = 0; i < concat.size(); ++i) {
        dequantize_q8_0_block(&concat[i], &deq[i * GGML_QK8_0]);
      }
      gpuErrchk(cudaMalloc(&w_gate_f32_dev_, deq.size() * sizeof(float)));
      gpuErrchk(cudaMemcpy(w_gate_f32_dev_, deq.data(),
                           deq.size() * sizeof(float), cudaMemcpyHostToDevice));
      Logger::info(
          "Dequantized and copied Gate projection (Q8_0->fp32) to GPU.");
    }
  }

  if (!w_up_f32_dev_) {
    bool all_have_q8_0 = true;
    for (int i = 0; i < num_layers; ++i) {
      if (layers[i].up_proj_q8_0.empty()) {
        all_have_q8_0 = false;
        break;
      }
    }
    if (all_have_q8_0) {
      std::vector<block_q8_0> concat;
      for (int i = 0; i < num_layers; ++i) {
        concat.insert(concat.end(), layers[i].up_proj_q8_0.begin(),
                      layers[i].up_proj_q8_0.end());
      }
      std::vector<float> deq(concat.size() * GGML_QK8_0);
      for (size_t i = 0; i < concat.size(); ++i) {
        dequantize_q8_0_block(&concat[i], &deq[i * GGML_QK8_0]);
      }
      gpuErrchk(cudaMalloc(&w_up_f32_dev_, deq.size() * sizeof(float)));
      gpuErrchk(cudaMemcpy(w_up_f32_dev_, deq.data(),
                           deq.size() * sizeof(float), cudaMemcpyHostToDevice));
      Logger::info("Dequantized and copied Up projection (Q8_0->fp32) to GPU.");
    }
  }

  if (!w_down_f32_dev_) {
    bool all_have_q8_0 = true;
    for (int i = 0; i < num_layers; ++i) {
      if (layers[i].down_proj_q8_0.empty()) {
        all_have_q8_0 = false;
        break;
      }
    }
    if (all_have_q8_0) {
      std::vector<block_q8_0> concat;
      for (int i = 0; i < num_layers; ++i) {
        concat.insert(concat.end(), layers[i].down_proj_q8_0.begin(),
                      layers[i].down_proj_q8_0.end());
      }
      std::vector<float> deq(concat.size() * GGML_QK8_0);
      for (size_t i = 0; i < concat.size(); ++i) {
        dequantize_q8_0_block(&concat[i], &deq[i * GGML_QK8_0]);
      }
      gpuErrchk(cudaMalloc(&w_down_f32_dev_, deq.size() * sizeof(float)));
      gpuErrchk(cudaMemcpy(w_down_f32_dev_, deq.data(),
                           deq.size() * sizeof(float), cudaMemcpyHostToDevice));
      Logger::info(
          "Dequantized and copied Down projection (Q8_0->fp32) to GPU.");
    }
  }
#endif
}

TinyLlamaModel::TinyLlamaModel(const ModelConfig& config,
                               const SafeTensorsLoader& loader)
    : config_(config) {
  Logger::info("Constructing TinyLlamaModel from SafeTensorsLoader.");
  initialize_weights(&loader, nullptr);
#ifdef HAS_CUDA
  initialize_gpu_and_rope();
#endif
  Logger::info("TinyLlamaModel construction from SafeTensorsLoader complete.");
}

TinyLlamaModel::TinyLlamaModel(const ModelConfig& config_in,
                               const std::string& weights_path)

{
  Logger::info("Constructing TinyLlamaModel from path: " + weights_path);
  bool is_gguf_detected_runtime = false;
  if (weights_path.size() >= 5 &&
      weights_path.substr(weights_path.size() - 5) == ".gguf") {
    is_gguf_detected_runtime = true;
  } else {
    std::ifstream file(weights_path, std::ios::binary);
    if (file.is_open()) {
      uint32_t magic = 0;
      file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
      if (magic == GGUF_MAGIC) {
        is_gguf_detected_runtime = true;
      }
      file.close();
    } else {
      Logger::warning("Could not open weights file to check magic number: " +
                      weights_path);
    }
  }

  if (is_gguf_detected_runtime) {
    Logger::info("Detected GGUF file. Loading metadata and mapping weights...");
    gguf_data_ = std::make_unique<GGUFData>(load_gguf_meta(weights_path));
    config_ = parse_model_config_from_gguf(*gguf_data_);
    config_.is_gguf_file_loaded = true;

    if (config_in.max_position_embeddings > 0 &&
        config_in.max_position_embeddings != config_.max_position_embeddings) {
      Logger::warning("Overriding GGUF max_position_embeddings (" +
                      std::to_string(config_.max_position_embeddings) +
                      ") with value from main: " +
                      std::to_string(config_in.max_position_embeddings));
      config_.max_position_embeddings = config_in.max_position_embeddings;
    }

    initialize_weights(nullptr, gguf_data_.get());
    Logger::info("GGUF weights mapped.");

  } else {
    config_ = config_in;
    config_.is_gguf_file_loaded = false;
    Logger::info(
        "Detected non-GGUF file. Using provided config and loading with "
        "SafeTensors loader...");
    SafeTensorsLoader loader(weights_path);
    initialize_weights(&loader, nullptr);
    Logger::info("SafeTensors weights loaded.");
  }

  initialize_gpu_and_rope();
  Logger::info("TinyLlamaModel construction from path complete.");
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
  Logger::info("Freed persistent GPU workspace buffers.");

  Logger::info("Finished freeing TinyLlamaModel CUDA weight memory.");
#endif
}

std::vector<float> TinyLlamaModel::lookup_embedding(int token_id) {
  int hs = config_.hidden_size;
  int vs = config_.vocab_size;
  bool log_initial = (token_id == config_.bos_token_id);

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
    if (log_initial) {
      log_vector_summary("[CPU_EMBED Q4_K] Output Embedding (Token " +
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
    if (log_initial) {
      log_vector_summary("[CPU_EMBED Q8_0] Output Embedding (Token " +
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
    if (log_initial) {
      log_vector_summary("[CPU_EMBED F32] Output Embedding (Token " +
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
    if (log_initial) {
      log_vector_summary("[CPU_EMBED BF16] Output Embedding (Token " +
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

std::vector<float> TinyLlamaModel::forward(std::vector<float>& input,
                                           int n_tokens, KVCache* kv_cache,
                                           const std::vector<int>* input_ids) {
  bool log_this_step = (n_tokens == 0);
  bool log_first_gen_step = (n_tokens == 23);

  int hs = config_.hidden_size;
  int is = config_.intermediate_size;
  int nhl = config_.num_hidden_layers;
  int vs = config_.vocab_size;
  int n_heads = config_.num_attention_heads;
  int n_kv_heads = config_.num_key_value_heads;
  int head_dim = hs / n_heads;
  int max_seq_len = config_.max_position_embeddings;
  float eps = config_.rms_norm_eps;

  if (log_this_step) Logger::info("[CPU_FWD STEP 0] Entered forward function.");
  if (log_first_gen_step)
    Logger::info("[CPU_FWD STEP 23] Entered forward function.");

  if (log_this_step || log_first_gen_step) {
    Logger::info("[CPU_FWD] forward called. n_tokens=" +
                 std::to_string(n_tokens));
    log_vector_summary(
        "[CPU_FWD] Input input (n_tokens=" + std::to_string(n_tokens) + ")",
        input);
    log_vector_summary_with_tail("[CPU_FWD] Initial Embedding Vector (input)",
                                 input, 10, 10);
  }

  if (n_tokens >= max_seq_len) {
    Logger::error("Position index exceeds max_position_embeddings");
    return std::vector<float>(vs, 0.0f);
  }
  if (!kv_cache) {
    Logger::error("KVCache is required for token-by-token forward pass");
    return std::vector<float>(vs, 0.0f);
  }
  if (input.size() != hs) {
    Logger::error("Input vector input has incorrect size. Expected " +
                  std::to_string(hs) + ", got " + std::to_string(input.size()));
    return std::vector<float>(vs, 0.0f);
  }

  std::vector<float> x_norm_vec1(hs);
  std::vector<float> q_vec(hs);
  std::vector<float> k_vec(n_kv_heads * head_dim);
  std::vector<float> v_vec(n_kv_heads * head_dim);
  std::vector<float> attn_out_vec(hs);
  std::vector<float> attn_proj_vec(hs);
  std::vector<float> x_norm_vec2(hs);
  std::vector<float> gate_vec(is);
  std::vector<float> up_vec(is);
  std::vector<float> silu_out_vec(is);
  std::vector<float> swiglu_result_vec(is);
  std::vector<float> mlp_out_vec(hs);

  if (log_this_step || log_first_gen_step)
    Logger::info("[CPU_FWD STEP " + std::to_string(n_tokens) +
                 "] Entering layer loop.");

  for (int l = 0; l < nhl; ++l) {
    bool log_this_layer = (log_this_step || log_first_gen_step) && (l == 0);
    if (log_this_layer)
      Logger::info("[CPU_FWD STEP " + std::to_string(n_tokens) +
                   "] Start of Layer " + std::to_string(l) + ".");
    if (log_this_layer)
      Logger::info("[CPU_FWD] ------ START Layer " + std::to_string(l) +
                   " (pos=" + std::to_string(n_tokens) + ") ------");

    const auto& lw = layers[l];

    std::vector<float> x_resid1_vec = input;

    const std::vector<float>& w_norm1_vec =
        lw.input_layernorm_f32.empty()
            ? bf16vec_to_float_vec(lw.input_layernorm)
            : lw.input_layernorm_f32;
    rmsnorm_vector_cpu(input, w_norm1_vec, x_norm_vec1, eps);
    if (log_this_layer)
      log_vector_summary(
          "[CPU_FWD L" + std::to_string(l) +
              "] RMSNorm1 Out (n_tokens=" + std::to_string(n_tokens) + ")",
          x_norm_vec1);
    if (log_this_layer)
      Logger::info("[CPU_FWD STEP " + std::to_string(n_tokens) +
                   "] Before QKV projections in Layer " + std::to_string(l) +
                   ".");

    if (!lw.q_proj_q8_0.empty()) {
      if (log_this_layer)
        Logger::info("[CPU_FWD L" + std::to_string(l) +
                     "] MatVec: Q_Proj (Q8_0)");
      std::vector<float> q_proj_f32(hs * hs);
      size_t num_blocks = (size_t)(hs * hs) / GGML_QK8_0;
      if (lw.q_proj_q8_0.size() != num_blocks) {
        Logger::error("Q_Proj Q8_0 block count mismatch. Expected " +
                      std::to_string(num_blocks) + " got " +
                      std::to_string(lw.q_proj_q8_0.size()));
        q_vec.assign(hs, 0.0f);
      } else {
        for (size_t i = 0; i < lw.q_proj_q8_0.size(); ++i)
          dequantize_q8_0_block(&lw.q_proj_q8_0[i],
                                &q_proj_f32[i * GGML_QK8_0]);
        matvec_f32_f32_vector_cpu(q_proj_f32, x_norm_vec1, q_vec, hs, hs);
      }
    } else if (!lw.q_proj_q6k.empty()) {
      if (log_this_layer)
        Logger::info("[CPU_FWD L" + std::to_string(l) +
                     "] MatVec: Q_Proj (Q6_K)");
      std::vector<float> q_proj_f32(hs * hs);
      for (size_t i = 0; i < lw.q_proj_q6k.size(); ++i)
        dequantize_q6_k(&lw.q_proj_q6k[i], &q_proj_f32[i * GGML_QK_K],
                        GGML_QK_K);
      matvec_f32_f32_vector_cpu(q_proj_f32, x_norm_vec1, q_vec, hs, hs);
    } else if (!lw.q_proj_q4k.empty()) {
      if (log_this_layer)
        Logger::info("[CPU_FWD L" + std::to_string(l) +
                     "] MatVec: Q_Proj (Q4_K)");
      std::vector<float> q_proj_f32(hs * hs);
      for (size_t i = 0; i < lw.q_proj_q4k.size(); ++i)
        dequantize_q4_k_m(&lw.q_proj_q4k[i], &q_proj_f32[i * GGML_QK_K],
                          GGML_QK_K);
      matvec_f32_f32_vector_cpu(q_proj_f32, x_norm_vec1, q_vec, hs, hs);
    } else if (!lw.q_proj_f32.empty()) {
      if (log_this_layer)
        Logger::info("[CPU_FWD L" + std::to_string(l) +
                     "] MatVec: Q_Proj (F32)");
      matvec_f32_f32_vector_cpu(lw.q_proj_f32, x_norm_vec1, q_vec, hs, hs);
    } else if (!lw.q_proj.empty()) {
      if (log_this_layer)
        Logger::warning("[CPU_FWD L" + std::to_string(l) +
                        "] MatVec: Q_Proj (BF16 Fallback)");
      matvec_bf16_f32_vector_cpu(lw.q_proj, x_norm_vec1, q_vec, hs, hs);
    } else {
      throw std::runtime_error("Layer " + std::to_string(l) +
                               ": No valid Q projection weights found!");
    }
    if (log_this_layer) log_vector_summary("  Output (q_vec)", q_vec);

    size_t k_proj_dim0 = n_kv_heads * head_dim;
    if (!lw.k_proj_q8_0.empty()) {
      if (log_this_layer)
        Logger::info("[CPU_FWD L" + std::to_string(l) +
                     "] MatVec: K_Proj (Q8_0)");
      std::vector<float> k_proj_f32(k_proj_dim0 * hs);
      size_t num_blocks = (k_proj_dim0 * hs) / GGML_QK8_0;
      if (lw.k_proj_q8_0.size() != num_blocks) {
        Logger::error("K_Proj Q8_0 block count mismatch");
        k_vec.assign(k_proj_dim0, 0.0f);
      } else {
        for (size_t i = 0; i < lw.k_proj_q8_0.size(); ++i)
          dequantize_q8_0_block(&lw.k_proj_q8_0[i],
                                &k_proj_f32[i * GGML_QK8_0]);
        matvec_f32_f32_vector_cpu(k_proj_f32, x_norm_vec1, k_vec, k_proj_dim0,
                                  hs);
      }
    } else if (!lw.k_proj_q6k.empty()) {
      if (log_this_layer)
        Logger::info("[CPU_FWD L" + std::to_string(l) +
                     "] MatVec: K_Proj (Q6_K)");
      std::vector<float> k_proj_f32(k_proj_dim0 * hs);
      for (size_t i = 0; i < lw.k_proj_q6k.size(); ++i)
        dequantize_q6_k(&lw.k_proj_q6k[i], &k_proj_f32[i * GGML_QK_K],
                        GGML_QK_K);
      matvec_f32_f32_vector_cpu(k_proj_f32, x_norm_vec1, k_vec, k_proj_dim0,
                                hs);
    } else if (!lw.k_proj_q4k.empty()) {
      if (log_this_layer)
        Logger::info("[CPU_FWD L" + std::to_string(l) +
                     "] MatVec: K_Proj (Q4_K)");
      std::vector<float> k_proj_f32(k_proj_dim0 * hs);
      for (size_t i = 0; i < lw.k_proj_q4k.size(); ++i)
        dequantize_q4_k_m(&lw.k_proj_q4k[i], &k_proj_f32[i * GGML_QK_K],
                          GGML_QK_K);
      matvec_f32_f32_vector_cpu(k_proj_f32, x_norm_vec1, k_vec, k_proj_dim0,
                                hs);
    } else if (!lw.k_proj_f32.empty()) {
      if (log_this_layer)
        Logger::info("[CPU_FWD L" + std::to_string(l) +
                     "] MatVec: K_Proj (F32)");
      matvec_f32_f32_vector_cpu(lw.k_proj_f32, x_norm_vec1, k_vec, k_proj_dim0,
                                hs);
    } else if (!lw.k_proj.empty()) {
      if (log_this_layer)
        Logger::warning("[CPU_FWD L" + std::to_string(l) +
                        "] MatVec: K_Proj (BF16 Fallback)");
      matvec_bf16_f32_vector_cpu(lw.k_proj, x_norm_vec1, k_vec, k_proj_dim0,
                                 hs);
    } else {
      throw std::runtime_error("Layer " + std::to_string(l) +
                               ": No valid K projection weights found!");
    }
    if (log_this_layer)
      log_vector_summary("  [K-Proj L" + std::to_string(l) + "] Output (k_vec)",
                         k_vec);

    size_t v_proj_dim0 = n_kv_heads * head_dim;
    if (log_this_layer)
      log_vector_summary(
          "  [V-Proj L" + std::to_string(l) + "] Input (x_norm_vec1)",
          x_norm_vec1);
    if (!lw.v_proj_q8_0.empty()) {
      if (log_this_layer)
        Logger::info("[CPU_FWD L" + std::to_string(l) +
                     "] MatVec: V_Proj (Q8_0)");
      std::vector<float> v_proj_f32(v_proj_dim0 * hs);
      size_t num_blocks = (v_proj_dim0 * hs) / GGML_QK8_0;
      if (lw.v_proj_q8_0.size() != num_blocks) {
        Logger::error("V_Proj Q8_0 block count mismatch");
        v_vec.assign(v_proj_dim0, 0.0f);
      } else {
        for (size_t i = 0; i < lw.v_proj_q8_0.size(); ++i)
          dequantize_q8_0_block(&lw.v_proj_q8_0[i],
                                &v_proj_f32[i * GGML_QK8_0]);
        matvec_f32_f32_vector_cpu(v_proj_f32, x_norm_vec1, v_vec, v_proj_dim0,
                                  hs);
      }
    } else if (!lw.v_proj_q6k.empty()) {
      if (log_this_layer)
        Logger::info("[CPU_FWD L" + std::to_string(l) +
                     "] MatVec: V_Proj (Q6_K)");
      std::vector<float> v_proj_f32(v_proj_dim0 * hs);
      for (size_t i = 0; i < lw.v_proj_q6k.size(); ++i)
        dequantize_q6_k(&lw.v_proj_q6k[i], &v_proj_f32[i * GGML_QK_K],
                        GGML_QK_K);
      matvec_f32_f32_vector_cpu(v_proj_f32, x_norm_vec1, v_vec, v_proj_dim0,
                                hs);
    } else if (!lw.v_proj_q4k.empty()) {
      if (log_this_layer)
        Logger::info("[CPU_FWD L" + std::to_string(l) +
                     "] MatVec: V_Proj (Q4_K)");
      std::vector<float> v_proj_f32(v_proj_dim0 * hs);
      for (size_t i = 0; i < lw.v_proj_q4k.size(); ++i)
        dequantize_q4_k_m(&lw.v_proj_q4k[i], &v_proj_f32[i * GGML_QK_K],
                          GGML_QK_K);
      matvec_f32_f32_vector_cpu(v_proj_f32, x_norm_vec1, v_vec, v_proj_dim0,
                                hs);
    } else if (!lw.v_proj_f32.empty()) {
      if (log_this_layer)
        Logger::info("[CPU_FWD L" + std::to_string(l) +
                     "] MatVec: V_Proj (F32)");
      matvec_f32_f32_vector_cpu(lw.v_proj_f32, x_norm_vec1, v_vec, v_proj_dim0,
                                hs);
    } else if (!lw.v_proj.empty()) {
      if (log_this_layer)
        Logger::warning("[CPU_FWD L" + std::to_string(l) +
                        "] MatVec: V_Proj (BF16 Fallback)");
      matvec_bf16_f32_vector_cpu(lw.v_proj, x_norm_vec1, v_vec, v_proj_dim0,
                                 hs);
    } else {
      throw std::runtime_error("Layer " + std::to_string(l) +
                               ": No valid V projection weights found!");
    }
    if (log_this_layer)
      log_vector_summary("  [V-Proj L" + std::to_string(l) + "] Output (v_vec)",
                         v_vec);

    size_t freqs_offset = (size_t)n_tokens * head_dim / 2;
    if (freqs_offset + head_dim / 2 > precomputed_freqs_cis_.size()) {
      throw std::runtime_error("RoPE freqs_cis access out of bounds. pos: " +
                               std::to_string(n_tokens) +
                               " head_dim: " + std::to_string(head_dim));
    }
    std::vector<std::pair<float, float>> current_freqs_cis(
        precomputed_freqs_cis_.begin() + freqs_offset,
        precomputed_freqs_cis_.begin() + freqs_offset + head_dim / 2);
    apply_rope_vector(q_vec, n_heads, head_dim, n_tokens, current_freqs_cis);
    apply_rope_vector(k_vec, n_kv_heads, head_dim, n_tokens, current_freqs_cis);
    if (log_this_layer)
      log_vector_summary(
          "  [K-Proj L" + std::to_string(l) + "] Output After RoPE (k_vec)",
          k_vec, head_dim);

    float* k_current_ptr = k_vec.data();
    float* v_current_ptr = v_vec.data();
    for (int kvh = 0; kvh < n_kv_heads; ++kvh) {
      size_t current_k_offset = (size_t)kvh * head_dim;
      size_t current_v_offset = (size_t)kvh * head_dim;
      size_t write_offset =
          (size_t)n_tokens * n_kv_heads * head_dim + kvh * head_dim;
      if (write_offset + head_dim <= kv_cache->layers[l].k.size()) {
        std::memcpy(&kv_cache->layers[l].k[write_offset],
                    k_current_ptr + current_k_offset, head_dim * sizeof(float));
        std::memcpy(&kv_cache->layers[l].v[write_offset],
                    v_current_ptr + current_v_offset, head_dim * sizeof(float));
      } else {
        Logger::error("KVCache write out of bounds: layer=" +
                      std::to_string(l) + ", pos=" + std::to_string(n_tokens) +
                      ", kv_head=" + std::to_string(kvh));
      }
    }
    kv_cache->seq_len = n_tokens + 1;

    std::fill(attn_out_vec.begin(), attn_out_vec.end(), 0.0f);
    int current_seq_len = n_tokens + 1;
    float scale = 1.0f / SAFE_SQRT(static_cast<float>(head_dim));
    for (int h = 0; h < n_heads; ++h) {
      std::vector<float> q_head_rope_vec(q_vec.begin() + h * head_dim,
                                         q_vec.begin() + (h + 1) * head_dim);
      int kv_head_idx = h / (n_heads / n_kv_heads);
      std::vector<float> k_cache_head_vec(current_seq_len * head_dim);
      std::vector<float> v_cache_head_vec(current_seq_len * head_dim);
      bool log_attn_details = log_this_layer && (h == 0);
      if (log_attn_details)
        log_vector_summary("  [Attn L0H0] Q Head Rope Vec", q_head_rope_vec);
      for (int j = 0; j < current_seq_len; ++j) {
        size_t cache_pos_offset =
            (size_t)j * n_kv_heads * head_dim + kv_head_idx * head_dim;
        if (cache_pos_offset + head_dim <= kv_cache->layers[l].k.size()) {
          std::memcpy(k_cache_head_vec.data() + j * head_dim,
                      &kv_cache->layers[l].k[cache_pos_offset],
                      head_dim * sizeof(float));
          std::memcpy(v_cache_head_vec.data() + j * head_dim,
                      &kv_cache->layers[l].v[cache_pos_offset],
                      head_dim * sizeof(float));
        } else {
          std::fill(k_cache_head_vec.begin() + j * head_dim,
                    k_cache_head_vec.begin() + (j + 1) * head_dim, 0.0f);
          std::fill(v_cache_head_vec.begin() + j * head_dim,
                    v_cache_head_vec.begin() + (j + 1) * head_dim, 0.0f);
          Logger::error(
              "Attention K/V access out of bounds: cache_pos_offset=" +
              std::to_string(cache_pos_offset));
        }
      }
      if (log_attn_details) {
        log_vector_summary_with_tail("  [Attn L0H0] K Cache Head Vec (SeqLen=" +
                                         std::to_string(current_seq_len) + ")",
                                     k_cache_head_vec, 5, head_dim);
        log_vector_summary("  [Attn L0H0] V Cache Head Vec (SeqLen=" +
                               std::to_string(current_seq_len) + ")",
                           v_cache_head_vec);
      }
      std::vector<float> scores_vec(current_seq_len);
      calculate_attention_scores(q_head_rope_vec, k_cache_head_vec, scores_vec,
                                 current_seq_len, head_dim, scale);
      if (log_attn_details)
        log_vector_summary("  [Attn L0H0] Scores Vec (Before Softmax)",
                           scores_vec);
      std::vector<float> probs_vec(current_seq_len);
      softmax_vector_cpu(scores_vec, probs_vec);
      if (log_attn_details)
        log_vector_summary("  [Attn L0H0] Probs Vec (After Softmax)",
                           probs_vec);
      std::vector<float> head_attn_out_vec(head_dim);
      weighted_sum_probs_v(probs_vec, v_cache_head_vec, head_attn_out_vec,
                           current_seq_len, head_dim);
      if (log_attn_details)
        log_vector_summary("  [Attn L0H0] Head Attn Out Vec",
                           head_attn_out_vec);
      size_t out_offset = h * head_dim;
      for (int i_val = 0; i_val < head_dim; ++i_val)
        attn_out_vec[out_offset + i_val] += head_attn_out_vec[i_val];
    }

    if (!lw.o_proj_q8_0.empty()) {
      if (log_this_layer)
        Logger::info("[CPU_FWD L" + std::to_string(l) +
                     "] MatVec: O_Proj (Q8_0)");
      std::vector<float> o_proj_f32(hs * hs);
      size_t num_q8_blocks_o = (hs * hs) / GGML_QK8_0;
      if (lw.o_proj_q8_0.size() != num_q8_blocks_o) {
        Logger::error("O_Proj Q8_0 block count mismatch. Expected " +
                      std::to_string(num_q8_blocks_o) + " got " +
                      std::to_string(lw.o_proj_q8_0.size()) +
                      " for tensor 'O_PROJ'");
      } else {
        for (size_t i = 0; i < lw.o_proj_q8_0.size(); ++i) {
          dequantize_q8_0_block(&lw.o_proj_q8_0[i],
                                &o_proj_f32[i * GGML_QK8_0]);
        }
        matvec_f32_f32_vector_cpu(o_proj_f32, attn_out_vec, attn_proj_vec, hs,
                                  hs);
      }
    } else if (!lw.o_proj_q6k.empty()) {
      if (log_this_layer)
        Logger::info("[CPU_FWD L" + std::to_string(l) +
                     "] MatVec: O_Proj (Q6_K)");
      std::vector<float> o_proj_f32(hs * hs);
      for (size_t i = 0; i < lw.o_proj_q6k.size(); ++i)
        dequantize_q6_k(&lw.o_proj_q6k[i], &o_proj_f32[i * GGML_QK_K],
                        GGML_QK_K);
      std::vector<float> attn_out_q8k_f32(hs);
      std::vector<block_q8_K> attn_out_q8k =
          quantize_fp32_to_q8_K(attn_out_vec);
      dequantize_q8_k(attn_out_q8k, attn_out_q8k_f32, hs, false);
      matvec_f32_f32_vector_cpu(o_proj_f32, attn_out_q8k_f32, attn_proj_vec, hs,
                                hs);
    } else if (!lw.o_proj_q4k.empty()) {
      if (log_this_layer)
        Logger::info("[CPU_FWD L" + std::to_string(l) +
                     "] MatVec: O_Proj (Q4_K)");
      std::vector<float> o_proj_f32(hs * hs);
      for (size_t i = 0; i < lw.o_proj_q4k.size(); ++i)
        dequantize_q4_k_m(&lw.o_proj_q4k[i], &o_proj_f32[i * GGML_QK_K],
                          GGML_QK_K);
      std::vector<float> attn_out_q8k_f32(hs);
      std::vector<block_q8_K> attn_out_q8k =
          quantize_fp32_to_q8_K(attn_out_vec);
      dequantize_q8_k(attn_out_q8k, attn_out_q8k_f32, hs, false);
      matvec_f32_f32_vector_cpu(o_proj_f32, attn_out_q8k_f32, attn_proj_vec, hs,
                                hs);
    } else if (!lw.o_proj_f32.empty()) {
      if (log_this_layer)
        Logger::info("[CPU_FWD L" + std::to_string(l) +
                     "] MatVec: O_Proj (F32)");
      matvec_f32_f32_vector_cpu(lw.o_proj_f32, attn_out_vec, attn_proj_vec, hs,
                                hs);
    } else if (!lw.o_proj.empty()) {
      if (log_this_layer)
        Logger::warning("[CPU_FWD L" + std::to_string(l) +
                        "] MatVec: O_Proj (BF16 Fallback)");
      matvec_bf16_f32_vector_cpu(lw.o_proj, attn_out_vec, attn_proj_vec, hs,
                                 hs);
    } else {
      throw std::runtime_error("Layer " + std::to_string(l) +
                               ": No valid O projection weights found!");
    }

    if (l == 0) {
      log_vector_summary("Layer 0 Attn Proj Out (attn_proj_vec)",
                         attn_proj_vec);
    }

#pragma omp parallel for
    for (int64_t i = 0; i < static_cast<int64_t>(hs); ++i) {
      input[i] = x_resid1_vec[i] + attn_proj_vec[i];
    }

    if (l == 0) {
      log_vector_summary("Layer 0 After Attn Residual (input)", input);
    }

    std::vector<float> x_resid2_vec = input;

    const std::vector<float>& w_norm2_vec =
        lw.post_attention_layernorm_f32.empty()
            ? bf16vec_to_float_vec(lw.post_attention_layernorm)
            : lw.post_attention_layernorm_f32;
    rmsnorm_vector_cpu(input, w_norm2_vec, x_norm_vec2, eps);

    if (!lw.gate_proj_q8_0.empty()) {
      if (log_this_layer)
        Logger::info("[CPU_FWD L" + std::to_string(l) +
                     "] MatVec: Gate_Proj (Q8_0)");
      std::vector<float> gate_proj_f32(is * hs);
      size_t num_q8_blocks_gate = (is * hs) / GGML_QK8_0;
      if (lw.gate_proj_q8_0.size() != num_q8_blocks_gate) {
        Logger::error("Gate_Proj Q8_0 block count mismatch. Expected " +
                      std::to_string(num_q8_blocks_gate) + " got " +
                      std::to_string(lw.gate_proj_q8_0.size()) +
                      " for tensor 'GATE_PROJ'");
      } else {
        for (size_t i = 0; i < lw.gate_proj_q8_0.size(); ++i) {
          dequantize_q8_0_block(&lw.gate_proj_q8_0[i],
                                &gate_proj_f32[i * GGML_QK8_0]);
        }
        matvec_f32_f32_vector_cpu(gate_proj_f32, x_norm_vec2, gate_vec, is, hs);
      }
    } else if (!lw.gate_proj_q6k.empty()) {
      if (log_this_layer)
        Logger::info("[CPU_FWD L" + std::to_string(l) +
                     "] MatVec: Gate_Proj (Q6_K)");

      std::vector<float> gate_proj_f32(is * hs);
      for (size_t i = 0; i < lw.gate_proj_q6k.size(); ++i)
        dequantize_q6_k(&lw.gate_proj_q6k[i], &gate_proj_f32[i * GGML_QK_K],
                        GGML_QK_K);
      if (log_this_layer)
        log_vector_summary("Layer 0 Gate Proj Weights (FP32)", gate_proj_f32,
                           10);
      std::vector<float> x_norm2_q8k_f32(hs);
      std::vector<block_q8_K> x_norm2_q8k = quantize_fp32_to_q8_K(x_norm_vec2);
      dequantize_q8_k(x_norm2_q8k, x_norm2_q8k_f32, hs, false);
      matvec_f32_f32_vector_cpu(gate_proj_f32, x_norm2_q8k_f32, gate_vec, is,
                                hs);
    } else if (!lw.gate_proj_f32.empty()) {
      if (log_this_layer)
        Logger::info("[CPU_FWD L" + std::to_string(l) +
                     "] MatVec: Gate_Proj (F32)");
      matvec_f32_f32_vector_cpu(lw.gate_proj_f32, x_norm_vec2, gate_vec, is,
                                hs);
    } else if (!lw.gate_proj.empty()) {
      if (log_this_layer)
        Logger::warning("[CPU_FWD L" + std::to_string(l) +
                        "] MatVec: Gate_Proj (BF16 Fallback)");
      matvec_bf16_f32_vector_cpu(lw.gate_proj, x_norm_vec2, gate_vec, is, hs);
    } else {
      throw std::runtime_error("Layer " + std::to_string(l) +
                               ": No valid Gate projection weights found!");
    }
    if (log_this_layer) log_vector_summary("Layer 0 Gate Vec", gate_vec);

    if (!lw.up_proj_q8_0.empty()) {
      if (log_this_layer)
        Logger::info("[CPU_FWD L" + std::to_string(l) +
                     "] MatVec: Up_Proj (Q8_0)");
      std::vector<float> up_proj_f32(is * hs);
      size_t num_q8_blocks_up = (is * hs) / GGML_QK8_0;
      if (lw.up_proj_q8_0.size() != num_q8_blocks_up) {
        Logger::error("Up_Proj Q8_0 block count mismatch. Expected " +
                      std::to_string(num_q8_blocks_up) + " got " +
                      std::to_string(lw.up_proj_q8_0.size()) +
                      " for tensor 'UP_PROJ'");
      } else {
        for (size_t i = 0; i < lw.up_proj_q8_0.size(); ++i) {
          dequantize_q8_0_block(&lw.up_proj_q8_0[i],
                                &up_proj_f32[i * GGML_QK8_0]);
        }
        matvec_f32_f32_vector_cpu(up_proj_f32, x_norm_vec2, up_vec, is, hs);
      }
    } else if (!lw.up_proj_q6k.empty()) {
      if (log_this_layer)
        Logger::info("[CPU_FWD L" + std::to_string(l) +
                     "] MatVec: Up_Proj (Q6_K)");

      std::vector<float> up_proj_f32(is * hs);
      for (size_t i = 0; i < lw.up_proj_q6k.size(); ++i)
        dequantize_q6_k(&lw.up_proj_q6k[i], &up_proj_f32[i * GGML_QK_K],
                        GGML_QK_K);
      std::vector<block_q8_K> x_norm2_q8k = quantize_fp32_to_q8_K(x_norm_vec2);
      std::vector<float> x_norm2_q8k_f32(hs);
      dequantize_q8_k(x_norm2_q8k, x_norm2_q8k_f32, hs, false);
      matvec_f32_f32_vector_cpu(up_proj_f32, x_norm2_q8k_f32, up_vec, is, hs);
    } else if (!lw.up_proj_f32.empty()) {
      if (log_this_layer)
        Logger::info("[CPU_FWD L" + std::to_string(l) +
                     "] MatVec: Up_Proj (F32)");
      matvec_f32_f32_vector_cpu(lw.up_proj_f32, x_norm_vec2, up_vec, is, hs);
    } else if (!lw.up_proj.empty()) {
      if (log_this_layer)
        Logger::warning("[CPU_FWD L" + std::to_string(l) +
                        "] MatVec: Up_Proj (BF16 Fallback)");
      matvec_bf16_f32_vector_cpu(lw.up_proj, x_norm_vec2, up_vec, is, hs);
    } else {
      throw std::runtime_error("Layer " + std::to_string(l) +
                               ": No valid Up projection weights found!");
    }
    if (log_this_layer) log_vector_summary("Layer 0 Up Vec", up_vec);

    silu_cpu(gate_vec, silu_out_vec);

#pragma omp parallel for
    for (int64_t i = 0; i < static_cast<int64_t>(is); ++i) {
      swiglu_result_vec[i] = silu_out_vec[i] * up_vec[i];
    }
    if (log_this_layer)
      log_vector_summary("Layer 0 SwiGLU Result Vec", swiglu_result_vec);

    if (!lw.down_proj_q8_0.empty()) {
      if (log_this_layer)
        Logger::info("[CPU_FWD L" + std::to_string(l) +
                     "] MatVec: Down_Proj (Q8_0)");
      std::vector<float> down_proj_f32(hs * is);
      size_t num_blocks = (size_t)(hs * is) / GGML_QK8_0;
      if (lw.down_proj_q8_0.size() != num_blocks) {
        Logger::error("Down_Proj Q8_0 block count mismatch. Expected " +
                      std::to_string(num_blocks) + " got " +
                      std::to_string(lw.down_proj_q8_0.size()) +
                      " for tensor 'DOWN_PROJ'");
        mlp_out_vec.assign(hs, 0.0f);
      } else {
        for (size_t i = 0; i < lw.down_proj_q8_0.size(); ++i) {
          dequantize_q8_0_block(&lw.down_proj_q8_0[i],
                                &down_proj_f32[i * GGML_QK8_0]);
        }
        matvec_f32_f32_vector_cpu(down_proj_f32, swiglu_result_vec, mlp_out_vec,
                                  hs, is);
      }
    } else if (!lw.down_proj_q6k.empty()) {
      if (log_this_layer)
        Logger::info("[CPU_FWD L" + std::to_string(l) +
                     "] MatVec: Down_Proj (Q6_K)");

      std::vector<float> down_proj_f32(hs * is);
      for (size_t i = 0; i < lw.down_proj_q6k.size(); ++i)
        dequantize_q6_k(&lw.down_proj_q6k[i], &down_proj_f32[i * GGML_QK_K],
                        GGML_QK_K);
      std::vector<float> swiglu_q8k_f32(is);
      std::vector<block_q8_K> swiglu_q8k =
          quantize_fp32_to_q8_K(swiglu_result_vec);
      dequantize_q8_k(swiglu_q8k, swiglu_q8k_f32, is, false);
      matvec_f32_f32_vector_cpu(down_proj_f32, swiglu_q8k_f32, mlp_out_vec, hs,
                                is);
    } else if (!lw.down_proj_q4k.empty()) {
      if (log_this_layer)
        Logger::info("[CPU_FWD L" + std::to_string(l) +
                     "] MatVec: Down_Proj (Q4_K)");
      std::vector<float> down_proj_f32(hs * is);
      for (size_t i = 0; i < lw.down_proj_q4k.size(); ++i)
        dequantize_q4_k_m(&lw.down_proj_q4k[i], &down_proj_f32[i * GGML_QK_K],
                          GGML_QK_K);
      std::vector<block_q8_K> swiglu_q8k =
          quantize_fp32_to_q8_K(swiglu_result_vec);
      std::vector<float> swiglu_q8k_f32(is);
      dequantize_q8_k(swiglu_q8k, swiglu_q8k_f32, is, false);
      matvec_f32_f32_vector_cpu(down_proj_f32, swiglu_q8k_f32, mlp_out_vec, hs,
                                is);
    } else if (!lw.down_proj_f32.empty()) {
      if (log_this_layer)
        Logger::info("[CPU_FWD L" + std::to_string(l) +
                     "] MatVec: Down_Proj (F32)");
      matvec_f32_f32_vector_cpu(lw.down_proj_f32, swiglu_result_vec,
                                mlp_out_vec, hs, is);
    } else if (!lw.down_proj.empty()) {
      if (log_this_layer)
        Logger::warning("[CPU_FWD L" + std::to_string(l) +
                        "] MatVec: Down_Proj (BF16 Fallback)");
      matvec_bf16_f32_vector_cpu(lw.down_proj, swiglu_result_vec, mlp_out_vec,
                                 hs, is);
    } else {
      throw std::runtime_error("Layer " + std::to_string(l) +
                               ": No valid Down projection weights found!");
    }

#pragma omp parallel for
    for (int64_t i = 0; i < static_cast<int64_t>(hs); ++i) {
      input[i] = x_resid2_vec[i] + mlp_out_vec[i];
    }

    if (l == 0) {
      log_vector_summary("Layer 0 MLP Down Proj Out (mlp_out_vec)",
                         mlp_out_vec);
    }

    if (l == 0) {
      log_vector_summary("Layer 0 End (After MLP Residual) (input)", input);
    }

    if (log_this_layer) {
      Logger::info("[CPU_FWD] ------ END Layer " + std::to_string(l) +
                   " (pos=" + std::to_string(n_tokens) + ") ------");
    }
  }

  if (log_this_step || log_first_gen_step)
    Logger::info("[CPU_FWD STEP " + std::to_string(n_tokens) +
                 "] After layer loop, before final RMSNorm.");

  const std::vector<float>& w_final_norm_vec =
      final_norm_f32.empty() ? bf16vec_to_float_vec(final_norm)
                             : final_norm_f32;
  std::vector<float> x_final_norm_vec(hs);
  rmsnorm_vector_cpu(input, w_final_norm_vec, x_final_norm_vec, eps);
  if (log_this_step || log_first_gen_step)
    log_vector_summary("[CPU_FWD] Output of Final CPU RMSNorm (pos=" +
                           std::to_string(n_tokens) + ")",
                       x_final_norm_vec);

  std::vector<float> logits(vs);
  bool lm_head_logged = false;
  if (!lm_head_q8_0.empty()) {
    if (log_this_step) Logger::info("[CPU_FWD] Using Q8_0 LM Head");
    std::vector<float> lm_head_f32(vs * hs);
    size_t num_blocks = (size_t)(vs * hs) / GGML_QK8_0;
    if (lm_head_q8_0.size() != num_blocks) {
      Logger::error("LM_Head Q8_0 block count mismatch. Expected " +
                    std::to_string(num_blocks) + " got " +
                    std::to_string(lm_head_q8_0.size()) +
                    " for tensor 'LM_HEAD'");
    } else {
      for (size_t i = 0; i < lm_head_q8_0.size(); ++i) {
        dequantize_q8_0_block(&lm_head_q8_0[i], &lm_head_f32[i * GGML_QK8_0]);
      }
      matvec_f32_f32_vector_cpu(lm_head_f32, x_final_norm_vec, logits, vs, hs);
      lm_head_logged = true;
    }
  } else if (!lm_head_q6k.empty()) {
    if (log_this_step) Logger::info("[CPU_FWD] Using Q6_K LM Head");
    std::vector<float> lm_head_f32(vs * hs);
    for (size_t i = 0; i < lm_head_q6k.size(); ++i)
      dequantize_q6_k(&lm_head_q6k[i], &lm_head_f32[i * GGML_QK_K], GGML_QK_K);
    matvec_f32_f32_vector_cpu(lm_head_f32, x_final_norm_vec, logits, vs, hs);
    lm_head_logged = true;
  } else if (!lm_head_q4k.empty()) {
    if (log_this_step) Logger::info("[CPU_FWD] Using Q4_K LM Head");
    std::vector<float> lm_head_f32(vs * hs);
    for (size_t i = 0; i < lm_head_q4k.size(); ++i)
      dequantize_q4_k_m(&lm_head_q4k[i], &lm_head_f32[i * GGML_QK_K],
                        GGML_QK_K);
    matvec_f32_f32_vector_cpu(lm_head_f32, x_final_norm_vec, logits, vs, hs);
    lm_head_logged = true;
  } else if (!lm_head_f32.empty()) {
    if (log_this_step) Logger::info("[CPU_FWD] Using F32 LM Head");
    matvec_f32_f32_vector_cpu(lm_head_f32, x_final_norm_vec, logits, vs, hs);
    lm_head_logged = true;
  } else if (!lm_head.empty()) {
    Logger::warning(
        "[CPU_FWD] Using BF16 LM head weights directly - Ensure F32 version "
        "wasn't expected.");
    if (log_this_step) Logger::info("[CPU_FWD] Using BF16 LM Head (Fallback)");
    matvec_bf16_f32_vector_cpu(lm_head, x_final_norm_vec, logits, vs, hs);
    lm_head_logged = true;
  }

  if (!lm_head_logged) {
    Logger::fatal(
        "No valid LM Head weights found or processed (Q8_0, Q6_K, Q4_K, F32, "
        "or BF16).");
    throw std::runtime_error("Missing or failed LM head weights processing");
  }
  if (log_this_step || log_first_gen_step) {
    Logger::info("Reached end of forward for n_tokens=0");
    log_vector_summary("[CPU_FWD] Final Logits (BEFORE RETURN, pos=" +
                           std::to_string(n_tokens) + ")",
                       logits, 15);
  }
  if (log_this_step || log_first_gen_step) {
    Logger::info("[CPU_FWD] forward complete. pos=" + std::to_string(n_tokens));
  }

  if (nhl > 0 && (n_tokens == 12 || n_tokens == 13)) {
    log_vector_summary_detailed("[CPU] Layer 0 Output (after layer 0, pos=" +
                                    std::to_string(n_tokens) + ")",
                                input, n_tokens, 0, 8);
  }

  return logits;
}

int TinyLlamaModel::get_vocab_size() const { return config_.vocab_size; }

#ifdef HAS_CUDA
std::vector<float> TinyLlamaModel::forward_device(
    int token_id, int pos, KVCache* kv_cache,
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
  int nhl = config_.num_hidden_layers;
  int is = config_.intermediate_size;
  float eps = config_.rms_norm_eps;
  int max_seq_len = config_.max_position_embeddings;
  bool log_this_pos = (pos == 13);

  if (log_this_pos) {
    Logger::info("[TM::fw_dev pos=" + std::to_string(pos) +
                 "] Entered. Token ID: " + std::to_string(token_id));
  }

  cublasStatus_t stream_status = cublasSetStream(cublas_handle_, stream);
  if (stream_status != CUBLAS_STATUS_SUCCESS) {
    Logger::error("cublasSetStream failed in forward_device");
    return {};
  }

  const void* embed_table_dev_ptr = nullptr;
  bool is_bf16_embedding = false;
  if (token_embedding_table_f32_dev_) {
    embed_table_dev_ptr = token_embedding_table_f32_dev_;
    is_bf16_embedding = false;
  } else if (token_embedding_table_dev_) {
    embed_table_dev_ptr = token_embedding_table_dev_;
    is_bf16_embedding = true;
  } else {
    Logger::error(
        "No embedding table found on GPU (FP32 or BF16) in forward_device.");
    return {};
  }
  lookup_embedding_cuda(embed_table_dev_ptr, x_dev_, token_id, hs, vs,
                        is_bf16_embedding, stream);
  if (log_this_pos) {
    std::vector<float> x_host_after_embed(hs);
    gpuErrchk(cudaMemcpy(x_host_after_embed.data(), x_dev_, hs * sizeof(float),
                         cudaMemcpyDeviceToHost));
    log_vector_summary_detailed("[TM::fw_dev pos=" + std::to_string(pos) +
                                    "] x_dev_ after embedding lookup",
                                x_host_after_embed, pos, -2, 8);
  }

  for (int l = 0; l < nhl; ++l) {
    if (log_this_pos) {
      Logger::info("[TM::fw_dev pos=" + std::to_string(pos) + "] Layer " +
                   std::to_string(l) + ": Calling layer.forward_device");
    }
    const auto& lw = layers[l];

    size_t layer_q_size = (size_t)hs * hs;
    size_t layer_k_size = (size_t)n_kv_heads * head_dim * hs;
    size_t layer_v_size = (size_t)n_kv_heads * head_dim * hs;
    size_t layer_o_size = (size_t)hs * hs;
    size_t layer_gate_size = (size_t)is * hs;
    size_t layer_up_size = (size_t)is * hs;
    size_t layer_down_size = (size_t)hs * is;
    const uint16_t* lw_q_proj_bf16_dev =
        w_q_dev_ ? w_q_dev_ + (size_t)l * layer_q_size : nullptr;
    const uint16_t* lw_k_proj_bf16_dev =
        w_k_dev_ ? w_k_dev_ + (size_t)l * layer_k_size : nullptr;
    const uint16_t* lw_v_proj_bf16_dev =
        w_v_dev_ ? w_v_dev_ + (size_t)l * layer_v_size : nullptr;
    const uint16_t* lw_o_proj_bf16_dev =
        w_o_dev_ ? w_o_dev_ + (size_t)l * layer_o_size : nullptr;
    const uint16_t* lw_gate_proj_bf16_dev =
        w_gate_dev_ ? w_gate_dev_ + (size_t)l * layer_gate_size : nullptr;
    const uint16_t* lw_up_proj_bf16_dev =
        w_up_dev_ ? w_up_dev_ + (size_t)l * layer_up_size : nullptr;
    const uint16_t* lw_down_proj_bf16_dev =
        w_down_dev_ ? w_down_dev_ + (size_t)l * layer_down_size : nullptr;

    const float* lw_q_proj_f32_dev =
        w_q_f32_dev_ ? w_q_f32_dev_ + (size_t)l * layer_q_size : nullptr;
    const float* lw_k_proj_f32_dev =
        w_k_f32_dev_ ? w_k_f32_dev_ + (size_t)l * layer_k_size : nullptr;
    const float* lw_v_proj_f32_dev =
        w_v_f32_dev_ ? w_v_f32_dev_ + (size_t)l * layer_v_size : nullptr;
    const float* lw_o_proj_f32_dev =
        w_o_f32_dev_ ? w_o_f32_dev_ + (size_t)l * layer_o_size : nullptr;
    const float* lw_gate_proj_f32_dev =
        w_gate_f32_dev_ ? w_gate_f32_dev_ + (size_t)l * layer_gate_size
                        : nullptr;
    const float* lw_up_proj_f32_dev =
        w_up_f32_dev_ ? w_up_f32_dev_ + (size_t)l * layer_up_size : nullptr;
    const float* lw_down_proj_f32_dev =
        w_down_f32_dev_ ? w_down_f32_dev_ + (size_t)l * layer_down_size
                        : nullptr;
    const float* lw_in_norm_dev = layers[l].input_layernorm_dev;
    const float* lw_post_norm_dev = layers[l].post_attention_layernorm_dev;

    gpuErrchk(cudaMemcpyAsync(x_resid1_dev_, x_dev_, hs * sizeof(float),
                              cudaMemcpyDeviceToDevice, stream));

    rmsnorm_vector_cuda(x_dev_, layers[l].input_layernorm_dev, x_norm_dev_, hs,
                        eps, stream);

    if (lw_q_proj_f32_dev && lw_k_proj_f32_dev && lw_v_proj_f32_dev) {
      matvec_f32_f32_cuda(cublas_handle_, lw_q_proj_f32_dev, x_norm_dev_,
                          q_dev_, hs, hs, stream);
      matvec_f32_f32_cuda(cublas_handle_, lw_k_proj_f32_dev, x_norm_dev_,
                          k_dev_, n_kv_heads * head_dim, hs, stream);
      matvec_f32_f32_cuda(cublas_handle_, lw_v_proj_f32_dev, x_norm_dev_,
                          v_dev_, n_kv_heads * head_dim, hs, stream);
    } else if (lw_q_proj_bf16_dev && lw_k_proj_bf16_dev && lw_v_proj_bf16_dev) {
      Logger::warning("Layer " + std::to_string(l) +
                      ": Using BF16 matvec path (less efficient) for QKV.");
      matvec_bf16_f32_cuda(cublas_handle_, lw_q_proj_bf16_dev, x_norm_dev_,
                           q_dev_, hs, hs, stream);
      matvec_bf16_f32_cuda(cublas_handle_, lw_k_proj_bf16_dev, x_norm_dev_,
                           k_dev_, n_kv_heads * head_dim, hs, stream);
      matvec_bf16_f32_cuda(cublas_handle_, lw_v_proj_bf16_dev, x_norm_dev_,
                           v_dev_, n_kv_heads * head_dim, hs, stream);
    } else {
      Logger::error(
          "Layer " + std::to_string(l) +
          ": No valid QKV projection weights found on GPU (FP32 or BF16).");
      return {};
    }

    if (log_this_pos) {
      std::vector<float> temp_q_host(hs);
      gpuErrchk(cudaMemcpy(temp_q_host.data(), q_dev_, hs * sizeof(float),
                           cudaMemcpyDeviceToHost));
      log_vector_summary_detailed("[TM::fw_dev pos=" + std::to_string(pos) +
                                      " L" + std::to_string(l) +
                                      "] q_dev_ after QKV Proj",
                                  temp_q_host, pos, l, 8);
    }
    if (log_this_pos) {
      std::vector<float> temp_q_host_rope(hs);
      gpuErrchk(cudaMemcpy(temp_q_host_rope.data(), q_dev_, hs * sizeof(float),
                           cudaMemcpyDeviceToHost));
      log_vector_summary_detailed("[TM::fw_dev pos=" + std::to_string(pos) +
                                      " L" + std::to_string(l) +
                                      "] q_dev_ after RoPE",
                                  temp_q_host_rope, pos, l, 8);
    }

    rope_cuda(q_dev_, n_heads, head_dim, all_freqs_cis_dev, pos, stream);
    rope_cuda(k_dev_, n_kv_heads, head_dim, all_freqs_cis_dev, pos, stream);

    for (int kvh = 0; kvh < n_kv_heads; ++kvh) {
      const float* current_k_head_ptr = k_dev_ + kvh * head_dim;
      const float* current_v_head_ptr = v_dev_ + kvh * head_dim;

      update_kv_cache_cuda(kv_cache->layers[l].k_dev, current_k_head_ptr, pos,
                           kvh, kv_cache->allocated_max_seq_len,
                           kv_cache->allocated_num_kv_heads,
                           kv_cache->allocated_head_dim, stream);

      update_kv_cache_cuda(kv_cache->layers[l].v_dev, current_v_head_ptr, pos,
                           kvh, kv_cache->allocated_max_seq_len,
                           kv_cache->allocated_num_kv_heads,
                           kv_cache->allocated_head_dim, stream);
    }

    float scale = 1.0f / SAFE_SQRT(static_cast<float>(head_dim));
    attention_cuda(q_dev_, kv_cache->layers[l].k_dev, kv_cache->layers[l].v_dev,
                   attn_out_dev_, n_heads, pos + 1, head_dim, scale,
                   kv_cache->allocated_max_seq_len,
                   kv_cache->allocated_num_kv_heads, stream);
    if (log_this_pos) {
      std::vector<float> temp_attn_out_host(hs);
      gpuErrchk(cudaMemcpy(temp_attn_out_host.data(), attn_out_dev_,
                           hs * sizeof(float), cudaMemcpyDeviceToHost));
      log_vector_summary_detailed("[TM::fw_dev pos=" + std::to_string(pos) +
                                      " L" + std::to_string(l) +
                                      "] attn_out_dev_ after Attention",
                                  temp_attn_out_host, pos, l, 8);
    }

    if (lw_o_proj_f32_dev) {
      matvec_f32_f32_cuda(cublas_handle_, lw_o_proj_f32_dev, attn_out_dev_,
                          attn_proj_dev_, hs, hs, stream);
    } else if (lw_o_proj_bf16_dev) {
      Logger::warning("Layer " + std::to_string(l) +
                      ": Using BF16 matvec path (less efficient) for O-Proj.");
      matvec_bf16_f32_cuda(cublas_handle_, lw_o_proj_bf16_dev, attn_out_dev_,
                           attn_proj_dev_, hs, hs, stream);
    } else {
      Logger::error(
          "Layer " + std::to_string(l) +
          ": No valid O projection weights found on GPU (FP32 or BF16).");
      return {};
    }
    if (log_this_pos) {
      std::vector<float> temp_attn_proj_host(hs);
      gpuErrchk(cudaMemcpy(temp_attn_proj_host.data(), attn_proj_dev_,
                           hs * sizeof(float), cudaMemcpyDeviceToHost));
      log_vector_summary_detailed("[TM::fw_dev pos=" + std::to_string(pos) +
                                      " L" + std::to_string(l) +
                                      "] attn_proj_dev_ after O-Proj",
                                  temp_attn_proj_host, pos, l, 8);
    }

    add_residual_cuda(attn_proj_dev_, x_resid1_dev_, x_dev_, hs, stream);

    gpuErrchk(cudaMemcpyAsync(x_resid2_dev_, x_dev_, hs * sizeof(float),
                              cudaMemcpyDeviceToDevice, stream));

    rmsnorm_vector_cuda(x_dev_, layers[l].post_attention_layernorm_dev,
                        x_norm_dev_, hs, eps, stream);
    if (log_this_pos) {
      std::vector<float> temp_x_host(hs);
      gpuErrchk(cudaMemcpy(temp_x_host.data(), x_norm_dev_, hs * sizeof(float),
                           cudaMemcpyDeviceToHost));
      log_vector_summary_detailed("[TM::fw_dev pos=" + std::to_string(pos) +
                                      " L" + std::to_string(l) +
                                      "] x_norm_dev_ after Input RMSNorm",
                                  temp_x_host, pos, l, 8);
    }

    if (lw_gate_proj_f32_dev && lw_up_proj_f32_dev) {
      matvec_f32_f32_cuda(cublas_handle_, lw_gate_proj_f32_dev, x_norm_dev_,
                          gate_vec_dev_, is, hs, stream);
      matvec_f32_f32_cuda(cublas_handle_, lw_up_proj_f32_dev, x_norm_dev_,
                          up_vec_dev_, is, hs, stream);
    } else if (lw_gate_proj_bf16_dev && lw_up_proj_bf16_dev) {
      Logger::warning(
          "Layer " + std::to_string(l) +
          ": Using BF16 matvec path (less efficient) for Gate/Up Proj.");
      matvec_bf16_f32_cuda(cublas_handle_, lw_gate_proj_bf16_dev, x_norm_dev_,
                           gate_vec_dev_, is, hs, stream);
      matvec_bf16_f32_cuda(cublas_handle_, lw_up_proj_bf16_dev, x_norm_dev_,
                           up_vec_dev_, is, hs, stream);
    } else {
      Logger::error(
          "Layer " + std::to_string(l) +
          ": No valid Gate/Up projection weights found on GPU (FP32 or BF16).");
      return {};
    }
    if (log_this_pos) {
      std::vector<float> temp_gate_host(is);
      gpuErrchk(cudaMemcpy(temp_gate_host.data(), gate_vec_dev_,
                           is * sizeof(float), cudaMemcpyDeviceToHost));
      log_vector_summary_detailed("[TM::fw_dev pos=" + std::to_string(pos) +
                                      " L" + std::to_string(l) +
                                      "] gate_vec_dev_ after Proj",
                                  temp_gate_host, pos, l, 8);
    }

    swiglu_cuda(gate_vec_dev_, up_vec_dev_, swiglu_vec_dev_, is, stream);
    if (log_this_pos) {
      std::vector<float> temp_swiglu_host(is);
      gpuErrchk(cudaMemcpy(temp_swiglu_host.data(), swiglu_vec_dev_,
                           is * sizeof(float), cudaMemcpyDeviceToHost));
      log_vector_summary_detailed("[TM::fw_dev pos=" + std::to_string(pos) +
                                      " L" + std::to_string(l) +
                                      "] swiglu_vec_dev_ after SwiGLU",
                                  temp_swiglu_host, pos, l, 8);
    }

    if (lw_down_proj_f32_dev) {
      matvec_f32_f32_cuda(cublas_handle_, lw_down_proj_f32_dev, swiglu_vec_dev_,
                          mlp_down_dev_, hs, is, stream);
    } else if (lw_down_proj_bf16_dev) {
      Logger::warning(
          "Layer " + std::to_string(l) +
          ": Using BF16 matvec path (less efficient) for Down Proj.");
      matvec_bf16_f32_cuda(cublas_handle_, lw_down_proj_bf16_dev,
                           swiglu_vec_dev_, mlp_down_dev_, hs, is, stream);
    } else {
      Logger::error(
          "Layer " + std::to_string(l) +
          ": No valid Down projection weights found on GPU (FP32 or BF16).");
      return {};
    }

    if (log_this_pos) {
      std::vector<float> temp_mlp_down_host(hs);
      gpuErrchk(cudaMemcpy(temp_mlp_down_host.data(), mlp_down_dev_,
                           hs * sizeof(float), cudaMemcpyDeviceToHost));
      log_vector_summary_detailed("[TM::fw_dev pos=" + std::to_string(pos) +
                                      " L" + std::to_string(l) +
                                      "] mlp_down_dev_ after Down Proj",
                                  temp_mlp_down_host, pos, l, 8);
    }

    add_residual_cuda(mlp_down_dev_, x_resid2_dev_, x_dev_, hs, stream);
    if (l == 0 && (pos == 12 || pos == 13)) {
      std::vector<float> x_host_layer0_output(hs);
      gpuErrchk(cudaMemcpy(x_host_layer0_output.data(), x_dev_,
                           hs * sizeof(float), cudaMemcpyDeviceToHost));
      log_vector_summary_detailed("[CUDA] Layer 0 Output (INSIDE LOOP, pos=" +
                                      std::to_string(pos) + ")",
                                  x_host_layer0_output, pos, 0, 8);
    }
    if (l == (config_.num_hidden_layers - 1) && (pos == 12 || pos == 13)) {
      std::vector<float> x_host_last_layer_output(hs);
      gpuErrchk(cudaMemcpy(x_host_last_layer_output.data(), x_dev_,
                           hs * sizeof(float), cudaMemcpyDeviceToHost));
      log_vector_summary_detailed(
          "[CPU] Last Layer (L" + std::to_string(l) +
              ") Output (INSIDE LOOP, pos=" + std::to_string(pos) + ")",
          x_host_last_layer_output, pos, l, 8);
    }
    if (l == (config_.num_hidden_layers - 1) && (pos == 12 || pos == 13)) {
      std::vector<float> x_host_last_layer_output(hs);
      gpuErrchk(cudaMemcpy(x_host_last_layer_output.data(), x_dev_,
                           hs * sizeof(float), cudaMemcpyDeviceToHost));
      log_vector_summary_detailed(
          "[CUDA] Last Layer (L" + std::to_string(l) +
              ") Output (INSIDE LOOP, pos=" + std::to_string(pos) + ")",
          x_host_last_layer_output, pos, l, 8);
    }
  }

  if (log_this_pos)
    Logger::info("[TM::fw_dev pos=" + std::to_string(pos) +
                 "] Processing final RMSNorm.");

  rmsnorm_vector_cuda(x_dev_, final_norm_dev, x_norm_dev_, hs, eps, stream);
  if (log_this_pos)
    Logger::info("[TM::fw_dev pos=" + std::to_string(pos) +
                 "] Processing LM Head.");

  if (lm_head_f32_dev_) {
    matvec_f32_f32_cuda(cublas_handle_, lm_head_f32_dev_, x_norm_dev_,
                        logits_dev_, vs, hs, stream);
  } else if (lm_head_dev_) {
    Logger::warning("Using BF16 matvec path (less efficient) for LM Head.");
    matvec_bf16_f32_cuda(cublas_handle_, lm_head_dev_, x_norm_dev_, logits_dev_,
                         vs, hs, stream);
  } else {
    Logger::error("No valid LM head weights found on GPU (FP32 or BF16).");
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
#endif

void map_gguf_weights(const GGUFData& gguf, TinyLlamaModel& model) {
  Logger::info("Mapping GGUF weights to model fields...");
  if (gguf.tensor_data.empty()) {
    Logger::warning("GGUF tensor data buffer is empty. Cannot map weights.");
    return;
  }
  const uint8_t* data_buffer_start = gguf.tensor_data.data();
  const uint8_t* data_buffer_end = data_buffer_start + gguf.tensor_data.size();
  Logger::info("map_gguf_weights: Total tensor data size available: " +
               std::to_string(gguf.tensor_data.size()) + " bytes.");

  for (const auto& pair : gguf.tensor_infos_map) {
    std::stringstream ss_map;
    const std::string& target_field = pair.first;
    const GGUFTensorInfo& info = pair.second;
    const uint8_t* tensor_data_ptr = data_buffer_start + info.offset;
    const uint8_t* tensor_data_end = tensor_data_ptr + info.size_in_bytes;

    ss_map << "Attempting to map tensor: '" << info.name
           << "', Type: " << info.type << ", Offset: " << info.offset
           << ", NumElem: " << info.num_elements
           << ", SizeBytes: " << info.size_in_bytes
           << ", SrcAddr: " << static_cast<const void*>(tensor_data_ptr)
           << ", ReadEndAddr: " << static_cast<const void*>(tensor_data_end)
           << ", DataBuffer: [" << static_cast<const void*>(data_buffer_start)
           << " - " << static_cast<const void*>(data_buffer_end) << "]";

    if (tensor_data_ptr < data_buffer_start ||
        tensor_data_end > data_buffer_end) {
      ss_map << ", InBounds: NO";
      Logger::error(ss_map.str());
      Logger::error("Tensor data out of bounds for: " + info.name);
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
        const block_q8_0* src = reinterpret_cast<const block_q8_0*>(
            gguf.tensor_data.data() + info_local.offset);

        if (reinterpret_cast<const uint8_t*>(src) < data_buffer_start ||
            reinterpret_cast<const uint8_t*>(src + num_blocks) >
                data_buffer_end) {
          Logger::error(
              "Tensor '" + info_local.name +
              "' (Q8_0) data is out of bounds. Offset: " +
              std::to_string(info_local.offset) + ", NumBlocks: " +
              std::to_string(num_blocks) + ", ExpectedSize: " +
              std::to_string(num_blocks * sizeof(block_q8_0)) +
              ", BufferSize: " + std::to_string(gguf.tensor_data.size()));
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
  config.bos_token_id = get_meta_value("tokenizer.ggml.bos_token_id", 1);
  config.eos_token_id = get_meta_value("tokenizer.ggml.eos_token_id", 2);

  config.architecture = get_meta_string("general.architecture", "unknown");
  config.model_name = get_meta_string("general.name", "unknown");
  bool has_pre_key = gguf.metadata.count("tokenizer.ggml.pre");

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
    }
  }

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
  if (config_.num_attention_heads == 0) {
    Logger::error("Cannot initialize RoPE frequencies: num_attention_heads is zero.");
    return;
  }
  int head_dim = config_.hidden_size / config_.num_attention_heads;
  if (head_dim == 0) {
    Logger::error("Cannot initialize RoPE frequencies: calculated head_dim is zero.");
    return;
  }
  if (head_dim % 2 != 0) {
    Logger::error("Cannot initialize RoPE frequencies: head_dim must be even.");
    return;
  }

  // This function precomputes RoPE frequencies for all positions and head dimensions.
  // The precomputed_freqs_cis_ vector stores pairs of (cos(angle), sin(angle)).
  // It's a flat vector: precomputed_freqs_cis_[(pos * head_dim / 2) + (dim_pair_idx)]

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
            // This should not happen if resize was correct
            return; 
        }
      }
    }
    Logger::info("Precomputed RoPE frequencies on CPU. Size: " + std::to_string(precomputed_freqs_cis_.size()));
  } else {
      Logger::info("RoPE frequencies already precomputed.");
  }
}
