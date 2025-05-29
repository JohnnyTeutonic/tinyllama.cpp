#include "model_utils.h"
#include "logger.h"
#include "utils.h"
#include "quantization.h"
#include "model_constants.h"
#include "model_macros.h"
#include <algorithm>
#include <cstring>
#include <cmath>

std::vector<float> TinyLlamaModel::lookup_embedding(int token_id) {
  int hs = config_.hidden_size;
  int vs = config_.vocab_size;

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
    
    if (token_id < 2) {
      float sum = 0.0f, min_val = embedding_vec[0], max_val = embedding_vec[0];
      for (int i = 0; i < hs; ++i) {
        sum += embedding_vec[i];
        min_val = std::min(min_val, embedding_vec[i]);
        max_val = std::max(max_val, embedding_vec[i]);
      }
      Logger::info("[Q8_0_EMBED_FINAL] Token " + std::to_string(token_id) + 
                   " embedding stats: sum=" + std::to_string(sum) + 
                   ", mean=" + std::to_string(sum / hs) + 
                   ", min=" + std::to_string(min_val) + 
                   ", max=" + std::to_string(max_val) + 
                   ", first_4=[" + std::to_string(embedding_vec[0]) + 
                   ", " + std::to_string(embedding_vec[1]) + 
                   ", " + std::to_string(embedding_vec[2]) + 
                   ", " + std::to_string(embedding_vec[3]) + "]");
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
    return embedding_vec;

  } else {
    Logger::error(
        "No valid embedding table found (Q4_K, Q8_0, Q6_K, F32, BF16) for token: " +
        std::to_string(token_id));

    return embedding_vec;
  }
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

  Logger::info("[ROPE_INIT] Initializing RoPE with head_dim=" + std::to_string(head_dim) +
               ", configured max_pos_emb=" + std::to_string(config_.max_position_embeddings) +
               ", using internal rope::MAX_SEQUENCE_LENGTH=" + std::to_string(rope::MAX_SEQUENCE_LENGTH) +
               ", configured rope_theta=" + std::to_string(config_.rope_theta));


  if (precomputed_freqs_cis_.empty()) { 
    int max_seq_len = rope::MAX_SEQUENCE_LENGTH;
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

int TinyLlamaModel::get_vocab_size() const { 
  return config_.vocab_size; 
} 