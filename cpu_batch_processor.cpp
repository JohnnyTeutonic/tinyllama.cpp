#include "cpu_batch_processor.h"
#include "utils.h"
#include "cpu_attention.h"
#include "model_macros.h"
#include "logger.h"
#include <cmath>
#include <algorithm>

CPUBatchProcessor::CPUBatchProcessor(TinyLlamaModel* model) : model_(model) {}

std::vector<float> CPUBatchProcessor::forward_cpu_batch(
    const std::vector<float>& batch_input_activations,
    int num_tokens_in_batch,
    int num_cpu_layers_to_process,
    int start_pos_in_sequence,
    KVCache* kv_cache,
    const std::vector<int>& prompt_lengths) {

    if (batch_input_activations.size() != (size_t)num_tokens_in_batch * model_->config_.hidden_size) {
        Logger::error("[CPU_BATCH_FWD] Input size mismatch. Expected: " +
                      std::to_string((size_t)num_tokens_in_batch * model_->config_.hidden_size) + " Got: " +
                      std::to_string(batch_input_activations.size()));
        return {};
    }

    int hs = model_->config_.hidden_size;
    int is = model_->config_.intermediate_size;
    int n_heads = model_->config_.num_attention_heads;
    int n_kv_heads = model_->config_.num_key_value_heads;
    if (n_heads == 0) {
        Logger::error("[CPU_BATCH_FWD] Error: num_attention_heads is zero.");
        return {};
    }
    int head_dim = hs / n_heads;
    float eps = model_->config_.rms_norm_eps;
    int max_pos_embeddings = model_->config_.max_position_embeddings;
    bool use_rope_adjacent_pairing = model_->config_.is_gguf_file_loaded;
    float attention_scale = 1.0f / SAFE_SQRT(static_cast<float>(head_dim));

    std::vector<float> current_batch_activations = batch_input_activations;

    std::vector<int> sequence_indices(num_tokens_in_batch);
    std::vector<int> position_in_sequence(num_tokens_in_batch);
    
    if (!prompt_lengths.empty()) {
        int token_offset = 0;
        for (size_t seq_idx = 0; seq_idx < prompt_lengths.size(); ++seq_idx) {
            for (int pos = 0; pos < prompt_lengths[seq_idx]; ++pos) {
                if (token_offset >= num_tokens_in_batch) {
                    Logger::error("[CPU_BATCH_FWD] Token offset exceeded num_tokens_in_batch");
                    return {};
                }
                sequence_indices[token_offset] = seq_idx;
                position_in_sequence[token_offset] = pos;
                token_offset++;
            }
        }
    } else {
        for (int token_idx = 0; token_idx < num_tokens_in_batch; ++token_idx) {
            sequence_indices[token_idx] = 0;
            position_in_sequence[token_idx] = start_pos_in_sequence + token_idx;
        }
    }

    for (int l = 0; l < num_cpu_layers_to_process; ++l) {
        model_->ensure_q_proj_dequantized(l);
        model_->ensure_k_proj_dequantized(l);
        model_->ensure_v_proj_dequantized(l);
        model_->ensure_o_proj_dequantized(l);
        model_->ensure_gate_proj_dequantized(l);
        model_->ensure_up_proj_dequantized(l);
        model_->ensure_down_proj_dequantized(l);
        
        const auto& lw = model_->layers[l];
        
        std::vector<float> batch_x_norm1(current_batch_activations.size());
        const std::vector<float>& w_input_norm_vec =
            lw.input_layernorm_f32.empty()
                ? bf16vec_to_float_vec(lw.input_layernorm)
                : lw.input_layernorm_f32;
        rmsnorm_batch_cpu(current_batch_activations, w_input_norm_vec, batch_x_norm1, num_tokens_in_batch, hs, eps);

        std::vector<float> residual_batch_component_attn = current_batch_activations;

        std::vector<float> q_batch((size_t)num_tokens_in_batch * hs);
        std::vector<float> k_batch((size_t)num_tokens_in_batch * n_kv_heads * head_dim);
        std::vector<float> v_batch((size_t)num_tokens_in_batch * n_kv_heads * head_dim);

        if (!lw.q_proj_f32.empty()) {
            matmul_f32_f32_batch_cpu(lw.q_proj_f32, batch_x_norm1, q_batch, num_tokens_in_batch, hs, hs);
        } else if (!lw.q_proj_q8_0.empty()) {
            matmul_q8_0_f32_batch_cpu(lw.q_proj_q8_0, batch_x_norm1, q_batch, num_tokens_in_batch, hs, hs);
        } else if (!lw.q_proj_q6k.empty()) {
            matmul_q6k_f32_batch_cpu(lw.q_proj_q6k, batch_x_norm1, q_batch, num_tokens_in_batch, hs, hs);
        } else if (!lw.q_proj_q4k.empty()) {
            matmul_q4k_f32_batch_cpu(lw.q_proj_q4k, batch_x_norm1, q_batch, num_tokens_in_batch, hs, hs);
        } else {
            Logger::error("[CPU_BATCH_FWD] Layer " + std::to_string(l) + ": No Q proj weights found for CPU");
            return {};
        }

        if (!lw.k_proj_f32.empty()) {
            matmul_f32_f32_batch_cpu(lw.k_proj_f32, batch_x_norm1, k_batch, num_tokens_in_batch, n_kv_heads * head_dim, hs);
        } else if (!lw.k_proj_q8_0.empty()) {
            matmul_q8_0_f32_batch_cpu(lw.k_proj_q8_0, batch_x_norm1, k_batch, num_tokens_in_batch, n_kv_heads * head_dim, hs);
        } else if (!lw.k_proj_q6k.empty()) {
            matmul_q6k_f32_batch_cpu(lw.k_proj_q6k, batch_x_norm1, k_batch, num_tokens_in_batch, n_kv_heads * head_dim, hs);
        } else if (!lw.k_proj_q4k.empty()) {
            matmul_q4k_f32_batch_cpu(lw.k_proj_q4k, batch_x_norm1, k_batch, num_tokens_in_batch, n_kv_heads * head_dim, hs);
        } else {
            Logger::error("[CPU_BATCH_FWD] Layer " + std::to_string(l) + ": No K proj weights found for CPU");
            return {};
        }

        if (!lw.v_proj_f32.empty()) {
            matmul_f32_f32_batch_cpu(lw.v_proj_f32, batch_x_norm1, v_batch, num_tokens_in_batch, n_kv_heads * head_dim, hs);
        } else if (!lw.v_proj_q8_0.empty()) {
            matmul_q8_0_f32_batch_cpu(lw.v_proj_q8_0, batch_x_norm1, v_batch, num_tokens_in_batch, n_kv_heads * head_dim, hs);
        } else if (!lw.v_proj_q6k.empty()) {
            matmul_q6k_f32_batch_cpu(lw.v_proj_q6k, batch_x_norm1, v_batch, num_tokens_in_batch, n_kv_heads * head_dim, hs);
        } else if (!lw.v_proj_q4k.empty()) {
            matmul_q4k_f32_batch_cpu(lw.v_proj_q4k, batch_x_norm1, v_batch, num_tokens_in_batch, n_kv_heads * head_dim, hs);
        } else {
            Logger::error("[CPU_BATCH_FWD] Layer " + std::to_string(l) + ": No V proj weights found for CPU");
            return {};
        }

        if (!prompt_lengths.empty()) {
            for (int t = 0; t < num_tokens_in_batch; ++t) {
                int current_token_pos = position_in_sequence[t];
                int seq_idx = sequence_indices[t];

                if (current_token_pos < 0 || current_token_pos >= max_pos_embeddings) {
                    Logger::warning("[CPU_BATCH_FWD] Token " + std::to_string(t) + " (seq=" + std::to_string(seq_idx) + 
                                    ", pos=" + std::to_string(current_token_pos) + ") is out of range. Skipping RoPE.");
                    continue;
                }

                std::vector<float> q_token(hs);
                std::vector<float> k_token(n_kv_heads * head_dim);
                
                std::copy(q_batch.begin() + (size_t)t * hs, 
                         q_batch.begin() + (size_t)(t + 1) * hs, 
                         q_token.begin());
                std::copy(k_batch.begin() + (size_t)t * n_kv_heads * head_dim, 
                         k_batch.begin() + (size_t)(t + 1) * n_kv_heads * head_dim, 
                         k_token.begin());

                apply_rope_vector(q_token, n_heads, head_dim, current_token_pos, model_->precomputed_freqs_cis_, max_pos_embeddings, use_rope_adjacent_pairing);
                apply_rope_vector(k_token, n_kv_heads, head_dim, current_token_pos, model_->precomputed_freqs_cis_, max_pos_embeddings, use_rope_adjacent_pairing);

                std::copy(q_token.begin(), q_token.end(), q_batch.begin() + (size_t)t * hs);
                std::copy(k_token.begin(), k_token.end(), k_batch.begin() + (size_t)t * n_kv_heads * head_dim);
            }
        } else {
            apply_rope_batch_cpu(q_batch, k_batch, num_tokens_in_batch, n_heads, n_kv_heads, head_dim, 
                                  start_pos_in_sequence, model_->precomputed_freqs_cis_, max_pos_embeddings, use_rope_adjacent_pairing);
        }

        if (kv_cache) {
            if (!prompt_lengths.empty()) {
                update_kv_cache_batch_cpu_sequence_aware(kv_cache, l, k_batch, v_batch, num_tokens_in_batch,
                                                         sequence_indices, position_in_sequence, n_kv_heads, head_dim);
            } else {
            update_kv_cache_batch_cpu(kv_cache, l, k_batch, v_batch, num_tokens_in_batch, 
                                      start_pos_in_sequence, n_kv_heads, head_dim);        
            }
        }
        
        std::vector<float> batch_attn_output((size_t)num_tokens_in_batch * hs);
        
        if (kv_cache && static_cast<size_t>(l) < kv_cache->layers.size()) {
            if (!prompt_lengths.empty()) {
                attention_batch_cpu_sequence_aware(q_batch, kv_cache->layers[l], batch_attn_output,
                                                  num_tokens_in_batch, sequence_indices, position_in_sequence,
                                                  n_heads, n_kv_heads, head_dim, attention_scale,
                                                  kv_cache->max_seq_len_config_);
            } else {
                attention_batch_cpu(q_batch, kv_cache->layers[l], batch_attn_output,
                                   num_tokens_in_batch, start_pos_in_sequence,
                                   n_heads, n_kv_heads, head_dim, attention_scale);
            }
        } else if (kv_cache) { 
            Logger::error("[CPU_BATCH_FWD] Layer " + std::to_string(l) + 
                          " is out of bounds for KV Cache access during attention. KVCache layers size: " + 
                          std::to_string(kv_cache->layers.size()) + 
                          ". Filling attention output with zeros.");
            std::fill(batch_attn_output.begin(), batch_attn_output.end(), 0.0f); 
        } else {
            Logger::error("[CPU_BATCH_FWD] KV Cache is null, cannot perform attention for layer " + std::to_string(l) +
                          ". Filling attention output with zeros.");
            std::fill(batch_attn_output.begin(), batch_attn_output.end(), 0.0f); 
        }

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

        for(size_t i=0; i < current_batch_activations.size(); ++i) {
            current_batch_activations[i] = residual_batch_component_attn[i] + batch_attn_proj_out[i];
        }
        
        std::vector<float> residual_batch_component_mlp = current_batch_activations;
        std::vector<float> batch_x_norm2(current_batch_activations.size());
        const std::vector<float>& w_post_attn_norm_vec =
            lw.post_attention_layernorm_f32.empty()
                ? bf16vec_to_float_vec(lw.post_attention_layernorm)
                : lw.post_attention_layernorm_f32;
        
        rmsnorm_batch_cpu(current_batch_activations, w_post_attn_norm_vec, batch_x_norm2, num_tokens_in_batch, hs, eps);
        
        std::vector<float> batch_gate_proj_out((size_t)num_tokens_in_batch * is);
        std::vector<float> batch_up_proj_out((size_t)num_tokens_in_batch * is);
        
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
        
        std::vector<float> batch_swiglu_out((size_t)num_tokens_in_batch * is);
        for (size_t i = 0; i < batch_gate_proj_out.size(); ++i) {
            float gate_val = batch_gate_proj_out[i];
            float silu_gate_val = gate_val / (1.0f + std::exp(-gate_val));
            batch_swiglu_out[i] = silu_gate_val * batch_up_proj_out[i];
        }
        
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
        }
        
        for(size_t i = 0; i < current_batch_activations.size(); ++i) {
            current_batch_activations[i] = residual_batch_component_mlp[i] + batch_mlp_down_proj_out[i];
        }
    }

    if (kv_cache && num_tokens_in_batch > 0) {
        kv_cache->seq_len = start_pos_in_sequence + num_tokens_in_batch;
    }
    return current_batch_activations;
} 