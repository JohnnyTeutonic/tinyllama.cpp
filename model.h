#ifndef MODEL_H
#define MODEL_H

#include <string>
#include <vector>
#include <unordered_map>
#include <nlohmann/json.hpp>
#include "safetensors_loader.h"
#include <functional>
#include <cstdint> // Include for uint16_t
#ifdef HAS_CUDA
#include <cuda_runtime.h> // Needed for cudaFree, cudaMalloc
#include <cublas_v2.h>    // <<< INCLUDE CUBLAS >>>
#include "cuda_kernels.h" // Needed for gpuErrchk
#endif
#include "quantization.h" // For block_q4_K, block_q6_K
#include <memory>

struct ModelConfig {
    int hidden_size;
    int intermediate_size;
    int num_attention_heads;
    int num_key_value_heads;
    int num_hidden_layers;
    int vocab_size;
    int max_position_embeddings;
    float rms_norm_eps;
    float rope_theta;
    std::string hidden_act;
    std::string torch_dtype;
    int bos_token_id;
    int eos_token_id;
};

struct GGUFData; // Forward declaration
struct ModelConfig; // Forward declaration needed for the parse function
ModelConfig parse_model_config_from_gguf(const GGUFData& gguf); // Forward declaration

// KVCache for autoregressive inference
struct KVCacheLayer {
    // --- ALWAYS define host vectors --- 
    std::vector<float> k; // Host vector for K cache (CPU path)
    std::vector<float> v; // Host vector for V cache (CPU path)

#ifdef HAS_CUDA
    // --- Conditionally define device pointers --- 
    float* k_dev = nullptr; // Device pointer for K cache of this layer
    float* v_dev = nullptr; // Device pointer for V cache of this layer
#endif
};

struct KVCache {
    std::vector<KVCacheLayer> layers; // One per hidden layer
    int seq_len = 0; // Current sequence length stored in the cache

#ifdef HAS_CUDA
    // Store dimensions needed for indexing and freeing memory
    int allocated_num_layers = 0;
    int allocated_max_seq_len = 0;
    int allocated_num_kv_heads = 0;
    int allocated_head_dim = 0;

    // Destructor to free CUDA memory
    ~KVCache() {
        if (allocated_num_layers > 0) { // Only free if initialized
            Logger::info("Freeing KVCache CUDA memory...");
            for (int l = 0; l < allocated_num_layers; ++l) {
                if (layers[l].k_dev) {
                    gpuErrchk(cudaFree(layers[l].k_dev));
                    layers[l].k_dev = nullptr;
                }
                if (layers[l].v_dev) {
                    gpuErrchk(cudaFree(layers[l].v_dev));
                    layers[l].v_dev = nullptr;
                }
            }
            Logger::info("KVCache CUDA memory freed.");
        }
    }
#endif

    // Initialize method declaration (remove implementation body)
    void initialize(int num_layers, int max_seq_len, int num_kv_heads, int head_dim);
};

// Forward diagnostic callback type
using ForwardDiagCallback = std::function<void(int layer, const std::string& name, const std::vector<float>& v)>;

// LayerWeights struct for transformer layers
struct LayerWeights {
    std::vector<uint16_t> input_layernorm;         // [hidden_size]
    std::vector<uint16_t> post_attention_layernorm;// [hidden_size]
    // Attention
    std::vector<uint16_t> q_proj; // [hidden_size, hidden_size]
    std::vector<uint16_t> k_proj; // [hidden_size, kv_dim]
    std::vector<uint16_t> v_proj; // [hidden_size, kv_dim]
    std::vector<uint16_t> o_proj; // [hidden_size, hidden_size]
    // MLP
    std::vector<uint16_t> gate_proj; // [intermediate_size, hidden_size]
    std::vector<uint16_t> up_proj;   // [intermediate_size, hidden_size]
    std::vector<uint16_t> down_proj; // [hidden_size, intermediate_size]

    // --- ADDED: FP32 and Quantized fields ---
    std::vector<float> input_layernorm_f32;
    std::vector<float> post_attention_layernorm_f32;
    std::vector<float> q_proj_f32, k_proj_f32, v_proj_f32, o_proj_f32;
    std::vector<float> gate_proj_f32, up_proj_f32, down_proj_f32;
    std::vector<block_q4_K> q_proj_q4k, k_proj_q4k, v_proj_q4k, o_proj_q4k;
    std::vector<block_q4_K> gate_proj_q4k, up_proj_q4k, down_proj_q4k;
    std::vector<block_q6_K> q_proj_q6k, k_proj_q6k, v_proj_q6k, o_proj_q6k;
    std::vector<block_q6_K> gate_proj_q6k, up_proj_q6k, down_proj_q6k;

#ifdef HAS_CUDA
    // Device pointers for RMSNorm weights
    float* input_layernorm_dev = nullptr;
    float* post_attention_layernorm_dev = nullptr;
#endif
};

class TinyLlamaModel {
public:
    // Construct from config and safetensors loader
    TinyLlamaModel(const ModelConfig& config, const SafeTensorsLoader& loader);
    // New: Construct from config and weights file path (auto-detect GGUF or safetensors)
    TinyLlamaModel(const ModelConfig& config, const std::string& weights_path);
    ~TinyLlamaModel(); // ADD Destructor declaration

    // --- Forward Pass (NOW uses std::vector<float>) --- 
    std::vector<float> forward(std::vector<float>& x_vec, int pos, KVCache* cache = nullptr, const std::vector<int>* attention_mask = nullptr);

    // New: Device-only forward pass for incremental GPU pipeline
    std::vector<float> forward_device(int token_id, int pos, KVCache* cache, const std::vector<int>* attention_mask = nullptr, cudaStream_t stream = 0);

    // Get model config
    const ModelConfig& get_config() const { return config_; }

    // Getter for lm_head weights
    const std::vector<uint16_t>& get_lm_head() const { return lm_head; }

    // Getter for embed_tokens weights
    const std::vector<uint16_t>& get_embed_tokens() const { return embed_tokens; }

    // Getter for layers (for debugging)
    std::vector<LayerWeights>& get_layers() { return layers; } // Return mutable reference

    // Lookup embedding for a token (returns std::vector<float>)
    std::vector<float> lookup_embedding(int token_id);

    int get_vocab_size() const;

    const GGUFData* get_gguf_data() const { return gguf_data_ ? gguf_data_.get() : nullptr; }

    friend void map_gguf_weights(const GGUFData& gguf, TinyLlamaModel& model); // Declare friend

private:
    ModelConfig config_;

    // Weight storage (changed to uint16_t for bfloat16)
    std::vector<uint16_t> embed_tokens; // [vocab_size, hidden_size]
    std::vector<uint16_t> lm_head;      // [vocab_size, hidden_size]
    std::vector<uint16_t> final_norm;   // [hidden_size]

    // --- ADDED: FP32 and Quantized fields ---
    std::vector<float> embed_tokens_f32, lm_head_f32, final_norm_f32;
    std::vector<block_q4_K> embed_tokens_q4k, lm_head_q4k, final_norm_q4k;
    std::vector<block_q6_K> embed_tokens_q6k, lm_head_q6k, final_norm_q6k;

    std::vector<LayerWeights> layers; // num_hidden_layers

#ifdef HAS_CUDA
    // Device pointer for final RMSNorm weights
    float* final_norm_dev = nullptr;
    float* all_freqs_cis_dev = nullptr; // NEW: Persistent device buffer for all RoPE freqs
    // --- START Persistent Device Weights (BF16) ---
    uint16_t* token_embedding_table_dev_ = nullptr;
    uint16_t* w_q_dev_ = nullptr;
    uint16_t* w_k_dev_ = nullptr;
    uint16_t* w_v_dev_ = nullptr;
    uint16_t* w_o_dev_ = nullptr;
    uint16_t* w_gate_dev_ = nullptr;
    uint16_t* w_up_dev_ = nullptr;
    uint16_t* w_down_dev_ = nullptr;
    uint16_t* lm_head_dev_ = nullptr;
    // --- END Persistent Device Weights ---
    // --- START Persistent Device Weights (FP32) ---
    float* token_embedding_table_f32_dev_ = nullptr;
    float* w_q_f32_dev_ = nullptr;
    float* w_k_f32_dev_ = nullptr;
    float* w_v_f32_dev_ = nullptr;
    float* w_o_f32_dev_ = nullptr;
    float* w_gate_f32_dev_ = nullptr;
    float* w_up_f32_dev_ = nullptr;
    float* w_down_f32_dev_ = nullptr;
    float* lm_head_f32_dev_ = nullptr;
    // --- END Persistent Device Weights (FP32) ---
    cublasHandle_t cublas_handle_ = nullptr; // <<< ADD CUBLAS HANDLE >>>
    
    // --- START: REMOVE DUPLICATE DECLARATIONS ---
    /*
    // GPU device pointers for persistent weights (BF16)
    uint16_t* token_embedding_table_dev_ = nullptr;
    uint16_t* lm_head_dev_ = nullptr;
    uint16_t* w_q_dev_ = nullptr;
    // ... (other BF16 weights)
    uint16_t* w_down_dev_ = nullptr;

    // GPU device pointers for persistent weights (FP32)
    float* token_embedding_table_f32_dev_ = nullptr;
    float* lm_head_f32_dev_ = nullptr;
    float* w_q_f32_dev_ = nullptr;
    // ... (other FP32 weights)
    float* w_down_f32_dev_ = nullptr;

    // GPU device pointers for layer norm weights
    // (Managed within LayerWeights struct for layers, plus final_norm_dev)
    float* final_norm_dev = nullptr;
    // NOTE: Layer norm weights are stored in LayerWeights::*_dev

    // GPU device pointer for RoPE frequencies
    float* all_freqs_cis_dev = nullptr;
    */
    // --- END: REMOVE DUPLICATE DECLARATIONS ---

    // --- START: Added Persistent Workspace Buffers ---
    float* x_dev_ = nullptr;           // Input/State vector
    float* x_norm_dev_ = nullptr;      // Output of RMSNorm
    float* x_resid1_dev_ = nullptr;    // Residual connection 1 (before attention)
    float* x_resid2_dev_ = nullptr;    // Residual connection 2 (before MLP)
    float* q_dev_ = nullptr;           // Q projection output
    float* k_dev_ = nullptr;           // K projection output (current token)
    float* v_dev_ = nullptr;           // V projection output (current token)
    float* attn_out_dev_ = nullptr;    // Attention output (weighted sum of V)
    float* attn_proj_dev_ = nullptr;   // Attention projection (O proj)
    float* gate_vec_dev_ = nullptr;    // MLP gate projection
    float* up_vec_dev_ = nullptr;      // MLP up projection
    float* swiglu_vec_dev_ = nullptr;  // Output of SwiGLU
    float* mlp_down_dev_ = nullptr;    // MLP down projection
    float* logits_dev_ = nullptr;      // Final logits output
    // --- END: Added Persistent Workspace Buffers ---
#endif

    // Precomputed RoPE cos/sin values
    std::vector<std::pair<float, float>> precomputed_freqs_cis_;

    // Internal helper for attention calculation (if still needed)
    /*std::vector<float> forward_from_qkv_host(
        const std::vector<float>& x_resid1_vec, 
        std::vector<float>& q_vec,
        std::vector<float>& k_vec,
        std::vector<float>& v_vec,
        int pos,
        KVCache* cache,
        const std::vector<int>* attention_mask);*/ // Commented out if unused

    std::unique_ptr<GGUFData> gguf_data_; // Only set if loaded from GGUF

    // --- ADDED: Private Helper Declarations ---
    void initialize_weights(const SafeTensorsLoader* loader, const GGUFData* gguf);
    void initialize_gpu_and_rope();
};

// Utility: parse ModelConfig from nlohmann::json
ModelConfig parse_model_config(const nlohmann::json& json);

// Helper function declaration
int argmax(const std::vector<float>& v);

// Convert a single bfloat16 (uint16_t) to float32
float bfloat16_to_float32(uint16_t b16);

// rmsnorm declaration
void rmsnorm(const std::vector<float>& x, const std::vector<uint16_t>& weight, float eps, std::vector<float>& out);

// matvec_bf16_f32 declaration
void matvec_bf16_f32(const std::vector<uint16_t>& mat, const std::vector<float>& vec, std::vector<float>& out, int M, int N);

// softmax declaration
void softmax(std::vector<float>& x);

// Forward declaration for KVCache structure
struct KVCache;

// Helper function declarations (make them non-static)
float bfloat16_to_float32(uint16_t b16);
std::vector<uint16_t> uint8_vector_to_uint16_vector(const std::vector<uint8_t>& bytes, size_t numel);

// --- FIX: Add declaration for logging helper ---
void log_vector_summary(const std::string& name, const std::vector<float>& v, int head_count = 5);

#endif // MODEL_H 