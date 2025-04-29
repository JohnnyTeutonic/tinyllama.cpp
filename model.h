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
#include "cuda_kernels.h" // Needed for gpuErrchk
#endif

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

// KVCache for autoregressive inference
struct KVCacheLayer {
#ifdef HAS_CUDA
    float* k_dev = nullptr; // Device pointer for K cache of this layer
    float* v_dev = nullptr; // Device pointer for V cache of this layer
#else
    std::vector<float> k; // Host vector for K cache (CPU path)
    std::vector<float> v; // Host vector for V cache (CPU path)
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
    ~TinyLlamaModel(); // ADD Destructor declaration

    // --- Forward Pass (NOW uses std::vector<float>) --- 
    std::vector<float> forward(std::vector<float>& x_vec, int pos, KVCache* cache = nullptr, const std::vector<int>* attention_mask = nullptr);

    // New: Device-only forward pass for incremental GPU pipeline
    std::vector<float> forward_device(int token_id, int pos, KVCache* cache, const std::vector<int>* attention_mask);

    // Get model config
    const ModelConfig& get_config() const { return config_; }

    // Getter for lm_head weights
    const std::vector<uint16_t>& get_lm_head() const { return lm_head; }

    // Getter for embed_tokens weights
    const std::vector<uint16_t>& get_embed_tokens() const { return embed_tokens; }

    // Getter for layers (for debugging)
    const std::vector<LayerWeights>& get_layers() const { return layers; }

    // Lookup embedding for a token (returns std::vector<float>)
    std::vector<float> lookup_embedding(int token_id);

    int get_vocab_size() const;

private:
    ModelConfig config_;

    // Weight storage (changed to uint16_t for bfloat16)
    std::vector<uint16_t> embed_tokens; // [vocab_size, hidden_size]
    std::vector<uint16_t> lm_head;      // [vocab_size, hidden_size]
    std::vector<uint16_t> final_norm;   // [hidden_size]

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
#endif

    // Precomputed RoPE cos/sin values
    std::vector<std::pair<float, float>> precomputed_freqs_cis_;

    std::vector<float> forward_from_qkv_host(
        const std::vector<float>& x_resid1_vec, // Input embedding for residual 1
        std::vector<float>& q_vec,
        std::vector<float>& k_vec,
        std::vector<float>& v_vec,
        int pos,
        KVCache* cache,
        const std::vector<int>* attention_mask);
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