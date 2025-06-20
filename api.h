#ifndef TINYLLAMA_API_H
#define TINYLLAMA_API_H

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <random>
#include <sstream>

#include "model.h"
#include "tokenizer.h"

struct KVCache;

namespace tinyllama {

/**
 * @brief Represents an active TinyLlama session holding the loaded model and
 * tokenizer.
 * 
 * This class provides a high-level interface for text generation using TinyLlama models.
 * It supports both GGUF and SafeTensors model formats, and handles model loading,
 * tokenization, and text generation with various sampling strategies.
 */
class TinyLlamaSession {
 public:
  /**
   * @brief Loads the model, config, and tokenizer from the specified directory
   * or GGUF file.
   * @param model_path Path to the directory containing model files OR a .gguf
   * file.
   * @param tokenizer_path Path to the tokenizer file.
   * @param threads Number of threads to use for model loading.
   * @param num_gpu_layers_from_cli Number of GPU layers to use from command-line arguments.
   * @param cli_use_mmap Whether to use mmap for loading the model.
   * @param use_kv_quant Whether to use INT8 quantization for the KVCache on GPU.
   * @param use_batch_generation Whether to enable single-token batch generation.
   * @param max_batch_size Maximum number of sequences for multi-prompt batch processing (default: 1).
   * @throws std::runtime_error if loading fails.
   */
  TinyLlamaSession(const std::string& model_path,
                   const std::string& tokenizer_path, int threads = 1,
                   int num_gpu_layers_from_cli = 0, bool cli_use_mmap = true,
                   bool use_kv_quant = false, bool use_batch_generation = false,
                   int max_batch_size = 1);

  /**
   * @brief Destructor to ensure proper cleanup (e.g., KVCache CUDA memory).
   */
  ~TinyLlamaSession();

  /**
   * @brief Generates text based on a given prompt.
   *
   * This method supports various text generation strategies through its sampling parameters.
   * For temperatures close to 0, it approaches deterministic/greedy sampling.
   * For higher temperatures with top-k and top-p, it produces more diverse outputs.
   *
   * The method can also apply Q:A formatting to the prompt, which is recommended for
   * both GGUF and SafeTensors models when used via the command-line interface.
   *
   * @param prompt The input prompt string.
   * @param steps The number of tokens to generate.
   * @param temperature Sampling temperature. Lower values (e.g., 0.1) make the output more focused and deterministic.
   * @param top_k Top-K sampling parameter. Limits sampling to K most likely tokens. Set to 0 or vocab_size to disable.
   * @param top_p Nucleus sampling parameter. Limits sampling to tokens comprising top P probability mass (0.0 to 1.0).
   * @param system_prompt Optional system prompt to guide the generation.
   * @param apply_q_a_format Whether to apply Q:A format ("Q: [prompt]\nA:"). Recommended true for command-line use.
   * @return The generated text string (excluding the prompt).
   * @throws std::runtime_error if generation fails.
   */
  std::string generate(const std::string& prompt, int steps = 128,
                      float temperature = 0.1f,
                      int top_k = 40,
                      float top_p = 0.9f,
                      const std::string& system_prompt = "",
                      bool apply_q_a_format = false);

  /**
   * @brief Generates text for multiple prompts in a single batch (parallel processing).
   *
   * This method processes multiple independent prompts simultaneously, providing significant
   * efficiency gains compared to sequential generate() calls. Each prompt is processed
   * independently with its own KV cache state.
   *
   * @param prompts Vector of input prompt strings to process in batch.
   * @param steps The number of tokens to generate for each prompt.
   * @param temperature Sampling temperature. Lower values make output more focused and deterministic.
   * @param top_k Top-K sampling parameter. Limits sampling to K most likely tokens.
   * @param top_p Nucleus sampling parameter. Limits sampling to tokens comprising top P probability mass.
   * @param system_prompt Optional system prompt applied to all prompts in the batch.
   * @param apply_q_a_format Whether to apply Q:A format to all prompts in the batch.
   * @return Vector of generated text strings, one for each input prompt.
   * @throws std::runtime_error if batch generation fails or if prompts vector is empty.
   */
  std::vector<std::string> generate_batch(const std::vector<std::string>& prompts,
                                         int steps = 128,
                                         float temperature = 0.1f,
                                         int top_k = 40,
                                         float top_p = 0.9f,
                                         const std::string& system_prompt = "",
                                         bool apply_q_a_format = false);

  const Tokenizer* get_tokenizer() const { return tokenizer_.get(); }
  const ModelConfig& get_config() const { return config_; }
  KVCache& get_kv_cache() { return kv_cache_; }

 private:
  TinyLlamaSession(const TinyLlamaSession&) = delete;
  TinyLlamaSession& operator=(const TinyLlamaSession&) = delete;

  // Parallel batch processing methods
  bool batch_prefill_parallel(const std::vector<std::vector<int>>& all_tokens,
                              const std::vector<int>& prompt_lengths,
                              std::vector<std::vector<float>>& batch_final_logits);

  bool batch_generation_parallel(const std::vector<int>& current_tokens,
                                 const std::vector<int>& token_positions,
                                 const std::vector<int>& original_sequence_indices,
                                 std::vector<std::vector<float>>& batch_logits);

  std::unique_ptr<TinyLlamaModel> model_;
  std::unique_ptr<Tokenizer> tokenizer_;
  ModelConfig config_;
  KVCache kv_cache_;
  int eos_token_id_;
  std::mt19937 rng_{std::random_device{}()};  // RNG for sampling
  int threads_;
  bool use_batch_generation_; // Enable batch generation
  std::stringstream generated_stream_; // Added for streaming output
  std::string generated_text_for_api_return_; // Added to accumulate full response
  
  // Batch processing support
  int max_batch_size_ = 1;  // Maximum number of sequences for batch processing
};

}  // namespace tinyllama

#endif