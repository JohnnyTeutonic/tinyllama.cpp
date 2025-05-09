#ifndef TINYLLAMA_API_H
#define TINYLLAMA_API_H

#include <memory>     
#include <stdexcept>  
#include <string>
#include <vector>

#include "model.h"
#include "tokenizer.h"





struct KVCache;  
                 

namespace tinyllama {

/**
 * @brief Represents an active TinyLlama session holding the loaded model and
 * tokenizer.
 */
class TinyLlamaSession {
 public:
  /**
   * @brief Loads the model, config, and tokenizer from the specified directory
   * or GGUF file.
   * @param model_path Path to the directory containing model files OR a .gguf
   * file.
   * @throws std::runtime_error if loading fails.
   */
  TinyLlamaSession(const std::string& model_path);

  /**
   * @brief Destructor to ensure proper cleanup (e.g., KVCache CUDA memory).
   */
  ~TinyLlamaSession();

  /**
   * @brief Generates text based on a given prompt.
   *
   * @param prompt The input prompt string.
   * @param steps The number of steps to generate.
   * @param temperature Sampling temperature. Lower values are more
   * deterministic.
   * @param system_prompt Optional system prompt to guide the generation.
   * @param apply_q_a_format Whether to apply Q&A format.
   * @return The generated text string (excluding the prompt).
   * @throws std::runtime_error if generation fails.
   */
  std::string generate(const std::string& prompt, int steps = 128,
                       float temperature = 0.7f,
                       const std::string& system_prompt = "",
                       bool apply_q_a_format = false);

  // Add public getters
  const Tokenizer* get_tokenizer() const { return tokenizer_.get(); }
  const ModelConfig& get_config() const { return config_; }

 private:
  
  
  

  
  TinyLlamaSession(const TinyLlamaSession&) = delete;
  TinyLlamaSession& operator=(const TinyLlamaSession&) = delete;

  
  std::unique_ptr<TinyLlamaModel> model_;  
  std::unique_ptr<Tokenizer> tokenizer_;
  ModelConfig config_;
  KVCache kv_cache_;
  int eos_token_id_;
};

}  

#endif  