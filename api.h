#ifndef TINYLLAMA_API_H
#define TINYLLAMA_API_H

#include <string>
#include <vector>
#include <memory> // For std::unique_ptr

// Forward declarations to avoid including heavy headers here
class Tokenizer;
struct ModelConfig;
class TinyLlamaModel;
struct KVCache;

namespace tinyllama {

/**
 * @brief Represents an active TinyLlama session holding the loaded model and tokenizer.
 */
class TinyLlamaSession {
public:
    /**
     * @brief Loads the model, config, and tokenizer from the specified directory.
     * @param model_dir Path to the directory containing config.json, tokenizer.json, and model.safetensors.
     * @throws std::runtime_error if loading fails.
     */
    explicit TinyLlamaSession(const std::string& model_dir);

    /**
     * @brief Destructor to ensure proper cleanup (e.g., KVCache CUDA memory).
     */
    ~TinyLlamaSession();

    /**
     * @brief Generates text based on a given prompt.
     * 
     * @param prompt The input prompt string.
     * @param max_new_tokens The maximum number of tokens to generate.
     * @param temperature Sampling temperature. Lower values are more deterministic.
     * @param top_k Top-k sampling parameter. Limits sampling to the k most likely tokens.
     * @param top_p Top-p (nucleus) sampling parameter. Limits sampling to a cumulative probability mass.
     * @return The generated text string (excluding the prompt).
     * @throws std::runtime_error if generation fails.
     */
    std::string generate(const std::string& prompt, 
                         int max_new_tokens = 100, 
                         float temperature = 0.7f, 
                         int top_k = 50, 
                         float top_p = 0.9f);

private:
    // Internal state - using PImpl idiom slightly to hide heavy includes
    struct SessionImpl; 
    std::unique_ptr<SessionImpl> pimpl_; 

    // Prevent copying/assignment
    TinyLlamaSession(const TinyLlamaSession&) = delete;
    TinyLlamaSession& operator=(const TinyLlamaSession&) = delete;
};

} // namespace tinyllama

#endif // TINYLLAMA_API_H 