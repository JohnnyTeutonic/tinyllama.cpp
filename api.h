#ifndef TINYLLAMA_API_H
#define TINYLLAMA_API_H

#include <string>
#include <vector>
#include <memory> // For std::unique_ptr
#include "model.h"
#include "tokenizer.h"
#include <stdexcept> // Required for std::runtime_error

// Forward declarations - Keep these for the direct members
// class Tokenizer; // Already included via tokenizer.h
// struct ModelConfig; // Already included via model.h
// class TinyLlamaModel; // Already included via model.h
struct KVCache; // Keep if KVCache definition isn't fully visible via model.h (safer to keep)

namespace tinyllama {

/**
 * @brief Represents an active TinyLlama session holding the loaded model and tokenizer.
 */
class TinyLlamaSession {
public:
    /**
     * @brief Loads the model, config, and tokenizer from the specified directory or GGUF file.
     * @param model_path Path to the directory containing model files OR a .gguf file.
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
     * @param temperature Sampling temperature. Lower values are more deterministic.
     * @param system_prompt Optional system prompt to guide the generation.
     * @return The generated text string (excluding the prompt).
     * @throws std::runtime_error if generation fails.
     */
    std::string generate(const std::string& prompt,
                         int steps = 128,
                         float temperature = 0.7f, // Adjusted default
                         const std::string& system_prompt = "");

private:
    // REMOVED PImpl - Direct members now
    // struct SessionImpl; 
    // std::unique_ptr<SessionImpl> pimpl_;

    // Prevent copying/assignment
    TinyLlamaSession(const TinyLlamaSession&) = delete;
    TinyLlamaSession& operator=(const TinyLlamaSession&) = delete;

    // Direct Members
    std::unique_ptr<TinyLlamaModel> model_; // Use unique_ptr for RAII
    std::unique_ptr<Tokenizer> tokenizer_;
    ModelConfig config_;
    KVCache kv_cache_;
    int eos_token_id_;
};

} // namespace tinyllama

#endif // TINYLLAMA_API_H 