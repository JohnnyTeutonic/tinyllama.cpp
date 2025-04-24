#include "prompt.h"

// TinyLlama default chat template
static const char* TINYLLAMA_TEMPLATE =
    "<|system|>\n{system}</s>\n";

// Format a prompt for TinyLlama inference.
std::string format_prompt(
    const std::string& system,
    const std::vector<std::string>& user_messages,
    const std::vector<std::string>& assistant_messages,
    const std::string& chat_template
) {
    // For now, ignore chat_template and use the default
    std::string prompt;
    // System message
    prompt += "<|system|>\n" + system + "</s>\n";
    // Interleave user/assistant turns
    size_t n_turns = user_messages.size();
    for (size_t i = 0; i < n_turns; ++i) {
        prompt += "<|user|>\n" + user_messages[i] + "</s>\n";
        if (i < assistant_messages.size()) {
            prompt += "<|assistant|>\n" + assistant_messages[i] + "</s>\n";
        }
    }
    // For next response, add assistant block start
    prompt += "<|assistant|>\n";
    return prompt;
} 