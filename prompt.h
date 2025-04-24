#ifndef PROMPT_H
#define PROMPT_H

#include <string>
#include <vector>

// Format a prompt for TinyLlama inference.
// - system: the system message (may be empty)
// - user_messages: vector of user messages (one per turn)
// - assistant_messages: vector of assistant messages (one per turn, may be empty for incomplete turn)
// - chat_template: if non-empty, use this template; otherwise use TinyLlama default
// Returns the formatted prompt string.
std::string format_prompt(
    const std::string& system,
    const std::vector<std::string>& user_messages,
    const std::vector<std::string>& assistant_messages,
    const std::string& chat_template = ""
);

#endif // PROMPT_H 