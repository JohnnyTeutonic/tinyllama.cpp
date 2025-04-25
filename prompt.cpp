#include "prompt.h"

// TinyLlama default chat template
static const char* TINYLLAMA_TEMPLATE =
    "<|system|>\n{system}</s>\n";

// Format a prompt for TinyLlama inference using Q: A: style only.
std::string format_prompt(const std::string& question) {
    return "Q: " + question + "\nA:";
} 