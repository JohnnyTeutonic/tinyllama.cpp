#ifndef PROMPT_H
#define PROMPT_H

#include <string>
#include <vector>

// Format a prompt for TinyLlama inference using Q: A: style only.
// - question: a single user question
// Returns the formatted prompt string in Q: A: format.
std::string format_prompt(const std::string& question);

#endif // PROMPT_H 