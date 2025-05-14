/**
 * @file main.cpp
 * @brief Command-line interface for TinyLlama text generation.
 *
 * This program provides a command-line interface for generating text using
 * TinyLlama models. It supports both GGUF and SafeTensors model formats.
 * The program automatically applies Q:A formatting to prompts, which is
 * required for proper model responses in both formats.
 *
 * Usage:
 *   tinyllama [model_path] [prompt] [steps] [temperature] [top_k] [top_p] [cpu_layers]
 *
 * Arguments:
 *   model_path: Path to model directory or .gguf file (default: data)
 *   prompt: Input text (default: "Hello, world!")
 *   steps: Number of tokens to generate (default: 64)
 *   temperature: Sampling temperature, lower is more deterministic (default: 0.1)
 *   top_k: Limit sampling to top K tokens (default: 40)
 *   top_p: Limit sampling to top P probability mass (default: 0.9)
 *   cpu_layers: Number of layers to offload to CPU (default: 0)
 *
 * Example:
 *   ./tinyllama data "What is the capital of France?" 64 0.1 40 0.9 0
 *
 * Note:
 *   The program always applies Q:A formatting to prompts ("Q: [prompt]\nA:")
 *   as this is required for proper responses from both GGUF and SafeTensors models.
 */

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "api.h"
#include "logger.h"

std::string trim_whitespace(const std::string& s) {
  auto start = std::find_if_not(
      s.begin(), s.end(), [](unsigned char c) { return std::isspace(c); });
  auto end = std::find_if_not(s.rbegin(), s.rend(), [](unsigned char c) {
               return std::isspace(c);
             }).base();
  return (start < end ? std::string(start, end) : std::string());
}

void print_usage(const char* program_name) {
  std::cout << "Usage: " << program_name
            << " <model_path> <tokenizer_path> <num_threads> <prompt|chat> "
               "[initial_prompt_string] [max_tokens] [n_gpu_layers] [use_mmap] [temperature]"
            << std::endl;
  std::cout << "\nArguments:\n"
               "  model_path          : Path to the model file (.gguf) or directory (SafeTensors).\n"
               "  tokenizer_path      : Path to the tokenizer file.\n"
               "  num_threads         : Number of threads to use for generation.\n"
               "  prompt|chat         : 'prompt' for single prompt generation or 'chat' for chat mode.\n"
               "  initial_prompt_string: (Optional) Initial prompt string. Default: \"Hello, world!\".\n"
               "  max_tokens          : (Optional) Maximum number of tokens to generate. Default: 256.\n"
               "  n_gpu_layers        : (Optional) Number of layers to offload to GPU (-1 for all, 0 for none). Default: -1.\n"
               "  use_mmap            : (Optional) Use mmap for GGUF files ('true' or 'false'). Default: true.\n"
               "  temperature         : (Optional) Sampling temperature. Default: 0.1.\n"
            << std::endl;
}

int main(int argc, char** argv) {
  if (argc > 1 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
    print_usage(argv[0]);
    return 0;
  }

  if (argc < 5) { // Minimum required: model_path, tokenizer_path, num_threads, mode
    std::cerr << "ERROR: Missing required arguments." << std::endl;
    print_usage(argv[0]);
    return 1;
  }

  std::string model_path_or_dir = argv[1];
  std::string tokenizer_path = argv[2];
  int num_threads = 4; // Default
  try {
    num_threads = std::stoi(argv[3]);
  } catch (const std::exception& e) {
    Logger::error("Invalid num_threads argument: " + std::string(argv[3]) +
                  ". Using default: " + std::to_string(num_threads));
  }

  std::string mode_str = argv[4];
  std::string initial_prompt_string = "Hello, world!"; // Default
  int max_tokens = 256; // Default for steps
  int n_gpu_layers = -1; // Default: all layers on GPU
  bool use_mmap = true; // Default: use mmap
  
  // Default sampling params for generate
  float temperature = 0.1f; // Default temperature
  int top_k = 40;           // Default, not exposed via CLI in this version
  float top_p = 0.9f;       // Default, not exposed via CLI in this version

  if (argc > 5) {
    initial_prompt_string = trim_whitespace(argv[5]);
  }

  if (argc > 6) {
    try {
      max_tokens = std::stoi(argv[6]);
    } catch (const std::exception& e) {
      Logger::error("Invalid max_tokens argument: " + std::string(argv[6]) +
                    ". Using default: " + std::to_string(max_tokens));
    }
  }

  if (argc > 7) {
    try {
      n_gpu_layers = std::stoi(argv[7]);
    } catch (const std::invalid_argument& ia) {
      std::cerr << "ERROR: Invalid n_gpu_layers: " << argv[7] << std::endl;
      return 1;
    }
  }

  if (argc > 8) {
    std::string mmap_str = argv[8];
    std::transform(mmap_str.begin(), mmap_str.end(), mmap_str.begin(), ::tolower);
    if (mmap_str == "false" || mmap_str == "0") {
      use_mmap = false;
    } else if (mmap_str == "true" || mmap_str == "1") {
      use_mmap = true;
    } else {
      std::cerr << "ERROR: Invalid use_mmap value: " << argv[8]
                << ". Expected 'true', 'false', '1', or '0'." << std::endl;
      return 1;
    }
  }

  if (argc > 9) { // Temperature is argv[9]
    try {
      temperature = std::stof(argv[9]);
      if (temperature < 0.0f) {
        Logger::warning("Temperature cannot be negative. Using default: 0.1");
        temperature = 0.1f;
      }
    } catch (const std::exception& e) {
      Logger::error("Invalid temperature argument: " + std::string(argv[9]) +
                    ". Using default: " + std::to_string(temperature));
    }
  }

  Logger::info("Using model path/directory: " + model_path_or_dir);
  Logger::info("Tokenizer path: " + tokenizer_path);
  Logger::info("Num threads: " + std::to_string(num_threads));
  Logger::info("Mode: " + mode_str);
  Logger::info("Initial prompt/string: \"" + initial_prompt_string + "\"");
  Logger::info("Max tokens: " + std::to_string(max_tokens));
  Logger::info("N GPU Layers: " + std::to_string(n_gpu_layers));
  Logger::info(std::string("Use mmap: ") + (use_mmap ? "true" : "false"));
  Logger::info("Temperature: " + std::to_string(temperature));

  try {
    tinyllama::TinyLlamaSession session(model_path_or_dir, tokenizer_path, num_threads, n_gpu_layers, use_mmap);
    Logger::info("TinyLlamaSession initialized successfully.");

    const ModelConfig& config = session.get_config();
    bool apply_qa_formatting; // Explicitly set based on tokenizer family

    if (config.tokenizer_family == ModelConfig::TokenizerFamily::LLAMA3_TIKTOKEN) {
        apply_qa_formatting = false;
        Logger::info("[Main.cpp] Llama 3 model detected (via tokenizer_family). Q&A prompt formatting will be DISABLED for this session.");
    } else {
        apply_qa_formatting = true; // Default for non-Llama 3 / other models in this CLI context
        Logger::info("[Main.cpp] Non-Llama 3 model detected (via tokenizer_family). Q&A prompt formatting will be ENABLED for this session.");
    }

    // Log the final decision clearly, this combines previous [MAIN_CPP_DIAGNOSTIC] logs
    Logger::info("[Main.cpp] Mode: '" + mode_str + "'. Final decision for apply_qa_formatting: " + std::string(apply_qa_formatting ? "true" : "false"));

    if (mode_str == "prompt") {
      std::string generated_text =
          session.generate(initial_prompt_string, max_tokens, temperature, top_k, top_p, "", apply_qa_formatting);
      std::cout << generated_text << std::endl;
    } else if (mode_str == "chat") {
      std::cout << "Entering chat mode. Type 'exit', 'quit' to end." << std::endl;
      std::string current_chat_prompt;
      if (!initial_prompt_string.empty() && initial_prompt_string != "Hello, world!") {
          current_chat_prompt = initial_prompt_string;
          std::cout << "AI: " << session.generate(current_chat_prompt, max_tokens, temperature, top_k, top_p, "", apply_qa_formatting) << std::endl;
      }
      while (true) {
        std::cout << "You: ";
        std::getline(std::cin, current_chat_prompt);
        if (current_chat_prompt == "exit" || current_chat_prompt == "quit") {
          break;
        }
        if (current_chat_prompt.empty()) {
          continue;
        }
        std::string ai_response = session.generate(current_chat_prompt, max_tokens, temperature, top_k, top_p, "", apply_qa_formatting);
        std::cout << "AI: " << ai_response << std::endl;
      }
    } else {
        std::cerr << "ERROR: Invalid mode '" << mode_str << "'. Expected 'prompt' or 'chat'." << std::endl;
        print_usage(argv[0]);
        return 1;
    }

  } catch (const std::exception& e) {
    Logger::error("Error: " + std::string(e.what()));
    return 1;
  }

  return 0;
}