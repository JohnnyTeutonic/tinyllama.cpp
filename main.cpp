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
 *   tinyllama <model_path> <tokenizer_path> <num_threads> <prompt|chat> [--system-prompt <system_prompt_string>] [initial_user_prompt] [max_tokens] [n_gpu_layers] [use_mmap] [temperature] [use_kv_quant]
 *
 * Arguments:
 *   model_path: Path to the model file (.gguf) or directory (SafeTensors).
 *   tokenizer_path: Path to the tokenizer file.
 *   num_threads: Number of threads to use for generation.
 *   prompt|chat: 'prompt' for single prompt generation or 'chat' for interactive chat mode.
 *   --system-prompt: (Optional) System prompt to guide the model. Default: empty.
 *   initial_user_prompt: (Optional) Initial user prompt string. Default: "Hello, world!"
 *   max_tokens: (Optional) Maximum number of tokens to generate. Default: 256
 *   n_gpu_layers: (Optional) Number of layers to offload to GPU (-1 for all, 0 for none). Default: -1 (all layers on GPU if available)
 *   use_mmap: (Optional) Use mmap for GGUF files ('true' or 'false'). Default: true
 *   temperature: (Optional) Sampling temperature (e.g., 0.1). Lower is more deterministic. Default: 0.1
 *   use_kv_quant: (Optional) Use INT8 KVCache quantization on GPU ('true' or 'false'). Default: false
 *
 * Example:
 *   ./tinyllama ./models/model.gguf ./models/tokenizer.model 4 prompt --system-prompt "You are a helpful assistant." "What is the capital of France?" 128 0 true 0.1 true
 *
 * Note:
 *   The program may automatically apply Q:A formatting to prompts (e.g., "Q: [prompt]\nA:")
 *   depending on the model type, as this can be required for proper responses.
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
               "[--system-prompt <system_prompt_string>] [initial_user_prompt] [max_tokens] [n_gpu_layers] [use_mmap] [temperature] [use_kv_quant] [use_batch_generation]"
            << std::endl;
  std::cout << "\nArguments:\n"
               "  model_path          : Path to the model file (.gguf) or directory (SafeTensors).\n"
               "  tokenizer_path      : Path to the tokenizer file.\n"
               "  num_threads         : Number of threads to use for generation.\n"
               "  prompt|chat         : 'prompt' for single prompt generation or 'chat' for chat mode.\n"
               "  --system-prompt     : (Optional) System prompt to guide the model. Default: empty.\n"
               "  initial_user_prompt : (Optional) Initial user prompt string. Default: \"Hello, world!\".\n"
               "  max_tokens          : (Optional) Maximum number of tokens to generate. Default: 256.\n"
               "  n_gpu_layers        : (Optional) Number of layers to offload to GPU (-1 for all, 0 for none). Default: -1.\n"
               "  use_mmap            : (Optional) Use mmap for GGUF files ('true' or 'false'). Default: true.\n"
               "  temperature         : (Optional) Sampling temperature. Default: 0.1.\n"
               "  use_kv_quant        : (Optional) Use INT8 KVCache quantization on GPU ('true' or 'false'). Default: false.\n"
               "  use_batch_generation: (Optional) Use GPU batch generation for tokens ('true' or 'false'). Default: false.\n"
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
  int num_threads = 4; // Default, will be overwritten if provided by argv[3]
  try {
      if (argc > 3) { // Ensure argv[3] (number of threads) exists
          num_threads = std::stoi(argv[3]);
      } else {
          Logger::warning("Number of threads not provided, using default: " + std::to_string(num_threads));
      }
  } catch (const std::exception& e) {
      Logger::warning("Could not parse num_threads from argv[3]: '" + (argc > 3 ? std::string(argv[3]) : "<not provided>") + "'. Using default: " + std::to_string(num_threads));
  }
  
  if (argc < 5) { // Check after attempting to parse num_threads
      std::cerr << "ERROR: Missing mode argument (prompt|chat) after num_threads." << std::endl;
      print_usage(argv[0]);
      return 1;
  }
  std::string mode_str = argv[4]; // Mode (prompt|chat) is argv[4]

  std::string system_prompt_str = ""; // Default empty system prompt
  std::string user_prompt_str = "Hello, world!"; // Default user prompt
  int max_tokens = 256; // Default for steps
  int n_gpu_layers = -1; // Default: all layers on GPU
  bool use_mmap = true; // Default: use mmap
  bool use_kv_quant = false; // Default: do not use KVCache quantization
  bool use_batch_generation = false; // Default: do not use batch generation
  
  // Default sampling params for generate
  float temperature = 0.1f; // Default temperature
  int top_k = 40;           // Default, not exposed via CLI in this version
  float top_p = 0.9f;       // Default, not exposed via CLI in this version

  int current_arg_idx = 5;
  while(current_arg_idx < argc) {
    std::string arg = argv[current_arg_idx];
    if (arg == "--system-prompt" || arg == "-sp") {
      if (current_arg_idx + 1 < argc) {
        system_prompt_str = argv[current_arg_idx + 1];
        current_arg_idx += 2;
      } else {
        std::cerr << "ERROR: --system-prompt requires a value." << std::endl;
        print_usage(argv[0]);
        return 1;
      }
    } else if (arg == "--max-tokens" || arg == "-mt") {
        if (current_arg_idx + 1 < argc) {
            try { max_tokens = std::stoi(argv[current_arg_idx+1]); }
            catch (const std::exception& e) { Logger::error("Invalid max_tokens: " + std::string(argv[current_arg_idx+1])); }
            current_arg_idx += 2;
        } else { std::cerr << "ERROR: --max-tokens requires a value." << std::endl; return 1;}
    } else if (arg == "--n-gpu-layers" || arg == "-ngl") {
        if (current_arg_idx + 1 < argc) {
            try { n_gpu_layers = std::stoi(argv[current_arg_idx+1]); }
            catch (const std::exception& e) { Logger::error("Invalid n_gpu_layers: " + std::string(argv[current_arg_idx+1])); }
            current_arg_idx += 2;
        } else { std::cerr << "ERROR: --n-gpu-layers requires a value." << std::endl; return 1;}
    } else if (arg == "--use-mmap") {
         if (current_arg_idx + 1 < argc) {
            std::string mmap_str_val = argv[current_arg_idx+1];
            std::transform(mmap_str_val.begin(), mmap_str_val.end(), mmap_str_val.begin(), ::tolower);
            if (mmap_str_val == "false" || mmap_str_val == "0") use_mmap = false;
            else if (mmap_str_val == "true" || mmap_str_val == "1") use_mmap = true;
            else { std::cerr << "ERROR: Invalid use_mmap value." << std::endl; return 1; }
            current_arg_idx += 2;
        } else { std::cerr << "ERROR: --use-mmap requires a value." << std::endl; return 1;}
    } else if (arg == "--temperature" || arg == "-t") {
        if (current_arg_idx + 1 < argc) {
            try { temperature = std::stof(argv[current_arg_idx+1]); }
            catch (const std::exception& e) { Logger::error("Invalid temperature: " + std::string(argv[current_arg_idx+1]));}
            current_arg_idx += 2;
        } else { std::cerr << "ERROR: --temperature requires a value." << std::endl; return 1;}
    } else if (arg == "--use-kv-quant" || arg == "-kvq") {
         if (current_arg_idx + 1 < argc) {
            std::string kvq_str_val = argv[current_arg_idx+1];
            std::transform(kvq_str_val.begin(), kvq_str_val.end(), kvq_str_val.begin(), ::tolower);
            if (kvq_str_val == "false" || kvq_str_val == "0") use_kv_quant = false;
            else if (kvq_str_val == "true" || kvq_str_val == "1") use_kv_quant = true;
            else { std::cerr << "ERROR: Invalid use_kv_quant value: " << argv[current_arg_idx+1] << std::endl; return 1; }
            current_arg_idx += 2;
        } else { std::cerr << "ERROR: --use-kv-quant requires a value." << std::endl; return 1;}
    } else if (arg == "--use-batch-generation" || arg == "-ubg") {
         if (current_arg_idx + 1 < argc) {
            std::string ubg_str_val = argv[current_arg_idx+1];
            std::transform(ubg_str_val.begin(), ubg_str_val.end(), ubg_str_val.begin(), ::tolower);
            if (ubg_str_val == "false" || ubg_str_val == "0") use_batch_generation = false;
            else if (ubg_str_val == "true" || ubg_str_val == "1") use_batch_generation = true;
            else { std::cerr << "ERROR: Invalid use_batch_generation value: " << argv[current_arg_idx+1] << std::endl; return 1; }
            current_arg_idx += 2;
        } else { std::cerr << "ERROR: --use-batch-generation requires a value." << std::endl; return 1;}
    } else {
        if (user_prompt_str == "Hello, world!") {
             user_prompt_str = trim_whitespace(argv[current_arg_idx]);
        } else if (argv[current_arg_idx][0] != '-') {
             std::cerr << "ERROR: Unexpected positional argument: " << argv[current_arg_idx] << std::endl;
             print_usage(argv[0]);
             return 1;
        } else {
            std::cerr << "ERROR: Unknown option: " << argv[current_arg_idx] << std::endl;
            print_usage(argv[0]);
            return 1;
        }
        current_arg_idx++;
    }
  }

  Logger::info("Using model path/directory: " + model_path_or_dir);
  Logger::info("Tokenizer path: " + tokenizer_path);
  Logger::info("Num threads: " + std::to_string(num_threads));
  Logger::info("Mode: " + mode_str);
  Logger::info("System Prompt: \"" + system_prompt_str + "\"");
  Logger::info("Initial User Prompt: \"" + user_prompt_str + "\"");
  Logger::info("Max tokens: " + std::to_string(max_tokens));
  Logger::info("N GPU Layers: " + std::to_string(n_gpu_layers));
  Logger::info(std::string("Use mmap: ") + (use_mmap ? "true" : "false"));
  Logger::info("Temperature: " + std::to_string(temperature));
  Logger::info(std::string("Use KVCache Quantization: ") + (use_kv_quant ? "true" : "false"));
  Logger::info(std::string("Use Batch Generation: ") + (use_batch_generation ? "true" : "false"));

  try {
    tinyllama::TinyLlamaSession session(model_path_or_dir, tokenizer_path, num_threads, n_gpu_layers, use_mmap, use_kv_quant, use_batch_generation, 1);
    Logger::info("TinyLlamaSession initialized successfully.");

    const ModelConfig& config = session.get_config();
    bool apply_qa_formatting_decision; // This will be true if no advanced template is used

    if (config.tokenizer_family == ModelConfig::TokenizerFamily::LLAMA3_TIKTOKEN || \
        (session.get_tokenizer() && !session.get_tokenizer()->get_gguf_chat_template().empty())) { // Use getter
        apply_qa_formatting_decision = false; // Llama 3 or GGUF template handles formatting
        Logger::info("[Main.cpp] Llama 3 model or GGUF chat template detected. Internal Q&A prompt formatting will be DISABLED.");
    } else {
        apply_qa_formatting_decision = true; // Default for other models without GGUF template
        Logger::info("[Main.cpp] Non-Llama 3 model and no GGUF chat template. Internal Q&A prompt formatting will be ENABLED.");
    }

    Logger::info("[Main.cpp] Mode: '" + mode_str + "'. Final decision for apply_qa_formatting_decision: " + std::string(apply_qa_formatting_decision ? "true" : "false"));

    if (mode_str == "prompt") {
      std::string generated_text =
          session.generate(user_prompt_str, max_tokens, temperature, top_k, top_p, system_prompt_str, apply_qa_formatting_decision);
      std::cout << generated_text << std::endl;
    } else if (mode_str == "chat") {
      std::cout << "Entering chat mode. System Prompt: \"" << system_prompt_str << "\". Type 'exit', 'quit' to end." << std::endl;
      std::string current_user_message;
      
      if (!user_prompt_str.empty() && (user_prompt_str != "Hello, world!" || !system_prompt_str.empty() )) {
          current_user_message = user_prompt_str;
          std::cout << "You: " << current_user_message << std::endl;
          std::string ai_response = session.generate(current_user_message, max_tokens, temperature, top_k, top_p, system_prompt_str, apply_qa_formatting_decision);
          std::cout << "AI: " << ai_response << std::endl;
      }

      while (true) {
        std::cout << "You: ";
        std::getline(std::cin, current_user_message);
        if (current_user_message == "exit" || current_user_message == "quit") {
          break;
        }
        if (current_user_message.empty()) {
          continue;
        }
        std::string ai_response = session.generate(current_user_message, max_tokens, temperature, top_k, top_p, system_prompt_str, apply_qa_formatting_decision);
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