#include <iomanip>  
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "api.h"
#include "logger.h"

int main(int argc, char** argv) {
  if (argc < 2) {
    Logger::error("Usage: " + std::string(argv[0]) +
                  " <path_to_gguf_file> [prompt] [steps] [temperature]");
    Logger::error("Example: " + std::string(argv[0]) +
                  " ./models/tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf \"Who is "
                  "awesome?\" 64 0.7");
    std::cerr << "Usage: " << argv[0]
              << " <path_to_gguf_file> [prompt] [steps] [temperature]"
              << std::endl;
    return 1;
  }

  std::string gguf_model_path = argv[1];
  std::string prompt = "What is the capital of France?";  
  int steps = 64;            
  float temperature = 0.7f;  

  if (argc > 2) {
    prompt = argv[2];  
  }
  if (argc > 3) {
    try {
      steps = std::stoi(argv[3]);
    } catch (const std::exception& e) {
      Logger::error("Invalid steps argument: " + std::string(argv[3]) +
                    ". Using default: " + std::to_string(steps));
    }
  }
  if (argc > 4) {
    try {
      temperature = std::stof(argv[4]);
    } catch (const std::exception& e) {
      Logger::error("Invalid temperature argument: " + std::string(argv[4]) +
                    ". Using default: " + std::to_string(temperature));
    }
  }

  
  Logger::info("Using GGUF model file: " + gguf_model_path);
  Logger::info("Raw Prompt (to be formatted by API): \"" + prompt + "\"");
  Logger::info("Steps (max new tokens): " + std::to_string(steps));
  Logger::info("Temperature: " + std::to_string(temperature));

  try {
    
    
    
    tinyllama::TinyLlamaSession session(gguf_model_path);
    Logger::info("TinyLlamaSession initialized successfully for GGUF model.");

    
    
    
    std::string generated_text = session.generate(prompt, steps, temperature, "", true);

    
    
    std::cout << "Q: " << prompt << "\nA:" << generated_text << std::endl;

  } catch (const std::exception& e) {
    Logger::error("An error occurred: " + std::string(e.what()));
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
