/**
 * @file server.cpp
 * @brief HTTP server implementation for TinyLlama chat interface.
 *
 * This server provides a REST API for interacting with TinyLlama models.
 * It handles both GGUF and SafeTensors models, applying appropriate prompt
 * formatting for each:
 * - For GGUF models: Applies Q:A format
 * - For SafeTensors models: Uses the tokenizer's chat template
 *
 * The server exposes a /chat endpoint that accepts POST requests with JSON body:
 * {
 *   "user_input": "string",      // Required: The prompt text
 *   "temperature": float,        // Optional: Sampling temperature (default: 0.1)
 *   "max_new_tokens": int,       // Optional: Max tokens to generate (default: 60)
 *   "top_k": int,               // Optional: Top-K sampling parameter (default: 40)
 *   "top_p": float              // Optional: Top-P sampling parameter (default: 0.9)
 * }
 *
 * Usage:
 *   tinyllama_server [model_path] [port] [host] [www_path]
 *     model_path: Path to model directory or .gguf file (default: data)
 *     port: Server port (default: 8080)
 *     host: Host to bind to (default: localhost)
 *     www_path: Path to static web files (default: ./www)
 */

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4244)
#pragma warning(disable : 4267)  
                                 
#pragma warning(disable : 4996)  
#endif

#include "httplib.h"

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <filesystem>         
#include <memory>             
#include <nlohmann/json.hpp>  
#include <string>
#include <thread>  
#include <vector>

#include "api.h"
#include "logger.h"
#include "tokenizer.h"
#include "model_macros.h"

using json = nlohmann::json;

int main(int argc, char** argv) {
  std::string model_dir = "data";  
  std::string host = "localhost";
  int port = 8080;
  std::string www_path = "./www";  
  
  if (argc > 1) {
    model_dir = argv[1];
  }
  if (argc > 2) {
    port = std::stoi(argv[2]);
  }
  if (argc > 3) {
    host = argv[3];
  }
  if (argc > 4) {
    www_path = argv[4];
  }

  Logger::info("Starting TinyLlama Chat Server...");
  
  std::shared_ptr<tinyllama::TinyLlamaSession> session;
  try {
    Logger::info("Loading model from: " + model_dir);
    session = std::make_shared<tinyllama::TinyLlamaSession>(model_dir, "tokenizer.json", 4, -1, true);
    Logger::info("Model loaded successfully.");
  } catch (const std::exception& e) {
    Logger::error(std::string("Failed to load model: ") + e.what());
    return 1;
  }
  
  httplib::Server svr;
  
  if (std::filesystem::exists(www_path) &&
      std::filesystem::is_directory(www_path)) {
    Logger::info("Serving static files from: " + www_path);
    bool mount_ok = svr.set_mount_point("/", www_path);
    if (!mount_ok) {
      Logger::error("Failed to mount static file directory: " + www_path);
      return 1;
    }
  } else {
    Logger::info("Static file directory not found: " + www_path +
                 ". Web client will not be served.");
  }

  svr.Post("/chat", [&session](const httplib::Request& req,
                               httplib::Response& res) {
        Logger::info("Received request for /chat");
    res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type");

        std::string user_input_from_client;

    float temperature = 0.1f;  // Lower temperature for more focused chat responses
        int max_new_tokens = 60;
    int top_k = 40;           // Default top-k value
    float top_p = 0.9f;       // Default top-p value

        try {
          json req_json = json::parse(req.body);
          if (req_json.contains("user_input")) {
            user_input_from_client = req_json["user_input"].get<std::string>();
          } else {
            throw std::runtime_error("Missing 'user_input' field in request JSON");
          }
          
          if (req_json.contains("max_new_tokens"))
            max_new_tokens = req_json["max_new_tokens"].get<int>();
          if (req_json.contains("temperature"))
            temperature = req_json["temperature"].get<float>();
      if (req_json.contains("top_k"))
        top_k = req_json["top_k"].get<int>();
      if (req_json.contains("top_p"))
        top_p = req_json["top_p"].get<float>();

          Logger::info("Processing user input: " +
                       user_input_from_client.substr(0, 100) + "...");

          const ModelConfig& config = session->get_config();
          std::string prompt_for_session_generate;
          bool use_q_a_format_for_session_generate = false;

          const Tokenizer* tokenizer = session->get_tokenizer(); 

          if (config.is_gguf_file_loaded) {
        prompt_for_session_generate = user_input_from_client;
        use_q_a_format_for_session_generate = true;
        Logger::info(
            "GGUF model detected. Using Q:A: format via session->generate.");
          } else {
        std::string system_prompt_text = "You are a helpful AI.";
            if (tokenizer) {
          prompt_for_session_generate = tokenizer->apply_chat_template(
              user_input_from_client, system_prompt_text, config);
          Logger::info(
              "Safetensors model detected. Applied chat template via "
              "tokenizer. Prompt: " +
              prompt_for_session_generate.substr(0, 200) + "...");
            } else {
          Logger::error(
              "CRITICAL: Tokenizer not available for Safetensors model in "
              "server. Cannot apply chat template.");

              prompt_for_session_generate = user_input_from_client; 
            }
        use_q_a_format_for_session_generate = false;
          }
          
          std::string reply = session->generate(
          prompt_for_session_generate, max_new_tokens, temperature, top_k, top_p, "",
          use_q_a_format_for_session_generate);
          Logger::info("Generated reply: " + reply.substr(0, 50) + "...");

          json res_json;
          res_json["reply"] = reply;
          
          res.set_content(res_json.dump(), "application/json");
          Logger::info("Response sent successfully.");

        } catch (const json::parse_error& e) {
          Logger::error("JSON parsing error: " + std::string(e.what()));
          res.status = 400;  
          json err_json;
          err_json["error"] = "Invalid JSON format: " + std::string(e.what());
          res.set_content(err_json.dump(), "application/json");
        } catch (const std::exception& e) {
          Logger::error("Generation error: " + std::string(e.what()));
          res.status = 500;  
          json err_json;
          err_json["error"] = "Internal server error: " + std::string(e.what());
          res.set_content(err_json.dump(), "application/json");
        }
      });
  
  svr.Options("/chat", [](const httplib::Request& req, httplib::Response& res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_header("Access-Control-Allow-Headers", "Content-Type");
    res.set_header("Access-Control-Allow-Methods", "POST, OPTIONS");
    res.status = 204;  
  });

  unsigned int num_threads = SAFE_MAX(1u, std::thread::hardware_concurrency() / 2);
  Logger::info("Starting server on " + host + ":" + std::to_string(port) +
               " with " + std::to_string(num_threads) + " threads.");

  svr.listen(host.c_str(), port);
  
  Logger::info("Server stopped.");
  return 0;
}