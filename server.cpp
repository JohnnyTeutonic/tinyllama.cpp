// Disable warnings specific to cpp-httplib in MSVC
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4244) // conversion from 'x' to 'y', possible loss of data
#pragma warning(disable : 4267) // 'var' : conversion from 'size_t' to 'type', possible loss of data
#pragma warning(disable : 4996) // The POSIX name for this item is deprecated
#endif

#define CPPHTTPLIB_OPENSSL_SUPPORT // Enable SSL if needed (requires OpenSSL dev libraries)
// Or #define CPPHTTPLIB_THREAD_POOL_COUNT 8 // Define thread pool count
#include "httplib.h"

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include "api.h"
#include "logger.h"
#include <nlohmann/json.hpp> // For JSON parsing/creation
#include <string>
#include <memory>   // For std::shared_ptr
#include <thread>   // For std::thread::hardware_concurrency
#include <vector>
#include <filesystem> // For checking www path

// Use nlohmann::json for convenience
using json = nlohmann::json;

int main(int argc, char** argv) {
    // --- Configuration --- 
    std::string model_dir = "data"; // Default model directory
    std::string host = "localhost";
    int port = 8080;
    std::string www_path = "./www"; // Directory for static client files

    // --- Argument Parsing (Simple) --- 
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

    // --- Load Model via API --- 
    std::shared_ptr<tinyllama::TinyLlamaSession> session;
    try {
        Logger::info("Loading model from: " + model_dir);
        session = std::make_shared<tinyllama::TinyLlamaSession>(model_dir);
        Logger::info("Model loaded successfully.");
    } catch (const std::exception& e) {
        Logger::error(std::string("Failed to load model: ") + e.what());
        return 1;
    }

    // --- HTTP Server Setup --- 
    httplib::Server svr;

    // --- Mount Static File Directory --- 
    if (std::filesystem::exists(www_path) && std::filesystem::is_directory(www_path)) {
        Logger::info("Serving static files from: " + www_path);
        bool mount_ok = svr.set_mount_point("/", www_path);
        if (!mount_ok) {
             Logger::error("Failed to mount static file directory: " + www_path);
             return 1;
        }
    } else {
         Logger::info("Static file directory not found: " + www_path + ". Web client will not be served.");
    }

    // --- API Endpoint: /chat --- 
    svr.Post("/chat", [&session](const httplib::Request& req, httplib::Response& res) {
        Logger::info("Received request for /chat");
        res.set_header("Access-Control-Allow-Origin", "*"); // Basic CORS
        res.set_header("Access-Control-Allow-Methods", "POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type");

        std::string user_message;
        std::vector<std::pair<std::string, std::string>> history; // Placeholder for future history
        float temperature = 0.7f;
        int max_new_tokens = 100;
        int top_k = 50;
        float top_p = 0.9f;

        try {
            // Parse request JSON
            json req_json = json::parse(req.body);
            if (req_json.contains("message")) {
                user_message = req_json["message"].get<std::string>();
            } else {
                 throw std::runtime_error("Missing 'message' field in request JSON");
            }
             // Remove parsing of unused sampling parameters
             // if (req_json.contains("temperature")) temperature = req_json["temperature"].get<float>();
             // if (req_json.contains("max_new_tokens")) max_new_tokens = req_json["max_new_tokens"].get<int>();
             // if (req_json.contains("top_k")) top_k = req_json["top_k"].get<int>();
             // if (req_json.contains("top_p")) top_p = req_json["top_p"].get<float>();
             
             // Still parse max_new_tokens as it's used in the loop limit
             if (req_json.contains("max_new_tokens")) max_new_tokens = req_json["max_new_tokens"].get<int>();
             
             // Optional: Add history parsing here if client sends it

             Logger::info("Processing message: " + user_message.substr(0, 50) + "...");
             
             // Construct the prompt using the expected Q&A format
             std::string prompt = "Q: " + user_message + "\nA:";
             Logger::info("Formatted prompt: " + prompt.substr(0, 100) + "..."); // Log formatted prompt

             // Call generate method (remove unused sampling parameters)
             std::string reply = session->generate(prompt, max_new_tokens /*, temperature, top_k, top_p */);
             Logger::info("Generated reply: " + reply.substr(0, 50) + "...");

            // Create response JSON
            json res_json;
            res_json["reply"] = reply;

            // Send response
            res.set_content(res_json.dump(), "application/json");
            Logger::info("Response sent successfully.");

        } catch (const json::parse_error& e) {
            Logger::error("JSON parsing error: " + std::string(e.what()));
            res.status = 400; // Bad Request
            json err_json;
            err_json["error"] = "Invalid JSON format: " + std::string(e.what());
            res.set_content(err_json.dump(), "application/json");
        } catch (const std::exception& e) {
            Logger::error("Generation error: " + std::string(e.what()));
            res.status = 500; // Internal Server Error
            json err_json;
            err_json["error"] = "Internal server error: " + std::string(e.what());
            res.set_content(err_json.dump(), "application/json");
        }
    });

    // --- Add OPTIONS handler for CORS preflight --- 
    svr.Options("/chat", [](const httplib::Request& req, httplib::Response& res){
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Headers", "Content-Type");
        res.set_header("Access-Control-Allow-Methods", "POST, OPTIONS");
        res.status = 204; // No Content
    });

    // --- Start Server --- 
    unsigned int num_threads = std::max(1u, std::thread::hardware_concurrency() / 2); // Use half the cores
    Logger::info("Starting server on " + host + ":" + std::to_string(port) + " with " + std::to_string(num_threads) + " threads.");
    
    svr.listen(host.c_str(), port);

    // Server runs until stopped (e.g., Ctrl+C)
    Logger::info("Server stopped.");
    return 0;
} 