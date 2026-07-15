/**
 * @file distributed_main.cpp
 * @brief Distributed inference main program for TinyLlama
 *
 * This program provides distributed inference capabilities using MPI.
 * It can run as either a master node (coordinating requests) or worker nodes
 * (processing inference requests).
 *
 * Usage:
 *   mpirun -np 4 ./tinyllama_distributed <model_path> <tokenizer_path> <role> [gpu_layers]
 *
 * Arguments:
 *   model_path: Path to the model file (.gguf) or directory (SafeTensors).
 *   tokenizer_path: Path to the tokenizer file.
 *   role: Node role - "master", "worker", or "hybrid"
 *   gpu_layers: (Optional) Number of layers to offload to GPU. Default: -1 (all layers)
 *
 * Examples:
 *   # Run with 4 nodes (1 master, 3 workers)
 *   mpirun -np 4 ./tinyllama_distributed ./models/model.gguf ./models/tokenizer.model worker -1
 *   
 *   # Run with hybrid nodes (all can be master or worker)
 *   mpirun -np 4 ./tinyllama_distributed ./models/model.gguf ./models/tokenizer.model hybrid -1
 *
 * Environment Variables:
 *   TINYLLAMA_LOG_LEVEL: Set logging level (DEBUG, INFO, WARNING, ERROR)
 */

#include <iostream>
#include <string>
#include <thread>
#include <chrono>
#include <signal.h>
#include <atomic>

#include "distributed_coordinator.h"
#include "logger.h"

using namespace tinyllama;

// Global flag for graceful shutdown
std::atomic<bool> shutdown_requested{false};

void signal_handler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        std::cout << "\nShutdown requested..." << std::endl;
        shutdown_requested = true;
    }
}

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <model_path> <tokenizer_path> <role> [gpu_layers]\n"
              << "\n"
              << "Arguments:\n"
              << "  model_path     Path to the model file (.gguf) or directory (SafeTensors)\n"
              << "  tokenizer_path Path to the tokenizer file\n"
              << "  role          Node role: 'master', 'worker', or 'hybrid'\n"
              << "  gpu_layers    (Optional) Number of GPU layers (-1 for all, 0 for none). Default: -1\n"
              << "\n"
              << "Examples:\n"
              << "  mpirun -np 4 " << program_name << " ./model.gguf ./tokenizer.model worker -1\n"
              << "  mpirun -np 4 " << program_name << " ./model.gguf ./tokenizer.model hybrid 32\n"
              << std::endl;
}

DistributedCoordinator::NodeRole parse_role(const std::string& role_str) {
    if (role_str == "master") {
        return DistributedCoordinator::NodeRole::MASTER;
    } else if (role_str == "worker") {
        return DistributedCoordinator::NodeRole::WORKER;
    } else if (role_str == "hybrid") {
        return DistributedCoordinator::NodeRole::HYBRID;
    } else {
        throw std::invalid_argument("Invalid role: " + role_str + ". Must be 'master', 'worker', or 'hybrid'");
    }
}

void run_interactive_master(DistributedCoordinator& coordinator) {
    std::cout << "\n=== TinyLlama Distributed Inference (Master Node) ===" << std::endl;
    std::cout << "Type 'quit' to exit, 'status' for cluster status, or enter prompts for inference." << std::endl;
    std::cout << "Cluster info:" << std::endl;
    
    auto cluster_info = coordinator.get_cluster_info();
    for (const auto& node : cluster_info) {
        std::cout << "  Node " << node.rank << " (" << node.hostname << "): ";
        switch (node.role) {
            case DistributedCoordinator::NodeRole::MASTER:
                std::cout << "MASTER";
                break;
            case DistributedCoordinator::NodeRole::WORKER:
                std::cout << "WORKER";
                break;
            case DistributedCoordinator::NodeRole::HYBRID:
                std::cout << "HYBRID";
                break;
        }
        std::cout << ", GPU: " << (node.has_gpu ? "yes" : "no")
                  << ", Layers: " << node.gpu_layers << std::endl;
    }
    std::cout << std::endl;

    std::string input;
    int request_counter = 0;

    while (!shutdown_requested) {
        std::cout << "tinyllama> ";
        std::getline(std::cin, input);

        if (input.empty()) continue;

        if (input == "quit" || input == "exit") {
            break;
        }

        if (input == "status") {
            auto stats = coordinator.get_load_stats();
            std::cout << "Cluster Status:" << std::endl;
            std::cout << "  Total requests: " << stats.total_requests << std::endl;
            std::cout << "  Completed: " << stats.completed_requests << std::endl;
            std::cout << "  Failed: " << stats.failed_requests << std::endl;
            std::cout << "  Active requests per node:" << std::endl;
            for (const auto& pair : stats.requests_per_node) {
                std::cout << "    Node " << pair.first << ": " << pair.second << std::endl;
            }
            continue;
        }

        // Create inference request
        DistributedCoordinator::InferenceRequest request;
        request.request_id = "req_" + std::to_string(++request_counter);
        request.prompt = input;
        request.max_tokens = 256;
        request.temperature = 0.1f;
        request.top_k = 40;
        request.top_p = 0.9f;
        request.system_prompt = "";

        std::cout << "Processing request..." << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();

        // Submit request
        bool submitted = coordinator.submit_request(request, 
            [start_time](const DistributedCoordinator::InferenceResult& result) {
                auto end_time = std::chrono::high_resolution_clock::now();
                auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

                std::cout << "\n--- Response ---" << std::endl;
                if (result.success) {
                    std::cout << result.generated_text << std::endl;
                    std::cout << "\nStats:" << std::endl;
                    std::cout << "  Tokens: " << result.tokens_generated << std::endl;
                    std::cout << "  Inference time: " << result.inference_time_ms << "ms" << std::endl;
                    std::cout << "  Total time: " << total_time.count() << "ms" << std::endl;
                    std::cout << "  Processed by node: " << result.processing_node << std::endl;
                } else {
                    std::cout << "Error: " << result.error_message << std::endl;
                }
                std::cout << "tinyllama> ";
                std::cout.flush();
            });

        if (!submitted) {
            std::cout << "Failed to submit request!" << std::endl;
        }

        // Give some time for the request to be processed
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

int main(int argc, char* argv[]) {
    // Set up signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // Parse command line arguments
    if (argc < 4) {
        print_usage(argv[0]);
        return 1;
    }

    std::string model_path = argv[1];
    std::string tokenizer_path = argv[2];
    std::string role_str = argv[3];
    int gpu_layers = (argc > 4) ? std::stoi(argv[4]) : -1;

    // Set up logging
    const char* log_level_env = std::getenv("TINYLLAMA_LOG_LEVEL");
    if (log_level_env) {
        std::string log_level(log_level_env);
        if (log_level == "DEBUG") {
            Logger::set_level(Logger::DEBUG);
        } else if (log_level == "INFO") {
            Logger::set_level(Logger::INFO);
        } else if (log_level == "WARNING") {
            Logger::set_level(Logger::WARNING);
        } else if (log_level == "ERROR") {
            Logger::set_level(Logger::ERROR);
        }
    }

    try {
        // Parse role
        DistributedCoordinator::NodeRole role = parse_role(role_str);

        // Create distributed coordinator
        DistributedCoordinator coordinator(role, model_path, tokenizer_path, gpu_layers);

        // Initialize MPI and discover cluster
        if (!coordinator.initialize()) {
            std::cerr << "Failed to initialize distributed coordinator" << std::endl;
            return 1;
        }

        int rank = coordinator.get_rank();
        int size = coordinator.get_size();

        Logger::log("Node " + std::to_string(rank) + "/" + std::to_string(size) + 
                   " initialized with role: " + role_str, Logger::INFO);

        // Run based on role and rank
        if (role == DistributedCoordinator::NodeRole::MASTER || 
            (role == DistributedCoordinator::NodeRole::HYBRID && rank == 0)) {
            
            // Master node - handle interactive input and coordinate requests
            std::thread master_thread([&coordinator]() {
                coordinator.run_master();
            });

            // Run interactive interface on main thread
            run_interactive_master(coordinator);

            // Signal shutdown and wait for master thread
            shutdown_requested = true;
            master_thread.join();

        } else if (role == DistributedCoordinator::NodeRole::WORKER ||
                   role == DistributedCoordinator::NodeRole::HYBRID) {
            
            // Worker node - process incoming requests
            Logger::log("Starting worker mode", Logger::INFO);
            coordinator.run_worker();
        }

        // Cleanup
        coordinator.shutdown();
        Logger::log("Node shutdown complete", Logger::INFO);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
