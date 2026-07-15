#pragma once

#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <map>
#include <atomic>

#ifdef HAS_MPI
#include <mpi.h>
#endif

namespace tinyllama {

/**
 * @brief Distributed coordinator for multi-node LLM inference
 * 
 * This class manages distributed inference across multiple nodes using MPI.
 * It handles model sharding, request routing, and result aggregation.
 */
class DistributedCoordinator {
public:
    enum class NodeRole {
        MASTER,     // Coordinates requests and aggregates results
        WORKER,     // Processes inference requests
        HYBRID      // Can act as both master and worker
    };

    struct NodeInfo {
        int rank;
        std::string hostname;
        NodeRole role;
        bool has_gpu;
        size_t gpu_memory_mb;
        int gpu_layers;
        std::atomic<bool> is_busy{false};
        std::atomic<int> active_requests{0};
    };

    struct InferenceRequest {
        std::string request_id;
        std::string prompt;
        int max_tokens;
        float temperature;
        int top_k;
        float top_p;
        std::string system_prompt;
        int target_node = -1;  // -1 for auto-selection
    };

    struct InferenceResult {
        std::string request_id;
        std::string generated_text;
        int tokens_generated;
        float inference_time_ms;
        int processing_node;
        bool success;
        std::string error_message;
    };

    /**
     * @brief Initialize the distributed coordinator
     * @param role The role of this node
     * @param model_path Path to the model file
     * @param tokenizer_path Path to the tokenizer file
     * @param gpu_layers Number of GPU layers for this node
     */
    DistributedCoordinator(NodeRole role, 
                          const std::string& model_path,
                          const std::string& tokenizer_path,
                          int gpu_layers = -1);

    ~DistributedCoordinator();

    /**
     * @brief Initialize MPI and discover network topology
     * @return true if initialization successful
     */
    bool initialize();

    /**
     * @brief Shutdown MPI and cleanup resources
     */
    void shutdown();

    /**
     * @brief Submit an inference request (master node only)
     * @param request The inference request
     * @param callback Callback function for result
     * @return true if request was submitted successfully
     */
    bool submit_request(const InferenceRequest& request,
                       std::function<void(const InferenceResult&)> callback);

    /**
     * @brief Process requests as a worker node
     * This function blocks and processes incoming requests
     */
    void run_worker();

    /**
     * @brief Run as master node, coordinating requests
     * This function blocks and coordinates the distributed system
     */
    void run_master();

    /**
     * @brief Get information about all nodes in the cluster
     */
    std::vector<NodeInfo> get_cluster_info() const;

    /**
     * @brief Get the rank of this node
     */
    int get_rank() const { return rank_; }

    /**
     * @brief Get the total number of nodes
     */
    int get_size() const { return size_; }

    /**
     * @brief Check if MPI is available and initialized
     */
    bool is_mpi_available() const;

    /**
     * @brief Get load balancing statistics
     */
    struct LoadStats {
        int total_requests;
        int completed_requests;
        int failed_requests;
        float average_response_time_ms;
        std::map<int, int> requests_per_node;
    };
    LoadStats get_load_stats() const;

private:
    NodeRole role_;
    std::string model_path_;
    std::string tokenizer_path_;
    int gpu_layers_;
    
    int rank_ = -1;
    int size_ = -1;
    bool mpi_initialized_ = false;
    
    std::vector<NodeInfo> cluster_nodes_;
    std::map<std::string, std::function<void(const InferenceResult&)>> pending_callbacks_;
    
    // Load balancing
    std::atomic<int> total_requests_{0};
    std::atomic<int> completed_requests_{0};
    std::atomic<int> failed_requests_{0};
    
    // MPI message tags
    static constexpr int TAG_REQUEST = 1;
    static constexpr int TAG_RESULT = 2;
    static constexpr int TAG_HEARTBEAT = 3;
    static constexpr int TAG_SHUTDOWN = 4;
    static constexpr int TAG_NODE_INFO = 5;

    /**
     * @brief Discover cluster topology and node capabilities
     */
    void discover_cluster();

    /**
     * @brief Select optimal node for processing request
     */
    int select_optimal_node(const InferenceRequest& request);

    /**
     * @brief Send request to worker node
     */
    bool send_request_to_worker(int worker_rank, const InferenceRequest& request);

    /**
     * @brief Receive and process request (worker node)
     */
    bool receive_and_process_request();

    /**
     * @brief Send result back to master
     */
    bool send_result_to_master(const InferenceResult& result);

    /**
     * @brief Receive result from worker (master node)
     */
    bool receive_result_from_worker();

    /**
     * @brief Serialize request for MPI transmission
     */
    std::vector<char> serialize_request(const InferenceRequest& request);

    /**
     * @brief Deserialize request from MPI transmission
     */
    InferenceRequest deserialize_request(const std::vector<char>& data);

    /**
     * @brief Serialize result for MPI transmission
     */
    std::vector<char> serialize_result(const InferenceResult& result);

    /**
     * @brief Deserialize result from MPI transmission
     */
    InferenceResult deserialize_result(const std::vector<char>& data);

    /**
     * @brief Process inference request locally
     */
    InferenceResult process_inference_locally(const InferenceRequest& request);

    /**
     * @brief Send heartbeat to all nodes
     */
    void send_heartbeat();

    /**
     * @brief Handle heartbeat from other nodes
     */
    void handle_heartbeat(int source_rank);
};

} // namespace tinyllama
