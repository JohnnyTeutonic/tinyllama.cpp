#include "distributed_coordinator.h"
#include "api.h"
#include "logger.h"
#include <chrono>
#include <thread>
#include <sstream>
#include <algorithm>
#include <cstring>

#ifdef HAS_MPI
#include <mpi.h>
#endif

namespace tinyllama {

DistributedCoordinator::DistributedCoordinator(NodeRole role,
                                             const std::string& model_path,
                                             const std::string& tokenizer_path,
                                             int gpu_layers)
    : role_(role)
    , model_path_(model_path)
    , tokenizer_path_(tokenizer_path)
    , gpu_layers_(gpu_layers) {
}

DistributedCoordinator::~DistributedCoordinator() {
    shutdown();
}

bool DistributedCoordinator::initialize() {
#ifdef HAS_MPI
    int provided;
    int result = MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
    if (result != MPI_SUCCESS) {
        Logger::log("Failed to initialize MPI", Logger::ERROR);
        return false;
    }

    if (provided < MPI_THREAD_MULTIPLE) {
        Logger::log("MPI does not support multithreading", Logger::WARNING);
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &size_);
    
    mpi_initialized_ = true;

    Logger::log("MPI initialized - Rank: " + std::to_string(rank_) + 
                ", Size: " + std::to_string(size_), Logger::INFO);

    // Discover cluster topology
    discover_cluster();

    return true;
#else
    Logger::log("MPI support not compiled in", Logger::ERROR);
    return false;
#endif
}

void DistributedCoordinator::shutdown() {
#ifdef HAS_MPI
    if (mpi_initialized_) {
        // Send shutdown signal to all nodes
        if (rank_ == 0) {
            for (int i = 1; i < size_; ++i) {
                int shutdown_signal = 1;
                MPI_Send(&shutdown_signal, 1, MPI_INT, i, TAG_SHUTDOWN, MPI_COMM_WORLD);
            }
        }
        
        MPI_Finalize();
        mpi_initialized_ = false;
        Logger::log("MPI finalized", Logger::INFO);
    }
#endif
}

bool DistributedCoordinator::is_mpi_available() const {
#ifdef HAS_MPI
    return mpi_initialized_;
#else
    return false;
#endif
}

void DistributedCoordinator::discover_cluster() {
#ifdef HAS_MPI
    if (!mpi_initialized_) return;

    cluster_nodes_.clear();
    cluster_nodes_.resize(size_);

    // Get hostname
    char hostname[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(hostname, &name_len);

    // Create node info for this node
    NodeInfo local_info;
    local_info.rank = rank_;
    local_info.hostname = std::string(hostname, name_len);
    local_info.role = role_;
    local_info.has_gpu = true; // Assume GPU available - could be detected
    local_info.gpu_memory_mb = 8192; // Default - could be detected
    local_info.gpu_layers = gpu_layers_;

    // Serialize local node info
    std::string serialized_info = local_info.hostname + "|" + 
                                 std::to_string(static_cast<int>(local_info.role)) + "|" +
                                 std::to_string(local_info.has_gpu ? 1 : 0) + "|" +
                                 std::to_string(local_info.gpu_memory_mb) + "|" +
                                 std::to_string(local_info.gpu_layers);

    // Gather all node info at rank 0
    std::vector<char> send_buffer(serialized_info.begin(), serialized_info.end());
    send_buffer.resize(256, '\0'); // Fixed size buffer

    std::vector<char> recv_buffer;
    if (rank_ == 0) {
        recv_buffer.resize(256 * size_);
    }

    MPI_Gather(send_buffer.data(), 256, MPI_CHAR, 
               recv_buffer.data(), 256, MPI_CHAR, 0, MPI_COMM_WORLD);

    // Parse and distribute node info
    if (rank_ == 0) {
        for (int i = 0; i < size_; ++i) {
            std::string node_data(recv_buffer.data() + i * 256);
            // Remove null terminators
            node_data = node_data.substr(0, node_data.find('\0'));
            
            std::istringstream ss(node_data);
            std::string hostname, role_str, has_gpu_str, memory_str, layers_str;
            
            if (std::getline(ss, hostname, '|') &&
                std::getline(ss, role_str, '|') &&
                std::getline(ss, has_gpu_str, '|') &&
                std::getline(ss, memory_str, '|') &&
                std::getline(ss, layers_str)) {
                
                cluster_nodes_[i].rank = i;
                cluster_nodes_[i].hostname = hostname;
                cluster_nodes_[i].role = static_cast<NodeRole>(std::stoi(role_str));
                cluster_nodes_[i].has_gpu = (std::stoi(has_gpu_str) == 1);
                cluster_nodes_[i].gpu_memory_mb = std::stoul(memory_str);
                cluster_nodes_[i].gpu_layers = std::stoi(layers_str);
            }
        }
    }

    // Broadcast cluster info to all nodes
    if (rank_ == 0) {
        for (int i = 1; i < size_; ++i) {
            MPI_Send(recv_buffer.data(), recv_buffer.size(), MPI_CHAR, i, TAG_NODE_INFO, MPI_COMM_WORLD);
        }
    } else {
        recv_buffer.resize(256 * size_);
        MPI_Status status;
        MPI_Recv(recv_buffer.data(), recv_buffer.size(), MPI_CHAR, 0, TAG_NODE_INFO, MPI_COMM_WORLD, &status);
        
        // Parse received cluster info
        for (int i = 0; i < size_; ++i) {
            std::string node_data(recv_buffer.data() + i * 256);
            node_data = node_data.substr(0, node_data.find('\0'));
            
            std::istringstream ss(node_data);
            std::string hostname, role_str, has_gpu_str, memory_str, layers_str;
            
            if (std::getline(ss, hostname, '|') &&
                std::getline(ss, role_str, '|') &&
                std::getline(ss, has_gpu_str, '|') &&
                std::getline(ss, memory_str, '|') &&
                std::getline(ss, layers_str)) {
                
                cluster_nodes_[i].rank = i;
                cluster_nodes_[i].hostname = hostname;
                cluster_nodes_[i].role = static_cast<NodeRole>(std::stoi(role_str));
                cluster_nodes_[i].has_gpu = (std::stoi(has_gpu_str) == 1);
                cluster_nodes_[i].gpu_memory_mb = std::stoul(memory_str);
                cluster_nodes_[i].gpu_layers = std::stoi(layers_str);
            }
        }
    }

    Logger::log("Discovered " + std::to_string(cluster_nodes_.size()) + " nodes in cluster", Logger::INFO);
    for (const auto& node : cluster_nodes_) {
        Logger::log("Node " + std::to_string(node.rank) + ": " + node.hostname + 
                   " (GPU: " + (node.has_gpu ? "yes" : "no") + 
                   ", Layers: " + std::to_string(node.gpu_layers) + ")", Logger::INFO);
    }
#endif
}

bool DistributedCoordinator::submit_request(const InferenceRequest& request,
                                          std::function<void(const InferenceResult&)> callback) {
#ifdef HAS_MPI
    if (!mpi_initialized_ || rank_ != 0) {
        Logger::log("Only master node (rank 0) can submit requests", Logger::ERROR);
        return false;
    }

    // Store callback for later
    pending_callbacks_[request.request_id] = callback;

    // Select optimal worker node
    int target_node = select_optimal_node(request);
    if (target_node == -1) {
        Logger::log("No available worker nodes", Logger::ERROR);
        InferenceResult error_result;
        error_result.request_id = request.request_id;
        error_result.success = false;
        error_result.error_message = "No available worker nodes";
        callback(error_result);
        return false;
    }

    // Send request to worker
    bool sent = send_request_to_worker(target_node, request);
    if (!sent) {
        Logger::log("Failed to send request to worker node " + std::to_string(target_node), Logger::ERROR);
        InferenceResult error_result;
        error_result.request_id = request.request_id;
        error_result.success = false;
        error_result.error_message = "Failed to send request to worker";
        callback(error_result);
        return false;
    }

    total_requests_++;
    return true;
#else
    // Fallback to local processing if MPI not available
    InferenceResult result = process_inference_locally(request);
    callback(result);
    return result.success;
#endif
}

int DistributedCoordinator::select_optimal_node(const InferenceRequest& request) {
    if (request.target_node != -1 && request.target_node < size_) {
        return request.target_node;
    }

    // Simple load balancing - select node with least active requests
    int best_node = -1;
    int min_load = INT_MAX;

    for (const auto& node : cluster_nodes_) {
        if (node.rank == 0) continue; // Skip master node
        if (node.role == NodeRole::WORKER || node.role == NodeRole::HYBRID) {
            int current_load = node.active_requests.load();
            if (current_load < min_load) {
                min_load = current_load;
                best_node = node.rank;
            }
        }
    }

    return best_node;
}

bool DistributedCoordinator::send_request_to_worker(int worker_rank, const InferenceRequest& request) {
#ifdef HAS_MPI
    std::vector<char> serialized = serialize_request(request);
    
    // Send size first
    int size = static_cast<int>(serialized.size());
    int result = MPI_Send(&size, 1, MPI_INT, worker_rank, TAG_REQUEST, MPI_COMM_WORLD);
    if (result != MPI_SUCCESS) {
        return false;
    }

    // Send data
    result = MPI_Send(serialized.data(), size, MPI_CHAR, worker_rank, TAG_REQUEST, MPI_COMM_WORLD);
    return result == MPI_SUCCESS;
#else
    return false;
#endif
}

void DistributedCoordinator::run_worker() {
#ifdef HAS_MPI
    if (!mpi_initialized_) {
        Logger::log("MPI not initialized", Logger::ERROR);
        return;
    }

    Logger::log("Starting worker node (rank " + std::to_string(rank_) + ")", Logger::INFO);

    while (true) {
        // Check for shutdown signal
        int flag;
        MPI_Status status;
        MPI_Iprobe(0, TAG_SHUTDOWN, MPI_COMM_WORLD, &flag, &status);
        if (flag) {
            int shutdown_signal;
            MPI_Recv(&shutdown_signal, 1, MPI_INT, 0, TAG_SHUTDOWN, MPI_COMM_WORLD, &status);
            Logger::log("Received shutdown signal", Logger::INFO);
            break;
        }

        // Check for incoming requests
        if (receive_and_process_request()) {
            // Request processed successfully
        } else {
            // No request available, sleep briefly
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    Logger::log("Worker node shutting down", Logger::INFO);
#endif
}

void DistributedCoordinator::run_master() {
#ifdef HAS_MPI
    if (!mpi_initialized_ || rank_ != 0) {
        Logger::log("Only rank 0 can run as master", Logger::ERROR);
        return;
    }

    Logger::log("Starting master node", Logger::INFO);

    while (true) {
        // Receive results from workers
        if (receive_result_from_worker()) {
            // Result processed successfully
        } else {
            // No result available, sleep briefly
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        // Send periodic heartbeats
        static auto last_heartbeat = std::chrono::steady_clock::now();
        auto now = std::chrono::steady_clock::now();
        if (now - last_heartbeat > std::chrono::seconds(5)) {
            send_heartbeat();
            last_heartbeat = now;
        }
    }
#endif
}

bool DistributedCoordinator::receive_and_process_request() {
#ifdef HAS_MPI
    int flag;
    MPI_Status status;
    
    // Check if there's a request available
    MPI_Iprobe(0, TAG_REQUEST, MPI_COMM_WORLD, &flag, &status);
    if (!flag) {
        return false;
    }

    // Receive size first
    int size;
    MPI_Recv(&size, 1, MPI_INT, 0, TAG_REQUEST, MPI_COMM_WORLD, &status);

    // Receive data
    std::vector<char> buffer(size);
    MPI_Recv(buffer.data(), size, MPI_CHAR, 0, TAG_REQUEST, MPI_COMM_WORLD, &status);

    // Deserialize request
    InferenceRequest request = deserialize_request(buffer);

    // Update active request count
    if (rank_ < static_cast<int>(cluster_nodes_.size())) {
        cluster_nodes_[rank_].active_requests++;
    }

    // Process request locally
    InferenceResult result = process_inference_locally(request);
    result.processing_node = rank_;

    // Send result back to master
    send_result_to_master(result);

    // Update active request count
    if (rank_ < static_cast<int>(cluster_nodes_.size())) {
        cluster_nodes_[rank_].active_requests--;
    }

    return true;
#else
    return false;
#endif
}

bool DistributedCoordinator::receive_result_from_worker() {
#ifdef HAS_MPI
    int flag;
    MPI_Status status;
    
    // Check if there's a result available from any worker
    MPI_Iprobe(MPI_ANY_SOURCE, TAG_RESULT, MPI_COMM_WORLD, &flag, &status);
    if (!flag) {
        return false;
    }

    // Receive size first
    int size;
    MPI_Recv(&size, 1, MPI_INT, status.MPI_SOURCE, TAG_RESULT, MPI_COMM_WORLD, &status);

    // Receive data
    std::vector<char> buffer(size);
    MPI_Recv(buffer.data(), size, MPI_CHAR, status.MPI_SOURCE, TAG_RESULT, MPI_COMM_WORLD, &status);

    // Deserialize result
    InferenceResult result = deserialize_result(buffer);

    // Find and call the callback
    auto callback_it = pending_callbacks_.find(result.request_id);
    if (callback_it != pending_callbacks_.end()) {
        callback_it->second(result);
        pending_callbacks_.erase(callback_it);
        
        if (result.success) {
            completed_requests_++;
        } else {
            failed_requests_++;
        }
    }

    return true;
#else
    return false;
#endif
}

bool DistributedCoordinator::send_result_to_master(const InferenceResult& result) {
#ifdef HAS_MPI
    std::vector<char> serialized = serialize_result(result);
    
    // Send size first
    int size = static_cast<int>(serialized.size());
    int mpi_result = MPI_Send(&size, 1, MPI_INT, 0, TAG_RESULT, MPI_COMM_WORLD);
    if (mpi_result != MPI_SUCCESS) {
        return false;
    }

    // Send data
    mpi_result = MPI_Send(serialized.data(), size, MPI_CHAR, 0, TAG_RESULT, MPI_COMM_WORLD);
    return mpi_result == MPI_SUCCESS;
#else
    return false;
#endif
}

std::vector<char> DistributedCoordinator::serialize_request(const InferenceRequest& request) {
    std::ostringstream oss;
    oss << request.request_id << "|"
        << request.prompt << "|"
        << request.max_tokens << "|"
        << request.temperature << "|"
        << request.top_k << "|"
        << request.top_p << "|"
        << request.system_prompt << "|"
        << request.target_node;
    
    std::string str = oss.str();
    return std::vector<char>(str.begin(), str.end());
}

DistributedCoordinator::InferenceRequest DistributedCoordinator::deserialize_request(const std::vector<char>& data) {
    std::string str(data.begin(), data.end());
    std::istringstream iss(str);
    
    InferenceRequest request;
    std::string temp;
    
    if (std::getline(iss, request.request_id, '|') &&
        std::getline(iss, request.prompt, '|') &&
        std::getline(iss, temp, '|')) {
        request.max_tokens = std::stoi(temp);
    }
    if (std::getline(iss, temp, '|')) {
        request.temperature = std::stof(temp);
    }
    if (std::getline(iss, temp, '|')) {
        request.top_k = std::stoi(temp);
    }
    if (std::getline(iss, temp, '|')) {
        request.top_p = std::stof(temp);
    }
    if (std::getline(iss, request.system_prompt, '|') &&
        std::getline(iss, temp)) {
        request.target_node = std::stoi(temp);
    }
    
    return request;
}

std::vector<char> DistributedCoordinator::serialize_result(const InferenceResult& result) {
    std::ostringstream oss;
    oss << result.request_id << "|"
        << result.generated_text << "|"
        << result.tokens_generated << "|"
        << result.inference_time_ms << "|"
        << result.processing_node << "|"
        << (result.success ? 1 : 0) << "|"
        << result.error_message;
    
    std::string str = oss.str();
    return std::vector<char>(str.begin(), str.end());
}

DistributedCoordinator::InferenceResult DistributedCoordinator::deserialize_result(const std::vector<char>& data) {
    std::string str(data.begin(), data.end());
    std::istringstream iss(str);
    
    InferenceResult result;
    std::string temp;
    
    if (std::getline(iss, result.request_id, '|') &&
        std::getline(iss, result.generated_text, '|') &&
        std::getline(iss, temp, '|')) {
        result.tokens_generated = std::stoi(temp);
    }
    if (std::getline(iss, temp, '|')) {
        result.inference_time_ms = std::stof(temp);
    }
    if (std::getline(iss, temp, '|')) {
        result.processing_node = std::stoi(temp);
    }
    if (std::getline(iss, temp, '|')) {
        result.success = (std::stoi(temp) == 1);
    }
    if (std::getline(iss, result.error_message)) {
        // Got error message
    }
    
    return result;
}

DistributedCoordinator::InferenceResult DistributedCoordinator::process_inference_locally(const InferenceRequest& request) {
    InferenceResult result;
    result.request_id = request.request_id;
    result.processing_node = rank_;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        // Use the existing TinyLlama API for local inference
        // This is a simplified version - you would integrate with your actual API
        
        // For now, create a mock result
        result.generated_text = "Generated response for: " + request.prompt;
        result.tokens_generated = request.max_tokens;
        result.success = true;
        
        // TODO: Replace with actual inference call
        // std::string response = tinyllama_generate(request.prompt, request.max_tokens, 
        //                                          request.temperature, request.top_k, request.top_p);
        // result.generated_text = response;
        
    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = e.what();
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    result.inference_time_ms = static_cast<float>(duration.count());
    
    return result;
}

void DistributedCoordinator::send_heartbeat() {
#ifdef HAS_MPI
    for (int i = 1; i < size_; ++i) {
        int heartbeat = 1;
        MPI_Send(&heartbeat, 1, MPI_INT, i, TAG_HEARTBEAT, MPI_COMM_WORLD);
    }
#endif
}

void DistributedCoordinator::handle_heartbeat(int source_rank) {
    // Update last seen time for the node
    if (source_rank >= 0 && source_rank < static_cast<int>(cluster_nodes_.size())) {
        // Could update a last_seen timestamp here
    }
}

std::vector<DistributedCoordinator::NodeInfo> DistributedCoordinator::get_cluster_info() const {
    return cluster_nodes_;
}

DistributedCoordinator::LoadStats DistributedCoordinator::get_load_stats() const {
    LoadStats stats;
    stats.total_requests = total_requests_.load();
    stats.completed_requests = completed_requests_.load();
    stats.failed_requests = failed_requests_.load();
    
    if (stats.completed_requests > 0) {
        stats.average_response_time_ms = 0.0f; // Would need to track this
    }
    
    for (const auto& node : cluster_nodes_) {
        stats.requests_per_node[node.rank] = node.active_requests.load();
    }
    
    return stats;
}

} // namespace tinyllama
