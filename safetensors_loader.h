#ifndef SAFETENSORS_LOADER_H
#define SAFETENSORS_LOADER_H

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

#include <functional>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <queue>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

struct ModelConfig; // Forward declaration

/**
 * @file safetensors_loader.h
 * @brief SafeTensors format loader for efficient tensor loading
 *
 * This file implements a loader for the SafeTensors format, which is a safe and
 * efficient way to serialize tensors. The implementation uses memory mapping for
 * efficient loading and provides both sequential and parallel loading capabilities.
 */

class ThreadPool;

/**
 * @brief Main class for loading tensors from SafeTensors format files
 * 
 * Provides functionality to load tensor data from SafeTensors files using memory
 * mapping for efficient access. Supports both sequential and parallel loading
 * of tensors.
 */
class SafeTensorsLoader {
 public:
  /**
   * @brief Information about a tensor stored in the SafeTensors file
   */
  struct TensorInfo {
    std::string name;        /**< Name of the tensor */
    std::string dtype;       /**< Data type of the tensor (e.g., "float32", "float16") */
    std::vector<size_t> shape; /**< Shape of the tensor as a vector of dimensions */
    size_t data_offset;      /**< Offset of tensor data in the file */
    size_t nbytes;           /**< Number of bytes occupied by the tensor */
  };

  /**
   * @brief Constructs a SafeTensorsLoader for a specific file
   * @param path Path to the SafeTensors file
   * @throws std::runtime_error if file cannot be opened or has invalid format
   */
  explicit SafeTensorsLoader(const std::string& path);
  
  /**
   * @brief Destructor that ensures proper cleanup of memory mappings
   */
  ~SafeTensorsLoader();

  // Disable copy operations to prevent double-free of memory mappings
  SafeTensorsLoader(const SafeTensorsLoader&) = delete;
  SafeTensorsLoader& operator=(const SafeTensorsLoader&) = delete;

  /**
   * @brief Gets a list of all tensor names in the file
   * @return Vector of tensor names
   */
  std::vector<std::string> tensor_names() const;

  /**
   * @brief Loads a single tensor's data
   * @param name Name of the tensor to load
   * @return Vector of bytes containing the tensor data
   * @throws std::runtime_error if tensor not found
   */
  std::vector<uint8_t> get_tensor_bytes(const std::string& name) const;

  /**
   * @brief Gets information about a specific tensor
   * @param name Name of the tensor
   * @return Reference to the tensor's information
   * @throws std::runtime_error if tensor not found
   */
  const TensorInfo& get_tensor_info(const std::string& name) const;

  /**
   * @brief Loads a single tensor's data using parallel processing
   * @param name Name of the tensor to load
   * @return Vector of bytes containing the tensor data
   * @throws std::runtime_error if tensor not found
   */
  std::vector<uint8_t> get_tensor_bytes_parallel(const std::string& name) const;

  /**
   * @brief Loads all tensors in parallel
   * @return Map of tensor names to their data
   */
  std::map<std::string, std::vector<uint8_t>> load_all_tensors_parallel() const;

  /**
   * @brief Loads model configuration from a JSON file corresponding to a .safetensors model path.
   *
   * Given the path to a .safetensors model, this method attempts to find a "config.json"
   * in the same directory. If found, it parses the JSON and populates the provided
   * ModelConfig object.
   *
   * @param model_path Path to the .safetensors model file.
   * @param config_to_populate Reference to a ModelConfig object to be filled.
   * @return True if config.json was found and successfully parsed, false otherwise.
   */
  static bool load_model_config_from_json(const std::string& model_path, ModelConfig& config_to_populate);

 private:
  std::map<std::string, TensorInfo> tensors_;  /**< Map of tensor names to their information */
  std::string file_path_;                      /**< Path to the SafeTensors file */
  size_t data_start_ = 0;                      /**< Offset where tensor data begins */

#ifdef _WIN32
  HANDLE file_handle_ = INVALID_HANDLE_VALUE;   /**< Windows file handle */
  HANDLE mapping_handle_ = NULL;                /**< Windows file mapping handle */
#else
  int fd_ = -1;                                /**< File descriptor for memory mapping */
#endif
  void* mapped_data_ = nullptr;                /**< Pointer to memory mapped data */
  size_t file_size_ = 0;                       /**< Total size of the file */

  /**
   * @brief Sets up memory mapping for the file
   * @throws std::runtime_error if mapping fails
   */
  void initialize_memory_mapping();

  /**
   * @brief Cleans up memory mapping resources
   */
  void cleanup_memory_mapping();

  /**
   * @brief Converts raw tensor data to the appropriate format
   * @param data Pointer to raw tensor data
   * @param size Size of the data in bytes
   * @param dtype Data type of the tensor
   * @return Converted tensor data
   */
  std::vector<uint8_t> convert_tensor_data(const uint8_t* data, size_t size,
                                          const std::string& dtype) const;
};

/**
 * @brief Thread pool for parallel tensor loading operations
 * 
 * Manages a pool of worker threads for parallel processing of tensor data.
 * Used by SafeTensorsLoader for parallel loading operations.
 */
class ThreadPool {
 public:
  /**
   * @brief Constructs a thread pool with specified number of threads
   * @param num_threads Number of worker threads to create
   */
  explicit ThreadPool(size_t num_threads);
  
  /**
   * @brief Destructor that ensures proper cleanup of threads
   */
  ~ThreadPool();

  /**
   * @brief Submits a task to the thread pool
   * @tparam F Function type
   * @tparam Args Argument types
   * @param f Function to execute
   * @param args Arguments for the function
   * @return Future containing the result of the function
   */
  template <class F, class... Args>
  std::future<typename std::result_of<F(Args...)>::type> submit(F&& f,
                                                               Args&&... args);

 private:
  std::vector<std::thread> workers_;           /**< Worker threads */
  std::queue<std::function<void()>> tasks_;    /**< Queue of pending tasks */
  std::mutex queue_mutex_;                     /**< Mutex for task queue access */
  std::condition_variable condition_;          /**< Condition variable for thread synchronization */
  bool stop_ = false;                          /**< Flag to stop worker threads */
};

#endif