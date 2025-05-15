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
#include <filesystem> // For directory operations, C++17

#include "logger.h" // Assuming Logger is accessible

struct ModelConfig; // Forward declaration

/**
 * @file safetensors_loader.h
 * @brief SafeTensors format loader for efficient tensor loading, supporting single and sharded models.
 */

class ThreadPool; // Forward declaration

/**
 * @brief Represents a memory-mapped SafeTensors file (shard).
 *
 * Handles opening, memory-mapping, and cleanup for a single SafeTensors file.
 * Used internally by SafeTensorsLoader to support sharded models.
 */
struct Shard {
    /**
     * @brief Path to the shard file.
     */
    std::string file_path;

    /**
     * @brief Pointer to the memory-mapped data.
     */
    void* mapped_data = nullptr;

    /**
     * @brief Size of the mapped file in bytes.
     */
    size_t file_size = 0;

    /**
     * @brief Size of the metadata block in bytes.
     */
    uint64_t metadata_size = 0;

    /**
     * @brief Pointer to the start of the metadata block.
     */
    const uint8_t* metadata_ptr = nullptr;

    /**
     * @brief Pointer to the start of the tensor data block.
     */
    const uint8_t* tensor_data_block_ptr = nullptr;

#ifdef _WIN32
    HANDLE file_handle_ = INVALID_HANDLE_VALUE; /**< Windows file handle */
    HANDLE mapping_handle_ = NULL;              /**< Windows file mapping handle */
#else
    int fd_ = -1;                               /**< File descriptor for memory mapping */
#endif

    /**
     * @brief Construct and memory-map a shard file.
     * @param fp Path to the shard file.
     * @throws std::runtime_error on failure.
     */
    explicit Shard(const std::string& fp);

    /**
     * @brief Destructor. Cleans up memory mapping and file handles.
     */
    ~Shard();

    /**
     * @brief Move constructor.
     */
    Shard(Shard&& other) noexcept;

    /**
     * @brief Move assignment operator.
     */
    Shard& operator=(Shard&& other) noexcept;

    /**
     * @brief Get a pointer to the raw tensor data within this shard.
     * @param local_offset Offset from the start of the tensor data block.
     * @param n_bytes Number of bytes to access.
     * @return Pointer to the tensor data.
     * @throws std::out_of_range if the requested range is invalid.
     */
    const uint8_t* get_tensor_raw_data(size_t local_offset, size_t n_bytes) const;
};


/**
 * @brief Main class for loading tensors from SafeTensors format files (single or sharded)
 *
 * Supports both single-file and multi-shard (sharded) SafeTensors models. Handles memory mapping,
 * tensor metadata parsing, and provides efficient access to tensor data. Can load models from a
 * single .safetensors file, a directory containing multiple shards, or a directory with an index file.
 */
class SafeTensorsLoader {
 public:
  /**
   * @brief Information about a tensor stored in the SafeTensors file(s)
   */
  struct TensorInfo {
    std::string name;        /**< Name of the tensor */
    std::string dtype;       /**< Data type of the tensor (e.g., "F32", "F16") */
    std::vector<size_t> shape; /**< Shape of the tensor as a vector of dimensions */
    size_t data_offset;      /**< Offset of tensor data relative to its shard's tensor_data_block_ptr */
    size_t nbytes;           /**< Number of bytes occupied by the tensor */
    std::string shard_key;   /**< Key (e.g., filename) of the shard this tensor belongs to */
  };

  /**
   * @brief Constructs a SafeTensorsLoader.
   *
   * The path can be to a single .safetensors file, or a directory containing
   * .safetensors file(s) and potentially an index.json.
   *
   * @param model_load_path Path to the model file or directory.
   * @throws std::runtime_error if files cannot be opened, are invalid, or sharding info is inconsistent.
   */
  explicit SafeTensorsLoader(const std::string& model_load_path);
  
  /**
   * @brief Destructor. Cleans up all memory-mapped shards.
   */
  ~SafeTensorsLoader();

  SafeTensorsLoader(const SafeTensorsLoader&) = delete;
  SafeTensorsLoader& operator=(const SafeTensorsLoader&) = delete;

  /**
   * @brief Get a list of all tensor names available in the loaded model.
   * @return Vector of tensor names.
   */
  std::vector<std::string> tensor_names() const;

  /**
   * @brief Get the raw bytes for a tensor, converting to FP32 if needed.
   * @param name Name of the tensor to load.
   * @return Vector of bytes containing the tensor data (FP32 format).
   * @throws std::runtime_error if tensor not found or conversion fails.
   */
  std::vector<uint8_t> get_tensor_bytes(const std::string& name) const;

  /**
   * @brief Get information about a specific tensor.
   * @param name Name of the tensor.
   * @return Reference to the tensor's information.
   * @throws std::runtime_error if tensor not found.
   */
  const TensorInfo& get_tensor_info(const std::string& name) const;

  /**
   * @brief Load all tensors in parallel.
   * @return Map of tensor names to their data (FP32 format).
   */
  std::map<std::string, std::vector<uint8_t>> load_all_tensors_parallel() const;

  /**
   * @brief Loads model configuration from a JSON file corresponding to a .safetensors model path.
   *
   * Given the path to a .safetensors model or directory, this method attempts to find a "config.json"
   * in the same directory. If found, it parses the JSON and populates the provided ModelConfig object.
   *
   * @param model_path_or_dir Path to the .safetensors model file or directory.
   * @param config_to_populate Reference to a ModelConfig object to be filled.
   * @return True if config.json was found and successfully parsed, false otherwise.
   */
  static bool load_model_config_from_json(const std::string& model_path_or_dir, ModelConfig& config_to_populate);

 private:
  std::string model_load_path_;                     /**< Original path provided to constructor (file or directory) */
  bool is_sharded_ = false;                         /**< True if model is loaded from multiple shard files */
  
  std::map<std::string, TensorInfo> tensors_;       /**< Global map of tensor names to their comprehensive info */
  std::map<std::string, std::unique_ptr<Shard>> loaded_shards_; /**< Map of shard keys (e.g., filenames) to Shard objects */
  
  // If sharded via an index file, this maps tensor names directly to their shard key.
  // If not sharded or sharded by pattern, this might be populated differently or less used.
  std::map<std::string, std::string> tensor_name_to_shard_key_map_; 

  /**
   * @brief Load tensors from a directory, handling index files and multiple shards.
   *
   * If an index file is found, parses it and loads the referenced shards. Otherwise, scans for
   * .safetensors files and loads them as individual shards.
   * @param directory_path Path to the directory containing model files.
   */
  void load_from_directory(const std::string& directory_path);

  /**
   * @brief Load a single .safetensors file as a shard.
   *
   * Memory-maps the file and parses its metadata to populate tensor information.
   * @param file_path Path to the .safetensors file.
   * @param shard_key_override Optional key to use for this shard (e.g., filename).
   */
  void load_single_file(const std::string& file_path, const std::string& shard_key_override = "");

  /**
   * @brief Parse the metadata of a shard and populate tensor information.
   *
   * Reads the metadata JSON from the shard and adds entries to the tensors_ map.
   * @param shard Reference to the Shard object.
   * @param shard_key Key identifying this shard (e.g., filename).
   */
  void parse_shard_metadata(Shard& shard, const std::string& shard_key);

  /**
   * @brief Convert raw tensor data to FP32 if needed.
   *
   * Handles conversion from F16/BF16 to FP32 as required by the tensor's dtype.
   * @param data Pointer to the raw tensor data.
   * @param size Size of the data in bytes.
   * @param dtype Data type string (e.g., "F32", "F16", "BF16").
   * @return Converted tensor data as a vector of bytes (FP32 format).
   */
  std::vector<uint8_t> convert_tensor_data(const uint8_t* data, size_t size,
                                           const std::string& dtype) const;

  /**
   * @brief Get the Shard object for a given tensor name.
   *
   * Looks up the shard key for the tensor and returns a pointer to the corresponding Shard.
   * @param tensor_name Name of the tensor.
   * @return Pointer to the Shard containing the tensor.
   * @throws std::logic_error if the shard is not found.
   */
  const Shard* get_shard_for_tensor(const std::string& tensor_name) const;
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

// Template implementation for ThreadPool::submit
template <class F, class... Args>
std::future<typename std::result_of<F(Args...)>::type> ThreadPool::submit(
    F&& f, Args&&... args) {
  using return_type = typename std::result_of<F(Args...)>::type;

  auto task = std::make_shared<std::packaged_task<return_type()>>(
      std::bind(std::forward<F>(f), std::forward<Args>(args)...));

  std::future<return_type> res = task->get_future();
  {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    if (stop_) throw std::runtime_error("submit on stopped ThreadPool");
    tasks_.emplace([task]() { (*task)(); });
  }
  condition_.notify_one();
  return res;
}

#endif // SAFETENSORS_LOADER_H