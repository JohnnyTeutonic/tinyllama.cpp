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
 * @brief Helper struct to manage memory mapping for a single SafeTensors shard.
 */
struct Shard {
    std::string file_path;                /**< Absolute path to the shard file */
    void* mapped_data = nullptr;          /**< Pointer to memory mapped data for this shard */
    size_t file_size = 0;                 /**< Total size of this shard file */
    uint64_t metadata_size = 0;           /**< Size of the metadata block in this shard */
    const uint8_t* metadata_ptr = nullptr;/**< Pointer to the start of metadata within mapped_data */
    const uint8_t* tensor_data_block_ptr = nullptr; /**< Pointer to the start of the tensor data block within mapped_data */


#ifdef _WIN32
    HANDLE file_handle_ = INVALID_HANDLE_VALUE; /**< Windows file handle */
    HANDLE mapping_handle_ = NULL;              /**< Windows file mapping handle */
#else
    int fd_ = -1;                               /**< File descriptor for memory mapping */
#endif

    /**
     * @brief Constructs and memory-maps a shard.
     * @param fp Absolute path to the shard file.
     * @throws std::runtime_error if file cannot be opened or mapped.
     */
    explicit Shard(const std::string& fp);

    /**
     * @brief Destructor, cleans up memory mapping for this shard.
     */
    ~Shard();

    // Disable copy operations
    Shard(const Shard&) = delete;
    Shard& operator=(const Shard&) = delete;
    // Enable move operations
    Shard(Shard&& other) noexcept;
    Shard& operator=(Shard&& other) noexcept;


    /**
     * @brief Gets a pointer to the raw byte data for a tensor within this shard.
     * @param local_offset Offset of the tensor data relative to tensor_data_block_ptr.
     * @param n_bytes Number of bytes for the tensor.
     * @return Const pointer to the tensor's data.
     * @throws std::out_of_range if the requested data is out of bounds.
     */
    const uint8_t* get_tensor_raw_data(size_t local_offset, size_t n_bytes) const;
};


/**
 * @brief Main class for loading tensors from SafeTensors format files (single or sharded)
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
   * The path can be to a single .safetensors file, or a directory containing
   * .safetensors file(s) and potentially an index.json.
   * @param model_load_path Path to the model file or directory.
   * @throws std::runtime_error if files cannot be opened, are invalid, or sharding info is inconsistent.
   */
  explicit SafeTensorsLoader(const std::string& model_load_path);
  
  ~SafeTensorsLoader();

  SafeTensorsLoader(const SafeTensorsLoader&) = delete;
  SafeTensorsLoader& operator=(const SafeTensorsLoader&) = delete;

  std::vector<std::string> tensor_names() const;
  std::vector<uint8_t> get_tensor_bytes(const std::string& name) const; // Converts to FP32 if F16/BF16
  const TensorInfo& get_tensor_info(const std::string& name) const;
  std::map<std::string, std::vector<uint8_t>> load_all_tensors_parallel() const;

  static bool load_model_config_from_json(const std::string& model_path_or_dir, ModelConfig& config_to_populate);

 private:
  std::string model_load_path_;                     /**< Original path provided to constructor (file or directory) */
  bool is_sharded_ = false;                         /**< True if model is loaded from multiple shard files */
  
  std::map<std::string, TensorInfo> tensors_;       /**< Global map of tensor names to their comprehensive info */
  std::map<std::string, std::unique_ptr<Shard>> loaded_shards_; /**< Map of shard keys (e.g., filenames) to Shard objects */
  
  // If sharded via an index file, this maps tensor names directly to their shard key.
  // If not sharded or sharded by pattern, this might be populated differently or less used.
  std::map<std::string, std::string> tensor_name_to_shard_key_map_; 

  void load_from_directory(const std::string& directory_path);
  void load_single_file(const std::string& file_path, const std::string& shard_key_override = "");
  void parse_shard_metadata(Shard& shard, const std::string& shard_key);

  std::vector<uint8_t> convert_tensor_data(const uint8_t* data, size_t size,
                                           const std::string& dtype) const;
  // Helper to get shard for a tensor
  const Shard* get_shard_for_tensor(const std::string& tensor_name) const;
};

class ThreadPool {
 public:
  explicit ThreadPool(size_t num_threads);
  ~ThreadPool();

  template <class F, class... Args>
  std::future<typename std::result_of<F(Args...)>::type> submit(F&& f,
                                                               Args&&... args);
 private:
  std::vector<std::thread> workers_;
  std::queue<std::function<void()>> tasks_;
  std::mutex queue_mutex_;
  std::condition_variable condition_;
  bool stop_ = false;
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