#ifndef SAFETENSORS_LOADER_H
#define SAFETENSORS_LOADER_H

#include <map>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>
#include <memory>
#include <thread>
#include <mutex>
#include <future>
#include <queue>
#include <functional>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>


class ThreadPool;

class SafeTensorsLoader {
 public:
  struct TensorInfo {
    std::string name;
    std::string dtype;
    std::vector<size_t> shape;
    size_t data_offset;
    size_t nbytes;
  };

  explicit SafeTensorsLoader(const std::string& path);
  ~SafeTensorsLoader();

  
  SafeTensorsLoader(const SafeTensorsLoader&) = delete;
  SafeTensorsLoader& operator=(const SafeTensorsLoader&) = delete;

  std::vector<std::string> tensor_names() const;
  std::vector<uint8_t> get_tensor_bytes(const std::string& name) const;
  const TensorInfo& get_tensor_info(const std::string& name) const;

  
  std::vector<uint8_t> get_tensor_bytes_parallel(const std::string& name) const;
  std::map<std::string, std::vector<uint8_t>> load_all_tensors_parallel() const;

 private:
  std::map<std::string, TensorInfo> tensors_;
  std::string file_path_;
  size_t data_start_ = 0;
  
  
  int fd_ = -1;
  void* mapped_data_ = nullptr;
  size_t file_size_ = 0;
  
  
  void initialize_memory_mapping();
  void cleanup_memory_mapping();
  std::vector<uint8_t> convert_tensor_data(const uint8_t* data, size_t size, const std::string& dtype) const;
};


class ThreadPool {
public:
  explicit ThreadPool(size_t num_threads);
  ~ThreadPool();

  template<class F, class... Args>
  std::future<typename std::result_of<F(Args...)>::type> submit(F&& f, Args&&... args);

private:
  std::vector<std::thread> workers_;
  std::queue<std::function<void()>> tasks_;
  std::mutex queue_mutex_;
  std::condition_variable condition_;
  bool stop_ = false;
};

#endif  