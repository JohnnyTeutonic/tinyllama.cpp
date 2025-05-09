#include "safetensors_loader.h"
#include <fstream>
#include <algorithm>

// Only include AVX headers if available
#ifdef __AVX2__
#include <immintrin.h>
#endif

SafeTensorsLoader::SafeTensorsLoader(const std::string& path) : file_path_(path) {
  std::ifstream file(path, std::ios::binary);
  if (!file)
    throw std::runtime_error("Failed to open safetensors file: " + path);

  // Get file size
  file.seekg(0, std::ios::end);
  file_size_ = file.tellg();
  file.seekg(0, std::ios::beg);

  // Read header length
  uint64_t header_len = 0;
  file.read(reinterpret_cast<char*>(&header_len), sizeof(header_len));
  if (file.gcount() != sizeof(header_len))
    throw std::runtime_error("Failed to read safetensors header length");

  // Read header
  std::vector<char> header_buf(header_len);
  file.read(header_buf.data(), header_len);
  if (file.gcount() != static_cast<std::streamsize>(header_len))
    throw std::runtime_error("Failed to read safetensors header");
  std::string header_json(header_buf.begin(), header_buf.end());
  nlohmann::json header = nlohmann::json::parse(header_json);

  // Parse tensor info
  for (auto it = header.begin(); it != header.end(); ++it) {
    const std::string& key = it.key();
    if (key == "__metadata__") continue;
    const auto& meta = it.value();
    TensorInfo info;
    info.name = key;
    info.dtype = meta["dtype"].get<std::string>();
    info.shape = meta["shape"].get<std::vector<size_t>>();
    info.data_offset = meta["data_offsets"][0].get<size_t>();
    info.nbytes = meta["data_offsets"][1].get<size_t>() - info.data_offset;
    tensors_[key] = info;
  }

  data_start_ = 8 + header_len;
  file.close();

  // Initialize memory mapping
  initialize_memory_mapping();
}

SafeTensorsLoader::~SafeTensorsLoader() {
  cleanup_memory_mapping();
}

void SafeTensorsLoader::initialize_memory_mapping() {
  fd_ = open(file_path_.c_str(), O_RDONLY);
  if (fd_ == -1) {
    throw std::runtime_error("Failed to open file for memory mapping: " + file_path_);
  }

  mapped_data_ = mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
  if (mapped_data_ == MAP_FAILED) {
    close(fd_);
    throw std::runtime_error("Failed to memory map file: " + file_path_);
  }
}

void SafeTensorsLoader::cleanup_memory_mapping() {
  if (mapped_data_ != nullptr) {
    munmap(mapped_data_, file_size_);
    mapped_data_ = nullptr;
  }
  if (fd_ != -1) {
    close(fd_);
    fd_ = -1;
  }
}

std::vector<std::string> SafeTensorsLoader::tensor_names() const {
  std::vector<std::string> names;
  for (const auto& kv : tensors_) names.push_back(kv.first);
  return names;
}

std::vector<uint8_t> SafeTensorsLoader::get_tensor_bytes(const std::string& name) const {
  auto it = tensors_.find(name);
  if (it == tensors_.end())
    throw std::runtime_error("Tensor not found: " + name);

  const auto& info = it->second;
  const uint8_t* data = static_cast<const uint8_t*>(mapped_data_) + data_start_ + info.data_offset;
  return convert_tensor_data(data, info.nbytes, info.dtype);
}

std::vector<uint8_t> SafeTensorsLoader::get_tensor_bytes_parallel(const std::string& name) const {
  return get_tensor_bytes(name); // Already optimized with memory mapping
}

std::map<std::string, std::vector<uint8_t>> SafeTensorsLoader::load_all_tensors_parallel() const {
  std::map<std::string, std::vector<uint8_t>> result;
  std::vector<std::future<std::pair<std::string, std::vector<uint8_t>>>> futures;
  
  // Determine optimal number of threads
  unsigned int num_threads = std::thread::hardware_concurrency();
  if (num_threads == 0) num_threads = 4; // Fallback if hardware_concurrency fails
  
  // Create thread pool
  ThreadPool pool(num_threads);
  
  // Submit tasks for each tensor
  for (const auto& kv : tensors_) {
    futures.push_back(pool.submit([this, &kv]() {
      return std::make_pair(kv.first, get_tensor_bytes(kv.first));
    }));
  }
  
  // Collect results
  for (auto& future : futures) {
    auto [name, data] = future.get();
    result[name] = std::move(data);
  }
  
  return result;
}

const SafeTensorsLoader::TensorInfo& SafeTensorsLoader::get_tensor_info(const std::string& name) const {
  auto it = tensors_.find(name);
  if (it == tensors_.end())
    throw std::runtime_error("Tensor not found: " + name);
  return it->second;
}

std::vector<uint8_t> SafeTensorsLoader::convert_tensor_data(const uint8_t* data, size_t size, const std::string& dtype) const {
  std::vector<uint8_t> result(size);
  
  // SIMD-accelerated conversion based on dtype
  if (dtype == "F16") {
#ifdef __AVX2__
    // Use AVX2 for F16 conversion if available
    if (__builtin_cpu_supports("avx2")) {
      const size_t simd_size = 16; // AVX2 processes 16 bytes at a time
      size_t i = 0;
      for (; i + simd_size <= size; i += simd_size) {
        __m256i src = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i));
        __m256i dst = _mm256_shuffle_epi8(src, _mm256_set_epi8(
          1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14,
          1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14
        ));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(result.data() + i), dst);
      }
      // Handle remaining bytes
      for (; i < size; i++) {
        result[i] = data[i];
      }
    } else {
      // Fallback to regular copy if AVX2 not available
      std::copy(data, data + size, result.begin());
    }
#else
    // Fallback to regular copy if AVX2 not enabled
    std::copy(data, data + size, result.begin());
#endif
  } else {
    // For other dtypes, use regular copy
    std::copy(data, data + size, result.begin());
  }
  
  return result;
}

// ThreadPool implementation
ThreadPool::ThreadPool(size_t num_threads) {
  for (size_t i = 0; i < num_threads; ++i) {
    workers_.emplace_back([this] {
      while (true) {
        std::function<void()> task;
        {
          std::unique_lock<std::mutex> lock(queue_mutex_);
          condition_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
          if (stop_ && tasks_.empty()) return;
          task = std::move(tasks_.front());
          tasks_.pop();
        }
        task();
      }
    });
  }
}

ThreadPool::~ThreadPool() {
  {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    stop_ = true;
  }
  condition_.notify_all();
  for (std::thread& worker : workers_) {
    worker.join();
  }
}

template<class F, class... Args>
std::future<typename std::result_of<F(Args...)>::type> ThreadPool::submit(F&& f, Args&&... args) {
  using return_type = typename std::result_of<F(Args...)>::type;
  auto task = std::make_shared<std::packaged_task<return_type()>>(
    std::bind(std::forward<F>(f), std::forward<Args>(args)...)
  );
  std::future<return_type> res = task->get_future();
  {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    if (stop_) throw std::runtime_error("submit on stopped ThreadPool");
    tasks_.emplace([task]() { (*task)(); });
  }
  condition_.notify_one();
  return res;
}