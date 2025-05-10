#include "safetensors_loader.h"

#include <algorithm>
#include <fstream>
#include <filesystem> // Required for path manipulation
#include "model.h"      // Required for ModelConfig definition
#include "logger.h"     // Required for Logger

#ifdef __AVX2__
#include <immintrin.h>
#endif

SafeTensorsLoader::SafeTensorsLoader(const std::string& path)
    : file_path_(path) {
  std::ifstream file(path, std::ios::binary);
  if (!file)
    throw std::runtime_error("Failed to open safetensors file: " + path);

  file.seekg(0, std::ios::end);
  file_size_ = file.tellg();
  file.seekg(0, std::ios::beg);

  uint64_t header_len = 0;
  file.read(reinterpret_cast<char*>(&header_len), sizeof(header_len));
  if (file.gcount() != sizeof(header_len))
    throw std::runtime_error("Failed to read safetensors header length");

  std::vector<char> header_buf(header_len);
  file.read(header_buf.data(), header_len);
  if (file.gcount() != static_cast<std::streamsize>(header_len))
    throw std::runtime_error("Failed to read safetensors header");
  std::string header_json(header_buf.begin(), header_buf.end());
  nlohmann::json header = nlohmann::json::parse(header_json);

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

  initialize_memory_mapping();
}

SafeTensorsLoader::~SafeTensorsLoader() { cleanup_memory_mapping(); }

void SafeTensorsLoader::initialize_memory_mapping() {
#ifdef _WIN32
  file_handle_ = CreateFileA(file_path_.c_str(), GENERIC_READ, FILE_SHARE_READ,
                            NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
  if (file_handle_ == INVALID_HANDLE_VALUE) {
    throw std::runtime_error("Failed to open file for memory mapping: " + file_path_);
  }

  mapping_handle_ = CreateFileMappingA(file_handle_, NULL, PAGE_READONLY,
                                      0, 0, NULL);
  if (mapping_handle_ == NULL) {
    CloseHandle(file_handle_);
    throw std::runtime_error("Failed to create file mapping: " + file_path_);
  }

  mapped_data_ = MapViewOfFile(mapping_handle_, FILE_MAP_READ, 0, 0, 0);
  if (mapped_data_ == NULL) {
    CloseHandle(mapping_handle_);
    CloseHandle(file_handle_);
    throw std::runtime_error("Failed to map view of file: " + file_path_);
  }
#else
  fd_ = open(file_path_.c_str(), O_RDONLY);
  if (fd_ == -1) {
    throw std::runtime_error("Failed to open file for memory mapping: " + file_path_);
  }

  mapped_data_ = mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
  if (mapped_data_ == MAP_FAILED) {
    close(fd_);
    throw std::runtime_error("Failed to memory map file: " + file_path_);
  }
#endif
}

void SafeTensorsLoader::cleanup_memory_mapping() {
#ifdef _WIN32
  if (mapped_data_ != nullptr) {
    UnmapViewOfFile(mapped_data_);
    mapped_data_ = nullptr;
  }
  if (mapping_handle_ != NULL) {
    CloseHandle(mapping_handle_);
    mapping_handle_ = NULL;
  }
  if (file_handle_ != INVALID_HANDLE_VALUE) {
    CloseHandle(file_handle_);
    file_handle_ = INVALID_HANDLE_VALUE;
  }
#else
  if (mapped_data_ != nullptr) {
    munmap(mapped_data_, file_size_);
    mapped_data_ = nullptr;
  }
  if (fd_ != -1) {
    close(fd_);
    fd_ = -1;
  }
#endif
}

std::vector<std::string> SafeTensorsLoader::tensor_names() const {
  std::vector<std::string> names;
  for (const auto& kv : tensors_) names.push_back(kv.first);
  return names;
}

std::vector<uint8_t> SafeTensorsLoader::get_tensor_bytes(
    const std::string& name) const {
  auto it = tensors_.find(name);
  if (it == tensors_.end())
    throw std::runtime_error("Tensor not found: " + name);

  const auto& info = it->second;
  const uint8_t* data = static_cast<const uint8_t*>(mapped_data_) +
                        data_start_ + info.data_offset;
  return convert_tensor_data(data, info.nbytes, info.dtype);
}

std::vector<uint8_t> SafeTensorsLoader::get_tensor_bytes_parallel(
    const std::string& name) const {
  return get_tensor_bytes(name);
}

std::map<std::string, std::vector<uint8_t>>
SafeTensorsLoader::load_all_tensors_parallel() const {
  std::map<std::string, std::vector<uint8_t>> result;
  std::vector<std::future<std::pair<std::string, std::vector<uint8_t>>>>
      futures;

  unsigned int num_threads = std::thread::hardware_concurrency();
  if (num_threads == 0) num_threads = 4;

  ThreadPool pool(num_threads);

  for (const auto& kv : tensors_) {
    futures.push_back(pool.submit([this, &kv]() {
      return std::make_pair(kv.first, get_tensor_bytes(kv.first));
    }));
  }

  for (auto& future : futures) {
    auto [name, data] = future.get();
    result[name] = std::move(data);
  }

  return result;
}

const SafeTensorsLoader::TensorInfo& SafeTensorsLoader::get_tensor_info(
    const std::string& name) const {
  auto it = tensors_.find(name);
  if (it == tensors_.end())
    throw std::runtime_error("Tensor not found: " + name);
  return it->second;
}

std::vector<uint8_t> SafeTensorsLoader::convert_tensor_data(
    const uint8_t* data, size_t size, const std::string& dtype) const {
  std::vector<uint8_t> result(size);

  if (dtype == "F16") {
#ifdef __AVX2__

    if (__builtin_cpu_supports("avx2")) {
      const size_t simd_size = 16;
      size_t i = 0;
      for (; i + simd_size <= size; i += simd_size) {
        __m256i src =
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i));
        __m256i dst = _mm256_shuffle_epi8(
            src, _mm256_set_epi8(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12,
                                 15, 14, 1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10,
                                 13, 12, 15, 14));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(result.data() + i), dst);
      }

      for (; i < size; i++) {
        result[i] = data[i];
      }
    } else {
      std::copy(data, data + size, result.begin());
    }
#else

    std::copy(data, data + size, result.begin());
#endif
  } else {
    std::copy(data, data + size, result.begin());
  }

  return result;
}

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

// Implementation of the new static method
bool SafeTensorsLoader::load_model_config_from_json(const std::string& model_weights_path, ModelConfig& config_to_populate) {
    namespace fs = std::filesystem;
    fs::path weights_fs_path(model_weights_path);
    fs::path config_json_path = weights_fs_path.parent_path() / "config.json";

    Logger::info("[SafetensorsLoader] Attempting to load model config from: " + config_json_path.string());

    if (!fs::exists(config_json_path)) {
        Logger::warning("[SafetensorsLoader] config.json not found at " + config_json_path.string());
        return false;
    }

    std::ifstream config_file(config_json_path);
    if (!config_file.is_open()) {
        Logger::error("[SafetensorsLoader] Failed to open config.json at " + config_json_path.string());
        return false;
    }

    try {
        nlohmann::json json_config;
        config_file >> json_config;
        config_file.close();

        Logger::info("[SafetensorsLoader] Successfully parsed config.json. Populating ModelConfig...");

        // Use .value() for safety, providing a default or relying on ModelConfig defaults if key is missing
        // For critical fields, consider checking with .contains() and throwing/logging an error if absent
        config_to_populate.hidden_size = json_config.value("hidden_size", 0);
        config_to_populate.intermediate_size = json_config.value("intermediate_size", 0);
        config_to_populate.num_attention_heads = json_config.value("num_attention_heads", 0);
        // num_key_value_heads might not always be present; often defaults to num_attention_heads
        config_to_populate.num_key_value_heads = json_config.value("num_key_value_heads", config_to_populate.num_attention_heads);
        if (json_config.contains("num_key_value_heads")) { // Explicitly set if present
             config_to_populate.num_key_value_heads = json_config.at("num_key_value_heads").get<int>();
        } else {
            config_to_populate.num_key_value_heads = config_to_populate.num_attention_heads; // Default if not found
        }
        config_to_populate.num_hidden_layers = json_config.value("num_hidden_layers", 0);
        config_to_populate.vocab_size = json_config.value("vocab_size", 0);
        config_to_populate.max_position_embeddings = json_config.value("max_position_embeddings", 0);
        config_to_populate.rms_norm_eps = json_config.value("rms_norm_eps", 1e-5f);
        config_to_populate.rope_theta = json_config.value("rope_theta", 10000.0f);
        config_to_populate.hidden_act = json_config.value("hidden_act", "");
        config_to_populate.torch_dtype = json_config.value("torch_dtype", "");
        config_to_populate.bos_token_id = json_config.value("bos_token_id", -1); // Default to -1 if not set
        config_to_populate.eos_token_id = json_config.value("eos_token_id", -1); // Default to -1 if not set
        
        // Architecture: often a list, take the first element if it's an array
        if (json_config.contains("architectures") && json_config["architectures"].is_array() && !json_config["architectures"].empty()) {
            config_to_populate.architecture = json_config["architectures"][0].get<std::string>();
        } else {
            config_to_populate.architecture = json_config.value("architecture", ""); // Fallback if not an array or not present
        }

        config_to_populate.model_name = json_config.value("model_type", ""); // Often called model_type in HF configs
        if (config_to_populate.model_name.empty()) { // Fallback if model_type is not present
             config_to_populate.model_name = json_config.value("name_or_path", "");
        }

        // These might not be standard in all config.json, provide defaults or leave empty
        config_to_populate.chat_template_type = json_config.value("chat_template_type", ""); 
        config_to_populate.pre_tokenizer_type = json_config.value("pre_tokenizer_type", "");
        config_to_populate.chat_template_string = json_config.value("chat_template", "");

        config_to_populate.is_gguf_file_loaded = false; // Explicitly false for safetensors

        Logger::info("[SafetensorsLoader] ModelConfig populated from config.json:");
        Logger::info("  hidden_size: " + std::to_string(config_to_populate.hidden_size));
        Logger::info("  vocab_size: " + std::to_string(config_to_populate.vocab_size));
        // Add more logs for other critical parameters if needed

        return true;

    } catch (const nlohmann::json::exception& e) {
        Logger::error("[SafetensorsLoader] Failed to parse config.json: " + std::string(e.what()));
        return false;
    } catch (const std::exception& e) {
        Logger::error("[SafetensorsLoader] An unexpected error occurred while processing config.json: " + std::string(e.what()));
        return false;
    }
}