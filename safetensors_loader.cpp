#include "safetensors_loader.h"

#include <algorithm>
#include <fstream>
#include <filesystem> // For path manipulation
#include "model.h"      // For ModelConfig definition and parse_model_config
#include "logger.h"     // For Logger

#ifdef __AVX2__
#include <immintrin.h>
#endif


// (Similar to the one in cuda_kernels.cu but for CPU)
inline float cpu_bf16_to_float32(uint16_t bf16_raw) {
    // BF16 is essentially the top 16 bits of an FP32 number.
    // So, we shift it left by 16 bits to align it with the FP32 format.
    // The lower 16 bits (mantissa of FP32) will be zero.
    unsigned int bits = ((unsigned int)bf16_raw) << 16;
    float result;
    memcpy(&result, &bits, sizeof(float));
    return result;
}



inline float cpu_f16_to_float32(uint16_t f16_raw) {
    // F16 format: 1 sign bit, 5 exponent bits, 10 mantissa bits
    // FP32 format: 1 sign bit, 8 exponent bits, 23 mantissa bits

    const uint32_t sign_mask_f16 = 0x8000;
    const uint32_t exp_mask_f16 = 0x7C00;
    const uint32_t mant_mask_f16 = 0x03FF;
    const uint32_t exp_bias_f16 = 15;

    // const uint32_t sign_mask_f32 = 0x80000000;
    // const uint32_t exp_mask_f32 = 0x7F800000;
    // const uint32_t mant_mask_f32 = 0x007FFFFF;
    const uint32_t exp_bias_f32 = 127;

    uint32_t sign_f16 = (f16_raw & sign_mask_f16);
    uint32_t exp_f16 = (f16_raw & exp_mask_f16) >> 10;
    uint32_t mant_f16 = (f16_raw & mant_mask_f16);

    uint32_t f32_bits;

    if (exp_f16 == 0x1F) { // F16 NaN or Inf
        f32_bits = (sign_f16 << 16) | 0x7F800000 | (mant_f16 << 13); // Propagate mantissa for NaN
    } else if (exp_f16 == 0) { // F16 zero or subnormal
        if (mant_f16 == 0) { // Zero
            f32_bits = (sign_f16 << 16);
        } else { // Subnormal F16 to normal or subnormal F32
            // Convert F16 subnormal to F32 normal if possible, or F32 subnormal.
            // This involves counting leading zeros in mantissa and adjusting exponent.
            // Simplified: treat as zero for now, or scale directly.
            // A more precise conversion:
            int32_t current_exp = 1 - exp_bias_f16; // Exponent for subnormals
            uint32_t current_mant = mant_f16;
            while (!(current_mant & 0x0400)) { // while leading bit (10th) is not 1
                current_mant <<= 1;
                current_exp--;
                if (current_exp < (1 - exp_bias_f32 - 23) ) { // underflow to zero
                    current_mant = 0; break;
                }
            }
            current_mant &= mant_mask_f16; // Remove implicit leading 1 after normalization
            
            if (current_mant == 0) { // Result is zero
                 f32_bits = (sign_f16 << 16);
            } else {
                int32_t f32_exp_val = current_exp + exp_bias_f32;
                if (f32_exp_val <= 0) { // F32 subnormal or zero
                    // requires shifting mantissa right by -f32_exp_val + 1
                    int shift = 1 - f32_exp_val;
                    f32_bits = (sign_f16 << 16) | ((current_mant << 13) >> shift) ;
                } else { // F32 normal
                     f32_bits = (sign_f16 << 16) | (static_cast<uint32_t>(f32_exp_val) << 23) | (current_mant << 13);
                }
            }
        }
    } else { // Normal F16
        uint32_t f32_exp = exp_f16 - exp_bias_f16 + exp_bias_f32;
        f32_bits = (sign_f16 << 16) | (f32_exp << 23) | (mant_f16 << 13);
    }

    float result;
    memcpy(&result, &f32_bits, sizeof(float));
    return result;
}


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
    const uint8_t* data, size_t n_bytes_original, const std::string& dtype) const {
  
  if (dtype == "BF16") {
    if (n_bytes_original % sizeof(uint16_t) != 0) {
        throw std::runtime_error("BF16 tensor data size is not a multiple of sizeof(uint16_t)");
    }
    size_t num_bf16_elements = n_bytes_original / sizeof(uint16_t);
    size_t new_fp32_n_bytes = num_bf16_elements * sizeof(float);
    std::vector<uint8_t> result_fp32_bytes(new_fp32_n_bytes);
    
    const uint16_t* bf16_data_ptr = reinterpret_cast<const uint16_t*>(data);
    float* fp32_data_ptr = reinterpret_cast<float*>(result_fp32_bytes.data());

    // Diagnostic: Log first few raw BF16 and converted FP32 values
    static bool bf16_diag_logged = false; // Only log for the first BF16 tensor encountered
    if (!bf16_diag_logged && dtype == "BF16" && num_bf16_elements > 5) {
        std::stringstream ss_diag;
        ss_diag << "[BF16_CONV_DIAG] First 5 BF16 values from tensor (name not available here). Raw uint16_t hex: ";
        for (size_t k=0; k < 5; ++k) ss_diag << "0x" << std::hex << bf16_data_ptr[k] << " ";
        ss_diag << "| Converted to FP32: ";
        for (size_t k=0; k < 5; ++k) ss_diag << std::fixed << std::setprecision(7) << cpu_bf16_to_float32(bf16_data_ptr[k]) << " ";
        Logger::debug(ss_diag.str());
        bf16_diag_logged = true; // Set flag so we don't log again
      }

    for (size_t i = 0; i < num_bf16_elements; ++i) {
        fp32_data_ptr[i] = cpu_bf16_to_float32(bf16_data_ptr[i]);
      }
    return result_fp32_bytes;

  } else if (dtype == "F16") {
    if (n_bytes_original % sizeof(uint16_t) != 0) {
        throw std::runtime_error("F16 tensor data size is not a multiple of sizeof(uint16_t)");
    }
    size_t num_f16_elements = n_bytes_original / sizeof(uint16_t);
    size_t new_fp32_n_bytes = num_f16_elements * sizeof(float);
    std::vector<uint8_t> result_fp32_bytes(new_fp32_n_bytes);

    const uint16_t* f16_data_ptr = reinterpret_cast<const uint16_t*>(data);
    float* fp32_data_ptr = reinterpret_cast<float*>(result_fp32_bytes.data());

    for (size_t i = 0; i < num_f16_elements; ++i) {
        fp32_data_ptr[i] = cpu_f16_to_float32(f16_data_ptr[i]);
    }
    Logger::info("[SafeTensorsLoader] Converted F16 tensor data to FP32. Original bytes: " + std::to_string(n_bytes_original) + ", New FP32 bytes: " + std::to_string(new_fp32_n_bytes));
    return result_fp32_bytes;
  } else {
    // For other dtypes (e.g., F32, I32), just copy bytes.
    std::vector<uint8_t> result(n_bytes_original);
    std::copy(data, data + n_bytes_original, result.begin());
    return result;
  }
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
bool SafeTensorsLoader::load_model_config_from_json(const std::string& model_path, ModelConfig& config_to_populate) {
    std::filesystem::path sf_path(model_path);
    std::filesystem::path config_json_path = sf_path.parent_path() / "config.json";

    Logger::info("[SafeTensorsLoader] Attempting to load model config from: " + config_json_path.string());

    if (!std::filesystem::exists(config_json_path)) {
        Logger::warning("[SafeTensorsLoader] config.json not found at " + config_json_path.string());
        return false;
    }

    std::ifstream config_file(config_json_path);
    if (!config_file.is_open()) {
        Logger::error("[SafeTensorsLoader] Failed to open config.json at " + config_json_path.string());
        return false;
    }

    try {
        nlohmann::json json_config;
        config_file >> json_config;
        config_file.close();

        // Use the existing parse_model_config function (it needs to be callable here)
        // If parse_model_config is not static or part of a class accessible here, 
        // we'll need to duplicate its logic or make it accessible.
        // For now, assuming parse_model_config from model.cpp can be used or adapted.
        // Let's replicate the relevant parts of parse_model_config directly here for simplicity,
        // as parse_model_config itself is not part of SafeTensorsLoader.

        config_to_populate.hidden_size = json_config.value("hidden_size", 0);
        config_to_populate.intermediate_size = json_config.value("intermediate_size", 0);
        config_to_populate.num_attention_heads = json_config.value("num_attention_heads", 0);
        config_to_populate.num_key_value_heads = json_config.value("num_key_value_heads", config_to_populate.num_attention_heads); // Default to num_attention_heads if not present
        config_to_populate.num_hidden_layers = json_config.value("num_hidden_layers", 0);
        config_to_populate.vocab_size = json_config.value("vocab_size", 0);
        config_to_populate.max_position_embeddings = json_config.value("max_position_embeddings", 0);
        config_to_populate.rms_norm_eps = json_config.value("rms_norm_eps", 1e-5f);
        config_to_populate.rope_theta = json_config.value("rope_theta", 10000.0f);
        config_to_populate.hidden_act = json_config.value("hidden_act", "silu");
        config_to_populate.torch_dtype = json_config.value("torch_dtype", "float16"); // Safetensors often bf16 or f16
        config_to_populate.bos_token_id = json_config.value("bos_token_id", 1);
        config_to_populate.eos_token_id = json_config.value("eos_token_id", 2);
        
        std::string model_architecture_str;
        if (json_config.contains("architectures") && json_config["architectures"].is_array() && !json_config["architectures"].empty()) {
            model_architecture_str = json_config["architectures"][0].get<std::string>();
            config_to_populate.architecture = model_architecture_str; // Keep this for info
        } else {
            // Fallback if "architectures" is not present or not a list
            model_architecture_str = json_config.value("architecture", ""); 
            config_to_populate.architecture = model_architecture_str;
        }
        
        std::string model_type_str = json_config.value("model_type", "");
        config_to_populate.model_name = model_type_str; // Populate model_name with model_type
        if (config_to_populate.model_name.empty()) {
             // Fallback for model_name if model_type was empty
             config_to_populate.model_name = json_config.value("name_or_path", "");
        }

        // Determine TokenizerFamily
        // Default to UNKNOWN, will be overridden if specific conditions are met.
        config_to_populate.tokenizer_family = ModelConfig::TokenizerFamily::UNKNOWN; 

        std::string tokenizer_class_str = json_config.value("tokenizer_class", "");
        Logger::info("[SafeTensorsLoader] Parsed from config.json - model_type: '" + model_type_str + 
                     "', architecture: '" + model_architecture_str + 
                     "', tokenizer_class: '" + tokenizer_class_str + "'");

        if (model_type_str == "llama" || model_architecture_str == "LlamaForCausalLM") {
            if (tokenizer_class_str == "LlamaTokenizer" || tokenizer_class_str == "CodeLlamaTokenizer") { // Llama 1/2 SentencePiece
                config_to_populate.tokenizer_family = ModelConfig::TokenizerFamily::LLAMA_SENTENCEPIECE;
            } else if (tokenizer_class_str == "LlamaHfTokenizer") { // Often seen with Tiktoken-based Llama tokenizers in HF format
                 // Further check if it's really Llama 3 like. For now, assume Llama 3 if LlamaHfTokenizer.
                 // This might need refinement if Llama 2 models also use LlamaHfTokenizer with SentencePiece vocab.
                 // A more robust check might involve looking at specific vocab entries or merge rules.
                 // For instance, Llama 3 tokenizers typically have a larger vocab size (e.g., 128256).
                if (config_to_populate.vocab_size > 100000) { // Heuristic for Llama 3 / Tiktoken based
                    config_to_populate.tokenizer_family = ModelConfig::TokenizerFamily::LLAMA3_TIKTOKEN;
                     Logger::info("[SafeTensorsLoader] LlamaHfTokenizer with large vocab suggests LLAMA3_TIKTOKEN.");
                } else {
                    config_to_populate.tokenizer_family = ModelConfig::TokenizerFamily::LLAMA_SENTENCEPIECE;
                    Logger::info("[SafeTensorsLoader] LlamaHfTokenizer with smaller vocab suggests LLAMA_SENTENCEPIECE.");
                }
            } else if (tokenizer_class_str.find("Llama3") != std::string::npos || tokenizer_class_str.find("llama3") != std::string::npos) { // Explicit Llama3 in class name
                 config_to_populate.tokenizer_family = ModelConfig::TokenizerFamily::LLAMA3_TIKTOKEN;
            } else if (tokenizer_class_str.empty() && (model_type_str == "llama" || model_architecture_str == "LlamaForCausalLM") ) {
                // If tokenizer_class is not specified, but model_type is llama, assume SentencePiece for Llama 1/2
                // This is a common case for original Llama models.
                config_to_populate.tokenizer_family = ModelConfig::TokenizerFamily::LLAMA_SENTENCEPIECE;
                Logger::info("[SafeTensorsLoader] model_type 'llama' and empty tokenizer_class, defaulting to LLAMA_SENTENCEPIECE.");
            }
        } else if (model_type_str.find("llama3") != std::string::npos || model_architecture_str.find("Llama3") != std::string::npos) {
             config_to_populate.tokenizer_family = ModelConfig::TokenizerFamily::LLAMA3_TIKTOKEN;
        }
        // Add more specific checks if needed for other model types or tokenizer classes.

        std::string family_log_str = "UNKNOWN_FAMILY";
        if(config_to_populate.tokenizer_family == ModelConfig::TokenizerFamily::LLAMA_SENTENCEPIECE) family_log_str = "LLAMA_SENTENCEPIECE";
        else if(config_to_populate.tokenizer_family == ModelConfig::TokenizerFamily::LLAMA3_TIKTOKEN) family_log_str = "LLAMA3_TIKTOKEN";
        Logger::info("[SafeTensorsLoader] Deduced TokenizerFamily: " + family_log_str);
        

        // GGUF specific fields, not typically in safetensors config.json, but ensure they are defaulted reasonably.
        config_to_populate.is_gguf_file_loaded = false; 
        // chat_template_type, pre_tokenizer_type, chat_template_string might not be in all safetensor configs
        config_to_populate.chat_template_type = json_config.value("chat_template_type", "");
        config_to_populate.pre_tokenizer_type = json_config.value("pre_tokenizer_type", "");
        config_to_populate.chat_template_string = json_config.value("chat_template", ""); // often 'chat_template'

        Logger::info("[SafeTensorsLoader] Successfully parsed config.json. Hidden size: " + std::to_string(config_to_populate.hidden_size));
        return true;

    } catch (const nlohmann::json::exception& e) {
        Logger::error("[SafeTensorsLoader] Failed to parse config.json: " + std::string(e.what()));
        return false;
    } catch (const std::exception& e) {
        Logger::error("[SafeTensorsLoader] An unexpected error occurred while parsing config.json: " + std::string(e.what()));
        return false;
    }
}