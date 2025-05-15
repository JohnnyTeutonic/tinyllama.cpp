#include "safetensors_loader.h"
#include "model.h"
#include "logger.h"
#include "model_macros.h" // For SAFE_MIN, SAFE_MAX (may be needed by cpu_f16_to_float32)

#include <fstream>
#include <stdexcept>
#include <nlohmann/json.hpp>
#include <algorithm> 
#include <cctype>    
#include <vector>
#include <string>
#include <map>
#include <memory>
#include <filesystem>


#ifndef _WIN32
#include <sys/stat.h> 
#include <cerrno> // For strerror
#else
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h> 
#endif

#ifdef __AVX2__
#include <immintrin.h>
#endif
inline float cpu_bf16_to_float32(uint16_t bf16_raw) {
    unsigned int bits = ((unsigned int)bf16_raw) << 16;
    float result;
    memcpy(&result, &bits, sizeof(float));
    return result;
}
inline float cpu_f16_to_float32(uint16_t f16_raw) {
    const uint32_t sign_mask_f16 = 0x8000;
    const uint32_t exp_mask_f16 = 0x7C00;
    const uint32_t mant_mask_f16 = 0x03FF;
    const int32_t exp_bias_f16 = 15;
    const int32_t exp_bias_f32 = 127;

    uint32_t sign_f32 = (static_cast<uint32_t>(f16_raw & sign_mask_f16)) << 16;
    int32_t exp_f16 = (f16_raw & exp_mask_f16) >> 10;
    uint32_t mant_f16 = (f16_raw & mant_mask_f16);

    uint32_t f32_bits;

    if (exp_f16 == 0x1F) { // F16 NaN or Inf
        f32_bits = sign_f32 | 0x7F800000U | (mant_f16 << 13); // Propagate mantissa for NaN
    } else if (exp_f16 == 0) { // F16 zero or subnormal
        if (mant_f16 == 0) { // Zero
            f32_bits = sign_f32;
        } else { // Subnormal F16 to normal or subnormal F32
            int32_t s = -1;
            mant_f16 <<= 1;
            while ((mant_f16 & 0x0400) == 0) {
                mant_f16 <<= 1;
                s--;
            }
            mant_f16 &= 0x03FF; // Clear leading 1
            int32_t f32_exp_val = (1 - exp_bias_f16) + s + exp_bias_f32;
            if (f32_exp_val <= 0) { // Result is subnormal F32 or zero
                int32_t shift = 1 - f32_exp_val;
                if (shift > 23) { // Underflow to zero
                     f32_bits = sign_f32;
                } else {
                     f32_bits = sign_f32 | ((mant_f16 << 13) >> shift) ; 
                }
            } else { // Result is normal F32
                 f32_bits = sign_f32 | (static_cast<uint32_t>(f32_exp_val) << 23) | (mant_f16 << 13);
            }
        }
    } else { // Normal F16
        int32_t f32_exp = exp_f16 - exp_bias_f16 + exp_bias_f32;
        f32_bits = sign_f32 | (static_cast<uint32_t>(f32_exp) << 23) | (mant_f16 << 13);
    }

    float result;
    memcpy(&result, &f32_bits, sizeof(float));
    return result;
}

Shard::Shard(const std::string& fp) : file_path(fp) {
    Logger::info("Shard: Initializing for file: " + file_path);
#ifdef _WIN32
    file_handle = CreateFileA(file_path.c_str(), GENERIC_READ, FILE_SHARE_READ,
                               NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (file_handle == INVALID_HANDLE_VALUE) {
        throw std::runtime_error("Shard: Failed to open file (Windows): " + file_path + " Error: " + std::to_string(GetLastError()));
    }

    LARGE_INTEGER size_li;
    if (!GetFileSizeEx(file_handle, &size_li)) {
        CloseHandle(file_handle);
        file_handle = INVALID_HANDLE_VALUE; 
        throw std::runtime_error("Shard: Failed to get file size (Windows): " + file_path);
    }
    file_size = static_cast<size_t>(size_li.QuadPart);
    if (file_size == 0) { 
        CloseHandle(file_handle);
        file_handle = INVALID_HANDLE_VALUE;
        throw std::runtime_error("Shard: File is empty: " + file_path);
    }

    mapping_handle = CreateFileMapping(file_handle, NULL, PAGE_READONLY, 0, 0, NULL);
    if (mapping_handle == NULL) {
        CloseHandle(file_handle);
        file_handle = INVALID_HANDLE_VALUE;
        throw std::runtime_error("Shard: Failed to create file mapping (Windows): " + file_path + " Error: " + std::to_string(GetLastError()));
    }

    mapped_data = MapViewOfFile(mapping_handle, FILE_MAP_READ, 0, 0, file_size);
    if (mapped_data == nullptr) {
        CloseHandle(mapping_handle);
        mapping_handle = NULL;
        CloseHandle(file_handle);
        file_handle = INVALID_HANDLE_VALUE;
        throw std::runtime_error("Shard: Failed to map view of file (Windows): " + file_path + " Error: " + std::to_string(GetLastError()));
    }
#else // POSIX
    fd_ = open(file_path.c_str(), O_RDONLY);
    if (fd_ == -1) {
        throw std::runtime_error("Shard: Failed to open file: " + file_path + " Error: " + strerror(errno));
    }

    struct stat sb;
    if (fstat(fd_, &sb) == -1) {
        close(fd_);
        fd_ = -1; 
        throw std::runtime_error("Shard: Failed to get file size: " + file_path + " Error: " + strerror(errno));
    }
    file_size = sb.st_size;
    if (file_size == 0) { 
        close(fd_);
        fd_ = -1;
        throw std::runtime_error("Shard: File is empty: " + file_path);
    }

    mapped_data = mmap(NULL, file_size, PROT_READ, MAP_SHARED, fd_, 0);
    if (mapped_data == MAP_FAILED) {
        close(fd_);
        fd_ = -1;
        mapped_data = nullptr; 
        throw std::runtime_error("Shard: Failed to memory map file: " + file_path + " Error: " + strerror(errno));
    }
#endif
    Logger::debug("Shard: Successfully mapped file: " + file_path + ", size: " + std::to_string(file_size));

    if (file_size < 8) { 
        throw std::runtime_error("Shard: File too small (" + std::to_string(file_size) + " bytes) to be a valid SafeTensors shard (min 8 bytes for metadata length): " + file_path);
    }
    metadata_size = *reinterpret_cast<const uint64_t*>(mapped_data);
    
    if (metadata_size == 0) {
        throw std::runtime_error("Shard: Metadata size is 0 in file header: " + file_path);
    }
    if (8 + metadata_size > file_size) {
        throw std::runtime_error("Shard: Declared metadata size (" + std::to_string(metadata_size) + ") plus header (8 bytes) exceeds file size (" + std::to_string(file_size) + ") in: " + file_path);
    }
    metadata_ptr = static_cast<const uint8_t*>(mapped_data) + 8;
    tensor_data_block_ptr = metadata_ptr + metadata_size;
    Logger::debug("Shard: Metadata size from header: " + std::to_string(metadata_size) + " for " + file_path);
}

Shard::~Shard() {
    Logger::debug("Shard: Cleaning up for file: " + (file_path.empty() ? "(moved or uninitialized)" : file_path) ); 
#ifdef _WIN32
    if (mapped_data != nullptr) {
        if (!UnmapViewOfFile(mapped_data)) {
            Logger::error("Shard: Failed to unmap view of file (Windows) for \"" + file_path + "\" Error: " + std::to_string(GetLastError()));
        }
    }
    if (mapping_handle != NULL) {
        if (!CloseHandle(mapping_handle)) {
             Logger::error("Shard: Failed to close mapping handle (Windows) for \"" + file_path + "\" Error: " + std::to_string(GetLastError()));
        }
    }
    if (file_handle != INVALID_HANDLE_VALUE) {
        if (!CloseHandle(file_handle)) {
            Logger::error("Shard: Failed to close file handle (Windows) for \"" + file_path + "\" Error: " + std::to_string(GetLastError()));
        }
    }
    mapped_data = nullptr; 
    file_handle = INVALID_HANDLE_VALUE; 
    mapping_handle = NULL; 
#else // POSIX
    if (mapped_data != nullptr && mapped_data != MAP_FAILED) {
        if (munmap(mapped_data, file_size) == -1) {
            Logger::error("Shard: Failed to munmap file: \"" + file_path + "\" Error: " + strerror(errno));
        }
    }
    if (fd_ != -1) {
        if (close(fd_) == -1) {
             Logger::error("Shard: Failed to close file descriptor for \"" + file_path + "\" Error: " + strerror(errno));
        }
    }
    mapped_data = nullptr;
    fd_ = -1;
#endif
}

Shard::Shard(Shard&& other) noexcept
    : file_path(std::move(other.file_path)),
      mapped_data(other.mapped_data),
      file_size(other.file_size),
      metadata_size(other.metadata_size),
      metadata_ptr(other.metadata_ptr),
      tensor_data_block_ptr(other.tensor_data_block_ptr)
#ifdef _WIN32
      , file_handle(other.file_handle)
      , mapping_handle(other.mapping_handle)
#else
      , fd_(other.fd_)
#endif
{
    other.mapped_data = nullptr;
    other.file_size = 0;
    other.metadata_size = 0;
    other.metadata_ptr = nullptr;
    other.tensor_data_block_ptr = nullptr;
#ifdef _WIN32
    other.file_handle = INVALID_HANDLE_VALUE;
    other.mapping_handle = NULL;
#else
    other.fd_ = -1;
#endif
}

Shard& Shard::operator=(Shard&& other) noexcept {
    if (this != &other) {
        this->~Shard(); 
        file_path = std::move(other.file_path);
        mapped_data = other.mapped_data;
        file_size = other.file_size;
        metadata_size = other.metadata_size;
        metadata_ptr = other.metadata_ptr;
        tensor_data_block_ptr = other.tensor_data_block_ptr;
#ifdef _WIN32
        file_handle = other.file_handle;
        mapping_handle = other.mapping_handle;
#else
        fd_ = other.fd_;
#endif
        other.mapped_data = nullptr;
        other.file_size = 0; 
        other.metadata_size = 0;
        other.metadata_ptr = nullptr;
        other.tensor_data_block_ptr = nullptr;
#ifdef _WIN32
        other.file_handle = INVALID_HANDLE_VALUE;
        other.mapping_handle = NULL;
#else
        other.fd_ = -1;
#endif
    }
    return *this;
}

const uint8_t* Shard::get_tensor_raw_data(size_t local_offset, size_t n_bytes) const {
    if (!mapped_data || (mapped_data == MAP_FAILED && fd_ != -1) || !tensor_data_block_ptr) { 
        throw std::logic_error("Shard not properly mapped or initialized to get tensor data: " + file_path);
    }
    const uint8_t* data_start = tensor_data_block_ptr + local_offset;
    const uint8_t* shard_data_block_end = tensor_data_block_ptr + (file_size - (8 + metadata_size));

    if (data_start < tensor_data_block_ptr || data_start + n_bytes > shard_data_block_end || n_bytes > (file_size - (8 + metadata_size))) { 
        throw std::out_of_range(
            "Tensor data (local_offset: " + std::to_string(local_offset) +
            ", n_bytes: " + std::to_string(n_bytes) +
            ") out of bounds for data block of shard: " + file_path +
            ". Shard data block size: " + std::to_string(file_size - (8 + metadata_size)) + " bytes."
        );
    }
    return data_start;
}

SafeTensorsLoader::SafeTensorsLoader(const std::string& model_load_path)
    : model_load_path_(model_load_path), is_sharded_(false) {
    Logger::info("SafeTensorsLoader: Initializing for path: " + model_load_path_);
    std::filesystem::path path_obj(model_load_path_);

    if (!std::filesystem::exists(path_obj)){
        throw std::runtime_error("SafeTensorsLoader: Provided model_load_path does not exist: " + model_load_path_);
    }

    if (std::filesystem::is_directory(path_obj)) {
        Logger::info("SafeTensorsLoader: Path is a directory. Attempting to load from directory.");
        load_from_directory(model_load_path_); 
    } else if (std::filesystem::is_regular_file(path_obj)) {
        Logger::info("SafeTensorsLoader: Path is a single file. Loading single file.");
        std::string file_key = path_obj.filename().string();
        load_single_file(model_load_path_, file_key);
        is_sharded_ = false; 
    } else {
        throw std::runtime_error("SafeTensorsLoader: model_load_path is not a valid file or directory: " + model_load_path_);
    }

    if (tensors_.empty() && loaded_shards_.empty()) {
        Logger::warning("SafeTensorsLoader: Initialization complete, but no tensors were loaded and no shards mapped. Check model path and format: " + model_load_path_);
    } else {
        Logger::info("SafeTensorsLoader: Initialization complete. Total unique tensors mapped: " + std::to_string(tensors_.size()) +
                     " from " + std::to_string(loaded_shards_.size()) + " shard(s).");
    }
}

SafeTensorsLoader::~SafeTensorsLoader() {
    Logger::info("SafeTensorsLoader: Destructing. Clearing " + std::to_string(loaded_shards_.size()) + " loaded shards.");
    loaded_shards_.clear(); 
    Logger::info("SafeTensorsLoader: All shards cleared.");
}

void SafeTensorsLoader::load_from_directory(const std::string& directory_path_str) {
    Logger::debug("SafeTensorsLoader::load_from_directory for '" + directory_path_str + "'.");
    std::filesystem::path dir_p(directory_path_str);
    std::filesystem::path index_json_path_v1 = dir_p / "model.safetensors.index.json";
    std::filesystem::path index_json_path_v2 = dir_p / "pytorch_model.bin.index.json"; 
    std::filesystem::path actual_index_path;

    bool index_found = false;
    if (std::filesystem::exists(index_json_path_v1) && std::filesystem::is_regular_file(index_json_path_v1)) {
        actual_index_path = index_json_path_v1;
        index_found = true;
    } else if (std::filesystem::exists(index_json_path_v2) && std::filesystem::is_regular_file(index_json_path_v2)) {
        actual_index_path = index_json_path_v2;
        index_found = true;
    }

    if (index_found) {
        Logger::info("SafeTensorsLoader: Found index file: " + actual_index_path.string());
        is_sharded_ = true; 
        std::ifstream f(actual_index_path.string());
        if (!f.is_open()) {
            throw std::runtime_error("SafeTensorsLoader: Failed to open index file: " + actual_index_path.string());
        }
        nlohmann::json index_json_data;
        try {
            index_json_data = nlohmann::json::parse(f);
        } catch (const nlohmann::json::parse_error& e) {
            f.close();
            throw std::runtime_error("SafeTensorsLoader: Failed to parse index JSON from " + actual_index_path.string() + ": " + e.what());
        }
        f.close();

        if (index_json_data.count("weight_map") && index_json_data["weight_map"].is_object()) {
            // First pass: populate tensor_name_to_shard_key_map_ and identify unique shards to load
            std::map<std::string, std::string> unique_shards_to_load; // shard_filename -> full_path
            for (auto const& [tensor_name, shard_filename_json] : index_json_data["weight_map"].items()) {
                if (!shard_filename_json.is_string()) {
                    Logger::warning("SafeTensorsLoader: Shard filename for tensor '" + tensor_name + "' in index is not a string. Skipping.");
                    continue;
                }
                std::string shard_filename = shard_filename_json.get<std::string>();
                tensor_name_to_shard_key_map_[tensor_name] = shard_filename; 
                if (unique_shards_to_load.find(shard_filename) == unique_shards_to_load.end()) {
                     unique_shards_to_load[shard_filename] = (dir_p / shard_filename).string();
                }
            }
           
            // Second pass: load each unique shard and parse its metadata
            for(const auto& pair : unique_shards_to_load){
                const std::string& shard_filename = pair.first;
                const std::string& full_shard_path = pair.second;
                if (loaded_shards_.find(shard_filename) == loaded_shards_.end()) {
                    Logger::info("SafeTensorsLoader: Loading and parsing shard (from index): " + full_shard_path + " (key:"+ shard_filename + ")");
                    load_single_file(full_shard_path, shard_filename); 
                } else {
                     Logger::debug("SafeTensorsLoader: Shard '" + shard_filename + "' already loaded/parsed (should not happen if unique_shards logic is correct).");
                }
            }

        } else {
            throw std::runtime_error("SafeTensorsLoader: Index file " + actual_index_path.string() + " does not contain a valid 'weight_map'.");
        }
    } else {
        Logger::info("SafeTensorsLoader: No index file found in " + directory_path_str + ". Scanning for *.safetensors files.");
        std::vector<std::filesystem::path> shard_files;
        for (const auto& entry : std::filesystem::directory_iterator(dir_p)) {
            if (entry.is_regular_file() && entry.path().extension() == ".safetensors") {
                shard_files.push_back(entry.path());
            }
        }

        if (shard_files.empty()) {
             Logger::warning("SafeTensorsLoader: No .safetensors files found directly in directory: " + directory_path_str + ". Checking for model.safetensors as last resort.");
            std::filesystem::path single_model_file = dir_p / "model.safetensors";
            if(std::filesystem::exists(single_model_file) && std::filesystem::is_regular_file(single_model_file)){
                Logger::info("SafeTensorsLoader: Found 'model.safetensors' in directory, loading it as a single non-sharded model.");
                load_single_file(single_model_file.string(), single_model_file.filename().string());
                is_sharded_ = false;
            } else {
                 Logger::info("SafeTensorsLoader: No .safetensors files or index.json found in directory: " + directory_path_str + ". No model weights will be loaded from this path directly.");
            }
        } else if (shard_files.size() == 1) {
            Logger::info("SafeTensorsLoader: Found single .safetensors file: " + shard_files[0].string() + ". Loading as non-sharded.");
            load_single_file(shard_files[0].string(), shard_files[0].filename().string());
            is_sharded_ = false;
        } else {
            Logger::info("SafeTensorsLoader: Found " + std::to_string(shard_files.size()) + " .safetensors files (no index). Loading all as individual shards.");
            is_sharded_ = true;
            for (const auto& p : shard_files) {
                load_single_file(p.string(), p.filename().string());
            }
        }
    }
}

void SafeTensorsLoader::load_single_file(const std::string& file_path, const std::string& shard_key_override) {
    std::string key_to_use = shard_key_override.empty() ? std::filesystem::path(file_path).filename().string() : shard_key_override;
    if (key_to_use.empty()) key_to_use = file_path; 

    if (loaded_shards_.count(key_to_use)) {
        Logger::debug("SafeTensorsLoader: Shard/file '" + key_to_use + "' (path: " + file_path + ") already processed/loaded.");
        return;
    }
    Logger::info("SafeTensorsLoader: Loading single file/shard: " + file_path + " with key: " + key_to_use);
    try {
        auto shard = std::make_unique<Shard>(file_path);
        parse_shard_metadata(*shard, key_to_use); 
        loaded_shards_[key_to_use] = std::move(shard);
    } catch (const std::exception& e) {
        throw std::runtime_error("SafeTensorsLoader: Error processing file/shard '" + file_path + "' (key: " + key_to_use + "): " + e.what());
    }
}

void SafeTensorsLoader::parse_shard_metadata(Shard& shard, const std::string& shard_key) {
    Logger::debug("SafeTensorsLoader: Parsing metadata for shard: " + shard_key + " (file: " + shard.file_path + ")");
    if (!shard.metadata_ptr || shard.metadata_size == 0) {
        throw std::runtime_error("Shard metadata is not available for parsing (nullptr or zero size): " + shard.file_path);
    }
    std::string metadata_json_str;
    try {
        metadata_json_str.assign(reinterpret_cast<const char*>(shard.metadata_ptr), shard.metadata_size);
    } catch (const std::length_error& le) {
        throw std::runtime_error("Error constructing metadata string for shard " + shard.file_path + ": " + le.what());
    }
    
    nlohmann::json metadata_root;
    try {
        metadata_root = nlohmann::json::parse(metadata_json_str);
    } catch (const nlohmann::json::parse_error& e) {
        throw std::runtime_error("Failed to parse metadata JSON for shard " + shard.file_path + " (key: " + shard_key + ") at offset 8, metadata_size: " + 
                                 std::to_string(shard.metadata_size) + ". Error: " + e.what() + 
                                 "\nJSON content snippet (first 200 chars): " + metadata_json_str.substr(0, 200));
    }

    size_t tensors_in_this_shard_count = 0;
    for (auto const& [tensor_name_str, info_json] : metadata_root.items()) {
        if (tensor_name_str == "__metadata__") continue; 

        TensorInfo tensor_info;
        tensor_info.name = tensor_name_str;
        try {
            tensor_info.dtype = info_json.at("dtype").get<std::string>();
            std::transform(tensor_info.dtype.begin(), tensor_info.dtype.end(), tensor_info.dtype.begin(),
                           [](unsigned char c){ return static_cast<char>(std::toupper(c)); });

            for (const auto& dim : info_json.at("shape")) {
                tensor_info.shape.push_back(dim.get<size_t>());
            }
            const auto& data_offsets_json = info_json.at("data_offsets");
            if (!data_offsets_json.is_array() || data_offsets_json.size() != 2) {
                 throw std::runtime_error("Tensor '" + tensor_name_str + "' 'data_offsets' must be an array of two numbers.");
            }
            size_t start_offset_in_data_block = data_offsets_json[0].get<size_t>();
            size_t end_offset_in_data_block = data_offsets_json[1].get<size_t>();
            
            tensor_info.data_offset = start_offset_in_data_block; 
            tensor_info.nbytes = end_offset_in_data_block - start_offset_in_data_block;
            tensor_info.shard_key = shard_key;

            if (tensors_.count(tensor_info.name)) {
                 Logger::warning("SafeTensorsLoader: Duplicate tensor name '" + tensor_info.name + "' encountered. " + 
                                 "Previous shard key: '" + tensors_[tensor_info.name].shard_key + "', New shard key: '" + shard_key + "'. " +
                                 "Overwriting with info from current shard being parsed. This can happen with unindexed multi-file loads or inconsistent index files.");
            }
            tensors_[tensor_info.name] = tensor_info;
            if (tensor_name_to_shard_key_map_.find(tensor_info.name) == tensor_name_to_shard_key_map_.end()){
                tensor_name_to_shard_key_map_[tensor_info.name] = shard_key;
            }

            tensors_in_this_shard_count++;

        } catch (const nlohmann::json::exception& e) {
            throw std::runtime_error("Failed to parse tensor info for '" + tensor_name_str + "' in shard " +
                                     shard.file_path + " (key: " + shard_key + "): " + e.what());
        }
    }
     Logger::debug("SafeTensorsLoader: Finished parsing metadata for shard: " + shard_key + ". Parsed " + std::to_string(tensors_in_this_shard_count) + " tensor entries from this shard.");
}

std::vector<std::string> SafeTensorsLoader::tensor_names() const {
    std::vector<std::string> names;
    names.reserve(tensors_.size());
    for (const auto& pair : tensors_) {
        names.push_back(pair.first);
    }
    return names;
}

const SafeTensorsLoader::TensorInfo& SafeTensorsLoader::get_tensor_info(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        throw std::runtime_error("Tensor not found in SafeTensorsLoader metadata: " + name);
    }
    return it->second;
}

const Shard* SafeTensorsLoader::get_shard_for_tensor(const std::string& tensor_name) const {
    auto map_it = tensor_name_to_shard_key_map_.find(tensor_name);
    std::string determined_shard_key;

    if (map_it != tensor_name_to_shard_key_map_.end()){
        determined_shard_key = map_it->second;
    } else {
        const auto& tensor_info_direct = get_tensor_info(tensor_name);
        determined_shard_key = tensor_info_direct.shard_key;
    }
    
    if (determined_shard_key.empty()){
         throw std::logic_error("Internal inconsistency: Could not determine shard key for tensor '" + tensor_name + "'.");
    }

    auto shard_it = loaded_shards_.find(determined_shard_key);
    if (shard_it == loaded_shards_.end()) {
        throw std::logic_error("Internal inconsistency: Shard key '" + determined_shard_key + "' for tensor '" + tensor_name + "' not found in loaded_shards_ map. Tensors map has it, but shard object itself is missing.");
    }
    return shard_it->second.get();
}

std::vector<uint8_t> SafeTensorsLoader::get_tensor_bytes(const std::string& name) const {
    const TensorInfo& info = get_tensor_info(name); 
    const Shard* shard = get_shard_for_tensor(name); 
    
    const uint8_t* raw_data_ptr = shard->get_tensor_raw_data(info.data_offset, info.nbytes);
    return convert_tensor_data(raw_data_ptr, info.nbytes, info.dtype);
}

std::map<std::string, std::vector<uint8_t>> SafeTensorsLoader::load_all_tensors_parallel() const {
    std::map<std::string, std::vector<uint8_t>> result_map;
    if (tensors_.empty()) {
        Logger::debug("SafeTensorsLoader::load_all_tensors_parallel: No tensors to load.");
        return result_map;
    }

    std::vector<std::future<std::pair<std::string, std::vector<uint8_t>>>> futures;
    unsigned int n_threads = std::max(1u, std::thread::hardware_concurrency());
    n_threads = std::min(n_threads, static_cast<unsigned int>(tensors_.size())); 
    if (n_threads > 16) n_threads = 16; 
    
    ThreadPool pool(n_threads);
    Logger::info("SafeTensorsLoader: Loading all " + std::to_string(tensors_.size()) + " tensors in parallel using " + std::to_string(n_threads) + " threads.");

    for (const auto& pair : tensors_) {
        const std::string& tensor_name = pair.first;
        futures.push_back(pool.submit([this, tensor_name]() {
            std::vector<uint8_t> data = this->get_tensor_bytes(tensor_name); 
            return std::make_pair(tensor_name, std::move(data));
        }));
    }

    for (auto& fut : futures) {
        try {
            std::pair<std::string, std::vector<uint8_t>> tensor_pair = fut.get();
            result_map[tensor_pair.first] = std::move(tensor_pair.second);
        } catch (const std::exception& e) {
            Logger::error("SafeTensorsLoader: Error loading a tensor in parallel task: " + std::string(e.what()));
            throw; 
        }
    }
    Logger::info("SafeTensorsLoader: Finished loading all tensors in parallel.");
    return result_map;
}

std::vector<uint8_t> SafeTensorsLoader::convert_tensor_data(const uint8_t* data_ptr, size_t n_bytes, const std::string& dtype_str_upper) const {
    if (dtype_str_upper == "F32") {
        return std::vector<uint8_t>(data_ptr, data_ptr + n_bytes);
    } else if (dtype_str_upper == "F16") {
        size_t num_elements = n_bytes / 2;
        std::vector<float> f32_vec(num_elements);
        const uint16_t* f16_ptr = reinterpret_cast<const uint16_t*>(data_ptr);
        for (size_t i = 0; i < num_elements; ++i) {
             f32_vec[i] = cpu_f16_to_float32(f16_ptr[i]);
        }
        std::vector<uint8_t> bytes_out(num_elements * sizeof(float));
        memcpy(bytes_out.data(), f32_vec.data(), bytes_out.size());
        return bytes_out;
    } else if (dtype_str_upper == "BF16") {
        size_t num_elements = n_bytes / 2;
        std::vector<float> f32_vec(num_elements);
        const uint16_t* bf16_ptr = reinterpret_cast<const uint16_t*>(data_ptr);
        for (size_t i = 0; i < num_elements; ++i) {
            f32_vec[i] = cpu_bf16_to_float32(bf16_ptr[i]);
        }
        std::vector<uint8_t> bytes_out(num_elements * sizeof(float));
        memcpy(bytes_out.data(), f32_vec.data(), bytes_out.size());
        return bytes_out;
    }
    throw std::runtime_error("SafeTensorsLoader: Unsupported tensor dtype for conversion: " + dtype_str_upper);
}

bool SafeTensorsLoader::load_model_config_from_json(const std::string& model_path_or_dir_str, ModelConfig& config_to_populate) {
    std::filesystem::path model_fs_path(model_path_or_dir_str);
    std::filesystem::path config_json_path;

    if (std::filesystem::is_directory(model_fs_path)) {
        config_json_path = model_fs_path / "config.json";
    } else if (std::filesystem::is_regular_file(model_fs_path)) {
        config_json_path = model_fs_path.parent_path() / "config.json";
    } else {
        Logger::error("SafeTensorsLoader::load_model_config_from_json: Provided model path is not a valid file or directory: " + model_path_or_dir_str);
        return false;
    }
    std::string config_json_path_str = config_json_path.string();

    std::ifstream f(config_json_path_str);
    if (!f.is_open()) {
        Logger::warning("SafeTensorsLoader: config.json not found at: " + config_json_path_str);
        return false;
    }

    try {
        nlohmann::json data = nlohmann::json::parse(f);
        f.close();
        
        config_to_populate.hidden_size = data.value("hidden_size", 0);
        config_to_populate.intermediate_size = data.value("intermediate_size", 0);
        config_to_populate.num_attention_heads = data.value("num_attention_heads", 0);
        config_to_populate.num_key_value_heads = data.value("num_key_value_heads", config_to_populate.num_attention_heads);
        config_to_populate.num_hidden_layers = data.value("num_hidden_layers", 0);
        config_to_populate.vocab_size = data.value("vocab_size", 0);
        config_to_populate.max_position_embeddings = data.value("max_position_embeddings", 2048); 
        config_to_populate.rms_norm_eps = data.value("rms_norm_eps", 1e-5f);
        config_to_populate.rope_theta = data.value("rope_theta", 10000.0f);
        config_to_populate.bos_token_id = data.value("bos_token_id", 1);
        config_to_populate.eos_token_id = data.value("eos_token_id", 2); 
        config_to_populate.pad_token_id = data.value("pad_token_id", -1); 
        config_to_populate.unk_token_id = data.value("unk_token_id", 0); 

        if (data.contains("architectures") && data["architectures"].is_array() && !data["architectures"].empty()) {
            config_to_populate.architecture = data["architectures"][0].get<std::string>();
        } else {
            config_to_populate.architecture = data.value("model_type", "unknown");
        }
        config_to_populate.model_name = data.value("model_type", config_to_populate.architecture);

        bool is_llama3_vocab_size_json = (config_to_populate.vocab_size == 128256);
        bool is_llama3_arch_hint_json = (config_to_populate.architecture.find("LlamaForCausalLM") != std::string::npos &&
                               config_to_populate.architecture.find("Llama2") == std::string::npos);

        if (is_llama3_vocab_size_json && is_llama3_arch_hint_json) {
            config_to_populate.tokenizer_family = ModelConfig::TokenizerFamily::LLAMA3_TIKTOKEN;
             if (config_to_populate.rope_theta == 10000.0f) { 
                float llama3_rope_candidate = data.value("rope_theta", 500000.0f);
                if (llama3_rope_candidate > 10000.0f) { 
                    config_to_populate.rope_theta = llama3_rope_candidate;
                } else if (config_to_populate.rope_theta == 10000.0f) { 
                     config_to_populate.rope_theta = 500000.0f;
                }
            }
        } else if (config_to_populate.vocab_size == 32000 || config_to_populate.architecture.find("Llama") != std::string::npos) {
            config_to_populate.tokenizer_family = ModelConfig::TokenizerFamily::LLAMA_SENTENCEPIECE;
        } else {
            config_to_populate.tokenizer_family = ModelConfig::TokenizerFamily::UNKNOWN;
        }
        config_to_populate.is_gguf_file_loaded = false; 

        Logger::info("SafeTensorsLoader: Successfully loaded and parsed model config from: " + config_json_path_str);
        return true;

    } catch (const nlohmann::json::exception& e) {
        Logger::error("SafeTensorsLoader: Failed to parse config.json: " + config_json_path_str + ". Error: " + e.what());
        return false;
    }
    return false; 
}


ThreadPool::ThreadPool(size_t num_threads) : stop_(false) {
    for (size_t i = 0; i < num_threads; ++i) {
        workers_.emplace_back([this] {
            while (true) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(this->queue_mutex_);
                    this->condition_.wait(lock, [this] {
                        return this->stop_ || !this->tasks_.empty();
                    });
                    if (this->stop_ && this->tasks_.empty()) return;
                    task = std::move(this->tasks_.front());
                    this->tasks_.pop();
                }
                if(task) task(); 
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
        if (worker.joinable()) { 
            worker.join();
        }
    }
}