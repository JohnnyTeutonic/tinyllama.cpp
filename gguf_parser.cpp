#include "gguf_parser.h"

#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <cstring> // For strerror
#include <cerrno>  // For errno

// mmap related includes are now in gguf_structs.h, but also here for clarity/safety
#include <sys/mman.h>   // For mmap, munmap
#include <sys/stat.h>   // For fstat, stat
#include <fcntl.h>      // For O_RDONLY
#include <unistd.h>     // For close, fstat, read, lseek

#include "logger.h"
#include "quantization.h"

size_t gguf_value_type_size(GGUFValueType type) {
  switch (type) {
    case GGUFValueType::UINT8:
      return sizeof(uint8_t);
    case GGUFValueType::INT8:
      return sizeof(int8_t);
    case GGUFValueType::UINT16:
      return sizeof(uint16_t);
    case GGUFValueType::INT16:
      return sizeof(int16_t);
    case GGUFValueType::UINT32:
      return sizeof(uint32_t);
    case GGUFValueType::INT32:
      return sizeof(int32_t);
    case GGUFValueType::FLOAT32:
      return sizeof(float);
    case GGUFValueType::BOOL:
      return sizeof(uint8_t);
    case GGUFValueType::UINT64:
      return sizeof(uint64_t);
    case GGUFValueType::INT64:
      return sizeof(int64_t);
    case GGUFValueType::FLOAT64:
      return sizeof(double);
    case GGUFValueType::STRING:
      return 0;
    case GGUFValueType::ARRAY:
      return 0;
    default:
      return 0;
  }
}

template <typename T>
void read_raw(std::ifstream& file, T& dest) {
  file.read(reinterpret_cast<char*>(&dest), sizeof(T));
  if (!file) {
    throw std::runtime_error(
        "GGUF Error: Failed to read data from file stream.");
  }
}

template void read_raw<uint8_t>(std::ifstream&, uint8_t&);
template void read_raw<int8_t>(std::ifstream&, int8_t&);
template void read_raw<uint16_t>(std::ifstream&, uint16_t&);
template void read_raw<int16_t>(std::ifstream&, int16_t&);
template void read_raw<uint32_t>(std::ifstream&, uint32_t&);
template void read_raw<int32_t>(std::ifstream&, int32_t&);
template void read_raw<float>(std::ifstream&, float&);
template void read_raw<uint64_t>(std::ifstream&, uint64_t&);
template void read_raw<int64_t>(std::ifstream&, int64_t&);
template void read_raw<double>(std::ifstream&, double&);
template void read_raw<GGUFValueType>(std::ifstream&, GGUFValueType&);

std::string read_gguf_string(std::ifstream& file) {
  uint64_t len;
  read_raw(file, len);
  if (len > 0) {
    if (len > GGUF_STRING_MAX_LENGTH) {
      throw std::runtime_error(
          "GGUF Error: String length exceeds sanity limit: " +
          std::to_string(len));
    }
    std::vector<char> buf(static_cast<size_t>(len));
    file.read(buf.data(), static_cast<std::streamsize>(len));
    if (!file) {
      throw std::runtime_error("GGUF Error: Failed to read string data.");
    }
    return std::string(buf.data(), static_cast<size_t>(len));
  } else {
    return "";
  }
}

GGUFData load_gguf_meta(const std::string& filename, bool use_mmap) {
  Logger::info("Attempting to load GGUF file: " + filename + (use_mmap ? " with mmap" : " without mmap"));
  std::ifstream metadata_file(filename, std::ios::binary);
  if (!metadata_file.is_open()) {
    throw std::runtime_error("Failed to open file for metadata: " + filename);
  }

  GGUFData result;
  // The file_descriptor for mmap will be opened separately and stored in result.
  // The GGUFData destructor will handle closing this fd and munmap.

  read_raw(metadata_file, result.header.magic);
  read_raw(metadata_file, result.header.version);
  read_raw(metadata_file, result.header.tensor_count);
  read_raw(metadata_file, result.header.metadata_kv_count);

  {
    std::stringstream ss;
    ss << "Read Header:\n"
       << "    Magic: 0x" << std::hex << result.header.magic << std::dec << "\n"
       << "    Version: " << result.header.version << "\n"
       << "    Tensor Count: " << result.header.tensor_count << "\n"
       << "    Metadata KV Count: " << result.header.metadata_kv_count;
    Logger::info(ss.str());
  }

  if (result.header.magic != GGUF_MAGIC) {
    throw std::runtime_error("Not a valid GGUF file (magic number mismatch).");
  }

  Logger::info("Reading Metadata (" +
               std::to_string(result.header.metadata_kv_count) + " pairs)...");
  for (uint64_t i = 0; i < result.header.metadata_kv_count; ++i) {
    std::string key;
    GGUFValueType value_type_enum;
    try {
      key = read_gguf_string(metadata_file);
      read_raw(metadata_file, value_type_enum);

      switch (value_type_enum) {
        case GGUFValueType::UINT8: {
          uint8_t val;
          read_raw(metadata_file, val);
          result.metadata[key] = val;
          break;
        }
        case GGUFValueType::INT8: {
          int8_t val;
          read_raw(metadata_file, val);
          result.metadata[key] = val;
          break;
        }
        case GGUFValueType::UINT16: {
          uint16_t val;
          read_raw(metadata_file, val);
          result.metadata[key] = val;
          break;
        }
        case GGUFValueType::INT16: {
          int16_t val;
          read_raw(metadata_file, val);
          result.metadata[key] = val;
          break;
        }
        case GGUFValueType::UINT32: {
          uint32_t val;
          read_raw(metadata_file, val);
          result.metadata[key] = val;
          break;
        }
        case GGUFValueType::INT32: {
          int32_t val;
          read_raw(metadata_file, val);
          result.metadata[key] = val;
          break;
        }
        case GGUFValueType::FLOAT32: {
          float val;
          read_raw(metadata_file, val);
          result.metadata[key] = val;
          break;
        }
        case GGUFValueType::BOOL: {
          uint8_t byte;
          read_raw(metadata_file, byte);
          result.metadata[key] = (byte != 0);
          break;
        }
        case GGUFValueType::STRING: {
          std::string val = read_gguf_string(metadata_file);
          result.metadata[key] = val;
          break;
        }
        case GGUFValueType::UINT64: {
          uint64_t val;
          read_raw(metadata_file, val);
          result.metadata[key] = val;
          break;
        }
        case GGUFValueType::INT64: {
          int64_t val;
          read_raw(metadata_file, val);
          result.metadata[key] = val;
          break;
        }
        case GGUFValueType::FLOAT64: {
          double val;
          read_raw(metadata_file, val);
          result.metadata[key] = val;
          break;
        }
        case GGUFValueType::ARRAY: {
          GGUFValueType array_type_enum;
          uint64_t count;
          read_raw(metadata_file, array_type_enum);
          read_raw(metadata_file, count);

          GGUFArray array_obj;
          array_obj.type = array_type_enum;
          array_obj.len = count;
          result.metadata[key] = array_obj;
          bool skipped_data = false;
          if (key == "tokenizer.ggml.tokens" &&
              array_type_enum == GGUFValueType::STRING) {
            Logger::info("Loading STRING array data ('" + key + "') with " +
                         std::to_string(count) + " elements...");
            result.tokenizer_tokens.reserve(static_cast<size_t>(count));
            for (uint64_t arr_i = 0; arr_i < count; ++arr_i) {
              result.tokenizer_tokens.push_back(read_gguf_string(metadata_file));
            }
            Logger::info("Loaded tokenizer_tokens. Size: " +
                         std::to_string(result.tokenizer_tokens.size()));
          } else if (key == "tokenizer.ggml.scores" &&
                     array_type_enum == GGUFValueType::FLOAT32) {
            Logger::info("Loading FLOAT32 array data ('" + key + "') with " +
                         std::to_string(count) + " elements...");
            result.tokenizer_scores.resize(static_cast<size_t>(count));
            metadata_file.read(reinterpret_cast<char*>(result.tokenizer_scores.data()),
                      static_cast<std::streamsize>(count * sizeof(float)));
            if (!metadata_file) {
              throw std::runtime_error(
                  "GGUF Error: Failed to read scores array data.");
            }
            Logger::info("Loaded tokenizer_scores. Size: " +
                         std::to_string(result.tokenizer_scores.size()));
          } else if (key == "tokenizer.ggml.token_type" &&
                     (array_type_enum == GGUFValueType::UINT32 || array_type_enum == GGUFValueType::INT32) ) {
            Logger::info("Loading " + std::string(array_type_enum == GGUFValueType::UINT32 ? "UINT32" : "INT32") + 
                         " array data ('" + key + "') with " +
                         std::to_string(count) + " elements...");
            result.tokenizer_token_types.resize(static_cast<size_t>(count));
            if (array_type_enum == GGUFValueType::UINT32) {
                metadata_file.read(
                    reinterpret_cast<char*>(result.tokenizer_token_types.data()),
                    static_cast<std::streamsize>(count * sizeof(uint32_t)));
            } else { // GGUFValueType::INT32
                std::vector<int32_t> temp_s32_types(static_cast<size_t>(count));
                metadata_file.read(
                    reinterpret_cast<char*>(temp_s32_types.data()),
                    static_cast<std::streamsize>(count * sizeof(int32_t)));
                for(size_t k=0; k < count; ++k) {
                    result.tokenizer_token_types[k] = static_cast<uint32_t>(temp_s32_types[k]);
                }
            }
            if (!metadata_file) {
              throw std::runtime_error(
                  "GGUF Error: Failed to read token_type array data.");
            }
            Logger::info("Loaded tokenizer_token_types. Size: " +
                         std::to_string(result.tokenizer_token_types.size()));
          } else if (key == "tokenizer.ggml.merges" &&
                     array_type_enum == GGUFValueType::STRING) {
            Logger::info("Loading STRING array data ('" + key + "') with " +
                         std::to_string(count) + " elements...");
            result.tokenizer_merges.reserve(static_cast<size_t>(count));
            for (uint64_t arr_i = 0; arr_i < count; ++arr_i) {
              result.tokenizer_merges.push_back(read_gguf_string(metadata_file));
            }
            Logger::info("Loaded tokenizer_merges. Size: " +
                         std::to_string(result.tokenizer_merges.size()));
          } else {
            skipped_data = true;
            Logger::info(
                "Skipping unhandled/non-tokenizer ARRAY data for key '" + key +
                "' (Type: " +
                std::to_string(static_cast<uint32_t>(array_type_enum)) +
                ", Count: " + std::to_string(count) + ")");

            if (array_type_enum == GGUFValueType::STRING) {
              for (uint64_t arr_i = 0; arr_i < count; ++arr_i) {
                try {
                  std::string discarded_str = read_gguf_string(metadata_file);
                } catch (const std::exception& e) {
                  Logger::error("Error skipping string element " +
                                std::to_string(arr_i) + " for key '" + key +
                                "': " + e.what());
                  throw;
                }
              }
            } else {
              size_t element_size = gguf_value_type_size(array_type_enum);
              if (element_size == 0) {
                throw std::runtime_error(
                    "Cannot skip array for key '" + key +
                    "' with unsupported or variable-sized element type: " +
                    std::to_string(static_cast<uint32_t>(array_type_enum)));
              }

              if (count > 0 &&
                  element_size > std::numeric_limits<uint64_t>::max() / count) {
                throw std::overflow_error(
                    "Array size overflow calculating skip amount for key '" +
                    key + "'");
              }
              uint64_t total_size_to_skip = count * element_size;
              if (total_size_to_skip > 0) {
                metadata_file.seekg(static_cast<std::streamoff>(total_size_to_skip),
                           std::ios::cur);
                if (!metadata_file) {
                  throw std::runtime_error(
                      "GGUF Error: Failed to seek past array data for key '" +
                      key + "'");
                }
              }
            }
          }
          break;
        }
        default: {
          throw std::runtime_error(
              "Unknown metadata type encountered: " +
              std::to_string(static_cast<uint32_t>(value_type_enum)) +
              " for key: " + key);
        }
      }
    } catch (const std::exception& e) {
      std::string error_key =
          key.empty() ? "(unknown key, error during key read)" : key;
      Logger::error(
          "Error reading metadata for key: '" + error_key +
          "' (type: " + std::to_string(static_cast<uint32_t>(value_type_enum)) +
          ") - " + e.what());
      throw;
    }
  }
  Logger::info("Finished reading metadata.");

  result.tensor_infos.reserve(static_cast<size_t>(result.header.tensor_count));
  Logger::info("Reading Tensor Info (" +
               std::to_string(result.header.tensor_count) + " tensors)...");
  uint64_t accumulated_offset_debug = 0;
  for (uint64_t i = 0; i < result.header.tensor_count; ++i) {
    GGUFTensorInfo info;
    try {
      info.name = read_gguf_string(metadata_file);

      uint32_t n_dims;
      read_raw(metadata_file, n_dims);
      if (n_dims > GGUF_MAX_TENSOR_DIMS) {
        throw std::runtime_error("Tensor '" + info.name +
                                 "' has unsupported number of dimensions: " +
                                 std::to_string(n_dims));
      }
      info.shape.resize(n_dims);
      for (uint32_t d = 0; d < n_dims; ++d) {
        read_raw(metadata_file, info.shape[d]);
      }

      uint32_t ggml_type_u32;
      read_raw(metadata_file, ggml_type_u32);
      info.type = static_cast<GGMLType>(ggml_type_u32);

      uint64_t pos_before_offset_read = metadata_file.tellg();

      read_raw(metadata_file, info.offset);

      std::stringstream ss_offset_log;
      ss_offset_log
          << "[GGUF_TENSOR_INFO] Tensor " << i << " ('" << info.name
          << "'):" << "\n  Raw offset from file: " << info.offset
          << "\n  File pos before offset read: " << pos_before_offset_read
          << "\n  Calculated accumulated_offset_debug (before this tensor): "
          << accumulated_offset_debug;

      info.num_elements = 1;
      for (uint64_t dim : info.shape) {
        if (dim > 0 &&
            info.num_elements > std::numeric_limits<uint64_t>::max() / dim) {
          throw std::overflow_error(
              "Tensor dimension overflow calculating num_elements for tensor "
              "'" +
              info.name + "'");
        }
        info.num_elements *= dim;
      }

      size_t type_size = ggml_type_size(info.type);
      size_t block_size = ggml_type_block_size(info.type);

      if (block_size == 0 && info.num_elements > 0) {
        throw std::runtime_error(
            "Tensor '" + info.name +
            "' has unknown or unsupported type: " + std::to_string(info.type));
      }

      if (block_size > 1) {
        if (info.num_elements % block_size != 0) {
          throw std::runtime_error("Tensor '" + info.name + "' num_elements (" +
                                   std::to_string(info.num_elements) +
                                   ") not divisible by block_size (" +
                                   std::to_string(block_size) + ") for type " +
                                   ggml_type_name(info.type));
        }
        uint64_t num_blocks = info.num_elements / block_size;
        if (type_size > 0 &&
            num_blocks > std::numeric_limits<uint64_t>::max() / type_size) {
          throw std::overflow_error(
              "Tensor size overflow calculating size_in_bytes for tensor '" +
              info.name + "'");
        }
        info.size_in_bytes = static_cast<size_t>(num_blocks * type_size);
      } else {
        if (type_size > 0 &&
            info.num_elements >
                std::numeric_limits<uint64_t>::max() / type_size) {
          throw std::overflow_error(
              "Tensor size overflow calculating size_in_bytes for tensor '" +
              info.name + "'");
        }
        info.size_in_bytes = static_cast<size_t>(info.num_elements * type_size);
      }

      ss_offset_log << "\n  Calculated size_in_bytes for this tensor: "
                    << info.size_in_bytes;
      Logger::info(ss_offset_log.str());
      accumulated_offset_debug += info.size_in_bytes;

      result.tensor_infos.push_back(info);
      {
        std::stringstream ss_tensor;
        ss_tensor << "Tensor " << i << ": Name='" << info.name
                  << "', Type=" << ggml_type_name(info.type) << ", Shape=[ ";
        for (size_t d = 0; d < info.shape.size(); ++d)
          ss_tensor << info.shape[d]
                    << (d == info.shape.size() - 1 ? "" : ", ");
        ss_tensor << " ], Offset=" << info.offset
                  << ", Size=" << info.size_in_bytes << " bytes";
        Logger::info(ss_tensor.str());
      }
    } catch (const std::exception& e) {
      std::string tensor_name =
          info.name.empty() ? ("(unknown, index " + std::to_string(i) + ")")
                            : info.name;
      Logger::error("Error reading tensor info for tensor " + tensor_name +
                    ": " + e.what());
      throw;
    }
  }
  Logger::info("Finished reading tensor info.");

  Logger::info("Populating tensor_infos_map...");
  for (const auto& tinfo : result.tensor_infos) {
    if (result.tensor_infos_map.count(tinfo.name)) {
      Logger::warning("Duplicate tensor name found in GGUF: '" + tinfo.name +
                      "'. Overwriting entry in map.");
    }
    result.tensor_infos_map[tinfo.name] = tinfo;
  }
  Logger::info("Finished populating tensor_infos_map. Map size: " +
               std::to_string(result.tensor_infos_map.size()));

  uint64_t alignment = GGUF_DEFAULT_ALIGNMENT;
  try {
    if (result.metadata.count("general.alignment")) {
      uint32_t align_val =
          std::get<uint32_t>(result.metadata["general.alignment"]);
      if (align_val > 0) {
        alignment = align_val;
      }
      Logger::info("Using alignment value from metadata: " +
                   std::to_string(alignment));
    } else {
      Logger::info(
          "Metadata key 'general.alignment' not found. Using default "
          "alignment: " +
          std::to_string(alignment));
    }
  } catch (const std::bad_variant_access& e) {
    Logger::warning(
        "Could not read 'general.alignment' metadata as uint32. Using default "
        "alignment: " +
        std::to_string(alignment));
  } catch (const std::exception& e) {
    Logger::warning("Error accessing 'general.alignment' metadata: " +
                    std::string(e.what()) +
                    ". Using default alignment: " + std::to_string(alignment));
  }
  result.data_alignment = alignment; // Store the determined alignment

  uint64_t current_pos_metadata_stream = metadata_file.tellg();
  Logger::info("[GGUF_LOAD] Current file position (metadata stream) before padding seek: " +
               std::to_string(current_pos_metadata_stream));
  uint64_t padding = (alignment - (current_pos_metadata_stream % alignment)) % alignment;
  Logger::info("[GGUF_LOAD] Calculated padding: " + std::to_string(padding));
  
  uint64_t actual_data_start_offset_in_file = current_pos_metadata_stream + padding;
  Logger::info(
      "[GGUF_LOAD] Calculated actual_data_start_offset_in_file (for mmap): " +
      std::to_string(actual_data_start_offset_in_file));
  
  metadata_file.close();
  Logger::info("[GGUF_LOAD] Metadata ifstream closed.");

  if (!use_mmap) {
    Logger::info("[GGUF_LOAD] mmap is disabled by configuration. Tensor data will not be memory-mapped from GGUF.");
    // GGUFData members (file_descriptor, mapped_tensor_data, etc.) are already defaulted
    // to safe values (-1, nullptr) by its constructor.
    // map_gguf_weights function in model.cpp already checks for mapped_tensor_data == nullptr.
    return result; 
  }

  result.file_descriptor = open(filename.c_str(), O_RDONLY);
  if (result.file_descriptor == -1) {
      throw std::runtime_error("GGUF Error: Failed to open file for mmap: " + filename + " - " + strerror(errno));
  }
  Logger::info("[GGUF_LOAD] File opened for mmap with fd: " + std::to_string(result.file_descriptor));

  struct stat file_stat;
  if (fstat(result.file_descriptor, &file_stat) == -1) {
      close(result.file_descriptor); 
      result.file_descriptor = -1;
      throw std::runtime_error("GGUF Error: Failed to fstat file for mmap: " + filename + " - " + strerror(errno));
  }
  uint64_t file_total_size = static_cast<uint64_t>(file_stat.st_size);

  if (file_total_size < actual_data_start_offset_in_file) {
    close(result.file_descriptor); 
    result.file_descriptor = -1;
    throw std::runtime_error(
        "GGUF Error: File total size (" + std::to_string(file_total_size) + 
        ") is less than calculated actual_data_start_offset_in_file (" + std::to_string(actual_data_start_offset_in_file) + ").");
  }

  uint64_t tensor_data_block_size_on_disk = file_total_size - actual_data_start_offset_in_file;
  Logger::info("[GGUF_LOAD] Calculated tensor_data_block_size_on_disk (for mmap length calculation): " +
               std::to_string(tensor_data_block_size_on_disk) + " bytes.");
  
  long page_size = sysconf(_SC_PAGE_SIZE);
  if (page_size == -1) {
      close(result.file_descriptor);
      result.file_descriptor = -1;
      throw std::runtime_error(std::string("GGUF Error: Failed to get page size using sysconf - ") + strerror(errno));
  }
  Logger::info("[GGUF_LOAD] System page size: " + std::to_string(page_size));

  uint64_t mmap_offset = (actual_data_start_offset_in_file / page_size) * page_size;
  result.offset_diff_for_mmap = static_cast<size_t>(actual_data_start_offset_in_file - mmap_offset);
  size_t mmap_length = static_cast<size_t>(tensor_data_block_size_on_disk + result.offset_diff_for_mmap);

  Logger::info("[GGUF_LOAD] Aligning mmap: actual_data_start_offset_in_file=" + std::to_string(actual_data_start_offset_in_file) +
               ", mmap_offset=" + std::to_string(mmap_offset) +
               ", offset_diff_for_mmap=" + std::to_string(result.offset_diff_for_mmap) +
               ", mmap_length=" + std::to_string(mmap_length));

  if (mmap_length > 0) { // Only mmap if there's something to map
    result.mapped_tensor_data_size = mmap_length; // Store the actual mapped length
    result.mapped_tensor_data = mmap(nullptr, result.mapped_tensor_data_size, 
                                         PROT_READ, MAP_SHARED, 
                                         result.file_descriptor, static_cast<off_t>(mmap_offset));
    
    if (result.mapped_tensor_data == MAP_FAILED) {
        int mmap_errno = errno; 
        close(result.file_descriptor); 
        result.file_descriptor = -1;
        result.mapped_tensor_data = nullptr; 
        result.mapped_tensor_data_size = 0; // Reset on failure
        result.offset_diff_for_mmap = 0;    // Reset on failure
        throw std::runtime_error("GGUF Error: mmap failed for tensor data. Aligned Offset: " + std::to_string(mmap_offset) +
                                 ", Mmap Length: " + std::to_string(mmap_length) + 
                                 " - Error: " + strerror(mmap_errno));
    }
    Logger::info("[GGUF_LOAD] Successfully mmapped tensor data block. Mapped Address: " + 
                 std::to_string(reinterpret_cast<uintptr_t>(result.mapped_tensor_data)) + 
                 ", Mapped Size: " + std::to_string(result.mapped_tensor_data_size) + 
                 " bytes from file offset " + std::to_string(mmap_offset));

    if (result.mapped_tensor_data_size >= (result.offset_diff_for_mmap + 16)) {
      std::stringstream ss_bytes;
      ss_bytes << "[GGUF_LOAD] First 16 bytes of *actual* tensor data (after offset_diff) in mmap: ";
      const uint8_t* actual_data_ptr = static_cast<const uint8_t*>(result.mapped_tensor_data) + result.offset_diff_for_mmap;
      for (int i = 0; i < 16; ++i)
        ss_bytes << "0x" << std::hex << static_cast<int>(actual_data_ptr[i]) << " ";
      Logger::info(ss_bytes.str());
    }

  } else {
    Logger::info("[GGUF_LOAD] Tensor data block size (or mmap_length) is 0. Nothing to mmap.");
    result.mapped_tensor_data = nullptr; // Ensure it's null if not mapped
    result.mapped_tensor_data_size = 0;
    result.offset_diff_for_mmap = 0;
    // fd will be closed by GGUFData destructor if not already closed by an error path.
  }
  
  Logger::info("GGUF metadata loaded and tensor data (if any) mmapped successfully.");
  return result;
}