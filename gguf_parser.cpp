#include "gguf_parser.h"

#include <iomanip>  
#include <iostream>  
#include <numeric>    
#include <sstream>    
#include <stdexcept>  
#include <vector>     

#include "logger.h"
#include "quantization.h"  


const uint32_t GGUF_MAGIC = 0x46554747;  





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
    
    if (len > (1ull << 30)) {  
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



GGUFData load_gguf_meta(const std::string& filename) {
  Logger::info("Attempting to load GGUF file: " + filename);
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file: " + filename);
  }

  GGUFData result;

  
  read_raw(file, result.header.magic);
  read_raw(file, result.header.version);
  read_raw(file, result.header.tensor_count);
  read_raw(file, result.header.metadata_kv_count);

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
      key = read_gguf_string(file);
      read_raw(file, value_type_enum);

      switch (value_type_enum) {
        case GGUFValueType::UINT8: {
          uint8_t val;
          read_raw(file, val);
          result.metadata[key] = val;
          break;
        }
        case GGUFValueType::INT8: {
          int8_t val;
          read_raw(file, val);
          result.metadata[key] = val;
          break;
        }
        case GGUFValueType::UINT16: {
          uint16_t val;
          read_raw(file, val);
          result.metadata[key] = val;
          break;
        }
        case GGUFValueType::INT16: {
          int16_t val;
          read_raw(file, val);
          result.metadata[key] = val;
          break;
        }
        case GGUFValueType::UINT32: {
          uint32_t val;
          read_raw(file, val);
          result.metadata[key] = val;
          break;
        }
        case GGUFValueType::INT32: {
          int32_t val;
          read_raw(file, val);
          result.metadata[key] = val;
          break;
        }
        case GGUFValueType::FLOAT32: {
          float val;
          read_raw(file, val);
          result.metadata[key] = val;
          break;
        }
        case GGUFValueType::BOOL: {
          uint8_t byte;
          read_raw(file, byte);
          result.metadata[key] = (byte != 0);
          break;
        }
        case GGUFValueType::STRING: {
          std::string val = read_gguf_string(file);
          result.metadata[key] = val;
          break;
        }
        case GGUFValueType::UINT64: {
          uint64_t val;
          read_raw(file, val);
          result.metadata[key] = val;
          break;
        }
        case GGUFValueType::INT64: {
          int64_t val;
          read_raw(file, val);
          result.metadata[key] = val;
          break;
        }
        case GGUFValueType::FLOAT64: {
          double val;
          read_raw(file, val);
          result.metadata[key] = val;
          break;
        }
        case GGUFValueType::ARRAY: {
          GGUFValueType array_type_enum;
          uint64_t count;
          read_raw(file, array_type_enum);
          read_raw(file, count);

          
          
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
              result.tokenizer_tokens.push_back(read_gguf_string(file));
            }
            Logger::info("Loaded tokenizer_tokens. Size: " +
                         std::to_string(result.tokenizer_tokens.size()));
          } else if (key == "tokenizer.ggml.scores" &&
                     array_type_enum == GGUFValueType::FLOAT32) {
            Logger::info("Loading FLOAT32 array data ('" + key + "') with " +
                         std::to_string(count) + " elements...");
            result.tokenizer_scores.resize(static_cast<size_t>(count));
            file.read(reinterpret_cast<char*>(result.tokenizer_scores.data()),
                      static_cast<std::streamsize>(count * sizeof(float)));
            if (!file) {
              throw std::runtime_error(
                  "GGUF Error: Failed to read scores array data.");
            }
            Logger::info("Loaded tokenizer_scores. Size: " +
                         std::to_string(result.tokenizer_scores.size()));
          } else if (key == "tokenizer.ggml.token_type" &&
                     array_type_enum == GGUFValueType::UINT32) {
            Logger::info("Loading UINT32 array data ('" + key + "') with " +
                         std::to_string(count) + " elements...");
            result.tokenizer_token_types.resize(static_cast<size_t>(count));
            file.read(
                reinterpret_cast<char*>(result.tokenizer_token_types.data()),
                static_cast<std::streamsize>(count * sizeof(uint32_t)));
            if (!file) {
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
              result.tokenizer_merges.push_back(read_gguf_string(file));
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
                  std::string discarded_str =
                      read_gguf_string(file);  
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
                file.seekg(static_cast<std::streamoff>(total_size_to_skip),
                           std::ios::cur);
                if (!file) {
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
      info.name = read_gguf_string(file);

      uint32_t n_dims;
      read_raw(file, n_dims);
      if (n_dims > 4) {  
        throw std::runtime_error("Tensor '" + info.name +
                                 "' has unsupported number of dimensions: " +
                                 std::to_string(n_dims));
      }
      info.shape.resize(n_dims);
      for (uint32_t d = 0; d < n_dims; ++d) {
        read_raw(file, info.shape[d]);
      }

      uint32_t ggml_type_u32;
      read_raw(file, ggml_type_u32);
      info.type = static_cast<GGMLType>(ggml_type_u32);

      uint64_t pos_before_offset_read = file.tellg();

      read_raw(file, info.offset);

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

  
  
  uint64_t alignment = 32;  
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

  uint64_t current_pos = file.tellg();
  Logger::info("[GGUF_LOAD] Current file position before padding seek: " +
               std::to_string(current_pos));
  uint64_t padding = (alignment - (current_pos % alignment)) % alignment;
  Logger::info("[GGUF_LOAD] Calculated padding: " + std::to_string(padding));
  if (padding > 0) {
    Logger::info("[GGUF_LOAD] Seeking past " + std::to_string(padding) +
                 " padding bytes to reach alignment " +
                 std::to_string(alignment) + ".");
    file.seekg(padding, std::ios::cur);
    if (!file) {
      throw std::runtime_error(
          "GGUF Error: Failed to seek past padding before tensor data.");
    }
  }
  uint64_t data_start_pos = file.tellg();  
                                           
  Logger::info(
      "[GGUF_LOAD] File position after padding seek (data_start_pos): " +
      std::to_string(data_start_pos));
  std::string target_tensor_name_debug = "blk.0.ffn_down.weight";
  uint64_t offset_of_d_in_block_q6k = 208;
  if (result.tensor_infos_map.count(target_tensor_name_debug)) {
    const GGUFTensorInfo& t_info_debug =
        result.tensor_infos_map.at(target_tensor_name_debug);
    if (t_info_debug.type == GGML_TYPE_Q6_K) {  
      uint64_t tensor_relative_offset =
          t_info_debug
              .offset;  
      uint64_t d_field_abs_file_offset =
          data_start_pos + tensor_relative_offset + offset_of_d_in_block_q6k;

      Logger::info("[GGUF_DEBUG_READ] Targeting tensor '" +
                   target_tensor_name_debug + "' (type Q6_K)");
      Logger::info("[GGUF_DEBUG_READ]   data_start_pos (file): " +
                   std::to_string(data_start_pos));
      Logger::info("[GGUF_DEBUG_READ]   tensor_relative_offset (metadata): " +
                   std::to_string(tensor_relative_offset));
      Logger::info("[GGUF_DEBUG_READ]   d_field_offset_in_block: " +
                   std::to_string(offset_of_d_in_block_q6k));
      Logger::info("[GGUF_DEBUG_READ]   Calculated d_field_abs_file_offset: " +
                   std::to_string(d_field_abs_file_offset));

      uint64_t seek_pos_debug =
          d_field_abs_file_offset - 4;  
      int bytes_to_read_debug = 8;      
      std::vector<uint8_t> debug_buffer(bytes_to_read_debug);

      uint64_t original_file_pos_before_debug_read =
          file.tellg();  
      file.seekg(seek_pos_debug, std::ios::beg);
      if (!file) {
        Logger::error("[GGUF_DEBUG_READ] Failed to seek to debug position: " +
                      std::to_string(seek_pos_debug));
      } else {
        Logger::info("[GGUF_DEBUG_READ] At debug seek position: " +
                     std::to_string(file.tellg()));
        file.read(reinterpret_cast<char*>(debug_buffer.data()),
                  bytes_to_read_debug);
        if (!file) {
          Logger::error("[GGUF_DEBUG_READ] Failed to read debug bytes. Read " +
                        std::to_string(file.gcount()) + " bytes.");
        } else {
          std::stringstream ss_debug_bytes;
          ss_debug_bytes << "[GGUF_DEBUG_READ]   Read " << file.gcount()
                         << " bytes from file @ " << seek_pos_debug << ": ";
          for (int k = 0; k < file.gcount(); ++k)
            ss_debug_bytes << "0x" << std::hex << (int)debug_buffer[k] << " ";
          Logger::info(ss_debug_bytes.str());
        }
        file.clear();  
        file.seekg(original_file_pos_before_debug_read, std::ios::beg);
        Logger::info(
            "[GGUF_DEBUG_READ] Seeked back to "
            "original_file_pos_before_debug_read: " +
            std::to_string(file.tellg()));
      }
    } else {
      Logger::warning("[GGUF_DEBUG_READ] Target tensor '" +
                      target_tensor_name_debug +
                      "' is not Q6_K. Skipping debug read.");
    }
  } else {
    Logger::warning("[GGUF_DEBUG_READ] Target tensor '" +
                    target_tensor_name_debug + "' not found for debug read.");
  }

  
  file.seekg(0, std::ios::end);  
  uint64_t file_end_pos = file.tellg();
  file.seekg(data_start_pos, std::ios::beg);  

  if (file_end_pos < data_start_pos) {
    throw std::runtime_error(
        "GGUF Error: File end position is before calculated data start "
        "position.");
  }

  uint64_t data_size = file_end_pos - data_start_pos;
  Logger::info("[GGUF_LOAD] Calculated tensor data block size: " +
               std::to_string(data_size) + " bytes.");

  if (data_size > 0) {
    
    try {
      result.tensor_data.resize(static_cast<size_t>(data_size));
    } catch (const std::bad_alloc& e) {
      throw std::runtime_error("Failed to allocate memory for tensor data: " +
                               std::to_string(data_size) + " bytes. " +
                               e.what());
    }

    
    Logger::info("[GGUF_LOAD] Reading " + std::to_string(data_size) +
                 " bytes into tensor_data vector...");
    file.read(reinterpret_cast<char*>(result.tensor_data.data()),
              static_cast<std::streamsize>(data_size));
    uint64_t pos_after_read = file.tellg();      
    std::streamsize bytes_read = file.gcount();  
    Logger::info("[GGUF_LOAD] File position after read: " +
                 std::to_string(pos_after_read));  
    Logger::info("[GGUF_LOAD] Bytes reported read by gcount(): " +
                 std::to_string(bytes_read));  
    Logger::info("[GGUF_LOAD] Stream state after read: good=" +
                 std::to_string(file.good()) +
                 ", eof=" + std::to_string(file.eof()) +
                 ", fail=" + std::to_string(file.fail()));  

    if (!file ||
        bytes_read != static_cast<std::streamsize>(
                          data_size)) {  
      
      if (file.eof()) {
        throw std::runtime_error(
            "GGUF Error: Reached EOF prematurely while reading tensor data. "
            "Expected " +
            std::to_string(data_size) + " bytes, read " +
            std::to_string(bytes_read) + ".");
      } else {
        throw std::runtime_error(
            "GGUF Error: Failed to read tensor data block from file. Stream "
            "state indicates error.");
      }
    }
    Logger::info("[GGUF_LOAD] Successfully read tensor data block.");
    
    if (data_size >= 16) {
      std::stringstream ss_bytes;
      ss_bytes << "[GGUF_LOAD] First 16 bytes of tensor_data buffer: ";
      for (int i = 0; i < 16; ++i)
        ss_bytes << "0x" << std::hex << (int)result.tensor_data[i] << " ";
      Logger::info(ss_bytes.str());
    }
  } else {
    Logger::info("[GGUF_LOAD] Tensor data block size is 0. Nothing to read.");
  }

  file.close();
  Logger::info("GGUF metadata and data loaded successfully.");
  return result;
}