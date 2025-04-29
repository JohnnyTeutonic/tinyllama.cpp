#include "gguf_parser.h"
#include "quantization.h" // For ggml_type_size/block_size/name
#include <iostream>       // For std::cout, std::cerr
#include <stdexcept>      // For std::runtime_error
#include <vector>         // For intermediate buffers
#include <iomanip>        // For std::hex
#include <numeric>        // For std::accumulate in tensor size calculation

// Define the magic constant
const uint32_t GGUF_MAGIC = 0x46554747; // "GGUF" little-endian

// --- GGUF File Reading Helpers Implementation ---

template<typename T>
void read_raw(std::ifstream& file, T& dest) {
    file.read(reinterpret_cast<char*>(&dest), sizeof(T));
    if (!file) {
        throw std::runtime_error("GGUF Error: Failed to read data from file stream.");
    }
}
// Explicit template instantiation for types used (optional but good practice)
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
        // Limit string length to avoid excessive memory allocation
        if (len > (1ull << 30)) { // Max 1GB string, adjust as needed
             throw std::runtime_error("GGUF Error: String length exceeds sanity limit: " + std::to_string(len));
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

// --- GGUF Loading Logic Implementation ---

GGUFData load_gguf_meta(const std::string& filename) {
    std::cout << "Attempting to load GGUF file: " << filename << "\n";
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    GGUFData result;

    // 1. Read Header
    read_raw(file, result.header.magic);
    read_raw(file, result.header.version);
    read_raw(file, result.header.tensor_count);
    read_raw(file, result.header.metadata_kv_count);

    std::cout << "  Read Header:\n";
    std::cout << "    Magic: 0x" << std::hex << result.header.magic << std::dec << "\n";
    std::cout << "    Version: " << result.header.version << "\n";
    std::cout << "    Tensor Count: " << result.header.tensor_count << "\n";
    std::cout << "    Metadata KV Count: " << result.header.metadata_kv_count << "\n";

    // 2. Validate Magic Number
    if (result.header.magic != GGUF_MAGIC) {
        throw std::runtime_error("Not a valid GGUF file (magic number mismatch).");
    }

    // TODO: Validate Version (Support specific versions if needed)

    // 3. Read Metadata Key-Value Pairs
    std::cout << "  Reading Metadata (" << result.header.metadata_kv_count << " pairs)...\n";
    for (uint64_t i = 0; i < result.header.metadata_kv_count; ++i) {
        std::string key = read_gguf_string(file);
        GGUFValueType value_type_enum;
        read_raw(file, value_type_enum);

        try {
            switch (value_type_enum) {
                case GGUFValueType::UINT8:   { uint8_t  val; read_raw(file, val); result.metadata[key] = val; break; }
                case GGUFValueType::INT8:    { int8_t   val; read_raw(file, val); result.metadata[key] = val; break; }
                case GGUFValueType::UINT16:  { uint16_t val; read_raw(file, val); result.metadata[key] = val; break; }
                case GGUFValueType::INT16:   { int16_t  val; read_raw(file, val); result.metadata[key] = val; break; }
                case GGUFValueType::UINT32:  { uint32_t val; read_raw(file, val); result.metadata[key] = val; break; }
                case GGUFValueType::INT32:   { int32_t  val; read_raw(file, val); result.metadata[key] = val; break; }
                case GGUFValueType::FLOAT32: { float    val; read_raw(file, val); result.metadata[key] = val; break; }
                case GGUFValueType::BOOL:    { uint8_t byte; read_raw(file, byte); result.metadata[key] = (byte != 0); break; }
                case GGUFValueType::STRING:  { std::string val = read_gguf_string(file); result.metadata[key] = val; break; }
                case GGUFValueType::UINT64:  { uint64_t val; read_raw(file, val); result.metadata[key] = val; break; }
                case GGUFValueType::INT64:   { int64_t  val; read_raw(file, val); result.metadata[key] = val; break; }
                case GGUFValueType::FLOAT64: { double   val; read_raw(file, val); result.metadata[key] = val; break; }
                case GGUFValueType::ARRAY: {
                     std::cerr << "    Skipping ARRAY metadata type for key: " << key << " (Not Implemented Yet)\n";
                     GGUFValueType array_type_enum; uint64_t count;
                     read_raw(file, array_type_enum); read_raw(file, count);
                     std::cerr << "      -> Array type: " << static_cast<uint32_t>(array_type_enum) << ", Count: " << count << "\n";

                     // TODO: Properly implement array reading and skipping
                     // For now, attempt to skip known problematic large arrays like tokenizer vocab/merges
                     // This is still fragile!
                     if (array_type_enum == GGUFValueType::STRING) {
                         std::cout << "      -> Attempting to skip string array...\n";
                         for(uint64_t arr_i = 0; arr_i < count; ++arr_i) {
                             try {
                                read_gguf_string(file); // Read and discard
                             } catch (const std::exception& e) {
                                std::cerr << "      -> Error skipping string element " << arr_i << ": " << e.what() << "\n";
                                throw; // Re-throw after logging inner error
                             }
                         }
                         std::cout << "      -> Finished skipping string array.\n";
                     } else {
                        // Need a reliable way to get size for other types
                        std::cerr << "      -> Cannot reliably skip non-string array type yet.\n";
                        // We could try seeking, but need the size of array_type_enum elements.
                        // For now, we might be stuck if we encounter other large arrays.
                        // throw std::runtime_error("Skipping non-string arrays not implemented yet.");
                     }
                    break;
                }
                default: {
                    throw std::runtime_error("Unknown metadata type encountered: " + std::to_string(static_cast<uint32_t>(value_type_enum)));
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Error reading metadata for key: '" << key << "' (type: " << static_cast<uint32_t>(value_type_enum) << ") - " << e.what() << "\n";
            throw; // Re-throw for now
        }
    }
    std::cout << "  Finished reading metadata.\n";

    // 4. Read Tensor Info
    result.tensor_infos.reserve(static_cast<size_t>(result.header.tensor_count));
    std::cout << "  Reading Tensor Info (" << result.header.tensor_count << " tensors)...\n";
    for (uint64_t i = 0; i < result.header.tensor_count; ++i) {
        GGUFTensorInfo info;
        info.name = read_gguf_string(file);

        uint32_t n_dims;
        read_raw(file, n_dims);
        if (n_dims > 4) { // Limit dimensions for sanity
            throw std::runtime_error("Tensor '" + info.name + "' has unsupported number of dimensions: " + std::to_string(n_dims));
        }
        info.shape.resize(n_dims);
        // Read shape elements individually (GGUF spec: n_dims * uint64)
        for(uint32_t d=0; d<n_dims; ++d) {
            read_raw(file, info.shape[d]);
        }

        uint32_t ggml_type_u32;
        read_raw(file, ggml_type_u32);
        info.type = static_cast<GGMLType>(ggml_type_u32);

        read_raw(file, info.offset);

        // Calculate num_elements and size_in_bytes
        info.num_elements = 1;
        // Use checked multiplication to prevent overflow on large tensors
        for(uint64_t dim : info.shape) {
            // Basic overflow check
            if (dim > 0 && info.num_elements > std::numeric_limits<uint64_t>::max() / dim) {
                throw std::overflow_error("Tensor dimension overflow calculating num_elements for tensor '" + info.name + "'");
            }
            info.num_elements *= dim;
        }

        size_t type_size = ggml_type_size(info.type); // From quantization.cpp
        size_t block_size = ggml_type_block_size(info.type); // From quantization.cpp

        if (block_size == 0 && info.num_elements > 0) { // Check for unknown type
             throw std::runtime_error("Tensor '" + info.name + "' has unknown or unsupported type: " + std::to_string(info.type));
        }

        if (block_size > 1) { // Quantized type
            if (info.num_elements % block_size != 0) {
                throw std::runtime_error("Tensor '" + info.name + "' num_elements (" + std::to_string(info.num_elements)
                                       + ") not divisible by block_size (" + std::to_string(block_size) + ") for type " + ggml_type_name(info.type));
            }
            // Check for overflow when calculating size
            uint64_t num_blocks = info.num_elements / block_size;
             if (type_size > 0 && num_blocks > std::numeric_limits<uint64_t>::max() / type_size) {
                 throw std::overflow_error("Tensor size overflow calculating size_in_bytes for tensor '" + info.name + "'");
             }
            info.size_in_bytes = static_cast<size_t>(num_blocks * type_size);
        } else { // Non-quantized type
             if (type_size > 0 && info.num_elements > std::numeric_limits<uint64_t>::max() / type_size) {
                 throw std::overflow_error("Tensor size overflow calculating size_in_bytes for tensor '" + info.name + "'");
             }
             info.size_in_bytes = static_cast<size_t>(info.num_elements * type_size);
        }

        result.tensor_infos.push_back(info);
        std::cout << "    Tensor " << i << ": Name='" << info.name << "', Type=" << ggml_type_name(info.type)
                  << ", Shape=[ ";
        for(size_t d=0; d<info.shape.size(); ++d) std::cout << info.shape[d] << (d==info.shape.size()-1 ? "" : ", ");
        std::cout << " ], Offset=" << info.offset << ", Size=" << info.size_in_bytes << " bytes\n";
    }
    std::cout << "  Finished reading tensor info.\n";

    // 5. TODO: Read Tensor Data (based on offsets and sizes)

    // 6. TODO: Align data section (padding)
    // uint64_t current_pos = file.tellg();
    // uint64_t alignment = 32; // Default GGUF alignment
    // // Get alignment from metadata if present
    // if (result.metadata.count("general.alignment")) {
    //     // Extract uint32 alignment value from variant
    // }
    // uint64_t padding = (alignment - (current_pos % alignment)) % alignment;
    // if (padding > 0) {
    //     std::cout << "  Seeking past " << padding << " padding bytes.\n";
    //     file.seekg(padding, std::ios::cur);
    // }

    file.close();
    std::cout << "GGUF metadata loaded successfully.\n";
    return result;
} 