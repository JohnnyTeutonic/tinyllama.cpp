#pragma once

#include <string>
#include <vector>
#include <map>
#include <fstream>
#include "gguf_structs.h" // Include the GGUF structure definitions

// --- GGUF Constants ---
extern const uint32_t GGUF_MAGIC; // Defined in gguf_parser.cpp

// --- Helper Function Declarations ---

template<typename T>
void read_raw(std::ifstream& file, T& dest);

std::string read_gguf_string(std::ifstream& file);

// --- Main Loading Function Declaration ---

// Structure to hold the loaded GGUF data
struct GGUFData {
    GGUFHeader header;
    std::map<std::string, GGUFMetadataValue> metadata;
    std::vector<GGUFTensorInfo> tensor_infos;
    // TODO: Add field for actual tensor data (e.g., vector<std::byte> or similar)
};

// Loads header, metadata, and tensor info from a GGUF file.
// Tensor data itself is not loaded by this function yet.
GGUFData load_gguf_meta(const std::string& filename);

// TODO: Declare a function to load the actual tensor data later.
// void load_tensor_data(std::ifstream& file, GGUFTensorInfo& info, std::vector<std::byte>& data_buffer); 