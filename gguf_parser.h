#pragma once

#include <cstddef>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "gguf_structs.h"

extern const uint32_t GGUF_MAGIC;

template <typename T>
void read_raw(std::ifstream& file, T& dest);

std::string read_gguf_string(std::ifstream& file);

GGUFData load_gguf_meta(const std::string& filename);
