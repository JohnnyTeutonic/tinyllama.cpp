#include "safetensors_loader.h"

#include <fstream>

SafeTensorsLoader::SafeTensorsLoader(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file)
    throw std::runtime_error("Failed to open safetensors file: " + path);

  
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
  
  file_path_ = path;
  data_start_ = 8 + header_len;
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
  const TensorInfo& info = it->second;
  std::ifstream file(file_path_, std::ios::binary);
  if (!file)
    throw std::runtime_error("Failed to open safetensors file: " + file_path_);
  file.seekg(data_start_ + info.data_offset, std::ios::beg);
  std::vector<uint8_t> data(info.nbytes);
  file.read(reinterpret_cast<char*>(data.data()), info.nbytes);
  if (file.gcount() != static_cast<std::streamsize>(info.nbytes))
    throw std::runtime_error("Failed to read tensor data: " + name);
  return data;
}

const SafeTensorsLoader::TensorInfo& SafeTensorsLoader::get_tensor_info(
    const std::string& name) const {
  auto it = tensors_.find(name);
  if (it == tensors_.end())
    throw std::runtime_error("Tensor not found: " + name);
  return it->second;
}