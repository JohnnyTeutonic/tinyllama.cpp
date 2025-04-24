#ifndef SAFETENSORS_LOADER_H
#define SAFETENSORS_LOADER_H

#include <string>
#include <vector>
#include <map>
#include <stdexcept>
#include <nlohmann/json.hpp>

class SafeTensorsLoader {
public:
    struct TensorInfo {
        std::string name;
        std::string dtype;
        std::vector<size_t> shape;
        size_t data_offset;
        size_t nbytes;
    };

    explicit SafeTensorsLoader(const std::string& path);
    std::vector<std::string> tensor_names() const;
    std::vector<uint8_t> get_tensor_bytes(const std::string& name) const;
    const TensorInfo& get_tensor_info(const std::string& name) const;

private:
    std::map<std::string, TensorInfo> tensors_;
    std::string file_path_;
    size_t data_start_ = 0;
};

#endif // SAFETENSORS_LOADER_H 