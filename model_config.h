#ifndef MODEL_CONFIG_H
#define MODEL_CONFIG_H

#include <nlohmann/json.hpp>
#include "model.h"

struct GGUFData;

ModelConfig parse_model_config(const nlohmann::json& json);

ModelConfig parse_model_config_from_gguf(const GGUFData& gguf);

#endif // MODEL_CONFIG_H 