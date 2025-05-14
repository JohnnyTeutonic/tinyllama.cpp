#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>

#include "model_constants.h"
#include "model.h"
#include "api.h"

namespace py = pybind11;

PYBIND11_MODULE(tinyllama_bindings, m) {
    m.doc() = "Python bindings for the tinyllama.cpp project, using TinyLlamaSession.";

    py::add_ostream_redirect(m, "ostream_redirect");

    py::class_<ModelConfig> model_config_class(m, "ModelConfig");
    model_config_class
        .def(py::init<>())
        .def_readwrite("hidden_size", &ModelConfig::hidden_size)
        .def_readwrite("intermediate_size", &ModelConfig::intermediate_size)
        .def_readwrite("num_attention_heads", &ModelConfig::num_attention_heads)
        .def_readwrite("num_key_value_heads", &ModelConfig::num_key_value_heads)
        .def_readwrite("num_hidden_layers", &ModelConfig::num_hidden_layers)
        .def_readwrite("vocab_size", &ModelConfig::vocab_size)
        .def_readwrite("max_position_embeddings", &ModelConfig::max_position_embeddings)
        .def_readwrite("rms_norm_eps", &ModelConfig::rms_norm_eps)
        .def_readwrite("rope_theta", &ModelConfig::rope_theta)
        .def_readwrite("hidden_act", &ModelConfig::hidden_act)
        .def_readwrite("torch_dtype", &ModelConfig::torch_dtype)
        .def_readwrite("bos_token_id", &ModelConfig::bos_token_id)
        .def_readwrite("eos_token_id", &ModelConfig::eos_token_id)
        .def_readwrite("architecture", &ModelConfig::architecture)
        .def_readwrite("model_name", &ModelConfig::model_name)
        .def_readwrite("chat_template_type", &ModelConfig::chat_template_type)
        .def_readwrite("pre_tokenizer_type", &ModelConfig::pre_tokenizer_type)
        .def_readwrite("chat_template_string", &ModelConfig::chat_template_string)
        .def_readwrite("is_gguf_file_loaded", &ModelConfig::is_gguf_file_loaded)
        .def_readonly("tokenizer_family", &ModelConfig::tokenizer_family)
        .def("__repr__",
             [](const ModelConfig &cfg) {
                 std::string tf_str = "UNKNOWN";
                 if (cfg.tokenizer_family == ModelConfig::TokenizerFamily::LLAMA_SENTENCEPIECE) tf_str = "LLAMA_SENTENCEPIECE";
                 else if (cfg.tokenizer_family == ModelConfig::TokenizerFamily::LLAMA3_TIKTOKEN) tf_str = "LLAMA3_TIKTOKEN";
                 return "<ModelConfig: vocab_size=" + std::to_string(cfg.vocab_size) +
                        ", hidden_size=" + std::to_string(cfg.hidden_size) +
                        ", tokenizer_family=" + tf_str +
                        ">";
             }
        );
    
    py::enum_<ModelConfig::TokenizerFamily>(model_config_class, "TokenizerFamily")
        .value("UNKNOWN", ModelConfig::TokenizerFamily::UNKNOWN)
        .value("LLAMA_SENTENCEPIECE", ModelConfig::TokenizerFamily::LLAMA_SENTENCEPIECE)
        .value("LLAMA3_TIKTOKEN", ModelConfig::TokenizerFamily::LLAMA3_TIKTOKEN)
        .export_values();

    py::class_<tinyllama::TinyLlamaSession>(m, "TinyLlamaSession")
        .def(py::init<const std::string &, const std::string &, int, int, bool>(), 
             py::arg("model_path"),
             py::arg("tokenizer_path"),
             py::arg("threads") = 1,
             py::arg("n_gpu_layers") = 0,
             py::arg("use_mmap") = false,
             "Loads the model, config, and tokenizer from the specified model_path (directory or .gguf file)." 
        )
        .def("generate", &tinyllama::TinyLlamaSession::generate,
             py::arg("prompt"),
             py::arg("steps") = 128,
             py::arg("temperature") = 0.1f,
             py::arg("top_k") = 40,
             py::arg("top_p") = 0.9f,
             py::arg("system_prompt") = "",
             py::arg("apply_q_a_format") = false,
             "Generates text based on a prompt with various sampling parameters."
        )
        .def("get_config", &tinyllama::TinyLlamaSession::get_config, 
             py::return_value_policy::reference_internal,
             "Returns a const reference to the session's ModelConfig."
        );
} 