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

    py::class_<ModelConfig>(m, "ModelConfig")
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
        .def("__repr__",
             [](const ModelConfig &cfg) {
                 return "<ModelConfig: vocab_size=" + std::to_string(cfg.vocab_size) +
                        ", hidden_size=" + std::to_string(cfg.hidden_size) +
                        ">";
             }
        );

    py::class_<tinyllama::TinyLlamaSession>(m, "TinyLlamaSession")
        .def(py::init<const std::string&>(), 
             py::arg("model_path"),
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