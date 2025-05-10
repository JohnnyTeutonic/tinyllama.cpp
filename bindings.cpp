#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>

#include "model_constants.h"
#include "model.h"
// #include "api.h"

namespace py = pybind11;

PYBIND11_MODULE(tinyllama_bindings, m) {
    m.doc() = "Python bindings for tinyllama.cpp project";

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
                        // ... add more important fields to the repr ...
                        ">";
             }
        );

    py::class_<TinyLlamaModel>(m, "TinyLlamaModel")
        .def(py::init<const ModelConfig&, const std::string&>(), 
             py::arg("config"), py::arg("weights_path"))
        .def("get_config", &TinyLlamaModel::get_config, py::return_value_policy::reference_internal)
        .def("get_vocab_size", &TinyLlamaModel::get_vocab_size)
        .def("lookup_embedding", &TinyLlamaModel::lookup_embedding, py::arg("token_id"))
        // Exposing the raw forward method. Users will need to manage KVCache and inputs carefully.
        // KVCache and its management are not bound yet.
        // The input x_vec (std::vector<float>&) would also be tricky from Python.
        // We will likely need a higher-level generate() method later.
        .def("forward_cpu", [](TinyLlamaModel &model, std::vector<float>& x_vec, int pos) {
            // Simplified call for now, ignoring KVCache and attention_mask for this basic binding
            // This is NOT a good generation loop, just a way to call one step of forward
            // WARNING: This binding of forward_cpu is very basic and likely not directly usable for generation without more work
            return model.forward(x_vec, pos, nullptr, nullptr);
        }, py::arg("x_vec"), py::arg("pos"), "Performs one forward pass on CPU (low-level)")
        ;

    // Note: KVCache is not bound yet, which is needed for efficient generation with the current forward method.
    // Binding SafeTensorsLoader is also skipped for now.
} 