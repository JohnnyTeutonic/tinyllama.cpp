#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>

#include "model_constants.h"
#include "model.h"
#include "api.h"

namespace py = pybind11;

PYBIND11_MODULE(tinyllama_bindings, m) {
    m.doc() = R"pbdoc(
        TinyLlama.cpp Python Bindings
        =============================
        
        Python bindings for the tinyllama.cpp inference engine supporting both GGUF and SafeTensors formats.
        
        Key Classes:
        - TinyLlamaSession: Main interface for model loading and text generation
        - ModelConfig: Configuration object containing model parameters and metadata
        
        Example Usage:
            import tinyllama_cpp
            
            # Basic CPU usage
            session = tinyllama_cpp.TinyLlamaSession(
                model_path="path/to/model.gguf",
                tokenizer_path="path/to/tokenizer.json",
                threads=4
            )
            
            # GPU usage with quantized KV cache
            session = tinyllama_cpp.TinyLlamaSession(
                model_path="path/to/model.gguf", 
                tokenizer_path="path/to/tokenizer.json",
                threads=4,
                n_gpu_layers=-1,
                use_kv_quant=True
            )
            
            # Generate text
            response = session.generate("What is AI?", steps=64, temperature=0.7)
            print(response)
            
            # Batch generation
            prompts = ["What is AI?", "Explain quantum computing", "Tell me a joke"]
            responses = session.generate_batch(prompts, steps=64)
            for prompt, response in zip(prompts, responses):
                print(f"Q: {prompt}\nA: {response}\n")
    )pbdoc";

    py::add_ostream_redirect(m, "ostream_redirect");

    py::class_<ModelConfig> model_config_class(m, "ModelConfig", R"pbdoc(
        Model configuration containing architecture parameters and metadata.
        
        This class holds all the configuration parameters for a loaded model,
        including architecture details, tokenizer information, and model metadata.
        Most fields are automatically populated when loading a model.
    )pbdoc");
    
    model_config_class
        .def(py::init<>(), "Create an empty ModelConfig object")
        .def_readwrite("hidden_size", &ModelConfig::hidden_size, "Hidden dimension size of the model")
        .def_readwrite("intermediate_size", &ModelConfig::intermediate_size, "Intermediate size in feed-forward layers")
        .def_readwrite("num_attention_heads", &ModelConfig::num_attention_heads, "Number of attention heads")
        .def_readwrite("num_key_value_heads", &ModelConfig::num_key_value_heads, "Number of key-value heads (for GQA)")
        .def_readwrite("num_hidden_layers", &ModelConfig::num_hidden_layers, "Number of transformer layers")
        .def_readwrite("vocab_size", &ModelConfig::vocab_size, "Vocabulary size")
        .def_readwrite("max_position_embeddings", &ModelConfig::max_position_embeddings, "Maximum sequence length")
        .def_readwrite("rms_norm_eps", &ModelConfig::rms_norm_eps, "RMS normalization epsilon")
        .def_readwrite("rope_theta", &ModelConfig::rope_theta, "RoPE theta parameter")
        .def_readwrite("hidden_act", &ModelConfig::hidden_act, "Activation function name")
        .def_readwrite("torch_dtype", &ModelConfig::torch_dtype, "Original PyTorch data type")
        .def_readwrite("bos_token_id", &ModelConfig::bos_token_id, "Beginning-of-sequence token ID")
        .def_readwrite("eos_token_id", &ModelConfig::eos_token_id, "End-of-sequence token ID")
        .def_readwrite("architecture", &ModelConfig::architecture, "Model architecture name")
        .def_readwrite("model_name", &ModelConfig::model_name, "Model name")
        .def_readwrite("chat_template_type", &ModelConfig::chat_template_type, "Chat template type")
        .def_readwrite("pre_tokenizer_type", &ModelConfig::pre_tokenizer_type, "Pre-tokenizer type")
        .def_readwrite("chat_template_string", &ModelConfig::chat_template_string, "Chat template string")
        .def_readwrite("is_gguf_file_loaded", &ModelConfig::is_gguf_file_loaded, "Whether model was loaded from GGUF format")
        .def_readonly("tokenizer_family", &ModelConfig::tokenizer_family, "Tokenizer family (LLAMA_SENTENCEPIECE or LLAMA3_TIKTOKEN)")
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
    
    py::enum_<ModelConfig::TokenizerFamily>(model_config_class, "TokenizerFamily", "Enumeration of supported tokenizer families")
        .value("UNKNOWN", ModelConfig::TokenizerFamily::UNKNOWN, "Unknown tokenizer family")
        .value("LLAMA_SENTENCEPIECE", ModelConfig::TokenizerFamily::LLAMA_SENTENCEPIECE, "Llama/Llama2 SentencePiece tokenizer")
        .value("LLAMA3_TIKTOKEN", ModelConfig::TokenizerFamily::LLAMA3_TIKTOKEN, "Llama3 TikToken-style tokenizer")
        .export_values();

    py::class_<tinyllama::TinyLlamaSession>(m, "TinyLlamaSession", R"pbdoc(
        Main interface for TinyLlama model inference.
        
        This class provides a high-level interface for loading models and generating text.
        It supports both GGUF and SafeTensors formats, CPU and GPU inference, and various
        sampling strategies for text generation.
        
        The session manages the model, tokenizer, and KV cache, providing both single
        prompt generation and efficient batch processing capabilities.
    )pbdoc")
        .def(py::init([](const py::object& model_path, const py::object& tokenizer_path, int threads, int n_gpu_layers, bool use_mmap, bool use_kv_quant, bool use_batch_generation, int max_batch_size) {
            // Convert path-like objects to strings
            std::string model_path_str = py::str(model_path);
            std::string tokenizer_path_str = py::str(tokenizer_path);
            return new tinyllama::TinyLlamaSession(model_path_str, tokenizer_path_str, threads, n_gpu_layers, use_mmap, use_kv_quant, use_batch_generation, max_batch_size);
        }), 
             py::arg("model_path"),
             py::arg("tokenizer_path"),
             py::arg("threads") = 1,
             py::arg("n_gpu_layers") = 0,
             py::arg("use_mmap") = true,
             py::arg("use_kv_quant") = false,
             py::arg("use_batch_generation") = false,
             py::arg("max_batch_size") = 1,
             R"pbdoc(
                Initialize a TinyLlama inference session.
                
                Args:
                    model_path (str or Path): Path to model directory (SafeTensors) or .gguf file
                    tokenizer_path (str or Path): Path to tokenizer.json file. For GGUF models with 
                                                embedded tokenizer, this can be the same as model_path
                    threads (int, optional): Number of CPU threads for inference. Defaults to 1.
                    n_gpu_layers (int, optional): Number of layers to offload to GPU. 
                                                 -1 = all layers, 0 = CPU only. Defaults to 0.
                    use_mmap (bool, optional): Use memory mapping for model loading. Defaults to True.
                    use_kv_quant (bool, optional): Use INT8 quantization for KV cache on GPU. 
                                                  Reduces VRAM usage. Defaults to False.
                    use_batch_generation (bool, optional): Enable optimized batch generation mode. 
                                                         Defaults to False.
                    max_batch_size (int, optional): Maximum number of sequences for batch processing. 
                                                   Defaults to 1.
                
                Raises:
                    RuntimeError: If model loading fails due to invalid paths, unsupported format,
                                or insufficient resources.
                
                Example:
                    # Basic CPU usage
                    session = TinyLlamaSession("model.gguf", "tokenizer.json", threads=4)
                    
                    # Using pathlib.Path objects
                    from pathlib import Path
                    session = TinyLlamaSession(
                        model_path=Path("model.gguf"),
                        tokenizer_path=Path("tokenizer.json"), 
                        threads=4,
                        n_gpu_layers=-1,
                        use_kv_quant=True
                    )
             )pbdoc"
        )
        .def("generate", &tinyllama::TinyLlamaSession::generate,
             py::arg("prompt"),
             py::arg("steps") = 128,
             py::arg("temperature") = 0.1f,
             py::arg("top_k") = 40,
             py::arg("top_p") = 0.9f,
             py::arg("system_prompt") = "",
             py::arg("apply_q_a_format") = true,
             R"pbdoc(
                Generate text based on a prompt using various sampling strategies.
                
                Args:
                    prompt (str): Input text prompt to generate from
                    steps (int, optional): Number of tokens to generate. Defaults to 128.
                    temperature (float, optional): Sampling temperature. Lower values (0.1) 
                                                  produce more focused/deterministic output, 
                                                  higher values (1.0+) more creative/random. 
                                                  Defaults to 0.1.
                    top_k (int, optional): Top-K sampling - limit to K most likely tokens. 
                                         Set to 0 to disable. Defaults to 40.
                    top_p (float, optional): Nucleus sampling - limit to tokens comprising 
                                           top P probability mass (0.0-1.0). Defaults to 0.9.
                    system_prompt (str, optional): System prompt to guide generation behavior. 
                                                  Defaults to empty string.
                    apply_q_a_format (bool, optional): Apply Q:A formatting to prompt. 
                                                     Recommended for most models. Defaults to True.
                
                Returns:
                    str: Generated text (excluding the original prompt)
                
                Raises:
                    RuntimeError: If generation fails due to tokenization errors or model issues.
                
                Example:
                    # Basic generation
                    response = session.generate("What is artificial intelligence?")
                    
                    # Creative generation with higher temperature
                    story = session.generate(
                        "Once upon a time", 
                        steps=200, 
                        temperature=0.8, 
                        top_k=50
                    )
                    
                    # Focused generation with system prompt
                    answer = session.generate(
                        "Explain quantum computing",
                        steps=100,
                        temperature=0.1,
                        system_prompt="You are a helpful physics teacher."
                    )
             )pbdoc"
        )
        .def("generate_batch", &tinyllama::TinyLlamaSession::generate_batch,
             py::arg("prompts"),
             py::arg("steps") = 128,
             py::arg("temperature") = 0.1f,
             py::arg("top_k") = 40,
             py::arg("top_p") = 0.9f,
             py::arg("system_prompt") = "",
             py::arg("apply_q_a_format") = true,
             R"pbdoc(
                Generate text for multiple prompts in parallel (batch processing).
                
                This method processes multiple independent prompts simultaneously, providing
                significant efficiency gains over sequential generate() calls. Each prompt
                maintains its own KV cache state and is processed independently.
                
                Args:
                    prompts (List[str]): List of input prompts to process in batch
                    steps (int, optional): Number of tokens to generate per prompt. Defaults to 128.
                    temperature (float, optional): Sampling temperature applied to all prompts. 
                                                  Defaults to 0.1.
                    top_k (int, optional): Top-K sampling parameter for all prompts. Defaults to 40.
                    top_p (float, optional): Nucleus sampling parameter for all prompts. Defaults to 0.9.
                    system_prompt (str, optional): System prompt applied to all prompts. 
                                                  Defaults to empty string.
                    apply_q_a_format (bool, optional): Apply Q:A formatting to all prompts. 
                                                     Defaults to True.
                
                Returns:
                    List[str]: List of generated text strings, one for each input prompt
                
                Raises:
                    RuntimeError: If batch generation fails or prompts list is empty.
                
                Example:
                    prompts = [
                        "What is machine learning?",
                        "Explain neural networks", 
                        "How does backpropagation work?"
                    ]
                    
                    responses = session.generate_batch(
                        prompts, 
                        steps=100, 
                        temperature=0.2
                    )
                    
                    for prompt, response in zip(prompts, responses):
                        print(f"Q: {prompt}")
                        print(f"A: {response}\n")
             )pbdoc"
        )
        .def("get_config", &tinyllama::TinyLlamaSession::get_config, 
             py::return_value_policy::reference_internal,
             R"pbdoc(
                Get the model configuration.
                
                Returns:
                    ModelConfig: Reference to the session's model configuration containing
                               architecture parameters, tokenizer info, and metadata.
                
                Example:
                    config = session.get_config()
                    print(f"Model has {config.num_hidden_layers} layers")
                    print(f"Vocabulary size: {config.vocab_size}")
                    print(f"Hidden size: {config.hidden_size}")
             )pbdoc"
        );
} 