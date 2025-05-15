import tinyllama_bindings
import os
import sys
import argparse

def print_config(config_obj, source_msg="Config"):
    print(f"--- {source_msg} ---")
    print(f"  Hidden size: {config_obj.hidden_size}")
    print(f"  Intermediate size: {config_obj.intermediate_size}")
    print(f"  Num attention heads: {config_obj.num_attention_heads}")
    print(f"  Num key_value heads: {config_obj.num_key_value_heads}")
    print(f"  Num hidden layers: {config_obj.num_hidden_layers}")
    print(f"  Vocab size: {config_obj.vocab_size}")
    print(f"  Max position embeddings: {config_obj.max_position_embeddings}")
    print(f"  RMS norm eps: {config_obj.rms_norm_eps}")
    print(f"  Rope theta: {config_obj.rope_theta}")
    print(f"  Hidden act: {config_obj.hidden_act}")
    print(f"  Torch dtype: {config_obj.torch_dtype}")
    print(f"  Architecture: {config_obj.architecture}")
    print(f"  Model name: {config_obj.model_name}")
    print(f"  Is GGUF loaded: {config_obj.is_gguf_file_loaded}")
    tf_str = "UNKNOWN_IN_PYTHON_TEST"
    print(f"  Tokenizer Family (raw value): {config_obj.tokenizer_family}") # This will print something like <TokenizerFamily.LLAMA3_TIKTOKEN: 2>
    print(f"  (See C++ repr for ModelConfig for stringified tokenizer_family)")
    print(f"  BOS token ID: {config_obj.bos_token_id}")
    print(f"  EOS token ID: {config_obj.eos_token_id}")
    print(f"  Chat template type: {config_obj.chat_template_type}")
    print(f"  Pre-tokenizer type: {config_obj.pre_tokenizer_type}")
    print(f"-------------------------")

def run_model_test(model_path, tokenizer_path_arg=None, threads_arg=1, n_gpu_layers_arg=0, use_mmap_arg=False, prompt_arg=None, test_temperature=0.1, test_steps=30):
    print(f"\n=============================================")
    print(f"  TESTING TINYLLAMA SESSION: {model_path}")
    print(f"  Tokenizer Path: {tokenizer_path_arg if tokenizer_path_arg else '(default from model or GGUF internal)'}")
    print(f"  Threads: {threads_arg}, N GPU Layers: {n_gpu_layers_arg}, Use Mmap: {use_mmap_arg}")
    print(f"=============================================")

    if not os.path.exists(model_path):
        print(f"ERROR: Model path not found: {model_path}")
        return False

    session = None
    try:
        print(f"\n--- Initializing TinyLlamaSession ---")
        with tinyllama_bindings.ostream_redirect(stdout=True, stderr=True):
            # Use provided args for session creation
            session = tinyllama_bindings.TinyLlamaSession(
                model_path,
                tokenizer_path_arg if tokenizer_path_arg else "", # Pass empty string if None, C++ handles it
                threads_arg,
                n_gpu_layers_arg,
                use_mmap_arg
            )
            print("--- TinyLlamaSession initialized successfully ---")
        
        cpp_loaded_config = session.get_config()
        print_config(cpp_loaded_config, "Config from TinyLlamaSession")
        print(f"Config C++ Repr: {repr(cpp_loaded_config)}") # Test the __repr__ we enhanced

        print("\n--- Testing ModelConfig.TokenizerFamily Enum Access ---")
        print(f"Attempting to compare config.tokenizer_family with known enum values...")
        is_llama3_detected = False
        if cpp_loaded_config.tokenizer_family == tinyllama_bindings.ModelConfig.TokenizerFamily.LLAMA3_TIKTOKEN:
            print("  Correctly identified as LLAMA3_TIKTOKEN by comparison.")
            is_llama3_detected = True
        elif cpp_loaded_config.tokenizer_family == tinyllama_bindings.ModelConfig.TokenizerFamily.LLAMA_SENTENCEPIECE:
            print("  Correctly identified as LLAMA_SENTENCEPIECE by comparison.")
        elif cpp_loaded_config.tokenizer_family == tinyllama_bindings.ModelConfig.TokenizerFamily.UNKNOWN:
            print("  Correctly identified as UNKNOWN by comparison.")
        else:
            print("  Could not match tokenizer_family to known enum values via direct comparison in Python.")
        print("--- Enum access test conceptualized ---")

        # --- Test Generation ---
        print(f"\n--- Testing Generation with intelligent apply_q_a_format ---")
        
        current_prompt = prompt_arg if prompt_arg else "What is the capital of France?"
        
        # Determine apply_q_a_format based on tokenizer_family
        apply_qa_for_generate = True # Default
        if is_llama3_detected:
            apply_qa_for_generate = False
            print(f"Python Test: Llama 3 detected (tokenizer_family), setting apply_q_a_format to False.")
        else:
            print(f"Python Test: Non-Llama 3 detected, setting apply_q_a_format to True.")

        print(f"Prompt: \"{current_prompt}\"")
        print(f"Generating up to {test_steps} tokens with temperature {test_temperature}...")
        print(f"Using apply_q_a_format: {apply_qa_for_generate}")
        
        generated_text = ""
        with tinyllama_bindings.ostream_redirect(stdout=True, stderr=True):
            generated_text = session.generate(
                current_prompt,
                test_steps,         # Use test_steps
                test_temperature,   # Use test_temperature
                top_k,              # Keep existing top_k, top_p or make them params too
                top_p,
                "",                 # system_prompt
                apply_qa_for_generate # Use determined value
            )
        
        print(f"\n--- Generation Result ---")
        print(f"Prompt: {current_prompt}")
        print(f"Generated Text: {generated_text}")
        print("--- Generation test completed ---")
        
        if not generated_text or generated_text.isspace():
            print("WARNING: Generated text is empty or whitespace.")

        print(f"\nSession test completed for {os.path.basename(model_path)}.")
        return True

    except Exception as e:
        print(f"!!! ERROR during TinyLlamaSession test for {model_path}: {e} !!!")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test TinyLlama Python bindings using TinyLlamaSession.")
    parser.add_argument("model_path", type=str, help="Path to the model directory or .gguf file.")
    parser.add_argument("--prompt", type=str, default=None, help="Optional prompt to use for generation.")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Optional tokenizer path for the session.")
    parser.add_argument("--threads", type=int, default=1, help="Number of threads for the session.")
    parser.add_argument("--n_gpu_layers", type=int, default=0, help="Number of GPU layers for the session.")
    parser.add_argument("--use_mmap", type=lambda x: (str(x).lower() == 'true'), default=False, help="Use mmap for GGUF files (true/false).")
    parser.add_argument("--test_temp", type=float, default=0.1, help="Temperature for generate test.")
    parser.add_argument("--test_steps", type=int, default=30, help="Number of steps for generate test.")
    args = parser.parse_args()

    print(f"Starting Python bindings test script for model: {args.model_path}")
    if args.prompt:
        print(f"Using custom prompt: \"{args.prompt}\"")
    
    test_passed = run_model_test(
        args.model_path, 
        args.tokenizer_path, 
        args.threads, 
        args.n_gpu_layers, 
        args.use_mmap, 
        args.prompt,
        args.test_temp,
        args.test_steps
    )

    print("\n=============================================")
    print("                TEST SUMMARY")
    print("=============================================")
    print(f"Session Test ({os.path.basename(args.model_path)}): {'PASSED' if test_passed else 'FAILED'}")
    print("=============================================")

    if not test_passed:
        sys.exit(1)
    sys.exit(0)