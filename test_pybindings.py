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
    print(f"  BOS token ID: {config_obj.bos_token_id}")
    print(f"  EOS token ID: {config_obj.eos_token_id}")
    print(f"  Chat template type: {config_obj.chat_template_type}")
    print(f"  Pre-tokenizer type: {config_obj.pre_tokenizer_type}")
    print(f"-------------------------")

def run_model_test(model_path, prompt_arg=None):
    print(f"\n=============================================")
    print(f"  TESTING TINYLLAMA SESSION: {model_path}")
    print(f"=============================================")

    if not os.path.exists(model_path):
        print(f"ERROR: Model path not found: {model_path}")
        return False

    session = None
    try:
        print(f"\n--- Initializing TinyLlamaSession with model: {model_path} ---")
        with tinyllama_bindings.ostream_redirect(stdout=True, stderr=True):
            session = tinyllama_bindings.TinyLlamaSession(model_path)
            print("--- TinyLlamaSession initialized successfully ---")
        
        cpp_loaded_config = session.get_config()
        print_config(cpp_loaded_config, "Config from TinyLlamaSession")

        # --- Test Generation ---
        print(f"\n--- Testing Generation using TinyLlamaSession.generate() ---")
        
        current_prompt = prompt_arg if prompt_arg else "What is the capital of France?"
        num_steps = 30
        temperature = 0.1 # Low temperature for more deterministic output
        top_k = 1 # Effectively greedy
        top_p = 0.9 # Does not matter much if top_k is 1

        print(f"Prompt: \"{current_prompt}\"")
        print(f"Generating up to {num_steps} tokens...")
        
        generated_text = ""
        with tinyllama_bindings.ostream_redirect(stdout=True, stderr=True):
            generated_text = session.generate(
                current_prompt,
                num_steps,
                temperature,
                top_k,
                top_p,
                "",
                True
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
    args = parser.parse_args()

    print(f"Starting Python bindings test script for model: {args.model_path}")
    if args.prompt:
        print(f"Using custom prompt: \"{args.prompt}\"")
    
    test_passed = run_model_test(args.model_path, args.prompt)

    print("\n=============================================")
    print("                TEST SUMMARY")
    print("=============================================")
    print(f"Session Test ({os.path.basename(args.model_path)}): {'PASSED' if test_passed else 'FAILED'}")
    print("=============================================")

    if not test_passed:
        sys.exit(1)
    sys.exit(0)