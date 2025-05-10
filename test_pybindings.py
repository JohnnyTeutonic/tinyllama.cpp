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
    print(f"  Chat template type: {config_obj.chat_template_type}")
    print(f"  Pre-tokenizer type: {config_obj.pre_tokenizer_type}")
    print(f"-------------------------")

def run_model_test(model_path):
    print(f"\n=============================================")
    print(f"  TESTING MODEL LOADING: {model_path}")
    print(f"=============================================")

    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        return False

    # Create a default ModelConfig. C++ side should override it with actual model config.
    py_initial_cfg = tinyllama_bindings.ModelConfig()
    py_initial_cfg.hidden_size = 1
    py_initial_cfg.vocab_size = 2

    print_config(py_initial_cfg, "Python Initial Default Config")

    model = None
    try:
        with tinyllama_bindings.ostream_redirect(stdout=True, stderr=True):
            print(f"\n--- Loading TinyLlamaModel with {os.path.basename(model_path)} ---")
            model = tinyllama_bindings.TinyLlamaModel(py_initial_cfg, model_path)
            print(f"--- TinyLlamaModel loaded successfully for {os.path.basename(model_path)} ---")
        
        cpp_loaded_config = model.get_config()
        print_config(cpp_loaded_config, "Config from C++ Model")

        if cpp_loaded_config.hidden_size == 1 or cpp_loaded_config.vocab_size == 2:
            print("WARNING: Config values from C++ model still look like Python defaults.")
            print("         This might indicate the model's config (config.json or GGUF metadata) was not loaded correctly by C++.")
        
        file_is_gguf = model_path.lower().endswith(".gguf")
        if cpp_loaded_config.is_gguf_file_loaded != file_is_gguf:
            print(f"ERROR: Mismatch in GGUF status! File is GGUF: {file_is_gguf}, C++ reports: {cpp_loaded_config.is_gguf_file_loaded}")
        print(f"\nTesting basic model functions:")
        print(f"  model.get_vocab_size(): {model.get_vocab_size()}")
        
        if model.get_vocab_size() > 0:
            test_token_id = 0 
            try:
                embedding = model.lookup_embedding(test_token_id)
                print(f"  model.lookup_embedding({test_token_id}) returned vector of size: {len(embedding)}")
                if len(embedding) != cpp_loaded_config.hidden_size:
                     print(f"  ERROR: Embedding size {len(embedding)} != hidden_size {cpp_loaded_config.hidden_size}")
                     return False
            except Exception as e_embed:
                print(f"  ERROR during lookup_embedding: {e_embed}")
                return False
        else: # vocab_size is 0 or less, which is an error state.
            print(f"  ERROR: vocab_size is {model.get_vocab_size()}, cannot test lookup_embedding.")
            return False
            
        print(f"Model test completed for {os.path.basename(model_path)}.")
        return True

    except Exception as e:
        print(f"!!! ERROR during model test for {model_path}: {e} !!!")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test TinyLlama Python bindings with a given model file.")
    parser.add_argument("model_path", type=str, help="Path to the .safetensors or .gguf model file.")
    args = parser.parse_args()

    print(f"Starting Python bindings test script for model: {args.model_path}")
    
    test_passed = run_model_test(args.model_path)

    print("\n=============================================")
    print("                TEST SUMMARY")
    print("=============================================")
    print(f"Model Test ({os.path.basename(args.model_path)}): {'PASSED' if test_passed else 'FAILED'}")
    print("=============================================")

    if not test_passed:
        sys.exit(1)
    sys.exit(0)