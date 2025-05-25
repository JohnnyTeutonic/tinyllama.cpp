#!/usr/bin/env python3
"""
GPU Capability Test Script
This script checks GPU capabilities and validates kernel launch parameters
to help diagnose CUDA errors in TinyLlama.cpp.
"""

import sys
import os
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir in sys.path:
    sys.path.remove(current_dir)

try:
    import tinyllama_cpp as tl
    import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False
    print("Warning: pynvml not available. Install with: pip install pynvml")

def check_gpu_capabilities():
    """Check GPU capabilities and limits."""
    print("=== GPU Capability Check ===")
    
    if HAS_PYNVML:
        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            print(f"Number of GPUs: {device_count}")
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                print(f"\nGPU {i}: {name}")
                print(f"  Total Memory: {memory_info.total / 1024**3:.2f} GB")
                print(f"  Free Memory: {memory_info.free / 1024**3:.2f} GB")
                print(f"  Used Memory: {memory_info.used / 1024**3:.2f} GB")
                
                # Get compute capability
                major = pynvml.nvmlDeviceGetCudaComputeCapability(handle)[0]
                minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)[1]
                print(f"  Compute Capability: {major}.{minor}")
                
        except Exception as e:
            print(f"Error checking GPU info: {e}")
    else:
        print("pynvml not available - install with: pip install pynvml")

def validate_model_parameters(model_path, tokenizer_path=None):
    """Validate model parameters against GPU limits."""
    print("\n=== Model Parameter Validation ===")
    
    if tokenizer_path is None:
        tokenizer_path = model_path
    
    try:
        # Create session with minimal GPU usage for testing
        session = tl.TinyLlamaSession(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            threads=1,
            n_gpu_layers=1,  # Only load 1 layer to GPU for testing
            use_mmap=True,
            use_kv_quant=False,
            use_batch_generation=False,
            max_batch_size=1
        )
        
        config = session.get_config()
        print(f"Model Configuration:")
        print(f"  Hidden Size: {config.hidden_size}")
        print(f"  Num Attention Heads: {config.num_attention_heads}")
        print(f"  Num KV Heads: {config.num_key_value_heads}")
        print(f"  Num Layers: {config.num_hidden_layers}")
        print(f"  Vocab Size: {config.vocab_size}")
        print(f"  Max Position Embeddings: {config.max_position_embeddings}")
        
        # Calculate derived parameters
        head_dim = config.hidden_size // config.num_attention_heads
        print(f"\nDerived Parameters:")
        print(f"  Head Dimension: {head_dim}")
        print(f"  KV Head Dimension: {config.hidden_size // config.num_key_value_heads if config.num_key_value_heads > 0 else 'N/A'}")
        
        # Check against CUDA limits
        print(f"\nCUDA Limit Validation:")
        max_threads_per_block = 1024
        max_shared_memory = 48 * 1024  # 48KB typical limit
        
        print(f"  Head Dim vs Max Threads per Block: {head_dim} <= {max_threads_per_block} : {'✓' if head_dim <= max_threads_per_block else '✗ PROBLEM!'}")
        
        # Estimate shared memory usage for different context lengths
        for context_len in [128, 512, 1024, 2048]:
            shared_mem_bytes = (context_len + head_dim) * 4  # 4 bytes per float
            status = "✓" if shared_mem_bytes <= max_shared_memory else "✗ PROBLEM!"
            print(f"  Shared Memory for context {context_len}: {shared_mem_bytes} bytes <= {max_shared_memory} bytes : {status}")
        
        return True
        
    except Exception as e:
        print(f"Error validating model: {e}")
        return False

def test_minimal_inference(model_path, tokenizer_path=None):
    """Test minimal inference with CPU-only mode."""
    print("\n=== CPU-Only Inference Test ===")
    
    if tokenizer_path is None:
        tokenizer_path = model_path
    
    try:
        # CPU-only session
        session = tl.TinyLlamaSession(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            threads=1,
            n_gpu_layers=0,  # CPU only
            use_mmap=True,
            use_kv_quant=False,
            use_batch_generation=False,
            max_batch_size=1
        )
        
        result = session.generate("Hello", steps=5, temperature=0.1)
        print(f"CPU inference successful: '{result[:50]}...'")
        return True
        
    except Exception as e:
        print(f"CPU inference failed: {e}")
        return False

def test_gpu_inference_gradual(model_path, tokenizer_path=None):
    """Test GPU inference with gradually increasing GPU layers."""
    print("\n=== Gradual GPU Inference Test ===")
    
    if tokenizer_path is None:
        tokenizer_path = model_path
    
    # First get model info
    try:
        cpu_session = tl.TinyLlamaSession(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            threads=1,
            n_gpu_layers=0,
            use_mmap=True,
            use_kv_quant=False,
            use_batch_generation=False,
            max_batch_size=1
        )
        config = cpu_session.get_config()
        total_layers = config.num_hidden_layers
        del cpu_session
        
        print(f"Model has {total_layers} layers. Testing GPU layer by layer...")
        
        for gpu_layers in [1, 2, 4, 8, total_layers // 2, total_layers]:
            if gpu_layers > total_layers:
                continue
                
            try:
                print(f"  Testing with {gpu_layers} GPU layers...")
                session = tl.TinyLlamaSession(
                    model_path=model_path,
                    tokenizer_path=tokenizer_path,
                    threads=1,
                    n_gpu_layers=gpu_layers,
                    use_mmap=True,
                    use_kv_quant=False,
                    use_batch_generation=False,
                    max_batch_size=1
                )
                
                result = session.generate("Hello", steps=3, temperature=0.1)
                print(f"    ✓ {gpu_layers} layers successful")
                del session
                
            except Exception as e:
                print(f"    ✗ {gpu_layers} layers failed: {e}")
                break
                
    except Exception as e:
        print(f"Error in gradual GPU test: {e}")

def main():
    parser = argparse.ArgumentParser(description="Test GPU capabilities and diagnose CUDA errors")
    parser.add_argument("model_path", help="Path to model file or directory")
    parser.add_argument("--tokenizer", help="Path to tokenizer file (optional)")
    
    args = parser.parse_args()
    
    print("TinyLlama GPU Diagnostics")
    print("=" * 50)
    
    # Check GPU capabilities
    check_gpu_capabilities()
    
    # Validate model parameters
    if validate_model_parameters(args.model_path, args.tokenizer):
        # Test CPU inference
        if test_minimal_inference(args.model_path, args.tokenizer):
            # Test GPU inference gradually
            test_gpu_inference_gradual(args.model_path, args.tokenizer)
    
    print("\n" + "=" * 50)
    print("Diagnostics complete!")

if __name__ == "__main__":
    main() 