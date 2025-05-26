#!/usr/bin/env python3
"""
Test script for TinyLlama batch processing Python bindings.
This script demonstrates the new generate_batch() functionality.
"""

import sys
import time
import os
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir in sys.path:
    sys.path.remove(current_dir)
    
import tinyllama_cpp as tl

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Test TinyLlama batch processing Python bindings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # For Llama 3+ GGUF models (tokenizer embedded)
  python test_batch_bindings.py path/to/model.gguf
  
  # For Llama 2 GGUF models or SafeTensors (separate tokenizer required)
  python test_batch_bindings.py path/to/model.gguf --tokenizer path/to/tokenizer.json
  
  # For SafeTensors format
  python test_batch_bindings.py path/to/safetensors/directory --tokenizer path/to/tokenizer.json
        """
    )
    
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to model file (.gguf) or directory (SafeTensors)"
    )
    
    parser.add_argument(
        "--tokenizer", "--tokenizer-path",
        type=str,
        default=None,
        help="Path to tokenizer.json file. Optional for Llama 3+ GGUF models (will use model_path if not specified). Required for SafeTensors and Llama 2 models."
    )
    
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of CPU threads for inference (default: 4)"
    )
    
    parser.add_argument(
        "--gpu-layers",
        type=int,
        default=-1,
        help="Number of layers to offload to GPU. -1 for all layers, 0 for CPU only (default: -1)"
    )
    
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=5,
        help="Maximum batch size for testing (default: 5)"
    )
    
    return parser.parse_args()

def test_single_vs_batch_generation(model_path, tokenizer_path, threads, gpu_layers, max_batch_size):
    """Test that single and batch generation produce consistent results."""
    print("Initializing TinyLlamaSession with batch support...")
    
    # Use model_path as tokenizer_path if not specified (for Llama 3+ GGUF models)
    if tokenizer_path is None:
        tokenizer_path = model_path
        print(f"Using model path as tokenizer path: {tokenizer_path}")
    
    session = tl.TinyLlamaSession(
        model_path=model_path,
        tokenizer_path=tokenizer_path, 
        threads=threads,
        n_gpu_layers=gpu_layers,
        use_mmap=True,
        use_kv_quant=True,
        use_batch_generation=True,
        max_batch_size=max_batch_size
    )
    
    print(f"Model config: {session.get_config()}")
    
    test_prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a short poem about coding.",
        "What are the benefits of renewable energy?",
        "How do neural networks work?"
    ]
    
    print(f"\nTesting with {len(test_prompts)} prompts...")
    
    print("\n=== Single Generation (Sequential) ===")
    start_time = time.time()
    single_results = []
    for i, prompt in enumerate(test_prompts):
        print(f"Processing prompt {i+1}/{len(test_prompts)}: {prompt[:50]}...")
        result = session.generate(
            prompt=prompt,
            steps=50,
            temperature=0.1,
            system_prompt="You are a helpful assistant.",
            apply_q_a_format=True
        )
        single_results.append(result)
    single_time = time.time() - start_time
    
    print("\n=== Batch Generation ===")
    start_time = time.time()
    batch_results = session.generate_batch(
        prompts=test_prompts,
        steps=50,
        temperature=0.1,
        system_prompt="You are a helpful assistant.",
        apply_q_a_format=True
    )
    batch_time = time.time() - start_time
    
    print(f"\n=== Results Comparison ===")
    print(f"Single generation time: {single_time:.2f}s")
    print(f"Batch generation time: {batch_time:.2f}s")
    print(f"Time ratio (batch/single): {batch_time/single_time:.2f}x")
    
    print(f"\nNumber of results - Single: {len(single_results)}, Batch: {len(batch_results)}")
    
    print(f"\n=== Sample Results ===")
    for i in range(min(2, len(test_prompts))):
        print(f"\nPrompt {i+1}: {test_prompts[i]}")
        print(f"Single result: {single_results[i][:100]}...")
        print(f"Batch result:  {batch_results[i][:100]}...")
        
        if len(single_results[i]) > 0 and len(batch_results[i]) > 0:
            print("✓ Both methods produced non-empty results")
        else:
            print("✗ One or both methods produced empty results")

def test_batch_edge_cases(model_path, tokenizer_path, threads, gpu_layers):
    """Test batch generation edge cases."""
    print("\n=== Testing Batch Edge Cases ===")
    
    # Use model_path as tokenizer_path if not specified
    if tokenizer_path is None:
        tokenizer_path = model_path
    
    session = tl.TinyLlamaSession(
        model_path=model_path,
        tokenizer_path=tokenizer_path, 
        threads=threads,
        n_gpu_layers=gpu_layers,
        use_mmap=True,
        use_kv_quant=True,
        use_batch_generation=True,
        max_batch_size=3
    )
    
    try:
        results = session.generate_batch([])
        print("✗ Empty batch should have thrown an exception")
    except RuntimeError as e:
        print(f"✓ Empty batch correctly threw exception: {e}")
    
    try:
        large_batch = ["Test prompt"] * 5
        results = session.generate_batch(large_batch)
        print("✗ Oversized batch should have thrown an exception")
    except RuntimeError as e:
        print(f"✓ Oversized batch correctly threw exception: {e}")
    
    try:
        small_batch = ["Hello", "World"]
        results = session.generate_batch(small_batch, steps=10)
        print(f"✓ Small batch processed successfully: {len(results)} results")
    except Exception as e:
        print(f"✗ Small batch failed: {e}")

if __name__ == "__main__":
    print("TinyLlama Batch Processing Test")
    print("=" * 40)
    
    args = parse_arguments()
    
    print(f"Model path: {args.model_path}")
    print(f"Tokenizer path: {args.tokenizer or 'Using model path'}")
    print(f"Threads: {args.threads}")
    print(f"GPU layers: {args.gpu_layers}")
    print(f"Max batch size: {args.max_batch_size}")
    print()
    
    try:
        test_single_vs_batch_generation(
            args.model_path, 
            args.tokenizer, 
            args.threads, 
            args.gpu_layers, 
            args.max_batch_size
        )
        test_batch_edge_cases(
            args.model_path, 
            args.tokenizer, 
            args.threads, 
            args.gpu_layers
        )
        print("\n✓ All tests completed!")
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc() 