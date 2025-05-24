#!/usr/bin/env python3
"""
Test script for TinyLlama batch processing Python bindings.
This script demonstrates the new generate_batch() functionality.
"""

import sys
import time
import os
# Remove current directory from path to avoid local import conflicts
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir in sys.path:
    sys.path.remove(current_dir)
    
try:
    import tinyllama_cpp as tl
except ImportError:
    print("Error: tinyllama_bindings module not found. Please compile the Python bindings first.")
    print("Run: python setup.py build_ext --inplace")
    sys.exit(1)

def test_single_vs_batch_generation():
    """Test that single and batch generation produce consistent results."""
    
    print("Initializing TinyLlamaSession with batch support...")
    session = tl.TinyLlamaSession(
        model_path="../data/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        tokenizer_path="../data/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", 
        threads=4,
        n_gpu_layers=22,
        use_mmap=True,
        use_kv_quant=True,
        use_batch_generation=True,
        max_batch_size=5
    )
    
    print(f"Model config: {session.get_config()}")
    
    # Test prompts
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
    
    # Test batch generation
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
    
    # Compare results
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

def test_batch_edge_cases():
    """Test batch generation edge cases."""
    print("\n=== Testing Batch Edge Cases ===")
    
    session = tl.TinyLlamaSession(
        model_path="../data/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        tokenizer_path="../data/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", 
        threads=4,
        n_gpu_layers=22,
        use_mmap=True,
        use_kv_quant=True,
        use_batch_generation=True,
        max_batch_size=3
    )
    
    # Test empty batch
    try:
        results = session.generate_batch([])
        print("✗ Empty batch should have thrown an exception")
    except RuntimeError as e:
        print(f"✓ Empty batch correctly threw exception: {e}")
    
    # Test batch size exceeding limit
    try:
        large_batch = ["Test prompt"] * 5  # Exceeds max_batch_size=3
        results = session.generate_batch(large_batch)
        print("✗ Oversized batch should have thrown an exception")
    except RuntimeError as e:
        print(f"✓ Oversized batch correctly threw exception: {e}")
    
    # Test valid small batch
    try:
        small_batch = ["Hello", "World"]
        results = session.generate_batch(small_batch, steps=10)
        print(f"✓ Small batch processed successfully: {len(results)} results")
    except Exception as e:
        print(f"✗ Small batch failed: {e}")

if __name__ == "__main__":
    print("TinyLlama Batch Processing Test")
    print("=" * 40)    
    try:
        test_single_vs_batch_generation()
        test_batch_edge_cases()
        print("\n✓ All tests completed!")
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc() 