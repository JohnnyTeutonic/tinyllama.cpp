import sys

# --- START: Direct GGUF Import Test ---
try:
    from gguf import GGUFReader
    print("Successfully imported GGUFReader at the top of the script.")
except ImportError as e_top:
    print(f"Failed to import from 'gguf' at the TOP of the script. Error: {e_top}")
    print("Please ensure the 'gguf' pip package (from llama.cpp/gguf-py) is installed in the Python environment being used to run this script.")
    sys.exit(1) 
# --- END: Direct GGUF Import Test ---

import torch
from safetensors.torch import load_file as load_safetensors_file
import numpy as np
import struct # For reading GGUF (though GGUFReader is preferred)

# --- Configuration for TinyLlama 1.1B ---
HIDDEN_SIZE = 2048
N_HEAD = 32
N_HEAD_KV = 4 # For GQA
HEAD_DIM = HIDDEN_SIZE // N_HEAD
# --- End Configuration ---

# --- Llama Permutation Function (adapted from convert_hf_to_gguf.py) ---
def permute_llama_tensor(weights: torch.Tensor, n_head_config: int, n_head_kv_config: int | None):
    """
    Applies the Llama-style permutation to Q, K, or V tensors.
    Also acts as its own inverse if applied again.
    For Q_proj: weights.shape is (HIDDEN_SIZE, HIDDEN_SIZE), n_head_config=N_HEAD, n_head_kv_config should be N_HEAD or None.
    For K_proj: weights.shape is (N_HEAD_KV * HEAD_DIM, HIDDEN_SIZE), n_head_config=N_HEAD, n_head_kv_config=N_HEAD_KV.
    Uses float32 for intermediate permutation steps to enhance stability for bfloat16.
    """
    original_dtype = weights.dtype
    if original_dtype == torch.bfloat16:
        print(f"    Permuting: original dtype {original_dtype}, converting to float32 for permutation.")
        weights = weights.to(torch.float32)

    num_groups_for_reshape = 0
    if weights.shape[0] == HIDDEN_SIZE: # Q_proj
        if n_head_kv_config is not None and n_head_config != n_head_kv_config:
            print(f"Warning: Unexpected n_head_kv_config for Q_proj like tensor: {n_head_kv_config}")
        num_groups_for_reshape = n_head_config 
    elif weights.shape[0] == N_HEAD_KV * HEAD_DIM: # K_proj or V_proj (GQA)
        if n_head_kv_config is None:
            raise ValueError("n_head_kv_config cannot be None for K/V-like GQA tensors")
        num_groups_for_reshape = n_head_kv_config 
    else:
        # If original_dtype was bfloat16, weights is now float32. Error message should reflect original intent.
        error_shape = weights.shape
        if original_dtype == torch.bfloat16:
            # Approximate original shape if it was bfloat16 for error message clarity
            # This is a bit of a simplification as the number of elements is the same.
            pass # Shape is already correct for element count.
        raise ValueError(f"Unexpected tensor shape {error_shape} (element-wise, original dtype {original_dtype}) for Llama permutation with N_HEAD={N_HEAD}, N_HEAD_KV={N_HEAD_KV}, HEAD_DIM={HEAD_DIM}.")

    permuted_weights = (
        weights.reshape(num_groups_for_reshape, 2, HEAD_DIM // 2, weights.shape[1])
        .swapaxes(1, 2)
        .reshape(weights.shape)
    )

    if original_dtype == torch.bfloat16:
        print(f"    Permuting: converting back to {original_dtype}.")
        permuted_weights = permuted_weights.to(original_dtype)
    
    return permuted_weights
# --- End Llama Permutation Function ---

# Helper function to convert bfloat16 to float32
def bf16_to_fp32_torch(bf16_val_int_tensor: torch.Tensor):
    # Assuming bf16_val_int_tensor is a BF16 tensor
    return bf16_val_int_tensor.to(torch.float32)

def bf16_tensor_to_fp32(bf16_tensor):
    return bf16_tensor.to(torch.float32)


def read_gguf_tensor_data(gguf_path, tensor_name_to_find, expected_shape, expected_gguf_dtype_str):
    """Reads a tensor, attempting to handle BF16 correctly and log details."""
    print(f"\nAttempting to read GGUF tensor: '{tensor_name_to_find}' (expecting type like {expected_gguf_dtype_str}).")
    tensor_torch = None # Initialize return value
    
    try:
        reader = GGUFReader(gguf_path, 'r')
        found_tensor_info = None
        found_tensor_index = -1

        # Find tensor info by name
        if not hasattr(reader, 'tensors'):
             print("--> ERROR: GGUFReader object does not have 'tensors' attribute.")
             return None
             
        for i, tensor_info in enumerate(reader.tensors):
            if tensor_info.name == tensor_name_to_find:
                found_tensor_info = tensor_info
                found_tensor_index = i
                break
        
        if not found_tensor_info:
            print(f"--> ERROR: GGUF tensor '{tensor_name_to_find}' - Info object not found in reader.tensors.")
            available_names = [t.name for t in reader.tensors]
            print(f"    Available tensor names: {available_names}")
            return None
            
        # Log metadata BEFORE attempting to load data
        actual_gguf_type_enum = found_tensor_info.tensor_type
        actual_gguf_type_name = actual_gguf_type_enum.name
        metadata_shape = found_tensor_info.shape 
        print(f"--> Found GGUF tensor info: '{tensor_name_to_find}' at index {found_tensor_index}.")
        print(f"    Metadata Type: {actual_gguf_type_name} (Enum: {actual_gguf_type_enum}) Shape: {metadata_shape}")

        # Attempt to get the tensor data (ReaderTensor wrapper)
        tensor_wrapper_obj = reader.get_tensor(found_tensor_index) 
        if tensor_wrapper_obj is None:
            print(f"--> ERROR: reader.get_tensor({found_tensor_index}) returned None for '{tensor_name_to_find}'.")
            return None
        if not hasattr(tensor_wrapper_obj, 'data') or not isinstance(tensor_wrapper_obj.data, np.ndarray):
             print(f"--> ERROR: Failed to get valid NumPy data via reader.get_tensor().data for '{tensor_name_to_find}'. Type was {type(tensor_wrapper_obj)}")
             return None

        tensor_data_np = tensor_wrapper_obj.data
        numpy_dtype = tensor_data_np.dtype
        numpy_shape = tensor_data_np.shape
        print(f"--> Extracted NumPy array: Shape={numpy_shape}, Dtype={numpy_dtype}")

        # --- Special handling for BF16 data that might be read as uint8 ---
        # This must happen BEFORE the generic shape validation if it's to correct the shape.
        if actual_gguf_type_name == 'BF16' and tensor_data_np.dtype == np.uint8:
            print(f"    NOTE: GGUF tensor '{tensor_name_to_find}' is BF16 but NumPy array is uint8. Attempting reinterpretation.")
            num_bytes_expected_for_bf16 = np.prod(metadata_shape) * 2 # 2 bytes per BF16 element
            if tensor_data_np.nbytes != num_bytes_expected_for_bf16:
                print(f"    --> ERROR: Byte count mismatch. Expected {num_bytes_expected_for_bf16} bytes for BF16 shape {metadata_shape}, "
                      f"but uint8 array has {tensor_data_np.nbytes} bytes.")
                return None
            try:
                # Reinterpret the bytes as uint16. This changes dtype and potentially the shape's last dimension.
                tensor_data_np = tensor_data_np.view(np.uint16)
                print(f"    Viewed as np.uint16: Shape={tensor_data_np.shape}, Dtype={tensor_data_np.dtype}")
                
                # Ensure the shape matches the metadata shape.
                # .view(np.uint16) on a (D1, D2*2) uint8 array should result in a (D1, D2) uint16 array.
                if list(tensor_data_np.shape) != list(metadata_shape):
                    print(f"    --> WARNING: Shape after .view(np.uint16) is {tensor_data_np.shape}, GGUF metadata shape is {metadata_shape}. Attempting explicit reshape.")
                    tensor_data_np = tensor_data_np.reshape(metadata_shape)
                    print(f"    Reshaped to metadata_shape: Shape={tensor_data_np.shape}, Dtype={tensor_data_np.dtype}")
                
                # Update numpy_dtype and numpy_shape for subsequent logic and logging
                numpy_dtype = tensor_data_np.dtype
                numpy_shape = tensor_data_np.shape
            except Exception as e_reinterpret:
                print(f"    --> ERROR: Failed to reinterpret uint8 data as uint16 for BF16 tensor '{tensor_name_to_find}': {e_reinterpret}")
                import traceback
                traceback.print_exc()
                return None

        # --- Shape Validation --- 
        if list(numpy_shape) != list(expected_shape): # Compare as lists for flexibility
            print(f"--> WARNING: SHAPE MISMATCH for '{tensor_name_to_find}'. Expected {expected_shape}, got {numpy_shape}. Will attempt reshape.")
            try:
                tensor_data_np = tensor_data_np.reshape(expected_shape)
                print(f"    Reshape successful to {tensor_data_np.shape}")
                numpy_shape = tensor_data_np.shape # Update shape after reshape
            except ValueError as e_reshape:
                print(f"--> ERROR: Reshape failed for '{tensor_name_to_find}': {e_reshape}. Cannot proceed.")
                return None
                
        # --- Type Handling & Conversion to PyTorch --- 
        # Use the gguf tensor_type enum value for decisions
        tensor_type_value = actual_gguf_type_enum.value

        # Mapping from GGUF type *values* (like those in convert_hf_to_gguf.py) to handle BF16
        # GGML_TYPE_F32 = 0, GGML_TYPE_F16 = 1, GGML_TYPE_BF16 = 19 (value might vary)
        # Check if the gguf library exposes these constants or use magic numbers carefully
        # Assuming BF16 has enum value 19 based on convert script context
        GGML_TYPE_VALUE_BF16 = 19 # This will be replaced by direct enum comparison
        GGML_TYPE_VALUE_F32 = 0
        GGML_TYPE_VALUE_F16 = 1
        GGML_TYPE_VALUE_Q8_0 = 8

        if expected_gguf_dtype_str == 'BF16':
            if actual_gguf_type_name == 'BF16':
                if numpy_dtype == np.uint16: # This should now be the case after uint8 reinterpretation
                    print(f"    OK: NumPy dtype is uint16 for BF16 GGUF. Converting via frombuffer to torch.bfloat16.")
                    # Get raw bytes from the (potentially reshaped) uint16 numpy array
                    raw_bytes = tensor_data_np.tobytes()
                    # Create a torch.int16 tensor from the buffer, then reshape, then view as bfloat16
                    int16_torch_tensor = torch.frombuffer(raw_bytes, dtype=torch.int16).reshape(tensor_data_np.shape)
                    tensor_torch = int16_torch_tensor.view(torch.bfloat16)
                else:
                     # This path should ideally not be hit if uint8 was handled, or if it was already uint16.
                     print(f"    ERROR: NumPy dtype is {numpy_dtype} for BF16 GGUF, expected uint16 after reinterpretation. Cannot proceed with frombuffer conversion.")
                     return None
            # Handle cases where file is F32/F16 even if we expected BF16
            elif actual_gguf_type_name == 'F32' and numpy_dtype == np.float32:
                 print(f"    NOTE: Expected BF16, but GGUF is F32. Loading as FP32.")
                 tensor_torch = torch.from_numpy(tensor_data_np)
            elif actual_gguf_type_name == 'F16' and numpy_dtype == np.float16:
                 print(f"    NOTE: Expected BF16, but GGUF is F16. Loading as FP16 -> FP32 for safety.")
                 tensor_torch = torch.from_numpy(tensor_data_np.astype(np.float32)) # Convert F16 numpy to FP32 torch
            elif actual_gguf_type_name == 'Q8_0': # Example of quantized type
                 print(f"    ERROR: Expected GGUF BF16, but Metadata Type is {actual_gguf_type_name}. File is likely quantized.")
                 return None
            else:
                 print(f"--> ERROR: GGUF metadata type {actual_gguf_type_name} incompatible with expected BF16 or readable alternatives.")
                 return None

        elif expected_gguf_dtype_str == 'F32':
             if actual_gguf_type_name == 'F32':
                 if numpy_dtype != np.float32:
                     print(f"    NOTE: NumPy dtype is {numpy_dtype}. Converting to torch.float32.")
                     tensor_torch = torch.from_numpy(tensor_data_np.astype(np.float32))
                 else:
                     print(f"    OK: NumPy dtype is float32. Creating torch.float32 tensor.")
                     tensor_torch = torch.from_numpy(tensor_data_np)
             # If expected F32 but GGUF is BF16, convert BF16 to F32
             elif actual_gguf_type_name == 'BF16' and numpy_dtype == np.uint16: # Assuming BF16 was processed to uint16
                 print(f"    NOTE: Expected F32, but GGUF is BF16. Converting BF16 (from int16 bytes) to F32.")
                 # Get raw bytes from the (potentially reshaped) uint16 numpy array
                 raw_bytes = tensor_data_np.tobytes()
                 # Create a torch.int16 tensor from the buffer, then reshape, view as bfloat16, then convert to float32
                 int16_torch_tensor = torch.frombuffer(raw_bytes, dtype=torch.int16).reshape(tensor_data_np.shape)
                 tensor_torch = int16_torch_tensor.view(torch.bfloat16).to(torch.float32)
             else:
                print(f"--> ERROR: GGUF metadata type {actual_gguf_type_name} incompatible with expected F32 or readable alternatives like BF16.")
                return None
        else:
            print(f"--> ERROR: Unsupported expected GGUF dtype for comparison: {expected_gguf_dtype_str}")
            return None
            
        if tensor_torch is None:
             print(f"--> ERROR: Failed to create PyTorch tensor for '{tensor_name_to_find}'.")
             return None
             
        print(f"--> Successfully converted '{tensor_name_to_find}' to PyTorch tensor: Shape={tensor_torch.shape}, Dtype={tensor_torch.dtype}")
        return tensor_torch

    except Exception as e:
        print(f"--> ERROR: Unexpected failure during GGUF processing for '{tensor_name_to_find}': {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        if reader and hasattr(reader, 'close'):
             try: reader.close()
             except: pass # Ignore errors during close


def compare_tensors(st_tensor, gguf_tensor_rev_permuted, name):
    print(f"\nComparing {name}")
    print(f"Comparing SafeTensors {name} (Original) (shape: {st_tensor.shape}, dtype: {st_tensor.dtype}) and GGUF {name} (Reverse-Permuted) (shape: {gguf_tensor_rev_permuted.shape}, dtype: {gguf_tensor_rev_permuted.dtype})")
    
    if st_tensor.dtype != gguf_tensor_rev_permuted.dtype:
        print(f"Dtype mismatch: ST={st_tensor.dtype}, GGUF={gguf_tensor_rev_permuted.dtype}. Comparing using allclose.")
        if not (st_tensor.is_floating_point() and gguf_tensor_rev_permuted.is_floating_point()):
             print("Cannot compare non-floating point tensors of different types.")
             return False
        st_comp = st_tensor.to(torch.float32)
        gguf_comp = gguf_tensor_rev_permuted.to(torch.float32)
        comparison_method = "allclose (FP32)"
        are_equal = torch.allclose(st_comp, gguf_comp, atol=1e-5, rtol=1e-3) 
    elif st_tensor.dtype == torch.bfloat16:
        comparison_method = "equal (BF16)"
        are_equal = torch.equal(st_tensor, gguf_tensor_rev_permuted)
    else: 
        comparison_method = "allclose"
        are_equal = torch.allclose(st_tensor, gguf_tensor_rev_permuted, atol=1e-5, rtol=1e-3)

    if are_equal:
        print(f"Tensors SafeTensors {name} (Original) and GGUF {name} (Reverse-Permuted) are THE SAME (using {comparison_method}).")
        return True
    else:
        print(f"Tensors SafeTensors {name} (Original) and GGUF {name} (Reverse-Permuted) are DIFFERENT (using {comparison_method}).")
        st_fp32 = st_tensor.to(torch.float32)
        gguf_fp32 = gguf_tensor_rev_permuted.to(torch.float32)
        abs_diff = torch.abs(st_fp32 - gguf_fp32)
        print(f"  Max absolute difference (FP32): {torch.max(abs_diff).item()}")
        print(f"  Mean absolute difference (FP32): {torch.mean(abs_diff).item()}")
        
        if comparison_method == "equal (BF16)":
             diff_indices = torch.nonzero(st_tensor != gguf_tensor_rev_permuted, as_tuple=True)
        else:
             diff_indices = torch.where(~torch.isclose(st_fp32, gguf_fp32, atol=1e-5, rtol=1e-3))

        if len(diff_indices[0]) > 0:
            print(f"  First 10 differing elements (indices, val_st(orig), val_gguf_rev(orig), abs_diff(fp32)):")
            for i in range(min(10, len(diff_indices[0]))):
                idx_tuple = tuple(dim[i].item() for dim in diff_indices)
                try: val1_orig = st_tensor[idx_tuple].item() 
                except: val1_orig = "N/A"
                try: val2_orig = gguf_tensor_rev_permuted[idx_tuple].item() 
                except: val2_orig = "N/A"
                abs_d_fp32 = abs_diff[idx_tuple].item()
                print(f"    {idx_tuple}: {val1_orig} vs {val2_orig} (fp32_diff: {abs_d_fp32:.6f})")
        return False

def main(safetensors_path, gguf_path):
    # Minimal imports at the top now
    print(f"Using SafeTensors file: {safetensors_path}")
    print(f"Using GGUF file (expecting BF16): {gguf_path}")
    print(f"N_HEAD: {N_HEAD}, N_HEAD_KV: {N_HEAD_KV}, HEAD_DIM: {HEAD_DIM}")

    st_q_proj_bf16 = None 
    st_k_proj_bf16 = None 
    # Initialize ALL potentially assigned variables to None
    gguf_q_proj = None
    gguf_k_proj = None
    gguf_q_proj_rev_permuted = None
    gguf_k_proj_rev_permuted = None
    q_match = False # Also initialize comparison results
    k_match = False

    st_q_proj_name = "model.layers.0.self_attn.q_proj.weight"
    st_k_proj_name = "model.layers.0.self_attn.k_proj.weight"
    
    gguf_q_proj_name = "blk.0.attn_q.weight"
    gguf_k_proj_name = "blk.0.attn_k.weight"

    # --- Load SafeTensors (Keep as BF16) --- 
    print("\n--- Loading SafeTensors (as BF16) ---")
    try:
        st_model = load_safetensors_file(safetensors_path, device="cpu")
        st_q_proj_bf16 = st_model.get(st_q_proj_name)
        st_k_proj_bf16 = st_model.get(st_k_proj_name)

        if st_q_proj_bf16 is not None:
            if st_q_proj_bf16.dtype != torch.bfloat16:
                 print(f"Warning: Expected SafeTensors {st_q_proj_name} to be bfloat16, but got {st_q_proj_bf16.dtype}")
            print(f"Loaded SafeTensors {st_q_proj_name} (shape: {st_q_proj_bf16.shape}, dtype: {st_q_proj_bf16.dtype})")
        else:
            print(f"SafeTensors tensor {st_q_proj_name} not found.")

        if st_k_proj_bf16 is not None:
            if st_k_proj_bf16.dtype != torch.bfloat16:
                 print(f"Warning: Expected SafeTensors {st_k_proj_name} to be bfloat16, but got {st_k_proj_bf16.dtype}")
            print(f"Loaded SafeTensors {st_k_proj_name} (shape: {st_k_proj_bf16.shape}, dtype: {st_k_proj_bf16.dtype})")
        else:
            print(f"SafeTensors tensor {st_k_proj_name} not found.")

    except Exception as e:
        print(f"Error loading SafeTensors file {safetensors_path}: {e}")
        st_q_proj_bf16 = None 
        st_k_proj_bf16 = None 

    # --- Load GGUF Tensors (Attempt BF16) AND REVERSE-PERMUTE --- 
    print("\n--- Loading GGUF Tensors (expecting BF16) ---")
    expected_q_shape = (HIDDEN_SIZE, HIDDEN_SIZE)
    expected_k_shape = (N_HEAD_KV * HEAD_DIM, HIDDEN_SIZE)

    # Attempt to load GGUF tensors
    gguf_q_proj = read_gguf_tensor_data(gguf_path, gguf_q_proj_name, expected_q_shape, "BF16")
    if gguf_q_proj is not None:
        print(f"DEBUG: gguf_q_proj[0,0] before perm: {gguf_q_proj[0,0].item()} (dtype: {gguf_q_proj.dtype})")
        print(f"DEBUG: gguf_q_proj[0,0].to(fp32) before perm: {gguf_q_proj[0,0].to(torch.float32).item()}")

    gguf_k_proj = read_gguf_tensor_data(gguf_path, gguf_k_proj_name, expected_k_shape, "BF16")
    if gguf_k_proj is not None:
        print(f"DEBUG: gguf_k_proj[0,0] before perm: {gguf_k_proj[0,0].item()} (dtype: {gguf_k_proj.dtype})")
        print(f"DEBUG: gguf_k_proj[0,0].to(fp32) before perm: {gguf_k_proj[0,0].to(torch.float32).item()}")

    # Attempt reverse permutation if loading succeeded
    if gguf_q_proj is not None:
        try:
            print(f"Applying reverse permutation to GGUF Q tensor (dtype: {gguf_q_proj.dtype})...")
            gguf_q_proj_rev_permuted = permute_llama_tensor(gguf_q_proj, N_HEAD, N_HEAD)
            print(f"Reverse-permuted GGUF Q (shape: {gguf_q_proj_rev_permuted.shape}, dtype: {gguf_q_proj_rev_permuted.dtype})")
        except Exception as e:
             print(f"--> ERROR during reverse permutation of GGUF Q tensor: {e}")
             gguf_q_proj_rev_permuted = None # Ensure it's None on error
    else:
         # gguf_q_proj is already None from read_gguf_tensor_data
         print(f"Skipping reverse permutation for GGUF Q tensor as it failed to load.")

    if gguf_k_proj is not None:
        try:
            print(f"Applying reverse permutation to GGUF K tensor (dtype: {gguf_k_proj.dtype})...")
            gguf_k_proj_rev_permuted = permute_llama_tensor(gguf_k_proj, N_HEAD, N_HEAD_KV)
            print(f"Reverse-permuted GGUF K (shape: {gguf_k_proj_rev_permuted.shape}, dtype: {gguf_k_proj_rev_permuted.dtype})")
        except Exception as e:
             print(f"--> ERROR during reverse permutation of GGUF K tensor: {e}")
             gguf_k_proj_rev_permuted = None # Ensure it's None on error
    else:
         # gguf_k_proj is already None from read_gguf_tensor_data
         print(f"Skipping reverse permutation for GGUF K tensor as it failed to load.")

    # --- Comparison (Original ST BF16 vs Reverse-Permuted GGUF BF16/FP32) --- 
    print("\n--- Comparison Results ---")
    
    if st_q_proj_bf16 is not None and gguf_q_proj_rev_permuted is not None:
        q_match = compare_tensors(st_q_proj_bf16, gguf_q_proj_rev_permuted, "Q_PROJ (Layer 0)")
    else:
        print("Skipping Q_PROJ comparison: one or both tensors unavailable.")

    if st_k_proj_bf16 is not None and gguf_k_proj_rev_permuted is not None:
        k_match = compare_tensors(st_k_proj_bf16, gguf_k_proj_rev_permuted, "K_PROJ (Layer 0)")
    else:
        print("Skipping K_PROJ comparison: one or both tensors unavailable.")

    print("\n--- Summary ---")
    # Use the comparison flags which are now reliably False if comparison didn't happen
    if st_q_proj_bf16 is None:
        print("Q_PROJ tensors: SafeTensors failed to load.")
    elif gguf_q_proj_rev_permuted is None:
        print("Q_PROJ tensors: GGUF tensor failed to load or process.")
    elif q_match:
        print("Q_PROJ tensors MATCH (Original ST vs Reverse-Permuted GGUF).")
    else:
        print("Q_PROJ tensors DO NOT MATCH (Original ST vs Reverse-Permuted GGUF).")

    if st_k_proj_bf16 is None:
        print("K_PROJ tensors: SafeTensors failed to load.")
    elif gguf_k_proj_rev_permuted is None:
        print("K_PROJ tensors: GGUF tensor failed to load or process.")
    elif k_match:
        print("K_PROJ tensors MATCH (Original ST vs Reverse-Permuted GGUF).")
    else:
        print("K_PROJ tensors DO NOT MATCH (Original ST vs Reverse-Permuted GGUF).")

if __name__ == "__main__":
    # Removed optional args for hidden_size/kv_dim as they are now constants
    if len(sys.argv) != 3:
        print("Usage: python compare_tensors.py <path_to_safetensors_bf16_file> <path_to_gguf_bf16_file>")
        sys.exit(1)
    
    safetensors_file_path = sys.argv[1]
    gguf_file_path = sys.argv[2]
    
    main(safetensors_file_path, gguf_file_path)
