import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import re
import logging
import numpy as np
from typing import Optional, List, Tuple

# Configure logging
logging.basicConfig(filename='pytorch/debugging.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s', 
                    filemode='w') # 'w' to overwrite the file each run

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # Standard float32 RMSNorm calculation
        norm = x.norm(2, dim=-1, keepdim=True)
        output = x * (self.weight / (norm / math.sqrt(x.shape[-1]) + self.eps))
        return output

# Real RoPE implementation (matches HuggingFace Llama)
def precompute_freqs_cis(dim, end, theta=10000.0, device=None):
    """Precompute complex exponentials for RoPE."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(end, device=device).float()
    freqs = torch.outer(t, freqs)  # (seq_len, dim/2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis  # (seq_len, dim/2)

# Helper function for direct RoPE manipulation
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_emb(x, freqs_cis):
    """Applies RoPE using direct real-valued manipulation."""
    # x: [bs, seq_len, n_heads, head_dim]
    # freqs_cis: [seq_len, head_dim / 2] (complex)

    # Get seq_len from freqs_cis to handle partial sequences
    seq_len_freq = freqs_cis.shape[0]
    
    # Ensure freqs_cis matches the sequence length of x
    # This handles cases where seq_len_x might be different (e.g., during kv caching, though not used here)
    # We take the portion of freqs_cis relevant to the input sequence length
    freqs_cis = freqs_cis[:seq_len_freq]

    # Extract real (cos) and imaginary (sin) parts
    # freqs_cos/sin shape: [seq_len, head_dim / 2]
    freqs_cos = freqs_cis.real
    freqs_sin = freqs_cis.imag

    # Add batch and head dimensions for broadcasting
    # Shape: [1, seq_len, 1, head_dim / 2]
    freqs_cos = freqs_cos.unsqueeze(0).unsqueeze(2)
    freqs_sin = freqs_sin.unsqueeze(0).unsqueeze(2)

    # Duplicate cos and sin to match the full head_dim for element-wise multiplication
    # Shape becomes: [1, seq_len, 1, head_dim]
    freqs_cos = torch.cat((freqs_cos, freqs_cos), dim=-1)
    freqs_sin = torch.cat((freqs_sin, freqs_sin), dim=-1)

    # Apply the rotation using the formula: x_rotated = x * cos + rotate_half(x) * sin
    # rotate_half operates on the last dimension (head_dim)
    x_rotated = (x * freqs_cos) + (rotate_half(x) * freqs_sin)

    # No need to cast back if input dtype was correct (e.g., float32)
    return x_rotated

# Old implementation using view_as_complex
# def apply_rotary_emb_complex(x, freqs_cis):
#     # x: (batch, seq, n_heads, head_dim)
#     x_ = x.float().reshape(*x.shape[:-1], -1, 2)
#     x_ = torch.view_as_complex(x_)
#     freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)  # (1, seq, 1, head_dim/2)
#     x_out = x_ * freqs_cis
#     x_out = torch.view_as_real(x_out).reshape_as(x)
#     return x_out.type_as(x)

def repeat_kv(x, n_rep):
    """Repeat K/V heads for GQA compatibility."""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x.unsqueeze(3)
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

class Llama2Block(nn.Module):
    def __init__(self, hidden_size, num_q_heads, num_kv_heads, mlp_hidden_size, rope_theta=10000.0, max_seq_len=2048, rms_norm_eps=1e-5):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.num_key_value_groups = num_q_heads // num_kv_heads
        self.head_dim = hidden_size // num_q_heads
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        # Norms (default to float32)
        self.rmsnorm1 = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.rmsnorm2 = RMSNorm(hidden_size, eps=rms_norm_eps)
        # Attention projections (no bias, default to float32)
        self.q_proj = nn.Linear(hidden_size, num_q_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_q_heads * self.head_dim, hidden_size, bias=False)
        # MLP (default to float32)
        self.gate_proj = nn.Linear(hidden_size, mlp_hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, mlp_hidden_size, bias=False)
        self.down_proj = nn.Linear(mlp_hidden_size, hidden_size, bias=False)
        # Precompute RoPE frequencies
        self.register_buffer("freqs_cis", precompute_freqs_cis(self.head_dim, self.max_seq_len, self.rope_theta), persistent=False)
        # --- Log precomputed freqs_cis for pos=1 --- 
        if self.freqs_cis.shape[0] > 1:
             freqs_cis_pos1 = self.freqs_cis[1]
             for i in range(min(5, freqs_cis_pos1.shape[0])): # Log first 5 complex numbers
                 cos_val = freqs_cis_pos1[i].real.item()
                 sin_val = freqs_cis_pos1[i].imag.item()
                 logging.info(f"PyTorch RoPE Precompute (Pos=1, FreqDim={i}): cos={cos_val:.6f} sin={sin_val:.6f}")
        # --- End log ---

    def load_weights(self, weights, prefix):
        def copy_weight(param, tensor, name):
            logging.info(f"Loading {name}:")
            logging.info(f"  - Tensor shape: {tensor.shape}, dtype: {tensor.dtype}")
            logging.info(f"  - Param shape: {param.shape}, dtype: {param.dtype}")
            if param.shape != tensor.shape:
                if param.shape == tensor.t().shape:
                    logging.info(f"  - Transposing tensor to match parameter shape.")
                    param.data.copy_(tensor.t())
                else:
                    logging.error(f"Shape mismatch for {name}: param {param.shape}, tensor {tensor.shape}")
                    raise ValueError(f"Shape mismatch for {name}: param {param.shape}, tensor {tensor.shape}")
            else:
                param.data.copy_(tensor)
            logging.info(f"  - Copied. Param mean: {param.data.mean():.4f}, std: {param.data.std():.4f}")
        # RMSNorms
        copy_weight(self.rmsnorm1.weight, weights[prefix + "input_layernorm.weight"], f"{prefix}input_layernorm.weight")
        copy_weight(self.rmsnorm2.weight, weights[prefix + "post_attention_layernorm.weight"], f"{prefix}post_attention_layernorm.weight")
        # Attention projections
        copy_weight(self.q_proj.weight, weights[prefix + "self_attn.q_proj.weight"], f"{prefix}self_attn.q_proj.weight")
        copy_weight(self.k_proj.weight, weights[prefix + "self_attn.k_proj.weight"], f"{prefix}self_attn.k_proj.weight")
        copy_weight(self.v_proj.weight, weights[prefix + "self_attn.v_proj.weight"], f"{prefix}self_attn.v_proj.weight")
        copy_weight(self.o_proj.weight, weights[prefix + "self_attn.o_proj.weight"], f"{prefix}self_attn.o_proj.weight")
        # MLP projections
        copy_weight(self.gate_proj.weight, weights[prefix + "mlp.gate_proj.weight"], f"{prefix}mlp.gate_proj.weight")
        copy_weight(self.up_proj.weight, weights[prefix + "mlp.up_proj.weight"], f"{prefix}mlp.up_proj.weight")
        copy_weight(self.down_proj.weight, weights[prefix + "mlp.down_proj.weight"], f"{prefix}mlp.down_proj.weight")

    def forward(self, x, attn_mask=None, layer_idx=None, pos: Optional[int] = None, k_cache=None, v_cache=None):
        bsz, seq_len, hidden_size = x.shape
        prefix = f"L{layer_idx}"
        residual = x 

        # --- Attention Block --- 
        h_norm1 = self.rmsnorm1(x)
        if layer_idx == 0 and seq_len > 1:
            log_tensor_stats(f"L0 P1 Input to RMSNorm1", x[:, 1, :])
            log_tensor_stats(f"L0 P1 Output of RMSNorm1", h_norm1[:, 1, :])

        if pos is None: # Original full-sequence logic
            q = self.q_proj(h_norm1).view(bsz, seq_len, self.num_q_heads, self.head_dim)
            k = self.k_proj(h_norm1).view(bsz, seq_len, self.num_kv_heads, self.head_dim)
            v = self.v_proj(h_norm1).view(bsz, seq_len, self.num_kv_heads, self.head_dim)
            
            freqs_cis = self.freqs_cis[:seq_len]
            q_rope = apply_rotary_emb(q, freqs_cis)
            k_rope = apply_rotary_emb(k, freqs_cis)
            
            # KVCache is not used here in the original logic path
            
            k_repeated = repeat_kv(k_rope, self.num_key_value_groups) # Repeat K for GQA after RoPE
            v_repeated = repeat_kv(v, self.num_key_value_groups) # Repeat V for GQA
            
            q_attn = q_rope.permute(0, 2, 1, 3) # [bs, n_heads, seq_len, head_dim]
            k_attn = k_repeated.permute(0, 2, 1, 3)      # [bs, n_heads, seq_len, head_dim]
            v_attn = v_repeated.permute(0, 2, 1, 3)      # [bs, n_heads, seq_len, head_dim]

            attn_scores = torch.matmul(q_attn, k_attn.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            if attn_mask is not None:
                if attn_mask.dim() == 2: attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
                if attn_mask.dim() == 3: attn_mask = attn_mask.unsqueeze(1)
                # Ensure mask dimensions match scores dimensions [bs, n_heads, seq_len, seq_len]
                attn_scores = attn_scores.masked_fill(~attn_mask[:, :, :seq_len, :seq_len], float('-inf'))
                # Logging for full sequence (optional)
                # if layer_idx == 0 and seq_len > 1:
                #    scores_log = attn_scores[0, 0, 1, :].detach().cpu().numpy()
                #    logging.info(f"L0 P1 H0 Scores (Full Seq): [ {' '.join(map(str, scores_log))} ]")
                
            attn_probs = F.softmax(attn_scores, dim=-1, dtype=torch.float32)
            # Logging for full sequence (optional)
            # if layer_idx == 0 and seq_len > 1:
            #    probs_log = attn_probs[0, 0, 1, :].detach().cpu().numpy()
            #    logging.info(f"L0 P1 H0 Probs (Full Seq): [ {' '.join(map(str, probs_log))} ]")

            attn_out = torch.matmul(attn_probs, v_attn.type_as(attn_probs))
            attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(bsz, seq_len, hidden_size)
        
        else: # Token-by-token logic with KVCache
            assert seq_len == 1, f"Expected seq_len=1 for token-by-token generation, but got {seq_len}"
            assert k_cache is not None and v_cache is not None, "KVCache must be provided for token-by-token generation"
            
            # Calculate Q, K, V for the single input token
            q = self.q_proj(h_norm1).view(bsz, 1, self.num_q_heads, self.head_dim) # seq_len is 1
            k = self.k_proj(h_norm1).view(bsz, 1, self.num_kv_heads, self.head_dim) # seq_len is 1
            v = self.v_proj(h_norm1).view(bsz, 1, self.num_kv_heads, self.head_dim) # seq_len is 1

            # Apply RoPE for the current position
            freqs_cis = self.freqs_cis[pos:pos+1] # Get freqs for the current position
            q_rope = apply_rotary_emb(q, freqs_cis)
            k_rope = apply_rotary_emb(k, freqs_cis)

            # --- Update KVCache --- 
            # Shapes: k_rope [bs, 1, n_kv_heads, head_dim], v [bs, 1, n_kv_heads, head_dim]
            # Cache shapes passed in: [bs, n_kv_heads, max_seq_len, head_dim]
            # We need to insert at the 'pos' index in the sequence dimension (dim=2)
            k_cache[:, :, pos:pos+1, :] = k_rope.permute(0, 2, 1, 3) # [bs, n_kv_heads, 1, head_dim]
            v_cache[:, :, pos:pos+1, :] = v.permute(0, 2, 1, 3)      # [bs, n_kv_heads, 1, head_dim]
                 
            # --- Retrieve K/V history --- 
            # Get keys/values up to the current position (inclusive)
            current_seq_len_cache = pos + 1
            keys = k_cache[:, :, :current_seq_len_cache, :] # [bs, n_kv_heads, current_seq_len_cache, head_dim]
            values = v_cache[:, :, :current_seq_len_cache, :] # [bs, n_kv_heads, current_seq_len_cache, head_dim]
            
            # Repeat K/V for Grouped Query Attention (GQA)
            # Input shape to repeat_kv: [bs, current_seq_len_cache, n_kv_heads, head_dim]
            keys_repeated = repeat_kv(keys.permute(0, 2, 1, 3), self.num_key_value_groups) # [bs, current_seq_len_cache, n_q_heads, head_dim]
            values_repeated = repeat_kv(values.permute(0, 2, 1, 3), self.num_key_value_groups) # [bs, current_seq_len_cache, n_q_heads, head_dim]

            # Reshape Q, K, V for attention calculation
            q_attn = q_rope.permute(0, 2, 1, 3)  # [bs, n_q_heads, 1, head_dim]
            # Permute K/V back to [bs, n_q_heads, current_seq_len_cache, head_dim]
            k_attn = keys_repeated.permute(0, 2, 1, 3)
            v_attn = values_repeated.permute(0, 2, 1, 3)

            # --- Calculate attention scores --- 
            # Q @ K^T -> [bs, n_q_heads, 1, head_dim] @ [bs, n_q_heads, head_dim, current_seq_len_cache] = [bs, n_q_heads, 1, current_seq_len_cache]
            attn_scores = torch.matmul(q_attn, k_attn.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            # --- Softmax --- 
            attn_probs = F.softmax(attn_scores, dim=-1, dtype=torch.float32) # Softmax over key sequence length `current_seq_len_cache`

            # --- Calculate attention output --- 
            # Prob @ V -> [bs, n_q_heads, 1, current_seq_len_cache] @ [bs, n_q_heads, current_seq_len_cache, head_dim] = [bs, n_q_heads, 1, head_dim]
            attn_out = torch.matmul(attn_probs, v_attn.type_as(attn_probs)) 
            
            # Reshape attn_out back to [bs, 1, hidden_size]
            attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(bsz, 1, hidden_size)

        # --- Common processing after attention --- 
        # (O Projection)
        attn_out_proj = self.o_proj(attn_out) 
        
        # --- First Residual Connection --- 
        h_resid1 = residual + attn_out_proj
        if layer_idx == 0 and seq_len > 1:
            log_tensor_stats(f"L0 P1 State after 1st Residual", h_resid1[:, 1, :])

        # --- MLP Block --- 
        residual2 = h_resid1
        h_norm2 = self.rmsnorm2(h_resid1)
        if layer_idx == 0 and seq_len > 1:
            log_tensor_stats(f"L0 P1 Input to RMSNorm2", h_resid1[:, 1, :])
            log_tensor_stats(f"L0 P1 Output of RMSNorm2", h_norm2[:, 1, :])
        
        gate = self.gate_proj(h_norm2)
        up = self.up_proj(h_norm2)
        if layer_idx == 0 and seq_len > 1:
            log_tensor_stats(f"L0 P1 Gate Proj Output", gate[:, 1, :])
            log_tensor_stats(f"L0 P1 Up Proj Output", up[:, 1, :])

        gate_activated = F.silu(gate)
        if layer_idx == 0 and seq_len > 1:
             log_tensor_stats(f"L0 P1 Gate after SiLU", gate_activated[:, 1, :])
        
        gate_mul_up = gate_activated * up
        if layer_idx == 0 and seq_len > 1:
             log_tensor_stats(f"L0 P1 Gate * Up", gate_mul_up[:, 1, :])

        mlp_out = self.down_proj(gate_mul_up)
        if layer_idx == 0 and seq_len > 1:
            log_tensor_stats(f"L0 P1 Down Proj Output (MLP Out)", mlp_out[:, 1, :])

        # --- Second Residual Connection --- 
        out = residual2 + mlp_out
        if layer_idx == 0 and seq_len > 1:
            log_tensor_stats(f"L0 P1 State after 2nd Residual (Layer Output)", out[:, 1, :])
            logging.info(f"--- PyTorch Layer 0 End ---")
            
        return out

# Add utility function for logging tensor stats
def log_tensor_stats(name, tensor):
    if tensor.numel() == 0: 
        logging.info(f"[PyTorch] {name}: EMPTY TENSOR")
        return
    t_np = tensor.detach().cpu().numpy().flatten()
    minv = t_np.min()
    maxv = t_np.max()
    mean = t_np.mean()
    finite = np.isfinite(t_np).all()
    logging.info(f"[PyTorch] {name}: shape={list(tensor.shape)}, min={minv:.6f}, max={maxv:.6f}, mean={mean:.6f}, all_finite={finite}")
    # Log first 5 elements
    first_5 = t_np[:5]
    logging.info(f"[PyTorch] {name} first 5: {' '.join(f'{x:.6f}' for x in first_5)}")

class TinyLlama(nn.Module):
    """TinyLlama model: embedding, transformer stack, output head. Uses config for hyperparameters."""
    def __init__(self, config=None, weights=None):
        super().__init__()
        # Get hyperparameters from config
        vocab_size = config.get('vocab_size')
        hidden_size = config.get('hidden_size')
        num_layers = config.get('num_hidden_layers')
        num_q_heads = config.get('num_attention_heads')
        num_kv_heads = config.get('num_key_value_heads')
        mlp_hidden_size = config.get('intermediate_size')
        max_seq_len = config.get('max_position_embeddings', 2048)
        rope_theta = config.get('rope_theta', 10000.0) # Get rope_theta or default
        rms_norm_eps = config.get('rms_norm_eps', 1e-5) # Get norm eps or default
        # --- Added for KVCache ---
        self.head_dim = hidden_size // num_q_heads
        self.max_seq_len = max_seq_len 
        self.n_layers = num_layers
        self.n_kv_heads = num_kv_heads
        # --- End KVCache additions ---

        # Build model (defaulting to float32)
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.blocks = nn.ModuleList([
            Llama2Block(hidden_size, num_q_heads, num_kv_heads, mlp_hidden_size, 
                        rope_theta=rope_theta, max_seq_len=max_seq_len, rms_norm_eps=rms_norm_eps)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.output_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # --- Initialize KVCache --- 
        self.init_kv_cache()
        # --- End KVCache Init ---

        # NO explicit weight tying here based on config

        # Load weights if provided (will load into float32 params)
        if weights is not None:
            # Load embedding weights 
            embed_weight = None
            embed_key = None
            for key in ["model.embed_tokens.weight", "embed_tokens.weight", "embedding.weight"]:
                 if key in weights:
                    embed_weight = weights[key]
                    embed_key = key
                    break
            if embed_weight is not None:
                 logging.info(f"Loading {embed_key}:")
                 logging.info(f"  - Tensor shape: {embed_weight.shape}, dtype: {embed_weight.dtype}")
                 logging.info(f"  - Param shape: {self.embedding.weight.shape}, dtype: {self.embedding.weight.dtype}")
                 self.embedding.weight.data.copy_(embed_weight)
                 logging.info(f"  - Copied (Embedding). Param mean: {self.embedding.weight.data.mean():.4f}, std: {self.embedding.weight.data.std():.4f}")
            else:
                 logging.warning("Warning: Embedding weights not found in safetensors file.")

            # Load output head weights separately if they exist and aren't tied
            output_head_key = "lm_head.weight" # Common key for untied head
            if output_head_key in weights:
                 output_weight = weights[output_head_key]
                 logging.info(f"Loading {output_head_key}:")
                 logging.info(f"  - Tensor shape: {output_weight.shape}, dtype: {output_weight.dtype}")
                 logging.info(f"  - Param shape: {self.output_head.weight.shape}, dtype: {self.output_head.weight.dtype}")
                 if self.output_head.weight.shape == output_weight.shape:
                    self.output_head.weight.data.copy_(output_weight)
                 elif self.output_head.weight.shape == output_weight.t().shape:
                     logging.info("  - Transposing tensor to match parameter shape.")
                     self.output_head.weight.data.copy_(output_weight.t())
                 else:
                      logging.error(f"Shape mismatch for {output_head_key}: param {self.output_head.weight.shape}, tensor {output_weight.shape}")
                      raise ValueError(f"Shape mismatch for {output_head_key}: param {self.output_head.weight.shape}, tensor {output_weight.shape}")
                 logging.info(f"  - Copied (Output Head). Param mean: {self.output_head.weight.data.mean():.4f}, std: {self.output_head.weight.data.std():.4f}")
            else:
                 # If lm_head.weight doesn't exist, and tie_word_embeddings was false, 
                 # it implies the model expects the output head initialized but potentially not used?
                 # Or maybe another key is used? For now, just warn.
                 logging.warning(f"Warning: Output head weights ({output_head_key}) not found. Output head will use initial weights.")

            # Load final norm weights
            norm_key = "model.norm.weight"
            if norm_key in weights:
                norm_weight = weights[norm_key]
                logging.info(f"Loading {norm_key}:")
                logging.info(f"  - Tensor shape: {norm_weight.shape}, dtype: {norm_weight.dtype}")
                logging.info(f"  - Param shape: {self.norm.weight.shape}, dtype: {self.norm.weight.dtype}")
                self.norm.weight.data.copy_(norm_weight)
                logging.info(f"  - Copied (Final Norm). Param mean: {self.norm.weight.data.mean():.4f}, std: {self.norm.weight.data.std():.4f}")
            else:
                logging.warning(f"Warning: Final norm weights ({norm_key}) not found.")
            # Load transformer block weights
            for i, block in enumerate(self.blocks):
                prefix = f"model.layers.{i}."
                block.load_weights(weights, prefix)

            # Print shapes of layer 0 projection weights after loading
            if len(self.blocks) > 0:
                print(f"PyTorch Layer 0 q_proj weight shape: {self.blocks[0].q_proj.weight.shape}")
                print(f"PyTorch Layer 0 k_proj weight shape: {self.blocks[0].k_proj.weight.shape}")

    def init_kv_cache(self, batch_size=1):
        """Initialize the K/V cache tensors."""
        logging.info(f"Initializing KVCache for batch_size={batch_size}, max_seq_len={self.max_seq_len}, layers={self.n_layers}, kv_heads={self.n_kv_heads}, head_dim={self.head_dim}")
        # Create empty lists to hold tensors for each layer
        k_cache = []
        v_cache = []
        # Desired shape: [batch_size, num_kv_heads, max_seq_len, head_dim]
        cache_shape = (batch_size, self.n_kv_heads, self.max_seq_len, self.head_dim)
        for _ in range(self.n_layers):
            k_cache.append(torch.zeros(cache_shape, dtype=self.embedding.weight.dtype, device=self.embedding.weight.device))
            v_cache.append(torch.zeros(cache_shape, dtype=self.embedding.weight.dtype, device=self.embedding.weight.device))
        
        # Register caches as buffers (part of state, not parameters)
        # We use lists of tensors, which register_buffer doesn't handle directly.
        # Instead, store them as regular attributes. Ensure they are moved to the correct device.
        self.k_cache = k_cache
        self.v_cache = v_cache
        logging.info(f"KVCache initialized. k_cache length: {len(self.k_cache)}, v_cache length: {len(self.v_cache)}")
        if self.k_cache: logging.info(f"Example k_cache[0] shape: {self.k_cache[0].shape}")

    def forward(self, input_ids, attention_mask=None, pos: Optional[int] = None):
        logging.info(f"[PyTorch] TinyLlama.forward called with input_ids shape: {input_ids.shape}, pos={pos}")
        if not torch.is_tensor(input_ids):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        # Ensure input_ids is on the same device as embedding weights
        input_ids = input_ids.to(self.embedding.weight.device)
        
        x = self.embedding(input_ids) # Output will be float32
        # Add batch dimension if needed
        if x.dim() == 2:
            x = x.unsqueeze(0)

        # Ensure attention_mask is on the correct device and dtype if provided
        if attention_mask is not None:
            attention_mask = attention_mask.to(x.device)

        for i, block in enumerate(self.blocks):
            x = block(x, attn_mask=attention_mask, layer_idx=i, pos=pos,
                      k_cache=self.k_cache[i] if pos is not None else None, 
                      v_cache=self.v_cache[i] if pos is not None else None) # Pass relevant cache slice
            # --- Log output from block (Strategy 1 target) ---
            if pos == 1 and i == 0: # Specifically for Strategy 1 comparison
                log_tensor_stats(f"PyT L{i} Output (pos=1)", x)
                logging.info(f"--- PyTorch Layer {i} End for pos=1 ---")
        
        x = self.norm(x)
        # --- Log final norm output ---
        # if pos is not None: # Log only if generating token-by-token
        #     log_tensor_stats(f"PyT Final Norm Output (pos={pos})", x)

        logits = self.output_head(x)
        # --- Log final logits ---
        # if pos is not None:
        #     log_tensor_stats(f"PyT Final Logits (pos={pos})", logits)
        
        logging.info(f"[PyTorch] TinyLlama.forward returning logits shape: {logits.shape}")
        return logits # Return float32 logits 