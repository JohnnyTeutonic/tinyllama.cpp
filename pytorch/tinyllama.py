import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import re
import logging
import numpy as np

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

    def forward(self, x, attn_mask=None, layer_idx=None):
        # Store input for first residual connection
        residual = x 
        # Get batch size and sequence length from input
        bsz, seq_len, _ = x.shape
        prefix = f"Layer {layer_idx} " if layer_idx is not None else ""
        # --- Attention Block --- 
        h = self.rmsnorm1(x)
        if layer_idx == 0:
            h_flat = h[0,0].detach().cpu().numpy().reshape(-1)
            logging.info(f"{prefix}RMSNorm1 Output (Input to Proj) first 5: {h_flat[:5]}")
            # --- START: Save RMSNorm1 output for C++ comparison ---
            ref_filename = "rmsnorm1_out_layer0_ref.bin"
            try:
                h_save = h[0,0].detach().cpu().numpy().astype(np.float32) # Use first token, ensure float32
                with open(ref_filename, 'wb') as f:
                    f.write(h_save.tobytes())
                logging.info(f"Saved reference RMSNorm1 output (Layer 0, Token 0) to {ref_filename}")
            except Exception as e:
                logging.error(f"Failed to save reference RMSNorm1 output to {ref_filename}: {e}")
            # --- END: Save RMSNorm1 output ---
        q = self.q_proj(h).view(bsz, seq_len, self.num_q_heads, self.head_dim)
        k = self.k_proj(h).view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(h).view(bsz, seq_len, self.num_kv_heads, self.head_dim)

        # Log Q/K/V projection stats for first token
        if layer_idx == 0:
            q_proj_flat = q[0,0].detach().cpu().numpy().reshape(-1)
            k_proj_flat = k[0,0].detach().cpu().numpy().reshape(-1)
            v_proj_flat = v[0,0].detach().cpu().numpy().reshape(-1)
            # Log raw projection output before RoPE
            logging.info(f"{prefix}Q Projection Output (Pre-RoPE) first 5: {q_proj_flat[:5]}")
            logging.info(f"{prefix}K Projection Output (Pre-RoPE) first 5: {k_proj_flat[:5]}")
            logging.info(f"{prefix}Q projection stats: min={q_proj_flat.min()}, max={q_proj_flat.max()}, mean={q_proj_flat.mean()}")
            logging.info(f"{prefix}K projection stats: min={k_proj_flat.min()}, max={k_proj_flat.max()}, mean={k_proj_flat.mean()}")
            logging.info(f"{prefix}V projection stats: min={v_proj_flat.min()}, max={v_proj_flat.max()}, mean={v_proj_flat.mean()}")
            logging.info(f"{prefix}Q before RoPE shape: [{q_proj_flat.shape[0]}] first 5: {q_proj_flat[:5]}") # This is redundant now but keep for format consistency if needed

        freqs_cis = self.freqs_cis[:seq_len]
        q_rope = apply_rotary_emb(q, freqs_cis)
        k_rope = apply_rotary_emb(k, freqs_cis)

        # Log Q/K after RoPE for first token
        if layer_idx == 0:
            q_rope_flat = q_rope[0,0].detach().cpu().numpy().reshape(-1)
            k_rope_flat = k_rope[0,0].detach().cpu().numpy().reshape(-1)
            logging.info(f"{prefix}Q after RoPE shape: [{q_rope_flat.shape[0]}] first 5: {q_rope_flat[:5]}")
            logging.info(f"{prefix}K after RoPE shape: [{k_rope_flat.shape[0]}] first 5: {k_rope_flat[:5]}")

        k = repeat_kv(k_rope, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups) # Note: Using original V here, not k_rope!
        q = q_rope

        if layer_idx == 0:
            # Log the Q and K heads for the first position (pos=0) that go into the first score calculation
            q_head0_pos0_log = q[0, 0, 0, :].detach().cpu().numpy() # Batch 0, SeqPos 0, Head 0
            k_head0_pos0_log = k_rope[0, 0, 0, :].detach().cpu().numpy() # Batch 0, SeqPos 0, Head 0 (before repeat_kv)
            logging.info(f"{prefix}Q Head 0, Pos 0 (Input to Score) first 5: {q_head0_pos0_log[:5]}")
            logging.info(f"{prefix}K Head 0, Pos 0 (Input to Score) first 5: {k_head0_pos0_log[:5]}")
            
        q = q.permute(0, 2, 1, 3) 
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            attn_scores = attn_scores.masked_fill(~attn_mask, float('-inf'))
        if layer_idx == 0:
            # Log scores for first batch, first head, first query pos attending to all key pos
            scores_log = attn_scores[0, 0, 0, :].detach().cpu().numpy()
            logging.info(f"{prefix}Attention Scores (Head 0, Pos 0): {scores_log}")
            
        attn_probs = F.softmax(attn_scores, dim=-1, dtype=torch.float32) # Explicit float32 for softmax
        if layer_idx == 0:
            # Log probabilities for the same
            probs_log = attn_probs[0, 0, 0, :].detach().cpu().numpy()
            logging.info(f"{prefix}Attention Probs (Head 0, Pos 0): {probs_log}")
            
        attn_out = torch.matmul(attn_probs, v.type_as(attn_probs)) # Cast V to match probs dtype
        attn_out_unprojected = attn_out.permute(0, 2, 1, 3).contiguous().view(bsz, seq_len, self.hidden_size)
        if layer_idx == 0:
             # Log first 5 values of the attention output vector *before* projection for the first token
            attn_out_log = attn_out_unprojected[0, 0, :self.head_dim].detach().cpu().numpy() # Log first head's output vector
            logging.info(f"{prefix}Attention Out (Head 0, Pos 0, Before o_proj) first {len(attn_out_log)}: {attn_out_log[:5]}") # Print first 5 values

        attn_out_proj = self.o_proj(attn_out_unprojected.type_as(x)) # Cast back to input type before o_proj

        # Log attention output stats for first token
        if layer_idx == 0:
            attn_out_flat = attn_out_proj[0,0].detach().cpu().numpy()
            logging.info(f"{prefix}attn_out (Projected) stats: min={attn_out_flat.min()}, max={attn_out_flat.max()}, mean={attn_out_flat.mean()}")

        # --- First Residual Connection --- 
        h = residual + attn_out_proj
        if layer_idx == 0:
            h_flat = h[0,0].detach().cpu().numpy()
            logging.info(f"{prefix}post-attn residual stats: min={h_flat.min()}, max={h_flat.max()}, mean={h_flat.mean()}")

        # --- MLP Block --- 
        # Store output of attention block for second residual connection
        residual = h 
        h = self.rmsnorm2(h) # Apply norm *after* first residual
        gate = F.silu(self.gate_proj(h))
        up = self.up_proj(h)
        mlp_out = self.down_proj(gate * up)
        if layer_idx == 0:
            mlp_out_flat = mlp_out[0,0].detach().cpu().numpy()
            logging.info(f"{prefix}MLP output stats: min={mlp_out_flat.min()}, max={mlp_out_flat.max()}, mean={mlp_out_flat.mean()}")
        # --- Second Residual Connection --- 
        out = residual + mlp_out
        if layer_idx == 0:
            out_flat = out[0,0].detach().cpu().numpy()
            logging.info(f"{prefix}post-MLP residual stats: min={out_flat.min()}, max={out_flat.max()}, mean={out_flat.mean()}")
        return out

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
        # torch_dtype_str = config.get('torch_dtype', 'float32') # REMOVED
        # self.dtype = getattr(torch, torch_dtype_str) # REMOVED

        # Build model (defaulting to float32)
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.blocks = nn.ModuleList([
            Llama2Block(hidden_size, num_q_heads, num_kv_heads, mlp_hidden_size, 
                        rope_theta=rope_theta, max_seq_len=max_seq_len, rms_norm_eps=rms_norm_eps)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.output_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
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

    def forward(self, input_ids, attention_mask=None):
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
            x = block(x, attn_mask=attention_mask, layer_idx=i)
        x = self.norm(x)
        logits = self.output_head(x)
        return logits # Return float32 logits 