import json
from tinyllama import TinyLlama
from utils import load_tokenizer, load_safetensors_weights, load_config
import torch
import torch.nn.functional as F
import logging
import re
import scipy.special
import numpy as np

# Configure logging (same as in tinyllama.py)
logging.basicConfig(filename='pytorch/debugging.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s', 
                    filemode='w') # 'w' to overwrite the file each run

def main():
    # Load tokenizer config for special tokens
    with open("data/tokenizer_config.json", "r", encoding="utf-8") as f:
        tokenizer_config = json.load(f)
    bos_token = tokenizer_config["bos_token"]
    eos_token = tokenizer_config["eos_token"]
    
    # Hard-coded paths to data directory
    weights_path = "data/model.safetensors"
    tokenizer_path = "data/tokenizer.model"
    config_path = "data/config.json"

    # Load config
    config = load_config(config_path)
    logging.info(f"Config: {config}")

    # Load tokenizer
    tokenizer = load_tokenizer(tokenizer_path)

    # Get EOS token ID for stopping generation AFTER tokenizer is loaded
    eos_token_id = tokenizer.eos_id()
    if eos_token_id is None:
        eos_token_id = tokenizer_config.get('eos_token_id', 2) # Default to 2 if not found
        logging.warning(f"Could not get eos_id directly from tokenizer, using ID: {eos_token_id}")

    # Load weights and model
    weights = load_safetensors_weights("data/model.safetensors")
    model = TinyLlama(config=config, weights=weights)
    logging.info(f"Model initialized with default dtype: {next(model.parameters()).dtype}")

    # Print first 10 values of output_head (lm_head) weights
    lm_head_flat = model.output_head.weight.view(-1)
    first_10_lm_head = lm_head_flat[:10].tolist()
    print("lm_head first 10 values:", first_10_lm_head)
    logging.info(f"lm_head first 10 values: {first_10_lm_head}")

    # Questions to test
    questions = [
        "What is the capital of France?",
        "Who wrote Hamlet?"
    ]

    # Only Q: A: style prompt
    prompt_template = "Q: {question}\nA:"

    def extract_first_sentence_or_line(text):
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        for sentence in sentences:
            cleaned = sentence.strip()
            if cleaned:
                return cleaned
        for line in text.splitlines():
            cleaned = line.strip()
            if cleaned:
                return cleaned
        return text.strip()

    # Run inference for each question
    for question in questions:
        print(f"\n==============================\nQuestion: {question}\n==============================")
        prompt = prompt_template.format(question=question)
        logging.info(f"\n===== Q: A: Style =====\nPrompt: {prompt}")
        print(f"\n===== Q: A: Style =====\nPrompt: {prompt}")

        # Tokenize prompt (Add BOS token)
        initial_input_ids = tokenizer.encode(f"{bos_token}{prompt}")
        print("Prompt token IDs:", initial_input_ids)
        logging.info(f"Prompt token IDs: {initial_input_ids}")
        input_ids_tensor = torch.tensor([initial_input_ids], dtype=torch.long).to(model.embedding.weight.device)
        logging.info(f"Initial Input IDs (with BOS): {initial_input_ids}")

        # Print embedding stats for first token
        first_token_id = initial_input_ids[0]
        embedding_vec = model.embedding.weight[first_token_id].detach().cpu().numpy()
        minv = float(embedding_vec.min())
        maxv = float(embedding_vec.max())
        mean = float(embedding_vec.mean())
        print(f"Embedding stats for first token: min={minv}, max={maxv}, mean={mean}")
        logging.info(f"Embedding stats for first token: min={minv}, max={maxv}, mean={mean}")

        # Print stats for first RMSNorm (input_layernorm) output for the first token
        rmsnorm_weight = model.blocks[0].rmsnorm1.weight.detach().cpu().numpy()
        rmsnorm_eps = model.blocks[0].rmsnorm1.eps
        # RMSNorm: y = x / sqrt(mean(x^2) + eps) * weight
        ssq = (embedding_vec ** 2).mean()
        denom = 1.0 / ((ssq + rmsnorm_eps) ** 0.5)
        rmsnorm_out = embedding_vec * denom * rmsnorm_weight
        minv_rms = float(rmsnorm_out.min())
        maxv_rms = float(rmsnorm_out.max())
        mean_rms = float(rmsnorm_out.mean())
        print(f"First RMSNorm output stats: min={minv_rms}, max={maxv_rms}, mean={mean_rms}")
        logging.info(f"First RMSNorm output stats: min={minv_rms}, max={maxv_rms}, mean={mean_rms}")
        # --- START: Save RMSNorm output for C++ comparison ---
        ref_filename = "rmsnorm_out_ref.bin"
        try:
            with open(ref_filename, 'wb') as f:
                f.write(rmsnorm_out.astype(np.float32).tobytes())
            logging.info(f"Saved reference RMSNorm output (Layer 0, Token 0) to {ref_filename}")
        except Exception as e:
            logging.error(f"Failed to save reference RMSNorm output to {ref_filename}: {e}")
        # --- END: Save RMSNorm output ---

        # Print stats for first Q projection (q_proj) output for the first token
        q_proj_weight = model.blocks[0].q_proj.weight.detach().cpu().numpy()
        # q_proj: out = weight @ rmsnorm_out
        q_proj_out = q_proj_weight @ rmsnorm_out
        minv_q = float(q_proj_out.min())
        maxv_q = float(q_proj_out.max())
        mean_q = float(q_proj_out.mean())
        print(f"First Q projection output stats: min={minv_q}, max={maxv_q}, mean={mean_q}")
        logging.info(f"First Q projection output stats: min={minv_q}, max={maxv_q}, mean={mean_q}")

        # Q before RoPE
        num_heads = config["num_attention_heads"]
        hidden_size = config["hidden_size"]
        kv_dim = (hidden_size // num_heads) * config["num_key_value_heads"]
        print(f"Q before RoPE shape: [{q_proj_out.shape[0]}] num_heads={num_heads} head_dim={hidden_size // num_heads} pos=0 first 5: {q_proj_out[:5]}")
        logging.info(f"Q before RoPE shape: [{q_proj_out.shape[0]}] num_heads={num_heads} head_dim={hidden_size // num_heads} pos=0 first 5: {q_proj_out[:5]}")
        # Q after RoPE (no-op for t=0, but keep for symmetry)
        q_rope = q_proj_out[:kv_dim].copy()  # Only use first kv_dim elements
        print(f"Q after RoPE shape: [{q_rope.shape[0]}] first 5: {q_rope[:5]}")
        logging.info(f"Q after RoPE shape: [{q_rope.shape[0]}] first 5: {q_rope[:5]}")

        # K projection
        k_proj_weight = model.blocks[0].k_proj.weight.detach().cpu().numpy()
        k_proj_out = k_proj_weight @ rmsnorm_out  # shape [kv_dim]
        minv_k = float(k_proj_out.min())
        maxv_k = float(k_proj_out.max())
        mean_k = float(k_proj_out.mean())
        print(f"First K projection output stats: min={minv_k}, max={maxv_k}, mean={mean_k}")
        logging.info(f"First K projection output stats: min={minv_k}, max={maxv_k}, mean={mean_k}")

        # V projection
        v_proj_weight = model.blocks[0].v_proj.weight.detach().cpu().numpy()
        v_proj_out = v_proj_weight @ rmsnorm_out  # shape [kv_dim]
        minv_v = float(v_proj_out.min())
        maxv_v = float(v_proj_out.max())
        mean_v = float(v_proj_out.mean())
        print(f"First V projection output stats: min={minv_v}, max={maxv_v}, mean={mean_v}")
        logging.info(f"First V projection output stats: min={minv_v}, max={maxv_v}, mean={mean_v}")

        # K after RoPE (no-op for t=0, but keep for symmetry)
        k_rope = k_proj_out.copy()
        print(f"K after RoPE shape: [{k_rope.shape[0]}] first 5: {k_rope[:5]}")
        logging.info(f"K after RoPE shape: [{k_rope.shape[0]}] first 5: {k_rope[:5]}")

        # Attention score (dot Q_rope, K_rope)
        attn_score = float(np.dot(q_rope, k_rope))
        print(f"First attention score (dot Q_rope, K_rope): {attn_score}")
        logging.info(f"First attention score (dot Q_rope, K_rope): {attn_score}")

        # Attention probability (softmax)
        attn_prob = float(scipy.special.softmax([attn_score])[0])
        print(f"First attention probability (after softmax): {attn_prob}")
        logging.info(f"First attention probability (after softmax): {attn_prob}")

        # Attention output (context vector, weighted sum of V)
        attn_out = attn_prob * v_proj_out
        minv_attn = float(attn_out.min())
        maxv_attn = float(attn_out.max())
        mean_attn = float(attn_out.mean())
        print(f"First attention output stats: min={minv_attn}, max={maxv_attn}, mean={mean_attn}")
        logging.info(f"First attention output stats: min={minv_attn}, max={maxv_attn}, mean={mean_attn}")

        # --- Autoregressive Generation Loop --- 
        model.eval()
        generated_ids = initial_input_ids.copy() # Start with prompt IDs
        max_new_tokens = 50 # Max tokens to generate
        logging.info(f"Starting generation loop (max_new_tokens={max_new_tokens}, eos_token_id={eos_token_id})")

        with torch.no_grad():
            for i in range(max_new_tokens):
                current_ids_tensor = torch.tensor([generated_ids], dtype=torch.long).to(model.embedding.weight.device)
                seq_len = current_ids_tensor.shape[1]
                attn_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=current_ids_tensor.device))
                logits = model(current_ids_tensor, attention_mask=attn_mask)
                last_token_logits = logits[:, -1, :]
                if i == 0:
                    first_10_logits = last_token_logits[0, :10].tolist()
                    print("First generated token logits (first 10):", first_10_logits)
                    logging.info(f"First generated token logits (first 10): {first_10_logits}")
                next_token_id = torch.argmax(last_token_logits, dim=-1).item()
                logging.info(f"Step {i+1}: Predicted token ID: {next_token_id}")
                generated_ids.append(next_token_id)
                if next_token_id == eos_token_id:
                    logging.info(f"EOS token ({eos_token_id}) generated. Stopping generation.")
                    break
            else:
                logging.warning(f"Max new tokens ({max_new_tokens}) reached before EOS token.")

        # Decode the full generated sequence
        try:
            full_decoded_text = tokenizer.decode(generated_ids)
            generated_part_ids = generated_ids[len(initial_input_ids):]
            generated_decoded_text = tokenizer.decode(generated_part_ids)
            cleaned_text = extract_first_sentence_or_line(generated_decoded_text)
            logging.info(f"Full Generated Sequence IDs: {generated_ids}")
            logging.info(f"Full Decoded Text:\n-------\n{full_decoded_text}\n-------")
            logging.info(f"Generated Part IDs: {generated_part_ids}")
            logging.info(f"Generated Decoded Text (raw):\n-------\n{generated_decoded_text}\n-------")
            logging.info(f"Generated Decoded Text (cleaned):\n-------\n{cleaned_text}\n-------")
            print(f"\nModel output (cleaned):\n{cleaned_text}\n")
        except Exception as e:
            logging.error(f"Error during decoding or logging decoded text: {e}", exc_info=True)
            print(f"Error during decoding: {e}")

if __name__ == "__main__":
    main() 