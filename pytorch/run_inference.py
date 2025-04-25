import json
from tinyllama import TinyLlama
from utils import load_tokenizer, load_safetensors_weights, load_config
import torch
import torch.nn.functional as F
import logging

# Configure logging (same as in tinyllama.py)
logging.basicConfig(filename='pytorch/debugging.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s', 
                    filemode='w') # 'w' to overwrite the file each run

def main():
    # Hard-coded conversation
    system = "You are a helpful assistant."
    user = "What is the capital of France?"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    # Load tokenizer config for chat template and special tokens
    # We load this first just to get the string values if needed
    with open("data/tokenizer_config.json", "r", encoding="utf-8") as f:
        tokenizer_config = json.load(f)
    bos_token = tokenizer_config["bos_token"]
    eos_token = tokenizer_config["eos_token"]
    
    # Construct prompt 
    #prompt = f"{bos_token}System: {system}\nUser: {user}{eos_token}" # Using config tokens
    prompt = f"What is the capital of France?{eos_token}" # Simple prompt
    logging.info(f"Using Prompt: {prompt}")

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
        # Fallback if not directly available (might be in added_tokens_decoder)
        eos_token_id = tokenizer_config.get('eos_token_id', 2) # Default to 2 if not found
        logging.warning(f"Could not get eos_id directly from tokenizer, using ID: {eos_token_id}")

    # Load weights
    weights = load_safetensors_weights(weights_path)

    # Initialize model using config and weights
    # Model will be float32 by default now
    model = TinyLlama(config=config, weights=weights)

    # Tokenize prompt (Add BOS token)
    logging.info(f"Prepending BOS token ({bos_token}) to prompt for encoding.")
    initial_input_ids = tokenizer.encode(f"{bos_token}{prompt}")
    input_ids_tensor = torch.tensor([initial_input_ids], dtype=torch.long).to(model.embedding.weight.device)
    logging.info(f"Initial Input IDs (with BOS): {initial_input_ids}")

    # --- Autoregressive Generation Loop --- 
    model.eval()
    generated_ids = initial_input_ids # Start with prompt IDs
    max_new_tokens = 100 # Max tokens to generate
    
    logging.info(f"Starting generation loop (max_new_tokens={max_new_tokens}, eos_token_id={eos_token_id})")

    with torch.no_grad():
        for i in range(max_new_tokens):
            current_ids_tensor = torch.tensor([generated_ids], dtype=torch.long).to(model.embedding.weight.device)
            
            # Create the causal attention mask for the current sequence length
            seq_len = current_ids_tensor.shape[1]
            attn_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=current_ids_tensor.device))
            
            # Pass the attention mask to the model
            logits = model(current_ids_tensor, attention_mask=attn_mask) 

            # Get logits for the last token
            last_token_logits = logits[:, -1, :] # Shape: [1, vocab_size]
            
            # Greedy decoding (get the most likely next token)
            next_token_id = torch.argmax(last_token_logits, dim=-1).item() # Get Python int
            logging.info(f"Step {i+1}: Predicted token ID: {next_token_id}")

            # Append predicted token ID
            generated_ids.append(next_token_id)

            # Check for EOS token
            if next_token_id == eos_token_id:
                logging.info(f"EOS token ({eos_token_id}) generated. Stopping generation.")
                break
        else: # Loop finished without hitting EOS
             logging.warning(f"Max new tokens ({max_new_tokens}) reached before EOS token.")

    # Decode the full generated sequence
    try:
        full_decoded_text = tokenizer.decode(generated_ids)
        logging.info(f"Full Generated Sequence IDs: {generated_ids}")
        logging.info(f"Full Decoded Text:\n-------\n{full_decoded_text}\n-------")

        # Decode only the generated part (after the initial prompt)
        generated_part_ids = generated_ids[len(initial_input_ids):]
        generated_decoded_text = tokenizer.decode(generated_part_ids)
        logging.info(f"Generated Part IDs: {generated_part_ids}")
        logging.info(f"Generated Decoded Text:\n-------\n{generated_decoded_text}\n-------")
    except Exception as e:
        logging.error(f"Error during decoding or logging decoded text: {e}", exc_info=True)

    # Check if special chat tokens are recognized by the tokenizer
    for special in ["<|system|>", "<|user|>", "<|assistant|>"]:
        ids = tokenizer.encode(special)
        logging.info(f"Token IDs for {special}: {ids}")


if __name__ == "__main__":
    main() 