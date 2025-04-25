import json
from tinyllama import TinyLlama
from utils import load_tokenizer, load_safetensors_weights, load_config
import torch
import torch.nn.functional as F
import logging
import re

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
        input_ids_tensor = torch.tensor([initial_input_ids], dtype=torch.long).to(model.embedding.weight.device)
        logging.info(f"Initial Input IDs (with BOS): {initial_input_ids}")

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