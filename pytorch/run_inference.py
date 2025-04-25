import json
from tinyllama import TinyLlama
from utils import load_tokenizer, load_safetensors_weights, load_config
import torch
import torch.nn.functional as F
import logging
import jinja2
import re

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
    
    # Hard-coded paths to data directory
    weights_path = "data/model.safetensors"
    tokenizer_path = "data/tokenizer.model"
    config_path = "data/config.json"

    # Load config
    config = load_config(config_path)
    logging.info(f"Config: {config}")

    # Load tokenizer
    tokenizer = load_tokenizer(tokenizer_path)

    # Construct prompt using apply_chat_template or fallback
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        logging.info(f"Using Prompt from apply_chat_template: {prompt}")
    else:
        # Fallback: render chat template using Jinja2
        chat_template = tokenizer_config.get("chat_template")
        if not chat_template:
            raise RuntimeError("No chat_template found in tokenizer_config.json and tokenizer has no apply_chat_template method.")
        template = jinja2.Template(chat_template)
        prompt = template.render(messages=messages, eos_token=eos_token, add_generation_prompt=True)
        logging.info(f"Using Prompt from Jinja2 chat_template: {prompt}")

    # Get EOS token ID for stopping generation AFTER tokenizer is loaded
    eos_token_id = tokenizer.eos_id()
    if eos_token_id is None:
        # Fallback if not directly available (might be in added_tokens_decoder)
        eos_token_id = tokenizer_config.get('eos_token_id', 2) # Default to 2 if not found
        logging.warning(f"Could not get eos_id directly from tokenizer, using ID: {eos_token_id}")

    # Load weights and model
    weights = load_safetensors_weights("data/model.safetensors")
    model = TinyLlama(config=config, weights=weights) # Revert to default float32
    # model = TinyLlama(config=config, weights=weights).to(dtype=torch.bfloat16) # bfloat16 made things worse
    logging.info(f"Model initialized with default dtype: {next(model.parameters()).dtype}") # Log actual parameter dtype

    # Tokenize prompt (prepend BOS token)
    # input_ids = tokenizer.encode(f"{bos_token}{prompt}")
    # input_ids_tensor = torch.tensor([input_ids], dtype=torch.long).to(model.embedding.weight.device)

    # Questions to test
    questions = [
        "What is the capital of France?",
        "Who wrote Hamlet?"
    ]

    # 3 best prompt variants, now as format strings
    prompt_variants = [
        ("Role-based (User/Assistant)", "User: {question}\nAssistant:"),
        ("Q: A: Style", "Q: {question}\nA:"),
        ("Classic Chatbot", "Human: {question}\nAI:")
    ]

    def extract_first_sentence_or_line(text):
        # Split into sentences using regex
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        for sentence in sentences:
            cleaned = sentence.strip()
            if cleaned:
                return cleaned
        # Fallback: split by newlines
        for line in text.splitlines():
            cleaned = line.strip()
            if cleaned:
                return cleaned
        return text.strip()

    # Run inference for each prompt variant and question
    for question in questions:
        print(f"\n==============================\nQuestion: {question}\n==============================")
        for prompt_name, prompt_template in prompt_variants:
            prompt = prompt_template.format(question=question)
            logging.info(f"\n===== Testing prompt variant: {prompt_name} =====\nPrompt: {prompt}")
            print(f"\n===== {prompt_name} =====\nPrompt: {prompt}")

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
                print(f"\nModel output for {prompt_name} (cleaned):\n{cleaned_text}\n")
            except Exception as e:
                logging.error(f"Error during decoding or logging decoded text: {e}", exc_info=True)
                print(f"Error during decoding for {prompt_name}: {e}")

    # Check if special chat tokens are recognized by the tokenizer
    for special in ["<|system|>", "<|user|>", "<|assistant|>"]:
        ids = tokenizer.encode(special)
        logging.info(f"Token IDs for {special}: {ids}")


if __name__ == "__main__":
    main() 