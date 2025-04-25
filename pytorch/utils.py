import torch
from safetensors.torch import load_file
import sentencepiece as spm
import json

def load_safetensors_weights(weights_path):
    """Load safetensors weights from the given path."""
    weights = load_file(weights_path)
    print(f"Loaded weights: {list(weights.keys())}")
    return weights

def load_tokenizer(tokenizer_path):
    """Load a sentencepiece tokenizer from the given path."""
    sp = spm.SentencePieceProcessor()
    sp.load(tokenizer_path)
    return sp

def load_config(config_path):
    """Load model config from a JSON file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config 