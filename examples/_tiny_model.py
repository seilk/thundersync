"""The randomly initialized one-layer Qwen3 model both examples train."""

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from thundersync import STREAM_ATTENTION_NAME, register_stream_attention

# Token ids below this bound are valid inputs; the examples' prompts and
# turns are arbitrary ids in this range rather than tokenized text.
VOCAB_SIZE = 64


def tiny_qwen3(device: torch.device) -> torch.nn.Module:
    """A tiny Qwen3 decoder whose attention runs through the streaming executor."""

    register_stream_attention()
    config = AutoConfig.for_model(
        "qwen3", hidden_size=32, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        vocab_size=VOCAB_SIZE, max_position_embeddings=128, tie_word_embeddings=False,
    )
    return AutoModelForCausalLM.from_config(
        config, attn_implementation=STREAM_ATTENTION_NAME, dtype=torch.float32,
    ).to(device)
