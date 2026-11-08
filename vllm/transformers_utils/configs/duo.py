# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from transformers import PreTrainedConfig


class DuoConfig(PreTrainedConfig):
    """s-sahoo/duo config.json, plus the standard names vLLM reads. A request is one BOS prompt token followed by
    one canvas of model_length tokens, hence max_position_embeddings = model_length + 1."""

    model_type = "DUO"

    def __init__(self, vocab_size=50258, hidden_dim=768, n_heads=12, n_blocks=12, cond_dim=128, model_length=1024,
                 max_denoising_steps=1024, **kwargs):
        self.vocab_size, self.hidden_dim, self.n_heads, self.n_blocks = vocab_size, hidden_dim, n_heads, n_blocks
        self.cond_dim, self.model_length, self.max_denoising_steps = cond_dim, model_length, max_denoising_steps
        self.hidden_size, self.num_hidden_layers = hidden_dim, n_blocks
        self.num_attention_heads = self.num_key_value_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.max_position_embeddings = model_length + 1
        self.canvas_length = model_length  # marks a diffusion model (ModelConfig.is_diffusion)
        super().__init__(**kwargs)
