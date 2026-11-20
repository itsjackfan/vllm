# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Duo (s-sahoo/duo): uniform-state diffusion LM, whole canvas denoised by ancestral sampling.

Numerical oracle: src/duo.py in the parent repo (cs2680-diffusion-kernel); keep the math identical to it.
Engine mapping (the read-only path of diffusion_gemma.py):
- request = 1 BOS prompt token (prefill; output unused) + one canvas of canvas_length draft tokens, canvas[0] = BOS
- every engine step recomputes the whole canvas and rejects all drafts, until step == T (the noise-removal step,
  T + 1 forwards in total); then the canvas is emitted and the request ends (max_tokens = canvas_length)
- attention is exact bidirectional SDPA inside each request's canvas; the KV cache is never read (see kv_anchor)
Kernel hooks: DuoBlock.forward (attention), DuoSampler._denoise (softmax / posterior / categorical over V).

Dims: N tokens in the step (flattened over requests), R requests, n decoding requests, L canvas length, V vocab,
D hidden, H heads, K head dim, C cond dim, F sinusoid freqs.
"""

import math
from collections.abc import Iterable
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.attention import Attention
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.sample.output import SamplerOutput


def layer_norm(x_ND, weight_D):
    return F.layer_norm(x_ND.float(), x_ND.shape[-1:]) * weight_D.float()


def rotary(x_NHK, cos_NK, sin_NK):
    x1, x2 = x_NHK.float().chunk(2, -1)
    return x_NHK.float() * cos_NK[:, None] + torch.cat([-x2, x1], -1) * sin_NK[:, None]


class Norm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))


class DuoTimestepEmbedder(nn.Module):
    """fp32, as in the reference (it runs outside the bf16 autocast region)."""

    def __init__(self, cond_dim, n_freq=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(n_freq, cond_dim, dtype=torch.float32), nn.SiLU(),
                                 nn.Linear(cond_dim, cond_dim, dtype=torch.float32))
        self.n_freq = n_freq

    def forward(self, sigma_N):
        half = self.n_freq // 2
        freqs_F = torch.exp(-math.log(10000) * torch.arange(half, device=sigma_N.device) / half)
        args_NF = sigma_N[:, None].float() * freqs_F[None]
        return self.mlp(torch.cat([args_NF.cos(), args_NF.sin()], -1))


class DuoBlock(nn.Module):
    def __init__(self, dim, n_heads, cond_dim):
        super().__init__()
        self.n_heads = n_heads
        self.norm1, self.norm2 = Norm(dim), Norm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(approximate="tanh"), nn.Linear(4 * dim, dim))
        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim)

    def forward(self, x_ND, cos_NK, sin_NK, c_NC, seg_lens):
        """x_ND: fp32 residual stream (the reference adds bf16 branches onto an fp32 stream under autocast)."""
        dt = self.attn_qkv.weight.dtype
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaLN_modulation(c_NC.to(dt)).float().chunk(6, -1)
        h_ND = (layer_norm(x_ND, self.norm1.weight) * (1 + scale1) + shift1).to(dt)
        q, k, v = rearrange(self.attn_qkv(h_ND), "n (three h k) -> three n h k", three=3, h=self.n_heads)
        q, k = rotary(q, cos_NK, sin_NK).to(dt), rotary(k, cos_NK, sin_NK).to(dt)
        # NOTE: one SDPA per request canvas (exact, no cache). Baseline for a fused varlen kernel.
        a_NHK = torch.cat([
            F.scaled_dot_product_attention(qs.transpose(0, 1), ks.transpose(0, 1), vs.transpose(0, 1)).transpose(0, 1)
            for qs, ks, vs in zip(q.split(seg_lens), k.split(seg_lens), v.split(seg_lens))])
        x_ND = x_ND + gate1 * self.attn_out(rearrange(a_NHK, "n h k -> n (h k)")).float()
        return x_ND + gate2 * self.mlp((layer_norm(x_ND, self.norm2.weight) * (1 + scale2) + shift2).to(dt)).float()


class DuoFinalLayer(nn.Module):
    def __init__(self, dim, vocab_size, cond_dim):
        super().__init__()
        self.norm_final = Norm(dim)
        self.linear = nn.Linear(dim, vocab_size)
        self.adaLN_modulation = nn.Linear(cond_dim, 2 * dim)

    def pre_logits(self, x_ND, c_NC):
        dt = self.linear.weight.dtype
        shift, scale = self.adaLN_modulation(c_NC.to(dt)).float().chunk(2, -1)
        return (layer_norm(x_ND, self.norm_final.weight) * (1 + scale) + shift).to(dt)


class DuoEmbedding(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty(vocab_size, dim))


class DuoForDiffusion(nn.Module):
    """Parameter names = checkpoint names minus `backbone.`."""

    @staticmethod
    def get_model_state_cls():
        return DuoModelState

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        cfg = vllm_config.model_config.hf_config
        D, H, C, V = cfg.hidden_size, cfg.num_attention_heads, cfg.cond_dim, cfg.vocab_size
        self.config = cfg
        self.vocab_embed = DuoEmbedding(V, D)
        self.sigma_map = DuoTimestepEmbedder(C)
        self.blocks = nn.ModuleList(DuoBlock(D, H, C) for _ in range(cfg.num_hidden_layers))
        self.output_layer = DuoFinalLayer(D, V, C)
        K = D // H
        self.register_buffer("inv_freq", 1.0 / 10000 ** (torch.arange(0, K, 2, dtype=torch.float32) / K), False)
        # Never called: registers one (small) KV cache layer so the scheduler / runner KV path works unchanged.
        # Duo recomputes the full canvas every step and attends within it only (DuoBlock.forward).
        # Lives on block 0: vLLM requires exactly one layer index in every attention layer name.
        self.blocks[0].kv_anchor = Attention(H, K, K**-0.5, cache_config=vllm_config.cache_config,
                                             prefix="blocks.0.kv_anchor")

    def embed_input_ids(self, input_ids):
        return self.vocab_embed.embedding[input_ids]

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, sigma=None, seg_lens=None,
                **kwargs):
        x_ND = self.embed_input_ids(input_ids).float()
        c_NC = F.silu(self.sigma_map(sigma))
        freqs_NK = (positions.float()[:, None] * self.inv_freq).repeat(1, 2)  # relative => prompt offset is harmless
        cos_NK, sin_NK = freqs_NK.cos(), freqs_NK.sin()
        for block in self.blocks:
            x_ND = block(x_ND, cos_NK, sin_NK, c_NC, seg_lens)
        return self.output_layer.pre_logits(x_ND, c_NC)

    def compute_logits(self, hidden_states):
        return self.output_layer.linear(hidden_states).float()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params, loaded = dict(self.named_parameters()), set()
        for name, w in weights:
            name = name.removeprefix("backbone.")
            if name in params:  # skips the checkpoint's rotary_emb.inv_freq buffer
                params[name].data.copy_(w)
                loaded.add(name)
        return loaded


def alpha(t, noise_eps):
    """Log-linear schedule alpha_t = 1 - (1 - eps) t (src/duo.py:alpha)."""
    return 1 - (1 - noise_eps) * t


def posterior(p_x0_nLV, x_nL, alpha_s, alpha_t):
    """q(x_s | x_t, x0 ~ p_x0), identical to src/duo.py:posterior."""
    V = p_x0_nLV.shape[-1]
    alpha_ts, d_alpha = alpha_t / alpha_s, alpha_s - alpha_t
    xt_nLV = F.one_hot(x_nL, V).to(p_x0_nLV.dtype)
    num = (alpha_t * V * p_x0_nLV * xt_nLV + (alpha_ts - alpha_t) * xt_nLV + d_alpha * p_x0_nLV
           + (1 - alpha_ts) * (1 - alpha_s) / V)
    return num / (alpha_t * V * p_x0_nLV.gather(-1, x_nL[..., None]) + 1 - alpha_t)


class DuoRequestStates:
    """Slot-indexed GPU state. step k < T: ancestral step t_k -> t_{k+1} on linspace(1, t_eps, T + 1);
    step T: noise removal (alpha_s = 1), then emit."""

    def __init__(self, max_num_reqs, canvas_length, vocab_size, max_steps, bos_id, device):
        self.L, self.V, self.T, self.bos_id = canvas_length, vocab_size, max_steps, bos_id
        self.canvas = torch.zeros(max_num_reqs, canvas_length, dtype=torch.int64, device=device)
        self.step = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
        self.max_steps = torch.full((max_num_reqs,), max_steps, dtype=torch.int32, device=device)

    def reset(self, slot):
        self.canvas[slot] = torch.randint(0, self.V, (self.L,), device=self.canvas.device)
        self.canvas[slot, 0] = self.bos_id
        self.step[slot] = 0

    def times(self, slots, t_eps, noise_eps):
        """-> (sigma_t, alpha_s, alpha_t), fp64, per slot."""
        k, T = self.step[slots].double(), self.max_steps[slots].double()
        alpha_t = alpha(1 - (1 - t_eps) * k / T, noise_eps)
        alpha_s = torch.where(k >= T, torch.ones_like(alpha_t), alpha(1 - (1 - t_eps) * (k + 1) / T, noise_eps))
        return -alpha_t.log(), alpha_s, alpha_t


class DuoModelState(DefaultModelState):
    num_new_sampled_tokens_per_step: int = 0
    t_eps, noise_eps, bos_id = 1e-5, 1e-3, 50256  # = configs/model/duo.yaml in the parent repo

    def __init__(self, vllm_config, model, encoder_cache, device):
        super().__init__(vllm_config, model, encoder_cache, device)
        hf, dc = self.model_config.hf_config, vllm_config.diffusion_config
        self.states = DuoRequestStates(self.max_num_reqs, dc.canvas_length, hf.vocab_size,
                                       dc.max_denoising_steps or hf.max_denoising_steps, self.bos_id, device)
        self._sigma_buf = torch.zeros(self.max_num_tokens, dtype=torch.float32, device=device)

    def get_supported_generation_tasks(self):
        return ("generate",)

    def custom_sampler(self, sampler):
        return DuoSampler(sampler, self.states, self.t_eps, self.noise_eps), None

    def add_request(self, req_index, new_req_data):
        super().add_request(req_index, new_req_data)
        self.states.reset(req_index)

    def prepare_inputs(self, input_batch, req_states):
        R, N, N_pad = input_batch.num_reqs, input_batch.num_tokens, input_batch.num_tokens_after_padding
        seg_lens = np.diff(input_batch.query_start_loc_np[: R + 1]).tolist()
        sigma_R = self.states.times(input_batch.idx_mapping[:R], self.t_eps, self.noise_eps)[0].float()
        sigma = self._sigma_buf[:N_pad]
        sigma[:N] = sigma_R.repeat_interleave(torch.tensor(seg_lens, device=self.device), output_size=N)
        if N_pad > N:
            seg_lens.append(N_pad - N)
        return {"sigma": sigma, "seg_lens": seg_lens}


_NO_PENALTIES_STATE = SimpleNamespace(output_bin_counts=None)


class DuoSampler:
    """Replaces (Sampler, RejectionSampler). Duck-typed like diffusion_gemma.DiffusionSampler."""

    def __init__(self, sampler, states, t_eps, noise_eps):
        self.sampling_states = sampler.sampling_states
        self.logprob_token_ids_state = sampler.logprob_token_ids_state
        self.req_states = sampler.req_states
        self.states, self.t_eps, self.noise_eps = states, t_eps, noise_eps
        R, L, device = states.canvas.shape[0], states.L, states.canvas.device
        self._sampled = torch.zeros(R, L, dtype=torch.int32, device=device)
        self._num_sampled = torch.zeros(R, dtype=torch.int32, device=device)

    @property
    def penalties_state(self):
        return _NO_PENALTIES_STATE

    def add_request(self, req_idx, sampling_params):
        self.sampling_states.add_request(req_idx, sampling_params)
        self.logprob_token_ids_state.add_request(req_idx, sampling_params)
        T = (sampling_params.extra_args or {}).get("diffusion_max_steps", self.states.T)
        self.states.max_steps[req_idx] = int(T)

    def apply_staged_writes(self):
        self.sampling_states.apply_staged_writes()
        self.logprob_token_ids_state.apply_staged_writes()

    def _denoise(self, logits_NV, slots_n):
        """One ancestral step for n canvases -> (new canvas [n, L], emit [n])."""
        st = self.states
        n = slots_n.shape[0]
        sigma, alpha_s, alpha_t = st.times(slots_n, self.t_eps, self.noise_eps)
        p_x0_nLV = logits_NV.view(n, st.L, -1).double().log_softmax(-1).exp()
        q_nLV = posterior(p_x0_nLV, st.canvas[slots_n], alpha_s[:, None, None], alpha_t[:, None, None])
        x_nL = (q_nLV / (1e-10 - (torch.rand_like(q_nLV) + 1e-10).log())).argmax(-1)
        x_nL[:, 0] = st.bos_id
        return x_nL, st.step[slots_n] >= st.max_steps[slots_n]

    def __call__(self, logits, input_batch, draft_logits=None):
        st, R = self.states, input_batch.num_reqs
        device = logits.device
        all_slots = input_batch.idx_mapping[:R]
        sampled, num_sampled = self._sampled[:R].zero_(), self._num_sampled[:R].zero_()
        num_logits_np = np.diff(input_batch.cu_num_logits_np[: R + 1])
        dec_np = np.where(num_logits_np > 0)[0]
        if input_batch.num_draft_tokens > 0 and len(dec_np):
            slots_n = async_tensor_h2d(input_batch.idx_mapping_np[dec_np].astype(np.int64), device=device)
            idx_n = async_tensor_h2d(dec_np.astype(np.int64), device=device)
            x_nL, emit_n = self._denoise(logits, slots_n)
            st.canvas[slots_n] = x_nL
            st.step[slots_n] += 1
            sampled[idx_n] = x_nL.to(sampled.dtype) * emit_n[:, None]
            num_sampled[idx_n] = (emit_n * st.L).to(num_sampled.dtype)
        self.req_states.draft_tokens[all_slots, : st.L] = st.canvas[all_slots]
        # all drafts of a denoise step are rejected so num_computed_tokens stays and the canvas is recomputed
        qsl = input_batch.query_start_loc[: R + 1]
        num_logits = async_tensor_h2d(num_logits_np.astype(np.int32), device=device)
        num_rejected = torch.where((num_logits > 0) & (num_sampled == 0), qsl[1:] - qsl[:-1], num_logits - num_sampled)
        if input_batch.num_draft_tokens == 0:
            sampled, num_rejected = sampled[:, :1], num_sampled
        return SamplerOutput(sampled_token_ids=sampled, logprobs_tensors=None, num_nans=None,
                             num_sampled=num_sampled, num_rejected=num_rejected)
