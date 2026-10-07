"""Performance patches for the local CLEF-Flash/MPS pipeline:

1. Metal row-substitution kernel for the gated-delta chunk scan (replaces the
   63-step sequential loop with one dispatch per layer).
2. torch_compilable_check no-op (removes two device syncs per forward; the
   checks are pure validation).
3. JointSchemaHead receives its attention mask on CPU (removes the end-of-forward
   device sync from the sequence-length read).
"""

import sys
from pathlib import Path

import torch

_release = Path("../cloudflare-clef-flash")
for p in (str(_release / "runtime"), str(_release)):
    if p not in sys.path:
        sys.path.insert(0, p)

from metal_chunk import row_substitution_inplace


def _patched_chunk_gated_delta_rule(
    query, key, value, g, beta, chunk_size=64, initial_state=None,
    output_final_state=False, use_qk_l2norm_in_kernel=False, **kwargs,
):
    import torch.nn.functional as F

    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = query * torch.rsqrt((query * query).sum(dim=-1, keepdim=True) + 1e-6)
        key = key * torch.rsqrt((key * key).sum(dim=-1, keepdim=True) + 1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]
    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    num_matrices = attn.reshape(-1, chunk_size, chunk_size).shape[0]
    row_substitution_inplace(attn, num_matrices)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def install():
    import transformers.models.qwen3_5.modeling_qwen3_5 as q

    q.torch_compilable_check = lambda *args, **kwargs: None
    q.torch_chunk_gated_delta_rule = _patched_chunk_gated_delta_rule


def patch_head_cpu_mask(model):
    """Route the head's attention-mask read through CPU (batch has no padding)."""
    head = model.head
    original = type(head).forward

    def forward(self, hidden_states, input_ids, attention_mask, records, output_embedding_weight):
        if attention_mask.device.type != "cpu":
            attention_mask = attention_mask.detach().to("cpu")
        return original(self, hidden_states, input_ids, attention_mask, records, output_embedding_weight)

    head.forward = forward.__get__(head)
    return model
