"""Portable attention for one query and ordered memory blocks."""

import torch
from torch.nn import functional as F


def dense_attention(attention, q, k, v, *, causal):
    mask = None
    if causal and q.shape[2] != k.shape[2]:
        mask = torch.arange(k.shape[2], device=q.device)[None, :] <= (
            torch.arange(q.shape[2], device=q.device)[:, None] + k.shape[2] - q.shape[2]
        )
    return F.scaled_dot_product_attention(
        q,
        k.repeat_interleave(q.shape[1] // k.shape[1], dim=1),
        v.repeat_interleave(q.shape[1] // v.shape[1], dim=1),
        is_causal=causal and mask is None,
        attn_mask=mask,
        dropout_p=0.0,
        scale=float(attention.scaling),
    ).transpose(1, 2)


def varlen_attention(
    q,
    k,
    v,
    cu_q,
    cu_k,
    *,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
):
    # Match qwen's packed reference SDPA arithmetic for reproducible inference.
    qi = torch.arange(q.shape[0], device=q.device)
    ki = torch.arange(k.shape[0], device=k.device)
    qb = torch.bucketize(qi, cu_q[1:], right=True)
    kb = torch.bucketize(ki, cu_k[1:], right=True)
    mask = qb[:, None] == kb[None, :]
    if causal:
        qpos = qi - cu_q[qb]
        kpos = ki - cu_k[kb]
        offset = (cu_k[1:] - cu_k[:-1] - cu_q[1:] + cu_q[:-1])[qb]
        mask = mask & (kpos[None, :] <= (qpos + offset)[:, None])
    groups = q.shape[1] // k.shape[1]
    out = F.scaled_dot_product_attention(
        q.transpose(0, 1)[None],
        k.repeat_interleave(groups, dim=1).transpose(0, 1)[None],
        v.repeat_interleave(groups, dim=1).transpose(0, 1)[None],
        attn_mask=mask, dropout_p=dropout_p, scale=softmax_scale)
    return out[0].transpose(0, 1)
