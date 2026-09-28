"""Nomic-embed-text-v1 attention with a pluggable kernel (same parameters as NomicBertAttention)."""
import torch
from transformers.models.nomic_bert.modeling_nomic_bert import NomicBertAttention, apply_rotary_pos_emb

from .attention import attention
from .kernels import TAU


class KernelAttention(NomicBertAttention):
    def __init__(self, config, kernel="softmax", tau=None, impl="efficient"):
        super().__init__(config)
        self.kernel = kernel
        self.tau = TAU[kernel] if tau is None else tau
        self.impl = impl

    def forward(self, hidden_states, attention_mask=None, position_embeddings=None, **kwargs):
        B, S, _ = hidden_states.shape
        shape = (B, S, -1, self.head_dim)
        q = self.q_proj(hidden_states).view(shape).transpose(1, 2)            # (B, H, S, D)
        k = self.k_proj(hidden_states).view(shape).transpose(1, 2)
        v = self.v_proj(hidden_states).view(shape).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, *position_embeddings)

        keep = None
        if attention_mask is not None:
            # sdpa gives a boolean mask whose rows all hold the key-keep pattern; an eager float mask is additive
            assert attention_mask.dtype == torch.bool, "expected a boolean attention mask"
            keep = attention_mask[:, 0, 0, :]
        out = attention(q, k, v, self.kernel, self.tau, keep, self.impl)
        return self.o_proj(out.transpose(1, 2).reshape(B, S, -1)), None


def use_kernel_attention(model, kernel, tau=None, impl="efficient"):
    """Swap every attention layer of a nomic SentenceTransformer in place, keeping its weights."""
    auto = model[0].auto_model
    for layer in auto.layers:
        new = KernelAttention(auto.config, kernel, tau, impl).to(layer.self_attn.q_proj.weight.device)
        new.load_state_dict(layer.self_attn.state_dict())
        layer.self_attn = new
    return model
