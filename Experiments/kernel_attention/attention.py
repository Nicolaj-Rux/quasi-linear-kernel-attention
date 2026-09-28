"""Kernel attention core: attention(q, k, v, kernel, tau, keep=None, impl) with q, k, v (B, H, S, D) fp32.

impl="naive" builds the full kernel matrix; impl="efficient" never materializes (P, D, S, S).
Without gradients, add_laplace runs on weighted_laplace_sum (forward only), else on the chunked compiled path.
"""

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint
from weighted_abs_sum import weighted_abs_sum_full
from weighted_laplace_sum import weighted_laplace_sum

from .kernels import EPS, KERNELS, dpfp_features

# many sequence lengths: compile each once instead of falling back to eager
torch._dynamo.config.cache_size_limit = 128
torch._dynamo.config.accumulated_cache_size_limit = 1024

CLAMP = 1e-6
CHUNK = 2 ** 28     # max entries of one query chunk's (P, width, chunk, S) intermediate
KEOPS_BELOW = 256   # keops is faster than cuda below this S

FEATURES = {
    "tri": lambda x: torch.cat([x.cos(), x.sin()], dim=1),
    "elu": lambda x: F.elu(x) + 1.0,
    "relu": torch.relu,
    "dpfp": dpfp_features,
}


def _normalized(kernel, q, k, v, tau, keep):
    """Full-matrix attention: q, k (P, D, S), v (P, S, D) -> (P, Sq, D)."""
    A = KERNELS[kernel](q, k, tau, keep)
    return A @ v / A.sum(dim=2, keepdim=True).clamp_min(CLAMP)


_normalized_compiled = torch.compile(_normalized, dynamic=True)


def _chunked(fn, kernel, q, k, v, tau, keep, width):
    # the compiled forward fuses the D-sum, but its backward materializes (P, D, chunk, S) (width = D):
    # when training, size chunks for that and checkpoint them so backward holds one chunk at a time
    train = torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad)
    c = max(1, CHUNK // (q.shape[0] * k.shape[2] * (width if train else 1)))
    starts = range(0, q.shape[2], c)
    if train and len(starts) > 1:
        return torch.cat([checkpoint(fn, kernel, q[:, :, i:i + c], k, v, tau, keep, use_reentrant=False)
                          for i in starts], dim=1)
    return torch.cat([fn(kernel, q[:, :, i:i + c], k, v, tau, keep) for i in starts], dim=1)


def _linear(phi, q, k, v, keep):
    fq, fk = phi(q), phi(k)                                     # (P, F, S)
    if keep is not None:
        fk = fk * keep[:, None, :]
    fq = fq.transpose(1, 2)                                     # (P, S, F)
    return fq @ (fk @ v) / (fq @ fk.sum(dim=2, keepdim=True)).clamp_min(CLAMP)


@torch.compiler.disable   # the extension/keops op cannot be traced; a graph break lets callers compile the rest
def _abs_sum(q, k, v):
    backend = "keops" if q.shape[2] < KEOPS_BELOW else "cuda"
    return weighted_abs_sum_full(q.contiguous(), k.contiguous(), v.contiguous(), backend=backend)


@torch.compiler.disable   # forward-only CUDA extension
def _add_laplace(q, k, v, n_pad):
    # a padded key (k=0, v=0) adds sum_d exp(-|q_d|) to the row sum
    out_w, out_u = weighted_laplace_sum(q.contiguous(), k.contiguous(), v.contiguous())
    den = out_u - n_pad * (-q.abs()).exp().sum(dim=1)
    return out_w / den.clamp_min(CLAMP)[:, :, None]


def _add_riesz(q, k, v, n_pad):
    # A[i,j] = |q_i|_1 + |k_j|_1 - |q_i - k_j|_1 + eps; a padded key (k=0, v=0) adds only eps to the row sum
    out_w, out_u = _abs_sum(q, k, v)
    aq, ak, S = q.abs().sum(dim=1), k.abs().sum(dim=1), q.shape[2]          # (P, S)
    num = (aq + EPS)[:, :, None] * v.sum(dim=1)[:, None, :] + (ak[:, :, None] * v).sum(dim=1)[:, None, :] - out_w
    den = (aq + EPS) * S + ak.sum(dim=1, keepdim=True) - out_u - EPS * n_pad
    return num / den.clamp_min(CLAMP)[:, :, None]


def _add_bump(q, k, v, n_pad):
    # max(0, 1 - |x|) = |x + 1| / 2 + |x - 1| / 2 - |x| with x = q_d - k_d;
    # a padded key (k=0, v=0) adds sum_d max(0, 1 - |q_d|) to the row sum
    w0, u0 = _abs_sum(q, k, v)
    wp, up = _abs_sum(q + 1, k, v)
    wm, um = _abs_sum(q - 1, k, v)
    num = (wp + wm) / 2 - w0
    den = (up + um) / 2 - u0 - n_pad * (1 - q.abs()).clamp_min(0).sum(dim=1)
    return num / den.clamp_min(CLAMP)[:, :, None]


def attention(q, k, v, kernel, tau, keep=None, impl="efficient"):
    assert q.dtype == k.dtype == v.dtype == torch.float32, "kernel attention runs in fp32 only"
    assert impl in ("naive", "efficient")
    B, H, S, D = q.shape
    if impl == "efficient" and kernel == "softmax":
        mask = None if keep is None else keep[:, None, None, :]
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=tau ** -2)

    P = B * H
    q = q.transpose(2, 3).reshape(P, D, S)
    k = k.transpose(2, 3).reshape(P, D, S)
    v = v.reshape(P, S, D)
    keep = None if keep is None else keep.repeat_interleave(H, dim=0)       # (P, S)

    # weighted_laplace_sum has no backward: add_laplace uses it only when no gradient is needed
    grad = torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad)
    if impl == "naive":
        out = _normalized(kernel, q, k, v, tau, keep)
    elif kernel in ("gauss", "laplace") or (kernel == "add_laplace" and grad):
        out = _chunked(_normalized_compiled, kernel, q, k, v, tau, keep, D)
    elif kernel == "riesz":
        out = _chunked(_normalized, kernel, q, k, v, tau, keep, 1)
    elif kernel in FEATURES:
        out = _linear(FEATURES[kernel], q / tau, k / tau, v, keep)
    else:
        q, k, n_pad = q / tau, k / tau, 0
        if keep is not None:
            k, v = k * keep[:, None, :], v * keep[:, :, None]
            n_pad = S - keep.sum(dim=1, keepdim=True)
        out = {"add_laplace": _add_laplace, "add_riesz": _add_riesz, "add_bump": _add_bump}[kernel](q, k, v, n_pad)
    return out.view(B, H, S, D)
