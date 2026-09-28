"""PyKeOps backend: out[p,m,c] = sum_{n,d} |k[p,d,n] - q[p,d,m]| * v[p,n,c] as an online map-reduce.

JIT compilation needs cuda.h / nvrtc.h under CUDA_PATH (or CUDA_HOME)/include.
"""

import torch
from pykeops.torch import LazyTensor


def _lazy_qk(q: torch.Tensor, k: torch.Tensor):
    # KeOps wants the reduction axes (M=i, N=j) at -3 / -2 and the feature vector at -1
    q_pmd = q.transpose(1, 2).contiguous()          # (P, M, D)
    k_pnd = k.transpose(1, 2).contiguous()          # (P, N, D)
    x_i = LazyTensor(q_pmd[:, :, None, :])          # (P, M, 1, D)   i-variable
    y_j = LazyTensor(k_pnd[:, None, :, :])          # (P, 1, N, D)   j-variable
    return x_i, y_j


def weighted_abs_sum(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """out[p,m,c] = sum_{n,d} |k[p,d,n] - q[p,d,m]| * v[p,n,c]"""
    x_i, y_j = _lazy_qk(q, k)
    v_j = LazyTensor(v.contiguous()[:, None, :, :])     # (P, 1, N, C)   j-variable

    a_ij = (x_i - y_j).abs().sum(-1)                    # L1 over D -> inner dim 1
    return (a_ij * v_j).sum(dim=2)                      # weighted sum over j=N -> (P, M, C)


def weighted_abs_sum_full(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """(out_w, out_u): the unweighted sum is a constant ones-channel appended
    to v, so both come out of ONE KeOps reduction (shared distance pass)."""
    C = v.shape[2]
    ones = torch.ones(v.shape[0], v.shape[1], 1, device=v.device, dtype=v.dtype)
    out = weighted_abs_sum(q, k, torch.cat([v, ones], dim=2))
    return out[:, :, :C].contiguous(), out[:, :, C].contiguous()


def weighted_sgn_sum(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """out[p,d,m] = sum_n sgn(k[p,d,n] - q[p,d,m]) * dot(v[p,n,:], g[p,m,:])

    KeOps sign(0) == 0, the CUDA walk uses sign(0) == -1 (ties only).
    """
    x_i, y_j = _lazy_qk(q, k)
    v_j = LazyTensor(v.contiguous()[:, None, :, :])     # (P, 1, N, C)
    g_i = LazyTensor(g.contiguous()[:, :, None, :])     # (P, M, 1, C)

    b_ij = (v_j * g_i).sum(-1)                          # dot over C -> scalar
    out = ((y_j - x_i).sign() * b_ij).sum(dim=2)        # (P, M, D)
    return out.transpose(1, 2).contiguous()             # (P, D, M)
