"""Naive reference kernels K(q, k, tau=1.0, mask=None) -> A: q, k (P, D, S), A (P, Sq, Sk), reduction over D.

Each returns K(q/tau, k/tau); `mask` is a (P, Sk) key-keep tensor, masked columns are zero.
"""

import torch
import torch.nn.functional as F

EPS = 1e-3        # Riesz offset, keeps the kernel strictly positive
DPFP_NU = 3       # number of DPFP shifts (nu in {1, ..., D-1})

# D^(1/4) for head dim 64 makes softmax exactly scaled dot-product attention
TAU = {
    "softmax": 64 ** 0.25,
    "gauss": 64 ** 0.25,
    "laplace": 6.0,
    "riesz": 1.0,
    "add_laplace": 0.5,
    "add_riesz": 1.0,
    "add_bump": 1.5,
    "tri": 64 ** 0.25,
    "elu": 1.0,
    "relu": 1.0,
    "dpfp": 1.0,
}


def _mask(A, mask):
    return A if mask is None else A * mask[:, None, :]


def softmax(q, k, tau=1.0, mask=None):
    """exp(<q, k>)"""
    q, k = q / tau, k / tau
    logits = torch.einsum("pdi,pdj->pij", q, k)
    if mask is not None:
        logits = logits.masked_fill(~mask[:, None, :], float("-inf"))
    # subtract the per-query max before exp (cancels in normalization, avoids overflow)
    return (logits - logits.amax(dim=2, keepdim=True)).exp()


def riesz(q, k, tau=1.0, mask=None):
    """||q||_2 + ||k||_2 - ||q - k||_2 + eps"""
    q, k = q / tau, k / tau
    qq = (q * q).sum(dim=1)                                     # (P, Sq) = ||q||^2
    kk = (k * k).sum(dim=1)                                     # (P, Sk) = ||k||^2
    gram = torch.einsum("pdi,pdj->pij", q, k)                  # (P, Sq, Sk) = <q_i, k_j>
    # avoids the (P, D, Sq, Sk) intermediate; clamp keeps sqrt finite when q ~= k
    dist = (qq[:, :, None] + kk[:, None, :] - 2 * gram).clamp(min=1e-6).sqrt()
    A = qq.sqrt()[:, :, None] + kk.sqrt()[:, None, :] - dist + EPS
    return _mask(A, mask)


def laplace(q, k, tau=1.0, mask=None):
    """exp(-||q - k||_1)"""
    q, k = q / tau, k / tau
    dist = (q.unsqueeze(3) - k.unsqueeze(2)).abs().sum(dim=1)   # (P, Sq, Sk)
    return _mask((-dist).exp(), mask)


def gauss(q, k, tau=1.0, mask=None):
    """exp(-0.5 ||q - k||_2^2)"""
    q, k = q / tau, k / tau
    dist2 = (q.unsqueeze(3) - k.unsqueeze(2)).pow(2).sum(dim=1)  # (P, Sq, Sk)
    return _mask((-0.5 * dist2).exp(), mask)


def add_riesz(q, k, tau=1.0, mask=None):
    """sum_d |q_d| + |k_d| - |q_d - k_d| + eps"""
    q, k = q / tau, k / tau
    aq = q.abs().sum(dim=1)
    ak = k.abs().sum(dim=1)
    d1 = (q.unsqueeze(3) - k.unsqueeze(2)).abs().sum(dim=1)
    A = aq[:, :, None] + ak[:, None, :] - d1 + EPS
    return _mask(A, mask)


def add_laplace(q, k, tau=1.0, mask=None):
    """sum_d exp(-|q_d - k_d|)"""
    q, k = q / tau, k / tau
    A = (-(q.unsqueeze(3) - k.unsqueeze(2)).abs()).exp().sum(dim=1)
    return _mask(A, mask)


def add_bump(q, k, tau=1.0, mask=None):
    """sum_d max(0, 1 - |q_d - k_d|)"""
    q, k = q / tau, k / tau
    A = (1.0 - (q.unsqueeze(3) - k.unsqueeze(2)).abs()).clamp(min=0.0).sum(dim=1)
    return _mask(A, mask)


def tri(q, k, tau=1.0, mask=None):
    """sum_d cos(q_d - k_d)  (RFA / Peng); sign-indefinite"""
    q, k = q / tau, k / tau
    # cos(q_d - k_d) = cos q_d cos k_d + sin q_d sin k_d, so the D-sum is two matmuls
    A = torch.einsum("pdi,pdj->pij", q.cos(), k.cos()) + torch.einsum("pdi,pdj->pij", q.sin(), k.sin())
    return _mask(A, mask)


def elu(q, k, tau=1.0, mask=None):
    """phi(q) . phi(k) with phi = elu(.) + 1  (Katharopoulos)"""
    q, k = q / tau, k / tau
    A = torch.einsum("pdi,pdj->pij", F.elu(q) + 1.0, F.elu(k) + 1.0)
    return _mask(A, mask)


def relu(q, k, tau=1.0, mask=None):
    """phi(q) . phi(k) with phi = relu  (Choromanski)"""
    q, k = q / tau, k / tau
    A = torch.einsum("pdi,pdj->pij", q.relu(), k.relu())
    return _mask(A, mask)


def dpfp_features(x):
    """DPFP feature map phi: (P, D, S) -> (P, 2D*nu, S)  (Schlag)."""
    r = torch.cat([x, -x], dim=1).relu()                               # (P, 2D, S)
    parts = [r * r.roll(-j, dims=1) for j in range(1, DPFP_NU + 1)]    # cyclic shifts
    return torch.cat(parts, dim=1)


def dpfp(q, k, tau=1.0, mask=None):
    """phi(q) . phi(k) with the DPFP feature map."""
    q, k = q / tau, k / tau
    A = torch.einsum("pfi,pfj->pij", dpfp_features(q), dpfp_features(k))
    return _mask(A, mask)


KERNELS = {
    "softmax": softmax,
    "gauss": gauss,
    "laplace": laplace,
    "riesz": riesz,
    "add_laplace": add_laplace,
    "add_riesz": add_riesz,
    "add_bump": add_bump,
    "tri": tri,
    "elu": elu,
    "relu": relu,
    "dpfp": dpfp,
}
