"""Public ops and backend dispatch: "auto", "keops" (small max(N,M)), "cuda", "cuda-fused", "cuda-global".

All backends require CUDA fp32 contiguous tensors; "auto" uses keops below KEOPS_CUTOFF, else cuda.
"""

import warnings

import torch

from weighted_abs_sum import _ext as _C

_BACKENDS = ("auto", "keops", "cuda", "cuda-fused", "cuda-global")
_PATH = {"cuda": -1, "cuda-fused": 1, "cuda-global": 0}

# "auto" uses keops strictly below this max(N,M)
KEOPS_CUTOFF = 512

_keops = {"checked": False, "mod": None}


def set_keops_cutoff(n):
    """Set the max(N,M) below which backend='auto' uses keops (0 disables)."""
    global KEOPS_CUTOFF
    KEOPS_CUTOFF = int(n)
    return KEOPS_CUTOFF


def _keops_module():
    if not _keops["checked"]:
        _keops["checked"] = True
        try:
            from weighted_abs_sum import keops_backend
            _keops["mod"] = keops_backend
        except Exception as e:  # noqa: BLE001 — any import/JIT-config failure disables keops
            warnings.warn(
                f"pykeops backend unavailable ({e!r}); backend='auto' will use "
                f"the CUDA extension everywhere. Check that CUDA_PATH/CUDA_HOME "
                f"point at a CUDA dir whose include/ has cuda.h and nvrtc.h."
            )
            _keops["mod"] = None
    return _keops["mod"]


def _keops_call(fn_name, args):
    """Run a keops op; on failure disable keops for this process and re-raise."""
    mod = _keops_module()
    try:
        return getattr(mod, fn_name)(*args)
    except Exception:
        _keops["mod"] = None
        warnings.warn("pykeops call failed; disabling the keops backend for this process")
        raise


def _resolve(backend, q, k, v):
    if backend not in _BACKENDS:
        raise ValueError(f"backend must be one of {_BACKENDS}, got {backend!r}")
    if backend != "auto":
        if backend == "keops" and _keops_module() is None:
            raise RuntimeError(
                "backend='keops' requested but pykeops is unavailable (see the "
                "warning above; CUDA_PATH/CUDA_HOME must expose cuda.h/nvrtc.h)"
            )
        return backend
    if not (q.is_cuda and q.dtype == torch.float32):
        return "cuda"   # let the extension raise its precise error message
    if max(q.shape[-1], k.shape[-1]) < KEOPS_CUTOFF and _keops_module() is not None:
        return "keops"
    return "cuda"


class _WeightedAbsSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, path):
        ctx.save_for_backward(q, k, v)
        ctx.set_materialize_grads(False)
        return _C.weighted_abs_sum_forward(q, k, v, path)

    @staticmethod
    def backward(ctx, g_w):
        q, k, v = ctx.saved_tensors
        grad_q, grad_k, grad_v = _C.weighted_abs_sum_backward(q, k, v, g_w.contiguous())
        return grad_q, grad_k, grad_v, None


class _WeightedAbsSumFull(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, path):
        ctx.save_for_backward(q, k, v)
        ctx.set_materialize_grads(False)
        return _C.weighted_abs_sum_full_forward(q, k, v, path)

    @staticmethod
    def backward(ctx, g_w, g_u):
        q, k, v = ctx.saved_tensors
        if g_u is None:     # out_u unused -> the cheaper plain backward
            grad_q, grad_k, grad_v = _C.weighted_abs_sum_backward(q, k, v, g_w.contiguous())
        else:
            if g_w is None:
                g_w = torch.zeros(q.shape[0], q.shape[2], v.shape[2],
                                  device=q.device, dtype=q.dtype)
            grad_q, grad_k, grad_v = _C.weighted_abs_sum_full_backward(
                q, k, v, g_w.contiguous(), g_u.contiguous())
        return grad_q, grad_k, grad_v, None


def weighted_abs_sum(q, k, v, backend="auto"):
    """out[p,m,c] = sum_{n,d} |k[p,d,n] - q[p,d,m]| * v[p,n,c]

    q: (P, D, M), k: (P, D, N), v: (P, N, C) -> (P, M, C). Differentiable.
    """
    b = _resolve(backend, q, k, v)
    if b == "keops":
        return _keops_call("weighted_abs_sum", (q, k, v))
    return _WeightedAbsSum.apply(q, k, v, _PATH[b])


def weighted_abs_sum_full(q, k, v, backend="auto"):
    """Returns (out_w, out_u): the weighted abs sum (P, M, C) and the
    unweighted abs sum out_u[p,m] = sum_{n,d} |k[p,d,n] - q[p,d,m]| (P, M),
    from one fused pass. Differentiable in both outputs.
    """
    b = _resolve(backend, q, k, v)
    if b == "keops":
        return _keops_call("weighted_abs_sum_full", (q, k, v))
    return _WeightedAbsSumFull.apply(q, k, v, _PATH[b])


def weighted_sgn_sum(q, k, v, g, backend="auto"):
    """out[p,d,m] = sum_n sgn(k[p,d,n] - q[p,d,m]) * dot(v[p,n,:], g[p,m,:])

    (P, D, M). Not differentiable through the cuda backend (it IS a gradient).
    """
    b = _resolve(backend, q, k, v)
    if b == "keops":
        return _keops_call("weighted_sgn_sum", (q, k, v, g))
    if b == "cuda-fused":
        raise ValueError("weighted_sgn_sum has no fused path; use 'cuda' or 'cuda-global'")
    return _C.weighted_sgn_sum_forward(q, k, v, g.contiguous())
