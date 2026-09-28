"""Runtime and peak memory of add_riesz attention (weighted_abs_sum backends, fp32) and add_laplace attention
(weighted_laplace_sum, fp32, forward only) vs PyTorch SDPA softmax backends.
Appends one row per measurement to --out, skipping rows already measured on this GPU; OOM or unsupported gives NaN.
add_laplace has no backward, so it is measured in fwd only.

    python -m speed_and_memory.bench [--sizes 128 1024 ...] [--out results.csv]
"""
import argparse
import csv
import datetime
import math
import socket
from pathlib import Path

import pykeops
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from weighted_abs_sum import weighted_abs_sum_full
from weighted_laplace_sum import weighted_laplace_sum

METHODS = [("add_laplace", "cuda-global", "fp32"),
           ("add_riesz", "cuda-global", "fp32"), ("add_riesz", "cuda-fused", "fp32"), ("add_riesz", "keops", "fp32"),
           ("softmax", "MEM_EFF", "fp32"), ("softmax", "MEM_EFF", "fp16"), ("softmax", "FLASH", "fp16"),
           ("softmax", "CUDNN", "fp16")]
SDPA = {"MEM_EFF": SDPBackend.EFFICIENT_ATTENTION, "FLASH": SDPBackend.FLASH_ATTENTION, "CUDNN": SDPBackend.CUDNN_ATTENTION}
DTYPES = {"fp32": torch.float32, "fp16": torch.float16}
FUSED_MAX = 4096   # the fused path is limited by shared memory
FORWARD_ONLY = {"add_laplace"}
LAPLACE_TAU = 0.5  # kernel_attention.TAU["add_laplace"]
KEY = ["method", "backend", "dtype", "B", "H", "D", "C", "N", "mode", "gpu"]

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--sizes", type=int, nargs="+", default=[2 ** i for i in range(7, 18)])
parser.add_argument("--runs", type=int, default=10)
parser.add_argument("--warmups", type=int, default=5)
parser.add_argument("--B", type=int, default=4)
parser.add_argument("--H", type=int, default=12)
parser.add_argument("--D", type=int, default=64)
parser.add_argument("--C", type=int, default=64)
parser.add_argument("--out", default=str(Path(__file__).with_name("results.csv")))
args = parser.parse_args()


def add_riesz_attention(q, k, v, backend):
    """add_riesz attention (eps = 0) through weighted_abs_sum_full with the given backend:
    s[m] = sum_n (|q_m|_1 + |k_n|_1 - |q_m - k_n|_1) v_n / sum_n (|q_m|_1 + |k_n|_1 - |q_m - k_n|_1)."""
    B, H, M, D = q.shape
    N, C = v.shape[2], v.shape[3]
    q = q.reshape(B * H, M, D).transpose(1, 2).contiguous()          # (P, D, M)
    k = k.reshape(B * H, N, D).transpose(1, 2).contiguous()          # (P, D, N)
    v = v.reshape(B * H, N, C).contiguous()                          # (P, N, C)
    q_l1, k_l1 = q.abs().sum(dim=1), k.abs().sum(dim=1)              # (P, M), (P, N)
    out_w, out_u = weighted_abs_sum_full(q, k, v, backend=backend)   # (P, M, C), (P, M)
    numer = (k_l1[:, :, None] * v).sum(dim=1)[:, None, :] + q_l1[:, :, None] * v.sum(dim=1)[:, None, :] - out_w
    denom = k_l1.sum(dim=1, keepdim=True) + q_l1 * N - out_u
    return (numer / denom.clamp(min=1e-6)[:, :, None]).reshape(B, H, M, C)


def add_laplace_attention(q, k, v):
    """s[m] = sum_n A_mn v_n / sum_n A_mn with A_mn = sum_d exp(-|q_md - k_nd| / tau), tau = 0.5."""
    B, H, M, D = q.shape
    N, C = v.shape[2], v.shape[3]
    q = (q / LAPLACE_TAU).reshape(B * H, M, D).transpose(1, 2).contiguous()     # (P, D, M)
    k = (k / LAPLACE_TAU).reshape(B * H, N, D).transpose(1, 2).contiguous()     # (P, D, N)
    v = v.reshape(B * H, N, C).contiguous()                                    # (P, N, C)
    out_w, out_u = weighted_laplace_sum(q, k, v)                               # (P, M, C), (P, M)
    return (out_w / out_u.clamp(min=1e-6)[:, :, None]).reshape(B, H, M, C)


def attention_fn(method, backend):
    if method == "add_riesz":
        return lambda q, k, v: add_riesz_attention(q, k, v, backend)
    if method == "add_laplace":
        return add_laplace_attention

    def sdpa(q, k, v):
        with sdpa_kernel(SDPA[backend]):   # only this backend: raises instead of falling back
            return F.scaled_dot_product_attention(q, k, v)
    return sdpa


def measure(attention, dtype, N, backward):
    """(mean ms, std ms, peak MB); peak = inputs and output (x2 with gradients) plus the first call's overhead."""
    shape = (args.B, args.H, N)
    q = torch.randn(*shape, args.D, dtype=dtype, device="cuda", requires_grad=backward)
    k = torch.randn(*shape, args.D, dtype=dtype, device="cuda", requires_grad=backward)
    v = torch.randn(*shape, args.C, dtype=dtype, device="cuda", requires_grad=backward)
    grad_out = torch.randn(*shape, args.C, dtype=dtype, device="cuda")

    def step():
        q.grad = k.grad = v.grad = None
        out = attention(q, k, v)
        if backward:
            out.backward(grad_out)
        return out

    io = (2 if backward else 1) * sum(t.numel() * t.element_size() for t in (q, k, v, grad_out))
    try:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        out = step()
        torch.cuda.synchronize()
        peak_mb = (torch.cuda.max_memory_allocated() - torch.cuda.memory_allocated() + io) / 2 ** 20
        del out
        for _ in range(args.warmups):
            step()
        times = []
        for _ in range(args.runs):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start.record()
            step()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        times = torch.tensor(times)
        return times.mean().item(), times.std().item(), peak_mb
    except (torch.OutOfMemoryError, RuntimeError) as e:   # RuntimeError: SDPA backend unsupported for this input/GPU
        print(f"    NaN: {type(e).__name__}: {str(e).splitlines()[0][:100]}")
        return math.nan, math.nan, math.nan


gpu = torch.cuda.get_device_name()
versions = dict(torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                pykeops=pykeops.__version__)
out_path = Path(args.out)
done = set()
if out_path.exists():
    done = {tuple(r[k] for k in KEY) for r in csv.DictReader(open(out_path))}

# sizes outermost: after a keops failure (OOM) weighted_abs_sum disables keops for the whole process, so every
# later keops call must be one that would fail anyway (larger N; keops memory grows ~linearly in N)
for N in args.sizes:
    for mode in ("fwd", "fwd+bwd"):
        for method, backend, dtype in METHODS:
            if mode == "fwd+bwd" and method in FORWARD_ONLY:
                continue   # no backward implemented
            row = dict(method=method, backend=backend, dtype=dtype, B=args.B, H=args.H, D=args.D, C=args.C, N=N,
                       mode=mode, gpu=gpu)
            if tuple(str(row[k]) for k in KEY) in done:
                continue
            if backend == "cuda-fused" and N > FUSED_MAX:
                mean, std, peak = math.nan, math.nan, math.nan
            else:
                mean, std, peak = measure(attention_fn(method, backend), DTYPES[dtype], N, mode == "fwd+bwd")
            torch.cuda.empty_cache()
            row.update(time_mean_ms=round(mean, 4), time_std_ms=round(std, 4), peak_mem_mb=round(peak, 1),
                       runs=args.runs, warmups=args.warmups, hostname=socket.gethostname(), **versions,
                       date=datetime.datetime.now().isoformat(timespec="seconds"))
            new = not out_path.exists()
            with open(out_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row))
                if new:
                    writer.writeheader()
                writer.writerow(row)
            print(f"{mode:<8} N={N:<6} {method:<7} {backend:<11} {dtype}  {mean:10.3f} ± {std:7.3f} ms  {peak:9.1f} MB",
                  flush=True)
