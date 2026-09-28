"""Associative recall (Schlag et al., Sec. 6.1): learn E, W_Q, W_K so that A = K(W_Q E, W_K E) normalized equals I_N.

    python -m associative_recall.train --kernel add_bump --N 600 --seed 0
"""
import argparse
import csv
import datetime
import random
import socket
import time
import uuid
from pathlib import Path

import numpy as np
import torch

from kernel_attention.kernels import KERNELS, TAU

CLAMP = 1e-6

parser = argparse.ArgumentParser()
parser.add_argument("--kernel", default="add_bump", choices=list(KERNELS))
parser.add_argument("--tau", type=float, default=None, help="default: the kernel's entry in kernel_attention.TAU")
parser.add_argument("--N", type=int, default=600)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--D", type=int, default=64)
parser.add_argument("--lr", type=float, default=0.03)
parser.add_argument("--eps_E", type=float, default=1e-4, help="Adam eps for E")
parser.add_argument("--eps_W", type=float, default=1e-3, help="Adam eps for W_Q, W_K")
parser.add_argument("--clip", type=float, default=2.0, help="grad-norm clipping")
parser.add_argument("--steps", type=int, default=120_000)
parser.add_argument("--stop_loss", type=float, default=1e-4)
parser.add_argument("--patience", type=int, default=30_000, help="stop after this many steps without improvement")
parser.add_argument("--patience_rel", type=float, default=0.01, help="relative loss drop that counts as improvement")
parser.add_argument("--eval_draws", type=int, default=20)
parser.add_argument("--results", default=str(Path(__file__).with_name("results.csv")))
args = parser.parse_args()
tau = TAU[args.kernel] if args.tau is None else args.tau
N = args.N


def attention(E, WQ, WK):
    """Attention and Gram matrix, both (N, N), rows indexed by the query."""
    Q = (E @ WQ.T).T[None]
    K = (E @ WK.T).T[None]
    G = KERNELS[args.kernel](Q, K, tau)[0]
    return G / G.sum(-1, keepdim=True).clamp(min=CLAMP), G


@torch.compile
def training_loss(E, WQ, WK, I):
    A, _ = attention(E, WQ, WK)
    return ((A - I) ** 2).sum() / N


@torch.no_grad()
def evaluate(E, WQ, WK, I):
    """Re-measure on random key/value permutations; eval_deviation must be float noise (no positional encoding)."""
    A, G = attention(E, WQ, WK)
    losses, accuracies = [], []
    for _ in range(args.eval_draws):
        sigma = torch.randperm(N, device=I.device)
        xi = torch.randperm(N, device=I.device)
        vhat = A[:, sigma] @ I[xi]
        target = xi[torch.argsort(sigma)]
        losses.append((((I[target] - vhat) ** 2).sum() / N).item())
        accuracies.append((vhat.argmax(-1) == target).float().mean().item())

    loss = (((A - I) ** 2).sum() / N).item()
    return dict(final_loss=loss,
                eval_loss=sum(losses) / len(losses),
                eval_accuracy=sum(accuracies) / len(accuracies),
                eval_deviation=max(abs(v - loss) for v in losses) / max(abs(loss), 1e-30),
                gram_rank=torch.linalg.matrix_rank(G, rtol=1e-5).item(),
                attn_rank=torch.linalg.matrix_rank(A, rtol=1e-5).item(),
                max_offdiag=(A - I).abs().max().item(),
                n_clamped_rows=(G.sum(-1) < CLAMP).sum().item())


random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
device = torch.device("cuda")
I = torch.eye(N, device=device)
E = torch.randn(N, args.D, device=device).requires_grad_()
WQ = torch.eye(args.D, device=device).requires_grad_()
WK = torch.eye(args.D, device=device).requires_grad_()

# Adam's default eps 1e-8 steps by ~lr even for tiny gradients and walks the additive kernels off their optimum
opt = torch.optim.Adam([{"params": [E], "eps": args.eps_E},
                        {"params": [WQ, WK], "eps": args.eps_W}], lr=args.lr)

best = mark = float("inf")
stalled, reason, used = 0, "budget", args.steps
start = time.time()
for step in range(args.steps):
    loss = training_loss(E, WQ, WK, I)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_([E, WQ, WK], args.clip)
    opt.step()

    value = loss.item()
    mark, stalled = (value, 0) if value < mark * (1 - args.patience_rel) else (mark, stalled + 1)
    best = min(best, value)
    if value < args.stop_loss:
        reason, used = "target", step + 1
        break
    if stalled >= args.patience:
        reason, used = "patience", step + 1
        break

row = dict(kernel=args.kernel, tau=tau, N=N, seed=args.seed, D=args.D, lr=args.lr, clip=args.clip,
           steps_used=used, stop_reason=reason, wall_s=round(time.time() - start, 1),
           **evaluate(E.detach(), WQ.detach(), WK.detach(), I))
row["best_loss"] = min(best, row["final_loss"])
row.update(stage="recall", run_id=uuid.uuid4().hex[:8], hostname=socket.gethostname(),
           gpu=torch.cuda.get_device_name(), date=datetime.datetime.now().isoformat(timespec="seconds"))

with open(args.results, "a", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(row))
    if handle.tell() == 0:
        writer.writeheader()
    writer.writerow(row)
print(" ".join(f"{k}={v}" for k, v in row.items()), flush=True)
