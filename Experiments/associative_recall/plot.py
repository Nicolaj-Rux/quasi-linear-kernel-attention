"""Loss vs N at each kernel's default tau, best seed per (kernel, N) (the paper's figure).

    python -m associative_recall.plot [--kernels softmax gauss ...] [--results ...] [--out ...]

Writes {out}.pdf.
"""
import argparse
import collections
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from kernel_attention.kernels import KERNELS, TAU

HERE = Path(__file__).parent
COLOR = dict(softmax="#1f77b4", gauss="#ff7f0e", laplace="#2ca02c", riesz="#17becf",
             add_laplace="#8c564b", add_riesz="#000000", add_bump="#9467bd", tri="#d62728",
             elu="#e377c2", relu="#7f7f7f", dpfp="#bcbd22")
MARKER = dict(softmax="o", gauss="s", laplace="^", riesz="<", add_laplace="P", add_riesz=">",
              add_bump="D", tri="v", elu="X", relu="*", dpfp="d")

parser = argparse.ArgumentParser()
parser.add_argument("--kernels", nargs="+", default=list(KERNELS), choices=list(KERNELS))
parser.add_argument("--results", default=str(HERE / "results.csv"))
parser.add_argument("--out", default=str(HERE / "loss_vs_N_best"))
args = parser.parse_args()

losses = collections.defaultdict(list)
for row in csv.DictReader(open(args.results)):
    if float(row["tau"]) == TAU[row["kernel"]]:
        losses[row["kernel"], int(row["N"])].append(float(row["final_loss"]))

fig, ax = plt.subplots(figsize=(11.0, 3.6))
for kernel in args.kernels:
    sizes = sorted(n for k, n in losses if k == kernel)
    best = [np.min(losses[kernel, n]) for n in sizes]
    ax.plot(sizes, best, color=COLOR[kernel], marker=MARKER[kernel], markersize=4, linewidth=1.5,
            label=f"{kernel} ({TAU[kernel]:.3g})")
ax.set_xlabel("sequence length $N$")
ax.set_ylabel(r"$\mathcal{L}/N$")
ax.set_ylim(-0.05, 1.05)
ax.set_xlim(0, None)
ax.grid(alpha=0.3)
ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.02, 1.0))
fig.tight_layout()
fig.savefig(f"{args.out}.pdf")
print(f"wrote {args.out}.pdf")
