"""LoCo nDCG@10 against GPU time per document at 8192 and 16384 tokens (the paper's figure).

    python -m text_transformer.visualize.plot_pareto

Reads results/pareto_summary.csv (rows of text_transformer.pareto), writes visualize/pareto_all.pdf.
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).parent
SEQ_LENS = [8192, 16384]
BATCH_SIZE = 16
KERNELS = ["softmax", "add_laplace", "add_riesz", "add_bump", "tri", "elu", "dpfp"]
COLOR = dict(softmax="#1f77b4", add_laplace="#8c564b", add_riesz="#000000", add_bump="#9467bd", tri="#d62728",
             elu="#e377c2", dpfp="#bcbd22")
MARKER = dict(softmax="o", add_laplace="P", add_riesz=">", add_bump="D", tri="v", elu="X", dpfp="d")

rows = [r for r in csv.DictReader(open(HERE.parent / "results" / "pareto_summary.csv"))
        if int(r["batch_size"]) == BATCH_SIZE]

fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.2))
for ax, seq_len in zip(axes, SEQ_LENS):
    points = {r["kernel"]: (float(r["ms_per_doc"]), float(r["avg"])) for r in rows if int(r["seq_len"]) == seq_len}
    for kernel in KERNELS:
        if kernel in points:
            ax.scatter(*points[kernel], color=COLOR[kernel], marker=MARKER[kernel], s=45, zorder=3)
    ax.set_title(f"{seq_len} tokens")
    ax.set_xlabel("GPU time / doc [ms]")
    ax.set_ylabel("LoCo nDCG@10")
    ax.set_ylim(0.775, 0.895)
    ax.grid(alpha=0.3)

present = {r["kernel"] for r in rows}
handles = [Line2D([], [], color=COLOR[k], marker=MARKER[k], linestyle="", markersize=7, label=k)
           for k in KERNELS if k in present]
axes[-1].legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
fig.tight_layout()
fig.savefig(HERE / "pareto_all.pdf")
print(f"wrote {HERE / 'pareto_all.pdf'}")
