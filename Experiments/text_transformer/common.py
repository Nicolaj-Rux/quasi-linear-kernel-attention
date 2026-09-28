"""Shared pieces of the text experiments: models, the long-context RoPE fix, checkpoints, nDCG, summary rows."""
import argparse
import csv
import datetime
import fcntl
import random
import socket
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from kernel_attention import KERNELS, TAU
from kernel_attention.nomic import use_kernel_attention
from paths import CKPT_DIR

MODEL = "nomic-ai/nomic-embed-text-v1"
TRAINED_CTX = 2048   # nomic trains at 2048 and reaches 8192 with Dynamic NTK

# everything fp32
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def tau_str(tau):
    return f"{tau:.4g}"


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def eval_parser(doc):
    p = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kernel", required=True, choices=list(KERNELS))
    p.add_argument("--tau", type=float, default=None, help="default: kernel_attention.TAU[kernel]")
    p.add_argument("--stage", default="e2e", choices=["lbl", "e2e"], help="checkpoint to evaluate (softmax: stock model)")
    p.add_argument("--ckpt_dir", default=str(CKPT_DIR), help="holds text_{stage}_{kernel}_tau{tau}_*.pt; not needed for softmax")
    p.add_argument("--results_dir", default=str(Path(__file__).with_name("results")))
    return p


def parse(p):
    args = p.parse_args()
    args.tau = TAU[args.kernel] if args.tau is None else args.tau
    if args.kernel == "softmax":
        args.stage = "stock"   # the teacher: stock weights, no checkpoint
    return args


def find_checkpoint(ckpt_dir, stage, kernel, tau):
    paths = sorted(Path(ckpt_dir).glob(f"text_{stage}_{kernel}_tau{tau_str(tau)}_*.pt"))
    if len(paths) != 1:
        raise SystemExit(f"expected exactly one text {stage} checkpoint for {kernel} tau {tau_str(tau)} "
                         f"in {ckpt_dir}, found {len(paths)}")
    print(f"loading {paths[0]}")
    return torch.load(paths[0], map_location="cuda")


def load_eval_model(args):
    """Nomic with kernel attention: stock weights for softmax (the teacher), else the trained checkpoint."""
    model = use_kernel_attention(SentenceTransformer(MODEL, device="cuda"), args.kernel, args.tau)
    run_id = "stock"
    if args.kernel != "softmax":
        ckpt = find_checkpoint(args.ckpt_dir, args.stage, args.kernel, args.tau)
        model.load_state_dict(ckpt["state_dict"])
        run_id = ckpt["run_id"]
    return model.eval(), run_id


def fix_long_context_rope(model):
    """Make Dynamic NTK fire above the trained 2048 (nomic ships max_position_embeddings=8192) and reset its
    cached base before every forward, so a long batch does not affect later short ones."""
    rot = model[0].auto_model.rotary_emb
    rot.config.max_position_embeddings = TRAINED_CTX
    rot.original_max_seq_len = TRAINED_CTX
    rot.max_seq_len_cached = TRAINED_CTX

    def reset(module, inputs):
        module.max_seq_len_cached = TRAINED_CTX
        module.register_buffer("inv_freq", module.original_inv_freq.to(inputs[0].device), persistent=False)

    rot.register_forward_pre_hook(reset)


def run_dir(args, benchmark, run_id):
    return Path(args.results_dir) / benchmark / f"{args.kernel}_tau{tau_str(args.tau)}_{run_id}"


def ndcg_at_10(sim, relevant):
    """Mean nDCG@10. sim: (queries, docs) similarities; relevant: per query the set of relevant doc indices."""
    k = min(10, sim.shape[1])
    top = np.argsort(-sim, axis=1)[:, :k]
    discount = 1.0 / np.log2(np.arange(k) + 2)
    return float(np.mean([(np.isin(row, list(rel)) * discount).sum() / discount[:min(k, len(rel))].sum()
                          for row, rel in zip(top, relevant)]))


def write_summary(args, benchmark, run_id, key, values):
    """Insert or replace this run's row (kernel, tau, stage, run id and `key` identify it) in {benchmark}_summary.csv."""
    row = dict(kernel=args.kernel, tau=args.tau, stage=args.stage, run_id=run_id, **key, **values,
               hostname=socket.gethostname(), gpu=torch.cuda.get_device_name(),
               date=datetime.datetime.now().isoformat(timespec="seconds"))
    ident = ["kernel", "tau", "stage", "run_id", *key]
    path = Path(args.results_dir) / f"{benchmark}_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)   # several evals may finish at the same time
        f.seek(0)
        rows = [r for r in csv.DictReader(f) if [r[k] for k in ident] != [str(row[k]) for k in ident]]
        f.seek(0)
        f.truncate()
        writer = csv.DictWriter(f, fieldnames=list(row), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows + [row])
    print(f"wrote {path}")
