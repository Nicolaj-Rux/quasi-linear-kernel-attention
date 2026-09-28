"""Distill the stock softmax nomic-embed-text-v1 into a kernel student: layer-by-layer (masked MSE on attention
outputs), then end-to-end (MSE on sentence embeddings); saves text_lbl_* and text_e2e_* checkpoints.

    python -m text_transformer.train --kernel add_bump
"""
import argparse
import time
import uuid
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader

from kernel_attention import KERNELS, TAU
from kernel_attention.nomic import use_kernel_attention
from paths import CKPT_DIR
from text_transformer import common

PREFIXES = ["classification: ", "clustering: ", "search_document: ", "search_query: "]

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--kernel", required=True, choices=[k for k in KERNELS if k != "softmax"])
p.add_argument("--tau", type=float, default=None, help="default: kernel_attention.TAU[kernel]")
p.add_argument("--ckpt_dir", default=str(CKPT_DIR))
p.add_argument("--seed", type=int, default=0)
p.add_argument("--batch_size", type=int, default=16)
p.add_argument("--max_length", type=int, default=512)
p.add_argument("--prefix_weights", type=float, nargs=4, default=[0.45, 0.1, 0.35, 0.1], help="for " + " ".join(PREFIXES))
p.add_argument("--lbl_steps", type=int, default=10_000, help="steps per layer")
p.add_argument("--lbl_lr", type=float, default=5e-4)
p.add_argument("--lbl_lr_min", type=float, default=5e-5)
p.add_argument("--e2e_steps", type=int, default=10_000)
p.add_argument("--e2e_lr", type=float, default=5e-5)
p.add_argument("--e2e_lr_min", type=float, default=1e-5)
p.add_argument("--e2e_warmup", type=float, default=0.1, help="fraction of e2e steps with linear warmup")
p.add_argument("--clip", type=float, default=2.0)
p.add_argument("--log_every", type=int, default=100)
args = p.parse_args()
args.tau = TAU[args.kernel] if args.tau is None else args.tau
name = f"{args.kernel}_tau{common.tau_str(args.tau)}"
ckpt_dir = Path(args.ckpt_dir)
if list(ckpt_dir.glob(f"text_e2e_{name}_*.pt")):
    print(f"text_e2e checkpoint for {name} exists in {ckpt_dir}, skipping")
    raise SystemExit   # exit code 0: already done counts as success
if list(ckpt_dir.glob(f"text_lbl_{name}_*.pt")):
    raise SystemExit(f"text_lbl checkpoint for {name} without text_e2e in {ckpt_dir} (crashed run?): delete it and rerun")

common.seed_everything(args.seed)
teacher = SentenceTransformer(common.MODEL, device="cuda").eval().requires_grad_(False)
student = use_kernel_attention(SentenceTransformer(common.MODEL, device="cuda"), args.kernel, args.tau)
student.eval().requires_grad_(False)
data = load_dataset("nomic-ai/nomic-embed-unsupervised-data", split="reddit_title_body", streaming=True)
loader = DataLoader(data, batch_size=args.batch_size, collate_fn=lambda batch: [ex["document"] for ex in batch])
prefix_rng = torch.Generator().manual_seed(args.seed)


def batches(steps):
    """`steps` tokenized batches from the start of the stream, each text with a sampled task prefix."""
    for _, texts in zip(range(steps), loader):
        ids = torch.multinomial(torch.tensor(args.prefix_weights), len(texts), replacement=True, generator=prefix_rng)
        yield teacher.tokenizer([PREFIXES[i] + t for i, t in zip(ids, texts)], padding=True, truncation=True,
                                max_length=args.max_length, return_tensors="pt").to("cuda")


def save(stage, parents):
    path = ckpt_dir / f"text_{stage}_{name}_{run_id}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(kernel=args.kernel, tau=args.tau, stage=stage, run_id=run_id, parent_run_ids=parents,
                    config=vars(args), state_dict=student.state_dict()), path)
    print(f"saved {path}", flush=True)


class StopForward(Exception):
    pass


captured = {}


def capture(key):
    def hook(module, inputs, output):
        captured[key] = output[0]
        raise StopForward   # nothing above this attention is needed for this layer's loss
    return hook


def forward_until_hook(model, features):
    try:
        model(features)
    except StopForward:
        pass


run_id = uuid.uuid4().hex[:8]
start, step = time.time(), 0

# stage 1: layer-by-layer
teacher_layers, student_layers = teacher[0].auto_model.layers, student[0].auto_model.layers
for layer in range(len(student_layers)):
    attn = student_layers[layer].self_attn.requires_grad_(True)
    opt = torch.optim.Adam(attn.parameters(), lr=args.lbl_lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.lbl_steps, eta_min=args.lbl_lr_min)
    hooks = [teacher_layers[layer].self_attn.register_forward_hook(capture("teacher")),
             attn.register_forward_hook(capture("student"))]
    for features in batches(args.lbl_steps):
        with torch.no_grad():
            forward_until_hook(teacher, features)
        forward_until_hook(student, features)
        mask = features["attention_mask"].unsqueeze(-1).to(captured["student"].dtype)
        loss = ((captured["student"] - captured["teacher"]) ** 2 * mask).sum() / (mask.sum() * captured["student"].shape[-1])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(attn.parameters(), args.clip)
        opt.step()
        sched.step()
        step += 1
        if step % args.log_every == 0:
            print(f"step {step}  lbl/mse {loss.item():.3e}  lbl/lr {sched.get_last_lr()[0]:.3e}  lbl/layer {layer}", flush=True)
    for hook in hooks:
        hook.remove()
    attn.requires_grad_(False)
    print(f"layer {layer}: mse {loss.item():.3e}  {(time.time() - start) / 60:.1f} min", flush=True)
save("lbl", [])

# stage 2: end-to-end
student.train().requires_grad_(True)
opt = torch.optim.Adam(student.parameters(), lr=args.e2e_lr)
warmup = int(args.e2e_warmup * args.e2e_steps)
sched = torch.optim.lr_scheduler.SequentialLR(opt, [
    torch.optim.lr_scheduler.LinearLR(opt, start_factor=1e-3, total_iters=warmup),
    torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.e2e_steps - warmup, eta_min=args.e2e_lr_min)], milestones=[warmup])
for features in batches(args.e2e_steps):
    with torch.no_grad():
        target = teacher(features)["sentence_embedding"]
    output = student(features)["sentence_embedding"]
    loss = 0.5 * F.mse_loss(output, target, reduction="sum") / output.shape[0]
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(student.parameters(), args.clip)
    opt.step()
    sched.step()
    step += 1
    if step % args.log_every == 0:
        print(f"step {step}  e2e/loss {loss.item():.4f}  e2e/lr {sched.get_last_lr()[0]:.3e}", flush=True)
print(f"e2e: loss {loss.item():.4f}  {(time.time() - start) / 60:.1f} min", flush=True)
save("e2e", [run_id])

peak = torch.cuda.max_memory_allocated() / 1e9
print(f"done {name}: {(time.time() - start) / 60:.1f} min, peak {peak:.2f} GB")
