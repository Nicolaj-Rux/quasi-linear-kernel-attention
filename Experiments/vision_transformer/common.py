"""Shared pieces of the vision stages: arguments, data, model, evaluation, checkpoints, results."""
import argparse
import csv
import datetime
import random
import socket
import uuid
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

from kernel_attention import KERNELS, TAU
from kernel_attention.vision_transformer import VisionTransformer
from paths import CKPT_DIR, DATA_DIR

MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)

# everything fp32: cuDNN would otherwise run the patch convolution in TF32 on Ampere
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False


def parser(doc, teacher=False):
    p = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kernel", required=True, choices=list(KERNELS))
    p.add_argument("--tau", type=float, default=None, help="default: kernel_attention.TAU[kernel]")
    if teacher:
        p.add_argument("--teacher_kernel", default="softmax", choices=list(KERNELS))
        p.add_argument("--teacher_tau", type=float, default=None, help="default: TAU[teacher_kernel]")
    p.add_argument("--data_dir", default=str(DATA_DIR))
    p.add_argument("--ckpt_dir", default=str(CKPT_DIR))
    p.add_argument("--results", default=str(Path(__file__).with_name("results.csv")))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--eval_batch_size", type=int, default=512)
    p.add_argument("--patch_size", type=int, default=4)
    p.add_argument("--num_layers", type=int, default=12)
    p.add_argument("--num_heads", type=int, default=3)
    p.add_argument("--hidden_dim", type=int, default=192)
    p.add_argument("--mlp_dim", type=int, default=768)
    return p


def parse(p):
    args = p.parse_args()
    args.tau = TAU[args.kernel] if args.tau is None else args.tau
    if getattr(args, "teacher_kernel", None) and args.teacher_tau is None:
        args.teacher_tau = TAU[args.teacher_kernel]
    return args


def tau_str(tau):
    return f"{tau:.4g}"


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _seed_worker(_):
    seed = torch.utils.data.get_worker_info().seed % 2 ** 32
    random.seed(seed)
    np.random.seed(seed)


def loader(args, train, batch_size, randaugment=False):
    ops = [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()] if train else []
    if randaugment:
        ops.append(T.RandAugment(num_ops=2, magnitude=9))
    dataset = torchvision.datasets.CIFAR10(args.data_dir, train=train, download=True,
                                           transform=T.Compose(ops + [T.ToTensor(), T.Normalize(MEAN, STD)]))
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, drop_last=train, num_workers=args.num_workers,
                      pin_memory=True, persistent_workers=True, worker_init_fn=_seed_worker,
                      generator=torch.Generator().manual_seed(args.seed))


def cycle(data):
    while True:
        yield from data


def build_model(args, kernel, tau):
    # naive: at S=65 the compiled full matrix is as fast as efficient, whose checkpointed chunks break torch.compile
    return VisionTransformer(image_size=32, patch_size=args.patch_size, num_layers=args.num_layers,
                             num_heads=args.num_heads, hidden_dim=args.hidden_dim, mlp_dim=args.mlp_dim,
                             kernel=kernel, tau=tau, impl="naive").cuda()


def param_groups(model, weight_decay):
    """No weight decay on biases and norm parameters."""
    decay = [p for n, p in model.named_parameters() if p.ndim > 1 and not n.endswith(".bias")]
    no_decay = [p for n, p in model.named_parameters() if p.ndim <= 1 or n.endswith(".bias")]
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


@torch.no_grad()
def evaluate(model, test_loader):
    """Top-1 test accuracy in %."""
    training = model.training
    model.eval()
    correct = total = 0
    for images, labels in test_loader:
        images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        correct += (model(images).argmax(dim=1) == labels).sum().item()
        total += labels.numel()
    model.train(training)
    return 100.0 * correct / total


def find_checkpoint(ckpt_dir, stage, kernel, tau):
    paths = sorted(Path(ckpt_dir).glob(f"vision_{stage}_{kernel}_tau{tau_str(tau)}_*.pt"))
    if len(paths) != 1:
        raise SystemExit(f"expected exactly one vision {stage} checkpoint for {kernel} tau {tau_str(tau)} "
                         f"in {ckpt_dir}, found {len(paths)}")
    print(f"loading {paths[0]}")
    return torch.load(paths[0], map_location="cuda")


def save_checkpoint(model, args, stage, run_id, parents):
    path = Path(args.ckpt_dir) / f"vision_{stage}_{args.kernel}_tau{tau_str(args.tau)}_{run_id}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(kernel=args.kernel, tau=args.tau, stage=stage, run_id=run_id, parent_run_ids=parents,
                    config=vars(args), state_dict=model.state_dict()), path)
    print(f"saved {path}")
    return path


def is_done(args, stage):
    if not Path(args.results).exists():
        return False
    return any(r["stage"] == stage and r["kernel"] == args.kernel and float(r["tau"]) == args.tau
               for r in csv.DictReader(open(args.results)))


def start_run():
    return uuid.uuid4().hex[:8]


def finish(run_id, args, stage, test_acc, parents, checkpoint, wall_min, status="ok"):
    """Append the result row."""
    row = dict(stage=stage, kernel=args.kernel, tau=args.tau, test_acc=round(test_acc, 2), status=status,
               run_id=run_id, parent_run_ids=";".join(parents), checkpoint=checkpoint.name if checkpoint else "",
               wall_min=round(wall_min, 1), seed=args.seed, hostname=socket.gethostname(),
               gpu=torch.cuda.get_device_name(), date=datetime.datetime.now().isoformat(timespec="seconds"))
    new = not Path(args.results).exists()
    with open(args.results, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)
    print(f"{stage} {args.kernel} tau {tau_str(args.tau)}: test acc {test_acc:.2f}% ({status})")
