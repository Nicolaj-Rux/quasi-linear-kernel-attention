"""Stage scratch: train a ViT on CIFAR-10 from scratch with one kernel; saves a checkpoint, appends to results.csv.

    python -m vision_transformer.scratch --kernel softmax
"""
import time

import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T2

from vision_transformer import common

p = common.parser(__doc__)
p.add_argument("--epochs", type=int, default=1000)
p.add_argument("--warmup_epochs", type=int, default=100)
p.add_argument("--batch_size", type=int, default=128)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--lr_min", type=float, default=1e-5)
p.add_argument("--weight_decay", type=float, default=0.1)
p.add_argument("--label_smoothing", type=float, default=0.1)
p.add_argument("--mixup_alpha", type=float, default=0.8)
p.add_argument("--cutmix_alpha", type=float, default=1.0)
p.add_argument("--clip", type=float, default=2.0)
p.add_argument("--log_every", type=int, default=100, help="steps")
p.add_argument("--eval_every", type=int, default=10, help="epochs between printed test accuracies")
args = common.parse(p)
if common.is_done(args, "scratch"):
    print(f"scratch {args.kernel} tau {common.tau_str(args.tau)} already in {args.results}, skipping")
    raise SystemExit   # exit code 0: already done counts as success

common.seed_everything(args.seed)
train_loader = common.loader(args, train=True, batch_size=args.batch_size, randaugment=True)
test_loader = common.loader(args, train=False, batch_size=args.eval_batch_size)
model = common.build_model(args, args.kernel, args.tau)
forward = torch.compile(model)   # 10-40% faster steps; weights stay in `model`
run_id = common.start_run()

mix = T2.RandomChoice([T2.MixUp(alpha=args.mixup_alpha, num_classes=10), T2.CutMix(alpha=args.cutmix_alpha, num_classes=10)])
opt = torch.optim.AdamW(common.param_groups(model, args.weight_decay), lr=args.lr)
warmup = args.warmup_epochs * len(train_loader)
total = args.epochs * len(train_loader)
sched = torch.optim.lr_scheduler.SequentialLR(opt, [
    torch.optim.lr_scheduler.LinearLR(opt, start_factor=args.lr_min / args.lr, total_iters=warmup),
    torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total - warmup, eta_min=args.lr_min)], milestones=[warmup])

step, status, start = 0, "ok", time.time()
model.train()
for epoch in range(args.epochs):
    for images, labels in train_loader:
        images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        images, soft = mix(images, labels)
        loss = F.cross_entropy(forward(images), soft, label_smoothing=args.label_smoothing)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()
        sched.step()
        step += 1
        if step % args.log_every == 0:
            print(f"step {step}  train/loss {loss.item():.4f}  train/grad_norm {grad_norm.item():.4f}  "
                  f"train/lr {sched.get_last_lr()[0]:.3e}  epoch {epoch}", flush=True)
            if not torch.isfinite(loss):
                status = f"diverged at step {step}"
                break
    if status != "ok":
        break
    print(f"epoch {epoch + 1}/{args.epochs}  loss {loss.item():.4f}  {(time.time() - start) / 60:.1f} min", flush=True)
    if (epoch + 1) % args.eval_every == 0:
        print(f"step {step}  test/acc_curve {common.evaluate(model, test_loader):.2f}", flush=True)

path = common.save_checkpoint(model, args, "scratch", run_id, [])
common.finish(run_id, args, "scratch", common.evaluate(model, test_loader), [], path, (time.time() - start) / 60, status)
