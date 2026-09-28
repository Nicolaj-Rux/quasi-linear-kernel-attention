"""Stage e2e: end-to-end distillation of the lbl student on KL(teacher || student) at temperature T, scaled by T^2.

    python -m vision_transformer.end_to_end --kernel gauss
"""
import time

import torch
import torch.nn.functional as F

from vision_transformer import common

p = common.parser(__doc__, teacher=True)
p.add_argument("--steps", type=int, default=50_000)
p.add_argument("--batch_size", type=int, default=512)
p.add_argument("--lr", type=float, default=2e-5)
p.add_argument("--lr_min", type=float, default=5e-7)
p.add_argument("--weight_decay", type=float, default=0.05)
p.add_argument("--temperature", type=float, default=4.0)
p.add_argument("--clip", type=float, default=2.0)
p.add_argument("--log_every", type=int, default=100, help="steps")
p.add_argument("--eval_every", type=int, default=5000, help="steps between printed test accuracies")
args = common.parse(p)
if common.is_done(args, "e2e"):
    print(f"e2e {args.kernel} tau {common.tau_str(args.tau)} already in {args.results}, skipping")
    raise SystemExit   # exit code 0: already done counts as success

common.seed_everything(args.seed)
start = time.time()
teacher_ckpt = common.find_checkpoint(args.ckpt_dir, "scratch", args.teacher_kernel, args.teacher_tau)
student_ckpt = common.find_checkpoint(args.ckpt_dir, "lbl", args.kernel, args.tau)
teacher = common.build_model(args, args.teacher_kernel, args.teacher_tau)
teacher.load_state_dict(teacher_ckpt["state_dict"])
teacher.eval().requires_grad_(False)
student = common.build_model(args, args.kernel, args.tau)
student.load_state_dict(student_ckpt["state_dict"])
parents = [student_ckpt["run_id"], teacher_ckpt["run_id"]]
run_id = common.start_run()
batches = common.cycle(common.loader(args, train=True, batch_size=args.batch_size))
test_loader = common.loader(args, train=False, batch_size=args.eval_batch_size)

teacher_forward, student_forward = torch.compile(teacher), torch.compile(student)   # weights stay in the modules
opt = torch.optim.AdamW(common.param_groups(student, args.weight_decay), lr=args.lr)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr_min)
T = args.temperature
student.train()
for step in range(1, args.steps + 1):
    images = next(batches)[0].cuda(non_blocking=True)
    with torch.no_grad():
        target = F.softmax(teacher_forward(images) / T, dim=-1)
    loss = F.kl_div(F.log_softmax(student_forward(images) / T, dim=-1), target, reduction="batchmean") * T ** 2
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(student.parameters(), args.clip)
    opt.step()
    sched.step()
    if step % args.log_every == 0:
        print(f"step {step}  e2e/kl {loss.item():.4f}  e2e/lr {sched.get_last_lr()[0]:.3e}", flush=True)
    if step % args.eval_every == 0:
        print(f"step {step}  test/acc_curve {common.evaluate(student, test_loader):.2f}", flush=True)
        print(f"step {step}/{args.steps}  kl {loss.item():.4f}  {(time.time() - start) / 60:.1f} min", flush=True)

path = common.save_checkpoint(student, args, "e2e", run_id, parents)
common.finish(run_id, args, "e2e", common.evaluate(student, test_loader), parents, path, (time.time() - start) / 60)
