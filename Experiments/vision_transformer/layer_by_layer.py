"""Stage lbl: per layer, train only the student attention to match the teacher's output (MSE, student-forced input).

    python -m vision_transformer.layer_by_layer --kernel gauss
"""
import time

import torch
import torch.nn.functional as F

from vision_transformer import common

p = common.parser(__doc__, teacher=True)
p.add_argument("--steps_per_layer", type=int, default=10_000)
p.add_argument("--batch_size", type=int, default=512)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--lr_min", type=float, default=1e-6)
p.add_argument("--clip", type=float, default=2.0)
p.add_argument("--log_every", type=int, default=500, help="steps")
args = common.parse(p)
if common.is_done(args, "lbl"):
    print(f"lbl {args.kernel} tau {common.tau_str(args.tau)} already in {args.results}, skipping")
    raise SystemExit   # exit code 0: already done counts as success

common.seed_everything(args.seed)
start = time.time()
teacher_ckpt = common.find_checkpoint(args.ckpt_dir, "scratch", args.teacher_kernel, args.teacher_tau)
teacher = common.build_model(args, args.teacher_kernel, args.teacher_tau)
teacher.load_state_dict(teacher_ckpt["state_dict"])
teacher.eval().requires_grad_(False)
student = common.build_model(args, args.kernel, args.tau)
student.load_state_dict(teacher_ckpt["state_dict"])
parents = [teacher_ckpt["run_id"]]
run_id = common.start_run()
batches = common.cycle(common.loader(args, train=True, batch_size=args.batch_size))


def attn_input(model, images, layer):
    """Input of layer `layer`'s attention: mirrors VisionTransformer.forward up to that layer's ln_1."""
    x = model._process_input(images)
    x = torch.cat([model.class_token.expand(x.shape[0], -1, -1), x], dim=1) + model.encoder.pos_embedding
    for j in range(layer):
        x = model.encoder.layers[j](x)
    return model.encoder.layers[layer].ln_1(x)


step = 0
for layer in range(args.num_layers):
    student.requires_grad_(False)
    attn = student.encoder.layers[layer].self_attention.requires_grad_(True)
    teacher_attn = teacher.encoder.layers[layer].self_attention
    opt = torch.optim.AdamW(attn.parameters(), lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps_per_layer, eta_min=args.lr_min)
    for _ in range(args.steps_per_layer):
        images = next(batches)[0].cuda(non_blocking=True)
        with torch.no_grad():
            target = teacher_attn(attn_input(teacher, images, layer))
            x = attn_input(student, images, layer)
        loss = F.mse_loss(attn(x), target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(attn.parameters(), args.clip)
        opt.step()
        sched.step()
        step += 1
        if step % args.log_every == 0:
            print(f"step {step}  lbl/mse {loss.item():.3e}  lbl/lr {sched.get_last_lr()[0]:.3e}  lbl/layer {layer}", flush=True)
    print(f"layer {layer}: mse {loss.item():.3e}  {(time.time() - start) / 60:.1f} min", flush=True)

student.requires_grad_(True)
path = common.save_checkpoint(student, args, "lbl", run_id, parents)
test_loader = common.loader(args, train=False, batch_size=args.eval_batch_size)
common.finish(run_id, args, "lbl", common.evaluate(student, test_loader), parents, path, (time.time() - start) / 60)
