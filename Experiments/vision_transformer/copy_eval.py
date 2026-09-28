"""Stage copy: the softmax scratch teacher's weights evaluated with the student kernel, no training, no checkpoint.

    python -m vision_transformer.copy_eval --kernel gauss
"""
import time

from vision_transformer import common

p = common.parser(__doc__, teacher=True)
args = common.parse(p)
if common.is_done(args, "copy"):
    print(f"copy {args.kernel} tau {common.tau_str(args.tau)} already in {args.results}, skipping")
    raise SystemExit   # exit code 0: already done counts as success

common.seed_everything(args.seed)
start = time.time()
teacher = common.find_checkpoint(args.ckpt_dir, "scratch", args.teacher_kernel, args.teacher_tau)
model = common.build_model(args, args.kernel, args.tau)
model.load_state_dict(teacher["state_dict"])
parents = [teacher["run_id"]]
run_id = common.start_run()
acc = common.evaluate(model, common.loader(args, train=False, batch_size=args.eval_batch_size))
common.finish(run_id, args, "copy", acc, parents, None, (time.time() - start) / 60)
