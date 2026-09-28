"""LoCo (hazyresearch/LoCoV1) nDCG@10 for one kernel; per-subset JSON in results/loco/, averages in results/loco_summary.csv.

    python -m text_transformer.loco_eval --kernel add_bump [--seq_lens 2048 4096 8192]
"""
import json
import time

import torch
from datasets import load_dataset

from text_transformer import common

SUBSETS = ["summ_screen_fd", "gov_report", "qmsum", "qasper_abstract", "qasper_title"]

p = common.eval_parser(__doc__)
p.add_argument("--seq_lens", type=int, nargs="+", default=[2048, 4096, 8192])
p.add_argument("--batch_size", type=int, default=1)
p.add_argument("--max_docs", type=int, default=None, help="smoke tests only: first N documents per subset")
p.add_argument("--max_queries", type=int, default=None, help="smoke tests only: first N answerable queries")
args = common.parse(p)

model, run_id = common.load_eval_model(args)
common.fix_long_context_rope(model)
subset_queries = {s: [] for s in SUBSETS}
subset_docs = {s: [] for s in SUBSETS}
for q in load_dataset("hazyresearch/LoCoV1-Queries", split="test"):
    subset_queries.get(q["dataset"], []).append(q)
for d in load_dataset("hazyresearch/LoCoV1-Documents", split="test"):
    subset_docs.get(d["dataset"], []).append(d)
out = common.run_dir(args, "loco", run_id)
out.mkdir(parents=True, exist_ok=True)

for seq_len in args.seq_lens:
    model.max_seq_length = seq_len
    results = []
    for subset in SUBSETS:
        path = out / f"seq{seq_len}_{subset}.json"
        if path.exists():
            res = json.loads(path.read_text())
            print(f"[{seq_len}] skip {subset}: {res['ndcg@10']:.4f}", flush=True)
        else:
            docs = subset_docs[subset][:args.max_docs]
            index = {d["pid"]: i for i, d in enumerate(docs)}
            # retrieval is within the subset's own pool; the full pool holds every answer
            qs = [q for q in subset_queries[subset] if any(pid in index for pid in q["answer_pids"])][:args.max_queries]
            print(f"[{seq_len}] run  {subset}: {len(docs)} docs, {len(qs)} queries", flush=True)
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            # documents first: they are the long texts, so an OOM surfaces immediately
            d_emb = model.encode([d["passage"] for d in docs], prompt="search_document: ", batch_size=args.batch_size,
                                 normalize_embeddings=True, show_progress_bar=False)
            q_emb = model.encode([q["query"] for q in qs], prompt="search_query: ", batch_size=args.batch_size,
                                 normalize_embeddings=True, show_progress_bar=False)
            relevant = [{index[pid] for pid in q["answer_pids"] if pid in index} for q in qs]
            res = {"subset": subset, "seq_len": seq_len, "num_docs": len(docs), "num_queries": len(qs),
                   "ndcg@10": round(common.ndcg_at_10(q_emb @ d_emb.T, relevant), 4),
                   "runtime_min": round((time.perf_counter() - start) / 60, 2),
                   "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
            path.write_text(json.dumps(res, indent=2))
            print(f"    ndcg@10 {res['ndcg@10']:.4f}  {res['runtime_min']:.2f} min  peak {res['peak_mem_gb']:.2f} GB", flush=True)
        results.append(res)

    values = {"avg": round(sum(r["ndcg@10"] for r in results) / len(results), 4)}
    values.update({r["subset"]: r["ndcg@10"] for r in results})
    values.update(runtime_min=round(sum(r["runtime_min"] for r in results), 2),
                  peak_mem_gb=max(r["peak_mem_gb"] for r in results))
    print(f"LoCo @ {seq_len}: {values}")
    common.write_summary(args, "loco", run_id, {"seq_len": seq_len}, values)
