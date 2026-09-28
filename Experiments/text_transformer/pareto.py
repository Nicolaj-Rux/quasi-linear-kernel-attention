"""LoCo with every document padded to exactly --seq_len, timed on the GPU; one row to results/pareto_summary.csv.

    python -m text_transformer.pareto --kernel add_bump [--seq_len 8192] [--batch_size 16] [--tau 1.5]
"""
import time

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

from text_transformer import common

SUBSETS = ["summ_screen_fd", "gov_report", "qmsum", "qasper_abstract", "qasper_title"]

p = common.eval_parser(__doc__)
p.add_argument("--seq_len", type=int, default=8192)
p.add_argument("--batch_size", type=int, default=16)
p.add_argument("--max_docs", type=int, default=None, help="smoke tests only: first N documents per subset")
p.add_argument("--max_queries", type=int, default=None, help="smoke tests only: first N answerable queries")
args = common.parse(p)
SEQ_LEN = args.seq_len

model, run_id = common.load_eval_model(args)
common.fix_long_context_rope(model)          # sets the NTK reference length the frequencies below use
model.max_seq_length = SEQ_LEN
tokenizer, auto_model = model[0].tokenizer, model[0].auto_model

# Dynamic NTK derives the RoPE base from the padded length, and a batch shares one base, so both padding
# and batch composition would change the embeddings. Build the frequencies per sequence from its true
# token count instead: any batch size then gives the same output as a single unpadded forward.
rotary = model[0].auto_model.rotary_emb
real_lens = [SEQ_LEN]


def rope_forward(x, position_ids):
    inv = torch.stack([ROPE_INIT_FUNCTIONS["dynamic"](rotary.config, x.device, seq_len=n)[0] for n in real_lens])
    freqs = inv[:, None, :] * position_ids[:, :, None]          # (B, 1, D/2) * (1, S, 1) -> (B, S, D/2)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


rotary.forward = rope_forward


@torch.no_grad()
def encode(texts, prompt, pad):
    """Mean-pooled, L2-normalized embeddings (padded to SEQ_LEN if pad) and the summed GPU ms of the forwards."""
    global real_lens
    embeddings, gpu_ms = [], 0.0
    for i in range(0, len(texts), args.batch_size):
        batch = tokenizer([prompt + t for t in texts[i:i + args.batch_size]], return_tensors="pt",
                          truncation=True, max_length=SEQ_LEN,
                          padding="max_length" if pad else True).to("cuda")
        real_lens = batch["attention_mask"].sum(dim=1).tolist()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        hidden = auto_model(**batch).last_hidden_state
        end.record()
        torch.cuda.synchronize()
        gpu_ms += start.elapsed_time(end)
        mask = batch["attention_mask"].unsqueeze(-1)
        embeddings.append(F.normalize((hidden * mask).sum(dim=1) / mask.sum(dim=1), dim=-1))
    return torch.cat(embeddings).cpu().numpy(), gpu_ms


subset_queries = {s: [] for s in SUBSETS}
subset_docs = {s: [] for s in SUBSETS}
for q in load_dataset("hazyresearch/LoCoV1-Queries", split="test"):
    subset_queries.get(q["dataset"], []).append(q)
for d in load_dataset("hazyresearch/LoCoV1-Documents", split="test"):
    subset_docs.get(d["dataset"], []).append(d)

# the padded shape is the same for every batch, so one warmup pays for all torch.compile work
encode([d["passage"] for d in subset_docs[SUBSETS[0]][:args.batch_size]], "search_document: ", pad=True)

torch.cuda.reset_peak_memory_stats()
scores, num_docs, doc_gpu_ms, doc_wall_s, query_gpu_ms = {}, 0, 0.0, 0.0, 0.0
for subset in SUBSETS:
    docs = subset_docs[subset][:args.max_docs]
    index = {d["pid"]: i for i, d in enumerate(docs)}
    # retrieval is within the subset's own pool; the full pool holds every answer
    qs = [q for q in subset_queries[subset] if any(pid in index for pid in q["answer_pids"])][:args.max_queries]
    print(f"{subset}: {len(docs)} docs at {SEQ_LEN}, {len(qs)} queries", flush=True)

    start = time.perf_counter()
    d_emb, ms = encode([d["passage"] for d in docs], "search_document: ", pad=True)
    doc_wall_s += time.perf_counter() - start
    doc_gpu_ms += ms
    num_docs += len(docs)
    q_emb, query_ms = encode([q["query"] for q in qs], "search_query: ", pad=False)
    query_gpu_ms += query_ms

    relevant = [{index[pid] for pid in q["answer_pids"] if pid in index} for q in qs]
    scores[subset] = round(common.ndcg_at_10(q_emb @ d_emb.T, relevant), 4)
    print(f"    ndcg@10 {scores[subset]:.4f}  docs {ms / 1000:.1f} s GPU  queries {query_ms / 1000:.1f} s GPU",
          flush=True)

values = {"avg": round(sum(scores.values()) / len(scores), 4), **scores,
          "doc_gpu_s": round(doc_gpu_ms / 1000, 2), "doc_wall_s": round(doc_wall_s, 2),
          "query_gpu_s": round(query_gpu_ms / 1000, 2), "num_docs": num_docs,
          "ms_per_doc": round(doc_gpu_ms / num_docs, 2),
          "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
print(f"LoCo @ {SEQ_LEN} padded, batch {args.batch_size}: {values}")
common.write_summary(args, "pareto", run_id, {"seq_len": SEQ_LEN, "batch_size": args.batch_size}, values)
