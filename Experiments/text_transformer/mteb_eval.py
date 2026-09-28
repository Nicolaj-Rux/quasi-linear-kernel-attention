"""MTEB(eng, v2) for one kernel; per-task JSON in results/mteb/, averages in results/mteb_summary.csv.

    python -m text_transformer.mteb_eval --kernel add_bump [--stage lbl] [--tasks STS17 ...]
"""
import json
import time

import mteb
import torch
from mteb import TaskResult

from text_transformer import common

# fastest first, so partial runs cover the most tasks
ORDER = [
    "STS17",
    "BIOSSES",
    "STSBenchmark",
    "STS13",
    "STS15",
    "TwentyNewsgroupsClustering.v2",
    "STS12",
    "ArXivHierarchicalClusteringS2S",
    "STS14",
    "MedrxivClusteringS2S.v2",
    "AskUbuntuDupQuestions",
    "ArXivHierarchicalClusteringP2P",
    "AmazonCounterfactualClassification",
    "SummEvalSummarization.v2",
    "SICK-R",
    "StackExchangeClustering.v2",
    "TwitterSemEval2015",
    "STS22.v2",
    "MTOPDomainClassification",
    "MassiveIntentClassification",
    "MassiveScenarioClassification",
    "TweetSentimentExtractionClassification",
    "Banking77Classification",
    "MedrxivClusteringP2P.v2",
    "BiorxivClusteringP2P.v2",
    "ToxicConversationsClassification",
    "StackExchangeClusteringP2P.v2",
    "TwitterURLCorpus",
    "ArguAna",
    "SprintDuplicateQuestions",
    "CQADupstackGamingRetrieval",
    "SCIDOCS",
    "ImdbClassification",
    "FiQA2018",
    "CQADupstackUnixRetrieval",
    "ClimateFEVERHardNegatives",
    "HotpotQAHardNegatives",
    "FEVERHardNegatives",
    "TRECCOVID",
    "MindSmallReranking",                       # needs ~75 GB RAM
    "Touche2020Retrieval.v3",
]
TYPES = ["Retrieval", "Classification", "Clustering", "PairClassification", "Reranking", "STS", "Summarization"]
PROMPTS = {"Classification": "classification: ", "Clustering": "clustering: ", "PairClassification": "classification: ",
           "STS": "classification: ", "Summarization": "classification: ", "query": "search_query: ",
           "document": "search_document: "}

p = common.eval_parser(__doc__)
p.add_argument("--batch_size", type=int, default=32)
p.add_argument("--tasks", nargs="+", default=ORDER, choices=ORDER, help="subset of the benchmark (default: all 41)")
args = common.parse(p)
# mteb reranks one query at a time on CPU, where one thread per core makes tiny matmuls ~200x slower
torch.set_num_threads(4)

tasks = mteb.get_benchmark("MTEB(eng, v2)").tasks
assert {t.metadata.name for t in tasks} == set(ORDER), "ORDER does not match the benchmark"
tasks = sorted((t for t in tasks if t.metadata.name in args.tasks), key=lambda t: ORDER.index(t.metadata.name))
model, run_id = common.load_eval_model(args)
model.max_seq_length = 512
model.prompts = PROMPTS
out = common.run_dir(args, "mteb", run_id)
out.mkdir(parents=True, exist_ok=True)

scores, runtime, peak = [], 0.0, 0.0
for i, task in enumerate(tasks, 1):
    name = task.metadata.name
    path, cost_path = out / f"{name}.json", out / f"{name}.cost.json"
    if path.exists():
        result, cost = TaskResult.from_disk(path), json.loads(cost_path.read_text())
        print(f"[{i}/{len(tasks)}] skip {name}: {result.get_score():.4f}", flush=True)
    else:
        print(f"[{i}/{len(tasks)}] run  {name} ({task.metadata.type})", flush=True)
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        result = mteb.evaluate(model, tasks=[task], cache=None, overwrite_strategy="always",
                               encode_kwargs={"batch_size": args.batch_size})[0]
        cost = dict(runtime_min=(time.perf_counter() - start) / 60, peak_mem_gb=torch.cuda.max_memory_allocated() / 1e9)
        cost_path.write_text(json.dumps(cost))   # before the result, so a result on disk always has its cost
        result.to_disk(path)
        print(f"    {task.metadata.main_score}: {result.get_score():.4f}  {cost['runtime_min']:.1f} min", flush=True)
    scores.append((task.metadata.type, result.get_score()))
    runtime += cost["runtime_min"]
    peak = max(peak, cost["peak_mem_gb"])

values = {"num_tasks": len(scores), "avg": round(sum(s for _, s in scores) / len(scores), 4)}
for t in TYPES:
    typed = [s for ty, s in scores if ty == t]
    values[t] = round(sum(typed) / len(typed), 4) if typed else ""
values.update(runtime_min=round(runtime, 1), peak_mem_gb=round(peak, 2))
print(values)
common.write_summary(args, "mteb", run_id, {}, values)
