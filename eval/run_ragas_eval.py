"""
Runs the full RAG pipeline over eval/test_queries.json and scores it
with RAGAS: faithfulness, answer_relevancy, context_precision, context_recall.

IMPORTANT: fill in "ground_truth" for each query in test_queries.json
before running this (context_precision/recall need a reference answer
to compare against). Faithfulness and answer_relevancy work even
without ground_truth, but you'll get the full picture with it filled in.

Usage:
    python eval/run_ragas_eval.py
Outputs:
    eval/results/ragas_raw.csv       - per-question scores
    eval/results/metrics_table.md    - README-ready summary table
"""
import json
import os
import sys
import time
from pathlib import Path
from statistics import mean

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from rag.pipeline import FinRAGPipeline

from datasets import Dataset
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

QUERIES_PATH = Path(__file__).parent / "test_queries.json"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
PROGRESS_PATH = RESULTS_DIR / "progress.jsonl"


def _load_existing_progress():
    """Load any previously completed question results, keyed by question id."""
    completed = {}
    if PROGRESS_PATH.exists():
        with open(PROGRESS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                completed[row["id"]] = row
    return completed


def _append_progress(row):
    with open(PROGRESS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_pipeline_over_queries():
    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        queries = json.load(f)

    completed = _load_existing_progress()
    if completed:
        print(f"Found {len(completed)} previously completed questions in "
              f"{PROGRESS_PATH} - skipping those and resuming the rest.\n")

    pipeline = None  # lazily init only if we actually need to call the model

    rows = []
    retrieval_latencies = []
    generation_latencies = []

    for q in queries:
        if q["id"] in completed:
            row = completed[q["id"]]
            rows.append(row)
            if row.get("retrieval_latency_ms") is not None:
                retrieval_latencies.append(row["retrieval_latency_ms"])
            if row.get("generation_latency_ms") is not None:
                generation_latencies.append(row["generation_latency_ms"])
            print(f"[{q['id']}/{len(queries)}] (already done, skipped) "
                  f"{q['question'][:60]}...")
            continue

        if pipeline is None:
            pipeline = FinRAGPipeline()

        print(f"[{q['id']}/{len(queries)}] {q['question'][:70]}...")
        result = pipeline.answer(q["question"], company_filter=q["company"])

        retrieval_latencies.append(result["retrieval_latency_ms"])
        generation_latencies.append(result["generation_latency_ms"])

        row = {
            "id": q["id"],
            "question": q["question"],
            "answer": result["answer"],
            "contexts": [c["text"] for c in result["contexts"]],
            "ground_truth": q.get("ground_truth", ""),
            "company": q["company"],
            "type": q["type"],
            "retrieval_latency_ms": result["retrieval_latency_ms"],
            "generation_latency_ms": result["generation_latency_ms"],
        }
        rows.append(row)
        _append_progress(row)  # save immediately - survives crashes/rate limits

    latency_summary = {
        "avg_retrieval_ms": mean(retrieval_latencies) if retrieval_latencies else 0,
        "p95_retrieval_ms": sorted(retrieval_latencies)[int(len(retrieval_latencies) * 0.95) - 1]
                             if retrieval_latencies else 0,
        "avg_generation_ms": mean(generation_latencies) if generation_latencies else 0,
        "p95_generation_ms": sorted(generation_latencies)[int(len(generation_latencies) * 0.95) - 1]
                              if generation_latencies else 0,
    }
    return rows, latency_summary


def score_with_ragas(rows):
    # RAGAS needs ground_truth filled in for context_precision/context_recall.
    has_ground_truth = all(r["ground_truth"] and r["ground_truth"] != "FILL_ME" for r in rows)

    # Optionally score only a subset of questions. RAGAS makes roughly
    # 4 judge-LLM calls per question per metric, so 50 questions x 4
    # metrics = ~200 calls - too many for a free-tier daily token
    # budget to reliably complete in one run. Set EVAL_SUBSET_SIZE in
    # .env (e.g. 20) to score a smaller, honest subset instead. Default
    # keeps all rows if not set.
    subset_size = os.getenv("EVAL_SUBSET_SIZE")
    if subset_size:
        n = int(subset_size)
        rows = rows[:n]
        print(f"\n[INFO] EVAL_SUBSET_SIZE={n} set - scoring only the first "
              f"{n} of {len(rows)} questions to fit within API limits. "
              f"Report this subset size honestly in your README.\n")

    # Truncate context chunks before scoring - RAGAS re-sends every
    # retrieved chunk to the judge LLM for each metric, so long chunks
    # multiply token cost across ~4 metrics x N questions. 1200 chars
    # is plenty for the judge to assess relevance/faithfulness.
    trimmed_rows = []
    for r in rows:
        r = dict(r)
        r["contexts"] = [c[:1200] for c in r["contexts"]]
        trimmed_rows.append(r)

    dataset = Dataset.from_list(trimmed_rows)

    metrics = [faithfulness, answer_relevancy]
    if has_ground_truth:
        metrics += [context_precision, context_recall]
    else:
        print("\n[WARN] ground_truth not filled in for all queries -> skipping "
              "context_precision and context_recall. Fill eval/test_queries.json "
              "with verified answers from the PDFs for the full metric set.\n")

    # Optionally restrict to a single named metric (e.g. to cheaply
    # re-score just one metric that failed/zeroed-out on a weaker model,
    # without re-running all four and burning tokens on the ones that
    # already worked). Set METRICS_ONLY=answer_relevancy in .env.
    metrics_only = os.getenv("METRICS_ONLY")
    if metrics_only:
        name_to_metric = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "context_recall": context_recall,
        }
        wanted = [m.strip() for m in metrics_only.split(",")]
        metrics = [name_to_metric[m] for m in wanted if m in name_to_metric]
        print(f"\n[INFO] METRICS_ONLY set - scoring only: {[m for m in wanted]}\n")

    # RAGAS defaults to OpenAI for its judge-LLM and embeddings. We don't
    # have an OpenAI key, so point it at Groq (judge LLM) and the same
    # local BGE model we already use for retrieval (embeddings) instead.
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from config import GROQ_API_KEY, EMBED_MODEL

    # Use a SEPARATE Groq key for judging if provided. RAGAS makes many
    # additional LLM calls to score each answer, which easily exhausts
    # a free-tier daily quota that was already spent generating the
    # pipeline's answers. Set GROQ_JUDGE_API_KEY in .env to a second,
    # unused Groq key to avoid this - falls back to the main key if not set.
    judge_api_key = os.getenv("GROQ_JUDGE_API_KEY", GROQ_API_KEY)

    # Use a smaller, faster, cheaper model just for judging. The
    # 70B model used for generation is overkill for scoring and burns
    # through the daily token budget fast. A smaller instant model
    # uses far fewer tokens per call and Groq's free tier gives it a
    # separate, typically larger daily allowance.
    judge_model = os.getenv("GROQ_JUDGE_MODEL", "llama-3.1-8b-instant")

    judge_llm = LangchainLLMWrapper(ChatGroq(
        api_key=judge_api_key,
        model=judge_model,
        temperature=0,
    ))
    judge_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
        model_name=EMBED_MODEL
    ))

    # The smaller judge model hits a PER-MINUTE token limit (clears in
    # seconds - Groq's own error tells us how long: "try again in 20s")
    # rather than the old per-DAY limit (which took hours to clear).
    # These need opposite retry strategies: a per-minute limit is worth
    # waiting out with a few retries; a per-day limit is not. Give each
    # call enough retries and wait time to ride out a short TPM cooldown,
    # and a longer timeout since small models under load can be slow to
    # respond even when they do succeed.
    from ragas.run_config import RunConfig
    run_config = RunConfig(
        max_workers=1,
        timeout=90,
        max_retries=3,
        max_wait=40,
    )

    result = evaluate(
        dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=run_config,
    )
    return result, has_ground_truth, len(rows)


def write_metrics_table(ragas_result, latency_summary, has_ground_truth, n_queries):
    df = ragas_result.to_pandas()

    def safe_mean(col):
        return df[col].mean() if col in df.columns else None

    faithfulness_score = safe_mean("faithfulness")
    relevancy_score = safe_mean("answer_relevancy")
    precision_score = safe_mean("context_precision")
    recall_score = safe_mean("context_recall")

    lines = []
    lines.append("## RAGAS Evaluation Results\n")
    lines.append(f"Evaluated on **{n_queries} queries** across 5 companies "
                  f"(TCS, Reliance, Infosys, Tata Motors, HDFC Bank).\n")
    lines.append("| Metric | Score | What it measures |")
    lines.append("|---|---|---|")
    lines.append(f"| Faithfulness | {faithfulness_score:.3f} | Are claims in the answer actually supported by the retrieved context? |" if faithfulness_score is not None else "| Faithfulness | N/A (not computed this run) | |")
    lines.append(f"| Answer Relevancy | {relevancy_score:.3f} | Does the answer actually address the question asked? |" if relevancy_score is not None else "| Answer Relevancy | N/A (not computed this run) | |")
    lines.append(f"| Context Precision | {precision_score:.3f} | Of the retrieved chunks, how many were actually relevant? |" if precision_score is not None else "| Context Precision | N/A (not computed this run) | |")
    lines.append(f"| Context Recall | {recall_score:.3f} | Did retrieval surface all the info needed to answer correctly? |" if recall_score is not None else "| Context Recall | N/A (not computed this run) | |")

    lines.append("\n### Latency (measured separately, not blended)\n")
    lines.append("| Stage | Avg | P95 |")
    lines.append("|---|---|---|")
    lines.append(f"| Retrieval (FAISS) | {latency_summary['avg_retrieval_ms']:.1f} ms | {latency_summary['p95_retrieval_ms']:.1f} ms |")
    lines.append(f"| Generation (Groq) | {latency_summary['avg_generation_ms']:.1f} ms | {latency_summary['p95_generation_ms']:.1f} ms |")

    lines.append("\n### Notes / what I'd improve next\n")
    lines.append("- _Fill this in after you see your real numbers._ "
                  "E.g.: \"Faithfulness dropped mainly on multi-hop questions "
                  "spanning two segments — retrieval only pulled one segment's "
                  "table. Next step: query decomposition for multi-hop questions.\"")

    out_path = RESULTS_DIR / "metrics_table.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nWrote metrics table -> {out_path}")
    print("\n".join(lines))


if __name__ == "__main__":
    rows, latency_summary = run_pipeline_over_queries()

    raw_df = pd.DataFrame(rows)
    raw_df.to_csv(RESULTS_DIR / "pipeline_outputs_raw.csv", index=False)

    ragas_result, has_gt, n_scored = score_with_ragas(rows)
    ragas_df = ragas_result.to_pandas()
    ragas_df.to_csv(RESULTS_DIR / "ragas_raw.csv", index=False)

    write_metrics_table(ragas_result, latency_summary, has_gt, n_scored)

    print(f"\nPipeline answered {len(rows)} questions; RAGAS scored {n_scored} "
          f"of them. If you want to re-run this eval fresh later (e.g. after "
          f"fixing test_queries.json), delete {PROGRESS_PATH} first, "
          f"otherwise the old answers will be reused.")