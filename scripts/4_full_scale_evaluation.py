"""Phase 2.0 - Full-Scale Retrieval Evaluation

Validates Phase 1 results at scale:
- 500 answerable + 200 unanswerable CLAPnq records
- Metrics by question type (FACTOID, NUMERIC, TEMPORAL, LOCATION)
- Failure pattern analysis
- Comparison with Phase 1 (20-query) baseline

Run: python scripts/4_full_scale_evaluation.py
Output: data/full_scale_evaluation.json
"""

import json
import sys
import time
import asyncio
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict

import numpy as np
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.metrics import DatasetMetrics
from src.data_loading.clapnq_loader import CLAPnqLoader
from src.retrieval.vector_store_faiss import VectorStoreFaiss


# ─────────────────────────────────────────────
# Question Type Classifier
# ─────────────────────────────────────────────

def classify_question_type(question: str) -> str:
    """Classify question into FACTOID / NUMERIC / TEMPORAL / LOCATION."""
    q = question.lower().strip()

    if any(p in q for p in [
        'how many', 'how much', 'what number', 'what percentage', 'how often',
        'how large', 'how long is', 'how wide', 'how tall', 'how far',
        'what size', 'what age', 'how old', 'total', 'amount of', 'number of',
    ]):
        return 'NUMERIC'

    if any(p in q for p in [
        'when', 'what year', 'what date', 'what time', 'how long did',
        'in what year', 'in what month', 'in what century', 'what decade',
    ]):
        return 'TEMPORAL'

    if any(p in q for p in [
        'where', 'which country', 'which city', 'which state', 'which continent',
        'which region', 'location of', 'located in', 'what country', 'what city',
        'what state', 'what continent',
    ]):
        return 'LOCATION'

    return 'FACTOID'


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    a, b = np.array(v1, dtype=np.float32), np.array(v2, dtype=np.float32)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na and nb else 0.0


# ─────────────────────────────────────────────
# Core Evaluation
# ─────────────────────────────────────────────

async def evaluate_answerable(
    vector_store: VectorStoreFaiss,
    records: List[Dict[str, Any]],
    k: int = 5,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Evaluate 500 answerable records with per-type breakdown."""
    logger.info(f"Evaluating {len(records)} answerable records (k={k})...")

    all_queries: List[Dict] = []
    per_type: Dict[str, List[Dict]] = defaultdict(list)
    failures: List[Dict] = []
    timings: List[float] = []

    for i, record in enumerate(records, 1):
        query = record.get("input", "")
        answer = record.get("output", [{}])[0].get("answer", "")
        if not query or not answer:
            continue

        q_type = classify_question_type(query)
        t0 = time.time()

        answer_emb = vector_store.embed_text(answer)
        chunks = await vector_store.search_with_embeddings(query, top_k=k)

        timings.append(time.time() - t0)

        retrieved = [
            {
                "chunk_id": c["chunk_id"],
                "score": cosine_similarity(answer_emb, c["embedding"]),
                "text": c["chunk_text"],
                "passage_title": c["passage_title"],
            }
            for c in chunks
        ]
        relevant_ids = [c["chunk_id"] for c in retrieved if c["score"] > threshold]

        entry = {
            "query_id": record.get("id", f"rec_{i}"),
            "query_text": query,
            "retrieved_chunks": retrieved,
            "relevant_chunk_ids": relevant_ids,
        }
        all_queries.append(entry)
        per_type[q_type].append(entry)

        if not relevant_ids:
            failures.append({"record_id": record.get("id"), "query": query, "question_type": q_type})

        if i % 50 == 0:
            logger.info(f"  {i}/{len(records)} ({i/len(records)*100:.0f}%)")

    # Overall metrics
    dm_overall = DatasetMetrics(k=k)
    overall = dm_overall.evaluate_batch(all_queries)

    # Per-type metrics
    per_type_out: Dict[str, Dict] = {}
    for q_type, queries in per_type.items():
        dm = DatasetMetrics(k=k)
        per_type_out[q_type] = {
            "num_queries": len(queries),
            "metrics": dm.evaluate_batch(queries),
        }

    n = len(all_queries)
    return {
        "num_evaluated": n,
        "overall_metrics": overall,
        "per_type_metrics": per_type_out,
        "failure_count": len(failures),
        "failure_rate": round(len(failures) / n, 4) if n else 0,
        "failures": failures[:20],
        "avg_query_time_ms": round(float(np.mean(timings)) * 1000, 2) if timings else 0,
        "p95_query_time_ms": round(float(np.percentile(timings, 95)) * 1000, 2) if timings else 0,
    }


async def evaluate_unanswerable(
    vector_store: VectorStoreFaiss,
    records: List[Dict[str, Any]],
    k: int = 5,
) -> Dict[str, Any]:
    """Evaluate 200 unanswerable records — expect 0 precision & recall."""
    logger.info(f"Evaluating {len(records)} unanswerable records...")

    queries: List[Dict] = []
    timings: List[float] = []

    for record in records:
        query = record.get("input", "")
        if not query:
            continue
        t0 = time.time()
        chunks = await vector_store.search_with_embeddings(query, top_k=k)
        timings.append(time.time() - t0)
        queries.append({
            "query_id": record.get("id", "unk"),
            "query_text": query,
            "retrieved_chunks": [
                {"chunk_id": c["chunk_id"], "score": 0.0,
                 "text": c["chunk_text"], "passage_title": c["passage_title"]}
                for c in chunks
            ],
            "relevant_chunk_ids": [],
        })

    dm = DatasetMetrics(k=k)
    overall = dm.evaluate_batch(queries)

    return {
        "num_evaluated": len(queries),
        "overall_metrics": overall,
        "avg_query_time_ms": round(float(np.mean(timings)) * 1000, 2) if timings else 0,
    }


# ─────────────────────────────────────────────
# Report Printer
# ─────────────────────────────────────────────

PHASE1_BASELINE = {"mrr": 0.7751, "ndcg": 0.9614, "recall": 1.0000, "precision": 0.4000}


def print_report(ans: Dict, unans: Dict) -> None:
    sep = "=" * 70
    agg = ans["overall_metrics"]

    mrr  = agg.get("mrr_mean", 0)
    ndcg = agg.get("ndcg_mean", 0)
    rec  = agg.get("recall_mean", 0)
    prec = agg.get("precision_mean", 0)

    print(f"\n{sep}")
    print("PHASE 2.0 — FULL-SCALE RETRIEVAL EVALUATION REPORT")
    print(sep)

    print(f"\n{'ANSWERABLE QUERIES':} (n={ans['num_evaluated']})")
    print("-" * 70)
    print(f"  {'Metric':<14} {'This run':>10}  {'Phase 1':>10}  {'Delta':>10}")
    print(f"  {'-'*14}  {'-'*10}  {'-'*10}  {'-'*10}")
    for label, val, base_key in [
        ("MRR",       mrr,  "mrr"),
        ("NDCG",      ndcg, "ndcg"),
        ("Recall@5",  rec,  "recall"),
        ("Precision", prec, "precision"),
    ]:
        delta = val - PHASE1_BASELINE[base_key]
        flag = "✅" if delta >= -0.05 else "⚠️ "
        print(f"  {flag} {label:<12} {val:>10.4f}  {PHASE1_BASELINE[base_key]:>10.4f}  {delta:>+10.4f}")

    print(f"\n  Failures: {ans['failure_count']} / {ans['num_evaluated']} "
          f"({ans['failure_rate']*100:.1f}%)")
    print(f"  Avg query: {ans['avg_query_time_ms']:.1f}ms  "
          f"p95: {ans['p95_query_time_ms']:.1f}ms")

    # Per-type table
    print(f"\n{'BY QUESTION TYPE':}")
    print("-" * 70)
    print(f"  {'Type':<12} {'N':>5}  {'MRR':>7}  {'NDCG':>7}  {'Recall':>7}")
    print(f"  {'-'*12}  {'-'*5}  {'-'*7}  {'-'*7}  {'-'*7}")
    for q_type, data in sorted(ans.get("per_type_metrics", {}).items()):
        m = data["metrics"]
        print(f"  {q_type:<12} {data['num_queries']:>5}  "
              f"{m.get('mrr_mean',0):>7.4f}  "
              f"{m.get('ndcg_mean',0):>7.4f}  "
              f"{m.get('recall_mean',0):>7.4f}")

    # Sample failures
    if ans.get("failures"):
        print(f"\nSAMPLE FAILURES (first 5)")
        print("-" * 70)
        for f in ans["failures"][:5]:
            print(f"  [{f['question_type']}] {f['query'][:65]}")

    # Unanswerable
    uagg = unans["overall_metrics"]
    print(f"\n{'UNANSWERABLE QUERIES':} (n={unans['num_evaluated']})")
    print("-" * 70)
    print(f"  Precision: {uagg.get('precision_mean',0):.4f}  (expect 0.0 — no false positives)")
    print(f"  Recall:    {uagg.get('recall_mean',0):.4f}  (expect 0.0 — no relevant chunks)")
    print(f"  Avg query: {unans['avg_query_time_ms']:.1f}ms")

    # Verdict
    passes = (
        mrr  >= PHASE1_BASELINE["mrr"]  * 0.90 and
        rec  >= 0.65 and
        ans["failure_rate"] <= 0.35 and
        uagg.get("precision_mean", 0) <= 0.05
    )
    print(f"\n{sep}")
    if passes:
        print("✅  VERDICT: Phase 1 results hold at scale → Ready for Phase 2.1 (Generation)")
    else:
        print("⚠️   VERDICT: Performance dropped at scale → Investigate before Phase 2.1")
    print(sep)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

async def main() -> None:
    logger.info("Phase 2.0 — Full-Scale Evaluation starting...")

    vector_store = VectorStoreFaiss()
    if not vector_store.index_path.exists():
        logger.error(f"FAISS index not found at {vector_store.index_path}. Run script 2 first.")
        return

    stats = vector_store.get_stats()
    logger.info(f"Vector store loaded: {stats.get('total_chunks', 0)} chunks")

    loader = CLAPnqLoader()
    answerable = await loader.load_answerable(
        "data/clapnq_train_answerable.jsonl", limit=500
    )
    unanswerable = await loader.load_unanswerable(
        "data/clapnq_train_unanswerable.jsonl", limit=200
    )
    logger.info(f"Dataset: {len(answerable)} answerable, {len(unanswerable)} unanswerable")

    t0 = time.time()
    ans_results   = await evaluate_answerable(vector_store, answerable)
    unans_results = await evaluate_unanswerable(vector_store, unanswerable)
    total_time    = round(time.time() - t0, 2)
    logger.info(f"Evaluation complete in {total_time}s")

    output = {
        "phase": "2.0",
        "description": "Full-scale retrieval evaluation (500 answerable + 200 unanswerable)",
        "config": {
            "answerable_limit": 500,
            "unanswerable_limit": 200,
            "k": 5,
            "similarity_threshold": 0.5,
            "embedding_model": "all-MiniLM-L6-v2",
        },
        "phase1_baseline": PHASE1_BASELINE,
        "answerable": ans_results,
        "unanswerable": unans_results,
        "total_evaluation_time_s": total_time,
    }

    out_path = Path("data/full_scale_evaluation.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Saved → {out_path}")

    print_report(ans_results, unans_results)


if __name__ == "__main__":
    asyncio.run(main())
