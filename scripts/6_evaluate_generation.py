#!/usr/bin/env python3
"""
Phase 2.5 — LLM-as-Judge Generation Evaluation

Runs the full RAG pipeline on CLAPnq answerable records and scores each
generated answer against the gold passage using an Ollama model as judge.

Generator and judge are intentionally separate models so the judge is
independent of the system being evaluated.

Why LLM-as-Judge and not Token F1 / ROUGE-L / BERTScore:
  - Token F1 / ROUGE-L: recall is capped at ~0.08 for a 13-token answer
    against a 120-token CLAPnq gold passage — correct concise answers score 0.15.
  - BERTScore: entity substitution blindness — "Berlin" scores 0.93 against
    a gold that says "Paris" because both are European capitals in embedding space.
  - LLM-as-Judge: reads question + gold + generated, evaluates factual accuracy
    and completeness directly — handles long gold, paraphrase, and entity errors.

See docs/PHASE2_5_EVALUATION.md for full metric analysis.

Usage:
    # 50-record spot-check, llama3 generator + mistral judge (defaults)
    python scripts/6_evaluate_generation.py

    # Full dataset
    python scripts/6_evaluate_generation.py --limit 0

    # Custom generator, different judge model
    python scripts/6_evaluate_generation.py --model llama3 --judge-model phi3

    # Anthropic generator, ollama judge
    python scripts/6_evaluate_generation.py --provider anthropic --model claude-sonnet-4-6

    # Custom output path
    python scripts/6_evaluate_generation.py --output data/my_eval.json
"""

import asyncio
import argparse
import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.vector_store_faiss import VectorStoreFaiss
from src.generation.rag_generator import RAGGenerator
from src.generation.providers import get_provider
from src.generation.providers.base import BaseLLMProvider


# ── Judge configuration ────────────────────────────────────────────────────────
# mistral is the default judge — different model from the llama3 generator so
# the judge is independent of the system being evaluated.
# Temperature 0.0 → deterministic scores across runs.

JUDGE_MODEL = "mistral"
JUDGE_MAX_TOKENS = 150
JUDGE_TEMPERATURE = 0.0

JUDGE_SYSTEM = (
    "You are an expert evaluator for a question-answering system. "
    "Always respond with valid JSON only — no prose, no markdown fences."
)

JUDGE_PROMPT_TEMPLATE = """You are evaluating a RAG system's answer quality.

QUESTION:
{question}

REFERENCE PASSAGE (gold answer from Wikipedia):
{gold_answer}

GENERATED ANSWER:
{generated_answer}

Evaluate the generated answer on two criteria:

1. FACTUAL ACCURACY: Are all facts in the generated answer supported by the
   reference passage? Penalise wrong entities (wrong person, wrong city, wrong
   date), wrong numbers, and unsupported claims.

2. COMPLETENESS: Does the generated answer address the core of the question?
   A concise correct answer is better than a verbose incomplete one.

Score from 1 to 5:
  5 — All facts correct, question fully answered
  4 — Facts correct, minor detail missing but core answered
  3 — Partially correct — main fact right but some errors or gaps
  2 — Significant factual error or question only tangentially addressed
  1 — Wrong answer, hallucinated facts, or completely off-topic

Return JSON only:
{{"score": <1-5>, "reasoning": "<one sentence>"}}"""


# ── Data loading ───────────────────────────────────────────────────────────────

def load_answerable_records(path: str, limit: int) -> List[Dict[str, Any]]:
    """Load CLAPnq answerable records. limit=0 means load all."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            gold = record.get("output", [{}])[0].get("answer", "").strip()
            if not gold:
                continue
            records.append({
                "id": record["id"],
                "question": record["input"],
                "gold": gold,
            })
            if limit and len(records) >= limit:
                break
    return records


# ── LLM judge ─────────────────────────────────────────────────────────────────

async def judge_answer(
    judge: BaseLLMProvider,
    question: str,
    gold: str,
    generated: str,
) -> Dict[str, Any]:
    """
    Call LLM judge and parse score + reasoning.
    Returns {"score": int 1-5, "reasoning": str}.
    Falls back to {"score": 0, "reasoning": "parse_error: ..."} on failure.
    """
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        gold_answer=gold,
        generated_answer=generated,
    )
    raw = ""
    try:
        raw = await judge.generate(
            prompt=prompt,
            system=JUDGE_SYSTEM,
            max_tokens=JUDGE_MAX_TOKENS,
            temperature=JUDGE_TEMPERATURE,
        )
        cleaned = re.sub(r"```[a-z]*\n?", "", raw).strip()
        parsed = json.loads(cleaned)
        score = int(parsed.get("score", 0))
        if not 1 <= score <= 5:
            raise ValueError(f"score out of range: {score}")
        return {"score": score, "reasoning": str(parsed.get("reasoning", ""))}
    except Exception as exc:
        logger.warning(f"Judge parse error ({exc}) | raw={raw!r:.120}")
        return {"score": 0, "reasoning": f"parse_error: {exc}"}


# ── Evaluation loop ────────────────────────────────────────────────────────────

async def evaluate(
    records: List[Dict[str, Any]],
    store: VectorStoreFaiss,
    generator: RAGGenerator,
    judge: BaseLLMProvider,
    top_k: int,
    reranker=None,
) -> List[Dict[str, Any]]:
    results = []
    total = len(records)

    for i, rec in enumerate(records, 1):
        q = rec["question"]
        gold = rec["gold"]

        chunks = await store.search(q, top_k=top_k, reranker=reranker)
        gen_result = await generator.generate(q, chunks)
        generated = gen_result["answer"]
        question_type = gen_result.get("question_type", "FACTOID")
        grounding = gen_result.get("grounding", {})

        verdict = await judge_answer(judge, q, gold, generated)

        results.append({
            "id": rec["id"],
            "question": q,
            "question_type": question_type,
            "generated": generated,
            "gold": gold,
            "judge_score": verdict["score"],
            "judge_reasoning": verdict["reasoning"],
            "num_chunks": gen_result.get("num_chunks", 0),
            "grounding_score": grounding.get("grounding_score"),
            "is_grounded": grounding.get("is_grounded"),
        })

        if i % 10 == 0 or i == total:
            valid = [r["judge_score"] for r in results if r["judge_score"] > 0]
            mean = sum(valid) / len(valid) if valid else 0.0
            logger.info(
                f"[{i}/{total}] mean_score={mean:.2f} | "
                f"last={verdict['score']} | {q[:55]}"
            )

    return results


# ── Aggregation ────────────────────────────────────────────────────────────────

def aggregate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Mean score and pass rate (score >= 4) per question type and overall."""
    by_type: Dict[str, List[int]] = defaultdict(list)
    all_scores: List[int] = []

    for r in results:
        s = r["judge_score"]
        if s == 0:
            continue
        by_type[r["question_type"]].append(s)
        all_scores.append(s)

    def stats(scores: List[int]) -> Dict[str, Any]:
        if not scores:
            return {"mean": 0.0, "pass_rate": 0.0, "n": 0}
        return {
            "mean": round(sum(scores) / len(scores), 3),
            "pass_rate": round(sum(1 for s in scores if s >= 4) / len(scores), 3),
            "n": len(scores),
        }

    return {
        "overall": stats(all_scores),
        "by_question_type": {qt: stats(sc) for qt, sc in sorted(by_type.items())},
        "parse_errors": sum(1 for r in results if r["judge_score"] == 0),
        "total_evaluated": len(results),
    }


def save_failures(
    results: List[Dict[str, Any]],
    path: str,
) -> int:
    """Write all failed records (score < 4) to a readable markdown file.

    Returns the number of failures written.
    """
    failures = [r for r in results if 1 <= r["judge_score"] <= 3]
    if not failures:
        logger.info("No failures (score < 4) — skipping failures file.")
        return 0

    lines = [
        "# Evaluation Failures (score < 4)",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
        f"Total failures: {len(failures)}",
        "",
    ]

    for i, r in enumerate(failures, 1):
        lines += [
            f"---",
            f"## [{i}] Score {r['judge_score']}/5 — {r['question_type']}",
            f"",
            f"**Question:** {r['question']}",
            f"",
            f"**Judge verdict:** {r['judge_reasoning']}",
            f"",
            f"**Generated answer:**",
            f"```",
            r["generated"].strip(),
            f"```",
            f"",
            f"**Gold answer (Wikipedia):**",
            f"```",
            r["gold"].strip(),
            f"```",
            f"",
        ]

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return len(failures)


def print_report(agg: Dict[str, Any]) -> None:
    print(f"\n{'='*60}")
    print("PHASE 2.5 — LLM-as-Judge Evaluation Results")
    print("="*60)

    ov = agg["overall"]
    print(f"\nOVERALL  mean={ov['mean']:.3f}  pass_rate={ov['pass_rate']:.1%}  n={ov['n']}")

    print(f"\n{'Type':<14} {'Mean':>6} {'Pass≥4':>8} {'N':>5} {'Failures':>9}")
    print("-"*48)
    for qt, s in agg["by_question_type"].items():
        failures = round(s["n"] * (1 - s["pass_rate"]))
        print(f"{qt:<14} {s['mean']:>6.3f} {s['pass_rate']:>7.1%} {s['n']:>5} {failures:>9}")

    if agg["parse_errors"]:
        print(f"\nParse errors (excluded from scores): {agg['parse_errors']}")

    print(f"\nRubric: 5=Excellent  4=Good  3=Partial  2=Poor  1=Fail")
    print("Pass rate = fraction of scores >= 4")


# ── CLI ────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase 2.5 — LLM-as-Judge generation evaluation")
    p.add_argument("--provider", default=None,
                   help="Generator provider: ollama | anthropic | openai (default: $LLM_PROVIDER or ollama)")
    p.add_argument("--model", default=None,
                   help="Generator model override (e.g. claude-sonnet-4-6)")
    p.add_argument("--judge-model", default=JUDGE_MODEL,
                   help=f"Ollama judge model — must be different from generator (default: {JUDGE_MODEL})")
    p.add_argument("--limit", type=int, default=50,
                   help="Max records to evaluate. 0 = full dataset (default: 50)")
    p.add_argument("--top-k", type=int, default=5,
                   help="Chunks to retrieve per query (default: 5)")
    p.add_argument("--score-threshold", type=float, default=0.3,
                   help="Min score for a chunk to reach the LLM (default: 0.3).")
    p.add_argument("--rerank", action="store_true",
                   help="Enable cross-encoder reranking after FAISS retrieval")
    p.add_argument("--rerank-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2",
                   help="Cross-encoder model (default: ms-marco-MiniLM-L-6-v2)")
    p.add_argument("--data-path", default="./data/clapnq_train_answerable.jsonl")
    p.add_argument("--db-path", default="./data/vectordb/chunks.db")
    p.add_argument("--index-path", default="./data/vectordb/chunks.faiss")
    p.add_argument("--output", default=None,
                   help="Output JSON path (default: data/generation_eval_<timestamp>.json)")
    p.add_argument("--failures-output", default=None,
                   help="Failures markdown path (default: data/failures_<timestamp>.md). "
                        "Pass 'none' to disable.")
    return p


async def main(args: argparse.Namespace) -> None:
    logger.info("="*60)
    logger.info("PHASE 2.5 — GENERATION EVALUATION (LLM-as-Judge)")
    logger.info("="*60)

    records = load_answerable_records(args.data_path, args.limit)
    logger.info(f"Loaded {len(records)} records")

    store = VectorStoreFaiss(db_path=args.db_path, index_path=args.index_path)

    gen_provider = get_provider(provider=args.provider, model=args.model)
    generator = RAGGenerator(
        provider=gen_provider,
        max_tokens=512,
        temperature=0.3,
        multi_passage=True,
        score_threshold=args.score_threshold,
    )
    logger.info(
        f"Generator : {gen_provider.provider_name} | "
        f"multi_passage=True | score_threshold={args.score_threshold}"
    )

    judge = get_provider(provider="ollama", model=args.judge_model)
    logger.info(f"Judge     : {judge.provider_name} | temp={JUDGE_TEMPERATURE}")

    reranker = None
    if args.rerank:
        from src.retrieval.reranker import CrossEncoderReranker
        reranker = CrossEncoderReranker(model_name=args.rerank_model, top_n=args.top_k)
        logger.info(f"Reranker  : {args.rerank_model} | top_n={args.top_k}")

    t0 = time.perf_counter()
    results = await evaluate(records, store, generator, judge, top_k=args.top_k, reranker=reranker)
    elapsed = time.perf_counter() - t0

    agg = aggregate(results)
    print_report(agg)
    logger.info(f"Done in {elapsed:.1f}s ({elapsed/len(records):.1f}s/record)")

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    output_path = args.output or f"./data/generation_eval_{ts}.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "num_records": len(records),
                "generator": gen_provider.provider_name,
                "judge": judge.provider_name,
                "top_k": args.top_k,
                "elapsed_seconds": round(elapsed, 1),
            },
            "summary": agg,
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {output_path}")

    if getattr(args, "failures_output", None) != "none":
        failures_path = args.failures_output or f"./data/failures_{ts}.md"
        n_failures = save_failures(results, failures_path)
        if n_failures:
            logger.info(f"Failures ({n_failures}) saved to {failures_path}")


if __name__ == "__main__":
    asyncio.run(main(build_parser().parse_args()))
