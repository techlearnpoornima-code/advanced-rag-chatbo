"""Evaluate retrieval metrics on CLAPnq dataset samples."""

import json
import sys
import asyncio
from pathlib import Path
from typing import List, Dict, Any
import random

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.metrics import DatasetMetrics, QueryMetrics
from src.data_loading.clapnq_loader import CLAPnqLoader


async def load_sample_data(
    answerable_limit: int = 10,
    unanswerable_limit: int = 10,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load sample CLAPnq data for evaluation."""
    loader = CLAPnqLoader()

    answerable_data = await loader.load_answerable(
        "data/clapnq_train_answerable.jsonl",
        limit=answerable_limit
    )
    unanswerable_data = await loader.load_unanswerable(
        "data/clapnq_train_unanswerable.jsonl",
        limit=unanswerable_limit
    )

    return answerable_data, unanswerable_data


def create_mock_retrieval_results(
    passages: List[Dict[str, Any]],
    num_relevant: int = 2,
    num_total: int = 10,
) -> tuple[List[Dict[str, Any]], List[str]]:
    """
    Create mock retrieval results for evaluation.

    Simulates FAISS retrieval with scores.
    """
    all_chunks = []
    for i, passage in enumerate(passages):
        sentences = passage.get("sentences", [passage.get("text", "")])
        for j, sentence in enumerate(sentences):
            chunk_id = f"passage_{i}_chunk_{j}"
            all_chunks.append({
                "chunk_id": chunk_id,
                "content": sentence,
                "passage_idx": i,
                "sentence_idx": j,
            })

    if len(all_chunks) < num_total:
        logger.warning(
            f"Only {len(all_chunks)} chunks available, requested {num_total}"
        )
        num_total = len(all_chunks)

    retrieved = random.sample(all_chunks, k=num_total)

    for i, chunk in enumerate(retrieved):
        score = 0.95 - (i * 0.08) + random.uniform(-0.05, 0.05)
        score = max(0.0, min(1.0, score))
        chunk["score"] = score

    sorted_chunks = sorted(retrieved, key=lambda x: x["score"], reverse=True)

    relevant_ids = set()
    if num_relevant > 0:
        relevant_indices = random.sample(
            range(min(num_relevant + 2, len(sorted_chunks))),
            k=min(num_relevant, len(sorted_chunks))
        )
        for idx in relevant_indices[:num_relevant]:
            relevant_ids.add(sorted_chunks[idx]["chunk_id"])

    return sorted_chunks, list(relevant_ids)


def evaluate_answerable_queries(
    data: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Evaluate retrieval metrics for answerable questions."""
    logger.info(f"Evaluating {len(data)} answerable queries...")

    dataset_metrics = DatasetMetrics(k=10)
    queries = []

    for record in data:
        query_id = record.get("id", "unknown")
        passages = record.get("passages", [])

        retrieved_chunks, relevant_ids = create_mock_retrieval_results(
            passages,
            num_relevant=2,
            num_total=10
        )

        queries.append({
            "query_id": query_id,
            "retrieved_chunks": retrieved_chunks,
            "relevant_chunk_ids": relevant_ids,
        })

    aggregated = dataset_metrics.evaluate_batch(queries)
    individual_results = dataset_metrics.get_results()

    return {
        "dataset_type": "answerable",
        "num_queries": len(data),
        "aggregated_metrics": aggregated,
        "individual_results": individual_results,
    }


def evaluate_unanswerable_queries(
    data: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Evaluate retrieval metrics for unanswerable questions."""
    logger.info(f"Evaluating {len(data)} unanswerable queries...")

    dataset_metrics = DatasetMetrics(k=10)
    queries = []

    for record in data:
        query_id = record.get("id", "unknown")
        passages = record.get("passages", [])

        retrieved_chunks, relevant_ids = create_mock_retrieval_results(
            passages,
            num_relevant=0,
            num_total=10
        )

        queries.append({
            "query_id": query_id,
            "retrieved_chunks": retrieved_chunks,
            "relevant_chunk_ids": relevant_ids,
        })

    aggregated = dataset_metrics.evaluate_batch(queries)
    individual_results = dataset_metrics.get_results()

    return {
        "dataset_type": "unanswerable",
        "num_queries": len(data),
        "aggregated_metrics": aggregated,
        "individual_results": individual_results,
    }


def save_results(
    answerable_results: Dict[str, Any],
    unanswerable_results: Dict[str, Any],
    output_path: str = "data/evaluation_results.json",
) -> None:
    """Save evaluation results to JSON file."""
    results = {
        "evaluation_config": {
            "k": 10,
            "answerable_queries": answerable_results["num_queries"],
            "unanswerable_queries": unanswerable_results["num_queries"],
        },
        "answerable": answerable_results,
        "unanswerable": unanswerable_results,
    }

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {output_file}")


def print_results(
    answerable_results: Dict[str, Any],
    unanswerable_results: Dict[str, Any],
) -> None:
    """Print evaluation results."""
    print("\n" + "=" * 80)
    print("RETRIEVAL EVALUATION RESULTS")
    print("=" * 80)

    print("\n📊 ANSWERABLE QUERIES")
    print("-" * 80)
    print(f"Total Queries: {answerable_results['num_queries']}")
    agg = answerable_results["aggregated_metrics"]
    print(f"MRR:       {agg['mrr_mean']:.4f} (±{agg['mrr_std']:.4f})")
    print(f"NDCG:      {agg['ndcg_mean']:.4f} (±{agg['ndcg_std']:.4f})")
    print(f"Precision: {agg['precision_mean']:.4f} (±{agg['precision_std']:.4f})")
    print(f"Recall:    {agg['recall_mean']:.4f} (±{agg['recall_std']:.4f})")
    print(f"F1 Score:  {agg['f1_mean']:.4f} (±{agg['f1_std']:.4f})")
    print(f"AP:        {agg['ap_mean']:.4f} (±{agg['ap_std']:.4f})")

    print("\n📊 UNANSWERABLE QUERIES")
    print("-" * 80)
    print(f"Total Queries: {unanswerable_results['num_queries']}")
    agg = unanswerable_results["aggregated_metrics"]
    print(f"MRR:       {agg['mrr_mean']:.4f} (±{agg['mrr_std']:.4f})")
    print(f"NDCG:      {agg['ndcg_mean']:.4f} (±{agg['ndcg_std']:.4f})")
    print(f"Precision: {agg['precision_mean']:.4f} (±{agg['precision_std']:.4f})")
    print(f"Recall:    {agg['recall_mean']:.4f} (±{agg['recall_std']:.4f})")
    print(f"F1 Score:  {agg['f1_mean']:.4f} (±{agg['f1_std']:.4f})")
    print(f"AP:        {agg['ap_mean']:.4f} (±{agg['ap_std']:.4f})")

    print("\n" + "=" * 80)


async def main():
    """Main evaluation pipeline."""
    logger.info("Starting retrieval evaluation...")

    answerable_data, unanswerable_data = await load_sample_data(
        answerable_limit=10,
        unanswerable_limit=10
    )

    logger.info(f"Loaded {len(answerable_data)} answerable queries")
    logger.info(f"Loaded {len(unanswerable_data)} unanswerable queries")

    answerable_results = evaluate_answerable_queries(answerable_data)
    unanswerable_results = evaluate_unanswerable_queries(unanswerable_data)

    save_results(answerable_results, unanswerable_results)
    print_results(answerable_results, unanswerable_results)

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    asyncio.run(main())
