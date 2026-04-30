"""Phase 1.3 - Evaluate Retrieval Quality (uses FAISS from Phase 1.2)

This script:
1. Loads pre-built FAISS index + SQLite from Phase 1.2
2. Loads CLAPnq evaluation queries (10 answerable + 10 unanswerable)
3. Runs semantic search against the index
4. Evaluates retrieval quality using metrics (MRR, NDCG, Precision, Recall, F1)
5. Stores results in data/evaluation_results_faiss.json

NO chunking/embedding generation - reuses Phase 1.2 output!
"""

import json
import sys
import asyncio
from pathlib import Path
from typing import List, Dict, Any

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.metrics import DatasetMetrics
from src.data_loading.clapnq_loader import CLAPnqLoader
from src.retrieval.vector_store_faiss import VectorStoreFaiss


async def load_evaluation_data(
    answerable_limit: int = 10,
    unanswerable_limit: int = 10,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load CLAPnq evaluation queries."""
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


def extract_relevant_chunks(record: Dict[str, Any]) -> List[str]:
    """Extract relevant chunk IDs from answer information."""
    relevant_chunks = []

    output = record.get("output", [{}])[0]
    answer = output.get("answer", "")

    if not answer:
        return relevant_chunks

    selected_sentences = output.get("selected_sentences", [])

    for sent_idx in selected_sentences:
        try:
            sent_idx = int(sent_idx)
            chunk_id = f"p0_c{sent_idx}"
            relevant_chunks.append(chunk_id)
        except (ValueError, TypeError):
            continue

    return relevant_chunks


async def evaluate_answerable_queries(
    vector_store: VectorStoreFaiss,
    records: List[Dict[str, Any]],
    k: int = 5,
) -> Dict[str, Any]:
    """Evaluate retrieval on answerable questions."""
    logger.info(f"Evaluating {len(records)} answerable queries (K={k})...")

    dataset_metrics = DatasetMetrics(k=k)
    queries = []

    for record in records:
        query = record.get("input", "")
        record_id = record.get("id", "unknown")

        if not query:
            continue

        retrieved_chunks = await vector_store.search(query, top_k=k)
        relevant_chunk_ids = extract_relevant_chunks(record)

        queries.append({
            "query_id": record_id,
            "query_text": query,
            "retrieved_chunks": retrieved_chunks,
            "relevant_chunk_ids": relevant_chunk_ids,
        })

    if not queries:
        logger.warning("No valid queries for answerable evaluation")
        return {
            "dataset_type": "answerable",
            "num_queries": 0,
            "aggregated_metrics": {},
            "individual_results": [],
        }

    aggregated = dataset_metrics.evaluate_batch(queries)
    individual_results = dataset_metrics.get_results()

    return {
        "dataset_type": "answerable",
        "num_queries": len(queries),
        "aggregated_metrics": aggregated,
        "individual_results": individual_results,
    }


async def evaluate_unanswerable_queries(
    vector_store: VectorStoreFaiss,
    records: List[Dict[str, Any]],
    k: int = 5,
) -> Dict[str, Any]:
    """Evaluate retrieval on unanswerable questions."""
    logger.info(f"Evaluating {len(records)} unanswerable queries (K={k})...")

    dataset_metrics = DatasetMetrics(k=k)
    queries = []

    for record in records:
        query = record.get("input", "")
        record_id = record.get("id", "unknown")

        if not query:
            continue

        retrieved_chunks = await vector_store.search(query, top_k=k)

        queries.append({
            "query_id": record_id,
            "query_text": query,
            "retrieved_chunks": retrieved_chunks,
            "relevant_chunk_ids": [],
        })

    if not queries:
        logger.warning("No valid queries for unanswerable evaluation")
        return {
            "dataset_type": "unanswerable",
            "num_queries": 0,
            "aggregated_metrics": {},
            "individual_results": [],
        }

    aggregated = dataset_metrics.evaluate_batch(queries)
    individual_results = dataset_metrics.get_results()

    return {
        "dataset_type": "unanswerable",
        "num_queries": len(queries),
        "aggregated_metrics": aggregated,
        "individual_results": individual_results,
    }


def save_results(
    answerable_results: Dict[str, Any],
    unanswerable_results: Dict[str, Any],
    output_path: str = "data/evaluation_results_faiss.json",
) -> None:
    """Save evaluation results to JSON file."""
    results = {
        "evaluation_type": "FAISS-based (Phase 1.2 index)",
        "evaluation_config": {
            "k": 5,
            "embedding_model": "all-MiniLM-L6-v2",
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
    print("FAISS-BASED RETRIEVAL EVALUATION RESULTS (Phase 1.3)")
    print("=" * 80)

    print("\n📊 ANSWERABLE QUERIES")
    print("-" * 80)
    if answerable_results["num_queries"] == 0:
        print("No queries evaluated")
    else:
        print(f"Total Queries: {answerable_results['num_queries']}")
        agg = answerable_results["aggregated_metrics"]
        print(f"MRR:       {agg.get('mrr_mean', 0):.4f} (±{agg.get('mrr_std', 0):.4f})")
        print(f"NDCG:      {agg.get('ndcg_mean', 0):.4f} (±{agg.get('ndcg_std', 0):.4f})")
        print(f"Precision: {agg.get('precision_mean', 0):.4f} (±{agg.get('precision_std', 0):.4f})")
        print(f"Recall:    {agg.get('recall_mean', 0):.4f} (±{agg.get('recall_std', 0):.4f})")
        print(f"F1 Score:  {agg.get('f1_mean', 0):.4f} (±{agg.get('f1_std', 0):.4f})")
        print(f"AP:        {agg.get('ap_mean', 0):.4f} (±{agg.get('ap_std', 0):.4f})")

    print("\n📊 UNANSWERABLE QUERIES")
    print("-" * 80)
    if unanswerable_results["num_queries"] == 0:
        print("No queries evaluated")
    else:
        print(f"Total Queries: {unanswerable_results['num_queries']}")
        agg = unanswerable_results["aggregated_metrics"]
        print(f"MRR:       {agg.get('mrr_mean', 0):.4f} (±{agg.get('mrr_std', 0):.4f})")
        print(f"NDCG:      {agg.get('ndcg_mean', 0):.4f} (±{agg.get('ndcg_std', 0):.4f})")
        print(f"Precision: {agg.get('precision_mean', 0):.4f} (±{agg.get('precision_std', 0):.4f})")
        print(f"Recall:    {agg.get('recall_mean', 0):.4f} (±{agg.get('recall_std', 0):.4f})")
        print(f"F1 Score:  {agg.get('f1_mean', 0):.4f} (±{agg.get('f1_std', 0):.4f})")
        print(f"AP:        {agg.get('ap_mean', 0):.4f} (±{agg.get('ap_std', 0):.4f})")

    print("\n" + "=" * 80)


async def main():
    """Main evaluation pipeline."""
    logger.info("Starting Phase 1.3 - Retrieval Evaluation...")

    vector_store = VectorStoreFaiss()

    if not vector_store.index_path.exists():
        logger.error(f"FAISS index not found at {vector_store.index_path}")
        logger.error("Run Phase 1.2 (script 2) first to build the index")
        return

    answerable_data, unanswerable_data = await load_evaluation_data(
        answerable_limit=10,
        unanswerable_limit=10
    )

    logger.info(f"Loaded {len(answerable_data)} answerable queries")
    logger.info(f"Loaded {len(unanswerable_data)} unanswerable queries")

    answerable_results = await evaluate_answerable_queries(
        vector_store,
        answerable_data,
        k=5
    )
    unanswerable_results = await evaluate_unanswerable_queries(
        vector_store,
        unanswerable_data,
        k=5
    )

    save_results(answerable_results, unanswerable_results)
    print_results(answerable_results, unanswerable_results)

    logger.info("Phase 1.3 complete!")


if __name__ == "__main__":
    asyncio.run(main())
