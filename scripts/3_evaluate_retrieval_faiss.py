"""Real FAISS-based retrieval evaluation on CLAPnq dataset."""

import json
import sys
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np

from loguru import logger
from sentence_transformers import SentenceTransformer
import faiss

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.metrics import DatasetMetrics
from src.data_loading.clapnq_loader import CLAPnqLoader
from src.chunking.semantic_chunker import SemanticChunker


class FAISSRetriever:
    """FAISS-based retrieval for evaluation."""

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 512,
    ):
        """
        Initialize FAISS retriever.

        Args:
            embedding_model: SentenceTransformers model name
            chunk_size: Max tokens per chunk
        """
        self.embedding_model = SentenceTransformer(embedding_model)
        self.chunk_size = chunk_size
        self.chunker = SemanticChunker(max_tokens=chunk_size)

        self.chunks = []
        self.chunk_embeddings = None
        self.index = None

    def add_documents(
        self,
        records: List[Dict[str, Any]],
    ) -> None:
        """
        Add documents to FAISS index.

        Args:
            records: CLAPnq records with passages
        """
        logger.info(f"Processing {len(records)} records...")

        all_chunks = []
        chunk_texts = []

        for record_idx, record in enumerate(records):
            passages = record.get("passages", [])

            if not passages:
                continue

            chunks, metadata = self.chunker.chunk_passages(passages)

            for chunk_idx, (chunk_text, chunk_meta) in enumerate(zip(chunks, metadata)):
                passage_idx = chunk_meta.get("passage_idx", 0)
                chunk_id = f"record_{record_idx}_passage_{passage_idx}_chunk_{chunk_idx}"
                title = passages[passage_idx].get("title", "") if passage_idx < len(passages) else ""

                all_chunks.append({
                    "chunk_id": chunk_id,
                    "content": chunk_text,
                    "title": title,
                    "record_id": record.get("id", "unknown"),
                    "passage_idx": passage_idx,
                    "chunk_idx": chunk_idx,
                })
                chunk_texts.append(chunk_text)

        logger.info(f"Created {len(all_chunks)} chunks from {len(records)} records")

        self.chunks = all_chunks
        self._build_index(chunk_texts)

    def _build_index(self, texts: List[str]) -> None:
        """Build FAISS index from chunk texts."""
        logger.info(f"Generating embeddings for {len(texts)} chunks...")

        embeddings = self.embedding_model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        embeddings = embeddings.astype(np.float32)
        logger.info(f"Embedding shape: {embeddings.shape}")

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        self.chunk_embeddings = embeddings

        logger.info(f"FAISS index built with {self.index.ntotal} vectors")

    def search(
        self,
        query: str,
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of retrieved chunks with scores
        """
        if self.index is None:
            logger.error("Index not built. Call add_documents first.")
            return []

        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
        ).astype(np.float32)

        distances, indices = self.index.search(query_embedding, k)

        results = []
        for dist_val, idx_val in zip(distances[0], indices[0]):
            idx = int(idx_val)
            if idx < 0 or idx >= len(self.chunks):
                continue

            chunk = self.chunks[idx]
            score = 1.0 / (1.0 + float(dist_val))

            results.append({
                "chunk_id": chunk["chunk_id"],
                "content": chunk["content"],
                "title": chunk["title"],
                "score": score,
                "distance": float(dist_val),
            })

        return results


async def load_evaluation_data(
    answerable_limit: int = 10,
    unanswerable_limit: int = 10,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load CLAPnq data for evaluation."""
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


def extract_relevant_chunks(
    record: Dict[str, Any],
) -> List[str]:
    """Extract relevant chunk IDs from answer information."""
    relevant_chunks = set()

    passages = record.get("passages", [])
    output = record.get("output", [{}])[0]
    answer = output.get("answer", "")

    if not answer:
        return list(relevant_chunks)

    selected_sentences = output.get("selected_sentences", [])

    for passage_idx, passage in enumerate(passages):
        sentences = passage.get("sentences", [])

        for sent_idx in selected_sentences:
            try:
                sent_idx = int(sent_idx)
                if 0 <= sent_idx < len(sentences):
                    chunk_id = f"record_*_passage_{passage_idx}_chunk_{sent_idx}"
                    relevant_chunks.add(chunk_id)
            except (ValueError, TypeError):
                continue

    return list(relevant_chunks)


def evaluate_answerable_queries(
    retriever: FAISSRetriever,
    records: List[Dict[str, Any]],
    k: int = 5,
) -> Dict[str, Any]:
    """Evaluate retrieval on answerable queries."""
    logger.info(f"Evaluating {len(records)} answerable queries with FAISS...")

    dataset_metrics = DatasetMetrics(k=k)
    queries = []

    for record in records:
        query = record.get("input", "")
        record_id = record.get("id", "unknown")

        if not query:
            continue

        retrieved_chunks = retriever.search(query, k=k)

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


def evaluate_unanswerable_queries(
    retriever: FAISSRetriever,
    records: List[Dict[str, Any]],
    k: int = 5,
) -> Dict[str, Any]:
    """Evaluate retrieval on unanswerable queries."""
    logger.info(f"Evaluating {len(records)} unanswerable queries with FAISS...")

    dataset_metrics = DatasetMetrics(k=k)
    queries = []

    for record in records:
        query = record.get("input", "")
        record_id = record.get("id", "unknown")

        if not query:
            continue

        retrieved_chunks = retriever.search(query, k=k)

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
        "evaluation_type": "FAISS-based (real retrieval)",
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
    print("FAISS-BASED RETRIEVAL EVALUATION RESULTS")
    print("=" * 80)

    print("\n📊 ANSWERABLE QUERIES (FAISS)")
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

    print("\n📊 UNANSWERABLE QUERIES (FAISS)")
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
    """Main FAISS evaluation pipeline."""
    logger.info("Starting FAISS-based retrieval evaluation...")

    answerable_data, unanswerable_data = await load_evaluation_data(
        answerable_limit=10,
        unanswerable_limit=10
    )

    logger.info(f"Loaded {len(answerable_data)} answerable queries")
    logger.info(f"Loaded {len(unanswerable_data)} unanswerable queries")

    retriever = FAISSRetriever()

    all_data = answerable_data + unanswerable_data
    retriever.add_documents(all_data)

    answerable_results = evaluate_answerable_queries(
        retriever,
        answerable_data,
        k=5
    )
    unanswerable_results = evaluate_unanswerable_queries(
        retriever,
        unanswerable_data,
        k=5
    )

    save_results(answerable_results, unanswerable_results)
    print_results(answerable_results, unanswerable_results)

    logger.info("FAISS evaluation complete!")


if __name__ == "__main__":
    asyncio.run(main())
