"""Retrieval evaluation metrics for CLAPnq RAG system."""

from typing import List, Dict, Any
import numpy as np
from loguru import logger


class RetrievalMetrics:
    """Compute retrieval evaluation metrics."""

    @staticmethod
    def mean_reciprocal_rank(
        retrieved_ranks: List[int],
    ) -> float:
        """
        Mean Reciprocal Rank (MRR).

        Average of reciprocal ranks of relevant documents.

        Args:
            retrieved_ranks: List of ranks (1-indexed) where relevant docs appear

        Returns:
            MRR score (0-1, higher is better)
        """
        if not retrieved_ranks:
            return 0.0
        return float(np.mean([1.0 / rank for rank in retrieved_ranks]))

    @staticmethod
    def ndcg_score(
        relevance_scores: List[float],
        k: int = 10,
    ) -> float:
        """
        Normalized Discounted Cumulative Gain (NDCG@k).

        Measures ranking quality accounting for position.

        Args:
            relevance_scores: Relevance scores for top-k retrieved items
            k: Number of items to consider

        Returns:
            NDCG score (0-1, higher is better)
        """
        relevance_scores = relevance_scores[:k]
        if not relevance_scores:
            return 0.0

        dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance_scores))
        ideal_scores = sorted(relevance_scores, reverse=True)
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_scores))

        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def precision_at_k(
        retrieved_relevant: int,
        k: int = 10,
    ) -> float:
        """
        Precision@k.

        Fraction of retrieved documents that are relevant.

        Args:
            retrieved_relevant: Number of relevant docs in top-k
            k: Number of items retrieved

        Returns:
            Precision score (0-1, higher is better)
        """
        return retrieved_relevant / k if k > 0 else 0.0

    @staticmethod
    def recall_at_k(
        retrieved_relevant: int,
        total_relevant: int,
        k: int = 10,
    ) -> float:
        """
        Recall@k.

        Fraction of relevant documents that were retrieved.

        Args:
            retrieved_relevant: Number of relevant docs in top-k
            total_relevant: Total number of relevant docs
            k: Number of items retrieved

        Returns:
            Recall score (0-1, higher is better)
        """
        return retrieved_relevant / total_relevant if total_relevant > 0 else 0.0

    @staticmethod
    def f1_score(
        precision: float,
        recall: float,
    ) -> float:
        """
        F1 Score.

        Harmonic mean of precision and recall.

        Args:
            precision: Precision score
            recall: Recall score

        Returns:
            F1 score (0-1, higher is better)
        """
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    @staticmethod
    def average_precision(
        relevant_positions: List[int],
        k: int = 10,
    ) -> float:
        """
        Average Precision (AP).

        Average of precision values at each relevant document position.

        Args:
            relevant_positions: Positions (1-indexed) of relevant documents
            k: Number of items to consider

        Returns:
            AP score (0-1, higher is better)
        """
        relevant_positions = [p for p in relevant_positions if p <= k]
        if not relevant_positions:
            return 0.0

        precisions = [
            (i + 1) / pos for i, pos in enumerate(relevant_positions)
        ]
        return sum(precisions) / len(relevant_positions)


class QueryMetrics:
    """Compute metrics for individual queries."""

    def __init__(self, k: int = 10):
        """
        Initialize query metrics calculator.

        Args:
            k: Number of top results to evaluate
        """
        self.k = k
        self.metrics = RetrievalMetrics()

    def evaluate_query(
        self,
        query_id: str,
        retrieved_chunks: List[Dict[str, Any]],
        relevant_chunk_ids: List[str],
    ) -> Dict[str, Any]:
        """
        Evaluate retrieval quality for a single query.

        Args:
            query_id: Unique query identifier
            retrieved_chunks: List of retrieved chunks with scores
            relevant_chunk_ids: List of relevant chunk IDs for this query

        Returns:
            Dict with MRR, NDCG, Precision, Recall, F1, AP
        """
        top_k = retrieved_chunks[:self.k]
        retrieved_ids = [chunk["chunk_id"] for chunk in top_k]
        retrieved_scores = [chunk.get("score", 1.0) for chunk in top_k]

        relevant_positions = [
            i + 1 for i, chunk_id in enumerate(retrieved_ids)
            if chunk_id in relevant_chunk_ids
        ]

        num_relevant_retrieved = len(relevant_positions)
        num_relevant_total = len(relevant_chunk_ids)

        mrr = self.metrics.mean_reciprocal_rank(
            relevant_positions if relevant_positions else [self.k + 1]
        )
        ndcg = self.metrics.ndcg_score(
            [1.0 if cid in relevant_chunk_ids else 0.0 for cid in retrieved_ids],
            k=self.k
        )
        precision = self.metrics.precision_at_k(num_relevant_retrieved, self.k)
        recall = self.metrics.recall_at_k(
            num_relevant_retrieved, num_relevant_total, self.k
        )
        f1 = self.metrics.f1_score(precision, recall)
        ap = self.metrics.average_precision(relevant_positions, self.k)

        return {
            "query_id": query_id,
            "mrr": float(mrr),
            "ndcg": float(ndcg),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "ap": float(ap),
            "num_relevant": num_relevant_total,
            "num_retrieved_relevant": num_relevant_retrieved,
        }


class DatasetMetrics:
    """Aggregate metrics across multiple queries."""

    def __init__(self, k: int = 10):
        """
        Initialize dataset metrics calculator.

        Args:
            k: Number of top results to evaluate
        """
        self.k = k
        self.query_metrics = QueryMetrics(k)
        self.results = []

    def evaluate_batch(
        self,
        queries: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Evaluate retrieval metrics across multiple queries.

        Args:
            queries: List of query evaluation dicts with:
                - query_id
                - retrieved_chunks
                - relevant_chunk_ids

        Returns:
            Aggregated metrics (mean, std, median)
        """
        self.results = []

        for query in queries:
            result = self.query_metrics.evaluate_query(
                query_id=query["query_id"],
                retrieved_chunks=query["retrieved_chunks"],
                relevant_chunk_ids=query["relevant_chunk_ids"],
            )
            self.results.append(result)

        return self._aggregate_results()

    def _aggregate_results(self) -> Dict[str, Any]:
        """Aggregate results across all queries."""
        if not self.results:
            return {}

        metrics = {
            "mrr": [],
            "ndcg": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "ap": [],
        }

        for result in self.results:
            for metric in metrics:
                metrics[metric].append(result[metric])

        aggregated = {}
        for metric, values in metrics.items():
            aggregated[f"{metric}_mean"] = float(np.mean(values))
            aggregated[f"{metric}_std"] = float(np.std(values))
            aggregated[f"{metric}_median"] = float(np.median(values))
            aggregated[f"{metric}_min"] = float(np.min(values))
            aggregated[f"{metric}_max"] = float(np.max(values))

        aggregated["num_queries"] = len(self.results)
        aggregated["total_relevant"] = sum(r["num_relevant"] for r in self.results)

        return aggregated

    def get_results(self) -> List[Dict[str, Any]]:
        """Get individual query results."""
        return self.results

    def get_aggregated(self) -> Dict[str, Any]:
        """Get aggregated results."""
        return self._aggregate_results()
