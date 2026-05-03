"""Cross-encoder reranker for two-stage retrieval."""

from typing import Any, Dict, List

from loguru import logger


class CrossEncoderReranker:
    """
    Re-ranks FAISS-retrieved chunks using a cross-encoder model.

    Two-stage pipeline:
      Stage 1 — FAISS (bi-encoder): fast approximate retrieval of top-N candidates
      Stage 2 — CrossEncoder      : precise (query, chunk) joint scoring, returns top-k

    The cross-encoder reads query and chunk together so it catches exact keyword
    matches and fine-grained relevance that bi-encoder embeddings miss.

    Example — grana question:
        FAISS alone   : chunk1 (plant cells, score=0.61) > chunk2 (grana, score=0.58)
        After rerank  : chunk2 (grana exact match, rerank=8.3) > chunk1 (rerank=2.1)

    Args:
        model_name: HuggingFace cross-encoder model. ms-marco-MiniLM-L-6-v2
                    is ~85 MB, fast, and strong for passage ranking tasks.
        top_n:      Number of chunks to return after reranking.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_n: int = 5,
    ) -> None:
        self.model_name = model_name
        self.top_n = top_n
        self._model = None  # lazy-loaded on first rerank() call

    @property
    def _cross_encoder(self):
        if self._model is None:
            from sentence_transformers.cross_encoder import CrossEncoder
            logger.info(f"Loading cross-encoder: {self.model_name}")
            self._model = CrossEncoder(self.model_name)
            logger.info("Cross-encoder ready")
        return self._model

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Re-score chunks against the query and return top_n sorted by rerank score.

        Each returned chunk gains:
          rerank_score — cross-encoder relevance score (higher = more relevant)
          faiss_score  — original FAISS cosine similarity (preserved for debugging)
          score        — set to rerank_score so downstream threshold filter works unchanged

        Args:
            query:  The user's question.
            chunks: Candidates from FAISS retrieval (any length).

        Returns:
            Up to top_n chunks sorted by rerank_score descending.
        """
        if not chunks:
            return chunks

        pairs = [[query, c["chunk_text"]] for c in chunks]
        raw_scores = self._cross_encoder.predict(pairs)

        reranked = []
        for chunk, raw in zip(chunks, raw_scores):
            reranked.append({
                **chunk,
                "faiss_score": chunk.get("score", 0.0),
                "rerank_score": round(float(raw), 4),
                "score": round(float(raw), 4),
            })

        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)

        logger.info(
            f"Reranked {len(chunks)} candidates → top-{self.top_n} | "
            f"best='{reranked[0].get('passage_title', '?')}' "
            f"rerank={reranked[0]['rerank_score']:.3f} "
            f"faiss={reranked[0]['faiss_score']:.3f}"
        )
        return reranked[: self.top_n]
