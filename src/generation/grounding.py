"""Answer grounding and hallucination detection for RAG generation."""

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from loguru import logger


@dataclass
class GroundingResult:
    """Result of grounding verification for a generated answer."""

    is_grounded: bool
    grounding_score: float          # 0.0–1.0 best-chunk hybrid score
    best_chunk_id: str              # chunk_id with highest grounding score
    evidence_phrases: List[str]     # multi-word answer phrases found in chunks
    chunk_scores: Dict[str, float]  # per chunk_id score (for debugging/inspection)


class GroundingVerifier:
    """
    Verify a generated answer is grounded in retrieved passages.

    Hybrid algorithm: lexical word-overlap + semantic cosine similarity.

      Lexical  = fraction of answer words that appear in chunk text.
      Semantic = cosine similarity between answer embedding and chunk embedding.
      Hybrid   = lexical_weight * lexical + semantic_weight * semantic
                 (weights are normalised; falls back to lexical-only if no embed_fn).

    The overall grounding_score is the maximum hybrid score across all chunks;
    a single highly-matching chunk is sufficient to ground the answer.

    Args:
        lexical_weight:      Contribution of lexical score (default 0.5).
        semantic_weight:     Contribution of semantic score (default 0.5).
        grounding_threshold: Minimum score for is_grounded=True (default 0.4).
        min_phrase_length:   Minimum word count for evidence phrases (default 3).
        embed_fn:            Optional callable (text: str) -> List[float].
                             If None, semantic scoring is disabled and lexical
                             weight is promoted to 1.0 automatically.
    """

    def __init__(
        self,
        lexical_weight: float = 0.5,
        semantic_weight: float = 0.5,
        grounding_threshold: float = 0.4,
        min_phrase_length: int = 3,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
    ) -> None:
        self.lexical_weight = lexical_weight
        self.semantic_weight = semantic_weight
        self.grounding_threshold = grounding_threshold
        self.min_phrase_length = min_phrase_length
        self.embed_fn = embed_fn

        if embed_fn is None and semantic_weight > 0:
            logger.warning(
                "GroundingVerifier: embed_fn not provided; "
                "semantic scoring disabled, lexical weight promoted to 1.0."
            )

    def verify(
        self,
        answer: str,
        chunks: List[Dict[str, Any]],
    ) -> GroundingResult:
        """
        Verify the answer is grounded in the provided chunks.

        Args:
            answer: Generated answer text.
            chunks: Source chunks fed to the LLM; each must have
                    'chunk_id' and 'chunk_text' fields.

        Returns:
            GroundingResult with verdict, score, best chunk id, and evidence phrases.
        """
        if not answer.strip() or not chunks:
            return GroundingResult(
                is_grounded=False,
                grounding_score=0.0,
                best_chunk_id="",
                evidence_phrases=[],
                chunk_scores={},
            )

        lex_w, sem_w = self._effective_weights()

        # Embed answer once; reused across all chunk comparisons.
        answer_embedding: Optional[List[float]] = None
        if sem_w > 0 and self.embed_fn is not None:
            try:
                answer_embedding = self.embed_fn(answer)
            except Exception as exc:
                logger.warning(f"Answer embedding failed ({exc}); falling back to lexical-only.")
                lex_w, sem_w = 1.0, 0.0

        chunk_scores: Dict[str, float] = {}
        best_score = 0.0
        best_chunk_id = chunks[0].get("chunk_id", "")
        best_chunk_text = chunks[0].get("chunk_text", "")

        for chunk in chunks:
            chunk_id = chunk.get("chunk_id", "")
            chunk_text = chunk.get("chunk_text", "")
            if not chunk_text:
                chunk_scores[chunk_id] = 0.0
                continue

            lex_score = self._lexical_score(answer, chunk_text)

            sem_score = 0.0
            if sem_w > 0 and answer_embedding is not None:
                sem_score = self._semantic_score(answer_embedding, chunk_text)

            hybrid = round(lex_w * lex_score + sem_w * sem_score, 4)
            chunk_scores[chunk_id] = hybrid

            if hybrid > best_score:
                best_score = hybrid
                best_chunk_id = chunk_id
                best_chunk_text = chunk_text

        logger.debug(
            f"Grounding: score={best_score:.4f} threshold={self.grounding_threshold} "
            f"best_chunk={best_chunk_id} chunks_checked={len(chunks)}"
        )

        return GroundingResult(
            is_grounded=best_score >= self.grounding_threshold,
            grounding_score=best_score,
            best_chunk_id=best_chunk_id,
            evidence_phrases=self._extract_evidence_phrases(answer, best_chunk_text),
            chunk_scores=chunk_scores,
        )

    # ── private helpers ────────────────────────────────────────────────────

    def _effective_weights(self) -> tuple[float, float]:
        """Normalised (lexical_weight, semantic_weight), adjusted when embed_fn is absent."""
        if self.embed_fn is None:
            return 1.0, 0.0
        total = self.lexical_weight + self.semantic_weight
        if total <= 0:
            return 0.5, 0.5
        return self.lexical_weight / total, self.semantic_weight / total

    def _lexical_score(self, answer: str, chunk_text: str) -> float:
        """Fraction of answer words that appear in chunk text (word-level overlap)."""
        answer_words = set(re.findall(r'\b\w+\b', answer.lower()))
        chunk_words = set(re.findall(r'\b\w+\b', chunk_text.lower()))
        if not answer_words:
            return 0.0
        return len(answer_words & chunk_words) / len(answer_words)

    def _semantic_score(self, answer_embedding: List[float], chunk_text: str) -> float:
        """Cosine similarity between pre-computed answer embedding and chunk text."""
        try:
            chunk_embedding = self.embed_fn(chunk_text)
            return _cosine_similarity(answer_embedding, chunk_embedding)
        except Exception as exc:
            logger.debug(f"Chunk embedding failed: {exc}")
            return 0.0

    def _extract_evidence_phrases(self, answer: str, chunk_text: str) -> List[str]:
        """Return multi-word answer phrases (>= min_phrase_length words) found verbatim in chunk_text."""
        chunk_lower = chunk_text.lower()
        words = re.findall(r'\b\w+\b', answer.lower())
        max_n = min(10, len(words))

        candidates: List[str] = []
        # Iterate longest-to-shortest to prefer longer, more specific phrases.
        for n in range(max_n, self.min_phrase_length - 1, -1):
            for i in range(len(words) - n + 1):
                phrase = " ".join(words[i : i + n])
                if phrase in chunk_lower:
                    candidates.append(phrase)

        # Keep longest non-redundant phrases; cap at 5.
        kept: List[str] = []
        for phrase in candidates:
            if not any(phrase in existing for existing in kept):
                kept.append(phrase)
            if len(kept) == 5:
                break
        return kept


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two embedding vectors."""
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    denom = float(np.linalg.norm(va)) * float(np.linalg.norm(vb))
    if denom == 0.0:
        return 0.0
    return float(np.dot(va, vb) / denom)
