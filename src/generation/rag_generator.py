"""RAG generation pipeline with multi-passage answer synthesis."""

from typing import Any, Callable, Dict, List, Optional

from loguru import logger

from src.generation.grounding import GroundingVerifier
from src.generation.providers.base import BaseLLMProvider
from src.generation.providers import get_provider
from src.generation.question_classifier import QuestionType, classify, needs_synthesis


SYSTEM_PROMPT = """You are a precise question-answering assistant.
Answer ONLY using the information in the provided passages.
Each passage is about a SPECIFIC entity (person, song, event, place). Do NOT mix information across different entities.
If the question asks about entity X, only use passages that are specifically about X.
If the passages do not contain enough information to answer, say so clearly.
Do not add facts from your training data."""


def _deduplicate_chunks(chunks: List[Dict[str, Any]], sim_threshold: float = 0.85) -> List[Dict[str, Any]]:
    """
    Remove near-duplicate chunks by checking text overlap ratio.

    Two chunks are duplicates when their shared words exceed sim_threshold
    of the shorter chunk's word count.
    """
    seen: List[set] = []
    unique = []
    for chunk in chunks:
        words = set(chunk["chunk_text"].lower().split())
        is_dup = any(
            len(words & s) / max(len(words), 1) >= sim_threshold
            for s in seen
        )
        if not is_dup:
            unique.append(chunk)
            seen.append(words)
    return unique


def _group_by_subtopic(chunks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group chunks by passage_title (each Wikipedia article = one subtopic).

    Falls back to a single "General" group if titles are absent.
    """
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for chunk in chunks:
        title = chunk.get("passage_title") or "General"
        groups.setdefault(title, []).append(chunk)
    return groups


def _build_context(groups: Dict[str, List[Dict[str, Any]]]) -> str:
    """
    Render grouped chunks into a labelled context block for the prompt.
    """
    lines = []
    for topic, topic_chunks in groups.items():
        lines.append(f"### {topic}")
        for c in topic_chunks:
            lines.append(c["chunk_text"].strip())
        lines.append("")
    return "\n".join(lines).strip()


def _build_synthesis_prompt(query: str, context: str) -> str:
    return f"""Use the passages below to answer the question.
The passages may cover multiple topics. ONLY use passages that directly answer the question.
If a passage is about a different person, song, or entity than what is asked, IGNORE it completely.
Do not mention entities or facts from irrelevant passages.
Write in clear prose. Be factual and concise.

QUESTION: {query}

PASSAGES:
{context}

ANSWER:"""


def _build_simple_prompt(query: str, context: str) -> str:
    return f"""Use the passage below to answer the question clearly and concisely.

QUESTION: {query}

PASSAGE:
{context}

ANSWER:"""


def _build_synthesis_prompt_deep(query: str, context: str) -> str:
    return f"""You are synthesizing information from multiple Wikipedia passages to answer a complex question.

Instructions:
- Read ALL passages carefully before writing your answer.
- Build a comprehensive answer that draws on evidence from MULTIPLE passages.
- Explicitly cite the passage topic when you use it, e.g. "According to the passage on [Topic], ..."
- Explain causes, effects, roles, and significance — not just bare facts.
- Write at least 3 sentences in coherent prose.
- Do NOT summarize only the first passage; integrate all relevant evidence.
- If a passage is irrelevant to the question, ignore it.

QUESTION: {query}

PASSAGES:
{context}

ANSWER:"""


# def _build_multi_hop_prompt(query: str, context: str) -> str:
#     return f"""You are answering a question that requires connecting facts across multiple passages.

# Instructions:
# - Identify the key facts from each passage that form a chain of evidence.
# - Explicitly cite the passage topic when you use it, e.g. "According to the passage on [Topic], ..."
# - Connect the facts step by step — show how information from one passage leads to or explains information from another.
# - Trace the full reasoning chain before stating your final answer.
# - Write in clear prose; do not use bullet points.

# QUESTION: {query}

# PASSAGES:
# {context}

# ANSWER:"""

def _build_multi_hop_prompt(query: str, context: str) -> str:
    return f"""Answer the question using the provided passages.

Combine evidence across the passages to form a single coherent answer.
Draw clear connections when one fact explains or causes another.
Answer directly and naturally, as if explaining to a user.
Do not mention passages, sources, or retrieval.
Do not describe your reasoning process.
Do not use bullet points.

QUESTION: {query}

PASSAGES:
{context}

ANSWER:"""


class RAGGenerator:
    """
    Retrieval-Augmented Generation with multi-passage synthesis.

    Pipeline:
      retrieve (caller) → deduplicate → group by subtopic → synthesize via LLM

    Usage:
        generator = RAGGenerator()  # uses LLM_PROVIDER env var (default: ollama)
        answer = await generator.generate(query, retrieved_chunks)
    """

    def __init__(
        self,
        provider: Optional[BaseLLMProvider] = None,
        max_tokens: int = 512,
        temperature: float = 0.3,
        dedup_threshold: float = 0.85,
        multi_passage: bool = True,
        score_threshold: float = 0.5,
        verify_grounding: bool = False,
        grounding_threshold: float = 0.4,
        lexical_weight: float = 0.5,
        semantic_weight: float = 0.5,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
    ):
        """
        Args:
            provider: LLM provider instance. If None, resolves from LLM_PROVIDER env.
            max_tokens: Max tokens in generated answer.
            temperature: Sampling temperature.
            dedup_threshold: Word-overlap ratio above which a chunk is a duplicate.
            multi_passage: If True, group by subtopic and synthesize. If False, use top chunk only.
            score_threshold: Minimum cosine similarity score for a chunk to be passed to the LLM.
                             Chunks below this threshold are dropped before synthesis to prevent
                             distractor entities from bleeding into the answer.
            verify_grounding: If True, run GroundingVerifier after generation and add
                              a 'grounding' key to the result dict.
            grounding_threshold: Minimum hybrid score for is_grounded=True (default 0.4).
            lexical_weight: Weight for word-overlap component of grounding score (default 0.5).
            semantic_weight: Weight for cosine-similarity component (default 0.5).
            embed_fn: Callable (text: str) -> List[float] for semantic grounding.
                      Pass VectorStoreFaiss.embed_text to reuse the loaded model.
                      If None and verify_grounding=True, grounding falls back to lexical-only.
        """
        self.provider = provider or get_provider()
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.dedup_threshold = dedup_threshold
        self.multi_passage = multi_passage
        self.score_threshold = score_threshold
        self._grounding_verifier: Optional[GroundingVerifier] = (
            GroundingVerifier(
                lexical_weight=lexical_weight,
                semantic_weight=semantic_weight,
                grounding_threshold=grounding_threshold,
                embed_fn=embed_fn,
            )
            if verify_grounding else None
        )

        logger.info(
            f"RAGGenerator using provider: {self.provider.provider_name}"
            + (" [grounding=ON]" if self._grounding_verifier else "")
        )

    async def generate(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Generate an answer from retrieved chunks.

        Args:
            query: User's question
            chunks: Retrieved chunks from vector store
                    Each dict must have at least: chunk_text, passage_title, chunk_id

        Returns:
            Dict with:
              answer       - generated text
              sources      - list of passage_title strings used
              chunk_ids    - list of chunk_id strings used
              num_chunks   - how many chunks fed to LLM
              provider     - provider_name string
              grounding    - (only when verify_grounding=True) dict with:
                             is_grounded, grounding_score, best_chunk_id,
                             evidence_phrases
        """
        question_type = classify(query)
        logger.info(f"Question type: {question_type.value} | query: {query[:60]}")

        if not chunks:
            return {
                "answer": "No relevant passages were retrieved for this question.",
                "sources": [],
                "chunk_ids": [],
                "num_chunks": 0,
                "provider": self.provider.provider_name,
                "question_type": question_type.value,
            }

        if self.multi_passage:
            answer, used_chunks = await self._synthesize(query, chunks, question_type)
        else:
            answer, used_chunks = await self._simple_generate(query, chunks[:1])

        result: Dict[str, Any] = {
            "answer": answer.strip(),
            "sources": list({c.get("passage_title", "") for c in used_chunks}),
            "chunk_ids": [c.get("chunk_id", "") for c in used_chunks],
            "num_chunks": len(used_chunks),
            "provider": self.provider.provider_name,
            "question_type": question_type.value,
        }

        if self._grounding_verifier is not None:
            gr = self._grounding_verifier.verify(answer, used_chunks)
            result["grounding"] = {
                "is_grounded": gr.is_grounded,
                "grounding_score": gr.grounding_score,
                "best_chunk_id": gr.best_chunk_id,
                "evidence_phrases": gr.evidence_phrases,
            }

        return result

    async def _synthesize(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        question_type: QuestionType = QuestionType.FACTOID,
    ):
        """Score-filter → deduplicate → group by subtopic → synthesize across relevant groups."""
        # Synthesis/multi-hop questions need broader evidence — use a lower threshold
        # so diverse supporting chunks are not dropped before the LLM sees them.
        effective_threshold = (
            max(0.0, self.score_threshold - 0.15)
            if needs_synthesis(question_type)
            else self.score_threshold
        )

        above_threshold = [c for c in chunks if c.get("score", 1.0) >= effective_threshold]
        if not above_threshold:
            above_threshold = chunks[:1]
            logger.warning(
                f"All {len(chunks)} chunks below score_threshold={effective_threshold}; "
                "using top-1 as fallback"
            )

        unique = _deduplicate_chunks(above_threshold, self.dedup_threshold)
        groups = _group_by_subtopic(unique)
        context = _build_context(groups)

        logger.info(
            f"Synthesis: {len(chunks)} chunks → {len(above_threshold)} above threshold "
            f"({effective_threshold:.2f}) → {len(unique)} unique → {len(groups)} topic(s) "
            f"[{question_type.value}]"
        )

        if question_type == QuestionType.MULTI_HOP:
            prompt = _build_multi_hop_prompt(query, context)
        elif needs_synthesis(question_type):
            prompt = _build_synthesis_prompt_deep(query, context)
        else:
            prompt = _build_synthesis_prompt(query, context)

        answer = await self.provider.generate(
            prompt=prompt,
            system=SYSTEM_PROMPT,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return answer, unique

    async def _simple_generate(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
    ):
        """Single-passage generation (top chunk only)."""
        context = chunks[0]["chunk_text"]
        prompt = _build_simple_prompt(query, context)
        answer = await self.provider.generate(
            prompt=prompt,
            system=SYSTEM_PROMPT,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return answer, chunks
