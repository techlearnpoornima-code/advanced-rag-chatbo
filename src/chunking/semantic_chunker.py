"""Sentence-based semantic chunking for CLAPnq passages."""

from typing import List, Dict, Any, Tuple

from loguru import logger


class SemanticChunker:
    """
    Chunk CLAPnq passages using sentence boundaries.

    Groups consecutive sentences until token limit reached,
    preserving semantic boundaries and answer spans.
    """

    def __init__(
        self,
        max_tokens: int = 512,
        min_tokens: int = 50,
        overlap_sentences: int = 1,
        preserve_boundaries: bool = True
    ):
        """
        Initialize semantic chunker.

        Args:
            max_tokens: Maximum tokens per chunk
            min_tokens: Minimum tokens per chunk
            overlap_sentences: Number of sentences to overlap between chunks
            preserve_boundaries: Preserve sentence boundaries
        """
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.overlap_sentences = overlap_sentences
        self.preserve_boundaries = preserve_boundaries

    def chunk_passages(
        self,
        passages: List[Dict[str, Any]]
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Split passages into semantic chunks.

        Args:
            passages: List of passage dicts with 'title', 'text', 'sentences'

        Returns:
            Tuple of:
            - chunks: List of chunk texts
            - metadata: List of metadata dicts for each chunk
        """
        chunks = []
        metadata = []

        logger.info("Chunking {} passages", len(passages))

        for passage_idx, passage in enumerate(passages):
            passage_chunks, passage_metadata = self._chunk_single_passage(
                passage, passage_idx
            )
            chunks.extend(passage_chunks)
            metadata.extend(passage_metadata)

        logger.info("Created {} chunks from {} passages", len(chunks), len(passages))
        return chunks, metadata

    def _chunk_single_passage(
        self,
        passage: Dict[str, Any],
        passage_idx: int
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Chunk a single passage."""
        sentences = passage.get('sentences', [])
        passage_title = passage.get('title', 'Unknown')

        if not sentences:
            logger.warning("Passage {} has no sentences", passage_idx)
            return [], []

        chunks = []
        metadata = []
        current_chunk_sentences = []
        current_token_count = 0

        for sent_idx, sentence in enumerate(sentences):
            sent_tokens = self._count_tokens(sentence)

            # Check if adding this sentence exceeds limit
            if (current_token_count + sent_tokens > self.max_tokens and
                    current_chunk_sentences):
                # Save current chunk
                chunk_text = " ".join(
                    [sentences[i] for i in current_chunk_sentences]
                )
                chunk_id = f"p{passage_idx}_c{len(chunks)}"

                chunks.append(chunk_text)
                metadata.append({
                    'chunk_id': chunk_id,
                    'passage_idx': passage_idx,
                    'passage_title': passage_title,
                    'sentence_indices': current_chunk_sentences.copy(),
                    'token_count': current_token_count,
                    'contains_answer': self._contains_answer_span(
                        current_chunk_sentences,
                        passage.get('output', [])
                    ),
                    'start_sentence': current_chunk_sentences[0],
                    'end_sentence': current_chunk_sentences[-1]
                })

                # Start new chunk with overlap
                if self.overlap_sentences > 0:
                    overlap_start = max(
                        0,
                        len(current_chunk_sentences) - self.overlap_sentences
                    )
                    overlap_indices = current_chunk_sentences[overlap_start:]
                    current_chunk_sentences = overlap_indices
                    current_token_count = sum(
                        self._count_tokens(sentences[i])
                        for i in current_chunk_sentences
                    )
                else:
                    current_chunk_sentences = []
                    current_token_count = 0

            # Add sentence to current chunk
            current_chunk_sentences.append(sent_idx)
            current_token_count += sent_tokens

        # Save final chunk
        if current_chunk_sentences:
            chunk_text = " ".join([sentences[i] for i in current_chunk_sentences])
            chunk_id = f"p{passage_idx}_c{len(chunks)}"

            chunks.append(chunk_text)
            metadata.append({
                'chunk_id': chunk_id,
                'passage_idx': passage_idx,
                'passage_title': passage_title,
                'sentence_indices': current_chunk_sentences.copy(),
                'token_count': current_token_count,
                'contains_answer': self._contains_answer_span(
                    current_chunk_sentences,
                    passage.get('output', [])
                ),
                'start_sentence': current_chunk_sentences[0],
                'end_sentence': current_chunk_sentences[-1]
            })

        return chunks, metadata

    def _contains_answer_span(
        self,
        chunk_sentence_indices: List[int],
        outputs: List[Dict[str, Any]]
    ) -> bool:
        """Check if chunk contains answer sentences."""
        if not outputs:
            return False

        chunk_set = set(chunk_sentence_indices)

        for output in outputs:
            answer_sentences = output.get('selected_sentences', [])
            if answer_sentences and any(s in chunk_set for s in answer_sentences):
                return True

        return False

    @staticmethod
    def _count_tokens(text: str) -> int:
        """Simple token count (whitespace-based)."""
        return len(text.split())

    def get_statistics(
        self,
        chunks: List[str],
        metadata: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compute chunking statistics."""
        if not chunks:
            return {}

        token_counts = [self._count_tokens(c) for c in chunks]
        answer_chunks = sum(1 for m in metadata if m.get('contains_answer', False))

        return {
            'total_chunks': len(chunks),
            'total_tokens': sum(token_counts),
            'chunks_with_answers': answer_chunks,
            'answer_chunk_ratio': answer_chunks / len(chunks) if chunks else 0,
            'token_stats': {
                'min': min(token_counts) if token_counts else 0,
                'max': max(token_counts) if token_counts else 0,
                'avg': sum(token_counts) / len(token_counts) if token_counts else 0
            }
        }
