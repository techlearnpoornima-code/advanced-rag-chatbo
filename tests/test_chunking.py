"""Tests for semantic chunking."""

import pytest
from src.chunking.semantic_chunker import SemanticChunker


class TestSemanticChunker:
    """Tests for SemanticChunker."""

    @pytest.fixture
    def chunker(self):
        return SemanticChunker(max_tokens=100)

    def test_chunker_initialization(self, chunker):
        """Chunker should initialize with config."""
        assert chunker.max_tokens == 100
        assert chunker.min_tokens == 50
        assert chunker.overlap_sentences == 1

    def test_chunk_simple_passage(self, chunker, sample_answerable_record):
        """Should chunk a simple passage."""
        passages = sample_answerable_record['passages']
        chunks, metadata = chunker.chunk_passages(passages)

        assert len(chunks) > 0
        assert len(metadata) == len(chunks)

    def test_metadata_structure(self, chunker, sample_answerable_record):
        """Metadata should have required fields."""
        passages = sample_answerable_record['passages']
        chunks, metadata = chunker.chunk_passages(passages)

        for meta in metadata:
            assert 'chunk_id' in meta
            assert 'passage_title' in meta
            assert 'sentence_indices' in meta
            assert 'token_count' in meta
            assert 'contains_answer' in meta

    def test_answer_detection(self, chunker, sample_answerable_record):
        """Should detect chunks containing answers."""
        passages = sample_answerable_record['passages']
        chunks, metadata = chunker.chunk_passages(passages)

        answer_chunks = [m for m in metadata if m['contains_answer']]
        assert len(answer_chunks) > 0

    def test_answer_not_in_unanswerable(self, chunker, sample_unanswerable_record):
        """Unanswerable passages should have no answer chunks."""
        passages = sample_unanswerable_record['passages']
        chunks, metadata = chunker.chunk_passages(passages)

        answer_chunks = [m for m in metadata if m['contains_answer']]
        assert len(answer_chunks) == 0

    def test_token_limit_respected(self, chunker, sample_answerable_record):
        """Chunks should not exceed token limit."""
        passages = sample_answerable_record['passages']
        chunks, metadata = chunker.chunk_passages(passages)

        for meta in metadata:
            assert meta['token_count'] <= chunker.max_tokens

    def test_statistics(self, chunker, sample_answerable_record):
        """Statistics should be computed correctly."""
        passages = sample_answerable_record['passages']
        chunks, metadata = chunker.chunk_passages(passages)

        stats = chunker.get_statistics(chunks, metadata)

        assert 'total_chunks' in stats
        assert 'total_tokens' in stats
        assert 'chunks_with_answers' in stats
        assert 'token_stats' in stats
