#!/usr/bin/env python3
"""
Phase 1.2 Verification - Test FAISS + SQLite Vector Store

Tests:
1. Vector store initialization (FAISS + SQLite)
2. Adding documents (embeddings + metadata)
3. Searching for similar chunks
4. Metadata filtering
5. Statistics computation
"""

import json
import sqlite3
from pathlib import Path

from loguru import logger

from src.retrieval.vector_store_faiss import VectorStoreFaiss


def test_vector_store_initialization():
    """Test vector store initialization."""
    logger.info("Test 1: Vector store initialization...")

    db_path = "./test_data/test_chunks.db"
    index_path = "./test_data/test_chunks.faiss"

    # Clean up old test files
    Path(db_path).unlink(missing_ok=True)
    Path(index_path).unlink(missing_ok=True)

    try:
        store = VectorStoreFaiss(db_path=db_path, index_path=index_path)
        assert store.index is not None, "FAISS index not initialized"
        assert store.chunk_count == 0, "Chunk count should be 0 initially"
        assert Path(db_path).exists(), "SQLite database file not created"

        logger.info("✓ Vector store initialized successfully")
        return True
    except Exception as e:
        logger.error(f"✗ Initialization failed: {e}")
        return False


def test_add_documents():
    """Test adding documents to vector store."""
    logger.info("Test 2: Adding documents...")

    db_path = "./test_data/test_chunks.db"
    index_path = "./test_data/test_chunks.faiss"

    Path(db_path).unlink(missing_ok=True)
    Path(index_path).unlink(missing_ok=True)

    try:
        store = VectorStoreFaiss(db_path=db_path, index_path=index_path)

        # Sample chunks
        chunks = [
            "Paris is the capital of France.",
            "London is the capital of England.",
            "Berlin is the capital of Germany."
        ]

        metadatas = [
            {
                "chunk_id": "chunk_0",
                "passage_id": "passage_0",
                "passage_title": "France",
                "sentence_indices": [0],
                "contains_answer": True,
                "token_count": 8,
                "source_file": "test.jsonl"
            },
            {
                "chunk_id": "chunk_1",
                "passage_id": "passage_1",
                "passage_title": "England",
                "sentence_indices": [0],
                "contains_answer": True,
                "token_count": 8,
                "source_file": "test.jsonl"
            },
            {
                "chunk_id": "chunk_2",
                "passage_id": "passage_2",
                "passage_title": "Germany",
                "sentence_indices": [0],
                "contains_answer": True,
                "token_count": 8,
                "source_file": "test.jsonl"
            }
        ]

        # Manually run async function
        import asyncio
        added = asyncio.run(store.add_documents(chunks, metadatas))

        assert added == 3, f"Expected 3 documents added, got {added}"
        assert store.chunk_count == 3, f"Expected chunk_count=3, got {store.chunk_count}"
        assert Path(index_path).exists(), "FAISS index not saved"

        logger.info("✓ Documents added successfully")
        return True
    except Exception as e:
        logger.error(f"✗ Adding documents failed: {e}")
        return False


def test_search():
    """Test searching vector store."""
    logger.info("Test 3: Searching vector store...")

    db_path = "./test_data/test_chunks.db"
    index_path = "./test_data/test_chunks.faiss"

    # Use existing store from previous test
    try:
        store = VectorStoreFaiss(db_path=db_path, index_path=index_path)

        import asyncio
        results = asyncio.run(store.search("What is the capital of France?", top_k=2))

        assert len(results) > 0, "No results returned"
        assert len(results) <= 2, f"Expected max 2 results, got {len(results)}"

        # First result should be about France (most similar)
        first_result = results[0]
        assert 'chunk_text' in first_result, "chunk_text not in result"
        assert 'passage_title' in first_result, "passage_title not in result"

        logger.info(f"✓ Search successful: retrieved {len(results)} results")
        logger.info(f"  Top result: {first_result['chunk_text']}")
        return True
    except Exception as e:
        logger.error(f"✗ Search failed: {e}")
        return False


def test_metadata_filtering():
    """Test filtering results by metadata."""
    logger.info("Test 4: Metadata filtering...")

    db_path = "./test_data/test_chunks.db"
    index_path = "./test_data/test_chunks.faiss"

    try:
        store = VectorStoreFaiss(db_path=db_path, index_path=index_path)

        import asyncio

        # Search with filter
        results = asyncio.run(store.search(
            "capital of France",
            top_k=5,
            filters={"passage_title": "France"}
        ))

        assert all(r['passage_title'] == 'France' for r in results), \
            "Filtering didn't work: got non-France results"

        logger.info(f"✓ Filtering successful: {len(results)} filtered results")
        return True
    except Exception as e:
        logger.error(f"✗ Filtering failed: {e}")
        return False


def test_statistics():
    """Test getting statistics."""
    logger.info("Test 5: Statistics computation...")

    db_path = "./test_data/test_chunks.db"
    index_path = "./test_data/test_chunks.faiss"

    try:
        store = VectorStoreFaiss(db_path=db_path, index_path=index_path)
        stats = store.get_stats()

        assert 'total_chunks' in stats, "total_chunks not in stats"
        assert 'chunks_with_answers' in stats, "chunks_with_answers not in stats"
        assert 'avg_token_count' in stats, "avg_token_count not in stats"
        assert stats['total_chunks'] == 3, f"Expected 3 chunks, got {stats['total_chunks']}"
        assert stats['chunks_with_answers'] == 3, f"Expected 3 with answers, got {stats['chunks_with_answers']}"

        logger.info("✓ Statistics computed successfully")
        logger.info(f"  Total chunks: {stats['total_chunks']}")
        logger.info(f"  Chunks with answers: {stats['chunks_with_answers']}")
        logger.info(f"  Avg tokens: {stats['avg_token_count']}")
        return True
    except Exception as e:
        logger.error(f"✗ Statistics failed: {e}")
        return False


def test_sqlite_schema():
    """Test SQLite schema and data integrity."""
    logger.info("Test 6: SQLite schema validation...")

    db_path = "./test_data/test_chunks.db"

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check table exists
        cursor.execute('''
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='chunks'
        ''')
        assert cursor.fetchone() is not None, "chunks table not found"

        # Check columns
        cursor.execute("PRAGMA table_info(chunks)")
        columns = {row[1] for row in cursor.fetchall()}

        expected_columns = {
            'chunk_id', 'passage_id', 'passage_title', 'chunk_text',
            'sentence_indices', 'contains_answer', 'token_count', 'source_file'
        }
        assert expected_columns.issubset(columns), f"Missing columns: {expected_columns - columns}"

        # Check data
        cursor.execute("SELECT COUNT(*) FROM chunks")
        count = cursor.fetchone()[0]
        assert count == 3, f"Expected 3 rows, got {count}"

        conn.close()
        logger.info("✓ SQLite schema valid")
        return True
    except Exception as e:
        logger.error(f"✗ Schema validation failed: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("="*70)
    logger.info("PHASE 1.2 VERIFICATION - FAISS + SQLite Vector Store")
    logger.info("="*70)
    logger.info("")

    # Create test directory
    Path("./test_data").mkdir(exist_ok=True)

    tests = [
        test_vector_store_initialization,
        test_add_documents,
        test_search,
        test_metadata_filtering,
        test_statistics,
        test_sqlite_schema
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            logger.error(f"Test crashed: {e}")
            results.append(False)
        logger.info("")

    # Summary
    logger.info("="*70)
    logger.info("VERIFICATION SUMMARY")
    logger.info("="*70)

    passed = sum(results)
    total = len(results)

    logger.info(f"Passed: {passed}/{total}")

    if passed == total:
        logger.info("✓ All tests passed! Phase 1.2 is ready.")
    else:
        logger.info(f"✗ {total - passed} test(s) failed.")

    # Cleanup
    import shutil
    shutil.rmtree("./test_data", ignore_errors=True)

    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
