#!/usr/bin/env python3
"""
Manual verification of Phase 1.1 implementation.

Tests data loading and chunking without pytest.
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loading.clapnq_loader import CLAPnqLoader
from src.chunking.semantic_chunker import SemanticChunker


def test_loader():
    """Test CLAPnqLoader."""
    print("\n" + "="*70)
    print("TEST 1: CLAPnqLoader Validation")
    print("="*70)

    loader = CLAPnqLoader()

    # Test good record
    good_record = {
        "id": "test_1",
        "input": "What is Python?",
        "passages": [{
            "title": "Python",
            "text": "Python is a language.",
            "sentences": ["Python is a language."]
        }],
        "output": [{
            "answer": "A programming language",
            "selected_sentences": [0],
            "meta": {}
        }]
    }

    assert loader._validate_record(good_record), "✗ Good record validation failed"
    print("✓ Valid record accepted")

    # Test bad record (missing input)
    bad_record = good_record.copy()
    del bad_record['input']
    assert not loader._validate_record(bad_record), "✗ Invalid record accepted"
    print("✓ Invalid record rejected")

    # Test statistics
    stats = loader.get_statistics([good_record])
    assert 'question_stats' in stats, "✗ Statistics missing question_stats"
    assert 'passage_stats' in stats, "✗ Statistics missing passage_stats"
    print("✓ Statistics computed correctly")

    print("\n✅ CLAPnqLoader tests passed")


def test_chunker():
    """Test SemanticChunker."""
    print("\n" + "="*70)
    print("TEST 2: SemanticChunker")
    print("="*70)

    chunker = SemanticChunker(max_tokens=100)

    # Test simple passage
    passage = {
        "title": "Test",
        "text": "Sentence 1. Sentence 2. Sentence 3. Sentence 4. Sentence 5.",
        "sentences": [
            "Sentence 1.",
            "Sentence 2.",
            "Sentence 3.",
            "Sentence 4.",
            "Sentence 5."
        ],
        "output": [{
            "answer": "Answer",
            "selected_sentences": [1, 2],
            "meta": {}
        }]
    }

    chunks, metadata = chunker.chunk_passages([passage])

    assert len(chunks) > 0, "✗ No chunks created"
    print(f"✓ Created {len(chunks)} chunks from 1 passage")

    assert len(metadata) == len(chunks), "✗ Metadata count mismatch"
    print("✓ Metadata matches chunks")

    # Verify metadata structure
    for meta in metadata:
        required = ['chunk_id', 'passage_title', 'sentence_indices', 'token_count']
        for key in required:
            assert key in meta, f"✗ Missing metadata key: {key}"
    print("✓ Metadata structure valid")

    # Check answer detection
    answer_chunks = [m for m in metadata if m['contains_answer']]
    assert len(answer_chunks) > 0, "✗ Answer not detected"
    print("✓ Answer detection working")

    # Check token limits
    for meta in metadata:
        assert meta['token_count'] <= chunker.max_tokens, "✗ Token limit exceeded"
    print("✓ Token limits respected")

    # Statistics
    stats = chunker.get_statistics(chunks, metadata)
    assert 'total_chunks' in stats, "✗ Statistics missing"
    print(f"✓ Statistics: {stats['total_chunks']} chunks, "
          f"{stats['chunks_with_answers']} with answers")

    print("\n✅ SemanticChunker tests passed")


def test_integration():
    """Test integration."""
    print("\n" + "="*70)
    print("TEST 3: Integration")
    print("="*70)

    loader = CLAPnqLoader()
    chunker = SemanticChunker()

    # Create sample records
    records = [
        {
            "id": "rec_1",
            "input": "Question 1?",
            "passages": [{
                "title": "Article 1",
                "text": "Text 1. Text 2. Text 3.",
                "sentences": ["Text 1.", "Text 2.", "Text 3."]
            }],
            "output": [{
                "answer": "Answer 1",
                "selected_sentences": [0],
                "meta": {}
            }]
        },
        {
            "id": "rec_2",
            "input": "Question 2?",
            "passages": [{
                "title": "Article 2",
                "text": "Content A. Content B.",
                "sentences": ["Content A.", "Content B."]
            }],
            "output": [{
                "answer": "",
                "selected_sentences": [],
                "meta": {}
            }]
        }
    ]

    # Validate
    for record in records:
        assert loader._validate_record(record), "✗ Validation failed"
    print(f"✓ Validated {len(records)} records")

    # Extract passages
    passages = [p for r in records for p in r.get('passages', [])]
    print(f"✓ Extracted {len(passages)} passages")

    # Chunk
    chunks, metadata = chunker.chunk_passages(passages)
    print(f"✓ Created {len(chunks)} chunks")

    # Statistics
    input_stats = loader.get_statistics(records)
    chunk_stats = chunker.get_statistics(chunks, metadata)

    print(f"\nInput: {input_stats['total_records']} records → {len(passages)} passages")
    print(f"Output: {len(chunks)} chunks")
    print(f"Expansion ratio: {len(chunks) / len(passages):.2f}x")

    print("\n✅ Integration test passed")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("PHASE 1.1 VERIFICATION")
    print("="*70)

    try:
        test_loader()
        test_chunker()
        test_integration()

        print("\n" + "="*70)
        print("ALL TESTS PASSED ✅")
        print("="*70)
        print("\nPhase 1.1 is ready for real data testing!")
        print("Next: python scripts/1_load_and_chunk.py")

        return 0

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
