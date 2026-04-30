#!/usr/bin/env python3
"""Phase 1.4 - Test Chunking Quality on CLAPnq Dataset

This script:
1. Loads chunks from Phase 1.1 (data/processed/chunks.jsonl)
2. Loads CLAPnq records for comparison
3. Measures chunking quality:
   - Answer coverage (% of questions with answer preserved in chunks)
   - Token distribution (chunk size compliance with 512-token limit)
   - Sentence integrity (no sentence boundary splits)
   - Chunk count statistics
4. Stores results in data/chunking_quality_results.json
5. Exits non-zero if critical thresholds fail
"""

import json
import sys
import asyncio
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

import numpy as np
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loading.clapnq_loader import CLAPnqLoader


def load_chunks(path: str) -> List[Dict[str, Any]]:
    """Load chunks from JSONL file."""
    chunks = []
    try:
        with open(path, 'r') as f:
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line))
        logger.info(f"Loaded {len(chunks)} chunks from {path}")
        return chunks
    except Exception as e:
        logger.error(f"Error loading chunks: {e}")
        raise


def compute_answer_coverage(
    chunks: List[Dict],
    records: List[Dict],
) -> Dict[str, Any]:
    """
    Compute answer coverage: % of answerable records with answer preserved in chunks.

    Uses two checks:
    1. Metadata flag: contains_answer=True
    2. Text-level fallback: answer text appears in chunk_text
    """
    logger.info(f"Computing answer coverage for {len(records)} answerable records...")

    passage_chunks = {}
    for chunk in chunks:
        pid = chunk["metadata"]["passage_idx"]
        if pid not in passage_chunks:
            passage_chunks[pid] = []
        passage_chunks[pid].append(chunk)

    records_with_answer_chunk = 0
    records_missing_answer = []

    for record in records:
        record_id = record.get("id", "unknown")
        answer_text = record.get("output", [{}])[0].get("answer", "")

        if not answer_text:
            continue

        found_answer = False
        for passage in record.get("passages", []):
            passage_title = passage.get("title", "")

            for chunk in chunks:
                if chunk["metadata"]["passage_title"] == passage_title:
                    if chunk["metadata"].get("contains_answer", False):
                        found_answer = True
                        break

                    if answer_text.lower() in chunk["chunk_text"].lower():
                        found_answer = True
                        break

            if found_answer:
                break

        if found_answer:
            records_with_answer_chunk += 1
        else:
            records_missing_answer.append((record_id, answer_text[:50]))

    coverage_rate = records_with_answer_chunk / len(records) if records else 0.0

    logger.info(f"Answer coverage: {records_with_answer_chunk}/{len(records)} = {coverage_rate:.4f}")
    if records_missing_answer:
        logger.warning(f"Missing answer in {len(records_missing_answer)} records")

    return {
        "records_with_answer_chunk": records_with_answer_chunk,
        "total_answerable_records": len(records),
        "coverage_rate": float(coverage_rate),
        "target": 0.95,
        "status": "PASS" if coverage_rate >= 0.95 else "FAIL",
        "records_missing_answer_count": len(records_missing_answer),
    }


def compute_token_distribution(chunks: List[Dict]) -> Dict[str, Any]:
    """Compute token size statistics."""
    logger.info(f"Computing token distribution for {len(chunks)} chunks...")

    token_counts = [chunk["metadata"]["token_count"] for chunk in chunks]
    exceeding_limit = sum(1 for tc in token_counts if tc > 512)

    stats = {
        "min": int(np.min(token_counts)),
        "max": int(np.max(token_counts)),
        "mean": float(np.mean(token_counts)),
        "std": float(np.std(token_counts)),
        "median": float(np.median(token_counts)),
        "q25": float(np.percentile(token_counts, 25)),
        "q75": float(np.percentile(token_counts, 75)),
        "total_chunks": len(chunks),
        "chunks_exceeding_limit": exceeding_limit,
        "max_tokens_config": 512,
        "status": "PASS" if exceeding_limit == 0 else "FAIL",
    }

    logger.info(f"Token stats: mean={stats['mean']:.1f}, max={stats['max']}, exceeding_limit={exceeding_limit}")
    return stats


def check_sentence_integrity(chunks: List[Dict]) -> Dict[str, Any]:
    """Verify sentence_indices form contiguous ranges."""
    logger.info(f"Checking sentence integrity for {len(chunks)} chunks...")

    violations = []

    for i, chunk in enumerate(chunks):
        sentence_indices = chunk["metadata"].get("sentence_indices", [])

        if not sentence_indices:
            violations.append(f"Chunk {i}: empty sentence_indices")
            continue

        sentence_indices_sorted = sorted(sentence_indices)
        for j in range(len(sentence_indices_sorted) - 1):
            if sentence_indices_sorted[j + 1] != sentence_indices_sorted[j] + 1:
                violations.append(
                    f"Chunk {i}: non-contiguous {sentence_indices_sorted}"
                )
                break

    violation_rate = len(violations) / len(chunks) if chunks else 0.0

    logger.info(f"Sentence integrity: {len(violations)} violations out of {len(chunks)} chunks")

    return {
        "total_chunks": len(chunks),
        "violations": len(violations),
        "violation_rate": float(violation_rate),
        "status": "PASS" if len(violations) == 0 else "FAIL",
    }


def compute_chunk_distribution(chunks: List[Dict]) -> Dict[str, Any]:
    """Compute chunks per passage statistics."""
    logger.info("Computing chunk distribution per passage...")

    passage_chunk_count = {}
    for chunk in chunks:
        pid = chunk["metadata"]["passage_idx"]
        passage_chunk_count[pid] = passage_chunk_count.get(pid, 0) + 1

    counts = list(passage_chunk_count.values())

    return {
        "total_passages": len(passage_chunk_count),
        "total_chunks": len(chunks),
        "min_chunks_per_passage": int(np.min(counts)) if counts else 0,
        "max_chunks_per_passage": int(np.max(counts)) if counts else 0,
        "mean_chunks_per_passage": float(np.mean(counts)) if counts else 0.0,
        "std_chunks_per_passage": float(np.std(counts)) if counts else 0.0,
    }


def save_results(
    answer_coverage: Dict,
    token_distribution: Dict,
    sentence_integrity: Dict,
    chunk_distribution: Dict,
    output_path: str = "data/chunking_quality_results.json",
) -> None:
    """Save results to JSON."""
    results = {
        "evaluation_date": datetime.now().isoformat(),
        "status": "COMPLETE",
        "config": {
            "max_tokens": 512,
            "min_tokens": 50,
            "chunks_file": "data/processed/chunks.jsonl",
        },
        "answer_coverage": answer_coverage,
        "token_distribution": token_distribution,
        "sentence_integrity": sentence_integrity,
        "chunk_distribution": chunk_distribution,
    }

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {output_file}")


def print_results(
    answer_coverage: Dict,
    token_distribution: Dict,
    sentence_integrity: Dict,
    chunk_distribution: Dict,
) -> None:
    """Print formatted results."""
    print("\n" + "=" * 80)
    print("CHUNKING QUALITY EVALUATION RESULTS (Phase 1.4)")
    print("=" * 80)

    print("\n📊 ANSWER COVERAGE")
    print("-" * 80)
    print(f"Records with answer preserved: {answer_coverage['records_with_answer_chunk']}/{answer_coverage['total_answerable_records']}")
    print(f"Coverage rate: {answer_coverage['coverage_rate']:.4f} (target: {answer_coverage['target']:.2f})")
    print(f"Status: {answer_coverage['status']}")

    print("\n📊 TOKEN DISTRIBUTION")
    print("-" * 80)
    print(f"Min: {token_distribution['min']}, Max: {token_distribution['max']}, Mean: {token_distribution['mean']:.1f}")
    print(f"Std Dev: {token_distribution['std']:.1f}")
    print(f"Chunks exceeding 512-token limit: {token_distribution['chunks_exceeding_limit']}")
    print(f"Status: {token_distribution['status']}")

    print("\n📊 SENTENCE INTEGRITY")
    print("-" * 80)
    print(f"Total chunks: {sentence_integrity['total_chunks']}")
    print(f"Boundary violations: {sentence_integrity['violations']}")
    print(f"Violation rate: {sentence_integrity['violation_rate']:.4f}")
    print(f"Status: {sentence_integrity['status']}")

    print("\n📊 CHUNK DISTRIBUTION")
    print("-" * 80)
    print(f"Total passages: {chunk_distribution['total_passages']}")
    print(f"Chunks per passage: min={chunk_distribution['min_chunks_per_passage']}, "
          f"max={chunk_distribution['max_chunks_per_passage']}, "
          f"mean={chunk_distribution['mean_chunks_per_passage']:.2f}")

    print("\n" + "=" * 80)
    statuses = [
        answer_coverage['status'],
        token_distribution['status'],
        sentence_integrity['status'],
    ]

    if all(s == "PASS" for s in statuses):
        print("✅ ALL QUALITY CHECKS PASSED")
    else:
        print("❌ SOME QUALITY CHECKS FAILED")

    print("=" * 80 + "\n")


async def main():
    """Main evaluation pipeline."""
    logger.info("Starting Phase 1.4 - Chunking Quality Evaluation...")

    chunks = load_chunks("data/chunks.jsonl")

    loader = CLAPnqLoader()
    answerable_records = await loader.load_answerable(
        "data/clapnq_train_answerable.jsonl",
        limit=100
    )

    logger.info(f"Loaded {len(answerable_records)} answerable records for evaluation")

    answer_coverage = compute_answer_coverage(chunks, answerable_records)
    token_distribution = compute_token_distribution(chunks)
    sentence_integrity = check_sentence_integrity(chunks)
    chunk_distribution = compute_chunk_distribution(chunks)

    save_results(
        answer_coverage,
        token_distribution,
        sentence_integrity,
        chunk_distribution,
    )

    print_results(
        answer_coverage,
        token_distribution,
        sentence_integrity,
        chunk_distribution,
    )

    critical_failures = [
        answer_coverage['status'] != "PASS",
        token_distribution['status'] != "PASS",
        sentence_integrity['status'] != "PASS",
    ]

    if any(critical_failures):
        logger.error("Critical quality checks failed!")
        return 1

    logger.info("Phase 1.4 complete!")
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
