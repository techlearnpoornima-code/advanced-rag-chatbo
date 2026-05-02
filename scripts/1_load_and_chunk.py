#!/usr/bin/env python3
"""
Phase 1.1 - Load CLAPnq Dataset and Apply Semantic Chunking

This script:
1. Loads CLAPnq records (answerable and unanswerable)
2. Validates record structure
3. Applies sentence-based semantic chunking
4. Computes statistics
5. Saves chunks to JSONL for Phase 1.2
"""

import asyncio
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

from loguru import logger

from src.data_loading.clapnq_loader import CLAPnqLoader
from src.chunking.semantic_chunker import SemanticChunker


async def main(args):
    """Main entry point."""
    logger.info("="*70)
    logger.info("PHASE 1.1 - LOAD & CHUNK CLAPnq DATASET")
    logger.info("="*70)

    # 1. Load data
    logger.info("")
    logger.info("STEP 1: Loading CLAPnq records...")
    loader = CLAPnqLoader()

    # Load answerable data
    answerable_path = args.data_dir / "clapnq_train_answerable.jsonl"
    logger.info("Loading answerable: {}", answerable_path)
    answerable_records = await loader.load_answerable(
        str(answerable_path),
        limit=args.limit_answerable
    )

    # Load unanswerable data
    unanswerable_path = args.data_dir / "clapnq_train_unanswerable.jsonl"
    logger.info("Loading unanswerable: {}", unanswerable_path)
    unanswerable_records = await loader.load_unanswerable(
        str(unanswerable_path),
        limit=args.limit_unanswerable
    )

    all_records = answerable_records + unanswerable_records
    logger.info("Total records loaded: {} (answerable: {}, unanswerable: {})",
                len(all_records), len(answerable_records), len(unanswerable_records))

    # 2. Compute input statistics
    logger.info("")
    logger.info("STEP 2: Computing input statistics...")
    input_stats = loader.get_statistics(all_records)
    logger.info("Input Statistics:")
    logger.info("  Total records: {}", input_stats.get('total_records'))
    logger.info("  Total passages: {}", input_stats.get('total_passages'))
    logger.info("  Question lengths (words): min={}, max={}, avg={:.1f}",
                input_stats['question_stats']['min'],
                input_stats['question_stats']['max'],
                input_stats['question_stats']['avg'])
    logger.info("  Passage lengths (words): min={}, max={}, avg={:.1f}",
                input_stats['passage_stats']['min'],
                input_stats['passage_stats']['max'],
                input_stats['passage_stats']['avg'])
    logger.info("  Answer lengths (words): min={}, max={}, avg={:.1f}",
                input_stats['answer_stats']['min'],
                input_stats['answer_stats']['max'],
                input_stats['answer_stats']['avg'])
    logger.info("  Answerable rate: {:.1%}", input_stats.get('answerable_rate', 0))

    # 3. Extract passages for chunking
    logger.info("")
    logger.info("STEP 3: Extracting passages...")
    passages = []

    for record in all_records:
        for passage in record.get('passages', []):
            passages.append(passage)

    logger.info("Total passages to chunk: {}", len(passages))

    # 4. Chunk passages
    logger.info("")
    logger.info("STEP 4: Applying semantic chunking...")
    chunker = SemanticChunker(
        max_tokens=args.max_tokens,
        min_tokens=args.min_tokens,
        overlap_sentences=args.overlap_sentences
    )

    chunks, chunk_metadata = chunker.chunk_passages(passages)
    logger.info("Total chunks created: {}", len(chunks))

    # 5. Compute output statistics
    logger.info("")
    logger.info("STEP 5: Computing chunking statistics...")
    chunk_stats = chunker.get_statistics(chunks, chunk_metadata)
    logger.info("Chunking Statistics:")
    logger.info("  Total chunks: {}", chunk_stats.get('total_chunks'))
    logger.info("  Total tokens: {}", chunk_stats.get('total_tokens'))
    logger.info("  Token stats (min/max/avg): {}/{}/{:.1f}",
                chunk_stats['token_stats']['min'],
                chunk_stats['token_stats']['max'],
                chunk_stats['token_stats']['avg'])

    # 6. Save chunks to JSONL
    logger.info("")
    logger.info("STEP 6: Saving chunks...")
    output_path = Path(args.output_dir) / args.output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for chunk_text, metadata in zip(chunks, chunk_metadata):
            record = {
                'chunk_text': chunk_text,
                'metadata': metadata
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    logger.info("Chunks saved to: {}", output_path)

    # 7. Summary
    logger.info("")
    logger.info("="*70)
    logger.info("PHASE 1.1 COMPLETE")
    logger.info("="*70)
    logger.info("Input:  {} records → {} passages", len(all_records), len(passages))
    logger.info("Output: {} chunks", len(chunks))
    logger.info("Ratio:  {:.2f}x (passages → chunks)", len(chunks) / len(passages))
    logger.info("")
    logger.info("Next step: Phase 1.2 - Build vector store")
    logger.info("  python scripts/2_build_vector_store.py --chunks {}", output_path)


def create_parser():
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="Phase 1.1 - Load CLAPnq dataset and apply semantic chunking"
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default="data",
        help="Directory containing CLAPnq JSONL files"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Directory to save chunked output"
    )

    parser.add_argument(
        "--output-file",
        type=str,
        default="chunks.jsonl",
        help="Output filename for chunks"
    )

    parser.add_argument(
        "--limit-answerable",
        type=int,
        default=None,
        help="Max answerable records to load (None = all)"
    )

    parser.add_argument(
        "--limit-unanswerable",
        type=int,
        default=None,
        help="Max unanswerable records to load (None = all)"
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens per chunk"
    )

    parser.add_argument(
        "--min-tokens",
        type=int,
        default=50,
        help="Minimum tokens per chunk"
    )

    parser.add_argument(
        "--overlap-sentences",
        type=int,
        default=1,
        help="Number of sentences to overlap between chunks"
    )

    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    asyncio.run(main(args))
