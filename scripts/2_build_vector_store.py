#!/usr/bin/env python3
"""
Phase 1.2 - Build Vector Store (FAISS + SQLite)

This script:
1. Loads chunks from Phase 1.1 (JSONL format)
2. Generates embeddings using SentenceTransformer
3. Builds FAISS HNSW index for fast semantic search
4. Stores metadata in SQLite for retrieval
5. Saves both index and database to disk
"""

import asyncio
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

from loguru import logger

from src.retrieval.vector_store_faiss import VectorStoreFaiss


async def load_chunks_from_jsonl(filepath: str, limit: int = None) -> tuple:
    """
    Load chunks from JSONL file produced by Phase 1.1.

    Args:
        filepath: Path to chunks JSONL file
        limit: Limit number of chunks to load (for testing)

    Returns:
        Tuple of (chunks_list, metadatas_list)
    """
    chunks = []
    metadatas = []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break

                record = json.loads(line.strip())
                chunks.append(record.get('chunk_text', ''))
                metadatas.append(record.get('metadata', {}))

        logger.info(f"Loaded {len(chunks)} chunks from {filepath}")
        return chunks, metadatas

    except FileNotFoundError:
        logger.error(f"Chunks file not found: {filepath}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in chunks file: {e}")
        raise


async def main(args):
    """Main entry point."""
    logger.info("="*70)
    logger.info("PHASE 1.2 - BUILD VECTOR STORE (FAISS + SQLite)")
    logger.info("="*70)

    # 1. Load chunks
    logger.info("")
    logger.info("STEP 1: Loading chunks from Phase 1.1...")
    chunks_file = Path(args.chunks_file)

    if not chunks_file.exists():
        logger.error(f"Chunks file not found: {chunks_file}")
        raise FileNotFoundError(f"Chunks file not found: {chunks_file}")

    chunks, metadatas = await load_chunks_from_jsonl(
        str(chunks_file),
        limit=args.limit
    )

    if not chunks:
        logger.error("No chunks loaded. Exiting.")
        return

    logger.info(f"Loaded {len(chunks)} chunks")

    # 2. Validate chunks
    logger.info("")
    logger.info("STEP 2: Validating chunks...")
    valid_chunks = []
    valid_metadatas = []

    for chunk, metadata in zip(chunks, metadatas):
        if not chunk or not chunk.strip():
            logger.warning(f"Skipping empty chunk: {metadata.get('chunk_id', 'unknown')}")
            continue

        if 'chunk_id' not in metadata:
            logger.warning(f"Skipping chunk without chunk_id: {chunk[:50]}")
            continue

        valid_chunks.append(chunk)
        valid_metadatas.append(metadata)

    logger.info(f"Validated {len(valid_chunks)} chunks (removed {len(chunks) - len(valid_chunks)} invalid)")

    # 3. Initialize vector store
    logger.info("")
    logger.info("STEP 3: Initializing vector store...")
    vector_store = VectorStoreFaiss(
        db_path=args.db_path,
        index_path=args.index_path,
        embedding_model=args.embedding_model
    )

    # 4. Add chunks to vector store
    logger.info("")
    logger.info("STEP 4: Adding chunks to vector store...")
    try:
        added = await vector_store.add_documents(
            valid_chunks,
            valid_metadatas,
            batch_size=args.batch_size
        )
        logger.info(f"Successfully added {added} chunks")

    except Exception as e:
        logger.error(f"Failed to add chunks: {e}")
        raise

    # 5. Get and display statistics
    logger.info("")
    logger.info("STEP 5: Computing statistics...")
    stats = vector_store.get_stats()

    logger.info("Vector Store Statistics:")
    logger.info("  Total chunks: {}", stats.get('total_chunks'))
    logger.info("  Chunks with answers: {}", stats.get('chunks_with_answers'))
    logger.info("  Avg tokens per chunk: {}", stats.get('avg_token_count'))
    logger.info("  Embedding dimension: {}", stats.get('embedding_dim'))
    logger.info("  Embedding model: {}", stats.get('embedding_model'))
    logger.info("  DB path: {}", stats.get('db_path'))
    logger.info("  Index path: {}", stats.get('index_path'))

    # 6. Test a sample search
    logger.info("")
    logger.info("STEP 6: Testing vector store with sample query...")
    sample_query = "What is the capital of France?"
    results = await vector_store.search(sample_query, top_k=3)

    logger.info(f"Sample query: {sample_query}")
    logger.info(f"Retrieved {len(results)} results:")
    for i, result in enumerate(results, 1):
        logger.info("  [{0}] {1} (passage: {2})",
                    i,
                    result.get('chunk_text', '')[:60] + "...",
                    result.get('passage_title', 'unknown'))

    # 7. Summary
    logger.info("")
    logger.info("="*70)
    logger.info("PHASE 1.2 COMPLETE")
    logger.info("="*70)
    logger.info("Input:  {} chunks", len(valid_chunks))
    logger.info("Output: FAISS index + SQLite database")
    logger.info("")
    logger.info("Vector Store Files:")
    logger.info("  - {}", stats.get('index_path'))
    logger.info("  - {}", stats.get('db_path'))
    logger.info("")
    logger.info("Next step: Phase 1.3 - Evaluate retrieval quality")
    logger.info("  python scripts/3_evaluate_retrieval.py --index {}", stats.get('index_path'))


def create_parser():
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="Phase 1.2 - Build vector store from chunks"
    )

    parser.add_argument(
        '--chunks-file',
        type=str,
        default='./data/processed/chunks.jsonl',
        help='Path to chunks JSONL file from Phase 1.1'
    )

    parser.add_argument(
        '--db-path',
        type=str,
        default='./data/vectordb/chunks.db',
        help='Path to SQLite database file'
    )

    parser.add_argument(
        '--index-path',
        type=str,
        default='./data/vectordb/chunks.faiss',
        help='Path to FAISS index file'
    )

    parser.add_argument(
        '--embedding-model',
        type=str,
        default='all-MiniLM-L6-v2',
        help='SentenceTransformer model for embeddings'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for embedding generation'
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of chunks to process (for testing)'
    )

    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise
