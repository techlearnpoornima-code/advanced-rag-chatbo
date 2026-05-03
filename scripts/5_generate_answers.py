#!/usr/bin/env python3
"""
Phase 2 - Generation Pipeline Demo

Runs the full RAG pipeline:
  1. Load query
  2. Retrieve top-k chunks from FAISS vector store
  3. Print all retrieved chunks
  4. Deduplicate & group by subtopic
  5. Synthesize answer via LLM (default: Ollama/llama3)

Usage:
    python scripts/5_generate_answers.py
    python scripts/5_generate_answers.py --provider anthropic --model claude-sonnet-4-6
    python scripts/5_generate_answers.py --no-multi-passage
    python scripts/5_generate_answers.py --verify-grounding
    python scripts/5_generate_answers.py --verify-grounding --grounding-threshold 0.5
"""

import asyncio
import argparse
import sys
import time
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.vector_store_faiss import VectorStoreFaiss
from src.generation.rag_generator import RAGGenerator
from src.generation.providers import get_provider


# DEMO_QUERIES = [
#     # SYNTHESIS — how the US ended WWII in the Pacific
#     "How did the United States use nuclear weapons to end World War II in the Pacific?",
#     # MULTI_HOP — French and Indian War → Treaty of Fontainebleau → Louisiana cession to Spain
#     "How did France's defeat in the French and Indian War that led to the Treaty of Fontainebleau cause it to cede Louisiana to Spain?",
#     # FACTOID — Jungle Book voice actor
#     "Who sang The Bare Necessities in the 1967 Disney film The Jungle Book?",
#     # TEMPORAL — Camp David Accords
#     "When were the Camp David Accords signed?",
#     # LOCATION — US state geography
#     "Where is Oklahoma located in the United States?",
# ]

DEMO_QUERIES = ["administers the oath of office to the president",
"what does the grana do in a plant cell",
"nba players to win back to back finals mvp",
"what event caused the united states to declare war and enter world war ii",
"what do the lyrics for american pie mean"
]


async def run_query(
    query: str,
    store: VectorStoreFaiss,
    generator: RAGGenerator,
    top_k: int,
    reranker=None,
) -> None:
    print(f"\n{'='*70}")
    print(f"QUERY: {query}")
    print("="*70)

    # Retrieve (+ optional cross-encoder rerank)
    t0 = time.perf_counter()
    chunks = await store.search(query, top_k=top_k, reranker=reranker)
    retrieval_ms = (time.perf_counter() - t0) * 1000

    # Print all retrieved chunks
    print(f"\nRETRIEVED CHUNKS ({len(chunks)} of top-{top_k})")
    print("-"*70)
    for i, chunk in enumerate(chunks, 1):
        print(f"[{i}] {chunk.get('passage_title', 'unknown')}")
        print(f"    {chunk['chunk_text'].strip()}")
        print()

    # Generate
    t1 = time.perf_counter()
    result = await generator.generate(query, chunks)
    generation_ms = (time.perf_counter() - t1) * 1000

    total_ms = retrieval_ms + generation_ms

    print(f"ANSWER ({result['provider']}) [{result.get('question_type', 'FACTOID')}]:")
    print("-"*70)
    print(result["answer"])
    print("-"*70)
    print(f"Sources : {', '.join(result['sources']) or 'none'}")
    print(f"Chunks  : {result['num_chunks']} used")
    print(f"Timing  : retrieval={retrieval_ms:.0f}ms | generation={generation_ms:.0f}ms | total={total_ms:.0f}ms")

    if "grounding" in result:
        g = result["grounding"]
        verdict = "GROUNDED" if g["is_grounded"] else "NOT GROUNDED"
        print(f"Grounding: {verdict} (score={g['grounding_score']:.4f}, best_chunk={g['best_chunk_id']})")
        if g["evidence_phrases"]:
            print(f"Evidence : {' | '.join(g['evidence_phrases'])}")


async def main(args: argparse.Namespace) -> None:
    logger.info("="*70)
    logger.info("PHASE 2 - GENERATION PIPELINE")
    logger.info("="*70)

    provider = get_provider(provider=args.provider, model=args.model)
    logger.info(f"LLM provider : {provider.provider_name}")

    store = VectorStoreFaiss(
        db_path=args.db_path,
        index_path=args.index_path,
    )

    generator = RAGGenerator(
        provider=provider,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        multi_passage=not args.no_multi_passage,
        verify_grounding=args.verify_grounding,
        grounding_threshold=args.grounding_threshold,
        embed_fn=store.embed_text if args.verify_grounding else None,
    )

    reranker = None
    if args.rerank:
        from src.retrieval.reranker import CrossEncoderReranker
        reranker = CrossEncoderReranker(model_name=args.rerank_model, top_n=args.top_k)
        logger.info(f"Reranker  : {args.rerank_model} | top_n={args.top_k}")

    queries = args.queries if args.queries else DEMO_QUERIES
    for query in queries:
        await run_query(query, store, generator, top_k=args.top_k, reranker=reranker)

    print(f"\n{'='*70}")
    print("Done.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase 2 - RAG generation demo")
    p.add_argument("--provider", default=None,
                   help="LLM provider: ollama | anthropic | openai (default: $LLM_PROVIDER or ollama)")
    p.add_argument("--model", default=None,
                   help="Model override (e.g. llama3, mistral, claude-sonnet-4-6)")
    p.add_argument("--top-k", type=int, default=5,
                   help="Chunks to retrieve per query (default: 10)")
    p.add_argument("--max-tokens", type=int, default=512,
                   help="Max tokens in generated answer (default: 512)")
    p.add_argument("--temperature", type=float, default=0.3,
                   help="Sampling temperature (default: 0.3)")
    p.add_argument("--no-multi-passage", action="store_true",
                   help="Disable multi-passage synthesis (use top chunk only)")
    p.add_argument("--verify-grounding", action="store_true",
                   help="Run grounding verification (hybrid lexical+semantic) after generation")
    p.add_argument("--grounding-threshold", type=float, default=0.4,
                   help="Minimum grounding score for is_grounded=True (default: 0.4)")
    p.add_argument("--rerank", action="store_true",
                   help="Enable cross-encoder reranking after FAISS retrieval")
    p.add_argument("--rerank-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2",
                   help="Cross-encoder model for reranking (default: ms-marco-MiniLM-L-6-v2)")
    p.add_argument("--db-path", default="./data/vectordb/chunks.db")
    p.add_argument("--index-path", default="./data/vectordb/chunks.faiss")
    p.add_argument("queries", nargs="*", help="Questions to answer (default: demo set)")
    return p


if __name__ == "__main__":
    asyncio.run(main(build_parser().parse_args()))
