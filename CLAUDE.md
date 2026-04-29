# CLAPnq-Based RAG System

**Project Purpose:** Production-ready Retrieval-Augmented Generation system optimized for long-form question answering using Natural Questions (CLAPnq) dataset and Wikipedia passages.

**Status:** Phase 1 Development - Dataset Analysis & Chunking Strategy

---

## Project Overview

### What is CLAPnq?

- **Real Google Search Queries** paired with Wikipedia passages
- **Long-form RAG benchmark** designed for cohesive, grounded answers
- **Gold standard** for evaluating RAG systems
- **3,745 total examples** (1,954 answerable + 1,791 unanswerable)

### Project Goals

1. **Phase 1:** Foundation with proper chunking (sentence-based semantic)
2. **Phase 2:** Advanced retrieval (hybrid search, cross-encoders, metadata filtering, RBAC)
3. **Phase 3:** Generation & long-form answer quality
4. **Phase 4:** Evaluation against CLAPnq benchmarks

---

## Tech Stack

- **Language:** Python 3.11+
- **Framework:** FastAPI (async web framework)
- **LLM:** Claude (Anthropic)
- **Vector DB:** ChromaDB (persistent storage)
- **Embeddings:** all-MiniLM-L6-v2 (SentenceTransformers)
- **Data:** CLAPnq Dataset (Natural Questions)
- **Evaluation:** Custom metrics + CLAPnq benchmarks

---

## Project Structure

```
advanced-rag-chatbot/
├── data/
│   ├── clapnq_train_answerable.jsonl      # 1,954 records with answers
│   └── clapnq_train_unanswerable.jsonl    # 1,791 records without answers
│
├── src/
│   ├── data_loading/
│   │   └── clapnq_loader.py               # Load CLAPnq JSONL files
│   ├── chunking/
│   │   └── semantic_chunker.py            # Sentence-based semantic chunking
│   ├── retrieval/
│   │   └── vector_store.py                # ChromaDB vector storage
│   ├── generation/
│   │   └── llm_client.py                  # Claude API integration
│   └── evaluation/
│       └── metrics.py                     # Retrieval & generation metrics
│
├── app/
│   ├── main.py                            # FastAPI application
│   ├── config.py                          # Configuration & settings
│   ├── models.py                          # Pydantic models
│   └── api/
│       └── routes.py                      # API endpoints
│
├── scripts/
│   ├── 1_load_and_chunk.py                # Phase 1.1 - Load data & chunk
│   ├── 2_build_vector_store.py            # Phase 1.2 - Build index
│   ├── 3_evaluate_retrieval.py            # Phase 1.3 - Eval metrics
│   └── generate_benchmarks.py             # Run against CLAPnq benchmark
│
├── docs/
│   ├── CLAPNQ_DATASET_ANALYSIS.md         # Dataset statistics & analysis
│   ├── PHASE_1_CHUNKING.md                # Chunking strategy details
│   └── ARCHITECTURE.md                    # System architecture
│
├── tests/
│   ├── test_data_loading.py
│   ├── test_chunking.py
│   └── test_retrieval.py
│
└── docker/
    └── Dockerfile                         # Docker deployment
```

---

## Dataset Details

### CLAPnq Files

| File | Records | Size | Purpose |
|------|---------|------|---------|
| `clapnq_train_answerable.jsonl` | 1,954 | 6.3 MB | Questions with answers |
| `clapnq_train_unanswerable.jsonl` | 1,791 | 4.4 MB | Negative examples |

### Key Statistics

- **Questions:** 4-21 words (avg 9.2 words)
- **Passages:** 62-10,598 words (avg 189-215 words)
- **Answers:** 0-243 words (avg 50 words when present)
- **Answerable:** 96% have answer spans
- **Unanswerable:** 100% negative examples (no answers)

See `docs/CLAPNQ_DATASET_ANALYSIS.md` for full analysis.

---

## Development Phases

### Phase 1: Foundation with CLAPnq ✅ IN PROGRESS
- **1.1** Load & parse CLAPnq dataset
- **1.2** Implement sentence-based semantic chunking
- **1.3** Build ChromaDB index with metadata
- **1.4** Create retrieval evaluation metrics
- **1.5** Implement baseline RAG pipeline

**Current Status:** Analyzing dataset (1.0 complete)

### Phase 2: Advanced Retrieval & Routing 📋 PLANNED
- **2.1** Query intent detection
- **2.2** Hybrid search (dense + sparse + BM25)
- **2.3** Cross-encoder reranking
- **2.4** Metadata filtering & RBAC roles
- **2.5** Multi-intent decomposition

### Phase 3: Generation & Long-form Answers 📋 PLANNED
- **3.1** Claude-based answer generation
- **3.2** Grounding checks (facts from passages)
- **3.3** Answer quality metrics

### Phase 4: Evaluation & Optimization 📋 PLANNED
- **4.1** Run against CLAPnq benchmark
- **4.2** Compare against baselines
- **4.3** Iterate on strategy

---

## Chunking Strategy

### Selected Approach: Sentence-Based Semantic Chunking

**Why?**
- Dataset provides pre-segmented sentences
- Answer spans align with sentence indices
- Handles 62-10,598 word passages gracefully
- Better retrieval quality

**Configuration:**
```python
CHUNKING_CONFIG = {
    "strategy": "sentence_based_semantic",
    "max_tokens": 512,
    "min_tokens": 50,
    "overlap_sentences": 1,
    "preserve_sentence_boundaries": True
}
```

**Expected Result:**
- Total chunks: ~9,000-12,000 from 3,745 records
- Average chunk: 512 tokens (3-4 sentences)
- Metadata: chunk_id, passage_title, sentence_indices, contains_answer flag

---

## Configuration

### Environment Variables (.env)

```bash
# LLM Configuration
ANTHROPIC_API_KEY=your_api_key_here
LLM_MODEL=claude-opus-4-7
LLM_TEMPERATURE=0.3

# Embedding Configuration
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_BATCH_SIZE=32

# ChromaDB Configuration
VECTOR_DB_PATH=./data/vectordb
CHROMA_SETTINGS={"anonymized_telemetry": false, "allow_reset": true}

# Chunking Configuration
CHUNK_SIZE_TOKENS=512
CHUNK_OVERLAP_SENTENCES=1

# Retrieval Configuration
RETRIEVAL_TOP_K=10
ENABLE_HYBRID_SEARCH=true
ENABLE_CROSS_ENCODER=false

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
ALLOWED_ORIGINS=["*"]
```

---

## Code Style & Conventions

### Python Standards

- **PEP 8** compliance (100 char line length)
- **Type hints** required for all functions
- **Async/await** for all I/O operations
- **Docstrings** Google-style for public methods
- **Logging** using loguru (structured logging)
- **Error handling** with specific exceptions

### File Organization

```python
"""Module docstring."""

# Imports (stdlib, third-party, local)
import json
from pathlib import Path
from typing import List, Dict, Any

import anthropic
from loguru import logger

from app.config import settings

# Constants
DEFAULT_CHUNK_SIZE = 512

# Classes
class MyClass:
    """Class docstring."""
    pass

# Functions
def my_function() -> str:
    """Function docstring."""
    pass

if __name__ == "__main__":
    main()
```

---

## Testing

### Test Strategy

- **Unit tests** for chunking, data loading, retrieval
- **Integration tests** for end-to-end pipeline
- **Benchmark tests** against CLAPnq metrics
- **Coverage goal:** 80%+

### Running Tests

```bash
# All tests
pytest tests/

# Specific test
pytest tests/test_chunking.py::test_sentence_based_chunking

# With coverage
pytest --cov=src --cov=app tests/
```

---

## Commands

### Common Development Tasks

```bash
# Load & chunk CLAPnq data
python scripts/1_load_and_chunk.py

# Build vector store
python scripts/2_build_vector_store.py

# Run retrieval evaluation
python scripts/3_evaluate_retrieval.py

# Start API server
python app/main.py

# Run tests
pytest tests/
```

---

## Important Notes for Claude Code

1. ✅ **Always use type hints** - required for all functions
2. ✅ **Use async/await** - all I/O operations must be async
3. ✅ **Log with context** - use loguru, include relevant information
4. ✅ **Test incrementally** - build Phase 1, verify before Phase 2
5. ✅ **Reference .md analysis** - check `docs/CLAPNQ_DATASET_ANALYSIS.md`
6. ✅ **Keep data files intact** - never modify CLAPnq dataset
7. ✅ **Document decisions** - add rationale to complex code
8. ❌ **Don't hardcode values** - use `app/config.py` for configuration
9. ❌ **Don't commit .env** - API keys stay local only
10. ❌ **Don't skip chunking tests** - quality depends on proper chunking

---

## Current Status

**Phase 1.0 COMPLETE:**
- ✅ Dataset downloaded & available
- ✅ Dataset analysis complete (see `docs/CLAPNQ_DATASET_ANALYSIS.md`)
- ✅ Chunking strategy selected (Sentence-Based Semantic)
- ✅ Project structure cleaned & ready

**Next Steps:**
- Implement data loader (Phase 1.1)
- Implement semantic chunker (Phase 1.1)
- Build vector store (Phase 1.2)
- Verify chunk quality (Phase 1.3)

---

## References

- **Dataset:** Natural Questions (NQ) / CLAPnq
- **Paper:** Strich et al., 2025 - CLAPnq: Coherent Long-form Answer Pairing for Natural Questions
- **Benchmark:** Use CLAPnq evaluation for MRR, NDCG, F1

---

*Last Updated: 2026-04-29*  
*Project Stage: Phase 1 - Foundation*  
*Ready for: Phase 1.1 Implementation*
