# CLAPnq-Based RAG System

Production-ready Retrieval-Augmented Generation system optimized for long-form question answering using Natural Questions (CLAPnq) dataset and Wikipedia passages.

**Status:** Phase 1 - Foundation (In Progress)

---

## 🎯 What is CLAPnq?

- **Real Google Search Queries** paired with Wikipedia passages
- **Long-form RAG Benchmark** designed for cohesive, grounded answers  
- **Gold Standard** for evaluating RAG systems
- **3,745 Examples** (1,954 answerable + 1,791 unanswerable)

---

## 📦 Dataset

| Dataset | Records | Size | Purpose |
|---------|---------|------|---------|
| `clapnq_train_answerable.jsonl` | 1,954 | 6.3 MB | Questions with answers |
| `clapnq_train_unanswerable.jsonl` | 1,791 | 4.4 MB | Negative examples |

**Key Stats:**
- Questions: 4-21 words (avg 9.2)
- Passages: 62-10,598 words (avg 189-215)
- Answers: 0-243 words (avg 50)

See [CLAPNQ_DATASET_ANALYSIS.md](docs/CLAPNQ_DATASET_ANALYSIS.md) for full analysis.

---

## 🏗️ Architecture

```
User Query
    ↓
Retrieval Module
├─ Dense Search (Embeddings + FAISS)
├─ Sparse Search (BM25)
└─ Cross-Encoder Reranking
    ↓
Vector Store (FAISS + SQLite Metadata)
    ↓
Retrieved Passages
    ↓
Generation Module
├─ Grounding Check
├─ Answer Generation
└─ Confidence Scoring
    ↓
Response
```

---

## 🚀 Development Phases

### Phase 1: Foundation ✅ IN PROGRESS
- [x] Dataset analysis
- [x] Chunking strategy (Sentence-Based Semantic)
- [ ] Data loader implementation
- [ ] Semantic chunker implementation
- [ ] Vector store setup
- [ ] Retrieval metrics

### Phase 2: Advanced Retrieval 📋 PLANNED
- Hybrid search (dense + sparse + BM25)
- Cross-encoder reranking
- Metadata filtering & RBAC
- Multi-intent decomposition

### Phase 3: Generation & Quality 📋 PLANNED
- Claude-based answer generation
- Grounding checks
- Answer quality metrics

### Phase 4: Evaluation 📋 PLANNED
- CLAPnq benchmark evaluation
- Baseline comparisons
- Optimization

---

## 📚 Documentation

- **[CLAUDE.md](CLAUDE.md)** - Project overview & commands
- **[CLAPNQ_DATASET_ANALYSIS.md](docs/CLAPNQ_DATASET_ANALYSIS.md)** - Dataset statistics & analysis
- **[.claude/rules/code-style.md](.claude/rules/code-style.md)** - Python conventions
- **[.claude/rules/testing.md](.claude/rules/testing.md)** - Testing guidelines
- **[.claude/rules/api-conventions.md](.claude/rules/api-conventions.md)** - API design

---

## 🛠️ Quick Start

### Setup

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your ANTHROPIC_API_KEY
```

### Phase 1.1 - Load & Chunk Data

```bash
python scripts/1_load_and_chunk.py \
  --dataset clapnq_train_answerable.jsonl \
  --output data/chunks.jsonl
```

### Phase 1.2 - Build Vector Store

```bash
python scripts/2_build_vector_store.py \
  --chunks data/chunks.jsonl \
  --model all-MiniLM-L6-v2
```

### Phase 1.3 - Evaluate Retrieval

```bash
python scripts/3_evaluate_retrieval.py \
  --queries data/test_queries.jsonl
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific category
pytest tests/test_chunking.py
pytest tests/test_data_loading.py

# With coverage
pytest --cov=src --cov=app tests/
```

---

## 📊 Chunking Strategy

**Approach:** Sentence-Based Semantic Chunking

- Groups consecutive sentences until token limit
- Preserves semantic boundaries
- Tracks answer span locations
- Handles variable passage lengths

**Configuration:**
```python
max_tokens: 512
min_tokens: 50
overlap_sentences: 1
preserve_boundaries: True
```

**Expected Output:**
- ~9,000-12,000 total chunks from 3,745 records
- Average chunk: 512 tokens (3-4 sentences)
- Metadata: chunk_id, passage_title, sentence_indices, contains_answer flag

---

## 🔧 Tech Stack

- **Python** 3.11+
- **FastAPI** - Web framework
- **FAISS** - Vector indexing & similarity search
- **SQLite** - Metadata storage
- **SentenceTransformers** - Embeddings
- **Claude (Anthropic)** - LLM
- **Pytest** - Testing

---

## 📝 Important Notes

✅ **Do:**
- Use type hints in all code
- Write docstrings for public functions
- Log with context using loguru
- Test before submitting
- Keep data files intact
- Follow conventions in `.claude/rules/`

❌ **Don't:**
- Hardcode configuration values
- Use blocking I/O in async functions
- Commit `.env` files
- Skip error handling
- Modify CLAPnq dataset

---

## 🎓 References

- **Dataset:** Natural Questions (NQ) / CLAPnq
- **Paper:** Strich et al., 2025
- **Benchmark Metrics:** MRR, NDCG, F1, Exact Match

---

## 📞 Need Help?

See [CLAUDE.md](CLAUDE.md) for detailed project documentation and available commands.

---

*Last Updated: 2026-04-29*  
*Phase: 1 (Foundation) - Ready for Phase 1.1 Implementation*
