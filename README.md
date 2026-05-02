# CLAPnq-Based RAG System

Production-ready Retrieval-Augmented Generation system optimized for long-form question answering using Natural Questions (CLAPnq) dataset and Wikipedia passages.

**Status:** Phase 2 - Generation & Quality (In Progress)

---

## What is CLAPnq?

- **Real Google Search Queries** paired with Wikipedia passages
- **Long-form RAG Benchmark** designed for cohesive, grounded answers
- **Gold Standard** for evaluating RAG systems
- **3,745 Examples** (1,954 answerable + 1,791 unanswerable)

---

## Dataset

| Dataset | Records | Size | Purpose |
|---------|---------|------|---------|
| `clapnq_train_answerable.jsonl` | 1,954 | 6.3 MB | Questions with answers |
| `clapnq_train_unanswerable.jsonl` | 1,791 | 4.4 MB | Negative examples |

**Key Stats:**
- Questions: 4-21 words (avg 9.2)
- Passages: 62-10,598 words (avg 189-215)
- Answers: 0-243 words (avg 50)

---

## Architecture

```
User Query
    ↓
Question Classifier (6 types)
├─ FACTOID / NUMERIC / TEMPORAL / LOCATION
└─ SYNTHESIS / MULTI_HOP
    ↓
Retrieval (FAISS + SQLite)
├─ Dense Embeddings (all-MiniLM-L6-v2)
├─ Score Threshold Filter
│   └─ Lowered 0.15 for SYNTHESIS/MULTI_HOP (broader evidence)
└─ Deduplication + Subtopic Grouping
    ↓
Generation (RAGGenerator)
├─ SYNTHESIS  → deep synthesis prompt (cite passage topics, explain causality)
├─ MULTI_HOP → chain-of-evidence prompt (connect facts step by step)
└─ FACTOID/NUMERIC/TEMPORAL/LOCATION → concise factual prompt
    ↓
LLM (Anthropic Claude / Ollama / OpenAI)
    ↓
Grounding Verification (GroundingVerifier)
├─ Lexical Score  — fraction of answer words found in source chunks
├─ Semantic Score — cosine similarity via shared embedding model
└─ Hybrid Score + Evidence Phrases
    ↓
Response {answer, sources, grounding, question_type}
```

---

## Development Phases

### Phase 1: Foundation ✅ COMPLETE

| Step | Task | Result |
|------|------|--------|
| 1.1 | Load & parse CLAPnq dataset | 3,745 records |
| 1.2 | Sentence-based semantic chunking | ~10,000 chunks, 512-token max |
| 1.3 | FAISS + SQLite vector store | HNSW index |
| 1.4 | Retrieval evaluation | MRR=0.775, NDCG=0.961, Recall@5=1.0 |

### Phase 2: Generation & Quality ✅ IN PROGRESS

| Step | Task | Status |
|------|------|--------|
| 2.0 | Full-scale retrieval evaluation (500+ queries) | ✅ |
| 2.1 | RAGGenerator with multi-passage synthesis | ✅ |
| 2.2 | Answer Grounding & Verification (hybrid lexical + semantic) | ✅ |
| 2.3 | Type-aware prompt optimization | ✅ |
| 2.4 | Question Type Classifier (6 types incl. SYNTHESIS & MULTI_HOP) | ✅ |
| 2.5 | End-to-end evaluation (ROUGE, BERTScore, F1 vs gold answers) | Planned |

### Phase 3: Evaluation & Optimization (Planned)
- Run against full CLAPnq benchmark
- ROUGE-L, BERTScore, token F1 vs gold answers
- Failure analysis by question type
- Iteration on generation quality

---

## Quick Start

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add ANTHROPIC_API_KEY to .env
```

### Step 1 — Load & Chunk Data

```bash
python scripts/1_load_and_chunk.py
# Output: data/chunks.jsonl
```

### Step 2 — Build Vector Store

```bash
python scripts/2_build_vector_store.py
# Output: data/vectordb/chunks.faiss + chunks.db
```

### Step 3 — Evaluate Retrieval

```bash
python scripts/3_evaluate_retrieval.py
# Metrics: MRR, NDCG, Recall@5, Precision@5
```

### Step 4 — Full-Scale Evaluation

```bash
python scripts/4_full_scale_evaluation.py
# Output: data/full_scale_evaluation.json
```

### Step 5 — Run Generation Pipeline

```bash
# Default (Ollama/llama3)
python scripts/5_generate_answers.py

# With Anthropic Claude
python scripts/5_generate_answers.py --provider anthropic --model claude-sonnet-4-6

# With grounding verification
python scripts/5_generate_answers.py --provider anthropic --verify-grounding

# Custom question
python scripts/5_generate_answers.py "How did the Roman Empire fall?"
```

---

## Generation Features

### Question Type Classification

The system classifies every query before generation to select the optimal prompt strategy:

| Type | Example | Behaviour |
|------|---------|-----------|
| `FACTOID` | "Who sang X?" | Concise single-entity answer |
| `NUMERIC` | "How many countries in Europe?" | Number + context |
| `TEMPORAL` | "When were the Camp David Accords signed?" | Date + context |
| `LOCATION` | "Where is Oklahoma?" | Place + context |
| `SYNTHESIS` | "How did France contribute to American independence?" | Multi-passage synthesis with passage citations |
| `MULTI_HOP` | "How did France's defeat that led to the Treaty of Fontainebleau cause it to cede Louisiana?" | Chain-of-evidence reasoning across passages |

### Answer Grounding

Every answer can be verified against source passages using hybrid lexical + semantic scoring:

```python
generator = RAGGenerator(
    verify_grounding=True,
    embed_fn=store.embed_text,   # reuses loaded SentenceTransformer — no extra model
    grounding_threshold=0.4,
)
result = await generator.generate(query, chunks)

# result["grounding"]:
# {
#   "is_grounded": True,
#   "grounding_score": 0.82,
#   "best_chunk_id": "p3_c1",
#   "evidence_phrases": ["france allied with the united states", "declared war on great britain"]
# }
```

### LLM Providers

| Provider | Models | Requirement |
|----------|--------|-------------|
| Ollama (default) | llama3, mistral | Local, no API key |
| Anthropic | claude-sonnet-4-6, claude-opus-4-7 | `ANTHROPIC_API_KEY` |
| OpenAI | gpt-4o, gpt-4o-mini | `OPENAI_API_KEY` |

Switch via `LLM_PROVIDER` env var or `--provider` CLI flag.

---

## Retrieval Performance (Phase 1 Results)

| Metric | Score | Target | Status |
|--------|-------|--------|--------|
| MRR | 0.7751 | 0.65 | +19% above target |
| NDCG | 0.9614 | 0.72 | +33% above target |
| Recall@5 | 1.0000 | 0.80 | Perfect |
| Precision@5 | 0.4000 | — | 4/10 chunks relevant |
| Query Time | ~40ms | <200ms | Fast |

---

## Tech Stack

- **Python** 3.11+
- **FastAPI** — Web framework
- **FAISS** — Vector indexing (HNSW)
- **SQLite** — Metadata storage
- **SentenceTransformers** (`all-MiniLM-L6-v2`) — Embeddings
- **Claude / Ollama / OpenAI** — LLM generation
- **NumPy** — Similarity computation
- **Pytest** — Testing

---

## Project Structure

```
advanced-rag-chatbot/
├── data/
│   ├── clapnq_train_answerable.jsonl
│   ├── clapnq_train_unanswerable.jsonl
│   ├── chunks.jsonl
│   └── vectordb/
│       ├── chunks.faiss
│       └── chunks.db
├── src/
│   ├── chunking/
│   │   └── semantic_chunker.py
│   ├── retrieval/
│   │   └── vector_store_faiss.py
│   ├── generation/
│   │   ├── rag_generator.py           # Main pipeline + type-aware routing
│   │   ├── grounding.py               # Hallucination detection
│   │   ├── question_classifier.py     # 6-type question classifier
│   │   └── providers/                 # Anthropic / Ollama / OpenAI
│   └── evaluation/
│       └── metrics.py
├── scripts/
│   ├── 1_load_and_chunk.py
│   ├── 2_build_vector_store.py
│   ├── 3_evaluate_retrieval.py
│   ├── 4_full_scale_evaluation.py
│   └── 5_generate_answers.py
├── app/
│   ├── main.py
│   ├── config.py
│   └── models.py
└── tests/
```

---

## Testing

```bash
pytest tests/
pytest tests/test_chunking.py
pytest --cov=src --cov=app tests/
```

---

## References

- **Dataset:** Natural Questions (NQ) / CLAPnq
- **Paper:** Strich et al., 2025 — CLAPnq: Coherent Long-form Answer Pairing for Natural Questions
- **Benchmark Metrics:** MRR, NDCG, F1, ROUGE-L, BERTScore

---

*Last Updated: 2026-05-02*
*Phase: 2 — Generation, Grounding & Type-Aware Routing implemented*
