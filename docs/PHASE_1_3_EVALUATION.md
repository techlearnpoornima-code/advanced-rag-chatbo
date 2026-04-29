# Phase 1.3 - Retrieval Evaluation Metrics

**Date:** 2026-04-29  
**Phase:** 1.3 - Evaluate Retrieval Quality  
**Status:** Planning & Analysis

---

## Objective

Measure how well our **FAISS + SQLite vector store** retrieves relevant passages from CLAPnq dataset using standard information retrieval metrics.

**Key Questions:**
- ✓ How accurately does our chunking preserve answer spans?
- ✓ Can the vector store find passages containing answers?
- ✓ How does retrieval quality vary by question type?
- ✓ Are there chunking gaps affecting retrieval?

---

## Evaluation Metrics

### **1. MRR (Mean Reciprocal Rank)**

**What it measures:** Position of the first relevant result

```
Query: "What is capital of France?"
Results:
  1. ❌ "European geography"         (not relevant)
  2. ❌ "French culture"             (not relevant)
  3. ✅ "Paris is the capital"       (relevant!) ← Rank 3

MRR = 1/3 = 0.333
```

**Formula:** `MRR = (1/N) * Σ(1/rank_i)` for N queries

**Interpretation:**
- 1.0 = Perfect (answer always at rank 1)
- 0.5 = Answer usually in top 2
- 0.33 = Answer usually in top 3
- 0.1 = Answer rarely in top 10

**Target for CLAPnq:** MRR ≥ 0.65 (baseline with SentenceTransformer)

---

### **2. NDCG@K (Normalized Discounted Cumulative Gain)**

**What it measures:** Ranking quality (considers partial relevance)

```
Results (with relevance scores):
  1. "European geography"            (relevance: 0.3)
  2. "French cities"                 (relevance: 0.6)
  3. "Paris is the capital"          (relevance: 1.0) ← Best
  4. "France info"                   (relevance: 0.4)
  5. ...

DCG = 0.3/log(2) + 0.6/log(3) + 1.0/log(4) + 0.4/log(5) + ...
iDCG = 1.0/log(2) + 0.6/log(3) + 0.3/log(4) + 0.0/...  (ideal order)
NDCG = DCG / iDCG
```

**Interpretation:**
- 1.0 = Perfect ranking
- 0.9+ = Excellent ranking
- 0.7-0.8 = Good ranking
- < 0.5 = Poor ranking

**Target for CLAPnq:** NDCG@10 ≥ 0.72

---

### **3. Recall@K**

**What it measures:** Percentage of questions where answer appears in top-K

```
10 questions, top-5 results:
  Q1: ✅ Answer in top-5
  Q2: ✅ Answer in top-5
  Q3: ❌ Answer NOT in top-5
  Q4: ✅ Answer in top-5
  ...

Recall@5 = 7/10 = 0.7 (70% of questions have answer in top-5)
```

**Target for CLAPnq:**
- Recall@5 ≥ 0.60
- Recall@10 ≥ 0.75

---

### **4. Precision@K**

**What it measures:** Percentage of retrieved results that are relevant

```
Top-10 results:
  1. ✅ Relevant
  2. ❌ Not relevant
  3. ✅ Relevant
  4. ✅ Relevant
  5. ❌ Not relevant
  6. ✅ Relevant
  7. ✅ Relevant
  8. ❌ Not relevant
  9. ✅ Relevant
  10. ✅ Relevant

Precision@10 = 8/10 = 0.8 (80% of top-10 are relevant)
```

---

### **5. Answer Coverage**

**What it measures:** % of questions where answer span is preserved in chunks

```
Total CLAPnq questions: 1,954 (answerable)

After chunking:
  1,891 questions: Answer span intact in at least 1 chunk ✓
  63 questions: Answer span lost during chunking ✗

Answer Coverage = 1,891 / 1,954 = 96.8%
```

**Target:** Answer Coverage ≥ 95%

---

## Evaluation Dataset

### **Size**
- **Answerable records:** 1,954 questions with answers
- **Unanswerable records:** 1,791 questions without answers
- **Total:** 3,745 records

### **Split Strategy**

| Split | Records | Purpose |
|-------|---------|---------|
| **Dev (Validation)** | 500 | Quick evaluation during development |
| **Test (Final)** | 500 | Final evaluation metrics |
| **Full** | 3,745 | Comprehensive evaluation |

---

## Implementation Plan

### **Phase 1.3.1: Relevance Labeling**

Create a function to determine if a retrieved chunk is relevant:

```python
def is_relevant(question, retrieved_chunk, answer_text):
    """
    Check if chunk contains answer to question.
    
    Args:
        question: Original question
        retrieved_chunk: Retrieved chunk text
        answer_text: Expected answer span
    
    Returns:
        bool: True if chunk contains answer
    """
    # Method 1: Exact answer span match
    if answer_text in retrieved_chunk:
        return True
    
    # Method 2: Fuzzy matching (for slight variations)
    if fuzz.token_set_ratio(answer_text, retrieved_chunk) > 0.85:
        return True
    
    return False
```

### **Phase 1.3.2: Metric Computation**

```python
class RetrievalMetrics:
    """Compute standard IR metrics"""
    
    def compute_mrr(self, rankings):
        """Mean Reciprocal Rank"""
        mrr_scores = []
        for ranks in rankings:
            if ranks:  # If any relevant
                mrr_scores.append(1.0 / (ranks[0] + 1))
            else:      # No relevant found
                mrr_scores.append(0.0)
        return sum(mrr_scores) / len(mrr_scores)
    
    def compute_ndcg(self, relevances, k=10):
        """NDCG@K"""
        # DCG: sum(relevance / log(rank+1))
        # iDCG: best possible DCG
        ...
    
    def compute_recall(self, rankings, k=10):
        """Recall@K: % queries with answer in top-K"""
        found = sum(1 for r in rankings if r and r[0] < k)
        return found / len(rankings)
    
    def compute_precision(self, rankings, k=10):
        """Precision@K"""
        ...
```

### **Phase 1.3.3: Evaluation Script**

```python
# scripts/3_evaluate_retrieval.py

async def main():
    """Evaluate vector store on CLAPnq benchmark"""
    
    # 1. Load vector store
    store = VectorStoreFaiss(...)
    
    # 2. Load CLAPnq test set
    records = load_clap_records(limit=500)
    
    # 3. For each question:
    for record in records:
        query = record['input']
        answer = record['output'][0]['answer']
        passage_title = record['passages'][0]['title']
        
        # 4. Retrieve top-10 chunks
        results = await store.search(query, top_k=10)
        
        # 5. Check relevance
        rankings = []
        for rank, result in enumerate(results):
            if is_relevant(query, result['chunk_text'], answer):
                rankings.append(rank)
        
        # 6. Track metrics
        track_metrics(rankings, ...)
    
    # 7. Compute final metrics
    metrics = {
        'mrr': compute_mrr(...),
        'ndcg@10': compute_ndcg(..., k=10),
        'recall@5': compute_recall(..., k=5),
        'recall@10': compute_recall(..., k=10),
        'answer_coverage': compute_answer_coverage(...)
    }
    
    return metrics
```

---

## Expected Results (Baseline)

### **With SentenceTransformer (384 dims) - Current**

```
Metric              Target      Expected
─────────────────────────────────────────
MRR                 0.65        0.62-0.65
NDCG@10             0.72        0.68-0.72
Recall@5            0.60        0.55-0.60
Recall@10           0.75        0.70-0.75
Answer Coverage     0.95        0.96-0.98
```

### **With BGE-large (1024 dims) - Phase 2 Target**

```
Metric              Target      Expected
─────────────────────────────────────────
MRR                 0.75        0.72-0.78
NDCG@10             0.80        0.78-0.82
Recall@5            0.70        0.68-0.72
Recall@10           0.85        0.82-0.87
Answer Coverage     0.95        0.96-0.98
```

---

## Analysis & Debugging

### **If MRR is Low (<0.60)**

**Possible causes:**
1. Embedding model too weak (SentenceTransformer 384-dim)
2. Chunking is too coarse (missing answer spans)
3. Question-answer semantic mismatch

**Fixes:**
- Upgrade embedding model (BGE, OpenAI)
- Adjust chunk size (smaller = more precise)
- Add question reformulation

### **If Answer Coverage is Low (<0.95)**

**Possible causes:**
1. Chunks split answer spans across boundaries
2. Answer span not recognized correctly

**Fixes:**
- Increase chunk size (more overlap)
- Use answer-aware chunking
- Add validation that answers are preserved

### **If Recall@10 is Low (<0.70)**

**Possible causes:**
1. Not enough chunks retrieved (k=10 too small)
2. Semantic gap between question and passage

**Fixes:**
- Increase k (retrieve top-20 instead)
- Add query expansion
- Use hybrid search (dense + sparse)

---

## Success Criteria

✅ **Phase 1.3 is successful if:**

1. **MRR ≥ 0.62** - Reasonable baseline performance
2. **NDCG@10 ≥ 0.68** - Good ranking quality
3. **Recall@10 ≥ 0.70** - 70% of questions have answer in top-10
4. **Answer Coverage ≥ 0.96** - Chunking preserves >96% of answers
5. **All metrics computed** - Complete evaluation pipeline

---

## Phase 1.3 Deliverables

```
src/evaluation/
├── retrieval_metrics.py     # Metric computation
├── relevance_scoring.py     # Relevance labeling
└── benchmark_runner.py      # End-to-end evaluation

scripts/
├── 3_evaluate_retrieval.py  # Main evaluation script
└── 4_generate_report.py     # Generate HTML report

docs/
└── EVALUATION_RESULTS.md    # Final results & analysis
```

---

## Next Steps After Phase 1.3

### **If metrics are good (MRR≥0.62):**
→ Proceed to **Phase 2.1 - Hybrid Search**

### **If metrics are poor (MRR<0.60):**
→ Debug & iterate on:
- Embedding model upgrade (BGE)
- Chunk size optimization
- Query expansion techniques

---

## Timeline & Effort

| Step | Time | Notes |
|------|------|-------|
| 1.3.1 Relevance labeling | 30 min | Simple string matching |
| 1.3.2 Metric computation | 1 hr | MRR, NDCG, Recall |
| 1.3.3 Evaluation script | 1 hr | Load store, run queries |
| Testing & debugging | 1-2 hrs | Handle edge cases |
| Report generation | 30 min | Create summary |
| **Total** | **4-5 hrs** | |

---

*Last Updated: 2026-04-29*  
*Ready for: Phase 1.3 Implementation*
