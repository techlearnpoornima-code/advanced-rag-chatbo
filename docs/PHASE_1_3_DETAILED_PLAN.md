# Phase 1.3 - Detailed Implementation Plan & Alternative Approaches

**Date:** 2026-04-29  
**Phase:** 1.3 - Retrieval Evaluation Metrics  
**Scope:** Detailed breakdown + 4 alternative evaluation strategies

---

## Part 1: Detailed Implementation Breakdown

### 1.3.1 Relevance Labeling - Three Approaches

Determining if a chunk is "relevant" to answer a question is critical but not trivial.

#### **Approach A: Exact String Match** (Simplest)

**Implementation:**
```python
def is_relevant_exact(question: str, chunk_text: str, answer_text: str) -> bool:
    """Check if answer_text appears exactly in chunk_text."""
    return answer_text in chunk_text
```

**Characteristics:**
- **Complexity:** O(n) string matching
- **Speed:** < 1ms per chunk
- **False Positives:** None (exact match is reliable)
- **False Negatives:** High (misses paraphrases, typos)
- **Example:**
  ```
  Question: "What is the capital of France?"
  Answer: "Paris"
  Chunk 1: "Paris is the capital of France..." ✅ RELEVANT
  Chunk 2: "The city of Paris..." ❌ IRRELEVANT (no "Paris" alone)
  Chunk 3: "Paris, the capital city..." ✅ RELEVANT
  ```

**When to use:** Quick baseline, clean datasets

#### **Approach B: Fuzzy Token Matching** (Recommended for Phase 1.3)

**Implementation:**
```python
from difflib import SequenceMatcher

def is_relevant_fuzzy(
    question: str, 
    chunk_text: str, 
    answer_text: str,
    threshold: float = 0.85
) -> bool:
    """Check if answer appears in chunk with fuzzy matching."""
    # Try exact match first
    if answer_text in chunk_text:
        return True
    
    # Fuzzy match: splits and compares token overlap
    answer_tokens = answer_text.lower().split()
    chunk_lower = chunk_text.lower()
    
    # Check if answer tokens appear in sequence
    for i in range(len(chunk_lower) - len(answer_text) + 1):
        chunk_slice = chunk_lower[i:i+len(answer_text)]
        similarity = SequenceMatcher(None, answer_text, chunk_slice).ratio()
        if similarity > threshold:
            return True
    
    return False
```

**Characteristics:**
- **Complexity:** O(n*m) where n=chunk, m=answer
- **Speed:** ~10-50ms per chunk (10x slower than exact)
- **False Positives:** Very low (fuzzy prevents noise)
- **False Negatives:** Low (catches paraphrases)
- **Example:**
  ```
  Question: "What is the capital of France?"
  Answer: "Paris"
  Chunk 1: "Paris is the capital..." ✅ RELEVANT (exact)
  Chunk 2: "Parisians love their city..." ✅ RELEVANT (fuzzy, ~0.90)
  Chunk 3: "The city Paris..." ✅ RELEVANT (fuzzy, ~0.85)
  Chunk 4: "Par is mentioned..." ❌ IRRELEVANT (<0.85)
  ```

**When to use:** Phase 1.3 baseline (good balance of speed + quality)

#### **Approach C: Semantic Similarity** (Phase 2 upgrade)

**Implementation:**
```python
import numpy as np
from sentence_transformers import util

def is_relevant_semantic(
    embedding_model,
    chunk_text: str,
    answer_text: str,
    threshold: float = 0.6
) -> bool:
    """Check if answer and chunk are semantically similar."""
    chunk_emb = embedding_model.encode(chunk_text, convert_to_tensor=True)
    answer_emb = embedding_model.encode(answer_text, convert_to_tensor=True)
    
    similarity = util.pytorch_cos_sim(chunk_emb, answer_emb)[0][0]
    return similarity > threshold
```

**Characteristics:**
- **Complexity:** O(embedding_time * 2)
- **Speed:** ~50-200ms per chunk (50x slower than exact)
- **False Positives:** Possible (semantic drift)
- **False Negatives:** Very low (catches all variations)
- **Example:**
  ```
  Question: "What is the capital of France?"
  Answer: "Paris"
  Chunk 1: "Paris is the capital..." ✅ RELEVANT (0.95)
  Chunk 2: "The main city of France..." ✅ RELEVANT (0.78 - semantic match)
  Chunk 3: "France's largest metropolis..." ⚠️ MAYBE (0.65 - on boundary)
  Chunk 4: "The Eiffel Tower..." ❌ IRRELEVANT (0.45 - too different)
  ```

**When to use:** Phase 2+ when quality matters more than speed

---

### Recommendation for Phase 1.3

**Use Approach B (Fuzzy Matching)** because:
1. ✅ 10-50x faster than semantic (scale to 1,954 questions * 10 chunks = 19,540 checks)
2. ✅ Sufficient accuracy for baseline evaluation
3. ✅ No extra model loading overhead
4. ✅ Can upgrade to semantic in Phase 2 if needed

---

### 1.3.2 Metric Computation - Complete Implementation

#### **MRR (Mean Reciprocal Rank)**

**Formula:**
```
MRR = (1/Q) * Σ(1/rank_i)

where:
  Q = number of queries
  rank_i = position of first relevant result for query i
```

**Implementation:**
```python
def compute_mrr(rankings: List[List[int]]) -> float:
    """
    Compute Mean Reciprocal Rank.
    
    Args:
        rankings: List of rankings per query
                  Each ranking is a list of positions (0-indexed)
                  where relevant results appear, or empty if none found
    
    Returns:
        MRR score (0.0 to 1.0)
    
    Example:
        rankings = [
            [2],      # Query 1: first relevant at rank 3 (0-indexed) → 1/(2+1)
            [0],      # Query 2: first relevant at rank 1 → 1/(0+1)
            [],       # Query 3: no relevant results → 0
        ]
        MRR = (1/3 + 1/2 + 0) / 3 = 0.278
    """
    mrr_scores = []
    for ranks in rankings:
        if ranks:  # If any relevant result found
            mrr_scores.append(1.0 / (ranks[0] + 1))
        else:      # No relevant result
            mrr_scores.append(0.0)
    
    return sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0
```

**Interpretation:**
- **1.0** = First result always relevant
- **0.5** = First relevant typically in top 2
- **0.33** = First relevant typically in top 3
- **0.1** = First relevant rarely in top 10

**For CLAPnq:** Target ≥ 0.62 (Phase 1), ≥ 0.75 (Phase 2)

---

#### **NDCG@K (Normalized Discounted Cumulative Gain)**

**Formula:**
```
DCG@K = Σ(rel_i / log2(i+2))  for i=0 to K-1

where:
  rel_i = relevance score of item at position i
  log2(i+2) = discount factor (position-based decay)

iDCG@K = ideal DCG (if items ordered by relevance)

NDCG@K = DCG@K / iDCG@K
```

**Implementation:**
```python
import math

def compute_ndcg(relevances: List[List[float]], k: int = 10) -> float:
    """
    Compute NDCG@K.
    
    Args:
        relevances: List of relevance scores per query
                    Each is a list of floats (0.0-1.0) for top-K results
        k: Cutoff (usually 10)
    
    Returns:
        NDCG@K score (0.0 to 1.0)
    
    Example:
        relevances = [
            [1.0, 0.8, 0.6, 0.4, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0],  # Query 1
            [0.8, 0.6, 0.4, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Query 2
        ]
        For each query:
          DCG = 1.0/log2(2) + 0.8/log2(3) + 0.6/log2(4) + ...
          iDCG = 1.0/log2(2) + 0.8/log2(3) + 0.6/log2(4) + ...
          NDCG = DCG / iDCG
    """
    ndcg_scores = []
    
    for query_relevances in relevances:
        # Compute DCG
        dcg = 0.0
        for i, rel in enumerate(query_relevances[:k]):
            dcg += rel / math.log2(i + 2)  # log2(i+2) so position 0 has log2(2)
        
        # Compute ideal DCG (sort relevances descending)
        ideal_relevances = sorted(query_relevances[:k], reverse=True)
        idcg = 0.0
        for i, rel in enumerate(ideal_relevances):
            idcg += rel / math.log2(i + 2)
        
        # Compute NDCG
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcg_scores.append(ndcg)
    
    return sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0
```

**Interpretation:**
- **1.0** = Perfect ranking
- **0.9+** = Excellent
- **0.7-0.8** = Good
- **< 0.5** = Poor

**For CLAPnq:** Target ≥ 0.68 (Phase 1), ≥ 0.80 (Phase 2)

---

#### **Recall@K**

**Formula:**
```
Recall@K = (queries with answer in top-K) / total_queries
```

**Implementation:**
```python
def compute_recall(rankings: List[List[int]], k: int = 10) -> float:
    """
    Compute Recall@K.
    
    Args:
        rankings: List of rankings per query (same as MRR)
        k: Cutoff (usually 5 or 10)
    
    Returns:
        Recall@K (0.0 to 1.0, percentage)
    
    Example:
        rankings = [
            [2],      # Query 1: answer at rank 3 (< 10) ✅
            [15],     # Query 2: answer at rank 16 (> 10) ❌
            [0],      # Query 3: answer at rank 1 (< 10) ✅
        ]
        Recall@10 = 2/3 = 0.667
    """
    found = sum(1 for ranks in rankings if ranks and ranks[0] < k)
    return found / len(rankings) if rankings else 0.0
```

**For CLAPnq:**
- Recall@5 ≥ 0.60 (Phase 1)
- Recall@10 ≥ 0.75 (Phase 1)

---

#### **Answer Coverage**

**Formula:**
```
Answer Coverage = (records with answer preserved) / total_records
```

**Implementation:**
```python
def compute_answer_coverage(metadata: List[Dict]) -> float:
    """
    Compute answer coverage during chunking.
    
    Args:
        metadata: List of chunk metadata dicts with 'contains_answer' flag
    
    Returns:
        Coverage percentage (0.0 to 1.0)
    
    Example:
        metadata = [
            {'chunk_id': '1', 'contains_answer': True},
            {'chunk_id': '2', 'contains_answer': True},
            {'chunk_id': '3', 'contains_answer': False},
            ...
        ]
        # Track unique passages with at least one answer chunk
        # Coverage = passages_with_answers / total_passages
    """
    pass  # Compute during Phase 1.1 chunking validation
```

**For CLAPnq:** Target ≥ 0.95

---

### 1.3.3 Evaluation Script Structure

```python
# scripts/3_evaluate_retrieval.py

async def main():
    # 1. Initialize
    store = VectorStoreFaiss(db_path="./data/vectordb/chunks.db")
    records = load_clap_records(limit=500)  # Test set
    
    # 2. Prepare metrics tracking
    metrics_tracker = {
        'mrr_rankings': [],
        'ndcg_relevances': [],
        'recall_rankings': [],
        'processing_times': []
    }
    
    # 3. For each question
    for i, record in enumerate(records):
        query = record['input']
        answer = record['output'][0]['answer']
        
        # 4. Retrieve top-10 chunks
        start = time.time()
        results = await store.search(query, top_k=10)
        elapsed = time.time() - start
        metrics_tracker['processing_times'].append(elapsed)
        
        # 5. Check relevance for each result
        relevant_ranks = []
        relevances = []
        
        for rank, chunk in enumerate(results):
            is_rel = is_relevant_fuzzy(
                question=query,
                chunk_text=chunk['chunk_text'],
                answer_text=answer
            )
            
            if is_rel:
                relevant_ranks.append(rank)
                relevances.append(1.0)
            else:
                relevances.append(0.0)
        
        # 6. Track metrics
        metrics_tracker['mrr_rankings'].append(relevant_ranks)
        metrics_tracker['ndcg_relevances'].append(relevances)
        metrics_tracker['recall_rankings'].append(relevant_ranks)
    
    # 7. Compute final metrics
    final_metrics = {
        'mrr': compute_mrr(metrics_tracker['mrr_rankings']),
        'ndcg@10': compute_ndcg(metrics_tracker['ndcg_relevances'], k=10),
        'recall@5': compute_recall(metrics_tracker['recall_rankings'], k=5),
        'recall@10': compute_recall(metrics_tracker['recall_rankings'], k=10),
        'avg_query_time_ms': mean(metrics_tracker['processing_times']) * 1000
    }
    
    return final_metrics
```

---

## Part 2: Alternative Evaluation Approaches

### **Alternative A: Tier-Based Relevance Scoring** (Fine-grained)

Instead of binary (relevant/not relevant), use 3-level scoring:

```python
def score_relevance_tier(chunk: Dict, answer: str) -> float:
    """
    Score chunk relevance on 3-tier scale.
    
    Tiers:
      1.0 = Direct Answer (chunk contains complete answer)
      0.5 = Semantic Match (chunk discusses answer concept)
      0.0 = Irrelevant (unrelated to answer)
    
    Example:
      Answer: "Paris"
      Chunk 1: "Paris is the capital of France." → 1.0
      Chunk 2: "France's largest city Paris..." → 1.0
      Chunk 3: "Paris is known for art and culture" → 0.5
      Chunk 4: "France is in Europe" → 0.0
    """
    if answer in chunk['chunk_text']:
        return 1.0
    elif semantic_similarity(answer, chunk['chunk_text']) > 0.7:
        return 0.5
    else:
        return 0.0
```

**Advantages:**
- ✅ Captures nuance (not all non-answers are equally bad)
- ✅ Better NDCG scores reflect this nuance

**Disadvantages:**
- ❌ Harder to define tier boundaries
- ❌ More computation (semantic similarity expensive)
- ❌ Harder to reproduce tier definitions

**Best for:** Phase 2+ when fine-tuning is needed

---

### **Alternative B: Question-Type Aware Evaluation** (Stratified)

Segment evaluation by question type and compute separate metrics:

```python
def categorize_question(question: str, answer: str) -> str:
    """Classify question into type."""
    # FACTOID: "Who/What/Where is/are..."
    if any(q in question.lower() for q in ['who', 'what', 'where']):
        return 'FACTOID'
    
    # NUMERIC: Answer is a number/date
    if answer[0].isdigit() or re.search(r'\d{4}', answer):
        return 'NUMERIC'
    
    # TEMPORAL: Time-related
    if any(t in question.lower() for t in ['when', 'how long', 'year']):
        return 'TEMPORAL'
    
    # LOCATION: Place-related
    if any(l in question.lower() for l in ['where', 'location', 'country', 'city']):
        return 'LOCATION'
    
    return 'OTHER'

# Evaluation code
metrics_by_type = {}
for question_type in ['FACTOID', 'NUMERIC', 'TEMPORAL', 'LOCATION', 'OTHER']:
    type_records = [r for r in records if categorize_question(...) == question_type]
    metrics_by_type[question_type] = evaluate_subset(type_records)

# Results
{
    'FACTOID': {'mrr': 0.65, 'recall@10': 0.72},
    'NUMERIC': {'mrr': 0.58, 'recall@10': 0.68},  # Harder questions
    'TEMPORAL': {'mrr': 0.62, 'recall@10': 0.70},
    'LOCATION': {'mrr': 0.71, 'recall@10': 0.78},  # Easier
    'OTHER': {'mrr': 0.60, 'recall@10': 0.68}
}
```

**Advantages:**
- ✅ Identifies which question types are hard
- ✅ Actionable insights (e.g., "improve NUMERIC question handling")

**Disadvantages:**
- ❌ More complex evaluation pipeline
- ❌ Some questions fit multiple categories

**Best for:** Detailed post-mortem analysis

---

### **Alternative C: Passage-Length Impact Analysis** (Diagnostic)

Analyze how passage length affects retrieval quality:

```python
def analyze_by_passage_length(records: List[Dict], results: List[Dict]):
    """
    Group results by passage length and compute metrics per bucket.
    
    Buckets:
      - SHORT: < 100 tokens
      - MEDIUM: 100-300 tokens
      - LONG: > 300 tokens
    """
    buckets = {'SHORT': [], 'MEDIUM': [], 'LONG': []}
    
    for record in records:
        passage = record['passages'][0]
        token_count = len(passage['text'].split())
        
        if token_count < 100:
            bucket = 'SHORT'
        elif token_count < 300:
            bucket = 'MEDIUM'
        else:
            bucket = 'LONG'
        
        buckets[bucket].append((record, results))
    
    # Compute metrics per bucket
    metrics_per_length = {}
    for bucket, data in buckets.items():
        metrics_per_length[bucket] = evaluate_subset(data)
    
    return metrics_per_length

# Results
{
    'SHORT': {'mrr': 0.78, 'recall@10': 0.85},  # Easier
    'MEDIUM': {'mrr': 0.65, 'recall@10': 0.73},
    'LONG': {'mrr': 0.52, 'recall@10': 0.65}   # Harder
}
```

**Advantages:**
- ✅ Identifies scalability issues
- ✅ Helps decide chunk size strategy

**Disadvantages:**
- ❌ Less actionable than question-type analysis
- ❌ Confounds with answer position (long passages have more potential positions)

**Best for:** Understanding chunking strategy effectiveness

---

### **Alternative D: Detailed Error Analysis Framework** (Post-mortem)

Categorize failures and count occurrences:

```python
def analyze_errors(records, results, metrics):
    """
    Categorize retrieval failures.
    
    Categories:
      - MISSING_ANSWER: Answer not in top-10 at all
      - ANSWER_LATE: Answer present but ranked > 5
      - SEMANTIC_GAP: Answer conceptually related but not retrieved
      - CHUNK_SPLIT: Answer split across chunks
    """
    errors = {
        'missing_answer': [],
        'answer_late': [],
        'semantic_gap': [],
        'chunk_split': []
    }
    
    for record in records:
        answer = record['output'][0]['answer']
        chunks = search(record['input'])
        
        if not any(answer in c['text'] for c in chunks):
            if semantic_similarity(...) > 0.6:
                errors['semantic_gap'].append(record)
            else:
                errors['missing_answer'].append(record)
        else:
            # Answer present - how far down?
            rank = next(i for i, c in enumerate(chunks) if answer in c['text'])
            if rank >= 5:
                errors['answer_late'].append((record, rank))
    
    # Summary
    return {
        'missing_answer': len(errors['missing_answer']),
        'answer_late': len(errors['answer_late']),
        'semantic_gap': len(errors['semantic_gap']),
        'total_failures': sum(len(e) for e in errors.values())
    }

# Results
{
    'missing_answer': 45,      # 9%
    'answer_late': 23,         # 5%
    'semantic_gap': 12,        # 2%
    'chunk_split': 8,          # 2%
    'total_failures': 88,      # Out of 500 = 17.6% failure rate
    'fixable_by_chunking': 8,  # chunk_split can be fixed
    'fixable_by_embeddings': 12  # semantic_gap needs better model
}
```

**Advantages:**
- ✅ Identifies root causes of failures
- ✅ Guides Phase 2 improvements

**Disadvantages:**
- ❌ Most complex to implement
- ❌ Requires manual inspection for validation

**Best for:** Detailed improvement planning for Phase 2

---

## Part 3: Strategy Comparison

### Quick Dev vs Comprehensive Final

| Aspect | Quick Dev (3 hours) | Comprehensive (6 hours) |
|--------|---|---|
| **Relevance Method** | Fuzzy matching (B) | Fuzzy + error analysis |
| **Metrics** | MRR, NDCG, Recall only | + Tier-based scores + by-type breakdown |
| **Dataset Size** | Test set (500 records) | Full test set (500) + sample analysis (50) |
| **Alternative Approaches** | Skip | All 4 alternatives implemented |
| **Error Analysis** | Basic counts | Detailed categorization (D) |
| **Output** | Core metrics only | Full diagnostic report |
| **Report Quality** | Summary stats | Deep insights + recommendations |
| **Effort** | 1 person, 1 day | 1 person, 1.5 days |

**Recommendation for Phase 1.3:** Start with Quick Dev, iterate to Comprehensive if findings warrant deeper analysis.

---

## Part 4: Implementation Checklist

### Phase 1.3.1 - Relevance Labeling
- [ ] Implement `is_relevant_exact()` function
- [ ] Implement `is_relevant_fuzzy()` function with SequenceMatcher
- [ ] Test on 10 sample Q&A pairs
- [ ] Benchmark speed: time 1000 checks
- [ ] Document approach choice rationale in code comments

### Phase 1.3.2 - Metric Computation
- [ ] Implement `compute_mrr()` with docstring examples
- [ ] Implement `compute_ndcg()` with log2 discount
- [ ] Implement `compute_recall()` for multiple k values
- [ ] Implement `compute_precision()` (if needed)
- [ ] Add unit tests for each metric with known values
- [ ] Verify formulas match CLAPnq paper if applicable

### Phase 1.3.3 - Evaluation Script
- [ ] Load vector store from disk
- [ ] Load CLAPnq test set (500 records)
- [ ] Loop through queries with timing
- [ ] Retrieve top-10 chunks per query
- [ ] Check relevance for each chunk
- [ ] Accumulate metrics
- [ ] Compute final aggregate metrics
- [ ] Print timing breakdown (loading, retrieval, eval)

### Phase 1.3.4 - Testing & Validation
- [ ] Test on subset (10 records) manually
- [ ] Verify MRR in [0.0, 1.0]
- [ ] Verify NDCG in [0.0, 1.0]
- [ ] Verify Recall matches manual inspection
- [ ] Run on full test set (500 records)
- [ ] Compare results against baseline expectations

### Phase 1.3.5 - Documentation & Reporting
- [ ] Generate `docs/EVALUATION_RESULTS.md` with metrics
- [ ] Create visualization of metric breakdown
- [ ] Document any anomalies or surprises
- [ ] Write recommendations for Phase 2
- [ ] Add implementation notes (approach chosen + why)

---

## Part 5: Failure Recovery

If metrics are poor (MRR < 0.60), use this diagnostic flow:

```
MRR < 0.60
├─ Check Recall@10
│  ├─ If Recall@10 < 0.60 → Vector store missing answers entirely
│  │  ├─ Step 1: Run error analysis (Alternative D) 
│  │  ├─ Step 2: Check if answers are in chunks at all (Phase 1.1 validation)
│  │  └─ Step 3: If yes, embeddings are weak → Phase 2.1 (upgrade model)
│  └─ If Recall@10 >= 0.70 → Answers present but ranked low
│     ├─ Step 1: Analyze ranking pattern (NDCG breakdown)
│     ├─ Step 2: If NDCG is also low → query-document mismatch
│     └─ Step 3: Phase 2.3 (cross-encoder reranking)
├─ Check Answer Coverage
│  └─ If < 0.95 → Chunking is splitting answers
│     ├─ Action: Increase overlap_sentences (Phase 1.1 retry)
│     └─ Or: Increase chunk_size (Phase 1.1 retry)
└─ Check Processing Time
   ├─ If > 500ms → FAISS query slow
   │  └─ Action: Profile FAISS, check index size
   └─ If ok → Everything working, metrics just reflect difficulty
```

---

## Part 6: Timeline & Effort Estimation

| Task | Time | Owner | Notes |
|------|------|-------|-------|
| 1.3.1 Relevance labeling | 30 min | Developer | Fuzzy matching approach |
| 1.3.2 Metric computation | 60 min | Developer | All 5 metrics with tests |
| 1.3.3 Evaluation script | 90 min | Developer | Main loop + result tracking |
| Testing & validation | 60 min | Developer | On subset + full set |
| Documentation | 30 min | Developer | Results + recommendations |
| **Total (Quick Dev)** | **4.5 hours** | | |
| Alternative approaches | 120 min | Optional | All 4 alternatives |
| Error analysis (Alt D) | 60 min | Optional | Detailed categorization |
| **Total (Comprehensive)** | **6.5 hours** | | |

---

## Expected Baseline Results

### Phase 1 Targets (all-MiniLM-L6-v2, 384 dims)

```
Metric           Target      Expectation
────────────────────────────────────────
MRR              0.62+       0.60-0.65
NDCG@10          0.68+       0.65-0.72
Recall@5         0.55+       0.50-0.60
Recall@10        0.70+       0.68-0.75
Answer Coverage  0.96+       0.95-0.98
Query time (ms)  <20         10-15 typical
```

---

## Next Steps After Phase 1.3

### If metrics are GOOD (MRR ≥ 0.62):
→ Proceed to Phase 2.1 - Hybrid Search
- Add BM25 sparse retrieval
- Combine dense + sparse with reciprocal rank fusion (RRF)
- Expected improvement: +5-10% on MRR

### If metrics are POOR (MRR < 0.60):
→ Debug before Phase 2:
1. Run Alternative D (error analysis)
2. Identify root cause from failure categories
3. If chunking issue: Fix Phase 1.1, rebuild Phase 1.2
4. If embedding weakness: Evaluate BGE-large upgrade path
5. Retry Phase 1.3 evaluation

---

*Prepared: 2026-04-29*  
*Status: Ready for Phase 1.3 Implementation*  
*Next: Execute evaluation script and collect baseline metrics*
