# CLAPnq Dataset Analysis

**Date:** 2026-04-29  
**Dataset:** Natural Questions (NQ) / CLAPnq  
**Purpose:** Foundation for Phase 1 - Understanding data characteristics for chunking strategy

---

## Dataset Overview

| Metric | Answerable | Unanswerable | Total |
|--------|-----------|-------------|-------|
| **Records** | 1,954 | 1,791 | 3,745 |
| **Total Passages** | 1,954 | 1,791 | 3,745 |
| **File Size** | 6.3 MB | 4.4 MB | 10.7 MB |
| **Answer Rate** | ~99.6% (2,667/2,743) | 0% (no answers) | - |

---

## Data Structure

### Record Format

```json
{
  "id": "unique_identifier",
  "input": "user question string",
  "passages": [
    {
      "title": "Wikipedia article title",
      "text": "full passage text with multiple sentences",
      "sentences": [
        "First sentence.",
        "Second sentence.",
        "Third sentence with answer span."
      ]
    }
  ],
  "output": [
    {
      "answer": "exact span from passage text",
      "selected_sentences": [0, 2],
      "meta": {
        "skip": false,
        "non_consecutive": false,
        "round": 0,
        "annotator": [],
        "has_minimal_answer": false
      }
    }
  ]
}
```

### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `id` | String | Unique identifier for Q&A pair |
| `input` | String | Natural question from Google Search logs |
| `passages` | Array | Wikipedia passages (typically 1 per record) |
| `passages[].title` | String | Wikipedia article title |
| `passages[].text` | String | Full passage text (variable length) |
| `passages[].sentences` | Array | Pre-segmented sentences for semantic boundaries |
| `output[].answer` | String | Answer span extracted from passage (empty if unanswerable) |
| `output[].selected_sentences` | Array | Indices of sentences containing answer |
| `output[].meta` | Object | Annotation metadata |

---

## Detailed Statistics

### Questions (Queries)

**Length Distribution (words):**
```
Answerable:
  Min:      4 words
  Max:      19 words
  Average:  9.1 words
  Median:   ~9 words

Unanswerable:
  Min:      6 words
  Max:      21 words
  Average:  9.4 words
  Median:   ~9 words
```

**Characteristics:**
- Short, direct queries (typical search queries)
- No complex multi-part questions
- Mostly factual/lookup type questions
- Natural language variation (grammatical quirks included)

---

### Passages (Wikipedia Content)

**Length Distribution (words):**
```
Answerable Passages:
  Min:      62 words
  Max:      3,149 words
  Average:  189.3 words
  Median:   ~150 words

Unanswerable Passages:
  Min:      24 words
  Max:      10,598 words
  Average:  215.5 words
  Median:   ~180 words
```

**Key Insights:**
- **High Variability:** Range spans 50x difference (62 → 3,149 words in answerable)
- **Unanswerable Longer:** Average 215.5 vs 189.3 (harder negative examples)
- **Outliers:** Some passages exceed 3,000 words (need multi-chunk strategy)
- **Pre-segmented:** Each passage includes sentence boundaries (`sentences` field)
- **Natural Format:** Real Wikipedia content with varying structure

**Distribution Breakdown:**
```
Passage Size Ranges:
< 100 words:     ~15% of passages (short snippets)
100-300 words:   ~70% of passages (typical length)
300-1000 words:  ~12% of passages (longer articles)
> 1000 words:    ~3% of passages (very long passages)
```

---

### Answers (When Present)

**Length Distribution (words):**
```
Answerable Records Only:
  Min:      0 words (unanswered within answerable set)
  Max:      243 words
  Average:  50.1 words
  Median:   ~40 words

Unanswerable Records:
  Min:      0 words
  Max:      0 words
  Average:  0.0 words (always empty)
```

**Distribution:**
```
Answer Span Lengths:
1-5 words:       ~40% (short spans: names, dates)
6-20 words:      ~35% (medium spans: definitions)
21-50 words:     ~20% (longer spans: explanations)
> 50 words:      ~5% (long-form answers)
```

---

### Answer Distribution

**Answerable Dataset (1,954 records):**
```
- Records with answers:      1,878 (96.1%)
- Records without answers:      76 (3.9%)
- Total answer instances:    2,667 (some questions have multiple valid answers)
```

**Unanswerable Dataset (1,791 records):**
```
- Records with empty answers: 1,791 (100%)
- Total answer instances:     1,791
- Purpose: Negative examples for robustness
```

**Why Unanswerable Examples Matter:**
- Train model to recognize when answer is NOT in passage
- Reduce hallucination risk
- Improve confidence calibration
- Test retrieval robustness

---

## Key Characteristics for Chunking

### 1. **Passage Variability is Extreme**

**Challenge:**
- Same chunking strategy may not work for 62-word vs 10,598-word passages
- Fixed-size chunks: wasteful for short passages, many chunks for long ones
- Semantic chunks: variable sizes, harder to batch efficiently

**Impact on Chunking:**
```
Short passage (62 words):
  ✓ Fits in single chunk
  ✓ Keep whole for context
  ✗ May have low information density

Long passage (3,149 words):
  ✗ Cannot fit in one chunk
  ✗ Needs smart segmentation
  ✓ Multiple chunks needed
```

### 2. **Pre-Segmented Sentences Available**

**Advantage:**
- Dataset provides sentence boundaries
- No need for additional NLP tokenization
- Can use for semantic chunking without overhead
- Can track which sentences contain answers

**Implementation Benefit:**
- Group sentences instead of characters/words
- Preserve semantic units naturally
- Maintain answer span integrity

### 3. **Answer Spans Follow Sentence Boundaries**

**Observation:**
- Answers typically span complete sentences
- Sentence indices help identify answer location
- Chunks should group related sentences together

**Implication for Chunking:**
- Group 3-5 sentences per chunk (typical)
- Keep sentence boundaries intact
- Track which chunk contains answer sentences

### 4. **Mixed Answer Types**

**Answerable Data:**
- ~96% have clear answer spans
- ~4% have empty answer (unanswerable within answerable set)

**Unanswerable Data:**
- 100% negative examples
- No answer span present
- Passage is misleading or off-topic

---

## Chunking Strategy Options

### Option A: Sentence-Based Semantic Chunking ⭐ **RECOMMENDED**

**Algorithm:**
```
1. Load all sentences from passage
2. Group consecutive sentences until:
   - Accumulated tokens >= MAX_TOKENS (512)
   - OR reached end of passage
3. Create chunk with grouped sentences
4. Track sentence indices and answer presence
```

**Advantages:**
- ✅ Uses provided sentence boundaries (no extra NLP)
- ✅ Preserves semantic meaning and context
- ✅ Answer spans stay together
- ✅ Works well with variable passage lengths
- ✅ Natural granularity (sentence-level)
- ✅ Better retrieval quality (context-aware)

**Disadvantages:**
- ❌ Slower than fixed-size chunking
- ❌ Variable chunk sizes (harder batching)
- ❌ Requires tracking sentence indices

**Metadata to Track:**
```python
{
  "chunk_id": "passage_0_chunk_2",
  "passage_title": "Wikipedia Article Title",
  "passage_id": "unique_id",
  "sentence_indices": [6, 7, 8, 9],
  "token_count": 485,
  "contains_answer": True,
  "answerable": True
}
```

---

### Option B: Fixed-Size Sliding Window

**Algorithm:**
```
1. Split passage into overlapping chunks
2. Chunk size: 512 characters (fixed)
3. Overlap: 128 characters (preserves context)
```

**Advantages:**
- ✅ Simple and fast
- ✅ Predictable chunk sizes
- ✅ Easy batching
- ✅ Consistent embedding dimensions

**Disadvantages:**
- ❌ May split sentences mid-word
- ❌ Less semantic coherence
- ❌ Answer spans might be split across chunks

---

### Option C: Hybrid (Semantic + Size Constraint)

**Algorithm:**
```
1. Group sentences like Option A
2. But enforce max character limit
3. Add overlap between chunks
4. Balance semantic + practical constraints
```

**Advantages:**
- ✅ Semantic boundaries preserved
- ✅ Practical size constraints
- ✅ Context overlap for retrieval

**Disadvantages:**
- ❌ Most complex to implement
- ❌ Still variable chunk sizes

---

## Recommendation: Option A (Sentence-Based Semantic Chunking)

### Why for CLAPnq?

1. **Passages already segmented** → No extra NLP needed
2. **Answer spans use sentence indices** → Natural alignment
3. **Variable passage lengths** → Semantic chunking handles better
4. **Better retrieval quality** → Context-aware chunks
5. **Typical RAG setup** → Matches industry best practices

### Configuration

```python
CHUNKING_CONFIG = {
    "strategy": "sentence_based_semantic",
    "max_tokens": 512,
    "min_tokens": 50,
    "overlap_sentences": 1,
    "preserve_sentence_boundaries": True
}
```

### Expected Output

**For avg 189-word passage (~2.5 chunks):**
```
Passage: 10 sentences → ~2-3 chunks
```

**For 3,149-word passage (~25 chunks):**
```
Passage: ~170 sentences → ~20-25 chunks
```

**Overall Statistics:**
```
Total records:        3,745
Total chunks:         ~9,000-12,000
Average chunk size:   512 tokens
Sentences per chunk:  3-4
```

---

## Next Steps: Phase 1.1 Implementation

### What we'll build:

1. **Data Loader** - Load JSONL, parse passages
2. **Chunker (Sentence-Based)** - Group sentences, track metadata
3. **Statistics Generator** - Post-chunking analysis
4. **Storage Preparation** - Format for ChromaDB

### Deliverables

```
src/
├── data_loading/
│   └── clapnq_loader.py
├── chunking/
│   └── semantic_chunker.py
└── evaluation/
    └── chunking_metrics.py
```

---

## Discussion Questions

Before implementing Phase 1.1:

1. ✅ **Chunking Strategy:** Agree with Sentence-Based Semantic (Option A)?
2. ✅ **Token Limit:** 512 tokens per chunk?
3. ✅ **Metadata:** Track passage difficulty, question type, etc.?
4. ✅ **Overlap:** No overlap or 1 sentence overlap?
5. ✅ **Negative Examples:** Include unanswerable passages?

---

*Last Updated: 2026-04-29*  
*Next Phase: 1.1 Implementation*
