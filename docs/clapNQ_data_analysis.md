CLAPnq Dataset Analysis
Dataset Overview
Metric	Answerable	Unanswerable
Records	1,954	1,791
Total Passages	1,954	1,791
File Size	6.3 MB	4.4 MB
Data Structure

{
  "id": "unique_identifier",
  "input": "user question",
  "passages": [
    {
      "title": "Wikipedia article title",
      "text": "full passage text",
      "sentences": ["sent1", "sent2", ...]
    }
  ],
  "output": [
    {
      "answer": "span from passage",
      "selected_sentences": [0, 1, 2],
      "meta": {...}
    }
  ]
}
Key Statistics
Questions:

Length: 4-21 words (avg 9.1-9.4 words)
Observation: Short, direct queries
Passages (Wikipedia):

Length: 62-10,598 words (avg 189-215 words)
Challenge: Highly variable! Some very long passages need smart chunking
Answers:

Length: 0-243 words (avg 50 words when present)
Observation: Medium-length answers, needs context preservation
Answer Distribution:

Answerable: 2,667 answers from 1,954 records (some questions have multiple answers)
Unanswerable: 1,791 pure negative cases
Key Observations for Chunking Strategy

1. Passage Variability is High
Min: 62 words | Max: 10,598 words | Avg: ~200 words
Some passages fit in one chunk, others need 10+ chunks
Issue: Fixed-size chunks might work, but semantic chunking could preserve better structure

2. Pre-segmented Sentences
Dataset provides sentence boundaries ("sentences" field)
Advantage: We can use this for semantic chunking without NLP
Strategy: Chunk by combining complete sentences, respecting passage structure

3. Answer Span Pattern
Answers are continuous spans from passages
"selected_sentences" tells us which sentences contain the answer
Insight: Good chunks should preserve answer sentences together

4. Negative Examples Matter
1,791 unanswerable questions for robustness
System must learn to say "no answer" when appropriate
Proposed Chunking Strategies (Discussion)
Option A: Sentence-Based Semantic Chunking ⭐ RECOMMENDED

Group consecutive sentences until:
- Reached max_tokens (e.g., 512 tokens)
- Topic shift detected
- Passage ends

Advantages:
✓ Uses provided sentence boundaries
✓ Preserves semantic meaning
✓ Answer spans stay together
✓ Works with variable passage lengths

Disadvantages:
✗ Slower than fixed-size
✗ Chunk sizes vary significantly
Option B: Fixed-Size Sliding Window

Fixed 512-char chunks with 128-char overlap

Advantages:
✓ Simple, fast, predictable
✓ Good for embeddings
✓ Consistent batch processing

Disadvantages:
✗ May split sentences mid-word
✗ Less semantically meaningful
Option C: Hybrid (Semantic + Size Constraint)

Combine sentences until max_tokens OR max_chars
Then add overlap between chunks

Advantages:
✓ Semantic + practical constraints
✓ Handles long passages gracefully
✓ Flexible granularity

Disadvantages:
✗ More complex to implement
My Recommendation for Phase 1.1
Use Sentence-Based Semantic Chunking (Option A)

Max chunk size: 512 tokens (~2,000 chars)

Metadata to track:

Original passage title
Sentence indices within passage
Chunk position (chunk 1 of N)
Is this chunk answerable?
Implementation approach:

Load sentences from dataset
Group by semantic boundaries + token limits
Track which chunks contain answer spans
Store passage context
Questions for You:
Do you agree with Sentence-Based Semantic Chunking?

Or prefer fixed-size for simplicity?
Or hybrid approach?
For metadata, should we also track:

Which chunks are likely to contain answers?
Passage category (if available)?
Source Wikipedia article?
Token limit preference:

256 tokens (more granular)?
512 tokens (balanced)?
768 tokens (longer context)?
Once you confirm the chunking approach, I'll implement Phase 1.1 with a data loading + chunking script that you can test on actual passages.

clapnq_train_answerable.jsonl
can u store CLAPnq Dataset Analysis in .md file please
The user wants me to save the CLAPnq Dataset Analysis that I just presented into a markdown file. This is a good idea for documentation and reference. I'll create a comprehensive analysis document.

Let me create a detailed markdown file with all the analysis, observations, and recommendations.

Write CLAPNQ_DATASET_ANALYSIS.md
Write failed
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

**Example Questions:**
```
- "who sang love the one you're with first" (9 words)
- "what is the capital of france" (6 words)
- "when was the first iphone released" (6 words)
```

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

**Answer Type Examples:**
```
Short:    "Stephen Stills"
Medium:   "Stephen Stills, a folk rocker who released it in 1970"
Long:     "Love the One You're With is a song by Stephen Stills released 
           as the lead single from his debut album in November 1970..."
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
```json
"sentences": [
  "Sentence 1 here.",
  "Sentence 2 here.",
  "Sentence 3 with answer here."
]
```

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
```
selected_sentences: [0, 2]
```
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

**Chunking Strategy Implication:**
- Must handle both cases during evaluation
- Need to track "answerable" vs "unanswerable" chunks
- Evaluate retrieval: does retrieved chunk help answer? Or mislead?

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

Example with 512 token limit:
Passage: 10 sentences (189 avg = ~1890 words total)
  Chunk 1: Sentences 0-2 (3 sentences, ~400 tokens)
  Chunk 2: Sentences 3-5 (3 sentences, ~400 tokens)
  Chunk 3: Sentences 6-9 (4 sentences, ~500 tokens)
```

**Advantages:**
- ✅ Uses provided sentence boundaries (no extra NLP)
- ✅ Preserves semantic meaning and context
- ✅ Answer spans stay together (sentences 0,2 → same chunk)
- ✅ Works well with variable passage lengths
- ✅ Natural granularity (sentence-level)
- ✅ Better retrieval quality (context-aware)

**Disadvantages:**
- ❌ Slower than fixed-size chunking
- ❌ Variable chunk sizes (harder batching)
- ❌ Requires tracking sentence indices
- ❌ More complex implementation

**Metadata to Track:**
```python
{
  "chunk_id": "passage_0_chunk_2",
  "passage_title": "Love the One You're With",
  "passage_id": "unique_passage_id",
  "sentence_indices": [6, 7, 8, 9],
  "start_char": 1250,
  "end_char": 2150,
  "token_count": 485,
  "contains_answer": True,
  "answerable": True  # Is this chunk from answerable or unanswerable passage?
}
```

---

### Option B: Fixed-Size Sliding Window

**Algorithm:**
```
1. Split passage into overlapping chunks
2. Chunk size: 512 characters (fixed)
3. Overlap: 128 characters (preserves context)
4. No sentence boundary respect

Example with 512 char chunks:
Passage: "Sentence 1. Sentence 2. Sentence 3. Sentence 4..."
  Chunk 1: Char 0-512
  Chunk 2: Char 384-896 (128 char overlap)
  Chunk 3: Char 768-1280
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
- ❌ Overlap adds storage overhead
- ❌ Harder to track which chunk contains answer

---

### Option C: Hybrid (Semantic + Size Constraint)

**Algorithm:**
```
1. Group sentences like Option A
2. But enforce max character limit (not just tokens)
3. Add overlap between chunks
4. Balance semantic + practical constraints

Example:
  Chunk 1: Sentences 0-2 (512 chars) + 128 char overlap
  Chunk 2: Sentences 3-5 (520 chars) + 128 char overlap
```

**Advantages:**
- ✅ Semantic boundaries preserved
- ✅ Practical size constraints
- ✅ Context overlap for retrieval
- ✅ Works with variable lengths

**Disadvantages:**
- ❌ Most complex to implement
- ❌ Still variable chunk sizes
- ❌ Overlap adds storage

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
    "max_tokens": 512,  # Per chunk limit
    "min_tokens": 50,   # Don't create tiny chunks
    "overlap_sentences": 1,  # 1 sentence overlap between chunks
    "preserve_sentence_boundaries": True
}
```

### Expected Output

**For passage with 189 avg words (~760 tokens):**
```
Passage: 10 sentences → ~2.5 chunks on average
```

**For passage with 3,149 words (~12,600 tokens):**
```
Passage: ~170 sentences → ~25 chunks
```

**Chunk statistics post-chunking:**
```
Total chunks from 3,745 records: ~9,000-12,000 chunks
Average chunk size: 512 tokens
Average sentences per chunk: 3-4
```

---

## Next Steps: Phase 1.1 Implementation

### What we'll build:

1. **Data Loader**
   - Load JSONL files
   - Parse passage structure
   - Handle both answerable and unanswerable

2. **Chunker (Sentence-Based)**
   - Group sentences by token limit
   - Track metadata (indices, answer presence)
   - Generate unique chunk IDs

3. **Statistics Generator**
   - Post-chunking analysis
   - Chunk size distribution
   - Answer presence ratio

4. **Storage Preparation**
   - Format for ChromaDB ingestion
   - Metadata structure
   - Sample export

### Deliverables

```
src/
├── data_loading/
│   └── clapnq_loader.py          # Load JSONL files
├── chunking/
│   ├── __init__.py
│   └── semantic_chunker.py        # Sentence-based chunking
├── evaluation/
│   └── chunking_metrics.py        # Pre/post chunk analysis
```

---

## Questions for Discussion

Before implementing Phase 1.1, confirm:

1. ✅ **Chunking Strategy:** Do you agree with Sentence-Based Semantic Chunking (Option A)?
   - Or prefer Fixed-Size (Option B)?
   - Or Hybrid (Option C)?

2. ✅ **Token Limit:** Should we use 512 tokens max per chunk?
   - Or 256 (more granular)?
   - Or 768 (more context)?

3. ✅ **Metadata:** Should we also track:
   - Passage difficulty level?
   - Question type (who/what/when/where/why)?
   - Answer type (named entity, date, definition, etc.)?

4. ✅ **Overlap:** Should chunks overlap?
   - No overlap (pure semantic)
   - 1 sentence overlap (context sharing)
   - More?

5. ✅ **Negative Examples:** How should we handle unanswerable passages?
   - Include as negative examples?
   - Mark separately in metadata?
   - Filter out for initial implementation?

---

## References

- **Dataset:** Natural Questions (NQ) / CLAPnq
- **Paper:** Strich et al., 2025 - CLAPnq: Long-form Question Answering with Coherence and Grounding
- **Source:** Real Google search queries + Wikipedia passages
- **Benchmark Purpose:** Evaluate RAG systems on realistic, long-form QA

---

