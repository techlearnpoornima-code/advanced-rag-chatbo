📊 Chunking Technique Examples
Sample Passage (from CLAPnq dataset)

Passage: "France"
=================================================================

Sentence 1: "France is a country located in Western Europe."
Sentence 2: "It is the largest country in the European Union by area."
Sentence 3: "The capital and most populous city is Paris."
Sentence 4: "Paris is famous for the Eiffel Tower and is known as the City of Light."
Sentence 5: "France has a rich cultural heritage spanning thousands of years."
Sentence 6: "The country is a major hub for art, fashion, and cuisine."

Token counts: S1=10, S2=13, S3=10, S4=16, S5=12, S6=12
Total: 73 words (~87 tokens with whitespace)
Answer span: Sentences 3 & 4 (answer to "What is the capital of France?")
OPTION A: Sentence-Based Semantic Chunking
(Recommended - Respects sentence boundaries)
Configuration: max_tokens=50, overlap=0 sentences


CHUNK 1:
┌─────────────────────────────────────────────┐
│ Sentence 1: "France is a country..."        │ (10 tokens)
│ Sentence 2: "It is the largest country..."  │ (13 tokens)
│ Sentence 3: "The capital is Paris."         │ (10 tokens)
│ Total: 33 tokens ✓ (under limit)            │
└─────────────────────────────────────────────┘

CHUNK 2:
┌─────────────────────────────────────────────┐
│ Sentence 4: "Paris is famous for..."        │ (16 tokens)
│ Sentence 5: "France has a rich..."          │ (12 tokens)
│ Total: 28 tokens ✓ (under limit)            │
└─────────────────────────────────────────────┘

CHUNK 3:
┌─────────────────────────────────────────────┐
│ Sentence 6: "The country is a major hub..." │ (12 tokens)
│ Total: 12 tokens ✓ (under limit)            │
└─────────────────────────────────────────────┘

Results:
  - 3 chunks from 1 passage
  - No sentence is split ✓
  - Answer (S3 & S4) is in CHUNKS 1 & 2
  - Clear semantic boundaries ✓
  - Variable chunk sizes (28-33 tokens)
Pros:
✅ No sentence splitting

✅ Preserves semantic meaning

✅ Answer spans preserved

✅ Natural reading flow

Cons:
❌ Variable sizes harder to batch

❌ Slight storage overhead

OPTION B: Fixed-Size Sliding Window
(Simple but may split sentences)
Configuration: chunk_size=40 chars, overlap=10 chars


Looking at character positions:
S1: "France is a country..." [0-47 chars]
S2: "It is the largest..." [48-120 chars]
S3: "The capital is Paris." [121-165 chars]
S4: "Paris is famous..." [166-248 chars]
S5: "France has a rich..." [249-324 chars]
S6: "The country..." [325-362 chars]

CHUNK 1: [0-40 chars]
┌─────────────────────────────────────────────┐
│ "France is a country located in Wes"        │
│ ⚠️ SENTENCE SPLIT MID-WORD!                 │
└─────────────────────────────────────────────┘

CHUNK 2: [30-70 chars] (30 char overlap)
┌─────────────────────────────────────────────┐
│ "country located in Western Europe. It"      │
│ ✓ Covers gap between sentences              │
└─────────────────────────────────────────────┘

CHUNK 3: [60-100 chars]
┌─────────────────────────────────────────────┐
│ "Europe. It is the largest country in"      │
│ ⚠️ SENTENCE MID-WORD SPLIT AGAIN            │
└─────────────────────────────────────────────┘

CHUNK 4: [90-130 chars]
┌─────────────────────────────────────────────┐
│ "t in the European Union by area. The"      │
│ ⚠️ SENTENCE 3 PARTIALLY IN TWO CHUNKS       │
└─────────────────────────────────────────────┘

... continues with more splits...

Results:
  - 8-10 chunks from 1 passage
  - Multiple sentence splits ❌
  - Answer span split across 2-3 chunks ❌
  - Inconsistent context ❌
  - Predictable sizes (all ~40 chars) ✓
Pros:
✅ Consistent chunk sizes

✅ Simple to implement

✅ Easy to batch

Cons:
❌ Splits sentences mid-word

❌ Loses semantic coherence

❌ Answer spans fragmented

❌ Poor readability

OPTION C: Hybrid (Semantic + Size Constraint)
(Balance between A & B)
Configuration: max_tokens=50, min_tokens=30, overlap=1 sentence


CHUNK 1: (Sentences grouped + overlap)
┌─────────────────────────────────────────────┐
│ Sentence 1: "France is a country..."        │ (10 tokens)
│ Sentence 2: "It is the largest country..."  │ (13 tokens)
│ Sentence 3: "The capital is Paris."         │ (10 tokens)
│ Total: 33 tokens ✓ (30-50 range)            │
│ Overlap marker: ► S3 will repeat in next    │
└─────────────────────────────────────────────┘

CHUNK 2: (With 1-sentence overlap)
┌─────────────────────────────────────────────┐
│ Sentence 3: "The capital is Paris." [REPEAT]│ (10 tokens - overlap)
│ Sentence 4: "Paris is famous for..."        │ (16 tokens)
│ Total: 26 tokens ✓ (meets min threshold)    │
│ Overlap marker: ► S4 will repeat in next    │
└─────────────────────────────────────────────┘

CHUNK 3: (With 1-sentence overlap)
┌─────────────────────────────────────────────┐
│ Sentence 4: "Paris is famous..." [REPEAT]   │ (16 tokens - overlap)
│ Sentence 5: "France has a rich..."          │ (12 tokens)
│ Total: 28 tokens ✓ (meets min threshold)    │
│ Overlap marker: ► S5 will repeat in next    │
└─────────────────────────────────────────────┘

CHUNK 4: (Final chunk)
┌─────────────────────────────────────────────┐
│ Sentence 5: "France has a rich..." [REPEAT] │ (12 tokens - overlap)
│ Sentence 6: "The country is..."             │ (12 tokens)
│ Total: 24 tokens ✓ (final chunk can be <30) │
└─────────────────────────────────────────────┘

Results:
  - 4 chunks from 1 passage
  - No sentence splitting ✓
  - Answer (S3 & S4) intact in chunks 1-3 ✓
  - Context overlap between chunks ✓
  - Some redundancy (S3 & S4 repeated) ⚠️
  - Variable but bounded sizes (24-33 tokens)
Pros:
✅ No sentence splitting

✅ Context preservation via overlap

✅ Respects semantic boundaries

✅ Better than pure sliding window

Cons:
⚠️ Storage redundancy (overlap)

⚠️ More complex logic

⚠️ Still variable sizes

Comparison Table
Aspect	Option A (Semantic)	Option B (Sliding)	Option C (Hybrid)
Chunks Created	3	8-10	4
Sentence Splits	✅ None	❌ Multiple	✅ None
Answer Integrity	✅ Preserved	❌ Fragmented	✅ Preserved
Semantic Coherence	✅ High	❌ Low	✅ High
Consistent Sizes	❌ Variable	✅ Fixed	⚠️ Mostly fixed
Storage Efficient	✅ No overlap	✅ No redundancy	❌ Has overlap
Batching Easy	❌ Variable	✅ Uniform	⚠️ Mostly uniform
Retrieval Quality	✅ Excellent	❌ Poor	✅ Very good
Implementation	⚠️ Medium	✅ Simple	❌ Complex
Real-World Impact on CLAPnq
For the Query: "What is the capital of France?"
Option A Results:


Query embedding → Find similar chunks → CHUNK 1 & 2 returned
  ↓
Both contain full context:
  - "The capital is Paris" (S3)
  - "Paris is famous for the Eiffel Tower..." (S4)
  ↓
LLM generates: "Paris is the capital of France and is famous for..."
✅ High-quality answer with context
Option B Results:


Query embedding → Find similar chunks → CHUNKS 3, 4, 5 returned
  ↓
Partial context:
  - Chunk 3: "...Europe. It is the largest country in"
  - Chunk 4: "t in the European Union by area. The"
  - Chunk 5: "capital is Paris. Paris is famous"
  ↓
LLM generates: "...the largest country... the capital is Paris..."
❌ Fragmented, less coherent answer
Our Choice: Option A (Sentence-Based Semantic)
Why for CLAPnq?
Dataset provides sentences - Pre-segmented boundaries reduce overhead
Answer spans use sentence indices - Natural alignment with chunking
Variable passages - Handles 62-word to 10,598-word gracefully
Better retrieval - Semantic chunks match query better
Cleaner chunks - No mid-sentence splits means better context
Configuration:

Max tokens:        512
Min tokens:        50
Overlap sentences: 1
Preserve boundaries: True
This balances:

✅ No sentence splits
✅ Semantic preservation
✅ Efficient storage
✅ Good retrieval quality
✅ Practical batch sizes
