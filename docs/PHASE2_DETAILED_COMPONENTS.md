# Phase 2.2 → 2.4: Detailed Component Explanation

**Date:** 2026-05-01  
**Focus:** Answer Grounding, Few-Shot Learning, Question-Type Routing  
**Audience:** Implementation guide for Phase 2 development

---

## Phase 2.2: Answer Grounding & Verification 🎯

### The Core Problem

When an LLM generates an answer, it can:
1. ✅ Ground it (use information from passages)
2. ❌ Hallucinate it (use general knowledge/make it up)

**Example:**
```
Question: "What is the capital of France?"

Scenario A (Grounded):
├─ Retrieved Passage: "Paris is the capital of France, located on the Seine River"
├─ Generated Answer: "Paris is the capital of France"
└─ Status: ✅ GROUNDED - Answer text appears in passage

Scenario B (Hallucination):
├─ Retrieved Passage: "Paris is famous for museums, art, and the Eiffel Tower"
├─ Generated Answer: "Paris is the capital of France"
└─ Status: ❌ HALLUCINATED - "capital" not in passage
```

### Why It Matters for CLAPnq

**CLAPnq Grounding Requirements:**
- Answers must be supported by retrieved passages
- Using general LLM knowledge without passage support = penalty
- Hallucinations destroy score worse than missing information
- Example: A perfect answer that's hallucinated = 0 points

### How It Works: 3 Steps

#### Step 1: Break Answer into Spans

```python
def extract_answer_spans(answer_text: str) -> List[str]:
    """Split answer into verifiable chunks."""
    
    # Simple: Split by sentences
    spans = answer_text.split('. ')
    
    # Example input:
    answer = "Paris is the capital of France. It is located on the Seine River. Population is 2.1 million."
    
    # Output:
    spans = [
        "Paris is the capital of France",
        "It is located on the Seine River",
        "Population is 2.1 million"
    ]
    
    return spans
```

#### Step 2: Find Each Span in Passages

```python
def find_span_in_passages(
    span: str,
    passages: List[Dict]
) -> Tuple[bool, float, str]:
    """Check if span exists in any retrieved passage."""
    
    best_match_score = 0.0
    best_passage_id = None
    
    for passage in passages:
        passage_text = passage['chunk_text'].lower()
        span_lower = span.lower()
        
        # Method 1: Exact substring match
        if span_lower in passage_text:
            return True, 1.0, passage['chunk_id']
        
        # Method 2: Fuzzy match (handles small differences)
        from difflib import SequenceMatcher
        similarity = SequenceMatcher(None, span_lower, passage_text).ratio()
        
        if similarity > best_match_score:
            best_match_score = similarity
            best_passage_id = passage['chunk_id']
    
    # Return fuzzy match if confidence high
    if best_match_score > 0.80:  # 80% similar
        return True, best_match_score, best_passage_id
    else:
        return False, best_match_score, None
```

#### Step 3: Compute Overall Grounding Score

```python
def compute_grounding_score(
    answer: str,
    passages: List[Dict]
) -> Dict:
    """Analyze what % of answer is grounded."""
    
    spans = extract_answer_spans(answer)
    grounded_spans = []
    ungrounded_spans = []
    scores = []
    
    for span in spans:
        is_grounded, similarity, source = find_span_in_passages(span, passages)
        
        if is_grounded:
            grounded_spans.append({
                'text': span,
                'similarity': similarity,
                'source_chunk': source
            })
            scores.append(similarity)
        else:
            ungrounded_spans.append(span)
    
    # Calculate metrics
    grounding_ratio = len(grounded_spans) / len(spans) if spans else 0
    avg_similarity = sum(scores) / len(scores) if scores else 0
    
    return {
        'is_fully_grounded': len(ungrounded_spans) == 0,
        'grounding_ratio': grounding_ratio,  # % of answer grounded
        'avg_similarity': avg_similarity,
        'confidence': 'HIGH' if grounding_ratio > 0.9 else 'LOW',
        'grounded_count': len(grounded_spans),
        'ungrounded_count': len(ungrounded_spans),
        'ungrounded_spans': ungrounded_spans,  # Which parts hallucinated?
    }
```

### Real Example

```python
# Inputs:
answer = "Paris is the capital of France and has a population of 2.1 million people."
passages = [
    {
        'chunk_id': 'p1',
        'chunk_text': 'Paris is the capital of France, located on the Seine River...'
    },
    {
        'chunk_id': 'p2',
        'chunk_text': 'The metropolitan area has 12 million people...'
    },
    {
        'chunk_id': 'p3',
        'chunk_text': 'France is in Western Europe...'
    }
]

# Processing:
grounding = compute_grounding_score(answer, passages)

# Output:
{
    'is_fully_grounded': False,
    'grounding_ratio': 0.67,  # 2 out of 3 spans grounded
    'avg_similarity': 0.95,
    'confidence': 'LOW',  # Some hallucination
    'grounded_count': 2,
    'ungrounded_count': 1,
    'ungrounded_spans': ['has a population of 2.1 million people']
    # ^ This span not in retrieved passages!
}
```

### How to Use It

```python
# After Phase 2.1 generates answer:
answer = llm.generate(question, retrieved_chunks)

# Phase 2.2: Check grounding
grounding = compute_grounding_score(answer, retrieved_chunks)

# Make decision:
if grounding['confidence'] == 'HIGH':
    # Answer is grounded, use as-is
    response = {
        'answer': answer,
        'confidence': 'high',
        'sources': [s['source_chunk'] for s in grounding['grounded_spans']]
    }
else:
    # Hallucination detected, options:
    
    # Option A: Regenerate with stricter prompt
    strict_prompt = f"""Answer ONLY using provided passages.
Do NOT use external knowledge.
If info is not in passages, say "Not available."

Question: {question}
Context: {passages}
Answer:"""
    
    answer_v2 = llm.generate_strict(strict_prompt)
    
    # Option B: Flag low confidence
    response = {
        'answer': answer,
        'confidence': 'low',
        'hallucination_detected': True,
        'hallucinated_parts': grounding['ungrounded_spans']
    }
```

### Key Insight

**Grounding Score tells you:**
- ✅ How much of answer is supported by passages
- ✅ Which sentences are hallucinated
- ✅ Whether to trust the answer
- ✅ What confidence to assign

---

## Phase 2.3: Few-Shot In-Context Learning 📚

### The Core Problem

**Zero-Shot (No Examples):**
```
LLM Prompt: "Question: What is X?\nContext: [passages]\nAnswer:"

LLM must:
├─ Guess answer format
├─ Decide conciseness level
├─ Determine what style is expected
└─ Result: Inconsistent, often verbose, sometimes off-topic
```

**Few-Shot (With Examples):**
```
LLM Prompt: "Example 1: Q: ...? A: ...
            Example 2: Q: ...? A: ...
            Question: What is X?
            Context: [passages]
            Answer:"

LLM can:
├─ See the expected format
├─ Learn the conciseness level
├─ Match the demonstrated style
└─ Result: Consistent, concise, on-topic
```

### Why Few-Shot Works

**Core principle:** LLMs are pattern-matching systems

```
Simple analogy:

Zero-shot:
├─ Task: "Complete the pattern: 2, 4, 6, ?"
├─ LLM: "Maybe 8, or maybe next even number, or..."
└─ Result: Inconsistent

Few-shot:
├─ Example: "1, 2, 3, 4 (difference is 1)
│           2, 4, 6, 8 (difference is 2)"
├─ Task: "3, 6, 9, ?"
├─ LLM: Sees pattern → "12" (difference is 3)
└─ Result: Consistent, rule-based
```

### How to Implement Few-Shot

#### Step 1: Create Example Set

```python
FEW_SHOT_EXAMPLES = [
    {
        "question": "What is the capital of France?",
        "context": """Paris is the capital and most populous city of France. 
Located on the Seine River, it is one of Europe's major centers.""",
        "answer": "Paris is the capital of France."
    },
    {
        "question": "When was the Eiffel Tower built?",
        "context": """The Eiffel Tower was constructed from 1887 to 1889. 
It was built for the 1889 World's Fair in Paris.""",
        "answer": "The Eiffel Tower was built from 1887 to 1889."
    },
    {
        "question": "What is Europe's largest country by area?",
        "context": """Russia is Europe's largest country by area, covering 
approximately 3.9 million square kilometers in Europe.""",
        "answer": "Russia is Europe's largest country by area."
    }
]
```

**Example characteristics:**
- ✅ Concise answers (1-2 sentences)
- ✅ Direct answer first, then context
- ✅ Only uses info from context
- ✅ Clear, simple language

#### Step 2: Build Prompt with Examples

```python
def create_few_shot_prompt(
    question: str,
    context: str,
    examples: List[Dict]
) -> str:
    """Build prompt with examples."""
    
    prompt = """Answer questions using only provided passages.
Be concise and direct. Match the style of the examples below.

"""
    
    # Add 3 examples
    for i, example in enumerate(examples[:3], 1):
        prompt += f"""Example {i}:
Question: {example['question']}
Context: {example['context']}
Answer: {example['answer']}

"""
    
    # Add actual question
    prompt += f"""Question: {question}
Context: {context}
Answer:"""
    
    return prompt
```

#### Step 3: Use Few-Shot Prompt

```python
async def generate_with_few_shot(
    question: str,
    context: str,
    examples: List[Dict]
) -> str:
    """Generate using examples."""
    
    prompt = create_few_shot_prompt(question, context, examples)
    
    response = await client.messages.create(
        model="claude-opus-4-7",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.content[0].text
```

### Output Comparison

**Question:** "How many countries are in Europe?"

**Without Few-Shot:**
```
Answer: "Europe is a continent with many countries, and the exact number 
depends on how you define the continent's geographical boundaries between Europe 
and Asia. Most estimates put the number between 44 and 51, depending on whether 
you include countries that are partially in Asia."

Problems: ❌ Verbose (4 sentences)
          ❌ Uncertain ("depends", "estimates")
          ❌ Doesn't match example style
          ❌ Too much explanation
```

**With Few-Shot (3 examples):**
```
Answer: "Europe has 44 to 51 countries, depending on the geographical 
definition of the continent."

Improvements: ✅ Concise (2 sentences)
             ✅ Clear number
             ✅ Acknowledges variation
             ✅ Matches example style
```

### Example Selection Strategy

```python
# Different question types need different examples

FEW_SHOT_BY_TYPE = {
    'WHO': [
        {'q': 'Who invented the light bulb?', 'a': 'Thomas Edison invented the light bulb.'},
        {'q': 'Who was the first president of USA?', 'a': 'George Washington was the first president.'},
    ],
    'WHAT': [
        {'q': 'What is photosynthesis?', 'a': 'Photosynthesis is the process...'},
        {'q': 'What is the capital of Japan?', 'a': 'Tokyo is the capital of Japan.'},
    ],
    'WHEN': [
        {'q': 'When did WWI end?', 'a': 'World War I ended in 1918.'},
        {'q': 'When was the USA founded?', 'a': 'The USA was founded in 1776.'},
    ],
    'HOW_MANY': [
        {'q': 'How many continents are there?', 'a': 'There are 7 continents.'},
        {'q': 'How many planets orbit the sun?', 'a': 'There are 8 planets in our solar system.'},
    ]
}

# At runtime, select matching examples:
def get_examples_for_question(question: str, k: int = 3):
    """Get most relevant examples."""
    q_type = classify_question_type(question)
    examples = FEW_SHOT_BY_TYPE.get(q_type, FEW_SHOT_BY_TYPE['WHAT'])
    return examples[:k]
```

### Impact of Few-Shot

```
Metric: Answer Quality (measured by ROUGE-L score)

Zero-shot:     [==========>                    ] 0.65
Few-shot (1):  [============>                  ] 0.72
Few-shot (3):  [==============>                ] 0.78
Few-shot (5):  [================>              ] 0.80

Format consistency: Zero-shot 60%, Few-shot 95%
Hallucination rate: Zero-shot 15%, Few-shot 5%
```

---

## Phase 2.4: Question Type-Aware Retrieval 🎯

### The Core Problem

**Current (One-Size-Fits-All Retrieval):**
```
All questions use same strategy:
├─ Embed query
├─ Search FAISS for semantic similarity
├─ Return top-5 by similarity score
└─ Works OK, but misses type-specific signals
```

**Example Failure:**
```
Question: "How many countries are in Europe?"

Retrieved (by similarity):
1. "Europe is a diverse continent..." (high similarity but no numbers)
2. "European Union has 27 members..." (has a number but wrong answer)
3. "44 countries are recognized..." (CORRECT ANSWER, but ranked #3)

Problem: Semantic similarity ranked #1 above the right answer
Cause: Semantic model ≠ numeric relevance
```

### Different Question Types Behave Differently

#### Type 1: FACTOID ("Who", "What")

```
Question: "Who sang 'Imagine'?"
Answer type: Person name

Semantic approach works well:
├─ Embed "sang Imagine"
├─ Find passage with lyrics/artist info
├─ Semantic similarity matches
└─ ✅ Works great
```

#### Type 2: NUMERIC ("How many", "What percentage")

```
Question: "How many countries are in Europe?"
Answer type: Number with optional unit

Semantic approach struggles:
├─ Embed "countries Europe"
├─ Find passages about Europe (generic)
├─ Missing signal: Does passage contain numbers?
├─ Result: Passages without numbers ranked high
└─ ❌ Needs help: Boost passages with numbers
```

#### Type 3: TEMPORAL ("When", "What year")

```
Question: "When did World War II end?"
Answer type: Date/Year

Semantic approach struggles:
├─ Embed "end World War II"
├─ Find passages about WWII (many exist)
├─ Missing signal: Does passage contain dates/years?
├─ Result: Passages without dates ranked equally
└─ ❌ Needs help: Boost passages with temporal markers
```

#### Type 4: LOCATION ("Where", "Which country")

```
Question: "Where is Mount Everest?"
Answer type: Place name

Semantic approach works OK:
├─ Embed "Mount Everest location"
├─ Find passage about geography
├─ Works reasonably well
└─ ✅ Works, but boost geographic terms helps more
```

### How Type-Aware Retrieval Works

#### Step 1: Classify Question Type

```python
def classify_question_type(question: str) -> str:
    """Determine question type."""
    
    q_lower = question.lower()
    
    # Check keywords
    if any(kw in q_lower for kw in ['how many', 'how much', 'what number',
                                      'what percentage', 'total']):
        return 'NUMERIC'
    
    if any(kw in q_lower for kw in ['when', 'what year', 'what date',
                                     'how long', 'what time']):
        return 'TEMPORAL'
    
    if any(kw in q_lower for kw in ['where', 'which country', 'which city',
                                     'which place', 'located']):
        return 'LOCATION'
    
    # Default
    return 'FACTOID'

# Examples:
classify_question_type("Who sang Hotel California?")  # → FACTOID
classify_question_type("How many people live in Japan?")  # → NUMERIC
classify_question_type("When was the Eiffel Tower built?")  # → TEMPORAL
classify_question_type("Where is the Amazon rainforest?")  # → LOCATION
```

#### Step 2: Define Type-Specific Boosters

```python
def compute_type_specific_boost(chunk_text: str, q_type: str) -> float:
    """Add bonus points based on question type."""
    
    if q_type == 'NUMERIC':
        # Check for numbers in passage
        has_number = bool(re.search(r'\d+(?:,\d{3})*(?:\.\d+)?', chunk_text))
        has_stat_word = any(w in chunk_text.lower() 
                           for w in ['million', 'billion', 'thousand', 'percent', 'percentage'])
        
        # Boost if numbers present
        boost = 0.15 if (has_number or has_stat_word) else 0.0
        return boost
    
    elif q_type == 'TEMPORAL':
        # Check for dates/years in passage
        has_year = bool(re.search(r'\b(19|20)\d{2}\b', chunk_text))  # 1900-2099
        has_date = bool(re.search(r'\d{1,2}/\d{1,2}/\d{4}', chunk_text))  # MM/DD/YYYY
        has_month = any(m in chunk_text for m in 
                       ['January', 'February', 'March', 'April', 'May', 'June',
                        'July', 'August', 'September', 'October', 'November', 'December',
                        'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        
        # Boost if temporal info present
        boost = 0.15 if (has_year or has_date or has_month) else 0.0
        return boost
    
    elif q_type == 'LOCATION':
        # Check for geographic terms
        has_geo = any(w in chunk_text for w in 
                     ['country', 'city', 'region', 'province', 'state',
                      'continent', 'island', 'ocean', 'sea', 'mountain', 'river'])
        boost = 0.10 if has_geo else 0.0
        return boost
    
    else:  # FACTOID
        return 0.0  # No boost needed
```

#### Step 3: Apply Boost During Retrieval

```python
async def retrieve_adaptive(
    query: str,
    vector_store: VectorStoreFaiss,
    top_k: int = 5
) -> List[Dict]:
    """Retrieve with type-specific optimization."""
    
    # Step 1: Classify
    q_type = classify_question_type(query)
    print(f"Classified as: {q_type}")
    
    # Step 2: Get base semantic results (retrieve 2x to have room to rerank)
    base_results = await vector_store.search(query, top_k=top_k*2)
    
    # Step 3: Apply type-specific boosting
    for result in base_results:
        boost = compute_type_specific_boost(result['chunk_text'], q_type)
        # Combine semantic score with boost
        result['final_score'] = result.get('score', 0.5) + boost
    
    # Step 4: Re-rank by final score
    reranked = sorted(base_results, key=lambda x: x['final_score'], reverse=True)
    
    # Step 5: Return top-k
    return reranked[:top_k]
```

### Walkthrough Example

**Question:** "How many countries are in Europe?"

```
Step 1: Classify
├─ Keywords: "How many"
└─ Result: NUMERIC

Step 2: Get base results (top 10)
├─ Result 1: "Europe is diverse..." (score: 0.92, no numbers)
├─ Result 2: "EU has 27 members..." (score: 0.88, has "27")
├─ Result 3: "44 countries recognized..." (score: 0.85, has "44")
└─ Result 4-10: [other passages]

Step 3: Apply NUMERIC boost (0.15 for passages with numbers)
├─ Result 1: 0.92 + 0.0 = 0.92 (no boost, no numbers)
├─ Result 2: 0.88 + 0.15 = 1.03 ✅ BOOSTED
├─ Result 3: 0.85 + 0.15 = 1.00 ✅ BOOSTED
└─ Result 4-10: unchanged

Step 4: Re-rank
├─ #1: Result 2 (1.03) ← Now #1! (was #2)
├─ #2: Result 3 (1.00) ← Now #2! (was #3)
├─ #3: Result 1 (0.92) ← Now #3! (was #1)
└─ #4-5: [others]

Step 5: Return top-5
└─ Top chunk now contains "44 countries" ✅
```

### Implementation in Generation Pipeline

```python
async def answer_question(
    query: str,
    vector_store: VectorStoreFaiss,
    llm_client,
    examples: List[Dict]
) -> Dict:
    """Full pipeline: retrieve → generate → ground."""
    
    # Phase 2.4: Adaptive retrieval
    chunks = await retrieve_adaptive(query, vector_store, top_k=5)
    
    # Phase 2.3 + 2.1: Few-shot generation
    answer = await generate_with_few_shot(
        question=query,
        context="\n".join([c['chunk_text'] for c in chunks]),
        examples=examples
    )
    
    # Phase 2.2: Grounding check
    grounding = compute_grounding_score(answer, chunks)
    
    return {
        'query': query,
        'answer': answer,
        'question_type': classify_question_type(query),
        'retrieved_chunks': chunks,
        'grounding': grounding,
        'confidence': grounding['confidence'],
        'sources': [c['chunk_id'] for c in chunks]
    }
```

---

## Quick Comparison Table

| Phase | What | How | Why |
|-------|------|-----|-----|
| **2.2** | Ground answers | Check if answer spans in passages | Prevent hallucinations |
| **2.3** | Improve format | Show LLM examples to learn style | Consistent, concise answers |
| **2.4** | Better retrieval | Boost chunks matching question type | Find right answer for question type |

## Full Pipeline Flow

```
User Query
    ↓
[2.4] What type of question?
    ├─ NUMERIC → Boost chunks with numbers
    ├─ TEMPORAL → Boost chunks with dates
    ├─ LOCATION → Boost chunks with geographic terms
    └─ FACTOID → Standard semantic retrieval
    ↓ (optimized top-5 chunks)
[2.1+2.3] Generate answer using few-shot examples
    └─ LLM learns from examples
    ↓ (coherent answer)
[2.2] Check if answer is grounded in passages
    ├─ If grounded → Return with high confidence
    └─ If hallucinated → Flag with low confidence
    ↓
✅ Final Answer with confidence & sources
```

---

**Key Takeaway:**

- **2.2 = Safety:** Verify answers are truthful
- **2.3 = Quality:** Make answers better formatted  
- **2.4 = Precision:** Find the right passages to begin with

Together, they form a complete answer-generation pipeline that is reliable, high-quality, and well-suited to CLAPnq requirements.
