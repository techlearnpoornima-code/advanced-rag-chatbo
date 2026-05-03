# Phase 2.5 — Generation Evaluation Strategy

**Why Token F1, ROUGE-L, and BERTScore each break on CLAPnq — and why LLM-as-Judge
is the right evaluation approach for this dataset.**

---

## The Core Problem: What Are CLAPnq Gold Answers?

Before choosing a metric, we need to understand what we are comparing against.

In most QA datasets (SQuAD, TriviaQA), gold answers are short entity labels:

```
Question : Who sang "Love the One You're With"?
Gold (SQuAD-style): Stephen Stills
```

CLAPnq gold answers are **full Wikipedia passage extracts** — the exact sentences the
annotator highlighted as the answer region:

```
Question : Who sang "Love the One You're With"?

Gold (CLAPnq):
  "Love the One You're With" is a song by folk rocker Stephen Stills.
  David Crosby and Graham Nash, Stills' fellow members of Crosby, Stills &
  Nash, provide background vocals on the song. The song was released as a
  single in 1970 from his debut album Stephen Stills. It reached number 14
  on the Billboard Hot 100 ...
```

Our RAG system produces a **concise, synthesized answer** — correct and useful.
The gold answer is a long passage extract — complete but verbose.

This asymmetry breaks every word-overlap metric, and even BERTScore has its own
failure mode. This is why LLM-as-Judge is the right choice.

---

## Why Token F1 Fails

### How Token F1 Works

```
Precision = |generated_tokens ∩ gold_tokens| / |generated_tokens|
Recall    = |generated_tokens ∩ gold_tokens| / |gold_tokens|
F1        = 2 × Precision × Recall / (Precision + Recall)
```

### Concrete Failure Example

```
Generated : "Stephen Stills sang Love the One You're With in 1970."
            → 13 tokens

Gold      : "Love the One You're With is a song by folk rocker Stephen Stills.
             David Crosby and Graham Nash, Stills' fellow members of Crosby,
             Stills & Nash, provide background vocals on the song. The song
             was released as a single in 1970 from his debut album Stephen Stills.
             It reached number 14 on the Billboard Hot 100 ..."
            → ~120 tokens
```

| | Value |
|---|---|
| Shared tokens | ~10 |
| Precision | 10 / 13 = **0.77** |
| Recall | 10 / 120 = **0.08** |
| **F1** | **0.15** ← answer is correct, score says it is wrong |

**Why F1 fails:** Recall is computed over 120 gold tokens, but our answer is concise.
Recall is mathematically capped at ~0.11 for a 13-token answer against a 120-token gold,
no matter how correct the answer is. F1 punishes correct brevity.

---

## Why ROUGE-L Fails

### How ROUGE-L Works

ROUGE-L finds the **Longest Common Subsequence (LCS)** of words, then computes
precision and recall over that LCS length.

```
ROUGE-L Recall    = LCS_length / gold_length
ROUGE-L Precision = LCS_length / generated_length
ROUGE-L F1        = weighted harmonic mean
```

### Concrete Failure Example — FACTOID

```
Generated : "Stephen Stills sang Love the One You're With in 1970."
Gold      : (120-token paragraph about the song, release, chart position...)

LCS ≈ "Stephen Stills ... Love the One You're With ... 1970"  → ~9 tokens

ROUGE-L Recall    = 9 / 120 = 0.075
ROUGE-L Precision = 9 / 13  = 0.69
ROUGE-L F1        ≈ 0.13    ← answer is correct, score says it is terrible
```

### Concrete Failure Example — SYNTHESIS

```
Question  : How did the United States use nuclear weapons to end WWII in the Pacific?

Generated : "The US dropped atomic bombs on Hiroshima on August 6 and Nagasaki on
             August 9, 1945. The devastation prompted Japan's surrender, ending
             the war in the Pacific."
            → ~35 tokens

Gold      : "During the final stages of World War II in 1945, the United States
             conducted atomic bombings of the Japanese cities of Hiroshima and
             Nagasaki, the first on August 6, 1945, and the second on August 9,
             1945. Within the first two to four months of the bombings, the acute
             effects of the atomic bombings killed 90,000–166,000 people..."
            → ~140 tokens

ROUGE-L F1 ≈ 0.18   ← correct synthesis answer scored as near-failure
```

The generated answer captures the correct causal chain. ROUGE-L penalises different
sentence structure ("dropped" vs "conducted atomic bombings") and the length gap.

### Why ROUGE-L Is Especially Wrong for MULTI_HOP

Multi-hop answers use connective language ("This caused...", "As a result of...") that
does not appear verbatim in the gold passage. ROUGE-L sees low LCS and reports a poor
score even when the reasoning chain is correct.

---

## Why BERTScore Alone Also Fails

BERTScore computes contextual embeddings and measures cosine similarity — it handles
paraphrase and the long-gold problem. But it has its own critical weakness.

### Entity Substitution Blindness

```
Gold      : "The capital of France is Paris."
Generated : "The capital of France is Berlin."   ← factually wrong
```

`all-MiniLM-L6-v2` encodes these two sentences almost identically because:
- 5 of 6 content words are identical
- "Paris" and "Berlin" are both European capital cities — semantically close in
  embedding space

```
Cosine similarity ≈ 0.92 – 0.95   ← incorrectly scored as excellent
```

BERTScore measures structural and contextual similarity, not factual correctness of
named entities. This applies to all entity types: wrong city, wrong person, wrong date
where the surrounding sentence structure matches.

### Full Comparison of All Failure Modes

| Error Type | Example | Token F1 | ROUGE-L | BERTScore |
|---|---|---|---|---|
| Concise answer vs long gold | "Stephen Stills sang it" vs 120-token passage | ✗ Crushed | ✗ Crushed | ✓ Handles |
| Wrong entity (similar type) | "Berlin" instead of "Paris" | ✓ Catches | ✓ Catches | ✗ Misses |
| Wrong entity (different type) | "1970" instead of "Paris" | ✓ Catches | ✓ Catches | ✓ Catches |
| Correct paraphrase | "bombing" vs "nuclear attack" | ✗ Misses | ✗ Partial | ✓ Handles |
| Synthesis vs long gold | Different phrasing, correct facts | ✗ Very low | ✗ Very low | ✓ Handles |

**No automated metric handles all cases.** CLAPnq's structure (long gold, all question
types) exposes the failure mode of every word-overlap and embedding-similarity metric.

---

## Recommended Approach: LLM-as-Judge

An LLM judge reads the question, the gold passage, and the generated answer, then
evaluates factual correctness directly — the way a human evaluator would.

This sidesteps every metric failure:
- Long gold is not a problem — the LLM reads the whole passage
- Entity substitution is caught — "Berlin" vs "Paris" is obvious to an LLM
- Paraphrase is handled — the LLM understands meaning, not just words
- Synthesis quality is assessed — the LLM can verify the reasoning chain

### Judge Prompt

```
You are an expert evaluator for a question-answering system.

Your task is to assess whether the generated answer correctly answers the question
based on the reference passage provided.

QUESTION:
{question}

REFERENCE PASSAGE (gold answer from Wikipedia):
{gold_answer}

GENERATED ANSWER:
{generated_answer}

Evaluate the generated answer on two criteria:

1. FACTUAL ACCURACY: Are all facts stated in the generated answer supported by
   the reference passage? Penalise wrong entities (wrong person, wrong date,
   wrong place), wrong numbers, and unsupported claims.

2. COMPLETENESS: Does the generated answer address the core of the question?
   It does not need to reproduce every detail from the reference — a concise
   correct answer is better than a verbose incomplete one.

Score from 1 to 5 using this rubric:

  5 — All facts are correct and the question is fully answered
  4 — Facts are correct, minor detail missing but core is answered
  3 — Partially correct — main entity/fact right but some errors or gaps
  2 — Significant factual error or the question is only tangentially addressed
  1 — Wrong answer, hallucinated facts, or completely off-topic

Return your response as JSON only:
{"score": <1-5>, "reasoning": "<one sentence explaining the score>"}
```

### Example Judge Outputs

**Example 1 — Correct FACTOID**
```
Question  : Who sang "Love the One You're With"?
Gold      : (120-token passage about Stephen Stills and the 1970 single)
Generated : "Stephen Stills sang Love the One You're With in 1970."

Judge output:
{"score": 5, "reasoning": "Correct artist and year, directly answers the question."}
```

**Example 2 — Wrong Entity (BERTScore would miss this)**
```
Question  : What is the capital of France?
Gold      : (passage about France mentioning Paris as the capital)
Generated : "The capital of France is Berlin."

Judge output:
{"score": 1, "reasoning": "Berlin is incorrect; the reference clearly states Paris."}
```

**Example 3 — Correct SYNTHESIS (Token F1 would give ~0.15)**
```
Question  : How did the US use nuclear weapons to end WWII in the Pacific?
Gold      : (140-token passage about Hiroshima, Nagasaki, and Japan's surrender)
Generated : "The US dropped atomic bombs on Hiroshima and Nagasaki in August 1945,
             prompting Japan's surrender."

Judge output:
{"score": 5, "reasoning": "Correct facts, correct causation, directly answers the question."}
```

**Example 4 — Partially Correct MULTI_HOP**
```
Question  : How did France's defeat in the French and Indian War cause it to
            cede Louisiana to Spain?
Gold      : (passage covering the Treaty of Fontainebleau and the cession)
Generated : "France lost territory after the war and ceded Louisiana."

Judge output:
{"score": 3, "reasoning": "Correct outcome but missing the Treaty of Fontainebleau
                            as the mechanism and Spain as the recipient."}
```

---

## Evaluation Pipeline

### Full Pipeline per Record

```
1. Load     →  question + gold from clapnq_train_answerable.jsonl
               gold = record["output"][0]["answer"]

2. Retrieve →  top-5 chunks from FAISS (VectorStoreFaiss.search)

3. Classify →  question type (FACTOID / NUMERIC / TEMPORAL / LOCATION /
               SYNTHESIS / MULTI_HOP) via QuestionClassifier

4. Generate →  answer via RAGGenerator (type-aware prompt routing)

5. Judge    →  call Claude with judge prompt
               parse JSON response → score (1–5) + reasoning

6. Record   →  {id, question, question_type, generated, gold,
               judge_score, judge_reasoning, grounding_score, is_grounded}
```

### LLM Judge Configuration

| Parameter | Value | Reason |
|---|---|---|
| Model | `claude-haiku-4-5` | Fast and cheap for structured scoring |
| Temperature | 0.0 | Deterministic scores |
| Max tokens | 100 | Only JSON output needed |
| Response format | JSON `{"score": int, "reasoning": str}` | Parseable |

Use `claude-haiku-4-5` for the judge (not `claude-sonnet-4-6`) — Haiku is sufficient
for this structured scoring task and costs ~10× less per call.

### Cost Estimate

| Sample size | Haiku calls | Approx cost |
|---|---|---|
| 50 records (spot-check) | 50 | ~$0.01 |
| 200 records (type-level stats) | 200 | ~$0.04 |
| Full 1,954 records | 1,954 | ~$0.40 |

The full dataset is affordable — run it end-to-end.

---

## Score Interpretation

| Score | Label | Meaning |
|---|---|---|
| 5 | Excellent | All facts correct, question fully answered |
| 4 | Good | Core correct, minor gap |
| 3 | Partial | Main entity right, some errors or missing links |
| 2 | Poor | Significant factual error |
| 1 | Fail | Wrong, hallucinated, or off-topic |

Aggregate as **mean score per question type** and **% of scores ≥ 4** (pass rate).

### Expected Score Ranges by Question Type

| Type | Expected Mean | Expected Pass Rate (≥4) |
|---|---|---|
| FACTOID | 4.0 – 4.8 | 80 – 95% |
| NUMERIC | 4.0 – 4.8 | 80 – 95% |
| TEMPORAL | 4.0 – 4.8 | 80 – 95% |
| LOCATION | 4.0 – 4.8 | 80 – 95% |
| SYNTHESIS | 3.2 – 4.2 | 55 – 75% |
| MULTI_HOP | 2.8 – 3.8 | 45 – 65% |

Lower SYNTHESIS/MULTI_HOP pass rates are expected — these require evidence from
multiple passages and are genuinely harder to answer correctly.

---

## Known Limitations

1. **Judge model bias**: Claude judging Claude-generated answers may be lenient.
   Mitigation: use a different provider (e.g. GPT-4o) as judge for a spot-check
   comparison on 50 records.

2. **Score subjectivity**: The rubric boundary between score 3 and 4 can vary.
   The `reasoning` field in each judge output helps audit borderline cases.

3. **No reference-free cases**: The judge uses the gold passage as reference.
   If the gold passage itself is incomplete or misleading, the judge may penalise
   a correct answer that goes beyond it.

4. **Unanswerable records excluded**: This evaluation covers only
   `clapnq_train_answerable.jsonl`. Unanswerable records require a separate
   abstention-rate evaluation where the expected answer is "I don't know."

5. **Grounding score is complementary, not redundant**: The existing `GroundingVerifier`
   score (lexical + semantic against retrieved chunks) measures a different thing —
   whether the answer is supported by the *retrieved passages*, not whether it is
   *factually correct* against the gold. Both scores together give a complete picture.

---

*Phase: 2.5 — Generation Evaluation*  
*Primary metric: LLM-as-Judge (Claude Haiku, score 1–5)*  
*Complementary: GroundingVerifier score (already implemented in RAGGenerator)*  
*Last Updated: 2026-05-02*
