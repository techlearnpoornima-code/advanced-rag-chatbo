"""Question type classifier for type-aware RAG generation."""

import re
from enum import Enum


class QuestionType(str, Enum):
    FACTOID = "FACTOID"       # Who/What/Which — single-entity lookup
    NUMERIC = "NUMERIC"       # How many/much — quantity answers
    TEMPORAL = "TEMPORAL"     # When/What year — date/time answers
    LOCATION = "LOCATION"     # Where/Which country — place answers
    SYNTHESIS = "SYNTHESIS"   # How/Why/Explain — multi-passage reasoning
    MULTI_HOP = "MULTI_HOP"  # Chained inference across 2+ fact-sets


# ── MULTI_HOP patterns (checked first — more specific than SYNTHESIS) ──────

_MULTI_HOP_PATTERNS = [
    r'\bthat (led|caused|resulted|triggered|prompted)\b',
    r'\bwhich (led|caused|resulted|triggered|prompted)\b',
    r'\bwho was responsible for .+ that\b',
    r'\bhow did .+\'s .+ (affect|influence|impact|lead to|result in)\b',
    r'\bwhat (connection|relationship|link) .+ (between|among)\b',
    r'\bwhat (event|action|decision) .+ (caused|triggered|led)\b',
    r'\bwhy did .+ (after|because|following|as a result of)\b',
    r'\bhow did the .+ of .+ (affect|change|influence)\b',
]

# ── SYNTHESIS patterns (multi-passage explanation/analysis) ────────────────

_SYNTHESIS_PREFIXES = (
    'how did', 'how was', 'how were', 'how does', 'how do',
    'why did', 'why was', 'why were', 'why is', 'why are',
    'explain', 'describe', 'what role did', 'what role does',
    'what was the impact', 'what is the impact',
    'what caused', 'what led to', 'what contributed to',
    'what were the causes', 'what were the effects',
    'what were the consequences', 'what factors',
    'what was the significance', 'what is the significance',
    'what was the relationship', 'what is the relationship',
    'in what ways', 'to what extent',
    'how important', 'how significant',
)

# ── NUMERIC patterns ───────────────────────────────────────────────────────

_NUMERIC_PREFIXES = (
    'how many', 'how much', 'how long is', 'how long was',
    'how tall', 'how wide', 'how far', 'how large', 'how big',
    'how old', 'what number', 'what percentage', 'what amount',
    'how often', 'how frequently', 'total number', 'number of',
)

# ── TEMPORAL patterns ──────────────────────────────────────────────────────

_TEMPORAL_PREFIXES = (
    'when did', 'when was', 'when were', 'when is',
    'what year', 'what date', 'what time', 'what century',
    'what decade', 'what month', 'in what year', 'in what century',
    'how long did', 'how long ago',
)

# ── LOCATION patterns ──────────────────────────────────────────────────────

_LOCATION_PREFIXES = (
    'where', 'which country', 'which city', 'which state',
    'which continent', 'which region', 'which island',
    'what country', 'what city', 'what state', 'what continent',
    'what region', 'location of', 'located in',
)


def classify(question: str) -> QuestionType:
    """
    Classify a question into one of six types.

    Classification order (most-specific first):
      MULTI_HOP -> SYNTHESIS -> NUMERIC -> TEMPORAL -> LOCATION -> FACTOID

    Args:
        question: Raw question string.

    Returns:
        QuestionType enum value.
    """
    q = question.lower().strip().rstrip('?')

    # Multi-hop: chained inference across fact-sets (most specific)
    for pattern in _MULTI_HOP_PATTERNS:
        if re.search(pattern, q):
            return QuestionType.MULTI_HOP

    # Synthesis: explanation / analysis / causation
    for prefix in _SYNTHESIS_PREFIXES:
        if q.startswith(prefix):
            return QuestionType.SYNTHESIS

    # Numeric: quantity / measurement
    for prefix in _NUMERIC_PREFIXES:
        if prefix in q:
            return QuestionType.NUMERIC

    # Temporal: date / time
    for prefix in _TEMPORAL_PREFIXES:
        if prefix in q:
            return QuestionType.TEMPORAL

    # Location: place
    for prefix in _LOCATION_PREFIXES:
        if q.startswith(prefix) or prefix in q:
            return QuestionType.LOCATION

    # Default: single-entity factoid
    return QuestionType.FACTOID


def needs_synthesis(qt: QuestionType) -> bool:
    """Return True for question types that require multi-passage synthesis."""
    return qt in (QuestionType.SYNTHESIS, QuestionType.MULTI_HOP)
