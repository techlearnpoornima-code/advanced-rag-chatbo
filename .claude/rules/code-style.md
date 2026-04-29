# Code Style Guidelines - CLAPnq RAG Project

## Python Style (PEP 8 + Extensions)

### General Rules
- Maximum line length: 100 characters
- Use 4 spaces for indentation (no tabs)
- Two blank lines between top-level functions/classes
- One blank line between methods

### Naming Conventions
```python
# Classes: PascalCase
class SemanticChunker:
    pass

# Functions/methods: snake_case
def chunk_documents(passages: List[str]) -> List[Dict]:
    pass

# Constants: UPPER_SNAKE_CASE
MAX_CHUNK_SIZE = 512
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Private methods: _leading_underscore
def _extract_sentences(text: str) -> List[str]:
    pass
```

### Type Hints (Required)
```python
from typing import Optional, List, Dict, Tuple

# Always use type hints for function signatures
async def load_clapnq(
    filepath: str,
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Load CLAPnq dataset from JSONL file."""
    pass

# Use Optional for nullable values
def get_chunk(chunk_id: str) -> Optional[Dict]:
    pass

# Use List, Dict from typing
def chunk_passages(passages: List[str]) -> List[Dict]:
    pass
```

### Async/Await (Required for I/O)
```python
# Always use async for I/O operations
async def fetch_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for texts."""
    # Use aiohttp for async HTTP
    # Use async db operations
    pass

# DON'T block in async functions
async def process_passages(passages: List[str]):
    # WRONG - blocks event loop
    embeddings = embedding_model.encode(passages)  # ❌
    
    # RIGHT - async or use thread pool
    embeddings = await generate_embeddings(passages)  # ✅
```

### Docstrings (Google Style)
```python
def chunk_documents(
    documents: List[Dict[str, str]],
    max_tokens: int = 512
) -> Tuple[List[str], List[Dict]]:
    """
    Split documents into semantic chunks.
    
    Groups sentences until token limit reached, preserving
    semantic boundaries. Used for CLAPnq passages.
    
    Args:
        documents: List of documents with 'text' and 'sentences' keys
        max_tokens: Maximum tokens per chunk (default 512)
        
    Returns:
        Tuple of (chunks, metadata_list):
        - chunks: List of chunk texts
        - metadata_list: List of metadata dicts with chunk info
        
    Raises:
        ValueError: If document format is invalid
        
    Example:
        >>> docs = [{"text": "...", "sentences": ["...", "..."]}]
        >>> chunks, meta = chunk_documents(docs)
        >>> len(chunks) > 0
        True
    """
    pass
```

### Error Handling
```python
from loguru import logger

async def risky_operation(data: List[Dict]) -> Dict:
    try:
        result = await process_data(data)
        logger.info(f"Processed {len(data)} items")
        return result
    except ValueError as e:
        logger.error(f"Invalid data format: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise RuntimeError(f"Processing failed: {e}") from e
```

### Imports
```python
# Order: stdlib, third-party, local
# Group by category, alphabetize within groups

# Standard library
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

# Third-party
import numpy as np
from loguru import logger
from pydantic import BaseModel, Field

# Local
from app.config import settings
from app.models import ChunkMetadata
from src.chunking.semantic_chunker import SemanticChunker
```

### Class Structure
```python
class SemanticChunker:
    """Chunk CLAPnq passages using sentence boundaries."""
    
    # Class variables
    DEFAULT_MAX_TOKENS = 512
    DEFAULT_MIN_TOKENS = 50
    
    def __init__(self, max_tokens: int = DEFAULT_MAX_TOKENS):
        """Initialize chunker with config."""
        self.max_tokens = max_tokens
        self._tokenizer = None  # Private attribute
    
    # Public methods first
    async def chunk_passages(self, passages: List[Dict]) -> List[str]:
        """Chunk passages into semantic units."""
        return [self._process_one(p) for p in passages]
    
    def get_statistics(self) -> Dict:
        """Return chunking statistics."""
        return {"chunks": len(self.chunks)}
    
    # Private methods last
    async def _process_one(self, passage: Dict) -> str:
        """Process single passage."""
        pass
    
    def _tokenize(self, text: str) -> int:
        """Count tokens in text."""
        pass
```

### Configuration Access
```python
from app.config import settings

# GOOD - use settings from config
max_tokens = settings.CHUNK_SIZE_TOKENS
model = settings.EMBEDDING_MODEL

# BAD - hardcoded values
max_tokens = 512  # ❌ Don't do this
```

### Logging
```python
from loguru import logger

# Use appropriate log levels
logger.debug("Tokenizer initialized with model: {}", model_name)
logger.info(f"Chunking {len(passages)} passages")
logger.warning(f"Chunk {i} exceeds target size: {actual} > {target}")
logger.error(f"Failed to load embeddings: {e}")

# Include context
logger.info(
    "Chunked passage {} into {} chunks (avg {} tokens)",
    passage_id, chunk_count, avg_tokens
)
```

### File Organization
```python
"""CLAPnq data loader module."""

import json
from pathlib import Path
from typing import List, Dict, Any

from loguru import logger

from app.config import settings

# Constants
DEFAULT_BATCH_SIZE = 100

# Type definitions
CLAPnqRecord = Dict[str, Any]

# Classes
class CLAPnqLoader:
    """Load and parse CLAPnq dataset."""
    pass

# Functions
def load_jsonl(filepath: str) -> List[CLAPnqRecord]:
    """Load JSONL file."""
    pass

if __name__ == "__main__":
    main()
```

## Don'ts

❌ Don't use `print()` - use `logger` instead
❌ Don't use bare `except:` - specify exception types
❌ Don't hardcode values - use `settings`
❌ Don't use blocking I/O in async functions
❌ Don't skip type hints
❌ Don't skip docstrings for public functions
❌ Don't use mutable default arguments

## DOs

✅ Use type hints everywhere
✅ Use async/await for I/O
✅ Log with context
✅ Handle errors explicitly
✅ Write docstrings
✅ Keep functions focused
✅ Use meaningful names
✅ Add comments for WHY, not WHAT
