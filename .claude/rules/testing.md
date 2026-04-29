# Testing Guidelines - CLAPnq RAG Project

## Testing Philosophy

- Write tests for all new features **before implementation** (TDD where possible)
- Aim for 80%+ code coverage
- Test behavior, not implementation
- Keep tests fast and isolated
- Use fixtures for test data (from CLAPnq dataset)

## Test Categories

### 1. Unit Tests - Chunking

Test chunking logic in isolation:

```python
import pytest
from src.chunking.semantic_chunker import SemanticChunker

class TestSemanticChunker:
    """Tests for SemanticChunker."""
    
    @pytest.fixture
    def chunker(self):
        return SemanticChunker(max_tokens=512)
    
    @pytest.fixture
    def sample_passage(self):
        return {
            "text": "Sentence 1. Sentence 2. Sentence 3.",
            "sentences": ["Sentence 1.", "Sentence 2.", "Sentence 3."],
            "title": "Test Passage"
        }
    
    def test_chunking_preserves_sentence_boundaries(self, chunker, sample_passage):
        """Chunks should not split sentences."""
        chunks, metadata = chunker.chunk_passages([sample_passage])
        
        # Verify no sentence is split across chunks
        for chunk in chunks:
            # Count sentence boundaries
            assert not self._is_sentence_split(chunk)
    
    def test_chunking_respects_max_tokens(self, chunker, sample_passage):
        """Each chunk must not exceed max_tokens."""
        chunks, metadata = chunker.chunk_passages([sample_passage])
        
        for chunk, meta in zip(chunks, metadata):
            assert meta['token_count'] <= chunker.max_tokens
    
    def test_answer_spans_stay_together(self, chunker):
        """If answer spans sentences 1-2, they stay in same chunk."""
        passage = {
            "text": "S1. S2. S3.",
            "sentences": ["S1.", "S2.", "S3."],
            "answer_sentences": [0, 1]  # Spans sentences 0 and 1
        }
        chunks, metadata = chunker.chunk_passages([passage])
        
        # Find chunk containing answer
        answer_chunks = [m for m in metadata if m['contains_answer']]
        assert len(answer_chunks) == 1  # Answer in one chunk
```

### 2. Unit Tests - Data Loading

Test CLAPnq data loading:

```python
import pytest
from src.data_loading.clapnq_loader import CLAPnqLoader

class TestCLAPnqLoader:
    """Tests for CLAPnqLoader."""
    
    @pytest.fixture
    def loader(self):
        return CLAPnqLoader()
    
    def test_load_answerable_data(self, loader):
        """Load answerable records correctly."""
        records = loader.load_answerable("data/clapnq_train_answerable.jsonl")
        
        assert len(records) > 0
        assert all('input' in r for r in records)
        assert all('passages' in r for r in records)
        assert all('output' in r for r in records)
    
    def test_load_unanswerable_data(self, loader):
        """Load unanswerable records correctly."""
        records = loader.load_unanswerable("data/clapnq_train_unanswerable.jsonl")
        
        assert len(records) > 0
        assert all(len(r['output'][0]['answer']) == 0 for r in records)
    
    def test_data_structure_validation(self, loader, sample_record):
        """Validate loaded record structure."""
        assert 'id' in sample_record
        assert 'input' in sample_record  # Question
        assert 'passages' in sample_record
        assert 'output' in sample_record
        assert len(sample_record['passages']) > 0
```

### 3. Integration Tests - Pipeline

Test end-to-end pipeline:

```python
import pytest
from src.data_loading.clapnq_loader import CLAPnqLoader
from src.chunking.semantic_chunker import SemanticChunker

class TestLoadAndChunkPipeline:
    """Integration test for load + chunk."""
    
    @pytest.mark.asyncio
    async def test_load_and_chunk_workflow(self):
        """Test loading data and chunking."""
        loader = CLAPnqLoader()
        chunker = SemanticChunker()
        
        # Load
        records = loader.load_answerable("data/clapnq_train_answerable.jsonl", limit=10)
        assert len(records) > 0
        
        # Chunk
        passages = [r['passages'][0] for r in records]
        chunks, metadata = chunker.chunk_passages(passages)
        
        # Verify
        assert len(chunks) > len(passages)  # More chunks than passages
        assert all('chunk_id' in m for m in metadata)
        assert all('passage_title' in m for m in metadata)
```

### 4. Evaluation Tests - Chunking Quality

```python
import pytest
from src.evaluation.metrics import ChunkingMetrics

class TestChunkingMetrics:
    """Test chunking quality metrics."""
    
    @pytest.fixture
    def metrics(self):
        return ChunkingMetrics()
    
    def test_chunk_size_distribution(self, metrics, sample_chunks):
        """Verify chunk size distribution."""
        stats = metrics.analyze_chunks(sample_chunks)
        
        # Check sizes
        assert stats['min_tokens'] >= 50
        assert stats['max_tokens'] <= 512
        assert stats['mean_tokens'] > 100
        assert stats['mean_tokens'] < 512
    
    def test_sentence_boundary_preservation(self, metrics, chunks):
        """Verify no sentences are split."""
        violations = metrics.count_split_sentences(chunks)
        assert violations == 0
    
    def test_answer_containment(self, metrics, chunks, metadata):
        """Verify answer-containing chunks are marked."""
        answer_chunks = [m for m in metadata if m['contains_answer']]
        assert len(answer_chunks) > 0
```

## Test Fixtures

Create reusable test data:

```python
# tests/conftest.py

import pytest

@pytest.fixture
def sample_passage():
    """Sample CLAPnq passage."""
    return {
        "title": "Sample Article",
        "text": "First sentence. Second sentence. Third sentence.",
        "sentences": ["First sentence.", "Second sentence.", "Third sentence."]
    }

@pytest.fixture
def sample_answerable_record():
    """Sample answerable CLAPnq record."""
    return {
        "id": "test_123",
        "input": "What is the answer?",
        "passages": [{
            "title": "Test",
            "text": "The answer is 42. This is important.",
            "sentences": ["The answer is 42.", "This is important."]
        }],
        "output": [{
            "answer": "42",
            "selected_sentences": [0],
            "meta": {}
        }]
    }

@pytest.fixture
def sample_unanswerable_record():
    """Sample unanswerable CLAPnq record."""
    return {
        "id": "test_456",
        "input": "Unanswerable question?",
        "passages": [{
            "title": "Irrelevant",
            "text": "This is not related.",
            "sentences": ["This is not related."]
        }],
        "output": [{
            "answer": "",
            "selected_sentences": [],
            "meta": {}
        }]
    }

@pytest.fixture
def small_dataset(sample_answerable_record, sample_unanswerable_record):
    """Small dataset for testing."""
    return [sample_answerable_record, sample_unanswerable_record]
```

## Running Tests

```bash
# All tests
pytest tests/

# Specific category
pytest tests/test_chunking.py
pytest tests/test_data_loading.py
pytest tests/test_evaluation.py

# With coverage
pytest --cov=src --cov=app tests/

# Specific test
pytest tests/test_chunking.py::TestSemanticChunker::test_chunking_preserves_sentence_boundaries

# With output
pytest -v tests/

# Stop on first failure
pytest -x tests/
```

## Coverage Goals

```
src/chunking/:        95%+ (critical path)
src/data_loading/:    90%+ (validation)
src/evaluation/:      80%+ (metrics)
src/retrieval/:       85%+ (critical)
Overall:             80%+
```

## Test Organization

```
tests/
├── conftest.py                    # Shared fixtures
├── test_data_loading.py           # CLAPnq loader tests
├── test_chunking.py               # Chunking logic tests
├── test_retrieval.py              # Vector store tests
├── test_evaluation.py             # Metrics tests
└── integration/
    └── test_e2e_pipeline.py       # End-to-end tests
```

## Best Practices

✅ Use descriptive test names: `test_chunking_preserves_sentence_boundaries`  
✅ Test one behavior per test  
✅ Use fixtures for setup  
✅ Mock external dependencies  
✅ Test edge cases (empty, large, malformed)  
✅ Keep tests fast (<1s per test)  
✅ Use parametrize for multiple cases  

❌ Don't test implementation details  
❌ Don't test third-party libraries  
❌ Don't hardcode test data  
❌ Don't skip error cases  
❌ Don't make tests depend on each other  
