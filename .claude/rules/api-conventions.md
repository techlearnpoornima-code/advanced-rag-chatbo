# API Design Conventions - CLAPnq RAG

## RESTful Principles

### Endpoint Naming

```
/api/v1/health              # Health check
/api/v1/retrieve            # Retrieve passages
/api/v1/generate            # Generate answer
/api/v1/query               # Full QA pipeline
/api/v1/chunks/{chunk_id}   # Get specific chunk
/api/v1/stats               # System statistics
/api/v1/evaluate            # Evaluation metrics
```

- Use plural nouns for collections
- Use kebab-case for multi-word resources
- Version APIs: `/api/v1/`
- No trailing slashes

---

## Core Endpoints

### 1. Health Check

```python
@router.get("/api/v1/health")
async def health_check() -> HealthResponse:
    """
    Check system health and component status.
    
    Returns:
        HealthResponse with status of all components
    """
    pass

# Response
{
  "status": "healthy",
  "version": "1.0.0",
  "components": {
    "vector_store": "ok",
    "embeddings": "ok",
    "llm": "ok"
  },
  "timestamp": "2026-04-29T10:30:00Z"
}
```

### 2. Retrieve Passages

```python
@router.post("/api/v1/retrieve")
async def retrieve_passages(request: RetrievalRequest) -> RetrievalResponse:
    """
    Retrieve relevant passages for a query.
    
    Uses hybrid search (dense + sparse + cross-encoder).
    """
    pass

# Request
{
  "query": "What is the capital of France?",
  "top_k": 10,
  "filters": {
    "source": "wikipedia",
    "min_confidence": 0.5
  }
}

# Response
{
  "query": "What is the capital of France?",
  "chunks": [
    {
      "chunk_id": "passage_0_chunk_1",
      "content": "France is a country... The capital is Paris...",
      "score": 0.95,
      "metadata": {
        "passage_title": "France",
        "sentence_indices": [1, 2],
        "contains_answer": true
      },
      "source_url": "https://..."
    }
  ],
  "processing_time_ms": 245,
  "timestamp": "2026-04-29T10:30:00Z"
}
```

### 3. Generate Answer

```python
@router.post("/api/v1/generate")
async def generate_answer(request: GenerationRequest) -> GenerationResponse:
    """
    Generate answer using retrieved passages.
    
    Ensures answer is grounded in retrieved text.
    """
    pass

# Request
{
  "query": "What is the capital of France?",
  "passages": [
    {
      "chunk_id": "chunk_1",
      "content": "France is...",
      "metadata": {}
    }
  ],
  "style": "concise"  # concise, detailed, long-form
}

# Response
{
  "answer": "Paris is the capital of France.",
  "grounded": true,
  "sources": ["chunk_1"],
  "confidence": 0.98,
  "generation_time_ms": 1200
}
```

### 4. Full QA Pipeline

```python
@router.post("/api/v1/query")
async def full_qa(request: QueryRequest) -> QueryResponse:
    """
    End-to-end QA pipeline.
    
    1. Retrieve passages
    2. Generate answer
    3. Evaluate grounding
    """
    pass

# Request
{
  "query": "What is the capital of France?",
  "top_k": 5,
  "style": "concise"
}

# Response
{
  "query": "What is the capital of France?",
  "answer": "Paris is the capital of France.",
  "retrieved_chunks": [
    {
      "chunk_id": "...",
      "content": "...",
      "score": 0.95
    }
  ],
  "confidence": 0.98,
  "grounded": true,
  "processing_time_ms": 1500
}
```

### 5. Statistics

```python
@router.get("/api/v1/stats")
async def get_stats() -> StatsResponse:
    """
    Get vector store and system statistics.
    """
    pass

# Response
{
  "vector_store": {
    "total_chunks": 10567,
    "total_passages": 3745,
    "embedding_model": "all-MiniLM-L6-v2",
    "vector_dimension": 384
  },
  "dataset": {
    "answerable_records": 1954,
    "unanswerable_records": 1791,
    "avg_chunk_size_tokens": 512
  },
  "performance": {
    "avg_retrieval_time_ms": 245,
    "avg_generation_time_ms": 1200
  }
}
```

---

## Request/Response Models

### Pydantic Models

```python
from pydantic import BaseModel, Field
from typing import Optional, List

class RetrievalRequest(BaseModel):
    """Request to retrieve passages."""
    
    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="User question"
    )
    top_k: int = Field(
        10,
        ge=1,
        le=100,
        description="Number of passages to retrieve"
    )
    filters: Optional[dict] = Field(
        None,
        description="Metadata filters (source, confidence, etc)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is the capital of France?",
                "top_k": 5,
                "filters": {"source": "wikipedia"}
            }
        }

class RetrievalResponse(BaseModel):
    """Response with retrieved passages."""
    
    query: str
    chunks: List[dict]
    processing_time_ms: float
    timestamp: str
```

---

## Error Handling

```python
from fastapi import HTTPException, status

# 400 - Bad Request
if not query.strip():
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Query cannot be empty"
    )

# 404 - Not Found
if chunk_id not in vector_store:
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Chunk {chunk_id} not found"
    )

# 500 - Internal Server Error
try:
    result = await generate_answer(query, passages)
except Exception as e:
    logger.error(f"Generation failed: {e}")
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Failed to generate answer"
    )

# 503 - Service Unavailable
if not vector_store.is_healthy():
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Vector store not available"
    )
```

---

## Response Format

Consistent response structure:

```python
# Success
{
  "query": "...",
  "answer": "...",
  "metadata": {...},
  "timestamp": "2026-04-29T10:30:00Z"
}

# Error (automatic from HTTPException)
{
  "detail": "Error message here"
}

# List responses
{
  "items": [...],
  "total": 100,
  "page": 1,
  "page_size": 20
}
```

---

## Testing Endpoints

```python
from fastapi.testclient import TestClient
from app.main import app

def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/api/v1/health")
    
    assert response.status_code == 200
    assert "status" in response.json()

def test_retrieve_endpoint():
    client = TestClient(app)
    response = client.post(
        "/api/v1/retrieve",
        json={
            "query": "What is the capital of France?",
            "top_k": 5
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "chunks" in data
    assert len(data["chunks"]) <= 5

def test_invalid_query():
    client = TestClient(app)
    response = client.post(
        "/api/v1/retrieve",
        json={"query": ""}  # Empty query
    )
    
    assert response.status_code == 400
```

---

## Best Practices

✅ Use Pydantic models for validation  
✅ Include examples in model schemas  
✅ Use proper HTTP status codes  
✅ Log all requests with IDs  
✅ Version your API  
✅ Document with OpenAPI/Swagger  
✅ Handle errors gracefully  
✅ Include processing time metadata  
✅ Add timestamp to responses  

❌ Don't expose internal errors  
❌ Don't use GET for state-changing ops  
❌ Don't hardcode URLs  
❌ Don't skip input validation  
