"""
Pydantic models for API request/response validation
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class QueryIntent(str, Enum):
    """Query intent types"""
    FACTUAL = "factual"
    HOWTO = "howto"
    COMPARISON = "comparison"
    REASONING = "reasoning"
    CODE = "code"
    DEBUG = "debug"
    CONVERSATIONAL = "conversational"


class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    query: str = Field(..., min_length=1, max_length=2000, description="User query")
    session_id: Optional[str] = Field(None, description="Session identifier for conversation history")
    stream: bool = Field(False, description="Enable streaming response")
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters (e.g., tags, date range)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "Compare asyncio vs threading in Python and show example code",
                "session_id": "user-123",
                "stream": False,
                "filters": {"tags": ["python", "concurrency"]}
            }
        }


class RetrievedChunk(BaseModel):
    """Model for retrieved document chunk"""
    content: str = Field(..., description="Chunk content")
    score: float = Field(..., description="Similarity score")
    metadata: Dict[str, Any] = Field(..., description="Document metadata")
    source_url: Optional[str] = Field(None, description="Source URL")


class IntentAnalysis(BaseModel):
    """Intent classification result"""
    primary_intent: QueryIntent = Field(..., description="Primary detected intent")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")
    sub_intents: Optional[List[QueryIntent]] = Field(None, description="Additional detected intents")
    is_multi_intent: bool = Field(False, description="Whether query has multiple intents")


class SubQuery(BaseModel):
    """Decomposed sub-query for multi-intent handling"""
    query: str = Field(..., description="Sub-query text")
    intent: QueryIntent = Field(..., description="Intent type")
    priority: int = Field(..., ge=1, description="Processing priority")


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    answer: str = Field(..., description="Generated answer")
    intent: IntentAnalysis = Field(..., description="Detected intent information")
    sources: List[RetrievedChunk] = Field(..., description="Retrieved source chunks")
    sub_queries: Optional[List[SubQuery]] = Field(None, description="Decomposed queries if multi-intent")
    processing_time: float = Field(..., description="Processing time in seconds")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "Here's a comparison of asyncio vs threading...",
                "intent": {
                    "primary_intent": "comparison",
                    "confidence": 0.95,
                    "sub_intents": ["code"],
                    "is_multi_intent": True
                },
                "sources": [
                    {
                        "content": "Asyncio is a Python library...",
                        "score": 0.89,
                        "metadata": {"tags": ["python", "asyncio"], "votes": 245},
                        "source_url": "https://stackoverflow.com/questions/123"
                    }
                ],
                "sub_queries": [
                    {"query": "Compare asyncio vs threading", "intent": "comparison", "priority": 1},
                    {"query": "Show asyncio example code", "intent": "code", "priority": 2}
                ],
                "processing_time": 1.23,
                "timestamp": "2024-01-15T10:30:00"
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="Application version")
    vector_db_status: str = Field(..., description="Vector database status")
    timestamp: datetime = Field(default_factory=datetime.now)


class MetricsResponse(BaseModel):
    """Metrics response"""
    total_queries: int = Field(..., description="Total queries processed")
    avg_response_time: float = Field(..., description="Average response time")
    intent_distribution: Dict[str, int] = Field(..., description="Query distribution by intent")
    cache_hit_rate: float = Field(..., description="Cache hit rate")


class IngestionRequest(BaseModel):
    """Request for data ingestion"""
    data_source: str = Field(..., description="Data source path or URL")
    max_documents: Optional[int] = Field(None, description="Maximum documents to process")
    filters: Optional[Dict[str, Any]] = Field(None, description="Ingestion filters")


class IngestionResponse(BaseModel):
    """Response for data ingestion"""
    status: str = Field(..., description="Ingestion status")
    documents_processed: int = Field(..., description="Number of documents processed")
    chunks_created: int = Field(..., description="Number of chunks created")
    processing_time: float = Field(..., description="Processing time in seconds")
    errors: List[str] = Field(default_factory=list, description="Any errors encountered")
