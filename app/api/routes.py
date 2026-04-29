"""
API Routes for RAG Chatbot
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any
from loguru import logger
import time

from app.models import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    MetricsResponse,
    IngestionRequest,
    IngestionResponse
)
from src.core.query_router import QueryRouter
from src.data_ingestion.stackoverflow_loader import (
    StackOverflowLoader,
    DocumentChunker,
    DEFAULT_TAGS
)
from app.config import settings

# Initialize router
api_router = APIRouter(prefix="/api/v1", tags=["RAG Chatbot"])

# Global instances (initialized once)
query_router = None
metrics = {
    "total_queries": 0,
    "total_response_time": 0.0,
    "intent_counts": {},
    "cache_hits": 0,
    "cache_misses": 0
}


def get_query_router() -> QueryRouter:
    """Get or create query router singleton"""
    global query_router
    if query_router is None:
        query_router = QueryRouter()
    return query_router


@api_router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Main chat endpoint
    
    Processes user query through RAG pipeline:
    1. Intent classification
    2. Multi-intent detection
    3. Retrieval
    4. Generation
    5. Response synthesis
    """
    try:
        router = get_query_router()
        
        # Process query
        response = await router.process_query(request)
        
        # Update metrics
        metrics["total_queries"] += 1
        metrics["total_response_time"] += response.processing_time
        
        intent_key = response.intent.primary_intent.value
        metrics["intent_counts"][intent_key] = metrics["intent_counts"].get(intent_key, 0) + 1
        
        return response
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint"""
    try:
        router = get_query_router()
        vector_store = router.get_vector_store()
        stats = vector_store.get_stats()
        
        return HealthResponse(
            status="healthy",
            version=settings.APP_VERSION,
            vector_db_status=f"OK ({stats.get('total_documents', 0)} docs)"
        )
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return HealthResponse(
            status="unhealthy",
            version=settings.APP_VERSION,
            vector_db_status=f"Error: {str(e)}"
        )


@api_router.get("/metrics", response_model=MetricsResponse)
async def get_metrics() -> MetricsResponse:
    """Get application metrics"""
    avg_response_time = (
        metrics["total_response_time"] / metrics["total_queries"]
        if metrics["total_queries"] > 0
        else 0.0
    )
    
    total_cache_ops = metrics["cache_hits"] + metrics["cache_misses"]
    cache_hit_rate = (
        metrics["cache_hits"] / total_cache_ops
        if total_cache_ops > 0
        else 0.0
    )
    
    return MetricsResponse(
        total_queries=metrics["total_queries"],
        avg_response_time=avg_response_time,
        intent_distribution=metrics["intent_counts"],
        cache_hit_rate=cache_hit_rate
    )


@api_router.post("/ingest", response_model=IngestionResponse)
async def ingest_data(
    request: IngestionRequest,
    background_tasks: BackgroundTasks
) -> IngestionResponse:
    """
    Ingest Stack Overflow data
    
    Can load from:
    - Stack Exchange API
    - Local JSON file
    """
    try:
        start_time = time.time()
        logger.info(f"Starting data ingestion from: {request.data_source}")
        
        loader = StackOverflowLoader()
        
        # Load documents
        if request.data_source == "api":
            # Load from Stack Exchange API
            tags = request.filters.get("tags", DEFAULT_TAGS[:5])  # Default to 5 tags
            max_docs = request.max_documents or settings.MAX_DOCUMENTS
            
            documents = await loader.load_from_api(
                tags=tags,
                max_questions=max_docs,
                min_score=settings.MIN_SCORE
            )
        else:
            # Load from JSON file
            from pathlib import Path
            filepath = Path(request.data_source)
            documents = loader.load_from_json(filepath)
        
        if not documents:
            raise HTTPException(status_code=400, detail="No documents loaded")
        
        # Chunk documents
        chunker = DocumentChunker(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        
        chunks, metadatas, ids = chunker.chunk_documents(documents)
        
        # Add to vector store
        router = get_query_router()
        vector_store = router.get_vector_store()
        
        chunks_added = await vector_store.add_documents(
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )
        
        processing_time = time.time() - start_time
        
        logger.info(f"Ingestion complete: {len(documents)} docs, {chunks_added} chunks")
        
        return IngestionResponse(
            status="success",
            documents_processed=len(documents),
            chunks_created=chunks_added,
            processing_time=processing_time,
            errors=[]
        )
        
    except Exception as e:
        logger.error(f"Ingestion error: {e}")
        return IngestionResponse(
            status="failed",
            documents_processed=0,
            chunks_created=0,
            processing_time=0.0,
            errors=[str(e)]
        )


@api_router.get("/stats")
async def get_stats() -> Dict[str, Any]:
    """Get vector store statistics"""
    try:
        router = get_query_router()
        vector_store = router.get_vector_store()
        return vector_store.get_stats()
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/clear")
async def clear_vector_store() -> Dict[str, str]:
    """Clear all documents from vector store (admin only)"""
    try:
        router = get_query_router()
        vector_store = router.get_vector_store()
        vector_store.clear()
        return {"status": "Vector store cleared successfully"}
    except Exception as e:
        logger.error(f"Clear error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
