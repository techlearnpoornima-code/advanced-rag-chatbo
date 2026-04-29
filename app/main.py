"""
FastAPI Application - Advanced RAG Chatbot
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from loguru import logger
import sys

from app.config import settings
from app.api.routes import api_router

# Configure logging
logger.remove()
logger.add(
    sys.stdout,
    level=settings.LOG_LEVEL,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
)

logger.add(
    settings.LOG_FILE,
    rotation="100 MB",
    retention="10 days",
    level=settings.LOG_LEVEL,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}"
)

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="""
    Advanced RAG Chatbot with Semantic Routing and Multi-Intent Handling
    
    ## Features
    - **Semantic Routing**: Intent-based query classification
    - **Multi-Intent Handling**: Automatic query decomposition
    - **Hybrid Search**: Dense + sparse retrieval
    - **Stack Overflow Data**: Code examples and technical Q&A
    - **Production Ready**: Monitoring, caching, error handling
    
    ## Intent Types
    - Factual: Definitions and facts
    - How-To: Step-by-step tutorials
    - Comparison: Side-by-side analysis
    - Reasoning: Conceptual explanations
    - Code: Code generation
    - Debug: Error resolution
    """,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=settings.ALLOWED_METHODS,
    allow_headers=settings.ALLOWED_HEADERS,
)

# Include API routes
app.include_router(api_router)


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Log level: {settings.LOG_LEVEL}")
    logger.info(f"Vector DB: {settings.VECTOR_DB_PATH}")
    logger.info(f"Embedding model: {settings.EMBEDDING_MODEL}")
    logger.info("Application started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down application")


@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to API docs"""
    return RedirectResponse(url="/docs")


@app.get("/ping")
async def ping():
    """Simple ping endpoint"""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        log_level=settings.LOG_LEVEL.lower()
    )
