"""
Vector Store Management with ChromaDB
Handles document storage, embedding, and retrieval
"""
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
from loguru import logger
from app.models import RetrievedChunk
from app.config import settings
import numpy as np


class VectorStore:
    """Vector database for document storage and retrieval"""
    
    def __init__(self):
        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.client = None
        self.collection = None
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize ChromaDB client and collection"""
        try:
            # Create persistent client
            self.client = chromadb.PersistentClient(
                path=str(settings.VECTOR_DB_PATH),
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name="stackoverflow_qa",
                metadata={"description": "Stack Overflow Q&A with code examples"}
            )
            
            logger.info(f"Vector store initialized with {self.collection.count()} documents")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise
    
    async def add_documents(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        ids: List[str]
    ) -> int:
        """
        Add documents to vector store
        
        Args:
            documents: List of document texts
            metadatas: List of metadata dicts
            ids: List of unique document IDs
            
        Returns:
            Number of documents added
        """
        try:
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(documents)} documents...")
            embeddings = self.embedding_model.encode(
                documents,
                show_progress_bar=True,
                batch_size=32
            ).tolist()
            
            # Add to collection
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Added {len(documents)} documents to vector store")
            return len(documents)
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return 0
    
    async def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedChunk]:
        """
        Search for relevant documents
        
        Args:
            query: Search query
            top_k: Number of results to return
            filters: Metadata filters (e.g., {"has_code": True})
            
        Returns:
            List of RetrievedChunk objects
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Build where clause from filters
            where = self._build_where_clause(filters) if filters else None
            
            # Search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where,
                include=["documents", "metadatas", "distances"]
            )
            
            # Convert to RetrievedChunk objects
            chunks = []
            if results['documents'] and len(results['documents'][0]) > 0:
                for i in range(len(results['documents'][0])):
                    metadata = results['metadatas'][0][i]

                    # Client-side tag filtering (tags stored as comma-separated string)
                    if filters and "tags" in filters:
                        doc_tags = [t.strip() for t in metadata.get('tags', '').split(',') if t.strip()]
                        requested_tags = filters["tags"]
                        # Check if any requested tag is in document tags
                        if not any(tag in doc_tags for tag in requested_tags):
                            continue

                    # Convert distance to similarity score (0-1)
                    distance = results['distances'][0][i]
                    score = 1 / (1 + distance)  # Normalize to 0-1

                    chunks.append(RetrievedChunk(
                        content=results['documents'][0][i],
                        score=float(score),
                        metadata=metadata,
                        source_url=metadata.get('url')
                    ))

            logger.info(f"Retrieved {len(chunks)} chunks for query: {query[:50]}...")
            return chunks
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
    
    def _build_where_clause(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Build ChromaDB where clause from filters"""
        conditions = []

        # Handle specific filter types
        if "has_code" in filters and filters["has_code"]:
            conditions.append({"has_code": True})

        if "min_score" in filters:
            conditions.append({"score": {"$gte": filters["min_score"]}})

        # Note: tags filtering is done client-side since tags are stored as comma-separated string
        # (ChromaDB operators don't work well with substring matching)

        if "has_accepted_answer" in filters:
            conditions.append({"has_accepted_answer": filters["has_accepted_answer"]})

        if "source_type" in filters:
            conditions.append({"source_type": filters["source_type"]})

        # Return properly formatted where clause
        if len(conditions) == 0:
            return None
        elif len(conditions) == 1:
            return conditions[0]
        else:
            return {"$and": conditions}
    
    async def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        alpha: float = 0.7
    ) -> List[RetrievedChunk]:
        """
        Hybrid search combining dense and sparse retrieval
        
        Args:
            query: Search query
            top_k: Number of results
            alpha: Weight for dense retrieval (1-alpha for sparse)
            
        Returns:
            Reranked chunks
        """
        # Dense retrieval (semantic)
        dense_results = await self.search(query, top_k=top_k * 2)
        
        # Sparse retrieval (keyword - simplified BM25)
        sparse_results = self._keyword_search(query, top_k=top_k * 2)
        
        # Combine with weighted fusion
        combined = self._reciprocal_rank_fusion(
            dense_results,
            sparse_results,
            alpha=alpha
        )
        
        return combined[:top_k]
    
    def _keyword_search(self, query: str, top_k: int) -> List[RetrievedChunk]:
        """Simple keyword-based search (BM25-like)"""
        # Simplified implementation - in production use a proper BM25
        query_terms = set(query.lower().split())
        
        # Get all documents (limited for performance)
        all_docs = self.collection.get(limit=1000)
        
        scored_docs = []
        for i, doc in enumerate(all_docs['documents']):
            doc_terms = set(doc.lower().split())
            overlap = len(query_terms & doc_terms)
            score = overlap / len(query_terms) if query_terms else 0
            
            if score > 0:
                scored_docs.append(RetrievedChunk(
                    content=doc,
                    score=score,
                    metadata=all_docs['metadatas'][i],
                    source_url=all_docs['metadatas'][i].get('url')
                ))
        
        scored_docs.sort(key=lambda x: x.score, reverse=True)
        return scored_docs[:top_k]
    
    def _reciprocal_rank_fusion(
        self,
        results_a: List[RetrievedChunk],
        results_b: List[RetrievedChunk],
        alpha: float = 0.7,
        k: int = 60
    ) -> List[RetrievedChunk]:
        """
        Reciprocal Rank Fusion (RRF) for combining results
        
        Args:
            results_a: Dense retrieval results
            results_b: Sparse retrieval results
            alpha: Weight for dense results
            k: RRF constant
        """
        # Create content -> chunk mapping
        content_to_chunk = {}
        scores = {}
        
        # Score dense results
        for rank, chunk in enumerate(results_a):
            content = chunk.content
            rrf_score = alpha / (k + rank + 1)
            scores[content] = scores.get(content, 0) + rrf_score
            content_to_chunk[content] = chunk
        
        # Score sparse results
        for rank, chunk in enumerate(results_b):
            content = chunk.content
            rrf_score = (1 - alpha) / (k + rank + 1)
            scores[content] = scores.get(content, 0) + rrf_score
            if content not in content_to_chunk:
                content_to_chunk[content] = chunk
        
        # Sort by combined score
        sorted_contents = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        # Reconstruct chunks with updated scores
        fused_chunks = []
        for content in sorted_contents:
            chunk = content_to_chunk[content]
            chunk.score = scores[content]
            fused_chunks.append(chunk)
        
        return fused_chunks
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        try:
            count = self.collection.count()
            return {
                "total_documents": count,
                "embedding_model": settings.EMBEDDING_MODEL,
                "collection_name": self.collection.name
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}
    
    def clear(self):
        """Clear all documents from collection"""
        try:
            self.client.delete_collection(name="stackoverflow_qa")
            self.collection = self.client.create_collection(
                name="stackoverflow_qa",
                metadata={"description": "Stack Overflow Q&A with code examples"}
            )
            logger.info("Vector store cleared")
        except Exception as e:
            logger.error(f"Error clearing vector store: {e}")
