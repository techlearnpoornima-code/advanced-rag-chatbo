"""
Vector Store with FAISS + SQLite
Handles document storage with dense embeddings (FAISS) + metadata (SQLite)
"""

import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from loguru import logger

from app.config import settings


class VectorStoreFaiss:
    """Vector database using FAISS for embeddings + SQLite for metadata"""

    def __init__(
        self,
        db_path: str = "./data/vectordb/chunks.db",
        index_path: str = "./data/vectordb/chunks.faiss",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize vector store with FAISS and SQLite.

        Args:
            db_path: Path to SQLite database file
            index_path: Path to FAISS index file
            embedding_model: SentenceTransformer model name
        """
        self.db_path = Path(db_path)
        self.index_path = Path(index_path)
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedding_model.get_embedding_dimension()

        # Create directories
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize FAISS index and SQLite DB
        self.index = None
        self.chunk_count = 0
        self._initialize_db()
        self._initialize_index()

        logger.info(f"Vector store initialized (db: {self.db_path}, index: {self.index_path})")

    def _initialize_db(self):
        """Initialize SQLite database with schema."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Drop old table if it has wrong schema (migration)
        cursor.execute('''
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='chunks'
        ''')

        if cursor.fetchone():
            # Check if old schema (chunk_id as PRIMARY KEY)
            cursor.execute("PRAGMA table_info(chunks)")
            columns = {row[1] for row in cursor.fetchall()}

            if 'faiss_index' not in columns:
                logger.info("Dropping old chunks table (schema migration)")
                cursor.execute('DROP TABLE IF EXISTS chunks')
                cursor.execute('DROP INDEX IF EXISTS idx_passage_id')
                cursor.execute('DROP INDEX IF EXISTS idx_contains_answer')

        # Create chunks table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chunks (
                faiss_index INTEGER PRIMARY KEY,
                chunk_id TEXT UNIQUE,
                passage_id TEXT,
                passage_title TEXT,
                chunk_text TEXT,
                sentence_indices TEXT,
                contains_answer BOOLEAN,
                token_count INTEGER,
                source_file TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Create index for common queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_passage_id ON chunks(passage_id)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_contains_answer ON chunks(contains_answer)
        ''')

        conn.commit()
        conn.close()
        logger.debug(f"SQLite database initialized: {self.db_path}")

    def _initialize_index(self):
        """Initialize or load FAISS index."""
        if self.index_path.exists():
            logger.info(f"Loading existing FAISS index from {self.index_path}")
            self.index = faiss.read_index(str(self.index_path))
            self.chunk_count = self.index.ntotal
            logger.info(f"Loaded index with {self.chunk_count} vectors")
        else:
            logger.info(f"Creating new FAISS HNSW index (dimension: {self.embedding_dim})")
            # HNSW: Hierarchical Navigable Small World
            # M=16: number of bi-directional links per node
            # efConstruction=200: construction-time parameter
            self.index = faiss.IndexHNSWFlat(self.embedding_dim, 16)
            self.index.hnsw.efConstruction = 200
            self.chunk_count = 0

    async def add_documents(
        self,
        chunks: List[str],
        metadatas: List[Dict[str, Any]],
        batch_size: int = 32
    ) -> int:
        """
        Add chunks to vector store (embeddings + metadata).

        Args:
            chunks: List of chunk texts
            metadatas: List of metadata dicts
            batch_size: Batch size for embedding generation

        Returns:
            Number of documents added
        """
        if not chunks:
            logger.warning("No chunks to add")
            return 0

        try:
            logger.info(f"Adding {len(chunks)} chunks to vector store...")

            # 1. Generate embeddings
            logger.info(f"Generating embeddings for {len(chunks)} chunks...")
            embeddings = self.embedding_model.encode(
                chunks,
                show_progress_bar=True,
                batch_size=batch_size
            )
            embeddings = embeddings.astype('float32')

            # 2. Add to FAISS index
            logger.info(f"Adding embeddings to FAISS index...")
            self.index.add(embeddings)

            # 3. Add metadata to SQLite
            logger.info(f"Storing metadata in SQLite...")
            self._save_metadata(chunks, metadatas)

            # 4. Save FAISS index to disk
            logger.info(f"Saving FAISS index to {self.index_path}...")
            faiss.write_index(self.index, str(self.index_path))

            self.chunk_count = self.index.ntotal
            logger.info(f"Successfully added {len(chunks)} chunks. Total: {self.chunk_count}")

            return len(chunks)

        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise

    def _save_metadata(self, chunks: List[str], metadatas: List[Dict[str, Any]]):
        """Save chunk metadata to SQLite."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Get current FAISS index offset
        cursor.execute('SELECT MAX(faiss_index) FROM chunks')
        max_index = cursor.fetchone()[0] or -1
        start_index = max_index + 1

        for idx, (chunk_text, metadata) in enumerate(zip(chunks, metadatas)):
            # Serialize sentence_indices and other JSON fields
            sentence_indices = json.dumps(metadata.get('sentence_indices', []))
            faiss_index = start_index + idx

            cursor.execute('''
                INSERT OR REPLACE INTO chunks
                (faiss_index, chunk_id, passage_id, passage_title, chunk_text,
                 sentence_indices, contains_answer, token_count, source_file)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                faiss_index,
                metadata.get('chunk_id'),
                metadata.get('passage_id'),
                metadata.get('passage_title'),
                chunk_text,
                sentence_indices,
                metadata.get('contains_answer', False),
                metadata.get('token_count', 0),
                metadata.get('source_file', '')
            ))

        conn.commit()
        conn.close()

    async def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks.

        Args:
            query: Search query text
            top_k: Number of results to return
            filters: Metadata filters (e.g., {"contains_answer": True})

        Returns:
            List of chunk results with metadata
        """
        try:
            # 1. Embed query
            query_embedding = self.embedding_model.encode(query).astype('float32')
            query_embedding = np.array([query_embedding])

            # 2. Search FAISS index
            distances, indices = self.index.search(query_embedding, k=min(top_k * 2, self.chunk_count))

            # 3. Fetch metadata from SQLite
            results = []
            for idx in indices[0]:
                if idx < 0:  # Invalid index
                    continue

                metadata = self._get_metadata_by_index(idx)
                if metadata is None:
                    continue

                # Apply filters
                if filters and not self._matches_filters(metadata, filters):
                    continue

                results.append(metadata)

            # Return top_k after filtering
            logger.info(f"Retrieved {len(results)} chunks for query: {query[:50]}...")
            return results[:top_k]

        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

    def _get_metadata_by_index(self, idx: int) -> Optional[Dict[str, Any]]:
        """Get metadata for chunk by FAISS index."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Fetch by FAISS index
            cursor.execute('''
                SELECT * FROM chunks WHERE faiss_index = ?
            ''', (int(idx),))

            row = cursor.fetchone()
            conn.close()

            if row:
                return {
                    'chunk_id': row['chunk_id'],
                    'passage_id': row['passage_id'],
                    'passage_title': row['passage_title'],
                    'chunk_text': row['chunk_text'],
                    'sentence_indices': json.loads(row['sentence_indices']),
                    'contains_answer': bool(row['contains_answer']),
                    'token_count': row['token_count'],
                    'source_file': row['source_file']
                }
            return None

        except Exception as e:
            logger.error(f"Error fetching metadata: {e}")
            return None

    def _matches_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if metadata matches filter criteria."""
        for key, value in filters.items():
            if key == 'contains_answer' and metadata.get(key) != value:
                return False
            elif key == 'passage_title' and value not in metadata.get(key, ''):
                return False
            elif key == 'passage_id' and metadata.get(key) != value:
                return False
            elif key not in metadata:
                return False

        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            # Total chunks
            cursor.execute('SELECT COUNT(*) FROM chunks')
            total_chunks = cursor.fetchone()[0]

            # Chunks with answers
            cursor.execute('SELECT COUNT(*) FROM chunks WHERE contains_answer = 1')
            chunks_with_answers = cursor.fetchone()[0]

            # Avg token count
            cursor.execute('SELECT AVG(token_count) FROM chunks')
            avg_tokens = cursor.fetchone()[0] or 0

            conn.close()

            return {
                'total_chunks': total_chunks,
                'chunks_with_answers': chunks_with_answers,
                'avg_token_count': int(avg_tokens),
                'embedding_dim': self.embedding_dim,
                'embedding_model': 'all-MiniLM-L6-v2',
                'db_path': str(self.db_path),
                'index_path': str(self.index_path)
            }

        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}

    def clear(self):
        """Clear all documents from vector store."""
        try:
            logger.warning("Clearing vector store...")

            # Clear SQLite
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute('DELETE FROM chunks')
            conn.commit()
            conn.close()

            # Clear FAISS index
            self._initialize_index()
            self.chunk_count = 0

            logger.info("Vector store cleared")

        except Exception as e:
            logger.error(f"Error clearing vector store: {e}")
            raise
