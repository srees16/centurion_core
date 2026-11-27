"""
ChromaDB vector store for the Centurion RAG pipeline.

- ``VectorStoreManager``: the one persistent collection (``RAGConfig.chroma_collection_name``) that
  ``PDFIngestionService``, ``RAGQueryEngine`` and ``HybridSearcher`` use.
- ``_normalize_for_embedding``: the code-chunk normalisation the PDF ingestion embeds instead of the stored text.

Tech stack: Python 3.11 · raw chromadb client · No LangChain.
"""

from __future__ import annotations

import logging
import re as _re
from typing import Any, Dict, List, Optional

from rag_pipeline.config import RAGConfig

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# 1. Lazy ChromaDB import
# ═══════════════════════════════════════════════════════════════════════════

def _import_chromadb():
    """Lazy import of chromadb to avoid ~1 s startup penalty."""
    import chromadb as _chromadb
    return _chromadb



# ═══════════════════════════════════════════════════════════════════════════
# 2. Pre-embedding normalisation for code chunks
# ═══════════════════════════════════════════════════════════════════════════


_TRAILING_COMMENT_RE = _re.compile(r"\s+#[^\n]*$", _re.MULTILINE)
_EXCESS_BLANK_RE = _re.compile(r"\n{3,}")
_DEF_FUNC_NAME_RE = _re.compile(r"(\bdef\s+)(\w+)")


def _normalize_for_embedding(text: str) -> str:
    """Normalise code text for embedding while preserving semantics.

    Transformations (applied in order):

    1. **Remove trailing comments** — inline ``# …`` after code.
    2. **Normalise whitespace** — collapse 3+ consecutive blank lines
       to 2; strip leading/trailing whitespace on each line.
    3. **Consistent indentation** — convert tabs to 4 spaces.
    4. **Lowercase function names** in ``def`` statements so that
       ``applyPtSlOnT1`` and ``applyptslont1`` produce the same
       embedding neighbourhood.

    The **original** (untouched) text should still be stored as the
    ``document`` in ChromaDB so that retrieval returns verbatim code.
    Only the *embedding vector* is computed from this normalised form.

    Returns:
        Normalised text suitable for embedding.
    """
    # 1. Remove trailing comments
    text = _TRAILING_COMMENT_RE.sub("", text)

    # 2. Tab 4 spaces
    text = text.expandtabs(4)

    # 3. Strip trailing whitespace per line, collapse excess blanks
    lines = [ln.rstrip() for ln in text.split("\n")]
    text = "\n".join(lines)
    text = _EXCESS_BLANK_RE.sub("\n\n", text)

    # 4. Lowercase function names in def statements
    text = _DEF_FUNC_NAME_RE.sub(
        lambda m: m.group(1) + m.group(2).lower(), text
    )

    return text.strip()



# ═══════════════════════════════════════════════════════════════════════════
# 3. VectorStoreManager
# ═══════════════════════════════════════════════════════════════════════════

class VectorStoreManager:
    """Manages a persistent ChromaDB collection for the RAG pipeline.

    **Backward-compatible** — this is the original single-collection
    manager used by ``PDFIngestionService``, ``RAGQueryEngine``,
    ``HybridSearcher``, and other pipeline components.

    Usage::

        config = RAGConfig()
        vs = VectorStoreManager(config)
        vs.add_documents(ids, texts, metadatas, embeddings)
        results = vs.query(query_embeddings, n_results=5)
    """

    def __init__(self, config: Optional[RAGConfig] = None) -> None:
        self._config = config or RAGConfig()
        self._client = None
        self._collection = None

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------
    @property
    def client(self):
        if self._client is None:
            _chromadb = _import_chromadb()
            from chromadb.config import Settings as ChromaSettings
            self._client = _chromadb.PersistentClient(
                path=self._config.chroma_persist_dir,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                ),
            )
            logger.info(
                "ChromaDB client initialised at %s",
                self._config.chroma_persist_dir,
            )
        return self._client

    @property
    def collection(self):
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self._config.chroma_collection_name,
                metadata={
                    "hnsw:space": "cosine",
                    "hnsw:M": self._config.hnsw_m,
                    "hnsw:construction_ef": self._config.hnsw_ef_construction,
                    "hnsw:search_ef": self._config.hnsw_ef_search,
                },
            )
            logger.info(
                "Collection '%s' ready (%d documents) "
                "[HNSW M=%d, ef_construction=%d, ef_search=%d]",
                self._config.chroma_collection_name,
                self._collection.count(),
                self._config.hnsw_m,
                self._config.hnsw_ef_construction,
                self._config.hnsw_ef_search,
            )
        return self._collection

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------
    def add_documents(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        embeddings: Optional[List[List[float]]] = None,
    ) -> None:
        """Insert or upsert documents into the collection."""
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        logger.info("Upserted %d documents into collection", len(ids))

    def query(
        self,
        query_texts: Optional[List[str]] = None,
        query_embeddings: Optional[List[List[float]]] = None,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Perform similarity search.

        Returns ChromaDB-style results dict with keys:
            ids, documents, metadatas, distances
        """
        include = include or ["documents", "metadatas", "distances"]
        kwargs: Dict[str, Any] = {
            "n_results": n_results,
            "include": include,
        }
        if query_texts is not None:
            kwargs["query_texts"] = query_texts
        if query_embeddings is not None:
            kwargs["query_embeddings"] = query_embeddings
        if where is not None:
            kwargs["where"] = where

        return self.collection.query(**kwargs)

    def delete_by_metadata(
        self, where: Dict[str, Any]
    ) -> None:
        """Delete documents matching a metadata filter."""
        results = self.collection.get(where=where)
        if results["ids"]:
            self.collection.delete(ids=results["ids"])
            logger.info(
                "Deleted %d documents matching %s", len(results["ids"]), where
            )

    def delete_by_ids(self, ids: List[str]) -> None:
        """Delete documents by their IDs."""
        if ids:
            self.collection.delete(ids=ids)
            logger.info("Deleted %d documents by ID", len(ids))

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------
    def count(self) -> int:
        """Return total document count in the collection."""
        return self.collection.count()

    def source_exists(self, source_name: str) -> bool:
        """Check whether any chunks from *source_name* are stored."""
        try:
            results = self.collection.get(
                where={"source": source_name},
                include=[],
            )
            return bool(results.get("ids"))
        except Exception:
            return False

    def get_source_file_hash(self, source_name: str) -> Optional[str]:
        """Return the ``file_hash`` stored for *source_name*, or None."""
        try:
            results = self.collection.get(
                where={"source": source_name},
                include=["metadatas"],
            )
            metas = results.get("metadatas") or []
            if metas and metas[0]:
                return metas[0].get("file_hash")
        except Exception:
            pass
        return None

    def get_source_chunk_count(self, source_name: str) -> int:
        """Return the number of chunks stored for *source_name*."""
        try:
            results = self.collection.get(
                where={"source": source_name},
                include=[],
            )
            return len(results.get("ids") or [])
        except Exception:
            return 0

    def list_sources(self) -> List[str]:
        """Return unique source filenames stored in the collection."""
        all_meta = self.collection.get(include=["metadatas"])
        sources = set()
        for meta in (all_meta.get("metadatas") or []):
            if meta and "source" in meta:
                sources.add(meta["source"])
        return sorted(sources)

    def get_collection_stats(self) -> Dict[str, Any]:
        """Return summary statistics for the collection."""
        count_val = self.count()
        sources = self.list_sources()
        return {
            "collection_name": self._config.chroma_collection_name,
            "total_documents": count_val,
            "total_sources": len(sources),
            "sources": sources,
            "persist_directory": self._config.chroma_persist_dir,
            "hnsw_m": self._config.hnsw_m,
            "hnsw_ef_construction": self._config.hnsw_ef_construction,
            "hnsw_ef_search": self._config.hnsw_ef_search,
        }

    def get_source_details(self, source_name: str) -> Dict[str, Any]:
        """Return metadata details for a specific indexed source.

        Reads stored chunk metadata to extract page count, chunk count,
        ingestion timestamp, and file size (from disk if available).
        """
        details: Dict[str, Any] = {
            "source": source_name,
            "chunks": 0,
            "page_count": None,
            "ingested_at": None,
            "file_size_bytes": None,
        }
        try:
            results = self.collection.get(
                where={"source": source_name},
                include=["metadatas"],
            )
            metas = results.get("metadatas") or []
            details["chunks"] = len(metas)
            if metas and metas[0]:
                first = metas[0]
                details["page_count"] = first.get("page_count")
                details["ingested_at"] = first.get("ingested_at")
        except Exception:
            pass

        # Try to get file size from the upload directory
        try:
            from pathlib import Path
            upload_path = Path(self._config.pdf_upload_dir) / source_name
            if upload_path.exists():
                details["file_size_bytes"] = upload_path.stat().st_size
        except Exception:
            pass

        return details

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------
    def reset_collection(self) -> None:
        """Drop and recreate the collection (destructive)."""
        self.client.delete_collection(self._config.chroma_collection_name)
        self._collection = None
        logger.warning(
            "Collection '%s' has been reset",
            self._config.chroma_collection_name,
        )
        _ = self.collection
