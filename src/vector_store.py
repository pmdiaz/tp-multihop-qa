"""Construcción y acceso al índice vectorial FAISS.

Decisión de diseño: granularidad a nivel de artículo completo (título + oraciones
concatenadas). Un vector por artículo, fiel a la estructura del corpus HotpotQA.

Funciones principales:

- ``get_embeddings``: instancia el modelo de embeddings (MiniLM o BGE).
- ``build_vector_store``: convierte el subset en documentos, embebe e indexa.
- ``save_vector_store`` / ``load_vector_store``: persistencia en disco.
- ``get_retriever``: devuelve un retriever FAISS listo para usar.
- ``build_bm25_retriever``: retriever BM25 (keyword) sobre el mismo corpus.
- ``build_hybrid_retriever``: EnsembleRetriever FAISS + BM25.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from . import config
from .data_utils import iter_unique_articles


def get_embeddings(model_name: str = config.EMBEDDING_MODEL) -> HuggingFaceEmbeddings:
    """Instancia el modelo de embeddings.

    Para BGE, activa normalización de vectores (recomendado por el modelo).
    """
    kwargs: dict = {}
    if "bge" in model_name.lower():
        kwargs["encode_kwargs"] = {"normalize_embeddings": True}
    return HuggingFaceEmbeddings(model_name=model_name, **kwargs)


def subset_to_documents(subset: list[dict[str, Any]]) -> list[Document]:
    """Convierte el subset en Documents de LangChain (uno por artículo único)."""
    docs = []
    for article in iter_unique_articles(subset):
        docs.append(
            Document(
                page_content=f"{article['title']}\n{article['text']}",
                metadata={"title": article["title"]},
            )
        )
    return docs


def build_vector_store(
    subset: list[dict[str, Any]],
    embeddings: HuggingFaceEmbeddings | None = None,
) -> FAISS:
    """Embebe los artículos del subset y construye el índice FAISS."""
    if embeddings is None:
        embeddings = get_embeddings()
    docs = subset_to_documents(subset)
    return FAISS.from_documents(docs, embeddings)


def save_vector_store(
    vs: FAISS,
    path: Path = config.VECTOR_STORE_DIR,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(path))


def load_vector_store(
    path: Path = config.VECTOR_STORE_DIR,
    embeddings: HuggingFaceEmbeddings | None = None,
) -> FAISS:
    if embeddings is None:
        embeddings = get_embeddings()
    return FAISS.load_local(
        str(path),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def get_retriever(vs: FAISS, k: int = config.DEFAULT_TOP_K):
    """Devuelve un retriever FAISS configurado con top-k."""
    return vs.as_retriever(search_kwargs={"k": k})


# ---------------------------------------------------------------------------
# BM25 + retrieval híbrido
# ---------------------------------------------------------------------------

def build_bm25_retriever(
    subset: list[dict[str, Any]],
    k: int = config.DEFAULT_TOP_K,
) -> BM25Retriever:
    """Construye un retriever BM25 (keyword matching) sobre el corpus.

    BM25 complementa a FAISS en queries con nombres propios exactos donde
    la similitud semántica falla pero el match lexical es preciso.
    """
    docs = subset_to_documents(subset)
    retriever = BM25Retriever.from_documents(docs)
    retriever.k = k
    return retriever


def build_hybrid_retriever(
    vs: FAISS,
    subset: list[dict[str, Any]],
    k: int = config.DEFAULT_TOP_K,
    bm25_weight: float = 0.4,
) -> EnsembleRetriever:
    """Combina FAISS (semántico) y BM25 (keyword) con pesos configurables.

    ``bm25_weight=0.4`` da 60% de peso a FAISS y 40% a BM25.
    Ajustar si el dataset tiene muchos nombres propios (subir BM25)
    o si las queries son más semánticas (bajar BM25).
    """
    faiss_retriever = get_retriever(vs, k=k)
    bm25_retriever = build_bm25_retriever(subset, k=k)
    return EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[bm25_weight, 1.0 - bm25_weight],
    )
