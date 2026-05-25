"""Fase 2b — Construcción del índice vectorial con BGE-large (mejora 1).

Uso:
    python scripts/02b_construir_vector_store_bge.py

Igual que 02_construir_vector_store.py pero usando BAAI/bge-large-en-v1.5
(335M params, MTEB score 63.6 vs 56.3 de MiniLM) y guardando el índice en
data/faiss_index_bge/ para no sobreescribir el índice original.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config
from src.data_utils import load_subset
from src.vector_store import (
    build_vector_store,
    get_embeddings,
    get_retriever,
    load_vector_store,
    save_vector_store,
    subset_to_documents,
)

VALIDATION_QUERIES = [
    "Who directed the 2007 film Atonement?",
    "What is the nationality of the CEO of Apple?",
    "In which country was the founder of Wikipedia born?",
]


def main() -> None:
    config.ensure_dirs()

    print("=" * 70)
    print("Fase 2b — Índice FAISS con BGE-large")
    print("=" * 70)

    subset = load_subset(config.SUBSET_PATH)
    docs = subset_to_documents(subset)
    print(f"\nSubset: {len(subset)} preguntas | {len(docs)} artículos únicos")

    print(f"\nCargando modelo de embeddings: {config.EMBEDDING_MODEL_BGE}")
    embeddings = get_embeddings(config.EMBEDDING_MODEL_BGE)

    print("\nConstruyendo índice FAISS con BGE...")
    vs = build_vector_store(subset, embeddings=embeddings)

    save_vector_store(vs, config.VECTOR_STORE_DIR_BGE)
    print(f"  Índice guardado en: {config.VECTOR_STORE_DIR_BGE.relative_to(ROOT)}")

    print("\nVerificando carga desde disco...")
    vs_loaded = load_vector_store(config.VECTOR_STORE_DIR_BGE, embeddings=embeddings)
    print("  OK — índice recargado correctamente.")

    print("\n" + "-" * 70)
    print(f"Validación cualitativa (top-{config.DEFAULT_TOP_K})")
    print("-" * 70)
    retriever = get_retriever(vs_loaded, k=config.DEFAULT_TOP_K)
    for query in VALIDATION_QUERIES:
        print(f"\nQuery: {query!r}")
        for i, doc in enumerate(retriever.invoke(query), 1):
            title = doc.metadata.get("title", "?")
            snippet = doc.page_content[:100].replace("\n", " ")
            print(f"  [{i}] {title}: {snippet}...")

    print("\n" + "=" * 70)
    print("Fase 2b completada.")
    print("=" * 70)


if __name__ == "__main__":
    main()
