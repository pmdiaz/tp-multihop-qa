"""Fase 2 — Construcción de la base de datos vectorial.

Uso:
    python scripts/02_construir_vector_store.py

Hace lo siguiente:
    1. Carga el subset persistido en la Fase 1.
    2. Extrae los artículos únicos y los convierte en Documents de LangChain.
    3. Embebe con sentence-transformers/all-MiniLM-L6-v2 e indexa con FAISS.
    4. Persiste el índice en data/faiss_index/.
    5. Validación cualitativa: consultas de prueba para verificar el retrieval.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config
from src.data_utils import iter_unique_articles, load_subset
from src.vector_store import (
    build_vector_store,
    get_embeddings,
    get_retriever,
    load_vector_store,
    save_vector_store,
    subset_to_documents,
)

# ---------------------------------------------------------------------------
# Queries de validación cualitativa
# ---------------------------------------------------------------------------
VALIDATION_QUERIES = [
    "Who directed the movie about a bear?",
    "What is the nationality of the CEO of Apple?",
    "In which country was the founder of Wikipedia born?",
]


def main() -> None:
    config.ensure_dirs()

    print("=" * 70)
    print("Fase 2 — Construcción del índice vectorial FAISS")
    print("=" * 70)

    # 1. Cargar subset
    print(f"\nCargando subset desde: {config.SUBSET_PATH.relative_to(ROOT)}")
    subset = load_subset(config.SUBSET_PATH)
    print(f"  {len(subset)} preguntas cargadas.")

    # 2. Artículos únicos
    docs = subset_to_documents(subset)
    n_articles = sum(1 for _ in iter_unique_articles(subset))
    print(f"\nArtículos únicos en el corpus: {n_articles}")
    print(f"  → {len(docs)} Documents creados (granularidad: artículo completo).")

    # 3. Embeddings
    print(f"\nCargando modelo de embeddings: {config.EMBEDDING_MODEL}")
    embeddings = get_embeddings()

    # 4. Construir índice
    print("\nConstruyendo índice FAISS...")
    vs = build_vector_store(subset, embeddings=embeddings)
    print("  Índice construido.")

    # 5. Persistir
    save_vector_store(vs, config.VECTOR_STORE_DIR)
    print(f"  Índice guardado en: {config.VECTOR_STORE_DIR.relative_to(ROOT)}")

    # 6. Round-trip check: recargar desde disco
    print("\nVerificando carga desde disco...")
    vs_loaded = load_vector_store(config.VECTOR_STORE_DIR, embeddings=embeddings)
    print("  OK — índice recargado correctamente.")

    # 7. Validación cualitativa
    print("\n" + "-" * 70)
    print(f"Validación cualitativa (top-{config.DEFAULT_TOP_K} resultados por query)")
    print("-" * 70)
    retriever = get_retriever(vs_loaded, k=config.DEFAULT_TOP_K)
    for query in VALIDATION_QUERIES:
        print(f"\nQuery: {query!r}")
        results = retriever.invoke(query)
        for i, doc in enumerate(results, 1):
            title = doc.metadata.get("title", "?")
            snippet = doc.page_content[:120].replace("\n", " ")
            print(f"  [{i}] {title}: {snippet}...")

    print("\n" + "=" * 70)
    print("Fase 2 completada.")
    print("=" * 70)


if __name__ == "__main__":
    main()
