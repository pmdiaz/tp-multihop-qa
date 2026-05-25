"""Fase 4 — Sistema RAG sobre la base de datos vectorial.

Uso:
    python scripts/04_rag.py [--limit N]

Opciones:
    --limit N   Procesar solo los primeros N ejemplos (útil para pruebas rápidas).

Hace lo siguiente:
    1. Carga el subset y el índice FAISS (construidos en Fases 1 y 2).
    2. Carga el LLM Gemma 4.
    3. Para cada pregunta recupera los top-MAX_K documentos del índice (una sola
       consulta a FAISS por pregunta, luego se recorta para k=3 y k=5).
    4. Para cada valor de k en TOP_K_VALUES genera y guarda predicciones en
       results/predictions_rag_k{k}.json con resume automático.
    5. Reporta la curva EM/F1 vs k al finalizar.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config
from src.data_utils import load_subset
from src.evaluation import evaluate
from src.llm import generate, load_model
from src.prompts import rag_messages
from src.vector_store import get_embeddings, load_vector_store


def output_path(k: int) -> Path:
    return config.RESULTS_DIR / f"predictions_rag_k{k}.json"


def load_existing(path: Path) -> dict[str, dict]:
    if path.exists():
        with path.open(encoding="utf-8") as f:
            return {r["id"]: r for r in json.load(f)}
    return {}


def save_records(records: dict[str, dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(list(records.values()), f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    config.ensure_dirs()

    print("=" * 70)
    print("Fase 4 — RAG")
    print("=" * 70)

    # ── Subset ──────────────────────────────────────────────────────────────
    subset = load_subset(config.SUBSET_PATH)
    if args.limit:
        subset = subset[: args.limit]
    print(f"\nSubset cargado: {len(subset)} preguntas.")

    # ── Vector store ─────────────────────────────────────────────────────────
    print(f"\nCargando índice FAISS desde: {config.VECTOR_STORE_DIR.relative_to(ROOT)}")
    embeddings = get_embeddings()
    vs = load_vector_store(config.VECTOR_STORE_DIR, embeddings=embeddings)
    print("  Índice cargado.")

    # ── LLM ──────────────────────────────────────────────────────────────────
    print(f"\nCargando modelo: {config.LLM_MODEL}")
    model, tokenizer = load_model()

    # ── Inferencia ───────────────────────────────────────────────────────────
    # Recuperamos max(TOP_K_VALUES) docs por pregunta una sola vez y recortamos
    # para k menores, evitando consultas redundantes a FAISS.
    max_k = max(config.TOP_K_VALUES)

    # Acumuladores por k: {id: record}
    done: dict[int, dict[str, dict]] = {
        k: load_existing(output_path(k)) for k in config.TOP_K_VALUES
    }

    # Preguntas pendientes: las que faltan en ALGUNO de los valores de k
    pending = [
        ex for ex in subset
        if any(ex["id"] not in done[k] for k in config.TOP_K_VALUES)
    ]
    print(f"\nYa procesadas (k={config.TOP_K_VALUES[0]}): {len(done[config.TOP_K_VALUES[0]])} | "
          f"Pendientes: {len(pending)}")

    print(f"\nIniciando inferencia (k values: {list(config.TOP_K_VALUES)})...\n")
    for ex in tqdm(pending, desc="Generando respuestas RAG"):
        # Una sola consulta a FAISS con el k máximo
        top_docs = vs.similarity_search(ex["question"], k=max_k)

        for k in config.TOP_K_VALUES:
            if ex["id"] in done[k]:
                continue

            contexts = [doc.page_content for doc in top_docs[:k]]
            messages = rag_messages(ex["question"], contexts)
            pred = generate(model, tokenizer, messages)

            done[k][ex["id"]] = {
                "id": ex["id"],
                "question": ex["question"],
                "gold": ex["answer"],
                "pred": pred,
                "type": ex.get("type", "unknown"),
                "level": ex.get("level", "unknown"),
                "k": k,
            }

        # Guardar tras cada pregunta para no perder progreso
        for k in config.TOP_K_VALUES:
            save_records(done[k], output_path(k))

    # ── Resultados ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Curva EM/F1 vs k — RAG")
    print("=" * 70)
    print(f"  {'k':>4}  {'N':>5}  {'EM':>8}  {'F1':>8}")
    print("  " + "-" * 34)
    for k in config.TOP_K_VALUES:
        records = list(done[k].values())
        m = evaluate(records)
        by_type = ", ".join(f"{t}→EM={v['em']*100:.1f}%" for t, v in sorted(m["by_type"].items()))
        by_level = ", ".join(f"{l}→EM={v['em']*100:.1f}%" for l, v in sorted(m["by_level"].items()))
        print(f"  {k:>4}  {m['n']:>5}  {m['em']*100:>7.1f}%  {m['f1']*100:>7.1f}%")
        print(f"        por type : {by_type}")
        print(f"        por level: {by_level}")

    print(f"\nPredicciones guardadas en: {config.RESULTS_DIR.relative_to(ROOT)}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
