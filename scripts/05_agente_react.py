"""Fase 5 — Agente ReAct sobre la base de datos vectorial.

Uso:
    python scripts/05_agente_react.py [--limit N] [--k K]

Opciones:
    --limit N   Procesar solo los primeros N ejemplos (útil para pruebas rápidas).
    --k K       Top-k documentos por búsqueda (default: DEFAULT_TOP_K del config).

Hace lo siguiente:
    1. Carga el subset y el índice FAISS.
    2. Carga el LLM Gemma 4.
    3. Para cada pregunta ejecuta el loop ReAct (Thought → Action → Observation)
       hasta que el agente llame a finish[] o se alcance max_steps.
    4. Guarda predicciones + trace en results/predictions_agent.json.
       Resume automático si se interrumpe.
    5. Reporta EM/F1 y estadísticas del comportamiento del agente.
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
from src.agent import run_agent
from src.data_utils import load_subset
from src.evaluation import evaluate
from src.llm import load_model
from src.vector_store import get_embeddings, load_vector_store

OUTPUT_PATH = config.RESULTS_DIR / "predictions_agent.json"


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
    parser.add_argument("--k", type=int, default=config.DEFAULT_TOP_K)
    args = parser.parse_args()

    config.ensure_dirs()

    print("=" * 70)
    print("Fase 5 — Agente ReAct")
    print(f"  max_steps={config.AGENT_MAX_STEPS}  k={args.k}")
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
    done = load_existing(OUTPUT_PATH)
    pending = [ex for ex in subset if ex["id"] not in done]
    print(f"\nYa procesadas: {len(done)} | Pendientes: {len(pending)}")
    print(f"\nIniciando inferencia (esto es más lento que RAG — hasta {config.AGENT_MAX_STEPS} "
          f"llamadas al LLM por pregunta)...\n")

    for ex in tqdm(pending, desc="Agente ReAct"):
        result = run_agent(
            model, tokenizer, vs,
            question=ex["question"],
            max_steps=config.AGENT_MAX_STEPS,
            k=args.k,
        )
        done[ex["id"]] = {
            "id": ex["id"],
            "question": ex["question"],
            "gold": ex["answer"],
            "pred": result["answer"],
            "type": ex.get("type", "unknown"),
            "level": ex.get("level", "unknown"),
            "steps": result["steps"],
            "finished": result["finished"],
            "trace": result["trace"],
        }
        save_records(done, OUTPUT_PATH)

    # ── Resultados ───────────────────────────────────────────────────────────
    records = list(done.values())
    m = evaluate(records)

    finished_count = sum(1 for r in records if r.get("finished"))
    avg_steps = sum(r.get("steps", 0) for r in records) / len(records) if records else 0

    # Estadísticas de uso de herramientas
    search_calls = sum(
        sum(1 for s in r.get("trace", []) if s.get("action_type") == "search")
        for r in records
    )
    lookup_calls = sum(
        sum(1 for s in r.get("trace", []) if s.get("action_type") == "lookup")
        for r in records
    )
    questions_with_lookup = sum(
        1 for r in records
        if any(s.get("action_type") == "lookup" for s in r.get("trace", []))
    )

    print("\n" + "=" * 70)
    print("Resultados — Agente ReAct")
    print("=" * 70)
    print(f"  N        : {m['n']}")
    print(f"  EM       : {m['em']*100:.1f}%")
    print(f"  F1       : {m['f1']*100:.1f}%")
    print(f"\n  Episodios terminados con finish[]: {finished_count}/{m['n']} "
          f"({finished_count/m['n']*100:.1f}%)")
    print(f"  Pasos promedio por pregunta      : {avg_steps:.1f}")
    print(f"\n  Llamadas a search[] (total)      : {search_calls}")
    print(f"  Llamadas a lookup[] (total)      : {lookup_calls}")
    print(f"  Preguntas que usaron lookup[]    : {questions_with_lookup}/{m['n']} "
          f"({questions_with_lookup/m['n']*100:.1f}%)")

    print("\nPor type:")
    for t, v in sorted(m["by_type"].items()):
        print(f"  {t:>12}: EM={v['em']*100:.1f}%  F1={v['f1']*100:.1f}%")

    print(f"\nPredicciones guardadas en: {OUTPUT_PATH.relative_to(ROOT)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
