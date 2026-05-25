"""Fase 5b — Agente ReAct con retrieval mejorado.

Uso:
    python scripts/05b_agente_mejorado.py [opciones]

Opciones:
    --embedding  mini|bge     Modelo de embeddings (default: mini)
    --hybrid                  Activar retrieval híbrido BM25+FAISS
    --multi-query             Activar multi-query (agrega la pregunta original)
    --limit N                 Procesar solo los primeros N ejemplos

Ejemplos:
    # Solo BGE
    python scripts/05b_agente_mejorado.py --embedding bge

    # BGE + híbrido + multi-query (todas las mejoras)
    python scripts/05b_agente_mejorado.py --embedding bge --hybrid --multi-query

El nombre del archivo de salida refleja las mejoras activas, por ejemplo:
    results/predictions_agent_bge_hybrid_mq.json

Comparar contra el baseline del agente:
    results/predictions_agent0.json   (MiniLM, sin híbrido, sin multi-query)
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
from src.vector_store import (
    build_hybrid_retriever,
    get_embeddings,
    load_vector_store,
)


def build_output_name(embedding: str, hybrid: bool, multi_query: bool) -> str:
    parts = ["predictions_agent", embedding]
    if hybrid:
        parts.append("hybrid")
    if multi_query:
        parts.append("mq")
    return "_".join(parts) + ".json"


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
    parser.add_argument("--embedding", choices=["mini", "bge"], default="mini")
    parser.add_argument("--hybrid", action="store_true")
    parser.add_argument("--multi-query", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    config.ensure_dirs()

    tag = build_output_name(args.embedding, args.hybrid, args.multi_query)
    output_path = config.RESULTS_DIR / tag

    print("=" * 70)
    print("Fase 5b — Agente ReAct con retrieval mejorado")
    print(f"  embedding  : {args.embedding}")
    print(f"  hybrid     : {args.hybrid}")
    print(f"  multi-query: {args.multi_query}")
    print(f"  output     : {tag}")
    print("=" * 70)

    # ── Subset ───────────────────────────────────────────────────────────────
    subset = load_subset(config.SUBSET_PATH)
    if args.limit:
        subset = subset[: args.limit]
    print(f"\nSubset cargado: {len(subset)} preguntas.")

    # ── Embeddings y vector store ─────────────────────────────────────────────
    if args.embedding == "bge":
        emb_model = config.EMBEDDING_MODEL_BGE
        vs_dir = config.VECTOR_STORE_DIR_BGE
    else:
        emb_model = config.EMBEDDING_MODEL
        vs_dir = config.VECTOR_STORE_DIR

    print(f"\nCargando embeddings: {emb_model}")
    embeddings = get_embeddings(emb_model)

    print(f"Cargando índice FAISS: {vs_dir.relative_to(ROOT)}")
    vs = load_vector_store(vs_dir, embeddings=embeddings)

    # ── Retriever ─────────────────────────────────────────────────────────────
    if args.hybrid:
        print("Construyendo retriever híbrido BM25+FAISS...")
        retriever = build_hybrid_retriever(vs, subset, k=config.DEFAULT_TOP_K)
        print("  OK.")
    else:
        retriever = vs

    # ── LLM ──────────────────────────────────────────────────────────────────
    print(f"\nCargando modelo: {config.LLM_MODEL}")
    model, tokenizer = load_model()

    # ── Inferencia ───────────────────────────────────────────────────────────
    done = load_existing(output_path)
    pending = [ex for ex in subset if ex["id"] not in done]
    print(f"\nYa procesadas: {len(done)} | Pendientes: {len(pending)}\n")

    for ex in tqdm(pending, desc="Agente mejorado"):
        result = run_agent(
            model, tokenizer, retriever,
            question=ex["question"],
            max_steps=config.AGENT_MAX_STEPS,
            k=config.DEFAULT_TOP_K,
            multi_query=args.multi_query,
        )
        done[ex["id"]] = {
            "id":       ex["id"],
            "question": ex["question"],
            "gold":     ex["answer"],
            "pred":     result["answer"],
            "type":     ex.get("type", "unknown"),
            "level":    ex.get("level", "unknown"),
            "steps":    result["steps"],
            "finished": result["finished"],
            "trace":    result["trace"],
        }
        save_records(done, output_path)

    # ── Resultados ───────────────────────────────────────────────────────────
    records = list(done.values())
    m = evaluate(records)

    finished_count = sum(1 for r in records if r.get("finished"))
    avg_steps = sum(r.get("steps", 0) for r in records) / len(records) if records else 0
    search_calls = sum(
        sum(1 for s in r.get("trace", []) if s.get("action_type") == "search")
        for r in records
    )
    lookup_calls = sum(
        sum(1 for s in r.get("trace", []) if s.get("action_type") == "lookup")
        for r in records
    )

    print("\n" + "=" * 70)
    print(f"Resultados — {tag}")
    print("=" * 70)
    print(f"  N    : {m['n']}")
    print(f"  EM   : {m['em']*100:.1f}%")
    print(f"  F1   : {m['f1']*100:.1f}%")
    print(f"\n  finish[] rate   : {finished_count}/{m['n']} ({finished_count/m['n']*100:.1f}%)")
    print(f"  Pasos promedio  : {avg_steps:.1f}")
    print(f"  Llamadas search : {search_calls}  lookup: {lookup_calls}")
    print("\nPor type:")
    for t, v in sorted(m["by_type"].items()):
        by_type = m["by_type"].get(t, {})
        print(f"  {t:>12}: EM={by_type['em']*100:.1f}%  F1={by_type['f1']*100:.1f}%")
    print(f"\nGuardado en: {output_path.relative_to(ROOT)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
