"""Fase 3 — Baseline closed-book (LLM sin contexto externo).

Uso:
    python scripts/03_baseline_closedbook.py [--limit N]

Opciones:
    --limit N   Procesar solo los primeros N ejemplos (útil para pruebas rápidas).

Hace lo siguiente:
    1. Carga el subset de 500 preguntas de la Fase 1.
    2. Carga el LLM Gemma 4 (requiere haber ejecutado `huggingface-cli login`).
    3. Para cada pregunta genera una respuesta sin ningún contexto externo.
    4. Guarda las predicciones en results/predictions_baseline.json.
       Si el archivo ya existe, retoma desde donde se dejó (resume automático).
    5. Reporta EM y F1 sobre las predicciones acumuladas.
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
from src.prompts import closed_book_messages

OUTPUT_PATH = config.RESULTS_DIR / "predictions_baseline.json"


def load_existing(path: Path) -> dict[str, dict]:
    """Carga predicciones ya guardadas, indexadas por id."""
    if path.exists():
        with path.open(encoding="utf-8") as f:
            records = json.load(f)
        return {r["id"]: r for r in records}
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
    print("Fase 3 — Baseline closed-book")
    print("=" * 70)

    subset = load_subset(config.SUBSET_PATH)
    if args.limit:
        subset = subset[: args.limit]
    print(f"\nSubset cargado: {len(subset)} preguntas.")

    # Resume automático
    done = load_existing(OUTPUT_PATH)
    pending = [ex for ex in subset if ex["id"] not in done]
    print(f"Ya procesadas: {len(done)} | Pendientes: {len(pending)}")

    if pending:
        print(f"\nCargando modelo: {config.LLM_MODEL}")
        model, processor = load_model()
        print("  Modelo cargado.\n")

        for ex in tqdm(pending, desc="Generando respuestas"):
            messages = closed_book_messages(ex["question"])
            pred = generate(model, processor, messages)
            done[ex["id"]] = {
                "id": ex["id"],
                "question": ex["question"],
                "gold": ex["answer"],
                "pred": pred,
                "type": ex.get("type", "unknown"),
                "level": ex.get("level", "unknown"),
            }
            # Guardar tras cada predicción para no perder progreso
            save_records(done, OUTPUT_PATH)

    # Métricas
    records = list(done.values())
    metrics = evaluate(records)

    print("\n" + "=" * 70)
    print("Resultados — Baseline closed-book")
    print("=" * 70)
    print(f"  N    : {metrics['n']}")
    print(f"  EM   : {metrics['em']:.4f} ({metrics['em']*100:.1f}%)")
    print(f"  F1   : {metrics['f1']:.4f} ({metrics['f1']*100:.1f}%)")

    print("\nPor type:")
    for k, v in sorted(metrics["by_type"].items()):
        print(f"  {k:>12}: EM={v['em']*100:.1f}%  F1={v['f1']*100:.1f}%")

    print("\nPor level:")
    for k, v in sorted(metrics["by_level"].items()):
        print(f"  {k:>12}: EM={v['em']*100:.1f}%  F1={v['f1']*100:.1f}%")

    print(f"\nPredicciones guardadas en: {OUTPUT_PATH.relative_to(ROOT)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
