"""Fase 1 — Exploración del dataset y construcción del subset estratificado.

Uso:
    python scripts/01_explorar_dataset.py

Hace lo siguiente:
    1. Descarga / lee desde caché el split de validación de HotpotQA.
    2. Imprime estadísticas globales (tamaño, distribución de type y level).
    3. Construye un subset estratificado de 500 preguntas con semilla fija.
    4. Persiste el subset en ``data/validation_subset_500.json``.
    5. Reporta cuántos artículos únicos hay en ese subset (insumo para la Fase 2).
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

# Permite ejecutar el script desde la raíz del proyecto
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config
from src.data_utils import (
    build_stratified_subset,
    describe_subset,
    iter_unique_articles,
    load_validation_split,
    save_subset,
)


def main() -> None:
    config.ensure_dirs()

    print("=" * 70)
    print("Fase 1 — Exploración del dataset HotpotQA")
    print("=" * 70)

    print(f"\nDescargando dataset: {config.DATASET_NAME} ({config.DATASET_CONFIG})...")
    ds = load_validation_split()
    print(f"  Split '{config.DATASET_SPLIT}' cargado: {len(ds)} ejemplos.\n")

    # Estadísticas globales
    types = Counter(ds["type"])
    levels = Counter(ds["level"])
    print("Distribución por type:")
    for t, c in types.most_common():
        print(f"  {t:>12} : {c:>6} ({100 * c / len(ds):.1f}%)")
    print("\nDistribución por level:")
    for l, c in levels.most_common():
        print(f"  {l:>12} : {c:>6} ({100 * c / len(ds):.1f}%)")

    # Subset estratificado
    print(f"\nConstruyendo subset estratificado de {config.N_SAMPLES} preguntas (seed={config.SEED})...")
    subset = build_stratified_subset(ds, n=config.N_SAMPLES, seed=config.SEED)
    stats = describe_subset(subset)
    print(f"  Tamaño final: {stats['n']}")
    print(f"  by_type : {stats['by_type']}")
    print(f"  by_level: {stats['by_level']}")

    # Persistencia
    save_subset(subset, config.SUBSET_PATH)
    print(f"\nSubset guardado en: {config.SUBSET_PATH.relative_to(ROOT)}")

    # Conteo de artículos únicos (insumo para la Fase 2)
    n_articles = sum(1 for _ in iter_unique_articles(subset))
    print(f"\nArtículos únicos en el subset (corpus para la base vectorial): {n_articles}")

    # Vistazo a un ejemplo
    print("\n" + "-" * 70)
    print("Ejemplo del subset:")
    print("-" * 70)
    ex = subset[0]
    print(f"  question : {ex['question']}")
    print(f"  answer   : {ex['answer']}")
    print(f"  type     : {ex['type']}")
    print(f"  level    : {ex['level']}")
    n_ctx = len(ex.get("context", {}).get("title", []))
    print(f"  contexts : {n_ctx} párrafos")


if __name__ == "__main__":
    main()
