"""Carga del dataset HotpotQA y construcción del subset estratificado.

Funciones principales:

- ``load_validation_split``: descarga / lee desde caché el split de validación.
- ``build_stratified_subset``: arma una muestra reproducible balanceada por
  ``type`` (bridge / comparison) y ``level`` (easy / medium / hard).
- ``save_subset`` / ``load_subset``: persistencia en JSON para reutilizar en
  todas las fases sin tener que recargar el dataset completo.
- ``iter_unique_articles``: itera sobre todos los artículos únicos
  (título + oraciones) que aparecen en el subset, necesarios para construir
  la base vectorial en la Fase 2.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

from datasets import Dataset, load_dataset

from . import config


# ---------------------------------------------------------------------------
# Carga del dataset
# ---------------------------------------------------------------------------
def load_validation_split() -> Dataset:
    """Devuelve el split ``validation`` de HotpotQA (config ``distractor``)."""
    return load_dataset(
        config.DATASET_NAME,
        config.DATASET_CONFIG,
        split=config.DATASET_SPLIT,
        trust_remote_code=True,
    )


# ---------------------------------------------------------------------------
# Subset estratificado
# ---------------------------------------------------------------------------
def build_stratified_subset(
    dataset: Dataset,
    n: int = config.N_SAMPLES,
    seed: int = config.SEED,
) -> list[dict[str, Any]]:
    """Devuelve una lista de ejemplos estratificada por ``type`` y ``level``.

    Se calculan las proporciones de cada combinación ``(type, level)`` en el
    dataset completo, y se muestrea de cada estrato la cantidad correspondiente
    para llegar a ``n`` ejemplos. La semilla fija garantiza reproducibilidad.
    """
    rng = random.Random(seed)

    # Indexar por (type, level)
    buckets: dict[tuple[str, str], list[int]] = defaultdict(list)
    for idx, row in enumerate(dataset):
        key = (row.get("type", "unknown"), row.get("level", "unknown"))
        buckets[key].append(idx)

    total = sum(len(v) for v in buckets.values())

    # Asignar cuotas proporcionales (al menos 1 por estrato no vacío)
    quotas: dict[tuple[str, str], int] = {}
    for key, indices in buckets.items():
        quotas[key] = max(1, round(n * len(indices) / total))

    # Ajustar para que la suma sea exactamente n
    diff = n - sum(quotas.values())
    keys_sorted = sorted(quotas, key=lambda k: -len(buckets[k]))
    i = 0
    while diff != 0 and keys_sorted:
        key = keys_sorted[i % len(keys_sorted)]
        if diff > 0:
            quotas[key] += 1
            diff -= 1
        else:
            if quotas[key] > 1:
                quotas[key] -= 1
                diff += 1
        i += 1

    # Muestrear por estrato
    chosen: list[int] = []
    for key, indices in buckets.items():
        k = min(quotas[key], len(indices))
        chosen.extend(rng.sample(indices, k))

    chosen.sort()
    return [dict(dataset[i]) for i in chosen]


# ---------------------------------------------------------------------------
# Persistencia
# ---------------------------------------------------------------------------
def save_subset(subset: list[dict[str, Any]], path: Path = config.SUBSET_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(subset, f, ensure_ascii=False)


def load_subset(path: Path = config.SUBSET_PATH) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Estadísticos y utilidades
# ---------------------------------------------------------------------------
def describe_subset(subset: list[dict[str, Any]]) -> dict[str, Any]:
    """Resumen del subset para incluir en el informe y/o validar la muestra."""
    types = Counter(row.get("type", "unknown") for row in subset)
    levels = Counter(row.get("level", "unknown") for row in subset)
    return {
        "n": len(subset),
        "by_type": dict(types),
        "by_level": dict(levels),
    }


def iter_unique_articles(subset: list[dict[str, Any]]) -> Iterator[dict[str, str]]:
    """Genera artículos únicos (``{title, text}``) presentes en los contextos.

    En HotpotQA cada ``context`` viene como ``{"title": [...], "sentences": [[...]]}``.
    Devolvemos un único registro por título, concatenando sus oraciones.
    """
    seen: set[str] = set()
    for row in subset:
        ctx = row.get("context", {})
        titles = ctx.get("title", [])
        sentences = ctx.get("sentences", [])
        for title, sents in zip(titles, sentences):
            if title in seen:
                continue
            seen.add(title)
            text = " ".join(sents).strip()
            yield {"title": title, "text": text}
