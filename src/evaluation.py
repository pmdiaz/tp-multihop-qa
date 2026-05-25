"""Métricas de evaluación oficiales de HotpotQA.

Implementa la normalización y cómputo de Exact Match (EM) y F1 a nivel de token,
siguiendo el paper original (Yang et al., 2018) y el script oficial de evaluación.

Normalización: lowercase → quitar artículos (a/an/the) → quitar puntuación
               → colapsar espacios → tokenizar por espacios.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def exact_match(pred: str, gold: str) -> int:
    return int(normalize_answer(pred) == normalize_answer(gold))


def token_f1(pred: str, gold: str) -> float:
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    precision = n_common / len(pred_tokens)
    recall = n_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def evaluate(records: list[dict[str, Any]]) -> dict[str, float]:
    """Calcula EM y F1 promedio para una lista de predicciones.

    Cada record debe tener las claves ``gold`` y ``pred``.
    Devuelve también breakdowns por ``type`` y ``level`` si están presentes.
    """
    ems, f1s = [], []
    by_type: dict[str, list] = {}
    by_level: dict[str, list] = {}

    for r in records:
        em = exact_match(r["pred"], r["gold"])
        f1 = token_f1(r["pred"], r["gold"])
        ems.append(em)
        f1s.append(f1)

        for dim, store in [("type", by_type), ("level", by_level)]:
            key = r.get(dim, "unknown")
            if key not in store:
                store[key] = {"em": [], "f1": []}
            store[key]["em"].append(em)
            store[key]["f1"].append(f1)

    result: dict[str, Any] = {
        "n": len(records),
        "em": sum(ems) / len(ems) if ems else 0.0,
        "f1": sum(f1s) / len(f1s) if f1s else 0.0,
        "by_type": {
            k: {"em": sum(v["em"]) / len(v["em"]), "f1": sum(v["f1"]) / len(v["f1"])}
            for k, v in by_type.items()
        },
        "by_level": {
            k: {"em": sum(v["em"]) / len(v["em"]), "f1": sum(v["f1"]) / len(v["f1"])}
            for k, v in by_level.items()
        },
    }
    return result
