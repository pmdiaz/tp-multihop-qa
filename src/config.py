"""Configuración global del proyecto.

Constantes reutilizadas por todas las fases: paths, semillas, nombres de modelos,
tamaños de muestra. Modificar acá para mantener consistencia en todo el pipeline.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = ROOT_DIR / "data"
RESULTS_DIR: Path = ROOT_DIR / "results"
REPORT_DIR: Path = ROOT_DIR / "report"

# Subset reproducible para evaluación
SUBSET_PATH: Path = DATA_DIR / "validation_subset_500.json"

# Índice vectorial
VECTOR_STORE_DIR: Path = DATA_DIR / "faiss_index"

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
DATASET_NAME: str = "hotpotqa/hotpot_qa"
DATASET_CONFIG: str = "distractor"
DATASET_SPLIT: str = "validation"

# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
SEED: int = 42
N_SAMPLES: int = 500

# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------
# Variante de Gemma 4. Cambiar a 4b o superior si el hardware lo permite.
LLM_MODEL: str = "google/gemma-4-2b-it"

# Modelo de embeddings (sentence-transformers)
EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Generación
# ---------------------------------------------------------------------------
MAX_NEW_TOKENS: int = 256
TEMPERATURE: float = 0.0  # respuestas determinísticas para reproducibilidad

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
TOP_K_VALUES: tuple[int, ...] = (3, 5, 10)
DEFAULT_TOP_K: int = 5

# ---------------------------------------------------------------------------
# Agente
# ---------------------------------------------------------------------------
AGENT_MAX_STEPS: int = 7


def ensure_dirs() -> None:
    """Crea las carpetas de trabajo si no existen."""
    for d in (DATA_DIR, RESULTS_DIR, REPORT_DIR, VECTOR_STORE_DIR):
        d.mkdir(parents=True, exist_ok=True)
