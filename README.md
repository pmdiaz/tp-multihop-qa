# TP NLP — Multi-hop QA con RAG y Agentes sobre HotpotQA

Trabajo práctico final de la materia **Procesamiento de Lenguaje Natural** de la Maestría en Ciencia de Datos, Universidad de San Andrés (2026).

Profesores: Juan Manuel Pérez, Bruno Bianchi.
Autores: Pablo Díaz (`pmdiaz@gmail.com`), Ezequiel Martinez (`martineze85@gmail.com`).

## Resumen

El objetivo es replicar y comparar distintas técnicas para resolver la tarea de **Multi-hop Question Answering** sobre el dataset [HotpotQA](https://huggingface.co/datasets/nlp-udesa/hotpot_qa_3k) (subset curado y provisto por la cátedra). Se implementan y evalúan cuatro enfoques sobre el mismo subconjunto de 150 preguntas:

1. **Baseline closed-book**: el LLM responde sin contexto externo.
2. **RAG**: el LLM responde con los `top-k` contextos recuperados desde una base vectorial (ChromaDB).
3. **Agente ReAct + ChromaDB**: agente que realiza los procesos *Thought → Action → Observation*, en ciclos, que consulta la base vectorial.
4. **Agente ReAct + Wikipedia**: misma arquitectura de agente, pero con acceso a Wikipedia en tiempo real.

Los resultados se reportan en términos de **Exact Match (EM)** y **F1** con la normalización oficial de HotpotQA, con desglose (breakdown) por tipo (`type`: bridge / comparison) y  nivel (`level`: easy / medium / hard).

## Decisiones de diseño

| Item | Decisión |
|---|---|
| Dataset | [`nlp-udesa/hotpot_qa_3k`](https://huggingface.co/datasets/nlp-udesa/hotpot_qa_3k) |
| LLM | `gemini-2.5-flash-lite` vía **LiteLLM** (baseline / RAG) y **LangChain** (agentes) |
| Embeddings | `jinaai/jina-embeddings-v5-text-nano` vía `transformers` |
| Vector store | **ChromaDB** persistente en `./chroma_db` |
| Subset de evaluación | **150 preguntas** del split `validation`, `seed=42` |
| Gestor de paquetes | **uv** (`pyproject.toml` + `uv.lock`) |

## Estructura del proyecto

```
tp-multihop-qa/
├── README.md
├── pyproject.toml               # Dependencias y entry points (uv)
├── uv.lock
├── .gitignore
├── src/
│   ├── bd_vectorial.py          # construye ChromaDB          →  uv run bd_vectorial
│   ├── baseline_llm_sin_bd.py   # LLM sin contexto            →  uv run baseline
│   ├── answer_rag.py            # pipeline RAG                →  uv run rag
│   ├── answer_agente.py         # agente ReAct + ChromaDB    →  uv run agente
│   ├── answer_agente_wiki.py    # agente ReAct + Wikipedia   →  uv run agente_wiki
│   ├── metricas.py              # EM / F1 con desglose       →  uv run metricas <archivo>
│   └── hotpot_qa/               # Paquete de soporte
│       ├── config.py            # Constantes globales (de ser necesarios)
│       └── data_utils.py        # Carga (de ser necesarios) 
├── chroma_db/                   # Base de datos ChromaDB (gitignored)

```

## Setup

### 1. Instalar uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Instalar dependencias

```bash
uv sync
```

### 3. Configurar la API key de Gemini

Obtener clave en [Google AI Studio](https://aistudio.google.com/api-keys):

```bash
export GEMINI_API_KEY="tu_api_key"
```

### 4. Construir la base vectorial

```bash
uv run bd_vectorial
```

Descarga `nlp-udesa/hotpot_qa_3k`, extrae todos los párrafos únicos, genera embeddings con `jina-embeddings-v5-text-nano` y los indexa en ChromaDB (`./chroma_db`). Solo hace falta correrlo una vez.

## Correr los distintos enfoques/modelos

Los cuatro enfoques/modelos deben correrse sobre el mismo subset (150 preguntas, `seed=42`). Correrlos secuencialmente para no superar el límite de 15 RPM de Gemini.

```bash
# Fase 3 — Baseline: LLM sin contexto externo
uv run baseline
# genera: resultados_baseline.json

# Fase 4 — RAG: recupera top-5 párrafos desde ChromaDB antes de responder
uv run rag
# genera: resultados_rag.json

# Fase 5a — Agente ReAct con ChromaDB como herramienta
uv run agente
# genera: resultados_agente_db.json

# Fase 5b — Agente ReAct con Wikipedia como herramienta
uv run agente_wiki
# genera: resultados_agente_wiki.json
```

Todos los scripts tienen **checkpoint automático**: si se interrumpen, al volver a correr retoman desde la última pregunta guardada.

## Calcular métricas

```bash
uv run metricas resultados_baseline.json
uv run metricas resultados_rag.json
uv run metricas resultados_agente_db.json
uv run metricas resultados_agente_wiki.json
```

El script calcula EM y F1 con la normalización oficial de HotpotQA (minúsculas, sin puntuación, sin artículos a/an/the) y muestra desglose por tipo y nivel. Para los archivos de agentes, también muestra estadísticas de uso de herramientas (promedio de llamadas por pregunta).

Ejemplo de salida:

```
Total preguntas: 150
EM: 23.33%   F1: 35.81%

Por tipo:
  bridge       (102)   EM: 21.57%   F1: 33.40%
  comparison    (48)   EM: 27.08%   F1: 40.62%

Por nivel:
  easy          (55)   EM: 30.91%   F1: 44.20%
  medium        (72)   EM: 19.44%   F1: 30.15%
  hard          (23)   EM: 13.04%   F1: 22.60%

Uso de herramientas (150 preguntas con traza):
  Promedio llamadas/pregunta: 1.84
  Sin llamadas (0):           12 (8.0%)
  1 llamada:    67 preguntas (44.7%)
  2 llamadas:   55 preguntas (36.7%)
  3 llamadas:   16 preguntas (10.7%)
```

## Dependencias principales

| Librería | Rol |
|---|---|
| `chromadb` | Base de datos vectorial |
| `transformers` | Modelo de embeddings Jina |
| `litellm` | Interfaz unificada para LLMs (baseline y RAG) |
| `langchain` / `langgraph` | Framework para agentes ReAct |
| `datasets` | Carga de `nlp-udesa/hotpot_qa_3k` |
| `wikipedia` | Acceso a Wikipedia en tiempo real (agente_wiki) |

Gestionadas con **uv** (`pyproject.toml` + `uv.lock`). No se usa `requirements.txt`.

## Referencias

- Yang, Z. et al. (2018). *HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering*. arXiv:1809.09600.
- Yao, S. et al. (2022). *ReAct: Synergizing Reasoning and Acting in Language Models*. arXiv:2210.03629.
- Abdin, M. et al. (2024). *Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone*. arXiv:2404.14219.
