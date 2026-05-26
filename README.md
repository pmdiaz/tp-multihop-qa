# TP NLP — Multi-hop QA con RAG y Agentes sobre HotpotQA

Trabajo práctico final de la materia **Procesamiento de Lenguaje Natural** de la Maestría en Ciencia de Datos, Universidad de San Andrés (2026).

Profesores: Juan Manuel Pérez, Bruno Bianchi.
Autor: Pablo Díaz (`pmdiaz@gmail.com`).

## Resumen

El objetivo del trabajo es replicar y comparar distintas técnicas para resolver la tarea de **Multi-hop Question Answering** sobre el dataset [HotpotQA](https://huggingface.co/datasets/nlp-udesa/hotpot_qa_3k) (subset curado por la cátedra). Se implementan y evalúan tres enfoques sobre el mismo subconjunto:

1. **Baseline closed-book**: un LLM responde sin contexto externo.
2. **RAG**: el LLM responde con los `top-k` contextos recuperados desde una base vectorial.
3. **Agente ReAct**: un agente con ciclo *Thought → Action → Observation* que interactúa con la base vectorial.

Los resultados se reportan en términos de **Exact Match (EM)** y **F1** con la normalización oficial de HotpotQA, y se comparan con los baselines del paper original ([Yang et al., 2018](https://arxiv.org/abs/1809.09600)) y con los resultados de [ReAct](https://arxiv.org/abs/2210.03629).

## Decisiones de diseño

| Item | Decisión |
|---|---|
| Dataset | [`nlp-udesa/hotpot_qa_3k`](https://huggingface.co/datasets/nlp-udesa/hotpot_qa_3k) (subset curado por la cátedra) |
| LLM | `gemini/gemini-3.1-flash-lite` vía **LiteLLM** |
| Embeddings | `all-MiniLM-L6-v2` (default de ChromaDB) |
| Vector store | **ChromaDB** persistente en `./chroma_db` |
| Tool del agente | **Base de datos vectorial** (no Wikipedia) |
| Subset de evaluación | **500 preguntas** del split `validation`, estratificadas por `type` y `level`, semilla fija |
| Gestor de paquetes | **uv** |

## Estructura del proyecto

```
tp-multihop-qa/
├── README.md
├── pyproject.toml             # Dependencias y entry points (uv)
├── uv.lock
├── .gitignore
├── src/
│   ├── bd_vectorial.py        # Construye la base vectorial       →  uv run bd_vectorial
│   ├── baseline_llm_sin_bd.py # LLM sin RAG (baseline)           →  uv run baseline
│   ├── baseline_metricas.py   # EM / F1 del baseline             →  uv run metricas
│   ├── answer_rag.py          # Pipeline RAG                     →  uv run rag
│   ├── rag_metricas.py        # EM / F1 del RAG                  →  uv run rag_metricas
│   └── hotpot_qa/             # Paquete de soporte
│       ├── config.py          # Constantes globales
│       └── data_utils.py      # Carga y subset estratificado
├── notebooks/                 # Análisis exploratorios
├── data/                      # Caché del dataset (gitignored)
├── chroma_db/                 # Base de datos ChromaDB (gitignored)
├── results/                   # Predicciones y métricas (gitignored)
└── report/                    # Informe final
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

### 3. Configurar la API key del LLM

El proyecto usa Gemini a través de LiteLLM. Obtené tu clave en [Google AI Studio](https://ai.google.dev/gemini-api/docs/api-key):

```bash
export GEMINI_API_KEY="tu_api_key"
```

### 4. Construir la base vectorial

```bash
uv run bd_vectorial
```

Descarga `nlp-udesa/hotpot_qa_3k`, extrae todos los párrafos únicos y los indexa en ChromaDB (`./chroma_db`). Solo hace falta correrlo una vez.

### 5. Correr los sistemas y calcular métricas

```bash
# Baseline: LLM sin contexto externo
uv run baseline        # genera resultados_baseline.json
uv run metricas        # calcula EM y F1 sobre resultados_baseline.json

# RAG: recupera top-k párrafos antes de responder
uv run rag             # genera resultados_rag.json
uv run rag_metricas    # calcula EM y F1 sobre resultados_rag.json
```

## Plan de trabajo

> Hoja de ruta detallada elaborada con el cronograma 22 mayo → 10 junio 2026.

### Fase 0 — Lectura crítica de papers (1–2 días)

Antes de tocar código, internalizar tres cosas de los papers de referencia:

- **HotpotQA** (Yang et al. 2018): estructura del dataset (preguntas *bridge* vs *comparison*, niveles *easy/medium/hard*), settings *distractor* vs *fullwiki*, y baselines reportados (BiDAF y variantes) que se usarán como referencia.
- **ReAct** (Yao et al. 2022): ciclo `Thought → Action → Observation`, definición de las acciones (`Search`, `Lookup`, `Finish`) y resultados publicados sobre HotpotQA.
- **Phi-3 / Gemma**: capacidades, tamaño y limitaciones de los LLMs candidatos, para justificar la elección en la metodología.

### Fase 1 — Setup del entorno y exploración del dataset

Cargar [`nlp-udesa/hotpot_qa_3k`](https://huggingface.co/datasets/nlp-udesa/hotpot_qa_3k) con la librería `datasets`. Inspeccionar los campos `question`, `answer`, `context`, `supporting_facts`, `type`, `level`. Construir un **subset reproducible de 500 preguntas** del split `validation`, estratificado por `type` y `level` con semilla fija. Persistir el subset como JSON para reutilizarlo en todas las fases.

### Fase 2 — Construcción de la base vectorial (`uv run bd_vectorial`)

Extraer todos los contextos únicos (título + oraciones) de todas las particiones del dataset e indexar en **ChromaDB** (`./chroma_db`). ChromaDB genera los embeddings automáticamente con `all-MiniLM-L6-v2`. Decisiones que documentar en el informe:

- **Granularidad**: artículo completo (título + todas las oraciones concatenadas).
- **Tamaño del corpus**: cantidad de pasajes únicos resultantes.
- **Validación cualitativa**: verificar con queries de prueba que el retrieval recupera los párrafos esperados.

### Fase 3 — Baseline closed-book (`uv run baseline`)

El LLM recibe **sólo la pregunta**, sin contexto externo. Sirve como piso del experimento para medir cuánto aporta el retrieval. Las respuestas se guardan en `resultados_baseline.json`. Métricas con `uv run metricas`.

### Fase 4 — Sistema RAG (`uv run rag`)

Pipeline canónico: pregunta → recuperar `top-k` desde ChromaDB → armar prompt con contextos → generar respuesta vía LiteLLM. El valor de `TOP_K` se ajusta directamente en `src/answer_rag.py`. Las respuestas se guardan en `resultados_rag.json`. Métricas con `uv run rag_metricas`.

### Fase 5 — Agente ReAct

Agente con ciclo `Thought → Action → Observation`. Herramientas mínimas:

- `search(query)` → top-k desde ChromaDB.
- `finish(answer)` → termina la ejecución.

Parámetros: `max_steps = 5–7`, prompt few-shot con ejemplos al estilo ReAct, captura del *reasoning trace* para análisis cualitativo posterior.

### Fase 6 — Evaluación con métricas oficiales

Implementación de **Exact Match** y **F1** con la normalización del paper de HotpotQA (lower, sin artículos `a/an/the`, sin puntuación, sin espacios extra). Corrida de los tres sistemas sobre el **mismo subset** de 500 preguntas. Persistencia en CSV con columnas `(qid, question, gold, pred_baseline, pred_rag, pred_agent, em_*, f1_*, type, level)`.

### Fase 7 — Análisis de resultados y comparación con el paper

- Tabla comparativa con EM/F1 de los tres enfoques + cifras del paper de HotpotQA (BiDAF) y de ReAct.
- Breakdown por `type` (bridge vs comparison) y `level` (easy/medium/hard).
- Análisis de errores: 5–10 casos donde RAG falla pero el agente acierta, y al revés.
- Costo computacional: cantidad promedio de llamadas al LLM por pregunta.

### Fase 8 — Redacción del informe (3–5 páginas)

Secciones: **Introducción** (½ pág) · **Metodología** (~1½ pág) · **Resultados** (~2 pág) · **Conclusiones** (½ pág). Formato Markdown o PDF. Referencias a los tres papers.

### Fase 9 — Verificación final y entrega

Revisión de reproducibilidad (semillas, `pyproject.toml`, instrucciones de uso), validación de que las métricas del informe coincidan con los outputs del código, empaquetado y subida al campus virtual antes del **miércoles 10 de junio de 2026, 23:59 hs**.

## Cronograma

| Fechas | Fase | Entregable parcial |
|---|---|---|
| 22–25 may | Fase 0–1 | Subset de 500 preguntas persistido |
| 26–28 may | Fase 2 | Base vectorial ChromaDB construida y validada |
| 29–30 may | Fase 3 | Predicciones baseline + métricas |
| 31 may – 2 jun | Fase 4 | Predicciones RAG con barrido de k |
| 3–5 jun | Fase 5 | Predicciones del agente + traces |
| 6–7 jun | Fase 6–7 | CSV consolidado + análisis |
| 8–9 jun | Fase 8 | Informe en draft |
| 10 jun | Fase 9 | Entrega final |

## Dependencias principales

| Librería | Rol |
|---|---|
| `chromadb` | Base de datos vectorial (embeddings automáticos con `all-MiniLM-L6-v2`) |
| `litellm` | Interfaz unificada para LLMs (Gemini, OpenAI, Anthropic, etc.) |
| `datasets` | Carga de `nlp-udesa/hotpot_qa_3k` |
| `langchain` | Utilidades de cadenas y agentes |

Gestionadas con **uv** (`pyproject.toml` + `uv.lock`). No se usa `requirements.txt`.

## Referencias

- Yang, Z. et al. (2018). *HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering*. arXiv:1809.09600.
- Yao, S. et al. (2022). *ReAct: Synergizing Reasoning and Acting in Language Models*. arXiv:2210.03629.
- Abdin, M. et al. (2024). *Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone*. arXiv:2404.14219.
