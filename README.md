# TP NLP — Multi-hop QA con RAG y Agentes sobre HotpotQA

Trabajo práctico final de la materia **Procesamiento de Lenguaje Natural** de la Maestría en Ciencia de Datos, Universidad de San Andrés (2026).

Profesores: Juan Manuel Pérez, Bruno Bianchi.
Autor: Pablo Díaz (`pmdiaz@gmail.com`).

## Resumen

El objetivo del trabajo es replicar y comparar distintas técnicas para resolver la tarea de **Multi-hop Question Answering** sobre el dataset [HotpotQA](https://huggingface.co/datasets/hotpotqa/hotpot_qa). Se implementan y evalúan tres enfoques sobre el mismo subconjunto del set de validación:

1. **Baseline closed-book**: un LLM responde sin contexto externo.
2. **RAG**: el LLM responde con los `top-k` contextos recuperados desde una base vectorial.
3. **Agente ReAct**: un agente con ciclo *Thought → Action → Observation* que interactúa con la base vectorial.

Los resultados se reportan en términos de **Exact Match (EM)** y **F1** con la normalización oficial de HotpotQA, y se comparan con los baselines del paper original ([Yang et al., 2018](https://arxiv.org/abs/1809.09600)) y con los resultados de [ReAct](https://arxiv.org/abs/2210.03629).

## Decisiones de diseño

| Item | Decisión |
|---|---|
| LLM | **Gemma 4** (local, vía `transformers`) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (a confirmar en Fase 2) |
| Vector store | **FAISS** vía LangChain |
| Tool del agente | **Base de datos vectorial** (no Wikipedia) |
| Subset de evaluación | **500 preguntas** del split `validation`, estratificadas por `type` y `level`, semilla fija |
| Config del dataset | `distractor` |

## Estructura del proyecto

```
tp-multihop-qa/
├── README.md                  # Este archivo
├── requirements.txt           # Dependencias Python
├── .gitignore
├── src/                       # Código fuente reutilizable
│   ├── config.py              # Constantes (semillas, paths, modelos)
│   ├── data_utils.py          # Carga del dataset y subset estratificado
│   ├── vector_store.py        # (Fase 2) Construcción del índice FAISS
│   ├── llm.py                 # (Fase 3) Wrapper del LLM
│   ├── rag.py                 # (Fase 4) Pipeline RAG
│   ├── agent.py               # (Fase 5) Agente ReAct
│   ├── prompts.py             # Templates de prompts
│   └── evaluation.py          # (Fase 6) Métricas EM / F1 oficiales
├── scripts/                   # Scripts ejecutables por fase
│   └── 01_explorar_dataset.py
├── notebooks/                 # Análisis exploratorios y reporting visual
├── data/                      # Caché del dataset, subset, índice (gitignored)
├── results/                   # Predicciones y métricas por sistema (gitignored)
└── report/                    # Informe final (Markdown / PDF)
```

## Setup

```bash
# 1. Crear entorno virtual
python -m venv venv
source venv/bin/activate

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. (Opcional) Pre-descargar el dataset y construir el subset
python scripts/01_explorar_dataset.py
```

Para usar Gemma 4 desde Hugging Face hace falta aceptar la licencia en el hub y autenticarse:

```bash
huggingface-cli login
```

## Plan de trabajo

> Hoja de ruta detallada elaborada con el cronograma 22 mayo → 10 junio 2026.

### Fase 0 — Lectura crítica de papers (1–2 días)

Antes de tocar código, internalizar tres cosas de los papers de referencia:

- **HotpotQA** (Yang et al. 2018): estructura del dataset (preguntas *bridge* vs *comparison*, niveles *easy/medium/hard*), settings *distractor* vs *fullwiki*, y baselines reportados (BiDAF y variantes) que se usarán como referencia.
- **ReAct** (Yao et al. 2022): ciclo `Thought → Action → Observation`, definición de las acciones (`Search`, `Lookup`, `Finish`) y resultados publicados sobre HotpotQA.
- **Phi-3 / Gemma**: capacidades, tamaño y limitaciones de los LLMs candidatos, para justificar la elección en la metodología.

### Fase 1 — Setup del entorno y exploración del dataset

Cargar `hotpotqa/hotpot_qa` (config `distractor`) con la librería `datasets`. Inspeccionar los campos `question`, `answer`, `context`, `supporting_facts`, `type`, `level`. Construir un **subset reproducible de 500 preguntas** del split `validation`, estratificado por `type` y `level` con semilla fija. Persistir el subset como JSON para reutilizarlo en todas las fases.

### Fase 2 — Construcción de la base de datos vectorial

Extraer todos los contextos únicos (artículos = título + oraciones) del subset elegido. Embebebir con `sentence-transformers/all-MiniLM-L6-v2` e indexar con **FAISS** vía LangChain. Decisiones que documentar en el informe:

- **Granularidad**: artículo completo (fiel a la consigna) vs. *chunking* por oración.
- **Tamaño del corpus**: cantidad de documentos únicos resultantes.
- **Validación cualitativa**: con queries de prueba, verificar que el retrieval recupera los párrafos esperados.

### Fase 3 — Baseline: LLM sin RAG ni agentes (closed-book)

Pipeline donde el LLM recibe **sólo la pregunta**. Sirve como piso del experimento para medir cuánto aporta el retrieval. Misma configuración de decoding (temperatura, max tokens) que los otros dos sistemas para que la comparación sea limpia.

### Fase 4 — Sistema RAG sobre la BD vectorial

Pipeline canónico: embebido de la pregunta → recuperar `top-k` → armar prompt con contextos → generar respuesta. Probar **k = 3, 5, 10** y reportar la curva. Implementación con `langchain.chains.RetrievalQA` o equivalente.

### Fase 5 — Agente estilo ReAct

Agente con ciclo `Thought → Action → Observation`. Herramientas mínimas:

- `search(query)` → top-k del FAISS store.
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

Revisión de reproducibilidad (semillas, `requirements.txt`, instrucciones de uso), validación de que las métricas del informe coincidan con los outputs del código, empaquetado y subida al campus virtual antes del **miércoles 10 de junio de 2026, 23:59 hs**.

## Cronograma

| Fechas | Fase | Entregable parcial |
|---|---|---|
| 22–25 may | Fase 0–1 | Subset de 500 preguntas persistido |
| 26–28 may | Fase 2 | Índice FAISS construido y validado |
| 29–30 may | Fase 3 | Predicciones baseline + métricas |
| 31 may – 2 jun | Fase 4 | Predicciones RAG con barrido de k |
| 3–5 jun | Fase 5 | Predicciones del agente + traces |
| 6–7 jun | Fase 6–7 | CSV consolidado + análisis |
| 8–9 jun | Fase 8 | Informe en draft |
| 10 jun | Fase 9 | Entrega final |

## Referencias

- Yang, Z. et al. (2018). *HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering*. arXiv:1809.09600.
- Yao, S. et al. (2022). *ReAct: Synergizing Reasoning and Acting in Language Models*. arXiv:2210.03629.
- Abdin, M. et al. (2024). *Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone*. arXiv:2404.14219.
