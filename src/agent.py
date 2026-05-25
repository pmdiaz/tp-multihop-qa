"""Agente estilo ReAct para Multi-hop QA.

Ciclo: Thought → Action → Observation (repetido hasta finish o max_steps).

Herramientas disponibles:
- search[query]   → top-k documentos del índice FAISS (búsqueda semántica).
- lookup[term]    → busca oraciones que contienen `term` en los documentos del
                    último search[]. No toca FAISS — opera en memoria.
- finish[answer]  → termina el episodio y devuelve la respuesta.

El scratchpad (historial del episodio) se construye como texto plano y se
incluye en el prompt en cada paso. Esto es más robusto con LLMs locales que
un esquema multi-turn de chat.
"""

from __future__ import annotations

import re
from typing import Any


from . import config
from .llm import generate
from .prompts import react_messages

# Cuántos caracteres de cada documento incluir en la Observation
_OBS_CHARS_PER_DOC = 400


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_action(text: str) -> tuple[str, str]:
    """Extrae (action_type, action_input) del texto generado.

    Acepta variantes de capitalización y espacios.
    Devuelve ("unknown", "") si no puede parsear.
    """
    m = re.search(r"Action:\s*(search|lookup|finish)\[([^\]]*)\]", text, re.IGNORECASE)
    if m:
        return m.group(1).lower(), m.group(2).strip()
    # Fallback: buscar finish sin corchetes al final de la línea
    m = re.search(r"Action:\s*finish[:\s]+(.+)", text, re.IGNORECASE)
    if m:
        return "finish", m.group(1).strip()
    return "unknown", ""


def _extract_first_step(text: str) -> str:
    """Recorta el texto hasta la primera línea Action: (inclusive).

    Evita que el modelo "alucine" múltiples pasos en una sola generación.
    """
    lines = text.splitlines()
    result: list[str] = []
    for line in lines:
        result.append(line)
        if re.match(r"\s*Action:", line, re.IGNORECASE):
            break
    return "\n".join(result)


# ---------------------------------------------------------------------------
# Ejecución de herramientas
# ---------------------------------------------------------------------------

def _retrieve(retriever_or_vs: Any, query: str, k: int) -> list:
    """Ejecuta una query contra FAISS o cualquier retriever LangChain."""
    if hasattr(retriever_or_vs, "invoke"):
        return retriever_or_vs.invoke(query)
    return retriever_or_vs.similarity_search(query, k=k)


def _execute_search(
    retriever_or_vs: Any,
    queries: list[str],
    k: int,
) -> tuple[str, list]:
    """Recupera documentos para una o varias queries y devuelve (obs, docs).

    Une los resultados de todas las queries deduplicando por título.
    Soporta tanto FAISS (similarity_search) como retrievers LangChain (invoke).
    """
    seen_titles: set[str] = set()
    all_docs: list = []

    for query in queries:
        for doc in _retrieve(retriever_or_vs, query, k):
            title = doc.metadata.get("title", "")
            if title not in seen_titles:
                seen_titles.add(title)
                all_docs.append(doc)

    result_docs = all_docs[:k]
    snippets = [
        f"{doc.metadata.get('title', '')}: {doc.page_content.replace(chr(10), ' ')[:_OBS_CHARS_PER_DOC]}"
        for doc in result_docs
    ]
    return " | ".join(snippets), result_docs


def _execute_lookup(last_docs: list, term: str) -> str:
    """Busca oraciones que contienen `term` en los docs del último search[].

    Divide cada documento en oraciones y devuelve las que hacen match.
    No consulta FAISS — opera completamente en memoria.
    """
    term_lower = term.lower()
    matches = []
    for doc in last_docs:
        title = doc.metadata.get("title", "")
        sentences = re.split(r"(?<=[.!?])\s+", doc.page_content.replace("\n", " "))
        for sent in sentences:
            if term_lower in sent.lower():
                matches.append(f"{title}: {sent.strip()}")
    if matches:
        return " | ".join(matches[:4])
    return f"No sentences containing '{term}' in last search results. Try search[] with a different query."


# ---------------------------------------------------------------------------
# Loop principal
# ---------------------------------------------------------------------------

def run_agent(
    model: Any,
    tokenizer: Any,
    retriever_or_vs: Any,
    question: str,
    max_steps: int = config.AGENT_MAX_STEPS,
    k: int = config.DEFAULT_TOP_K,
    multi_query: bool = False,
) -> dict[str, Any]:
    """Ejecuta el loop ReAct y devuelve un dict con la respuesta y el trace.

    Campos del resultado:
    - ``answer``: respuesta final (str)
    - ``trace``: lista de pasos, cada uno con thought_action y observation
    - ``steps``: número de pasos ejecutados
    - ``finished``: True si el agente llamó a finish[], False si se agotó max_steps

    ``retriever_or_vs`` acepta FAISS directo o cualquier retriever LangChain
    (EnsembleRetriever, etc.). ``multi_query=True`` agrega la pregunta original
    como segunda query en cada búsqueda, uniendo resultados por título.
    """
    scratchpad = ""
    trace: list[dict[str, str]] = []
    seen_queries: set[str] = set()
    last_docs: list = []  # documentos del último search[], usados por lookup[]

    for step in range(max_steps):
        messages = react_messages(question, scratchpad)
        raw = generate(model, tokenizer, messages, max_new_tokens=200)
        step_text = _extract_first_step(raw)

        action_type, action_input = _parse_action(step_text)
        trace.append({"step": step + 1, "action_type": action_type, "thought_action": step_text})

        if action_type == "finish":
            scratchpad += step_text
            return {
                "answer": action_input,
                "trace": trace,
                "steps": step + 1,
                "finished": True,
            }

        if action_type == "search":
            query = action_input
            if query in seen_queries:
                observation = "Already searched this query. Try a different query."
            else:
                seen_queries.add(query)
                queries = [query]
                if multi_query and query.lower() != question.lower():
                    queries.append(question)
                observation, last_docs = _execute_search(retriever_or_vs, queries, k)
            trace[-1]["observation"] = observation
            scratchpad += f"{step_text}\nObservation: {observation}\n"

        elif action_type == "lookup":
            if not last_docs:
                observation = "No previous search results to look up. Use search[] first."
            else:
                observation = _execute_lookup(last_docs, action_input)
            trace[-1]["observation"] = observation
            scratchpad += f"{step_text}\nObservation: {observation}\n"

        else:
            # No se pudo parsear la acción — forzar finish
            observation = "Could not parse action. Stopping."
            trace[-1]["observation"] = observation
            break

    # Fallback: el agente no llamó a finish en max_steps
    # Intentar extraer la respuesta del último paso usando el LLM
    fallback_messages = [
        {
            "role": "user",
            "content": (
                f"Based on the following research, answer the question with only "
                f"a short answer (a name, date, yes/no, etc.).\n\n"
                f"Question: {question}\n\n"
                f"Research:\n{scratchpad}\n\nAnswer:"
            ),
        }
    ]
    answer = generate(model, tokenizer, fallback_messages, max_new_tokens=64)
    return {
        "answer": answer.strip(),
        "trace": trace,
        "steps": max_steps,
        "finished": False,
    }
