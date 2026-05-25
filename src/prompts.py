"""Templates de prompts reutilizados por las tres fases de inferencia.

Todos los prompts siguen el formato de chat de Gemma 4 (lista de mensajes).
La función devuelve una lista de dicts lista para pasar a
``processor.apply_chat_template``.
"""

from __future__ import annotations


def closed_book_messages(question: str) -> list[dict]:
    """Prompt para el baseline sin contexto externo."""
    return [
        {
            "role": "user",
            "content": (
                "Answer the following question with a short, concise answer. "
                "Reply with only the answer — no explanation, no full sentences.\n\n"
                f"Question: {question}\nAnswer:"
            ),
        }
    ]


def rag_messages(question: str, contexts: list[str]) -> list[dict]:
    """Prompt para el sistema RAG con contextos recuperados."""
    context_block = "\n\n".join(
        f"[{i + 1}] {ctx}" for i, ctx in enumerate(contexts)
    )
    return [
        {
            "role": "user",
            "content": (
                "Use the following context passages to answer the question. "
                "Reply with only the answer — no explanation, no full sentences.\n\n"
                f"Context:\n{context_block}\n\n"
                f"Question: {question}\nAnswer:"
            ),
        }
    ]


# ---------------------------------------------------------------------------
# ReAct
# ---------------------------------------------------------------------------

_REACT_SYSTEM = """\
Answer questions by reasoning step by step and using search tools when needed.

You have access to two tools:
  search[query]  — searches a knowledge base and returns relevant passages.
  lookup[term]   — finds sentences containing 'term' inside the last search results.
                   Use this to zoom in on a specific fact without a new search.

Always follow this exact format for each step:
Thought: <your reasoning about what to do next>
Action: search[<query>]  OR  lookup[<term>]  OR  finish[<answer>]
Observation: <result, provided by the system>

Rules:
- Only use search[], lookup[], or finish[] as actions.
- Use lookup[] when you need a specific detail from what you already retrieved.
- Use search[] when you need information not yet in your results.
- The answer in finish[] must be as short as possible (a name, a year, yes/no, etc.).
- Do not repeat a search query you already used.\
"""

_REACT_EXAMPLES = """\
---
Question: What year was the director of Atonement (2007) born?
Thought: I need to find who directed Atonement (2007).
Action: search[Atonement 2007 film director]
Observation: Atonement: A 2007 British romantic war drama film directed by Joe Wright. It stars Keira Knightley and James McAvoy.
Thought: The director is Joe Wright. His birth year might already be in these results.
Action: lookup[born]
Observation: Joe Wright: Joe Wright (born 25 August 1972) is a British film and theatre director.
Thought: Joe Wright was born in 1972.
Action: finish[1972]

---
Question: Were Scott Derrickson and Ed Wood from the same country?
Thought: I need to find the country of origin of Scott Derrickson and Ed Wood separately.
Action: search[Scott Derrickson director nationality]
Observation: Scott Derrickson is an American director and screenwriter, best known for Doctor Strange and The Black Phone.
Thought: Scott Derrickson is American. Now I'll check Ed Wood.
Action: search[Ed Wood filmmaker nationality]
Observation: Ed Wood (October 10, 1924 – December 10, 1978) was an American filmmaker, actor, and author.
Thought: Both are American, so they are from the same country.
Action: finish[yes]
---\
"""


def react_messages(question: str, scratchpad: str) -> list[dict]:
    """Prompt para un paso del agente ReAct.

    ``scratchpad`` contiene el historial acumulado de Thought/Action/Observation
    del episodio actual. Se pasa vacío en el primer paso.
    """
    body = f"{_REACT_SYSTEM}\n\n{_REACT_EXAMPLES}\n\nQuestion: {question}\n{scratchpad}"
    return [{"role": "user", "content": body}]
