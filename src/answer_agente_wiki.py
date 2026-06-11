import json
import os
import time
import wikipedia
from datasets import load_dataset
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from tqdm import tqdm


def main():
    wikipedia.set_lang("en")
    wikipedia.set_user_agent("User-Agent: InfinitumBotty/0.1 (https://github.com/ctx77/InfinitumBotty) python-via-wikipedia-module/1.4.0")

    @tool
    def buscar_en_wikipedia(query: str) -> str:
        """Busca en Wikipedia y devuelve el contenido del artículo más relevante. Llamá esta herramienta una vez por entidad o sub-pregunta."""
        try:
            resultados = wikipedia.search(query, results=3)
            if not resultados:
                return "Sin resultados."
            try:
                pagina = wikipedia.page(resultados[0], auto_suggest=False)
                return pagina.summary[:2000]
            except wikipedia.DisambiguationError as e:
                pagina = wikipedia.page(e.options[0], auto_suggest=False)
                return pagina.summary[:2000]
        except Exception:
            return "No se encontro informacion."

    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)
    tools = [buscar_en_wikipedia]

    system_prompt = (
        "Sos un asistente de QA que responde preguntas con razonamiento multi-paso usando Wikipedia.\n\n"
        "FLUJO OBLIGATORIO para cada pregunta:\n"
        "1. Identificá las entidades y conceptos clave en la pregunta.\n"
        "2. Usá `buscar_en_wikipedia` con una query enfocada en la primera entidad o sub-pregunta.\n"
        "3. Si la respuesta requiere información de más de una entidad (pregunta de múltiples pasos), "
        "repetí el paso 2 con una query distinta para cada entidad antes de responder.\n"
        "4. Una vez que tenés toda la información necesaria, respondé con la respuesta EXACTA y CONCISA: "
        "solo el dato pedido, sin oraciones adicionales, explicaciones no solicitadas ni puntuación extra."
    )

    agente = create_react_agent(llm, tools)

    muestra = load_dataset("nlp-udesa/hotpot_qa_3k", split="validation").shuffle(seed=42).select(range(500))

    output_file = "resultados_agente_wiki_500.json"
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            resultados = json.load(f)
        ya_procesados = {r["id"] for r in resultados}
        print(f"Retomando: {len(ya_procesados)} preguntas ya procesadas")
    else:
        resultados = []
        ya_procesados = set()

    print("Iniciando Agente con Wikipedia...")
    with tqdm(total=len(muestra), desc="Procesando preguntas", initial=len(ya_procesados)) as pbar:
        for fila in muestra:
            if fila['id'] in ya_procesados:
                continue

            pregunta = fila['question']
            respuesta_real = fila['answer']

            try:
                resultado = agente.invoke(
                    {"messages": [("system", system_prompt), ("user", pregunta)]},
                    config={"recursion_limit": 10}
                )
                prediccion = resultado["messages"][-1].content
                mensajes = resultado["messages"]
            except Exception:
                prediccion = ""
                mensajes = []

            queries = []
            for msg in mensajes:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        queries.append(next(iter(tc["args"].values()), ""))

            resultados.append({
                "id": fila['id'],
                "pregunta": pregunta,
                "respuesta_real": respuesta_real,
                "respuesta_modelo": prediccion,
                "type": fila['type'],
                "level": fila['level'],
                "traza": {"num_llamadas": len(queries), "queries": queries}
            })

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(resultados, f, indent=4, ensure_ascii=False)

            time.sleep(4.5)
            pbar.update(1)

    print(f"Respuestas guardadas en {output_file}")


if __name__ == "__main__":
    main()
