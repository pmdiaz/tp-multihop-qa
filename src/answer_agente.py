import json
import os
import time
import chromadb
import torch
from transformers import AutoModel
from datasets import load_dataset
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from tqdm import tqdm

def main():
    # carga del modelo local jinaai para vectorizar las preguntas
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    print(f"Cargando modelo Jina en {device}...")
    model = AutoModel.from_pretrained(
        "jinaai/jina-embeddings-v5-text-nano",
        trust_remote_code=True,
        dtype=torch.bfloat16,
    ).to(device=device)

    # conexión a la bd
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_collection(name="hotpotqa_corpus")
    print(f"Base de datos cargada: {collection.count()} documentos")

    @tool
    def buscar_en_db(query: str) -> str:
        """Busca en la base de datos vectorial local. Usar para cualquier pregunta."""
        # embedding de la consulta
        vector_query = model.encode(
            [query],
            task="retrieval",
            prompt_name="query"
        ).tolist()

        TOP_K = 5
        # "top-k" de 5 párrafos más relevantes
        resultados = collection.query(
            query_embeddings=vector_query,
            n_results=TOP_K
        )

        contextos = []
        if resultados['documents']:
            for doc in resultados['documents'][0]:
                contextos.append(doc)
        return " | ".join(contextos)

    # inicialización del llm
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)
    tools = [buscar_en_db]

    system_prompt = (
        "Sos un asistente de QA que responde preguntas con razonamiento multi-hop usando una base de datos vectorial.\n\n"
        "FLUJO OBLIGATORIO para cada pregunta:\n"
        "1. Identificá las entidades y conceptos clave en la pregunta.\n"
        "2. Usá `buscar_en_db` con una query enfocada en la primera entidad o sub-pregunta.\n"
        "3. Si la respuesta requiere información de más de una entidad (pregunta de múltiples pasos), "
        "repetí el paso 2 con una query distinta para cada entidad o sub-pregunta antes de responder.\n"
        "4. Una vez que tenés toda la información necesaria, respondé con la respuesta EXACTA y CONCISA: "
        "solo el dato pedido, sin oraciones adicionales, explicaciones no solicitadas ni puntuación extra."
    )

    # creación del agente
    agente = create_react_agent(llm, tools)

    muestra = load_dataset("nlp-udesa/hotpot_qa_3k", split="validation").shuffle(seed=42).select(range(500))

    output_file = "resultados_agente_db_500.json"
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            resultados_agente_db = json.load(f)
        ya_procesados = {r["id"] for r in resultados_agente_db}
        print(f"Retomando: {len(ya_procesados)}/{len(muestra)} preguntas ya procesadas")
    else:
        resultados_agente_db = []
        ya_procesados = set()

    print("Iniciando Agente con bd..")
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

            resultados_agente_db.append({
                "id": fila['id'],
                "pregunta": pregunta,
                "respuesta_real": respuesta_real,
                "respuesta_modelo": prediccion,
                "type": fila['type'],
                "level": fila['level'],
                "traza": {"num_llamadas": len(queries), "queries": queries}
            })

            # guardar después de cada pregunta para no perder progreso
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(resultados_agente_db, f, indent=4, ensure_ascii=False)

            time.sleep(4.5)
            pbar.update(1)

    print(f"Respuestas guardadas en {output_file}")


if __name__ == "__main__":
    main()
