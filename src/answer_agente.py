# Responde preguntas usando un agente. Recupera los top-k párrafos más relevantes de la base 
# vectorial construida en bd_vectorial.py y los usa como contexto para el LLM.
# Se guardan las respuestas en un JSON para calcular métricas luego.

# Para usar Gemini:
# 1 Entrar a https://aistudio.google.com/api-keys
# 2 Create "API key" y copiar.
# 3 En Terminal: export GEMINI_API_KEY="api_key_aca"

import json
import time
import chromadb
import torch
from transformers import AutoModel
from datasets import load_dataset
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

# carga del modelo local jinaai para vectorizar las preguntas
device = "cuda" if torch.cuda.is_available() else "cpu"
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

def main():
    # inicialización del llm
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)
    tools = [buscar_en_db]

    system_prompt = "Sos un asistente de QA. Usá la herramienta provista para buscar el contexto y responder la pregunta de forma MUY CORTA Y CONCISA. Devolvé SOLO la respuesta exacta, sin explicaciones ni oraciones extra."

    # creación del agente  
    agente = create_react_agent(llm, tools)

    # carga del dataset (validación) y muestra de 50 preguntas
    muestra = load_dataset("nlp-udesa/hotpot_qa_3k", split="validation").shuffle(seed=42).select(range(150))
    
    resultados_agente_db = []
    
    print("Iniciando evaluación del Agente con bd..")

    for fila in muestra:
        pregunta = fila['question']
        respuesta_real = fila['answer']
        
        resultado = agente.invoke({"messages": [("system", system_prompt), ("user", pregunta)]})
        prediccion = resultado["messages"][-1].content
            
        resultados_agente_db.append({
            "id": fila['id'],
            "pregunta": pregunta,
            "respuesta_real": respuesta_real,
            "respuesta_modelo": prediccion
        })
        
        time.sleep(4.5)  # esperar para no superar el límite de 15 requests per minute

    # Guardar resultados en JSON para calcular métricas
    with open("resultados_agente_db.json", "w", encoding="utf-8") as f:
        json.dump(resultados_agente_db, f, indent=4, ensure_ascii=False)

    print("Listo. Respuestas guardadas en resultados_agente_db.json")

if __name__ == "__main__":
    main()