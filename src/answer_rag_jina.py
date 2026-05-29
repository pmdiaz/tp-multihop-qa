# Responde preguntas usando RAG (Retrieval-Augmented Generation).
# Recupera los top-k párrafos más relevantes de la base vectorial
# construida en bd_vectorial.py y los usa como contexto para el LLM.
# Se guardan las respuestas en un JSON para calcular métricas luego.

# Para usar Gemini:
# 1 Entrar a https://aistudio.google.com/api-keys
# 2 Create "API key" y copiar.
# 3 En Terminal: export GEMINI_API_KEY="api_key_aca"

import json
import time
import torch
from transformers import AutoModel
from datasets import load_dataset
from litellm import completion
import chromadb

def main():
    # Carga del dataset (validación) y muestra de 50 preguntas
    muestra = load_dataset("nlp-udesa/hotpot_qa_3k", split="validation").select(range(50))

    # Carga del modelo jinaai para los embeddings igual que en la bd
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Cargando modelo jina en {device}...")
    model = AutoModel.from_pretrained(
        "jinaai/jina-embeddings-v5-text-nano",
        trust_remote_code=True,
        dtype=torch.bfloat16,
    ).to(device=device)

    # Conexión a la bd
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_collection(name="hotpotqa_corpus")

    print(f"Base vectorial cargada: {collection.count()} documentos")

    TOP_K = 5  # cantidad de párrafos a recuperar por pregunta
    resultados_rag = []

    for fila in muestra:
        # Vectorizo la pregunta usando jinaai con la task correspondiente
        vector_query = model.encode(
            [fila['question']], 
            task="retrieval", 
            prompt_name="query"
        ).tolist()

        # recupero los k párrafos más relevantes enviando los embeddings
        resultados_busqueda = collection.query(
            query_embeddings=vector_query,
            n_results=TOP_K,
            include=["documents"]
        )
        contextos = resultados_busqueda['documents'][0]

        # se arma el prompt con los contextos recuperados para que conteste
        contexto_texto = "\n\n".join(
            f"[{i+1}] {ctx}" for i, ctx in enumerate(contextos)
        )
        prompt = (
            f"Usá únicamente los siguientes párrafos para responder la pregunta. " #usa los contextos de la bd
            f"Dá solo la respuesta, sin explicaciones.\n\n" #menos texto
            f"Contexto:\n{contexto_texto}\n\n"
            f"Pregunta: {fila['question']}\n"
            f"Respuesta:"
        )

        response = completion(
            model="gemini/gemini-3.1-flash-lite",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        resultados_rag.append({
            "id": fila['id'],
            "pregunta": fila['question'],
            "respuesta_real": fila['answer'],
            "respuesta_modelo": response.choices[0].message.content.strip(),
            "contextos_recuperados": contextos
        })

        time.sleep(4.5)  # esperar para no superar el límite de 15 requests per minute

    # se guardan resultados en JSON para calcular métricas
    with open("resultados_rag.json", "w", encoding="utf-8") as f:
        json.dump(resultados_rag, f, indent=4, ensure_ascii=False)

    print("Listo. Respuestas guardadas en resultados_rag.json")

if __name__ == "__main__":
    main()