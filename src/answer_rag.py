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
from datasets import load_dataset
from litellm import completion
import chromadb


def main():
    # Carga del dataset (validación) y muestra de 50 preguntas
    muestra = load_dataset("nlp-udesa/hotpot_qa_3k", split="validation").select(range(50))

    # Conexión a la base vectorial construida por bd_vectorial.py
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_collection(name="hotpotqa_corpus")

    print(f"Base vectorial cargada: {collection.count()} documentos")

    TOP_K = 5  # cantidad de párrafos a recuperar por pregunta

    resultados_rag = []

    for fila in muestra:
        # Recuperar los k párrafos más relevantes para la pregunta
        resultados_busqueda = collection.query(
            query_texts=[fila['question']],
            n_results=TOP_K,
            include=["documents"]
        )
        contextos = resultados_busqueda['documents'][0]  # lista de k párrafos

        # Armar el prompt con los contextos recuperados
        contexto_texto = "\n\n".join(
            f"[{i+1}] {ctx}" for i, ctx in enumerate(contextos)
        )
        prompt = (
            f"Usá únicamente los siguientes párrafos para responder la pregunta. "
            f"Dá solo la respuesta, sin explicaciones.\n\n"
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

    # Guardar resultados en JSON para calcular métricas
    with open("resultados_rag.json", "w", encoding="utf-8") as f:
        json.dump(resultados_rag, f, indent=4, ensure_ascii=False)

    print("Listo. Respuestas guardadas en resultados_rag.json")


if __name__ == "__main__":
    main()
