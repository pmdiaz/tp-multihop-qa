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
import torch
from transformers import AutoModel
from tqdm import tqdm

def main():
    # Mismo modelo de embeddings que usa bd_vectorial.py para la ingesta
    model_name = "jinaai/jina-embeddings-v5-text-nano"
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        dtype=torch.bfloat16,
    ).to(device=device)

    # Carga del dataset (validación), misma muestra que el baseline
    muestra = load_dataset("nlp-udesa/hotpot_qa_3k", split="validation").shuffle(seed=42).select(range(150))

    # Conexión a la base vectorial construida por bd_vectorial.py
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_collection(name="hotpotqa_corpus")

    print(f"Base vectorial cargada: {collection.count()} documentos")

    TOP_K = 5  # cantidad de párrafos a recuperar por pregunta

    resultados_rag = []
    with tqdm(total=len(muestra), desc="Procesando preguntas") as pbar:
        for fila in muestra:
            # Encodear la pregunta con Jina (mismo espacio vectorial que la ingesta)
            q_embedding = model.encode([fila['question']], task="retrieval")

            # Recuperar los k párrafos más relevantes para la pregunta
            resultados_busqueda = collection.query(
                query_embeddings=q_embedding.tolist(),
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
                "contextos_recuperados": contextos,
                "type": fila['type'],
                "level": fila['level']
            })

            time.sleep(4.5)  # esperar para no superar el límite de 15 requests per minute
            pbar.update(1)

    # Guardar resultados en JSON para calcular métricas
    with open("resultados_rag.json", "w", encoding="utf-8") as f:
        json.dump(resultados_rag, f, indent=4, ensure_ascii=False)

    print("Listo. Respuestas guardadas en resultados_rag.json")


if __name__ == "__main__":
    main()
