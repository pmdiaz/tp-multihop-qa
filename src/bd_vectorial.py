from datasets import load_dataset
import chromadb
from transformers import AutoModel
import torch


def main():
    # Carga del dataset reducido
    dataset = load_dataset("nlp-udesa/hotpot_qa_3k")

    corpus_unico = set() # se hace set para no tener repetidos

    # Función para extraer párrafos
    def extraer_contextos(split_name):
        for fila in dataset[split_name]:
            titulos = fila['context']['title']
            oraciones_por_titulo = fila['context']['sentences']


            for titulo, oraciones in zip(titulos, oraciones_por_titulo):
                parrafo = " ".join(oraciones)
                documento = f"{titulo}: {parrafo}"
                corpus_unico.add(documento)


    # Iteración sobre todas las particiones que tenga el dataset (se deja por las dudas)
    for split in dataset.keys():
        extraer_contextos(split)

    # Conversión a lista final
    corpus = list(corpus_unico)

    print(f"Documentos únicos extraídos: {len(corpus)}") #chequear tamaño
    print("Ejemplo:", corpus[0]) #chequear un ejemplo

    # Para embeddings, se usa el modelo jinaai de HF
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
   
    # Creación de carpeta llamada 'chroma_db' para guardar la base de datos
    chroma_client = chromadb.PersistentClient(path="./chroma_db")

    # Creación de una colección
    nombre_coleccion = "hotpotqa_corpus"

    collection = chroma_client.get_or_create_collection(name=nombre_coleccion)

    # ID de cada documento, se debe agregar a la db para que funcione
    ids = [str(i) for i in range(len(corpus))]

    # Batches y embeddings
    batch_size = 32  # batches chicos para no saturar la memoria de MPS/GPU
    for i in range(0, len(corpus), batch_size):
        batch_docs = corpus[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]

        embeddings = model.encode(batch_docs, task="retrieval")

        collection.add(
            documents=batch_docs,
            ids=batch_ids,
            embeddings=embeddings.tolist()
        )


    print(f"Lista la BD")


if __name__ == "__main__":
    main()

