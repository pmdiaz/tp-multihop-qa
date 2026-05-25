from datasets import load_dataset
import chromadb

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

# Iteración sobre todas las particiones que tenga el dataset (ej. 'train', 'validation')
for particion in dataset.keys():
    extraer_contextos(particion)

# Conversión a lista final
corpus = list(corpus_unico)

print(f"Documentos únicos extraídos: {len(corpus)}") #chequar tamaño
print("Ejemplo:", corpus[0]) #chequear un ejemplo


# Creación de carpeta llamada 'chroma_db' para guardar la base de datos
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Creación de una colección 
nombre_coleccion = "hotpotqa_corpus"

# Eliminación de colección de una ejecución previa para evitar duplicados
if nombre_coleccion in [c.name for c in chroma_client.list_collections()]:
    chroma_client.delete_collection(name=nombre_coleccion)

collection = chroma_client.create_collection(name=nombre_coleccion)

# ID de cada documento
ids = [str(i) for i in range(len(corpus))]

# Batches y embeddings
batch_size = 5000 # batches de 5000 elementos pero se puede cambiar
for i in range(0, len(corpus), batch_size):
    batch_docs = corpus[i:i + batch_size]
    batch_ids = ids[i:i + batch_size]
    
    collection.add( #al hacer esto, chromaDB forma los embeddings con el modelo all-MiniLM-L6-v2
        documents=batch_docs,
        ids=batch_ids
    )