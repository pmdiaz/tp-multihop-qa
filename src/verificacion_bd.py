import chromadb

# 1. Conectarse a la base de datos existente
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# 2. Acceder a la colección que creaste
collection = chroma_client.get_collection(name="hotpotqa_corpus")

# 3. Verificar la cantidad total de documentos indexados
print(f"Cantidad de elementos en la colección: {collection.count()}")

# 4. Traer los primeros 2 elementos para inspeccionar la estructura interna
primeros_elementos = collection.peek(limit=2)
print("\nEstructura de los documentos guardados:")
for i, doc in enumerate(primeros_elementos['documents']):
    print(f"\nDocumento {i+1}:")
    print(f"ID: {primeros_elementos['ids'][i]}")
    print(f"Texto: {doc[:200]}...")  # Muestra solo los primeros 200 caracteres