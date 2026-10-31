from rag.retriever import load_documents
from rag.chunker import chunk_documents
from rag.embeddings import create_embeddings
from rag.faiss_store import create_faiss_index, search_faiss


# ============================================================
# 1. Load documents
# ============================================================

documents = load_documents()

print("Documents loaded:", len(documents))


# ============================================================
# 2. Create chunks
# ============================================================

chunks = chunk_documents(
    documents,
    chunk_size=500,
    overlap=100
)

print("Total chunks:", len(chunks))


# ============================================================
# 3. Create embeddings for chunks
# ============================================================

texts = [
    chunk["content"]
    for chunk in chunks
]

embeddings = create_embeddings(texts)

print("Embeddings shape:", embeddings.shape)


# ============================================================
# 4. Create FAISS index
# ============================================================

index = create_faiss_index(embeddings)

print("FAISS index created!")
print("Vectors stored in FAISS:", index.ntotal)


# ============================================================
# 5. User question
# ============================================================

query = "What is supervised learning?"

print("\nQuestion:")
print(query)


# ============================================================
# 6. Create embedding for question
# ============================================================

query_embedding = create_embeddings(
    [query]
)[0]


# ============================================================
# 7. Search FAISS
# ============================================================

distances, indices = search_faiss(
    index,
    query_embedding,
    top_k=2
)


# ============================================================
# 8. Display results
# ============================================================

print("\nFAISS Search Results")
print("====================")

for rank, index_number in enumerate(indices):

    chunk = chunks[index_number]

    print(f"\nRank: {rank + 1}")
    print("File:", chunk["filename"])
    print("Chunk ID:", chunk["chunk_id"])
    print("Distance:", distances[rank])
    print("Content:")
    print(chunk["content"])