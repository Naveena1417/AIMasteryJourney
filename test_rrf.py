from rag.retriever import load_documents
from rag.chunker import chunk_documents
from rag.embeddings import create_embeddings
from rag.faiss_store import create_faiss_index, search_faiss
from rag.bm25_search import create_bm25_index, search_bm25
from rag.rrf import reciprocal_rank_fusion


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
# 3. Create embeddings
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

faiss_index = create_faiss_index(embeddings)

print("FAISS index created!")


# ============================================================
# 5. Create BM25 index
# ============================================================

bm25_index = create_bm25_index(chunks)

print("BM25 index created!")


# ============================================================
# 6. User question
# ============================================================

query = "What is supervised learning?"

print("\nQuestion:")
print(query)


# ============================================================
# 7. Query embedding
# ============================================================

query_embedding = create_embeddings(
    [query]
)[0]


# ============================================================
# 8. FAISS search
# ============================================================

faiss_distances, faiss_indices = search_faiss(
    faiss_index,
    query_embedding,
    top_k=3
)

print("\nFAISS Results")
print("=============")

for rank, index in enumerate(faiss_indices):

    print(
        f"Rank {rank + 1}: "
        f"{chunks[index]['filename']} "
        f"Chunk {chunks[index]['chunk_id']}"
    )


# ============================================================
# 9. BM25 search
# ============================================================

bm25_scores, bm25_indices = search_bm25(
    bm25_index,
    query,
    top_k=3
)

print("\nBM25 Results")
print("============")

for rank, index in enumerate(bm25_indices):

    print(
        f"Rank {rank + 1}: "
        f"{chunks[index]['filename']} "
        f"Chunk {chunks[index]['chunk_id']}"
    )


# ============================================================
# 10. RRF
# ============================================================

rrf_results = reciprocal_rank_fusion(
    faiss_indices,
    bm25_indices
)


# ============================================================
# 11. Display RRF results
# ============================================================

print("\nRRF Combined Results")
print("====================")

for rank, (index, score) in enumerate(
    rrf_results[:3]
):

    chunk = chunks[index]

    print(f"\nRank: {rank + 1}")
    print("File:", chunk["filename"])
    print("Chunk ID:", chunk["chunk_id"])
    print("RRF Score:", score)
    print("Content:")
    print(chunk["content"])