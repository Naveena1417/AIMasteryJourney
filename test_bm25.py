from rag.retriever import load_documents
from rag.chunker import chunk_documents
from rag.bm25_search import create_bm25_index, search_bm25


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
# 3. Create BM25 index
# ============================================================

bm25 = create_bm25_index(chunks)

print("BM25 index created!")


# ============================================================
# 4. User question
# ============================================================

query = "What is supervised learning?"

print("\nQuestion:")
print(query)


# ============================================================
# 5. Search using BM25
# ============================================================

scores, indices = search_bm25(
    bm25,
    query,
    top_k=2
)


# ============================================================
# 6. Display results
# ============================================================

print("\nBM25 Search Results")
print("===================")

for rank, index_number in enumerate(indices):

    chunk = chunks[index_number]

    print(f"\nRank: {rank + 1}")
    print("File:", chunk["filename"])
    print("Chunk ID:", chunk["chunk_id"])
    print("BM25 Score:", scores[index_number])
    print("Content:")
    print(chunk["content"])