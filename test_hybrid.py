from rag.hybrid_retriever import (
    build_hybrid_retriever,
    hybrid_search
)


# ============================================================
# 1. Build hybrid retriever
# ============================================================

print("Building hybrid retriever...")

chunks, faiss_index, bm25_index = (
    build_hybrid_retriever()
)

print("Hybrid retriever ready!")


# ============================================================
# 2. User question
# ============================================================

query = "What is supervised learning?"

print("\nQuestion:")
print(query)


# ============================================================
# 3. Hybrid search
# ============================================================

results = hybrid_search(
    query,
    chunks,
    faiss_index,
    bm25_index,
    top_k=3
)


# ============================================================
# 4. Display results
# ============================================================

print("\nHybrid Search Results")
print("=====================")

for rank, result in enumerate(
    results,
    start=1
):

    print(f"\nRank: {rank}")

    print(
        "File:",
        result["filename"]
    )

    print(
        "Chunk ID:",
        result["chunk_id"]
    )

    print(
        "RRF Score:",
        result["score"]
    )

    print("Content:")
    print(result["content"])