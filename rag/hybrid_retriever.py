from rag.retriever import load_documents
from rag.chunker import chunk_documents
from rag.embeddings import create_embeddings
from rag.faiss_store import create_faiss_index, search_faiss
from rag.bm25_search import create_bm25_index, search_bm25
from rag.rrf import reciprocal_rank_fusion


def build_hybrid_retriever():
    """
    Build FAISS + BM25 hybrid retrieval system.
    """

    # Load documents
    documents = load_documents()

    # Create chunks
    chunks = chunk_documents(
        documents,
        chunk_size=500,
        overlap=100
    )

    # Create embeddings
    texts = [
        chunk["content"]
        for chunk in chunks
    ]

    embeddings = create_embeddings(texts)

    # Create FAISS index
    faiss_index = create_faiss_index(
        embeddings
    )

    # Create BM25 index
    bm25_index = create_bm25_index(
        chunks
    )

    return (
        chunks,
        faiss_index,
        bm25_index
    )


def hybrid_search(
    query,
    chunks,
    faiss_index,
    bm25_index,
    top_k=3
):
    """
    Perform FAISS + BM25 + RRF hybrid search.
    """

    # --------------------------------------------------------
    # FAISS search
    # --------------------------------------------------------

    query_embedding = create_embeddings(
        [query]
    )[0]

    _, faiss_indices = search_faiss(
        faiss_index,
        query_embedding,
        top_k=top_k
    )

    # --------------------------------------------------------
    # BM25 search
    # --------------------------------------------------------

    _, bm25_indices = search_bm25(
        bm25_index,
        query,
        top_k=top_k
    )

    # --------------------------------------------------------
    # RRF
    # --------------------------------------------------------

    rrf_results = reciprocal_rank_fusion(
        faiss_indices,
        bm25_indices
    )

    # --------------------------------------------------------
    # Return top results
    # --------------------------------------------------------

    results = []

    for index, score in rrf_results[:top_k]:

        results.append(
            {
                "filename": chunks[index]["filename"],
                "chunk_id": chunks[index]["chunk_id"],
                "content": chunks[index]["content"],
                "score": score
            }
        )

    return results