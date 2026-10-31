import faiss
import numpy as np


def create_faiss_index(embeddings):
    """
    Create a FAISS index using embedding vectors.
    """

    # Embedding dimension
    dimension = embeddings.shape[1]

    # FAISS index
    index = faiss.IndexFlatL2(dimension)

    # FAISS expects float32
    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    # Add embeddings to index
    index.add(embeddings)

    return index


def search_faiss(
    index,
    query_embedding,
    top_k=2
):
    """
    Search the FAISS index.
    """

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    # FAISS expects 2D array
    query_embedding = query_embedding.reshape(
        1, -1
    )

    distances, indices = index.search(
        query_embedding,
        top_k
    )

    return distances[0], indices[0]