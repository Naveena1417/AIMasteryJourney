from rank_bm25 import BM25Okapi


def create_bm25_index(chunks):
    """
    Create a BM25 keyword search index.
    """

    tokenized_chunks = [
        chunk["content"].lower().split()
        for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized_chunks)

    return bm25


def search_bm25(
    bm25,
    query,
    top_k=2
):
    """
    Search chunks using BM25 keyword matching.
    """

    query_tokens = query.lower().split()

    scores = bm25.get_scores(query_tokens)

    ranked_indices = scores.argsort()[::-1]

    top_indices = ranked_indices[:top_k]

    return scores, top_indices