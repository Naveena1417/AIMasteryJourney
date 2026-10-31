def reciprocal_rank_fusion(
    faiss_indices,
    bm25_indices,
    k=60
):
    """
    Combine FAISS and BM25 rankings
    using Reciprocal Rank Fusion (RRF).
    """

    scores = {}

    # ========================================================
    # FAISS ranking
    # ========================================================

    for rank, index in enumerate(faiss_indices):

        rrf_score = 1 / (k + rank + 1)

        scores[index] = scores.get(index, 0) + rrf_score

    # ========================================================
    # BM25 ranking
    # ========================================================

    for rank, index in enumerate(bm25_indices):

        rrf_score = 1 / (k + rank + 1)

        scores[index] = scores.get(index, 0) + rrf_score

    # ========================================================
    # Sort by combined RRF score
    # ========================================================

    ranked_results = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked_results