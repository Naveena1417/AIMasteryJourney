from sentence_transformers import SentenceTransformer


# Embedding model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)


def create_embeddings(texts):
    """
    Convert text into numerical vectors.
    """

    embeddings = model.encode(
        texts,
        convert_to_numpy=True
    )

    return embeddings