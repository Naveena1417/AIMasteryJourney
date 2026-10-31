# ============================================================
# DOCUMENT CHUNKING
# ============================================================


def chunk_text(text, chunk_size=500, overlap=100):
    """
    Split a document into smaller chunks.

    chunk_size:
        Maximum number of characters in each chunk.

    overlap:
        Number of characters shared between chunks.
    """

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start = end - overlap

    return chunks


def chunk_documents(documents, chunk_size=500, overlap=100):
    """
    Chunk all documents.
    """

    all_chunks = []

    for document in documents:

        chunks = chunk_text(
            document["content"],
            chunk_size=chunk_size,
            overlap=overlap
        )

        for index, chunk in enumerate(chunks):

            all_chunks.append(
                {
                    "filename": document["filename"],
                    "chunk_id": index,
                    "content": chunk
                }
            )

    return all_chunks