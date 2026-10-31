from rag.retriever import load_documents
from rag.chunker import chunk_documents


# Load documents
documents = load_documents()

print("Documents loaded:", len(documents))


# Create chunks
chunks = chunk_documents(
    documents,
    chunk_size=500,
    overlap=100
)


print("Total chunks:", len(chunks))


print("\nFirst 3 chunks")
print("================")


for chunk in chunks[:3]:

    print("\nFile:", chunk["filename"])
    print("Chunk ID:", chunk["chunk_id"])
    print("Content:")
    print(chunk["content"])