from rag.retriever import load_documents
from rag.chunker import chunk_documents
from rag.embeddings import create_embeddings


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


# Get chunk text
texts = [
    chunk["content"]
    for chunk in chunks
]


# Create embeddings
embeddings = create_embeddings(texts)


print("\nEmbedding created successfully!")
print("==============================")

print("Number of embeddings:", len(embeddings))
print("Embedding dimension:", embeddings.shape[1])

print("\nFirst embedding:")
print(embeddings[0])