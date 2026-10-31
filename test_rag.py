from rag.retriever import load_documents, retrieve_documents


documents = load_documents()

print("Documents loaded:", len(documents))

query = "What is machine learning?"

results = retrieve_documents(
    query,
    documents,
    top_k=2
)

print("\nRetrieved Documents:")
print("====================")

for result in results:

    print("\nFile:", result["filename"])
    print("Score:", result["score"])
    print("Content:")
    print(result["content"])