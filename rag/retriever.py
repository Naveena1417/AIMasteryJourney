import os


DOCUMENTS_FOLDER = "documents"


def load_documents():
    documents = []

    for filename in os.listdir(DOCUMENTS_FOLDER):

        if filename.endswith(".txt"):

            file_path = os.path.join(DOCUMENTS_FOLDER, filename)

            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()

            documents.append({
                "filename": filename,
                "content": content
            })

    return documents


def retrieve_documents(query, documents, top_k=2):

    query_words = query.lower().split()

    scored_documents = []

    for document in documents:

        content = document["content"].lower()

        score = 0

        for word in query_words:

            if word in content:
                score += 1

        scored_documents.append(
            {
                "filename": document["filename"],
                "content": document["content"],
                "score": score
            }
        )

    scored_documents.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return scored_documents[:top_k]