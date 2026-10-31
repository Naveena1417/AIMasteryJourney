import ollama

from rag.hybrid_retriever import (
    build_hybrid_retriever,
    hybrid_search
)


MODEL_NAME = "qwen2.5:3b"


# ============================================================
# RESEARCH AGENT
# ============================================================

def research_question(question):
    """
    Answer AI/ML research questions using Hybrid RAG.
    """

    # --------------------------------------------------------
    # STEP 1 — Build Hybrid Retriever
    # --------------------------------------------------------

    chunks, faiss_index, bm25_index = build_hybrid_retriever()

    # --------------------------------------------------------
    # STEP 2 — Retrieve relevant documents
    # --------------------------------------------------------

    results = hybrid_search(
        question,
        chunks,
        faiss_index,
        bm25_index,
        top_k=3
    )

    # --------------------------------------------------------
    # STEP 3 — Create context
    # --------------------------------------------------------

    context_parts = []

    for result in results:

        context_parts.append(
            f"Source: {result['filename']}\n"
            f"Content: {result['content']}"
        )

    context = "\n\n".join(context_parts)

    # --------------------------------------------------------
    # STEP 4 — Ask Ollama
    # --------------------------------------------------------

    prompt = f"""
You are an AI and Machine Learning Research Assistant.

Answer the user's question using ONLY the provided context.

Context:
{context}

User Question:
{question}

Rules:
- Give a clear and simple answer.
- Stay grounded in the provided context.
- Do not invent facts.
- If the context does not contain enough information, say:
  "The provided documents do not contain enough information to answer this question."
"""

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response["message"]["content"].strip()

    # --------------------------------------------------------
    # STEP 5 — Return answer + sources
    # --------------------------------------------------------

    return {
        "answer": answer,
        "sources": results
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    question = "What is supervised learning?"

    result = research_question(question)

    print("\n🔬 Research Agent Answer")
    print("========================")

    print(result["answer"])

    print("\n📚 Sources")
    print("==========")

    for source in result["sources"]:

        print(
            f"- {source['filename']} "
            f"(score: {source['score']:.4f})"
        )