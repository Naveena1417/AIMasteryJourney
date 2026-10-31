import ollama


# ============================================================
# 1. MODEL
# ============================================================

MODEL_NAME = "qwen2.5:3b"


# ============================================================
# 2. TEST PROMPT
# ============================================================

prompt = """
Explain Machine Learning in simple words for a beginner.
Give the answer in 3 simple sentences.
"""


# ============================================================
# 3. CALL LOCAL LLM
# ============================================================

try:

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response["message"]["content"]

    print("\n🤖 Ollama LLM Answer:")
    print("======================")
    print(answer)

except Exception as e:

    print("\n❌ Ollama connection failed")
    print("Error:", e)