import os

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# ============================================================
# 1. LOAD TOKEN
# ============================================================

load_dotenv(override=True)

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    print("❌ HF_TOKEN not found")
    exit()

print("✅ HF_TOKEN loaded")

# ============================================================
# 2. HUGGING FACE CLIENT
# ============================================================

client = InferenceClient(
    api_key=HF_TOKEN,
    provider="auto"
)

# ============================================================
# 3. TEST PROMPT
# ============================================================

prompt = """
Explain Machine Learning in simple words for a beginner.
Give the answer in 3 simple sentences.
"""

# ============================================================
# 4. CALL LLM
# ============================================================

try:

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=150
    )

    answer = response.choices[0].message.content

    print("\n🤖 LLM Answer:")
    print("================")
    print(answer)

except Exception as e:

    print("\n❌ LLM connection failed")
    print("Error:", e)