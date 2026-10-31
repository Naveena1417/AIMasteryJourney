import os
import requests

from dotenv import load_dotenv

# Load .env
load_dotenv(override=True)

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    print("❌ HF_TOKEN not found")
    exit()

print("✅ HF_TOKEN loaded")

# Ask Hugging Face for available models
url = "https://router.huggingface.co/v1/models"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

response = requests.get(
    url,
    headers=headers
)

print("Status:", response.status_code)

if response.status_code != 200:
    print("❌ Failed to get models")
    print(response.text)
    exit()

data = response.json()

print("\nAvailable Models:")
print("=================")

for model in data.get("data", [])[:20]:
    print(model["id"])