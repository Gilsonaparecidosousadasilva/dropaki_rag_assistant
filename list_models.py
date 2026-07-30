import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()  # <-- isso carrega o .env pro ambiente

api_key = os.environ.get("GOOGLE_API_KEY")
print("Chave carregada?", bool(api_key))  # debug rápido

genai.configure(api_key=api_key)

print("Modelos disponíveis que suportam embedContent:\n")
for m in genai.list_models():
    if 'embedContent' in m.supported_generation_methods:
        print(f"- {m.name}")