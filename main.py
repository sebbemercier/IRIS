from fastapi import FastAPI
import httpx
import torch
from transformers import pipeline

app = FastAPI(title="IRIS | L'Analyseur")

ATLAS_URL = "http://localhost:8001"

print("Chargement du SLM IRIS (TinyLlama)...")
pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")

@app.get("/analyze/{product_id}")
async def analyze_product(product_id: int):
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{ATLAS_URL}/product/{product_id}")
        product_data = response.json()
    
    if "error" in product_data: return product_data

    prompt = f"<|system|>\nTu es IRIS, une IA d'analyse de marché. Analyse ce produit et donne un score de tendance de 1 à 100 avec une courte explication.</s>\n<|user|>\nProduit: {product_data['name']}\nDescription: {product_data['description']}</s>\n<|assistant|>\n"
    
    outputs = pipe(prompt, max_new_tokens=50, do_sample=True, temperature=0.5)
    analysis = outputs[0]["generated_text"].split("<|assistant|>\n")[-1].strip()
    
    return {
        "product": product_data["name"],
        "ai_analysis": analysis
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
