# Copyright 2026 The OpenSLM Project
import json
import random

def generate_iris_data():
    data = []
    for _ in range(1000):
        stock = random.randint(0, 100)
        velocity = random.randint(1, 20)
        days_left = stock // velocity if velocity > 0 else 100
        
        status = "CRITICAL" if days_left < 3 else "HEALTHY"
        
        data.append({
            "instruction": f"Analyse: Stock={stock}, Ventes/jour={velocity}",
            "response": f"Status: {status}. Rupture prévue dans {days_left} jours."
        })
    
    with open("IRIS/data/train_analytics.jsonl", "w") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")
    print("IRIS: Dataset analytique généré.")

if __name__ == "__main__":
    generate_iris_data()
