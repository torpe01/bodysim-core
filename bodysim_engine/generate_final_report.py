import requests
import json
import os
from datetime import datetime

def get_improved_report(drug, risks, clint, fup):
    url = "http://localhost:11434/api/generate"
    
    # IMPROVED PROMPT: We add clinical context about pro-drugs and flux
    prompt = f"""
    [CONTEXT] You are a Clinical Pharmacologist interpreting a PBPK simulation.
    [DATA] Drug: {drug}, CLint: {clint} L/h, Protein Binding: {fup}.
    [RESULT] The math engine flagged 100% Risk in Liver/GI due to high metabolic flux.
    
    [TASK] 
    1. Distinguish between 'Metabolic Flux' and 'Toxic Damage'. 
    2. Explain if this high CLint is a sign of rapid pro-drug activation (like Aspirin to Salicylate).
    3. Provide an 'Honest Summary' that reduces the 'danger of toxicity' alarm if it's just normal metabolism.
    
    [FORMAT] Provide a 3-sentence expert summary.
    """
    
    payload = {
        "model": "gemma2:2b",
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2}
    }
    
    print(f"[AI] Refining medical interpretation for {drug}...")
    response = requests.post(url, json=payload)
    return response.json().get('response', "AI Error")

# Data from your trial
trial_data = {
    "drug_name": "Caffeine",
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "physics_results": {
        "clint": 12.0, # Approximate lower clearance for Caffeine
        "fup": 0.64,   # Lower protein binding
        "risk_flags": ["Liver: 11%", "GI: 2%"] # Your actual engine results!
    }
}

# Get the refined summary
ai_summary = get_improved_report(
    trial_data["drug_name"], 
    "Liver/GI: 100%", 
    trial_data["physics_results"]["clint"],
    trial_data["physics_results"]["fup"]
)

# Add summary to our data
trial_data["clinical_interpretation"] = ai_summary

# Save to JSON
file_path = f"reports/trial_{trial_data['drug_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(file_path, "w") as f:
    json.dump(trial_data, f, indent=4)

print(f"\n--- REFINED REPORT SAVED TO {file_path} ---")
print(f"AI SUMMARY: {ai_summary}")