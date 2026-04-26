import json

class AIReporter:
    def __init__(self, model_name="meditron:7b"): # Or Llama-3 via Ollama
        self.model_name = model_name

    def generate_clinical_summary(self, drug_name, population_results):
        # Extract the risk data
        risks = population_results.get("risk_summary", {})
        admet = population_results.get("admet_predictions", {})
        
        prompt = f"""
        ACT AS A CLINICAL PHARMACOLOGIST.
        Review this Virtual Trial for {drug_name}.
        
        AI PREDICTIONS:
        - logP: {admet.get('logp')}
        - Clearance (CLint): {admet.get('clint')} L/h
        - Protein Binding (fup): {admet.get('fup')}
        
        SIMULATION RESULTS:
        - {risks}
        
        QUESTION: 
        The simulation shows 100% risk. Is this a real toxic threat, 
        or is this a known characteristic of the drug's fast metabolism?
        Provide a 3-sentence executive summary.
        """
        
        print(f"\n[AI Reporter] Analyzing results for {drug_name}...")
        # Here we would call the LLM API. For now, let's simulate the logic:
        return self._simulated_llm_response(drug_name, admet)

    def _simulated_llm_response(self, drug_name, admet):
        if float(admet.get('clint', 0)) > 500:
            return (f"DECISION: FALSE POSITIVE. {drug_name} shows high organ exposure due to "
                    "ultra-rapid metabolic conversion (CLint > 500). The 100% GI/Liver risk "
                    "is a signature of first-pass activation, not systemic toxicity.")
        return "DECISION: POTENTIAL TOXICITY. Further investigation required."

# Test the reporter
if __name__ == "__main__":
    reporter = AIReporter()
    print(reporter._simulated_llm_response("Aspirin", {"clint": 2000}))