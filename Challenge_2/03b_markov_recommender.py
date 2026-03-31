import pandas as pd
import numpy as np
from collections import defaultdict
import json
import warnings
warnings.filterwarnings('ignore')

def train_markov_model(seq_path='sequences_pde.parquet', output_model='markov_transitions.json'):
    print(f"Loading sequence data from {seq_path}...")
    df = pd.read_parquet(seq_path)
    
    # We want to train our pathways on historical data (2008-2009)
    # The `YEAR` column was appended during preprocessing
    train_df = df[df['YEAR'] <= 2009].copy()
    
    print(f"Training Markov Chain on {len(train_df)} transitions from 2008-2009...")
    
    # We want to calculate P(Drug B | Drug A)
    # To incorporate Chronic Diseases (e.g. SP_DIABETES), we can build conditional transition matrices.
    # For a hackathon baseline, we'll build a Global matrix, and one for Diabetes as an example.
    
    global_transitions = defaultdict(lambda: defaultdict(int))
    diabetes_transitions = defaultdict(lambda: defaultdict(int))
    
    # Iterate through patients to build sequences
    for patient, group in train_df.groupby('DESYNPUF_ID'):
        # Sort chronologically
        patient_seq = group.sort_values('SRVC_DT')
        drugs = patient_seq['PROD_SRVC_ID'].tolist()
        has_diabetes = patient_seq['SP_DIABETES'].max() == 1 # 1 if they had diabetes in any train year
        
        for i in range(len(drugs) - 1):
            current_drug = drugs[i]
            next_drug = drugs[i+1]
            
            global_transitions[current_drug][next_drug] += 1
            if has_diabetes:
                diabetes_transitions[current_drug][next_drug] += 1
                
    def to_probabilities(counts_dict):
        prob_dict = {}
        for current_drug, next_drugs in counts_dict.items():
            total = sum(next_drugs.values())
            # Convert counts to probabilities, sort by highest prob, keep top 10
            sorted_probs = sorted(
                [(d, count/total) for d, count in next_drugs.items()],
                key=lambda x: x[1], reverse=True
            )[:10]
            prob_dict[current_drug] = sorted_probs
        return prob_dict
        
    print("Normalizing probabilities...")
    model = {
        'global': to_probabilities(global_transitions),
        'diabetes': to_probabilities(diabetes_transitions)
    }
    
    with open(output_model, 'w') as f:
        json.dump(model, f)
        
    print(f"Transition model saved to {output_model} with {len(model['global'])} source drugs.")
    return model

def predict_next_drugs(model, current_drug, has_diabetes=False, top_k=5):
    """
    Given an NDC-11 code and patient context, predict the Top K next drugs.
    """
    pathway = 'diabetes' if has_diabetes else 'global'
    
    transitions = model[pathway].get(current_drug, [])
    # Fallback to global if strict cohort has no data
    if not transitions and has_diabetes:
        transitions = model['global'].get(current_drug, [])
        
    return transitions[:top_k]

if __name__ == "__main__":
    model = train_markov_model()
    
    # Quick sanity check / Demo
    sample_drugs = list(model['global'].keys())[:3]
    print("\n--- Model Inference Demo ---")
    for drug in sample_drugs:
        preds = predict_next_drugs(model, drug, has_diabetes=False, top_k=3)
        print(f"If patient is on Drug {drug}, next most likely distinct drugs:")
        for next_drug, prob in preds:
            print(f"  -> {next_drug} ({prob*100:.1f}%)")
