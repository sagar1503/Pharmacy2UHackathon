import pandas as pd
import json

def evaluate_recommender(seq_path='sequences_pde.parquet', model_path='markov_transitions.json', K=5):
    print(f"Loading sequence data...")
    df = pd.read_parquet(seq_path)
    
    # Isolate 2010 sequences as hold-out set
    test_df = df[df['YEAR'] == 2010].copy()
    
    print(f"Loading transition model...")
    with open(model_path, 'r') as f:
        model = json.load(f)
        
    print(f"Evaluating Recommendations (Recall@{K}) on {len(test_df)} transitions in 2010...")
    
    hits = 0
    total_evals = 0
    
    for patient, group in test_df.groupby('DESYNPUF_ID'):
        # Sort chronologically
        patient_seq = group.sort_values('SRVC_DT')
        drugs = patient_seq['PROD_SRVC_ID'].tolist()
        has_diabetes = patient_seq['SP_DIABETES'].max() == 1
        
        # Determine which pathway logic to use
        pathway = 'diabetes' if has_diabetes else 'global'
        
        for i in range(len(drugs) - 1):
            current_drug = drugs[i]
            actual_next_drug = drugs[i+1]
            
            # Get predictions from our trained model
            transitions = model[pathway].get(current_drug, [])
            if not transitions and has_diabetes:
                # Fallback to global
                transitions = model['global'].get(current_drug, [])
                
            # Keep top K predictions
            top_k_preds = [d[0] for d in transitions[:K]]
            
            if len(top_k_preds) > 0:
                total_evals += 1
                if actual_next_drug in top_k_preds:
                    hits += 1
                    
    if total_evals == 0:
        print("No evaluations could be made.")
        return
        
    recall_at_k = hits / total_evals
    print(f"\n--- Evaluation Results ---")
    print(f"Total Transitions Evaluated: {total_evals}")
    print(f"Recall@{K}: {recall_at_k:.4f} (The actual next drug was in the Top {K} recommendations {recall_at_k*100:.1f}% of the time)")

if __name__ == "__main__":
    evaluate_recommender(K=5)
