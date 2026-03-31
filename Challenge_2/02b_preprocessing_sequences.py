import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def load_and_merge_data(pde_path, ben_paths):
    """
    Loads PDE data and merges it with the yearly beneficiary summary files 
    so chronical conditions are accurate for the year the prescription was made.
    """
    print("Loading raw PDE data...")
    pde = pd.read_csv(pde_path, dtype={'PROD_SRVC_ID': str})
    pde['SRVC_DT'] = pd.to_datetime(pde['SRVC_DT'].astype(str), format='%Y%m%d', errors='coerce')
    
    # Extract year for the merge
    pde['YEAR'] = pde['SRVC_DT'].dt.year
    
    # Standard Quality Filters
    pde = pde.dropna(subset=['SRVC_DT', 'DAYS_SUPLY_NUM', 'QTY_DSPNSD_NUM'])
    pde = pde[(pde['DAYS_SUPLY_NUM'] > 0) & (pde['QTY_DSPNSD_NUM'] > 0)]
    
    # We will keep the full NDC-11 code for pathway analysis
    pde['PROD_SRVC_ID'] = pde['PROD_SRVC_ID'].astype(str)
    
    print("Loading and appending Beneficiary files (2008-2010)...")
    ben_cols = [
        'DESYNPUF_ID', 'SP_ALZHDMTA', 'SP_CHF', 'SP_CHRNKIDN', 'SP_CNCR', 
        'SP_COPD', 'SP_DEPRESSN', 'SP_DIABETES', 'SP_ISCHMCHT', 
        'SP_OSTEOPRS', 'SP_RA_OA', 'SP_STRKETIA'
    ]
    
    all_ben = []
    for year, path in ben_paths.items():
        try:
            ben_yr = pd.read_csv(path, usecols=ben_cols)
            ben_yr['YEAR'] = year
            all_ben.append(ben_yr)
        except FileNotFoundError:
            print(f"Warning: {path} not found. Skipping year {year}.")
            
    ben_df = pd.concat(all_ben, ignore_index=True)
    
    print("Merging PDE with temporally accurate Chronic Conditions...")
    # Merge on Patient ID and the Year the prescription was dispensed
    merged = pd.merge(pde, ben_df, on=['DESYNPUF_ID', 'YEAR'], how='left')
    
    return merged

def build_sequences(df, output_path='sequences_pde.parquet'):
    print("Sorting chronologically to build patient sequences...")
    # Sort strictly by patient and service date
    df = df.sort_values(by=['DESYNPUF_ID', 'SRVC_DT'])
    
    # We can create a unified "Patient State" per day (baskets if multiple drugs dispensed same day)
    # But for a simple Sequence/Markov approach, we just need the ordered sequence of distinct drugs over time.
    
    # Since patients often refill the *same* drug back-to-back, we should collapse consecutive 
    # identical drugs to focus on pathway *transitions* (Drug A -> Drug B), unless we want to predict refills too.
    # For Next-Best Recommendation (Challenge B), we typically want to predict the *next distinct* drug.
    
    print("Filtering out consecutive identical prescription events for pure transition modeling...")
    df['PREV_DRUG'] = df.groupby('DESYNPUF_ID')['PROD_SRVC_ID'].shift(1)
    
    # Keep row if it's the first drug (PREV_DRUG is null) OR if it is different from the previous drug.
    transitions = df[df['PROD_SRVC_ID'] != df['PREV_DRUG']].copy()
    
    print(f"Reduced from {len(df)} discrete events to {len(transitions)} transition events.")
    
    # Calculate days since last transition
    transitions['PREV_SRVC_DT'] = transitions.groupby('DESYNPUF_ID')['SRVC_DT'].shift(1)
    transitions['DAYS_SINCE_LAST_NEW_DRUG'] = (transitions['SRVC_DT'] - transitions['PREV_SRVC_DT']).dt.days
    
    print(f"Saving sequence data to {output_path}...")
    transitions.to_parquet(output_path, index=False)
    print("Done!")

if __name__ == "__main__":
    pde_file = 'DE1_0_2008_to_2010_Prescription_Drug_Events_Sample_1.csv'
    beneficiary_files = {
        2008: 'DE1_0_2008_Beneficiary_Summary_File_Sample_1.csv',
        2009: 'DE1_0_2009_Beneficiary_Summary_File_Sample_1.csv',
        2010: 'DE1_0_2010_Beneficiary_Summary_File_Sample_1.csv'
    }
    
    merged_data = load_and_merge_data(pde_file, beneficiary_files)
    build_sequences(merged_data, 'sequences_pde.parquet')
