import unittest
import pandas as pd
import numpy as np
import os
import json
import tempfile

# Import the functions directly
from importlib.machinery import SourceFileLoader
prep = SourceFileLoader("prep", "02b_preprocessing_sequences.py").load_module()
markov = SourceFileLoader("markov", "03b_markov_recommender.py").load_module()
eval_mod = SourceFileLoader("eval_mod", "04b_evaluate_recommender.py").load_module()

class TestRecommendationPipeline(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        
        # 1. Create Mock PDE Data
        self.pde_data = pd.DataFrame({
            'DESYNPUF_ID': ['P1', 'P1', 'P1', 'P2', 'P2'],
            'PROD_SRVC_ID': ['DrugA', 'DrugA', 'DrugB', 'DrugC', 'DrugD'],
            'SRVC_DT': ['20080101', '20080201', '20080301', '20100501', '20100601'],
            'DAYS_SUPLY_NUM': [30, 30, 30, 30, 30],
            'QTY_DSPNSD_NUM': [1, 1, 1, 1, 1]
        })
        self.pde_path = os.path.join(self.test_dir.name, 'mock_pde.csv')
        self.pde_data.to_csv(self.pde_path, index=False)
        
        # 2. Create Mock Beneficiary Data
        self.ben_2008 = pd.DataFrame({
            'DESYNPUF_ID': ['P1'],
            'SP_DIABETES': [1],
            'SP_ALZHDMTA': [0], 'SP_CHF': [0], 'SP_CHRNKIDN': [0], 'SP_CNCR': [0],
            'SP_COPD': [0], 'SP_DEPRESSN': [0], 'SP_ISCHMCHT': [0],
            'SP_OSTEOPRS': [0], 'SP_RA_OA': [0], 'SP_STRKETIA': [0]
        })
        self.ben2008_path = os.path.join(self.test_dir.name, 'mock_ben_2008.csv')
        self.ben_2008.to_csv(self.ben2008_path, index=False)
        
        self.ben_2010 = pd.DataFrame({
            'DESYNPUF_ID': ['P2'],
            'SP_DIABETES': [0],
            'SP_ALZHDMTA': [0], 'SP_CHF': [0], 'SP_CHRNKIDN': [0], 'SP_CNCR': [0],
            'SP_COPD': [0], 'SP_DEPRESSN': [0], 'SP_ISCHMCHT': [0],
            'SP_OSTEOPRS': [0], 'SP_RA_OA': [0], 'SP_STRKETIA': [0]
        })
        self.ben2010_path = os.path.join(self.test_dir.name, 'mock_ben_2010.csv')
        self.ben_2010.to_csv(self.ben2010_path, index=False)
        
        self.ben_paths = {
            2008: self.ben2008_path,
            2010: self.ben2010_path
        }
        
        self.seq_out = os.path.join(self.test_dir.name, 'mock_seq.parquet')
        self.model_out = os.path.join(self.test_dir.name, 'mock_model.json')

    def tearDown(self):
        self.test_dir.cleanup()

    def test_01_merge_and_sequence(self):
        # Test merging
        merged = prep.load_and_merge_data(self.pde_path, self.ben_paths)
        self.assertEqual(len(merged), 5)
        # Check P1 got Diabetes=1 from 2008
        p1_diabetes = merged[merged['DESYNPUF_ID'] == 'P1']['SP_DIABETES'].values[0]
        self.assertEqual(p1_diabetes, 1.0)
        
        # Test sequencing
        prep.build_sequences(merged, self.seq_out)
        self.assertTrue(os.path.exists(self.seq_out))
        
        seq_df = pd.read_parquet(self.seq_out)
        # P1 transitions: A -> A -> B. Since consecutive identical is removed, it should be: [A, B]
        # P2 transitions: C -> D. It should be [C, D]
        # So we should have 4 transition target rows total. (A starts, B starts, C starts, D starts)
        self.assertEqual(len(seq_df), 4)
        
        p1_drugs = seq_df[seq_df['DESYNPUF_ID'] == 'P1']['PROD_SRVC_ID'].tolist()
        self.assertEqual(p1_drugs, ['DrugA', 'DrugB'])

    def test_02_markov_training(self):
        # Since markov logic reads 2008-2009 for training, we need to make sure the sequence file exists
        merged = prep.load_and_merge_data(self.pde_path, self.ben_paths)
        prep.build_sequences(merged, self.seq_out)
        
        model = markov.train_markov_model(self.seq_out, self.model_out)
        
        self.assertTrue(os.path.exists(self.model_out))
        self.assertIn('global', model)
        self.assertIn('diabetes', model)
        
        # M1 transition DrugA -> DrugB in 2008
        self.assertIn('DrugA', model['global'])
        self.assertEqual(model['global']['DrugA'][0][0], 'DrugB')
        self.assertEqual(model['global']['DrugA'][0][1], 1.0) # 100% prob

    def test_03_prediction(self):
        mock_model = {
            'global': {'DrugX': [['DrugY', 0.8], ['DrugZ', 0.2]]},
            'diabetes': {'DrugX': [['DrugZ', 0.9], ['DrugY', 0.1]]}
        }
        
        # Global patient
        preds = markov.predict_next_drugs(mock_model, 'DrugX', has_diabetes=False, top_k=1)
        self.assertEqual(preds[0][0], 'DrugY')
        
        # Diabetic patient
        preds_diab = markov.predict_next_drugs(mock_model, 'DrugX', has_diabetes=True, top_k=1)
        self.assertEqual(preds_diab[0][0], 'DrugZ')

if __name__ == '__main__':
    unittest.main()
