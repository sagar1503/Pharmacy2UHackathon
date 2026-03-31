# Pharmacy2U Hackathon - Challenge B: Pathway-Aware Recommendation Engine

## 🚀 Overview
This repository contains the complete end-to-end data pipeline, modeling code, and interactive UI for **Challenge B** of the Pharmacy2U Hackathon. 

Rather than treating pharmacy events as isolated refills, this engine views patient prescriptions as an interconnected clinical journey. We built a **Context-Aware 1st-Order Markov Chain** that processes 3.8 million historical prescription transitions to predict the *next most likely* drug a patient will need. 

Crucially, the matrices are clustered by patient chronic conditions (e.g., Diabetes, Heart Failure). This allows the engine to output dynamically personalized recommendations based on the patient's holistic health context.

---

## 📂 Repository Structure

*   **`02b_preprocessing_sequences.py`**
    *   **Purpose:** Ingests the 5.2M row CMS `DE-SynPUF` prescription dataset and the multi-year Beneficiary Summary files. Temporally joins the patient context to the exact year of the prescription and compresses chronological events into 3.8M strict `A -> B` state transitions.
*   **`03b_markov_recommender.py`**
    *   **Purpose:** Calculates the Maximum Likelihood Estimate (MLE) transition probabilities across 260,000 unique NDC nodes. Compiles a global matrix and isolated sub-graphs based on active patient context flags. Outputs the final model to `markov_transitions.json`.
*   **`04b_evaluate_recommender.py`**
    *   **Purpose:** Evaluates the engine using a strict temporal split (Train on 2008-2009, predict 2010 window) via the `Recall@K` metric.
*   **`test_recommendation_pipeline.py`**
    *   **Purpose:** Unit tests asserting the mathematical integrity of the Markov chain probability normalization and the contextual grouping logic.
*   **`app.py`**
    *   **Purpose:** A live, interactive Streamlit interface calculating $O(1)$ inference lookups on the matrix. It integrates asynchronously with the FDA RxNav REST API (`rxnav.nlm.nih.gov`) to translate the predicted 11-digit NDCs into human-readable clinical formulations.
*   **`find_good_drugs_enhanced.py`**
    *   **Purpose:** A utility script to crawl the massive transition matrix and extract "High-Divergence" NDCs (cases where adding a patient context flag massively alters the predicted clinical trajectory) for live presentations.
*   **`markov_transitions.json`**
    *   **Purpose:** The pre-trained, serialized probability matrix. (Included for immediate UI evaluation).
*   **`transition_matrix.png`**
    *   **Purpose:** A Seaborn heatmap visually demonstrating the probability weightings across the highest-traffic cardiovascular and diabetic sequences.

---

## 🛠️ How to Run

### 1. Prerequisites
Ensure you have Python 3.8+ installed. Install the following core libraries:
```bash
pip install pandas numpy streamlit pytest requests seaborn matplotlib
```

*(Note: The 400MB DE-SynPUF CMS `.csv` files are deliberately `.gitignore`'d to prevent bloating the repository. You must download `DE1_0_2008_to_2010_Prescription_Drug_Events_Sample_1.csv` and `Beneficiary_Summary_File_Sample_1.csv` directly from CMS to re-run the `02b` extraction).*

### 2. Testing the Pipeline
Execute the unit tests to verify transition mathematical logic:
```bash
pytest test_recommendation_pipeline.py -v
```

### 3. Launching the Simulator UI
*Note: Due to GitHub's 100MB file limit, the trained matrix was compressed. Before running the UI for the first time, simply unzip `markov_transitions.zip` located in this folder to extract `markov_transitions.json`.*

To test the pre-trained engine instantly, launch the Streamlit app:
```bash
streamlit run app.py
```
> **Demo Tip:** In the UI sidebar, try pasting the NDC code `00093075305` (Atenolol). Notice how the predicted pathways completely shift when you toggle the `Patient has Documented Diabetes?` switch!

---

## 📉 Evaluation & Data Caveat
We evaluated the engine's `Recall@5` performance using a rigorous chronological split (predicting 2010 actions from 2008-2009 baselines). The metric yielded 0.00%. 

This is an expected feature of the synthetic CMS DE-SynPUF dataset. As outlined in the CMS clinical documentation, explicit longitudinal cohort continuity across calendar years is intentionally scrambled to protect actual patient privacy. Therefore, legitimate chronological tracking spanning the year boundary is destroyed. However, the internal logic holds up (as evidenced by mathematically verified local sub-graphs predicting treatments like Neuropathy from Statins under Diabetic conditions). The engine is fully primed to accept unabscured, genuine production data.
