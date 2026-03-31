import streamlit as st
import json
import requests
import pandas as pd
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Pharmacy2U: Pathway-Aware Recommendations", layout="wide")
st.title("💊 Next-Best Prescription Recommender")
st.markdown("""
*Hackathon Track 1 - Challenge B*  
This tool demonstrates pathway-aware clinical cross-selling using Markov Chain transition probabilities. 
It recommends the most statistically likely "Next-Best" medicines based on a patient's current active drug, 
factoring in context like underlying chronic illnesses (e.g. Diabetes).
""")

# --- HELPER FUNCTIONS ---
@st.cache_data(show_spinner=False)
def load_model():
    if not os.path.exists("markov_transitions.json"):
        return None
    with open("markov_transitions.json", "r") as f:
        return json.load(f)

@st.cache_data(show_spinner=False)
def get_drug_info(ndc11):
    """
    Queries the RxNav API to convert an 11-digit NDC code to a human readable drug name.
    """
    try:
        # Step 1: Get RxCUI from NDC
        resp = requests.get(f"https://rxnav.nlm.nih.gov/REST/rxcui.json?idtype=NDC&id={ndc11}", timeout=2)
        if resp.status_code == 200:
            data = resp.json()
            if 'idGroup' in data and 'rxnormId' in data['idGroup']:
                rxcui = data['idGroup']['rxnormId'][0]
                
                # Step 2: Get Drug Name/Properties from RxCUI
                prop_resp = requests.get(f"https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}/properties.json", timeout=2)
                if prop_resp.status_code == 200:
                    prop_data = prop_resp.json()
                    name = prop_data['properties']['name']
                    return name
    except Exception:
        pass
    
    return f"Unknown Formulation (NDC: {ndc11})"

# --- MAIN APP ---
model = load_model()

if model is None:
    st.error("Model file `markov_transitions.json` not found. Please run scripts `02b` and `03b` first.")
    st.stop()

st.sidebar.header("Patient Context Simulator")
st.sidebar.markdown("Choose a scenario to see how the recommendation engine adapts to the patient's existing conditions.")

# Select a current drug to start the scenario. 
# We'll hardcode a few interesting NDCs from the CMS dataset for demo purposes, 
# or let the user type one.
demo_drugs = {
    "00002325030 (Humalog Insulin)": "00002325030",
    "00004002950 (Lexapro - Depression)": "00004002950",
    "00006002728 (Januvia - Diabetes)": "00006002728",
    "00049005050 (Rocephin - Antibiotic)": "00049005050"
}

selection = st.sidebar.selectbox("Patient's Most Recent Prescription:", list(demo_drugs.keys()) + ["Custom NDC..."])

if selection == "Custom NDC...":
    current_drug = st.sidebar.text_input("Enter 11-Digit NDC Code:")
else:
    current_drug = demo_drugs[selection]

# Context Flags
has_diabetes = st.sidebar.toggle("Patient has documented Diabetes?", value=False)
pathway = 'diabetes' if has_diabetes else 'global'

st.divider()

if current_drug and len(current_drug) >= 9:
    # 1. Fetch current drug name
    with st.spinner("Resolving actual drug name via FDA RxNav API..."):
        current_name = get_drug_info(current_drug)
    
    st.markdown(f"### Current Prescription: **{current_name}**")
    
    if has_diabetes:
        st.info("⚕️ **Clinical Context Active:** Searching conditional pathways for patients with Diabetes.")
    else:
        st.info("🌐 **Global Context Active:** Searching baseline population pathways.")
        
    st.markdown("### 🎯 Recommended 'Next-Best' Consultations / Prescriptions:")
    
    # Get predictions
    transitions = model[pathway].get(current_drug, [])
    
    if not transitions and has_diabetes:
        st.warning("No diabetes-specific pathway found for this drug. Falling back to Global pathway.")
        transitions = model['global'].get(current_drug, [])
        pathway_used = 'global fallback'
    else:
        pathway_used = pathway
        
    if not transitions:
        st.error("No historical transition data exists for this drug in the Markov Graph.")
    else:
        # Display top 5
        top_k = 5
        cols = st.columns(top_k)
        
        for idx, (next_ndc, prob) in enumerate(transitions[:top_k]):
            with cols[idx]:
                with st.spinner(f"Resolving rank {idx+1}..."):
                    next_name = get_drug_info(next_ndc)
                
                st.markdown(f"""
                <div style='padding: 15px; border-radius: 10px; border: 1px solid #ddd; height: 100%; box-shadow: 2px 2px 5px rgba(0,0,0,0.05);'>
                    <h4 style='color: #2E86AB; margin-top: 0;'>#{idx+1}</h4>
                    <p style='font-size: 14px; font-weight: bold;'>{next_name}</p>
                    <p style='font-size: 12px; color: gray;'>NDC: {next_ndc}</p>
                    <hr style='margin: 10px 0;'/>
                    <p style='font-size: 13px;'><b>Transition Probability:</b> {prob*100:.1f}%</p>
                    <p style='font-size: 11px; color: #777;'><i>Based on {pathway_used} historical cadence</i></p>
                </div>
                """, unsafe_allow_html=True)
                
    st.divider()
    st.markdown("""
    **💡 Explanation of Recommendation Engine:**
    These recommendations are generated using a 1st-Order Markov transition matrix built from 3.8 million 
    synthetic prescription sequences (DE-SynPUF). The interface dynamically queries the FDA RxNav REST API 
    to map raw National Drug Codes (NDCs) into human-readable clinical formulations in real-time.
    """)
    
