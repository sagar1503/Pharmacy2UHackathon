# 💊 Late Prescription Refill Risk Prediction

> **Pharmacy2U Hackathon — Challenge A**  
> Predicting which patient–drug pairs are likely to refill late using synthetic CMS DE‑SynPUF data (2008–2010).

⚠️ **Note:** Data are fully synthetic and the work is for modelling/product‑thinking only, not clinical use.

---

## 🎯 Project Overview

This repository implements an end‑to‑end machine learning solution to identify patients at risk of late prescription refills. We build predictive models by engineering features from prescription histories and patient demographics, then evaluate using PR‑AUC with a time‑based split to prevent data leakage.

**What we do:**

✅ Build a **late refill** label from prescription histories  
✅ Engineer refill‑behaviour and patient‑level features  
✅ Train baseline and XGBoost models  
✅ Evaluate with **PR‑AUC** using **time‑based split**  

---

## 📊 Data Overview

**Source:** CMS 2008–2010 **DE‑SynPUF Sample 1** (synthetic Medicare‑style claims)

### Files Used

| File | Purpose | Key Columns |
|------|---------|------------|
| **Prescription Drug Events (PDE)** | Prescription history | `DESYNPUF_ID`, `SRVC_DT`, `PROD_SRVC_ID`, `QTY_DSPNSD_NUM`, `DAYS_SUPLY_NUM`, `PTNT_PAY_AMT`, `TOT_RX_CST_AMT` |
| **Beneficiary Summary 2010** | Patient demographics & conditions | `BENE_BIRTH_DT`, `BENE_SEX_IDENT_CD`, `BENE_RACE_CD`, `BENE_ESRD_IND`, `BENE_HI_CVRAGE_TOT_MONS`, `SP_*` flags |

**Dataset size:** **1,142,880** usable rows after labelling and merging

---

## 🏷️ Label Definition: "Late Refill"

We define late refill at the **patient–drug‑class level** using the middle 4 digits of the NDC code (`DRUG_MID_4`).

### Labelling Process

For each `(DESYNPUF_ID, DRUG_MID_4)` ordered by `SRVC_DT`:

1. **Compute run‑out date**  
   ```
   runout_date = SRVC_DT + DAYS_SUPLY_NUM
   ```

2. **Calculate refill gap**  
   ```
   refill_gap = next_fill_date − runout_date (in days)
   ```

3. **Mark as late** if gap > grace window (e.g., > 14 days)

4. **Exclude** the final fill in each sequence (no next fill available)

### Label Distribution

- **Total labelled rows:** 1,142,880
- **Late refill rate:** 88–92% (highly imbalanced dataset)

---

## 🔧 Features

### Refill‑Behaviour Features (8 features)

Per fill:

| Feature | Description |
|---------|-------------|
| `DAYS_SUPLY_NUM` | Days of supply dispensed |
| `QTY_DSPNSD_NUM` | Quantity dispensed |
| `PTNT_PAY_AMT` | Patient out‑of‑pocket cost |
| `TOT_RX_CST_AMT` | Total prescription cost |
| `fill_count` | Number of fills so far for that patient–drug |
| `prev_gap_days` | Previous refill gap |
| `avg_gap` | Running average refill gap |
| `num_drugs` | Count of distinct drugs per patient (polypharmacy proxy) |

### Patient‑Level Features (17 features)

Merged on `DESYNPUF_ID`:

**Demographics** (4)
- `age` — calculated from birth date
- `BENE_SEX_IDENT_CD` — sex
- `BENE_RACE_CD` — race
- `BENE_HI_CVRAGE_TOT_MONS` — Part A coverage months

**Chronic Conditions** (11)
- `SP_ALZHDMTA`, `SP_CHF`, `SP_CHRNKIDN`, `SP_CNCR`, `SP_COPD`, `SP_DEPRESSN`, `SP_DIABETES`, `SP_ISCHMCHT`, `SP_OSTEOPRS`, `SP_RA_OA`, `SP_STRKETIA`
- (Recoded: 1 = condition present, 0 = absent)

**Summaries** (2)
- `num_conditions` — count of chronic conditions
- `esrd_flag` — ESRD indicator (1 = yes, 0 = no)

---

## 🤖 Modelling & Results

### Validation Strategy

- **Time‑based split** on `SRVC_DT` (~90% early data for training, ~10% later data for testing)
- **Primary metric:** PR‑AUC (average precision) — more informative than ROC‑AUC for imbalanced classification

### Model Performance

| Model | Features | PR‑AUC | Notes |
|-------|----------|--------|-------|
| **Logistic Regression** | Refill‑only (7) | **0.7919** | Baseline |
| **XGBoost V1** | Refill‑only (7) | **0.8003** | Core features |
| **XGBoost V2** | Refill + aggregates (9) | **0.8008** | +avg_gap, num_drugs |
| **XGBoost V3** | Refill optimized (8) | **0.8008** | Removed zero‑importance feature |
| **XGBoost V4** | All features (25) | **0.9272** | 📈 **+0.12 lift with demographics & conditions** |

### Classification Metrics (V4, Threshold = 0.5)

| Class | Precision | Recall | F1  | Support |
|-------|-----------|--------|-----|---------|
| **On‑time** | 0.21 | 0.48 | 0.29 | 13,518 |
| **Late** | 0.91 | 0.75 | 0.83 | 100,770 |
| **Overall Accuracy** | — | — | — | **0.72** |

---

## 📈 Threshold Tuning

We swept thresholds from 0.10 to 0.90 to find an optimal operating point balancing precision, recall, and intervention volume.

| Threshold | Precision | Recall | F1 | % Patients Flagged |
|:---------:|:---------:|:------:|:--:|:------------------:|
| **0.10** | 0.88 | 0.999 | 0.94 | 99.8% |
| **0.35** ✓ | 0.90 | 0.895 | 0.90 | 87.3% |
| **0.50** | 0.91 | 0.752 | 0.83 | 72.5% |
| **0.60** | 0.94 | 0.211 | 0.34 | 19.8% |

### Recommended Threshold: **0.35**

✨ **Sweet spot** for production deployment:
- **Precision:** ~90% (high confidence in flagged cases)
- **Recall:** ~89% (catch most at‑risk patients)
- **F1:** ~90% (excellent balance)
- **Intervention volume:** 87% of patients (vs. 73% at threshold 0.5)

---

## 🔍 Feature Importance (XGBoost V4)

### Top Drivers of Late Refill Risk

| Rank | Feature | Category | Impact |
|:----:|---------|----------|--------|
| **1** | `DAYS_SUPLY_NUM` | Refill Behaviour | **Dominant driver** — days of supply |
| **2** | `fill_count` | Refill Behaviour | More history → clearer patterns |
| **3–5** | `prev_gap_days`, `avg_gap`, `num_drugs` | Refill Behaviour | Timing irregularity & polypharmacy |
| **6** | `age` | Demographics | Older patients at higher risk |
| **7** | `num_conditions` | Chronic Conditions | Multimorbidity signal |
| **8** | `esrd_flag` | Chronic Conditions | End‑stage renal disease (high‑need cohort) |
| **9–15** | `SP_DIABETES`, `SP_COPD`, `SP_CHF`, `SP_ALZHDMTA`, `SP_DEPRESSN`, `SP_OSTEOPRS`, `SP_RA_OA` | Chronic Conditions | Individual disease signals |
| **16–17** | `BENE_RACE_CD`, `BENE_HI_CVRAGE_TOT_MONS` | Demographics | Smaller but consistent effects |
| **18–20** | `QTY_DSPNSD_NUM`, `PTNT_PAY_AMT`, `TOT_RX_CST_AMT` | Refill Behaviour | Cost/quantity context |

### Key Insight

💡 **Refill behaviour** (days' supply, gaps, and history) contributes most of the signal. **Demographics and chronic disease burden** add meaningful incremental signal, explaining the **0.12 PR‑AUC jump** from V3 (~0.80) to V4 (~0.93).

---

## 📁 Repository Structure

```
.
├── data/                                   # (gitignored) raw CSVs from CMS
│   ├── DE1_0_2008_to_2010_Prescription_Drug_Events_Sample_1.csv
│   └── DE1_0_2010_Beneficiary_Summary_File_Sample_1.csv
│
├── notebooks/
│   └── Pharmacy2U_Project_A.ipynb          # main notebook: EDA → features → models
│
├── src/
│   ├── data_prep.py                        # loading, labelling, feature engineering
│   ├── models.py                           # model training & evaluation helpers
│   └── plotting.py                         # threshold + feature-importance plots
│
├── requirements.txt                        # Python dependencies
└── README.md                               # this file
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Git (optional, for cloning)

### Installation

1. **Download the CMS DE‑SynPUF data** from the [CMS website](https://www.cms.gov/Research-Statistics-Data-and-Systems/Downloadable-Public-Use-Files/SynPUF/index) and place CSVs in the `data/` folder.

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the notebook:**
   ```bash
   jupyter notebook notebooks/Pharmacy2U_Project_A.ipynb
   ```

### Running the Full Pipeline

The main notebook (`Pharmacy2U_Project_A.ipynb`) reproduces the entire workflow:

✔️ Data loading and exploratory data analysis (EDA)  
✔️ Late refill label construction  
✔️ Feature engineering (refill + patient features)  
✔️ Logistic regression baseline  
✔️ XGBoost V1–V4 training  
✔️ PR‑AUC evaluation with time‑based split  
✔️ Threshold sweep and optimization  
✔️ Feature importance plots  

---

## ⚠️ Limitations

- **Synthetic US Medicare data** — not representative of Pharmacy2U's UK population or modern pharmacy operations.

- **Late refill as a proxy** — does not capture intent, dose changes, clinical nuance, or patient‑prescriber communication.

- **Limited feature set** — no prescriber, hospitalisation, or social‑determinants data; predictions rely on claims and demographic attributes only.

- **Imbalanced dataset** — 88–92% late refills; standard accuracy is not informative; PR‑AUC is more meaningful.

---

## 📝 Next Steps & Recommendations

### For Production Deployment

1. **Validate on Pharmacy2U's UK data** with similar feature engineering.
2. **Integrate with pharmacy workflows** — flag at point of refill review.
3. **A/B test** the threshold (0.35 recommended) and measure impact on patient adherence.
4. **Monitor model drift** — retrain quarterly or on new cohorts.

### For Model Improvement

1. **Add prescriber features** (e.g., specialty, refill patterns).
2. **Include patient engagement signals** (e.g., app usage, communication history).
3. **Incorporate social determinants** (e.g., neighbourhood deprivation, transport access).
4. **Explore time‑series models** (e.g., LSTMs) for sequential refill patterns.
5. **Calibrate probabilities** — modern calibration methods for better decision‑making.

---

## 📚 References & Data Dictionary

- **CMS DE‑SynPUF Documentation:** [https://www.cms.gov/Research-Statistics-Data-and-Systems/Downloadable-Public-Use-Files/SynPUF](https://www.cms.gov/Research-Statistics-Data-and-Systems/Downloadable-Public-Use-Files/SynPUF)
- **NDC (National Drug Code):** [https://www.fda.gov/drugs/drug-approvals-and-databases/national-drug-code-directory](https://www.fda.gov/drugs/drug-approvals-and-databases/national-drug-code-directory)
- **Codebook (column definitions)**: https://www.cms.gov/files/document/de-10-codebook.pdf-0

---

## 📧 Contact & Acknowledgments

**Hackathon:** Pharmacy2U2‑day Hackathon  
**Challenge:** A – Late Prescription Refill Risk  
**Data:** CMS DE‑SynPUF (synthetic, for educational use)

---

<div align="center">

**Built with ❤️ for better patient medication adherence**

</div>
