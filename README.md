# ⚖️ AI System to Detect Hidden Bias in Datasets

> **A research-grade AI fairness auditing system** that automatically detects, quantifies, explains, and mitigates hidden bias in machine learning pipelines — built as a Final Year B.Tech Project.

---

## 🎯 Project Overview

Most machine learning systems are evaluated purely on accuracy. This project proves that **accuracy is not enough** — a model can be 86% accurate and still be illegally discriminatory against women and minorities.

This system acts as a **complete AI auditing pipeline** that:
- Detects bias before and after model training
- Quantifies it using 3 industry-standard fairness metrics
- Explains *why* it occurs using SHAP (SHapley Additive exPlanations)
- Mitigates it using 4 complementary strategies
- Presents everything in an interactive web dashboard

---

## 🔬 Key Research Findings

| Finding | Evidence |
|---------|----------|
| All 3 models biased on Adult dataset | DIR = 0.268–0.333 (legal threshold: 0.80) |
| Higher accuracy ≠ fairer model | Decision Tree: best accuracy (0.863), still 🚨 BIASED |
| German RF paradox | Lowest accuracy (0.695) but ✅ FAIR — accuracy paradox proven |
| #1 bias driver identified | `marital_status_Married-civ-spouse` SHAP diff = **+0.995** between Male/Female |
| Proxy discrimination found | `relationship` (Cramer's V = 0.647 with sex) encodes gender indirectly |
| Legal violation confirmed | DIR < 0.80 = adverse impact under EEOC 29 CFR Part 1607 |

---

## 📊 Datasets

| Dataset | Records | Features | Task | Sensitive Attributes |
|---------|---------|----------|------|----------------------|
| **Adult Income** (UCI) | 48,842 | 14 | Predict income >$50K | Sex, Race |
| **German Credit** (UCI) | 1,000 | 20 | Predict credit risk | Sex, Age Group |

---

## 🏗️ Project Structure

```
bias_detection_system/
│
├── data/                        # Cached datasets (auto-downloaded)
│   ├── adult.csv
│   └── german_credit.csv
│
├── src/                         # Core source modules
│   ├── data_loader.py           # Dataset loading + documentation
│   ├── preprocessor.py          # Imputation, encoding, scaling
│   ├── model_trainer.py         # LR, DT, RF training + metrics
│   ├── bias_detector.py         # Group-wise bias analysis
│   ├── fairness_metrics.py      # DPD, EOD, DIR computation
│   ├── bias_mitigator.py        # 4 mitigation strategies
│   ├── explainer.py             # SHAP + feature importance
│   ├── auto_sensitive.py        # Auto sensitive attribute detection
│   └── visualizer.py            # All charts and dashboards
│
├── app/
│   └── streamlit_app.py         # Interactive web UI
│
├── reports/
│   └── figures/                 # Generated charts (PNG)
│
├── models/                      # Saved trained models (PKL)
├── config.py                    # All constants and settings
├── main.py                      # Master pipeline runner
└── requirements.txt
```

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/ai-bias-detection-system.git
cd ai-bias-detection-system

# 2. Create and activate virtual environment
python -m venv bias_env

# Windows (PowerShell)
.\bias_env\Scripts\Activate.ps1

# Mac/Linux
source bias_env/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### Run the full pipeline (command line)
```bash
# Audit the Adult Income dataset
python main.py --dataset adult

# Audit the German Credit dataset
python main.py --dataset german
```

### Launch the interactive dashboard
```bash
streamlit run app/streamlit_app.py
```
Open your browser at `http://localhost:8501`

### Run individual phases
```bash
python test_phase2.py       # Dataset loading
python test_phase4.py       # Model training
python test_phase5_6.py     # Bias detection + fairness metrics
python test_phase7.py       # Auto sensitive attribute detection
python test_phase8.py       # Bias mitigation
python test_phase9.py       # SHAP explainability
python test_phase10_11.py   # Multi-model comparison + visualization
```

---

## 🧩 Core Features

### 1. 🔍 Bias Detection
Group-wise performance analysis that computes prediction rates, TPR, FPR, and accuracy separately for each demographic group — revealing disparate treatment before and after model training.

### 2. 📐 Fairness Metrics (Phase 5 & 6)

| Metric | Formula | Ideal | Legal Threshold |
|--------|---------|-------|-----------------|
| **Demographic Parity Difference** | P(Ŷ=1\|priv) − P(Ŷ=1\|unpriv) | 0.0 | \|DPD\| < 0.10 |
| **Equal Opportunity Difference** | TPR(priv) − TPR(unpriv) | 0.0 | \|EOD\| < 0.10 |
| **Disparate Impact Ratio** | P(Ŷ=1\|unpriv) / P(Ŷ=1\|priv) | 1.0 | DIR > 0.80 (EEOC) |

### 3. 🤖 Auto Sensitive Attribute Detection (Phase 7)
Automatically discovers bias-prone features using 3 statistical methods — no human labeling required:
- **Chi-squared + Cramer's V**: Target correlation analysis
- **Proxy Detection**: Finds features encoding sensitive attributes indirectly
- **Disparate Impact Scan**: Runs DIR on every feature automatically

### 4. 🛠️ Bias Mitigation — 4 Strategies (Phase 8)

| Strategy | Stage | Method |
|----------|-------|--------|
| **Re-sampling (SMOTE)** | Pre-processing | Creates synthetic samples for underrepresented groups |
| **Re-weighting** | Pre-processing | Kamiran & Calders (2012) sample weight correction |
| **Fair Classifier** | In-processing | Fairlearn's ExponentiatedGradient with DemographicParity constraint |
| **Threshold Adjustment** | Post-processing | Hardt et al. (2016) group-specific decision thresholds |

### 5. 🧠 Explainability (Phase 9)
- **Model Feature Importance**: Which features drive predictions?
- **SHAP Summary**: Mean |SHAP| per feature across all predictions
- **Group-Differential SHAP**: Compares SHAP values between privileged and unprivileged groups — directly answers *why* the model discriminates

### 6. 📊 Streamlit Dashboard (Phase 12)
5-tab interactive web application with dataset overview, model training, bias detection charts, mitigation comparison, and a downloadable CSV audit report.

---

## 📈 Results Summary

### Adult Income Dataset — Sex Bias

| Model | Accuracy | DIR (sex) | Status |
|-------|----------|-----------|--------|
| Logistic Regression | 0.849 | 0.293 | 🚨 BIASED |
| Decision Tree | **0.863** | 0.333 | 🚨 BIASED |
| Random Forest | 0.860 | 0.268 | 🚨 BIASED |

**After Threshold Adjustment:** DIR improved from 0.293 → 0.452 with only 0.003 accuracy loss.

### German Credit Dataset — Age Bias

| Model | Accuracy | DIR (age) | Status |
|-------|----------|-----------|--------|
| Logistic Regression | 0.700 | 0.934 | 🚨 BIASED |
| Decision Tree | 0.700 | 0.822 | 🚨 BIASED |
| Random Forest | **0.695** | **0.895** | ✅ FAIR |

**Key Insight:** The least accurate model (RF) is the fairest — proving the accuracy-fairness trade-off empirically.

---

## 🛠️ Tech Stack

| Category | Libraries |
|----------|-----------|
| Data Processing | `pandas`, `numpy`, `scipy` |
| Machine Learning | `scikit-learn` |
| Fairness | `fairlearn`, `aif360` |
| Explainability | `shap`, `lime` |
| Resampling | `imbalanced-learn` (SMOTE) |
| Visualization | `matplotlib`, `seaborn`, `plotly` |
| Dashboard | `streamlit` |
| Model Persistence | `joblib` |

---

## 📚 Research References

1. **Hardt, M., Price, E., & Srebro, N.** (2016). *Equality of Opportunity in Supervised Learning.* NeurIPS.
2. **Kamiran, F., & Calders, T.** (2012). *Data preprocessing techniques for classification without discrimination.* KAIS.
3. **Agarwal, A., et al.** (2018). *A Reductions Approach to Fair Classification.* ICML.
4. **Lundberg, S., & Lee, S.** (2017). *A Unified Approach to Interpreting Model Predictions.* NeurIPS.
5. **Feldman, M., et al.** (2015). *Certifying and Removing Disparate Impact.* KDD.
6. **EEOC 80% Rule** — 29 CFR Part 1607 (Uniform Guidelines on Employee Selection Procedures)

---

## 👥 Authors

**[Your Name]** & **[Friend's Name]**
B.Tech Final Year Project — [Your College Name], [Year]

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.

---

*"A model can be 86% accurate and still be illegally discriminatory. Accuracy is not fairness."*
