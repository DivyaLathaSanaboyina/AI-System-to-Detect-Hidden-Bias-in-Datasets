# config.py
"""
Master configuration file for the Bias Detection System.
All constants, paths, and hyperparameters are defined here.
Changing one value here propagates everywhere — clean and maintainable.
"""

import os

# ─── Project Paths ────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
REPORTS_DIR     = os.path.join(BASE_DIR, "reports")
FIGURES_DIR     = os.path.join(REPORTS_DIR, "figures")
MODELS_DIR      = os.path.join(BASE_DIR, "models")

# Create directories if they don't exist
for directory in [DATA_DIR, REPORTS_DIR, FIGURES_DIR, MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)

# ─── Dataset Settings ─────────────────────────────────────────────────────────
DATASETS = {
    "adult": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        "filename": "adult.csv",
        "target_column": "income",          # What we're predicting
        "positive_label": ">50K",           # What counts as "favorable outcome"
        "sensitive_attributes": ["sex", "race"],  # Known bias-prone features
        "privileged_groups": {
            "sex": "Male",
            "race": "White"
        }
    },
    "german": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data",
        "filename": "german_credit.csv",
        "target_column": "credit_risk",
        "positive_label": 1,                # 1 = Good credit
        "sensitive_attributes": ["sex", "age_group"],
        "privileged_groups": {
            "sex": "male",
            "age_group": "adult"
        }
    }
}

# ─── Model Settings ───────────────────────────────────────────────────────────
RANDOM_STATE    = 42            # For reproducibility (CRITICAL in research)
TEST_SIZE       = 0.2           # 80/20 train-test split
CV_FOLDS        = 5             # Cross-validation folds

# Hyperparameters for each model
MODEL_PARAMS = {
    "logistic_regression": {
        "max_iter": 1000,
        "random_state": RANDOM_STATE,
        "C": 1.0                # Regularization strength
    },
    "decision_tree": {
        "max_depth": 10,
        "random_state": RANDOM_STATE,
        "min_samples_split": 20
    },
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": RANDOM_STATE,
        "n_jobs": -1            # Use all CPU cores
    }
}

# ─── Fairness Thresholds ──────────────────────────────────────────────────────
# These are industry-standard thresholds from research literature
FAIRNESS_THRESHOLDS = {
    "demographic_parity_difference": 0.10,  # Max allowed: 10% gap
    "equal_opportunity_difference":  0.10,
    "disparate_impact_ratio":        0.80,  # 80% rule (EEOC guideline)
}

# ─── SHAP Settings ────────────────────────────────────────────────────────────
SHAP_SAMPLE_SIZE = 100          # Rows to use for SHAP (speed vs accuracy)

# ─── Bias Mitigation ──────────────────────────────────────────────────────────
RESAMPLING_STRATEGY = "SMOTE"   # Synthetic Minority Over-sampling
THRESHOLD_RANGE = (0.3, 0.7)    # Range to search for optimal threshold

# ─── Visualization ────────────────────────────────────────────────────────────
FIGURE_DPI   = 150
FIGURE_SIZE  = (12, 6)
COLOR_PALETTE = {
    "privileged":   "#2196F3",  # Blue
    "unprivileged": "#F44336",  # Red
    "neutral":      "#4CAF50",  # Green
    "warning":      "#FF9800",  # Orange
}