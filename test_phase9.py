# test_phase9.py
"""Phase 9: SHAP Explainability — Why does bias occur?"""

import logging
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

import numpy as np
from src.data_loader    import DataLoader
from src.preprocessor   import Preprocessor
from src.model_trainer  import ModelTrainer
from src.explainer      import Explainer

def run_explanation(dataset_name, sensitive_col, privileged_val, model_name):
    print(f"\n{'═'*68}")
    print(f"  🧠 EXPLANATION: {dataset_name.upper()} | {model_name} | [{sensitive_col}]")
    print(f"{'═'*68}\n")

    # ── Load + preprocess ──────────────────────────────────────────────────
    loader = DataLoader(dataset_name)
    df     = loader.load()
    prep   = Preprocessor(dataset_name)
    X_train, X_test, y_train, y_test = prep.fit_transform(df)

    # ── Train models ───────────────────────────────────────────────────────
    trainer = ModelTrainer()
    trainer.train_all(X_train, X_test, y_train, y_test, dataset_name)
    model   = trainer.get_model(model_name)

    # ── SHAP explanation ───────────────────────────────────────────────────
    explainer = Explainer(
        model        = model,
        X_train      = X_train,
        X_test       = X_test,
        feature_names= prep.get_feature_names(),
        dataset_name = f"{dataset_name}_{model_name.lower().replace(' ','_')}",
        sample_size  = 300
    )

    sensitive_test = prep.sensitive_test[sensitive_col].values

    results = explainer.run_full_explanation(
        sensitive_test = sensitive_test,
        privileged_val = privileged_val,
        sensitive_col  = sensitive_col
    )

    return results

# ══════════════════════════════════════════════════════════════════════════════
# ADULT INCOME — LOGISTIC REGRESSION (most interpretable)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "🔵 "*20)
print("PHASE 9: ADULT INCOME — SHAP with Logistic Regression")
print("🔵 "*20)

adult_lr_results = run_explanation(
    dataset_name  = "adult",
    sensitive_col = "sex",
    privileged_val= "Male",
    model_name    = "Logistic Regression"
)

# ══════════════════════════════════════════════════════════════════════════════
# ADULT INCOME — RANDOM FOREST (highest AUC)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "🔵 "*20)
print("PHASE 9: ADULT INCOME — SHAP with Random Forest")
print("🔵 "*20)

adult_rf_results = run_explanation(
    dataset_name  = "adult",
    sensitive_col = "sex",
    privileged_val= "Male",
    model_name    = "Random Forest"
)

# ══════════════════════════════════════════════════════════════════════════════
# GERMAN — LOGISTIC REGRESSION on age_group
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "🟡 "*20)
print("PHASE 9: GERMAN CREDIT — SHAP on age_group")
print("🟡 "*20)

german_results = run_explanation(
    dataset_name  = "german",
    sensitive_col = "age_group",
    privileged_val= "adult",
    model_name    = "Logistic Regression"
)

# ══════════════════════════════════════════════════════════════════════════════
# KEY INSIGHT SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "🔴 "*20)
print("🔑 SHAP-BASED BIAS EXPLANATION — Research Summary")
print("🔴 "*20)
print("""
  HOW TO READ THE GROUP-DIFFERENTIAL SHAP CHART:
  ─────────────────────────────────────────────────────────────────────
  • Each bar = one feature
  • Bar direction:
    → RIGHT (blue) = this feature benefits MALES more than females
    → LEFT  (red)  = this feature benefits FEMALES more than males

  • Expected findings for Adult/sex:
    → 'relationship_Husband' has large POSITIVE SHAP for males
      (being a husband pushes income prediction UP — for men only)
    → 'relationship_Wife' has NEGATIVE SHAP for females
      (being a wife pulls income prediction DOWN)
    → 'marital_status_Married-civ-spouse' favors males
      (married men earn more in 1994 data; married women earn less)

  These features are the MECHANISM of discrimination.
  The model didn't learn to discriminate against sex directly —
  it learned to discriminate against 'Wife' and 'married female'
  because that's what the 1994 census data taught it.

  ─────────────────────────────────────────────────────────────────────
  SHAP vs Feature Importance — Key Difference:
    Feature Importance: "capital_gain matters a lot overall"
    SHAP:               "capital_gain pushes THIS person's prediction
                         by +0.23 toward >50K"

  SHAP is per-prediction, directional, and group-comparable.
  That's why it's the gold standard for bias explanation.
  ─────────────────────────────────────────────────────────────────────
""")

print("✅ Phase 9 complete!")
print("   Figures saved to: reports/figures/")
print("   → Next: Phase 10 & 11 — Multi-model comparison + Visualization")