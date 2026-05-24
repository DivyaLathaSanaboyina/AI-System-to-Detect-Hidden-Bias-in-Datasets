# test_phase8.py
"""Phase 8: Bias Mitigation — Before vs After Comparison"""

import logging
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

from src.data_loader    import DataLoader
from src.preprocessor   import Preprocessor
from src.bias_mitigator import BiasMitigator

def run_mitigation(dataset_name, sensitive_col, privileged_val):
    print(f"\n{'═'*68}")
    print(f"  🔧 MITIGATION PIPELINE: {dataset_name.upper()} | [{sensitive_col}]")
    print(f"{'═'*68}\n")

    loader = DataLoader(dataset_name)
    df     = loader.load()

    prep   = Preprocessor(dataset_name)
    X_train, X_test, y_train, y_test = prep.fit_transform(df)

    mitigator = BiasMitigator(
        X_train         = X_train,
        X_test          = X_test,
        y_train         = y_train,
        y_test          = y_test,
        sensitive_train = prep.sensitive_train[sensitive_col],
        sensitive_test  = prep.sensitive_test[sensitive_col],
        sensitive_col   = sensitive_col,
        privileged_val  = privileged_val,
    )

    results = mitigator.run_all()
    mitigator.print_comparison(results)
    return results

# ── Adult: sex attribute ───────────────────────────────────────────────────────
print("\n" + "🔵 "*20)
print("PHASE 8: ADULT INCOME — MITIGATION ON [sex]")
print("🔵 "*20)

adult_sex_results = run_mitigation("adult", "sex", "Male")

# ── Adult: race attribute ──────────────────────────────────────────────────────
print("\n" + "🔵 "*20)
print("PHASE 8: ADULT INCOME — MITIGATION ON [race]")
print("🔵 "*20)

adult_race_results = run_mitigation("adult", "race", "White")

# ── German: age_group attribute ───────────────────────────────────────────────
print("\n" + "🟡 "*20)
print("PHASE 8: GERMAN CREDIT — MITIGATION ON [age_group]")
print("🟡 "*20)

german_age_results = run_mitigation("german", "age_group", "adult")

# ── Final research summary ────────────────────────────────────────────────────
print("\n" + "🔴 "*20)
print("🔑 MITIGATION STRATEGY GUIDE — When to use what")
print("🔴 "*20)
print("""
  ┌─────────────────────────────────────────────────────────────────┐
  │ STRATEGY          │ STAGE  │ BEST WHEN                         │
  ├─────────────────────────────────────────────────────────────────┤
  │ Re-sampling       │ Pre    │ Minority group is underrepresented │
  │ (SMOTE)           │        │ in training data                  │
  ├─────────────────────────────────────────────────────────────────┤
  │ Re-weighting      │ Pre    │ You want to keep all data + no    │
  │ (Kamiran)         │        │ synthetic samples; fast & clean   │
  ├─────────────────────────────────────────────────────────────────┤
  │ Fair Classifier   │ In     │ Most principled approach; when    │
  │ (ExpGrad)         │        │ retraining is acceptable          │
  ├─────────────────────────────────────────────────────────────────┤
  │ Threshold Adjust  │ Post   │ Model already deployed; works     │
  │ (Hardt)           │        │ with ANY model; fastest to deploy │
  └─────────────────────────────────────────────────────────────────┘

  ACCURACY vs FAIRNESS TRADE-OFF (expected for Adult/sex):
    Baseline:    Acc≈0.849 | DIR≈0.29  → 🚨 Biased
    Resampling:  Acc≈0.830 | DIR≈0.45  → improvement
    Reweighting: Acc≈0.835 | DIR≈0.55  → improvement
    FairClass:   Acc≈0.820 | DIR≈0.75  → near-fair
    Threshold:   Acc≈0.815 | DIR≈0.80  → ✅ Fair threshold

  Every mitigation strategy costs some accuracy — that's the price
  of fairness. The question is: what trade-off is acceptable?
  This is a BUSINESS and ETHICAL decision, not a technical one.
""")

print("✅ Phase 8 complete!")
print("   → Next: Phase 9 — Explainability (SHAP + Feature Importance)")