# test_phase5_6.py
"""
Phase 5 & 6: Bias Detection + Fairness Metrics
Run this after Phase 4 to get the complete bias audit report.
"""

import logging
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

from src.data_loader    import DataLoader
from src.preprocessor   import Preprocessor
from src.model_trainer  import ModelTrainer
from src.bias_detector  import BiasDetector

# ══════════════════════════════════════════════════════════════════════════════
# HELPER: run one full pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_bias_audit(dataset_name: str, privileged_groups: dict):
    print(f"\n{'═'*68}")
    print(f"  🔎 BIAS AUDIT PIPELINE: {dataset_name.upper()}")
    print(f"{'═'*68}\n")

    # ── Load + preprocess ──────────────────────────────────────────────────
    loader = DataLoader(dataset_name)
    df     = loader.load()

    prep   = Preprocessor(dataset_name)
    X_train, X_test, y_train, y_test = prep.fit_transform(df)

    # ── Train models ───────────────────────────────────────────────────────
    trainer = ModelTrainer()
    results = trainer.train_all(X_train, X_test, y_train, y_test,
                                dataset_name=dataset_name)

    # ── Bias audit for each model ──────────────────────────────────────────
    detector = BiasDetector(
        sensitive_df = prep.sensitive_test,
        y_true       = y_test,
        dataset_name = dataset_name
    )

    all_reports = []
    for model_name in results:
        y_pred, _ = trainer.get_predictions(model_name)
        report    = detector.run_full_audit(
            model_name       = model_name,
            y_pred           = y_pred,
            privileged_groups= privileged_groups
        )
        detector.print_report(report)
        all_reports.append(report)

    # ── Cross-model fairness comparison table ──────────────────────────────
    comparison_df = detector.compare_models(all_reports)
    print(f"\n{'═'*68}")
    print(f"  📊 CROSS-MODEL FAIRNESS COMPARISON — {dataset_name.upper()}")
    print(f"{'═'*68}")
    print(comparison_df.to_string(index=False))

    # ── Highlight the accuracy-fairness trade-off ──────────────────────────
    print(f"\n{'─'*68}")
    print("  ⚖️  ACCURACY vs FAIRNESS TRADE-OFF SUMMARY")
    print(f"{'─'*68}")
    print(f"  {'Model':<25} {'Accuracy':>10} {'Worst DPD':>11} {'Worst DIR':>11} {'Biased?':>10}")
    print(f"  {'─'*65}")

    acc_map = {n: r["accuracy"] for n, r in results.items()}
    for report in all_reports:
        model = report["model"]
        acc   = acc_map[model]
        # Take first attribute's summary for display
        first_attr = list(report["attributes"].values())[0]
        summary    = first_attr["fairness_metrics"].get("summary", {})
        dpd        = summary.get("worst_dpd", float("nan"))
        dir_       = summary.get("worst_dir", float("nan"))
        biased     = "🚨 YES" if summary.get("is_biased") else "✅ NO"
        print(f"  {model:<25} {acc:>10.3f} {dpd:>+11.3f} {dir_:>11.3f} {biased:>10}")

    print(f"\n  Key insight: Higher accuracy does NOT mean less bias!")
    print(f"{'═'*68}\n")

    return all_reports, trainer, prep


# ══════════════════════════════════════════════════════════════════════════════
# RUN BOTH DATASETS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "🔵 "*20)
print("PHASE 5 & 6: ADULT INCOME — BIAS DETECTION")
print("🔵 "*20)

adult_reports, adult_trainer, adult_prep = run_bias_audit(
    dataset_name      = "adult",
    privileged_groups = {"sex": "Male", "race": "White"}
)

print("\n" + "🟡 "*20)
print("PHASE 5 & 6: GERMAN CREDIT — BIAS DETECTION")
print("🟡 "*20)

german_reports, german_trainer, german_prep = run_bias_audit(
    dataset_name      = "german",
    privileged_groups = {"sex": "male", "age_group": "adult"}
)

# ══════════════════════════════════════════════════════════════════════════════
# FINAL INSIGHT SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "🔴 "*20)
print("🔑 RESEARCH-LEVEL FINDINGS SUMMARY")
print("🔴 "*20)
print("""
  METRIC INTERPRETATION GUIDE:
  ─────────────────────────────────────────────────────────────────────
  Demographic Parity Diff (DPD):
    → Measures: Do both groups get predicted '1' at the same rate?
    → Formula:  P(Ŷ=1|Male) - P(Ŷ=1|Female)
    → Ideal: 0.0 | Threshold: |DPD| < 0.10
    → Positive value means privileged group gets MORE positive predictions

  Equal Opportunity Diff (EOD):
    → Measures: Among true positives, are both groups found equally?
    → Formula:  TPR(Male) - TPR(Female)
    → Ideal: 0.0 | Threshold: |EOD| < 0.10
    → Critical for high-stakes decisions (loans, hiring, bail)

  Disparate Impact Ratio (DIR):
    → Measures: Relative positive rate between groups
    → Formula:  P(Ŷ=1|Female) / P(Ŷ=1|Male)
    → Ideal: 1.0 | Legal threshold: DIR > 0.80 (EEOC 80% rule)
    → DIR < 0.80 = evidence of illegal discrimination in US employment

  ─────────────────────────────────────────────────────────────────────
  WHY BIAS EXISTS IN THESE DATASETS:
    Adult:  1994 census data captures a period of rampant wage discrimination.
            The model learns that 'Female' and 'Black' correlate with <=50K
            because they DID in 1994 — and encodes that as a 'rule'.

    German: Age bias occurs because younger people have shorter credit
            histories, fewer assets, and lower income — all proxies for age.
            The model learns age discrimination 'for free' through proxies.
  ─────────────────────────────────────────────────────────────────────
  NEXT STEP: Phase 7 (Auto-detect sensitive attributes) +
             Phase 8 (Bias Mitigation — fix the bias we just measured)
""")

print("✅ Phase 5 & 6 complete!")