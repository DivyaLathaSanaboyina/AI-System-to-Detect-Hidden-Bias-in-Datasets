# test_phase7.py
"""Phase 7: Auto Sensitive Attribute Detection"""

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

from src.data_loader       import DataLoader
from src.auto_sensitive    import AutoSensitiveDetector

# ══════════════════════════════════════════════════════════════════════════════
# ADULT INCOME DATASET
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "🔵 "*20)
print("PHASE 7: ADULT INCOME — AUTO SENSITIVE DETECTION")
print("🔵 "*20)

adult_df = DataLoader("adult").load()

adult_detector = AutoSensitiveDetector(
    df              = adult_df,
    target_col      = "income",
    known_sensitive = ["sex", "race"]   # Used for proxy detection
)
adult_results = adult_detector.run_full_detection()
adult_detector.print_report(adult_results)

# Show what the system would discover WITHOUT prior knowledge
print("\n🧪 BLIND TEST — What if we didn't know sensitive attributes?")
print("─"*60)
blind_detector = AutoSensitiveDetector(
    df              = adult_df,
    target_col      = "income",
    known_sensitive = []     # No prior knowledge given
)
blind_results  = blind_detector.run_full_detection()
di             = blind_results["disparate_impact"]
flagged_di     = di[di["flagged"]].head(5) if not di.empty else None

print("Top 5 features with worst Disparate Impact (blind scan):")
if flagged_di is not None and not flagged_di.empty:
    for _, row in flagged_di.iterrows():
        print(f"  {row['feature']:<25} DIR={row['disparate_impact']:.3f}  "
              f"({row['min_rate_group']} vs {row['max_rate_group']})")

# ══════════════════════════════════════════════════════════════════════════════
# GERMAN CREDIT DATASET
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "🟡 "*20)
print("PHASE 7: GERMAN CREDIT — AUTO SENSITIVE DETECTION")
print("🟡 "*20)

german_df = DataLoader("german").load()

german_detector = AutoSensitiveDetector(
    df              = german_df,
    target_col      = "credit_risk",
    known_sensitive = ["sex", "age_group"]
)
german_results = german_detector.run_full_detection()
german_detector.print_report(german_results)

# ══════════════════════════════════════════════════════════════════════════════
# RESEARCH INSIGHT
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "🔴 "*20)
print("🔑 THE PROXY PROBLEM — Key Research Insight")
print("🔴 "*20)
print("""
  "Fairness Through Unawareness" is a MYTH.

  Simply removing 'sex' and 'race' from your feature set does NOT
  make your model fair, because proxy features remain:

  In the Adult dataset:
    → 'relationship' contains 'Wife' — a direct proxy for sex=Female
    → 'marital_status' is correlated with sex
    → 'occupation' has gender-segregated distribution

  A model trained WITHOUT sex/race but WITH relationship/occupation
  will STILL discriminate against women — it just does it indirectly.

  This is called INDIRECT DISCRIMINATION and is illegal under
  EU's GDPR Article 22 and US Equal Credit Opportunity Act.

  Our auto-detector finds these proxies automatically.
  This is what separates a toy project from a real AI audit system.
""")

print("✅ Phase 7 complete!")
print("   → Next: Phase 8 — Bias Mitigation (fixing what we found)")