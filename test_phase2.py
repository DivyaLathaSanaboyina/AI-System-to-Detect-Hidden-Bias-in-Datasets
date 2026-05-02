# test_phase2.py
"""Run this to verify Phase 2 works correctly."""

import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ── Clear corrupted cache if it exists ────────────────────────────────────────
for f in ["data/adult.csv", "data/german_credit.csv"]:
    if os.path.exists(f):
        os.remove(f)
        print(f"🗑️  Cleared cache: {f}")

from src.data_loader import DataLoader

print("\n" + "🔵 " * 20)
print("TESTING ADULT INCOME DATASET")
print("🔵 " * 20)

adult_loader = DataLoader("adult")
adult_df = adult_loader.load()
adult_loader.describe()

print("\n" + "🟡 " * 20)
print("TESTING GERMAN CREDIT DATASET")
print("🟡 " * 20)

german_loader = DataLoader("german")
german_df = german_loader.load()
german_loader.describe()

# ── Verification checks ───────────────────────────────────────────────────────
print("\n📋 VERIFICATION CHECKS:")
print(f"  Adult shape:              {adult_df.shape}")
print(f"  Adult income values:      {adult_df['income'].unique()}")
print(f"  Adult missing vals:       {adult_df.isnull().sum().sum()}")
print(f"  German shape:             {german_df.shape}")
print(f"  German credit_risk vals:  {sorted(german_df['credit_risk'].unique())}")
print(f"  German sex values:        {sorted(german_df['sex'].unique())}")
print(f"  German age_group values:  {sorted(german_df['age_group'].unique())}")
print("\n✅ Phase 2 verification complete!")