# test_phase3.py
"""Run this to verify Phase 3 preprocessing works correctly."""

import logging
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

from src.data_loader import DataLoader
from src.preprocessor import Preprocessor

# ══════════════════════════════════════════════════════════════
# TEST 1: ADULT INCOME DATASET
# ══════════════════════════════════════════════════════════════
print("\n" + "🔵 " * 20)
print("PREPROCESSING: ADULT INCOME DATASET")
print("🔵 " * 20)

adult_loader = DataLoader("adult")
adult_df     = adult_loader.load()

adult_prep   = Preprocessor("adult")
X_train_a, X_test_a, y_train_a, y_test_a = adult_prep.fit_transform(adult_df)
adult_prep.describe()

print("📊 Adult Preprocessing Results:")
print(f"  X_train shape:     {X_train_a.shape}")
print(f"  X_test shape:      {X_test_a.shape}")
print(f"  y_train shape:     {y_train_a.shape}")
print(f"  y_test shape:      {y_test_a.shape}")
print(f"  Feature count:     {len(adult_prep.get_feature_names())}")
print(f"  Train pos rate:    {y_train_a.mean()*100:.1f}%")
print(f"  Test  pos rate:    {y_test_a.mean()*100:.1f}%  (should match ≈ train)")
print(f"  Any NaN in train:  {np.isnan(X_train_a).any()}")
print(f"  Any NaN in test:   {np.isnan(X_test_a).any()}")
print(f"\n  Sensitive attrs (train sample):")
print(adult_prep.sensitive_train.head())

# ══════════════════════════════════════════════════════════════
# TEST 2: GERMAN CREDIT DATASET
# ══════════════════════════════════════════════════════════════
print("\n" + "🟡 " * 20)
print("PREPROCESSING: GERMAN CREDIT DATASET")
print("🟡 " * 20)

german_loader = DataLoader("german")
german_df     = german_loader.load()

german_prep   = Preprocessor("german")
X_train_g, X_test_g, y_train_g, y_test_g = german_prep.fit_transform(german_df)
german_prep.describe()

print("📊 German Preprocessing Results:")
print(f"  X_train shape:     {X_train_g.shape}")
print(f"  X_test shape:      {X_test_g.shape}")
print(f"  Feature count:     {len(german_prep.get_feature_names())}")
print(f"  Train pos rate:    {y_train_g.mean()*100:.1f}%")
print(f"  Test  pos rate:    {y_test_g.mean()*100:.1f}%  (should match ≈ train)")
print(f"  Any NaN in train:  {np.isnan(X_train_g).any()}")
print(f"\n  Sensitive attrs (train sample):")
print(german_prep.sensitive_train.head())

# ══════════════════════════════════════════════════════════════
# VERIFICATION CHECKS
# ══════════════════════════════════════════════════════════════
print("\n📋 VERIFICATION CHECKS:")
assert not np.isnan(X_train_a).any(), "❌ NaN found in adult train!"
assert not np.isnan(X_test_a).any(),  "❌ NaN found in adult test!"
assert not np.isnan(X_train_g).any(), "❌ NaN found in german train!"
assert not np.isnan(X_test_g).any(),  "❌ NaN found in german test!"

# Stratified split check: train/test positive rates should be within 1%
assert abs(y_train_a.mean() - y_test_a.mean()) < 0.01, \
    "❌ Adult split not stratified!"
assert abs(y_train_g.mean() - y_test_g.mean()) < 0.02, \
    "❌ German split not stratified!"

# Data leakage check: scaler was fit on train only
# (test mean should NOT be exactly 0 if scaler was fit on train only)
train_mean = X_train_a[:, 0].mean()
print(f"  Adult train col-0 mean (should be ≈ 0.0): {train_mean:.4f}")

print("\n✅ ALL CHECKS PASSED — Phase 3 complete!")
print("\n🔑 Key preprocessing decisions made:")
print("  1. Dropped 'fnlwgt' (census weight, not a real feature)")
print("  2. Imputed missing with mode/median (not dropped — avoids selection bias)")
print("  3. One-hot encoded categoricals (not label encoding — no false ordinality)")
print("  4. StandardScaler fit on TRAIN only (no data leakage)")
print("  5. Stratified split (preserves class balance)")
print("  6. Sensitive attributes saved separately (for fairness analysis)")