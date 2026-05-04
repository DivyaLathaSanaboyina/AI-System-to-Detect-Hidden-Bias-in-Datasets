# test_phase4.py
"""Run this to verify Phase 4 model training works correctly."""

import logging
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

from src.data_loader   import DataLoader
from src.preprocessor  import Preprocessor
from src.model_trainer import ModelTrainer

# ══════════════════════════════════════════════════════════════
# ADULT INCOME DATASET
# ══════════════════════════════════════════════════════════════
print("\n" + "🔵 " * 20)
print("MODEL TRAINING: ADULT INCOME DATASET")
print("🔵 " * 20 + "\n")

adult_loader = DataLoader("adult")
adult_df     = adult_loader.load()

adult_prep   = Preprocessor("adult")
X_train_a, X_test_a, y_train_a, y_test_a = adult_prep.fit_transform(adult_df)

adult_trainer = ModelTrainer()
adult_results = adult_trainer.train_all(
    X_train_a, X_test_a, y_train_a, y_test_a,
    dataset_name="adult"
)
adult_trainer.print_comparison()

# ══════════════════════════════════════════════════════════════
# GERMAN CREDIT DATASET
# ══════════════════════════════════════════════════════════════
print("\n" + "🟡 " * 20)
print("MODEL TRAINING: GERMAN CREDIT DATASET")
print("🟡 " * 20 + "\n")

german_loader = DataLoader("german")
german_df     = german_loader.load()

german_prep   = Preprocessor("german")
X_train_g, X_test_g, y_train_g, y_test_g = german_prep.fit_transform(german_df)

german_trainer = ModelTrainer()
german_results = german_trainer.train_all(
    X_train_g, X_test_g, y_train_g, y_test_g,
    dataset_name="german"
)
german_trainer.print_comparison()

# ══════════════════════════════════════════════════════════════
# THE KEY INSIGHT — print this for your presentation
# ══════════════════════════════════════════════════════════════
print("\n" + "🔴 " * 20)
print("🔑 THE CRITICAL INSIGHT ABOUT ACCURACY vs FAIRNESS")
print("🔴 " * 20)
print("""
  A classifier that always predicts '<=50K' (majority class) 
  would achieve 76.1% accuracy on Adult dataset — WITH ZERO effort.
  
  This is called the 'accuracy paradox':
    → High accuracy ≠ Fair model
    → A model can be 87% accurate AND discriminate against women
  
  This is WHY we need fairness metrics (Phase 5 & 6).
  Accuracy alone is a MISLEADING metric for biased datasets.
""")

# Show the comparison DataFrame
print("📊 Adult Results DataFrame:")
print(adult_trainer.get_comparison_dataframe().round(4).to_string())

print("\n📊 German Results DataFrame:")
print(german_trainer.get_comparison_dataframe().round(4).to_string())

print("\n✅ Phase 4 complete! Models trained and saved to models/")
print("   → Next: Phase 5 — Bias Detection on these exact predictions")