# src/model_trainer.py
"""
Model Trainer Module — Phase 4

Trains multiple classifiers and evaluates them on both
accuracy AND fairness metrics side by side.

Key design insight: We store both the trained models AND
their predictions on the test set, because fairness metrics
need the raw predictions (not just accuracy scores).
"""

import logging
import time
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any

from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier
from sklearn.metrics         import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
import joblib
import os

logger = logging.getLogger(__name__)

# ── Inline config ──────────────────────────────────────────────────────────────
RANDOM_STATE = 42
MODELS_DIR   = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
)
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_PARAMS = {
    "logistic_regression": {
        "max_iter":     1000,
        "random_state": RANDOM_STATE,
        "C":            1.0,
        "solver":       "lbfgs"
    },
    "decision_tree": {
        "max_depth":         10,
        "random_state":      RANDOM_STATE,
        "min_samples_split": 20,
        "min_samples_leaf":  10
    },
    "random_forest": {
        "n_estimators": 100,
        "max_depth":    10,
        "random_state": RANDOM_STATE,
        "n_jobs":       -1,
        "min_samples_split": 20
    }
}


class ModelTrainer:
    """
    Trains Logistic Regression, Decision Tree, and Random Forest
    classifiers and computes comprehensive evaluation metrics.

    Usage:
        trainer = ModelTrainer()
        results = trainer.train_all(X_train, X_test, y_train, y_test)
        trainer.print_comparison()
    """

    def __init__(self):
        self.models: Dict[str, Any]   = {}   # Trained model objects
        self.results: Dict[str, Dict] = {}   # Metrics per model
        self._build_models()

    def _build_models(self):
        """Instantiates all models with configured hyperparameters."""
        self.models = {
            "Logistic Regression": LogisticRegression(
                **MODEL_PARAMS["logistic_regression"]
            ),
            "Decision Tree": DecisionTreeClassifier(
                **MODEL_PARAMS["decision_tree"]
            ),
            "Random Forest": RandomForestClassifier(
                **MODEL_PARAMS["random_forest"]
            ),
        }
        logger.info(f"  🏗️  Initialized {len(self.models)} models")

    def train_all(
        self,
        X_train: np.ndarray,
        X_test:  np.ndarray,
        y_train: np.ndarray,
        y_test:  np.ndarray,
        dataset_name: str = "dataset"
    ) -> Dict[str, Dict]:
        """
        Trains all models and computes evaluation metrics.

        Args:
            X_train, X_test: Feature arrays
            y_train, y_test: Binary label arrays
            dataset_name: Used for saving model files

        Returns:
            Dict mapping model_name → metrics dict
        """
        logger.info(f"🚀 Training {len(self.models)} models...")

        for model_name, model in self.models.items():
            logger.info(f"  ⏳ Training: {model_name}...")
            start = time.time()

            # ── Train ──────────────────────────────────────────────────────
            model.fit(X_train, y_train)
            train_time = time.time() - start

            # ── Predict ────────────────────────────────────────────────────
            y_pred      = model.predict(X_test)
            y_pred_prob = model.predict_proba(X_test)[:, 1]

            # ── Metrics ────────────────────────────────────────────────────
            metrics = self._compute_metrics(
                y_test, y_pred, y_pred_prob, train_time
            )
            self.results[model_name] = {
                **metrics,
                "y_pred":      y_pred,
                "y_pred_prob": y_pred_prob,
                "model":       model
            }

            logger.info(
                f"     ✅ {model_name}: "
                f"Acc={metrics['accuracy']:.3f} | "
                f"F1={metrics['f1']:.3f} | "
                f"AUC={metrics['roc_auc']:.3f} | "
                f"Time={train_time:.2f}s"
            )

            # ── Save model to disk ─────────────────────────────────────────
            safe_name = model_name.lower().replace(" ", "_")
            model_path = os.path.join(
                MODELS_DIR, f"{dataset_name}_{safe_name}.pkl"
            )
            joblib.dump(model, model_path)

        logger.info(f"✅ All models trained. Saved to {MODELS_DIR}/")
        return self.results

    def _compute_metrics(
        self,
        y_true:      np.ndarray,
        y_pred:      np.ndarray,
        y_pred_prob: np.ndarray,
        train_time:  float
    ) -> Dict:
        """
        Computes a comprehensive set of classification metrics.

        WHY these specific metrics?
        - Accuracy: Overall correctness (misleading with imbalance)
        - Precision: Of predicted positives, how many are real?
        - Recall: Of real positives, how many did we catch?
        - F1: Harmonic mean of Precision+Recall (better for imbalance)
        - ROC-AUC: Model's ability to rank positives above negatives
        - Confusion matrix: Full breakdown of TP/FP/TN/FN
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        return {
            "accuracy":      accuracy_score(y_true, y_pred),
            "precision":     precision_score(y_true, y_pred, zero_division=0),
            "recall":        recall_score(y_true, y_pred, zero_division=0),
            "f1":            f1_score(y_true, y_pred, zero_division=0),
            "roc_auc":       roc_auc_score(y_true, y_pred_prob),
            "tp": int(tp), "fp": int(fp),
            "tn": int(tn), "fn": int(fn),
            "train_time_sec": round(train_time, 3),
        }

    def get_model(self, model_name: str):
        """Returns a trained model object by name."""
        if model_name not in self.models:
            raise ValueError(
                f"Model '{model_name}' not found. "
                f"Available: {list(self.models.keys())}"
            )
        return self.models[model_name]

    def get_predictions(self, model_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (y_pred, y_pred_prob) for a given model."""
        r = self.results[model_name]
        return r["y_pred"], r["y_pred_prob"]

    def print_comparison(self) -> None:
        """
        Prints a formatted comparison table of all models.
        This is the output you show in presentations/interviews.
        """
        if not self.results:
            print("No results yet. Call train_all() first.")
            return

        print("\n" + "=" * 72)
        print("  📊 MODEL COMPARISON REPORT")
        print("=" * 72)
        print(
            f"  {'Model':<22} {'Accuracy':>9} {'Precision':>10} "
            f"{'Recall':>8} {'F1':>8} {'AUC':>8} {'Time(s)':>8}"
        )
        print("-" * 72)

        # Find best model per metric
        best = {
            metric: max(self.results.items(), key=lambda x: x[1][metric])[0]
            for metric in ["accuracy", "f1", "roc_auc"]
        }

        for name, r in self.results.items():
            # Mark best model with ★
            acc_mark = " ★" if name == best["accuracy"] else "  "
            f1_mark  = " ★" if name == best["f1"]       else "  "
            auc_mark = " ★" if name == best["roc_auc"]  else "  "

            print(
                f"  {name:<22} "
                f"{r['accuracy']:>8.3f}{acc_mark}"
                f"{r['precision']:>10.3f}"
                f"{r['recall']:>8.3f}"
                f"{r['f1']:>7.3f}{f1_mark}"
                f"{r['roc_auc']:>7.3f}{auc_mark}"
                f"{r['train_time_sec']:>8.2f}"
            )

        print("=" * 72)
        print("  ★ = Best in category\n")

        # Confusion matrix for each model
        print("  📋 CONFUSION MATRICES (Test Set)")
        print("-" * 72)
        for name, r in self.results.items():
            total   = r["tp"] + r["fp"] + r["tn"] + r["fn"]
            tpr = r["tp"] / (r["tp"] + r["fn"]) if (r["tp"] + r["fn"]) > 0 else 0
            fpr = r["fp"] / (r["fp"] + r["tn"]) if (r["fp"] + r["tn"]) > 0 else 0
            print(f"\n  {name}:")
            print(f"    {'':>16} Predicted 0   Predicted 1")
            print(f"    {'Actual 0':>16}  TN={r['tn']:>6}    FP={r['fp']:>6}")
            print(f"    {'Actual 1':>16}  FN={r['fn']:>6}    TP={r['tp']:>6}")
            print(f"    True Positive Rate (Recall): {tpr:.3f}")
            print(f"    False Positive Rate:         {fpr:.3f}")

        print("\n" + "=" * 72)
        print("  ⚠️  NOTE: Accuracy alone doesn't tell the fairness story.")
        print("  A model can be 87% accurate AND deeply biased.")
        print("  → Proceed to Phase 5 to measure FAIRNESS metrics.")
        print("=" * 72 + "\n")

    def get_comparison_dataframe(self) -> pd.DataFrame:
        """
        Returns model comparison as a DataFrame.
        Useful for visualization and reports.
        """
        rows = []
        for name, r in self.results.items():
            rows.append({
                "Model":     name,
                "Accuracy":  r["accuracy"],
                "Precision": r["precision"],
                "Recall":    r["recall"],
                "F1":        r["f1"],
                "ROC-AUC":   r["roc_auc"],
                "TP":        r["tp"],
                "FP":        r["fp"],
                "TN":        r["tn"],
                "FN":        r["fn"],
            })
        return pd.DataFrame(rows).set_index("Model")