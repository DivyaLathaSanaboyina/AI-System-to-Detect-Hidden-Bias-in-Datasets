# src/explainer.py
"""
Explainability Module — Phase 9

Answers the question: WHY does bias occur?

Uses three complementary explainability methods:

  1. Feature Importance (model-intrinsic)
     → Which features does the model rely on most?
     → Fast, but doesn't show directionality

  2. SHAP (SHapley Additive exPlanations)
     → How much does each feature push a prediction up or down?
     → Based on cooperative game theory (Shapley values)
     → Direction + magnitude for every feature × every prediction

  3. Group-Differential SHAP
     → Compare SHAP values between privileged and unprivileged groups
     → Directly answers: "Which features cause different treatment?"

References:
  Lundberg & Lee (2017) "A Unified Approach to Interpreting Model Predictions"
  Shapley (1953) "A value for n-person games"
"""

import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend — safe for all environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

FIGURES_DIR  = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "reports", "figures"
)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Color palette
COLOR = {
    "privileged":   "#2196F3",   # Blue
    "unprivileged": "#F44336",   # Red
    "positive":     "#4CAF50",   # Green
    "negative":     "#FF5722",   # Orange-red
    "neutral":      "#9E9E9E",   # Grey
}


class Explainer:
    """
    Generates SHAP explanations and feature importance analysis
    to explain WHY a model is biased.

    Usage:
        explainer = Explainer(model, X_train, X_test,
                              feature_names, dataset_name)
        explainer.run_full_explanation(
            sensitive_test, privileged_val="Male"
        )
    """

    def __init__(
        self,
        model,
        X_train:       np.ndarray,
        X_test:        np.ndarray,
        feature_names: List[str],
        dataset_name:  str = "dataset",
        sample_size:   int = 200
    ):
        """
        Args:
            model:         Trained sklearn model (LR, DT, RF)
            X_train:       Training features (for SHAP background)
            X_test:        Test features (to explain)
            feature_names: Column names after encoding
            dataset_name:  Used for saving figures
            sample_size:   Number of test samples for SHAP (speed vs detail)
        """
        self.model         = model
        self.feature_names = feature_names
        self.dataset_name  = dataset_name
        self.sample_size   = min(sample_size, len(X_test))

        # Sample for efficiency
        idx = np.random.RandomState(42).choice(
            len(X_test), self.sample_size, replace=False
        )
        self.X_background = X_train[:100]        # SHAP background distribution
        self.X_explain    = X_test[idx]           # Samples to explain
        self.explain_idx  = idx

        self.shap_values  = None
        self._model_type  = self._detect_model_type()

        logger.info(
            f"Explainer ready: {self._model_type} model | "
            f"{self.sample_size} samples | "
            f"{len(feature_names)} features"
        )

    def _detect_model_type(self) -> str:
        """Detects model type for choosing the right SHAP explainer."""
        name = type(self.model).__name__
        if "LogisticRegression" in name:  return "linear"
        if "Forest" in name:              return "tree"
        if "Tree" in name:                return "tree"
        return "generic"

    # ══════════════════════════════════════════════════════════════════════════
    # METHOD 1: FEATURE IMPORTANCE
    # ══════════════════════════════════════════════════════════════════════════

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Extracts model-intrinsic feature importance.

        For Logistic Regression: |coefficient| (magnitude = importance)
        For Decision Tree/RF:    .feature_importances_ (Gini impurity reduction)

        Returns DataFrame sorted by importance descending.
        """
        model = self.model

        # Handle Fairlearn wrapper
        if hasattr(model, "_predictors"):
            model = model._predictors[0]

        if hasattr(model, "coef_"):
            # Logistic Regression
            importance = np.abs(model.coef_[0])
            kind = "LR |Coefficient|"
        elif hasattr(model, "feature_importances_"):
            # Tree-based models
            importance = model.feature_importances_
            kind = "Gini Importance"
        else:
            logger.warning("Model has no feature importance attribute.")
            return pd.DataFrame()

        df = pd.DataFrame({
            "feature":    self.feature_names,
            "importance": importance,
            "kind":       kind
        }).sort_values("importance", ascending=False).head(top_n)

        df["rank"] = range(1, len(df) + 1)
        return df

    # ══════════════════════════════════════════════════════════════════════════
    # METHOD 2: SHAP VALUES
    # ══════════════════════════════════════════════════════════════════════════

    def compute_shap_values(self) -> Optional[np.ndarray]:
        """
        Computes SHAP values using the appropriate explainer type.

        TreeExplainer: Fast, exact for tree-based models
        LinearExplainer: Exact for linear models
        KernelExplainer: Model-agnostic but slow (fallback)

        SHAP value interpretation:
          Positive SHAP → pushes prediction TOWARD positive class (income >50K)
          Negative SHAP → pushes prediction TOWARD negative class (income <=50K)
          Magnitude     → how much influence this feature has
        """
        import shap
        warnings.filterwarnings("ignore")

        try:
            if self._model_type == "tree":
                logger.info("  Using TreeExplainer (exact, fast)...")
                explainer = shap.TreeExplainer(self.model)
                shap_vals = explainer.shap_values(self.X_explain)
                # RF returns list [class0_shap, class1_shap] — take class 1
                # Handle both list output and 3D array output
                if isinstance(shap_vals, list):
                    shap_vals = shap_vals[1]
                elif shap_vals.ndim == 3:
                    # Shape (n_samples, n_features, n_classes) → take class 1
                    shap_vals = shap_vals[:, :, 1]

            elif self._model_type == "linear":
                logger.info("  Using LinearExplainer (exact)...")
                explainer = shap.LinearExplainer(
                    self.model, self.X_background,
                    feature_perturbation="interventional"
                )
                shap_vals = explainer.shap_values(self.X_explain)

            else:
                logger.info("  Using KernelExplainer (model-agnostic, slower)...")
                explainer = shap.KernelExplainer(
                    self.model.predict_proba,
                    shap.sample(self.X_background, 50)
                )
                shap_vals = explainer.shap_values(
                    self.X_explain[:50], nsamples=100
                )[:, :, 1]

            self.shap_values = shap_vals
            logger.info(
                f"  ✅ SHAP values computed: shape={shap_vals.shape}"
            )
            return shap_vals

        except Exception as e:
            logger.error(f"SHAP computation failed: {e}")
            return None

    def get_mean_abs_shap(self, top_n: int = 20) -> pd.DataFrame:
        """
        Returns mean |SHAP| per feature — overall feature importance
        as seen by SHAP (more trustworthy than model-intrinsic importance).
        """
        if self.shap_values is None:
            self.compute_shap_values()
        if self.shap_values is None:
            return pd.DataFrame()

        mean_shap = np.abs(self.shap_values).mean(axis=0)
        df = pd.DataFrame({
            "feature":   self.feature_names,
            "mean_shap": mean_shap
        }).sort_values("mean_shap", ascending=False).head(top_n)
        df["rank"] = range(1, len(df) + 1)
        return df

    # ══════════════════════════════════════════════════════════════════════════
    # METHOD 3: GROUP-DIFFERENTIAL SHAP (Bias explanation)
    # ══════════════════════════════════════════════════════════════════════════

    def group_differential_shap(
        self,
        sensitive_test: np.ndarray,
        privileged_val: str,
        top_n: int = 15
    ) -> pd.DataFrame:
        """
        The core bias explanation method.

        For each feature, computes:
          mean_SHAP(privileged group) - mean_SHAP(unprivileged group)

        Large positive difference → feature benefits privileged group MORE
        Large negative difference → feature benefits unprivileged group MORE

        This directly answers: "Which features are treated differently
        between demographic groups by this model?"

        This is essentially a feature-level audit of group discrimination.
        """
        if self.shap_values is None:
            self.compute_shap_values()
        if self.shap_values is None:
            return pd.DataFrame()

        # Align sensitive_test with our sample indices
        sens_sample = sensitive_test[self.explain_idx]

        mask_priv   = (sens_sample == privileged_val)
        mask_unpriv = ~mask_priv

        if mask_priv.sum() == 0 or mask_unpriv.sum() == 0:
            logger.warning("One group has 0 samples in SHAP subset.")
            return pd.DataFrame()

        # Mean SHAP per group
        shap_priv   = self.shap_values[mask_priv].mean(axis=0)
        shap_unpriv = self.shap_values[mask_unpriv].mean(axis=0)

        diff = shap_priv - shap_unpriv

        df = pd.DataFrame({
            "feature":         self.feature_names,
            "shap_privileged": shap_priv,
            "shap_unprivileged": shap_unpriv,
            "shap_difference": diff,
            "abs_difference":  np.abs(diff)
        }).sort_values("abs_difference", ascending=False).head(top_n)

        df["direction"] = df["shap_difference"].apply(
            lambda x: "Favors Privileged" if x > 0 else "Favors Unprivileged"
        )
        return df

    # ══════════════════════════════════════════════════════════════════════════
    # VISUALIZATIONS
    # ══════════════════════════════════════════════════════════════════════════

    def plot_feature_importance(
        self,
        top_n: int = 20,
        save: bool = True
    ) -> str:
        """Bar chart of top-N feature importances."""
        fi = self.get_feature_importance(top_n)
        if fi.empty:
            logger.warning("No feature importance to plot.")
            return ""

        fig, ax = plt.subplots(figsize=(10, 7))
        bars = ax.barh(
            fi["feature"][::-1],
            fi["importance"][::-1],
            color=COLOR["privileged"],
            alpha=0.85,
            edgecolor="white"
        )
        ax.set_xlabel("Importance", fontsize=12)
        ax.set_title(
            f"Top {top_n} Feature Importances\n"
            f"{self.dataset_name} | {type(self.model).__name__}",
            fontsize=13, fontweight="bold"
        )
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()

        path = os.path.join(
            FIGURES_DIR,
            f"{self.dataset_name}_feature_importance.png"
        )
        if save:
            fig.savefig(path, dpi=150, bbox_inches="tight")
            logger.info(f"  💾 Saved: {path}")
        plt.close(fig)
        return path

    def plot_shap_summary(self, save: bool = True) -> str:
        """
        SHAP summary bar plot — mean |SHAP| per feature.
        Shows which features matter most globally.
        """
        mean_shap = self.get_mean_abs_shap(20)
        if mean_shap.empty:
            return ""

        fig, ax = plt.subplots(figsize=(10, 7))
        colors = [COLOR["positive"] if i < 5 else COLOR["neutral"]
                  for i in range(len(mean_shap))]
        ax.barh(
            mean_shap["feature"][::-1],
            mean_shap["mean_shap"][::-1],
            color=colors[::-1],
            alpha=0.85,
            edgecolor="white"
        )
        ax.set_xlabel("Mean |SHAP Value|", fontsize=12)
        ax.set_title(
            f"SHAP Feature Importance (Mean |SHAP|)\n"
            f"{self.dataset_name} | {type(self.model).__name__}",
            fontsize=13, fontweight="bold"
        )
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()

        path = os.path.join(
            FIGURES_DIR, f"{self.dataset_name}_shap_summary.png"
        )
        if save:
            fig.savefig(path, dpi=150, bbox_inches="tight")
            logger.info(f"  💾 Saved: {path}")
        plt.close(fig)
        return path

    def plot_group_differential_shap(
        self,
        sensitive_test: np.ndarray,
        privileged_val: str,
        sensitive_col:  str,
        top_n: int = 15,
        save: bool = True
    ) -> str:
        """
        THE KEY BIAS EXPLANATION CHART.

        Shows SHAP differences between privileged and unprivileged groups
        for each feature. This is the visual answer to "WHY is it biased?"
        """
        diff_df = self.group_differential_shap(
            sensitive_test, privileged_val, top_n
        )
        if diff_df.empty:
            return ""

        fig, ax = plt.subplots(figsize=(11, 8))

        colors = [
            COLOR["privileged"] if v > 0 else COLOR["unprivileged"]
            for v in diff_df["shap_difference"]
        ]

        ax.barh(
            diff_df["feature"][::-1],
            diff_df["shap_difference"][::-1],
            color=colors[::-1],
            alpha=0.85,
            edgecolor="white",
            height=0.7
        )
        ax.axvline(x=0, color="black", linewidth=1.2, linestyle="--")

        # Legend
        patch_priv   = mpatches.Patch(
            color=COLOR["privileged"],
            label=f"Favors {privileged_val} (privileged)"
        )
        patch_unpriv = mpatches.Patch(
            color=COLOR["unprivileged"],
            label=f"Favors unprivileged group"
        )
        ax.legend(
            handles=[patch_priv, patch_unpriv],
            loc="lower right", fontsize=10
        )

        ax.set_xlabel(
            "SHAP Difference (Privileged − Unprivileged)", fontsize=12
        )
        ax.set_title(
            f"Group-Differential SHAP: [{sensitive_col}]\n"
            f"Why does the model treat groups differently?",
            fontsize=13, fontweight="bold"
        )
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()

        path = os.path.join(
            FIGURES_DIR,
            f"{self.dataset_name}_{sensitive_col}_group_shap.png"
        )
        if save:
            fig.savefig(path, dpi=150, bbox_inches="tight")
            logger.info(f"  💾 Saved: {path}")
        plt.close(fig)
        return path

    def plot_group_shap_comparison(
        self,
        sensitive_test: np.ndarray,
        privileged_val: str,
        sensitive_col:  str,
        top_n: int = 12,
        save: bool = True
    ) -> str:
        """
        Side-by-side SHAP bar chart comparing privileged vs unprivileged
        group mean SHAP values per feature.
        """
        if self.shap_values is None:
            self.compute_shap_values()
        if self.shap_values is None:
            return ""

        sens_sample = sensitive_test[self.explain_idx]
        mask_priv   = (sens_sample == privileged_val)
        mask_unpriv = ~mask_priv

        shap_priv   = self.shap_values[mask_priv].mean(axis=0)
        shap_unpriv = self.shap_values[mask_unpriv].mean(axis=0)

        # Pick top features by total absolute SHAP
        total_abs = np.abs(shap_priv) + np.abs(shap_unpriv)
        top_idx   = np.argsort(total_abs)[-top_n:][::-1]

        feats   = [self.feature_names[i] for i in top_idx]
        s_priv  = shap_priv[top_idx]
        s_unpriv= shap_unpriv[top_idx]

        x     = np.arange(len(feats))
        width = 0.38

        fig, ax = plt.subplots(figsize=(13, 7))
        ax.bar(x - width/2, s_priv,   width, label=f"{privileged_val} (privileged)",
               color=COLOR["privileged"],   alpha=0.85, edgecolor="white")
        ax.bar(x + width/2, s_unpriv, width, label="Unprivileged group",
               color=COLOR["unprivileged"], alpha=0.85, edgecolor="white")

        ax.axhline(y=0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(feats, rotation=40, ha="right", fontsize=9)
        ax.set_ylabel("Mean SHAP Value", fontsize=12)
        ax.set_title(
            f"Mean SHAP per Group: [{sensitive_col}]\n"
            f"Positive = pushes toward favorable outcome",
            fontsize=13, fontweight="bold"
        )
        ax.legend(fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()

        path = os.path.join(
            FIGURES_DIR,
            f"{self.dataset_name}_{sensitive_col}_shap_comparison.png"
        )
        if save:
            fig.savefig(path, dpi=150, bbox_inches="tight")
            logger.info(f"  💾 Saved: {path}")
        plt.close(fig)
        return path

    # ══════════════════════════════════════════════════════════════════════════
    # FULL PIPELINE + REPORT
    # ══════════════════════════════════════════════════════════════════════════

    def run_full_explanation(
        self,
        sensitive_test: np.ndarray,
        privileged_val: str,
        sensitive_col:  str
    ) -> Dict:
        """
        Runs the complete explainability pipeline and returns
        a dict with results + saved figure paths.
        """
        logger.info(f"🔍 Running full SHAP explanation pipeline...")

        # Feature importance
        fi = self.get_feature_importance(20)
        logger.info(f"  ✅ Feature importance computed")

        # SHAP
        self.compute_shap_values()
        mean_shap = self.get_mean_abs_shap(20)

        # Group differential
        diff_df = self.group_differential_shap(
            sensitive_test, privileged_val, top_n=15
        )

        # Plots
        paths = {}
        paths["feature_importance"] = self.plot_feature_importance()
        paths["shap_summary"]       = self.plot_shap_summary()
        paths["group_differential"] = self.plot_group_differential_shap(
            sensitive_test, privileged_val, sensitive_col
        )
        paths["group_comparison"]   = self.plot_group_shap_comparison(
            sensitive_test, privileged_val, sensitive_col
        )

        # Print text report
        self._print_explanation_report(fi, mean_shap, diff_df,
                                        sensitive_col, privileged_val)

        return {
            "feature_importance":   fi,
            "mean_shap":            mean_shap,
            "group_differential":   diff_df,
            "figure_paths":         paths
        }

    def _print_explanation_report(
        self,
        fi:            pd.DataFrame,
        mean_shap:     pd.DataFrame,
        diff_df:       pd.DataFrame,
        sensitive_col: str,
        privileged_val:str
    ) -> None:
        """Prints a text-based explainability report."""
        model_name = type(self.model).__name__

        print("\n" + "═"*68)
        print(f"  🧠 EXPLAINABILITY REPORT")
        print(f"  Model: {model_name} | Dataset: {self.dataset_name}")
        print(f"  Sensitive: [{sensitive_col}] | Privileged: '{privileged_val}'")
        print("═"*68)

        # Top features (model-intrinsic)
        if not fi.empty:
            print(f"\n  📊 TOP 10 MODEL FEATURES (intrinsic importance):")
            print(f"  {'Rank':<6} {'Feature':<35} {'Importance':>12}")
            print("  " + "─"*55)
            for _, row in fi.head(10).iterrows():
                print(f"  {int(row['rank']):<6} {row['feature']:<35} "
                      f"{row['importance']:>12.4f}")

        # Top SHAP features
        if not mean_shap.empty:
            print(f"\n  🎯 TOP 10 SHAP FEATURES (mean |SHAP|):")
            print(f"  {'Rank':<6} {'Feature':<35} {'Mean |SHAP|':>12}")
            print("  " + "─"*55)
            for _, row in mean_shap.head(10).iterrows():
                print(f"  {int(row['rank']):<6} {row['feature']:<35} "
                      f"{row['mean_shap']:>12.4f}")

        # Group differential — THE KEY BIAS EXPLANATION
        if not diff_df.empty:
            print(f"\n  🚨 GROUP-DIFFERENTIAL SHAP — WHY THE BIAS EXISTS:")
            print(f"  (Positive = feature benefits '{privileged_val}' more)")
            print(f"  {'Feature':<35} {'Priv SHAP':>10} {'Unpriv SHAP':>12} "
                  f"{'Diff':>8} {'Direction'}")
            print("  " + "─"*78)
            for _, row in diff_df.head(10).iterrows():
                arrow = "↑ priv" if row["shap_difference"] > 0 else "↑ unpriv"
                print(
                    f"  {row['feature']:<35} "
                    f"{row['shap_privileged']:>10.4f} "
                    f"{row['shap_unprivileged']:>12.4f} "
                    f"{row['shap_difference']:>+8.4f} "
                    f"  {arrow}"
                )

        print(f"\n  💾 Figures saved to: reports/figures/")
        print("═"*68 + "\n")