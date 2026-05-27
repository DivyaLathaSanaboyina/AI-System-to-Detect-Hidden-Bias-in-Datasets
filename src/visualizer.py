# src/visualizer.py
"""
Visualizer Module — Phases 10 & 11

Generates all comparison charts and the final visual audit report.
All figures saved to reports/figures/.

Charts produced:
  1. Multi-model accuracy vs fairness scatter
  2. Fairness metrics comparison heatmap
  3. Mitigation before/after comparison
  4. Bias detection radar chart
  5. Complete audit dashboard (combined figure)
"""

import logging
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

FIGURES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "reports", "figures"
)
os.makedirs(FIGURES_DIR, exist_ok=True)

COLOR = {
    "lr":           "#2196F3",
    "dt":           "#FF9800",
    "rf":           "#4CAF50",
    "privileged":   "#1565C0",
    "unprivileged": "#C62828",
    "fair":         "#2E7D32",
    "biased":       "#B71C1C",
    "neutral":      "#78909C",
    "bg":           "#FAFAFA",
}
MODEL_COLORS = {
    "Logistic Regression": COLOR["lr"],
    "Decision Tree":       COLOR["dt"],
    "Random Forest":       COLOR["rf"],
}


class Visualizer:
    """
    Generates all bias audit visualizations.

    Usage:
        viz = Visualizer(dataset_name="adult")
        viz.plot_accuracy_vs_fairness(model_results, fairness_results)
        viz.plot_fairness_heatmap(fairness_df)
        viz.plot_mitigation_comparison(mitigation_results)
        viz.plot_audit_dashboard(...)
    """

    def __init__(self, dataset_name: str = "dataset"):
        self.dataset_name = dataset_name
        plt.rcParams.update({
            "font.family":   "DejaVu Sans",
            "axes.spines.top":   False,
            "axes.spines.right": False,
            "figure.facecolor":  "white",
        })

    def _save(self, fig, filename: str) -> str:
        path = os.path.join(FIGURES_DIR, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor="white")
        plt.close(fig)
        logger.info(f"  💾 Saved: {path}")
        return path

    # ══════════════════════════════════════════════════════════════════════════
    # CHART 1: Accuracy vs Fairness Scatter
    # ══════════════════════════════════════════════════════════════════════════

    def plot_accuracy_vs_fairness(
        self,
        model_perf:  Dict,   # {model_name: {accuracy, f1, ...}}
        model_fair:  Dict,   # {model_name: {worst_dir, worst_dpd, ...}}
        sensitive_col: str = "sex"
    ) -> str:
        """
        Scatter plot: X=Accuracy, Y=Disparate Impact Ratio.
        Each point = one model. Ideal = top-right corner (high acc, high DIR).
        """
        fig, ax = plt.subplots(figsize=(9, 7))

        for model_name, perf in model_perf.items():
            fair = model_fair.get(model_name, {})
            acc  = perf.get("accuracy", 0)
            dir_ = fair.get("worst_dir", 0)
            if dir_ is None: dir_ = 0

            color  = MODEL_COLORS.get(model_name, COLOR["neutral"])
            ax.scatter(acc, dir_, s=250, color=color,
                       zorder=5, edgecolors="white", linewidths=2)
            ax.annotate(
                model_name,
                (acc, dir_),
                textcoords="offset points",
                xytext=(8, 4),
                fontsize=10,
                color=color,
                fontweight="bold"
            )

        # Fairness threshold line
        ax.axhline(y=0.80, color=COLOR["fair"], linestyle="--",
                   linewidth=1.8, label="Fairness threshold (DIR=0.80)")
        ax.axhspan(0, 0.80, alpha=0.06, color=COLOR["biased"])
        ax.axhspan(0.80, ax.get_ylim()[1] if ax.get_ylim()[1] > 0.80 else 1.2,
                   alpha=0.06, color=COLOR["fair"])

        ax.set_xlabel("Accuracy", fontsize=12, fontweight="bold")
        ax.set_ylabel("Disparate Impact Ratio (DIR)", fontsize=12,
                      fontweight="bold")
        ax.set_title(
            f"Accuracy vs Fairness Trade-off\n"
            f"{self.dataset_name.upper()} | Sensitive: [{sensitive_col}]",
            fontsize=13, fontweight="bold", pad=15
        )
        ax.legend(fontsize=10)
        ax.set_ylim(bottom=0)

        # Annotations
        ax.text(0.02, 0.97, "🚨 Biased Zone",
                transform=ax.transAxes, fontsize=9,
                color=COLOR["biased"], va="top")
        ax.text(0.02, 0.55, "✅ Fair Zone",
                transform=ax.transAxes, fontsize=9,
                color=COLOR["fair"], va="top")

        plt.tight_layout()
        return self._save(
            fig,
            f"{self.dataset_name}_{sensitive_col}_accuracy_vs_fairness.png"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # CHART 2: Fairness Metrics Heatmap
    # ══════════════════════════════════════════════════════════════════════════

    def plot_fairness_heatmap(
        self,
        fairness_data: List[Dict],
        sensitive_cols: List[str]
    ) -> str:
        """
        Heatmap: rows=models, cols=fairness metrics.
        Color: green=fair, red=biased.
        """
        models  = [d["model"] for d in fairness_data]
        metrics = ["DPD", "EOD", "DIR"]

        # Build matrix
        data = []
        for d in fairness_data:
            row = []
            for attr in sensitive_cols[:1]:  # Use first sensitive attr
                attr_data = d["attributes"].get(attr, {})
                summary   = attr_data.get("fairness_metrics", {}).get("summary", {})
                row += [
                    abs(summary.get("worst_dpd", 0) or 0),
                    abs(summary.get("worst_eod", 0) or 0),
                    1 - min(summary.get("worst_dir", 1) or 1, 1),
                ]
            data.append(row)

        mat = np.array(data)
        col_labels = [f"{m} [{sensitive_cols[0]}]" for m in metrics]

        fig, ax = plt.subplots(figsize=(9, 5))
        im = ax.imshow(mat, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=0.5)

        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, fontsize=11)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models, fontsize=11)

        # Cell values
        for i in range(len(models)):
            for j in range(len(col_labels)):
                val = mat[i, j]
                ax.text(j, i, f"{val:.3f}",
                        ha="center", va="center",
                        fontsize=11, fontweight="bold",
                        color="white" if val > 0.25 else "black")

        plt.colorbar(im, ax=ax, label="Bias Magnitude (higher=worse)")
        ax.set_title(
            f"Fairness Metrics Heatmap — {self.dataset_name.upper()}\n"
            f"(Darker red = more biased)",
            fontsize=13, fontweight="bold", pad=12
        )
        plt.tight_layout()
        return self._save(fig, f"{self.dataset_name}_fairness_heatmap.png")

    # ══════════════════════════════════════════════════════════════════════════
    # CHART 3: Mitigation Before/After
    # ══════════════════════════════════════════════════════════════════════════

    def plot_mitigation_comparison(
        self,
        mitigation_results: List[Dict],
        sensitive_col: str = "sex"
    ) -> str:
        """
        Side-by-side bars showing Accuracy and DIR before vs after mitigation.
        The core "before vs after" chart for your presentation.
        """
        strategies = [r["strategy"] for r in mitigation_results]
        # Shorten labels
        short = {
            "Baseline (No Mitigation)":                  "Baseline",
            "Re-sampling (SMOTE)":                       "SMOTE",
            "Re-weighting (Kamiran & Calders)":          "Reweighting",
            "Fair Classifier (ExponentiatedGradient)":   "FairClassifier",
            "Fair Classifier (fallback: strong reweighting)": "Fair(fallback)",
            "Threshold Adjustment (Hardt et al.)":       "Threshold Adj",
        }
        labels   = [short.get(s, s[:18]) for s in strategies]
        accs     = [r.get("accuracy") or 0 for r in mitigation_results]
        dirs     = [r.get("dir")      or 0 for r in mitigation_results]

        x     = np.arange(len(labels))
        width = 0.38

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # ── Accuracy bars ──────────────────────────────────────────────────
        bar_colors = [
            COLOR["biased"] if i == 0 else COLOR["lr"]
            for i in range(len(labels))
        ]
        bars1 = ax1.bar(x, accs, width * 1.8, color=bar_colors, alpha=0.85,
                        edgecolor="white")
        ax1.set_ylim(min(accs) * 0.95 if accs else 0, max(accs) * 1.05 if accs else 1)
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax1.set_ylabel("Accuracy", fontsize=12)
        ax1.set_title("Accuracy (higher = better)", fontsize=12,
                      fontweight="bold")
        for bar, val in zip(bars1, accs):
            ax1.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.001,
                     f"{val:.3f}", ha="center", va="bottom",
                     fontsize=9, fontweight="bold")

        # ── DIR bars ───────────────────────────────────────────────────────
        dir_colors = [
            COLOR["fair"] if d >= 0.80 else COLOR["biased"]
            for d in dirs
        ]
        bars2 = ax2.bar(x, dirs, width * 1.8, color=dir_colors, alpha=0.85,
                        edgecolor="white")
        ax2.axhline(y=0.80, color="black", linestyle="--",
                    linewidth=1.5, label="Fair threshold (0.80)")
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax2.set_ylabel("Disparate Impact Ratio (DIR)", fontsize=12)
        ax2.set_title("Fairness — DIR (higher = fairer)", fontsize=12,
                      fontweight="bold")
        ax2.legend(fontsize=9)
        ax2.set_ylim(0, max(max(dirs) * 1.15, 1.0))
        for bar, val in zip(bars2, dirs):
            ax2.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.01,
                     f"{val:.3f}", ha="center", va="bottom",
                     fontsize=9, fontweight="bold")

        fig.suptitle(
            f"Bias Mitigation: Before vs After\n"
            f"{self.dataset_name.upper()} | [{sensitive_col}]",
            fontsize=13, fontweight="bold", y=1.02
        )
        plt.tight_layout()
        return self._save(
            fig,
            f"{self.dataset_name}_{sensitive_col}_mitigation_comparison.png"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # CHART 4: Group Performance Bars
    # ══════════════════════════════════════════════════════════════════════════

    def plot_group_performance(
        self,
        audit_reports:  List[Dict],
        sensitive_col:  str,
        privileged_val: str
    ) -> str:
        """
        Grouped bar chart: for each model, shows positive prediction rate
        for privileged vs unprivileged group side by side.
        """
        model_names = [r["model"] for r in audit_reports]
        priv_rates  = []
        unpriv_rates= []

        for report in audit_reports:
            attr_data   = report["attributes"].get(sensitive_col, {})
            group_stats = attr_data.get("group_stats", {})
            p_rate  = group_stats.get(privileged_val, {}).get("pos_pred_rate", 0)
            u_rates = [v["pos_pred_rate"]
                       for k, v in group_stats.items()
                       if k != privileged_val]
            u_rate  = np.mean(u_rates) if u_rates else 0
            priv_rates.append(p_rate)
            unpriv_rates.append(u_rate)

        x     = np.arange(len(model_names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        b1 = ax.bar(x - width/2, priv_rates,   width,
                    label=f"Privileged ({privileged_val})",
                    color=COLOR["privileged"], alpha=0.85, edgecolor="white")
        b2 = ax.bar(x + width/2, unpriv_rates, width,
                    label="Unprivileged group",
                    color=COLOR["unprivileged"], alpha=0.85, edgecolor="white")

        ax.set_xticks(x)
        ax.set_xticklabels(model_names, fontsize=11)
        ax.set_ylabel("Positive Prediction Rate", fontsize=12)
        ax.set_title(
            f"Group Prediction Rates: [{sensitive_col}]\n"
            f"{self.dataset_name.upper()} — Gap shows demographic parity violation",
            fontsize=12, fontweight="bold"
        )
        ax.legend(fontsize=10)
        ax.set_ylim(0, max(priv_rates + unpriv_rates) * 1.2)

        # Value labels
        for bar in list(b1) + list(b2):
            ax.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f"{bar.get_height():.1%}",
                ha="center", va="bottom", fontsize=9, fontweight="bold"
            )

        plt.tight_layout()
        return self._save(
            fig,
            f"{self.dataset_name}_{sensitive_col}_group_performance.png"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # CHART 5: Complete Audit Dashboard
    # ══════════════════════════════════════════════════════════════════════════

    def plot_audit_dashboard(
        self,
        model_perf:        Dict,
        audit_reports:     List[Dict],
        mitigation_results:List[Dict],
        sensitive_col:     str,
        privileged_val:    str
    ) -> str:
        """
        Master dashboard: 2×2 grid combining all key charts.
        This is your single-slide presentation figure.
        """
        fig = plt.figure(figsize=(18, 12))
        fig.patch.set_facecolor("white")
        gs  = gridspec.GridSpec(2, 2, figure=fig,
                                hspace=0.45, wspace=0.35)

        # ── Panel A: Group prediction rates per model ──────────────────────
        ax_a = fig.add_subplot(gs[0, 0])
        model_names  = [r["model"] for r in audit_reports]
        priv_rates   = []
        unpriv_rates = []
        for report in audit_reports:
            attr_data   = report["attributes"].get(sensitive_col, {})
            gs_data     = attr_data.get("group_stats", {})
            p = gs_data.get(privileged_val, {}).get("pos_pred_rate", 0)
            u = np.mean([v["pos_pred_rate"] for k, v in gs_data.items()
                         if k != privileged_val]) if gs_data else 0
            priv_rates.append(p)
            unpriv_rates.append(u)

        x = np.arange(len(model_names))
        w = 0.35
        ax_a.bar(x-w/2, priv_rates,   w, color=COLOR["privileged"],
                 alpha=0.85, label=f"Privileged ({privileged_val})",
                 edgecolor="white")
        ax_a.bar(x+w/2, unpriv_rates, w, color=COLOR["unprivileged"],
                 alpha=0.85, label="Unprivileged", edgecolor="white")
        ax_a.set_xticks(x)
        ax_a.set_xticklabels(model_names, rotation=15, ha="right",
                              fontsize=9)
        ax_a.set_ylabel("Positive Prediction Rate")
        ax_a.set_title("A. Group Prediction Rates", fontweight="bold")
        ax_a.legend(fontsize=8)
        ax_a.spines[["top","right"]].set_visible(False)

        # ── Panel B: Fairness metrics per model ────────────────────────────
        ax_b = fig.add_subplot(gs[0, 1])
        dpds, eods, dirs_ = [], [], []
        for report in audit_reports:
            attr = report["attributes"].get(sensitive_col, {})
            s    = attr.get("fairness_metrics", {}).get("summary", {})
            dpds.append(abs(s.get("worst_dpd") or 0))
            eods.append(abs(s.get("worst_eod") or 0))
            dirs_.append(s.get("worst_dir") or 0)

        x2 = np.arange(len(model_names))
        w2 = 0.25
        ax_b.bar(x2-w2,   dpds,  w2, color="#E53935", alpha=0.8,
                 label="|DPD|", edgecolor="white")
        ax_b.bar(x2,      eods,  w2, color="#FB8C00", alpha=0.8,
                 label="|EOD|", edgecolor="white")
        ax_b.bar(x2+w2,   dirs_, w2, color="#43A047", alpha=0.8,
                 label="DIR",   edgecolor="white")
        ax_b.axhline(0.10, color="#E53935", linestyle=":",
                     linewidth=1, alpha=0.7)
        ax_b.axhline(0.80, color="#43A047", linestyle=":",
                     linewidth=1, alpha=0.7)
        ax_b.set_xticks(x2)
        ax_b.set_xticklabels(model_names, rotation=15, ha="right",
                              fontsize=9)
        ax_b.set_title("B. Fairness Metrics per Model", fontweight="bold")
        ax_b.legend(fontsize=8)
        ax_b.spines[["top","right"]].set_visible(False)

        # ── Panel C: Accuracy vs Fairness scatter ──────────────────────────
        ax_c = fig.add_subplot(gs[1, 0])
        for i, report in enumerate(audit_reports):
            mname = report["model"]
            acc   = model_perf.get(mname, {}).get("accuracy", 0)
            attr  = report["attributes"].get(sensitive_col, {})
            s     = attr.get("fairness_metrics", {}).get("summary", {})
            dir_  = s.get("worst_dir") or 0
            color = MODEL_COLORS.get(mname, COLOR["neutral"])
            ax_c.scatter(acc, dir_, s=200, color=color, zorder=5,
                         edgecolors="white", linewidths=2)
            ax_c.annotate(mname, (acc, dir_),
                          textcoords="offset points", xytext=(5,3),
                          fontsize=8, color=color, fontweight="bold")

        ax_c.axhline(0.80, color=COLOR["fair"], linestyle="--",
                     linewidth=1.5, label="Fair threshold")
        ax_c.set_xlabel("Accuracy")
        ax_c.set_ylabel("Disparate Impact Ratio")
        ax_c.set_title("C. Accuracy vs Fairness", fontweight="bold")
        ax_c.legend(fontsize=8)
        ax_c.spines[["top","right"]].set_visible(False)

        # ── Panel D: Mitigation DIR comparison ────────────────────────────
        ax_d = fig.add_subplot(gs[1, 1])
        short = {
            "Baseline (No Mitigation)":               "Baseline",
            "Re-sampling (SMOTE)":                    "SMOTE",
            "Re-weighting (Kamiran & Calders)":       "Reweighting",
            "Fair Classifier (ExponentiatedGradient)":"FairClassifier",
            "Fair Classifier (fallback: strong reweighting)": "Fair(fallback)",
            "Threshold Adjustment (Hardt et al.)":    "Threshold Adj",
        }
        mit_labels = [short.get(r["strategy"], r["strategy"][:14])
                      for r in mitigation_results]
        mit_dirs   = [r.get("dir") or 0 for r in mitigation_results]
        mit_accs   = [r.get("accuracy") or 0 for r in mitigation_results]

        bar_colors = [
            COLOR["biased"] if d < 0.80 else COLOR["fair"]
            for d in mit_dirs
        ]
        xm = np.arange(len(mit_labels))
        bars = ax_d.bar(xm, mit_dirs, 0.65, color=bar_colors,
                        alpha=0.85, edgecolor="white")
        ax_d.axhline(0.80, color="black", linestyle="--",
                     linewidth=1.5, label="Fair threshold (0.80)")
        ax_d.set_xticks(xm)
        ax_d.set_xticklabels(mit_labels, rotation=30, ha="right",
                              fontsize=8)
        ax_d.set_ylabel("Disparate Impact Ratio (DIR)")
        ax_d.set_title("D. Mitigation Impact on Fairness", fontweight="bold")
        ax_d.legend(fontsize=8)
        ax_d.spines[["top","right"]].set_visible(False)
        for bar, val in zip(bars, mit_dirs):
            ax_d.text(bar.get_x() + bar.get_width()/2,
                      bar.get_height() + 0.01,
                      f"{val:.3f}", ha="center", va="bottom",
                      fontsize=8, fontweight="bold")

        fig.suptitle(
            f"AI Bias Audit Dashboard — {self.dataset_name.upper()}\n"
            f"Sensitive Attribute: [{sensitive_col}] | "
            f"Privileged Group: '{privileged_val}'",
            fontsize=15, fontweight="bold", y=1.01
        )
        return self._save(
            fig,
            f"{self.dataset_name}_{sensitive_col}_audit_dashboard.png"
        )