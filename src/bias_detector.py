# src/bias_detector.py
"""
Bias Detector + Fairness Metrics Module — Phase 5 & 6

Implements group-wise performance analysis and three core
fairness metrics used in academic research and industry:

  1. Demographic Parity Difference (DPD)
  2. Equal Opportunity Difference  (EOD)
  3. Disparate Impact Ratio        (DIR)

Mathematical references:
  - Hardt et al. (2016) "Equality of Opportunity in Supervised Learning"
  - EEOC 80% rule (29 CFR Part 1607)
  - Barocas & Hardt "Fairness in Machine Learning" textbook
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Fairness thresholds (industry standard) ────────────────────────────────────
FAIRNESS_THRESHOLDS = {
    "demographic_parity_difference": 0.10,
    "equal_opportunity_difference":  0.10,
    "disparate_impact_ratio":        0.80,
}


class BiasDetector:
    """
    Measures and reports bias in model predictions
    across demographic groups.

    Usage:
        detector = BiasDetector(sensitive_df, y_true, dataset_name)
        report   = detector.run_full_audit(model_name, y_pred, privileged_groups)
        detector.print_report(report)
    """

    def __init__(
        self,
        sensitive_df: pd.DataFrame,
        y_true:       np.ndarray,
        dataset_name: str = "dataset"
    ):
        self.sensitive_df   = sensitive_df.reset_index(drop=True)
        self.y_true         = np.array(y_true)
        self.dataset_name   = dataset_name
        self.sensitive_cols = list(sensitive_df.columns)

    # ══════════════════════════════════════════════════════════════════════════
    # CORE PUBLIC API
    # ══════════════════════════════════════════════════════════════════════════

    def run_full_audit(
        self,
        model_name:        str,
        y_pred:            np.ndarray,
        privileged_groups: Optional[Dict[str, str]] = None
    ) -> Dict:
        """
        Runs the complete bias audit for one model.

        Args:
            model_name:        Name for reporting
            y_pred:            Binary predictions array
            privileged_groups: e.g. {"sex": "Male", "race": "White"}

        Returns:
            Nested dict with all fairness metrics and group stats
        """
        y_pred = np.array(y_pred)
        report = {
            "model":            model_name,
            "dataset":          self.dataset_name,
            "n_samples":        len(y_pred),
            "overall_pos_rate": float(y_pred.mean()),
            "attributes":       {}
        }

        for attr in self.sensitive_cols:
            if attr not in self.sensitive_df.columns:
                continue
            priv_group = (privileged_groups or {}).get(attr, None)
            report["attributes"][attr] = self._audit_attribute(
                attr, y_pred, priv_group
            )

        return report

    # ══════════════════════════════════════════════════════════════════════════
    # ATTRIBUTE-LEVEL AUDIT
    # ══════════════════════════════════════════════════════════════════════════

    def _audit_attribute(
        self,
        attr:       str,
        y_pred:     np.ndarray,
        priv_value: Optional[str]
    ) -> Dict:
        """
        Computes group-wise stats + all 3 fairness metrics
        for a single sensitive attribute.
        """
        groups      = self.sensitive_df[attr].values
        unique_vals = sorted(self.sensitive_df[attr].unique())

        group_stats = {}
        for val in unique_vals:
            mask    = (groups == val)
            y_t_grp = self.y_true[mask]
            y_p_grp = y_pred[mask]
            n       = int(mask.sum())

            tp = int(((y_p_grp == 1) & (y_t_grp == 1)).sum())
            fp = int(((y_p_grp == 1) & (y_t_grp == 0)).sum())
            tn = int(((y_p_grp == 0) & (y_t_grp == 0)).sum())
            fn = int(((y_p_grp == 0) & (y_t_grp == 1)).sum())

            pos_pred_rate = float(y_p_grp.mean()) if n > 0 else 0.0
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            acc = (tp + tn) / n  if n > 0 else 0.0

            group_stats[str(val)] = {
                "n":               n,
                "n_positive_true": int(y_t_grp.sum()),
                "n_predicted_pos": int(y_p_grp.sum()),
                "pos_pred_rate":   round(pos_pred_rate, 4),
                "true_pos_rate":   round(tpr, 4),
                "false_pos_rate":  round(fpr, 4),
                "accuracy":        round(acc, 4),
                "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                "is_privileged":   (str(val) == str(priv_value))
                                    if priv_value else None
            }

        fairness = self._compute_fairness_metrics(group_stats, priv_value)
        return {
            "privileged_group": priv_value,
            "group_stats":      group_stats,
            "fairness_metrics": fairness
        }

    # ══════════════════════════════════════════════════════════════════════════
    # FAIRNESS METRICS — DPD, EOD, DIR
    # ══════════════════════════════════════════════════════════════════════════

    def _compute_fairness_metrics(
        self,
        group_stats: Dict,
        priv_value:  Optional[str]
    ) -> Dict:
        """
        Computes DPD, EOD, and DIR.

        DPD = P(Ŷ=1|privileged) - P(Ŷ=1|unprivileged)   [ideal: 0]
        EOD = TPR(privileged)   - TPR(unprivileged)       [ideal: 0]
        DIR = P(Ŷ=1|unprivileged) / P(Ŷ=1|privileged)   [ideal: 1]
        """
        if not group_stats:
            return {}

        group_names = list(group_stats.keys())

        # Determine privileged key
        if priv_value and str(priv_value) in group_stats:
            priv_key    = str(priv_value)
            unpriv_keys = [g for g in group_names if g != priv_key]
        else:
            priv_key    = max(group_names,
                              key=lambda g: group_stats[g]["pos_pred_rate"])
            unpriv_keys = [g for g in group_names if g != priv_key]

        priv_ppr = group_stats[priv_key]["pos_pred_rate"]
        priv_tpr = group_stats[priv_key]["true_pos_rate"]

        metrics = {"privileged_group": priv_key, "pairwise": {}}
        worst_dpd, worst_eod, worst_dir = 0.0, 0.0, 1.0

        for uk in unpriv_keys:
            unpriv_ppr = group_stats[uk]["pos_pred_rate"]
            unpriv_tpr = group_stats[uk]["true_pos_rate"]

            dpd  = priv_ppr - unpriv_ppr
            eod  = priv_tpr - unpriv_tpr
            dir_ = (unpriv_ppr / priv_ppr) if priv_ppr > 0 else 0.0

            dpd_biased = abs(dpd)  > FAIRNESS_THRESHOLDS["demographic_parity_difference"]
            eod_biased = abs(eod)  > FAIRNESS_THRESHOLDS["equal_opportunity_difference"]
            dir_biased = dir_      < FAIRNESS_THRESHOLDS["disparate_impact_ratio"]

            metrics["pairwise"][f"{priv_key}_vs_{uk}"] = {
                "demographic_parity_difference": round(dpd,  4),
                "equal_opportunity_difference":  round(eod,  4),
                "disparate_impact_ratio":        round(dir_, 4),
                "dpd_biased": dpd_biased,
                "eod_biased": eod_biased,
                "dir_biased": dir_biased,
                "any_bias":   dpd_biased or eod_biased or dir_biased,
            }

            if abs(dpd) > abs(worst_dpd): worst_dpd = dpd
            if abs(eod) > abs(worst_eod): worst_eod = eod
            if dir_ < worst_dir:          worst_dir = dir_

        metrics["summary"] = {
            "worst_dpd": round(worst_dpd, 4),
            "worst_eod": round(worst_eod, 4),
            "worst_dir": round(worst_dir, 4),
            "is_biased": (
                abs(worst_dpd) > FAIRNESS_THRESHOLDS["demographic_parity_difference"] or
                abs(worst_eod) > FAIRNESS_THRESHOLDS["equal_opportunity_difference"] or
                worst_dir      < FAIRNESS_THRESHOLDS["disparate_impact_ratio"]
            )
        }
        return metrics

    # ══════════════════════════════════════════════════════════════════════════
    # REPORTING
    # ══════════════════════════════════════════════════════════════════════════

    def print_report(self, report: Dict) -> None:
        """Prints a human-readable bias audit report."""
        print("\n" + "═"*68)
        print(f"  🔍 BIAS AUDIT: {report['model'].upper()}")
        print(f"  Dataset: {report['dataset']} | "
              f"N={report['n_samples']:,} | "
              f"Overall Pos Rate: {report['overall_pos_rate']:.1%}")
        print("═"*68)

        for attr, attr_data in report["attributes"].items():
            print(f"\n  📌 Attribute: [{attr.upper()}]  "
                  f"(privileged='{attr_data['privileged_group']}')")

            # Group stats table
            print(f"\n  {'Group':<20} {'N':>6} {'Pred+ Rate':>11} "
                  f"{'TPR':>8} {'FPR':>8} {'Accuracy':>10}")
            print("  " + "─"*65)

            for grp, s in attr_data["group_stats"].items():
                tag = " ◀ priv" if s.get("is_privileged") else ""
                print(f"  {grp:<20} {s['n']:>6,} "
                      f"{s['pos_pred_rate']:>10.1%} "
                      f"{s['true_pos_rate']:>8.1%} "
                      f"{s['false_pos_rate']:>8.1%} "
                      f"{s['accuracy']:>9.1%}{tag}")

            # Fairness metrics table
            fm = attr_data["fairness_metrics"]
            print(f"\n  {'Comparison':<32} {'DPD':>8} {'EOD':>8} "
                  f"{'DIR':>8} {'Biased?':>10}")
            print("  " + "─"*68)

            for pair, v in fm["pairwise"].items():
                flag   = "🚨 YES" if v["any_bias"] else "✅ NO"
                d_warn = "⚠" if v["dpd_biased"] else " "
                e_warn = "⚠" if v["eod_biased"] else " "
                r_warn = "⚠" if v["dir_biased"] else " "
                label  = pair.replace("_vs_", " vs ")[:31]
                print(f"  {label:<32} "
                      f"{v['demographic_parity_difference']:>+7.3f}{d_warn} "
                      f"{v['equal_opportunity_difference']:>+7.3f}{e_warn} "
                      f"{v['disparate_impact_ratio']:>7.3f}{r_warn} "
                      f"{flag:>10}")

            s = fm["summary"]
            verdict = "🚨 BIASED" if s["is_biased"] else "✅ FAIR"
            print(f"\n  ▶ Worst DPD={s['worst_dpd']:+.3f} | "
                  f"EOD={s['worst_eod']:+.3f} | "
                  f"DIR={s['worst_dir']:.3f} | "
                  f"Verdict: {verdict}")

        print("\n" + "─"*68)
        print("  Thresholds: |DPD| < 0.10 | |EOD| < 0.10 | DIR > 0.80")
        print("═"*68 + "\n")

    def compare_models(self, model_reports: List[Dict]) -> pd.DataFrame:
        """
        Returns a DataFrame comparing fairness metrics across models.
        Useful for spotting the accuracy-fairness trade-off.
        """
        rows = []
        for report in model_reports:
            for attr, attr_data in report["attributes"].items():
                s = attr_data["fairness_metrics"].get("summary", {})
                rows.append({
                    "Model":     report["model"],
                    "Attribute": attr,
                    "Worst DPD": s.get("worst_dpd"),
                    "Worst EOD": s.get("worst_eod"),
                    "Worst DIR": s.get("worst_dir"),
                    "Is Biased": s.get("is_biased"),
                })
        return pd.DataFrame(rows)