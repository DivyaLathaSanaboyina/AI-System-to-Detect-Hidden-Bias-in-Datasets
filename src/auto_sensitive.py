# src/auto_sensitive.py
"""
Auto Sensitive Attribute Detector — Phase 7

Automatically discovers bias-prone features using three
complementary statistical methods:

  Method 1: Target Correlation Analysis
            → Which features are strongly associated with the outcome?
            → Uses Chi-squared (categorical) + Point-Biserial (continuous)

  Method 2: Proxy Detection
            → Which features are proxies for known sensitive attributes?
            → Uses Cramer's V (categorical association)

  Method 3: Disparate Impact Scan
            → Which features create subgroup disparities > threshold?
            → Runs DIR automatically on every categorical feature

Research basis:
  - Feldman et al. (2015) "Certifying and Removing Disparate Impact"
  - Kamiran & Calders (2012) "Data Preprocessing Techniques for
    Classification without Discrimination"
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

from scipy import stats
from scipy.stats import chi2_contingency, pointbiserialr

logger = logging.getLogger(__name__)

# ── Thresholds ─────────────────────────────────────────────────────────────────
CORRELATION_THRESHOLD   = 0.15   # Min correlation to flag as sensitive
PROXY_THRESHOLD         = 0.20   # Min Cramer's V to flag as proxy
DI_THRESHOLD            = 0.80   # Disparate Impact ratio threshold
PVALUE_THRESHOLD        = 0.05   # Statistical significance level


class AutoSensitiveDetector:
    """
    Automatically detects potentially sensitive or bias-prone
    features in a dataset without requiring prior labeling.

    Usage:
        detector = AutoSensitiveDetector(df, target_col="income")
        results  = detector.run_full_detection()
        detector.print_report(results)
    """

    def __init__(
        self,
        df:             pd.DataFrame,
        target_col:     str,
        known_sensitive: Optional[List[str]] = None
    ):
        """
        Args:
            df:              Raw DataFrame (before encoding)
            target_col:      Name of target/label column
            known_sensitive: Already-known sensitive attributes
                             (used for proxy detection)
        """
        self.df              = df.copy()
        self.target_col      = target_col
        self.known_sensitive = known_sensitive or []
        self.feature_cols    = [c for c in df.columns if c != target_col]

        # Separate column types
        self.cat_cols = [
            c for c in self.feature_cols
            if df[c].dtype == "object" or df[c].nunique() <= 10
        ]
        self.num_cols = [
            c for c in self.feature_cols
            if c not in self.cat_cols
        ]

        # Binary encode target if needed
        self._y = self._encode_target()

        logger.info(
            f"AutoSensitiveDetector initialized: "
            f"{len(self.cat_cols)} categorical, "
            f"{len(self.num_cols)} numerical features"
        )

    def _encode_target(self) -> np.ndarray:
        """Encodes target as binary 0/1."""
        y_raw = self.df[self.target_col]
        if y_raw.dtype == "object":
            # Use most frequent as positive label heuristic
            most_freq = y_raw.value_counts().index[0]
            return (y_raw != most_freq).astype(int).values
        return (y_raw > y_raw.median()).astype(int).values

    # ══════════════════════════════════════════════════════════════════════════
    # METHOD 1: TARGET CORRELATION ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════

    def detect_target_correlation(self) -> pd.DataFrame:
        """
        Identifies features strongly correlated with the target.

        For CATEGORICAL features:
          → Chi-squared test of independence
          → Cramer's V as effect size (0=no association, 1=perfect)

        For CONTINUOUS features:
          → Point-biserial correlation (r between continuous X and binary y)

        Why this matters:
          A feature correlated with the target AND with a sensitive attribute
          acts as a "proxy discriminator" — the model uses it to discriminate
          indirectly even if the sensitive attribute is removed.
        """
        results = []

        # ── Categorical features ───────────────────────────────────────────
        for col in self.cat_cols:
            if self.df[col].isnull().all():
                continue
            try:
                contingency = pd.crosstab(
                    self.df[col].fillna("MISSING"),
                    self._y
                )
                chi2, pval, dof, _ = chi2_contingency(contingency)

                # Cramer's V: normalized chi-squared effect size
                n      = len(self.df)
                k      = min(contingency.shape) - 1
                cramers_v = np.sqrt(chi2 / (n * max(k, 1))) if k > 0 else 0.0

                results.append({
                    "feature":       col,
                    "type":          "categorical",
                    "method":        "Chi-squared + Cramer's V",
                    "statistic":     round(cramers_v, 4),
                    "p_value":       round(pval, 6),
                    "significant":   pval < PVALUE_THRESHOLD,
                    "effect_size":   self._effect_label(cramers_v),
                    "flagged":       cramers_v >= CORRELATION_THRESHOLD
                                     and pval < PVALUE_THRESHOLD
                })
            except Exception as e:
                logger.debug(f"Chi2 failed for {col}: {e}")

        # ── Continuous features ────────────────────────────────────────────
        for col in self.num_cols:
            try:
                valid = self.df[col].dropna()
                valid_y = self._y[self.df[col].notna()]
                if len(valid) < 10:
                    continue

                r, pval = pointbiserialr(valid, valid_y)
                abs_r   = abs(r)

                results.append({
                    "feature":       col,
                    "type":          "continuous",
                    "method":        "Point-biserial correlation",
                    "statistic":     round(abs_r, 4),
                    "p_value":       round(pval, 6),
                    "significant":   pval < PVALUE_THRESHOLD,
                    "effect_size":   self._effect_label(abs_r),
                    "flagged":       abs_r >= CORRELATION_THRESHOLD
                                     and pval < PVALUE_THRESHOLD
                })
            except Exception as e:
                logger.debug(f"PBR failed for {col}: {e}")

        df_results = pd.DataFrame(results)
        if not df_results.empty:
            df_results = df_results.sort_values("statistic", ascending=False)
        return df_results

    # ══════════════════════════════════════════════════════════════════════════
    # METHOD 2: PROXY DETECTION
    # ══════════════════════════════════════════════════════════════════════════

    def detect_proxies(self) -> pd.DataFrame:
        """
        Identifies features that are proxies for known sensitive attributes.

        Uses Cramer's V to measure association between each feature
        and each known sensitive attribute.

        Example: In the Adult dataset:
          - 'relationship' contains 'Wife' — strongly correlated with sex
          - 'occupation' is correlated with both sex and race
          These features allow the model to discriminate by proxy
          even if 'sex' and 'race' are explicitly removed.

        This is the "fairness through unawareness fallacy" —
        removing sensitive attributes doesn't prevent discrimination
        if correlated proxies remain in the data.
        """
        if not self.known_sensitive:
            return pd.DataFrame(
                columns=["feature", "sensitive_attr",
                         "cramers_v", "flagged_as_proxy"]
            )

        results = []
        for col in self.cat_cols:
            if col in self.known_sensitive:
                continue
            for sens_attr in self.known_sensitive:
                if sens_attr not in self.df.columns:
                    continue
                try:
                    ct      = pd.crosstab(
                        self.df[col].fillna("MISSING"),
                        self.df[sens_attr].fillna("MISSING")
                    )
                    chi2, _, _, _ = chi2_contingency(ct)
                    n = len(self.df)
                    k = min(ct.shape) - 1
                    v = np.sqrt(chi2 / (n * max(k, 1))) if k > 0 else 0.0

                    results.append({
                        "feature":          col,
                        "sensitive_attr":   sens_attr,
                        "cramers_v":        round(v, 4),
                        "effect_size":      self._effect_label(v),
                        "flagged_as_proxy": v >= PROXY_THRESHOLD
                    })
                except Exception as e:
                    logger.debug(f"Proxy check failed {col}↔{sens_attr}: {e}")

        df_results = pd.DataFrame(results)
        if not df_results.empty:
            df_results = df_results.sort_values("cramers_v", ascending=False)
        return df_results

    # ══════════════════════════════════════════════════════════════════════════
    # METHOD 3: DISPARATE IMPACT SCAN
    # ══════════════════════════════════════════════════════════════════════════

    def scan_disparate_impact(self) -> pd.DataFrame:
        """
        Runs an automated Disparate Impact scan across all
        categorical features.

        For each categorical feature:
          1. Group the dataset by that feature's values
          2. Compute positive outcome rate per group
          3. Calculate DIR = min_group_rate / max_group_rate
          4. Flag if DIR < 0.80 (EEOC threshold)

        This reveals features that CREATE disparate impact
        in the GROUND TRUTH data — before any model is applied.
        These are features where the historical data itself is biased.
        """
        results = []

        for col in self.cat_cols:
            try:
                groups     = self.df.groupby(col)
                group_rates = {}

                for val, group_idx in groups.groups.items():
                    y_group = self._y[group_idx]
                    if len(y_group) >= 5:   # Min group size for reliability
                        group_rates[str(val)] = float(y_group.mean())

                if len(group_rates) < 2:
                    continue

                max_rate = max(group_rates.values())
                min_rate = min(group_rates.values())

                if max_rate == 0:
                    continue

                dir_ratio   = min_rate / max_rate
                max_grp     = max(group_rates, key=group_rates.get)
                min_grp     = min(group_rates, key=group_rates.get)
                rate_spread = max_rate - min_rate

                results.append({
                    "feature":          col,
                    "n_groups":         len(group_rates),
                    "max_rate_group":   max_grp,
                    "max_rate":         round(max_rate, 4),
                    "min_rate_group":   min_grp,
                    "min_rate":         round(min_rate, 4),
                    "rate_spread":      round(rate_spread, 4),
                    "disparate_impact": round(dir_ratio, 4),
                    "flagged":          dir_ratio < DI_THRESHOLD
                })
            except Exception as e:
                logger.debug(f"DI scan failed for {col}: {e}")

        df_results = pd.DataFrame(results)
        if not df_results.empty:
            df_results = df_results.sort_values(
                "disparate_impact", ascending=True
            )
        return df_results

    # ══════════════════════════════════════════════════════════════════════════
    # FULL DETECTION PIPELINE
    # ══════════════════════════════════════════════════════════════════════════

    def run_full_detection(self) -> Dict:
        """
        Runs all three detection methods and returns a consolidated report.

        Returns:
            Dict with keys: target_correlation, proxies, disparate_impact,
                            auto_flagged_features, summary
        """
        logger.info("🔍 Running auto sensitive attribute detection...")

        logger.info("  → Method 1: Target correlation analysis...")
        target_corr = self.detect_target_correlation()

        logger.info("  → Method 2: Proxy detection...")
        proxies = self.detect_proxies()

        logger.info("  → Method 3: Disparate impact scan...")
        di_scan = self.scan_disparate_impact()

        # Consolidate: collect all flagged features across methods
        flagged = set()

        if not target_corr.empty:
            flagged.update(
                target_corr[target_corr["flagged"]]["feature"].tolist()
            )
        if not proxies.empty:
            flagged.update(
                proxies[proxies["flagged_as_proxy"]]["feature"].tolist()
            )
        if not di_scan.empty:
            flagged.update(
                di_scan[di_scan["flagged"]]["feature"].tolist()
            )

        # Score each feature by how many methods flagged it
        feature_scores = {}
        for feat in flagged:
            score = 0
            reasons = []

            if not target_corr.empty:
                row = target_corr[target_corr["feature"] == feat]
                if not row.empty and row.iloc[0]["flagged"]:
                    score += 1
                    reasons.append(
                        f"target_corr={row.iloc[0]['statistic']:.3f}"
                    )

            if not proxies.empty:
                proxy_rows = proxies[
                    (proxies["feature"] == feat) &
                    (proxies["flagged_as_proxy"])
                ]
                if not proxy_rows.empty:
                    score += 1
                    attrs = proxy_rows["sensitive_attr"].tolist()
                    reasons.append(f"proxy_for={attrs}")

            if not di_scan.empty:
                row = di_scan[di_scan["feature"] == feat]
                if not row.empty and row.iloc[0]["flagged"]:
                    score += 1
                    reasons.append(
                        f"DI={row.iloc[0]['disparate_impact']:.3f}"
                    )

            feature_scores[feat] = {
                "methods_flagged": score,
                "confidence": ["Low", "Medium", "High"][min(score - 1, 2)],
                "reasons":    reasons
            }

        # Sort by confidence
        sorted_features = sorted(
            feature_scores.items(),
            key=lambda x: x[1]["methods_flagged"],
            reverse=True
        )

        return {
            "target_correlation": target_corr,
            "proxies":            proxies,
            "disparate_impact":   di_scan,
            "auto_flagged_features": dict(sorted_features),
            "summary": {
                "total_features_scanned": len(self.feature_cols),
                "features_flagged":       len(flagged),
                "high_confidence":  sum(
                    1 for v in feature_scores.values()
                    if v["methods_flagged"] == 3
                ),
                "medium_confidence": sum(
                    1 for v in feature_scores.values()
                    if v["methods_flagged"] == 2
                ),
                "low_confidence": sum(
                    1 for v in feature_scores.values()
                    if v["methods_flagged"] == 1
                ),
            }
        }

    # ══════════════════════════════════════════════════════════════════════════
    # REPORTING
    # ══════════════════════════════════════════════════════════════════════════

    def print_report(self, results: Dict) -> None:
        """Prints a comprehensive auto-detection report."""
        summary = results["summary"]

        print("\n" + "═"*68)
        print("  🤖 AUTO SENSITIVE ATTRIBUTE DETECTION REPORT")
        print("═"*68)
        print(f"  Features scanned:  {summary['total_features_scanned']}")
        print(f"  Features flagged:  {summary['features_flagged']}")
        print(f"  High confidence:   {summary['high_confidence']}  "
              f"(flagged by ALL 3 methods)")
        print(f"  Medium confidence: {summary['medium_confidence']}  "
              f"(flagged by 2 methods)")
        print(f"  Low confidence:    {summary['low_confidence']}  "
              f"(flagged by 1 method)")

        # ── Method 1: Target correlation ───────────────────────────────────
        print(f"\n{'─'*68}")
        print("  📊 METHOD 1: Target Correlation (top 10 flagged)")
        print(f"{'─'*68}")
        tc = results["target_correlation"]
        if not tc.empty:
            flagged_tc = tc[tc["flagged"]].head(10)
            if flagged_tc.empty:
                print("  No features flagged.")
            else:
                print(f"  {'Feature':<25} {'Type':<12} {'Statistic':>10} "
                      f"{'p-value':>10} {'Effect':>8}")
                print("  " + "─"*65)
                for _, row in flagged_tc.iterrows():
                    print(f"  {row['feature']:<25} {row['type']:<12} "
                          f"{row['statistic']:>10.4f} "
                          f"{row['p_value']:>10.6f} "
                          f"{row['effect_size']:>8}")

        # ── Method 2: Proxy detection ──────────────────────────────────────
        print(f"\n{'─'*68}")
        print("  🔗 METHOD 2: Proxy Detection (top 10 flagged)")
        print(f"{'─'*68}")
        px = results["proxies"]
        if not px.empty:
            flagged_px = px[px["flagged_as_proxy"]].head(10)
            if flagged_px.empty:
                print("  No proxies detected.")
            else:
                print(f"  {'Feature':<25} {'Proxies For':<20} "
                      f"{'Cramers V':>10} {'Effect':>8}")
                print("  " + "─"*65)
                for _, row in flagged_px.iterrows():
                    print(f"  {row['feature']:<25} "
                          f"{row['sensitive_attr']:<20} "
                          f"{row['cramers_v']:>10.4f} "
                          f"{row['effect_size']:>8}")

        # ── Method 3: Disparate impact scan ───────────────────────────────
        print(f"\n{'─'*68}")
        print("  ⚖️  METHOD 3: Disparate Impact Scan (top 10 worst)")
        print(f"{'─'*68}")
        di = results["disparate_impact"]
        if not di.empty:
            flagged_di = di[di["flagged"]].head(10)
            if flagged_di.empty:
                print("  No disparate impact detected.")
            else:
                print(f"  {'Feature':<22} {'Min Group':<20} "
                      f"{'Max Group':<20} {'DIR':>7} {'Flagged':>8}")
                print("  " + "─"*65)
                for _, row in flagged_di.iterrows():
                    print(f"  {row['feature']:<22} "
                          f"{str(row['min_rate_group'])[:19]:<20} "
                          f"{str(row['max_rate_group'])[:19]:<20} "
                          f"{row['disparate_impact']:>7.3f} "
                          f"{'🚨':>8}")

        # ── Final ranked list ──────────────────────────────────────────────
        print(f"\n{'─'*68}")
        print("  🏆 AUTO-DETECTED SENSITIVE FEATURES (Ranked by Confidence)")
        print(f"{'─'*68}")
        flagged = results["auto_flagged_features"]
        if not flagged:
            print("  No sensitive features detected.")
        else:
            print(f"  {'Feature':<25} {'Confidence':<12} "
                  f"{'Methods':>8}  Evidence")
            print("  " + "─"*65)
            for feat, info in flagged.items():
                conf_icon = (
                    "🔴" if info["confidence"] == "High"   else
                    "🟡" if info["confidence"] == "Medium" else "🟢"
                )
                reasons_str = " | ".join(info["reasons"])
                print(f"  {feat:<25} "
                      f"{conf_icon} {info['confidence']:<10} "
                      f"{info['methods_flagged']:>8}  "
                      f"{reasons_str}")

        print("\n" + "═"*68 + "\n")

    @staticmethod
    def _effect_label(v: float) -> str:
        """Labels effect size as Negligible/Small/Medium/Large."""
        if v < 0.10:  return "Negligible"
        if v < 0.20:  return "Small"
        if v < 0.40:  return "Medium"
        return "Large"