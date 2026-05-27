# app/streamlit_app.py
"""
AI Bias Detection System — Streamlit Dashboard
Phase 12: Interactive Web UI

Run with:
    streamlit run app/streamlit_app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Bias Detection System",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem; font-weight: 800;
        background: linear-gradient(90deg, #1565C0, #C62828);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .metric-card {
        background: #f8f9fa; border-radius: 10px;
        padding: 1rem; border-left: 4px solid #1565C0;
        margin: 0.3rem 0;
    }
    .biased-card  { border-left-color: #C62828 !important; }
    .fair-card    { border-left-color: #2E7D32 !important; }
    .section-header {
        font-size: 1.2rem; font-weight: 700;
        color: #1565C0; border-bottom: 2px solid #1565C0;
        padding-bottom: 0.3rem; margin: 1rem 0 0.5rem 0;
    }
    .finding-box {
        background: #fff8e1; border: 1px solid #f9a825;
        border-radius: 8px; padding: 0.8rem; margin: 0.3rem 0;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Imports ────────────────────────────────────────────────────────────────────
from src.data_loader   import DataLoader
from src.preprocessor  import Preprocessor
from src.model_trainer import ModelTrainer
from src.bias_detector import BiasDetector
from src.bias_mitigator import BiasMitigator
from src.auto_sensitive import AutoSensitiveDetector

FAIRNESS_THRESHOLDS = {
    "demographic_parity_difference": 0.10,
    "equal_opportunity_difference":  0.10,
    "disparate_impact_ratio":        0.80,
}

# ══════════════════════════════════════════════════════════════════════════════
# CACHED PIPELINE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_and_preprocess(dataset_name):
    loader = DataLoader(dataset_name)
    df     = loader.load()
    prep   = Preprocessor(dataset_name)
    X_train, X_test, y_train, y_test = prep.fit_transform(df)
    s_train = prep.sensitive_train
    s_test  = prep.sensitive_test
    return df, X_train, X_test, y_train, y_test, s_train, s_test

@st.cache_data(show_spinner=False)
def train_models(dataset_name, _X_train, _X_test, _y_train, _y_test):
    trainer = ModelTrainer()
    results = trainer.train_all(
        _X_train, _X_test, _y_train, _y_test,
        dataset_name=dataset_name
    )
    return trainer, results

@st.cache_data(show_spinner=False)
def run_bias_audit(dataset_name, _X_test, _y_test,
                   _sensitive_test, _results, privileged_groups):
    detector = BiasDetector(
        sensitive_df = _sensitive_test,
        y_true       = _y_test,
        dataset_name = dataset_name
    )
    reports = []
    preds   = {}
    for model_name, r in _results.items():
        y_pred = r["y_pred"]
        report = detector.run_full_audit(
            model_name        = model_name,
            y_pred            = y_pred,
            privileged_groups = privileged_groups
        )
        reports.append(report)
        preds[model_name] = y_pred
    return reports, detector

# ══════════════════════════════════════════════════════════════════════════════
# HELPER CHARTS
# ══════════════════════════════════════════════════════════════════════════════

def make_group_bar_chart(audit_reports, sensitive_col, privileged_val):
    model_names  = [r["model"] for r in audit_reports]
    priv_rates, unpriv_rates = [], []
    for report in audit_reports:
        gs = report["attributes"].get(sensitive_col, {}).get("group_stats", {})
        p  = gs.get(privileged_val, {}).get("pos_pred_rate", 0)
        u  = np.mean([v["pos_pred_rate"] for k,v in gs.items()
                      if k != privileged_val]) if gs else 0
        priv_rates.append(p)
        unpriv_rates.append(u)

    fig, ax = plt.subplots(figsize=(8, 4))
    x, w = np.arange(len(model_names)), 0.35
    ax.bar(x-w/2, priv_rates,   w, label=f"Privileged ({privileged_val})",
           color="#1565C0", alpha=0.85, edgecolor="white")
    ax.bar(x+w/2, unpriv_rates, w, label="Unprivileged",
           color="#C62828", alpha=0.85, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(model_names, fontsize=9)
    ax.set_ylabel("Positive Prediction Rate"); ax.legend(fontsize=9)
    ax.set_title("Group Prediction Rates", fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    return fig

def make_fairness_bar_chart(audit_reports, sensitive_col):
    model_names = [r["model"] for r in audit_reports]
    dpds, eods, dirs_ = [], [], []
    for report in audit_reports:
        s = report["attributes"].get(sensitive_col, {}).get(
            "fairness_metrics", {}).get("summary", {})
        dpds.append(abs(s.get("worst_dpd") or 0))
        eods.append(abs(s.get("worst_eod") or 0))
        dirs_.append(s.get("worst_dir") or 0)

    fig, ax = plt.subplots(figsize=(8, 4))
    x, w = np.arange(len(model_names)), 0.25
    ax.bar(x-w, dpds,  w, label="|DPD|", color="#E53935", alpha=0.8)
    ax.bar(x,   eods,  w, label="|EOD|", color="#FB8C00", alpha=0.8)
    ax.bar(x+w, dirs_, w, label="DIR",   color="#43A047", alpha=0.8)
    ax.axhline(0.10, color="#E53935", linestyle=":", alpha=0.7)
    ax.axhline(0.80, color="#43A047", linestyle=":", alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(model_names, fontsize=9)
    ax.legend(fontsize=9)
    ax.set_title("Fairness Metrics per Model", fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    return fig

def make_accuracy_vs_fairness(model_perf, model_fair, sensitive_col):
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"Logistic Regression": "#2196F3",
              "Decision Tree": "#FF9800", "Random Forest": "#4CAF50"}
    for name, perf in model_perf.items():
        fair = model_fair.get(name, {})
        acc  = perf.get("accuracy", 0)
        dir_ = fair.get("worst_dir") or 0
        c    = colors.get(name, "#9E9E9E")
        ax.scatter(acc, dir_, s=200, color=c, zorder=5,
                   edgecolors="white", linewidths=2)
        ax.annotate(name, (acc, dir_), textcoords="offset points",
                    xytext=(6, 3), fontsize=9, color=c, fontweight="bold")
    ax.axhline(0.80, color="#2E7D32", linestyle="--",
               linewidth=1.8, label="Fair threshold (0.80)")
    ax.set_xlabel("Accuracy", fontsize=11)
    ax.set_ylabel("Disparate Impact Ratio", fontsize=11)
    ax.set_title("Accuracy vs Fairness Trade-off", fontweight="bold")
    ax.legend(fontsize=9); ax.set_ylim(bottom=0)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    return fig

def make_mitigation_chart(mit_results, sensitive_col):
    short = {
        "Baseline (No Mitigation)":               "Baseline",
        "Re-sampling (SMOTE)":                    "SMOTE",
        "Re-weighting (Kamiran & Calders)":        "Reweighting",
        "Fair Classifier (ExponentiatedGradient)": "FairClassifier",
        "Fair Classifier (fallback: strong reweighting)": "Fair(fallback)",
        "Threshold Adjustment (Hardt et al.)":     "Threshold Adj",
    }
    labels = [short.get(r["strategy"], r["strategy"][:14])
              for r in mit_results]
    dirs   = [r.get("dir") or 0 for r in mit_results]
    accs   = [r.get("accuracy") or 0 for r in mit_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    bar_c = ["#C62828" if i == 0 else "#1565C0" for i in range(len(labels))]
    ax1.bar(range(len(labels)), accs, color=bar_c, alpha=0.85,
            edgecolor="white")
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax1.set_ylabel("Accuracy"); ax1.set_title("Accuracy", fontweight="bold")
    ax1.spines[["top","right"]].set_visible(False)

    dir_c = ["#2E7D32" if d >= 0.80 else "#C62828" for d in dirs]
    ax2.bar(range(len(labels)), dirs, color=dir_c, alpha=0.85,
            edgecolor="white")
    ax2.axhline(0.80, color="black", linestyle="--",
                linewidth=1.5, label="Fair (0.80)")
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax2.set_ylabel("DIR"); ax2.set_title("Fairness (DIR)", fontweight="bold")
    ax2.legend(fontsize=8); ax2.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.image("https://img.icons8.com/color/96/scales.png", width=60)
    st.markdown("## ⚖️ Bias Audit System")
    st.markdown("---")

    dataset_choice = st.selectbox(
        "📊 Select Dataset",
        ["Adult Income", "German Credit"],
        help="Adult: 48K US census records\nGerman: 1K credit applications"
    )
    dataset_name = "adult" if dataset_choice == "Adult Income" else "german"

    if dataset_name == "adult":
        sensitive_col  = st.selectbox("🎯 Sensitive Attribute", ["sex", "race"])
        privileged_val = "Male" if sensitive_col == "sex" else "White"
        privileged_groups = {"sex": "Male", "race": "White"}
    else:
        sensitive_col  = st.selectbox("🎯 Sensitive Attribute",
                                       ["age_group", "sex"])
        privileged_val = "adult" if sensitive_col == "age_group" else "male"
        privileged_groups = {"sex": "male", "age_group": "adult"}

    st.markdown("---")
    run_mitigation = st.checkbox("⚙️ Run Bias Mitigation", value=True)
    st.markdown("---")
    st.markdown("""
    **Fairness Thresholds:**
    - |DPD| < 0.10
    - |EOD| < 0.10
    - DIR  > 0.80 *(EEOC 80% rule)*
    """)
    st.markdown("---")
    st.markdown("*Built as a Final Year B.Tech Project*")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<p class="main-header">⚖️ AI System to Detect Hidden Bias</p>',
            unsafe_allow_html=True)
st.markdown(
    f"**Dataset:** {dataset_choice} &nbsp;|&nbsp; "
    f"**Sensitive Attribute:** `{sensitive_col}` &nbsp;|&nbsp; "
    f"**Privileged Group:** `{privileged_val}`"
)
st.markdown("---")

# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner("Loading and preprocessing dataset..."):
    df, X_train, X_test, y_train, y_test, s_train, s_test = \
        load_and_preprocess(dataset_name)

# ── Tab layout ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dataset Overview",
    "🤖 Model Training",
    "🔍 Bias Detection",
    "🔧 Mitigation",
    "📋 Audit Report"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: DATASET OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-header">Dataset Summary</p>',
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{len(df):,}")
    c2.metric("Features", f"{df.shape[1]-1}")
    c3.metric("Missing Values", f"{df.isnull().sum().sum():,}")
    target = "income" if dataset_name == "adult" else "credit_risk"
    pos_label = ">50K" if dataset_name == "adult" else 1
    pos_rate = (df[target] == pos_label).mean() if isinstance(pos_label, str) \
               else (df[target] == pos_label).mean()
    c4.metric("Positive Rate", f"{pos_rate:.1%}")

    st.markdown('<p class="section-header">Sample Data</p>',
                unsafe_allow_html=True)
    st.dataframe(df.head(8), use_container_width=True)

    st.markdown('<p class="section-header">Bias Signal in Raw Data</p>',
                unsafe_allow_html=True)
    st.info("⚠️ Before training any model, the raw data already shows disparate outcomes by group. This is historical bias encoded in data.")

    if sensitive_col in df.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        groups = df.groupby(sensitive_col)[target]
        group_names, rates = [], []
        for gname, gdata in groups:
            rate = (gdata == pos_label).mean() if isinstance(pos_label, str) \
                   else gdata.mean()
            group_names.append(str(gname))
            rates.append(rate)

        colors = ["#1565C0" if g == privileged_val else "#C62828"
                  for g in group_names]
        ax.bar(group_names, rates, color=colors, alpha=0.85,
               edgecolor="white")
        ax.axhline(np.mean(rates), color="black", linestyle="--",
                   linewidth=1.5, label=f"Overall mean: {np.mean(rates):.1%}")
        ax.set_ylabel("Favorable Outcome Rate")
        ax.set_title(f"Raw Outcome Rates by [{sensitive_col}]",
                     fontweight="bold")
        ax.legend(); ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-header">Training 3 Models</p>',
                unsafe_allow_html=True)

    with st.spinner("Training Logistic Regression, Decision Tree, Random Forest..."):
        trainer, results = train_models(
            dataset_name, X_train, X_test, y_train, y_test
        )

    st.success(f"✅ All 3 models trained successfully!")

    # Metrics table
    rows = []
    for name, r in results.items():
        rows.append({
            "Model": name,
            "Accuracy": f"{r['accuracy']:.3f}",
            "Precision": f"{r['precision']:.3f}",
            "Recall": f"{r['recall']:.3f}",
            "F1 Score": f"{r['f1']:.3f}",
            "ROC-AUC": f"{r['roc_auc']:.3f}",
            "Train Time": f"{r['train_time_sec']:.2f}s"
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True,
                 hide_index=True)

    st.markdown('<div class="finding-box">⚠️ <b>Key Insight:</b> '
                'Accuracy alone is misleading. A model predicting only the '
                'majority class achieves high accuracy with zero usefulness. '
                'This is the <i>accuracy paradox</i> — why we need fairness metrics.'
                '</div>', unsafe_allow_html=True)

    # Confusion matrices
    st.markdown('<p class="section-header">Confusion Matrices</p>',
                unsafe_allow_html=True)
    cols = st.columns(3)
    for i, (name, r) in enumerate(results.items()):
        with cols[i]:
            st.markdown(f"**{name}**")
            cm_df = pd.DataFrame(
                [[r["tn"], r["fp"]], [r["fn"], r["tp"]]],
                index=["Actual 0", "Actual 1"],
                columns=["Pred 0", "Pred 1"]
            )
            st.dataframe(cm_df)
            tpr = r["tp"]/(r["tp"]+r["fn"]) if (r["tp"]+r["fn"])>0 else 0
            st.caption(f"TPR: {tpr:.3f} | FPR: {r['fp']/(r['fp']+r['tn']):.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: BIAS DETECTION
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-header">Fairness Metrics Analysis</p>',
                unsafe_allow_html=True)

    with st.spinner("Running bias audit..."):
        audit_reports, detector = run_bias_audit(
            dataset_name, X_test, y_test, s_test,
            results, privileged_groups
        )

    # Summary verdict per model
    st.markdown("**Model Verdicts:**")
    vcols = st.columns(3)
    model_perf, model_fair = {}, {}
    for i, report in enumerate(audit_reports):
        mname = report["model"]
        r     = results[mname]
        attr  = report["attributes"].get(sensitive_col, {})
        s     = attr.get("fairness_metrics", {}).get("summary", {})
        dpd   = s.get("worst_dpd") or 0
        eod   = s.get("worst_eod") or 0
        dir_  = s.get("worst_dir") or 0
        biased = s.get("is_biased", True)

        model_perf[mname] = {"accuracy": r["accuracy"]}
        model_fair[mname] = {"worst_dir": dir_}

        with vcols[i]:
            card_class = "biased-card" if biased else "fair-card"
            verdict    = "🚨 BIASED" if biased else "✅ FAIR"
            st.markdown(
                f'<div class="metric-card {card_class}">'
                f'<b>{mname}</b><br>'
                f'Acc: {r["accuracy"]:.3f}<br>'
                f'DPD: {dpd:+.3f}<br>'
                f'EOD: {eod:+.3f}<br>'
                f'DIR: {dir_:.3f}<br>'
                f'<b>{verdict}</b></div>',
                unsafe_allow_html=True
            )

    st.markdown("---")

    # Charts
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Group Prediction Rates**")
        st.pyplot(make_group_bar_chart(
            audit_reports, sensitive_col, privileged_val
        ))
        plt.close()
    with col_b:
        st.markdown("**Fairness Metrics**")
        st.pyplot(make_fairness_bar_chart(audit_reports, sensitive_col))
        plt.close()

    st.markdown("**Accuracy vs Fairness Trade-off**")
    st.pyplot(make_accuracy_vs_fairness(model_perf, model_fair, sensitive_col))
    plt.close()

    # Detailed table
    st.markdown('<p class="section-header">Detailed Fairness Table</p>',
                unsafe_allow_html=True)
    comparison_df = detector.compare_models(audit_reports)
    comparison_df = comparison_df[comparison_df["Attribute"] == sensitive_col]
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: MITIGATION
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<p class="section-header">Bias Mitigation Strategies</p>',
                unsafe_allow_html=True)

    if not run_mitigation:
        st.info("Enable 'Run Bias Mitigation' in the sidebar to see results.")
    else:
        with st.spinner("Running 4 mitigation strategies..."):
            mitigator = BiasMitigator(
                X_train         = X_train,
                X_test          = X_test,
                y_train         = y_train,
                y_test          = y_test,
                sensitive_train = s_train[sensitive_col],
                sensitive_test  = s_test[sensitive_col],
                sensitive_col   = sensitive_col,
                privileged_val  = privileged_val,
            )
            mit_results = mitigator.run_all()

        # Strategy explanation
        st.markdown("""
        | Stage | Strategy | How It Works |
        |-------|----------|-------------|
        | Pre-processing | **Re-sampling (SMOTE)** | Creates synthetic samples for underrepresented group |
        | Pre-processing | **Re-weighting** | Assigns higher loss weight to underrepresented group |
        | In-processing | **Fair Classifier** | Optimizes accuracy subject to fairness constraint |
        | Post-processing | **Threshold Adjustment** | Sets group-specific decision boundaries |
        """)

        # Results table
        rows = []
        for r in mit_results:
            biased = (abs(r.get("dpd") or 0) > 0.10 or
                      abs(r.get("eod") or 0) > 0.10 or
                      (r.get("dir") or 1.0) < 0.80)
            rows.append({
                "Strategy":  r["strategy"],
                "Accuracy":  f"{r.get('accuracy') or 0:.3f}",
                "F1":        f"{r.get('f1') or 0:.3f}",
                "DPD":       f"{r.get('dpd') or 0:+.3f}",
                "EOD":       f"{r.get('eod') or 0:+.3f}",
                "DIR":       f"{r.get('dir') or 0:.3f}",
                "Fair?":     "✅ YES" if not biased else "🚨 NO"
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True,
                     hide_index=True)

        st.markdown("**Before vs After Mitigation:**")
        st.pyplot(make_mitigation_chart(mit_results, sensitive_col))
        plt.close()

        st.markdown('<div class="finding-box">💡 <b>Key Insight:</b> '
                    'Every mitigation strategy trades some accuracy for fairness. '
                    'The threshold adjustment is recommended for production — '
                    'it works with any already-deployed model and requires '
                    'no retraining.</div>',
                    unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: AUDIT REPORT
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<p class="section-header">📋 Final Audit Report</p>',
                unsafe_allow_html=True)

    best_model = max(results.keys(), key=lambda k: results[k]["roc_auc"])

    st.markdown(f"""
    ### Executive Summary

    **Dataset:** {dataset_choice} ({len(df):,} records)
    **Audit Date:** {pd.Timestamp.now().strftime("%Y-%m-%d")}
    **Sensitive Attribute Audited:** `{sensitive_col}`
    **Privileged Group:** `{privileged_val}`

    ---

    ### Findings
    """)

    for report in audit_reports:
        mname = report["model"]
        attr  = report["attributes"].get(sensitive_col, {})
        s     = attr.get("fairness_metrics", {}).get("summary", {})
        biased = s.get("is_biased", True)
        dpd = s.get("worst_dpd") or 0
        eod = s.get("worst_eod") or 0
        dir_= s.get("worst_dir") or 0

        status = "🚨 **BIASED**" if biased else "✅ **FAIR**"
        st.markdown(f"""
        **{mname}** — {status}
        - Demographic Parity Diff: `{dpd:+.3f}` (threshold: ±0.10)
        - Equal Opportunity Diff: `{eod:+.3f}` (threshold: ±0.10)
        - Disparate Impact Ratio: `{dir_:.3f}` (threshold: >0.80)
        """)

    st.markdown("---")
    st.markdown("""
    ### Recommendations

    1. **Immediate Action:** Apply threshold adjustment to the deployed model
       — it achieves fairness with minimal accuracy cost and requires no retraining.

    2. **Data Collection:** Gather more representative data for underrepresented groups
       to reduce historical bias at the source.

    3. **Ongoing Monitoring:** Re-run this audit quarterly — model fairness can drift
       as data distributions shift over time.

    4. **Proxy Features:** Consider removing or transforming `relationship` and
       `marital_status` which act as proxies for sex (Cramer's V > 0.45).

    5. **Legal Compliance:** Current models show DIR < 0.80 which may constitute
       adverse impact under EEOC guidelines (29 CFR Part 1607).
    """)

    # Download report as CSV
    report_data = []
    for report in audit_reports:
        attr = report["attributes"].get(sensitive_col, {})
        s    = attr.get("fairness_metrics", {}).get("summary", {})
        report_data.append({
            "Model":     report["model"],
            "Dataset":   dataset_name,
            "Attribute": sensitive_col,
            "DPD":       s.get("worst_dpd"),
            "EOD":       s.get("worst_eod"),
            "DIR":       s.get("worst_dir"),
            "Is_Biased": s.get("is_biased"),
            "Accuracy":  results[report["model"]]["accuracy"],
        })
    report_df = pd.DataFrame(report_data)
    st.download_button(
        label="⬇️ Download Audit Report (CSV)",
        data=report_df.to_csv(index=False),
        file_name=f"bias_audit_{dataset_name}_{sensitive_col}.csv",
        mime="text/csv"
    )