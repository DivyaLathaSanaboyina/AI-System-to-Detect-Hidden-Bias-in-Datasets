# src/data_loader.py
"""
DataLoader Module — Phase 2
Handles loading, column documentation, and initial validation
of both Adult Income and German Credit datasets.
"""

import os
import logging
import pandas as pd
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Inline config (mirrors config.py) ─────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(DATA_DIR, exist_ok=True)

DATASETS = {
    "adult": {
        "filename": "adult.csv",
        "target_column": "income",
        "positive_label": ">50K",
        "sensitive_attributes": ["sex", "race"],
        "privileged_groups": {"sex": "Male", "race": "White"}
    },
    "german": {
        "filename": "german_credit.csv",
        "target_column": "credit_risk",
        "positive_label": 1,
        "sensitive_attributes": ["sex", "age_group"],
        "privileged_groups": {"sex": "male", "age_group": "adult"}
    }
}


class DataLoader:
    """
    Loads and documents datasets for bias analysis.

    Usage:
        loader = DataLoader("adult")
        df = loader.load()
        loader.describe()
    """

    ADULT_COLUMNS = [
        "age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week", "native_country",
        "income",
    ]

    GERMAN_COLUMNS = [
        "checking_account", "duration", "credit_history", "purpose",
        "credit_amount", "savings_account", "employment", "installment_rate",
        "personal_status_sex", "other_debtors", "residence_since", "property",
        "age", "other_installments", "housing", "existing_credits", "job",
        "num_dependents", "telephone", "foreign_worker", "credit_risk",
    ]

    def __init__(self, dataset_name: str):
        if dataset_name not in DATASETS:
            raise ValueError(
                f"Unknown dataset: '{dataset_name}'. "
                f"Choose from: {list(DATASETS.keys())}"
            )
        self.dataset_name = dataset_name
        self.config = DATASETS[dataset_name]
        self.df: Optional[pd.DataFrame] = None

    def load(self) -> pd.DataFrame:
        """
        Loads the dataset from local cache if available,
        otherwise downloads from UCI repository.
        """
        filepath = os.path.join(DATA_DIR, self.config["filename"])

        if os.path.exists(filepath):
            logger.info(f"📂 Loading {self.dataset_name} from cache: {filepath}")
            self.df = pd.read_csv(filepath)
        else:
            logger.info(f"🌐 Downloading {self.dataset_name} dataset...")
            if self.dataset_name == "adult":
                self.df = self._download_adult()
            else:
                self.df = self._download_german()
            self.df.to_csv(filepath, index=False)
            logger.info(f"💾 Saved to {filepath}")

        logger.info(
            f"✅ Loaded {self.dataset_name}: "
            f"{self.df.shape[0]:,} rows × {self.df.shape[1]} columns"
        )
        return self.df

    def _download_adult(self) -> pd.DataFrame:
        """
        Downloads Adult Income dataset from UCI.
        Handles two quirks of the UCI source:
          1. The test file has a junk first line (skipped via skiprows=1)
          2. The test file income labels have trailing dots: '>50K.' → '>50K'
        """
        train_url = (
            "https://archive.ics.uci.edu/ml/machine-learning-databases"
            "/adult/adult.data"
        )
        test_url = (
            "https://archive.ics.uci.edu/ml/machine-learning-databases"
            "/adult/adult.test"
        )

        logger.info("  → Downloading train split...")
        train_df = pd.read_csv(
            train_url,
            names=self.ADULT_COLUMNS,
            sep=",",
            skipinitialspace=True,
            na_values="?"
        )

        logger.info("  → Downloading test split...")
        test_df = pd.read_csv(
            test_url,
            names=self.ADULT_COLUMNS,
            sep=",",
            skipinitialspace=True,
            na_values="?",
            skiprows=1
        )

        # Clean income labels in both splits
        for frame in [train_df, test_df]:
            frame["income"] = (
                frame["income"]
                .astype(str)
                .str.strip()
                .str.replace(".", "", regex=False)
            )

        combined = pd.concat([train_df, test_df], ignore_index=True)
        logger.info(f"  → Combined shape: {combined.shape}")
        return combined

    def _download_german(self) -> pd.DataFrame:
        """
        Downloads German Credit dataset from UCI.
        Space-separated file with coded categorical values.
        Target recoded: 1=Good credit, 0=Bad credit.
        """
        url = (
            "https://archive.ics.uci.edu/ml/machine-learning-databases"
            "/statlog/german/german.data"
        )
        logger.info("  → Downloading German Credit dataset...")

        df = pd.read_csv(
            url,
            names=self.GERMAN_COLUMNS,
            sep=" ",
            skipinitialspace=True
        )

        # Recode target: 1=Good → 1, 2=Bad → 0
        df["credit_risk"] = df["credit_risk"].map({1: 1, 2: 0})

        # Extract sex from combined personal_status_sex column
        # A91=male divorced, A92=female divorced/separated,
        # A93=male single, A94=male married/widowed, A95=female single
        df["sex"] = df["personal_status_sex"].map({
            "A91": "male",
            "A92": "female",
            "A93": "male",
            "A94": "male",
            "A95": "female"
        })

        # Create binary age group sensitive attribute
        df["age_group"] = (df["age"] >= 25).map(
            {True: "adult", False: "young"}
        )

        logger.info(f"  → German shape: {df.shape}")
        return df

    def get_target_column(self) -> str:
        return self.config["target_column"]

    def get_sensitive_attributes(self) -> List[str]:
        return self.config["sensitive_attributes"]

    def get_privileged_groups(self) -> Dict[str, str]:
        return self.config["privileged_groups"]

    def get_feature_columns(self) -> List[str]:
        if self.df is None:
            raise RuntimeError("Call .load() before accessing features.")
        return [c for c in self.df.columns if c != self.config["target_column"]]

    def describe(self) -> None:
        """Prints a comprehensive dataset summary."""
        if self.df is None:
            raise RuntimeError("Call .load() first.")

        target = self.config["target_column"]
        sensitive = self.config["sensitive_attributes"]
        privileged = self.config["privileged_groups"]
        positive_label = self.config["positive_label"]

        print("\n" + "=" * 65)
        print(f"  📊 DATASET REPORT: {self.dataset_name.upper()}")
        print("=" * 65)

        print(f"\n{'Shape':<25} {self.df.shape[0]:,} rows × {self.df.shape[1]} columns")
        print(f"{'Target column':<25} '{target}'")
        print(f"{'Sensitive attributes':<25} {sensitive}")

        # Missing values
        missing = self.df.isnull().sum()
        missing_cols = missing[missing > 0]
        print(f"\nMissing Values:")
        if missing_cols.empty:
            print("  ✅ No missing values")
        else:
            for col, count in missing_cols.items():
                pct = 100 * count / len(self.df)
                print(f"  ⚠️  {col:<25} {count:>5} ({pct:.1f}%)")

        # Target distribution
        print(f"\nTarget Distribution ('{target}'):")
        target_counts = self.df[target].value_counts()
        for val, count in target_counts.items():
            pct = 100 * count / len(self.df)
            bar = "█" * int(pct / 2)
            print(f"  {str(val):<15} {count:>6,} ({pct:5.1f}%) {bar}")

        # Sensitive attribute distributions
        print(f"\nSensitive Attribute Distributions:")
        for attr in sensitive:
            if attr in self.df.columns:
                priv = privileged.get(attr, "")
                print(f"\n  [{attr}]  (privileged group: '{priv}')")
                for val, count in self.df[attr].value_counts().items():
                    pct = 100 * count / len(self.df)
                    marker = " ← privileged" if str(val) == str(priv) else ""
                    print(f"    {str(val):<20} {count:>6,} ({pct:5.1f}%){marker}")

        # Outcome rates — first signal of bias
        print(f"\nFavorable Outcome Rate by Sensitive Group:")
        print(f"  (Gap > 10% is a ⚠️  BIAS SIGNAL)\n")
        for attr in sensitive:
            if attr in self.df.columns:
                print(f"  [{attr}]")
                overall_rate = (
                    (self.df[target] == positive_label).mean() * 100
                    if isinstance(positive_label, str)
                    else self.df[target].mean() * 100
                )
                for group_val, group_data in self.df.groupby(attr)[target]:
                    if isinstance(positive_label, str):
                        rate = (group_data == positive_label).mean() * 100
                    else:
                        rate = group_data.mean() * 100
                    gap = abs(rate - overall_rate)
                    flag = " ⚠️  BIAS SIGNAL" if gap > 10 else ""
                    print(f"    {str(group_val):<20} {rate:5.1f}%{flag}")

        print("\n" + "=" * 65 + "\n")