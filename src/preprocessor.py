# src/preprocessor.py
"""
Preprocessor Module — Phase 3

Handles all data cleaning, encoding, and scaling.
Every decision is documented with a fairness justification.

Design principle: The preprocessor PRESERVES sensitive attributes
in a separate structure so they can be used for bias analysis
even after encoding/transformation.
"""

import logging
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)

# ── Inline config ──────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE    = 0.2

DATASET_CONFIGS = {
    "adult": {
        "target_column":       "income",
        "positive_label":      ">50K",
        "sensitive_attributes": ["sex", "race"],
        "drop_columns":        ["fnlwgt"],   # Census weight — not a real feature
        "categorical_columns": [
            "workclass", "education", "marital_status",
            "occupation", "relationship", "race", "sex", "native_country"
        ],
        "continuous_columns": [
            "age", "education_num", "capital_gain",
            "capital_loss", "hours_per_week"
        ],
    },
    "german": {
        "target_column":       "credit_risk",
        "positive_label":      1,
        "sensitive_attributes": ["sex", "age_group"],
        "drop_columns":        ["personal_status_sex"],  # Replaced by extracted 'sex'
        "categorical_columns": [
            "checking_account", "credit_history", "purpose",
            "savings_account", "employment", "other_debtors",
            "property", "other_installments", "housing", "job",
            "telephone", "foreign_worker", "sex", "age_group"
        ],
        "continuous_columns": [
            "duration", "credit_amount", "installment_rate",
            "residence_since", "age", "existing_credits", "num_dependents"
        ],
    }
}


class Preprocessor:
    """
    Transforms raw datasets into ML-ready format while preserving
    sensitive attribute information for fairness analysis.

    Usage:
        preprocessor = Preprocessor("adult")
        splits = preprocessor.fit_transform(df)
        X_train, X_test, y_train, y_test = splits
        
        # Access sensitive attributes for bias analysis
        s_train = preprocessor.sensitive_train
        s_test  = preprocessor.sensitive_test
    """

    def __init__(self, dataset_name: str):
        if dataset_name not in DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset: '{dataset_name}'")
        self.dataset_name  = dataset_name
        self.config        = DATASET_CONFIGS[dataset_name]
        self.scaler        = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names: List[str] = []

        # These store sensitive attributes SEPARATELY from X
        # so we can always analyze fairness even after encoding
        self.sensitive_train: Optional[pd.DataFrame] = None
        self.sensitive_test:  Optional[pd.DataFrame] = None

    def fit_transform(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Full preprocessing pipeline:
          1. Drop irrelevant columns
          2. Separate target and features
          3. Save sensitive attributes (pre-encoding)
          4. Handle missing values
          5. Encode categoricals
          6. Scale continuous features
          7. Train/test split

        Args:
            df: Raw DataFrame from DataLoader

        Returns:
            (X_train, X_test, y_train, y_test) as numpy arrays
        """
        logger.info(f"🔧 Starting preprocessing for: {self.dataset_name}")
        df = df.copy()

        # ── Step 1: Drop irrelevant columns ───────────────────────────────────
        drop_cols = [c for c in self.config["drop_columns"] if c in df.columns]
        if drop_cols:
            df.drop(columns=drop_cols, inplace=True)
            logger.info(f"  ✂️  Dropped columns: {drop_cols}")

        # ── Step 2: Separate target ────────────────────────────────────────────
        target_col    = self.config["target_column"]
        positive_label = self.config["positive_label"]

        y_raw = df[target_col].copy()
        X_raw = df.drop(columns=[target_col])

        # Encode target as binary 0/1
        if isinstance(positive_label, str):
            y = (y_raw == positive_label).astype(int).values
        else:
            y = (y_raw == positive_label).astype(int).values

        pos_rate = y.mean() * 100
        logger.info(
            f"  🎯 Target encoded: 1=favorable ({pos_rate:.1f}%), "
            f"0=unfavorable ({100-pos_rate:.1f}%)"
        )

        # ── Step 3: Save sensitive attributes BEFORE encoding ─────────────────
        # This is critical — we preserve the original string labels
        # (e.g., "Male"/"Female") for human-readable bias reports
        sensitive_cols = [
            c for c in self.config["sensitive_attributes"]
            if c in X_raw.columns
        ]
        sensitive_df = X_raw[sensitive_cols].copy()
        logger.info(f"  🔒 Preserved sensitive attributes: {sensitive_cols}")

        # ── Step 4: Handle missing values ─────────────────────────────────────
        X_raw = self._impute_missing(X_raw)

        # ── Step 5: Encode categorical columns ────────────────────────────────
        X_encoded = self._encode_categoricals(X_raw)

        # ── Step 6: Scale continuous columns ──────────────────────────────────
        # Note: We fit scaler on ALL data here, then refit on train only
        # after splitting (see below). This two-step is for logging purposes.
        self.feature_names = list(X_encoded.columns)
        logger.info(f"  📐 Total features after encoding: {len(self.feature_names)}")

        # ── Step 7: Train/test split ───────────────────────────────────────────
        # Stratified split: preserves class distribution in both splits
        (X_train_raw, X_test_raw,
         y_train, y_test,
         s_train, s_test) = train_test_split(
            X_encoded, y, sensitive_df,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y              # Ensures balanced class split
        )

        # ── Step 8: Scale (fit on train ONLY — prevents data leakage) ─────────
        # Data leakage = using test set info during training = invalid results
        cont_cols = [
            c for c in self.config["continuous_columns"]
            if c in X_train_raw.columns
        ]

        X_train = X_train_raw.copy()
        X_test  = X_test_raw.copy()

        if cont_cols:
            X_train[cont_cols] = self.scaler.fit_transform(X_train_raw[cont_cols])
            X_test[cont_cols]  = self.scaler.transform(X_test_raw[cont_cols])
            logger.info(f"  ⚖️  Scaled {len(cont_cols)} continuous features")

        # Store sensitive attributes for later bias analysis
        self.sensitive_train = s_train.reset_index(drop=True)
        self.sensitive_test  = s_test.reset_index(drop=True)

        X_train_arr = X_train.values.astype(np.float32)
        X_test_arr  = X_test.values.astype(np.float32)

        logger.info(
            f"  ✅ Preprocessing complete:\n"
            f"     Train: {X_train_arr.shape} | "
            f"     Test:  {X_test_arr.shape}"
        )

        # Save for reference
        self.X_train_df = X_train
        self.X_test_df  = X_test

        return X_train_arr, X_test_arr, y_train, y_test

    def _impute_missing(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Imputes missing values:
          - Categorical columns → most frequent value (mode)
          - Continuous columns  → median (robust to outliers)

        WHY NOT drop rows?
        Dropping rows with missing values disproportionately removes
        certain demographic groups, introducing selection bias.
        Imputation is always preferred in fairness-critical contexts.
        """
        X = X.copy()
        cat_cols  = [c for c in self.config["categorical_columns"] if c in X.columns]
        cont_cols = [c for c in self.config["continuous_columns"]   if c in X.columns]

        missing_before = X.isnull().sum().sum()

        if missing_before == 0:
            logger.info("  ✅ No missing values — skipping imputation")
            return X

        # Categorical: fill with mode
        if cat_cols:
            cat_imputer = SimpleImputer(strategy="most_frequent")
            X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])

        # Continuous: fill with median
        if cont_cols:
            num_imputer = SimpleImputer(strategy="median")
            X[cont_cols] = num_imputer.fit_transform(X[cont_cols])

        missing_after = X.isnull().sum().sum()
        logger.info(
            f"  🩹 Imputation: {missing_before} → {missing_after} missing values"
        )
        return X

    def _encode_categoricals(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        One-hot encodes all categorical columns.

        WHY one-hot and not label encoding?
        Label encoding (0,1,2,3...) implies ordinal relationships.
        'Exec-managerial' is not "greater than" 'Craft-repair'.
        One-hot encoding treats categories as truly unordered.

        drop_first=True avoids the "dummy variable trap"
        (perfect multicollinearity in linear models).
        """
        cat_cols = [
            c for c in self.config["categorical_columns"]
            if c in X.columns
        ]

        if not cat_cols:
            return X

        X_encoded = pd.get_dummies(
            X,
            columns=cat_cols,
            drop_first=True,       # Avoids multicollinearity
            dtype=np.float32
        )

        logger.info(
            f"  🔡 One-hot encoded {len(cat_cols)} categorical columns → "
            f"{X_encoded.shape[1]} total features"
        )
        return X_encoded

    def get_feature_names(self) -> List[str]:
        """Returns feature names after encoding (for SHAP, reports)."""
        return self.feature_names

    def get_preprocessing_report(self) -> Dict:
        """
        Returns a summary dict of all preprocessing decisions.
        Used for the final audit report.
        """
        return {
            "dataset":           self.dataset_name,
            "dropped_columns":   self.config["drop_columns"],
            "imputation": {
                "categorical":   "most_frequent",
                "continuous":    "median"
            },
            "encoding":          "one_hot (drop_first=True)",
            "scaling":           "StandardScaler (fit on train only)",
            "train_test_split": {
                "test_size":     TEST_SIZE,
                "stratified":    True,
                "random_state":  RANDOM_STATE
            },
            "total_features":    len(self.feature_names),
        }

    def describe(self) -> None:
        """Prints a summary of preprocessing decisions."""
        report = self.get_preprocessing_report()
        print("\n" + "=" * 55)
        print("  🔧 PREPROCESSING REPORT")
        print("=" * 55)
        print(f"  Dataset:        {report['dataset']}")
        print(f"  Dropped cols:   {report['dropped_columns']}")
        print(f"  Imputation:     categorical=mode, continuous=median")
        print(f"  Encoding:       One-hot (drop_first=True)")
        print(f"  Scaling:        StandardScaler (train-only fit)")
        print(f"  Split:          {int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)} "
              f"stratified, seed={RANDOM_STATE}")
        print(f"  Total features: {report['total_features']}")
        print("=" * 55 + "\n")