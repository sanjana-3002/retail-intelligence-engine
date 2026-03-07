import pandas as pd
import numpy as np
import joblib
import warnings
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay,
)
from xgboost import XGBClassifier
import shap
from lifelines import CoxPHFitter, KaplanMeierFitter

warnings.filterwarnings("ignore")

MODELS_DIR = Path("models")
PROCESSED_DIR = Path("data/processed")

CHURN_FEATURES = [
    "recency", "frequency", "monetary",
    "velocity_decay_ratio", "category_hhi", "spend_cv",
    "return_rate", "cancellation_rate",
    "customer_tenure_days", "avg_items_per_order", "country_count",
]


def define_churn_label(master):
    """churned = 1 if recency > 180 days, else 0."""
    df = master.copy()
    df["churned"] = (df["recency"] > 180).astype(int)

    n_churned = df["churned"].sum()
    n_active = (df["churned"] == 0).sum()
    ratio = n_churned / len(df)

    print(f"churned=1 : {n_churned:,}  ({ratio * 100:.1f}%)")
    print(f"churned=0 : {n_active:,}  ({(1 - ratio) * 100:.1f}%)")
    print(f"churn ratio (1:0) : 1 : {n_active / n_churned:.2f}")

    return df


def prepare_churn_features(master_with_label):
    """Return X (feature matrix), y (churn label) using the canonical feature set."""
    df = master_with_label.copy()
    missing = [f for f in CHURN_FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    X = df[CHURN_FEATURES].fillna(0).astype(float)
    y = df["churned"].astype(int)

    print(f"feature matrix : {X.shape[0]:,} rows × {X.shape[1]} features")
    print(f"churn rate     : {y.mean() * 100:.1f}%")

    return X, y
