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


def fit_xgboost_classifier(X_train, y_train, X_test, y_test):
    """Fit XGBClassifier with class-weight balancing and early stopping."""
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"scale_pos_weight : {scale_pos_weight:.3f}")

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        eval_metric="auc",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbosity=0,
    )
    model.fit(
        X_train, y_train,
        early_stopping_rounds=20,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    best_iter = model.best_iteration
    print(f"best iteration   : {best_iter}")
    return model


def evaluate_classifier(model, X_test, y_test):
    """Return metrics dict; also prints AUC, precision, recall, F1."""
    import matplotlib.pyplot as plt

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, output_dict=True)

    print(f"AUC-ROC   : {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["active", "churned"]))

    # confusion matrix
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["active", "churned"])
    disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
    axes[0].set_title("Confusion Matrix")

    RocCurveDisplay.from_predictions(y_test, y_prob, ax=axes[1], name="XGBoost")
    axes[1].plot([0, 1], [0, 1], "k--", lw=1)
    axes[1].set_title(f"ROC Curve  (AUC = {auc:.3f})")

    plt.tight_layout()
    plt.show()

    return {
        "auc": auc,
        "precision": report["churned"]["precision"],
        "recall": report["churned"]["recall"],
        "f1": report["churned"]["f1-score"],
        "y_prob": y_prob,
    }


def get_shap_values(model, X_test):
    """Compute SHAP values for X_test using TreeExplainer."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    print(f"shap_values shape : {np.array(shap_values).shape}")
    return explainer, shap_values


def get_top_shap_drivers(explainer, customer_features, n=3):
    """
    Return the top-n SHAP drivers for a single customer.

    Parameters
    ----------
    explainer        : shap.TreeExplainer fitted on the XGBoost churn model
    customer_features: pd.DataFrame with exactly one row (the customer's features)
    n                : number of top drivers to return

    Returns
    -------
    list of (feature_name, shap_value) tuples sorted by |shap_value| descending
    """
    sv = explainer.shap_values(customer_features)
    sv_flat = np.array(sv).flatten()
    feature_names = list(customer_features.columns)

    pairs = sorted(
        zip(feature_names, sv_flat),
        key=lambda x: abs(x[1]),
        reverse=True,
    )
    return pairs[:n]


def fit_cox_model(df_cox):
    """
    Fit a Cox Proportional Hazards model.

    Duration column : customer_tenure_days
    Event column    : churned  (0 = censored / still active, 1 = churned)
    """
    cph = CoxPHFitter()
    cph.fit(
        df_cox,
        duration_col="customer_tenure_days",
        event_col="churned",
        show_progress=False,
    )
    print("Cox PH model fitted.")
    return cph


def predict_survival_days(cph, customer_features):
    """
    Return the median survival time (days) for a single customer.

    Parameters
    ----------
    cph               : fitted CoxPHFitter
    customer_features : pd.DataFrame with one row containing Cox covariates

    Returns
    -------
    float — median expected days until churn (or np.inf if > observation window)
    """
    sf = cph.predict_survival_function(customer_features)
    # find the time at which survival probability first drops to or below 0.5
    for t, prob in sf.iterrows():
        if prob.iloc[0] <= 0.5:
            return float(t)
    return float("inf")
