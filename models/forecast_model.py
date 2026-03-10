import pandas as pd
import numpy as np
import joblib
import warnings
import matplotlib.pyplot as plt
from pathlib import Path

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.ensemble import IsolationForest

warnings.filterwarnings("ignore")

MODELS_DIR = Path("models")
PROCESSED_DIR = Path("data/processed")


def build_weekly_revenue(df_all):
    """Aggregate weekly revenue per product category; return top-5 categories."""
    df = df_all.copy()
    df["category"] = df["stock_code"].astype(str).str[:2]
    df = df[df["revenue"] > 0]
    df["week"] = df["invoice_date"].dt.to_period("W")
    weekly_df = (
        df.groupby(["category", "week"], as_index=False)["revenue"].sum()
    )
    top5_categories = (
        weekly_df.groupby("category")["revenue"]
        .sum()
        .nlargest(5)
        .index.tolist()
    )
    return weekly_df, top5_categories


def run_adf_test(series, category):
    """Run Augmented Dickey-Fuller test and print result. Returns p-value."""
    result = adfuller(series.dropna())
    adf_stat = result[0]
    p_value = result[1]
    stationary = p_value <= 0.05
    print(
        f"[ADF] Category={category}  ADF stat={adf_stat:.4f}  "
        f"p={p_value:.4f}  stationary={stationary}"
    )
    return p_value
