import pandas as pd
import numpy as np
import joblib
import warnings
import matplotlib.pyplot as plt
from pathlib import Path

from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.plotting import (
    plot_frequency_recency_matrix,
    plot_probability_alive_matrix,
)

warnings.filterwarnings("ignore")

MODELS_DIR = Path("models")
PROCESSED_DIR = Path("data/processed")


def prepare_clv_data(master_df):
    df = master_df.copy()

    # BG/NBD needs frequency_repeat = total orders - 1 (repeat purchases only)
    df["frequency_repeat"] = df["frequency"] - 1

    # recency  = days from first purchase to last purchase
    # T        = customer_tenure_days (same variable per spec)
    df["recency_bgnbd"] = df["customer_tenure_days"]
    df["T_bgnbd"] = df["customer_tenure_days"]

    # keep only customers who made at least 2 purchases
    clv_df = df[df["frequency_repeat"] >= 1].copy()

    print(f"eligible customers (>=2 purchases) : {len(clv_df):,}")
    print(f"one-time buyers dropped             : {len(df) - len(clv_df):,}")

    return clv_df


def fit_bgnbd(clv_df):
    bgf = BetaGeoFitter(penalizer_coef=0.01)
    bgf.fit(
        clv_df["frequency_repeat"],
        clv_df["recency_bgnbd"],
        clv_df["T_bgnbd"],
    )
    print("BG/NBD fitted:")
    print(bgf)
    return bgf
