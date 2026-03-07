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


def predict_purchases(bgf, clv_df):
    predicted_purchases_90d = bgf.conditional_expected_number_of_purchases_up_to_time(
        90,
        clv_df["frequency_repeat"],
        clv_df["recency_bgnbd"],
        clv_df["T_bgnbd"],
    )

    prob_alive = bgf.conditional_probability_alive(
        clv_df["frequency_repeat"],
        clv_df["recency_bgnbd"],
        clv_df["T_bgnbd"],
    )

    print(f"median predicted purchases (90d) : {predicted_purchases_90d.median():.3f}")
    print(f"median prob alive                : {prob_alive.median():.3f}")

    return predicted_purchases_90d, prob_alive


def check_gg_assumption(clv_df):
    # gamma-gamma requires frequency and monetary to be independent
    corr = clv_df["frequency_repeat"].corr(clv_df["monetary"])
    print(f"Pearson correlation (frequency vs monetary): {corr:.4f}")

    if abs(corr) > 0.3:
        print(
            "WARNING: |correlation| > 0.3 — the Gamma-Gamma independence assumption "
            "may be violated. CLV estimates could be biased."
        )
    else:
        print("assumption check passed: frequency and monetary look approximately independent")

    return corr


def fit_gamma_gamma(clv_df):
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(clv_df["frequency_repeat"], clv_df["monetary"])
    print("Gamma-Gamma fitted:")
    print(ggf)
    return ggf


def score_clv(bgf, ggf, clv_df):
    expected_avg_order_value = ggf.conditional_expected_average_profit(
        clv_df["frequency_repeat"],
        clv_df["monetary"],
    )

    # 365-day CLV — time=12 months, 1% monthly discount rate
    clv_365d = ggf.customer_lifetime_value(
        bgf,
        clv_df["frequency_repeat"],
        clv_df["recency_bgnbd"],
        clv_df["T_bgnbd"],
        clv_df["monetary"],
        time=12,
        discount_rate=0.01,
    )

    # 90-day CLV — time=3 months
    clv_90d = ggf.customer_lifetime_value(
        bgf,
        clv_df["frequency_repeat"],
        clv_df["recency_bgnbd"],
        clv_df["T_bgnbd"],
        clv_df["monetary"],
        time=3,
        discount_rate=0.01,
    )

    print(f"median CLV 90d  : £{clv_90d.median():.2f}")
    print(f"median CLV 365d : £{clv_365d.median():.2f}")

    return expected_avg_order_value, clv_90d, clv_365d


def save_models(bgf, ggf):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bgf, MODELS_DIR / "bgf.pkl")
    joblib.dump(ggf, MODELS_DIR / "ggf.pkl")
    print("saved models/bgf.pkl and models/ggf.pkl")


def build_clv_scores(master_df):
    clv_df = prepare_clv_data(master_df)

    bgf = fit_bgnbd(clv_df)
    predicted_purchases_90d, prob_alive = predict_purchases(bgf, clv_df)

    check_gg_assumption(clv_df)
    ggf = fit_gamma_gamma(clv_df)
    expected_avg_order_value, clv_90d, clv_365d = score_clv(bgf, ggf, clv_df)

    scores = pd.DataFrame({
        "customer_id":               clv_df["customer_id"].values,
        "frequency_repeat":          clv_df["frequency_repeat"].values,
        "prob_alive":                prob_alive.values,
        "predicted_purchases_90d":   predicted_purchases_90d.values,
        "expected_avg_order_value":  expected_avg_order_value.values,
        "clv_90d":                   clv_90d.values,
        "clv_365d":                  clv_365d.values,
    })

    save_models(bgf, ggf)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    scores.to_csv(PROCESSED_DIR / "clv_scores.csv", index=False)
    print(f"\nsaved clv_scores.csv — {len(scores):,} customers")
    print(scores[["clv_90d", "clv_365d", "prob_alive"]].describe().round(2))

    return scores, bgf, ggf
