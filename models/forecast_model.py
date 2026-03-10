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


def fit_sarima(series, category):
    """Fit SARIMAX(1,1,1)(1,1,0,52) to a weekly revenue series. Saves model pkl."""
    p_value = run_adf_test(series, category)
    if p_value > 0.05:
        series = series.diff().dropna()

    model = SARIMAX(
        series,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 0, 52),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    results = model.fit(disp=False)
    print(f"[SARIMA] Category={category}  AIC={results.aic:.2f}")
    joblib.dump(results, MODELS_DIR / f"sarima_{category}.pkl")
    return results


def forecast_sarima(results, steps=12):
    """Generate SARIMA forecast with 95% confidence intervals."""
    forecast = results.get_forecast(steps=steps)
    ci = forecast.conf_int()
    fc_df = pd.DataFrame({
        "sarima_forecast": forecast.predicted_mean.values,
        "lower_ci": ci.iloc[:, 0].values,
        "upper_ci": ci.iloc[:, 1].values,
    })
    return fc_df


def fit_prophet_model(series, category):
    """Fit Prophet with yearly + weekly seasonality; return 12-week forecast tail."""
    prophet_df = pd.DataFrame({
        "ds": series.index.to_timestamp(),
        "y": series.values,
    })
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=12, freq="W")
    forecast = model.predict(future)
    return forecast.tail(12).reset_index(drop=True)


def plot_dual_forecast(category, history, sarima_fc, prophet_fc):
    """Plot historical revenue alongside SARIMA and Prophet 12-week forecasts."""
    fig, ax = plt.subplots(figsize=(12, 5))

    # History
    history_vals = history.values
    history_idx = range(len(history_vals))
    ax.plot(history_idx, history_vals, color="black", linewidth=1.5, label="History")

    # SARIMA forecast
    fc_start = len(history_vals)
    fc_idx = range(fc_start, fc_start + len(sarima_fc))
    ax.plot(fc_idx, sarima_fc["sarima_forecast"], color="blue", linewidth=1.5, label="SARIMA")
    ax.fill_between(
        fc_idx,
        sarima_fc["lower_ci"],
        sarima_fc["upper_ci"],
        color="blue",
        alpha=0.15,
        label="SARIMA 95% CI",
    )

    # Prophet forecast
    ax.plot(fc_idx, prophet_fc["yhat"].values, color="red", linestyle="--", linewidth=1.5, label="Prophet")

    ax.set_title(f"Demand Forecast — Category {category}", fontsize=13)
    ax.set_xlabel("Week")
    ax.set_ylabel("Revenue (£)")
    ax.legend()
    plt.tight_layout()
    plt.show()
