import os
os.chdir('/Users/sanjanawaghray/Documents/projects/retail-intelligence-engine')

import matplotlib
matplotlib.use('Agg')

import sys
sys.path.append('.')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

from models.data_pipeline import load_raw_data, extract_signal_before_cleaning, clean_data
from models.forecast_model import (
    build_weekly_revenue,
    run_adf_test,
    fit_sarima,
    forecast_sarima,
    fit_prophet_model,
    plot_dual_forecast,
    save_demand_forecasts,
    build_anomaly_features,
    fit_isolation_forest,
    score_anomalies,
    save_anomaly_flags,
    PROCESSED_DIR,
)

sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.figsize'] = (12, 5)

# ── Cell: load-data ──────────────────────────────────────────────────────────
df_raw = load_raw_data()
_, df_all = clean_data(df_raw)
print(f'df_all shape: {df_all.shape}')
print(df_all.dtypes)

# ── Cell: build-weekly-revenue ────────────────────────────────────────────────
weekly_df, top5 = build_weekly_revenue(df_all)
print('Top 5 categories by total revenue:')
print(top5)
print(f'\nweekly_df shape: {weekly_df.shape}')
print(weekly_df.head())

# ── Cell: adf-test ────────────────────────────────────────────────────────────
adf_results = {}
for cat in top5:
    series = weekly_df[weekly_df['category'] == cat].set_index('week')['revenue']
    p = run_adf_test(series, cat)
    adf_results[cat] = p

print('\nADF p-values:')
for cat, p in adf_results.items():
    status = 'stationary' if p <= 0.05 else 'non-stationary -> will difference'
    print(f'  {cat}: p={p:.4f}  ({status})')

# ── Cell: sarima-fit ──────────────────────────────────────────────────────────
sarima_forecasts = {}
sarima_history = {}

for cat in top5:
    series = weekly_df[weekly_df['category'] == cat].set_index('week')['revenue']
    try:
        results = fit_sarima(series, cat)
        fc = forecast_sarima(results, steps=12)
        sarima_forecasts[cat] = fc
        sarima_history[cat] = series
        print(f'{cat}: forecast range £{fc["sarima_forecast"].min():.0f} – £{fc["sarima_forecast"].max():.0f}')
    except Exception as e:
        print(f'SARIMA failed for {cat}: {e}')

# ── Cell: prophet-fit ─────────────────────────────────────────────────────────
prophet_forecasts = {}

for cat in top5:
    series = weekly_df[weekly_df['category'] == cat].set_index('week')['revenue']
    try:
        fc = fit_prophet_model(series, cat)
        prophet_forecasts[cat] = fc
        print(f'{cat}: Prophet yhat range £{fc["yhat"].min():.0f} – £{fc["yhat"].max():.0f}')
    except Exception as e:
        print(f'Prophet failed for {cat}: {e}')

# ── Cell: dual-forecast-plots ─────────────────────────────────────────────────
# Override plot_dual_forecast to save instead of show
def plot_dual_forecast_agg(category, history, sarima_fc, prophet_fc):
    try:
        fig, ax = plt.subplots(figsize=(12, 5))
        history_vals = history.values
        history_idx = range(len(history_vals))
        ax.plot(history_idx, history_vals, color='black', linewidth=1.5, label='History')
        fc_start = len(history_vals)
        fc_idx = range(fc_start, fc_start + len(sarima_fc))
        ax.plot(fc_idx, sarima_fc['sarima_forecast'], color='blue', linewidth=1.5, label='SARIMA')
        ax.fill_between(
            fc_idx,
            sarima_fc['lower_ci'],
            sarima_fc['upper_ci'],
            color='blue', alpha=0.15, label='SARIMA 95% CI',
        )
        ax.plot(fc_idx, prophet_fc['yhat'].values, color='red', linestyle='--',
                linewidth=1.5, label='Prophet')
        ax.set_title(f'Demand Forecast — Category {category}', fontsize=13)
        ax.set_xlabel('Week')
        ax.set_ylabel('Revenue (£)')
        ax.legend()
        plt.tight_layout()
        plt.savefig(f'data/processed/chart_forecast_{category}.png', dpi=80)
        plt.close()
    except Exception as e:
        print(f'forecast plot for {category} skipped: {e}')
        plt.close('all')

for cat in top5:
    if cat in sarima_forecasts and cat in prophet_forecasts:
        plot_dual_forecast_agg(
            category=cat,
            history=sarima_history[cat],
            sarima_fc=sarima_forecasts[cat],
            prophet_fc=prophet_forecasts[cat],
        )

# ── Cell: save-demand-forecasts ───────────────────────────────────────────────
records = []
for cat in top5:
    if cat not in sarima_forecasts or cat not in prophet_forecasts:
        continue
    s_fc = sarima_forecasts[cat]
    p_fc = prophet_forecasts[cat].reset_index(drop=True)
    for i in range(len(s_fc)):
        records.append({
            'category':         cat,
            'date':             str(p_fc['ds'].iloc[i].date()),
            'sarima_forecast':  round(s_fc['sarima_forecast'].iloc[i], 2),
            'prophet_forecast': round(p_fc['yhat'].iloc[i], 2),
            'lower_ci':         round(s_fc['lower_ci'].iloc[i], 2),
            'upper_ci':         round(s_fc['upper_ci'].iloc[i], 2),
        })

save_demand_forecasts(records)
print(pd.DataFrame(records).head(10))

# ── Cell: anomaly-features ────────────────────────────────────────────────────
X, df_flagged = build_anomaly_features(df_all)
print(f'feature matrix shape: {X.shape}')
feat_cols = ['order_value', 'order_value_zscore', 'quantity', 'hour_of_day', 'day_of_week', 'is_new_country']
print(pd.DataFrame(X, columns=feat_cols).describe().round(3))

# ── Cell: isolation-forest-fit ───────────────────────────────────────────────
iso_forest = fit_isolation_forest(X)
print('isolation forest fitted and saved to models/isolation_forest.pkl')

# ── Cell: score-anomalies ─────────────────────────────────────────────────────
df_scored = score_anomalies(iso_forest, X, df_flagged)
print(f'\ntotal transactions   : {len(df_scored):,}')
print(f'flagged anomalies    : {df_scored["anomaly_flag"].sum():,}')
print(f'anomaly rate         : {df_scored["anomaly_flag"].mean()*100:.2f}%')

# ── Cell: anomaly-histogram ───────────────────────────────────────────────────
try:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(df_scored['anomaly_score'], bins=80, color='#5c85d6', edgecolor='white', linewidth=0.3)
    threshold_val = df_scored[df_scored['anomaly_flag'] == 1]['anomaly_score'].max()
    ax.axvline(threshold_val, color='red', linestyle='--', linewidth=1.2, label='anomaly threshold')
    ax.set_title('Isolation Forest — Anomaly Score Distribution', fontsize=13)
    ax.set_xlabel('Anomaly Score (lower = more anomalous)')
    ax.set_ylabel('Transaction Count')
    ax.legend()
    plt.tight_layout()
    plt.savefig('data/processed/chart_anomaly_hist.png', dpi=80)
    plt.close()
except Exception as e:
    print(f'anomaly histogram skipped: {e}')
    plt.close('all')

# ── Cell: save-anomaly-flags ──────────────────────────────────────────────────
save_anomaly_flags(df_scored)
print('\nTop 10 most anomalous transactions:')
print(df_scored.nsmallest(10, 'anomaly_score')[
    ['invoice_no', 'customer_id', 'stock_code', 'quantity', 'unit_price', 'anomaly_score']
].to_string(index=False))

# ── Cell: summary ─────────────────────────────────────────────────────────────
demand_fc = pd.read_csv(PROCESSED_DIR / 'demand_forecasts.csv')
anomaly_fc = pd.read_csv(PROCESSED_DIR / 'anomaly_flags.csv')

print('=== Phase 4 Complete ===')
print(f'categories forecast      : {demand_fc["category"].nunique()}')
print(f'forecast horizon         : 12 weeks')
print(f'total transactions scored: {len(anomaly_fc):,}')
print(f'anomalies flagged        : {anomaly_fc["anomaly_flag"].sum():,}  ({anomaly_fc["anomaly_flag"].mean()*100:.2f}%)')
print()
print('Saved artefacts:')
for cat in top5:
    print(f'  models/sarima_{cat}.pkl')
print('  models/isolation_forest.pkl')
print('  data/processed/demand_forecasts.csv')
print('  data/processed/anomaly_flags.csv')
