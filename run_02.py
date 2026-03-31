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
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.plotting import plot_frequency_recency_matrix, plot_probability_alive_matrix
from models.data_pipeline import (
    load_raw_data, extract_signal_before_cleaning,
    clean_data, build_master_customer_table
)
from models.clv_model import (
    prepare_clv_data, fit_bgnbd, predict_purchases,
    check_gg_assumption, fit_gamma_gamma, score_clv,
    save_models, build_clv_scores
)
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.figsize'] = (12, 5)

# ── Cell: load-data ──────────────────────────────────────────────────────────
df_raw = load_raw_data()
return_features, cancel_features = extract_signal_before_cleaning(df_raw)
df_customers, _ = clean_data(df_raw)
master = build_master_customer_table(df_customers, return_features, cancel_features)
clv_df = prepare_clv_data(master)
print(clv_df[['frequency_repeat', 'recency_bgnbd', 'T_bgnbd', 'monetary']].describe().round(2))

# ── Cell: fit-bgnbd ──────────────────────────────────────────────────────────
# Override fit_bgnbd to try increasing penalizers until convergence
def fit_bgnbd_robust(clv_df):
    from lifetimes import BetaGeoFitter
    for pen in [0.01, 0.1, 0.5, 1.0, 5.0]:
        try:
            bgf = BetaGeoFitter(penalizer_coef=pen)
            bgf.fit(
                clv_df['frequency_repeat'],
                clv_df['recency_bgnbd'],
                clv_df['T_bgnbd'],
            )
            print(f'BG/NBD converged with penalizer_coef={pen}')
            print(bgf)
            return bgf
        except Exception as e:
            print(f'penalizer_coef={pen} failed: {e}')
    raise RuntimeError('BG/NBD failed to converge at all penalizer levels')

bgf = fit_bgnbd_robust(clv_df)
# Override predict_purchases to fix numpy.ndarray .median() bug
def predict_purchases_robust(bgf, clv_df):
    predicted_purchases_90d = bgf.conditional_expected_number_of_purchases_up_to_time(
        90,
        clv_df['frequency_repeat'],
        clv_df['recency_bgnbd'],
        clv_df['T_bgnbd'],
    )
    prob_alive = bgf.conditional_probability_alive(
        clv_df['frequency_repeat'],
        clv_df['recency_bgnbd'],
        clv_df['T_bgnbd'],
    )
    # convert to pandas Series if numpy arrays
    if hasattr(predicted_purchases_90d, 'values'):
        pass
    else:
        predicted_purchases_90d = pd.Series(predicted_purchases_90d)
    if hasattr(prob_alive, 'values'):
        pass
    else:
        prob_alive = pd.Series(prob_alive)
    print(f'median predicted purchases (90d) : {predicted_purchases_90d.median():.3f}')
    print(f'median prob alive                : {prob_alive.median():.3f}')
    return predicted_purchases_90d, prob_alive

predicted_purchases_90d, prob_alive = predict_purchases_robust(bgf, clv_df)

# ── Cell: chart-freq-rec ─────────────────────────────────────────────────────
try:
    fig, ax = plt.subplots(figsize=(12, 7))
    plot_frequency_recency_matrix(bgf, T=90, ax=ax)
    ax.set_title('BG/NBD — Expected Purchases in Next 90 Days')
    plt.tight_layout()
    plt.savefig('data/processed/chart_freq_rec.png', dpi=80)
    plt.close()
except Exception as e:
    print(f"chart-freq-rec skipped: {e}")
    plt.close('all')

# ── Cell: chart-prob-alive ───────────────────────────────────────────────────
try:
    fig, ax = plt.subplots(figsize=(12, 7))
    plot_probability_alive_matrix(bgf, ax=ax)
    ax.set_title('BG/NBD — Probability Customer is Still Alive')
    plt.tight_layout()
    plt.savefig('data/processed/chart_prob_alive.png', dpi=80)
    plt.close()
except Exception as e:
    print(f"chart-prob-alive skipped: {e}")
    plt.close('all')

# ── Cell: prob-alive-dist ────────────────────────────────────────────────────
try:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(prob_alive, bins=60, color='steelblue', edgecolor='white')
    ax.axvline(0.5, color='red', linestyle='--', linewidth=1.5, label='50% threshold')
    ax.set_title('Distribution of P(Alive) Across Repeat Customers')
    ax.set_xlabel('Probability Alive')
    ax.legend()
    pct_alive = (prob_alive > 0.5).mean() * 100
    ax.text(0.55, ax.get_ylim()[1] * 0.85, f'{pct_alive:.1f}% of customers\nhave P(alive) > 0.5', fontsize=11)
    plt.tight_layout()
    plt.savefig('data/processed/chart_prob_alive_dist.png', dpi=80)
    plt.close()
except Exception as e:
    print(f"prob-alive-dist skipped: {e}")
    plt.close('all')

# ── Cell: gg-assumption-check ────────────────────────────────────────────────
corr = check_gg_assumption(clv_df)

# ── Cell: gg-assumption-plot ─────────────────────────────────────────────────
try:
    fig, ax = plt.subplots(figsize=(8, 6))
    freq_clip = clv_df['frequency_repeat'].clip(upper=clv_df['frequency_repeat'].quantile(0.97))
    mon_clip = clv_df['monetary'].clip(upper=clv_df['monetary'].quantile(0.97))
    ax.scatter(freq_clip, mon_clip, alpha=0.2, s=8, color='steelblue')
    ax.set_xlabel('Frequency (repeat purchases)')
    ax.set_ylabel('Monetary (mean order £)')
    ax.set_title(f'Frequency vs Monetary  (Pearson r = {corr:.3f})')
    if abs(corr) > 0.3:
        ax.text(0.05, 0.92, 'WARNING: |r| > 0.3 — independence assumption may be violated',
                transform=ax.transAxes, color='red', fontsize=10)
    plt.tight_layout()
    plt.savefig('data/processed/chart_gg_assumption.png', dpi=80)
    plt.close()
except Exception as e:
    print(f"gg-assumption-plot skipped: {e}")
    plt.close('all')

# ── Cell: fit-gg ─────────────────────────────────────────────────────────────
ggf = fit_gamma_gamma(clv_df)

# ── Cell: score-clv ──────────────────────────────────────────────────────────
expected_avg_order_value, clv_90d, clv_365d = score_clv(bgf, ggf, clv_df)
scores = clv_df[['customer_id', 'frequency_repeat', 'monetary', 'total_revenue']].copy()
scores['prob_alive'] = prob_alive.values
scores['predicted_purchases_90d'] = predicted_purchases_90d.values
scores['expected_avg_order_value'] = expected_avg_order_value.values
scores['clv_90d'] = clv_90d.values
scores['clv_365d'] = clv_365d.values
print(scores[['clv_90d', 'clv_365d', 'prob_alive', 'expected_avg_order_value']].describe().round(2))

# ── Cell: chart-clv-hist ─────────────────────────────────────────────────────
try:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    clv_90_pos = scores[scores['clv_90d'] > 0]['clv_90d']
    axes[0].hist(clv_90_pos, bins=80, color='steelblue', edgecolor='none', alpha=0.85)
    axes[0].set_yscale('log')
    axes[0].set_title('CLV 90-Day Distribution (log y-axis)')
    axes[0].set_xlabel('Predicted CLV — 90 days (£)')
    axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'£{x:,.0f}'))
    clv_365_pos = scores[scores['clv_365d'] > 0]['clv_365d']
    axes[1].hist(clv_365_pos, bins=80, color='seagreen', edgecolor='none', alpha=0.85)
    axes[1].set_yscale('log')
    axes[1].set_title('CLV 365-Day Distribution (log y-axis)')
    axes[1].set_xlabel('Predicted CLV — 365 days (£)')
    axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'£{x:,.0f}'))
    plt.suptitle('Predicted CLV Distributions — Both Horizons', fontsize=13)
    plt.tight_layout()
    plt.savefig('data/processed/chart_clv_hist.png', dpi=80)
    plt.close()
except Exception as e:
    print(f"chart-clv-hist skipped: {e}")
    plt.close('all')

# ── Cell: CLV vs actual (skip plotly show) ───────────────────────────────────
corr_clv = scores['clv_365d'].corr(scores['total_revenue'])
print(f'Pearson correlation (CLV 365d vs actual revenue): {corr_clv:.3f}')

# ── Cell: clv-segments ───────────────────────────────────────────────────────
scores['clv_tier'] = pd.qcut(
    scores['clv_365d'],
    q=[0, 0.25, 0.5, 0.75, 1.0],
    labels=['Low', 'Mid', 'High', 'Top']
)

tier_summary = (
    scores.groupby('clv_tier', observed=True)
    .agg(
        customer_count=('customer_id', 'count'),
        median_clv_365d=('clv_365d', 'median'),
        total_predicted_revenue=('clv_365d', 'sum'),
        median_prob_alive=('prob_alive', 'median'),
    )
    .reset_index()
)
print(tier_summary.round(2))

# ── Cell: tier-bar ───────────────────────────────────────────────────────────
try:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(tier_summary['clv_tier'].astype(str), tier_summary['customer_count'],
                color=['#d9534f', '#f0ad4e', '#5bc0de', '#5cb85c'])
    axes[0].set_title('Customer Count by CLV Tier')
    axes[0].set_xlabel('CLV Tier')
    axes[0].set_ylabel('Customers')
    for i, v in enumerate(tier_summary['customer_count']):
        axes[0].text(i, v + v * 0.01, f'{v:,}', ha='center', fontsize=10)
    axes[1].bar(tier_summary['clv_tier'].astype(str), tier_summary['total_predicted_revenue'],
                color=['#d9534f', '#f0ad4e', '#5bc0de', '#5cb85c'])
    axes[1].set_title('Total Predicted Revenue by CLV Tier (365d)')
    axes[1].set_xlabel('CLV Tier')
    axes[1].set_ylabel('Revenue (£)')
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'£{x/1e6:.1f}M'))
    plt.suptitle('CLV Tier Breakdown', fontsize=13)
    plt.tight_layout()
    plt.savefig('data/processed/chart_clv_tiers.png', dpi=80)
    plt.close()
except Exception as e:
    print(f"tier-bar skipped: {e}")
    plt.close('all')

# ── Cell: alive-by-tier ──────────────────────────────────────────────────────
try:
    fig, ax = plt.subplots(figsize=(9, 5))
    scores.boxplot(column='prob_alive', by='clv_tier', ax=ax, patch_artist=True)
    ax.set_title('P(Alive) Distribution by CLV Tier')
    ax.set_xlabel('CLV Tier')
    ax.set_ylabel('Probability Alive')
    plt.suptitle('')
    plt.tight_layout()
    plt.savefig('data/processed/chart_alive_by_tier.png', dpi=80)
    plt.close()
except Exception as e:
    print(f"alive-by-tier skipped: {e}")
    plt.close('all')

# ── Cell: save-all ───────────────────────────────────────────────────────────
save_models(bgf, ggf)

Path('data/processed').mkdir(parents=True, exist_ok=True)
scores[['customer_id', 'frequency_repeat', 'prob_alive',
        'predicted_purchases_90d', 'expected_avg_order_value',
        'clv_90d', 'clv_365d']].to_csv(
    Path('data/processed/clv_scores.csv'), index=False
)
print('saved models/bgf.pkl, models/ggf.pkl, data/processed/clv_scores.csv')

# ── Cell: summary ────────────────────────────────────────────────────────────
print('=== Phase 2 Complete ===')
print(f'repeat customers scored  : {len(scores):,}')
print(f'median CLV 90d           : £{scores["clv_90d"].median():.2f}')
print(f'median CLV 365d          : £{scores["clv_365d"].median():.2f}')
print(f'mean   CLV 365d          : £{scores["clv_365d"].mean():.2f}')
print(f'% customers P(alive)>0.5 : {(scores["prob_alive"] > 0.5).mean() * 100:.1f}%')
print(f'\nCLV tier distribution:')
print(tier_summary[['clv_tier', 'customer_count', 'median_clv_365d']].to_string(index=False))
