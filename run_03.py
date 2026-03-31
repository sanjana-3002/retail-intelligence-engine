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

from sklearn.model_selection import train_test_split
from lifelines import CoxPHFitter, KaplanMeierFitter
import joblib
from pathlib import Path

from models.data_pipeline import (
    load_raw_data, extract_signal_before_cleaning,
    clean_data, build_master_customer_table,
)
from models.churn_model import (
    define_churn_label, prepare_churn_features,
    fit_xgboost_classifier, evaluate_classifier,
    get_shap_values, get_top_shap_drivers,
    fit_cox_model, predict_survival_days,
    compute_treatment_proxy, fit_uplift_tlearner, score_uplift,
    build_full_intelligence_table,
    CHURN_FEATURES,
)

sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.figsize'] = (12, 5)

# ── Cell: load-data ──────────────────────────────────────────────────────────
df_raw = load_raw_data()
return_features, cancel_features = extract_signal_before_cleaning(df_raw)
df_customers, df_all = clean_data(df_raw)
master = build_master_customer_table(df_customers, return_features, cancel_features)

clv_scores = pd.read_csv('data/processed/clv_scores.csv')
print(f'\nmaster shape     : {master.shape}')
print(f'clv_scores shape : {clv_scores.shape}')

# ── Cell: define-churn-label ─────────────────────────────────────────────────
master_labelled = define_churn_label(master)

try:
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = master_labelled['churned'].value_counts().sort_index()
    bars = ax.bar(['Active (0)', 'Churned (1)'], counts.values,
                  color=['#5cb85c', '#d9534f'], width=0.5)
    ax.set_title('Churn Label Distribution', fontsize=13)
    ax.set_ylabel('Customer Count')
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + val * 0.01,
                f'{val:,}', ha='center', fontsize=11)
    plt.tight_layout()
    plt.savefig('tmp_plot.png')
    plt.close()
except Exception as e:
    print(f'churn-dist plot skipped: {e}')
    plt.close('all')

# ── Cell: prepare-features ───────────────────────────────────────────────────
X, y = prepare_churn_features(master_labelled)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f'train size : {len(X_train):,}  (churned: {y_train.sum():,})')
print(f'test size  : {len(X_test):,}   (churned: {y_test.sum():,})')

# ── Cell: fit-xgboost ────────────────────────────────────────────────────────
# Override to fix early_stopping_rounds API change in newer XGBoost
from xgboost import XGBClassifier as _XGBClassifier

def fit_xgboost_classifier_fixed(X_train, y_train, X_test, y_test):
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f'scale_pos_weight : {scale_pos_weight:.3f}')
    model = _XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        eval_metric='auc',
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=20,
        random_state=42,
        verbosity=0,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    print(f'best iteration   : {model.best_iteration}')
    return model

xgb_model = fit_xgboost_classifier_fixed(X_train, y_train, X_test, y_test)

# ── Cell: evaluate-classifier ────────────────────────────────────────────────
# Override evaluate_classifier to save plots instead of showing them
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay,
)

def evaluate_classifier_agg(model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, output_dict=True)
    print(f'AUC-ROC   : {auc:.4f}')
    print(classification_report(y_test, y_pred, target_names=['active', 'churned']))
    try:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=['active', 'churned'])
        disp.plot(ax=axes[0], colorbar=False, cmap='Blues')
        axes[0].set_title('Confusion Matrix')
        RocCurveDisplay.from_predictions(y_test, y_prob, ax=axes[1], name='XGBoost')
        axes[1].plot([0, 1], [0, 1], 'k--', lw=1)
        axes[1].set_title(f'ROC Curve  (AUC = {auc:.3f})')
        plt.tight_layout()
        plt.savefig('data/processed/chart_churn_eval.png', dpi=80)
        plt.close()
    except Exception as e:
        print(f'eval plot skipped: {e}')
        plt.close('all')
    # classification_report output_dict uses target_names if given, else str(label)
    # check possible keys for the positive (churned) class
    for churn_key in ['churned', '1', 1]:
        if churn_key in report:
            break
    else:
        # fallback: use second key (after accuracy / macro / weighted)
        churn_key = [k for k in report if isinstance(k, str) and k not in ('accuracy', 'macro avg', 'weighted avg')][-1]
    return {
        'auc': auc,
        'precision': report[churn_key]['precision'],
        'recall': report[churn_key]['recall'],
        'f1': report[churn_key]['f1-score'],
        'y_prob': y_prob,
    }

metrics = evaluate_classifier_agg(xgb_model, X_test, y_test)
print(f"\nAUC-ROC   : {metrics['auc']:.4f}")
print(f"Precision : {metrics['precision']:.4f}")
print(f"Recall    : {metrics['recall']:.4f}")
print(f"F1        : {metrics['f1']:.4f}")

# ── Cell: save-churn-model ───────────────────────────────────────────────────
Path('models').mkdir(parents=True, exist_ok=True)
joblib.dump(xgb_model, 'models/churn_xgb.pkl')

churn_scores = master_labelled[['customer_id']].copy()
churn_scores['churn_probability'] = xgb_model.predict_proba(X)[:, 1]
print('saved models/churn_xgb.pkl')
print(churn_scores['churn_probability'].describe().round(4))

# ── Cell: shap-values ────────────────────────────────────────────────────────
import shap

explainer, shap_values = get_shap_values(xgb_model, X_test)

# Plot 1 — SHAP beeswarm summary
try:
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig('data/processed/chart_shap_summary.png', dpi=80)
    plt.close()
except Exception as e:
    print(f'shap summary plot skipped: {e}')
    plt.close('all')

# Plot 2 — SHAP bar chart
try:
    shap.summary_plot(shap_values, X_test, plot_type='bar', show=False)
    plt.tight_layout()
    plt.savefig('data/processed/chart_shap_bar.png', dpi=80)
    plt.close()
except Exception as e:
    print(f'shap bar plot skipped: {e}')
    plt.close('all')

# Plot 3 — SHAP waterfall for one at-risk customer
y_prob_test = metrics['y_prob']
at_risk_idx = np.where(y_prob_test > 0.7)[0]
print(f'customers with churn_prob > 0.7 in test set: {len(at_risk_idx):,}')

if len(at_risk_idx) > 0:
    try:
        idx = at_risk_idx[0]
        shap_exp = shap.Explanation(
            values=shap_values[idx],
            base_values=explainer.expected_value,
            data=X_test.iloc[idx].values,
            feature_names=CHURN_FEATURES,
        )
        shap.plots.waterfall(shap_exp, show=False)
        plt.tight_layout()
        plt.savefig('data/processed/chart_shap_waterfall.png', dpi=80)
        plt.close()
    except Exception as e:
        print(f'shap waterfall plot skipped: {e}')
        plt.close('all')

# save shap values
np.save('data/processed/shap_values.npy', shap_values)
print('saved data/processed/shap_values.npy')

# ── Cell: cox-model ──────────────────────────────────────────────────────────
cox_features = ['recency', 'frequency', 'monetary', 'velocity_decay_ratio',
                'category_hhi', 'spend_cv', 'return_rate', 'cancellation_rate',
                'avg_items_per_order', 'country_count', 'customer_tenure_days', 'churned']

df_cox = master_labelled[cox_features].fillna(0).copy()
df_cox['customer_tenure_days'] = df_cox['customer_tenure_days'].clip(lower=1)

cph = fit_cox_model(df_cox)
cph.print_summary()

# ── Cell: cox-forest-plot ────────────────────────────────────────────────────
try:
    fig, ax = plt.subplots(figsize=(10, 7))
    cph.plot(ax=ax)
    ax.set_title('Cox PH — Hazard Ratio Forest Plot (95% CI)')
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    plt.tight_layout()
    plt.savefig('data/processed/chart_cox_forest.png', dpi=80)
    plt.close()
except Exception as e:
    print(f'cox forest plot skipped: {e}')
    plt.close('all')

joblib.dump(cph, 'models/cox_model.pkl')
print('saved models/cox_model.pkl')

# ── Cell: kaplan-meier ───────────────────────────────────────────────────────
try:
    km_df = master_labelled[['customer_id', 'customer_tenure_days', 'churned',
                              'category_hhi']].copy()
    km_df['customer_id'] = km_df['customer_id'].astype(str)
    clv_scores_km = clv_scores[['customer_id', 'clv_365d']].copy()
    clv_scores_km['customer_id'] = clv_scores_km['customer_id'].astype(str)
    km_df = km_df.merge(clv_scores_km, on='customer_id', how='left')
    km_df['customer_tenure_days'] = km_df['customer_tenure_days'].clip(lower=1)
    km_df = km_df.dropna(subset=['clv_365d'])

    clv_q75 = km_df['clv_365d'].quantile(0.75)
    hhi_q75 = km_df['category_hhi'].quantile(0.75)

    fig, ax = plt.subplots(figsize=(11, 6))

    groups = {
        f'High CLV (> £{clv_q75:.0f})': km_df[km_df['clv_365d'] >  clv_q75],
        f'Low CLV (\u2264 £{clv_q75:.0f})':  km_df[km_df['clv_365d'] <= clv_q75],
        f'High HHI (> {hhi_q75:.2f})':  km_df[km_df['category_hhi'] >  hhi_q75],
        f'Low HHI (\u2264 {hhi_q75:.2f})':   km_df[km_df['category_hhi'] <= hhi_q75],
    }
    colors = ['#2196F3', '#90CAF9', '#F44336', '#EF9A9A']

    for (label, grp), color in zip(groups.items(), colors):
        kmf = KaplanMeierFitter()
        kmf.fit(grp['customer_tenure_days'], event_observed=grp['churned'], label=label)
        kmf.plot_survival_function(ax=ax, ci_show=False, color=color)

    ax.set_title('Kaplan-Meier Survival Curves — CLV and HHI Groups', fontsize=13)
    ax.set_xlabel('Customer Tenure (days)')
    ax.set_ylabel('Survival Probability')
    ax.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig('data/processed/chart_km_survival.png', dpi=80)
    plt.close()
except Exception as e:
    print(f'kaplan-meier plot skipped: {e}')
    plt.close('all')

# ── Cell: uplift-model ───────────────────────────────────────────────────────
master_with_treatment = compute_treatment_proxy(df_customers, master_labelled)
print(master_with_treatment[['customer_id', 'treatment', 'churned']].head())

X_uplift, y_uplift = prepare_churn_features(master_with_treatment)
treatment_col = master_with_treatment['treatment'].values

model_t, model_c = fit_uplift_tlearner(X_uplift, y_uplift, treatment_col)

master_uplift = score_uplift(model_t, model_c, X_uplift, master_with_treatment)
print(master_uplift['customer_segment'].value_counts())

# ── Cell: uplift-scatter (skip plotly, use matplotlib) ───────────────────────
try:
    segment_colors = {
        'persuadable':  '#2196F3',
        'lost_cause':   '#F44336',
        'sleeping_dog': '#FF9800',
        'sure_thing':   '#4CAF50',
    }
    plot_df = master_uplift[['customer_id', 'uplift_score', 'customer_segment']].copy()
    plot_df = plot_df.merge(churn_scores, on='customer_id', how='left')
    plot_df = plot_df.sample(min(4000, len(plot_df)), random_state=42)

    fig, ax = plt.subplots(figsize=(10, 7))
    for seg, color in segment_colors.items():
        subset = plot_df[plot_df['customer_segment'] == seg]
        ax.scatter(subset['churn_probability'], subset['uplift_score'],
                   c=color, alpha=0.55, s=15, label=seg)
    ax.axhline(0.1, linestyle='--', color='grey', linewidth=1)
    ax.axvline(0.5, linestyle='--', color='grey', linewidth=1)
    ax.set_xlabel('Churn Probability (XGBoost)')
    ax.set_ylabel('Uplift Score (T-Learner)')
    ax.set_title('Churn Probability vs Uplift Score by Customer Segment')
    ax.legend()
    plt.tight_layout()
    plt.savefig('data/processed/chart_uplift_scatter.png', dpi=80)
    plt.close()
except Exception as e:
    print(f'uplift scatter plot skipped: {e}')
    plt.close('all')

# ── Cell: save-uplift-scores ─────────────────────────────────────────────────
uplift_out = master_uplift[['customer_id', 'uplift_score', 'customer_segment']]
uplift_out.to_csv('data/processed/uplift_scores.csv', index=False)
print(f'saved uplift_scores.csv — {len(uplift_out):,} customers')
print(uplift_out['customer_segment'].value_counts())

# ── Cell: cox-survival-days ──────────────────────────────────────────────────
cox_input = df_cox[cph.params_.index.tolist()].copy()
survival_days_vals = []
for i in range(len(cox_input)):
    row = cox_input.iloc[[i]]
    survival_days_vals.append(predict_survival_days(cph, row))

cox_survival = master_labelled[['customer_id']].copy()
cox_survival['predicted_survival_days'] = survival_days_vals
print(pd.Series(survival_days_vals).describe())

# ── Cell: build-full-intelligence-table ──────────────────────────────────────
# Build the full intelligence table manually to avoid column duplication bug
# in build_full_intelligence_table when master already contains customer_id

def normalize_cid(df):
    df = df.copy()
    df['customer_id'] = df['customer_id'].astype(str)
    return df

ml_norm = normalize_cid(master_labelled)
clv_norm = normalize_cid(clv_scores)
churn_norm = normalize_cid(churn_scores)
cox_norm = normalize_cid(cox_survival)
uplift_norm = normalize_cid(uplift_out)

full_intel = ml_norm.copy()
for right_df in [clv_norm, churn_norm, cox_norm, uplift_norm]:
    extra_cols = [c for c in right_df.columns if c != 'customer_id' and c not in full_intel.columns]
    if extra_cols:
        full_intel = full_intel.merge(
            right_df[['customer_id'] + extra_cols].drop_duplicates('customer_id'),
            on='customer_id',
            how='left',
        )

Path('data/processed').mkdir(parents=True, exist_ok=True)
out_path = Path('data/processed/full_customer_intelligence.csv')
full_intel.to_csv(out_path, index=False)
print(f'full_customer_intelligence.csv : {full_intel.shape[0]:,} rows x {full_intel.shape[1]} columns')
print(f'saved to {out_path}')
print(full_intel.head())

# ── Cell: summary ────────────────────────────────────────────────────────────
print('=== Phase 3 Complete ===')
print(f'customers scored          : {len(full_intel):,}')
print(f'XGBoost AUC-ROC           : {metrics["auc"]:.4f}')
print(f'XGBoost F1 (churned)      : {metrics["f1"]:.4f}')
print(f'mean churn probability    : {churn_scores["churn_probability"].mean():.3f}')
print(f'% churned (label)         : {master_labelled["churned"].mean() * 100:.1f}%')
print()
print('Uplift segment breakdown:')
print(master_uplift['customer_segment'].value_counts().to_string())
print()
print('Saved artefacts:')
print('  models/churn_xgb.pkl')
print('  models/cox_model.pkl')
print('  data/processed/shap_values.npy')
print('  data/processed/uplift_scores.csv')
print('  data/processed/full_customer_intelligence.csv')
