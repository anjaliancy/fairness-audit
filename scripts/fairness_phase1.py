# Auto-generated from fairness_phase1.ipynb
# Source notebook exported as a plain Python script.

# %% [markdown]
# # Phase 1 — Fairness Pre-Processing: Disparate Impact Remover
#
# **Purpose:** Apply AIF360's Disparate Impact Remover (DIR) to the training data before fine/coarse classing and WOE feature engineering.
#
# **Pipeline position:**
# ```
# Raw Data → Geographic Encoding → Data Cleaning → DIR Sweep → Repaired Dataset → WOE/Classing
# ```
#
# **What this notebook does:**
# 1. Encodes ZIP / State / MSA as US Census Divisions (removes raw redlining vectors)
# 2. Cleans sentinel values and drops null columns
# 3. Measures baseline fairness (Disparate Impact Ratio, Statistical Parity Difference)
# 4. Sweeps DIR repair levels λ ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}
# 5. Reports performance–fairness trade-off table
# 6. Saves the selected repaired dataset for downstream use

# %% [markdown]
# ## 0. Imports & Configuration

# %%
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin

from geographic_encoding import add_geographic_region_features
from fairness_preprocessing import (
    apply_disparate_impact_repair,
    evaluate_disparate_impact_and_spd,
    sweep_repair_levels,
)

pd.set_option('display.float_format', '{:.4f}'.format)
pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 120)

DATA_PATH       = 'data/classing/main_alt_train_full_stratified.csv'
OUTPUT_DIR      = 'data/classing'
LABEL_COL       = 'target'
FAVORABLE_LABEL = 0   # 0 = non-default (good outcome)
REPAIR_LEVELS   = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

# %% [markdown]
# ## 1. Load Raw Data

# %%
df_raw = pd.read_csv(DATA_PATH)
print(f'Shape: {df_raw.shape}')
print(f'Default rate: {df_raw[LABEL_COL].mean():.4f}')
df_raw.head(3)

# %% [markdown]
# ## 2. Geographic Encoding
#
# Replace raw identifiers with Census Division labels.
#
# | Raw column | Problem | Replacement |
# |------------|---------|-------------|
# | `State` | 54 categories — model learns racial composition per state | `census_division` (9 categories) + `is_southern_region` (binary) |
# | `ZIP` | 3-digit ZIP3 prefix — classic redlining vector | `zip3_division` (Census Division derived from ZIP3 prefix) |
# | `MSA` | 406 CBSA codes — encodes neighbourhood composition | `msa_division` (Census Division derived from CBSA) |
#
# Lookup maps are **fit on the training set only** and reused on the test set.

# %%
df_encoded, geo_maps = add_geographic_region_features(
    df_raw,
    zip_col='ZIP',
    state_col='State',
    msa_col='MSA',
    drop_originals=True,
)

print('Census Division distribution:')
print(df_encoded['census_division'].value_counts().sort_index().to_string())
print()
print('is_southern_region distribution:')
print(df_encoded['is_southern_region'].value_counts().to_string())

# %%
for col in ['ZIP', 'State', 'MSA']:
    assert col not in df_encoded.columns, f'{col} still present'
print('Raw geographic IDs removed.')
print(f'Encoded shape: {df_encoded.shape}')

# %% [markdown]
# ## 3. Data Cleaning
#
# - Drop 100% null columns
# - Recode Credit Score sentinel 9999 → NaN
# - Drop `high_elec_pct` (r = 0.993 with `avg_elec_cost` — redundant)
# - Drop non-modelling ID column (`Loan Sequence Number`)

# %%
null_pct = df_encoded.isnull().mean()
fully_null = null_pct[null_pct == 1.0].index.tolist()
print(f'Dropping 100% null columns: {fully_null}')
df_clean = df_encoded.drop(columns=fully_null)

sentinel_count = (df_clean['Credit Score'] == 9999).sum()
print(f'Credit Score = 9999 (sentinel): {sentinel_count} records -> recoded to NaN')
df_clean['Credit Score'] = df_clean['Credit Score'].replace(9999, np.nan)

# Drop known leakage/redundant/non-model columns first.
drop_cols = [
    c for c in ['high_elec_pct', 'Loan Sequence Number', 'Defaulter (Y/N)', 'max_deliquency']
    if c in df_clean.columns
]

# Drop all delinquency-related columns (for leakage control).
delinquency_cols = [
    c for c in df_clean.columns
    if 'deliq' in c.lower() or 'delinquen' in c.lower()
]

drop_cols = sorted(list(set(drop_cols + delinquency_cols)))
print(f'Dropping redundant/ID/leakage columns: {drop_cols}')
df_clean = df_clean.drop(columns=drop_cols)

# Drop columns with only one distinct non-null value (no predictive signal).
constant_cols = [
    c for c in df_clean.columns
    if c != LABEL_COL and df_clean[c].nunique(dropna=True) <= 1
]
print(f'Dropping constant-value columns: {constant_cols}')
df_clean = df_clean.drop(columns=constant_cols)

print(f'Final shape: {df_clean.shape}')
remaining_nulls = df_clean.isnull().sum()
print('Remaining nulls:')
print(remaining_nulls[remaining_nulls > 0].to_string())

# %% [markdown]
# ## 4. Encode Categorical Features for AIF360
#
# AIF360 `StandardDataset` requires numeric features. String columns are ordinal-encoded here. WOE encoding replaces these in the downstream classing step.

# %%
cat_cols = [
    c for c in df_clean.select_dtypes(include=['object', 'str']).columns
    if c != LABEL_COL
]
print(f'Categorical columns to label-encode: {cat_cols}')

df_model = df_clean.copy()
for col in cat_cols:
    df_model[col] = pd.Categorical(df_model[col]).codes.astype(float)
    df_model[col] = df_model[col].replace(-1, np.nan)

print('Encoding complete.')
print(df_model.dtypes.value_counts().to_string())

# %% [markdown]
# ## 5. Define Protected Attributes
#
# **Primary — `is_southern_region`**  
# - 1 = South Atlantic / East South Central / West South Central (unprivileged)  
# - 0 = All other Census Divisions (privileged)  
# - Rationale: Southern states show default rates up to 2× the national average (LA 26%, MS 20%, FL 18%, GA 18%)
#
# **Secondary — `high_poverty_area`**  
# - 1 = poverty_rate above dataset median (unprivileged)  
# - 0 = at or below median (privileged)

# %%
poverty_median = df_model['poverty_rate'].median()
df_model['high_poverty_area'] = (df_model['poverty_rate'] > poverty_median).astype(int)
print(f'poverty_rate median: {poverty_median:.4f}')
print(df_model['high_poverty_area'].value_counts().to_string())

PROTECTED_ATTRIBUTES = ['is_southern_region', 'high_poverty_area']
PRIVILEGED_VALUES    = {'is_southern_region': 0, 'high_poverty_area': 0}
print(f'\nProtected attributes : {PROTECTED_ATTRIBUTES}')
print(f'Privileged values    : {PRIVILEGED_VALUES}')

# %%
# ── Impute remaining nulls before any AIF360 call ─────────────────────────────
# AIF360 silently drops rows with NaN; impute in-place on df_model so that
# §6 baseline, §7 sweep, and §10 repair all operate on all 70,000 rows.
# Protected attributes: mode-fill.  All other columns: median-fill.
_null_before  = int(df_model.isna().sum().sum())
_imputed_mask = df_model.isna().any(axis=1)

for _col in df_model.columns:
    if _col == LABEL_COL:
        continue
    if _col in PROTECTED_ATTRIBUTES:
        _mode = df_model[_col].mode(dropna=True)
        _fill = _mode.iloc[0] if not _mode.empty else 0.0
    else:
        _fill = df_model[_col].median(skipna=True)
        if pd.isna(_fill):
            _fill = 0.0
    df_model[_col] = df_model[_col].fillna(_fill)

df_model['was_imputed'] = _imputed_mask.astype(int)

print(f'Nulls before imputation : {_null_before}')
print(f'Nulls after imputation  : {df_model.isna().sum().sum()}')
print(f'Rows imputed            : {_imputed_mask.sum():,}  ({100 * _imputed_mask.mean():.1f}%)')
print(f'df_model shape          : {df_model.shape}')

# %% [markdown]
# ## 6. Baseline Fairness Measurement
#
# | Metric | Threshold | Interpretation |
# |--------|-----------|----------------|
# | DIR < 0.8 | 4/5ths rule | Unprivileged group selected at < 80% the rate of privileged |
# | SPD < 0 | Negative = disadvantaged | Magnitude > 0.05 is material |

# %%
baseline_fairness = evaluate_disparate_impact_and_spd(
    df=df_model,
    label_name=LABEL_COL,
    protected_attributes=PROTECTED_ATTRIBUTES,
    favorable_label=FAVORABLE_LABEL,
    privileged_values=PRIVILEGED_VALUES,
)

print('=== BASELINE FAIRNESS (no repair) ===')
print(baseline_fairness.to_string(index=False))
print()
print('4/5ths rule threshold: DIR >= 0.80')
for _, row in baseline_fairness.iterrows():
    flag = 'VIOLATION' if (pd.notna(row['disparate_impact_ratio'])
                           and row['disparate_impact_ratio'] < 0.8) else 'OK'
    print(f'  {row["protected_attribute"]:25s}  '
          f'DIR={row["disparate_impact_ratio"]:.4f}  [{flag}]')

# %% [markdown]
# ## 7. DIR Repair Sweep
#
# Sweep λ ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}. At each level: apply DIR → train logistic regression → measure AUC/KS and DIR/SPD.  
# The LR here is for **quantifying performance degradation only**. The production model (LR/RF/XGB with WOE) is built in Phase 2.

# %%
class MedianImputePipeline(BaseEstimator, ClassifierMixin):
    """Sklearn pipeline wrapper that median-imputes NaNs before predict."""
    def __init__(self, pipe, train_medians):
        self.pipe = pipe
        self.train_medians = train_medians
    def predict_proba(self, X):
        return self.pipe.predict_proba(X.fillna(self.train_medians))
    def predict(self, X):
        return self.pipe.predict(X.fillna(self.train_medians))

def robust_model_builder(X_train, y_train):
    medians = X_train.median(numeric_only=True)
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(max_iter=1000, random_state=42,
                                  class_weight='balanced')),
    ])
    pipe.fit(X_train.fillna(medians), y_train)
    return MedianImputePipeline(pipe, medians)

print('Running DIR sweep across repair levels:', REPAIR_LEVELS)
print('(This may take ~2-3 minutes)')

# %%
print('=== IMPUTATION SUMMARY ===')
print(f'df_model shape: {df_model.shape}')
print()
print('was_imputed distribution:')
print(df_model['was_imputed'].value_counts().sort_index().to_string())
print()
for val in sorted(df_model['was_imputed'].unique()):
    count = (df_model['was_imputed'] == val).sum()
    pct = 100 * count / len(df_model)
    label = 'HAD MISSING VALUES' if val == 1 else 'NO MISSING VALUES'
    print(f'  {label:25s}: {count:6,} records ({pct:5.2f}%)')

# %%
# Verify no leaky delinquency columns survived into the modelling dataframe
remaining_delinquency_cols = [
    c for c in df_model.columns
    if 'deliq' in c.lower() or 'delinquen' in c.lower()
]
print('Delinquency-related columns in df_model:', remaining_delinquency_cols)
assert len(remaining_delinquency_cols) == 0, 'Leaky columns still present!'
print('OK — no leakage columns in df_model.')

# %%
sweep_results = sweep_repair_levels(
    train_df=df_model,
    label_name=LABEL_COL,
    protected_attributes=PROTECTED_ATTRIBUTES,
    model_builder=robust_model_builder,
    repair_levels=REPAIR_LEVELS,
    favorable_label=FAVORABLE_LABEL,
    privileged_values=PRIVILEGED_VALUES,
    sequential_repair=True,
    fairness_on='train',
)
print('Sweep complete.')
sweep_results

# %% [markdown]
# ## 8. Results — Performance–Fairness Trade-Off Table

# %%
display_cols = [
    'repair_level', 'protected_attribute',
    'disparate_impact_ratio', 'delta_di_vs_baseline',
    'statistical_parity_difference', 'delta_spd_vs_baseline',
    'auc', 'delta_auc_vs_baseline',
    'ks',  'delta_ks_vs_baseline',
]
print('=== FAIRNESS-PERFORMANCE TRADE-OFF TABLE ===')
print(sweep_results[display_cols].to_string(index=False))

# %%
print('=== REPAIR LEVEL RECOMMENDATION (first λ achieving DIR >= 0.80) ===')
for attr in PROTECTED_ATTRIBUTES:
    subset = sweep_results[sweep_results['protected_attribute'] == attr].copy()
    compliant = subset[subset['disparate_impact_ratio'] >= 0.80]
    if compliant.empty:
        print(f'  {attr:25s}: No repair level achieves DIR >= 0.80 — flag for review')
    else:
        best = compliant.iloc[0]
        print(
            f'  {attr:25s}: lambda={best["repair_level"]:.1f}  '
            f'DIR={best["disparate_impact_ratio"]:.4f}  '
            f'delta_AUC={best["delta_auc_vs_baseline"]:+.4f}  '
            f'delta_KS={best["delta_ks_vs_baseline"]:+.4f}'
        )

# %% [markdown]
# ## 9. Visualisation — Repair Level vs DIR / AUC

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('DIR Sweep: Fairness–Performance Trade-Off', fontsize=13, fontweight='bold')

colors = {'is_southern_region': '#e15759', 'high_poverty_area': '#4e79a7'}

for attr in PROTECTED_ATTRIBUTES:
    subset = sweep_results[sweep_results['protected_attribute'] == attr]
    axes[0].plot(
        subset['repair_level'], subset['disparate_impact_ratio'],
        marker='o', label=attr, color=colors.get(attr)
    )
    axes[1].plot(
        subset['repair_level'], subset['auc'],
        marker='o', label=attr, color=colors.get(attr)
    )

axes[0].axhline(0.80, color='black', linestyle='--', linewidth=1.2,
                label='4/5ths threshold (0.80)')
axes[0].set_xlabel('Repair Level (λ)')
axes[0].set_ylabel('Disparate Impact Ratio')
axes[0].set_title('Disparate Impact Ratio vs Repair Level')
axes[0].legend()
axes[0].set_ylim(0, 1.2)
axes[0].grid(alpha=0.3)

axes[1].set_xlabel('Repair Level (λ)')
axes[1].set_ylabel('AUC')
axes[1].set_title('AUC vs Repair Level')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('data/classing/dir_sweep_results.png', dpi=150, bbox_inches='tight')
plt.show()
print('Plot saved to data/classing/dir_sweep_results.png')

# %% [markdown]
# ## 10. Apply Selected Repair Level & Save
#
# Select the first λ where DIR ≥ 0.80 for the primary protected attribute (`is_southern_region`). The repaired dataset feeds directly into fine/coarse classing.

# %%
primary_attr = 'is_southern_region'
sweep_primary = sweep_results[sweep_results['protected_attribute'] == primary_attr]
compliant_primary = sweep_primary[sweep_primary['disparate_impact_ratio'] >= 0.80]

if compliant_primary.empty:
    SELECTED_LAMBDA = 1.0
    print(f'WARNING: No repair level achieves DIR >= 0.80 for {primary_attr}. '
          f'Defaulting to lambda=1.0.')
else:
    SELECTED_LAMBDA = float(compliant_primary.iloc[0]['repair_level'])

print(f'Selected repair level: lambda = {SELECTED_LAMBDA}')

# %%
result = apply_disparate_impact_repair(
    train_df=df_model,
    label_name=LABEL_COL,
    protected_attributes=PROTECTED_ATTRIBUTES,
    repair_level=SELECTED_LAMBDA,
    favorable_label=FAVORABLE_LABEL,
    privileged_values=PRIVILEGED_VALUES,
    sequential_repair=True,
)

print('=== FAIRNESS METRICS: BEFORE vs AFTER REPAIR ===')
before = result.fairness_report_train_original
after  = result.fairness_report_train
comparison = before.merge(after, on='protected_attribute', suffixes=('_before', '_after'))
comparison['dir_change'] = (
    comparison['disparate_impact_ratio_after']
    - comparison['disparate_impact_ratio_before']
)
comparison['spd_change'] = (
    comparison['statistical_parity_difference_after']
    - comparison['statistical_parity_difference_before']
)
print(comparison[[
    'protected_attribute',
    'disparate_impact_ratio_before', 'disparate_impact_ratio_after', 'dir_change',
    'statistical_parity_difference_before', 'statistical_parity_difference_after', 'spd_change',
]].to_string(index=False))

# %%
output_path = f'{OUTPUT_DIR}/train_dir_repaired_lambda{int(SELECTED_LAMBDA * 10):02d}.csv'
result.train_repaired.to_csv(output_path, index=False)
print(f'Repaired dataset saved to: {output_path}')
print(f'Shape: {result.train_repaired.shape}')
print(f'Columns: {result.train_repaired.columns.tolist()}')

# %% [markdown]
# ## 11. Summary
#
# | Step | Result |
# |------|--------|
# | Geographic encoding | ZIP / State / MSA replaced with Census Division (9 categories + binary Southern flag) |
# | Null columns dropped | Pre-HARP Loan Sequence Number, HARP Indicator, MI Cancellation Indicator |
# | Sentinel values | Credit Score 9999 → NaN (11 records) |
# | Redundant feature dropped | `high_elec_pct` (r = 0.993 with `avg_elec_cost`) |
# | Protected attributes | `is_southern_region`, `high_poverty_area` |
# | Output | Saved to `data/classing/train_dir_repaired_lambda{selected}.csv` |
#
# **Next steps:**
# - Feed the repaired dataset into fine/coarse classing → WOE feature engineering
# - Compare KS/AUC of models trained on original vs repaired data (Phase 2)
# - Document the DIR-induced performance delta as justification for any AUC reduction
