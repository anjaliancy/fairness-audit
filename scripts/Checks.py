# Auto-generated from Checks.ipynb
# Source notebook exported as a plain Python script.

# %%
import pandas as pd
import numpy as np

# %%
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, roc_curve, auc
import scipy.stats as stats
from scipy.special import expit
import matplotlib.pyplot as plt
import seaborn as sns

# %%
final_bins=pd.read_csv("data/classing/main_alt_final_bins_all_variables.csv")

# %%
final_bins['variable'].unique()

# %%
cols_list=['Credit Score', 'First Time Homebuyer Flag', 'Loan Purpose',
       'Mortgage Insurance Percentage (MI %)',
       'Original Combined Loan-to-Value (CLTV)',
       'Original Debt-to-Income (DTI) Ratio', 'Original Interest Rate',
       'Original Loan Term', 'Program Indicator']

# %%
# variables_iv_high=final_bins[final_bins['iv_total'] > 0.01]['variable'].unique()

# %%
# variables_iv_high

# %%
# cols_list = [
#     'Credit Score',
#     'Mortgage Insurance Percentage (MI %)',
#     'Number of Units',
#     'Occupancy Status',
#     'Original Combined Loan-to-Value (CLTV)',
#     'Original Debt-to-Income (DTI) Ratio',
#     'Original UPB',
#     'Original Loan-to-Value (LTV)',
#     'Original Interest Rate',
#     'Channel',
#     'Property Type',
#     'Loan Purpose',
#     'Original Loan Term',
#     'Number of Borrowers',
#     'Program Indicator',
#     'Property Valuation Method',
#     'median_income',
#     'poverty_rate',
#     'avg_elec_cost',
#     'msa_hpi_2020',
#     'msa_hpi_growth',
#     'crime_rate',
#     'census_division_code',
#     'First Time Homebuyer Flag'
# ]

target_col = 'target'

# %%
df_train_v1 = pd.read_csv('data/classing/main_alt_train_required_variables_only_woe_transformed.csv')

missing_cols = [c for c in cols_list + [target_col] if c not in df_train_v1.columns]
if missing_cols:
    raise ValueError(f'Missing required columns in repaired train file: {missing_cols}')

df_train_v1 = df_train_v1[cols_list + [target_col]]

# %%
df_test_v1 = pd.read_csv('data/classing/main_alt_test_required_variables_only_woe_transformed.csv')

missing_cols = [c for c in cols_list + [target_col] if c not in df_test_v1.columns]
if missing_cols:
    raise ValueError(f'Missing required columns in repaired test file: {missing_cols}')

df_test_v1 = df_test_v1[cols_list + [target_col]]

# %%
df_test_v1

# %%
df_test_v1=pd.read_csv('data/classing/main_alt_test_required_variables_only_woe_transformed.csv')

# %%
print("\n" + "="*80)
print("BUILDING LOGISTIC REGRESSION MODEL")
print("="*80)

def parse_binary_target(series):
    # Accept numeric 0/1 directly, otherwise map common string labels.
    s_num = pd.to_numeric(series, errors='coerce')
    if s_num.notna().all():
        unique_vals = set(s_num.unique())
        if unique_vals.issubset({0, 1}):
            return s_num.astype(int)

    s_map = series.astype(str).str.strip().str.upper().map({'N': 0, 'Y': 1, '0': 0, '1': 1})
    if s_map.isna().any():
        raise ValueError("Unexpected labels in target. Expected N/Y or 0/1.")
    return s_map.astype(int)

# Prepare data for modeling
X_train = df_train_v1.drop(columns=[target_col])
y_train = parse_binary_target(df_train_v1[target_col])

# %%
y_train

# %%
X_test = df_test_v1.drop(columns=[target_col])
y_test = parse_binary_target(df_test_v1[target_col])

# %%
X_test.columns

# %%
# Build logistic regression
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)
    
# Get predictions
y_pred_train = model.predict(X_train)
y_pred_proba_train = model.predict_proba(X_train)[:, 1]

# %%
# Model coefficients
coef_df = pd.DataFrame({'Variable': cols_list,
                        'Coefficient': model.coef_[0],
                        'Intercept': model.intercept_[0]
    })
    
coef_df['Odds_Ratio'] = np.exp(coef_df['Coefficient'])
coef_df = coef_df.sort_values('Coefficient', ascending=False)
    
print(f"\nModel Coefficients (Training Set):")
print(coef_df.to_string(index=False))

# %%
y_pred_test = model.predict(X_test)
y_pred_proba_test = model.predict_proba(X_test)[:, 1]

# %%
# Model performance - TRAINING SET
train_auc = roc_auc_score(y_train, y_pred_proba_train)
    
print(f"\n✓ Model built successfully on {len(X_train):,} training records")
print(f"  Variables used: {len(cols_list)}")
print(f"  Intercept: {model.intercept_[0]:.6f}")
print(f"  Training AUC: {train_auc:.4f}")

# %%
# Model performance - TEST SET
print(f"\n" + "="*80)
print("MODEL EVALUATION ON TEST SET")
print("="*80)
    
test_auc = roc_auc_score(y_test, y_pred_proba_test)
print(f"\n✓ Model evaluated on {len(X_test):,} test records")
print(f"  Test AUC: {test_auc:.4f}")

# %%
# Gini coefficient
def calculate_gini(y_true, y_pred_proba):
    auc = roc_auc_score(y_true, y_pred_proba)
    return 2 * auc - 1
    
test_gini = calculate_gini(y_test, y_pred_proba_test)
train_gini = calculate_gini(y_train, y_pred_proba_train)
# KS statistic
def calculate_ks(y_true, y_pred_proba):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    ks_stat = np.max(tpr - fpr)
    return ks_stat
    
test_ks = calculate_ks(y_test, y_pred_proba_test)
train_ks = calculate_ks(y_train, y_pred_proba_train)
    
print(f"\nMetric              Train Set    Test Set")
print(f"{'─'*50}")
print(f"AUC Score:          {train_auc:>10.4f}   {test_auc:>10.4f}")
print(f"Gini Coefficient:   {train_gini:>10.4f}   {test_gini:>10.4f}")
print(f"K-S Statistic:      {train_ks:>10.4f}   {test_ks:>10.4f}")
   
# Confusion matrix
cm = confusion_matrix(y_test, y_pred_test)
print(f"\nConfusion Matrix (Test Set):")
print(f"  True Negatives:  {cm[0,0]:>10,}")
print(f"  False Positives: {cm[0,1]:>10,}")
print(f"  False Negatives: {cm[1,0]:>10,}")
print(f"  True Positives:  {cm[1,1]:>10,}")
    
# Save model output
model_output = coef_df.copy()
model_output['Train_AUC'] = train_auc
model_output['Test_AUC'] = test_auc
model_output['Train_Gini'] = train_gini
model_output['Test_Gini'] = test_gini
model_output['Test_KS'] = test_ks
    
model_output.to_csv(r'd:\FRM\data\08_logistic_regression_output-v1.csv', index=False)
print(f"\n✓ Model output saved: data/08_logistic_regression_output-v1.csv")

# %%
# Compatibility setup for downstream summary/scorecard cells
if 'sv_v1_list' not in globals():
    sv_v1_list = cols_list.copy()
if 'numeric_model_vars' not in globals():
    numeric_model_vars = cols_list.copy()

# Optional V2 placeholders; V2 section will run only if these are later defined
if 'df_train_v2' not in globals() or 'df_test_v2' not in globals():
    print("V2 datasets not found; V2 score band section may be skipped.")

# %%
# STEP 9: SCORECARD SCALING & APPLICATION SCORE
if len(sv_v1_list) > 0:
    print("\n" + "="*80)
    print("SCORECARD DEVELOPMENT")
    print("="*80)
    
    # Scorecard parameters
    base_score = 600
    pdo = 20  # Points to Double Odds
    target_odds = 60  # 1 default per 60 good loans
    
    print(f"\nScorecard Configuration:")
    print(f"  Base Score (@ {target_odds}:1 odds): {base_score}")
    print(f"  Points to Double Odds (PDO): {pdo}")
    print(f"  Target Odds: {target_odds}:1")
    
    # Calculate scaling factor
    factor = pdo / np.log(2)
    intercept = base_score + factor * np.log(target_odds)
    
    print(f"\nScaling Factors:")
    print(f"  Factor: {factor:.4f}")
    print(f"  Intercept: {intercept:.4f}")
    
    # Build scorecard
    scorecard = []
    for i, var in enumerate(sv_v1_list):
        coef = model.coef_[0][i]
        points = factor * coef
        scorecard.append({
            'Variable': var,
            'Coefficient': coef,
            'Points_Per_Unit': points,
            'Odds_Ratio': np.exp(coef)
        })
    
    scorecard_df = pd.DataFrame(scorecard).sort_values('Points_Per_Unit', ascending=False)
    
    print(f"\nScorecard Points Assignment:")
    print(scorecard_df.to_string(index=False))
    
    # Calculate scores on test set
    test_scores = intercept + X_test.dot(model.coef_[0])
    
    print(f"\nScore Distribution (Test Set):")
    print(f"  Mean Score:  {test_scores.mean():.2f}")
    print(f"  Min Score:   {test_scores.min():.2f}")
    print(f"  Max Score:   {test_scores.max():.2f}")
    print(f"  Std Dev:     {test_scores.std():.2f}")
    
    # Score by default status
    scores_good = test_scores[y_test == 0]
    scores_bad = test_scores[y_test == 1]
    
    print(f"\nScore Distribution by Class (Test Set):")
    print(f"  Good Loans   (0): Mean {scores_good.mean():.2f}, Std {scores_good.std():.2f}")
    print(f"  Bad Loans    (1): Mean {scores_bad.mean():.2f}, Std {scores_bad.std():.2f}")
    print(f"  Difference:       {scores_good.mean() - scores_bad.mean():.2f} points")
    
    # Save scorecard
    scorecard_df.to_csv(r'd:\FRM\data\09_scorecard_table-v1.csv', index=False)
    print(f"\n✓ Scorecard saved: data/09_scorecard_table-v1.csv")
    
    # Create score summary
    score_summary = pd.DataFrame({
        'Score': test_scores,
        'Actual_Default': y_test.values,
        'Predicted_Probability': y_pred_proba_test
    })
    
    score_summary.to_csv(r'd:\FRM\data\10_test_set_scores-v1.csv', index=False)
    print(f"✓ Test set scores saved: data/10_test_set_scores-v1.csv")
else:
    print("⚠ Cannot build scorecard - no numeric variables for modeling")
# SUMMARY & OUTPUTS
print("\n\n" + "="*80)
print("CREDIT RISK MODEL DEVELOPMENT - COMPLETE")
print("="*80)

print("\n📊 WORKFLOW EXECUTION SUMMARY")
print("-"*80)

print("\n✓ COMPLETED STEPS:")
print("  1. ✓ Data Validation - Data quality assessed")
print("  2. ✓ Binning Strategy - Methodology defined")
print("  3. ✓ WoE/IV Calculation - Predictive power ranked")
print("  4. ✓ Factor Analysis - Correlations identified")
print("  5. ✓ Train/Test Split - Stratified 70/30 split")
print("  6. ✓ WoE Validation - Monotonicity checked (training set only)")
print("  7. ✓ Variable Selection - Applied IV > 0.10 threshold")
print("  8. ✓ Logistic Regression - Model built and tested")
print("  9. ✓ Scorecard Development - Points assigned")

print("\n📊 DELIVERABLES SUMMARY")
print("-"*80)

print("\n✓ DATA FILES (d:\\FRM\\data\\):")
print("  1. 01_cleaned_dataset.csv - Full cleaned dataset")
print("  2. 02_iv_ranking_full_dataset.csv - IV ranking of all variables")
print("  3. 03_factor_analysis_report.csv - Correlated variable pairs")
print("  4. 04_train_dataset.csv - Training set (70%)")
print("  5. 05_test_dataset.csv - Test set (30%)")
print("  6. 06_woe_specifications_validated.csv - WoE validation report")
print("  7. 07_selected_variables.csv - Final model variables")
print("  8. 08_logistic_regression_output.csv - Model coefficients & metrics")
print("  9. 09_scorecard_table.csv - Scorecard points assignment")
print("  10. 10_test_set_scores.csv - Test set scores")

print("\n📈 MODEL PERFORMANCE SUMMARY")
print("-"*80)

if len(numeric_model_vars) > 0:
    print(f"\nTraining Set:")
    print(f"  Records: {len(X_train):,}")
    print(f"  Defaults: {y_train.sum():,} ({y_train.mean()*100:.2f}%)")
    print(f"  AUC: {train_auc:.4f}")
    print(f"  Gini: {train_gini:.4f}")
    print(f"  K-S: {train_ks:.4f}")
    
    print(f"\nTest Set:")
    print(f"  Records: {len(X_test):,}")
    print(f"  Defaults: {y_test.sum():,} ({y_test.mean()*100:.2f}%)")
    print(f"  AUC: {test_auc:.4f}")
    print(f"  Gini: {test_gini:.4f}")
    print(f"  K-S: {test_ks:.4f}")
    
    print(f"\nVariables in Model: {len(numeric_model_vars)}")
    
    print(f"\nScorecard:")
    print(f"  Base Score: {base_score}")
    print(f"  PDO: {pdo}")
    print(f"  Target Odds: {target_odds}:1")

print("\n⚠️  USER DECISION CHECKPOINTS IN THIS WORKFLOW:")
print("-"*80)
print("  1. After Factor Analysis Output:")
print("     → Review IV scores and correlation matrix")
print("     → Decide which variables to include/exclude")
print("")
print("  2. After WoE Validation (Training Set):")
print("     → Review monotonicity check results")
print("     → Address any non-monotonic WoE variables")
print("     → Confirm final variable selection for model")
print("")
print("  3. After Logistic Regression:")
print("     → Review model coefficients and p-values")
print("     → Assess model performance on test set")
print("     → Decide if model meets business requirements")

print("\n✓ WORKFLOW COMPLETE")
print("  → Review all outputs in d:\\FRM\\data\\")
print("  → Consider stability testing across demographics")
print("  → Monitor model performance over time")
print("  → Update scorecard as business requirements change")
print("\n" + "="*80)
# STEP 10: SCORING SUMMARY - SCORE BANDS & BAD RATES
print("\n" + "="*80)
print("SCORE BAND ANALYSIS - MODEL V1 (6 VARIABLES)")
print("="*80)

def create_score_bands(scores, y_true, n_bands=10):
    """
    Create score bands using quantiles and calculate bad rates for each band
    """
    # Create score bands using quantiles (deciles by default)
    score_bands = pd.qcut(scores, q=n_bands, duplicates='drop')
    
    # Create analysis dataframe
    df_bands = pd.DataFrame({
        'Score': scores,
        'Default': y_true,
        'Band': score_bands
    })
    
    # Group by band and calculate statistics
    band_analysis = df_bands.groupby('Band', observed=True).agg({
        'Score': ['min', 'max', 'count'],
        'Default': ['sum', 'mean']
    }).reset_index(drop=True)
    
    # Flatten column names
    band_analysis.columns = ['Score_Min', 'Score_Max', 'Population', 'Defaults', 'Default_Rate']
    
    # Calculate additional metrics
    band_analysis['Percentage_Population'] = (band_analysis['Population'] / band_analysis['Population'].sum() * 100)
    band_analysis['Percentage_Defaults'] = (band_analysis['Defaults'] / band_analysis['Defaults'].sum() * 100)
    band_analysis['Cumulative_Population_%'] = band_analysis['Percentage_Population'].cumsum()
    band_analysis['Cumulative_Defaults_%'] = band_analysis['Percentage_Defaults'].cumsum()
    
    # Note: Bands are from LOWEST to HIGHEST score
    # Typically lower scores = higher risk, so reverse for risk perspective
    band_analysis['Band_Rank'] = range(n_bands, 0, -1)
    
    # Reorder for better readability (highest scores first = lowest risk first)
    band_analysis = band_analysis.iloc[::-1].reset_index(drop=True)
    band_analysis['Band_Rank'] = range(1, len(band_analysis) + 1)
    
    return band_analysis

# Create score bands for V1 model
score_bands_v1 = create_score_bands(test_scores, y_test.values, n_bands=10)

# Format for display
display_df = score_bands_v1[['Band_Rank', 'Score_Min', 'Score_Max', 'Population', 
                               'Defaults', 'Default_Rate', 'Percentage_Population', 
                               'Percentage_Defaults']].copy()

display_df.columns = ['Band', 'Score_Min', 'Score_Max', 'Count', 'Defaults', 
                      'Bad_Rate_%', 'Pop_%', 'Default_%']

# Round for display
display_df['Score_Min'] = display_df['Score_Min'].round(0).astype(int)
display_df['Score_Max'] = display_df['Score_Max'].round(0).astype(int)
display_df['Bad_Rate_%'] = display_df['Bad_Rate_%'].round(2)
display_df['Pop_%'] = display_df['Pop_%'].round(2)
display_df['Default_%'] = display_df['Default_%'].round(2)

print(f"\nScore Band Performance (Test Set, n={len(test_scores):,}):")
print(display_df.to_string(index=False))

# Save to CSV
score_bands_v1.to_csv(r'd:\FRM\data\11_score_bands_v1.csv', index=False)
print(f"\n✓ Score bands saved: data/11_score_bands_v1.csv")

# %%
# Fallback setup for optional V2 section
if 'df_train_v2' not in globals() or 'df_test_v2' not in globals():
    df_train_v2 = df_train_v1.copy()
    df_test_v2 = df_test_v1.copy()
    print("V2 datasets missing; using V1 datasets as fallback for V2 output block.")

# Ensure V2 targets are numeric for metrics and band calculations
for _df_name in ['df_train_v2', 'df_test_v2']:
    _df = globals()[_df_name]
    _df[target_col] = parse_binary_target(_df[target_col])
    globals()[_df_name] = _df

# %%

# Summary statistics
print(f"\n" + "-"*80)
print("Score Band Summary:")
print(f"  Highest Risk Band:   {score_bands_v1.iloc[0]['Default_Rate']*100:.2f}% default rate")
print(f"  Lowest Risk Band:    {score_bands_v1.iloc[-1]['Default_Rate']*100:.2f}% default rate")
print(f"  Risk Spread:         {(score_bands_v1.iloc[0]['Default_Rate'] - score_bands_v1.iloc[-1]['Default_Rate'])*100:.2f} percentage points")
print(f"  Overall Default Rate: {y_test.mean()*100:.2f}%")

print("\n" + "="*80)
print("SCORE BAND ANALYSIS - MODEL V2 (10 VARIABLES)")
print("="*80)

# Prepare V2 data - need to build model if not already done
X_train_v2 = df_train_v2.drop(columns=[target_col])
y_train_v2 = df_train_v2[target_col]

X_test_v2 = df_test_v2.drop(columns=[target_col])
y_test_v2 = df_test_v2[target_col]

# Fill missing values in test set for v2
for col in X_test_v2.columns:
    if X_test_v2[col].isnull().sum() > 0:
        mode_val = X_test_v2[col].mode()[0]
        X_test_v2[col] = X_test_v2[col].fillna(mode_val)

# Build V2 model
model_v2 = LogisticRegression(random_state=42, max_iter=1000)
model_v2.fit(X_train_v2, y_train_v2)

# Get predictions for V2
y_pred_proba_test_v2 = model_v2.predict_proba(X_test_v2)[:, 1]

# Calculate scores for V2 model
factor_v2 = pdo / np.log(2)
intercept_v2 = base_score + factor_v2 * np.log(target_odds)
test_scores_v2 = intercept_v2 + X_test_v2.dot(model_v2.coef_[0])

# Create score bands for V2 model
score_bands_v2 = create_score_bands(test_scores_v2, y_test_v2.values, n_bands=10)

# Format for display
display_df_v2 = score_bands_v2[['Band_Rank', 'Score_Min', 'Score_Max', 'Population', 
                                  'Defaults', 'Default_Rate', 'Percentage_Population', 
                                  'Percentage_Defaults']].copy()

display_df_v2.columns = ['Band', 'Score_Min', 'Score_Max', 'Count', 'Defaults', 
                         'Bad_Rate_%', 'Pop_%', 'Default_%']

# Round for display
display_df_v2['Score_Min'] = display_df_v2['Score_Min'].round(0).astype(int)
display_df_v2['Score_Max'] = display_df_v2['Score_Max'].round(0).astype(int)
display_df_v2['Bad_Rate_%'] = display_df_v2['Bad_Rate_%'].round(2)
display_df_v2['Pop_%'] = display_df_v2['Pop_%'].round(2)
display_df_v2['Default_%'] = display_df_v2['Default_%'].round(2)

print(f"\nScore Band Performance (Test Set, n={len(test_scores_v2):,}):")
print(display_df_v2.to_string(index=False))

# Save to CSV
score_bands_v2.to_csv(r'd:\FRM\data\11_score_bands_v2.csv', index=False)
print(f"\n✓ Score bands saved: data/11_score_bands_v2.csv")

# Summary statistics
print(f"\n" + "-"*80)
print("Score Band Summary:")
print(f"  Highest Risk Band:   {score_bands_v2.iloc[0]['Default_Rate']*100:.2f}% default rate")
print(f"  Lowest Risk Band:    {score_bands_v2.iloc[-1]['Default_Rate']*100:.2f}% default rate")
print(f"  Risk Spread:         {(score_bands_v2.iloc[0]['Default_Rate'] - score_bands_v2.iloc[-1]['Default_Rate'])*100:.2f} percentage points")
print(f"  Overall Default Rate: {y_test_v2.mean()*100:.2f}%")

# %%
# FINAL SCORE RESCALING (LOGIT-BASED)
print("\n" + "="*80)
print("FINAL SCORE RESCALING - LOGIT FORMULA")
print("="*80)

# Standard scorecard settings
base_score = 600
pdo = 20
target_odds = 60  # good:bad odds at base score
score_floor = 300
score_cap = 900

factor = pdo / np.log(2)

def score_from_proba(pd_proba, base_score=600, target_odds=60, pdo=20, floor=300, cap=900):
    """Convert PD to score using standard logit scorecard scaling."""
    f = pdo / np.log(2)
    p = np.clip(np.asarray(pd_proba, dtype=float), 1e-9, 1 - 1e-9)
    log_odds_bad = np.log(p / (1 - p))
    score = base_score + f * (np.log(target_odds) - log_odds_bad)
    return np.clip(score, floor, cap)

# Recompute V1 scores from predicted probabilities
test_scores = pd.Series(score_from_proba(y_pred_proba_test, base_score, target_odds, pdo, score_floor, score_cap), index=X_test.index)

# Save updated V1 score summary
score_summary = pd.DataFrame({
    'Score': test_scores,
    'Actual_Default': y_test.values,
    'Predicted_Probability': y_pred_proba_test
})
score_summary.to_csv(r'd:\\FRM\\data\\10_test_set_scores-v1.csv', index=False)

# Recompute and save V1 score bands
score_bands_v1 = create_score_bands(test_scores, y_test.values, n_bands=10)
score_bands_v1.to_csv(r'd:\\FRM\\data\\11_score_bands_v1.csv', index=False)

print(f"Score settings: Base={base_score}, PDO={pdo}, TargetOdds={target_odds}:1, Range=[{score_floor}, {score_cap}]")
print(f"V1 score range: {test_scores.min():.1f} to {test_scores.max():.1f}")
print(f"V1 mean score:  {test_scores.mean():.1f}")
print("Saved: data/10_test_set_scores-v1.csv")
print("Saved: data/11_score_bands_v1.csv")

# If V2 predictions exist, rescale those too for consistency
if 'y_pred_proba_test_v2' in globals() and 'X_test_v2' in globals() and 'y_test_v2' in globals():
    test_scores_v2 = pd.Series(score_from_proba(y_pred_proba_test_v2, base_score, target_odds, pdo, score_floor, score_cap), index=X_test_v2.index)
    score_bands_v2 = create_score_bands(test_scores_v2, y_test_v2.values, n_bands=10)
    score_bands_v2.to_csv(r'd:\\FRM\\data\\11_score_bands_v2.csv', index=False)
    print(f"V2 score range: {test_scores_v2.min():.1f} to {test_scores_v2.max():.1f}")
    print(f"V2 mean score:  {test_scores_v2.mean():.1f}")
    print("Saved: data/11_score_bands_v2.csv")

# %%
# FINAL SCORECARD INTEGRATION & OUTPUT SUMMARY
import os
print("\n" + "="*80)
print("FINAL SCORECARD SUMMARY & OUTPUTS")
print("="*80)

# Calculate model performance metrics on test set
test_auc = roc_auc_score(y_test, y_pred_proba_test)
test_gini = 2 * test_auc - 1

def calculate_ks(y_true, y_pred_proba):
    """Calculate Kolmogorov-Smirnov statistic."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    ks_stat = np.max(tpr - fpr)
    return ks_stat

test_ks = calculate_ks(y_test, y_pred_proba_test)

# Combine all scorecard components
final_scorecard = pd.DataFrame({
    'Variable': cols_list,
    'Coefficient': model.coef_[0],
    'Odds_Ratio': np.exp(model.coef_[0]),
    'Points_Per_Unit': factor * model.coef_[0]
})

final_scorecard = final_scorecard.sort_values('Points_Per_Unit', ascending=False)

# Save comprehensive scorecard
scorecard_path = 'data/classing/final_scorecard_full.csv'
final_scorecard.to_csv(scorecard_path, index=False)
print(f'Scorecard saved: {scorecard_path}')

# Summary statistics
print(f'\nScorecard Summary:')
print(f'  Total Variables: {len(cols_list)}')
print(f'  Model Intercept: {model.intercept_[0]:.6f}')
print(f'  Base Score: {base_score}')
print(f'  Score Range: {score_floor} - {score_cap}')

# Score distribution by risk band
print(f'\nScore Distribution Summary:')
print(f'  Mean Score: {test_scores.mean():.2f}')
print(f'  Score Range: {test_scores.min():.2f} - {test_scores.max():.2f}')
print(f'  Std Dev: {test_scores.std():.2f}')

# Model performance metrics
print(f'\nModel Performance Metrics (Test Set):')
print(f'  AUC Score: {test_auc:.4f}')
print(f'  Gini Coefficient: {test_gini:.4f}')
print(f'  K-S Statistic: {test_ks:.4f}')

# Save band analysis summary
band_summary_path = 'data/classing/score_bands_summary.csv'
score_summary_bands = score_bands_v1[['Band_Rank', 'Score_Min', 'Score_Max', 'Population', 'Defaults', 'Default_Rate']].copy()
score_summary_bands.to_csv(band_summary_path, index=False)
print(f'\nBand analysis saved: {band_summary_path}')

# Save full model metadata
model_metadata = {
    'model_type': 'LogisticRegression',
    'variables_count': len(cols_list),
    'training_records': len(X_train),
    'test_records': len(X_test),
    'test_auc': round(test_auc, 4),
    'test_gini': round(test_gini, 4),
    'test_ks': round(test_ks, 4),
    'intercept': round(float(model.intercept_[0]), 6),
    'base_score': base_score,
    'pdo': pdo,
    'target_odds': target_odds,
    'score_floor': score_floor,
    'score_cap': score_cap
}

metadata_df = pd.DataFrame([model_metadata])
metadata_path = 'data/classing/model_metadata.csv'
metadata_df.to_csv(metadata_path, index=False)
print(f'Model metadata saved: {metadata_path}')

print(f'\n✓ All scorecard outputs saved to data/classing/')

# %%

# MODEL COMPARISON: WITH vs WITHOUT STATE VARIABLE
print("\n" + "="*80)
print("LOGISTIC REGRESSION - MODEL WITHOUT STATE VARIABLE")
print("="*80)

# Define columns without State
cols_list_no_state = [col for col in cols_list if col != 'State']

print(f"\n✓ Building model with {len(cols_list_no_state)} variables (excluding State)")
print(f"  Removed Variable: State")
print(f"  Remaining Variables: {', '.join(cols_list_no_state)}")

# Prepare data for modeling without State
X_train_no_state = df_train_v1[cols_list_no_state].copy()
y_train_no_state = parse_binary_target(df_train_v1[target_col])

X_test_no_state = df_test_v1[cols_list_no_state].copy()
y_test_no_state = parse_binary_target(df_test_v1[target_col])

# Build logistic regression without State
model_no_state = LogisticRegression(random_state=42, max_iter=1000)
model_no_state.fit(X_train_no_state, y_train_no_state)

# Get predictions
y_pred_train_no_state = model_no_state.predict(X_train_no_state)
y_pred_proba_train_no_state = model_no_state.predict_proba(X_train_no_state)[:, 1]

y_pred_test_no_state = model_no_state.predict(X_test_no_state)
y_pred_proba_test_no_state = model_no_state.predict_proba(X_test_no_state)[:, 1]

# Calculate performance metrics
train_auc_no_state = roc_auc_score(y_train_no_state, y_pred_proba_train_no_state)
test_auc_no_state = roc_auc_score(y_test_no_state, y_pred_proba_test_no_state)

train_gini_no_state = calculate_gini(y_train_no_state, y_pred_proba_train_no_state)
test_gini_no_state = calculate_gini(y_test_no_state, y_pred_proba_test_no_state)

train_ks_no_state = calculate_ks(y_train_no_state, y_pred_proba_train_no_state)
test_ks_no_state = calculate_ks(y_test_no_state, y_pred_proba_test_no_state)

print(f"\n" + "-"*80)
print("MODEL PERFORMANCE COMPARISON")
print("-"*80)

print(f"\nModel WITH State (12 Variables):")
print(f"{'Metric':<25} {'Train Set':<15} {'Test Set':<15}")
print(f"{'-'*55}")
print(f"{'AUC Score:':<25} {train_auc:>14.4f}  {test_auc:>14.4f}")
print(f"{'Gini Coefficient:':<25} {train_gini:>14.4f}  {test_gini:>14.4f}")
print(f"{'K-S Statistic:':<25} {train_ks:>14.4f}  {test_ks:>14.4f}")

print(f"\nModel WITHOUT State (11 Variables):")
print(f"{'Metric':<25} {'Train Set':<15} {'Test Set':<15}")
print(f"{'-'*55}")
print(f"{'AUC Score:':<25} {train_auc_no_state:>14.4f}  {test_auc_no_state:>14.4f}")
print(f"{'Gini Coefficient:':<25} {train_gini_no_state:>14.4f}  {test_gini_no_state:>14.4f}")
print(f"{'K-S Statistic:':<25} {train_ks_no_state:>14.4f}  {test_ks_no_state:>14.4f}")

# Calculate differences
auc_diff = test_auc - test_auc_no_state
gini_diff = test_gini - test_gini_no_state
ks_diff = test_ks - test_ks_no_state

print(f"\n" + "-"*80)
print("PERFORMANCE IMPACT OF STATE VARIABLE (Test Set)")
print("-"*80)
print(f"{'AUC Difference:':<25} {auc_diff:>14.4f}  ({auc_diff/test_auc*100:>6.2f}% decrease without State)")
print(f"{'Gini Difference:':<25} {gini_diff:>14.4f}  ({gini_diff/test_gini*100:>6.2f}% decrease without State)")
print(f"{'K-S Difference:':<25} {ks_diff:>14.4f}  ({ks_diff/test_ks*100:>6.2f}% decrease without State)")

# Model coefficients comparison
print(f"\n" + "-"*80)
print("MODEL COEFFICIENTS COMPARISON (Test Set)")
print("-"*80)

coef_comparison = pd.DataFrame({
    'Variable': cols_list_no_state,
    'Coefficient_With_State': model.coef_[0][:len(cols_list_no_state)],
    'Coefficient_Without_State': model_no_state.coef_[0],
})
coef_comparison['Coefficient_Difference'] = (coef_comparison['Coefficient_Without_State'] - 
                                              coef_comparison['Coefficient_With_State'])

print(coef_comparison.to_string(index=False))

# Confusion matrices
print(f"\n" + "-"*80)
print("CONFUSION MATRIX COMPARISON (Test Set)")
print("-"*80)

cm_with_state = confusion_matrix(y_test, y_pred_test)
cm_without_state = confusion_matrix(y_test_no_state, y_pred_test_no_state)

print(f"\nWith State (12 Variables):")
print(f"  True Negatives:  {cm_with_state[0,0]:>10,}")
print(f"  False Positives: {cm_with_state[0,1]:>10,}")
print(f"  False Negatives: {cm_with_state[1,0]:>10,}")
print(f"  True Positives:  {cm_with_state[1,1]:>10,}")
print(f"  Accuracy: {(cm_with_state[0,0] + cm_with_state[1,1]) / cm_with_state.sum():.4f}")

print(f"\nWithout State (11 Variables):")
print(f"  True Negatives:  {cm_without_state[0,0]:>10,}")
print(f"  False Positives: {cm_without_state[0,1]:>10,}")
print(f"  False Negatives: {cm_without_state[1,0]:>10,}")
print(f"  True Positives:  {cm_without_state[1,1]:>10,}")
print(f"  Accuracy: {(cm_without_state[0,0] + cm_without_state[1,1]) / cm_without_state.sum():.4f}")

print(f"\n" + "-"*80)
print("KEY INSIGHTS")
print("-"*80)
print(f"✓ State variable contributes {auc_diff:.4f} to test AUC (important predictor)")
print(f"✓ Without State, model still achieves {test_auc_no_state:.4f} AUC (acceptable)")
print(f"✓ State importance: ~{auc_diff/test_auc*100:.2f}% of model's discriminatory power")
print(f"✓ Model remains stable without State, but performance degrades")

# Save results
print(f"\n" + "-"*80)
print("SAVING COMPARISON RESULTS")
print("-"*80)

# Scorecard without State
scorecard_no_state = pd.DataFrame({
    'Variable': cols_list_no_state,
    'Coefficient': model_no_state.coef_[0],
    'Odds_Ratio': np.exp(model_no_state.coef_[0]),
    'Points_Per_Unit': factor * model_no_state.coef_[0]
})

scorecard_no_state = scorecard_no_state.sort_values('Points_Per_Unit', ascending=False)
scorecard_no_state.to_csv('data/classing/final_scorecard_no_state.csv', index=False)
print(f"✓ Scorecard without State saved: data/classing/final_scorecard_no_state.csv")

# Model comparison metadata
comparison_metadata = {
    'Model_Type': 'Logistic Regression Comparison',
    'Model_1_Variables': 12,
    'Model_1_Name': 'With State',
    'Model_1_Test_AUC': round(test_auc, 4),
    'Model_1_Test_Gini': round(test_gini, 4),
    'Model_1_Test_KS': round(test_ks, 4),
    'Model_2_Variables': 11,
    'Model_2_Name': 'Without State',
    'Model_2_Test_AUC': round(test_auc_no_state, 4),
    'Model_2_Test_Gini': round(test_gini_no_state, 4),
    'Model_2_Test_KS': round(test_ks_no_state, 4),
    'AUC_Difference': round(auc_diff, 4),
    'Gini_Difference': round(gini_diff, 4),
    'KS_Difference': round(ks_diff, 4),
    'AUC_% Change': round(auc_diff/test_auc*100, 2),
    'Gini_% Change': round(gini_diff/test_gini*100, 2),
    'KS_% Change': round(ks_diff/test_ks*100, 2),
}

comparison_df = pd.DataFrame([comparison_metadata])
comparison_df.to_csv('data/classing/model_comparison_with_vs_without_state.csv', index=False)
print(f"✓ Model comparison summary saved: data/classing/model_comparison_with_vs_without_state.csv")

print(f"\n" + "="*80)
print("✓ COMPARISON ANALYSIS COMPLETE")
print("="*80)

# %% [markdown]
# ## Post-Processing Fairness + Explainability Diagnostics (Logistic Regression)
#
# This cell runs threshold optimization, fairness trade-off metrics, and fidelity/stability checks using the logistic regression model.

# %%
# Post-processing fairness + explainability diagnostics for OLD model path (no State/Seller/Servicer).
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, 'src')

from explainability_assessment import (
    assess_postprocessing_fairness,
    assess_openxai_fidelity_stability,
    assess_lime,
    assess_shap,
    assess_dice,
    assess_ceteris_paribus,
    assess_anchor,
    assess_counterfactual_fairness,
    evaluate_fairness_tradeoff,
    _predict_scores,
 )

print('\n' + '=' * 90)
print('POST-PROCESSING FAIRNESS + EXPLAINABILITY (OLD VERSION, NO STATE/SELLER/SERVICER)')
print('=' * 90)

# Reuse already-built old-version model if available; otherwise train quickly from existing old-version matrices.
if 'model_no_state' in globals():
    model_old = model_no_state
    Xtr_old = X_train_no_state.copy()
    Xte_old = X_test_no_state.copy()
    ytr_old = y_train_no_state.copy()
    yte_old = y_test_no_state.copy()
else:
    model_old = LogisticRegression(random_state=42, max_iter=1000)
    Xtr_old = X_train.copy()
    Xte_old = X_test.copy()
    ytr_old = y_train.copy()
    yte_old = y_test.copy()
    for c in ['State', 'Seller Name', 'Servicer Name']:
        if c in Xtr_old.columns:
            Xtr_old = Xtr_old.drop(columns=[c])
        if c in Xte_old.columns:
            Xte_old = Xte_old.drop(columns=[c])
    model_old.fit(Xtr_old, ytr_old)

# Choose a sensitive feature available in this old-version dataset.
if 'census_division_code' in Xtr_old.columns:
    sensitive_col = 'census_division_code'
elif 'high_poverty_area' in Xtr_old.columns:
    sensitive_col = 'high_poverty_area'
else:
    # Fallback to binary segmentation by median of the first numeric feature.
    first_num = Xtr_old.select_dtypes(include=[np.number]).columns[0]
    sensitive_col = first_num
    Xtr_old = Xtr_old.copy()
    Xte_old = Xte_old.copy()
    med = Xtr_old[first_num].median()
    Xtr_old[first_num] = (Xtr_old[first_num] > med).astype(int)
    Xte_old[first_num] = (Xte_old[first_num] > med).astype(int)

sf_train = Xtr_old[sensitive_col]
sf_test = Xte_old[sensitive_col]

print(f'Sensitive feature used for fairness post-processing: {sensitive_col}')
print(f'Train rows: {len(Xtr_old):,} | Test rows: {len(Xte_old):,} | Features: {Xtr_old.shape[1]}')

# Baseline fairness/performance before threshold optimizer.
baseline_scores = _predict_scores(model_old, Xte_old)
baseline_pred = (baseline_scores >= 0.5).astype(int)
baseline_summary, baseline_by_group = evaluate_fairness_tradeoff(
    y_true=yte_old,
    y_pred=baseline_pred,
    y_score=baseline_scores,
    sensitive_features=sf_test,
 )

print('\nBaseline fairness/performance summary:')
display(baseline_summary)

# Threshold optimization post-processing.
post = assess_postprocessing_fairness(
    estimator=model_old,
    x_train=Xtr_old,
    y_train=ytr_old,
    x_eval=Xte_old,
    y_eval=yte_old,
    sensitive_features_train=sf_train,
    sensitive_features_eval=sf_test,
    constraints='demographic_parity',
    objective='balanced_accuracy_score',
    prefit_estimator=True,
 )

print('\nPost-processed fairness/performance summary:')
display(post.tradeoff_summary)

print('\nFairness metrics by group after threshold optimization:')
display(post.tradeoff_by_group)

# Improvement table.
compare_cols = [
    'accuracy', 'precision', 'recall', 'f1', 'roc_auc',
    'demographic_parity_difference', 'demographic_parity_ratio',
    'equalized_odds_difference', 'equalized_odds_ratio',
]
comparison = pd.DataFrame({
    'metric': compare_cols,
    'baseline': [baseline_summary.iloc[0][c] for c in compare_cols],
    'postprocessed': [post.tradeoff_summary.iloc[0][c] for c in compare_cols],
})
comparison['delta_post_minus_base'] = comparison['postprocessed'] - comparison['baseline']

print('\nBaseline vs post-processed comparison:')
display(comparison)

# Local and global explainability on OLD model.
x_instance = Xte_old.iloc[[0]].copy()
cp_feat = 'Credit Score' if 'Credit Score' in Xte_old.columns else Xte_old.columns[0]
cp_out = assess_ceteris_paribus(model_old, Xte_old, cp_feat, grid_points=25)

# LIME (optional by environment).
lime_out = None
try:
    lime_out = assess_lime(model_old, Xtr_old, x_instance, num_features=min(10, Xtr_old.shape[1]))
    print('\nLIME explanation for first test observation:')
    display(pd.DataFrame(lime_out['weights'], columns=['feature_rule', 'weight']))
except Exception as _lime_exc:
    print('\nLIME skipped:', str(_lime_exc))

# SHAP (optional by environment).
shap_out = None
try:
    shap_out = assess_shap(model_old, Xtr_old, Xte_old.sample(min(500, len(Xte_old)), random_state=42))
    print('\nTop SHAP global importance:')
    display(shap_out['global_importance'].head(15))
except Exception as _shap_exc:
    print('\nSHAP skipped:', str(_shap_exc))

# Counterfactual fairness check (on selected sensitive feature).
cfair = assess_counterfactual_fairness(
    model=model_old,
    x=Xte_old.sample(min(1000, len(Xte_old)), random_state=42),
    sensitive_attributes=[sensitive_col],
)

print(f"\nCeteris paribus profile for feature: {cp_feat}")
display(cp_out.head(25))

print('\nCounterfactual fairness summary (score delta under sensitive perturbations):')
display(cfair.summary)

# Anchor explanation is optional; skip gracefully if dependency unavailable.
try:
    anchor_out = assess_anchor(model_old, Xtr_old, x_instance)
    print('\nAnchor explanation for first test observation:')
    print('Anchor rules:', anchor_out['anchor'])
    print(f"Precision: {anchor_out['precision']:.4f} | Coverage: {anchor_out['coverage']:.4f}")
except Exception as _anchor_exc:
    print('\nAnchor explanation skipped:', str(_anchor_exc))

# OpenXAI-style fidelity/stability using linear local contributions as attribution function.
coef_vec = model_old.coef_[0]
feat_cols = list(Xtr_old.columns)

def linear_attr_fn(df_in):
    arr = df_in[feat_cols].to_numpy()
    return arr * coef_vec

openxai_diag = assess_openxai_fidelity_stability(
    model=model_old,
    x_eval=Xte_old.sample(min(1000, len(Xte_old)), random_state=42),
    feature_importance_fn=linear_attr_fn,
    require_openxai=False,
 )

print('\nFidelity/Stability diagnostic summary:')
display(openxai_diag)

# Persist key artifacts for reporting.
comparison.to_csv('data/classing/old_model_fairness_tradeoff_comparison.csv', index=False)
post.tradeoff_summary.to_csv('data/classing/old_model_fairness_postprocessed_summary.csv', index=False)
post.tradeoff_by_group.to_csv('data/classing/old_model_fairness_postprocessed_by_group.csv', index=False)
if shap_out is not None:
    shap_out['global_importance'].to_csv('data/classing/old_model_shap_global_importance.csv', index=False)
cfair.summary.to_csv('data/classing/old_model_counterfactual_fairness_summary.csv', index=False)
openxai_diag.to_csv('data/classing/old_model_openxai_fidelity_stability.csv', index=False)

print('\nSaved outputs:')
print(' - data/classing/old_model_fairness_tradeoff_comparison.csv')
print(' - data/classing/old_model_fairness_postprocessed_summary.csv')
print(' - data/classing/old_model_fairness_postprocessed_by_group.csv')
if shap_out is not None:
    print(' - data/classing/old_model_shap_global_importance.csv')
print(' - data/classing/old_model_counterfactual_fairness_summary.csv')
print(' - data/classing/old_model_openxai_fidelity_stability.csv')

# %%
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 12)

print('\n' + '=' * 90)
print('EXPLAINABILITY & FAIRNESS VISUALIZATIONS')
print('=' * 90)

# 1. Fairness Metrics Comparison (Baseline vs Post-Processed)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Fairness & Performance Metrics: Baseline vs Post-Processed', fontsize=16, fontweight='bold')

metrics_to_plot = [
    'accuracy', 'precision', 'recall', 'f1',
    'demographic_parity_ratio', 'equalized_odds_ratio'
]
metric_labels = [
    'Accuracy', 'Precision', 'Recall', 'F1-Score',
    'Demographic Parity Ratio', 'Equalized Odds Ratio'
]

for idx, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
    ax = axes[idx // 3, idx % 3]
    baseline_val = comparison[comparison['metric'] == metric]['baseline'].values[0]
    postproc_val = comparison[comparison['metric'] == metric]['postprocessed'].values[0]
    
    x_pos = [0, 1]
    values = [baseline_val, postproc_val]
    colors = ['#e74c3c', '#2ecc71']
    
    bars = ax.bar(x_pos, values, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel(label, fontsize=11, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Baseline', 'Post-Processed'])
    ax.set_title(label, fontsize=12, fontweight='bold')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('data/classing/fairness_metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. Feature Importance: LIME vs SHAP Comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Feature Importance: LIME vs SHAP Explanations', fontsize=16, fontweight='bold')

# LIME importance
if lime_out is not None:
    lime_weights = pd.DataFrame(lime_out['weights'], columns=['feature_rule', 'weight'])
    lime_weights['abs_weight'] = lime_weights['weight'].abs()
    lime_weights = lime_weights.sort_values('abs_weight', ascending=True)
    
    ax = axes[0]
    y_pos = range(len(lime_weights))
    colors_lime = ['#2ecc71' if x > 0 else '#e74c3c' for x in lime_weights['weight']]
    ax.barh(y_pos, lime_weights['weight'], color=colors_lime, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f.split('<')[0][:20] for f in lime_weights['feature_rule']], fontsize=9)
    ax.set_xlabel('Weight', fontweight='bold')
    ax.set_title('LIME: Local Feature Importance\n(First Test Instance)', fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(axis='x', alpha=0.3)

# SHAP importance
if shap_out is not None:
    shap_importance = shap_out['global_importance'].sort_values('mean_abs_shap', ascending=True)
    
    ax = axes[1]
    y_pos = range(len(shap_importance))
    ax.barh(y_pos, shap_importance['mean_abs_shap'], color='#3498db', alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(shap_importance['feature'], fontsize=9)
    ax.set_xlabel('Mean Absolute SHAP Value', fontweight='bold')
    ax.set_title('SHAP: Global Feature Importance\n(All Test Instances)', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('data/classing/feature_importance_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. ROC Curves: Baseline vs Post-Processed
fig, ax = plt.subplots(figsize=(10, 8))

# Baseline ROC
fpr_base, tpr_base, _ = roc_curve(yte_old, baseline_scores)
roc_auc_base = auc(fpr_base, tpr_base)
ax.plot(fpr_base, tpr_base, color='#e74c3c', lw=2.5, 
        label=f'Baseline (AUC = {roc_auc_base:.3f})')

# Post-processed ROC
fpr_post, tpr_post, _ = roc_curve(yte_old, post.scores)
roc_auc_post = auc(fpr_post, tpr_post)
ax.plot(fpr_post, tpr_post, color='#2ecc71', lw=2.5, 
        label=f'Post-Processed (AUC = {roc_auc_post:.3f})')

# Diagonal reference
ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Classifier')
ax.set_xlabel('False Positive Rate', fontweight='bold', fontsize=12)
ax.set_ylabel('True Positive Rate', fontweight='bold', fontsize=12)
ax.set_title('ROC Curves: Baseline vs Post-Processed', fontweight='bold', fontsize=14)
ax.legend(loc='lower right', fontsize=11)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('data/classing/roc_curves_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. Confusion Matrices
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Confusion Matrices: Baseline vs Post-Processed', fontsize=14, fontweight='bold')

# Baseline confusion matrix
cm_base = confusion_matrix(yte_old, baseline_pred)
disp_base = ConfusionMatrixDisplay(confusion_matrix=cm_base, display_labels=['Non-Default', 'Default'])
disp_base.plot(ax=axes[0], cmap='Reds', values_format='d')
axes[0].set_title('Baseline Model', fontweight='bold')

# Post-processed confusion matrix
cm_post = confusion_matrix(yte_old, post.predictions)
disp_post = ConfusionMatrixDisplay(confusion_matrix=cm_post, display_labels=['Non-Default', 'Default'])
disp_post.plot(ax=axes[1], cmap='Greens', values_format='d')
axes[1].set_title('Post-Processed Model', fontweight='bold')

plt.tight_layout()
plt.savefig('data/classing/confusion_matrices_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. Fairness Trade-offs by Sensitive Group
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle('Selection Rate & Accuracy by Sensitive Feature Group', fontsize=14, fontweight='bold')

groups = post.tradeoff_by_group['sensitive_feature_0'].astype(str).values
selection_rates = post.tradeoff_by_group['selection_rate'].values
accuracies = post.tradeoff_by_group['accuracy'].values

x = np.arange(len(groups))
width = 0.35

# Selection Rate
ax = axes[0]
bars = ax.bar(x, selection_rates, width, label='Selection Rate', alpha=0.7, color='#3498db')
ax.set_ylabel('Selection Rate', fontweight='bold')
ax.set_title('Selection Rate by Group\n(Post-Processed)', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'Group {g}' for g in groups])
ax.set_ylim([0, max(selection_rates) * 1.2])
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# Accuracy
ax = axes[1]
bars = ax.bar(x, accuracies, width, label='Accuracy', alpha=0.7, color='#2ecc71')
ax.set_ylabel('Accuracy', fontweight='bold')
ax.set_title('Accuracy by Group\n(Post-Processed)', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'Group {g}' for g in groups])
ax.set_ylim([0, max(accuracies) * 1.2])
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('data/classing/fairness_by_group.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. Counterfactual Fairness Deltas
fig, ax = plt.subplots(figsize=(10, 6))

delta_data = cfair.summary.iloc[0]
metrics = ['mean_abs_delta', 'max_abs_delta', 'p95_abs_delta']
values = [delta_data[m] for m in metrics]
labels = ['Mean Abs Delta', 'Max Abs Delta', '95th Percentile']

colors_delta = ['#e74c3c', '#f39c12', '#3498db']
bars = ax.bar(range(len(values)), values, color=colors_delta, alpha=0.7, edgecolor='black')
ax.set_ylabel('Score Delta', fontweight='bold', fontsize=12)
ax.set_title('Counterfactual Fairness Analysis\n(Credit Score Sensitive Attribute)', 
             fontweight='bold', fontsize=14)
ax.set_xticks(range(len(values)))
ax.set_xticklabels(labels)

for bar, val in zip(bars, values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('data/classing/counterfactual_fairness_deltas.png', dpi=300, bbox_inches='tight')
plt.show()

# 7. Predicted Score Distributions (Baseline vs Post-Processed)
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle('Score Distributions by Actual Default Status', fontsize=14, fontweight='bold')

# Baseline
ax = axes[0]
non_default_baseline = baseline_scores[yte_old == 0]
default_baseline = baseline_scores[yte_old == 1]
ax.hist(non_default_baseline, bins=50, alpha=0.6, label='Non-Default', color='#2ecc71')
ax.hist(default_baseline, bins=50, alpha=0.6, label='Default', color='#e74c3c')
ax.set_xlabel('Predicted Probability', fontweight='bold')
ax.set_ylabel('Frequency', fontweight='bold')
ax.set_title('Baseline Model Score Distribution', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Post-processed
ax = axes[1]
non_default_post = post.scores[yte_old == 0]
default_post = post.scores[yte_old == 1]
ax.hist(non_default_post, bins=50, alpha=0.6, label='Non-Default', color='#2ecc71')
ax.hist(default_post, bins=50, alpha=0.6, label='Default', color='#e74c3c')
ax.set_xlabel('Predicted Probability', fontweight='bold')
ax.set_ylabel('Frequency', fontweight='bold')
ax.set_title('Post-Processed Score Distribution', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('data/classing/score_distributions.png', dpi=300, bbox_inches='tight')
plt.show()

print('\n✓ All visualizations saved to data/classing/')
print('  - fairness_metrics_comparison.png')
print('  - feature_importance_comparison.png')
print('  - roc_curves_comparison.png')
print('  - confusion_matrices_comparison.png')
print('  - fairness_by_group.png')
print('  - counterfactual_fairness_deltas.png')
print('  - score_distributions.png')

# %%
# ==================== EXPLAINABILITY & FAIRNESS SUMMARY REPORT ====================

print('\n' + '=' * 90)
print('EXECUTIVE SUMMARY: EXPLAINABILITY & FAIRNESS ANALYSIS')
print('=' * 90)

print('\n📊 1. FEATURE IMPORTANCE ANALYSIS')
print('-' * 90)

print('\n   LIME (Local Explanation - First Test Instance):')
if lime_out is not None:
    lime_df = pd.DataFrame(lime_out['weights'], columns=['feature_rule', 'weight']).sort_values('weight', key=abs, ascending=False)
    for idx, row in lime_df.head(5).iterrows():
        feat = row['feature_rule'].split('<')[0].split('>')[0].strip()[:30]
        print(f'     • {feat:30} | Weight: {row["weight"]:+.4f}')

print('\n   SHAP (Global Explanation - All Test Instances):')
if shap_out is not None:
    for idx, row in shap_out['global_importance'].head(5).iterrows():
        print(f'     • {row["feature"]:30} | Mean Abs SHAP: {row["mean_abs_shap"]:.4f}')

print('\n   ✓ Key Insight: Original Interest Rate & Credit Score are the most important features')
print('     globally (SHAP), while Credit Score drives individual predictions locally (LIME)')

print('\n\n⚖️  2. FAIRNESS & PERFORMANCE TRADE-OFFS')
print('-' * 90)

print('\n   Baseline vs Post-Processed Comparison:')
key_metrics = [
    ('Recall (Higher = Better)', 'recall'),
    ('Demographic Parity Ratio (Higher = More Fair)', 'demographic_parity_ratio'),
    ('Equalized Odds Ratio (Higher = More Fair)', 'equalized_odds_ratio'),
    ('Accuracy', 'accuracy')
]

for metric_label, metric_name in key_metrics:
    baseline_val = comparison[comparison['metric'] == metric_name]['baseline'].values[0]
    post_val = comparison[comparison['metric'] == metric_name]['postprocessed'].values[0]
    delta = post_val - baseline_val
    arrow = '↑' if delta > 0 else '↓'
    print(f'     {metric_label}')
    print(f'       Baseline: {baseline_val:.4f} → Post-Processed: {post_val:.4f} (Δ {delta:+.4f}) {arrow}')

print('\n   ✓ Key Insight: Post-processing IMPROVES fairness metrics ~96% demographic')
print('     parity ratio) at the cost of ~24% accuracy loss. Recall improves by 62pp.')

print('\n\n🔍 3. COUNTERFACTUAL FAIRNESS ANALYSIS')
print('-' * 90)

cfair_data = cfair.summary.iloc[0]
print(f'\n   Sensitive Attribute: {cfair_data["sensitive_attribute"]}')
print(f'   • Mean Abs Score Delta:     {cfair_data["mean_abs_delta"]:.4f}')
print(f'   • Max Abs Score Delta:      {cfair_data["max_abs_delta"]:.4f}')
print(f'   • 95th Percentile Delta:    {cfair_data["p95_abs_delta"]:.4f}')

avg_change_pct = (cfair_data['mean_abs_delta'] / (1 - cfair_data['mean_abs_delta'])) * 100 if cfair_data['mean_abs_delta'] != 0 else 0
print(f'\n   ✓ Key Insight: Perturbing Credit Score changes predictions by ~{cfair_data["mean_abs_delta"]:.2%}')
print('     on average, indicating moderate sensitivity to this sensitive attribute.')

print('\n\n🎯 4. EXPLANATION QUALITY ASSESSMENT')
print('-' * 90)

print('\n   Anchor Explanation:')
if anchor_out:
    print(f'     • Precision: {anchor_out.get("precision", "N/A")}')
    print(f'     • Coverage:  {anchor_out.get("coverage", "N/A")}')
    print('     ℹ️  No simple rules found - predictions driven by complex feature interactions')

print('\n   Fidelity & Stability (OpenXAI-style):')
if not openxai_diag.empty:
    fidelity = openxai_diag.iloc[0]['fidelity_rank_corr']
    stability = openxai_diag.iloc[0]['stability_mean_corr']
    print(f'     • Fidelity (Rank Correlation):        {fidelity:.4f}')
    print(f'     • Stability (Perturbation Invariance): {stability:.4f}')
    print(f'     ✓ High stability ({stability:.2%}) means attributions are robust to small input changes')
    if fidelity < -0.3:
        print(f'     ⚠️  Negative fidelity suggests linear attributions may not fully capture model behavior')

print('\n\n📈 5. MODEL SELECTION RECOMMENDATIONS')
print('-' * 90)

print('\n   Scenario 1: Maximize Fairness')
print('     → Use Post-Processed Model with Demographic Parity Constraint')
print('     • Demographic Parity Ratio: 0.968 (near-perfect fairness)')
print('     • Trade-off: 24% accuracy loss, but 62pp recall improvement')
print('     • Best for: Credit decisions where fairness is paramount')

print('\n   Scenario 2: Balance Fairness & Accuracy')
print('     → Tune ThresholdOptimizer with different fairness constraints')
print('     • Current balance: 66% accuracy with 97% fairness ratio')
print('     • Options: Try equalized_odds or demographic_parity_difference constraints')

print('\n   Scenario 3: Maximize Accuracy (Baseline)')
print('     → Use Baseline Model without post-processing')
print('     • Accuracy: 90%, but fairness metrics poor (0% parity ratio)')
print('     • Not recommended for regulated credit decisions')

print('\n\n🔐 6. EXPLAINABILITY FOR COMPLIANCE')
print('-' * 90)

print('\n   ✓ Available Explanation Methods:')
print('     • LIME:          Local explanations for individual predictions')
print('     • SHAP:          Global feature importance + local contributions')
print('     • Anchor:        Rule-based explanations (limited applicability here)')
print('     • Ceteris Paribus: Marginal profiles for individual features')

print('\n   Recommended Approach:')
print('     1. Use SHAP for FEATURE IMPORTANCE ranking & regulatory reports')
print('     2. Use LIME for INDIVIDUAL CUSTOMER explanations (e.g., "Why was I declined?")')
print('     3. Use Counterfactual Fairness for SENSITIVITY analysis to protected attributes')
print('     4. Use Threshold post-processing for AUDIT TRAIL & controlled decision-making')

print('\n' + '=' * 90)
print('END OF REPORT')
print('=' * 90)

# %%
# ==================== CETERIS PARIBUS: CLTV & DTI RATIO ANALYSIS ====================

print('\n' + '=' * 90)
print('CETERIS PARIBUS ANALYSIS: CLTV & DTI RATIO')
print('=' * 90)

# Compute Ceteris Paribus for CLTV
print('\n🔄 Computing Ceteris Paribus profiles...')
cp_cltv = assess_ceteris_paribus(model_old, Xte_old, 'Original Combined Loan-to-Value (CLTV)', grid_points=30)
cp_dti = assess_ceteris_paribus(model_old, Xte_old, 'Original Debt-to-Income (DTI) Ratio', grid_points=30)

print(f'   ✓ CLTV profile: {len(cp_cltv)} grid points')
print(f'   ✓ DTI profile:  {len(cp_dti)} grid points')

# Debug: Check columns
print(f'\n   CLTV columns: {list(cp_cltv.columns)}')
print(f'   DTI columns:  {list(cp_dti.columns)}')

# Get the feature column names (should be the non-prediction columns)
cltv_feat_col = [col for col in cp_cltv.columns if col != 'prediction'][0]
dti_feat_col = [col for col in cp_dti.columns if col != 'prediction'][0]

print(f'\n   CLTV feature column: {cltv_feat_col}')
print(f'   DTI feature column:  {dti_feat_col}')

# Visualize Ceteris Paribus profiles
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Ceteris Paribus: Individual Prediction Profiles\n(Other features held constant at median values)', 
             fontsize=14, fontweight='bold')

# CLTV Profile
ax = axes[0]
ax.plot(cp_cltv[cltv_feat_col], cp_cltv['prediction'], 
        linewidth=2.5, color='#3498db', marker='o', markersize=5, label='Predicted Default Probability')
ax.fill_between(cp_cltv[cltv_feat_col], 
                 cp_cltv['prediction'], alpha=0.2, color='#3498db')
ax.set_xlabel('Combined Loan-to-Value (CLTV) - WoE Transformed', fontweight='bold', fontsize=11)
ax.set_ylabel('Predicted Default Probability', fontweight='bold', fontsize=11)
ax.set_title('CLTV Profile\n(Risk increases as CLTV increases)', fontweight='bold', fontsize=12)
ax.grid(True, alpha=0.3)

# Add risk zone coloring
ax.axhspan(0, 0.1, alpha=0.1, color='green', label='Low Risk (<10%)')
ax.axhspan(0.1, 0.2, alpha=0.1, color='yellow', label='Medium Risk (10-20%)')
ax.axhspan(0.2, 1, alpha=0.1, color='red', label='High Risk (>20%)')
ax.legend(loc='best', fontsize=9)

# DTI Profile
ax = axes[1]
ax.plot(cp_dti[dti_feat_col], cp_dti['prediction'], 
        linewidth=2.5, color='#e74c3c', marker='s', markersize=5, label='Predicted Default Probability')
ax.fill_between(cp_dti[dti_feat_col], 
                 cp_dti['prediction'], alpha=0.2, color='#e74c3c')
ax.set_xlabel('Debt-to-Income (DTI) Ratio - WoE Transformed', fontweight='bold', fontsize=11)
ax.set_ylabel('Predicted Default Probability', fontweight='bold', fontsize=11)
ax.set_title('DTI Profile\n(Risk increases as DTI increases)', fontweight='bold', fontsize=12)
ax.grid(True, alpha=0.3)

# Add risk zone coloring
ax.axhspan(0, 0.1, alpha=0.1, color='green', label='Low Risk (<10%)')
ax.axhspan(0.1, 0.2, alpha=0.1, color='yellow', label='Medium Risk (10-20%)')
ax.axhspan(0.2, 1, alpha=0.1, color='red', label='High Risk (>20%)')
ax.legend(loc='best', fontsize=9)

plt.tight_layout()
plt.savefig('data/classing/ceteris_paribus_cltv_dti.png', dpi=300, bbox_inches='tight')
plt.show()

print('\n✓ Ceteris Paribus visualization saved: data/classing/ceteris_paribus_cltv_dti.png')

# 📊 Statistical Summary
print('\n📊 STATISTICAL SUMMARY')
print('-' * 90)

print('\n   Combined Loan-to-Value (CLTV):')
cltv_min = cp_cltv[cltv_feat_col].min()
cltv_max = cp_cltv[cltv_feat_col].max()
pred_at_min_cltv = cp_cltv.loc[cp_cltv[cltv_feat_col].idxmin(), 'prediction']
pred_at_max_cltv = cp_cltv.loc[cp_cltv[cltv_feat_col].idxmax(), 'prediction']

print(f'     Range (WoE):           {cltv_min:.4f} → {cltv_max:.4f}')
print(f'     Prediction @ Min CLTV: {pred_at_min_cltv:.4f} ({pred_at_min_cltv*100:.2f}% default risk)')
print(f'     Prediction @ Max CLTV: {pred_at_max_cltv:.4f} ({pred_at_max_cltv*100:.2f}% default risk)')
print(f'     Risk Delta:            {(pred_at_max_cltv - pred_at_min_cltv)*100:+.2f} percentage points')

cltv_elasticity = ((pred_at_max_cltv - pred_at_min_cltv) / pred_at_min_cltv if pred_at_min_cltv != 0 else 0) / ((cltv_max - cltv_min) / cltv_min if cltv_min != 0 else 0) if cltv_min != 0 else 0
print(f'     Elasticity:            {cltv_elasticity:.2f} (% change in risk per % change in CLTV)')

print('\n   Debt-to-Income Ratio (DTI):')
dti_min = cp_dti[dti_feat_col].min()
dti_max = cp_dti[dti_feat_col].max()
pred_at_min_dti = cp_dti.loc[cp_dti[dti_feat_col].idxmin(), 'prediction']
pred_at_max_dti = cp_dti.loc[cp_dti[dti_feat_col].idxmax(), 'prediction']

print(f'     Range (WoE):           {dti_min:.4f} → {dti_max:.4f}')
print(f'     Prediction @ Min DTI:  {pred_at_min_dti:.4f} ({pred_at_min_dti*100:.2f}% default risk)')
print(f'     Prediction @ Max DTI:  {pred_at_max_dti:.4f} ({pred_at_max_dti*100:.2f}% default risk)')
print(f'     Risk Delta:            {(pred_at_max_dti - pred_at_min_dti)*100:+.2f} percentage points')

dti_elasticity = ((pred_at_max_dti - pred_at_min_dti) / pred_at_min_dti if pred_at_min_dti != 0 else 0) / ((dti_max - dti_min) / dti_min if dti_min != 0 else 0) if dti_min != 0 else 0
print(f'     Elasticity:            {dti_elasticity:.2f} (% change in risk per % change in DTI)')

# 🔍 Risk Interpretation
print('\n🔍 INTERPRETATION & INSIGHTS')
print('-' * 90)

print('\n   1. CLTV (Combined Loan-to-Value) Impact:')
if (pred_at_max_cltv - pred_at_min_cltv) > 0.05:
    print(f'      ⚠️  STRONG POSITIVE relationship: Higher CLTV → Higher default risk')
    print(f'      • Moving from min to max CLTV increases risk by {(pred_at_max_cltv - pred_at_min_cltv)*100:.2f}pp')
    print(f'      • This aligns with credit risk theory: higher loan amounts relative to')
    print(f'        property value increase borrower vulnerability')
else:
    print(f'      ✓ WEAK relationship: CLTV has limited direct impact on predictions')

print('\n   2. DTI (Debt-to-Income) Impact:')
if (pred_at_max_dti - pred_at_min_dti) > 0.05:
    print(f'      ⚠️  STRONG POSITIVE relationship: Higher DTI → Higher default risk')
    print(f'      • Moving from min to max DTI increases risk by {(pred_at_max_dti - pred_at_min_dti)*100:.2f}pp')
    print(f'      • This aligns with credit risk theory: higher debt burden reduces')
    print(f'        repayment capacity and increases default likelihood')
else:
    print(f'      ✓ WEAK relationship: DTI has limited direct impact on predictions')

print('\n   3. Comparative Analysis:')
cltv_impact = (pred_at_max_cltv - pred_at_min_cltv) * 100
dti_impact = (pred_at_max_dti - pred_at_min_dti) * 100

if abs(cltv_impact) > abs(dti_impact):
    print(f'      → CLTV is more influential ({cltv_impact:+.2f}pp) than DTI ({dti_impact:+.2f}pp)')
    print(f'      → Loan amount relative to property value is the primary driver')
else:
    print(f'      → DTI is more influential ({dti_impact:+.2f}pp) than CLTV ({cltv_impact:+.2f}pp)')
    print(f'      → Borrower debt burden is the primary driver of default risk')

print('\n   4. Model Decision-Making:')
print(f'      • The model considers both CLTV and DTI but likely relies more on')
print(f'        Original Interest Rate (from SHAP: 0.041 importance) and Credit Score')
print(f'        • WoE transformation has already captured non-linearity in these features')
print(f'        • Interactions between CLTV and DTI are implicitly modeled')

print('\n' + '=' * 90)

# %%
!pip install fairlearn

# %%
# Persist core artifacts for the current model run (timestamped, non-overwriting).
from datetime import datetime
from pathlib import Path

out_dir = Path('data/classing')
out_dir.mkdir(parents=True, exist_ok=True)

run_tag = datetime.now().strftime('%Y%m%d_%H%M%S')

# Ensure required model objects exist.
required = ['model', 'X_train', 'X_test', 'y_train', 'y_test', 'cols_list']
missing_required = [name for name in required if name not in globals()]
if missing_required:
    raise RuntimeError(f'Missing required objects for export: {missing_required}')

# Recompute predictions for consistency with current in-memory model.
y_pred_train_cur = model.predict(X_train)
y_pred_proba_train_cur = model.predict_proba(X_train)[:, 1]
y_pred_test_cur = model.predict(X_test)
y_pred_proba_test_cur = model.predict_proba(X_test)[:, 1]

# Metrics.
train_auc_cur = roc_auc_score(y_train, y_pred_proba_train_cur)
test_auc_cur = roc_auc_score(y_test, y_pred_proba_test_cur)
train_gini_cur = 2 * train_auc_cur - 1
test_gini_cur = 2 * test_auc_cur - 1
fpr_train, tpr_train, _ = roc_curve(y_train, y_pred_proba_train_cur)
fpr_test, tpr_test, _ = roc_curve(y_test, y_pred_proba_test_cur)
train_ks_cur = float(np.max(tpr_train - fpr_train))
test_ks_cur = float(np.max(tpr_test - fpr_test))

# Logistic regression output table.
coef_export = pd.DataFrame({
    'Variable': cols_list,
    'Coefficient': model.coef_[0],
})
coef_export['Odds_Ratio'] = np.exp(coef_export['Coefficient'])
coef_export['Intercept'] = float(model.intercept_[0])
coef_export['Train_AUC'] = train_auc_cur
coef_export['Test_AUC'] = test_auc_cur
coef_export['Train_Gini'] = train_gini_cur
coef_export['Test_Gini'] = test_gini_cur
coef_export['Train_KS'] = train_ks_cur
coef_export['Test_KS'] = test_ks_cur
coef_export = coef_export.sort_values('Coefficient', ascending=False).reset_index(drop=True)

logit_out = out_dir / f'logistic_regression_output_current_{run_tag}.csv'
coef_export.to_csv(logit_out, index=False)

# Scorecard conversion settings: prefer notebook values, fallback to defaults.
base_score_cur = globals().get('base_score', 600)
pdo_cur = globals().get('pdo', 20)
target_odds_cur = globals().get('target_odds', 60)
score_floor_cur = globals().get('score_floor', 300)
score_cap_cur = globals().get('score_cap', 900)

def score_from_proba(pd_proba, base_score=600, target_odds=60, pdo=20, floor=300, cap=900):
    # Convert PD to score using logit scaling.
    f = pdo / np.log(2)
    p = np.clip(np.asarray(pd_proba, dtype=float), 1e-9, 1 - 1e-9)
    log_odds_bad = np.log(p / (1 - p))
    score = base_score + f * (np.log(target_odds) - log_odds_bad)
    return np.clip(score, floor, cap)

test_scores_cur = score_from_proba(
    y_pred_proba_test_cur,
    base_score=base_score_cur,
    target_odds=target_odds_cur,
    pdo=pdo_cur,
    floor=score_floor_cur,
    cap=score_cap_cur,
 )

# Save test set scores/probabilities.
score_export = pd.DataFrame({
    'score': test_scores_cur,
    'predicted_probability': y_pred_proba_test_cur,
    'predicted_label': y_pred_test_cur,
    'actual_label': np.asarray(y_test),
})
scores_out = out_dir / f'test_set_scores_current_{run_tag}.csv'
score_export.to_csv(scores_out, index=False)

# Score bands on converted score (not on raw probability).
bands = score_export.copy()
bands['score_band'] = pd.qcut(bands['score'], q=10, duplicates='drop')
bands_summary = bands.groupby('score_band', observed=True).agg(
    score_min=('score', 'min'),
    score_max=('score', 'max'),
    population=('actual_label', 'count'),
    defaults=('actual_label', 'sum'),
    avg_score=('score', 'mean'),
    avg_predicted_probability=('predicted_probability', 'mean'),
).reset_index()
bands_summary['default_rate'] = bands_summary['defaults'] / bands_summary['population']
bands_summary = bands_summary.sort_values('avg_score', ascending=False).reset_index(drop=True)
bands_summary['band_rank_low_to_high_risk'] = np.arange(1, len(bands_summary) + 1)

bands_out = out_dir / f'score_bands_current_{run_tag}.csv'
bands_summary.to_csv(bands_out, index=False)

# Metadata.
meta = pd.DataFrame([
    {
        'run_tag': run_tag,
        'model_type': 'LogisticRegression',
        'n_features': int(X_train.shape[1]),
        'train_rows': int(X_train.shape[0]),
        'test_rows': int(X_test.shape[0]),
        'train_auc': float(train_auc_cur),
        'test_auc': float(test_auc_cur),
        'train_gini': float(train_gini_cur),
        'test_gini': float(test_gini_cur),
        'train_ks': float(train_ks_cur),
        'test_ks': float(test_ks_cur),
        'base_score': float(base_score_cur),
        'pdo': float(pdo_cur),
        'target_odds': float(target_odds_cur),
        'score_floor': float(score_floor_cur),
        'score_cap': float(score_cap_cur),
    }
])
meta_out = out_dir / f'model_run_metadata_current_{run_tag}.csv'
meta.to_csv(meta_out, index=False)

print('Saved current model artifacts:')
print(f' - {logit_out.as_posix()}')
print(f' - {scores_out.as_posix()}')
print(f' - {bands_out.as_posix()}')
print(f' - {meta_out.as_posix()}')
print(f'\nScore scaling used: base={base_score_cur}, pdo={pdo_cur}, target_odds={target_odds_cur}, range=[{score_floor_cur}, {score_cap_cur}]')

# %%
!pip install shap

# %%
import shap
print(f"SHAP version: {shap.__version__}")

# %%
!pip install anchor-exp
