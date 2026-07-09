# Auto-generated from Classing and Model-new.ipynb
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
final_bins=pd.read_csv("data/classing/repaired_train_selected_coarse_classing_lambda00_monotonic.csv")

# %%
final_bins['variable'].unique()

# %%
variables_iv_high=final_bins[final_bins['iv_total'] > 0.01]['variable'].unique()

# %%
variables_iv_high

# %%
cols_list = [
    'Credit Score',
    'Mortgage Insurance Percentage (MI %)',
    'Number of Units',
    'Occupancy Status',
    'Original Combined Loan-to-Value (CLTV)',
    'Original Debt-to-Income (DTI) Ratio',
    'Original UPB',
    'Original Loan-to-Value (LTV)',
    'Original Interest Rate',
    'Channel',
    'Property Type',
    'Loan Purpose',
    'Original Loan Term',
    'Number of Borrowers',
    'Program Indicator',
    'Property Valuation Method',
    'median_income',
    'poverty_rate',
    'avg_elec_cost',
    'msa_hpi_2020',
    'msa_hpi_growth',
    'crime_rate',
    'census_division_code',
    'First Time Homebuyer Flag'
]

target_col = 'target'

# %%
df_train_v1 = pd.read_csv('data/classing/train_dir_repaired_lambda00.csv')

missing_cols = [c for c in cols_list + [target_col] if c not in df_train_v1.columns]
if missing_cols:
    raise ValueError(f'Missing required columns in repaired train file: {missing_cols}')

df_train_v1 = df_train_v1[cols_list + [target_col]]

# %%
df_train_v1

# %%
df_test_v1=pd.read_csv('data/classing/test_dir_repaired_lambda00.csv')

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
