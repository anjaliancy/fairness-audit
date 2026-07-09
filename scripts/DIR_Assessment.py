# Auto-generated from DIR Assessment.ipynb
# Source notebook exported as a plain Python script.

# %%
import os
import sys
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Ensure local src/ is importable when running from notebook
repo_root = os.getcwd()
if repo_root not in sys.path:
    sys.path.append(repo_root)

from src.fairness_preprocessing import apply_disparate_impact_repair, sweep_repair_levels

# 1) Load stratified train/test datasets from classing folder
train_path = "data/classing/main_alt_train_full_stratified.csv"
test_path = "data/classing/main_alt_test_full_stratified.csv"

train_raw = pd.read_csv(train_path)
test_raw = pd.read_csv(test_path)

# 2) Build a clean subset from stratified files to avoid full-table null issues in AIF360
label_name = "target"  # Adjust if your target column has a different name
protected_attributes = ["First Time Homebuyer Flag","State"]

candidate_predictors = [
    "Credit Score",
    "Original Combined Loan-to-Value (CLTV)",
    "Original Debt-to-Income (DTI) Ratio",
    "Original UPB",
    "Original Interest Rate",
    "Number of Borrowers",
    "Mortgage Insurance Percentage (MI %)",
    "Loan Purpose",
    "Occupancy Status",
    "Channel",
    "Property Type",
    "State",
]

keep_cols = [label_name] + protected_attributes + [c for c in candidate_predictors if c in train_raw.columns]
keep_cols = list(dict.fromkeys(keep_cols))

train_df = train_raw[keep_cols].copy()
test_df = test_raw[keep_cols].copy()

# Fill nulls from train statistics
for col in train_df.columns:
    if col == label_name:
        continue
    if pd.api.types.is_numeric_dtype(train_df[col]):
        fill_val = train_df[col].median()
    else:
        mode_series = train_df[col].mode(dropna=True)
        fill_val = mode_series.iloc[0] if not mode_series.empty else "UNKNOWN"
    train_df[col] = train_df[col].fillna(fill_val)
    test_df[col] = test_df[col].fillna(fill_val)

# 3) Required fairness inputs
privileged_values = {
    "First Time Homebuyer Flag": "N",  # adjust per policy
}
repair_level = 0.4
favorable_label = 0  # In credit default modeling, often 0=non-default

# 4) Apply Disparate Impact Remover before classing/WOE/model training
repair_result = apply_disparate_impact_repair(
    train_df=train_df,
    test_df=test_df,
    label_name=label_name,
    protected_attributes=protected_attributes,
    repair_level=repair_level,
    favorable_label=favorable_label,
    privileged_values=privileged_values,
    sequential_repair=True,
)

print("Fairness BEFORE repair (train):")
display(repair_result.fairness_report_train_original)

print("Fairness AFTER repair (train):")
display(repair_result.fairness_report_train)

print("Fairness AFTER repair (test):")
display(repair_result.fairness_report_test)

# 5) Sweep repair levels and compare fairness/performance tradeoff
def feature_builder(train_data, test_data, target_col):
    y_train = train_data[target_col]
    y_test = test_data[target_col]

    X_train = train_data.drop(columns=[target_col])
    X_test = test_data.drop(columns=[target_col])

    combined = pd.concat([X_train, X_test], axis=0)
    combined_encoded = pd.get_dummies(combined, drop_first=True)

    X_train_enc = combined_encoded.iloc[: len(X_train), :].reset_index(drop=True)
    X_test_enc = combined_encoded.iloc[len(X_train) :, :].reset_index(drop=True)

    return X_train_enc, X_test_enc, y_train.reset_index(drop=True), y_test.reset_index(drop=True)


model_builder = lambda X, y: LogisticRegression(max_iter=1000).fit(X, y)

tradeoff_df = sweep_repair_levels(
    train_df=train_df,
    test_df=test_df,
    label_name=label_name,
    protected_attributes=protected_attributes,
    model_builder=model_builder,
    feature_builder=feature_builder,
    repair_levels=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    favorable_label=favorable_label,
    privileged_values=privileged_values,
    fairness_on="test",
)

print("Fairness vs performance tradeoff by repair level:")
display(tradeoff_df)

# %%
# Diagnostics: why fairness metrics may be NaN
attr = protected_attributes[0]
priv = privileged_values[attr]

print(f"Protected attribute: {attr}")
print(f"Privileged value used: {priv}")

for split_name, df in [("train", train_df), ("test", test_df)]:
    print(f"\n--- {split_name.upper()} SPLIT ---")
    vc = df[attr].value_counts(dropna=False)
    print("Group counts:")
    print(vc)

    grp = (
        df.groupby(attr, dropna=False)[label_name]
          .agg(count="count", favorable_rate="mean")
          .reset_index()
          .sort_values(by="count", ascending=False)
    )
    print("Per-group favorable rate (mean of target):")
    print(grp)

    priv_mask = df[attr] == priv
    unpriv_mask = ~priv_mask

    priv_count = int(priv_mask.sum())
    unpriv_count = int(unpriv_mask.sum())
    priv_rate = df.loc[priv_mask, label_name].mean() if priv_count > 0 else float("nan")
    unpriv_rate = df.loc[unpriv_mask, label_name].mean() if unpriv_count > 0 else float("nan")

    print(f"Privileged count: {priv_count}, Unprivileged count: {unpriv_count}")
    print(f"Privileged favorable rate: {priv_rate}")
    print(f"Unprivileged favorable rate: {unpriv_rate}")

    if priv_count == 0 or unpriv_count == 0:
        print("Reason for NaN: one of the groups is missing in this split.")
    elif priv_rate == 0:
        print("Reason for NaN DI ratio: privileged favorable rate is zero (division by zero).")
    elif pd.isna(priv_rate) or pd.isna(unpriv_rate):
        print("Reason for NaN: group rates are undefined due to missing values.")
    else:
        print("Both groups exist and rates are defined for this split.")

# %%
# Diagnostics on repaired outputs
attr = protected_attributes[0]

print("Repaired train protected-value counts:")
print(repair_result.train_repaired[attr].value_counts(dropna=False))

print("\nRepaired test protected-value counts:")
print(repair_result.test_repaired[attr].value_counts(dropna=False))

print("\nUnique protected values in repaired train:", sorted(repair_result.train_repaired[attr].dropna().unique().tolist()))
print("Unique protected values in repaired test:", sorted(repair_result.test_repaired[attr].dropna().unique().tolist()))
