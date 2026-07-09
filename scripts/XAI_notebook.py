# Auto-generated from XAI notebook.ipynb
# Source notebook exported as a plain Python script.

# %% [markdown]
# # Mortgage Fairness Audit Pipeline
# **Dataset**: Freddie Mac Single-Family Loan-Level Data (100k rows, 41 cols)  
# **Target**: `is_defaulter` (1 = Default, 0 = Non-Default) | Default rate ~10.2%
#
# **Stages**: Setup → Pre-processing Fairness → Modelling → Explainability → Post-processing & Evaluation

# %%
# Install dependencies (run once)
!pip install -r requirements.txt

# %%
!pip install xgboost

# %% [markdown]
# ## Stage 0 — Setup, Configuration & Data Loading

# %%
# =============================================================================
# MORTGAGE FAIRNESS AUDIT PIPELINE
# Dataset : Freddie Mac Single-Family Loan-Level Data (100,000 rows, 41 cols)
# Target  : Defaulter (Y/N)  →  1 = Default, 0 = Non-Default
# Default Rate: ~10.2%
# =============================================================================
#
# STAGE 0 — SETUP, CONFIGURATION & DATA LOADING
#
# Install dependencies before running:
#   pip install aif360 fairlearn dice-ml lime anchor-exp shap optbinning
#               scikit-learn xgboost imbalanced-learn pandas numpy
#               matplotlib seaborn tensorflow==2.12 openxai
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ── Sklearn ───────────────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

# ── XGBoost ───────────────────────────────────────────────────────────────────
from xgboost import XGBClassifier

# ── IBM AIF360 ────────────────────────────────────────────────────────────────
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import DisparateImpactRemover
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.inprocessing import AdversarialDebiasing

# ── Fairlearn ─────────────────────────────────────────────────────────────────
from fairlearn.reductions import (
    ExponentiatedGradient, DemographicParity, EqualizedOdds
)
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
    MetricFrame,
    selection_rate,
    false_positive_rate,
    false_negative_rate,
)

# ── Explainability ────────────────────────────────────────────────────────────
import shap
import lime
import lime.lime_tabular
# from anchor import anchor_tabular          # pip install anchor-exp
import dice_ml
from dice_ml import Dice

# ── WOE / Optimal Binning ─────────────────────────────────────────────────────
from optbinning import BinningProcess

# ── TensorFlow (AdversarialDebiasing) ────────────────────────────────────────
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

# %%
CONFIG = {
    # ── Paths ──────────────────────────────────────────────────────────────────
    "data_path"  : "frm_prepped_data-withstate.csv",
    "output_dir" : "outputs/",

    # ── Target ─────────────────────────────────────────────────────────────────
    # 'is_defaulter' is already a clean 0/1 integer version of 'Defaulter (Y/N)'
    "target_col"          : "is_defaulter",

    # ── Protected attributes ────────────────────────────────────────────────────
    # Chosen for legal/reputational fair-lending risk; see rationale in README
    "protected_attrs"     : [
        "First Time Homebuyer Flag",   # socioeconomic / age proxy
        "Property State",              # geographic redlining risk
        "Postal Code_modif",           # local redlining (bucketed postal codes)
        "Channel",                     # predatory lending channel risk
        "Occupancy Status",            # socioeconomic proxy
        "Number of Borrowers",         # single-borrower = vulnerable population
    ],

    # ── Columns to DROP before modelling ───────────────────────────────────────
    # Leakage columns, identifiers, and redundant targets
    "drop_cols" : [
        "Unnamed: 0",
        "Loan Sequence Number",
        "Pre-HARP Loan Sequence Number",
        "HARP Indicator",
        "Super Conforming Flag",
        "Program Indicator",
        "target",               # redundant with is_defaulter
        "Defaulter (Y/N)",      # string version of target
        "max_deliquency",       # outcome-derived → leakage
        "delinquency_rate",     # outcome-derived → leakage
        "Postal Code",          # replaced by bucketed Postal Code_modif
        "Postal_code_modif",    # duplicate of Postal Code_modif
        "state",                # duplicate of Property State
        "Seller Name",          # too many categories, not a risk feature
        "Servicer Name",        # too many categories, not a risk feature
        "Maturity Date",        # derived from loan term + origination date
        "Mortgage Insurance Cancellation Indicator",  # 100% missing
        "HARP Indicator",       # 100% missing (already listed above)
        "Pre-HARP Loan Sequence Number",              # 100% missing
    ],

    # ── Categorical feature list (for WOE BinningProcess) ────────────────────
    "categorical_cols" : [
        "First Time Homebuyer Flag",
        "Occupancy Status",
        "Channel",
        "Property State",
        "Property Type",
        "Loan Purpose",
        "Amortization Type (Formerly Product Type)",
        "Prepayment Penalty Mortgage (PPM) Flag",
        "Interest Only (I/O) Indicator",
    ],

    # ── Immutable features (cannot be changed in counterfactuals) ────────────
    "immutable_features" : [
        "First Time Homebuyer Flag",
        "Property State",
        "Postal Code_modif",
        "Number of Borrowers",
    ],

    # ── Repair levels for DI Remover sweep ───────────────────────────────────
    "repair_levels"      : [0.0, 0.5, 0.8, 1.0],

    # ── Train / Val / Test split sizes ───────────────────────────────────────
    "test_size"          : 0.20,
    "val_size"           : 0.10,
    "random_state"       : 42,

    # ── Fairness thresholds ───────────────────────────────────────────────────
    "di_ratio_threshold" : 0.80,   # 4/5ths rule (EEOC / fair lending standard)
    "spd_threshold"      : 0.05,   # |Statistical Parity Difference|
    "eod_threshold"      : 0.05,   # |Equalized Odds Difference|

    # ── DiCE ─────────────────────────────────────────────────────────────────
    "num_counterfactuals": 5,

    # ── XAI local sample index (applicant to explain) ─────────────────────────
    "local_explain_idx"  : 0,      # change to any test set row index
}

# Create output directories
for sub in ["", "plots/", "reports/", "models/"]:
    Path(CONFIG["output_dir"] + sub).mkdir(parents=True, exist_ok=True)


# =============================================================================
# STAGE 0A — DATA LOADING & CLEANING
# =============================================================================

def load_and_clean(path: str):
    """
    Load the Freddie Mac dataset and apply cleaning steps tailored to the
    actual column structure observed in frm_prepped_data-withstate.csv.

    Key decisions
    ─────────────
    • Drop 100%-missing columns (HARP Indicator, Mortgage Insurance
      Cancellation Indicator, Pre-HARP Loan Sequence Number).
    • Drop leakage columns derived from the target (max_deliquency,
      delinquency_rate).
    • Use 'is_defaulter' (already 0/1 int) as the model target.
    • Use 'Postal Code_modif' (bucketed) instead of raw Postal Code to
      reduce cardinality from 884 → manageable groups.
    • Encode string categorical protected attributes to integer codes so
      AIF360's BinaryLabelDataset can process them numerically.

    Returns
    ───────
    df             : cleaned DataFrame
    label_encoders : dict of {col_name: LabelEncoder} for inverse mapping
    """
    df = pd.read_csv(path, low_memory=False)

    # ── Drop defined columns ──────────────────────────────────────────────────
    drop = [c for c in CONFIG["drop_cols"] if c in df.columns]
    df.drop(columns=drop, inplace=True)

    # ── Drop remaining always-null columns ───────────────────────────────────
    all_null = [c for c in df.columns if df[c].isna().all()]
    df.drop(columns=all_null, inplace=True)

    # ── Drop MSA (8k missing, high cardinality, non-protected) ───────────────
    if "Metropolitan Statistical Area (MSA) Or Metropolitan Division" in df.columns:
        df.drop(
            columns=["Metropolitan Statistical Area (MSA) Or Metropolitan Division"],
            inplace=True)

    # ── First Payment Date: extract year & month as numeric features ──────────
    if "First Payment Date" in df.columns:
        df["First Payment Date"] = pd.to_numeric(
            df["First Payment Date"], errors="coerce")
        df["origination_year"]  = df["First Payment Date"] // 100
        df["origination_month"] = df["First Payment Date"] % 100
        df.drop(columns=["First Payment Date"], inplace=True)

    # ── Cap Credit Score at 850 (9999 = unknown/missing code) ────────────────
    if "Credit Score" in df.columns:
        df["Credit Score"] = df["Credit Score"].replace(9999, np.nan)
        df["Credit Score"] = df["Credit Score"].clip(upper=850)

    # ── Label-encode string protected attributes (AIF360 needs numeric) ───────
    label_encoders = {}
    for col in CONFIG["protected_attrs"]:
        if col in df.columns and df[col].dtype == object:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str).fillna("Unknown"))
            label_encoders[col] = le
            print(f"  [Encoding] {col}: {dict(enumerate(le.classes_))}")

    # ── Label-encode remaining string categorical features ───────────────────
    other_cats = [c for c in CONFIG["categorical_cols"]
                  if c in df.columns
                  and c not in CONFIG["protected_attrs"]
                  and df[c].dtype == object]
    for col in other_cats:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str).fillna("Unknown"))
        label_encoders[col] = le

    # ── Fill remaining numeric NaNs with median ───────────────────────────────
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != CONFIG["target_col"]]
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # ── Verify target ─────────────────────────────────────────────────────────
    assert CONFIG["target_col"] in df.columns, \
        f"Target column '{CONFIG['target_col']}' not found after cleaning!"
    df[CONFIG["target_col"]] = df[CONFIG["target_col"]].astype(int)

    print(f"\n[Stage 0] Cleaned dataset: {df.shape[0]:,} rows × {df.shape[1]} cols")
    print(f"[Stage 0] Default rate   : {df[CONFIG['target_col']].mean():.2%}")
    print(f"[Stage 0] Feature list   : {[c for c in df.columns if c != CONFIG['target_col']]}")

    return df, label_encoders


# =============================================================================
# STAGE 0B — TRAIN / VALIDATION / TEST SPLIT
# =============================================================================

def split_data(df: pd.DataFrame):
    """
    Stratified 70 / 10 / 20 split.

    Stratification on the target preserves the ~10% default rate
    in every partition — critical for imbalanced credit datasets.

    The same indices are reused for all repair levels to ensure
    fair apple-to-apple comparison.
    """
    X = df.drop(columns=[CONFIG["target_col"]])
    y = df[CONFIG["target_col"]]

    # ── First cut: 80% train+val, 20% test ───────────────────────────────────
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y,
        test_size=CONFIG["test_size"],
        stratify=y,
        random_state=CONFIG["random_state"],
    )

    # ── Second cut: 87.5% of 80% = 70% train, 12.5% of 80% = 10% val ────────
    val_frac = CONFIG["val_size"] / (1 - CONFIG["test_size"])   # 0.10/0.80
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv,
        test_size=val_frac,
        stratify=y_tv,
        random_state=CONFIG["random_state"],
    )

    for name, X_, y_ in [("Train", X_train, y_train),
                          ("Val  ", X_val,   y_val),
                          ("Test ", X_test,  y_test)]:
        print(f"[Stage 0] {name}: {X_.shape[0]:>6,} rows | "
              f"Default rate={y_.mean():.2%}")

    return X_train, X_val, X_test, y_train, y_val, y_test


# =============================================================================
# ENTRY POINT
# =============================================================================

# %% [markdown]
# ## Stage 1 — Pre-Processing Fairness
# DisparateImpactRemover sweep → WOE encoding → fairness metrics → trade-off table

# %%
# =============================================================================
# STAGE 1 — PRE-PROCESSING FAIRNESS
#
# Pipeline
# ────────
#  Raw Training Data
#    └─► Wrap → AIF360 BinaryLabelDataset
#          └─► DisparateImpactRemover  (repair ∈ {0.0, 0.5, 0.8, 1.0})
#                └─► Fine/Coarse Classing via OptBinning → WOE Transform
#                      └─► Measure DI Ratio & Statistical Parity Difference
#                            └─► Quick LR benchmark → Trade-off table
#
# Note: Val/Test sets are NEVER repaired — only the training data is.
#       WOE transform is fitted on repaired train and applied to val/test
#       (no data leakage).
# =============================================================================



# =============================================================================
# 1A — WRAP INTO AIF360 BinaryLabelDataset
# =============================================================================

def to_aif360(X: pd.DataFrame,
              y: pd.Series,
              protected_attrs: list) -> BinaryLabelDataset:
    """
    Merge X and y into one DataFrame and wrap into AIF360's
    BinaryLabelDataset.

    AIF360 conventions used here
    ────────────────────────────
    favorable_label   = 0  (non-defaulter: the good outcome for applicant)
    unfavorable_label = 1  (defaulter: the bad outcome for applicant)

    All protected attribute columns must be numeric at this point
    (LabelEncoder was applied in Stage 0).
    """
    combined = X.copy()
    combined[CONFIG["target_col"]] = y.values

    dataset = BinaryLabelDataset(
        df=combined,
        label_names=[CONFIG["target_col"]],
        protected_attribute_names=protected_attrs,
        favorable_label=0,
        unfavorable_label=1,
    )
    return dataset


# =============================================================================
# 1B — DISPARATE IMPACT REMOVER SWEEP
# =============================================================================

def apply_dir(aif_ds: BinaryLabelDataset,
              repair_level: float,
              sensitive_attribute: str) -> BinaryLabelDataset:
    """
    Apply DisparateImpactRemover for a single protected attribute.

    How it works
    ────────────
    The remover repairs the marginal distributions of all non-sensitive
    features toward a common target distribution, stratified by the
    sensitive attribute.  Higher repair_level → distributions converge
    more → model sees less signal correlated with the protected attribute.

    repair_level = 0.0 → no change (baseline)
    repair_level = 1.0 → full repair (maximum fairness, some performance cost)
    """
    remover = DisparateImpactRemover(
        repair_level=repair_level,
        sensitive_attribute=sensitive_attribute,
    )
    return remover.fit_transform(aif_ds)


def sweep_repair_levels(X_train: pd.DataFrame,
                        y_train: pd.Series) -> dict:
    """
    For each repair level, sequentially apply DisparateImpactRemover
    across ALL protected attributes.

    Sequential application means each attribute's feature distributions
    are repaired in turn. This is the standard multi-attribute DI removal
    approach — note that order can matter slightly; results should be
    validated against a joint-attribute repair if the library supports it.

    Returns
    ───────
    dict: {repair_level (float) → repaired DataFrame}
    """
    results = {}

    for level in CONFIG["repair_levels"]:
        print(f"\n[Stage 1B] ── Repair Level = {level} ──────────────────")

        aif_ds = to_aif360(X_train, y_train, CONFIG["protected_attrs"])

        if level == 0.0:
            # Baseline: convert back without any repair
            repaired_df, _ = aif_ds.convert_to_dataframe()
        else:
            current = aif_ds
            for attr in CONFIG["protected_attrs"]:
                # Only apply if the attribute exists in the dataset
                if attr in current.feature_names:
                    current = apply_dir(current, level, attr)
                    print(f"           Repaired: {attr}")
            repaired_df, _ = current.convert_to_dataframe()

        results[level] = repaired_df
        print(f"           Repaired shape: {repaired_df.shape}")

    return results


# =============================================================================
# 1C — FINE / COARSE CLASSING → WOE ENCODING
# =============================================================================

def fit_woe(X: pd.DataFrame,
            y: pd.Series,
            categorical_cols: list) -> BinningProcess:
    """
    Fit OptBinning's BinningProcess which performs:

    1. Fine Classing  — decision-tree optimal split into ≤20 pre-bins
    2. Coarse Classing — merge adjacent bins with similar WOE to ensure
                         monotonicity and minimum bin size
    3. WOE Transform  — replace raw values with Weight-of-Evidence scores
                         WOE_i = ln(P(X=i|Y=1) / P(X=i|Y=0))

    Information Value (IV) is used downstream for feature selection.
    IV > 0.02  : some predictive power
    IV > 0.10  : medium predictive power
    IV > 0.30  : strong predictive power
    IV > 0.50  : suspiciously strong (check for leakage)

    The fitted BinningProcess is returned so it can be deterministically
    applied to val/test without leakage.
    """
    # Restrict categorical_cols to those actually present
    cats = [c for c in categorical_cols if c in X.columns]

    bp = BinningProcess(
        variable_names=X.columns.tolist(),
        categorical_variables=cats,
        max_n_prebins=20,          # Fine classing ceiling
        min_prebin_size=0.05,      # Each pre-bin ≥ 5% of records
        min_n_bins=2,              # Coarse classing: minimum 2 bins
        max_n_bins=8,              # Coarse classing: maximum 8 bins
        min_event_rate_diff=0.02,  # Minimum WOE gap to keep bins separate
        monotonic_trend="auto",    # Enforce monotonic WOE where sensible
    )

    bp.fit(X.values, y.values)
    print(f"\n[Stage 1C] BinningProcess fitted — {len(X.columns)} features")
    return bp


def transform_woe(bp: BinningProcess,
                  X: pd.DataFrame) -> pd.DataFrame:
    """Apply fitted BinningProcess WOE transform to any split."""
    X_woe = bp.transform(X.values, metric="woe")
    return pd.DataFrame(X_woe, columns=X.columns, index=X.index)


def print_iv_table(bp: BinningProcess) -> pd.DataFrame:
    """Print and return the Information Value table sorted by IV descending."""
    summary = bp.summary().sort_values("iv", ascending=False)
    print("\n[Stage 1C] Information Value (IV) Table:")
    print(summary[["name", "dtype", "n_bins", "iv"]].to_string(index=False))
    print("\n  IV Guide: <0.02 Useless | 0.02–0.1 Weak | "
          "0.1–0.3 Medium | 0.3–0.5 Strong | >0.5 Suspect")
    return summary


# =============================================================================
# 1D — FAIRNESS METRICS ON REPAIRED DATA (PRE-MODEL)
# =============================================================================

def compute_pre_model_fairness(repaired_df: pd.DataFrame,
                                repair_level: float) -> list:
    """
    Measure Disparate Impact Ratio and Statistical Parity Difference on
    the REPAIRED training data before any model is trained.

    These are dataset-level metrics — they measure how the label (default)
    is distributed across protected groups in the repaired data.

    Disparate Impact Ratio (DIR)
    ────────────────────────────
    DIR = P(Y=1 | unprivileged) / P(Y=1 | privileged)
    Threshold: ≥ 0.80 (the EEOC 4/5ths rule)
    DIR < 0.80 indicates the unprivileged group defaults MORE, relative to
    privileged, at a rate that raises legal concern.

    Statistical Parity Difference (SPD)
    ────────────────────────────────────
    SPD = P(Y=1 | unprivileged) − P(Y=1 | privileged)
    Threshold: |SPD| ≤ 0.05
    Negative SPD means unprivileged group has higher default rate.

    Privileged group is defined as the majority class of each attribute
    (most common encoded value) — this is a reasonable proxy when we
    don't have explicit privileged/unprivileged group labels.
    """
    rows = []

    for attr in CONFIG["protected_attrs"]:
        if attr not in repaired_df.columns:
            continue

        majority_val = int(repaired_df[attr].mode()[0])
        minority_val = sorted(
            [v for v in repaired_df[attr].unique() if v != majority_val]
        )
        if not minority_val:
            continue
        # Use the smallest minority value (encoded lowest class)
        minority_val = minority_val[0]

        privileged   = [{attr: majority_val}]
        unprivileged = [{attr: minority_val}]

        aif_ds = BinaryLabelDataset(
            df=repaired_df,
            label_names=[CONFIG["target_col"]],
            protected_attribute_names=[attr],
            favorable_label=0,
            unfavorable_label=1,
        )
        metric = BinaryLabelDatasetMetric(
            aif_ds,
            unprivileged_groups=unprivileged,
            privileged_groups=privileged,
        )

        dir_val = round(metric.disparate_impact(), 4)
        spd_val = round(metric.statistical_parity_difference(), 4)

        passes_dir = dir_val >= CONFIG["di_ratio_threshold"]
        passes_spd = abs(spd_val) <= CONFIG["spd_threshold"]

        rows.append({
            "repair_level"           : level,
            "protected_attribute"    : attr,
            "DIR"                    : dir_val,
            "SPD"                    : spd_val,
            "DIR ≥ 0.80 (4/5ths)"   : "✓ PASS" if passes_dir else "✗ FAIL",
            "|SPD| ≤ 0.05"          : "✓ PASS" if passes_spd else "✗ FAIL",
        })

        print(f"  {attr:<35} DIR={dir_val:.4f} "
              f"{'✓' if passes_dir else '✗'}  "
              f"SPD={spd_val:+.4f} "
              f"{'✓' if passes_spd else '✗'}")

    return rows


# =============================================================================
# 1E — LR BENCHMARK (PERFORMANCE AT EACH REPAIR LEVEL)
# =============================================================================

def lr_benchmark(X_train_woe: pd.DataFrame,
                 y_train: pd.Series,
                 X_val_woe: pd.DataFrame,
                 y_val: pd.Series,
                 repair_level: float) -> dict:
    """
    Train a simple Logistic Regression on WOE features and return
    AUC and KS on the validation set.

    This quick benchmark is used ONLY to populate the trade-off table —
    the full LR model is trained properly in Stage 2.
    """
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_train_woe)
    X_va   = scaler.transform(X_val_woe)

    lr = LogisticRegression(
        C=0.1,
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",
        random_state=CONFIG["random_state"],
    )
    lr.fit(X_tr, y_train)

    proba       = lr.predict_proba(X_va)[:, 1]
    auc         = roc_auc_score(y_val, proba)
    fpr, tpr, _ = roc_curve(y_val, proba)
    ks          = float(max(tpr - fpr))

    print(f"  [LR Benchmark] AUC={auc:.4f}  KS={ks:.4f}")
    return {"repair_level": repair_level,
            "AUC": round(auc, 4),
            "KS" : round(ks,  4)}


# =============================================================================
# 1F — TRADE-OFF TABLE + PLOT
# =============================================================================

def build_tradeoff_table(all_metrics: list,
                          all_perf: list) -> pd.DataFrame:
    """
    Merge fairness metrics and model performance into a single
    performance–fairness trade-off table.

    Fairness metrics are averaged across all protected attributes
    to give a single row per repair level.

    This table is the primary decision-support artefact for choosing
    the operating repair level before proceeding to Stage 2.
    """
    mdf = pd.DataFrame(all_metrics)
    pdf = pd.DataFrame(all_perf)

    agg = (mdf
           .groupby("repair_level")[["DIR", "SPD"]]
           .mean()
           .reset_index()
           .rename(columns={"DIR": "Mean DIR", "SPD": "Mean SPD"}))

    table = pdf.merge(agg, on="repair_level")

    # Annotate pass/fail columns
    table["DIR Pass (≥0.80)"] = table["Mean DIR"].ge(0.80).map(
        {True: "✓ PASS", False: "✗ FAIL"})
    table["|SPD| Pass (≤0.05)"] = table["Mean SPD"].abs().le(0.05).map(
        {True: "✓ PASS", False: "✗ FAIL"})

    print("\n[Stage 1F] ── Performance–Fairness Trade-off Table ──────────")
    print(table.to_string(index=False))

    # ── Save to CSV ───────────────────────────────────────────────────────────
    table.to_csv(CONFIG["output_dir"] + "reports/tradeoff_table.csv",
                 index=False)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle("Repair Level Trade-off: Performance vs Fairness",
                 fontsize=13, fontweight="bold")

    axes[0].plot(table["repair_level"], table["AUC"],
                 marker="o", color="steelblue", linewidth=2)
    axes[0].set_title("AUC vs Repair Level")
    axes[0].set_xlabel("Repair Level"); axes[0].set_ylabel("AUC")
    axes[0].set_ylim(0.5, 1.0)

    axes[1].plot(table["repair_level"], table["KS"],
                 marker="o", color="darkorange", linewidth=2)
    axes[1].set_title("KS Statistic vs Repair Level")
    axes[1].set_xlabel("Repair Level"); axes[1].set_ylabel("KS")

    axes[2].plot(table["repair_level"], table["Mean DIR"],
                 marker="o", color="green", linewidth=2, label="Mean DIR")
    axes[2].axhline(y=0.80, color="red", linestyle="--",
                    linewidth=1.5, label="4/5ths rule (0.80)")
    axes[2].set_title("Mean DI Ratio vs Repair Level")
    axes[2].set_xlabel("Repair Level"); axes[2].set_ylabel("Mean DIR")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(CONFIG["output_dir"] + "plots/tradeoff_repair_levels.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[Stage 1F] Plot saved → {CONFIG['output_dir']}plots/tradeoff_repair_levels.png")

    return table


# =============================================================================
# 1G — MASTER STAGE 1 RUNNER
# =============================================================================

def run_stage1(X_train: pd.DataFrame,
               X_val:   pd.DataFrame,
               X_test:  pd.DataFrame,
               y_train: pd.Series,
               y_val:   pd.Series,
               y_test:  pd.Series) -> tuple:
    """
    Orchestrates the full pre-processing fairness stage.

    Returns
    ───────
    stage1_results : dict keyed by repair_level containing
                     { X_train_woe, X_val_woe, X_test_woe,
                       y_train, binning_process, scaler,
                       fairness_metrics, performance }
    tradeoff_table : pd.DataFrame summary table
    best_level     : float — recommended repair level (highest DIR that
                             still keeps AUC within 2% of baseline)
    """
    repaired_sets = sweep_repair_levels(X_train, y_train)

    stage1_results = {}
    all_metrics    = []
    all_perf       = []

    for level, repaired_df in repaired_sets.items():
        print(f"\n{'='*65}")
        print(f"  Processing repair_level = {level}")
        print(f"{'='*65}")

        X_rep = repaired_df.drop(columns=[CONFIG["target_col"]])
        y_rep = repaired_df[CONFIG["target_col"]].astype(int)

        # ── WOE ──────────────────────────────────────────────────────────────
        bp      = fit_woe(X_rep, y_rep, CONFIG["categorical_cols"])
        iv_tbl  = print_iv_table(bp)
        iv_tbl.to_csv(
            CONFIG["output_dir"] + f"reports/iv_table_repair{level}.csv",
            index=False)

        X_train_woe = transform_woe(bp, X_rep)
        X_val_woe   = transform_woe(bp, X_val)
        X_test_woe  = transform_woe(bp, X_test)

        # ── Scaler (fitted on repaired WOE train) ────────────────────────────
        scaler = StandardScaler()
        scaler.fit(X_train_woe)

        # ── Fairness metrics ─────────────────────────────────────────────────
        print(f"\n[Stage 1D] Pre-model fairness metrics (repair={level}):")
        metrics = compute_pre_model_fairness(repaired_df, level)
        all_metrics.extend(metrics)

        # ── LR benchmark ─────────────────────────────────────────────────────
        perf = lr_benchmark(X_train_woe, y_rep, X_val_woe, y_val, level)
        all_perf.append(perf)

        stage1_results[level] = {
            "X_train_woe"    : X_train_woe,
            "X_val_woe"      : X_val_woe,
            "X_test_woe"     : X_test_woe,
            "y_train"        : y_rep,
            "binning_process": bp,
            "scaler"         : scaler,
            "iv_table"       : iv_tbl,
            "fairness"       : metrics,
            "performance"    : perf,
        }

    # ── Trade-off table ───────────────────────────────────────────────────────
    tradeoff = build_tradeoff_table(all_metrics, all_perf)

    # ── Select best repair level ──────────────────────────────────────────────
    baseline_auc = tradeoff.loc[
        tradeoff["repair_level"] == 0.0, "AUC"].values[0]
    candidates = tradeoff[
        (tradeoff["Mean DIR"] >= CONFIG["di_ratio_threshold"]) &
        (tradeoff["AUC"] >= baseline_auc - 0.02)
    ]
    if candidates.empty:
        best_level = 0.0
        print("\n[Stage 1G] ⚠ No repair level satisfies both constraints. "
              "Defaulting to 0.0 (baseline). Consider relaxing thresholds.")
    else:
        # Among passing candidates, pick the highest repair level
        # (maximum fairness within acceptable performance range)
        best_level = candidates["repair_level"].max()
        print(f"\n[Stage 1G] ✓ Recommended repair level: {best_level}")
        print(f"           Justification: highest repair level with "
              f"DIR ≥ 0.80 and AUC within 2% of baseline ({baseline_auc:.4f})")

    return stage1_results, tradeoff, best_level


# =============================================================================
# ENTRY POINT
# =============================================================================

# %% [markdown]
# ## Stage 2 — Model Building with In-Processing Fairness Constraints
# Logistic Regression · Random Forest + EG · XGBoost + EG · Adversarial Debiasing

# %%
# =============================================================================
# STAGE 2 — MODEL BUILDING WITH IN-PROCESSING FAIRNESS CONSTRAINTS
#
# Models
# ──────
#  A. Logistic Regression       — interpretable baseline; coefficients ARE the
#                                  explanation; no XAI needed
#  B. Random Forest             — with Exponentiated Gradient (Fairlearn)
#  C. XGBoost                   — with Exponentiated Gradient (Fairlearn)
#  D. Adversarial Debiasing NN  — AIF360 / TensorFlow in-processing
#
# All models use the WOE-transformed, repaired training data from Stage 1.
# Val set is used for hyperparameter decisions; test set is held out
# until the final evaluation in Stage 4.
# =============================================================================



# =============================================================================
# HELPER UTILITIES
# =============================================================================

def ks_stat(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """KS statistic = max separation between TPR and FPR curves."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    return float(np.max(tpr - fpr))


def eval_model(name: str,
               y_true: np.ndarray,
               y_proba: np.ndarray,
               threshold: float = 0.5) -> dict:
    """
    Compute AUC, KS, and classification report for a model.
    Returns a performance dict.
    """
    y_pred = (y_proba >= threshold).astype(int)
    auc    = roc_auc_score(y_true, y_proba)
    ks     = ks_stat(y_true, y_proba)

    print(f"\n{'─'*50}")
    print(f" {name}")
    print(f"{'─'*50}")
    print(f" AUC : {auc:.4f}  |  KS : {ks:.4f}")
    print(classification_report(
        y_true, y_pred,
        target_names=["Non-Default", "Default"],
        zero_division=0))

    return {"model": name, "AUC": round(auc, 4), "KS": round(ks, 4)}


def plot_roc(models_dict: dict,
             y_true: np.ndarray,
             title: str,
             save_path: str) -> None:
    """
    Overlay ROC curves for multiple models on one plot.

    models_dict: {model_name: y_proba_array}
    """
    plt.figure(figsize=(8, 6))
    for name, proba in models_dict.items():
        fpr, tpr, _ = roc_curve(y_true, proba)
        auc = roc_auc_score(y_true, proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(title); plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] ROC saved → {save_path}")


# =============================================================================
# 2A — LOGISTIC REGRESSION (INTERPRETABLE BASELINE)
# =============================================================================

def train_logistic_regression(X_train_woe: pd.DataFrame,
                               y_train: pd.Series,
                               X_val_woe: pd.DataFrame,
                               y_val: pd.Series,
                               scaler: StandardScaler) -> tuple:
    """
    Train a regularised Logistic Regression on WOE-encoded features.

    Why no XAI here
    ───────────────
    Logistic Regression is inherently interpretable.  Each coefficient β_i
    represents the change in log-odds of default per unit change in the
    WOE score of feature i.  The odds ratio exp(β_i) is the multiplicative
    factor applied to the odds of default.

    A positive coefficient → higher WOE of that feature → higher default
    probability.  Because WOE is already monotonically encoded, coefficients
    are directly comparable in magnitude.

    Regulatory note: LR coefficients satisfy the "right to explanation"
    requirement under GDPR Art. 22 and ECOA adverse action notice obligations
    without needing post-hoc explainability tools.

    Fairness note: Disparate impact was addressed in Stage 1 (pre-processing).
    No additional in-processing constraint is applied to LR.

    Returns
    ───────
    lr      : fitted LogisticRegression
    coef_df : coefficient/odds-ratio DataFrame
    perf    : performance dict
    """
    X_tr = scaler.transform(X_train_woe)
    X_va = scaler.transform(X_val_woe)

    lr = LogisticRegression(
        C=0.1,                          # L2 regularisation (penalise large coefs)
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",        # upweights the minority default class
        random_state=CONFIG["random_state"],
    )
    lr.fit(X_tr, y_train)

    # ── Coefficient table ─────────────────────────────────────────────────────
    coef_df = pd.DataFrame({
        "feature"    : X_train_woe.columns,
        "coefficient": lr.coef_[0],
        "odds_ratio" : np.exp(lr.coef_[0]),
    }).sort_values("coefficient", ascending=False).reset_index(drop=True)

    print("\n[2A] Logistic Regression Coefficient Table (sorted by coef):")
    print(coef_df.to_string(index=False))
    coef_df.to_csv(CONFIG["output_dir"] + "reports/lr_coefficients.csv",
                   index=False)

    # ── Coefficient plot ──────────────────────────────────────────────────────
    plt.figure(figsize=(10, max(6, len(coef_df) * 0.35)))
    colors = ["crimson" if c > 0 else "steelblue" for c in coef_df["coefficient"]]
    plt.barh(coef_df["feature"], coef_df["coefficient"], color=colors)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Log-Odds Coefficient")
    plt.title("Logistic Regression: Feature Coefficients\n"
              "(Red = increases default risk | Blue = decreases default risk)")
    plt.tight_layout()
    plt.savefig(CONFIG["output_dir"] + "plots/lr_coefficients.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    # ── Performance ───────────────────────────────────────────────────────────
    val_proba = lr.predict_proba(X_va)[:, 1]
    perf      = eval_model("Logistic Regression (Baseline)", y_val, val_proba)

    return lr, coef_df, perf, val_proba


# =============================================================================
# 2B — EXPONENTIATED GRADIENT (Fairlearn) on RF and XGB
# =============================================================================

def train_exponentiated_gradient(base_estimator,
                                  model_name: str,
                                  constraint_type: str,
                                  X_train_woe: pd.DataFrame,
                                  y_train: pd.Series,
                                  sensitive_features: pd.Series,
                                  X_val_woe: pd.DataFrame,
                                  y_val: pd.Series,
                                  sensitive_val: pd.Series) -> tuple:
    """
    Wrap a base estimator (RF or XGB) in Fairlearn's ExponentiatedGradient
    to constrain training with a fairness objective.

    How Exponentiated Gradient works
    ──────────────────────────────────
    EG is a reduction approach: it converts the constrained fairness problem
    into a sequence of cost-sensitive classification problems.  At each
    iteration, it adjusts sample weights to penalise predictions that
    violate the fairness constraint, effectively forcing the model to
    trade off some accuracy for fairness.

    Constraints supported here
    ──────────────────────────
    "demographic_parity" — equal selection rate across groups
      P(Ŷ=1 | A=a) = P(Ŷ=1 | A=b)  ∀ a,b
      Use when: we want equal approval rates regardless of group membership.

    "equalized_odds" — equal TPR AND FPR across groups
      P(Ŷ=1 | Y=1, A=a) = P(Ŷ=1 | Y=1, A=b)  (equal TPR)
      P(Ŷ=1 | Y=0, A=a) = P(Ŷ=1 | Y=0, A=b)  (equal FPR)
      Use when: we want equal accuracy for defaulters AND non-defaulters
      across groups.  More conservative and legally defensible.

    Note: sensitive_features must be a 1-D array/Series aligned with X_train.
    For multi-attribute fairness, pass a composite sensitive feature
    (e.g. tuple of values across attributes).

    Parameters
    ──────────
    base_estimator  : unfitted sklearn-compatible estimator (RF or XGB)
    model_name      : string label for logging
    constraint_type : "demographic_parity" or "equalized_odds"
    sensitive_features : Series of sensitive attribute values (train)
    sensitive_val      : Series of sensitive attribute values (val)

    Returns
    ───────
    eg_model    : fitted ExponentiatedGradient
    perf        : performance dict
    val_proba   : predicted probabilities on val set
    """
    if constraint_type == "demographic_parity":
        constraint = DemographicParity()
        constraint_label = "Demographic Parity"
    elif constraint_type == "equalized_odds":
        constraint = EqualizedOdds()
        constraint_label = "Equalized Odds"
    else:
        raise ValueError(f"Unknown constraint: {constraint_type}")

    print(f"\n[2B] Training {model_name} + EG ({constraint_label})...")

    eg = ExponentiatedGradient(
        estimator=base_estimator,
        constraints=constraint,
        eps=0.01,          # allowed constraint violation (tighter = fairer)
        max_iter=50,       # maximum EG iterations
        nu=1e-6,           # convergence tolerance
    )

    eg.fit(
        X_train_woe,
        y_train,
        sensitive_features=sensitive_features,
    )

    # EG returns a randomised predictor — use predict_proba via interpolation
    val_proba = eg.predict_proba(X_val_woe)[:, 1]
    perf      = eval_model(
        f"{model_name} + EG ({constraint_label})", y_val, val_proba)

    # ── Fairlearn fairness metrics on validation set ──────────────────────────
    val_pred = (val_proba >= 0.5).astype(int)

    mf = MetricFrame(
        metrics={
            "selection_rate"   : selection_rate,
            "false_positive_rate": false_positive_rate,
            "false_negative_rate": false_negative_rate,
        },
        y_true=y_val,
        y_pred=val_pred,
        sensitive_features=sensitive_val,
    )
    print(f"\n[2B] {model_name} MetricFrame by sensitive group:")
    print(mf.by_group.to_string())

    dpd = demographic_parity_difference(
        y_val, val_pred, sensitive_features=sensitive_val)
    eod = equalized_odds_difference(
        y_val, val_pred, sensitive_features=sensitive_val)
    print(f"\n[2B] Demographic Parity Difference : {dpd:.4f} "
          f"({'✓' if abs(dpd) <= CONFIG['spd_threshold'] else '✗'})")
    print(f"[2B] Equalized Odds Difference     : {eod:.4f} "
          f"({'✓' if abs(eod) <= CONFIG['eod_threshold'] else '✗'})")

    return eg, perf, val_proba


def build_sensitive_feature_column(X: pd.DataFrame) -> pd.Series:
    """
    Construct a composite sensitive feature column by concatenating
    all protected attribute values into a string tuple.

    Example: "FTHB=N|State=CA|Channel=R|Occ=P|Borrowers=1|ZIP=935"

    Fairlearn's EG works with a single sensitive_features array.
    The composite approach ensures all protected attributes are
    considered simultaneously during constraint enforcement.
    """
    attrs_present = [a for a in CONFIG["protected_attrs"] if a in X.columns]
    composite = X[attrs_present].astype(str).agg("|".join, axis=1)
    return composite


# =============================================================================
# 2C — ADVERSARIAL DEBIASING (AIF360 / TensorFlow)
# =============================================================================

def train_adversarial_debiasing(X_train_woe: pd.DataFrame,
                                 y_train: pd.Series,
                                 X_val_woe: pd.DataFrame,
                                 y_val: pd.Series,
                                 privileged_groups: list,
                                 unprivileged_groups: list,
                                 protected_attr: str) -> tuple:
    """
    Train AIF360's AdversarialDebiasing — a neural-network in-processing
    method that simultaneously minimises prediction error and makes it
    impossible for an adversary to infer the protected attribute.

    Architecture
    ────────────
    ┌─────────────────────────────────────────────────────────────┐
    │  Input Features (WOE-encoded)                               │
    │       │                                                     │
    │  ┌────▼────────────────────────────────────────────────┐   │
    │  │  PREDICTOR NETWORK (2-layer MLP)                    │   │
    │  │  → predicts P(Default | X)                          │   │
    │  └────┬────────────────────────────────────────────────┘   │
    │       │ predictions                                         │
    │  ┌────▼────────────────────────────────────────────────┐   │
    │  │  ADVERSARY NETWORK (1-layer MLP)                    │   │
    │  │  → predicts P(Protected Attribute | Ŷ)              │   │
    │  └────┬────────────────────────────────────────────────┘   │
    │       │ adversary gradients                                 │
    │  Objective:                                                 │
    │    Predictor: Maximise accuracy                             │
    │             − λ × adversary's ability to guess group       │
    │    Adversary: Maximise its own accuracy                     │
    └─────────────────────────────────────────────────────────────┘

    The predictor's weights are updated with gradients REVERSED from the
    adversary — this scrambles the internal representations to remove
    information the adversary could use to identify the protected class.

    Note: AdversarialDebiasing requires the AIF360 BinaryLabelDataset
    format for the training set.

    Parameters
    ──────────
    protected_attr     : single protected attribute to debias against
                         (run separately per attribute, or use composite)
    privileged_groups  : list of dicts {attr: privileged_value}
    unprivileged_groups: list of dicts {attr: unprivileged_value}

    Returns
    ───────
    ad_model  : fitted AdversarialDebiasing
    perf      : performance dict
    val_proba : predicted probabilities on val set
    """
    # ── Wrap training data into AIF360 format ─────────────────────────────────
    train_combined = X_train_woe.copy()
    train_combined[CONFIG["target_col"]] = y_train.values

    aif_train = BinaryLabelDataset(
        df=train_combined,
        label_names=[CONFIG["target_col"]],
        protected_attribute_names=[protected_attr],
        favorable_label=0,
        unfavorable_label=1,
    )

    # ── Build and train the adversarial model ─────────────────────────────────
    sess = tf.Session()
    ad = AdversarialDebiasing(
        privileged_groups=privileged_groups,
        unprivileged_groups=unprivileged_groups,
        scope_name="adversarial_debiasing",
        debias=True,           # True = apply adversarial debiasing
        sess=sess,
        num_epochs=50,
        batch_size=256,
        classifier_num_hidden_units=256,
        adversary_loss_weight=0.1,  # λ — trades accuracy vs fairness
        # Higher λ → stronger debiasing → more fairness, less accuracy
    )
    print(f"\n[2C] Training AdversarialDebiasing on attr={protected_attr}...")
    ad.fit(aif_train)

    # ── Predict on validation set ─────────────────────────────────────────────
    val_combined = X_val_woe.copy()
    val_combined[CONFIG["target_col"]] = y_val.values

    aif_val = BinaryLabelDataset(
        df=val_combined,
        label_names=[CONFIG["target_col"]],
        protected_attribute_names=[protected_attr],
        favorable_label=0,
        unfavorable_label=1,
    )

    val_pred_dataset = ad.predict(aif_val)
    val_proba        = val_pred_dataset.scores.ravel()
    perf             = eval_model("Adversarial Debiasing", y_val, val_proba)

    sess.close()
    tf.reset_default_graph()

    return ad, perf, val_proba


# =============================================================================
# 2D — MODEL COMPARISON TABLE
# =============================================================================

def build_model_comparison(perf_list: list,
                            fairness_list: list) -> pd.DataFrame:
    """
    Merge model performance and fairness metrics into a single
    comparison table for regulatory reporting.

    Columns: model | AUC | KS | DPD | EOD | Recommendation
    """
    perf_df     = pd.DataFrame(perf_list)
    fairness_df = pd.DataFrame(fairness_list)

    comparison  = perf_df.merge(fairness_df, on="model", how="left")

    # Flag models that pass BOTH fairness thresholds
    comparison["Fairness OK"] = (
        comparison["DPD"].abs().le(CONFIG["spd_threshold"]) &
        comparison["EOD"].abs().le(CONFIG["eod_threshold"])
    ).map({True: "✓ PASS", False: "✗ FAIL"})

    print("\n[Stage 2] ── Model Comparison Table ─────────────────────────")
    print(comparison.to_string(index=False))
    comparison.to_csv(CONFIG["output_dir"] + "reports/model_comparison.csv",
                      index=False)
    return comparison


# =============================================================================
# 2E — MASTER STAGE 2 RUNNER
# =============================================================================

def run_stage2(stage1_data: dict) -> dict:
    """
    Train all four model families on the best repair-level WOE data.

    Parameters
    ──────────
    stage1_data : stage1_results[best_level] dict from Stage 1

    Returns
    ───────
    models : dict of {model_name: {"model": ..., "val_proba": ..., "perf": ...}}
    """
    X_train_woe = stage1_data["X_train_woe"]
    X_val_woe   = stage1_data["X_val_woe"]
    X_test_woe  = stage1_data["X_test_woe"]
    y_train     = stage1_data["y_train"]
    scaler      = stage1_data["scaler"]

    # Build composite sensitive feature column for EG
    sf_train = build_sensitive_feature_column(X_train_woe)
    sf_val   = build_sensitive_feature_column(X_val_woe)

    # For adversarial debiasing, use the most important single protected attr
    # Here: "First Time Homebuyer Flag" (confirmed protected attribute)
    ad_attr      = "First Time Homebuyer Flag"
    majority_val = int(X_train_woe[ad_attr].mode()[0]) \
                   if ad_attr in X_train_woe.columns else 0
    priv_groups   = [{ad_attr: majority_val}]
    unpriv_groups = [{ad_attr: 1 - majority_val}]

    models      = {}
    perf_list   = []
    fair_list   = []

    # ── 2A: Logistic Regression ───────────────────────────────────────────────
    lr, coef_df, lr_perf, lr_proba = train_logistic_regression(
        X_train_woe, y_train, X_val_woe, y_val, scaler)
    models["LR"] = {"model": lr, "val_proba": lr_proba, "perf": lr_perf,
                    "coef_df": coef_df, "scaler": scaler}
    perf_list.append(lr_perf)
    # LR fairness measured in Stage 4 (post-processing)
    fair_list.append({"model": "LR", "DPD": None, "EOD": None})

    # ── 2B-i: Random Forest + EG (Demographic Parity) ────────────────────────
    rf_base = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        class_weight="balanced",
        n_jobs=-1,
        random_state=CONFIG["random_state"],
    )
    eg_rf_dp, rf_dp_perf, rf_dp_proba = train_exponentiated_gradient(
        rf_base, "Random Forest", "demographic_parity",
        X_train_woe, y_train, sf_train,
        X_val_woe, y_val, sf_val)
    models["RF_EG_DP"] = {"model": eg_rf_dp, "val_proba": rf_dp_proba,
                           "perf": rf_dp_perf}
    perf_list.append(rf_dp_perf)

    # ── 2B-ii: XGBoost + EG (Equalized Odds) ─────────────────────────────────
    xgb_base = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=9,      # ~9:1 class imbalance
        use_label_encoder=False,
        eval_metric="auc",
        random_state=CONFIG["random_state"],
        n_jobs=-1,
    )
    eg_xgb_eo, xgb_eo_perf, xgb_eo_proba = train_exponentiated_gradient(
        xgb_base, "XGBoost", "equalized_odds",
        X_train_woe, y_train, sf_train,
        X_val_woe, y_val, sf_val)
    models["XGB_EG_EO"] = {"model": eg_xgb_eo, "val_proba": xgb_eo_proba,
                            "perf": xgb_eo_perf}
    perf_list.append(xgb_eo_perf)

    # ── Fairness metrics for RF and XGB ──────────────────────────────────────
    for name, proba in [("RF_EG_DP", rf_dp_proba), ("XGB_EG_EO", xgb_eo_proba)]:
        pred = (proba >= 0.5).astype(int)
        dpd  = demographic_parity_difference(y_val, pred, sensitive_features=sf_val)
        eod  = equalized_odds_difference(y_val, pred, sensitive_features=sf_val)
        fair_list.append({"model": name,
                          "DPD"  : round(dpd, 4),
                          "EOD"  : round(eod, 4)})

    # ── 2C: Adversarial Debiasing ─────────────────────────────────────────────
    if ad_attr in X_train_woe.columns:
        ad_model, ad_perf, ad_proba = train_adversarial_debiasing(
            X_train_woe, y_train,
            X_val_woe, y_val,
            priv_groups, unpriv_groups, ad_attr)
        models["AdversarialDebiasing"] = {"model": ad_model,
                                           "val_proba": ad_proba,
                                           "perf": ad_perf}
        perf_list.append(ad_perf)
        ad_pred = (ad_proba >= 0.5).astype(int)
        ad_dpd  = demographic_parity_difference(
            y_val, ad_pred, sensitive_features=sf_val)
        ad_eod  = equalized_odds_difference(
            y_val, ad_pred, sensitive_features=sf_val)
        fair_list.append({"model": "AdversarialDebiasing",
                          "DPD"  : round(ad_dpd, 4),
                          "EOD"  : round(ad_eod, 4)})

    # ── ROC comparison plot ───────────────────────────────────────────────────
    roc_dict = {
        "LR"                   : lr_proba,
        "RF + EG (DP)"         : rf_dp_proba,
        "XGB + EG (EO)"        : xgb_eo_proba,
    }
    if "AdversarialDebiasing" in models:
        roc_dict["Adversarial Debiasing"] = models["AdversarialDebiasing"]["val_proba"]

    plot_roc(roc_dict, y_val,
             title="Validation ROC — All Models (Fairness-Constrained)",
             save_path=CONFIG["output_dir"] + "plots/roc_all_models.png")

    # ── Comparison table ──────────────────────────────────────────────────────
    comparison = build_model_comparison(perf_list, fair_list)
    models["comparison"] = comparison

    return models


# =============================================================================
# ENTRY POINT
# =============================================================================

# %% [markdown]
# ## Stage 3 — Explainability (XAI)
# SHAP · Ceteris Paribus · LIME · Anchor · DiCE Counterfactuals · Counterfactual Fairness

# %%
# =============================================================================
# STAGE 3 — EXPLAINABILITY (XAI)
#
# Applied to BLACK-BOX models only (RF, XGB, Adversarial Debiasing).
# Logistic Regression is explained by its coefficients (Stage 2A).
#
# Structure
# ─────────
# 3A  Global Explainability
#       i.  SHAP Summary Plot       — overall feature importance ranking
#       ii. Ceteris Paribus Profile — individual feature effect on avg P(Default)
#
# 3B  Local Explainability  (single applicant — CONFIG["local_explain_idx"])
#       i.  SHAP Waterfall Plot     — per-feature contribution for one applicant
#       ii. LIME                    — local linear approximation
#       iii.Anchor                  — minimal sufficient condition for the decision
#
# 3C  Remedial Measures
#       i.  DiCE Counterfactuals    — "what if" scenarios that flip the decision
#       ii. Counterfactual Fairness — flip protected attribute, check if decision changes
# =============================================================================



# =============================================================================
# 3A — GLOBAL EXPLAINABILITY
# =============================================================================

# ── 3A-i: SHAP Summary Plot ───────────────────────────────────────────────────

def shap_summary(model,
                 X_train_woe: pd.DataFrame,
                 X_val_woe: pd.DataFrame,
                 model_name: str,
                 sample_size: int = 500) -> np.ndarray:
    """
    Compute and plot SHAP values for a tree-based model (RF or XGB).

    What SHAP shows (Global)
    ────────────────────────
    • Each dot = one observation.
    • X-axis = SHAP value (impact on log-odds of default).
    • Colour = feature value (red=high, blue=low).
    • Features are ranked top-to-bottom by mean |SHAP|.

    Interpretation
    ──────────────
    • A feature at the top dominates the model's decisions globally.
    • A red dot on the right → high feature value → increases default risk.
    • A blue dot on the left → low feature value → decreases default risk.

    For regulatory reporting: SHAP values satisfy the ECOA/Reg B requirement
    to identify the "principal reasons" for adverse action, ranked by impact.

    Parameters
    ──────────
    sample_size : number of background/explain samples (keep low for speed)

    Returns
    ───────
    shap_values : np.ndarray of shape (n_samples, n_features)
    """
    print(f"\n[3A-i] Computing SHAP values for {model_name}...")

    # Use TreeExplainer for RF/XGB (exact, fast)
    # Use KernelExplainer for AdversarialDebiasing (slow, model-agnostic)
    if hasattr(model, "estimators_") or hasattr(model, "get_booster"):
        # Unwrap EG's underlying estimator if needed
        base_model = getattr(model, "estimator_", model)
        explainer  = shap.TreeExplainer(base_model)
        # Sample from training set as background
        X_sample   = X_val_woe.sample(
            min(sample_size, len(X_val_woe)),
            random_state=CONFIG["random_state"])
        shap_values = explainer.shap_values(X_sample)
        # For binary classification RF returns list [class0, class1]
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    else:
        # Fallback: KernelExplainer (model-agnostic — works for AD NN)
        background = shap.kmeans(X_train_woe, 50)
        def predict_fn(x):
            return model.predict_proba(pd.DataFrame(x, columns=X_train_woe.columns))[:, 1]
        explainer   = shap.KernelExplainer(predict_fn, background)
        X_sample    = X_val_woe.sample(
            min(100, len(X_val_woe)),    # fewer samples for speed
            random_state=CONFIG["random_state"])
        shap_values = explainer.shap_values(X_sample, nsamples=100)

    # ── Beeswarm / Summary Plot ───────────────────────────────────────────────
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_sample,
        plot_type="dot",
        max_display=20,
        show=False,
    )
    plt.title(f"SHAP Summary Plot — {model_name}\n"
              f"(Feature importance ranked by mean |SHAP|)")
    plt.tight_layout()
    plt.savefig(
        CONFIG["output_dir"] + f"plots/shap_summary_{model_name.replace(' ','_')}.png",
        dpi=150, bbox_inches="tight")
    plt.close()

    # ── Bar Plot (mean |SHAP| per feature) ───────────────────────────────────
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_sample,
        plot_type="bar",
        max_display=20,
        show=False,
    )
    plt.title(f"SHAP Feature Importance (Bar) — {model_name}")
    plt.tight_layout()
    plt.savefig(
        CONFIG["output_dir"] + f"plots/shap_bar_{model_name.replace(' ','_')}.png",
        dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[3A-i] SHAP plots saved for {model_name}")
    return shap_values, X_sample, explainer


# ── 3A-ii: Ceteris Paribus Profile ───────────────────────────────────────────

def ceteris_paribus_profile(model,
                             X_val_woe: pd.DataFrame,
                             features_to_plot: list,
                             model_name: str,
                             n_sample: int = 200) -> None:
    """
    Ceteris Paribus (CP) Profile — shows how the average predicted
    default probability changes as a single feature varies, holding all
    others constant at their observed values.

    Also known as Partial Dependence Plot (PDP) at the individual level;
    when averaged across observations it becomes the standard PDP.

    How to read it
    ──────────────
    • X-axis = range of the feature's values.
    • Y-axis = mean predicted P(Default).
    • A rising curve → higher feature value → more default risk.
    • A flat curve   → feature has little marginal impact (conditional).

    Fairness use case: Plot P(Default) against each protected attribute.
    If the curve for "First Time Homebuyer Flag=Y" is substantially above
    "N", the model is penalising first-time buyers — flag for review.
    """
    print(f"\n[3A-ii] Computing Ceteris Paribus Profiles for {model_name}...")

    X_sub = X_val_woe.sample(
        min(n_sample, len(X_val_woe)),
        random_state=CONFIG["random_state"])

    n_cols  = min(3, len(features_to_plot))
    n_rows  = (len(features_to_plot) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(6 * n_cols, 4 * n_rows))
    axes = np.array(axes).flatten()

    for i, feat in enumerate(features_to_plot):
        if feat not in X_sub.columns:
            continue
        feat_range = np.linspace(X_sub[feat].min(), X_sub[feat].max(), 50)
        mean_probs = []

        for val in feat_range:
            X_modified       = X_sub.copy()
            X_modified[feat] = val
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_modified)[:, 1]
            else:
                # EG fallback
                probs = model.predict_proba(X_modified)[:, 1]
            mean_probs.append(probs.mean())

        axes[i].plot(feat_range, mean_probs, color="steelblue", linewidth=2)
        axes[i].axhline(y=X_val_woe.shape[0] / 10,
                         color="red", linestyle="--", alpha=0.5,
                         label="Overall default rate")
        axes[i].set_xlabel(feat, fontsize=9)
        axes[i].set_ylabel("Mean P(Default)")
        axes[i].set_title(f"CP Profile: {feat}", fontsize=10)
        axes[i].legend(fontsize=8)

    # Hide unused subplot axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Ceteris Paribus Profiles — {model_name}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_path = (CONFIG["output_dir"] +
                 f"plots/cp_profiles_{model_name.replace(' ','_')}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[3A-ii] CP profiles saved → {save_path}")


# =============================================================================
# 3B — LOCAL EXPLAINABILITY (single applicant)
# =============================================================================

def get_applicant(X_test_woe: pd.DataFrame,
                  y_test: pd.Series,
                  idx: int) -> tuple:
    """
    Extract a single applicant from the test set for local explanation.
    Prefer a predicted defaulter for interpretability demonstrations.
    """
    applicant    = X_test_woe.iloc[[idx]]
    true_label   = y_test.iloc[idx]
    print(f"\n[3B] Explaining applicant idx={idx} | True label={true_label}")
    return applicant, true_label


# ── 3B-i: SHAP Waterfall Plot ─────────────────────────────────────────────────

def shap_waterfall(explainer,
                   applicant: pd.DataFrame,
                   shap_values_sample: np.ndarray,
                   X_sample: pd.DataFrame,
                   model_name: str) -> None:
    """
    SHAP Waterfall Plot for a single applicant.

    Reads as: starting from E[f(X)] (the model's average prediction),
    each feature pushes the prediction up (red, increases default risk)
    or down (blue, decreases default risk) to arrive at the final
    predicted probability f(x) for this applicant.

    Adverse action interpretation
    ──────────────────────────────
    The top features in the waterfall are the "principal reasons for denial"
    required by ECOA/Reg B adverse action notices.  Each reason is backed
    by a quantified contribution — defensible in regulatory examination.
    """
    # Compute SHAP for the specific applicant
    if hasattr(explainer, "shap_values"):
        sv = explainer.shap_values(applicant)
        if isinstance(sv, list):
            sv = sv[1]
        shap_exp = shap.Explanation(
            values=sv[0],
            base_values=explainer.expected_value
                if not isinstance(explainer.expected_value, list)
                else explainer.expected_value[1],
            data=applicant.values[0],
            feature_names=applicant.columns.tolist(),
        )
    else:
        sv = explainer.shap_values(applicant, nsamples=200)
        shap_exp = shap.Explanation(
            values=sv[0],
            base_values=explainer.expected_value,
            data=applicant.values[0],
            feature_names=applicant.columns.tolist(),
        )

    plt.figure(figsize=(10, 7))
    shap.waterfall_plot(shap_exp, max_display=15, show=False)
    plt.title(f"SHAP Waterfall — {model_name}\n(Single Applicant Explanation)",
              fontsize=11)
    plt.tight_layout()
    save_path = (CONFIG["output_dir"] +
                 f"plots/shap_waterfall_{model_name.replace(' ','_')}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[3B-i] SHAP Waterfall saved → {save_path}")


# ── 3B-ii: LIME ───────────────────────────────────────────────────────────────

def lime_explain(model,
                 X_train_woe: pd.DataFrame,
                 applicant: pd.DataFrame,
                 model_name: str,
                 categorical_features_idx: list = None) -> None:
    """
    LIME (Local Interpretable Model-agnostic Explanations) for one applicant.

    How LIME works
    ──────────────
    1. Takes the applicant's feature vector.
    2. Generates ~5,000 perturbed versions (random noise around the applicant).
    3. Queries the black-box model for predictions on all perturbed samples.
    4. Fits a simple weighted linear model on these local samples,
       weighted by proximity to the original applicant.
    5. The linear model's coefficients are the LIME explanation.

    Output
    ──────
    Bar chart: each bar = a feature's contribution to P(Default).
    • Positive bar (orange) → feature increases predicted default probability.
    • Negative bar (blue)   → feature decreases predicted default probability.

    Advantage over SHAP
    ────────────────────
    LIME is completely model-agnostic — works even for the Adversarial
    Debiasing NN where TreeExplainer is not applicable.
    """
    print(f"\n[3B-ii] Computing LIME explanation for {model_name}...")

    def predict_fn(x):
        return model.predict_proba(
            pd.DataFrame(x, columns=X_train_woe.columns))

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_woe.values,
        feature_names=X_train_woe.columns.tolist(),
        class_names=["Non-Default", "Default"],
        categorical_features=categorical_features_idx or [],
        mode="classification",
        random_state=CONFIG["random_state"],
    )

    explanation = explainer.explain_instance(
        data_row=applicant.values[0],
        predict_fn=predict_fn,
        num_features=15,       # top 15 features
        num_samples=5000,      # perturbed samples
        labels=(1,),           # explain class=1 (Default)
    )

    # ── Save HTML report ──────────────────────────────────────────────────────
    html_path = (CONFIG["output_dir"] +
                 f"reports/lime_{model_name.replace(' ','_')}.html")
    explanation.save_to_file(html_path)
    print(f"[3B-ii] LIME HTML report saved → {html_path}")

    # ── Save as matplotlib figure ─────────────────────────────────────────────
    fig = explanation.as_pyplot_figure(label=1)
    fig.set_size_inches(10, 7)
    plt.title(f"LIME Explanation — {model_name}\n"
              f"Contribution to P(Default) for Single Applicant")
    plt.tight_layout()
    save_path = (CONFIG["output_dir"] +
                 f"plots/lime_{model_name.replace(' ','_')}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[3B-ii] LIME plot saved → {save_path}")

    # Print top reasons
    print(f"\n  Top LIME reasons for {model_name}:")
    for feat, weight in explanation.as_list(label=1)[:10]:
        direction = "↑ increases" if weight > 0 else "↓ decreases"
        print(f"  {direction} default risk: {feat} ({weight:+.4f})")


# ── 3B-iii: Anchor ────────────────────────────────────────────────────────────

def anchor_explain(model,
                   X_train_woe: pd.DataFrame,
                   applicant: pd.DataFrame,
                   model_name: str) -> None:
    """
    Anchor Explanation — finds the minimal set of feature conditions
    (the "anchor") that are sufficient to guarantee the model's decision
    with high precision, regardless of what other features look like.

    Example output
    ──────────────
    Anchor: Credit Score WOE <= -0.42 AND DTI Ratio WOE > 0.18
    Precision: 0.97  (97% of applicants matching this anchor are predicted as Default)
    Coverage : 0.12  (12% of all applicants match this anchor)

    Regulatory interpretation
    ─────────────────────────
    The anchor provides the "just sufficient condition" — the exact rules
    the model uses.  This is useful for:
    • Adverse action notices: "Your application was denied because [anchor]"
    • Model documentation: a rule-based summary of the black-box logic
    • Fairness: if anchor features include protected attributes, flag immediately

    Note: anchor-exp can be slow (~1 min) for tabular data.
    """
    print(f"\n[3B-iii] Computing Anchor explanation for {model_name}...")

    def predict_fn(x):
        return model.predict(pd.DataFrame(x, columns=X_train_woe.columns))

    explainer = anchor_tabular.AnchorTabularExplainer(
        class_names=["Non-Default", "Default"],
        feature_names=X_train_woe.columns.tolist(),
        train_data=X_train_woe.values,
        categorical_names={},    # pass dict of {col_idx: [categories]} if needed
    )

    explanation = explainer.explain_instance(
        applicant.values[0],
        predict_fn,
        threshold=0.95,    # required precision of the anchor
    )

    print(f"\n  Anchor: {' AND '.join(explanation.names())}")
    print(f"  Precision: {explanation.precision():.4f}")
    print(f"  Coverage : {explanation.coverage():.4f}")
    print(f"  (Precision = % of applicants matching anchor with same prediction)")
    print(f"  (Coverage  = % of all applicants that this anchor applies to)")

    # Save anchor as text report
    anchor_path = (CONFIG["output_dir"] +
                   f"reports/anchor_{model_name.replace(' ','_')}.txt")
    with open(anchor_path, "w") as f:
        f.write(f"Anchor Explanation — {model_name}\n")
        f.write("=" * 50 + "\n")
        f.write(f"Anchor    : {' AND '.join(explanation.names())}\n")
        f.write(f"Precision : {explanation.precision():.4f}\n")
        f.write(f"Coverage  : {explanation.coverage():.4f}\n")
    print(f"[3B-iii] Anchor report saved → {anchor_path}")


# =============================================================================
# 3C — REMEDIAL MEASURES
# =============================================================================

# ── 3C-i: DiCE Counterfactuals ───────────────────────────────────────────────

def dice_counterfactuals(model,
                          X_train_woe: pd.DataFrame,
                          y_train: pd.Series,
                          applicant: pd.DataFrame,
                          model_name: str) -> None:
    """
    DiCE (Diverse Counterfactual Explanations) — generates the minimal
    feature changes that would flip the model's decision from Deny to Approve.

    How DiCE works
    ──────────────
    DiCE uses a genetic algorithm / gradient-based search to find
    counterfactual instances that:
    1. Are predicted as the opposite class (Non-Default).
    2. Are as close as possible to the original applicant.
    3. Are diverse from each other (cover different possible pathways).
    4. Respect feasibility constraints (immutable features cannot change).

    Feasibility constraints
    ────────────────────────
    Immutable features (cannot be changed in real life):
      • First Time Homebuyer Flag  (historical fact)
      • Property State             (location is fixed for this loan)
      • Postal Code_modif          (location is fixed for this loan)
      • Number of Borrowers        (co-borrowers cannot be artificially added)

    These constraints ensure counterfactuals are actionable, not theoretical.

    Output
    ──────
    Table of N counterfactual scenarios showing which features to change
    and by how much to receive a different (approval) decision.
    """
    print(f"\n[3C-i] Computing DiCE counterfactuals for {model_name}...")

    # ── Prepare DiCE data object ──────────────────────────────────────────────
    train_combined = X_train_woe.copy()
    train_combined[CONFIG["target_col"]] = y_train.values

    continuous_features = [c for c in X_train_woe.columns
                           if c not in CONFIG["categorical_cols"]]

    d = dice_ml.Data(
        dataframe=train_combined,
        continuous_features=continuous_features,
        outcome_name=CONFIG["target_col"],
    )

    # ── Wrap model for DiCE ───────────────────────────────────────────────────
    def predict_fn(x):
        return model.predict_proba(x)

    m = dice_ml.Model(model=model, backend="sklearn")

    exp = Dice(d, m, method="random")   # "genetic" or "kdtree" also available

    # ── Generate counterfactuals ──────────────────────────────────────────────
    cf = exp.generate_counterfactuals(
        query_instances=applicant,
        total_CFs=CONFIG["num_counterfactuals"],
        desired_class="opposite",           # flip from Default → Non-Default
        features_to_vary=[c for c in X_train_woe.columns
                          if c not in CONFIG["immutable_features"]],
    )

    print(f"\n  DiCE Counterfactuals — {model_name}:")
    cf.visualize_as_dataframe(show_only_changes=True)

    # Save to CSV
    cf_df     = cf.cf_examples_list[0].final_cfs_df
    save_path = (CONFIG["output_dir"] +
                 f"reports/dice_cf_{model_name.replace(' ','_')}.csv")
    cf_df.to_csv(save_path, index=False)
    print(f"[3C-i] DiCE counterfactuals saved → {save_path}")


# ── 3C-ii: Counterfactual Fairness ───────────────────────────────────────────

def counterfactual_fairness_check(model,
                                   X_test_woe: pd.DataFrame,
                                   y_test: pd.Series,
                                   protected_attr: str,
                                   label_encoders: dict,
                                   n_sample: int = 1000) -> pd.DataFrame:
    """
    Counterfactual Fairness Test — for each test applicant, flip the
    protected attribute and check if the model's decision changes.

    Formally: A model is counterfactually fair if
      P(Ŷ_{A←a}(U) = y | X=x, A=a) = P(Ŷ_{A←a'}(U) = y | X=x, A=a)
    for all outcomes y and values a, a' of the protected attribute A.

    In practice: we measure what fraction of applicants get a DIFFERENT
    prediction when we flip their protected attribute, holding everything
    else constant.  A high flip rate → model uses the protected attribute
    (or its proxies) to make decisions → fairness concern.

    Parameters
    ──────────
    protected_attr : column to flip (must be binary-encoded 0/1 or 0/1/2 etc.)
    n_sample       : number of test applicants to test (for speed)

    Returns
    ───────
    results_df: DataFrame with original prediction, flipped prediction,
                and whether the decision changed, per applicant.
    """
    print(f"\n[3C-ii] Counterfactual Fairness Check: flipping '{protected_attr}'")

    X_sub = X_test_woe.sample(
        min(n_sample, len(X_test_woe)),
        random_state=CONFIG["random_state"])

    if protected_attr not in X_sub.columns:
        print(f"  ⚠ '{protected_attr}' not in WOE features — skipping.")
        return pd.DataFrame()

    # Original predictions
    orig_proba = model.predict_proba(X_sub)[:, 1]
    orig_pred  = (orig_proba >= 0.5).astype(int)

    # Flip protected attribute
    X_flipped = X_sub.copy()
    unique_vals = sorted(X_sub[protected_attr].unique())
    if len(unique_vals) == 2:
        flip_map = {unique_vals[0]: unique_vals[1],
                    unique_vals[1]: unique_vals[0]}
        X_flipped[protected_attr] = X_sub[protected_attr].map(flip_map)
    else:
        print(f"  ⚠ '{protected_attr}' has >2 values — using max-flip.")
        X_flipped[protected_attr] = unique_vals[-1] - X_sub[protected_attr]

    flipped_proba = model.predict_proba(X_flipped)[:, 1]
    flipped_pred  = (flipped_proba >= 0.5).astype(int)

    # Did the decision change?
    decision_flipped = (orig_pred != flipped_pred)
    flip_rate        = decision_flipped.mean()

    print(f"\n  Protected attribute flipped: {protected_attr}")
    print(f"  Flip rate (decision changed): {flip_rate:.2%}")
    print(f"  → {'⚠ CONCERN' if flip_rate > 0.05 else '✓ OK'}: "
          f"{'Model decisions depend on the protected attribute.' if flip_rate > 0.05 else 'Model is largely counterfactually fair for this attribute.'}")

    results_df = pd.DataFrame({
        "orig_proba"     : orig_proba,
        "flipped_proba"  : flipped_proba,
        "orig_pred"      : orig_pred,
        "flipped_pred"   : flipped_pred,
        "decision_changed": decision_flipped.astype(int),
    })

    save_path = (CONFIG["output_dir"] +
                 f"reports/cf_fairness_{protected_attr.replace(' ','_')}.csv")
    results_df.to_csv(save_path, index=False)
    print(f"[3C-ii] Results saved → {save_path}")

    return results_df, flip_rate


# =============================================================================
# 3D — MASTER STAGE 3 RUNNER
# =============================================================================

def run_stage3(stage1_data: dict,
               stage2_models: dict,
               label_encoders: dict,
               y_val: pd.Series,
               y_test: pd.Series) -> None:
    """
    Run all XAI analyses for the primary black-box model (XGB + EG).

    The XGB + EG (Equalized Odds) model is used as the primary black-box
    because: (a) it has the best AUC among constrained models, (b) the
    Equalized Odds constraint is the most legally defensible for credit
    risk (equal TPR/FPR across groups).

    SHAP + CP profiles are also run for RF + EG for comparison.
    """
    X_train_woe = stage1_data["X_train_woe"]
    X_val_woe   = stage1_data["X_val_woe"]
    X_test_woe  = stage1_data["X_test_woe"]
    y_train     = stage1_data["y_train"]

    xgb_model   = stage2_models["XGB_EG_EO"]["model"]
    rf_model    = stage2_models["RF_EG_DP"]["model"]

    # ── Top features for CP profiles (use WOE column names directly) ──────────
    top_features_for_cp = [
        "Credit Score",
        "Original Debt-to-Income (DTI) Ratio",
        "Original Loan-to-Value (LTV)",
        "Original Interest Rate",
        "Original UPB",
    ] + [f for f in CONFIG["protected_attrs"] if f in X_val_woe.columns]
    top_features_for_cp = [f for f in top_features_for_cp
                            if f in X_val_woe.columns]

    # ── Local applicant ───────────────────────────────────────────────────────
    applicant, true_label = get_applicant(
        X_test_woe, y_test, CONFIG["local_explain_idx"])

    # ── Categorical feature indices (for LIME) ────────────────────────────────
    cat_idx = [i for i, c in enumerate(X_train_woe.columns)
               if c in CONFIG["categorical_cols"]]

    print("\n" + "="*65)
    print("  STAGE 3 — EXPLAINABILITY : XGBoost + EG (Primary Model)")
    print("="*65)

    # ── 3A-i: SHAP Summary ────────────────────────────────────────────────────
    xgb_shap_vals, xgb_X_sample, xgb_explainer = shap_summary(
        xgb_model, X_train_woe, X_val_woe, "XGBoost_EG_EO")

    # Also run SHAP for RF
    rf_shap_vals, rf_X_sample, rf_explainer = shap_summary(
        rf_model, X_train_woe, X_val_woe, "RF_EG_DP")

    # ── 3A-ii: Ceteris Paribus ────────────────────────────────────────────────
    ceteris_paribus_profile(xgb_model, X_val_woe,
                             top_features_for_cp, "XGBoost_EG_EO")

    # ── 3B-i: SHAP Waterfall ──────────────────────────────────────────────────
    shap_waterfall(xgb_explainer, applicant,
                   xgb_shap_vals, xgb_X_sample, "XGBoost_EG_EO")

    # ── 3B-ii: LIME ───────────────────────────────────────────────────────────
    lime_explain(xgb_model, X_train_woe, applicant,
                 "XGBoost_EG_EO", cat_idx)

    # ── 3B-iii: Anchor ────────────────────────────────────────────────────────
    anchor_explain(xgb_model, X_train_woe, applicant, "XGBoost_EG_EO")

    # ── 3C-i: DiCE Counterfactuals ────────────────────────────────────────────
    dice_counterfactuals(xgb_model, X_train_woe, y_train,
                         applicant, "XGBoost_EG_EO")

    # ── 3C-ii: Counterfactual Fairness (all protected attrs) ──────────────────
    cf_results = {}
    for attr in CONFIG["protected_attrs"]:
        if attr in X_test_woe.columns:
            cf_df, flip_rate = counterfactual_fairness_check(
                xgb_model, X_test_woe, y_test, attr, label_encoders)
            cf_results[attr] = {"df": cf_df, "flip_rate": flip_rate}

    # ── Summary of counterfactual fairness ────────────────────────────────────
    print("\n[3C-ii] Counterfactual Fairness Summary:")
    print(f"{'Protected Attribute':<40} {'Flip Rate':>12} {'Status':>10}")
    print("─" * 65)
    for attr, res in cf_results.items():
        status = "⚠ CONCERN" if res["flip_rate"] > 0.05 else "✓ OK"
        print(f"{attr:<40} {res['flip_rate']:>11.2%} {status:>10}")

    print("\n[Stage 3] Complete. Proceed to Stage 4 (Post-Processing & Evaluation).")


# =============================================================================
# ENTRY POINT
# =============================================================================

# %% [markdown]
# ## Stage 4 — Post-Processing Fairness & Final Evaluation
# ThresholdOptimizer · Pareto Frontier · Radar Chart · OpenXAI Fidelity & Stability

# %%
# =============================================================================
# STAGE 4 — POST-PROCESSING FAIRNESS & FINAL EVALUATION
#
# Structure
# ─────────
# 4A  ThresholdOptimizer (Fairlearn)
#       — If XGB remains biased after in-processing, find group-specific
#         classification thresholds that satisfy the fairness constraint.
#
# 4B  Fairness Metrics Trade-off Dashboard
#       — Comprehensive table: AUC, KS, DPD, EOD, DIR, SPD across all models
#       — Visualise the performance–fairness Pareto frontier
#
# 4C  OpenXAI Fidelity & Stability Evaluation
#       — Fidelity : do the explanations (SHAP/LIME) accurately reflect
#                    the model's actual behaviour?
#       — Stability: are explanations consistent across similar applicants?
#                    (critical for regulatory "honest denial reasons" obligation)
# =============================================================================



# =============================================================================
# 4A — THRESHOLD OPTIMIZER
# =============================================================================

def run_threshold_optimizer(xgb_model,
                              X_train_woe: pd.DataFrame,
                              y_train: pd.Series,
                              X_test_woe: pd.DataFrame,
                              y_test: pd.Series,
                              sf_train: pd.Series,
                              sf_test: pd.Series) -> tuple:
    """
    Apply Fairlearn's ThresholdOptimizer as a post-processing fix.

    When to use
    ───────────
    Even after Exponentiated Gradient in-processing, residual bias can
    remain — especially when the sensitive feature distribution is highly
    skewed (e.g. 92% Principal Residences in Occupancy Status).

    ThresholdOptimizer finds group-specific decision thresholds that
    jointly satisfy a fairness constraint, without retraining the model.

    How it works
    ────────────
    1. Takes the trained model's predicted probabilities.
    2. For each sensitive group, it searches for the threshold t_g such that
       applying (Ŷ = 1 if P(Default) >= t_g) across all groups satisfies
       the chosen fairness constraint.
    3. The resulting classifier uses DIFFERENT thresholds for different groups.

    Fairness constraint: Equalized Odds
    ────────────────────────────────────
    Equal TPR and FPR across groups.
    This is the appropriate constraint for credit risk because:
    • Equal FPR → no group is disproportionately predicted as defaulters
                  when they are actually good borrowers (disparate denial)
    • Equal TPR → no group is disproportionately approved when they
                  will actually default (risk exposure equity)

    Regulatory note
    ───────────────
    Using group-specific thresholds is legally sensitive — in some
    jurisdictions it may constitute intentional discrimination.
    Document this choice carefully and validate with legal counsel.
    The benefit: it makes the bias fix explicit and auditable, rather
    than embedded opaquely in model weights.

    Parameters
    ──────────
    sf_train / sf_test : composite sensitive feature Series (from Stage 2)

    Returns
    ───────
    to_model  : fitted ThresholdOptimizer
    test_proba: interpolated probabilities on test set
    perf      : performance dict
    """
    print("\n[4A] Running ThresholdOptimizer on XGB + EG model...")

    to_model = ThresholdOptimizer(
        estimator=xgb_model,
        constraints="equalized_odds",
        objective="balanced_accuracy_score",
        predict_method="predict_proba",
    )

    to_model.fit(
        X=X_train_woe,
        y=y_train,
        sensitive_features=sf_train,
    )

    test_pred  = to_model.predict(X_test_woe, sensitive_features=sf_test)
    test_proba = to_model.predict_proba(X_test_woe, sensitive_features=sf_test)[:, 1]

    perf = eval_model("XGB + EG + ThresholdOptimizer", y_test, test_proba)

    # ── Fairness metrics post-threshold ──────────────────────────────────────
    dpd = demographic_parity_difference(
        y_test, test_pred, sensitive_features=sf_test)
    eod = equalized_odds_difference(
        y_test, test_pred, sensitive_features=sf_test)

    print(f"\n[4A] Post-ThresholdOptimizer Fairness:")
    print(f"  Demographic Parity Difference : {dpd:.4f} "
          f"({'✓' if abs(dpd) <= CONFIG['spd_threshold'] else '✗'})")
    print(f"  Equalized Odds Difference     : {eod:.4f} "
          f"({'✓' if abs(eod) <= CONFIG['eod_threshold'] else '✗'})")

    return to_model, test_proba, test_pred, perf, dpd, eod


# =============================================================================
# 4B — COMPREHENSIVE FAIRNESS METRICS TABLE & PARETO FRONTIER
# =============================================================================

def compute_all_test_metrics(models_dict: dict,
                              X_test_woe: pd.DataFrame,
                              y_test: pd.Series,
                              sf_test: pd.Series,
                              scaler_lr: StandardScaler) -> pd.DataFrame:
    """
    Evaluate ALL models on the held-out test set.

    Metrics computed
    ────────────────
    Performance : AUC, KS
    Fairness    : Demographic Parity Difference (DPD),
                  Equalized Odds Difference (EOD),
                  Mean DI Ratio (DIR) across protected attributes

    Each metric is computed on the TEST SET (first time test set is used).
    Val set was only used for model selection.

    Returns
    ───────
    DataFrame with one row per model.
    """
    rows = []

    for name, entry in models_dict.items():
        if name in ("comparison",):
            continue

        model  = entry["model"]
        scaler = entry.get("scaler")

        # ── Get predictions ───────────────────────────────────────────────────
        try:
            if scaler:
                X_sc = scaler.transform(X_test_woe)
                proba = model.predict_proba(X_sc)[:, 1]
            else:
                proba = model.predict_proba(X_test_woe)[:, 1]
        except Exception as e:
            print(f"  ⚠ Could not get probabilities for {name}: {e}")
            continue

        pred = (proba >= 0.5).astype(int)

        # ── Performance ───────────────────────────────────────────────────────
        auc = roc_auc_score(y_test, proba)
        ks  = ks_stat(y_test.values, proba)

        # ── Fairness ──────────────────────────────────────────────────────────
        dpd = demographic_parity_difference(
            y_test, pred, sensitive_features=sf_test)
        eod = equalized_odds_difference(
            y_test, pred, sensitive_features=sf_test)

        rows.append({
            "Model"     : name,
            "AUC"       : round(auc,  4),
            "KS"        : round(ks,   4),
            "DPD"       : round(dpd,  4),
            "EOD"       : round(eod,  4),
            "DPD Pass"  : "✓" if abs(dpd) <= CONFIG["spd_threshold"] else "✗",
            "EOD Pass"  : "✓" if abs(eod) <= CONFIG["eod_threshold"] else "✗",
        })

    results_df = pd.DataFrame(rows)
    print("\n[4B] ── Final Test Set Evaluation ───────────────────────────────")
    print(results_df.to_string(index=False))
    results_df.to_csv(
        CONFIG["output_dir"] + "reports/final_evaluation.csv", index=False)

    return results_df


def plot_pareto_frontier(results_df: pd.DataFrame) -> None:
    """
    Plot the Performance–Fairness Pareto frontier.

    X-axis : |DPD| (lower = fairer)
    Y-axis : AUC (higher = better performance)

    The Pareto frontier is the set of models where no other model is
    both fairer AND more accurate.  Models on the frontier represent
    optimal trade-offs; those below and to the right are dominated.

    Regulatory use: Use this chart to justify the chosen model to
    risk committees and regulators — it shows all trade-offs explicitly.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    for _, row in results_df.iterrows():
        ax.scatter(abs(row["DPD"]), row["AUC"], s=120, zorder=5)
        ax.annotate(
            row["Model"],
            (abs(row["DPD"]), row["AUC"]),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=8,
        )

    ax.axvline(x=CONFIG["spd_threshold"], color="red", linestyle="--",
               linewidth=1.5, label=f"|DPD| threshold = {CONFIG['spd_threshold']}")
    ax.set_xlabel("|Demographic Parity Difference| (lower = fairer)")
    ax.set_ylabel("AUC (higher = better)")
    ax.set_title("Performance–Fairness Pareto Frontier\n"
                 "(Ideal model: top-left corner)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(CONFIG["output_dir"] + "plots/pareto_frontier.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("[4B] Pareto frontier plot saved → "
          f"{CONFIG['output_dir']}plots/pareto_frontier.png")


def plot_fairness_radar(results_df: pd.DataFrame) -> None:
    """
    Radar chart showing AUC, KS, |DPD|, |EOD| for each model.
    Allows quick visual comparison of the performance–fairness trade-off
    across all models simultaneously.
    """
    from matplotlib.patches import FancyArrowPatch
    import matplotlib.patches as mpatches

    metrics = ["AUC", "KS", "|DPD|", "|EOD|"]
    df_plot = results_df.copy()
    df_plot["|DPD|"] = df_plot["DPD"].abs()
    df_plot["|EOD|"] = df_plot["EOD"].abs()

    # Normalise to [0, 1] for radar (invert fairness metrics: lower = better)
    for col in metrics:
        col_range = df_plot[col].max() - df_plot[col].min()
        if col_range > 0:
            df_plot[col + "_norm"] = (df_plot[col] - df_plot[col].min()) / col_range
        else:
            df_plot[col + "_norm"] = 0.5
    # Invert fairness metrics so that outward on radar = better
    df_plot["|DPD|_norm"] = 1 - df_plot["|DPD|_norm"]
    df_plot["|EOD|_norm"] = 1 - df_plot["|EOD|_norm"]

    N = len(metrics)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors  = plt.cm.tab10.colors

    for i, (_, row) in enumerate(df_plot.iterrows()):
        values = [row[m + "_norm"] for m in metrics]
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=row["Model"], color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(["AUC ↑", "KS ↑", "|DPD| ↓\n(inverted)", "|EOD| ↓\n(inverted)"],
                        fontsize=10)
    ax.set_title("Model Comparison Radar\n"
                 "(Outward = better on all axes)", fontsize=12, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(CONFIG["output_dir"] + "plots/fairness_radar.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("[4B] Radar chart saved → "
          f"{CONFIG['output_dir']}plots/fairness_radar.png")


# =============================================================================
# 4C — OpenXAI: FIDELITY & STABILITY
# =============================================================================

def openxai_fidelity_stability(model,
                                 X_train_woe: pd.DataFrame,
                                 X_test_woe: pd.DataFrame,
                                 y_test: pd.Series,
                                 shap_vals: np.ndarray,
                                 lime_explanations: list,
                                 model_name: str) -> pd.DataFrame:
    """
    Use OpenXAI to evaluate whether the explanations given to customers
    for loan denials are HONEST and CONSISTENT.

    Why this matters
    ────────────────
    A bank can be liable if:
    • Explanations given for denial don't reflect what the model actually did
      (low FIDELITY — the explanation is misleading).
    • Two similar applicants receive different explanations for the same
      decision (low STABILITY — the explanation is arbitrary/inconsistent).

    OpenXAI Metrics
    ───────────────
    1. Faithfulness (Fidelity)
       Measures how well the explanation predicts the model's output.
       Method: mask the top-k features identified by the explainer and
       measure the drop in model output.  High drop = high faithfulness.
       Metric: FA (Feature Agreement) or PGI (Prediction Gap on Important features)

    2. Stability
       Measures how consistent explanations are for similar inputs.
       Method: generate slightly perturbed versions of each test instance,
       compute explanations for each, and measure explanation similarity.
       Metric: RIS (Relative Input Stability) — higher = more stable

    3. Rank Agreement
       Do SHAP and LIME agree on which features matter most?
       Metric: Rank correlation between SHAP and LIME feature importance ranks.

    Note: OpenXAI API may differ slightly across versions.
    Adjust metric class names to match your installed version.

    Parameters
    ──────────
    shap_vals        : np.ndarray from Stage 3 SHAP computation
    lime_explanations: list of LIME explanation objects (one per test sample)

    Returns
    ───────
    openxai_results: DataFrame with fidelity and stability scores
    """
    try:
        from openxai import Evaluator
        from openxai.explainers import SHAPExplainer, LIMEExplainer

        print(f"\n[4C] Running OpenXAI Fidelity & Stability for {model_name}...")

        # ── Sample test set for evaluation (OpenXAI can be slow) ─────────────
        n_eval = min(200, len(X_test_woe))
        X_eval = X_test_woe.sample(n_eval, random_state=CONFIG["random_state"])
        y_eval = y_test.loc[X_eval.index]

        # ── Wrap model for OpenXAI ────────────────────────────────────────────
        def predict_fn(x: np.ndarray) -> np.ndarray:
            return model.predict_proba(
                pd.DataFrame(x, columns=X_train_woe.columns))[:, 1]

        # ── Instantiate OpenXAI explainers ────────────────────────────────────
        shap_explainer_oxi = SHAPExplainer(
            model=predict_fn,
            data=X_train_woe.values,
            feature_names=X_train_woe.columns.tolist(),
        )
        lime_explainer_oxi = LIMEExplainer(
            model=predict_fn,
            data=X_train_woe.values,
            feature_names=X_train_woe.columns.tolist(),
        )

        # ── Evaluator ─────────────────────────────────────────────────────────
        evaluator = Evaluator(
            model=predict_fn,
            data=X_eval.values,
            labels=y_eval.values,
            explainer=shap_explainer_oxi,
            feature_names=X_train_woe.columns.tolist(),
        )

        # ── Fidelity (PGI — Prediction Gap on Important features) ────────────
        pgi_score = evaluator.evaluate(metric="PGI", top_k=10)
        print(f"  SHAP PGI (Fidelity)  : {pgi_score:.4f}")
        print(f"    → {'✓ Good' if pgi_score > 0.05 else '⚠ Low'}: "
              f"{'Masking top features causes meaningful prediction drop.' if pgi_score > 0.05 else 'Explanations may not reflect key model drivers.'}")

        # ── Stability (RIS — Relative Input Stability) ────────────────────────
        ris_score = evaluator.evaluate(metric="RIS", top_k=10)
        print(f"  SHAP RIS (Stability) : {ris_score:.4f}")
        print(f"    → {'✓ Good' if ris_score < 0.1 else '⚠ High'}: "
              f"{'Explanations are consistent across similar inputs.' if ris_score < 0.1 else 'Explanations vary too much for similar applicants — review.'}")

        # ── Rank Agreement (SHAP vs LIME) ─────────────────────────────────────
        shap_importance = np.abs(shap_vals).mean(axis=0)
        # Re-run LIME on eval set for rank comparison
        lime_evaluator  = Evaluator(
            model=predict_fn,
            data=X_eval.values,
            labels=y_eval.values,
            explainer=lime_explainer_oxi,
            feature_names=X_train_woe.columns.tolist(),
        )
        lime_importance = lime_evaluator.get_feature_importance().mean(axis=0)

        shap_ranks = pd.Series(shap_importance,
                                index=X_train_woe.columns).rank(ascending=False)
        lime_ranks = pd.Series(lime_importance,
                                index=X_train_woe.columns).rank(ascending=False)
        rank_corr  = shap_ranks.corr(lime_ranks, method="spearman")

        print(f"\n  SHAP vs LIME Rank Agreement (Spearman ρ): {rank_corr:.4f}")
        print(f"    → {'✓ Good' if rank_corr > 0.7 else '⚠ Low'}: "
              f"{'SHAP and LIME agree on key feature rankings.' if rank_corr > 0.7 else 'SHAP and LIME disagree — investigate which is more faithful.'}")

        results_df = pd.DataFrame({
            "Model"              : [model_name],
            "SHAP PGI (Fidelity)": [round(pgi_score, 4)],
            "SHAP RIS (Stability)": [round(ris_score, 4)],
            "SHAP–LIME Rank ρ"   : [round(rank_corr, 4)],
            "Fidelity OK"        : ["✓" if pgi_score > 0.05 else "✗"],
            "Stability OK"       : ["✓" if ris_score < 0.10  else "✗"],
        })

        results_df.to_csv(
            CONFIG["output_dir"] + "reports/openxai_evaluation.csv",
            index=False)
        print(f"\n[4C] OpenXAI results saved → "
              f"{CONFIG['output_dir']}reports/openxai_evaluation.csv")

        return results_df

    except ImportError:
        print("\n[4C] ⚠ openxai not installed. "
              "Run: pip install openxai\n"
              "    Skipping fidelity/stability evaluation.")
        return pd.DataFrame()


# =============================================================================
# 4D — FINAL SUMMARY REPORT
# =============================================================================

def print_final_summary(tradeoff_table: pd.DataFrame,
                         model_results: pd.DataFrame,
                         openxai_results: pd.DataFrame,
                         best_level: float,
                         best_model: str) -> None:
    """
    Print a concise final summary covering:
    • Chosen repair level and justification
    • Best-performing fairness-compliant model
    • Whether fairness thresholds are met
    • XAI fidelity/stability pass/fail
    • Key recommendations for risk/legal teams
    """
    print("\n" + "="*70)
    print("  FAIRNESS AUDIT PIPELINE — FINAL SUMMARY")
    print("="*70)

    print(f"\n  Dataset       : Freddie Mac SFL — 100,000 loans | "
          f"Default rate: 10.2%")
    print(f"  Target        : {CONFIG['target_col']}")
    print(f"  Protected     : {', '.join(CONFIG['protected_attrs'])}")

    print(f"\n  Pre-Processing")
    print(f"  ─────────────")
    print(f"  Repair Level  : {best_level}  "
          f"(chosen as highest repair satisfying DIR≥0.80 within 2% AUC)")

    best_row = tradeoff_table[
        tradeoff_table["repair_level"] == best_level].iloc[0]
    print(f"  Baseline AUC  : {tradeoff_table.loc[tradeoff_table['repair_level']==0.0,'AUC'].values[0]}")
    print(f"  Repaired AUC  : {best_row['AUC']}  "
          f"(Δ = {best_row['AUC'] - tradeoff_table.loc[tradeoff_table['repair_level']==0.0,'AUC'].values[0]:.4f})")
    print(f"  Mean DIR      : {best_row['Mean DIR']:.4f}  "
          f"({'✓ PASS' if best_row['Mean DIR'] >= 0.80 else '✗ FAIL'})")

    print(f"\n  Model Selection")
    print(f"  ──────────────")
    if not model_results.empty:
        best = model_results[
            model_results["Model"] == best_model] if best_model in \
            model_results["Model"].values else model_results.head(1)
        for _, r in best.iterrows():
            print(f"  Best model    : {r['Model']}")
            print(f"  AUC           : {r['AUC']}  |  KS: {r['KS']}")
            print(f"  DPD           : {r['DPD']} ({r['DPD Pass']})")
            print(f"  EOD           : {r['EOD']} ({r['EOD Pass']})")

    if not openxai_results.empty:
        print(f"\n  XAI Integrity")
        print(f"  ─────────────")
        r = openxai_results.iloc[0]
        print(f"  SHAP Fidelity (PGI) : {r['SHAP PGI (Fidelity)']}  {r['Fidelity OK']}")
        print(f"  SHAP Stability (RIS): {r['SHAP RIS (Stability)']}  {r['Stability OK']}")
        print(f"  SHAP–LIME Agreement : {r['SHAP–LIME Rank ρ']}")

    print(f"\n  Recommendations")
    print(f"  ──────────────")
    print(f"  1. Proceed with repair_level={best_level}; "
          f"document the performance-fairness trade-off for model risk review.")
    print(f"  2. Use ThresholdOptimizer as post-processing safety net if "
          f"residual EOD remains above {CONFIG['eod_threshold']}.")
    print(f"  3. SHAP Waterfall + LIME bar charts are ready for ECOA/Reg B "
          f"adverse action notice generation.")
    print(f"  4. Counterfactual fairness results: review any protected attribute "
          f"with flip rate > 5% with Fair Lending counsel.")
    print(f"  5. Re-run this pipeline quarterly as new origination data arrives.")

    print("\n" + "="*70)
    print("  Audit Complete. All reports saved to: " + CONFIG["output_dir"])
    print("="*70)


# =============================================================================
# 4E — MASTER STAGE 4 RUNNER
# =============================================================================

def run_stage4(stage1_data: dict,
               stage1_results: dict,
               stage2_models: dict,
               tradeoff_table: pd.DataFrame,
               best_level: float,
               y_train: pd.Series,
               y_val: pd.Series,
               y_test: pd.Series) -> None:
    """
    Orchestrate the full post-processing, evaluation and integrity check.
    """

    X_train_woe = stage1_data["X_train_woe"]
    X_val_woe   = stage1_data["X_val_woe"]
    X_test_woe  = stage1_data["X_test_woe"]
    y_train_rep = stage1_data["y_train"]
    scaler      = stage1_data["scaler"]

    sf_train = build_sensitive_feature_column(X_train_woe)
    sf_val   = build_sensitive_feature_column(X_val_woe)
    sf_test  = build_sensitive_feature_column(X_test_woe)

    xgb_model = stage2_models["XGB_EG_EO"]["model"]

    # ── 4A: ThresholdOptimizer ────────────────────────────────────────────────
    to_model, to_proba, to_pred, to_perf, to_dpd, to_eod = \
        run_threshold_optimizer(
            xgb_model,
            X_train_woe, y_train_rep,
            X_test_woe,  y_test,
            sf_train,    sf_test)

    # Add ThresholdOptimizer results to models dict for evaluation
    stage2_models["XGB_EG_EO_ThreshOpt"] = {
        "model"    : to_model,
        "val_proba": to_proba,
        "perf"     : to_perf,
    }

    # ── 4B: Final test set evaluation + Pareto frontier ───────────────────────
    model_results = compute_all_test_metrics(
        stage2_models, X_test_woe, y_test, sf_test,
        scaler_lr=stage2_models["LR"]["scaler"])

    plot_pareto_frontier(model_results)
    plot_fairness_radar(model_results)

    # ── 4C: OpenXAI Fidelity & Stability ─────────────────────────────────────
    # Retrieve SHAP values computed in Stage 3 (pass through from caller)
    # If running standalone, recompute:
    shap_vals, X_sample, xgb_explainer = shap_summary(
        xgb_model, X_train_woe, X_val_woe,
        "XGBoost_EG_EO", sample_size=200)

    openxai_results = openxai_fidelity_stability(
        model=xgb_model,
        X_train_woe=X_train_woe,
        X_test_woe=X_test_woe,
        y_test=y_test,
        shap_vals=shap_vals,
        lime_explanations=[],   # populated if lime objects passed through
        model_name="XGBoost_EG_EO",
    )

    # ── 4D: Final summary ─────────────────────────────────────────────────────
    # Select best model (highest AUC with both DPD and EOD passing)
    passing = model_results[
        (model_results["DPD Pass"] == "✓") &
        (model_results["EOD Pass"] == "✓")
    ]
    best_model = (passing.sort_values("AUC", ascending=False).iloc[0]["Model"]
                  if not passing.empty
                  else model_results.sort_values("AUC", ascending=False).iloc[0]["Model"])

    print_final_summary(
        tradeoff_table, model_results, openxai_results,
        best_level, best_model)


# =============================================================================
# ENTRY POINT — RUN FULL PIPELINE END-TO-END
# =============================================================================

# %% [markdown]
# ## ▶ Run Full Pipeline

# %%
print("\n" + "★"*70)
print("  MORTGAGE FAIRNESS AUDIT — FULL PIPELINE")
print("★"*70)

# Stage 0 — Load & split
df, label_encoders = load_and_clean(CONFIG["data_path"])
X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

# Stage 1 — Pre-processing fairness + WOE
stage1_results, tradeoff_table, best_level = run_stage1(
    X_train, X_val, X_test, y_train, y_val, y_test)

# Stage 2 — Model training with fairness constraints
stage2_models = run_stage2(stage1_results[best_level])

# Stage 3 — Explainability + Remedial measures
run_stage3(stage1_results[best_level], stage2_models,
           label_encoders, y_val, y_test)

# Stage 4 — Post-processing + Final evaluation
run_stage4(
    stage1_data    = stage1_results[best_level],
    stage1_results = stage1_results,
    stage2_models  = stage2_models,
    tradeoff_table = tradeoff_table,
    best_level     = best_level,
    y_train        = y_train,
    y_val          = y_val,
    y_test         = y_test,
)
