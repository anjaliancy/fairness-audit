"""
Fairness preprocessing utilities for credit risk modeling.

This module applies AIF360's Disparate Impact Remover (DIR) before feature
engineering (for example WOE) and before model training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from aif360.algorithms.preprocessing import DisparateImpactRemover
from aif360.datasets import StandardDataset
from aif360.metrics import BinaryLabelDatasetMetric
from sklearn.metrics import roc_auc_score, roc_curve


@dataclass
class FairnessRepairResult:
    """Container for repaired datasets and fairness diagnostics."""

    train_repaired: pd.DataFrame
    test_repaired: Optional[pd.DataFrame]
    fairness_report_train_original: pd.DataFrame
    fairness_report_test_original: Optional[pd.DataFrame]
    fairness_report_train: pd.DataFrame
    fairness_report_test: Optional[pd.DataFrame]
    repair_level: float
    protected_attributes: List[str]


def _as_list(values: Any) -> List[Any]:
    if isinstance(values, (list, tuple, np.ndarray, pd.Series)):
        return list(values)
    return [values]


def _infer_privileged_value(series: pd.Series) -> Any:
    # Fallback inference when privileged value is not explicitly supplied.
    if pd.api.types.is_bool_dtype(series):
        return True
    if pd.api.types.is_numeric_dtype(series):
        uniq = np.sort(series.dropna().unique())
        if len(uniq) == 0:
            raise ValueError("Cannot infer privileged value from an empty protected attribute.")
        return uniq[-1]

    mode = series.dropna().mode()
    if mode.empty:
        raise ValueError("Cannot infer privileged value from an empty protected attribute.")
    return mode.iloc[0]


def _build_standard_dataset(
    df: pd.DataFrame,
    label_name: str,
    protected_attributes: Sequence[str],
    favorable_label: Any,
    privileged_values: Optional[Dict[str, Any]] = None,
) -> Tuple[StandardDataset, Dict[str, List[Any]]]:
    df = df.copy()

    # AIF360 StandardDataset relies on NumPy dtype checks that can fail on
    # pandas extension dtypes such as StringDtype/Int64/boolean.
    extension_cols = [
        col
        for col in df.columns
        if pd.api.types.is_extension_array_dtype(df[col].dtype)
    ]
    for col in extension_cols:
        if pd.api.types.is_numeric_dtype(df[col].dtype):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = df[col].astype(object)

    if label_name not in df.columns:
        raise ValueError(f"Label column '{label_name}' is not present in DataFrame.")

    missing_attrs = [attr for attr in protected_attributes if attr not in df.columns]
    if missing_attrs:
        raise ValueError(f"Protected attributes not found in DataFrame: {missing_attrs}")

    privileged_values = privileged_values or {}
    privileged_classes: List[List[Any]] = []
    raw_resolved_privileged_values: Dict[str, List[Any]] = {}

    for attr in protected_attributes:
        if attr in privileged_values:
            values = _as_list(privileged_values[attr])
        else:
            values = [_infer_privileged_value(df[attr])]
        raw_resolved_privileged_values[attr] = values

    favorable_class_for_dataset = favorable_label

    # StandardDataset requires numerical values. Encode protected attributes first
    # so privileged group semantics remain explicit.
    for attr in protected_attributes:
        if pd.api.types.is_numeric_dtype(df[attr].dtype):
            df[attr] = pd.to_numeric(df[attr], errors="coerce")
            transformed_privileged = [float(v) for v in raw_resolved_privileged_values[attr]]
        else:
            privileged_set = set(raw_resolved_privileged_values[attr])
            df[attr] = df[attr].isin(privileged_set).astype(float)
            transformed_privileged = [1.0]
        privileged_classes.append(transformed_privileged)

    # Encode label if needed (for non-numeric labels).
    if pd.api.types.is_numeric_dtype(df[label_name].dtype):
        df[label_name] = pd.to_numeric(df[label_name], errors="coerce")
    else:
        df[label_name] = (df[label_name] == favorable_label).astype(float)
        favorable_class_for_dataset = 1.0

    # Encode all remaining non-numeric features to numeric category codes.
    for col in df.columns:
        if col == label_name or col in protected_attributes:
            continue
        if pd.api.types.is_numeric_dtype(df[col].dtype):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = pd.Categorical(df[col]).codes.astype(float)

    dataset = StandardDataset(
        df=df,
        label_name=label_name,
        favorable_classes=[favorable_class_for_dataset],
        protected_attribute_names=list(protected_attributes),
        privileged_classes=privileged_classes,
    )

    # Return raw privileged values so callers can safely reuse them on similarly
    # encoded/unencoded dataframes (for example, train -> test construction).
    return dataset, raw_resolved_privileged_values


def _dataset_to_dataframe(
    dataset: StandardDataset,
    feature_columns: Sequence[str],
    label_name: str,
) -> pd.DataFrame:
    features_df = pd.DataFrame(dataset.features, columns=feature_columns, index=None)
    labels_series = pd.Series(dataset.labels.ravel(), name=label_name)
    repaired_df = pd.concat([features_df, labels_series], axis=1)
    return repaired_df


def _fairness_report(
    dataset: StandardDataset,
    protected_attributes: Sequence[str],
) -> pd.DataFrame:
    records = []
    protected_attributes = list(protected_attributes)
    for idx, attr in enumerate(protected_attributes):
        privileged_values = dataset.privileged_protected_attributes[idx].tolist()
        unprivileged_values = dataset.unprivileged_protected_attributes[idx].tolist()

        privileged_groups = [{attr: val} for val in privileged_values]
        unprivileged_groups = [{attr: val} for val in unprivileged_values]

        if len(unprivileged_groups) == 0:
            # If only one group exists, fairness metrics are undefined.
            records.append(
                {
                    "protected_attribute": attr,
                    "disparate_impact_ratio": np.nan,
                    "statistical_parity_difference": np.nan,
                    "note": "Only one group present; fairness metrics undefined.",
                }
            )
            continue

        metric = BinaryLabelDatasetMetric(
            dataset,
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups,
        )

        records.append(
            {
                "protected_attribute": attr,
                "disparate_impact_ratio": metric.disparate_impact(),
                "statistical_parity_difference": metric.statistical_parity_difference(),
                "note": "",
            }
        )

    return pd.DataFrame(records)


def evaluate_disparate_impact_and_spd(
    df: pd.DataFrame,
    label_name: str,
    protected_attributes: Sequence[str],
    favorable_label: Any = 0,
    privileged_values: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Compute Disparate Impact ratio and Statistical Parity Difference from a dataframe.

    This can be used on both original and repaired data for side-by-side comparison.
    """
    dataset, resolved_privileged = _build_standard_dataset(
        df=df,
        label_name=label_name,
        protected_attributes=protected_attributes,
        favorable_label=favorable_label,
        privileged_values=privileged_values,
    )
    return _fairness_report(dataset, protected_attributes)


def apply_disparate_impact_repair(
    train_df: pd.DataFrame,
    label_name: str,
    protected_attributes: Sequence[str],
    test_df: Optional[pd.DataFrame] = None,
    repair_level: float = 0.5,
    favorable_label: Any = 0,
    privileged_values: Optional[Dict[str, Any]] = None,
    sequential_repair: bool = True,
) -> FairnessRepairResult:
    """
    Apply AIF360 Disparate Impact Remover before feature engineering/modeling.

    Parameters
    ----------
    train_df:
        Training dataframe containing features, target, and protected attributes.
    label_name:
        Target/label column name.
    protected_attributes:
        Protected attribute column names to repair.
    test_df:
        Optional holdout dataframe to transform with the same fitted repairer.
    repair_level:
        DIR repair level in [0, 1], where 0 = no repair and 1 = full repair.
    favorable_label:
        Favorable class used by `StandardDataset` (credit context often 0=non-default).
    privileged_values:
        Mapping like {"sex": 1, "race": "White"}. If omitted, values are inferred.
    sequential_repair:
        DIR supports one sensitive attribute at a time. If multiple attributes are
        supplied and this is True, repair is applied sequentially per attribute.

    Returns
    -------
    FairnessRepairResult
        Repaired train/test dataframes plus DI ratio and SPD diagnostics.
    """
    if not 0.0 <= repair_level <= 1.0:
        raise ValueError("repair_level must be between 0 and 1.")

    if len(protected_attributes) == 0:
        raise ValueError("At least one protected attribute must be specified.")

    protected_attributes = list(protected_attributes)

    train_dataset, resolved_privileged = _build_standard_dataset(
        df=train_df,
        label_name=label_name,
        protected_attributes=protected_attributes,
        favorable_label=favorable_label,
        privileged_values=privileged_values,
    )

    test_dataset = None
    if test_df is not None:
        test_dataset, _ = _build_standard_dataset(
            df=test_df,
            label_name=label_name,
            protected_attributes=protected_attributes,
            favorable_label=favorable_label,
            privileged_values=resolved_privileged,
        )

    if len(protected_attributes) > 1 and not sequential_repair:
        raise ValueError(
            "DisparateImpactRemover handles one sensitive attribute at a time. "
            "Use sequential_repair=True to apply it per protected attribute."
        )

    repaired_train = train_dataset
    repaired_test = test_dataset

    fairness_train_original = _fairness_report(train_dataset, protected_attributes)
    fairness_test_original = (
        _fairness_report(test_dataset, protected_attributes)
        if test_dataset is not None
        else None
    )

    attrs_to_repair = protected_attributes if sequential_repair else [protected_attributes[0]]
    for attr in attrs_to_repair:
        remover = DisparateImpactRemover(repair_level=repair_level, sensitive_attribute=attr)
        repaired_train = remover.fit_transform(repaired_train)
        if repaired_test is not None:
            # AIF360 DisparateImpactRemover provides fit_transform but not transform.
            repaired_test = remover.fit_transform(repaired_test)

    feature_columns = [col for col in train_df.columns if col != label_name]
    train_repaired_df = _dataset_to_dataframe(repaired_train, feature_columns, label_name)
    test_repaired_df = (
        _dataset_to_dataframe(repaired_test, feature_columns, label_name)
        if repaired_test is not None
        else None
    )

    fairness_train = _fairness_report(repaired_train, protected_attributes)
    fairness_test = (
        _fairness_report(repaired_test, protected_attributes)
        if repaired_test is not None
        else None
    )

    return FairnessRepairResult(
        train_repaired=train_repaired_df,
        test_repaired=test_repaired_df,
        fairness_report_train_original=fairness_train_original,
        fairness_report_test_original=fairness_test_original,
        fairness_report_train=fairness_train,
        fairness_report_test=fairness_test,
        repair_level=repair_level,
        protected_attributes=protected_attributes,
    )


def compare_model_performance(
    baseline_model: Any,
    repaired_model: Any,
    x_test_baseline: pd.DataFrame,
    x_test_repaired: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """
    Compare predictive power between baseline and repaired-data models.

    Works with logistic regression and black-box classifiers as long as they expose
    `predict_proba`, `decision_function`, or `predict`.
    """

    def _score_model(model: Any, x_data: pd.DataFrame) -> np.ndarray:
        if hasattr(model, "predict_proba"):
            return model.predict_proba(x_data)[:, 1]
        if hasattr(model, "decision_function"):
            return model.decision_function(x_data)
        return model.predict(x_data)

    def _ks_statistic(y_true: pd.Series, y_score: np.ndarray) -> float:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        return float(np.max(tpr - fpr))

    baseline_scores = _score_model(baseline_model, x_test_baseline)
    repaired_scores = _score_model(repaired_model, x_test_repaired)

    baseline_auc = roc_auc_score(y_test, baseline_scores)
    repaired_auc = roc_auc_score(y_test, repaired_scores)

    baseline_ks = _ks_statistic(y_test, baseline_scores)
    repaired_ks = _ks_statistic(y_test, repaired_scores)

    return pd.DataFrame(
        [
            {
                "model_variant": "baseline",
                "auc": baseline_auc,
                "ks": baseline_ks,
            },
            {
                "model_variant": "repaired",
                "auc": repaired_auc,
                "ks": repaired_ks,
            },
            {
                "model_variant": "delta_repaired_minus_baseline",
                "auc": repaired_auc - baseline_auc,
                "ks": repaired_ks - baseline_ks,
            },
        ]
    )


def sweep_repair_levels(
    train_df: pd.DataFrame,
    label_name: str,
    protected_attributes: Sequence[str],
    model_builder: Callable[[pd.DataFrame, pd.Series], Any],
    test_df: Optional[pd.DataFrame] = None,
    repair_levels: Sequence[float] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    feature_builder: Optional[
        Callable[
            [pd.DataFrame, pd.DataFrame, str],
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
        ]
    ] = None,
    favorable_label: Any = 0,
    privileged_values: Optional[Dict[str, Any]] = None,
    sequential_repair: bool = True,
    fairness_on: str = "test",
) -> pd.DataFrame:
    """
    Run Disparate Impact Remover sweep and return fairness/performance tradeoff table.

    Parameters
    ----------
    train_df, test_df:
        Original train/test dataframes before WOE or model training.
    label_name:
        Target column.
    protected_attributes:
        Protected attribute columns for fairness evaluation.
    model_builder:
        Callable that returns a fitted model, e.g. lambda X, y: LogisticRegression().fit(X, y).
    repair_levels:
        DIR repair levels to evaluate (0=no repair, 1=full repair).
    feature_builder:
        Optional callable to produce model-ready matrices from repaired data.
        Signature: (train_df, test_df, label_name) -> (X_train, X_test, y_train, y_test)
        Use this for WOE/fine-coarse pipelines.
    fairness_on:
        "train" or "test" indicating where DI/SPD should be reported.

    Returns
    -------
    pd.DataFrame
        One row per (repair_level, protected_attribute) with DI/SPD, AUC/KS and deltas.
    """
    if fairness_on not in {"train", "test"}:
        raise ValueError("fairness_on must be either 'train' or 'test'.")

    if test_df is None:
        eval_test_df = train_df.copy()
    else:
        eval_test_df = test_df.copy()

    if feature_builder is None:

        def _default_feature_builder(
            cur_train_df: pd.DataFrame,
            cur_test_df: pd.DataFrame,
            cur_label_name: str,
        ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
            x_train = cur_train_df.drop(columns=[cur_label_name])
            y_train = cur_train_df[cur_label_name]
            x_test = cur_test_df.drop(columns=[cur_label_name])
            y_test = cur_test_df[cur_label_name]
            return x_train, x_test, y_train, y_test

        builder = _default_feature_builder
    else:
        builder = feature_builder

    base_x_train, base_x_test, base_y_train, base_y_test = builder(
        train_df.copy(), eval_test_df.copy(), label_name
    )
    baseline_model = model_builder(base_x_train, base_y_train)

    baseline_scores = (
        baseline_model.predict_proba(base_x_test)[:, 1]
        if hasattr(baseline_model, "predict_proba")
        else (
            baseline_model.decision_function(base_x_test)
            if hasattr(baseline_model, "decision_function")
            else baseline_model.predict(base_x_test)
        )
    )
    base_fpr, base_tpr, _ = roc_curve(base_y_test, baseline_scores)
    baseline_auc = roc_auc_score(base_y_test, baseline_scores)
    baseline_ks = float(np.max(base_tpr - base_fpr))

    fairness_baseline_df = evaluate_disparate_impact_and_spd(
        df=train_df if fairness_on == "train" else eval_test_df,
        label_name=label_name,
        protected_attributes=protected_attributes,
        favorable_label=favorable_label,
        privileged_values=privileged_values,
    )

    baseline_di_map = dict(
        zip(
            fairness_baseline_df["protected_attribute"],
            fairness_baseline_df["disparate_impact_ratio"],
        )
    )
    baseline_spd_map = dict(
        zip(
            fairness_baseline_df["protected_attribute"],
            fairness_baseline_df["statistical_parity_difference"],
        )
    )

    rows = []
    for level in repair_levels:
        repaired = apply_disparate_impact_repair(
            train_df=train_df,
            test_df=eval_test_df,
            label_name=label_name,
            protected_attributes=protected_attributes,
            repair_level=float(level),
            favorable_label=favorable_label,
            privileged_values=privileged_values,
            sequential_repair=sequential_repair,
        )

        rep_train_df = repaired.train_repaired
        rep_test_df = repaired.test_repaired if repaired.test_repaired is not None else repaired.train_repaired

        rep_x_train, rep_x_test, rep_y_train, rep_y_test = builder(rep_train_df, rep_test_df, label_name)
        repaired_model = model_builder(rep_x_train, rep_y_train)

        repaired_scores = (
            repaired_model.predict_proba(rep_x_test)[:, 1]
            if hasattr(repaired_model, "predict_proba")
            else (
                repaired_model.decision_function(rep_x_test)
                if hasattr(repaired_model, "decision_function")
                else repaired_model.predict(rep_x_test)
            )
        )
        repaired_fpr, repaired_tpr, _ = roc_curve(rep_y_test, repaired_scores)
        repaired_auc = roc_auc_score(rep_y_test, repaired_scores)
        repaired_ks = float(np.max(repaired_tpr - repaired_fpr))

        fairness_current = (
            repaired.fairness_report_train
            if fairness_on == "train"
            else repaired.fairness_report_test
        )
        if fairness_current is None:
            fairness_current = repaired.fairness_report_train

        for _, metric_row in fairness_current.iterrows():
            attr = metric_row["protected_attribute"]
            di_ratio = metric_row["disparate_impact_ratio"]
            spd = metric_row["statistical_parity_difference"]
            base_di = baseline_di_map.get(attr, np.nan)
            base_spd = baseline_spd_map.get(attr, np.nan)

            rows.append(
                {
                    "repair_level": float(level),
                    "protected_attribute": attr,
                    "disparate_impact_ratio": di_ratio,
                    "statistical_parity_difference": spd,
                    "delta_di_vs_baseline": di_ratio - base_di,
                    "delta_spd_vs_baseline": spd - base_spd,
                    "auc": repaired_auc,
                    "ks": repaired_ks,
                    "delta_auc_vs_baseline": repaired_auc - baseline_auc,
                    "delta_ks_vs_baseline": repaired_ks - baseline_ks,
                }
            )

    result_df = pd.DataFrame(rows)
    result_df = result_df.sort_values(["repair_level", "protected_attribute"]).reset_index(drop=True)
    return result_df