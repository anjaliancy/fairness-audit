"""Explainability and fairness assessment utilities for tabular credit models.

This module provides lightweight wrappers around common XAI methods:
- Anchor explanations
- Counterfactual fairness checks
- DiCE counterfactuals
- LIME local explanations
- SHAP global/local attributions
- Ceteris Paribus profiles

Most methods rely on optional third-party libraries. Each function raises an
informative ImportError when its optional dependency is missing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


ArrayLike = Union[np.ndarray, pd.Series, List[float]]


@dataclass
class CounterfactualFairnessResult:
    """Container for counterfactual fairness summary outputs."""

    individual_deltas: pd.DataFrame
    summary: pd.DataFrame


@dataclass
class PostProcessingFairnessResult:
    """Container for threshold-optimization and fairness trade-off outputs."""

    threshold_optimizer: Any
    predictions: np.ndarray
    scores: np.ndarray
    tradeoff_summary: pd.DataFrame
    tradeoff_by_group: pd.DataFrame


def _to_dataframe(x: Union[pd.DataFrame, np.ndarray], columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        return x.copy()
    return pd.DataFrame(x, columns=columns)


def _predict_scores(model: Any, x: pd.DataFrame, positive_class: int = 1) -> np.ndarray:
    """Return probability scores if available, else fallback to predictions."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        proba = np.asarray(proba)
        if proba.ndim == 2 and proba.shape[1] > positive_class:
            return proba[:, positive_class]
        return proba.ravel()
    preds = model.predict(x)
    return np.asarray(preds).ravel()


def assess_anchor(
    model: Any,
    x_train: pd.DataFrame,
    x_instance: Union[pd.DataFrame, pd.Series, np.ndarray],
    feature_names: Optional[Sequence[str]] = None,
    class_names: Optional[Sequence[str]] = None,
    threshold: float = 0.95,
) -> Dict[str, Any]:
    """Generate an Anchor explanation for a single tabular instance.

    Parameters
    ----------
    model:
        Trained classifier with a ``predict`` method.
    x_train:
        Background data used to fit the anchor explainer.
    x_instance:
        One row to explain.
    feature_names:
        Optional feature name override.
    class_names:
        Optional class names for readability.
    threshold:
        Precision threshold used by Anchor.
    """
    try:
        from anchor import anchor_tabular
    except ImportError as exc:
        raise ImportError("anchor-exp is required for assess_anchor. Install with `pip install anchor-exp`.") from exc

    x_train_df = _to_dataframe(x_train, columns=feature_names)
    fnames = list(feature_names) if feature_names is not None else list(x_train_df.columns)

    if isinstance(x_instance, pd.Series):
        x0 = x_instance.to_frame().T
    else:
        x0 = _to_dataframe(x_instance, columns=fnames)

    explainer = anchor_tabular.AnchorTabularExplainer(
        class_names=list(class_names) if class_names is not None else ["class_0", "class_1"],
        feature_names=fnames,
        train_data=x_train_df.values,
    )

    explanation = explainer.explain_instance(
        x0.iloc[0].values,
        classifier_fn=lambda arr: np.asarray(model.predict(arr)),
        threshold=threshold,
    )

    return {
        "anchor": list(explanation.names()),
        "precision": float(explanation.precision()),
        "coverage": float(explanation.coverage()),
        "raw": explanation,
    }


def assess_counterfactual_fairness(
    model: Any,
    x: pd.DataFrame,
    sensitive_attributes: Sequence[str],
    perturbation_map: Optional[Dict[str, Iterable[Any]]] = None,
    positive_class: int = 1,
) -> CounterfactualFairnessResult:
    """Assess counterfactual fairness by perturbing sensitive attributes only.

    For each sensitive attribute, predictions are recalculated after changing
    only that attribute while keeping all other features fixed.

    Returns
    -------
    CounterfactualFairnessResult
        ``individual_deltas`` contains row-level prediction changes.
        ``summary`` contains aggregate absolute and signed changes.
    """
    x_df = _to_dataframe(x)
    base_scores = _predict_scores(model, x_df, positive_class=positive_class)

    rows: List[pd.DataFrame] = []
    for attr in sensitive_attributes:
        if attr not in x_df.columns:
            raise KeyError(f"Sensitive attribute '{attr}' is not in input dataframe.")

        values = sorted(x_df[attr].dropna().unique().tolist())
        if perturbation_map and attr in perturbation_map:
            values = list(perturbation_map[attr])

        for candidate in values:
            x_cf = x_df.copy()
            x_cf[attr] = candidate
            cf_scores = _predict_scores(model, x_cf, positive_class=positive_class)
            delta = cf_scores - base_scores

            tmp = pd.DataFrame(
                {
                    "row_id": np.arange(len(x_df)),
                    "sensitive_attribute": attr,
                    "counterfactual_value": candidate,
                    "base_score": base_scores,
                    "counterfactual_score": cf_scores,
                    "delta": delta,
                    "abs_delta": np.abs(delta),
                }
            )
            rows.append(tmp)

    deltas = pd.concat(rows, ignore_index=True)

    summary = (
        deltas.groupby("sensitive_attribute", as_index=False)
        .agg(
            mean_abs_delta=("abs_delta", "mean"),
            max_abs_delta=("abs_delta", "max"),
            mean_delta=("delta", "mean"),
            p95_abs_delta=("abs_delta", lambda s: float(np.quantile(s, 0.95))),
        )
        .sort_values("mean_abs_delta", ascending=False)
        .reset_index(drop=True)
    )

    return CounterfactualFairnessResult(individual_deltas=deltas, summary=summary)


def assess_dice(
    model: Any,
    x_train: pd.DataFrame,
    query_instance: Union[pd.DataFrame, pd.Series, Dict[str, Any]],
    target_name: str,
    continuous_features: Sequence[str],
    desired_class: Union[int, str] = "opposite",
    total_cfs: int = 4,
) -> Dict[str, Any]:
    """Generate DiCE counterfactual examples for a single instance."""
    try:
        import dice_ml
    except ImportError as exc:
        raise ImportError("dice-ml is required for assess_dice. Install with `pip install dice-ml`.") from exc

    data_df = _to_dataframe(x_train).copy()
    if target_name not in data_df.columns:
        raise KeyError(
            f"target_name '{target_name}' must exist in x_train for DiCE data object construction."
        )

    data = dice_ml.Data(
        dataframe=data_df,
        continuous_features=list(continuous_features),
        outcome_name=target_name,
    )
    m = dice_ml.Model(model=model, backend="sklearn")
    exp = dice_ml.Dice(data, m, method="random")

    if isinstance(query_instance, pd.Series):
        query_df = query_instance.to_frame().T
    elif isinstance(query_instance, dict):
        query_df = pd.DataFrame([query_instance])
    else:
        query_df = _to_dataframe(query_instance)

    cfs = exp.generate_counterfactuals(
        query_df,
        total_CFs=total_cfs,
        desired_class=desired_class,
    )

    return {"counterfactuals": cfs, "counterfactuals_df": cfs.cf_examples_list[0].final_cfs_df}


def assess_lime(
    model: Any,
    x_train: pd.DataFrame,
    x_instance: Union[pd.DataFrame, pd.Series, np.ndarray],
    feature_names: Optional[Sequence[str]] = None,
    class_names: Optional[Sequence[str]] = None,
    positive_class: int = 1,
    num_features: int = 10,
) -> Dict[str, Any]:
    """Generate a LIME explanation for one tabular instance."""
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError as exc:
        raise ImportError("lime is required for assess_lime. Install with `pip install lime`.") from exc

    x_train_df = _to_dataframe(x_train, columns=feature_names)
    fnames = list(feature_names) if feature_names is not None else list(x_train_df.columns)

    if isinstance(x_instance, pd.Series):
        x0 = x_instance.to_frame().T
    else:
        x0 = _to_dataframe(x_instance, columns=fnames)

    explainer = LimeTabularExplainer(
        training_data=x_train_df.values,
        feature_names=fnames,
        class_names=list(class_names) if class_names is not None else ["class_0", "class_1"],
        mode="classification",
    )

    exp = explainer.explain_instance(
        x0.iloc[0].values,
        predict_fn=lambda arr: np.asarray(model.predict_proba(arr)),
        num_features=num_features,
    )

    return {
        "weights": exp.as_list(label=positive_class),
        "raw": exp,
    }


def assess_shap(
    model: Any,
    x_background: pd.DataFrame,
    x_eval: pd.DataFrame,
    max_background: int = 500,
) -> Dict[str, Any]:
    """Compute SHAP values and global mean absolute attributions."""
    try:
        import shap
    except ImportError as exc:
        raise ImportError("shap is required for assess_shap. Install with `pip install shap`.") from exc

    x_bg = _to_dataframe(x_background)
    x_ev = _to_dataframe(x_eval, columns=x_bg.columns)
    if len(x_bg) > max_background:
        x_bg = x_bg.sample(max_background, random_state=42)

    def _predict_fn(arr: np.ndarray) -> np.ndarray:
        df = pd.DataFrame(arr, columns=x_bg.columns)
        return _predict_scores(model, df)

    explainer = shap.Explainer(_predict_fn, x_bg)
    shap_values = explainer(x_ev)

    vals = np.asarray(shap_values.values)
    if vals.ndim == 3:
        vals = vals[:, :, 0]

    global_importance = (
        pd.DataFrame(
            {
                "feature": list(x_ev.columns),
                "mean_abs_shap": np.abs(vals).mean(axis=0),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "shap_values": shap_values,
        "global_importance": global_importance,
    }


def assess_ceteris_paribus(
    model: Any,
    x_reference: pd.DataFrame,
    feature: str,
    grid_points: int = 25,
    positive_class: int = 1,
) -> pd.DataFrame:
    """Compute a simple ceteris paribus profile by varying one feature.

    The profile is computed around a representative row (column medians/modes).
    """
    x_ref = _to_dataframe(x_reference)
    if feature not in x_ref.columns:
        raise KeyError(f"Feature '{feature}' not found in x_reference.")

    rep = x_ref.copy().iloc[[0]].copy()
    for col in x_ref.columns:
        if pd.api.types.is_numeric_dtype(x_ref[col]):
            rep[col] = x_ref[col].median()
        else:
            mode = x_ref[col].mode(dropna=True)
            rep[col] = mode.iloc[0] if not mode.empty else x_ref[col].iloc[0]

    if pd.api.types.is_numeric_dtype(x_ref[feature]):
        fmin = float(x_ref[feature].quantile(0.01))
        fmax = float(x_ref[feature].quantile(0.99))
        grid = np.linspace(fmin, fmax, grid_points)
    else:
        grid = x_ref[feature].dropna().unique().tolist()

    records: List[Dict[str, Any]] = []
    for val in grid:
        x_tmp = rep.copy()
        x_tmp[feature] = val
        score = float(_predict_scores(model, x_tmp, positive_class=positive_class)[0])
        records.append({"feature": feature, "feature_value": val, "prediction": score})

    return pd.DataFrame(records)


def optimize_thresholds_by_group(
    estimator: Any,
    x_train: pd.DataFrame,
    y_train: ArrayLike,
    x_eval: pd.DataFrame,
    sensitive_features_train: Union[pd.Series, np.ndarray, List[Any]],
    sensitive_features_eval: Union[pd.Series, np.ndarray, List[Any]],
    constraints: str = "demographic_parity",
    objective: str = "balanced_accuracy_score",
    prefit_estimator: bool = True,
) -> Tuple[Any, np.ndarray, np.ndarray]:
    """Fit Fairlearn ThresholdOptimizer and return group-aware predictions.

    This is useful when a base model (for example XGBoost) has unfair outcomes
    across groups and post-processing needs to enforce fairness constraints.
    """
    try:
        from fairlearn.postprocessing import ThresholdOptimizer
    except ImportError as exc:
        raise ImportError(
            "fairlearn is required for optimize_thresholds_by_group. "
            "Install with `pip install fairlearn`."
        ) from exc

    x_train_df = _to_dataframe(x_train)
    x_eval_df = _to_dataframe(x_eval, columns=x_train_df.columns)
    y_train_arr = np.asarray(y_train)
    sf_train = np.asarray(sensitive_features_train)
    sf_eval = np.asarray(sensitive_features_eval)

    if not prefit_estimator and hasattr(estimator, "fit"):
        estimator.fit(x_train_df, y_train_arr)

    threshold_optimizer = ThresholdOptimizer(
        estimator=estimator,
        constraints=constraints,
        objective=objective,
        prefit=prefit_estimator,
        predict_method="predict_proba",
    )
    threshold_optimizer.fit(
        x_train_df,
        y_train_arr,
        sensitive_features=sf_train,
    )

    group_preds = threshold_optimizer.predict(
        x_eval_df,
        sensitive_features=sf_eval,
    )
    base_scores = _predict_scores(estimator, x_eval_df)
    return threshold_optimizer, np.asarray(group_preds), np.asarray(base_scores)


def evaluate_fairness_tradeoff(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    y_score: ArrayLike,
    sensitive_features: Union[pd.Series, np.ndarray, List[Any]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute fairness and predictive-performance metrics with group breakdown."""
    try:
        from fairlearn.metrics import (
            MetricFrame,
            demographic_parity_difference,
            demographic_parity_ratio,
            equalized_odds_difference,
            equalized_odds_ratio,
            selection_rate,
            true_positive_rate,
            false_positive_rate,
        )
    except ImportError as exc:
        raise ImportError(
            "fairlearn is required for evaluate_fairness_tradeoff. "
            "Install with `pip install fairlearn`."
        ) from exc

    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    y_score_arr = np.asarray(y_score)
    sf_arr = np.asarray(sensitive_features)

    global_metrics = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "precision": float(precision_score(y_true_arr, y_pred_arr, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred_arr, zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred_arr, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true_arr, y_score_arr)) if len(np.unique(y_true_arr)) > 1 else np.nan,
        "demographic_parity_difference": float(
            demographic_parity_difference(y_true=y_true_arr, y_pred=y_pred_arr, sensitive_features=sf_arr)
        ),
        "demographic_parity_ratio": float(
            demographic_parity_ratio(y_true=y_true_arr, y_pred=y_pred_arr, sensitive_features=sf_arr)
        ),
        "equalized_odds_difference": float(
            equalized_odds_difference(y_true=y_true_arr, y_pred=y_pred_arr, sensitive_features=sf_arr)
        ),
        "equalized_odds_ratio": float(
            equalized_odds_ratio(y_true=y_true_arr, y_pred=y_pred_arr, sensitive_features=sf_arr)
        ),
    }

    metric_frame = MetricFrame(
        metrics={
            "selection_rate": selection_rate,
            "accuracy": accuracy_score,
            "precision": lambda yt, yp: precision_score(yt, yp, zero_division=0),
            "recall": lambda yt, yp: recall_score(yt, yp, zero_division=0),
            "f1": lambda yt, yp: f1_score(yt, yp, zero_division=0),
            "tpr": true_positive_rate,
            "fpr": false_positive_rate,
        },
        y_true=y_true_arr,
        y_pred=y_pred_arr,
        sensitive_features=sf_arr,
    )

    summary_df = pd.DataFrame([global_metrics])
    by_group_df = metric_frame.by_group.reset_index().rename(columns={"index": "group"})
    return summary_df, by_group_df


def assess_postprocessing_fairness(
    estimator: Any,
    x_train: pd.DataFrame,
    y_train: ArrayLike,
    x_eval: pd.DataFrame,
    y_eval: ArrayLike,
    sensitive_features_train: Union[pd.Series, np.ndarray, List[Any]],
    sensitive_features_eval: Union[pd.Series, np.ndarray, List[Any]],
    constraints: str = "demographic_parity",
    objective: str = "balanced_accuracy_score",
    prefit_estimator: bool = True,
) -> PostProcessingFairnessResult:
    """Run threshold optimization and return fairness/performance trade-off tables."""
    threshold_optimizer, y_pred, y_score = optimize_thresholds_by_group(
        estimator=estimator,
        x_train=x_train,
        y_train=y_train,
        x_eval=x_eval,
        sensitive_features_train=sensitive_features_train,
        sensitive_features_eval=sensitive_features_eval,
        constraints=constraints,
        objective=objective,
        prefit_estimator=prefit_estimator,
    )
    tradeoff_summary, tradeoff_by_group = evaluate_fairness_tradeoff(
        y_true=y_eval,
        y_pred=y_pred,
        y_score=y_score,
        sensitive_features=sensitive_features_eval,
    )
    return PostProcessingFairnessResult(
        threshold_optimizer=threshold_optimizer,
        predictions=y_pred,
        scores=y_score,
        tradeoff_summary=tradeoff_summary,
        tradeoff_by_group=tradeoff_by_group,
    )


def assess_openxai_fidelity_stability(
    model: Any,
    x_eval: pd.DataFrame,
    feature_importance_fn: Any,
    n_samples: int = 200,
    random_state: int = 42,
    require_openxai: bool = False,
) -> pd.DataFrame:
    """Estimate explanation fidelity and stability with an OpenXAI-compatible flow.

    Parameters
    ----------
    model:
        Trained model with ``predict`` and optionally ``predict_proba``.
    x_eval:
        Evaluation dataframe.
    feature_importance_fn:
        Callable that accepts a dataframe and returns feature attributions
        as an array of shape (n_rows, n_features).
    require_openxai:
        If True, raises if OpenXAI is not installed.

    Notes
    -----
    OpenXAI APIs vary across versions. This function provides robust, version-
    agnostic metrics that mirror fidelity/stability intent:
    - fidelity: rank correlation between absolute attribution sum and model score
    - stability: attribution consistency under small Gaussian perturbations
    """
    if require_openxai:
        try:
            import openxai  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "openxai is required (require_openxai=True). Install with `pip install openxai`."
            ) from exc

    x_df = _to_dataframe(x_eval)
    if len(x_df) > n_samples:
        x_df = x_df.sample(n=n_samples, random_state=random_state)

    attributions = np.asarray(feature_importance_fn(x_df))
    if attributions.ndim != 2 or attributions.shape[0] != len(x_df):
        raise ValueError("feature_importance_fn must return shape (n_rows, n_features).")

    base_scores = _predict_scores(model, x_df)
    attr_strength = np.abs(attributions).sum(axis=1)

    # Spearman-like correlation using rank transform only (no scipy dependency).
    base_rank = pd.Series(base_scores).rank(method="average").to_numpy()
    attr_rank = pd.Series(attr_strength).rank(method="average").to_numpy()
    fidelity_corr = float(np.corrcoef(base_rank, attr_rank)[0, 1])

    rng = np.random.default_rng(seed=random_state)
    noise = rng.normal(loc=0.0, scale=0.01, size=x_df.shape)
    x_perturbed = x_df.copy()
    numeric_cols = [c for c in x_df.columns if pd.api.types.is_numeric_dtype(x_df[c])]
    if numeric_cols:
        x_perturbed.loc[:, numeric_cols] = x_df[numeric_cols].to_numpy() + noise[:, : len(numeric_cols)]

    attrs_perturbed = np.asarray(feature_importance_fn(x_perturbed))
    if attrs_perturbed.shape != attributions.shape:
        raise ValueError("feature_importance_fn output shape changed after perturbation.")

    row_corrs: List[float] = []
    for i in range(len(x_df)):
        a = attributions[i]
        b = attrs_perturbed[i]
        if np.std(a) == 0 or np.std(b) == 0:
            row_corrs.append(np.nan)
        else:
            row_corrs.append(float(np.corrcoef(a, b)[0, 1]))

    stability_mean_corr = float(np.nanmean(row_corrs))
    stability_p10_corr = float(np.nanquantile(row_corrs, 0.10))

    return pd.DataFrame(
        [
            {
                "fidelity_rank_corr": fidelity_corr,
                "stability_mean_corr": stability_mean_corr,
                "stability_p10_corr": stability_p10_corr,
                "n_samples": int(len(x_df)),
            }
        ]
    )


def run_all_assessments(
    model: Any,
    x_train: pd.DataFrame,
    x_eval: pd.DataFrame,
    sensitive_attributes: Sequence[str],
    y_train: Optional[ArrayLike] = None,
    y_eval: Optional[ArrayLike] = None,
    sensitive_features_train: Optional[Union[pd.Series, np.ndarray, List[Any]]] = None,
    sensitive_features_eval: Optional[Union[pd.Series, np.ndarray, List[Any]]] = None,
    target_name_for_dice: Optional[str] = None,
    continuous_features_for_dice: Optional[Sequence[str]] = None,
    feature_importance_fn_for_openxai: Optional[Any] = None,
    explain_index: int = 0,
) -> Dict[str, Any]:
    """Run all configured assessments and return a consolidated dictionary.

    Notes
    -----
    - DiCE requires ``target_name_for_dice`` and ``continuous_features_for_dice``.
    - Anchor, LIME, SHAP, and DiCE need optional dependencies installed.
    - Failures are captured per-method and returned in ``errors``.
    """
    x_train_df = _to_dataframe(x_train)
    x_eval_df = _to_dataframe(x_eval, columns=x_train_df.columns)
    x_instance = x_eval_df.iloc[[explain_index]]

    out: Dict[str, Any] = {"results": {}, "errors": {}}

    try:
        out["results"]["counterfactual_fairness"] = assess_counterfactual_fairness(
            model=model,
            x=x_eval_df,
            sensitive_attributes=sensitive_attributes,
        )
    except Exception as exc:  # noqa: BLE001
        out["errors"]["counterfactual_fairness"] = str(exc)

    try:
        out["results"]["anchor"] = assess_anchor(
            model=model,
            x_train=x_train_df,
            x_instance=x_instance,
        )
    except Exception as exc:  # noqa: BLE001
        out["errors"]["anchor"] = str(exc)

    try:
        out["results"]["lime"] = assess_lime(
            model=model,
            x_train=x_train_df,
            x_instance=x_instance,
        )
    except Exception as exc:  # noqa: BLE001
        out["errors"]["lime"] = str(exc)

    try:
        out["results"]["shap"] = assess_shap(
            model=model,
            x_background=x_train_df,
            x_eval=x_eval_df,
        )
    except Exception as exc:  # noqa: BLE001
        out["errors"]["shap"] = str(exc)

    if target_name_for_dice and continuous_features_for_dice:
        try:
            train_for_dice = x_train_df.copy()
            if target_name_for_dice not in train_for_dice.columns:
                raise KeyError(
                    f"DiCE target '{target_name_for_dice}' not found in training dataframe."
                )
            out["results"]["dice"] = assess_dice(
                model=model,
                x_train=train_for_dice,
                query_instance=x_instance,
                target_name=target_name_for_dice,
                continuous_features=continuous_features_for_dice,
            )
        except Exception as exc:  # noqa: BLE001
            out["errors"]["dice"] = str(exc)

    try:
        cp_profiles: Dict[str, pd.DataFrame] = {}
        for feat in x_train_df.columns[: min(5, x_train_df.shape[1])]:
            cp_profiles[feat] = assess_ceteris_paribus(model, x_eval_df, feat)
        out["results"]["ceteris_paribus"] = cp_profiles
    except Exception as exc:  # noqa: BLE001
        out["errors"]["ceteris_paribus"] = str(exc)

    if (
        y_train is not None
        and y_eval is not None
        and sensitive_features_train is not None
        and sensitive_features_eval is not None
    ):
        try:
            out["results"]["postprocessing_fairness"] = assess_postprocessing_fairness(
                estimator=model,
                x_train=x_train_df,
                y_train=y_train,
                x_eval=x_eval_df,
                y_eval=y_eval,
                sensitive_features_train=sensitive_features_train,
                sensitive_features_eval=sensitive_features_eval,
            )
        except Exception as exc:  # noqa: BLE001
            out["errors"]["postprocessing_fairness"] = str(exc)

    if feature_importance_fn_for_openxai is not None:
        try:
            out["results"]["openxai_fidelity_stability"] = assess_openxai_fidelity_stability(
                model=model,
                x_eval=x_eval_df,
                feature_importance_fn=feature_importance_fn_for_openxai,
            )
        except Exception as exc:  # noqa: BLE001
            out["errors"]["openxai_fidelity_stability"] = str(exc)

    return out
