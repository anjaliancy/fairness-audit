# Fairness Assessment Export

> Extracted and reorganized from a 5-person group project (NUS-ISS, Graduate Certificate in Intelligent Financial Risk Management, April 2026). This repo covers my individually-led workstream: the full fairness and explainability pipeline, disparate impact audit, and logistic regression scorecard. Team members' work on alternative-data acquisition, factor analysis, and the stacked ensemble models is not included here.

This repository packages the fairness, DIR, anchor, explainability, and counterfactual work from the FRM project in a GitHub-friendly layout.

## Quick Start

1. Read this README first to understand the folder layout.
2. Open the `scripts/` folder for the notebook-to-Python exports.
3. Review `reports/` for the generated fairness and explainability outputs.
4. Use `requirements.txt` to recreate the Python environment if you want to rerun the analysis.

## Top-Level Layout

- `scripts/` - notebook exports and supporting Python scripts
- `src/` - reusable helper modules used by the assessment workflows
- `reports/` - curated generated outputs, grouped by topic
- `requirements.txt` - dependency snapshot from the workspace root

## Repo Index

- `scripts/` - converted analysis notebooks and report-generation scripts
- `src/` - shared utilities for fairness repair, explainability, and scorecard logic
- `reports/anchor/` - anchor explanation PDFs
- `reports/counterfactual/` - DiCE and counterfactual fairness outputs
- `reports/explainability/` - explainability summaries and charts
- `reports/fairness/` - DIR, fairness, and post-processing comparison outputs

## Script Exports

The original notebooks were converted to plain Python files:

- `scripts/fairness_preprocessing.py`
- `scripts/fairness_inprocessing.py`
- `scripts/fairness_postprocessing.py`
- `scripts/dir_audit.py`
- `scripts/explainability_shap_lime.py`
- `scripts/explainability_validation.py`
- `scripts/data_quality_checks.py`
- `scripts/scorecard_woe_iv.py`
- `scripts/scorecard_woe_iv_v2.py`

## Helper Modules

- `src/explainability_assessment.py`
- `src/fairness_preprocessing.py`
- `src/geographic_encoding.py`
- `src/scorecard_functions.py`
- `src/woe_iv_functions.py`

## Reports Layout

The generated outputs are split by purpose so the repository is easier to browse:

- `reports/anchor/` - anchor explanation PDFs
- `reports/counterfactual/` - DiCE and counterfactual fairness outputs
- `reports/explainability/` - explainability dashboards, comparison charts, and summary tables
- `reports/fairness/` - DIR and fairness diagnostics

## Notes

- The exports are plain `.py` files generated from the original `.ipynb` sources.
- Large raw datasets, model binaries, and intermediate build artifacts were intentionally left out.
- The repo is ready to upload as-is, and the grouped report folders are meant to make review easier on GitHub.
- If you want the repository even more presentation-ready, the next step would be to add a single entry script or a small walkthrough notebook.
