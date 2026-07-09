# Fairness Assessment Export

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

- `scripts/fairness_phase1.py`
- `scripts/fairness_phasev2.py`
- `scripts/fairness_phasev3.py`
- `scripts/DIR_Assessment.py`
- `scripts/XAI_notebook.py`
- `scripts/XAI_Checks.py`
- `scripts/Checks.py`
- `scripts/Classing_and_Model.py`
- `scripts/Classing_and_Model-new.py`

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
