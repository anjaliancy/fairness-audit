# Fairness Assessment Export

This folder packages the fairness, DIR, anchor, explainability, and counterfactual work from the FRM project in GitHub-friendly form.

## Contents

- `scripts/` - notebook exports and supporting scripts
- `src/` - reusable helper modules used by the assessment workflows
- `reports/` - curated generated outputs, figures, and summary tables
- `requirements.txt` - dependency snapshot from the workspace root

## Included notebook exports

- `fairness_phase1.py`
- `fairness_phasev2.py`
- `fairness_phasev3.py`
- `DIR_Assessment.py`
- `XAI_notebook.py`
- `XAI_Checks.py`
- `Checks.py`
- `Classing_and_Model.py`
- `Classing_and_Model-new.py`

## Included helper modules

- `src/explainability_assessment.py`
- `src/fairness_preprocessing.py`
- `src/geographic_encoding.py`
- `src/scorecard_functions.py`
- `src/woe_iv_functions.py`

## Included report artifacts

The `reports/` folder contains the main explainability, anchor, DIR, and counterfactual outputs that were referenced in the notebooks, including PDFs, PNGs, and summary CSV/TXT files.

## Notes

- The notebook exports are plain `.py` files generated from the original `.ipynb` sources.
- Large raw datasets, model binaries, and intermediate build artifacts were intentionally left out to keep the export GitHub-friendly.
- If you want a different level of pruning, the package can be slimmed down further or expanded to include more of the generated data tables.
