# Early-Warning Food Insecurity Prediction System

This project builds an **early-warning, household-level food insecurity risk prediction** workflow using an LSMS-style survey dataset. It trains and compares multiple classification models, selects a probability threshold, and produces a **risk ranking** (probability + risk tier + suggested action) intended for **decision support**.

## What you get

- **Trained model comparison** (multiple classifiers, evaluated with Macro F1, ROC-AUC, Average Precision, etc.)
- **Interpretable outputs** (feature importance + optional SHAP)
- **Risk ranking export**: `household_food_insecurity_risk_ranking.csv` (ignored by git as a generated artifact)

## Repository contents

- **Notebook (main entrypoint)**: `Machine_Learning_project_LSMS.ipynb`
  - End-to-end workflow: load data → clean/feature selection → train/validate/test → threshold search → interpretability → risk ranking export
- **Utilities**: `util_functions.py`
  - Metric reporting, best-threshold search, positive-class probability helper, risk tier assignment, and action recommendation mapping
- **Environment**: `pyproject.toml` + `uv.lock`
  - Python dependencies managed via `uv` (recommended)

## Data

- **Expected input file**: `hhlevel18.dta` (Stata file)
  - By default, the notebook loads from `./hhlevel18.dta` (project root).
- **Target column**: `foodinsecurty`
  - `0` = not food insecure, `1` = food insecure
  - Rows with missing target values are dropped in the notebook.
- **Leakage-aware feature exclusion**
  - The notebook excludes some variables considered too close to outcomes or post-intervention signals (e.g., `lnfood_cons_ann`, `Assistfood`, `Assistinkind`).

If your dataset file name/location differs, update the `file_path` variable in the notebook’s data-loading cell.

## Setup

### Option A: `uv` (recommended)

1. Install `uv` (Astral).
2. From the project root:

```bash
uv sync
```

### Option B: `pip` + venv

Create and activate a virtual environment, then install the dependencies listed in `pyproject.toml` using your preferred workflow.

## Run (Jupyter)

Launch JupyterLab:

```bash
uv run jupyter lab
```

Open `Machine_Learning_project_LSMS.ipynb` and run cells top-to-bottom.

## Output files

- **Main artifact**: `household_food_insecurity_risk_ranking.csv`
  - Contains each household’s predicted probability, risk group (Low/Medium/High), and recommended action.
  - Treated as generated output and ignored by git (see `.gitignore`).

## Customization

- **Risk tiers**: adjust thresholds in `util_functions.py` (`assign_risk_group`)
- **Recommended actions**: edit mappings in `util_functions.py` (`recommend_action`)
- **Decision threshold selection**: the notebook can search thresholds (e.g., maximize Macro F1). You can change the optimized metric or the threshold grid in `util_functions.py` (`find_best_threshold`).

## Troubleshooting

- **“Dataset not found” error**: ensure `hhlevel18.dta` is present in the project root, or update the notebook `file_path`.
- **Jupyter kernel issues**: after `uv sync`, re-launch using `uv run jupyter lab` so Jupyter uses the environment dependencies.

## Reproducibility notes

- Dependency versions are pinned via `uv.lock`.
- A fixed `random_state` is used in train/validation/test splits inside the notebook.