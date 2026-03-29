# Setup Instructions

**Aayush Nepal** — EAS 510

## What you need

- Python 3.9+
- pip

## Setup steps

### 1. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:
```powershell
# Windows
.venv\Scripts\Activate.ps1
```
```bash
# Mac/Linux
source .venv/bin/activate
```

### 2. Install packages

```bash
pip install -r requirements.txt
```

This installs: scikit-learn, xgboost, Pillow, numpy, pandas, matplotlib, seaborn, opencv-python, scipy

### 3. Check that the data is there

You should see 3 folders inside `data/original_pdfs/`:
- `word_pdfs_png/` (398 images)
- `google_docs_pdfs_png/` (396 images)
- `python_pdfs_png/` (100 images)

### 4. Run the pipeline

Run these in order from the project root:

```bash
python src/augmentation.py      # ~30 seconds, creates augmented images
python src/classification.py    # ~1 second, trains models
python src/analysis.py          # ~30 seconds, generates all results
```

## What gets generated

- `results/performance_metrics.csv` — accuracy/precision/recall/F1 for every model and condition
- `results/significance_tests.csv` — McNemar's test results
- `results/confusion_matrices/` — 24 heatmap images
- `results/robustness_plots/robustness_curves.png` — line chart showing accuracy drop

## My environment

- Windows 11
- Python 3.10
- PowerShell
