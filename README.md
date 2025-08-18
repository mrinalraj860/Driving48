# Project Overview

This repository consists of **three primary folders**:

1. **FineDiving** *(conversion & analysis)*  
2. **Diving48** *(analysis)*  
3. **FineDivingCheckOnD48** *(cross-dataset checks & analysis)*

Each folder contains most of the analysis performed. For many models, I attempted to keep all intermediate artifacts and result files; however, **due to size constraints**, not everything could be included in the repository.

---

## Repository Structure

```
project-root/
├─ FineDiving/
│  ├─ Pre-processing/
│  │  ├─ ...               # Analysis of all preprocessing steps and diagnostics
│  ├─ ...                  # Model code, reports, and selected results
├─ Diving48/
│  ├─ ...                  # Analyses, scripts, and selected results
├─ FineDivingCheckOnD48/
│  ├─ ...                  # Cross-dataset checks & experiments
├─ Running.py              # Main entry point
└─ README.md
```

---

## Folder Details

### `FineDiving/`
- Contains conversion scripts and analysis specific to FineDiving.
- **`Pre-processing/`**: a dedicated folder with **analysis of all preprocessing work** (data cleaning, normalization, trajectory extraction summaries, diagnostics, and figures).
- Selected results and reports are included; large artifacts were omitted to keep the repo lightweight.

### `Diving48/`
- Analyses, experiments, and scripts for Diving48.
- Selected results are kept where possible; some large logs/checkpoints are excluded.

### `FineDivingCheckOnD48/`
- Cross-dataset experiments that check FineDiving-derived methods against Diving48 (and vice versa), plus utilities for comparison.

---

## Main Entry Point

The **main file is `Running.py`** at the repository root.

```bash
# Show available options
python Running.py --help

# Example: run with a chosen dataset
python Running.py --dataset finediving
# or
python Running.py --dataset diving48
```

> Note: Some runs expect preprocessed artifacts produced inside `FineDiving/Pre-processing/`. If missing, please generate them by following the notes inside that folder first.

---

## Results & Artifacts

- Where feasible, results (metrics, figures, and summaries) are kept inside each dataset folder.  
- **Large files (e.g., full checkpoints, exhaustive logs, and full prediction dumps)** are not included due to repository size limits.
- If a referenced artifact is missing, it was likely excluded for size—see any included `REPORT.md`/`RESULTS.md` or the folder’s README for pointers to reproduce or regenerate.

---

## Reproducibility Notes

- Preprocessing configurations and important parameters are documented in `FineDiving/Pre-processing/`.
- Scripts and configs inside each folder should allow you to reproduce the key analyses that are still included.
