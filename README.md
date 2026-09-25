# Paper result reproducibility code

This private branch holds code and compact numeric inputs for the empirical results reported in *Predictive Geometry in Language Models*. It contains no manuscript source, PDF, or figures. It excludes other project experiments and the retired low-FPR/TPR exports.

## Scope

| Reported result | Code | Compact input |
|---|---|---|
| Eight randomized exposure phases, including alignment/localization analysis | `src/exposure_observability.py`, `src/run_exposure_observability.py`, `src/exposure_geometry_extension.py`, `src/run_exposure_geometry_extension.py`, `src/execute_exposure_geometry_extension.py`, `experiments/studies/study3/specification/software/` | `results/exposure_summary.csv`, `results/localization_summary.csv` |
| Retrospective binary/gap attribution | `reviewer_revision/run_gap_attribution.py`, `reviewer_revision/summarize_gap_attribution.py` | `results/gap_attribution.csv` |
| Final-checkpoint allocation-label AUC | `experiments/studies/geometric_trajectory_v1/` | `results/trajectory_*_auc.csv` |
| Protected output-row edit and ordinary-training comparison | `experiments/studies/context_selective_retrieval_v3/` | `results/editing_*.csv` |

The `src/cats_identification.py` and `src/run_cats_agnews.py` files are imported by the reported pipelines. Their presence does not add the older CATS experiments to this release's result scope.

## Check the compact results

Python 3.10 or later is sufficient for this check:

```sh
python3 scripts/verify_results.py
```

The script checks eight positive exposure slopes, the exact mean-response bridge and binary shares, equal target-stratum AUC aggregation, and editing/locality arithmetic. It prints the headline results. The AUC inputs contain **AUC only**; their source reports' empirical TPR columns are not included.

If the original study repository is available, verify the nine source result files used to extract the compact inputs:

```sh
python3 scripts/verify_results.py --source-root /path/to/original-study-repository
```

`results/source_hashes.json` records those exact source-file SHA-256 values. `results/code_hashes.json` records the bundled Python code hashes. The compact CSVs are selected columns and outcomes from those sources, not new estimates. One unused helper's personal absolute-path defaults were replaced with generic placeholders; its scientific calculations were left intact.
The initial setting's R p-values in `exposure_summary.csv` apply the reported three-size selection correction to the raw 0.00001 values in the source report.

## Run the study code

The full study pipelines require the frozen document assignments, model checkpoints, acquisition measurements, and some licensed or externally stored source data. Those large inputs are not in this branch. Set local paths or stage the original hash-verified artifacts before a full rerun; several preserved source files still have historical default paths. The results check above runs without them. A clean rerun of model training, acquisition, classifier fitting, randomization tests, or editing is **not** established by the compact result check.

The source dependencies are listed in `requirements-review.txt` and `requirements-mimir.txt`. For code-level tests, install them in a virtual environment and set:

```sh
export PYTHONPATH=src:experiments/studies:experiments/studies/study3/specification/software:.
python3 -m pytest -q tests experiments/studies/geometric_trajectory_v1/test_trajectory.py experiments/studies/study3/specification/software/test_study3.py experiments/studies/context_selective_retrieval_v3
```

Some tests need external artifacts and may not run in a clean clone. The compact-result check is self-contained.

## Release preparation

This repository is private. Historical code can contain local paths and infrastructure defaults. Review those, dependency licenses, and data rights before making any branch public or submitting an anonymous artifact.
