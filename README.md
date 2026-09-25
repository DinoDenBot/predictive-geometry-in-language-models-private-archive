# Predictive Geometry in Language Models

Code and compact result inputs for the empirical findings reported in the paper. The manuscript, PDF, and figures are not part of this repository. For the allocation-label ranking experiment, only ROC AUC is included; the earlier low-FPR/TPR analyses are outside the paper's reported results.

## Quick start

Run the result check from the repository root. It uses only the Python standard library and does not download models or data:

```sh
python3 scripts/verify_results.py
```

The command prints the eight exposure-phase slopes, the binary and gap shares, the four-setting AUC comparison, and the editing locality ratios. It also checks the arithmetic and SHA-256 hashes of the bundled code. A successful run verifies the compact inputs; it does not rerun model training or inference.

## Results and code

| Finding reported in the paper | Implementation | Compact result inputs |
| --- | --- | --- |
| Randomized exposure responses across eight phases, including alignment and localization | `src/exposure_observability.py`, `src/exposure_geometry_extension.py`, their `run_*` and `execute_*` entry points, and `experiments/studies/study3/specification/software/` | `results/exposure_summary.csv`, `results/localization_summary.csv` |
| Retrospective decomposition of the mean response into binary motion and the gap | `reviewer_revision/run_gap_attribution.py`, `reviewer_revision/summarize_gap_attribution.py` | `results/gap_attribution.csv` |
| Final-checkpoint ranking of the controlled positive-dose label | `experiments/studies/geometric_trajectory_v1/` | `results/trajectory_*_auc.csv` |
| Protected output-row edit and ordinary-training comparison | `experiments/studies/context_selective_retrieval_v3/` | `results/editing_*.csv` |

`src/cats_identification.py` and `src/run_cats_agnews.py` are retained because the reported study code imports functions from them. Results from the older CATS studies are not included.

The AUC files contain the setting, target, phase, and property summaries used in the paper. They omit empirical TPR columns from the original reports. The initial exposure study's R p-values include the reported three-size selection correction; its source report records the uncorrected values.

## Provenance check

The compact inputs were extracted from nine original result files. If those files are available in the original study repository, verify their hashes as well:

```sh
python3 scripts/verify_results.py --source-root /path/to/original-study-repository
```

`results/source_hashes.json` names the source files and records their SHA-256 hashes. `results/code_hashes.json` records the bundled Python files. One imported helper has generic placeholders in place of personal path defaults; its calculations are unchanged.

## Full experiment reruns

The full pipelines also need frozen document assignments, model checkpoints, source documents, and acquired prediction measurements. These large or restricted inputs are not bundled here. Some preserved entry points contain historical storage defaults that must be set for a new machine. Consequently, this repository supports inspection of the implementation and checking of the compact reported results, while a clean rerun of training, acquisition, classifier fitting, randomization tests, or editing requires those additional artifacts.

The pinned Python dependencies are in `requirements-review.txt` and `requirements-mimir.txt`. To run code-level tests, create a Python 3.12 virtual environment and install the dependencies:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-review.txt
export PYTHONPATH=src:experiments/studies:experiments/studies/study3/specification/software:.
.venv/bin/python -m pytest -q tests \
  experiments/studies/geometric_trajectory_v1/test_trajectory.py \
  experiments/studies/study3/specification/software/test_study3.py \
  experiments/studies/context_selective_retrieval_v3
```

Some tests require external artifacts. The quick result check above is self-contained.

## Anonymous release

Review remaining machine-specific defaults, dependency licenses, and data rights before publishing. If releasing from this private working repository, use a clean release history so earlier private commits are not exposed.
