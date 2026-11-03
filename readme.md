# Mini-project 2: Selecting Reliable LLM Reasoning

**Author: HE YUXIN | Student ID: 3036897306**

This directory contains the final **v14** model, the files needed for offline evaluation and training reproduction, the project report, and recorded experimental evidence. The selector combines a question-conditioned low-rank scorer (weight 0.75) with the v2 baseline (weight 0.25).

| Split | Correct / total | Accuracy |
|---|---:|---:|
| Validation | 753 / 798 | 94.3609022556391% |
| Test | 388 / 467 | **83.08351177730192%** |

The test result improves on v2 by eight questions, or 1.7131 percentage points. The paired confidence interval includes zero. The final configuration was selected on validation and frozen before test evaluation; no test result was used to retune the model.

## 1. Directory contents

```text
MP2_Submission_v14/
  readme.md                    Installation and reproduction instructions
  report.pdf                   Project report
  requirements.txt             Pinned runtime and training dependencies
  selector.py                  Final predict(bank, arrays, checkpoint) interface
  features.py                  Original numerical and text features
  math_consistency.py          Mathematical consistency features
  common.py                    Original data loader and numeric parser
  predict.py                   Original prediction runner
  score_predictions.py         Original independent scorer
  check_data.py                Original data integrity checker
  evaluate.py                  Complete four-thread prediction and scoring
  verify.py                    Asset, interface and permutation checks
  analyze.py                   Post-freeze validation reliability analysis
  train.py                     Rebuild the selected model from training data
  search.py                    Replay the final v14 development search
  training_utils.py            Shared training utilities
  test_interactions.py         Interaction and derivative tests
  test_math_consistency.py      Arithmetic parsing boundary tests
  checkpoint/                  All six final inference assets
  data/                        Original course data, embeddings and licenses
  evidence/                    Scores, resources, search and freeze records
  MANIFEST.json                Submission file sizes and SHA-256 hashes
```

The package excludes virtual environments, IDE settings, obsolete models, rejected checkpoints, cached predictions and temporary files. The training, verification and analysis scripts support reproducibility.

## 2. Environment setup

Use **Python 3.12**; the recorded runs used Python 3.12.14. Dependencies are pinned in `requirements.txt`:

```text
numpy==2.1.3
scipy==1.15.3
scikit-learn==1.6.1
joblib==1.4.2
threadpoolctl==3.5.0
```

Run the following commands from this submission directory.

**Windows PowerShell:**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If `py -3.12` is unavailable, install Python 3.12 or create a Python 3.12 virtual environment in PyCharm. You may reuse an existing environment with the required versions.

**Linux/macOS:**

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

No GPU, CUDA, PyTorch, CatBoost, API key or additional pretrained model is required. After dependencies are installed, prediction and scoring run offline.

## 3. Reproduce the frozen result

**Use the directory containing this README as the working directory. There is no `code/` subdirectory in this package. Retraining is not required.**

Windows PowerShell:

```powershell
# Check data and the model interface.
.\.venv\Scripts\python.exe check_data.py
.\.venv\Scripts\python.exe verify.py

# Evaluate validation and test using the supplied frozen checkpoint.
.\.venv\Scripts\python.exe evaluate.py --split validation --method selector.py --checkpoint checkpoint --output-dir results
.\.venv\Scripts\python.exe evaluate.py --split test --method selector.py --checkpoint checkpoint --output-dir results
```

On Linux/macOS, replace `.\.venv\Scripts\python.exe` with `.venv/bin/python`. If the environment is activated, `python` can be used instead.

Expected test score:

```json
{
  "split": "test",
  "correct": 388,
  "n": 467,
  "accuracy": 83.08351177730192
}
```

For each evaluated split, `results/` contains:

- `*-predictions.json`: selected candidate indices;
- `*-score.json`: the independent official score;
- `*-resources.json`: complete process timings, asset sizes and hashes.

These outputs do not overwrite the records in `evidence/`. The `accuracy` field is a percentage.

`evaluate.py` launches fresh prediction and scoring processes using the selected Python interpreter. It limits OMP, OpenBLAS, MKL and NumExpr to four threads. Use `total_seconds` in the resource JSON for complete evaluation time, including startup and preprocessing.

Recorded complete test times were 13.48 seconds in the development directory and 13.28 seconds in the standalone directory on an AMD Ryzen 7 5800H. Timing varies across machines; the course reference CPU determines resource eligibility.

The six files under `checkpoint/` total **6,246,341 bytes (5.957 MiB)**. Keep the entire `checkpoint/v2/` subdirectory. Course-supplied data and ordinary source files are excluded from the additional inference asset limit.

## 4. Run in PyCharm

1. Open **MP2_Submission_v14** itself as the project.
2. Select a Python 3.12 interpreter with the dependencies installed.
3. Open **Run > Edit Configurations > + > Python** and enter:

| Field | Value |
|---|---|
| Name | `MP2_Final_Test` |
| Script path | `$PROJECT_DIR$/evaluate.py` |
| Parameters | `--split test --method selector.py --checkpoint checkpoint --output-dir results` |
| Working directory | `$PROJECT_DIR$` |
| Python interpreter | The configured Python 3.12 environment |

Environment variables:

```text
OMP_NUM_THREADS=4;OPENBLAS_NUM_THREADS=4;MKL_NUM_THREADS=4;NUMEXPR_NUM_THREADS=4;PYTHONUNBUFFERED=1
```

Use **Run** for timing reproduction. Running `selector.py` directly does not perform evaluation. To evaluate validation, change `--split test` to `--split validation`.

If imports fail, check the run configuration's interpreter and install `requirements.txt` into that interpreter. If `predict.py` or the checkpoint cannot be found, check the working directory. Explicitly calling `.\.venv\Scripts\python.exe` in the terminal avoids relying on automatic environment activation.

## 5. Reproduce training and the final search

### Fixed-configuration retraining

```powershell
.\.venv\Scripts\python.exe train.py --output retraining/checkpoint
.\.venv\Scripts\python.exe verify.py --checkpoint retraining/checkpoint
.\.venv\Scripts\python.exe evaluate.py --split validation --method selector.py --checkpoint retraining/checkpoint --output-dir retraining/results
```

`train.py` reads training inputs and references only. It reconstructs the selected pairwise linear model (C=.01), the 64-unit/20-epoch neural baseline, the shared TF-IDF/SVD encoder, and the rank-32 low-rank scorer (factor L2=.01, 40 epochs). It does not load validation/test data or overwrite the submitted `checkpoint/`.

Training statistics are saved to `retraining/training_summary.json`. Fixed retraining was verified locally. Different BLAS implementations or platforms may introduce small numerical or serialization differences; the submitted frozen checkpoint defines the reported result.

### Replay the v14 development search

```powershell
.\.venv\Scripts\python.exe search.py --baseline checkpoint/v2 --output retraining/search
```

The search uses an 80/20 question-level stratified split within training. TF-IDF, SVD and normalization fit only the internal training portion. It compares 12 linear configurations and four low-rank configurations with internal epoch selection, refits the three family winners on all training data, and selects fusion on official validation. There are 19 supervised model fits and four preprocessing fits; the seed is 7506. Test data are never loaded. Outputs are written to `retraining/search/`.

Earlier tuning is summarized in `evidence/baseline_training.json`, `evidence/historical_tuning.json` and the report. This minimal package includes the selected baseline recipe and the complete final-stage search, rather than all historical trial scripts and rejected weights. Use training/validation for development, not repeated test-based checkpoint selection.

## 6. Tests, analysis and new audit inputs

```powershell
.\.venv\Scripts\python.exe test_interactions.py
.\.venv\Scripts\python.exe test_math_consistency.py
.\.venv\Scripts\python.exe verify.py
.\.venv\Scripts\python.exe analyze.py --output results/analysis.json
```

Checks cover analytic gradients, arithmetic boundaries, empty inputs, integer outputs, new IDs, batching and candidate permutation. `analyze.py` reproduces post-freeze validation reliability, hard-subset coverage and paired statistics; it does not train or select a model.

For a new audit bank:

```text
python predict.py --method selector.py --checkpoint checkpoint --split audit --data-dir /path/to/audit_inputs --output results/audit-predictions.json
```

When calling `predict.py` directly, set the four thread variables listed above. Prediction does not read references, retrieve answers by ID or use cached predictions.

## 7. Data, provenance and recorded evidence

- `data/` contains the unchanged course train/validation/test banks, references and fixed embeddings for offline evaluation and training reproduction.
- `common.py`, `predict.py`, `score_predictions.py` and `check_data.py` are byte-identical to the supplied tools; see `evidence/official_tools.json`.
- `evidence/asset_manifest.json` records SHA-256 hashes for all six checkpoint files. `freeze.json` preserves the original pre-test freeze record.
- `evidence/decision.json` and `internal_selection.json` record the final grid and selection. Score, resource and paired-comparison JSON files contain measured results.
- `MANIFEST.json` identifies this standalone submission snapshot. Development records may contain the original working paths; standalone reproduction is recorded separately.
- Earlier v1/v2 test results were known before later development. Rounds v3-v13 did not produce a new test-driven search. The final v14 configuration was selected on validation, frozen, and then scored on test without subsequent retuning.

Questions originate from GSM8K. Retain `data/LICENSE.txt` and `data/NOTICE.txt`. The course prepared candidate texts and fixed embeddings with Qwen3-0.6B. No additional pretrained weights are included. Numerical and learning implementations use NumPy, SciPy and scikit-learn; methodological references are listed in `report.pdf`.
