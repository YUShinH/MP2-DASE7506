# Mini-project 2: Selecting Reliable LLM Reasoning

**Author:** HE YUXIN  
**Student ID:** 3036897306  
**Course:** DASE7506

This project selects one answer from four candidate mathematical solutions. The final v14 selector combines a question-conditioned low-rank scorer (weight 0.75) with a pairwise linear/neural baseline (weight 0.25). Evaluation runs on CPU using the supplied frozen checkpoint; retraining is not required.

## 1. Repository, checkpoint and report

- **Code:** [YUShinH/MP2-DASE7506](https://github.com/YUShinH/MP2-DASE7506)
- **Final checkpoint:** [checkpoint/](https://github.com/YUShinH/MP2-DASE7506/tree/main/checkpoint)
- **Required baseline weights:** [checkpoint/v2/](https://github.com/YUShinH/MP2-DASE7506/tree/main/checkpoint/v2)
- **Report:** [report.pdf](https://github.com/YUShinH/MP2-DASE7506/blob/main/report.pdf)

**The complete final checkpoint is in the repository root's `checkpoint/` directory.** Download the repository using **Code > Download ZIP**, or clone it:

```bash
git clone https://github.com/YUShinH/MP2-DASE7506.git
cd MP2-DASE7506
```

Keep all six checkpoint files below, including the entire `v2/` subdirectory. These are components of the final fused model; removing `v2/` prevents inference. Total inference asset size: **6,246,341 bytes (5.957 MiB)**.

```text
checkpoint/
  config.json
  text_encoder.joblib
  text_model.npz
  v2/
    config.json
    neural.npz
    pair.npz
```

Download actual binary files using the repository ZIP, `git clone`, or GitHub's **Download raw file** action. Do not save the HTML file-view page as a checkpoint file.

## 2. Environment and input data

Use **Python 3.12**. Recorded results used Python 3.12.14 and the versions pinned in `requirements.txt`:

```text
numpy==2.1.3
scipy==1.15.3
scikit-learn==1.6.1
joblib==1.4.2
threadpoolctl==3.5.0
```

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Linux/macOS:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

No GPU, CUDA, PyTorch, CatBoost or API key is needed. Prediction and scoring run offline after installing dependencies and obtaining the input data.

Run commands from the repository root, where `evaluate.py`, `selector.py` and `checkpoint/` are located. The relevant evaluation layout is:

```text
MP2-DASE7506/
  readme.md
  report.pdf
  requirements.txt
  selector.py
  features.py
  math_consistency.py
  common.py
  predict.py
  score_predictions.py
  check_data.py
  evaluate.py
  verify.py
  checkpoint/
  data/
    manifest.json
    LICENSE.txt
    NOTICE.txt
    gsm8k_train.jsonl
    gsm8k_validation.jsonl
    gsm8k_test.jsonl
    candidates/
      gsm8k_train.jsonl
      gsm8k_train.npz
      gsm8k_validation.jsonl
      gsm8k_validation.npz
      gsm8k_test.jsonl
      gsm8k_test.npz
  evidence/                   Required by the additional verify.py checks
  results/                    Evaluation outputs
```

**Accuracy evaluation requires the original course data, including `data/candidates/`.** The JSONL files contain candidate solutions, the NPZ files contain the supplied embeddings, and references are needed for scoring. If any data files are missing from your downloaded copy, copy the unchanged `code/data/` directory from the [official course repository](https://github.com/xudongwu-0/DASE7506-MP2) into this repository as `data/`. Preserve the manifest and licenses. Avoid nested paths such as `data/data/` or `checkpoint/checkpoint/`.

## 3. Reproduce accuracy using the final checkpoint

The following Windows commands use the virtual environment created above:

```powershell
.\.venv\Scripts\python.exe check_data.py
.\.venv\Scripts\python.exe evaluate.py --split test --method selector.py --checkpoint checkpoint --output-dir results
.\.venv\Scripts\python.exe evaluate.py --split validation --method selector.py --checkpoint checkpoint --output-dir results
```

On Linux/macOS, replace `.\.venv\Scripts\python.exe` with `.venv/bin/python`. When the appropriate environment is activated, `python` can be used instead.

| Split | Correct / total | Accuracy | Score file |
|---|---:|---:|---|
| Test | 388 / 467 | **83.08351177730192%** | `results/test-score.json` |
| Validation | 753 / 798 | **94.3609022556391%** | `results/validation-score.json` |

Open the score JSON after the run. Its `accuracy` field is a percentage:

```json
{
  "split": "test",
  "correct": 388,
  "n": 467,
  "accuracy": 83.08351177730192
}
```

`evaluate.py` launches fresh prediction and independent official scoring using the same Python interpreter. Existing uploaded result files are records; the commands above calculate fresh results. Each run writes predictions, score and resource JSON files to `results/`.

The evaluator sets OMP, OpenBLAS, MKL and NumExpr to four threads. Complete local test evaluation took approximately 13.3 seconds on an AMD Ryzen 7 5800H; runtime varies by machine. Use `total_seconds` in `results/test-resources.json` to inspect complete prediction and scoring time.

The final test result improves on the v2 baseline by eight questions (1.7131 percentage points). The paired confidence interval includes zero. Earlier v1/v2 test results were known during development; final v14 settings were selected on validation, frozen, and then evaluated on test without subsequent retuning.

## 4. Verify checkpoint integrity and the inference interface

### Checkpoint-only verification

This check requires neither Python nor the dataset. In PowerShell at the repository root:

```powershell
Get-ChildItem ./checkpoint -Recurse -File | Get-FileHash -Algorithm SHA256 | Format-List Path,Hash
```

Compare all six SHA-256 values with the following table; hexadecimal letter case does not matter. These hashes identify the frozen local submission assets.

| File relative to checkpoint/ | Bytes | Expected SHA-256 |
|---|---:|---|
| `config.json` | 132 | `3c1aca8cae7695aaa38eff832c1a363df919dcb88f57bfac9083d01c691a67a0` |
| `text_encoder.joblib` | 5463666 | `444a1c32c1435f839c174a69388af4912c0ffd42d6e59edf0a5dc4ac71ae836f` |
| `text_model.npz` | 199670 | `266072cda7da188a3866459eb77f50ceb8e708bec058df07a2cbc4af28b8e8ca` |
| `v2/config.json` | 709 | `b5a93564082c1375e2ee57fc61a15cd3265816439b89bf7a81d80dec01cd3ea0` |
| `v2/neural.npz` | 567980 | `8158f9f2fcb779d3699025bb0cf51adc6df2d5d572c849725f56e76f99e43388` |
| `v2/pair.npz` | 14184 | `3770b5d82ab4f1ac596232b492a5e8a137fbd8857e54ea4aae49c465f1ad1c8b` |

### Full interface verification

With the original `data/` and the submission's `evidence/asset_manifest.json` and `evidence/official_tools.json` present, run:

```text
python verify.py --checkpoint checkpoint
```

Expected success message:

```text
PASS: asset hashes/size, official tools, output contract, empty/new IDs, batching and candidate permutation
```

This checks asset hashes and size, unchanged official tools, valid integer outputs, empty inputs, ID independence, batching consistency and candidate permutation. If the evidence files are absent from your copy, restore them from the complete submission package before running `verify.py`. The checkpoint-only SHA-256 check above remains usable without those supporting files.

## 5. Run in PyCharm

Open the downloaded **MP2-DASE7506** directory as the project. Select a Python 3.12 interpreter with `requirements.txt` installed. In **Run > Edit Configurations > + > Python**, enter:

| Field | Value |
|---|---|
| Name | `MP2_Final_Test` |
| Script path | `$PROJECT_DIR$/evaluate.py` |
| Script parameters / Parameters | `--split test --method selector.py --checkpoint checkpoint --output-dir results` |
| Working directory | `$PROJECT_DIR$` |
| Python interpreter | The Python 3.12 environment containing the dependencies |
| Environment variables | `PYTHONUNBUFFERED=1` |

The **Parameters** field is directly below the script path. Enable **Script parameters** under **Modify options** if hidden. Click **Apply**, then **Run**.

For validation, duplicate the configuration and change the parameters to:

```text
--split validation --method selector.py --checkpoint checkpoint --output-dir results
```

For integrity and interface checks, create another configuration pointing to `verify.py`, with parameters `--checkpoint checkpoint` and the same working directory. Running `selector.py` directly does not calculate accuracy.

`$PROJECT_DIR$` is valid only if the opened project root directly contains `evaluate.py`, `checkpoint/` and `data/`. If you opened a parent folder, select the actual submission folder for the working directory and use the full path to `evaluate.py`. The original local submission folder may be named `MP2_Submission_v14`; its contents serve the same role as the downloaded repository root.

Common problems:

- **ModuleNotFoundError:** install `requirements.txt` into the interpreter selected by this run configuration.
- **FileNotFoundError:** check the working directory and restore missing original data or checkpoint files.
- **Hash mismatch:** download the affected file again and compare all six checkpoint hashes. Do not mix checkpoints from different experiments.
- **verify.py AssertionError:** inspect the traceback to identify the failed integrity or interface check; ensure supporting evidence and official tools are unchanged.

## 6. Training reproduction and additional checks

The complete submission includes `train.py`, `search.py`, `training_utils.py`, the test scripts and experimental evidence. These are unnecessary for evaluating the frozen checkpoint but support reproduction.

Fixed-configuration retraining:

```text
python train.py --output retraining/checkpoint
python verify.py --checkpoint retraining/checkpoint
python evaluate.py --split validation --method selector.py --checkpoint retraining/checkpoint --output-dir retraining/results
```

`train.py` reads training inputs and references only. It reconstructs the selected pairwise linear model (C=.01), the 64-unit/20-epoch neural baseline, the shared TF-IDF/SVD encoder, and the rank-32 low-rank scorer (factor L2=.01, 40 epochs). It writes a separate checkpoint and does not overwrite the supplied model. Fixed retraining was verified locally; numerical and serialization differences can arise across platforms, so the supplied frozen checkpoint defines the reported result.

Replay the final development search:

```text
python search.py --baseline checkpoint/v2 --output retraining/search
```

The search uses an 80/20 question-level stratified split within training. Preprocessing fits only the internal training portion. It compares 12 linear and four low-rank configurations, refits the three family winners on all training data, and selects fusion on official validation. There are 19 supervised model fits and four preprocessing fits, with seed 7506. Test data are not loaded by this search.

Additional checks and post-freeze analysis:

```text
python test_interactions.py
python test_math_consistency.py
python analyze.py --output results/analysis.json
```

The analysis reproduces validation reliability, hard-subset coverage and paired statistics without selecting or training a model. See `report.pdf` for methods, tuning, ablations, failed attempts, uncertainty and limitations.

## 7. Data and provenance

Course data and the original tools are provided by [DASE7506-MP2](https://github.com/xudongwu-0/DASE7506-MP2). Questions originate from GSM8K, and candidate solutions and fixed embeddings were prepared with Qwen3-0.6B. Retain `data/LICENSE.txt` and `data/NOTICE.txt`. No additional pretrained weights are required.

`evidence/` records checkpoint hashes, official-tool hashes, selection decisions, scores and resources. `MANIFEST.json` identifies the packaged local submission snapshot; subsequent GitHub edits or added output files can differ from that snapshot. Development evidence can contain historical local paths. The report contains methodological references and the AI assistance declaration.
