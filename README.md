# Reproducible Evaluation of Phishing URL Detection

This repository reproduces and critically evaluates the public
[Phishing-URL-Detection](https://github.com/vaibhavbichave/Phishing-URL-Detection)
project. It uses a fixed, auditable workflow to compare Logistic Regression,
Random Forest, and Gradient Boosting on the included phishing-feature dataset. The analysis treats
phishing as the positive class, reports security-relevant errors, adds
stratified cross-validation, and distinguishes dataset performance from
real-world deployment claims.

Submission repository: https://github.com/raghdans/phishing_url_detection_submission2

## Repository structure

```text
raghdansproject/
|-- .gitattributes
|-- .gitignore
|-- README.md
|-- requirements.txt
|-- data/
|   `-- phishing.csv
|-- notebooks/
|   `-- phishing_url_detection_analysis.ipynb
|-- reports/
|   |-- phishing_url_detection_report.pdf
|   |-- cross_validation_results.csv
|   |-- threshold_tuning.csv
|   |-- feature_importance.csv
|   `-- final_test_results.csv
|-- src/
|   |-- __init__.py
|   |-- analysis.py
|   `-- generate_report.py
`-- tests/
    `-- test_analysis.py
```

The notebook is in the exact location shown above. Paths are resolved from the
repository root, so the analysis does not depend on Google Colab or a user's
home directory.

## Reproduce the analysis

Python 3.11 or 3.12 is recommended. From the repository root:

```bash
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

Install the pinned dependencies and execute the notebook:

```bash
python -m pip install -r requirements.txt
jupyter nbconvert --to notebook --execute notebooks/phishing_url_detection_analysis.ipynb --output phishing_url_detection_analysis.ipynb --output-dir notebooks --ExecutePreprocessor.timeout=600
python -m src.generate_report
```

Run the automated checks:

```bash
python -m pytest -q
```

The committed notebook is already executed and contains the reference output.
Re-executing it overwrites that notebook with results produced from the bundled
dataset and the fixed random seed.

## Methodology

- Data: 11,054 rows, 30 engineered URL/website predictors, one identifier, and
  one target.
- Identifier handling: `Index` is retained only for tracing errors and is not a
  model feature.
- Target: original `-1` (phishing) is mapped to `1`; original `1` (legitimate)
  is mapped to `0`. Therefore recall means phishing-detection recall.
- Holdout: stratified, group-aware 80/20 split with `random_state=42`.
- Leakage control: identical feature vectors are assigned to the same group and
  never cross development/test or cross-validation boundaries.
- Robustness: five-fold stratified group cross-validation with mean, standard
  deviation, and 95% normal-approximation intervals.
- Models: scaled Logistic Regression, class-weighted Random Forest, and
  Gradient Boosting, all implemented as leakage-resistant pipelines.
- Selection: model and decision-threshold selection use development data only;
  the 20% final test partition remains untouched until both are fixed.
- Threshold: selected from out-of-fold development probabilities by maximizing
  phishing recall subject to at least 95% phishing precision.
- Metrics: accuracy, phishing precision, phishing recall, F1, Matthews
  correlation coefficient, ROC-AUC, confusion matrices, and error counts.

## Reproducibility decisions

- All direct dependencies are version-pinned in `requirements.txt`.
- Randomness is controlled with `random_state=42`.
- Preprocessing is fitted inside each model pipeline.
- Input schema and labels are validated before modeling.
- The notebook can be executed non-interactively from the repository root.
- Unit tests cover path-independent loading, label semantics, identifier
  exclusion, and deterministic splitting.

## Reference results

The dataset contains 5,785 unique predictor vectors among 11,054 rows; 5,269
rows repeat a vector and 64 exact-vector groups contain conflicting labels.
Group-aware partitioning prevents those vectors from crossing evaluation
boundaries.

| Model | Development CV F1 | Final test F1 | Final ROC-AUC |
| --- | ---: | ---: | ---: |
| Random Forest | 0.9474 | 0.9338 | 0.9889 |
| Gradient Boosting | 0.9424 | - | - |
| Logistic Regression | 0.9177 | - | - |

Random Forest was selected using development cross-validation. Its threshold
was selected at 0.55 using out-of-fold development predictions. On the untouched
group-separated test set it missed 81 phishing samples and blocked 49 legitimate
samples. Only the selected model is evaluated on the final test set.

## Scope and limitations

The dataset contains precomputed, mostly ordinal indicators rather than raw
URLs. It has no collection timestamps, and its provenance and sampling process
are incompletely documented by the source project. Random splits therefore do
not demonstrate performance on future phishing campaigns. The results should
be read as a reproducible benchmark on this dataset, not evidence that the
model is ready for production. A deployment study would require temporally
separated data, fresh adversarial samples, feature-collection monitoring,
probability calibration, and threshold selection based on operational costs.

## Source and license note

The dataset was obtained from the source repository linked above. That
repository should be consulted for its original attribution and licensing
terms. This educational reproduction does not assert ownership of the data.
