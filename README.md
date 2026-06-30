# Phishing URL Detection - Reproducible Critical Evaluation

This repository contains a reproducible Data Science in Cybersecurity project on phishing URL detection. It reproduces and extends the selected source while correcting the evaluation workflow so that the final test set remains untouched during model and threshold selection.

## Selected Source

- Original repository: https://github.com/vaibhavbichave/Phishing-URL-Detection
- Dataset: `data/phishing.csv`
- Topic: Phishing Detection

## Improvements Over the Original Submission

- visible project files instead of code hidden inside a ZIP archive
- complete setup and execution instructions
- missing-value, duplicate-row, duplicate-column, and constant-column checks
- Logistic Regression, Random Forest, and Gradient Boosting comparison
- 5-fold stratified cross-validation on the development set
- out-of-fold threshold selection focused on phishing recall
- untouched 20% final test set
- ROC-AUC and Matthews Correlation Coefficient
- feature-importance and error analysis

## Dataset

- 11,054 rows and 32 raw columns
- 30 modeling features after dropping `Index` and the target
- no missing values, duplicated rows, duplicated columns, or constant columns
- original `-1` = phishing, mapped to positive class `1`
- original `1` = legitimate, mapped to class `0`

## Evaluation Design

The data is first split into an 80% development set and an untouched 20% final test set. Model selection uses 5-fold stratified cross-validation only on the development set. Random Forest is selected by mean F1. Its threshold is selected from out-of-fold development probabilities by maximizing phishing recall while requiring at least 95% phishing precision. The selected model is then trained on the full development set and evaluated once on the final test set.

## Development Cross-Validation

| model | accuracy_mean | accuracy_std | precision_mean | recall_mean | f1_mean | roc_auc_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random Forest | 0.9705 | 0.0010 | 0.9734 | 0.9597 | 0.9665 | 0.9956 |
| Gradient Boosting | 0.9485 | 0.0050 | 0.9509 | 0.9321 | 0.9414 | 0.9898 |
| Logistic Regression | 0.9267 | 0.0059 | 0.9283 | 0.9045 | 0.9162 | 0.9787 |

## Threshold Selection

The selected out-of-fold threshold is **0.35**. On development predictions it achieved 0.9538 precision and 0.9745 recall. Threshold selection did not use the final test set.

## Final Test Result

| model | threshold | accuracy | precision | recall | F1 | MCC | ROC-AUC | missed phishing | blocked legitimate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Random Forest | 0.35 | 0.9738 | 0.9666 | 0.9745 | 0.9705 | 0.9469 | 0.9966 | 25 | 33 |

## Repository Structure

```text
.
|-- README.md
|-- requirements.txt
|-- data/
|   `-- phishing.csv
|-- notebooks/
|   `-- phishing_url_detection_enhanced.ipynb
|-- reports/
|   |-- phishing_url_detection_report_enhanced.md
|   |-- phishing_url_detection_report_enhanced.pdf
|   |-- model_results.csv
|   |-- cross_validation_results.csv
|   |-- threshold_tuning.csv
|   `-- feature_importance.csv
`-- src/
    `-- __init__.py
```

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/phishing_url_detection_enhanced.ipynb
```

When using Google Colab, upload `phishing.csv` to `/content/phishing.csv`. The notebook automatically checks the repository path, the current directory, and `/content/phishing.csv`.

## Submission Note

Upload the folders and files directly to GitHub so the notebook, report, dataset, requirements, figures, and result files are visible and inspectable. The ZIP archive is optional and should not replace the visible repository files.
