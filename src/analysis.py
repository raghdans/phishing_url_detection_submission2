"""Reusable, testable components for the phishing URL experiment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedGroupKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
TARGET = "class"
IDENTIFIER_COLUMNS = ("Index",)


def repository_root(start: Path | None = None) -> Path:
    """Find the repository root from the root, notebook folder, or a child path."""
    current = (start or Path.cwd()).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "data" / "phishing.csv").is_file():
            return candidate
    raise FileNotFoundError("Could not find data/phishing.csv in this path or its parents")


def load_dataset(root: Path | None = None) -> pd.DataFrame:
    """Load and validate the committed dataset."""
    project_root = repository_root(root)
    frame = pd.read_csv(project_root / "data" / "phishing.csv")
    if TARGET not in frame.columns:
        raise ValueError(f"Required target column {TARGET!r} is missing")
    labels = set(frame[TARGET].dropna().unique())
    if labels != {-1, 1}:
        raise ValueError(f"Expected labels {{-1, 1}}, found {sorted(labels)}")
    if frame[TARGET].isna().any():
        raise ValueError("Target contains missing values")
    return frame


def prepare_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return predictors, phishing-positive labels, and traceable row identifiers."""
    identifiers = (
        frame["Index"].copy()
        if "Index" in frame.columns
        else pd.Series(frame.index, index=frame.index, name="row_id")
    )
    drop_columns = [TARGET, *(c for c in IDENTIFIER_COLUMNS if c in frame.columns)]
    predictors = frame.drop(columns=drop_columns).select_dtypes(include=[np.number]).copy()
    if predictors.empty:
        raise ValueError("No numeric predictor columns were found")
    target = frame[TARGET].map({-1: 1, 1: 0}).astype("int8")
    target.name = "is_phishing"
    return predictors, target, identifiers


def split_data(
    predictors: pd.DataFrame, target: pd.Series, test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create the deterministic stratified holdout split."""
    return train_test_split(
        predictors,
        target,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=target,
    )


def feature_groups(predictors: pd.DataFrame) -> pd.Series:
    """Assign identical predictor vectors to the same exact-match group."""
    codes, _ = pd.factorize(pd.MultiIndex.from_frame(predictors), sort=True)
    return pd.Series(codes, index=predictors.index, name="feature_group")


def grouped_split(
    predictors: pd.DataFrame, target: pd.Series, groups: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Create a deterministic split with no identical vector crossing partitions."""
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    candidates = list(splitter.split(predictors, target, groups))
    overall_rate = float(target.mean())
    development_index, test_index = min(
        candidates,
        key=lambda split: (
            abs(len(split[1]) / len(target) - 0.20)
            + abs(float(target.iloc[split[1]].mean()) - overall_rate)
        ),
    )
    return (
        predictors.iloc[development_index],
        predictors.iloc[test_index],
        target.iloc[development_index],
        target.iloc[test_index],
        groups.iloc[development_index],
        groups.iloc[test_index],
    )


def build_models(feature_names: list[str]) -> dict[str, Pipeline]:
    """Construct leakage-resistant model pipelines."""
    linear_preprocessor = ColumnTransformer(
        [("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")),
                               ("scaler", StandardScaler())]), feature_names)],
        remainder="drop",
    )
    tree_preprocessor = ColumnTransformer(
        [("numeric", SimpleImputer(strategy="median"), feature_names)],
        remainder="drop",
    )
    return {
        "Logistic Regression": Pipeline([
            ("preprocess", linear_preprocessor),
            ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]),
        "Random Forest": Pipeline([
            ("preprocess", tree_preprocessor),
            ("model", RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )),
        ]),
        "Gradient Boosting": Pipeline([
            ("preprocess", tree_preprocessor),
            ("model", GradientBoostingClassifier(random_state=RANDOM_STATE)),
        ]),
    }


def holdout_metrics(target: pd.Series, prediction: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    """Calculate metrics with phishing encoded as the positive class."""
    return {
        "accuracy": accuracy_score(target, prediction),
        "phishing_precision": precision_score(target, prediction, pos_label=1),
        "phishing_recall": recall_score(target, prediction, pos_label=1),
        "f1": f1_score(target, prediction, pos_label=1),
        "mcc": matthews_corrcoef(target, prediction),
        "roc_auc": roc_auc_score(target, probability),
    }


def cross_validation_summary(
    models: dict[str, Pipeline],
    predictors: pd.DataFrame,
    target: pd.Series,
    groups: pd.Series | None = None,
) -> pd.DataFrame:
    """Return five-fold scores and normal-approximation uncertainty intervals."""
    scoring = {
        "accuracy": "accuracy",
        "phishing_precision": make_scorer(precision_score, pos_label=1),
        "phishing_recall": make_scorer(recall_score, pos_label=1),
        "f1": make_scorer(f1_score, pos_label=1),
        "mcc": make_scorer(matthews_corrcoef),
        "roc_auc": "roc_auc",
    }
    folds = (
        StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        if groups is not None
        else StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    )
    rows: list[dict[str, float | str]] = []
    for model_name, model in models.items():
        scores = cross_validate(
            model,
            predictors,
            target,
            groups=groups,
            cv=folds,
            scoring=scoring,
            n_jobs=1,
        )
        for metric in scoring:
            values = scores[f"test_{metric}"]
            mean = float(values.mean())
            std = float(values.std(ddof=1))
            margin = 1.96 * std / np.sqrt(len(values))
            rows.append({
                "model": model_name,
                "metric": metric,
                "mean": mean,
                "std": std,
                "ci95_low": mean - margin,
                "ci95_high": mean + margin,
            })
    return pd.DataFrame(rows)


def out_of_fold_threshold_selection(
    model: Pipeline,
    predictors: pd.DataFrame,
    target: pd.Series,
    precision_floor: float = 0.95,
    groups: pd.Series | None = None,
) -> tuple[pd.DataFrame, float, str, np.ndarray]:
    """Select a security-oriented threshold using development data only."""
    folds = (
        StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        if groups is not None
        else StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    )
    probabilities = cross_val_predict(
        model,
        predictors,
        target,
        groups=groups,
        cv=folds,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]
    rows = []
    for threshold in np.arange(0.20, 0.81, 0.05):
        prediction = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(target, prediction, labels=[0, 1]).ravel()
        rows.append({
            "threshold": float(threshold),
            "phishing_precision": precision_score(target, prediction, pos_label=1),
            "phishing_recall": recall_score(target, prediction, pos_label=1),
            "f1": f1_score(target, prediction, pos_label=1),
            "missed_phishing": int(fn),
            "blocked_legitimate": int(fp),
        })
    table = pd.DataFrame(rows)
    eligible = table.loc[table["phishing_precision"].ge(precision_floor)]
    if eligible.empty:
        chosen = table.sort_values(["f1", "phishing_recall"], ascending=False).iloc[0]
        rule = "maximum out-of-fold F1 because no threshold met the precision floor"
    else:
        chosen = eligible.sort_values(["phishing_recall", "f1"], ascending=False).iloc[0]
        rule = f"maximum out-of-fold phishing recall with precision >= {precision_floor:.2f}"
    return table, float(chosen["threshold"]), rule, probabilities
