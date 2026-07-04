from pathlib import Path

import pandas as pd

from src.analysis import feature_groups, grouped_split, load_dataset, prepare_data, repository_root, split_data


ROOT = Path(__file__).resolve().parents[1]


def test_repository_root_resolves_from_notebook_folder():
    assert repository_root(ROOT / "notebooks") == ROOT


def test_dataset_schema_and_semantics():
    frame = load_dataset(ROOT)
    predictors, target, identifiers = prepare_data(frame)
    assert len(frame) == len(predictors) == len(target) == len(identifiers)
    assert "Index" not in predictors.columns
    assert "class" not in predictors.columns
    assert set(target.unique()) == {0, 1}
    assert target.loc[frame["class"].eq(-1)].eq(1).all()


def test_split_is_deterministic_and_stratified():
    predictors, target, _ = prepare_data(load_dataset(ROOT))
    first = split_data(predictors, target)
    second = split_data(predictors, target)
    pd.testing.assert_index_equal(first[0].index, second[0].index)
    assert abs(first[3].mean() - target.mean()) < 0.01


def test_grouped_split_prevents_exact_vector_leakage():
    predictors, target, _ = prepare_data(load_dataset(ROOT))
    groups = feature_groups(predictors)
    first = grouped_split(predictors, target, groups)
    second = grouped_split(predictors, target, groups)
    assert set(first[4]).isdisjoint(set(first[5]))
    pd.testing.assert_index_equal(first[0].index, second[0].index)
    assert abs(first[3].mean() - target.mean()) < 0.02
