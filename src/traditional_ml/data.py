"""Dataset loading helpers for the traditional ML tutorial.

The notebook intentionally skips exploratory data analysis so it can focus on
MLflow concepts. These helpers keep the dataset-loading story small and
repeatable: load a built-in sklearn dataset, split it deterministically, and
return the pieces with enough metadata to build examples and signatures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split

# Fixed defaults keep notebook screenshots, test expectations, and local MLflow
# runs stable across repeated executions.
DEFAULT_RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.2


@dataclass(frozen=True)
class DatasetSplit:
    """A deterministic train/test split with feature metadata.

    Keeping the split in one object avoids passing six related variables through
    every helper. The notebook still shows the individual variables inline, but
    the hardened code benefits from a small typed container.
    """

    x_train: NDArray[Any]
    x_test: NDArray[Any]
    y_train: NDArray[Any]
    y_test: NDArray[Any]
    feature_names: list[str]
    target_names: list[str]

    def input_example(self, rows: int = 5) -> NDArray[Any]:
        # MLflow input examples should look like real inference input. A few rows
        # from the test set are enough for the model UI and serving example.
        return self.x_test[:rows]


def load_iris_split(
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> DatasetSplit:
    """Load Iris and return a deterministic stratified train/test split.

    Iris is the first model in the tutorial because it is small, fast, and
    familiar. The resulting split has 4 numeric features.
    """

    iris = load_iris()
    return _split_dataset(
        features = iris.data,
        target = iris.target,
        feature_names = list(iris.feature_names),
        target_names = list(iris.target_names),
        test_size = test_size,
        random_state = random_state,
    )


def load_wine_split(
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> DatasetSplit:
    """Load Wine and return a deterministic stratified train/test split.

    Wine is used as the second model in the pyfunc-router section. It has 13
    numeric features, which makes it a useful contrast with Iris during routing.
    """

    wine = load_wine()
    return _split_dataset(
        features = wine.data,
        target = wine.target,
        feature_names = list(wine.feature_names),
        target_names = list(wine.target_names),
        test_size = test_size,
        random_state = random_state,
    )


def _split_dataset(
    features: NDArray[Any],
    target: NDArray[Any],
    feature_names: list[str],
    target_names: list[str],
    test_size: float,
    random_state: int,
) -> DatasetSplit:
    # Convert to arrays before splitting so the returned object is consistent
    # even if sklearn changes the exact dataset container type in the future.
    x_train, x_test, y_train, y_test = train_test_split(
        np.asarray(features),
        np.asarray(target),
        test_size = test_size,
        random_state = random_state,
        # Stratification preserves class proportions in train and test sets.
        # That matters for small teaching datasets where random splits can
        # otherwise omit or underrepresent a class.
        stratify = target,
    )

    return DatasetSplit(
        x_train = x_train,
        x_test = x_test,
        y_train = y_train,
        y_test = y_test,
        feature_names = feature_names,
        target_names = target_names,
    )
