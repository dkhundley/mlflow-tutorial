"""Model builders and evaluation helpers for the traditional ML tutorial.

These helpers mirror the model code shown in the notebook. They are deliberately
small: the goal is not to build the best classifier possible, but to create
clear examples for MLflow tracking, signatures, model registration, and serving.
"""

from __future__ import annotations

import sys
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Literal

from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Optuna chooses between these solvers during the Iris hyperparameter search.
# The Literal type makes the accepted values visible to readers and type checkers.
SolverName = Literal['lbfgs', 'saga']


def build_iris_pipeline(
    c_value: float = 1.0,
    solver: SolverName = 'lbfgs',
    max_iter: int = 400,
    random_state: int = 42,
) -> Pipeline:
    """Build the logistic regression pipeline used for the Iris demo.

    A Pipeline is useful in MLflow examples because the preprocessing and model
    are logged as one model artifact. Serving the model later therefore applies
    the same scaling that was used during training.
    """

    return Pipeline(
        [
            # Logistic Regression is sensitive to feature scale, so we fit a
            # StandardScaler before the classifier.
            ('scaler', StandardScaler()),
            (
                'classifier',
                LogisticRegression(
                    # `C` controls inverse regularization strength. Optuna will
                    # search this value on a log scale in the tuning helper.
                    C = c_value,
                    solver = solver,
                    max_iter = max_iter,
                    random_state = random_state,
                ),
            ),
        ]
    )


def build_iris_pipeline_from_trial(trial: Any) -> Pipeline:
    """Build an Iris model from an Optuna trial or FixedTrial.

    Optuna Trial objects expose `suggest_*` methods during search. FixedTrial
    exposes the same interface for one final training run after the best
    parameters are known, which lets the notebook reuse one pipeline builder.
    """

    return build_iris_pipeline(
        c_value = trial.suggest_float('C', 1e-3, 1e2, log = True),
        solver = trial.suggest_categorical('solver', ['lbfgs', 'saga']),
        max_iter = trial.suggest_int('max_iter', 300, 1000),
        random_state = 42,
    )


def build_wine_model(random_state: int = 42) -> RandomForestClassifier:
    """Build the random forest model used for the Wine demo.

    The Wine section is about multi-model serving rather than tuning, so a
    straightforward Random Forest gives a reliable second model without adding
    another optimization loop.
    """

    return RandomForestClassifier(
        # A moderately large forest keeps the example accurate while still being
        # quick enough for local notebook execution.
        n_estimators = 300,
        max_depth = None,
        random_state = random_state,
    )


def calculate_classification_metrics(
    y_true: NDArray[Any],
    y_pred: NDArray[Any],
    prefix: str = '',
) -> dict[str, float]:
    """Calculate the core classification metrics logged by the notebook.

    Macro-averaged precision/recall/F1 treat each class equally. That is useful
    in tutorials because the metric names are easy to compare across datasets.
    """

    metric_prefix = f'{prefix}_' if prefix else ''
    return {
        f'{metric_prefix}accuracy': float(accuracy_score(y_true, y_pred)),
        f'{metric_prefix}precision_macro': float(
            precision_score(y_true, y_pred, average = 'macro', zero_division = 0)
        ),
        f'{metric_prefix}recall_macro': float(
            recall_score(y_true, y_pred, average = 'macro', zero_division = 0)
        ),
        f'{metric_prefix}f1_macro': float(f1_score(y_true, y_pred, average = 'macro')),
    }


def build_model_pip_requirements(
    extra_packages: list[str] | None = None,
) -> list[str]:
    """Return pinned package requirements for logged local tutorial models.

    MLflow can infer dependencies, but explicit requirements make the model
    package easier to explain and reduce warning noise during local logging.
    """

    # These are the packages the logged sklearn and pyfunc models need at
    # inference time. Optional extras can be appended by callers for extensions.
    packages = [
        'mlflow',
        'scikit-learn',
        'skops',
        'numpy',
        'scipy',
        'joblib',
        'threadpoolctl',
        'pandas',
        'cloudpickle',
    ]
    if extra_packages:
        packages.extend(extra_packages)

    requirements = []
    for package_name in packages:
        try:
            # Pinning the current installed version captures the environment that
            # actually produced the local model artifact.
            requirements.append(f'{package_name}=={version(package_name)}')
        except PackageNotFoundError:
            # Keep this helper tolerant of optional packages. If a package is not
            # installed, it simply should not be added to the model environment.
            continue

    return requirements


def build_model_conda_env(
    name: str,
    pip_requirements: list[str] | None = None,
) -> dict[str, Any]:
    """Build an explicit model environment for MLflow model logging.

    The tutorial serves models with `--no-conda`, but logging the environment is
    still valuable because the MLflow UI shows what would be needed to recreate
    the model elsewhere.
    """

    requirements = pip_requirements or build_model_pip_requirements()
    return {
        'name': name,
        'channels': ['conda-forge'],
        'dependencies': [
            # Use the exact Python patch version from the local `.venv` so the
            # logged model environment matches the teaching environment.
            f'python={sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}',
            'pip',
            {'pip': requirements},
        ],
    }
