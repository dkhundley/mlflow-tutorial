"""Training and registration helpers for the traditional ML tutorial.

This module contains the durable version of the notebook's MLflow workflow:
Optuna tuning, final model training, model registration, loading from the
registry, and packaging a multi-model pyfunc router.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Any, Callable

import mlflow
import optuna
import pandas as pd
from mlflow.models import infer_signature

from traditional_ml.data import DatasetSplit
from traditional_ml.models import (
    build_iris_pipeline_from_trial,
    build_model_conda_env,
    build_model_pip_requirements,
    build_wine_model,
    calculate_classification_metrics,
)
from traditional_ml.router import MultiModelRouter, build_router_examples


@dataclass(frozen=True)
class RegisteredModelResult:
    """Result metadata for a logged and registered MLflow model.

    MLflow returns a model-info object, but the notebook usually wants a smaller
    teaching-friendly bundle: metrics, model URI, and registered version.
    """

    model_info: Any
    metrics: dict[str, float]
    model_uri: str
    registered_model_version: str | None


def utc_run_timestamp() -> str:
    """Return a UTC timestamp suitable for MLflow run names.

    Using UTC avoids local-time ambiguity while still making run names readable
    in the MLflow UI.
    """

    return pd.Timestamp.now(tz = 'UTC').strftime('%Y%m%d_%H%M%S')


def make_iris_objective(split: DatasetSplit) -> Callable[[optuna.Trial], float]:
    """Create an Optuna objective that logs nested trial runs to MLflow.

    Optuna calls the returned function once per trial. Each trial starts a nested
    MLflow run, trains a candidate pipeline, and returns F1 macro so Optuna knows
    whether that candidate was better than previous ones.
    """

    def objective(trial: optuna.Trial) -> float:
        # The trial object owns the hyperparameter suggestions. The model builder
        # knows which parameters to ask Optuna for.
        model = build_iris_pipeline_from_trial(trial)

        with mlflow.start_run(nested = True, run_name = f'optuna_trial_{trial.number}'):
            model.fit(split.x_train, split.y_train)
            predictions = model.predict(split.x_test)

        # We optimize the same F1 macro metric shown later in the final model
        # metrics so learners can connect tuning and evaluation.
        metrics = calculate_classification_metrics(split.y_test, predictions)
        return metrics['f1_macro']

    return objective


def run_iris_hyperparameter_search(
    split: DatasetSplit,
    run_timestamp: str,
    n_trials: int = 30,
    study_name: str = 'iris_logreg_optimization',
) -> dict[str, Any]:
    """Run deterministic Optuna hyperparameter tuning for the Iris model.

    The sampler seed makes the search repeatable for tests and demonstrations.
    The returned dictionary can be passed directly into an Optuna FixedTrial for
    final model training.
    """

    sampler = optuna.samplers.TPESampler(seed = 42)
    study = optuna.create_study(
        direction = 'maximize',
        study_name = study_name,
        sampler = sampler,
    )

    # The parent MLflow run groups all nested Optuna trial runs together in the
    # UI, making the tuning section easier to inspect.
    with mlflow.start_run(run_name = f'optuna_hyperparameter_search_{run_timestamp}'):
        study.optimize(
            make_iris_objective(split),
            n_trials = n_trials,
            show_progress_bar = True,
        )

    return dict(study.best_params)


def train_register_iris_model(
    split: DatasetSplit,
    best_params: dict[str, Any],
    run_timestamp: str,
    registered_model_name: str = 'iris_model',
) -> RegisteredModelResult:
    """Train, log, and register the final Iris classifier.

    The notebook first searches for parameters, then trains one final model with
    those parameters. This function mirrors that concept and logs only the final
    model as a registered model.
    """

    # FixedTrial lets us reuse the same pipeline builder that Optuna used during
    # search, but with the known best parameters.
    fixed_trial = optuna.trial.FixedTrial(best_params)
    model = build_iris_pipeline_from_trial(fixed_trial)

    with mlflow.start_run(run_name = f'iris_model_training_{run_timestamp}'):
        model.fit(split.x_train, split.y_train)
        predictions = model.predict(split.x_test)
        metrics = calculate_classification_metrics(split.y_test, predictions)

        mlflow.log_metrics(metrics)
        model_info = mlflow.sklearn.log_model(
            sk_model = model,
            name = 'model',
            # Signatures tell MLflow what input/output shape and dtype the model
            # expects. That becomes important when serving via `/invocations`.
            signature = infer_signature(split.x_test, predictions),
            # Input examples make the MLflow UI and generated serving examples
            # more concrete for learners.
            input_example = split.input_example(),
            conda_env = build_model_conda_env('iris_model_env'),
            serialization_format = 'skops',
            registered_model_name = registered_model_name,
        )

    return _registered_model_result(model_info = model_info, metrics = metrics)


def train_register_wine_model(
    split: DatasetSplit,
    run_timestamp: str,
    registered_model_name: str = 'wine_model',
) -> RegisteredModelResult:
    """Train, log, and register the Wine classifier.

    This second model exists to demonstrate pyfunc routing. It follows the same
    MLflow logging pattern as the Iris model so the two registered models can be
    bundled together later.
    """

    model = build_wine_model()

    with mlflow.start_run(run_name = f'wine_model_training_{run_timestamp}'):
        model.fit(split.x_train, split.y_train)
        predictions = model.predict(split.x_test)
        metrics = calculate_classification_metrics(
            split.y_test,
            predictions,
            prefix = 'wine',
        )

        mlflow.log_metrics(metrics)
        model_info = mlflow.sklearn.log_model(
            sk_model = model,
            name = 'model',
            signature = infer_signature(split.x_test, predictions),
            input_example = split.input_example(),
            conda_env = build_model_conda_env('wine_model_env'),
            serialization_format = 'skops',
            registered_model_name = registered_model_name,
        )

    return _registered_model_result(model_info = model_info, metrics = metrics)


def latest_registered_model_version(
    model_name: str,
    client: mlflow.tracking.MlflowClient | None = None,
) -> int:
    """Return the latest registered version number for a model name.

    The tutorial serves "whatever we just registered" locally. Looking up the
    latest version keeps the notebook from hardcoding version numbers after each
    rerun.
    """

    resolved_client = client or mlflow.tracking.MlflowClient()
    versions = resolved_client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise ValueError(f"No registered versions found for model '{model_name}'.")

    return max(int(version.version) for version in versions)


def load_latest_sklearn_model(model_name: str) -> tuple[Any, int, str]:
    """Load the latest registered sklearn model by name.

    This helper backs the "load and invoke the registered model directly in
    Python" section before the notebook moves on to HTTP serving.
    """

    latest_version = latest_registered_model_version(model_name)
    model_uri = f'models:/{model_name}/{latest_version}'
    return mlflow.sklearn.load_model(model_uri), latest_version, model_uri


def register_multi_model_router(
    iris_split: DatasetSplit,
    wine_split: DatasetSplit,
    run_timestamp: str,
    repo_root: Path,
    iris_model_name: str = 'iris_model',
    wine_model_name: str = 'wine_model',
    registered_model_name: str = 'multi_model_router',
) -> RegisteredModelResult:
    """Register a pyfunc model that routes Iris and Wine requests.

    A pyfunc model can include arbitrary artifacts. Here the artifacts are two
    already-registered sklearn model packages. The router loads both at runtime
    and chooses which one to call for each request row.
    """

    with tempfile.TemporaryDirectory() as artifact_staging_dir:
        staging_root = Path(artifact_staging_dir)
        # Download each registered model into a uniquely named directory. MLflow
        # model artifacts are often named simply `artifacts`; without staging,
        # both children can collide when packaged into one pyfunc model.
        iris_artifact_path = mlflow.artifacts.download_artifacts(
            artifact_uri = f'models:/{iris_model_name}/latest',
            dst_path = str(staging_root / 'iris_model'),
        )
        wine_artifact_path = mlflow.artifacts.download_artifacts(
            artifact_uri = f'models:/{wine_model_name}/latest',
            dst_path = str(staging_root / 'wine_model'),
        )
        # MLflow 3.14 may return these paths with trailing slashes. Normalizing
        # through Path removes the trailing slash so MLflow preserves the final
        # directory names (`iris_model` and `wine_model`) instead of treating
        # both artifact roots as `.` and overwriting one child model.
        iris_artifact_path = str(Path(iris_artifact_path))
        wine_artifact_path = str(Path(wine_artifact_path))
        input_example, output_example = build_router_examples(
            iris_features = iris_split.x_test[0],
            wine_features = wine_split.x_test[0],
        )

        with mlflow.start_run(run_name = f'pyfunc_multi_model_router_{run_timestamp}'):
            model_info = mlflow.pyfunc.log_model(
                name = 'model',
                python_model = MultiModelRouter(),
                # These keys become `context.artifacts[...]` in the router's
                # `load_context` method.
                artifacts = {
                    'iris_model': iris_artifact_path,
                    'wine_model': wine_artifact_path,
                },
                signature = infer_signature(input_example, output_example),
                input_example = input_example,
                pip_requirements = build_model_pip_requirements(),
                # Include source code so a loaded model can find the router
                # module outside the original notebook/session context.
                code_paths = [str(repo_root / 'src')],
                registered_model_name = registered_model_name,
            )

    return _registered_model_result(model_info = model_info, metrics = {})


def _registered_model_result(
    model_info: Any,
    metrics: dict[str, float],
) -> RegisteredModelResult:
    # `registered_model_version` is only populated when MLflow registration is
    # requested. Keep the type optional so this wrapper remains honest.
    registered_version = getattr(model_info, 'registered_model_version', None)
    return RegisteredModelResult(
        model_info = model_info,
        metrics = metrics,
        model_uri = model_info.model_uri,
        registered_model_version = registered_version,
    )
