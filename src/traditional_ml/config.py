"""Local MLflow configuration for the traditional ML tutorial.

The notebook teaches MLflow in a purely local setting, so this module keeps all
tracking metadata and model artifacts in the repository root. In production,
these values would normally point to a real database and a durable object store
such as S3, but local files are easier to reason about for a tutorial.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import mlflow
import mlflow.sklearn

# Keep one shared experiment name between the notebook and hardened helpers so
# runs, registered models, and server examples all land in the same local demo.
DEFAULT_EXPERIMENT_NAME = 'Local Demo'


@dataclass(frozen=True)
class MlflowLocalConfig:
    """Resolved local MLflow paths and URIs for this repository.

    MLflow needs two separate storage concepts:
    - a backend store for metadata such as experiments, runs, and model versions;
    - an artifact root for larger files such as model packages.

    For this tutorial, the backend store is `mlflow.db` and the artifact root is
    `mlruns/`, both at the repo root so they are not accidentally created under
    `notebooks/` based on where Jupyter happens to run from.
    """

    repo_root: Path
    backend_store_path: Path
    artifact_root: Path

    @classmethod
    def from_repo_root(cls, repo_root: Path | str | None = None) -> 'MlflowLocalConfig':
        # Allow callers to pass an explicit root in tests, but default to finding
        # the current repo root from the process working directory.
        resolved_root = find_repo_root(repo_root)
        return cls(
            repo_root = resolved_root,
            backend_store_path = resolved_root / 'mlflow.db',
            artifact_root = resolved_root / 'mlruns',
        )

    @property
    def backend_store_uri(self) -> str:
        # MLflow expects SQLite locations in URI form, not as raw filesystem paths.
        return f'sqlite:///{self.backend_store_path}'

    @property
    def artifact_root_uri(self) -> str:
        # Experiment artifact locations are stored as URIs. `as_uri()` gives a
        # portable `file://...` URI while preserving local-only behavior.
        return self.artifact_root.resolve().as_uri()

    @property
    def tracking_uri(self) -> str:
        # For a local file-backed run, the tracking URI and backend store URI are
        # the same SQLite database.
        return self.backend_store_uri

    def ensure_local_paths(self) -> None:
        # SQLite will create `mlflow.db` on demand, but the artifact directory has
        # to exist before MLflow can use it as a default artifact root.
        self.artifact_root.mkdir(parents = True, exist_ok = True)


def find_repo_root(start: Path | str | None = None) -> Path:
    """Find the repository root from a working directory or file path.

    Notebooks often run with `Path.cwd()` set to either the repo root or the
    `notebooks/` directory depending on how Jupyter was launched. Walking upward
    from the starting path makes the helper resilient to either case.
    """

    current = Path(start or Path.cwd()).resolve()
    if current.is_file():
        current = current.parent

    # The pair of `pyproject.toml` and `notebooks/` is a lightweight repo-root
    # marker that is stable for this tutorial project.
    for candidate in [current, *current.parents]:
        if (candidate / 'pyproject.toml').is_file() and (candidate / 'notebooks').is_dir():
            return candidate

    raise FileNotFoundError(
        f'Could not find repo root from {current}. Expected pyproject.toml and notebooks/.'
    )


def configure_mlflow(
    config: MlflowLocalConfig | None = None,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    enable_autolog: bool = True,
) -> MlflowLocalConfig:
    """Configure MLflow to use the repo-root local backend and artifact store.

    This does the same conceptual work as the setup cells in the notebook:
    choose the local backend, create/select the experiment, and optionally turn
    on sklearn autologging so each model fit is captured by MLflow.
    """

    resolved_config = config or MlflowLocalConfig.from_repo_root()
    resolved_config.ensure_local_paths()

    # This environment variable reduces noisy model-logging output in local
    # demos. It does not change model behavior.
    os.environ['MLFLOW_RECORD_ENV_VARS_IN_MODEL_LOGGING'] = 'false'
    mlflow.set_tracking_uri(resolved_config.tracking_uri)

    # Creating the experiment explicitly lets us attach the repo-root artifact
    # location. If we only called `set_experiment`, MLflow could create the
    # experiment with whatever default artifact root happens to be active.
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(
            experiment_name,
            artifact_location = resolved_config.artifact_root_uri,
        )

    mlflow.set_experiment(experiment_name)

    if enable_autolog:
        # Autologging is useful in the tuning section because each sklearn fit
        # automatically records parameters and metrics. We disable model logging
        # here so only the final explicitly logged model is registered.
        mlflow.sklearn.autolog(log_models = False, serialization_format = 'skops')

    return resolved_config
