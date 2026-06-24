"""Public helpers used by the traditional ML MLflow tutorial notebook.

The notebook intentionally shows teaching code inline. This package provides the
same concepts as importable, testable Python helpers so behavior can be hardened
without making the notebook less self-contained.
"""

# Re-export the most common helpers so notebook or test code can import from
# `traditional_ml` directly instead of reaching into each implementation module.
from traditional_ml.config import (
    DEFAULT_EXPERIMENT_NAME,
    MlflowLocalConfig,
    configure_mlflow,
    find_repo_root,
)
from traditional_ml.data import DatasetSplit, load_iris_split, load_wine_split
from traditional_ml.models import (
    build_iris_pipeline,
    build_iris_pipeline_from_trial,
    build_model_conda_env,
    build_model_pip_requirements,
    build_wine_model,
    calculate_classification_metrics,
)
from traditional_ml.router import MultiModelRouter, build_router_examples
from traditional_ml.training import (
    latest_registered_model_version,
    load_latest_sklearn_model,
    register_multi_model_router,
    run_iris_hyperparameter_search,
    train_register_iris_model,
    train_register_wine_model,
    utc_run_timestamp,
)

__all__ = [
    'DEFAULT_EXPERIMENT_NAME',
    'DatasetSplit',
    'MlflowLocalConfig',
    'MultiModelRouter',
    'build_iris_pipeline',
    'build_iris_pipeline_from_trial',
    'build_model_conda_env',
    'build_model_pip_requirements',
    'build_router_examples',
    'build_wine_model',
    'calculate_classification_metrics',
    'configure_mlflow',
    'find_repo_root',
    'latest_registered_model_version',
    'load_iris_split',
    'load_latest_sklearn_model',
    'load_wine_split',
    'register_multi_model_router',
    'run_iris_hyperparameter_search',
    'train_register_iris_model',
    'train_register_wine_model',
    'utc_run_timestamp',
]
