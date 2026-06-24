import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest

from traditional_ml.config import MlflowLocalConfig, configure_mlflow, find_repo_root
from traditional_ml.data import load_iris_split, load_wine_split
from traditional_ml.models import build_iris_pipeline, calculate_classification_metrics
from traditional_ml.router import MultiModelRouter
from traditional_ml.serving import (
    find_free_port,
    invoke_model_endpoint,
    start_model_server,
    start_tracking_server,
)
from traditional_ml.training import (
    register_multi_model_router,
    train_register_iris_model,
    train_register_wine_model,
)


class StubModel:
    def __init__(self, prediction: float) -> None:
        self.prediction = prediction

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.asarray([self.prediction + features.shape[1] * 0])


def test_find_repo_root_from_nested_path(tmp_path: Path) -> None:
    (tmp_path / 'pyproject.toml').write_text('[project]\nname = "demo"\n')
    notebook_dir = tmp_path / 'notebooks'
    notebook_dir.mkdir()

    assert find_repo_root(notebook_dir) == tmp_path


def test_local_config_uses_repo_root_backend_paths(tmp_path: Path) -> None:
    (tmp_path / 'pyproject.toml').write_text('[project]\nname = "demo"\n')
    (tmp_path / 'notebooks').mkdir()

    config = MlflowLocalConfig.from_repo_root(tmp_path)

    assert config.backend_store_path == tmp_path / 'mlflow.db'
    assert config.artifact_root == tmp_path / 'mlruns'
    assert config.backend_store_uri.startswith('sqlite:///')
    assert 'notebooks' not in config.backend_store_uri


def test_dataset_splits_are_deterministic() -> None:
    first_iris_split = load_iris_split()
    second_iris_split = load_iris_split()
    wine_split = load_wine_split()

    assert first_iris_split.x_train.shape == (120, 4)
    assert first_iris_split.x_test.shape == (30, 4)
    assert wine_split.x_train.shape == (142, 13)
    assert wine_split.x_test.shape == (36, 13)
    np.testing.assert_array_equal(first_iris_split.x_test, second_iris_split.x_test)
    np.testing.assert_array_equal(first_iris_split.y_test, second_iris_split.y_test)


def test_iris_model_builder_and_metrics() -> None:
    split = load_iris_split()
    model = build_iris_pipeline()

    model.fit(split.x_train, split.y_train)
    predictions = model.predict(split.x_test)
    metrics = calculate_classification_metrics(split.y_test, predictions)

    assert set(metrics) == {
        'accuracy',
        'precision_macro',
        'recall_macro',
        'f1_macro',
    }
    assert all(0.0 <= value <= 1.0 for value in metrics.values())


def test_router_predicts_for_supported_models() -> None:
    router = MultiModelRouter()
    router.models = {
        'iris': StubModel(1.0),
        'wine': StubModel(2.0),
    }

    result = router.predict(
        None,
        pd.DataFrame(
            [
                {'model_name': 'iris', 'features_json': json.dumps([1, 2, 3, 4])},
                {'model_name': 'wine', 'features_json': [1, 2, 3]},
            ]
        ),
    )

    assert result.to_dict(orient = 'records') == [
        {'model_name': 'iris', 'prediction': 1.0},
        {'model_name': 'wine', 'prediction': 2.0},
    ]


def test_router_rejects_bad_input() -> None:
    router = MultiModelRouter()
    router.models = {'iris': StubModel(1.0), 'wine': StubModel(2.0)}

    with pytest.raises(ValueError, match = 'Missing required columns'):
        router.predict(None, pd.DataFrame([{'model_name': 'iris'}]))

    with pytest.raises(ValueError, match = 'Unsupported model_name'):
        router.predict(
            None,
            pd.DataFrame([{'model_name': 'digits', 'features_json': '[1, 2]'}]),
        )

    with pytest.raises(ValueError, match = 'valid JSON'):
        router.predict(
            None,
            pd.DataFrame([{'model_name': 'iris', 'features_json': 'not-json'}]),
        )


def test_notebook_is_self_contained_and_root_backed() -> None:
    notebook_path = Path('notebooks/traditional_ml.ipynb')
    notebook = json.loads(notebook_path.read_text(encoding = 'utf-8'))
    notebook_text = notebook_path.read_text(encoding = 'utf-8')

    assert '/Users/' not in notebook_text
    assert 'notebooks/mlflow.db' not in notebook_text
    assert 'Iris Demo' not in notebook_text
    assert "MLFLOW_BACKEND_STORE_PATH = REPO_ROOT / 'mlflow.db'" in notebook_text
    assert "MLFLOW_ARTIFACT_ROOT = REPO_ROOT / 'mlruns'" in notebook_text
    assert 'def build_model_pipeline' in notebook_text
    assert 'def objective' in notebook_text
    assert 'class MultiModelRouter' in notebook_text
    assert 'def start_tracking_server' in notebook_text
    assert 'def start_model_server' in notebook_text
    assert 'dst_path = str(artifact_staging_root /' in notebook_text
    assert 'iris_artifact_path = str(Path(iris_artifact_path))' in notebook_text

    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            assert cell['execution_count'] is None
            assert cell['outputs'] == []


def test_local_tracking_and_model_server_smoke(tmp_path: Path) -> None:
    config = MlflowLocalConfig(
        repo_root = tmp_path,
        backend_store_path = tmp_path / 'mlflow.db',
        artifact_root = tmp_path / 'mlruns',
    )
    configure_mlflow(
        config,
        experiment_name = 'Smoke Test',
        enable_autolog = False,
    )
    split = load_iris_split()
    result = train_register_iris_model(
        split,
        best_params = {'C': 1.0, 'solver': 'lbfgs', 'max_iter': 400},
        run_timestamp = 'smoke',
        registered_model_name = 'smoke_iris_model',
    )

    tracking_server = None
    model_server = None
    try:
        tracking_server = start_tracking_server(
            config,
            port = find_free_port(),
            name = 'smoke tracking server',
            pid_file = tmp_path / 'tracking.pid',
            log_file = tmp_path / 'tracking.log',
        )
        model_server = start_model_server(
            f'models:/smoke_iris_model/{result.registered_model_version}',
            tracking_uri = tracking_server.url,
            port = find_free_port(),
            name = 'smoke model server',
            pid_file = tmp_path / 'model.pid',
            log_file = tmp_path / 'model.log',
        )

        response = invoke_model_endpoint(
            port = model_server.port,
            payload = {'inputs': split.x_test[:2].tolist()},
            timeout_seconds = 30,
        )
    finally:
        if model_server is not None:
            model_server.stop()
        if tracking_server is not None:
            tracking_server.stop()

    assert 'predictions' in response
    assert len(response['predictions']) == 2
    assert mlflow.get_tracking_uri() == config.tracking_uri


def test_multi_model_router_registration_preserves_child_models(tmp_path: Path) -> None:
    config = MlflowLocalConfig(
        repo_root = tmp_path,
        backend_store_path = tmp_path / 'mlflow.db',
        artifact_root = tmp_path / 'mlruns',
    )
    configure_mlflow(
        config,
        experiment_name = 'Router Smoke Test',
        enable_autolog = False,
    )
    iris_split = load_iris_split()
    wine_split = load_wine_split()
    train_register_iris_model(
        iris_split,
        best_params = {'C': 1.0, 'solver': 'lbfgs', 'max_iter': 400},
        run_timestamp = 'router_smoke',
        registered_model_name = 'router_smoke_iris_model',
    )
    train_register_wine_model(
        wine_split,
        run_timestamp = 'router_smoke',
        registered_model_name = 'router_smoke_wine_model',
    )
    router_result = register_multi_model_router(
        iris_split = iris_split,
        wine_split = wine_split,
        run_timestamp = 'router_smoke',
        repo_root = Path.cwd(),
        iris_model_name = 'router_smoke_iris_model',
        wine_model_name = 'router_smoke_wine_model',
        registered_model_name = 'router_smoke_multi_model_router',
    )

    router_model = mlflow.pyfunc.load_model(
        f'models:/router_smoke_multi_model_router/{router_result.registered_model_version}'
    )
    python_model = router_model._model_impl.python_model
    assert python_model.models['iris'].n_features_in_ == 4
    assert python_model.models['wine'].n_features_in_ == 13

    predictions = router_model.predict(
        pd.DataFrame(
            [
                {
                    'model_name': 'iris',
                    'features_json': json.dumps(iris_split.x_test[0].tolist()),
                },
                {
                    'model_name': 'wine',
                    'features_json': json.dumps(wine_split.x_test[0].tolist()),
                },
            ]
        )
    )

    assert predictions['model_name'].tolist() == ['iris', 'wine']
    assert len(predictions['prediction'].tolist()) == 2
