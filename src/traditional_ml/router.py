"""PyFunc router for serving multiple local tutorial models.

MLflow pyfunc models provide a generic Python interface around arbitrary
inference logic. Here we use that interface to put two independent sklearn
models behind one HTTP endpoint and route each row based on `model_name`.
"""

from __future__ import annotations

import json
from typing import Any

import mlflow
import numpy as np
import pandas as pd

# Every routed request needs a model selector and a serialized feature vector.
REQUIRED_ROUTER_COLUMNS = {'model_name', 'features_json'}

# Keep the valid model names centralized so validation and error messages agree.
SUPPORTED_MODEL_NAMES = ('iris', 'wine')


class MultiModelRouter(mlflow.pyfunc.PythonModel):
    """Route rows to one of several bundled sklearn models.

    MLflow calls `load_context` when the model package is loaded and `predict`
    for each inference request. The child model artifact paths come from the
    `artifacts={...}` mapping passed to `mlflow.pyfunc.log_model`.
    """

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        # Load with the sklearn flavor so each child model returns the same kind
        # of object it had during training. This avoids nesting pyfunc wrappers
        # inside another pyfunc wrapper.
        self.models = {
            'iris': mlflow.sklearn.load_model(context.artifacts['iris_model']),
            'wine': mlflow.sklearn.load_model(context.artifacts['wine_model']),
        }

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        # `context` and `params` are part of the MLflow pyfunc signature, but the
        # tutorial router does not need them. Deleting them makes that explicit
        # and avoids unused-variable lint warnings.
        del context, params

        missing = REQUIRED_ROUTER_COLUMNS.difference(model_input.columns)
        if missing:
            raise ValueError(f'Missing required columns: {sorted(missing)}')

        predictions = []
        for _, row in model_input.iterrows():
            # Validate the selector before parsing features so bad model names
            # fail with a direct, teaching-friendly error.
            selected_model = _normalize_model_name(row['model_name'])
            features = _parse_features(row['features_json'])
            routed_prediction = self.models[selected_model].predict(features)[0]
            predictions.append(
                {
                    'model_name': selected_model,
                    'prediction': float(routed_prediction),
                }
            )

        return pd.DataFrame(predictions)


def build_router_examples(
    iris_features: NDArrayLike,
    wine_features: NDArrayLike,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build representative router input and output examples.

    MLflow uses these examples for signature inference and for the model UI. The
    feature vectors are JSON strings because one table column has to hold vectors
    with different lengths: 4 Iris features or 13 Wine features.
    """

    input_example = pd.DataFrame(
        [
            {'model_name': 'iris', 'features_json': json.dumps(list(iris_features))},
            {'model_name': 'wine', 'features_json': json.dumps(list(wine_features))},
        ]
    )
    output_example = pd.DataFrame(
        [
            {'model_name': 'iris', 'prediction': 0.0},
            {'model_name': 'wine', 'prediction': 0.0},
        ]
    )
    return input_example, output_example


type NDArrayLike = list[float] | np.ndarray


def _normalize_model_name(value: Any) -> str:
    # Human-entered payloads often vary in casing and whitespace. Normalizing
    # lets values such as " Iris " still route to the Iris model.
    selected_model = str(value).strip().lower()
    if selected_model not in SUPPORTED_MODEL_NAMES:
        raise ValueError(
            f"Unsupported model_name '{selected_model}'. "
            f'Valid options: {list(SUPPORTED_MODEL_NAMES)}'
        )
    return selected_model


def _parse_features(value: Any) -> np.ndarray:
    # The served endpoint sends `features_json` as a string, while direct Python
    # tests may pass a raw list. Supporting both keeps the router easy to test.
    if isinstance(value, str):
        try:
            feature_values = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError('features_json must contain valid JSON.') from error
    else:
        feature_values = value

    features = np.asarray(feature_values, dtype = float)
    if features.ndim != 1:
        # Each row must represent exactly one feature vector. A nested matrix
        # would be ambiguous because routing happens row by row.
        raise ValueError('features_json must contain one numeric feature vector.')

    # sklearn estimators expect a 2D matrix of shape `(n_rows, n_features)`.
    # Since we process one routed row at a time, reshape the vector to one row.
    return features.reshape(1, -1)
