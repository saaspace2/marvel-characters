"""Reproducibility test for model training.

Trains the pipeline twice on identical data with the same random_state and
confirms identical predictions both times. Self-contained -- doesn't need
real Databricks tables, since it's testing the deterministic property of
the pipeline construction itself, not real-world performance.
"""

from types import SimpleNamespace

import pandas as pd

from marvel_characters.config import Tags
from marvel_characters.models.basic_model import BasicModel


def _build_and_train(X, y):
    config = SimpleNamespace(
        num_features=["Height", "Weight"],
        cat_features=["Gender"],
        target="Alive",
        parameters={"n_estimators": 10, "max_depth": 3, "random_state": 42, "verbose": -1},
        catalog_name="test_catalog",
        schema_name="test_schema",
        experiment_name_basic="/exp/repro-test",
    )
    tags = Tags(git_sha="repro-test", branch="repro-test")
    model = BasicModel(config, tags, spark=None)
    model.prepare_features()
    model.pipeline.fit(X, y)
    return model.pipeline


def test_training_is_reproducible_with_fixed_seed():
    X = pd.DataFrame(
        {
            "Height": [1.7, 1.8, 1.6, 1.9, 1.75, 1.65, 1.85, 1.7],
            "Weight": [70.0, 80.0, 60.0, 90.0, 72.0, 58.0, 88.0, 69.0],
            "Gender": ["Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female"],
        }
    )
    y = pd.Series([1, 0, 1, 0, 1, 0, 1, 1])

    pipeline_1 = _build_and_train(X, y)
    pipeline_2 = _build_and_train(X, y)

    preds_1 = pipeline_1.predict(X)
    preds_2 = pipeline_2.predict(X)

    assert list(preds_1) == list(preds_2), "Training twice with the same seed produced different predictions"
