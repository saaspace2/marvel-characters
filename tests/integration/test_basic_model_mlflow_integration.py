"""Optional integration test exercising MLflow logging + Unity Catalog registration.

Skipped by default because experiment_name_basic must point to a workspace
path you have write access to (e.g. "/Users/<you>/..." or a /Shared folder
you control). Update EXPERIMENT_PATH below to a real path, then remove the
@pytest.mark.skip line to run it.
"""

import pytest

from marvel_characters.data_processor import DataProcessor
from marvel_characters.models.basic_model import BasicModel

EXPERIMENT_PATH = "/Shared/marvel-characters-integration-tests"  # <-- update this to a path you can write to


@pytest.mark.skip(reason="Set EXPERIMENT_PATH to a workspace path you can write to, then remove this skip marker.")
def test_log_model_and_register_against_real_mlflow(spark, raw_marvel_df, integration_config, integration_tags):
    integration_config.experiment_name_basic = EXPERIMENT_PATH

    dp = DataProcessor(raw_marvel_df.copy(), integration_config, spark)
    dp.preprocess()
    train_set, test_set = dp.split_data(test_size=0.3, random_state=42)
    dp.save_to_catalog(train_set, test_set)

    model = BasicModel(integration_config, integration_tags, spark)
    model.load_data()
    model.prepare_features()
    model.train()
    model.log_model()

    assert model.run_id is not None
    assert model.metrics is not None

    version = model.register_model()
    assert version is not None
