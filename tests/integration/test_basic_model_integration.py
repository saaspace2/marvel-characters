"""Integration test for BasicModel against real Delta tables on the serverless cluster.

Covers load_data() and train() with real data flowing through the sklearn
pipeline. log_model()/register_model() are intentionally excluded here since
they need MLflow experiment/registry permissions that vary by workspace --
see test_basic_model_mlflow_integration.py (opt-in, skipped by default).
"""

from marvel_characters.data_processor import DataProcessor
from marvel_characters.models.basic_model import BasicModel


def test_load_data_and_train_against_real_tables(spark, raw_marvel_df, integration_config, integration_tags):
    # Arrange: get real train/test tables into the isolated schema first
    dp = DataProcessor(raw_marvel_df.copy(), integration_config, spark)
    dp.preprocess()
    train_set, test_set = dp.split_data(test_size=0.3, random_state=42)
    dp.save_to_catalog(train_set, test_set)

    # Act
    model = BasicModel(integration_config, integration_tags, spark)
    model.load_data()
    model.prepare_features()
    model.train()

    # Assert: pipeline actually fit and can predict on real held-out data
    predictions = model.pipeline.predict(model.X_test)
    assert len(predictions) == len(model.X_test)
    assert model.train_data_version is not None
    assert model.test_data_version is not None
