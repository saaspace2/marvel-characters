"""Integration tests for DataProcessor against a real Databricks serverless cluster.

Unlike the unit tests (where spark=None), these exercise save_to_catalog()
and enable_change_data_feed() against real Delta tables. They always run
inside the isolated, auto-cleaned-up schema from conftest.py -- never real
dev data.
"""

from marvel_characters.data_processor import DataProcessor


def test_full_round_trip_preprocess_split_save_and_read_back(spark, raw_marvel_df, integration_config):
    dp = DataProcessor(raw_marvel_df.copy(), integration_config, spark)
    dp.preprocess()

    assert len(dp.df) > 0  # confirm preprocessing didn't drop every row

    train_set, test_set = dp.split_data(test_size=0.3, random_state=42)
    assert len(train_set) + len(test_set) == len(dp.df)

    dp.save_to_catalog(train_set, test_set)

    # Read back from the REAL Delta tables to confirm the write actually landed
    catalog = integration_config.catalog_name
    schema = integration_config.schema_name

    train_read_back = spark.table(f"{catalog}.{schema}.train_set").toPandas()
    test_read_back = spark.table(f"{catalog}.{schema}.test_set").toPandas()

    assert len(train_read_back) == len(train_set)
    assert len(test_read_back) == len(test_set)
    assert "update_timestamp_utc" in train_read_back.columns


def test_enable_change_data_feed_does_not_raise(spark, raw_marvel_df, integration_config):
    dp = DataProcessor(raw_marvel_df.copy(), integration_config, spark)
    dp.preprocess()
    train_set, test_set = dp.split_data()
    dp.save_to_catalog(train_set, test_set)

    # Should not raise -- confirms the ALTER TABLE statements are valid against a real table
    dp.enable_change_data_feed()

    catalog = integration_config.catalog_name
    schema = integration_config.schema_name
    props = spark.sql(f"SHOW TBLPROPERTIES {catalog}.{schema}.train_set").toPandas()
    cdf_enabled = props[props["key"] == "delta.enableChangeDataFeed"]["value"].iloc[0]
    assert cdf_enabled == "true"
