"""Unit tests for marvel_characters.monitoring.

monitoring.py is almost entirely Spark/Databricks I/O, so a 'unit test' here
means mocking Spark and the workspace client entirely and checking control
flow / branching logic only — not real data processing. These tests cover
the early-return and exception-branch behavior; the actual Delta/Spark
transformations need a real cluster and belong in integration tests.
"""

from unittest.mock import MagicMock

from databricks.sdk.errors import NotFound

from marvel_characters.monitoring import create_or_refresh_monitoring


def _fake_config():
    config = MagicMock()
    config.catalog_name = "test_catalog"
    config.schema_name = "test_schema"
    return config


def test_returns_early_and_does_not_query_monitors_when_no_inference_records():
    config = _fake_config()
    spark = MagicMock()
    spark.sql.return_value.count.return_value = 0
    workspace = MagicMock()

    create_or_refresh_monitoring(config, spark, workspace)

    workspace.quality_monitors.get.assert_not_called()
    workspace.quality_monitors.run_refresh.assert_not_called()


def test_refreshes_existing_monitor_when_found(monkeypatch):
    config = _fake_config()
    spark = MagicMock()

    # First .count() call (raw inference table) must be > 0 to proceed past the early return.
    inf_table = MagicMock()
    inf_table.count.return_value = 5
    spark.sql.return_value = inf_table

    # Chain of withColumn/select/dropna/withColumn calls all return MagicMocks that
    # support further chaining, ending with a count() > 0 so the "valid prediction" branch is taken.
    chained = inf_table.withColumn.return_value
    chained.withColumn.return_value = chained
    chained.select.return_value = chained
    chained.dropna.return_value = chained
    chained.count.return_value = 3
    chained.write.format.return_value.mode.return_value.saveAsTable.return_value = None

    spark.table.return_value.count.return_value = 3

    workspace = MagicMock()  # quality_monitors.get succeeds -> no NotFound -> refresh path

    create_or_refresh_monitoring(config, spark, workspace)

    workspace.quality_monitors.run_refresh.assert_called_once()


def test_creates_monitor_when_not_found(monkeypatch):
    config = _fake_config()
    spark = MagicMock()

    inf_table = MagicMock()
    inf_table.count.return_value = 5
    spark.sql.return_value = inf_table

    chained = inf_table.withColumn.return_value
    chained.withColumn.return_value = chained
    chained.select.return_value = chained
    chained.dropna.return_value = chained
    chained.count.return_value = 3
    chained.write.format.return_value.mode.return_value.saveAsTable.return_value = None

    spark.table.return_value.count.return_value = 3

    workspace = MagicMock()
    workspace.quality_monitors.get.side_effect = NotFound("not found")

    import marvel_characters.monitoring as monitoring_module

    create_table_mock = MagicMock()
    monkeypatch.setattr(monitoring_module, "create_monitoring_table", create_table_mock)

    create_or_refresh_monitoring(config, spark, workspace)

    create_table_mock.assert_called_once()
