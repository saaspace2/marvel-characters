from types import SimpleNamespace
 
import pandas as pd
import pytest
 
from marvel_characters.data_processor import DataProcessor
 
 
@pytest.fixture
def fake_config():
    """Minimal stand-in for ProjectConfig — only the attrs preprocess() reads."""
    return SimpleNamespace(
        num_features=["Height", "Weight"],
        cat_features=["Universe", "Identity", "Gender", "Marital_Status", "Teams", "Origin", "Magic", "Mutant"],
        target="Alive",
    )
 
 
@pytest.fixture
def raw_df():
    """Raw input shaped like the source data, BEFORE preprocess() runs."""
    return pd.DataFrame(
        {
            "Height (m)": [1.8, 1.6, None],
            "Weight (kg)": [90.0, 55.0, 70.0],
            "Universe": ["Earth-616", None, "Earth-616"],
            "Teams": ["Avengers", None, "X-Men"],
            "Origin": ["Human mutate", "Asgardian god", "Human"],
            "Identity": ["Public", "Secret", None],
            "Gender": ["Male", "Female", "Unknown-Value"],
            "Marital Status": ["Single", "Widow", None],
            "Alive": ["Alive", "Dead", "Alive"],
            "PageID": [101, 102, 103],
        }
    )
 
 
def test_preprocess_renames_height_and_weight(raw_df, fake_config):
    dp = DataProcessor(raw_df.copy(), fake_config, spark=None)
    dp.preprocess()
    assert "Height" in dp.df.columns
    assert "Weight" in dp.df.columns
    assert "Height (m)" not in dp.df.columns
 
 
def test_preprocess_fills_universe_nulls(raw_df, fake_config):
    dp = DataProcessor(raw_df.copy(), fake_config, spark=None)
    dp.preprocess()
    assert dp.df["Universe"].isnull().sum() == 0
 
 
def test_preprocess_teams_becomes_binary(raw_df, fake_config):
    dp = DataProcessor(raw_df.copy(), fake_config, spark=None)
    dp.preprocess()
    assert set(dp.df["Teams"].astype(int).unique()).issubset({0, 1})
 
 
def test_preprocess_gender_collapses_unexpected_values(raw_df, fake_config):
    dp = DataProcessor(raw_df.copy(), fake_config, spark=None)
    dp.preprocess()
    assert set(dp.df["Gender"]).issubset({"Male", "Female", "Other"})
 
 
def test_preprocess_marital_status_widow_becomes_widowed(raw_df, fake_config):
    dp = DataProcessor(raw_df.copy(), fake_config, spark=None)
    dp.preprocess()
    assert "Widow" not in set(dp.df["Marital_Status"])
 
 
def test_preprocess_origin_normalizes_to_known_categories(raw_df, fake_config):
    dp = DataProcessor(raw_df.copy(), fake_config, spark=None)
    dp.preprocess()
    allowed = {"Human", "Mutant", "Asgardian", "Alien", "Symbiote", "Robot", "Cosmic Being", "Other"}
    assert set(dp.df["Origin"]).issubset(allowed)
 
 
def test_preprocess_alive_becomes_binary_target(raw_df, fake_config):
    dp = DataProcessor(raw_df.copy(), fake_config, spark=None)
    dp.preprocess()
    assert set(dp.df["Alive"].unique()).issubset({0, 1})
 
 
def test_preprocess_id_renamed_from_pageid(raw_df, fake_config):
    dp = DataProcessor(raw_df.copy(), fake_config, spark=None)
    dp.preprocess()
    assert "Id" in dp.df.columns
    assert "PageID" not in dp.df.columns
    assert dp.df["Id"].dtype == object  # cast to str
 
 
def test_preprocess_keeps_only_configured_columns(raw_df, fake_config):
    dp = DataProcessor(raw_df.copy(), fake_config, spark=None)
    dp.preprocess()
    expected = set(fake_config.num_features + fake_config.cat_features + [fake_config.target, "Id"])
    assert set(dp.df.columns) == expected
