"""Unit tests for marvel_characters.models.custom_model."""

from unittest.mock import MagicMock

import numpy as np

from marvel_characters.models.custom_model import MarvelModelWrapper, adjust_predictions


def test_adjust_predictions_maps_one_to_alive():
    assert adjust_predictions([1]) == {"Survival prediction": ["alive"]}


def test_adjust_predictions_maps_zero_to_dead():
    assert adjust_predictions([0]) == {"Survival prediction": ["dead"]}


def test_adjust_predictions_handles_mixed_list():
    result = adjust_predictions([1, 0, 1])
    assert result == {"Survival prediction": ["alive", "dead", "alive"]}


def test_adjust_predictions_handles_numpy_array_input():
    result = adjust_predictions(np.array([1, 0]))
    assert result == {"Survival prediction": ["alive", "dead"]}


def test_adjust_predictions_empty_input_returns_empty_list():
    assert adjust_predictions([]) == {"Survival prediction": []}


def test_wrapper_predict_delegates_to_underlying_model_and_adjusts(monkeypatch):
    wrapper = MarvelModelWrapper()
    wrapper.model = MagicMock()
    wrapper.model.predict.return_value = np.array([1, 0])

    result = wrapper.predict(context=None, model_input=None)

    assert result == {"Survival prediction": ["alive", "dead"]}
    wrapper.model.predict.assert_called_once_with(None)
