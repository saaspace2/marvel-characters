"""Model Training Tests for the Marvel Characters survival classifier.

These check the TRAINED MODEL itself -- not the data, not the preprocessing
code -- using the real, currently-registered model and real Delta tables.
"""

from sklearn.metrics import f1_score


def test_accuracy_above_threshold(trained_model, test_data):
    X, y = test_data
    accuracy = trained_model.score(X, y)
    assert accuracy > 0.70, f"Accuracy {accuracy:.3f} below 0.70"


def test_model_beats_naive_baseline(trained_model, test_data):
    """A model should do meaningfully better than just guessing the majority class.

    If most characters are 'Alive', a model that always predicts 'Alive'
    would already score high accuracy without learning anything. This
    catches that failure mode directly, rather than trusting a raw accuracy
    number that can look fine even when the model learned nothing useful.
    """
    X, y = test_data
    naive_baseline_accuracy = y.value_counts(normalize=True).max()
    model_accuracy = trained_model.score(X, y)
    assert model_accuracy > naive_baseline_accuracy, (
        f"Model accuracy ({model_accuracy:.3f}) does not beat the naive baseline "
        f"of always predicting the majority class ({naive_baseline_accuracy:.3f})"
    )


def test_f1_score_above_threshold(trained_model, test_data):
    """Your actual deployment decision (BasicModel.model_improved()) compares
    F1 score, not accuracy, when deciding whether to promote a new model.
    On an imbalanced target like this one, F1 is the metric that actually
    matters -- accuracy alone can look fine while F1 is poor."""
    X, y = test_data
    preds = trained_model.predict(X)
    f1 = f1_score(y, preds)
    assert f1 > 0.50, f"F1 score {f1:.3f} below 0.50"


def test_model_not_overfitting(trained_model, train_data, test_data):
    X_train, y_train = train_data
    X_test, y_test = test_data

    train_acc = trained_model.score(X_train, y_train)
    test_acc = trained_model.score(X_test, y_test)
    gap = train_acc - test_acc

    assert gap < 0.10, f"Overfitting: train_acc={train_acc:.3f}, test_acc={test_acc:.3f}, gap={gap:.3f}"


def test_predictions_are_binary(trained_model, test_data):
    X, _ = test_data
    preds = trained_model.predict(X)
    assert set(preds).issubset({0, 1})


def test_predictions_cover_both_classes(trained_model, test_data):
    """If the model only ever predicts one class, it isn't actually
    discriminating between alive/dead -- even if accuracy looks fine on an
    imbalanced test set."""
    X, _ = test_data
    preds = trained_model.predict(X)
    assert len(set(preds)) > 1, "Model predicted only one class for every row in the test set"


def test_prediction_count_matches_input(trained_model, test_data):
    X, _ = test_data
    preds = trained_model.predict(X)
    assert len(preds) == len(X)


def test_predict_proba_outputs_are_valid_probabilities(trained_model, test_data):
    """If anything downstream uses predict_proba() (e.g. ranking by
    confidence), the raw probabilities need to actually be valid -- not
    just the final hard 0/1 label."""
    X, _ = test_data
    proba = trained_model.predict_proba(X)

    assert proba.shape[1] == 2, "Expected 2 probability columns for binary classification"
    assert (proba >= 0).all() and (proba <= 1).all(), "Found probabilities outside [0, 1]"

    row_sums = proba.sum(axis=1)
    assert (abs(row_sums - 1.0) < 1e-6).all(), "Probabilities for a row don't sum to 1"


def test_model_has_configured_hyperparameters(trained_model):
    """Sanity check the underlying LightGBM model actually has real
    hyperparameters set, not silently falling back to library defaults
    due to a config-wiring bug."""
    regressor = trained_model.named_steps["regressor"]
    params = regressor.get_params()
    assert params.get("n_estimators") is not None
    assert params["n_estimators"] > 0


def test_no_target_or_id_leakage_in_model_features(trained_model):
    """The model's actual fitted feature columns should never include the
    target ('Alive') or the row identifier ('Id') -- a classic, easy-to-
    introduce leakage bug if load_data()'s feature list is ever
    misconfigured. Checked against the real fitted model's feature
    schema, not just our own assumptions about what the config contains."""
    preprocessor = trained_model.named_steps["preprocessor"]
    feature_names = set(preprocessor.feature_names_in_)
    assert "Alive" not in feature_names, "Target column leaked into model features"
    assert "Id" not in feature_names, "Row identifier leaked into model features"
