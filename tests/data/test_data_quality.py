"""Data Quality Tests for the Marvel Characters train/test tables.

These check the DATA ITSELF (the actual rows sitting in your real dev
tables), not your code's logic. Think of this as a produce inspector
checking incoming vegetables before they reach the kitchen -- separate from
checking whether the chef's knife skills (your code) are good.

All tests here are read-only: nothing is written or modified.
"""

EXPECTED_GENDER_VALUES = {"Male", "Female", "Other"}
EXPECTED_ORIGIN_VALUES = {
    "Human", "Mutant", "Asgardian", "Alien", "Symbiote", "Robot", "Cosmic Being", "Other",
}
EXPECTED_MARITAL_STATUS_VALUES = {"Single", "Married", "Widowed", "Engaged", "Unknown"}
EXPECTED_IDENTITY_VALUES = {"Public", "Secret", "Unknown"}

REQUIRED_COLUMNS = [
    "Height", "Weight", "Universe", "Identity", "Gender",
    "Marital_Status", "Teams", "Origin", "Magic", "Mutant", "Alive",
]


# ---------------------------------------------------------------------------
# Structural checks: are the right columns there, with the right shape?
# ---------------------------------------------------------------------------

def test_required_columns_exist(train_data):
    missing = [c for c in REQUIRED_COLUMNS if c not in train_data.columns]
    assert len(missing) == 0, f"Missing columns: {missing}"


def test_required_columns_exist_in_test_set(test_data):
    missing = [c for c in REQUIRED_COLUMNS if c not in test_data.columns]
    assert len(missing) == 0, f"Missing columns: {missing}"


def test_no_duplicate_ids(train_data):
    assert train_data["Id"].nunique() == len(train_data)


def test_no_fully_duplicate_rows(train_data):
    """Catches accidental double-writes -- the exact same row appearing twice."""
    assert train_data.duplicated().sum() == 0


def test_no_overlap_between_train_and_test(train_data, test_data):
    """The same character should never appear in both the train and test sets.

    If it did, the model would effectively be 'tested' on data it already saw
    during training, making the evaluation metrics misleadingly optimistic.
    """
    overlap = set(train_data["Id"]) & set(test_data["Id"])
    assert len(overlap) == 0, f"{len(overlap)} character(s) leaked between train and test"


# ---------------------------------------------------------------------------
# Missing-value checks
# ---------------------------------------------------------------------------

def test_no_nulls_in_numeric_features(train_data):
    for col in ["Height", "Weight"]:
        assert train_data[col].isnull().sum() == 0, f"{col} has null values"


def test_no_nulls_in_target(train_data):
    """A null target would silently break model training -- catch it here, not there."""
    assert train_data["Alive"].isnull().sum() == 0


# ---------------------------------------------------------------------------
# Value range / sanity checks
# ---------------------------------------------------------------------------

def test_height_within_reasonable_range(train_data):
    """Generous bounds since this dataset includes legitimately gigantic
    cosmic-scale characters (Galactus-tier beings, Titans, frost giants --
    some canonically hundreds of meters tall, e.g. 'Great_One' at 804.67m).
    This is a guardrail against data-entry/unit-conversion errors, not a
    claim about realistic biology."""
    assert (train_data["Height"] > 0).all(), "Found non-positive height"
    assert (train_data["Height"] < 1000).all(), "Found a suspiciously huge height (data error?)"


def test_weight_within_reasonable_range(train_data):
    assert (train_data["Weight"] > 0).all(), "Found non-positive weight"
    assert (train_data["Weight"] < 2000).all(), "Found a suspiciously huge weight (data error?)"


def test_target_is_binary(train_data):
    unique_vals = set(train_data["Alive"].dropna().unique())
    assert unique_vals.issubset({0, 1})


def test_teams_magic_mutant_are_binary_flags(train_data):
    for col in ["Teams", "Magic", "Mutant"]:
        unique_vals = set(train_data[col].dropna().astype(int).unique())
        assert unique_vals.issubset({0, 1}), f"{col} contains non-binary values: {unique_vals}"


# ---------------------------------------------------------------------------
# Categorical value checks -- catches new/unexpected categories slipping
# through preprocessing (e.g. a typo, or a new value type added upstream)
# ---------------------------------------------------------------------------

def test_gender_values_within_expected_categories(train_data):
    unexpected = set(train_data["Gender"].unique()) - EXPECTED_GENDER_VALUES
    assert not unexpected, f"Unexpected Gender values: {unexpected}"


def test_origin_values_within_expected_categories(train_data):
    unexpected = set(train_data["Origin"].unique()) - EXPECTED_ORIGIN_VALUES
    assert not unexpected, f"Unexpected Origin values: {unexpected}"


def test_marital_status_values_within_expected_categories(train_data):
    unexpected = set(train_data["Marital_Status"].unique()) - EXPECTED_MARITAL_STATUS_VALUES
    assert not unexpected, f"Unexpected Marital_Status values: {unexpected}"


def test_identity_values_within_expected_categories(train_data):
    unexpected = set(train_data["Identity"].unique()) - EXPECTED_IDENTITY_VALUES
    assert not unexpected, f"Unexpected Identity values: {unexpected}"


# ---------------------------------------------------------------------------
# Class balance -- not a hard correctness check, but a useful early warning
# ---------------------------------------------------------------------------

def test_class_balance_acceptable(train_data):
    """If 99% of characters are 'Alive', the model could cheat by always
    guessing 'Alive' and still look accurate. This isn't a bug, but it's
    worth knowing about before trusting accuracy as a metric."""
    alive_pct = train_data["Alive"].mean()
    assert 0.10 <= alive_pct <= 0.90, f"Alive class ratio is {alive_pct:.2%}, dangerously imbalanced"
