"""Training time bound.

A sudden, large increase in training time could indicate an accidental
config change (e.g. n_estimators exploding) or a performance regression
elsewhere in the pipeline. This actually runs load_data() + prepare_features()
+ train() against real Delta tables on the serverless cluster -- it's
slower than the other (read-only-on-already-trained-model) tests in this
folder, since it performs real training, but it's still read-only with
respect to your tables (train() only fits in memory, never writes).
"""

import time

from marvel_characters.models.basic_model import BasicModel

# Deliberately generous guess -- if this fails, check whether it's a real
# regression (e.g. n_estimators accidentally set very high) before just
# raising the number.
MAX_TRAINING_SECONDS = 120


def test_training_completes_within_time_bound(spark, basic_model_config, basic_model_tags):
    model = BasicModel(basic_model_config, basic_model_tags, spark)

    start = time.time()
    model.load_data()
    model.prepare_features()
    model.train()
    elapsed = time.time() - start

    assert elapsed < MAX_TRAINING_SECONDS, (
        f"Training took {elapsed:.1f}s, exceeding the {MAX_TRAINING_SECONDS}s bound"
    )
