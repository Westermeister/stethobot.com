# Copyright (c) 2021 The Stethobot Authors. All rights reserved.

"""Tests for the diagnostic model."""

from sklearn import ensemble

from stethobot.api.model import DiagnosticModel


# The following code will produce a reference scikit-learn model to compare our model to.
# Both models are trained on a modified version of UCI's famous Iris dataset, with the third class "virginica" missing.
# In addition, the "setosa" and "versicolor" classes have been renamed "positive" and "negative" respectively.
# These changes make the dataset compatible with our model's API.

# Preprocess the dataset.
x = []
y = []
with open("./tests/iris.csv") as f:
    lines = f.readlines()
    for line in lines[1:]:
        line = line.rstrip()
        split = line.split(",")
        x.append([float(i) for i in split[1:-1]])
        y.append(split[-1])

# Fit the reference model.
reference = ensemble.RandomForestClassifier(
    n_estimators=100,
    criterion="gini",
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features="sqrt",
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    min_impurity_split=None,
    bootstrap=True,
    oob_score=True,
    n_jobs=-1,
    random_state=None,
    verbose=0,
    warm_start=False,
    class_weight=None,
    ccp_alpha=0.0,
    max_samples=None,
)
reference.fit(x, y)

# Fit our custom model with the same parameters (most are built-in to it).
custom = DiagnosticModel(num_trees=100, num_split=-1)
custom.fit(x, y)


def test_smoke():
    """Simple smoke test before we do any comparisons to the reference model.

    Check that various types are handled correctly and that the output is formatted correctly."""
    # First column will be integers, second floats, third a mix, and fourth strings.
    # A few NoneTypes are also thrown in to be imputed.
    # We also duplicate the final row.
    sample = [
        [0, 0.1, 0, "0"],
        [1, 1.3, 1.3, None],
        [2, 2.6, None, "0"],
        [None, 3.9, 3.9, "0"],
        [4, 4.1, 4, "0"],
        [5, None, 5.8, "1"],
        [6, 6.6, 6, "1"],
        [None, 7.2, 7.2, None],
        [9, 9.5, 9.5, "1"],
        [9, 9.5, 9.5, "1"],
    ]
    diagnoses = [
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "positive",
        "positive",
        "positive",
        "positive",
        "positive",
    ]
    model = DiagnosticModel()
    # This shouldn't result in any errors.
    results = model.fit(sample, diagnoses)
    # Each result is an accuracy statistic between 0 and 1.
    for result in results:
        assert result >= 0 and result <= 1


def test_out_of_bag_accuracy():
    """Compare out-of-bag accuracy of reference and template. They shouldn't be too far off."""
    numerator = abs(reference.oob_score_ - custom._oob_accuracy)
    denominator = (reference.oob_score_ + custom._oob_accuracy) / 2
    percent_diff = (numerator / denominator) * 100
    assert percent_diff < 10


def test_predictions():
    """Compare predictions of reference and custom models. They should be the same."""
    # These are taken straight from the training sample. The first two are positive; the last two are negative.
    observations = [
        [5.1, 3.5, 1.4, 0.2],
        [4.9, 3.0, 1.4, 0.2],
        [5.1, 2.5, 3.0, 1.1],
        [5.7, 2.8, 4.1, 1.3],
    ]
    reference_predictions = []
    custom_predictions = []
    for obs in observations:
        # Predict with reference model.
        reference_prediction = reference.predict([obs])[0]
        reference_predictions.append(reference_prediction)
        # Predict with custom model.
        custom_prediction = custom.diagnose(obs)
        label = max(custom_prediction, key=lambda key: custom_prediction[key])
        custom_predictions.append(label)
    for reference_prediction, custom_prediction in zip(
        reference_predictions, custom_predictions
    ):
        assert reference_prediction == custom_prediction
