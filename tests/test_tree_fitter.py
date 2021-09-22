# Copyright (c) 2021 The Stethobot Authors. All rights reserved.

"""Tests for the diagnostic model's helper class."""

import queue

from stethobot.api.model import TreeFitter

# Some arbitrary testing data. 10 positives and 5 negatives.
SAMPLE = [
    [1.2, 2, "3"],
    [4.9, 5, "6"],
    [7, 8, "9"],
    [0.3, 0, "0"],
    [0.2, 0, "1"],
    [0.7, 1, "0"],
    [0.1, 1, "1"],
    [1, 0, "0"],
    [1.0, 0, "1"],
    [1.1, 1, "0"],
    [1, 1, "1"],
    [2, 2, "2"],
    [2.4, 2, "3"],
    [2.6, 3, "2"],
    [2, 3, "3"],
]
LABELS = [
    "positive",
    "positive",
    "positive",
    "positive",
    "positive",
    "positive",
    "positive",
    "positive",
    "positive",
    "positive",
    "negative",
    "negative",
    "negative",
    "negative",
    "negative",
]


def test_setup():
    """Ensure important attributes are initialized correctly in the constructor."""
    # Call the constructor with some testing data.
    tree_fitter = TreeFitter(SAMPLE, LABELS, int(len(SAMPLE[0]) ** 0.5))

    # Test the stratified bootstrap attributes.
    effective_sample = tree_fitter._effective_sample
    effective_labels = tree_fitter._effective_labels
    assert len(effective_sample) == len(effective_labels) == 10
    negative_count = 0
    positive_count = 0
    for label in effective_labels:
        if label == "negative":
            negative_count += 1
        else:
            positive_count += 1
    assert negative_count == positive_count == 5

    # Test the label set attribute.
    assert tree_fitter._label_set == set(["positive", "negative"])


def test_run():
    """Ensure output is structured correctly."""
    tree_fitter = TreeFitter(SAMPLE, LABELS, int(len(SAMPLE[0]) ** 0.5))
    q = queue.Queue()
    tree_fitter(q)
    result = q.get()

    # Validate the tree.

    assert "tree" in result
    assert type(result["tree"]) is list
    assert len(result["tree"]) > 0
    for node in result["tree"]:
        assert type(node) is dict

        # Validate each key in the node.

        # Validate the prediction.
        assert "prediction" in node
        assert type(node["prediction"]) is dict or node["prediction"] is None
        if type(node["prediction"]) is dict:
            assert (
                node["prediction"]["positive"] >= 0
                and node["prediction"]["positive"] <= 1
            )
            assert (
                node["prediction"]["negative"] >= 0
                and node["prediction"]["negative"] <= 1
            )

        # Validate the predictor.
        assert "predictor" in node
        assert type(node["predictor"]) is int or node["predictor"] is None
        if type(node["predictor"]) is int:
            assert node["predictor"] >= 0 and node["predictor"] <= 2

        # Validate the value.
        assert "value" in node
        assert (
            type(node["value"]) is int
            or type(node["value"]) is float
            or type(node["value"]) is str
            or node["value"] is None
        )

        # Validate the true/false keys.
        assert "true" in node
        assert type(node["true"]) is int or node["true"] is None
        assert "false" in node
        assert type(node["false"]) is int or node["false"] is None

    # Validate the root node.

    root = result["tree"][-1]
    assert root["prediction"] is None
    assert type(root["predictor"]) is int
    assert (
        type(root["value"]) is int
        or type(root["value"]) is float
        or type(root["value"]) is str
    )
    assert type(root["true"]) is int
    assert type(root["false"]) is int

    # Validate the tree's metadata.

    assert "metadata" in result
    assert "ignored_observation_indices" in result["metadata"]
    assert type(result["metadata"]["ignored_observation_indices"]) is set
    for index in result["metadata"]["ignored_observation_indices"]:
        assert type(index) is int
