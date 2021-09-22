# Copyright (c) 2021 The Stethobot Authors. All rights reserved.

"""Script for building a diabetes diagnosis model."""

import sys

# Get around Python's import system.
sys.path.append("..")

from model import DiagnosticModel


def main() -> None:
    """Produce diabetes diagnosis model."""
    # Load sample file's lines into list, excluding trailing newline chars.
    with open("sample") as f:
        lines = [line.rstrip() for line in f.readlines()]

    # Turn each CSV record (i.e. each line) into a list, thereby making a list of lists of strings.
    sample = [line.split(",") for line in lines]

    # First line is just column descriptions (i.e. headers). We don't need them.
    sample.pop(0)

    # There are some missing values in the sample. Set them to NoneType to mark them to be imputed.
    for obs in sample:
        for predictor in range(1, 8):
            if obs[predictor] == "0":
                obs[predictor] = None

    # For the rest of the values, we convert them from stringified floats to ACTUAL floats.
    for obs in sample:
        for predictor in range(len(sample[0])):
            if obs[predictor] is not None:
                obs[predictor] = float(obs[predictor])

    # Now for the label column, it will be either 1 or 0.
    # We convert the labels to "+" and "-" respectively, and split them off into a separate list.
    labels = []
    for i in range(len(sample)):
        label = int(sample[i].pop())
        label = "positive" if label == 1 else "negative"
        labels.append(label)

    # Finally, train the model and serialize it into a JSON file.
    brf = DiagnosticModel(num_trees=30)
    acc, sens, spec = brf.fit(sample, labels)
    brf_json = brf.to_json()
    with open("model.json", "w") as f:
        f.write(brf_json)

    # For convenience (so we don't have to open the JSON file), print out relevant stats directly to the console.
    print(f"     Acc: {acc}")
    print(f"    Sens: {sens}")
    print(f"    Spec: {spec}")


if __name__ == "__main__":
    main()
