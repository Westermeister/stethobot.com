# Copyright (c) 2021 The Stethobot Authors. All rights reserved.

"""Script for building a heart disease diagnosis model."""

import sys

# Get around Python's import system.
sys.path.append("..")

from model import DiagnosticModel


def main() -> None:
    """Produce heart disease diagnosis model."""
    #  Load sample file's lines into list, excluding trailing newline chars.
    with open("sample") as f:
        lines = [line.rstrip() for line in f.readlines()]

    # Turn each CSV record (i.e. each line) into a list, thereby making a list of lists of strings.
    sample = [line.split(",") for line in lines]

    # The sample contains missing values. Mark them as NoneType for imputation.
    # The rest of the values can be converted from stringified floats to ACTUAL floats.
    for i in range(len(sample)):
        for j in range(len(sample[0]) - 1):
            if sample[i][j] == "?":
                sample[i][j] = None
            else:
                try:
                    sample[i][j] = float(sample[i][j])
                except ValueError:
                    pass

    # Labels range from 0 to 4. Only 0 is counted as a negative test result.
    # Split off the labels as "+" and "-" into a separate list.
    labels = []
    for i in range(len(sample)):
        label = int(sample[i].pop())
        label = "positive" if label > 0 else "negative"
        labels.append(label)

    # Train the model and serialize it into a JSON file.
    brf = DiagnosticModel(num_trees=100)
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
