# Copyright (c) 2021 The Stethobot Authors. All rights reserved.

"""Script for building a breast cancer diagnosis model."""

import sys

# Get around Python's import system.
sys.path.append("..")

from model import DiagnosticModel


def main() -> None:
    """Produce breast cancer diagnosis model."""
    # Load sample file's lines into list, excluding trailing newline chars.
    with open("sample") as f:
        lines = [line.rstrip() for line in f.readlines()]

    # Turn each CSV record (i.e. each line) into a list, thereby making a list of lists of strings.
    sample = [line.split(",") for line in lines]

    # The sample currently contains ID numbers as the first column of each row.
    # This is useless for diagnosis, so we remove them.
    for i in range(len(sample)):
        del sample[i][0]

    # The next column is the label, which we'll deal with shortly.
    # For now, we convert the types of the other columns - which are stringified floats - into ACTUAL floats.
    for i in range(len(sample)):
        for j in range(1, len(sample[0])):
            try:
                sample[i][j] = float(sample[i][j])
            except ValueError:
                pass

    # Now we deal with the label column, which is either "M" for malignant, or "B" for benign.
    # We convert the labels to "+" or "-" respectively, and split them off into a separate list.
    labels = []
    for i in range(len(sample)):
        label = sample[i].pop(0)
        label = "positive" if label == "M" else "negative"
        labels.append(label)

    # Finally, we feed in the preprocessed data to train the model and serialize it into a JSON file.
    brf = DiagnosticModel(num_trees=20)
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
