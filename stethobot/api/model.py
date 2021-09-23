# Copyright (c) 2021 The Stethobot Authors. All rights reserved.

"""Implements a balanced random forest with an API similar to scikit-learn."""

import json
import multiprocessing
import random
import statistics


class TreeFitter:

    """Responsible for fitting an individual tree of a balanced random forest.

    Although a class, it's better to think of it as a glorified function with two phases: a setup phase i.e. the
    constructor, and a run phase i.e. the __call__ method. This is a convenient architecture because it plays nicely
    with the multiprocessing module (which wants callable objects), while also allowing for per-tree attributes (which
    is required for the algorithm).

    Attributes:
        _tree: A list of dicts which encodes the tree. All dicts have the same keys. The dicts represent the tree...
            ...nodes, and the last dict of the list is the root node. To run the tree, we start at the root and...
            ...travel until a leaf node. This happens in a similar manner to a linked list, using the dicts' keys...
            ...instead of pointers. The keys are described below:
            1. The first key is "prediction", which is either a dict (indicating a leaf node) or NoneType. If the...
            ...value is a dict, then it maps the string "positive"/"negative" labels to float probabilities...
            ...and all the other keys of the dict will have NoneType values. Otherwise, none will have NoneType values.
            2. The second key is "predictor". Its value is an integer index that encodes the predictor to be CHECKED.
            3. The third key is the "value" key. Its value is a string/integer/float, which is the value of the...
            ...predictor to be CHECKED. If a string, the CHECK is equivalence; otherwise, the CHECK is less-than.
            4. The fourth key is the "true" key. Its value is the integer index of the next node to travel to, but...
            ...only if the CHECK resulted in True.
            5. The fifth and final key is the "false" key. It is the same as the "true" key, but corresponds to...
            ...a CHECK value of False.
        _num_split: Integer number of predictors chosen at each split. Must be bounded between 1 and the total number
            of predictors in the sample.
        _label_set: A set which contains all unique labels found in the sample.
        _effective_sample: List of lists of integers/floats/strings; the result of a stratified bootstrap on the
            original sample.
        _effective_labels: A list of string labels for all the observations within the effective sample.
        _ignored_observation_indices: Set of integer indices of the sample observations not in the effective sample.
            This is useful for computing out-of-bag accuracy. It is "balanced" by downsampling every class to the size
            of the smallest class of the sample. This keeps the out-of-bag computation from being biased.
    """

    def __init__(self, sample, labels, num_split):
        """Prepare important state attributes for use during fitting.

        Args:
            sample: A list of lists of integers/floats/strings. Each individual list encodes an observation, and each
                individual value encodes a predictor's value. All lists must have the same length.
            labels: A list of strings corresponding to the labels of each observation. Should be the same length as the
                sample argument above.
        """
        # Initially empty; this will be filled after the __call__ method.
        self._tree = []

        # Validation for this is done in the higher-level class.
        self._num_split = num_split

        # Now we'll initialize the label set.
        # While we're doing that, we also initialize observation-by-class bins for the upcoming stratified bootstrap.
        self._label_set = set()
        observation_bins = {}
        for observation, label in zip(sample, labels):
            self._label_set.add(label)
            if label not in observation_bins:
                observation_bins[label] = []
            observation_bins[label].append(observation)

        # Next up, we perform the stratified bootstrap.
        # In doing so, we initialize two more attributes: the "effective" sample and "effective" labels.

        self._effective_sample = []
        self._effective_labels = []
        least_frequent_label = min(
            observation_bins, key=lambda key: len(observation_bins[key])
        )
        max_size = len(observation_bins[least_frequent_label])
        for label in observation_bins:
            chosen_observations = random.choices(observation_bins[label], k=max_size)
            self._effective_sample.extend(chosen_observations)
            self._effective_labels.extend([label] * max_size)

        # Finally, we balance the list of ignored observations' indices.
        # We do this by sampling without replacement up to the number of unobserved indices for the smallest class.
        # Basically, if class A has 10 ignored observations, and class B has 5 ignored observations...
        # ...then we want to downsample the 10 from class A down to just 5, thus making it balanced.

        # First, we collect lists of the ignored observations' indices across each label.
        ignored_indices_per_label = {label: [] for label in observation_bins}
        indices = [i for i in range(len(sample))]
        for index, observation, label in zip(indices, sample, labels):
            if observation not in self._effective_sample:
                ignored_indices_per_label[label].append(index)

        # Second, we find out which of these lists is the smallest, thus discovering the "smallest" label.
        smallest_label = None
        for label in ignored_indices_per_label:
            if smallest_label is None:
                smallest_label = label
            else:
                if len(ignored_indices_per_label[label]) < len(
                    ignored_indices_per_label[smallest_label]
                ):
                    smallest_label = label

        # Third, we use the number of ignored indices in the "smallest" label to perform the balancing.
        self._ignored_observation_indices = set()
        max_size = len(ignored_indices_per_label[smallest_label])
        for label in ignored_indices_per_label:
            sampled_wo_replacement = random.sample(
                ignored_indices_per_label[label], max_size
            )
            for index in sampled_wo_replacement:
                self._ignored_observation_indices.add(index)

    def __call__(self, queue):
        """Fit the tree.

        Args:
            queue: A "multiprocessing.Queue" object that collects the fit tree and metadata.
        """
        # Fill _tree attribute with the fit tree.
        self._build_tree(self._effective_sample, self._effective_labels)
        result = {
            "tree": self._tree,
            "metadata": {
                "ignored_observation_indices": self._ignored_observation_indices
            },
        }
        queue.put(result)

    def _build_tree(self, sample, labels):
        """Recursively build the tree.

        Args:
            sample: A list of lists of integers/floats/strings. Each individual list encodes an observation, and each
                individual value encodes a predictor's value. All lists must have the same length. This argument will
                change with each recursive call. It should initially be given the effective sample as computed in the
                constructor.
            labels: A list of strings corresponding to the labels of each observation. Should be the same length as the
                sample argument above.

        Returns:
            The integer index of the last node added. This value is only important for implementing the recursion.
            The actual item of interest (the fit tree) will be stored into the _tree attribute after the recursion is
            complete.
        """
        # Try to find a predictor/value pair to split on to decrease gini impurity.
        try:
            predictor, value = self._find_best_random_split(sample, labels)

        # If we can't, generate a leaf node.
        except ValueError:
            # Compute the probabilities for each label.
            label_counts = self._label_count(labels)
            for label in self._label_set:
                if label not in label_counts:
                    label_counts[label] = 0
            for label in label_counts:
                label_counts[label] = label_counts[label] / len(sample)

            # Add the leaf node with the probabilities as the prediction, then return its index.
            node = {
                "prediction": label_counts,
                "predictor": None,
                "value": None,
                "true": None,
                "false": None,
            }
            self._tree.append(node)
            return len(self._tree) - 1

        # If we can, then let's roll with it!
        true_subsample, true_labels, false_subsample, false_labels = self._split(
            sample, labels, predictor, value
        )

        # Make a recursive call for each half.
        true_index = self._build_tree(true_subsample, true_labels)
        false_index = self._build_tree(false_subsample, false_labels)
        self._tree.append(
            {
                "prediction": None,
                "predictor": predictor,
                "value": value,
                "true": true_index,
                "false": false_index,
            }
        )

        # After the recursion is completely done, this will be the index of the last node i.e. the root node.
        return len(self._tree) - 1

    def _find_best_random_split(self, sample, labels):
        """Find which split (on a random subset of predictors) that reduces gini impurity the most.

        Args:
            sample: A list of lists of integers/floats/strings. Each individual list encodes an observation, and each
                individual value encodes a predictor's value. All lists must have the same length.
            labels: A list of strings corresponding to the labels of each observation. Should be the same length as the
                sample argument above.

        Returns:
            A tuple (predictor, value) pair, where the predictor is an integer index, and the value is an
            integer/float/string. They represent the best possible split available.

        Raises:
            ValueError: The given sample cannot be split to reduce gini impurity.
        """
        # Get the predictors as indices.
        all_predictors = list(range(len(sample[0])))

        # Randomly sample w/o replacement a subset of predictors.
        chosen_predictors = random.sample(all_predictors, self._num_split)

        # Before we begin the search, compute the current impurity and prepare a container for the results.
        original_impurity = self._gini_impurity(labels)
        split_results = []

        for predictor in chosen_predictors:
            # Within a single predictor, it doesn't make sense to split again on the same value.
            # Hence, we keep track of seen values with a set.
            seen_values = set()

            for observation in sample:
                value = observation[predictor]
                if value not in seen_values:
                    # Try to produce a split i.e. both true and false subsamples with size > 0.
                    try:
                        (
                            true_subsample,
                            true_labels,
                            false_subsample,
                            false_labels,
                        ) = self._split(sample, labels, predictor, value)
                    except ValueError:
                        continue

                    # Compute the impurities of the resulting splits.
                    true_impurity = self._gini_impurity(true_labels)
                    false_impurity = self._gini_impurity(false_labels)

                    # Compute weights based off the splits' sizes.
                    true_weight = len(true_subsample) / len(sample)
                    false_weight = len(false_subsample) / len(sample)

                    # Combine them together to get weighted average gini impurity.
                    weighted_avg_impurity = (true_impurity * true_weight) + (
                        false_impurity * false_weight
                    )

                    # Measure the decreased impurity, add the split as a candidate, and record the value.
                    decreased_impurity = original_impurity - weighted_avg_impurity
                    split_result = {
                        "predictor": predictor,
                        "value": value,
                        "decreased_impurity": decreased_impurity,
                    }
                    split_results.append(split_result)
                    seen_values.add(value)

        # There are two failure conditions:
        # 1. There was literally no way to physically split the sample i.e. same values for each...
        #    ...observation across all predictors.
        # 2. There were ways to physically split the sample, but none resulted in impurity decreases.
        if len(split_results) == 0 or all(
            [split["decreased_impurity"] <= 0 for split in split_results]
        ):
            raise ValueError("Given sample cannot be split to decrease gini impurity!")

        # Otherwise, we're good to go.
        best_split = max(split_results, key=lambda x: x["decreased_impurity"])
        return best_split["predictor"], best_split["value"]

    def _gini_impurity(self, labels):
        """Compute the gini impurity of a list of labels.

        Args:
            labels: List of string labels.

        Returns:
            A float; the gini impurity as a probability.
        """
        label_count = self._label_count(labels)
        gini_impurity = 1
        for label in label_count:
            label_prob = label_count[label] / len(labels)
            gini_impurity -= label_prob ** 2
        return gini_impurity

    @staticmethod
    def _split(sample, labels, predictor, value):
        """Split the given sample's observations by a given predictor's value.

        This method does not deal with gini impurity.
        All it does is physically split the sample.

        Args:
            sample: A list of lists of integers/floats/strings. Each individual list encodes an observation, and each
                individual value encodes a predictor's value. All lists must have the same length.
            labels: A list of strings corresponding to the labels of each observation. Should be the same length as the
                sample argument above.
            predictor: Integer index of a sample observation to split on.
            value: Integer/float/string value of the given predictor to split on.

        Returns:
            A 4-tuple (true_subsample, true_labels, false_subsample, false_labels). The subsamples are typed as the
            sample argument is, and the labels are typed as the labels argument is.

        Raises:
            ValueError: There's no way to split the sample.
        """
        true_subsample = []
        true_labels = []
        false_subsample = []
        false_labels = []
        for observation, label in zip(sample, labels):
            if type(value) is str:
                if observation[predictor] == value:
                    true_subsample.append(observation)
                    true_labels.append(label)
                else:
                    false_subsample.append(observation)
                    false_labels.append(label)
            elif observation[predictor] < value:
                true_subsample.append(observation)
                true_labels.append(label)
            else:
                false_subsample.append(observation)
                false_labels.append(label)
        if len(true_subsample) == 0 or len(false_subsample) == 0:
            raise ValueError("Given predictor/value does not produce a split.")
        return true_subsample, true_labels, false_subsample, false_labels

    @staticmethod
    def _label_count(labels):
        """Count the number of each label in a given label list.

        Args:
            labels: List of string labels.

        Returns:
            A dict {"label": int} for each label.
        """
        label_count = {}
        for label in labels:
            if label not in label_count:
                label_count[label] = 0
            label_count[label] += 1
        return label_count


class DiagnosticModel:

    """Balanced random forest implementation for binary classification.

    Attributes:
        _forest: List of lists, each of which encodes a decision tree. Together, encodes the balanced random forest.
        _num_trees: Integer number of trees within the forest. Must be >= 1.
        _num_split: Integer number of predictors chosen at each split. Default is -1, which means "sqrt(predictors)".
            Otherwise, must be bounded between 1 and the total number of predictors in the sample.
        _oob_accuracy: Float out-of-bag estimate for generalization accuracy, between 0 and 1.
        _sensitivity: Float true positive rate, between 0 and 1.
        _specificity: Float true negative rate, between 0 and 1.
    """

    def __init__(self, num_trees=100, num_split=-1):
        """Initialize attributes.

        Raises:
            ValueError: Number of trees - or chosen predictors at each split - is invalid.
        """
        if num_trees <= 0:
            raise ValueError("Number of trees must be positive.")

        # This is only partial validation; we need to see the sample before we can fully validate it.
        if num_split < -1 or num_split == 0:
            raise ValueError("Number of splits must be either -1 or positive.")

        self._num_trees = num_trees
        self._num_split = num_split
        self._forest = []
        self._oob_accuracy = None
        self._sensitivity = None
        self._specificity = None

    def fit(self, sample, diagnoses):
        """Fit the model to a given sample of observations and a list of diagnoses.

        Args:
            sample: A 2D, rectangular list of integers/floats/strings/NoneTypes. Columns can have any number of
                NoneTypes, but must have one or more non-NoneTypes; these must be either integers/floats or strings,
                but not a mix of both. Any NoneTypes within a column will be imputed based off of the non-NoneTypes.
            diagnoses: A list of strings, all of which must be either "positive" and "negative".

        Returns:
            A 3-tuple (out_of_bag_accuracy, sensitivity, specificity) where all are floats between 0 and 1.

        Raises:
            ValueError: Either num_split is invalid, or the labels contain a non-"positive" or non-"negative" string.
        """
        # Validate the number of splits now that we can see the sample.
        if self._num_split == -1:
            self._num_split = int(len(sample[0]) ** 0.5)
        elif self._num_split > len(sample[0]):
            raise ValueError(
                "Cannot split on more predictors than there are in sample."
            )

        # Next, validate the diagnosis labels.
        for diagnosis in diagnoses:
            if diagnosis != "positive" and diagnosis != "negative":
                raise ValueError('All labels must be either "positive" or "negative"!')

        # Remove any duplicate inputs.
        deduped_sample, deduped_diagnoses = self._dedupe(sample, diagnoses)

        # Impute the sample unknown values (if any).
        effective_sample, effective_diagnoses = self._impute(
            deduped_sample, deduped_diagnoses
        )

        # Fit all trees in parallel.
        jobs, results = [], []
        job_retvals = multiprocessing.Queue()
        for _ in range(self._num_trees):
            tree_fitter = TreeFitter(
                effective_sample, effective_diagnoses, self._num_split
            )
            job = multiprocessing.Process(target=tree_fitter, args=(job_retvals,))
            jobs.append(job)
            job.start()
        for job in jobs:
            result = job_retvals.get()
            results.append(result)
            self._forest.append(result["tree"])
        for job in jobs:
            job.join()

        # Compute and return the statistics from the fitting process.
        fitting_statistics = self._compute_statistics(
            results, effective_sample, effective_diagnoses
        )
        self._oob_accuracy = fitting_statistics[0]
        self._sensitivity = fitting_statistics[1]
        self._specificity = fitting_statistics[2]
        return self._oob_accuracy, self._sensitivity, self._specificity

    def diagnose(self, observation):
        """Predict a diagnosis for a given observation.

        Args:
            observation: A list of integers/floats/strings representing the input data.

        Returns:
            A 2-tuple (diagnosis, confidence_score) where the first is a string - either "positive" or "negative" - and
            the second is an integer between 0 and 100 representing the model's confidence in its diagnosis.

        Raises:
            AttributeError: The model hasn't been fitted or loaded.
        """
        if (
            len(self._forest) == 0
            or self._oob_accuracy is None
            or self._sensitivity is None
            or self._specificity is None
        ):
            raise AttributeError("Model is uninitialized.")
        result = self._run_forest(self._forest, observation)
        diagnosis = max(result, key=result.get)
        confidence_score = int(result[diagnosis] * 100)
        return diagnosis, confidence_score

    def to_json(self):
        """Encode the model as JSON.

        Returns:
            A JSON string.

        Raises:
            AttributeError: The model hasn't been fitted or loaded.
        """
        if (
            len(self._forest) == 0
            or self._oob_accuracy is None
            or self._sensitivity is None
            or self._specificity is None
        ):
            raise AttributeError("Model is uninitialized.")
        return json.dumps(
            {
                "model": self._forest,
                "metadata": {
                    "number_of_trees": self._num_trees,
                    "predictors_each_split": self._num_split,
                    "out_of_bag_accuracy": self._oob_accuracy,
                    "sensitivity": self._sensitivity,
                    "specificity": self._specificity,
                },
            }
        )

    def from_json(self, model):
        """Load model (i.e. instance attributes) from JSON.

        Args:
            model: JSON string that encodes the model.
        """
        result = json.loads(model)
        self._num_trees = result["metadata"]["number_of_trees"]
        self._num_split = result["metadata"]["predictors_each_split"]
        self._forest = result["model"]
        self._oob_accuracy = result["metadata"]["out_of_bag_accuracy"]
        self._sensitivity = result["metadata"]["sensitivity"]
        self._specificity = result["metadata"]["specificity"]

    @staticmethod
    def _dedupe(sample, diagnoses):
        """Remove duplicates in the input data.

        Args:
            sample: A 2D, rectangular list of integers/floats/strings/NoneTypes.
            diagnoses: A list of strings, all of which must be either "positive" and "negative".

        Returns:
            A 2-tuple (sample, diagnoses) that is the same as the input arguments, but with duplicate observations in
            the sample, as well as their corresponding diagnoses, removed.
        """
        # Concatenate inputs for convenience.
        dataset = []
        for observation_index in range(len(sample)):
            dataset.append(sample[observation_index])
            dataset[-1].append(diagnoses[observation_index])

        # Sets can hash (and therefore remove) tuples, but not lists.
        dataset_as_tuples = [tuple(data) for data in dataset]
        deduped_dataset = list(set(dataset_as_tuples))

        # Extract the observations and diagnoses back into separate lists.
        sample_result = []
        diagnoses_result = []
        for data in deduped_dataset:
            data_as_list = list(data)
            diagnoses_result.append(data_as_list.pop())
            sample_result.append(data_as_list)
        return sample_result, diagnoses_result

    @staticmethod
    def _impute(sample, diagnoses):
        """Impute missing values in the sample.

        Args:
            sample: A 2D, rectangular list of integers/floats/strings/NoneTypes. Columns can have any number of
                NoneTypes, but must have one or more non-NoneTypes; these must be either integers/floats or strings,
                but not a mix of both. The former indicates a numerical column, while the latter indicates a
                categorical column. Any NoneTypes within a column will be imputed based off of the non-NoneTypes.
            diagnoses: A list of strings, all of which must be either "positive" and "negative". The reason this is
                required is because the ordering of the sample gets mixed up due to the imputation algorithm, so the
                diagnoses have to be rematched with their observations.

        Returns:
            The sample with missing NoneType values replaced by median or mode, along with corresponding diagnoses.
            Median is used for numerical predictors, and mode for categorical predictors.
        """
        # Group observations by their diagnostic label.
        observation_bins = {}
        for observation, diagnosis in zip(sample, diagnoses):
            if diagnosis not in observation_bins:
                observation_bins[diagnosis] = []
            observation_bins[diagnosis].append(observation)

        # Impute the sample.
        effective_sample = []
        effective_diagnoses = []
        for diagnosis in observation_bins:
            chosen_observations = observation_bins[diagnosis]
            # For each predictor...
            for predictor in range(len(chosen_observations[0])):
                # ...gather all of its values...
                values = []
                for observation in chosen_observations:
                    if observation[predictor] is not None:
                        values.append(observation[predictor])

                # ...find the "typical" value...
                typical_value = None
                if any(type(value) is str for value in values):
                    typical_value = statistics.mode(values)
                else:
                    typical_value = statistics.median(values)

                # ...and fill in the missing values that that "typical" value.
                for index in range(len(chosen_observations)):
                    if chosen_observations[index][predictor] is None:
                        chosen_observations[index][predictor] = typical_value
            effective_sample.extend(chosen_observations)
            effective_diagnoses.extend([diagnosis] * len(chosen_observations))
        return effective_sample, effective_diagnoses

    def _compute_statistics(self, trees, sample, diagnoses):
        """Compute the out-of-bag accuracy, sensitivity, and specificity.

        Args:
            trees: See the output of TreeFitter's __call__ method for the typing information.
            sample: A list of lists of integers/floats/strings. Each individual list encodes an observation, and each
                individual value encodes a predictor's value. All lists must have the same length.
            diagnoses: A list of strings, all of which must be either "positive" and "negative".

        Returns:
            A 3-tuple (out_of_bag_accuracy, sensitivity, specificity), all of which are floats between 0 and 1.

        Raises:
            ValueError: The minority class of the sample was so small or that there were so many duplicates that no OOB
                accuracy calculation could be made.
        """
        # For sensitivity and specificity measurements.
        true_pos = 0
        true_neg = 0
        total_pos = 0
        total_neg = 0

        # For out-of-bag accuracy measurement.
        num_oob_correct = 0
        num_oob_attempts = 0

        indices = [i for i in range(len(sample))]
        for index, observation, diagnosis in zip(indices, sample, diagnoses):
            # Build a forest out of the trees that didn't include the given observation.
            oob_forest = []
            for tree in trees:
                if index in tree["metadata"]["ignored_observation_indices"]:
                    oob_forest.append(tree["tree"])

            # Maybe all of the trees ended up fitting on the observation? Just move onto the next observation.
            if len(oob_forest) == 0:
                continue
            else:
                num_oob_attempts += 1

            # Run our makeshift forest and get its prediction.
            oob_forest_pred = self._run_forest(oob_forest, observation)
            predicted_diagnosis = max(
                oob_forest_pred, key=lambda key: oob_forest_pred[key]
            )
            if predicted_diagnosis == diagnosis:
                num_oob_correct += 1

            # Adjust positive/negative counts for sensitivity and specificity.
            if diagnosis == "positive":
                total_pos += 1
            else:
                total_neg += 1
            if diagnosis == "positive" and predicted_diagnosis == "positive":
                true_pos += 1
            if diagnosis == "negative" and predicted_diagnosis == "negative":
                true_neg += 1

        # Validate that we got some out-of-bag accuracy estimation before returning.
        if num_oob_attempts == 0:
            raise ValueError(
                "OOB acc. calc. failed: too few observations for minority class, or too many duplicates."
            )
        return (
            (num_oob_correct / num_oob_attempts),
            (true_pos / total_pos),
            (true_neg / total_neg),
        )

    def _run_forest(self, forest, observation):
        """Run the given forest on an observation.

        Args:
            forest: A list of trees to be run. See docs on the "_tree" attribute from TreeFitter for the format of each
                individual tree.
            observation: A list of integers/floats/strings representing the input data.

        Returns:
            A dict of float prediction confidence scores (summing to 1) for both positive and negative diagnoses.
            e.g. {"positive": 0.75, "negative": 0.25}
        """
        diagnoses = []
        for tree in forest:
            diagnosis = self._run_tree(tree, observation)
            diagnoses.append(diagnosis)
        return self._average_trees(diagnoses)

    @staticmethod
    def _run_tree(tree, observation):
        """Run a given tree to produce its predicted diagnosis.

        Args:
            tree: The tree to run. See docs on the "_tree" attribute from TreeFitter for the format.
            observation: A list of integers/floats/strings. This is the input data to run through the tree.

        Returns:
            A dict of float prediction confidence scores (summing to 1) for both positive and negative diagnoses.
            e.g. {"positive": 0.75, "negative": 0.25}
        """
        root = tree[-1]
        current_node = root
        while current_node["prediction"] is None:
            predictor = current_node["predictor"]
            predictor_type = type(current_node["value"])
            value = current_node["value"]
            next_tree_index = 0
            if predictor_type is str and observation[predictor] == value:
                next_tree_index = current_node["true"]
            elif observation[predictor] < value:
                next_tree_index = current_node["true"]
            else:
                next_tree_index = current_node["false"]
            current_node = tree[next_tree_index]
        return current_node["prediction"]

    @staticmethod
    def _average_trees(diagnoses):
        """Compute the mean of several predicted diagnoses.

        Args:
            diagnoses: List of predicted diagnoses. Each should be a dict with "positive" and "negative" keys.
                Both keys should lead to floats which together sum up to 1.

        Returns:
            The average prediction as a dict with the same typing as the argument.
        """
        average = {"positive": 0, "negative": 0}
        for diagnosis in diagnoses:
            average["positive"] += diagnosis["positive"]
            average["negative"] += diagnosis["negative"]
        average["positive"] /= len(diagnoses)
        average["negative"] /= len(diagnoses)
        return average
