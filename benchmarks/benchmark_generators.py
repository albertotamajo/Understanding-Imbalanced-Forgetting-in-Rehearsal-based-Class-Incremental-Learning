"""
This module provides some high-level benchmark generators.
"""
from typing import Sequence, Optional, Dict, Union, Set, List, Any
import random
from benchmarks.scenarios import NCScenario
from avalanche.benchmarks.utils.classification_dataset import SupportedDataset, as_supervised_classification_dataset
from avalanche.benchmarks.utils.data_attribute import DataAttribute
from avalanche.benchmarks.utils import classification_subset


def nc_benchmark(
    train_dataset: SupportedDataset,
    test_dataset: SupportedDataset,
    n_experiences: int,
    task_labels: bool,
    *,
    val_dataset: Optional[Union[SupportedDataset, float]] = None,
    shuffle: bool = True,
    seed: Optional[int] = None,
    fixed_class_order: Optional[Sequence[int]] = None,
    per_exp_classes: Optional[Dict[int, int]] = None,
    class_ids_from_zero_from_first_exp: bool = False,
    class_ids_from_zero_in_each_exp: bool = False,
    train_transform=None,
    eval_transform=None,
    include_indices=False,
    reproducibility_data: Optional[Dict[str, Any]] = None,
) -> NCScenario:
    """
    This function is identical to `avalanche.benchmarks.nc_benchmark` except for  four modifications:
        - it does not allow to pass multiple train or test datasets (only one for both is accepted)
        - it provides support for a validation dataset. Read the doc of the `val_dataset` parameter for more
          information about this.
        - it allows to include the sample indices as DataAttribute. Read the doc of the `include_indices` parameter for
          more information about this.
        - it discards the `one_dataset_per_exp` parameter included in the `avalanche.benchmarks.nc_benchmark` function
          because it is only useful when providing multiple train\test datasets, which this function does not allow to.

    This is the high-level benchmark instances generator for the
    "New Classes" (NC) case. Given a train, a test and (optionally) a validation dataset creates
    the continual stream of data as a series of experiences. Each experience
    will contain all the instances belonging to a certain set of classes and a
    class won't be assigned to more than one experience.

    This function returns a CL benchmark object consisting of two or three streams:
        - If `val_dataset` is None, the benchmark consists of two streams: `train` and `test`
        - Otherwise, the benchmark consists of three streams: `train`, `test` and `val`.

    The `train`, `test` and `val` streams can be accessed directly by using the `train_stream`, `test_stream` and
    `val_stream` attributes of the returned benchmark object, respectively.

    This is the reference helper function for creating instances of Class- or
    Task-Incremental benchmarks.

    The ``task_labels`` parameter determines if each incremental experience has
    an increasing task label or if, at the contrary, a default task label 0
    has to be assigned to all experiences. This can be useful when
    differentiating between Single-Incremental-Task and Multi-Task scenarios.

    There are other important parameters that can be specified in order to tweak
    the behaviour of the resulting benchmark. Please take a few minutes to read
    and understand them as they may save you a lot of work.

    This generator features a integrated reproducibility mechanism that allows
    the user to store and later re-load a benchmark. For more info see the
    ``reproducibility_data`` parameter.

    :param train_dataset: a single train dataset.
    :param test_dataset: a single test dataset.
    :param n_experiences: The number of incremental experience
    :param task_labels: If True, each experience will have an ascending task
            label. If False, the task label will be 0 for all the experiences.
    :param val_dataset: a single validation dataset or a float number between 0 and 1 (both excluded). If a float
        number is provided, it is used to apportion a fraction of the samples in the train dataset to validation
        (these samples are not used during training but only for validation). The fraction of samples apportioned to
        validation are selected randomly from each class independently, i.e. if the float number is 0.3, then for each
        class in the training dataset, 30% of its inputs are selected randomly. If None, no fraction of the samples
        in the train dataset are apportioned to validation, i.e. a validation stream is not added to the returned
        instance. Defaults to None.
    :param shuffle: If True, the class (or experience) order will be shuffled.
        Defaults to True.
    :param seed: A valid int used to initialize the random number generator before apportioning a fraction of training
        samples to validation (if `val_dataset` is a float) and before shuffling the class order in
        the incremental experiences (if `shuffle` is set to True). Note that the seed is set twice with the same value,
        immediately before creating the validation set (if `val_dataset` is a float) and immediately before shuffling
        the class order in the incremental experiences (if `shuffle` is set to True). If None, no random number
        generator seed is set, preventing results from being replicated. Default is None.
    :param fixed_class_order: If not None, the class order to use (overrides
        the shuffle argument). Very useful for enhancing reproducibility.
        Defaults to None.
    :param per_exp_classes: If not None, a dictionary whose keys are
        (0-indexed) experience IDs and their values are the number of classes
        to include in the respective experiences. The dictionary doesn't
        have to contain a key for each experience! All the remaining experiences
        will contain an equal amount of the remaining classes. The
        remaining number of classes must be divisible without remainder
        by the remaining number of experiences. For instance,
        if you want to include 50 classes in the first experience
        while equally distributing remaining classes across remaining
        experiences, just pass the "{0: 50}" dictionary as the
        per_experience_classes parameter. Defaults to None.
    :param class_ids_from_zero_from_first_exp: If True, original class IDs
        will be remapped so that they will appear as having an ascending
        order. For instance, if the resulting class order after shuffling
        (or defined by fixed_class_order) is [23, 34, 11, 7, 6, ...] and
        class_ids_from_zero_from_first_exp is True, then all the patterns
        belonging to class 23 will appear as belonging to class "0",
        class "34" will be mapped to "1", class "11" to "2" and so on.
        This is very useful when drawing confusion matrices and when dealing
        with algorithms with dynamic head expansion. Defaults to False.
        Mutually exclusive with the ``class_ids_from_zero_in_each_exp``
        parameter.
    :param class_ids_from_zero_in_each_exp: If True, original class IDs
        will be mapped to range [0, n_classes_in_exp) for each experience.
        Defaults to False. Mutually exclusive with the
        ``class_ids_from_zero_from_first_exp`` parameter.
    :param train_transform: The transformation to apply to the training data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation for a
        comprehensive list of possible transformations). Defaults to None.
    :param eval_transform: The transformation to apply to the test data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation for a
        comprehensive list of possible transformations). Defaults to None.
    :param include_indices: whether to include the indices of the samples in the train, test and (if `val_dataset` is
        not None) validation datasets as DataAttribute. The index of
        each sample is the index to be used to retrieve the given sample from the underlying dataset. The name of the
        DataAttribute is set to `dataset_indices` and its `use_in_getitem` attribute is set to True. Therefore, the
        index of a sample is returned when calling `__getitem__`. If `val_dataset` is a float, the index of each sample
        in the validation dataset refers to the index to be used to retrieve the given sample from the dataset
        underlying the train dataset.
    :param reproducibility_data: If not None, overrides all the other
        benchmark definition options. This is usually a dictionary containing
        data used to reproduce a specific experiment. One can use the
        ``get_reproducibility_data`` method to get (and even distribute)
        the experiment setup so that it can be loaded by passing it as this
        parameter. In this way one can be sure that the same specific
        experimental setup is being used (for reproducibility purposes).
        Beware that, in order to reproduce an experiment, the same train and
        test datasets must be used. Defaults to None.

    :return: A properly initialized :class:`NCScenario` instance.
    """

    if class_ids_from_zero_from_first_exp and class_ids_from_zero_in_each_exp:
        raise ValueError(
            "Invalid mutually exclusive options "
            "class_ids_from_zero_from_first_exp and "
            "classes_ids_from_zero_in_each_exp set at the "
            "same time"
        )

    if isinstance(train_dataset, (list, tuple)):
        # Multi-dataset setting
        raise ValueError("A list cannot be passed for train_dataset")

    if isinstance(test_dataset, (list, tuple)):
        # Multi-dataset setting
        raise ValueError("A list cannot be passed for test_dataset")

    if isinstance(val_dataset, float):
        if not 0 < val_dataset < 1:
            raise ValueError("The val_dataset parameter must be between 0 and 1")
    elif isinstance(val_dataset, (list, tuple)):
        raise ValueError("A list cannot be passed for val_dataset")


    seq_train_dataset = as_supervised_classification_dataset(train_dataset)
    seq_test_dataset = as_supervised_classification_dataset(test_dataset)

    transform_groups = dict(train=(train_transform, None), eval=(eval_transform, None))

    # Set transformation groups
    final_train_dataset = as_supervised_classification_dataset(
        seq_train_dataset,
        transform_groups=transform_groups,
        initial_transform_group="train",
    )
    if include_indices:
        final_train_dataset = final_train_dataset.update_data_attribute(
            name="dataset_indices",
            new_value=DataAttribute(data=[i for i in range(len(final_train_dataset))],
                                    name="dataset_indices",
                                    use_in_getitem=True
                                    )
        )

    final_test_dataset = as_supervised_classification_dataset(
        seq_test_dataset,
        transform_groups=transform_groups,
        initial_transform_group="eval",
    )
    if include_indices:
        final_test_dataset = final_test_dataset.update_data_attribute(
            name="dataset_indices",
            new_value=DataAttribute(data=[i for i in range(len(final_test_dataset))],
                                    name="dataset_indices",
                                    use_in_getitem=True
                                    )
        )

    if val_dataset is None:
        final_val_dataset = val_dataset
    else:
        # if val_dataset is a dataset, then perform the same operations as for the train and test datasets
        if not isinstance(val_dataset, float):
            seq_val_dataset = as_supervised_classification_dataset(val_dataset)
            # set transformation groups
            final_val_dataset = as_supervised_classification_dataset(
                seq_val_dataset,
                transform_groups=transform_groups,
                initial_transform_group="eval",
            )
            if include_indices:
                final_val_dataset = final_val_dataset.update_data_attribute(
                    name="dataset_indices",
                    new_value=DataAttribute(data=[i for i in range(len(final_val_dataset))],
                                            name="dataset_indices",
                                            use_in_getitem=True
                                            )
                )
        else:
            # if val_dataset is a float, then apportion the respective fraction of training samples to validation
            if seed is not None:
                random.seed(seed)  # set random seed
            # all indexes in the original training dataset
            all_train_indxs: Set[int] = set([i for i in range(len(final_train_dataset))])
            val_indxs: Set[int] = set()
            targets = list(final_train_dataset.targets.val_to_idx.keys())
            # sort the targets to ensure consistent replication across different runs with the same seed
            targets.sort()
            for t in targets:
                indxs = final_train_dataset.targets.val_to_idx[t]
                n_samples = round(len(indxs) * val_dataset)  # round to the closest integer
                sampled_indxs = random.sample(indxs, n_samples)
                val_indxs.update(sampled_indxs)
            train_indxs = list(all_train_indxs - val_indxs)  # get the train indexes only
            train_indxs.sort()
            val_indxs: List[int] = list(val_indxs)
            val_indxs.sort()
            # `classification_subset` behaves just like `subset` but in addition it allows to set the initial transform
            # group to eval, which is the correct one for the validation dataset
            final_val_dataset = classification_subset(dataset=final_train_dataset, indices=val_indxs,
                                                      initial_transform_group="eval")
            final_train_dataset = final_train_dataset.subset(train_indxs)

    return NCScenario(
        train_dataset=final_train_dataset,
        test_dataset=final_test_dataset,
        n_experiences=n_experiences,
        task_labels=task_labels,
        val_dataset=final_val_dataset,
        shuffle=shuffle,
        seed=seed,
        fixed_class_order=fixed_class_order,
        per_experience_classes=per_exp_classes,
        class_ids_from_zero_from_first_exp=class_ids_from_zero_from_first_exp,
        class_ids_from_zero_in_each_exp=class_ids_from_zero_in_each_exp,
        reproducibility_data=reproducibility_data,
    )
