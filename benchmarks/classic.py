"""
This module includes a set of some mainstream benchmarks
"""
from __future__ import annotations
from typing import Union, Optional, Any, Callable, Tuple, Sequence, Dict, TYPE_CHECKING
from pathlib import Path

from benchmarks.datasets import ImageNetSampleCache

from avalanche.benchmarks.classic.cimagenet import _default_train_transform, _default_eval_transform
from avalanche.benchmarks.classic.ccifar100 import _default_cifar100_train_transform, _default_cifar100_eval_transform
from avalanche.benchmarks.classic.ctiny_imagenet import (_default_train_transform as _default_ctiny_imagenet_train_transform,
                                                         _default_eval_transform as _default_ctiny_imagenet_eval_transform)
from avalanche.benchmarks import nc_benchmark
from benchmarks.benchmark_generators import nc_benchmark as nc_benchmark_indices
from avalanche.benchmarks.datasets.external_datasets.cifar import get_cifar100_dataset
from avalanche.benchmarks.classic.ctiny_imagenet import _get_tiny_imagenet_dataset

if TYPE_CHECKING:
    from avalanche.benchmarks.scenarios.new_classes.nc_scenario import NCScenario


def SplitImageNetSampleCache(
    dataset_root: Union[str, Path],
    *,
    n_experiences=10,
    per_exp_classes=None,
    return_task_id=False,
    seed=0,
    fixed_class_order=None,
    shuffle: bool = True,
    class_ids_from_zero_in_each_exp: bool = False,
    class_ids_from_zero_from_first_exp: bool = False,
    train_cache_mode: bool = False,
    eval_cache_mode: bool = False,
    cached_train_transform: Optional[Any] = _default_train_transform,
    cached_eval_transform: Optional[Any] = _default_eval_transform,
    train_transform: Optional[Any] = None,
    eval_transform: Optional[Any] = None
) -> Tuple[NCScenario, ImageNetSampleCache, ImageNetSampleCache]:
    """
    Creates a CL benchmark using the ImageNet dataset.

    This function is identical to `avalanche.benchmarks.classic.SplitImageNet` but it constructs the train set and
    validation set of ImageNet using :class:`ImageNetSampleCache` rather
    than :class:`avalanche.benchmarks.datasets.ImageNet`. The only difference between the two classes is
    that :class:`ImageNetSampleCache` is equipped with caching functionalities. Apart from this, the two classes are
    identical. This function has more arguments than `avalanche.benchmarks.classic.SplitImageNet` to control the caching
    functionalities of the train and eval split of ImageNet.

    If the dataset is not present in the computer, **this method will NOT be
    able automatically download** and store it.

    The returned benchmark will return experiences containing all patterns of a
    subset of classes, which means that each class is only seen "once".
    This is one of the most common scenarios in the Continual Learning
    literature. Common names used in literature to describe this kind of
    scenario are "Class Incremental", "New Classes", etc. By default,
    an equal amount of classes will be assigned to each experience.

    This generator doesn't force a choice on the availability of task labels,
    a choice that is left to the user (see the `return_task_id` parameter for
    more info on task labels).

    The benchmark instance returned by this method will have two fields,
    `train_stream` and `test_stream`, which can be iterated to obtain
    training and test :class:`Experience`. Each Experience contains the
    `dataset` and the associated task label.

    The benchmark API is quite simple and is uniform across all benchmark
    generators. It is recommended to check the tutorial of the "benchmark" API,
    which contains usage examples ranging from "basic" to "advanced".

    :param dataset_root: Base path where Imagenet data is stored.
    :param n_experiences: The number of experiences in the current benchmark.
    :param per_exp_classes: Is not None, a dictionary whose keys are
        (0-indexed) experience IDs and their values are the number of classes
        to include in the respective experiences. The dictionary doesn't
        have to contain a key for each experience! All the remaining exps
        will contain an equal amount of the remaining classes. The
        remaining number of classes must be divisible without remainder
        by the remaining number of experiences. For instance,
        if you want to include 50 classes in the first experience
        while equally distributing remaining classes across remaining
        experiences, just pass the "{0: 50}" dictionary as the
        per_experience_classes parameter. Defaults to None.
    :param return_task_id: if True, a progressive task id is returned for every
        experience. If False, all experiences will have a task ID of 0.
    :param seed: A valid int used to initialize the random number generator.
        Can be None.
    :param fixed_class_order: A list of class IDs used to define the class
        order. If None, value of ``seed`` will be used to define the class
        order. If non-None, ``seed`` parameter will be ignored.
        Defaults to None.
    :param shuffle: If true, the class order in the incremental experiences is
        randomly shuffled. Default to True.
    :param class_ids_from_zero_in_each_exp: If True, original class IDs
        will be mapped to range [0, n_classes_in_exp) for each experience.
        Defaults to False. Mutually exclusive with the
        ``class_ids_from_zero_from_first_exp`` parameter.
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
    :param train_cache_mode: (optional) whether the samples retrieved when `__getitem__` is called must be cached in
        the train split. Default is False.
    :param eval_cache_mode: (optional) whether the samples retrieved when `__getitem__` is called must be cached in
        the eval split. Default is False.
    :param cached_train_transform: (optional) a function/transform applied in the train split that takes in an PIL
        image and returns a transformed version. This transformation is applied before caching a given sample.
        Therefore, a transformed version of the sample is cached if a transformation is provided. If None, a sample is
        cached without applying any transformation. If no cached transformation is provided, the default one is applied.
    :param cached_eval_transform: (optional) a function/transform applied in the eval split that takes in an PIL
        image and returns a transformed version. This transformation is applied before caching a given sample.
        Therefore, a transformed version of the sample is cached if a transformation is provided. If None, a sample is
        cached without applying any transformation. If no cached transformation is provided, the default one is applied.
    :param train_transform: a function/transform applied after the `cached_train_transform`. Unlike
        `cached_train_transform`, this transformation is not applied before caching a given sample. Therefore, this
        transformation is always applied whenever a given sample is retrieved. This transformation is useful
        when some randomised transformations must be applied and it is important that every time a sample is
        retrieved a different transformation is applied to it. If None, no transformation is applied. Default is None.
    :param eval_transform: a function/transform applied after the `cached_eval_transform`. Unlike
        `cached_eval_transform`, this transformation is not applied before caching a given sample. Therefore, this
        transformation is always applied whenever a given sample is retrieved. This transformation is useful
        when some randomised transformations must be applied and it is important that every time a sample is
        retrieved a different transformation is applied to it. If None, no transformation is applied. Default is None.

    :returns: a properly initialized :class:`NCScenario` instance, the underlying ImageNet train dataset,
        the underlying ImageNet eval dataset
    """

    train_set, test_set = _get_imagenetsamplecache_dataset(dataset_root, train_cache_mode=train_cache_mode,
                                                           eval_cache_mode=eval_cache_mode,
                                                           cached_train_transform=cached_train_transform,
                                                           cached_eval_transform=cached_eval_transform)

    return nc_benchmark(
        train_dataset=train_set,
        test_dataset=test_set,
        n_experiences=n_experiences,
        task_labels=return_task_id,
        per_exp_classes=per_exp_classes,
        seed=seed,
        fixed_class_order=fixed_class_order,
        shuffle=shuffle,
        class_ids_from_zero_in_each_exp=class_ids_from_zero_in_each_exp,
        class_ids_from_zero_from_first_exp=class_ids_from_zero_from_first_exp,
        train_transform=train_transform,
        eval_transform=eval_transform,
    ), train_set, test_set


def _get_imagenetsamplecache_dataset(root: str,
                                     train_cache_mode: bool = False,
                                     eval_cache_mode: bool = False,
                                     cached_train_transform: Optional[Callable] = None,
                                     cached_eval_transform: Optional[Callable] = None)\
        -> Tuple[ImageNetSampleCache, ImageNetSampleCache]:
    """
    Get the train and eval split of Imagenet.
    :param root: root directory of the ImageNet Dataset
    :param train_cache_mode: (optional) whether the samples retrieved when `__getitem__` is called must be cached in
        the train split. Default is False.
    :param eval_cache_mode: (optional) whether the samples retrieved when `__getitem__` is called must be cached in
        the eval split. Default is False.
    :param cached_train_transform: (optional) a function/transform applied in the train split that takes in an PIL
        image and returns a transformed version. This transformation is applied before caching a given sample.
        Therefore, a transformed version of the sample is cached if a transformation is provided. If None, a sample is
        cached without applying any transformation. Default is None.
    :param cached_eval_transform: (optional) a function/transform applied in the eval split that takes in an PIL
        image and returns a transformed version. This transformation is applied before caching a given sample.
        Therefore, a transformed version of the sample is cached if a transformation is provided. If None, a sample is
        cached without applying any transformation. Default is None.
    :return: train split, eval split
    """
    train_set = ImageNetSampleCache(root, split="train", cache_mode=train_cache_mode,
                                    cached_transform=cached_train_transform)
    test_set = ImageNetSampleCache(root, split="val", cache_mode=eval_cache_mode,
                                   cached_transform=cached_eval_transform)
    return train_set, test_set


def SplitCIFAR100(
    n_experiences: int,
    *,
    val_dataset: Optional[float] = None,
    first_exp_with_half_classes: bool = False,
    per_exp_classes: Optional[Dict[int, int]] = None,
    return_task_id=False,
    seed: Optional[int] = None,
    fixed_class_order: Optional[Sequence[int]] = None,
    shuffle: bool = True,
    class_ids_from_zero_in_each_exp: bool = False,
    class_ids_from_zero_from_first_exp: bool = False,
    train_transform: Optional[Any] = _default_cifar100_train_transform,
    eval_transform: Optional[Any] = _default_cifar100_eval_transform,
    include_indices: bool = False,
    dataset_root: Optional[Union[str, Path]] = None
):
    """
    This is identical to `avalanche.benchmarks.classic.SplitCIFAR100` but it has two additional functionalities:
        - it allows to include the indices of the samples in the dataset as DataAttribute. Read the doc of the
          `include_indices` parameter for more on this.
        - it allows to apportion a fraction of the training dataset to validation. Read the doc of the `val_dataset`
          parameter for more on this.
        - it allows to specify the number of classes to include in each experience. Read the doc of the
          `per_exp_classes` parameter for more on this.

    Creates a CL benchmark using the CIFAR100 dataset.

    If the dataset is not present in the computer, this method will
    automatically download and store it.

    The returned benchmark will return experiences containing all patterns of a
    subset of classes, which means that each class is only seen "once".
    This is one of the most common scenarios in the Continual Learning
    literature. Common names used in literature to describe this kind of
    scenario are "Class Incremental", "New Classes", etc. By default,
    an equal amount of classes will be assigned to each experience unless the `per_exp_classes` is specified or
    the `first_exp_with_half_classes` is set to True.

    This generator doesn't force a choice on the availability of task labels,
    a choice that is left to the user (see the `return_task_id` parameter for
    more info on task labels).

    The benchmark instance returned by this method will have two fields,
    `train_stream` and `test_stream`, which can be iterated to obtain
    training and test :class:`Experience`. Each Experience contains the
    `dataset` and the associated task label. If `val_dataset` is not None, then the benchmark instance will have an
    additional field `val_stream`, which can be iterated to obtain validation :class:`Experience`.

    The benchmark API is quite simple and is uniform across all benchmark
    generators. It is recommended to check the tutorial of the "benchmark" API,
    which contains usage examples ranging from "basic" to "advanced".

    :param n_experiences: The number of incremental experiences in the current
        benchmark. The value of this parameter should be a divisor of 100 if
        first_task_with_half_classes is False, a divisor of 50 otherwise.
    :param val_dataset: a float number between 0 and 1 (both excluded). It is used to apportion a fraction of the
        samples in the train dataset to validation (these samples are not used during training but only for validation).
        The fraction of samples apportioned to validation are selected randomly from each class independently, i.e. if
        the float number is 0.3, then for each class in the training dataset, 30% of its inputs are selected randomly.
        If None, no fraction of the samples in the train dataset are apportioned to validation, i.e. a validation stream
        is not added to the returned instance. Defaults to None.
    :param first_exp_with_half_classes: A boolean value that indicates if a
        first pretraining batch containing half of the classes should be used.
        If it's True, a pretraining experience with half of the classes (50 for
        cifar100) is used. If this parameter is False no pretraining task
        will be used, and the dataset is simply split into the number of
        experiences defined by the parameter n_experiences. Default to False.
    :param per_exp_classes: if not None, a dictionary whose keys are (0-indexed) experience IDs and their values are the
        number of classes to include in the respective experiences. The dictionary doesn't have to contain a key for
        each experience! All the remaining experiences will contain an equal amount of the remaining classes.
        The remaining number of classes must be divisible without remainder by the remaining number of experiences.
        For instance, if you want to include 50 classes in the first experience while equally distributing remaining
        classes across remaining experiences, just pass "{0: 50}". It is *only* considered when
        `first_exp_with_half_classes` is False. Defaults to None
    :param return_task_id: if True, a progressive task id is returned for every
        experience. If False, all experiences will have a task ID of 0.
    :param seed: A valid int used to initialize the random number generator before apportioning a fraction of training
        samples to validation (if `val_dataset` is not None) and before shuffling the class order in
        the incremental experiences (if `shuffle` is set to True). Note that the seed is set twice with the same value,
        immediately before creating the validation set (if `val_dataset` is not None) and immediately before shuffling
        the class order in the incremental experiences (if `shuffle` is set to True). If None, no random number
        generator seed is set, preventing results from being replicated. Default is None.
    :param fixed_class_order: A list of class IDs used to define the class
        order. If None, value of ``seed`` will be used to define the class
        order. If non-None, ``seed`` parameter will be ignored.
        Defaults to None.
    :param shuffle: If true, the class order in the incremental experiences is
        randomly shuffled. Default to True.
    :param class_ids_from_zero_in_each_exp: If True, original class IDs
        will be mapped to range [0, n_classes_in_exp) for each experience.
        Defaults to False. Mutually exclusive with the
        ``class_ids_from_zero_from_first_exp`` parameter.
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
    :param train_transform: The transformation to apply to the training data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation for a
        comprehensive list of possible transformations).
        If no transformation is passed, the default train transformation
        will be used.
    :param eval_transform: The transformation to apply to the test data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation for a
        comprehensive list of possible transformations).
        If no transformation is passed, the default test transformation
        will be used.
    :param include_indices: whether to include the indices of the samples in the train, test and (if `val_dataset` is
        not None) validation datasets as DataAttribute. The index of
        each sample is the index to be used to retrieve the given sample from the underlying dataset. The name of the
        DataAttribute is set to `dataset_indices` and its `use_in_getitem` attribute is set to True. Therefore, the
        index of a sample is returned when calling `__getitem__`. The index of each sample in the validation dataset
        refers to the index to be used to retrieve the given sample from the dataset underlying the train dataset.
    :param dataset_root: The root path of the dataset. Defaults to None, which
        means that the default location for 'cifar100' will be used.

    :returns: A properly initialized :class:`NCScenario` instance.
    """
    if val_dataset is not None:
        if isinstance(val_dataset, float):
            if not 0 < val_dataset < 1:
                raise ValueError("The val_dataset parameter must be between 0 and 1")
        else:
            raise ValueError("The val_dataset parameter must be a float")

    cifar_train, cifar_test = get_cifar100_dataset(dataset_root)

    return nc_benchmark_indices(
        train_dataset=cifar_train,
        test_dataset=cifar_test,
        n_experiences=n_experiences,
        task_labels=return_task_id,
        val_dataset=val_dataset,
        seed=seed,
        fixed_class_order=fixed_class_order,
        shuffle=shuffle,
        per_exp_classes={0: 50} if first_exp_with_half_classes else per_exp_classes,
        class_ids_from_zero_in_each_exp=class_ids_from_zero_in_each_exp,
        class_ids_from_zero_from_first_exp=class_ids_from_zero_from_first_exp,
        train_transform=train_transform,
        eval_transform=eval_transform,
        include_indices=include_indices
    )


def SplitTinyImageNet(
    n_experiences: int,
    *,
    val_dataset: Optional[float] = None,
    first_exp_with_half_classes: bool = False,
    per_exp_classes: Optional[Dict[int, int]] = None,
    return_task_id = False,
    seed: Optional[int] = None,
    fixed_class_order: Optional[Sequence[int]] = None,
    shuffle: bool = True,
    class_ids_from_zero_in_each_exp: bool = False,
    class_ids_from_zero_from_first_exp: bool = False,
    train_transform: Optional[Any] = _default_ctiny_imagenet_train_transform,
    eval_transform: Optional[Any] = _default_ctiny_imagenet_eval_transform,
    include_indices: bool = False,
    dataset_root: Optional[Union[str, Path]] = None
):
    """
    This is identical to `avalanche.benchmarks.classic.SplitTinyImageNet` but it has several additional functionalities.
    Some of them are:
        - it allows to include the indices of the samples in the dataset as DataAttribute. Read the doc of the
          `include_indices` parameter for more on this.
        - it allows to apportion a fraction of the training dataset to validation. Read the doc of the `val_dataset`
          parameter for more on this.
        - it allows to specify the number of classes to include in each experience. Read the doc of the
          `per_exp_classes` parameter for more on this.

    Creates a CL benchmark using the Tiny ImageNet dataset.

    If the dataset is not present in the computer, this method will
    automatically download and store it.

    The returned benchmark will return experiences containing all patterns of a
    subset of classes, which means that each class is only seen "once".
    This is one of the most common scenarios in the Continual Learning
    literature. Common names used in literature to describe this kind of
    scenario are "Class Incremental", "New Classes", etc. By default,
    an equal amount of classes will be assigned to each experience unless the `per_exp_classes` is specified or
    the `first_exp_with_half_classes` is set to True.

    This generator doesn't force a choice on the availability of task labels,
    a choice that is left to the user (see the `return_task_id` parameter for
    more info on task labels).

    The benchmark instance returned by this method will have two fields,
    `train_stream` and `test_stream`, which can be iterated to obtain
    training and test :class:`Experience`. Each Experience contains the
    `dataset` and the associated task label. If `val_dataset` is not None, then the benchmark instance will have an
    additional field `val_stream`, which can be iterated to obtain validation :class:`Experience`.

    The benchmark API is quite simple and is uniform across all benchmark
    generators. It is recommended to check the tutorial of the "benchmark" API,
    which contains usage examples ranging from "basic" to "advanced".

    :param n_experiences: The number of incremental experiences in the current
        benchmark. The value of this parameter should be a divisor of 200 if
        first_task_with_half_classes is False, a divisor of 100 otherwise.
    :param val_dataset: a float number between 0 and 1 (both excluded). It is used to apportion a fraction of the
        samples in the train dataset to validation (these samples are not used during training but only for validation).
        The fraction of samples apportioned to validation are selected randomly from each class independently, i.e. if
        the float number is 0.3, then for each class in the training dataset, 30% of its inputs are selected randomly.
        If None, no fraction of the samples in the train dataset are apportioned to validation, i.e. a validation stream
        is not added to the returned instance. Defaults to None.
    :param first_exp_with_half_classes: A boolean value that indicates if a
        first pretraining batch containing half of the classes should be used.
        If it's True, a pretraining experience with half of the classes (100 for
        tiny imagenet) is used. If this parameter is False no pretraining task
        will be used, and the dataset is simply split into the number of
        experiences defined by the parameter n_experiences. Default to False.
    :param per_exp_classes: if not None, a dictionary whose keys are (0-indexed) experience IDs and their values are the
        number of classes to include in the respective experiences. The dictionary doesn't have to contain a key for
        each experience! All the remaining experiences will contain an equal amount of the remaining classes.
        The remaining number of classes must be divisible without remainder by the remaining number of experiences.
        For instance, if you want to include 50 classes in the first experience while equally distributing remaining
        classes across remaining experiences, just pass "{0: 50}". It is *only* considered when
        `first_exp_with_half_classes` is False. Defaults to None
    :param return_task_id: if True, a progressive task id is returned for every
        experience. If False, all experiences will have a task ID of 0.
    :param seed: A valid int used to initialize the random number generator before apportioning a fraction of training
        samples to validation (if `val_dataset` is not None) and before shuffling the class order in
        the incremental experiences (if `shuffle` is set to True). Note that the seed is set twice with the same value,
        immediately before creating the validation set (if `val_dataset` is not None) and immediately before shuffling
        the class order in the incremental experiences (if `shuffle` is set to True). If None, no random number
        generator seed is set, preventing results from being replicated. Default is None.
    :param fixed_class_order: A list of class IDs used to define the class
        order. If None, value of ``seed`` will be used to define the class
        order. If non-None, ``seed`` parameter will be ignored.
        Defaults to None.
    :param shuffle: If true, the class order in the incremental experiences is
        randomly shuffled. Default to True.
    :param class_ids_from_zero_in_each_exp: If True, original class IDs
        will be mapped to range [0, n_classes_in_exp) for each experience.
        Defaults to False. Mutually exclusive with the
        ``class_ids_from_zero_from_first_exp`` parameter.
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
    :param train_transform: The transformation to apply to the training data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation for a
        comprehensive list of possible transformations).
        If no transformation is passed, the default train transformation
        will be used.
    :param eval_transform: The transformation to apply to the test data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation for a
        comprehensive list of possible transformations).
        If no transformation is passed, the default test transformation
        will be used.
    :param include_indices: whether to include the indices of the samples in the train, test and (if `val_dataset` is
        not None) validation datasets as DataAttribute. The index of
        each sample is the index to be used to retrieve the given sample from the underlying dataset. The name of the
        DataAttribute is set to `dataset_indices` and its `use_in_getitem` attribute is set to True. Therefore, the
        index of a sample is returned when calling `__getitem__`. The index of each sample in the validation dataset
        refers to the index to be used to retrieve the given sample from the dataset underlying the train dataset.
    :param dataset_root: The root path of the dataset. Defaults to None, which
        means that the default location for 'tinyimagenet' will be used.

    :returns: A properly initialized :class:`NCScenario` instance.
    """
    if val_dataset is not None:
        if isinstance(val_dataset, float):
            if not 0 < val_dataset < 1:
                raise ValueError("The val_dataset parameter must be between 0 and 1")
        else:
            raise ValueError("The val_dataset parameter must be a float")

    train_set, test_set = _get_tiny_imagenet_dataset(dataset_root)

    return nc_benchmark_indices(
        train_dataset=train_set,
        test_dataset=test_set,
        n_experiences=n_experiences,
        task_labels=return_task_id,
        val_dataset=val_dataset,
        seed=seed,
        fixed_class_order=fixed_class_order,
        shuffle=shuffle,
        per_exp_classes={0: 100} if first_exp_with_half_classes else per_exp_classes,
        class_ids_from_zero_in_each_exp=class_ids_from_zero_in_each_exp,
        class_ids_from_zero_from_first_exp=class_ids_from_zero_from_first_exp,
        train_transform=train_transform,
        eval_transform=eval_transform,
        include_indices=include_indices
    )

