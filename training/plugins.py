"""
This module includes a set of plugins
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from abc import ABC
from typing import Optional, Literal, Dict, Tuple, List, Iterable, Set, Union, Type, Callable, FrozenSet, TYPE_CHECKING
from training.storage_policy import (BufferDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
                                     ClassBalancedBufferDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate)
from training.subset_generator import SubsetGeneratorGradientBasedReplayBufferSelection, RandomSubsetGenerator
from metrics.gradient import (GradientLossWRTInputLastLayer, GradientLossWRTWeightsBiasesClassificationHead,
                              GradientCrossEntropyLossMSELossWRTWeightsBiasesClassificationHead)
from metrics.forgetting import ForgettingEvents
from metrics.list_accumulator import ListAccumulator
from utils import balance_dataset
from benchmarks.datasets import SampleCacheMixin
from benchmarks.utils.data_loader import MultiDatasetBalancedDataLoader
from coreset.coreset import CRAIGCoreset
from training.logits_storage import LogitsStorageDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate
from training.prob_vecs_storage import ProbVecsStorageDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate
from training.loss_functions import CrossEntropyLossMSELossLogits
import training.subset_generator as sg

from avalanche.core import SupervisedPlugin, Template, CallbackResult
from avalanche.benchmarks.utils import concat_datasets
from avalanche.training.plugins import EvaluationPlugin
from avalanche.benchmarks.utils.data_loader import collate_from_data_or_kwargs
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler
from sklearn.metrics import pairwise_distances
import time
import copy
import torch
import torch.nn as nn
import numpy as np
import random

if TYPE_CHECKING:
    from training.templates import DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate
    from training.storage_policy import BufferDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate
    from avalanche.benchmarks.utils import ClassificationDataset, AvalancheDataset
    from models.dynamic_networks import (IncrementalClassifierOutlierDetectorWithInitFeatureExtractor,
                                         DynamicNetworkWithInitFeatureExtractorAndIncrementalOutlierSubNetworks)
    from avalanche.benchmarks.scenarios import CLExperience
    from torch.optim import Optimizer


class DynamicNetworkInitFeatureExtractorIncrementalOutlierPlugin(SupervisedPlugin, ABC):
    """
    Abstract class for plugins for the template
    :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
    """

    def __init__(self):
        super().__init__()

    def before_train_datasets_adaptation(self, strategy: Template, *args, **kwargs) -> CallbackResult:
        """
        Called before `train_datasets_adaptation` by the
        :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
        """
        pass

    def after_train_datasets_adaptation(self, strategy: Template, *args, **kwargs) -> CallbackResult:
        """
        Called after `train_datasets_adaptation` by the template
        :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
        """
        pass

    def before_train_dataloader(self, strategy: Template, *args, **kwargs) -> CallbackResult:
        """
        Called before `make_train_dataloader` by the template
        :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
        """
        pass

    def after_train_dataloader(self, strategy: Template, *args, **kwargs) -> CallbackResult:
        """
        Called after `make_train_dataloader` by the template
        :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
        """
        pass

    def before_training_subnetwork(self, strategy: Template, *args, **kwargs) -> CallbackResult:
        """
        Called before the training of the current subnetwork by the template
        :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
        """
        pass

    def after_training_subnetwork(self, strategy: Template, *args, **kwargs) -> CallbackResult:
        """
        Called after the training of the current subnetwork by the template
        :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
        """
        pass


class ReplayPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate(
    DynamicNetworkInitFeatureExtractorIncrementalOutlierPlugin, supports_distributed=True):
    """
    Experience replay plugin for :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`.

    Handles an external memory buffer with a specific storage policy.
    It implements `before_training_exp` and `after_training_exp` callbacks.

    The `before_training_exp` callback is implemented in order to create a dataloader for each subnetwork in the model
    :class:`DynamicNetworkWithInitFeatureExtractorAndIncrementalOutlierSubNetworks` stored
    inside :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` before training on a new experience.
    The dataloader of each subnetwork balances the data in each mini-batch from two datasets: the dataset of the
    subnetwork and a concatenation of all other subnetworks' datasets. The first dataset is used for training the
    incremental classifier of a subnetwork while both of them are used for the training of the outlier detector.
    The dataset of a subnetwork is defined as the concatenation of the dataset of classes a
    subnetwork must be trained on in the new experience (if any) and its rehearsal buffer (if it exists).
    At least one of these two *must* exist before training on a new experience.

    The `after_training_exp` callback is implemented in order to add new patterns to the external memory buffer with
    a specific storage policy after training on a new experience.

    .note::
        All subnetworks are trained on each experience because even if there is no new classes that a subnetwork must be
        trained on, it can be trained on the data of its rehearsal buffer and all the other subnetwork datasets, which
        include the new data in the new experience.

    .note::
        This plugin allows balancing the dataset of each subnetwork that is used for training the respective
        incremental classifier. The dataset of a subnetwork that is used for training the respective incremental
        classifier comprises the samples of classes in the new experience that the subnetwork must be trained on and the
        rehearsal buffer of the subnetwork. In case of balancing, this dataset is balanced by ensuring that each class
        has an identical number of samples, matching the number of samples of the class with the fewest samples in the
        original dataset.

    .warning::
        The storage policy must be a subclass
        of :class:`BufferDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
    """

    def __init__(self, storage_policy: BufferDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
                 batch_size: Optional[int] = None, balance_subnetwork_dataset: bool = False):
        """
        Create a new ReplayPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate
        :param storage_policy: the policy that controls how to add new exemplars in the external memory buffer. It *must*
            be a subclass of :class:`BufferDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
        :param batch_size: (optional) the size of the training data batch. If set to `None`, it will be set equal to
            the strategy's training batch size.
        :param balance_subnetwork_dataset: (optional) flag that determines whether the dataset of each subnetwork
            that is used for training the respective incremental classifier needs to be balanced. The dataset of a
            subnetwork that is used for training the respective incremental classifier comprises the samples of classes
            in the new experience that the subnetwork must be trained on and the rehearsal buffer of the subnetwork.
            If True,the dataset of each subnetwork that is used for training the respective incremental classifier is
            balanced by ensuring that each class has an identical number of samples, matching the number of samples of
            the class with the fewest samples in the original dataset. Default is False.
        """
        super().__init__()
        self.storage_policy: BufferDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate = storage_policy
        """storage policy that controls how new patterns are added to the rehearsal buffer"""

        self.batch_size: Optional[int] = batch_size
        """the size of the training data batch"""

        self.balance_subnetwork_dataset: bool = balance_subnetwork_dataset
        """
        flag that determines whether the dataset of each subnetwork that is used for training the
        respective incremental classifier needs to be balanced
        """

    def before_training_exp(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
                            num_workers=0, shuffle=True, pin_memory=None, persistent_workers=False, drop_last=False,
                            termination_dataset=-2, oversample_small_datasets=False, balance_large_datasets=True,
                            **kwargs):
        """
        Callback that sets the dataloader of each subnetwork in the model
        :class:`DynamicNetworkWithInitFeatureExtractorAndIncrementalOutlierSubNetworks` stored inside the strategy
        before training on a new experience.
        The dataloader of each subnetwork balances the data in each mini-batch from two datasets: the dataset of the
        subnetwork and a concatenation of all other subnetworks' datasets.

        The dataset of a subnetwork is a concatenation of the dataset of classes a subnetwork must be trained on in the
        new experience (if any) and its rehearsal buffer (if it exists). At least one of these two
        *must* exist before training on a new experience. Otherwise, a RuntimeError is raised.

        If `balance_subnetwork_dataset` is True, the dataset of each subnetwork is balanced.

        .note::
            If there is no rehearsal data in the buffer then the dataloader is not changed by this plugin.
        :param strategy: a :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` strategy
        :param num_workers: (optional) number of thread workers for the data loading.
        :param shuffle: (optional) True if the data should be shuffled, False otherwise. Default is True.
        :param pin_memory: (optional) If True, the data loader will copy Tensors into CUDA
            pinned memory before returning them. Defaults to True.
        :param persistent_workers: (optional) If True, the data loader will not shut down the worker processes after a
            dataset has been consumed once. This allows to maintain the workers Dataset instances alive.
            Defaults to False.
        :param drop_last: (optional) set to True to drop the last incomplete batch, if the dataset size is not divisible
            by the batch size. If False and the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. Defaults to False
        :param termination_dataset: (optional) an integer number denoting the index of the dataset to be used for
            determining when to stop iterating (the iteration is stopped when the end of the respective dataset is hit).
            0 for stopping iterating when the end of the dataset the respective subnetwork must be trained on is hit.
            1 for stopping iterating when the end of the concatenation of all the other subnetworks' datasets is hit.
            -1 for stopping iterating when the end of the larger dataset is hit.
            -2 for stopping iterating when the end of the smaller dataset is hit.
            Default is -2.
        :param oversample_small_datasets: (optional) if True, the smaller dataset is oversampled to match the
            `termination_dataset`. Otherwise, once data from a dataset is completely iterated, the dataset will be
            skipped in the subsequent mini-batches. Default is False.
        :param balance_large_datasets: (optional) if True, the larger dataset is randomly reduced in size to match the
            `termination_dataset` while ensuring that each class has an identical or as similar as possible
            number of samples. Note that if True, the larger dataset is randomly reduced in size every
            time `__iter__` is called. Therefore, a different smaller dataset is created out of the larger dataset
            in each epoch. If False, the larger dataset is not reduced in size and the
            samples used for it in each dataloader iteration depends on the argument `shuffle`. If `shuffle` is False
            and `balance_large_datasets` is False, the same samples of the larger dataset are used for each dataloader
            iteration. Default is True.
        :param kwargs:
        :return: None
        """
        # if there is no rehearsal data in the buffer then the dataloader is not changed by this plugin. Therefore,
        # if `make_train_dataloader` is not overridden, `strategy.dataloader` is computed as described in
        # `make_train_dataloader` of DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate
        if len(self.storage_policy.buffer) > 0:
            if strategy.adapted_dataset is None:
                raise RuntimeError("It is impossible to build train dataloaders if the adapted dataset is None")

            batch_size = self.batch_size
            if batch_size is None:  # if batch size is None, then the batch size is set to `strategy.train_mb_size`
                batch_size = strategy.train_mb_size

            other_dataloader_args = strategy.obtain_common_dataloader_parameters(batch_size=batch_size,
                                                                                 num_workers=num_workers,
                                                                                 shuffle=shuffle,
                                                                                 pin_memory=pin_memory,
                                                                                 persistent_workers=persistent_workers,
                                                                                 drop_last=drop_last)
            dataloader = {}  # holds the dataloader of each subnetwork
            sub_datasets = {}  # holds the dataset of each subnetwork (excluding the outlier dataset)
            # this dummy empty dataset is used to preserve the class of the datasets in `strategy.adapted_dataset` when
            # performing the following concatenation operations
            dummy_empty_dataset = (list(strategy.adapted_dataset.values())[0]).subset([])

            # looping over all subnetworks to build their datasets
            for sub_id in strategy.model.subnetworks.keys():
                curr_dataset = strategy.adapted_dataset[sub_id] if sub_id in strategy.adapted_dataset.keys() \
                    else dummy_empty_dataset
                # it is an empty dataset if there is no rehearsal buffer for a given subnetwork
                rehearsal_dataset = self.storage_policy.buffer_subnetwork(strategy, sub_id)
                # if the rehearsal dataset is empty, replace it with the empy dummy dataset
                if len(rehearsal_dataset) == 0:
                    rehearsal_dataset = dummy_empty_dataset
                dataset = curr_dataset.concat(rehearsal_dataset)
                if len(dataset) == 0:
                    raise RuntimeError("No new data or rehearsal data for a given subnetwork. However, at the start of "
                                       "a new experience, there must be either new data or "
                                       "rehearsal data for a given subnetwork.")
                # if true, balance `dataset` by ensuring each class has the same number of samples
                if self.balance_subnetwork_dataset:
                    dataset = balance_dataset(dataset)
                sub_datasets[sub_id] = dataset

            # looping over all subnetworks to build their dataloaders
            for sub_id in strategy.model.subnetworks.keys():
                dataset = sub_datasets[sub_id]
                outlier_dataset = concat_datasets([sub_dataset for sub_id2, sub_dataset in sub_datasets.items()
                                                   if sub_id2 != sub_id])
                if len(outlier_dataset) > 0:
                    dataloader[sub_id] = MultiDatasetBalancedDataLoader(datasets=[dataset, outlier_dataset],
                                                                        termination_dataset=termination_dataset,
                                                                        oversample_small_datasets=oversample_small_datasets,
                                                                        balance_large_datasets=balance_large_datasets,
                                                                        distributed_sampling=False,
                                                                        **other_dataloader_args)
                # the outlier dataset can be empty only if there is one subnetwork in the model: the one being trained
                else:
                    collate_from_data_or_kwargs(dataset, other_dataloader_args)
                    dataloader[sub_id] = DataLoader(dataset, **other_dataloader_args)

            strategy.dataloader = dataloader

    def after_training_exp(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate, **kwargs):
        """
        Callback that updates the rehearsal buffer after training on a new experience with new patterns using the
        storage policy stored in `storage_policy`.
        :param strategy: a :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` strategy
        :param kwargs:
        :return: None
        """
        self.storage_policy.update(strategy, **kwargs)


class ReplayNoOutlierPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate(
    DynamicNetworkInitFeatureExtractorIncrementalOutlierPlugin, supports_distributed=True):
    """
    Experience replay plugin for :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`.

    This plugin differs from :class:`ReplayPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` because
    it should be used only when there is one or multiple subnetworks whose outlier detector network is just a dummy
    network of no use. Simply put, this plugin should be used when a subnetwork must be trained only on the classes it
    is assigned to and does not have to be trained on the classes the other subnetwork are trained on.

    This plugin handles an external memory buffer with a specific storage policy.

    It implements `before_training_exp` and `after_training_exp` callbacks.

    The `before_training_exp` callback is implemented in order to create a dataloader for each subnetwork in the model
    :class:`DynamicNetworkWithInitFeatureExtractorAndIncrementalOutlierSubNetworks` stored
    inside :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` that undertakes training a new
    experience. The dataloader of each subnetwork that undertakes training on the new experience balances the data in
    each mini-batch from two datasets: the dataset of classes the subnetwork must be trained on in the new experience
    and its rehearsal buffer (if any).

    The `after_training_exp` callback is implemented in order to add new patterns to the external memory buffer with
    a specific storage policy after training on a new experience.

    .note::
        This plugin totally overrides the dataloaders stored in the `dataloader` dictionary attribute computed by the
        `make_train_dataloader` method of :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`. The
        `make_train_dataloader` method constructs the dataloaders by balancing between the dataset a subnetwork must
        be trained on and the concatenation of all the other subnetworks' datasets, while this plugin balances the data
        in each mini-batch from two datasets: the dataset of classes a subnetwork must be trained on in the new
        experience and its rehearsal buffer (if any).

    .note::
        A dataloader is created **only** for those subnetworks that undertake training on the current experience (there
        is a dataset of classes a given subnetwork must be trained on in the current experience).

    .warning::
        The storage policy must be a subclass
        of :class:`BufferDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
    """
    def __init__(self, storage_policy: BufferDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
                 batch_size: Optional[int] = None):
        """
        Create a new ReplayNoOutlierPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate
        :param storage_policy: the policy that controls how to add new exemplars in the external memory buffer.
            It *must* be a subclass of :class:`BufferDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
        :param batch_size: (optional) the size of the training data batch. If set to `None`, it will be set equal to
            the strategy's training batch size.
        """
        super().__init__()
        self.storage_policy: BufferDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate = storage_policy
        """storage policy that controls how new patterns are added to the rehearsal buffer"""

        self.batch_size: Optional[int] = batch_size
        """the size of the training data batch"""

    def before_training_exp(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
                            num_workers=0, shuffle=True, pin_memory=None, persistent_workers=False, drop_last=False,
                            termination_dataset=0, oversample_small_datasets=True, balance_large_datasets=False,
                            **kwargs):
        """
        Callback that sets the dataloader of each subnetwork in the model
        :class:`DynamicNetworkWithInitFeatureExtractorAndIncrementalOutlierSubNetworks` stored inside the strategy
        that undertakes training a new experience.

        The dataloader of each subnetwork that undertakes training on the new experience balances the data in
        each mini-batch from two datasets: the dataset of classes the subnetwork must be trained on in the new
        experience and its rehearsal buffer (if any).

        :param strategy: a :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` strategy
        :param num_workers: (optional) number of thread workers for the data loading.
        :param shuffle: (optional) True if the data should be shuffled, False otherwise. Default is True.
        :param pin_memory: (optional) If True, the data loader will copy Tensors into CUDA
            pinned memory before returning them. Defaults to True.
        :param persistent_workers: (optional) If True, the data loader will not shut down the worker processes after a
            dataset has been consumed once. This allows to maintain the workers Dataset instances alive.
            Defaults to False.
        :param drop_last: (optional) set to True to drop the last incomplete batch, if the dataset size is not divisible
            by the batch size. If False and the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. Defaults to False
        :param termination_dataset: (optional) an integer number denoting the index of the dataset to be used for
            determining when to stop iterating (the iteration is stopped when the end of the respective dataset is hit).
            0 for stopping iterating when the end of the dataset of classes a subnetwork must be trained on in the new
            experience is hit.
            1 for stopping iterating when the end of the rehearsal buffer is hit.
            -1 for stopping iterating when the end of the larger dataset is hit.
            -2 for stopping iterating when the end of the smaller dataset is hit.
            Default is 0.
        :param oversample_small_datasets: (optional) if True, the smaller dataset is oversampled to match the
            `termination_dataset`. Otherwise, once data from a dataset is completely iterated, the dataset will be
            skipped in the subsequent mini-batches. Default is True.
        :param balance_large_datasets: (optional) if True, the larger dataset is randomly reduced in size to match the
            `termination_dataset` while ensuring that each class has an identical or as similar as possible
            number of samples. Note that if True, the larger dataset is randomly reduced in size every
            time `__iter__` is called. Therefore, a different smaller dataset is created out of the larger dataset
            in each epoch. If False, the larger dataset is not reduced in size and the
            samples used for it in each dataloader iteration depends on the argument `shuffle`. If `shuffle` is False
            and `balance_large_datasets` is False, the same samples of the larger dataset are used for each dataloader
            iteration. Default is False.
        :param kwargs: some keyword arguments
        """
        if strategy.adapted_dataset is None:
            raise RuntimeError("It is impossible to build train dataloaders if the adapted dataset is None")

        batch_size = self.batch_size
        if batch_size is None:  # if batch size is None, then the batch size is set to `strategy.train_mb_size`
            batch_size = strategy.train_mb_size

        other_dataloader_args = strategy.obtain_common_dataloader_parameters(batch_size=batch_size,
                                                                             num_workers=num_workers,
                                                                             shuffle=shuffle,
                                                                             pin_memory=pin_memory,
                                                                             persistent_workers=persistent_workers,
                                                                             drop_last=drop_last)
        dataloader = {}  # holds the dataloader of each subnetwork
        # strategy.adapted_dataset is a dictionary containing the subnetwork IDs as keys and the respective avalanche
        # datasets as values.The dataset of each subnetwork has the `targets_task_labels` DataAttribute set to the
        # respective subnetworkID.
        for sub_id, dataset in strategy.adapted_dataset.items():
            # it is an empty dataset if there is no rehearsal buffer for a given subnetwork
            rehearsal_dataset = self.storage_policy.buffer_subnetwork(strategy, sub_id)
            if len(rehearsal_dataset) == 0:  # if empty
                collate_from_data_or_kwargs(dataset, other_dataloader_args)
                dataloader[sub_id] = DataLoader(dataset, **other_dataloader_args)
            else:
                dataloader[sub_id] = MultiDatasetBalancedDataLoader(datasets=[dataset, rehearsal_dataset],
                                                                    termination_dataset=termination_dataset,
                                                                    oversample_small_datasets=oversample_small_datasets,
                                                                    balance_large_datasets=balance_large_datasets,
                                                                    distributed_sampling=False,
                                                                    **other_dataloader_args)
        strategy.dataloader = dataloader

    def after_training_exp(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate, **kwargs):
        """
        Callback that updates the rehearsal buffer after training on a new experience with new patterns using the
        storage policy stored in `storage_policy`.
        :param strategy: a :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` strategy
        :param kwargs: some keyword arguments
        """
        self.storage_policy.update(strategy, **kwargs)


class SubsampleTrainDatasetPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate(
    DynamicNetworkInitFeatureExtractorIncrementalOutlierPlugin, supports_distributed=True):
    """
    Subsample train dataset plugin for :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`.
    It implements the `after_train_dataset_adaptation` callback.

    The `after_train_dataset_adaptation` callback is called after the `adapted_dataset` attribute of the underlying
    strategy is initialised with the train dataset of the current experience. It randomly selects a specific
    fraction of samples for each class and only keeps those in the `adapted_dataset`, or only keeps the samples with a
    given "dataset_indices" DataAttribute in the `adapted_dataset`. This subsampling procedure is carried out so that
    the subsequent dataloaders built for each subnetwork only contain these specific samples rather than all of the
    samples in the original train dataset.

    .note::
        The operations of this plugin do not modify the original train dataset instance of the current experience
        in-place. Therefore, such an instance contains all the original samples.

    .note::
        This plugin *assumes* that the "dataset_indices" DataAttribute is present in the datasets of the training
        experiences.

    .note::
        When using :class:` ReplayPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
        or :class:`ReplayNoOutlierPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`, the replay buffer
        will be populated using only the samples kept by this plugin.
    """
    def __init__(self, fraction: Union[float, Iterable[int]], seed: Optional[int] = None,
                 save_kept_dataset_indices: bool = False):
        """
        Create a new SubsampleTrainDatasetPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate
        :param fraction: the fraction of samples to keep for each class or an iterable of the "dataset_indices"
            DataAttributes for the samples to keep. If a fraction, it must be a float number between 0 (excluded)
            and 1 (included). If a fraction, the samples to keep are selected randomly.
        :param seed: a valid int used to initialize the random number generator before selecting a fraction of samples
            to keep for each class. Setting the random number generator seed does not affect other random operations
            because the state of the random generator is saved prior to setting the random seed and it is reset back
            once the random subsampling operation is performed. If None, no random number generator seed is set,
            preventing results from being replicated. Only used when `fraction` is not an iterable. Default is None.
        :param save_kept_dataset_indices: whether to save the `dataset_indices` DataAttribute of the samples kept.
            Default is False.
        """
        if isinstance(fraction, float):
            if not 0 < fraction <= 1:
                raise ValueError("`fraction` must be between 0 (excluded) and 1 (included)")

        super().__init__()

        self._fraction: Union[float, Iterable[int]] = fraction
        """
        the fraction of samples to keep for each class or an iterable of the "dataset_indices" DataAttributes for the
        samples to keep. If a fraction, it is a float number between 0 (excluded) and 1 (included). If a fraction, the
        samples to keep are selected randomly.
        """

        self._seed: Optional[int] = seed
        """
        a valid int used to initialize the random number generator before selecting a fraction of samples to keep for
        each class. Setting the random number generator seed does not affect other random operations because the state
        of the random generator is saved prior to setting the random seed and it is reset back once the random
        subsampling operation is performed. If None, no random number generator seed is set, preventing results from
        being replicated. Only used when `self._fraction` is not an iterable.
        """

        self._save_kept_dataset_indices: bool = save_kept_dataset_indices
        """
        whether to save the `dataset_indices` DataAttribute of the samples kept.
        """

        self.samples_kept_dataset_indices: List[List[int]] = []
        """
        if `self._save_kept_dataset_indices` is True, it is a list of lists where the ith list contains the 
        `dataset_indices` DataAttribute of the samples kept for the ith training experience. Otherwise, it is just an
        empty list. 
        """

    def after_train_dataset_adaptation(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
                                       **kwargs):
        """
        Callback called after the `adapted_dataset` attribute of the underlying strategy is initialised with the train
        dataset of the current experience. It randomly selects a specific fraction of samples for each class and only
        keeps those in the `adapted_dataset` if `self._fraction` is a float. Otherwise, it only keeps those samples with
        a "dataset_indices" DataAttribute contained in `self._fraction`. This subsampling procedure is carried out so
        that the subsequent dataloaders built for each subnetwork only contain these specific samples rather than all
        of the samples in the original train dataset.

        .note::
            It does not modify the original train dataset instance of the current experience in-place. Therefore,
            such an instance contains all the original samples.
        .note
            It raises a :class:`RuntimeError` if the `dataset_indices` DataAttribute is not present in the dataset of
            the current training experience
        :param strategy: a strategy
        :param kwargs: some keyword arguments
        """
        if "dataset_indices" not in strategy.use_in_getitem_indices.keys():
            raise RuntimeError("The dataset of the current training experience must have the `dataset_indices` "
                               "DataAttribute")

        if not isinstance(self._fraction, float):
            # create a subset of `strategy.adapted_dataset` only containing the selected samples
            strategy.adapted_dataset = strategy.adapted_dataset.subset(
                [i for i, d_index in enumerate(strategy.adapted_dataset.dataset_indices) if d_index in self._fraction])
        else:
            if self._seed is not None:
                saved_state = random.getstate()  # save the current state of the random generator
                random.seed(self._seed)  # set random seed

            # Get sample idxs per class
            cl_idxs: Dict[int, List[int]] = defaultdict(list)
            for idx, target in enumerate(strategy.adapted_dataset.targets):
                # Conversion to int may fix issues when target
                # is a single-element torch.tensor
                target = int(target)
                cl_idxs[target].append(idx)

            for t, t_indxs in cl_idxs.items():
                n_samples = round(len(t_indxs) * self._fraction)  # round to the closest integer
                sampled_indxs = random.sample(t_indxs, n_samples)
                t_indxs[:] = sampled_indxs

            kept_indxs = list(itertools.chain(*list(cl_idxs.values())))  # the indexes of the samples to be kept
            kept_indxs.sort()  # sort the list of the indexes of the samples to be kept
            # create a subset of `strategy.adapted_dataset` only containing the selected samples
            strategy.adapted_dataset = strategy.adapted_dataset.subset(kept_indxs)

            if self._seed is not None:
                random.setstate(saved_state)  # restore the original state

        if self._save_kept_dataset_indices:
            self.samples_kept_dataset_indices.append(list(strategy.adapted_dataset.dataset_indices))


class BalanceDataloaderEpochPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate(
    DynamicNetworkInitFeatureExtractorIncrementalOutlierPlugin, supports_distributed=True):
    """
    Balance dataloader epoch plugin for :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`.
    It implements the `after_training_epoch` callback.

    At the end of a specific training epoch, this plugin changes the dataloader of the subnetwork currently being
    trained. It creates a balanced dataset out of the dataset of the subnetwork currently being trained and creates a
    dataloader out of it. In the subsequent epochs, the subnetwork currently being trained will be trained on this
    new dataloader containing the balanced dataset and the concatenation of all other subnetworks' datasets.

    The dataset of a subnetwork is defined as the concatenation of the dataset of
    classes a subnetwork must be trained on in the new experience (if any) and its rehearsal buffer (if it exists).

    This plugin balances the dataset of the subnetwork currently being trained by ensuring that each class has an
    identical number of samples, matching the number of samples of the class with the fewest samples in the original
    dataset. The samples of each class to filter out are picked randomly.

    .note::
        The dataset of classes a subnetwork must be trained on in the new experience is retrieved from the attribute
        `adapted_dataset` of :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`.

    warning::
        This plugin does not change the dataloaders stored in the attribute `dataloader` of
        :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`. It only replaces the old dataloader
        stored in `curr_train_dataloader` with the new dataloader. The `curr_train_dataloader` attribute is used in
        each epoch to retrieve mini-batches for the subnetwork currently being trained.
    """

    def __init__(self, epoch: int,
                 storage_policy: BufferDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
                 batch_size: Optional[int] = None):
        """
        Create a new BalanceDataloaderEpochPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate
        :param epoch: epoch number. At the end of this epoch, this plugin changes the dataloader of the subnetwork
            currently being trained. It creates a balanced dataset out of the dataset of the subnetwork currently being
            trained and creates a dataloader out of it. In the subsequent epochs, the subnetwork currently being
            trained will be trained on this new dataloader containing the balanced dataset and the concatenation of all
            other subnetworks' datasets. Note that the epochs are numbered starting from 0.
        :param storage_policy: the policy that controls the external memory buffer.
            It *must* be a subclass of :class:`BufferDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
        :param batch_size: (optional) the size of the training data batch. If set to `None`, it will be set equal to
            the strategy's training batch size.
        """
        super().__init__()
        self.epoch: int = epoch
        """
        epoch number. At the end of this epoch, this plugin changes the dataloader of the subnetwork currently being
        trained. It creates a balanced training dataset for the subnetwork currently being trained and creates a
        dataloader out of it.
        """

        self.storage_policy: BufferDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate = storage_policy
        """the policy that controls the external memory buffer."""

        self.batch_size: Optional[int] = batch_size
        """the size of the training data batch."""

    def after_training_epoch(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
                             num_workers=0, shuffle=True, pin_memory=None, persistent_workers=False, drop_last=False,
                             termination_dataset=-2, oversample_small_datasets=False, balance_large_datasets=True,
                             **kwargs):
        """
        Callback called after each training epoch. If the current epoch in the strategy is equal to `self.epoch`, the
        dataloader of the subnetwork currently being trained is changed. A balanced version of the dataset of the
        subnetwork currently being trained is created and a dataloader created out of it. In the subsequent epochs,
        the subnetwork currently being trained will be trained on this new dataloader containing the balanced dataset
        and the concatenation of all other subnetworks' datasets.

        .warning::
            The old dataloader stored in `strategy.dataloader[current subnetwork ID]` is not changed. This plugin only
            changes the attribute `strategy.curr_train_dataloader`, which is used in each epoch to
            retrieve mini-batches for the training of the current subnetwork.
        :param strategy: a :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` strategy
        :param num_workers: (optional) number of thread workers for the data loading.
        :param shuffle: (optional) True if the data should be shuffled, False otherwise. Default is True.
        :param pin_memory: (optional) If True, the data loader will copy Tensors into CUDA
            pinned memory before returning them. Defaults to True.
        :param persistent_workers: (optional) If True, the data loader will not shut down the worker processes after a
            dataset has been consumed once. This allows to maintain the workers Dataset instances alive.
            Defaults to False.
        :param drop_last: (optional) set to True to drop the last incomplete batch, if the dataset size is not divisible
            by the batch size. If False and the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. Defaults to False
        :param termination_dataset: (optional) an integer number denoting the index of the dataset to be used for
            determining when to stop iterating (the iteration is stopped when the end of the respective dataset is hit).
            0 for stopping iterating when the end of the dataset the respective subnetwork must be trained on is hit.
            1 for stopping iterating when the end of the concatenation of all the other subnetworks' datasets is hit.
            -1 for stopping iterating when the end of the larger dataset is hit.
            -2 for stopping iterating when the end of the smaller dataset is hit.
            Default is -2.
        :param oversample_small_datasets: (optional) if True, the smaller dataset is oversampled to match the
            `termination_dataset`. Otherwise, once data from a dataset is completely iterated, the dataset will be
            skipped in the subsequent mini-batches. Default is False.
        :param balance_large_datasets: (optional) if True, the larger dataset is randomly reduced in size to match the
            `termination_dataset` while ensuring that each class has an identical or as similar as possible
            number of samples. Note that if True, the larger dataset is randomly reduced in size every
            time `__iter__` is called. Therefore, a different smaller dataset is created out of the larger dataset
            in each epoch. If False, the larger dataset is not reduced in size and the
            samples used for it in each dataloader iteration depends on the argument `shuffle`. If `shuffle` is False
            and `balance_large_datasets` is False, the same samples of the larger dataset are used for each dataloader
            iteration. Default is True.
        :param kwargs:
        :return: None
        """
        if self.epoch == strategy.curr_epoch:
            batch_size = self.batch_size
            if batch_size is None:  # if batch size is None, then the batch size is set to `strategy.train_mb_size`
                batch_size = strategy.train_mb_size
            curr_sub_id = strategy.curr_sub_id

            other_dataloader_args = strategy.obtain_common_dataloader_parameters(batch_size=batch_size,
                                                                                 num_workers=num_workers,
                                                                                 shuffle=shuffle,
                                                                                 pin_memory=pin_memory,
                                                                                 persistent_workers=persistent_workers,
                                                                                 drop_last=drop_last)
            # this dummy empty dataset is used to preserve the class of the datasets in `strategy.adapted_dataset` when
            # performing the following concatenation operations in case there are empty datasets
            dummy_empty_dataset = (list(strategy.adapted_dataset.values())[0]).subset([])
            # the dataset of classes the current subnetwork must be trained on in the new experience
            curr_dataset = strategy.adapted_dataset[curr_sub_id] if curr_sub_id in strategy.adapted_dataset.keys() \
                else dummy_empty_dataset
            # it is an empty dataset if there is no rehearsal buffer for a given subnetwork
            rehearsal_dataset = self.storage_policy.buffer_subnetwork(strategy, curr_sub_id)
            if len(rehearsal_dataset) == 0:  # if the rehearsal dataset is empty, replace it with the empy dummy dataset
                rehearsal_dataset = dummy_empty_dataset
            # concatenate the dataset of classes the current subnetwork must be trained on in the new experience with
            # the current subnetwork's rehearsal buffer
            dataset = curr_dataset.concat(rehearsal_dataset)
            if len(dataset) == 0:
                raise RuntimeError("No new data or rehearsal data for a given subnetwork. However, there must be "
                                   "either new data or rehearsal data for a given subnetwork.")
            # balance `dataset` by ensuring each class has the same number of samples
            dataset = balance_dataset(dataset)

            outlier_dataset = dummy_empty_dataset
            for sub_id in strategy.model.subnetworks.keys():
                if sub_id != curr_sub_id:  # if this subnetwork is not the subnetwork currently being trained
                    # the dataset of classes the subnetwork must be trained on in the new experience
                    curr_sub_outlier_dataset = strategy.adapted_dataset[sub_id] \
                        if sub_id in strategy.adapted_dataset.keys() else dummy_empty_dataset
                    # it is an empty dataset if there is no rehearsal buffer for a given subnetwork
                    rehearsal_sub_outlier_dataset = self.storage_policy.buffer_subnetwork(strategy, sub_id)
                    # if the rehearsal sub outlier dataset is empty, replace it with the empy dummy dataset
                    if len(rehearsal_sub_outlier_dataset) == 0:
                        rehearsal_sub_outlier_dataset = dummy_empty_dataset
                    # concatenate the dataset of classes the subnetwork must be trained on in the new experience with
                    # the subnetwork's rehearsal buffer
                    sub_outlier_dataset = curr_sub_outlier_dataset.concat(rehearsal_sub_outlier_dataset)
                    # concatenate `sub_outlier_dataset` with the datasets of the other subnetworks
                    if len(sub_outlier_dataset) == 0:
                        raise RuntimeError(
                            "No new data or rehearsal data for a given subnetwork. However, there must be "
                            "either new data or rehearsal data for a given subnetwork.")
                    outlier_dataset = outlier_dataset.concat(sub_outlier_dataset)

            if len(outlier_dataset) > 0:
                strategy.curr_train_dataloader = MultiDatasetBalancedDataLoader(datasets=[dataset, outlier_dataset],
                                                                                termination_dataset=termination_dataset,
                                                                                oversample_small_datasets=oversample_small_datasets,
                                                                                balance_large_datasets=balance_large_datasets,
                                                                                distributed_sampling=False,
                                                                                **other_dataloader_args)
            # the outlier dataset can be empty only if there is one subnetwork in the model: the one being trained
            else:
                collate_from_data_or_kwargs(dataset, other_dataloader_args)
                strategy.curr_train_dataloader = DataLoader(dataset, **other_dataloader_args)


class SampleCachePluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate(
    DynamicNetworkInitFeatureExtractorIncrementalOutlierPlugin, supports_distributed=True):
    """
    Sample cache plugin for :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`.

    This plugin allows to cache samples of the train dataset either before or during training/evaluation as
    mini-batches are retrieved from the dataset. It also allows to cache samples of the eval dataset either before
    evaluation or during evaluation as mini-batches are retrieved from the dataset.

    The train dataset and eval dataset must both be instances of :class:`SampleCacheMixin`.

    It implements the `after_train_dataset_adaptation` callback. If the samples of the train dataset
    need to be cached before starting the training/evaluation process on any experience, this callback caches
    the samples of the train dataset contained in the current training experience before starting the training process
    on the current experience. Otherwise, if the samples of the train dataset need to be cached during
    training/evaluation, this callback sets the cache mode of the train dataset to True.

    It implements the `after_training_exp` callback. This callback empties the cache of the train dataset after the
    training process on the current experience ends if after training/evaluation on any experience, the cached samples
    of the train dataset need to be uncached. It also sets the cache mode of the train dataset to False.
    If :class:`ReplayPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` is used in
    the current :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` strategy, this plugin will only
    uncache those samples that are not stored in the replay buffer.

    It implements the `after_eval_dataset_adaptation` callback. If the samples of the underlying dataset
    need to be cached before starting the evaluation process on the current experience, this callback
    caches the samples of the underlying dataset contained in the current experience. Otherwise, if the
    samples of the underlying need to be cached during evaluation, this callback sets the cache mode of
    the underlying dataset to True.

    It implements the `after_eval_exp` callback. This callback empties the cache of the underlying dataset
    if after evaluating on an experience, the cached samples need to be uncached. It also sets the cache mode of the
    underlying dataset to False. If :class:`ReplayPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
    is used in the current :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` strategy and the
    underlying dataset is the train dataset, this plugin will only uncache those samples that are not stored in the
    replay buffer.

    .note::
        This plugin assumes that both the train dataset and eval dataset can be used during the evaluation process.
        Thus, the underlying dataset mentioned in `after_eval_dataset_adaptation` and `after_eval_exp` could be either
        the train dataset or the eval dataset. It also assumes that only the train dataset is used during the training
        process.

    .warning::
        This plugin must be appended to the strategy's `plugin` list
        after :class:`ReplayPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`, if the latter is used.
        Otherwise, this plugin will not see the new samples added to the replay buffer.
    """
    def __init__(self, train_dataset: Optional[SampleCacheMixin] = None,
                 eval_dataset: Optional[SampleCacheMixin] = None, cache_before_exp_train: bool = False,
                 cache_before_exp_eval: bool = False, cache_n_subprocesses_train: int = 0,
                 cache_n_subprocesses_eval: int = 0, uncache_after_exp_train: bool = True,
                 uncache_after_exp_eval: bool = True):
        """
        Create a new SampleCachePluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate
        :param train_dataset: (optional) the underlying training dataset used by the current strategy. Default is None.
        :param eval_dataset: (optional) the underlying eval dataset used by the current strategy. Default is None.
        :param cache_before_exp_train: (optional) whether to cache the samples of the train dataset in the current
            experience before starting the training/evaluation process on it. If False, the samples of the train
            dataset in the current experience are cached during training/evaluation as the mini-batches are retrieved
            from the train dataset. Default is False.
        :param cache_before_exp_eval: (optional) whether to cache the samples of the eval dataset in the current
            experience before starting the evaluation process on it. If False, the samples in the current experience
            are cached during evaluation as the mini-batches are retrieved from the eval dataset. Default is False.
        :param cache_n_subprocesses_train: (optional) the number of subprocesses used to load and cache the
            samples of the train dataset when `cache_before_exp_train` is True. Default is 0 (the current process
            performs all the loading and caching operations)
        :param cache_n_subprocesses_eval: (optional) the number of subprocesses used to load and cache the
            samples of the eval dataset when `cache_before_exp_eval` is True. Default is 0 (the current process performs
            all the loading and caching operations)
        :param uncache_after_exp_train: (optional) whether to empty the cache of the train dataset after the
            training/evaluation process on the current experience is done and the underlying dataset was the train
            dataset.
            If :class:`ReplayPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` is used in
            the current :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` strategy, only
            those samples that are not stored in the replay buffer are uncached. Default is True
        :param uncache_after_exp_eval: (optional) whether to empty the cache of the eval dataset after the eval process
            on the current eval experience is done and the underlying dataset was the eval dataset.
        """
        super().__init__()
        if (train_dataset is not None) and (not isinstance(train_dataset, SampleCacheMixin)):
            raise ValueError("The train dataset must be of type `SampleCacheMixin` or None")
        if (eval_dataset is not None) and (not isinstance(eval_dataset, SampleCacheMixin)):
            raise ValueError("The eval dataset must be of type `SampleCacheMixin or None`")

        self.train_dataset: SampleCacheMixin = train_dataset
        """the underlying training dataset used by the current strategy"""
        self.train_dataset.cache_mode = False  # setting the cache mode to False in case it is True

        self.eval_dataset: SampleCacheMixin = eval_dataset
        """the underlying eval dataset used by the current strategy"""
        self.eval_dataset.cache_mode = False  # setting the cache mode to False in case it is True

        self.cache_before_exp_train: bool = cache_before_exp_train
        """
        boolean flag that indicates whether to cache the samples of the train dataset in the current experience
        before starting the training/evaluation process on it.
        """

        self.cache_before_exp_eval: bool = cache_before_exp_eval
        """
        boolean flag that indicates whether to cache the samples of the eval dataset in the current eval experience
        before starting the evaluation process on it.
        """

        self.cache_n_subprocesses_train: int = cache_n_subprocesses_train
        """
        the number of subprocesses used to load and cache the training samples when `cache_before_exp_train` is True and
        the underlying dataset is the train dataset.
        """

        self.cache_n_subprocesses_eval: int = cache_n_subprocesses_eval
        """
        the number of subprocesses used to load and cache the eval samples when `cache_before_exp_eval` is True and the
        underlying dataset is the eval dataset
        """

        self.uncache_after_exp_train: bool = uncache_after_exp_train
        """
        whether to empty the cache of the train dataset after the training/evaluation process on the current
        experience is done and the underlying dataset was the train dataset
        """

        self.uncache_after_exp_eval: bool = uncache_after_exp_eval
        """
        whether to empty the cache of the eval dataset after the eval process on the current eval experience is done and
        the underlying dataset was the eval dataset
        """

    def after_train_dataset_adaptation(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
                                       **kwargs):
        """
        Callback called after the train dataset adaptation performed for each experience.

        If `cache_before_exp_train` is True, this callback caches the samples of the train dataset contained in the
        current experience. Otherwise, this callback sets the cache mode of the train dataset to True.
        :param strategy: strategy
        :return:
        """
        if self.train_dataset is not None:
            if self.cache_before_exp_train:
                self._cache_train_dataset(strategy, mode="training")
            else:
                self.train_dataset.cache_mode = True

    def after_training_exp(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate, **kwargs):
        """
        Callback called after the training process on the current experience.

        This callback empties the cache of the train dataset if `uncache_after_exp_train` is True. It also sets the
        cache mode of the train dataset to False.
        If :class:`ReplayPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` is used in
        the current :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` strategy, this plugin will
        only uncache those samples that are not stored in the replay buffer.
        :param strategy: strategy
        :return:
        """
        if self.train_dataset is not None:
            if self.uncache_after_exp_train:
                self._uncache_train_dataset(strategy)
            self.train_dataset.cache_mode = False

    def after_eval_dataset_adaptation(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
                                      **kwargs):
        """
        Callback called after the eval dataset adaptation performed for each experience.

        If the underlying dataset is the train dataset and  `cache_before_exp_train` is True, this callback caches the
        samples of the train dataset contained in the current experience before starting the evaluation process on
        it. Otherwise, if the underlying dataset is the train dataset and `cache_before_exp_train` is
        not True, this callback sets the cache mode of the train dataset to True.

        If the underlying dataset is the eval dataset and  `cache_before_exp_eval` is True, this callback caches the
        samples of the eval dataset contained in the current experience before starting the evaluation process on
        it. Otherwise, if the underlying dataset is the eval dataset and `cache_before_exp_eval` is
        not True, this callback sets the cache mode of the eval dataset to True.

        .note::
            During the eval phase, the underlying dataset could be either the `train_dataset` or the `eval_dataset`.
            The underlying dataset used in the current eval phase is retrieved by
            using :meth:`SampleCachePluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate._get_dataset`
        :param strategy: strategy
        :return:
        """
        # the evaluation phase could be done both on the train dataset and the eval dataset. This method retrieves the
        # underlying dataset used during the current evaluation phase
        underlying_dataset = self._get_dataset(strategy.adapted_dataset)
        if underlying_dataset is self.train_dataset:
            if self.cache_before_exp_train:
                self._cache_train_dataset(strategy, mode="eval")
            else:
                self.train_dataset.cache_mode = True
        elif underlying_dataset is self.eval_dataset:
            if self.cache_before_exp_eval:
                self._cache_eval_dataset(strategy)
            else:
                self.eval_dataset.cache_mode = True

    def after_eval_exp(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate, **kwargs):
        """
        Callback called after the evaluation process on the current experience.

        This callback empties the cache of the train dataset if the underlying dataset is the train dataset and
        `uncache_after_exp_train` is True.
        If :class:`ReplayPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` is used in
        the current :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`, only those samples that are
        not stored in the replay buffer are uncached.

        It empties the cache of the eval dataset if the underlying dataset is the eval dataset and
        `uncache_after_exp_eval` is True.

        It also sets the cache mode of the underlying dataset to False.

        .note::
            During the eval phase, the underlying dataset could be either the `train_dataset` or the `eval_dataset`.
            The underlying dataset used in the current eval phase is retrieved by
            using :meth:`SampleCachePluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate._get_dataset`
        :param strategy: strategy
        :return:
        """
        # the evaluation phase could be done both on the train dataset and the eval dataset. This method retrieves the
        # underlying dataset used during the current evaluation phase
        underlying_dataset = self._get_dataset(strategy.adapted_dataset)
        if underlying_dataset is self.train_dataset:
            if self.uncache_after_exp_train:
                self._uncache_train_dataset(strategy)
            self.train_dataset.cache_mode = False
        elif underlying_dataset is self.eval_dataset:
            if self.uncache_after_exp_eval:
                self._uncache_eval_dataset(strategy)
            self.eval_dataset.cache_mode = False

    def _cache_train_dataset(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
                             mode: Literal["training", "eval"] = "training"):
        """
        Cache the samples of the train dataset in the current experience.
        :param strategy: a strategy
        :param mode: the mode of the current experience, ``training`` or ``eval``
        :return:
        """
        indices = strategy.adapted_dataset._flat_data._indices  # this field contains the indices
        print(f"Caching the samples of the current {mode} experience...")
        t1 = time.time()
        self.train_dataset.cache_samples(indices, n_subprocesses=self.cache_n_subprocesses_train)
        t2 = time.time()
        print(f"Caching completed in {(t2 - t1) / 60} minutes")

    def _uncache_train_dataset(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate):
        """
        Uncache the samples of the train dataset in the current experience.

        If :class:`ReplayPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` is used in
        the current :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` strategy, this method will
        only uncaches those samples that are not stored in the replay buffer.
        :param strategy: a stategy
        :return:
        """
        bools = [isinstance(plugin, ReplayPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate)
                 for plugin in strategy.plugins]
        # if in the current strategy there is a replay plugin, then uncache only those samples not stored
        # in the buffer
        if any(bools):
            replay_plugin = strategy.plugins[bools.index(True)]  # get the replay plugin
            storage_policy = replay_plugin.storage_policy
            indices_buffer = storage_policy.buffer._flat_data._indices
            indices_cached = self.train_dataset.cache.keys()
            # indices of samples currently cached and not stored in the replay buffer
            indices_to_remove = set(indices_cached) - set(indices_buffer)
            self.train_dataset.uncache_samples(list(indices_to_remove))
        else:
            self.train_dataset.uncache_samples()  # uncaches all samples

    def _cache_eval_dataset(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate):
        """
        Cache the samples of the eval dataset in the current eval experience
        :param strategy: a strategy
        :return:
        """
        indices = strategy.adapted_dataset._flat_data._indices  # this field contains the indices
        print("Caching the samples of the current eval experience...")
        t1 = time.time()
        self.eval_dataset.cache_samples(indices, n_subprocesses=self.cache_n_subprocesses_eval)
        t2 = time.time()
        print(f"Caching completed in {(t2 - t1) / 60} minutes")

    def _uncache_eval_dataset(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate):
        """
        Uncache the samples of the eval dataset
        :param strategy:
        :return:
        """
        self.eval_dataset.uncache_samples()  # uncaches all samples

    def _get_dataset(self, dataset: ClassificationDataset):
        """
        Get the underlying dataset from a given dataset. The underlying dataset should be either the `train_dataset`
        (if it is not None) or the `eval_dataset` (if it is not None). If both the `train_dataset` and the
        `eval_dataset` are not None and the underlying dataset is neither of them, a RuntimeError is raised.

        .warning::
            the way the underlying dataset is retrieved from the provided dataset was tested when
            using :class:`SplitImageNetSampleCache`. Therefore, this method is not guaranteed to work on other Cl
            benchmarks.
        :param dataset: dataset from which the underlying dataset is retrieved.
        :return: the underlying dataset if it is either the `train_dataset` or `eval_dataset` or None if one of
            `train_dataset` and `eval_dataset` is None and the underlying dataset is not the not-None dataset.
        """
        underlying_dataset = dataset._datasets[0]._datasets[0]._datasets[0]._datasets[0]
        if self.train_dataset is underlying_dataset:
            return self.train_dataset
        elif self.eval_dataset is underlying_dataset:
            return self.eval_dataset
        elif (self.train_dataset is None) or (self.eval_dataset is None):
            return None
        else:
            raise RuntimeError("The underlying dataset is neither the training dataset nor the eval dataset")


class PeriodicEvalPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate(
    DynamicNetworkInitFeatureExtractorIncrementalOutlierPlugin, supports_distributed=True):
    """
    Periodic eval plugin for :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`.

    This plugin schedules a periodic evaluation during the training loop of the currently trained subnetwork on the
    `_eval_streams` attribute of the :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`.

    This plugin is automatically configured and added
    by the :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`

    .note::
        Only the currently trained subnetwork is evaluated periodically; thus, some evaluation metrics, such as
        the selection accuracy are meaningless when such evaluation is performed because they rely on the results of
        multiple subnetworks. Other evaluation metrics, like the incremental classifier loss, are instead meaningful.

    .note::
        It might happen that the currently trained subnetwork is evaluated on an eval experience which precedes the
        experience in which the currently trained subnetwork was created. In that case, the dataset of that experience
        only contains outlier samples.
    """
    def __init__(self, eval_every: int = -1, peval_mode: Literal["epoch", "iteration"] = "epoch",
                 do_initial: bool = False):
        """
        Create a new PeriodicEvalPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate
        :param eval_every: the frequency of the calls to `eval` inside the training loop of the currently trained
            subnetwork. -1 disables the evaluation. 0 means `eval` is called only at the end of the learning loop.
            Values > 0 mean that `eval` is called every `eval_every` epochs or iterations according to `peval_mode`
            and at the end of the learning loop. Default is `-1`.
        :param peval_mode: one of {'epoch', 'iteration'}. Decides whether the periodic evaluation during the training
            loop of a subnetwork should execute every `eval_every` epochs or iteration. Default is `epoch`.
        :param do_initial: whether to evaluate before the training loop of a subnetwork. Occasionally this might be
            needed.
        """
        super().__init__()
        assert peval_mode in {"epoch", "iteration"}
        self.eval_every: int = eval_every
        self.peval_mode: Literal["epoch", "iteration"] = peval_mode
        self.do_initial: bool = do_initial and eval_every > -1

    def before_training_subnetwork(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
                                   **kwargs):
        """
        Callback called before the training loop of the soon-to-be-trained subnetwork.
        Evaluate the soon-to-be-trained subnetwork before its learning loop if `do_initial` is True.

        Occasionally this might be needed.
        :param strategy: a strategy
        :param kwargs:
        :return:
        """
        if self.do_initial:
            self._peval(strategy, **kwargs)

    def after_training_epoch(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate, **kwargs):
        """
        Callback called at the end of each training epoch.

        If `peval_mode` is `epoch`, the evaluation of the currently trained subnetwork is performed according to the
        value of `eval_every` and `strategy.clock.train_exp_epochs`.
        """
        if self.peval_mode == "epoch":
            self._maybe_peval(strategy, strategy.clock.train_exp_epochs, **kwargs)

    def after_training_iteration(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
                                 **kwargs):
        """
        Callback called at the end of each training iteration.

        If `peval_mode` is `iteration`, the evaluation of the currently trained subnetwork is performed according to the
        value of `eval_every` and `strategy.clock.train_exp_iterations`.
        """
        if self.peval_mode == "iteration":
            self._maybe_peval(strategy, strategy.clock.train_exp_iterations, **kwargs)

    def after_training_subnetwork(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
                                  **kwargs):
        """
        Callback called at the end of the training loop of the currently trained subnetwork.

        Evaluate the currently trained subnetwork at the end of its training loop. The evaluation is not performed by
        this callback if the last periodic evaluation already evaluated the final performance of the current
        subnetwork or if `eval_every` is `-1`.
        """
        if self.eval_every >= 0:
            evaluate_final: bool = True

            if self.peval_mode == "epoch":
                counter = strategy.clock.train_exp_epochs
            else:
                counter = strategy.clock.train_exp_iterations

            # if this holds then the last periodic evaluation already evaluated the final performance of the network
            if counter % self.eval_every == 0:
                evaluate_final = False

            if evaluate_final:
                self._peval(strategy, **kwargs)

    def _maybe_peval(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate, counter, **kwargs):
        """
        Maybe evaluate the currently trained subnetwork on all the evaluation experiences stored in
        `strategy._eval_streams`. The ID of the currently trained subnetwork is stored in `strategy.curr_sub_id`.
        The evaluation is performed according to the counter provided and the value of `eval_every`.
        :param strategy: strategy
        :param counter: counter
        :param kwargs: custom arguments
        :return:
        """
        if self.eval_every > 0 and counter % self.eval_every == 0:
            self._peval(strategy, **kwargs)

    def _peval(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate, **kwargs):
        """
        Evaluate the currently trained subnetwork on all the evaluation experiences stored in `strategy._eval_streams`.
        The ID of the currently trained subnetwork is stored in `strategy.curr_sub_id`
        :param strategy: a strategy
        :param kwargs: custom arguments
        :return:
        """
        for el in strategy._eval_streams:
            # the subnetwork_ids keyword argument is propagated up to `strategy.eval_epoch`, where it will be used for
            # computing the forward pass only for the subnetwork currently being trained
            strategy.eval(el, subnetwork_ids=strategy.curr_sub_id, **kwargs)


class WeightDecaySchedulerPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate(
    DynamicNetworkInitFeatureExtractorIncrementalOutlierPlugin, supports_distributed=True):
    """
    This plugin allows changing the weight decay of all parameter groups in all the optimizers in
    the :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` strategy before training on a new
    experience.

    A dictionary containing the experience IDs as keys and the respective weight decays as values is used to update the
    weight decay of all parameter groups in all the optimizers before training on a new experience.

    If the ID of an experience is not in the dictionary, then the weight decay of all parameter groups in all the
    optimizers is left unchanged.
    """
    def __init__(self, weight_decays: Dict[int, float]):
        """
        Create a new WeightDecaySchedulerPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate
        :param weight_decays: a dictionary containing the experience IDs as keys and the corresponding weight decays as
            values. Note that it is not necessary to provide <key, value> pairs for all experiences.
        """
        super().__init__()

        self.weight_decays: Dict[int, float] = weight_decays
        """ a dictionary containing the experience IDs as keys and the respective weight decays as values"""

    def before_training_exp(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate, **kwargs):
        """
        Callback called before training on an experience.

        If the current experience ID is a key in `weight_decays`, the weight decay of all parameter groups in all the
        optimizers in the strategy is updated according to the respective value in `weight_decays`.

        If the current experience ID is not a key in `weight_decays`, the weight decay of the optimizers is not changed.
        :param strategy: a strategy
        :param kwargs: custom arguments
        :return:
        """
        curr_experience = strategy.experience.current_experience
        if curr_experience in self.weight_decays.keys():
            for optimizer in strategy.optimizer.values():
                for param_group in optimizer.param_groups:
                    param_group["weight_decay"] = self.weight_decays[curr_experience]


class SampleTrackerPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate(
    DynamicNetworkInitFeatureExtractorIncrementalOutlierPlugin, supports_distributed=True):
    """
    Sample tracker plugin for :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`.

    This plugin tracks the samples used during training and evaluation. It stores the tracked data into three
    attributes:
        - `train_samples`
        - `val_samples`
        - `test_samples`

    `train_samples` tracks the samples used in the training experiences. It is a dictionary of
    <train experience ID, <subnetwork ID, sample data>> pairs. The sample data is a tuple of three elements
    where the first element is a tensor of sample indices, the second element is a tensor of respective target labels
    and the third is a tensor of the respective order in which the target labels appear in the list of classes seen by
    the respective subnetwork accessible through `strategy.model.subnetworks_classes[self.curr_sub_id]`.
    The index of each sample is the index to be used to retrieve the given sample from the underlying training dataset.
    This plugin metric assumes that the dataset used has the "dataset_indices" DataAttribute. The latter stores the
    dataset index of each sample. If this DataAttribute is not present, a :class:`RuntimeError` is raised. The
    "dataset_indices" DataAttribute does not need to have the attribute use_in_getitem=True.

    `val_samples` tracks the samples used in the validation experiences. It is a dictionary of
    <val experience ID, <subnetwork ID, sample data>> pairs. The sample data is a tuple of three elements
    where the first element is a tensor of sample indices, the second element is a tensor of respective target labels
    and the third is a tensor of the respective order in which the target labels appear in the list of classes seen by
    the respective subnetwork accessible through `strategy.model.subnetworks_classes[self.curr_sub_id]`.
    The index of each sample is the index to be used to retrieve the given sample from the underlying val dataset.
    This plugin metric assumes that the dataset used has the "dataset_indices" DataAttribute. The latter stores the
    dataset index of each sample. If this DataAttribute is not present, a :class:`RuntimeError` is raised. The
    "dataset_indices" DataAttribute does not need to have the attribute use_in_getitem=True.

    `test_samples` tracks the samples used in the text experiences. It is a dictionary of
    <test experience ID, <subnetwork ID, sample data>> pairs. The sample data is a tuple of three elements
    where the first element is a tensor of sample indices, the second element is a tensor of respective target labels
    and the third is a tensor of the respective order in which the target labels appear in the list of classes seen by
    the respective subnetwork accessible through `strategy.model.subnetworks_classes[self.curr_sub_id]`.
    The index of each sample is the index to be used to retrieve the given sample from the underlying test dataset.
    This plugin metric assumes that the dataset used has the "dataset_indices" DataAttribute. The latter stores the
    dataset index of each sample. If this DataAttribute is not present, a :class:`RuntimeError` is raised. The
    "dataset_indices" DataAttribute does not need to have the attribute use_in_getitem=True.

    This plugin implements the `after_train_datasets_adaptation` and `after_eval_dataset_adaptation` callbacks.

    The `after_train_datasets_adaptation` callback is called after setting up the training dataset for each subnetwork
    for the current training experience. If the ID of the current training experience has never been met before, this
    callback stores the sample data of the current training experience in the attribute `train_samples`. Otherwise, if
    the training experience has been met before, this callback does nothing.

    The `after_eval_dataset_adaptation` callback is called after setting up the eval dataset for the current experience.
    Train, val and test experiences can be provided to the eval phase; this callback detects whether a training or
    val or test experience is being used and checks whether the ID of the experience has been met before using the
    `train_samples` or `val_samples` or `test_samples` accordingly. If the ID has not been met before,  this callback
    stores the sample data of the current experience in the attribute `train_samples` or `val_samples` or `test_samples`
    according to the nature of the experience. Otherwise, if the experience has been met before, this callback does
    nothing.
    """
    def __init__(self):
        super().__init__()

        self.train_samples: Dict[int, Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = {}
        """
        a dictionary containing the training experience IDs as keys and the respective dictionaries as values. The
        dictionary of each training experience contains the subnetwork IDs as keys and a tuple of three elements as
        values. The three-element tuple of each subnetwork contains a tensor of sample indices as first element,
        a tensor of respective target labels as second element and a tensor of the respective order in which the
        target labels appear in the list of classes seen by the respective subnetwork accessible through 
        `strategy.model.subnetworks_classes[self.curr_sub_id]` as third element 
        """
        self.val_samples: Dict[int, Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = {}
        """
        a dictionary containing the val experience IDs as keys and the respective dictionaries as values. The
        dictionary of each val experience contains the subnetwork IDs as keys and a tuple of three elements as
        values. The three-element tuple of each subnetwork contains a tensor of sample indices as first element,
        a tensor of respective target labels as second element and a tensor of the respective order in which the
        target labels appear in the list of classes seen by the respective subnetwork accessible through 
        `strategy.model.subnetworks_classes[self.curr_sub_id]` as third element 
        """
        self.test_samples: Dict[int, Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = {}
        """
        a dictionary containing the test experience IDs as keys and the respective dictionaries as values. The
        dictionary of each test experience contains the subnetwork IDs as keys and a tuple of three elements as
        values. The three-element tuple of each subnetwork contains a tensor of sample indices as first element,
        a tensor of respective target labels as second element and a tensor of the respective order in which the
        target labels appear in the list of classes seen by the respective subnetwork accessible through 
        `strategy.model.subnetworks_classes[self.curr_sub_id]` as third element 
        """

    def after_train_datasets_adaptation(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
                                        **kwargs):
        """
        Callback called after setting up the training dataset for each subnetwork for the current training experience.

        If the ID of the current training experience has never been met before, this callback stores the sample data
        of the current training experience in the attribute `train_samples`. Otherwise, if the training experience has
        been met before, this callback does nothing.

        .note::
            This callback assumes that **only** training experiences are used during the training phase

        .note::
            This callback raises a :class:`RuntimeError` if the "dataset_indices" DataAttribute is not present in the
            dataset of all subnetworks

        .note::
            This callback raises a :class:`RuntimeError` if the dataset of a subnetwork contains at least one sample
            that belongs to a class the subnetwork should not be trained on

        :param strategy: a strategy
        :param kwargs: keyword arguments
        """
        strategy_id = strategy.experience.current_experience
        # if the ID of the current training experience has never been met before, store the sample data
        if strategy_id not in self.train_samples.keys():
            # create a dictionary for storing data of the current training experience
            self.train_samples[strategy_id]: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
            # `strategy.adapted_dataset` is a dictionary containing the subnetwork IDs as keys and the respective
            # avalanche datasets as values. The dataset of each subnetwork has the `targets_task_labels` DataAttribute
            # set to the respective subnetwork ID.
            for sub_id, sub_dataset in strategy.adapted_dataset.items():
                # if the "dataset_indices" DataAttribute is not in the dataset
                if not any([data_attribute.name == "dataset_indices" for data_attribute
                            in sub_dataset._data_attributes.values()]):
                    raise RuntimeError("The dataset_indices DataAttribute must be in the dataset of all subnetworks "
                                       "for this plugin to be used")
                sub_classes = strategy.model.subnetworks_classes[sub_id]
                # sub_dataset.dataset_indices.data is an instance of FlatData and is converted into a tensor
                sub_dataset_indices = torch.tensor(sub_dataset.dataset_indices.data, dtype=torch.int64)
                # sub_dataset.targets.data is an instance of FlatData and is converted into a tensor
                sub_dataset_targets = torch.tensor(sub_dataset.targets.data, dtype=torch.int64)
                order = copy.deepcopy(sub_dataset_targets)
                order.apply_(lambda x: sub_classes.index(x) if x in sub_classes else -1)
                # if there is a -1 then there is a target label not present in
                # `strategy.model.subnetworks_classes[sub_id]`
                if torch.any(order == -1):
                    raise RuntimeError(f"The dataset of subnetwork {sub_id} contains at least one sample that belongs "
                                       f"to a class the subnetwork should not be trained on")
                # moving the `order` tensor to the same device of the other tensors
                self.train_samples[strategy_id][int(sub_id)] = (sub_dataset_indices, sub_dataset_targets, order)

    def after_eval_dataset_adaptation(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
                                      **kwargs):
        """
        Callback called after setting up the eval dataset for the current experience.

        Train, val and test experiences can be provided to the eval phase; this callback detects whether a training or
        val or test experience is being used and checks whether the ID of the experience has been met before using the
        `train_samples` or `val_samples` or `test_samples` accordingly. If the ID has not been met before,  this
        callback stores the sample data of the current experience in the attribute `train_samples` or `val_samples` or
        `test_samples` according to the nature of the experience. Otherwise, if the experience has been met before, this
        callback does nothing.

        .note::
            This callback raises a :class:`RuntimeError` if the current experience is neither a training nor a val
            nor a test experience. This is checked using `strategy.experience.origin_stream.name`. For training
            experiences, the output must be `train`; for val experiences, the output must be `val`; for test
            experiences, the output must be `test`.

        .note::
            This callback raises a :class:`RuntimeError` if the "dataset_indices" DataAttribute is not present in the
            dataset of the current experience

        .note::
            This callback assumes that the `targets_task_labels` DataAttribute of each sample in the
            experience dataset is set to the ID of the respective subnetwork (in the form of an int rather than an int
            wrapped within a string)

        :param strategy: a strategy
        :param kwargs: some keyword arguments
        """
        sample_data = None
        if strategy.experience.origin_stream.name == "train":
            sample_data = self.train_samples
        elif strategy.experience.origin_stream.name == "val":
            sample_data = self.val_samples
        elif strategy.experience.origin_stream.name == "test":
            sample_data = self.test_samples

        if sample_data is None:
            raise RuntimeError("The current experience is neither a training nor a val nor a test experience")

        strategy_id = strategy.experience.current_experience
        # if the ID of the current experience has never been met before, store the sample data
        if strategy_id not in sample_data.keys():
            # if the "dataset_indices" DataAttribute is not in the dataset
            if not any([data_attribute.name == "dataset_indices" for data_attribute
                        in strategy.adapted_dataset._data_attributes.values()]):
                raise RuntimeError("The dataset_indices DataAttribute must be in the dataset for this plugin to be used")
            # create a dictionary for storing data of the current experience
            sample_data[strategy_id]: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
            for sub_id, sub_classes in strategy.model.subnetworks_classes.items():
                sub_id = int(sub_id)
                # strategy.adapted_dataset.targets_task_labels.data is an instance of FlatData and is converted into a
                # tensor
                sub_ids = torch.tensor(strategy.adapted_dataset.targets_task_labels.data, dtype=torch.int64)
                # strategy.adapted_dataset.dataset_indices.data is an instance of FlatData and is converted into a
                # tensor
                indices = torch.tensor(strategy.adapted_dataset.dataset_indices.data, dtype=torch.int64)
                sub_indices = indices[sub_ids == sub_id]
                # strategy.adapted_dataset.targets.data is an instance of FlatData and is converted into a
                # tensor
                targets = torch.tensor(strategy.adapted_dataset.targets.data, dtype=torch.int64)
                sub_targets = targets[sub_ids == sub_id]
                order = copy.deepcopy(sub_targets)
                order.apply_(lambda x: sub_classes.index(x) if x in sub_classes else -1)
                # if there is at least a -1 then there is at least a sample in the dataset whose `targets_task_labels`
                # DataAttribute has been assigned to this subnetwork but the class of the sample has not been assigned
                # to this subnetwork
                if torch.any(order == -1):
                    raise RuntimeError(f"There is at least one sample in the dataset whose `targets_task_labels` "
                                       f"DataAttribute has been assigned to subnetwork {sub_id} but the class of the "
                                       f"sample has not been assigned to subnetwork {sub_id}")
                # moving the `order` tensor to the same device of the other tensors
                sample_data[strategy_id][sub_id] = (sub_indices, sub_targets, order)


class TrainMBSchedulerPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate(
    DynamicNetworkInitFeatureExtractorIncrementalOutlierPlugin, supports_distributed=True):
    """
    Train mini-batch scheduler plugin for :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`.

    This plugin allows to schedule the size of the mini-batch used during training according to the ID of the current
    training experience.

    This plugin implements the `before_train_dataloader` callback.

    The `before_train_dataloader` callback sets the value of the attribute `strategy.train_mb_size` according to the
    given schedule and the ID of the current training experience

    .note::
        If at a given training experience, the value of `strategy.train_mb_size` is changed and the schedule does not
        change its value in subsequent training experiences, then those subsequent training experiences will keep using
        the same size of the training mini-batch
    """
    def __init__(self, train_mb_sizes: Dict[int, int]):
        """
        Create a new TrainMBSchedulerPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate
        :param train_mb_sizes: a dictionary containing the experience IDs as keys and the corresponding train mini-batch
            sizes as values. Note that it is not necessary to provide <key, value> pairs for all experiences.
        """
        super().__init__()

        self.train_mb_sizes: Dict[int, int] = train_mb_sizes
        """ a dictionary containing the experience IDs as keys and the respective train mini-batch sizes as values"""

    def before_train_dataloader(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate, **kwargs):
        """
        Callback called before creating the training dataloader.

        If the current experience ID is a key in `train_mb_sizes`, the attribute `strategy.train_mb_size` is updated
        according to the respective value in `train_mb_sizes`.

        If the current experience ID is not a key in `train_mb_sizes`, the attribute `strategy.train_mb_size` is not
        changed
        :param strategy: a strategy
        :param kwargs: custom arguments
        """
        curr_experience = strategy.experience.current_experience
        if curr_experience in self.train_mb_sizes.keys():
            strategy.train_mb_size = self.train_mb_sizes[curr_experience]


class TrainEpochsSchedulerPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate(
    DynamicNetworkInitFeatureExtractorIncrementalOutlierPlugin, supports_distributed=True):
    """
    Train epochs scheduler plugin for :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`.

    This plugin allows to schedule the number of training epochs used during training according to the ID of the current
    training experience.

    This plugin implements the `before_training_exp` callback.

    The `before_training_exp` callback sets the value of the attribute `strategy.train_epochs` according to the
    given schedule and the ID of the current training experience

    .note::
        If at a given training experience, the value of `strategy.train_epochs` is changed and the schedule does not
        change its value in subsequent training experiences, then those subsequent training experiences will keep using
        the same number of training epochs
    """
    def __init__(self, train_epochs: Dict[int, int]):
        """
        Create a new TrainEpochsSchedulerPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate
        :param train_epochs: a dictionary containing the experience IDs as keys and the corresponding number of training
            epochs as values. Note that it is not necessary to provide <key, value> pairs for all experiences.
        """
        super().__init__()

        self.train_epochs: Dict[int, int] = train_epochs
        """ a dictionary containing the experience IDs as keys and the respective number of training epochs as values"""

    def before_training_exp(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate, **kwargs):
        """
        Callback called before training on the current experience.

        If the current experience ID is a key in `train_epochs`, the attribute `strategy.train_epochs` is updated
        according to the respective value in `train_epochs`.

        If the current experience ID is not a key in `train_epochs`, the attribute `strategy.train_epochs` is not
        changed
        :param strategy: a strategy
        :param kwargs: custom arguments
        """
        curr_experience = strategy.experience.current_experience
        if curr_experience in self.train_epochs.keys():
            strategy.train_epochs = self.train_epochs[curr_experience]


class LogitsEndExperiencePluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate(
    LogitsStorageDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
    DynamicNetworkInitFeatureExtractorIncrementalOutlierPlugin, supports_distributed=True):
    """
    Logits at the end of experience plugin for :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`.

    This plugin computes the logits for the samples in the current experience at the end of the current
    experience's training. Optionally, one can also decide whether to recompute the logits for the samples in the
    previously encountered experiences, overwriting their past logits. The logits can be computed for the samples in
    the train, validation, test datasets, or any combination of these.

    .note::
        This class subclasses :class:`LogitsStorageDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`.
        Have a look at its documentation for more information.

    .note::
        This plugin *assumes* that the "dataset_indices" DataAttribute is present in the datasets of the **training**,
        **validation** and **test** experiences

    .note::
        This plugin *assumes* that there are no samples in the new experience (whose logits must be computed)
        with a "dataset_indices" DataAttribute identical to another sample in the same or past experience whose logits
        must or have been previously computed, respectively. If this is not satisfied, a :class:`RuntimeError` is
        raised.

    .note::
        This plugin metric *only* works when there is *only* one subnetwork

    .note::
        If the logits must be computed for the samples in the validation or test datasets, or both of them,
        every time the training process for a training experience with a given ID starts, the respective validation or
        test experience must be contained in `strategy._eval_streams`. Otherwise, a :class:`RuntimeError` is raised.

    .note::
        This plugin *assumes* that the output of a subnetwork's incremental classifier is the raw logits before passing
        through the softmax layer

    .note::
        This plugin *assumes* that the single subnetwork will never be trained on a training experience with an ID
        that matches the ID of a previously encountered training experience. If this happens,
         a :class:`RuntimeError` is raised.
    """

    def __init__(self, dataset_type: Union[Literal["train", "val", "test"], Iterable[Literal["train", "val", "test"]]],
                 recompute_logits: bool = False, device: Optional[Union[Literal["cpu", "cuda"], torch.device]] = None):
        """
        Create a new LogitsEndExperiencePluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate
        :param dataset_type: `train`, `val`, `test` or an iterable containing any combination of them. Decides whether
            to compute the logits for the samples in the train, validation, test datasets, or any combination of these.
        :param recompute_logits: whether to recompute the logits for the samples in the previously encountered
            experiences, overwriting their past logits. Default is False.
        :param device: device where to store the tensors of sample logits and their respective indices.
            If None, the tensors of sample logits and their respective indices are stored on the device specified by
            the `device` attribute of the :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
            strategy. Default is None.
        """
        super().__init__(dataset_type=dataset_type)

        self.logits: Dict[Literal["train", "val", "test"],
                          Optional[Tuple[torch.Tensor, torch.Tensor]]] = {v: None for v in self._dataset_type}
        """
        a dictionary containing `train`, `val`, `test` or any combination of them as keys, depending on the values
        provided at initialisation, and None or a 2-tuple as values. If a 2-tuple, the first element is a 2D tensor of
        logits for the samples in the experiences encountered so far. These samples belong to the train, validation or
        test datasets, depending on the respective key. If the logits for the samples in the previously encountered
        experiences are not recomputed at the end of each new experience training, i.e `self.recompute_logits` is False,
        they are padded with nan values to match the size of the logits for the samples in the new experience.
        The second element is a 1D tensor of respective dataset indices. Each index is the index to be used to retrieve
        the given sample from the underlying dataset and is obtained from the "dataset_indices" DataAttribute.
        If None, no logits have been collected yet. All values must be all None or all a 2-tuple where the first and
        second elements are tensors.
        """

        self.recompute_logits: bool = recompute_logits
        """
        whether to recompute the logits for the samples in the previously encountered experiences, overwriting their
        past logits.
        """

        self.device: Optional[Union[Literal["cpu", "cuda"], torch.device]] = device
        """
        device where to store the tensors of sample logits and their respective indices in `self.logits`.
        If None, the tensors of logits and their respective indices are stored on the device specified by the
        `device` attribute of the :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` strategy.
        """

        self._exps: Dict[Literal["train", "val", "test"], List] = {v: [] for v in self._dataset_type}
        """
        a dictionary containing `train`, `val`, `test` or any combination of them as keys, depending on the values
        provided at initialisation, and lists as values. Each list contains the training, validation or test
        experiences, depending on the respective key, encountered so far. The order of the experiences in the lists
        reflects the order with which these experiences were encountered. All lists must have the same number of
        experiences and all experiences at the same index across these lists must have matching IDs.
        """

        self._logits_metric: ListAccumulator = ListAccumulator()
        """
        metric for keeping track of the logits for the samples in a dataset
        """

        self._indices_metric: ListAccumulator = ListAccumulator()
        """
        metric for keeping track of the dataset indices for the samples in a dataset
        """

        self._eval_called: bool = False
        """
        boolean flag indicating whether `strategy.eval` was called by this plugin
        """

    def before_training_exp(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
                            **kwargs):
        """
        Callback called before training on the new experience

        This callback appends the new training experience to `self._exps` if it is a training experience
        never encountered before and `train` is a key in `self._dataset_type`. A :class:`RuntimeError` is raised if the
        new training experience has already been encountered.
        If `val` and/or `test` are keys in `self._dataset_type`, the validation and/or test experiences with the same ID
        as the new training experience are appended to `self._exps`. The new validation and/or test experiences are
        taken from `strategy._eval_streams`. A :class:`RuntimeError` is raised if `strategy._eval_streams` does not
        contain the new validation and/or test experiences. If logits for the samples in the previously encountered
        experiences have been collected, each sample logit is padded by appending to its end as many nan values as the
        number of new classes the single subnetwork is going to be trained during the new experience. Finally, this
        callback moves the tensors of logits and indices in `self.logits` (if any) to the appropriate device according
        to the value of `self.device`.

        .note::
            This callback raises a :class:`RuntimeError` if not all values in `self.logits` are all None or all a
            2-tuple where the first and second elements are Pytorch tensors.

        .note::
            This callback raises a :class:`RuntimeError` if not all lists in `self._exps` have the same number of
            experiences or not all experiences at the same index across these lists have matching IDs.
        :param strategy: a strategy
        :param kwargs: some keyword arguments
        """
        if strategy.experience.origin_stream.name == "train":  # if it is a training experience
            new_exp_id = strategy.experience.current_experience  # get the ID of the training experience

            # if this experience has been encountered before, then raise a :class:`RuntimeError`
            if new_exp_id in [exp.current_experience for exp in list(self._exps.values())[0]]:
                raise RuntimeError(f"A training experience with ID {new_exp_id} has already been encountered before")
            # if not all values in `self.logits` are all None or all a 2-tuple where the first and second elements are
            # Pytorch tensors, then raise a :class:`RuntimeError`
            if not self._check_logits():
                raise RuntimeError("All values in `self.logits` must be all None or all a 2-tuple where the first and "
                                   "second elements are Pytorch tensors")
            # if not all lists in `self._exps` have the same number of experiences or not all experiences
            # at the same index across these lists have matching IDs, then raise a :class:`RuntimeError`
            if not self._check_exps():
                raise RuntimeError("All lists in `self._exps` must have the same number of experiences and all "
                                   "experiences at the same index across these lists must have matching IDs")

            # append the new training, validation or test experience, or any combination of them, according to the
            # keys in `self._exps`
            for dataset_type, ls_exp in self._exps.items():
                if dataset_type == "train":
                    ls_exp.append(strategy.experience)  # append the new training experience
                else:
                    # retrieve the val or test experience according to `dataset_type` with the same ID as the new
                    # training experience from `strategy._eval_streams`
                    # `strategy._eval_streams` is a list of lists
                    rspctive_exp = [exp for exp_list in strategy._eval_streams for exp in exp_list
                                    if exp.origin_stream.name == dataset_type and exp.current_experience == new_exp_id]
                    if len(rspctive_exp) == 0:
                        raise RuntimeError(f"There is no respective {dataset_type} experience in "
                                           f"`strategy._eval_streams` for the new training experience")
                    ls_exp.append(rspctive_exp[0])  # append the new val or test experience

            # if logits for the samples in the previously encountered experiences have been collected, then pad them
            # by appending to their ends as many nan values as the number of new classes the single subnetwork is going
            # to be trained during the current experience
            if list(self.logits.values())[0] is not None:
                # get the total number of classes seen by the single subnetwork (both the old and new classes)
                n_seen_classes = len(list(strategy.model.subnetworks_classes.values())[0])
                # get the old number of classes by looking at the size of the sample logits vectors
                n_old_classes = list(self.logits.values())[0][0].shape[1]
                self._pad_with_nan(n_seen_classes - n_old_classes)

            # move the tensors of logits and indices in `self.logits` to the appropriate device
            self._to_device(self._get_device(strategy.device))

    def after_train_datasets_adaptation(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
                                        **kwargs):
        """
        Callback called after the train datasets adaptation

        This callback verifies that there is a single subnetwork by checking that `strategy.adapted_dataset` contains
        the dataset of a single subnetwork. If there are multiple subnetworks, this callback raises
        a :class`RuntimeError`.
        :param strategy: a strategy
        :param kwargs: some keyword arguments
        """
        if len(strategy.adapted_dataset) > 1:
            raise RuntimeError("This plugin only works when there is a single subnetwork")

    def after_training_exp(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate, **kwargs):
        """
        Callback called after training on the new experience.

        This callback calls `strategy.eval` to collect the logits and the dataset indices of the samples in the current
        experience. If `self.recompute_logits` is True, the logits and the dataset indices of the samples in the
        previously encountered experiences are collected as well. The logits and the dataset indices are collected
        for the samples in the train, validation, test datasets, or any combination of these, depending on the keys in
        `self._dataset_type`. The `after_eval_iteration` callback is used to capture the logits and the dataset indices
        of the samples.

        During the `strategy.eval` calls performed by this callback, the metrics in `EvaluationPlugin` are not
        computed as they would be meaningless and their computation would only be a computational overhead.

        .note::
            If there is one or more samples in the new experience (whose logits must be computed)
            with a "dataset_indices" DataAttribute identical to another sample in the same or past experience whose
            logits must or have been previously computed, respectively, a :class:`RuntimeError` is raised.

        :param strategy:  a strategy
        :param kwargs: some keyword arguments
        """
        self._eval_called = True  # indicate that the next `strategy.eval` calls are performed by this plugin
        # get the evaluator
        evaluator: EvaluationPlugin = [plg for plg in strategy.plugins if isinstance(plg, EvaluationPlugin)][0]
        # this will deactivate the evaluator, i.e. metrics are not computed during the next eval calls.
        evaluator.active = False
        for dataset_type, exps in self._exps.items():
            # if the logits for the samples in the previous experiences must not be computed, then only keep the last
            # experience in `exps`.
            if not self.recompute_logits:
                exps = exps[-1:]
            for exp in exps:
                strategy.eval(exp, **kwargs)  # perform the eval phase for this experience
            # after conclusion of all the eval phases, get the logits and the respective indices of the samples in the
            # experiences in `exps`
            logits, indices = self._logits_metric.result(), self._indices_metric.result()
            self._logits_metric.reset()  # reset the metric to its initial state
            self._indices_metric.reset()  # reset the metric to its initial state
            # concatenate all the 2d tensors of logits vertically and move to the appropriate device. To avoid any
            # possible issues, the resulting tensor is cloned
            logits = torch.cat(logits, dim=0).to(self._get_device(strategy.device)).detach().clone()
            # concatenate all the 1d tensors of indices and move to the appropriate device. To avoid any
            # possible issues, the resulting tensor is cloned
            indices = torch.cat(indices).to(self._get_device(strategy.device)).detach().clone()
            # if no logits have been previously collected or the logits of past experiences are recomputed
            if self.logits[dataset_type] is None or self.recompute_logits:
                self.logits[dataset_type] = (logits, indices)
            else:  # if logits have been previosly collected, concatenate the old logits with the new ones and the old
                # indices with the new ones
                old_logits, old_indices = self.logits[dataset_type]
                new_logits = torch.cat((old_logits, logits), dim=0)
                new_indices = torch.cat((old_indices, indices))
                self.logits[dataset_type] = (new_logits, new_indices)

            if len(torch.unique(self.logits[dataset_type][1])) < len(self.logits[dataset_type][1]):
                raise RuntimeError("One or more samples in the new experience have a `dataset_indices` DataAttribute "
                                   "identical to another sample in the same or past experience")
        self._eval_called = False  # reset the flag to false
        # active the evaluator back again, so that metrics are computed when eval is not called by this plugin
        evaluator.active = True

    def after_eval_iteration(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate, **kwargs):
        """
        Callback called after each eval iteration.

        Capture the logits and the dataset indices of the samples in the current mini-batch.
        This callback gets executed only if the current eval phase was called by this plugin.
        This is achieved by looking at the value of `self._eval_called`. This callback **should not** be executed
        when the current eval phase was not called by this plugin.
        :param strategy: a strategy
        :param kwargs: some keyword arguments
        """
        if self._eval_called:
            curr_sub_id = strategy.curr_sub_id  # store the current state of the strategy's subnetwork id
            subs_id = list(strategy.mb_output.keys())
            if len(subs_id) > 1:
                raise RuntimeError("This plugin must be used only when there is one subnetwork")
            strategy.curr_sub_id = subs_id[0]
            if "dataset_indices" not in strategy.use_in_getitem_indices.keys():
                raise RuntimeError("The dataset of the current training experience must have the `dataset_indices` "
                                   "DataAttribute")
            self._logits_metric.update(strategy.mb_output_incremental_classifier)
            self._indices_metric.update(strategy.mb_dataset_indices)
            strategy.curr_sub_id = curr_sub_id  # re-store the state of the strategy's subnetwork id

    def _get_incremental_classifier_logits(self, dataset_indices: torch.Tensor,
                                           dataset_type: Literal["train", "val", "test"]) -> torch.Tensor:
        """
        Get the logits of the given samples.
        .note::
            Safely assume that `dataset_indices` is a 1D tensor containing some elements, the logits of all the given
            samples are in the storage and `dataset_type` is contained in `self._dataset_type`.
            These checks are performed by `get_incremental_classifier_logits`, which successively invokes this method.
        .note:
            The returned tensor does not need to be allocated to the same device of `dataset_indices` because
            `get_incremental_classifier_logits` will do it.
        .note::
            Logits of different samples may vary in size because they are computed at different time steps.
            Additionally, their sizes might be smaller than the size of the logit outputs of the current subnetwork's
            incremental classifier. Therefore, to ensure consistency, the logits must be padded with NaN values when
            returned by this method so that their sizes match the size of the logit outputs of the current subnetwork's
            incremental classifier.
        :param dataset_indices: a 1D tensor of the `dataset_indices` DataAttribute of the samples whose logits must be
            retrieved.
        :param dataset_type: `train`, `val` or `test`. The type of dataset the given samples belong to.
        :return: a 2D tensor of logits for the given samples.
        """
        # get the stored logits and indices for the given dataset type
        logits, stored_indices = self.logits[dataset_type]
        # move `dataset_indices` to the same device as `logits`
        dataset_indices = dataset_indices.to(logits.device)
        matches = (stored_indices.unsqueeze(0) == dataset_indices.unsqueeze(1)).int()
        indices = matches.argmax(dim=1)
        return logits[indices]

    def _contains_incremental_classifier_logits(self, dataset_indices: torch.Tensor,
                                                dataset_type: Literal["train", "val", "test"]) -> torch.Tensor:
        """
        Check whether the logits of the given samples are in the storage.

        .note::
            Safely assume that `dataset_indices` is a 1D tensor containing some elements and `dataset_type` is contained
            in `self._dataset_type`. These checks are performed by `contains_incremental_classifier_logits`,
            which successively invokes this method.
        .note:
            The returned tensor does not need to be allocated to the same device of `dataset_indices` because
            `contains_incremental_classifier_logits` will do it.
        :param dataset_indices: a 1D tensor of the `dataset_indices` DataAttribute of the samples whose logits must be
            checked for presence.
        :param dataset_type: `train`, `val` or `test`. The type of dataset the given samples belong to.
        :return: a 1D tensor of boolean values of same size as `dataset_indices` indicating whether the logits of
            the respective samples are in the storage.
        """
        value = self.logits[dataset_type]
        # if no logits have been collected yet, return a tensor of same size as `dataset_indices` containing only False
        if value is None:
            return torch.zeros(len(dataset_indices), dtype=torch.bool)
        else:  # check whether each element in dataset_indices is in the storage
            # `dataset_indices` is moved to the same device as `value[1]`
            return torch.isin(dataset_indices.to(value[1].device), value[1])

    def _to_device(self, device: Union[Literal["cpu", "cuda"], torch.device]):
        """
        Move the tensors of logits and their respective indices in `self.logits` to the given device.
        :param device: device where to store the tensors of sample logits and their respective indices
        """
        for dataset_type, value in self.logits.items():
            if value is not None:
                self.logits[dataset_type] = (value[0].to(device), value[1].to(device))

    def _get_device(self, device: Union[Literal["cpu", "cuda"], torch.device]):
        """
        Get a device. If `self.device` is not None, `self.device` is returned. Otherwise, the provided device is
        returned.
        :return: `self.device` if `self.device` is not None. The provided device, otherwise.
        """
        _device = self.device
        if _device is None:
            _device = device
        return _device

    def _pad_with_nan(self, n: int):
        """
        Pad the tensors of logits in `self.logits` with nan values. The given number of nan values is appended to the
        end of each sample logit.
        :param n: the number of nan values to append to the end of each sample logit
        """
        if not n >= 0:
            raise ValueError("`n` must be greater than or equal to 0")
        if n > 0:
            for dataset_type, value in self.logits.items():
                if value is not None:
                    logits, indices = value
                    # create a tensor of nan values with the same number of rows, same dtype and same device as `logits`
                    nan_padding = torch.full(size=(len(logits), n), fill_value=float('nan'), dtype=logits.dtype,
                                             device=logits.device)
                    # create the padded logits
                    padded_logits = torch.cat((logits, nan_padding), dim=1)
                    # insert the padded logits
                    self.logits[dataset_type] = (padded_logits, indices)

    def _check_logits(self) -> bool:
        """
        Check if all values in `self.logits` are all None or all a 2-tuple where the first and second elements are
        Pytorch tensors
        :return: True, if all values in `self.logits` are all None or all a 2-tuple where the first and second elements
            are Pytorch tensors. False, otherwise
        """
        return all([value is None for value in self.logits.values()]) or all(
            [isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], torch.Tensor) and
             isinstance(value[1], torch.Tensor) for value in self.logits.values()]
        )

    def _check_exps(self) -> bool:
        """
        Check if all lists in `self._exps` have the same number of experiences. Additionally, ensure that experiences
        at the same index across these lists have matching IDs.
        :return: True, if all lists in `self._exps` have the same number of experiences and experiences at the same
            index across these lists have matching IDs. False, otherwise
        """
        # collect the IDs of the experiences in each list. It is a list of lists containing IDs
        exp_ids: List[List[int]] = [[exp.current_experience for exp in ls] for ls in self._exps.values()]
        # check whether all lists of IDs are identical
        return all([exp_id == exp_ids[0] for exp_id in exp_ids])


class SoftmaxProbsStartExperiencePluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate(
    ProbVecsStorageDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
    DynamicNetworkInitFeatureExtractorIncrementalOutlierPlugin, supports_distributed=True):
    """
    Softmax probabilities at the start of experience plugin
    for :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`.

    This plugin computes the softmax probabilities of the incremental classifier of the single subnetwork for the
    samples in the new experience prior to start the first training epoch on the new experience. This is performed for
    all new experiences except for the first experience.
    Note that the new output nodes (if there are any new classes in the new experience) are already present when the
    softmax probabilities are computed. The softmax probabilities can be computed for the samples in the train,
    validation, test datasets, or any combination of these. At the end of the new experience training, the softmax
    probabilities for the samples in the new experience are deleted.

    `use_incremental_classifier_prob_vecs = True` *must* be provided as a keyword argument when invoking `train` and
    `eval` of a :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` strategy. Otherwise,
    a :class:`RuntimeError` is raised by this plugin. When the above-mentioned keyword argument is provided, the
    strategy uses the softmax probabilities stored in this plugin for the samples in the new experience to compute the
    loss rather than using the standard one-hot encoding vectors. By default, the strategy will use standard one-hot
    encoding vectors for all samples not stored in this plugin. A scheduler function can be used to modify
    the softmax probabilities prior to each training epoch. This way one can gradually make the initial softmax
    probabilities approach the standard one-hot encoding vectors for the samples in the new experience.
    Using the standard one-hot encoding vectors right from the start leads to lots of catastrophic forgetting and this
    approach can mitigate catastrophic forgetting by using a sort of gradual learning during training. Note that the
    scheduler function is applied to the samples of any type of dataset for which softmax probabilities are computed
    (train, validation and test).

    .note::
        The evaluations performed before and after the training on the new experience use standard one-hot encoding
        vectors for all samples when computing the loss because the softmax probabilities for the samples in the new
        experience are computed immediately before the first training epoch and deleted when the training ends.

    .note::
        This class subclasses :class:`ProbVecsStorageDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`.
        Have a look at its documentation for more information.

    .note::
        This plugin *assumes* that the "dataset_indices" DataAttribute is present in the datasets of the **training**,
        **validation** and **test** experiences.

    .note::
        This plugin metric *only* works when there is *only* one subnetwork

    .note::
        If the softmax probabilities must be computed for the samples in the validation or test datasets, or both of
        them, every time the training process for a training experience with a given ID starts, the respective
        validation or test experience must be contained in `strategy._eval_streams`. Otherwise, a :class:`RuntimeError`
        is raised.

    .note::
        This plugin *assumes* that there are no samples in the new experience, which is not the first one,
        with a "dataset_indices" DataAttribute identical to another sample in the same experience.
        If this is not satisfied, a :class:`RuntimeError` is raised.

    .note::
        This plugin *assumes* that the output of a subnetwork's incremental classifier is the raw logits before passing
        through the softmax layer
    """

    def __init__(self, dataset_type: Union[Literal["train", "val", "test"], Iterable[Literal["train", "val", "test"]]],
                 scheduler: Optional[Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]] = None,
                 device: Optional[Union[Literal["cpu", "cuda"], torch.device]] = None):
        """
        Create a new SoftmaxProbsStartExperiencePluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate
        :param dataset_type: `train`, `val`, `test` or an iterable containing any combination of them. Decides whether
            to compute the softmax probabilities for the samples in the train, validation, test datasets, or any
            combination of these.
        :param scheduler: a callable that takes in a deep copy of the currently stored 2D tensor of sample softmax
            probabilities, a deep copy of the 1D tensor of their target labels and the current training epoch number
            and returns a new 2D tensor of sample softmax probabilities that overwrites the current one. Training epoch 
            numbering starts from 0. Note that the class targets are not the real class targets but the order in which
            classes appear in the list of classes seen by the single subnetwork accessible through
            `strategy.model.subnetworks_classes[id_of_the_single_subnetwork]`. If None, the softmax probabilities
            currently stored are not modified during training.
        :param device: device where to store the tensors of sample softmax probabilities, their respective indices and
            targets. If None, the tensors of sample softmax probabilities, their respective indices and targets are
            stored on the device specified by the `device` attribute of
            the :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` strategy. Default is None.
        """
        super().__init__(dataset_type=dataset_type)

        self.soft_probs: Dict[Literal["train", "val", "test"],
                              Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = {v: None
                                                                                            for v in self._dataset_type}
        """
        a dictionary containing `train`, `val`, `test` or any combination of them as keys, depending on the values
        provided at initialisation, and None or a 3-tuple as values. If a 3-tuple, the first element is a 2D tensor of
        softmax probabilities for the samples in the new experience. These samples belong to the train, validation or
        test datasets, depending on the respective key. The second element is a 1D tensor of respective dataset indices.
        Each index is the index to be used to retrieve the given sample from the underlying dataset and is obtained
        from the "dataset_indices" DataAttribute. The third element is a 1D tensor of respective target labels.
        The class targets are not the real class targets but the order in which classes appear in the list of classes
        seen by the single subnetwork accessible through
        `strategy.model.subnetworks_classes[id_of_the_single_subnetwork].
        All values must be all None or all a 3-tuple where
        the first, second and third elements are tensors. All values are all a 3-tuple when a new training experience,
        which is not the first one, arrives. They are all None during the first training experience and after the
        training process on a new experience ends.
        """

        self.scheduler: Optional[Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]] = scheduler
        """
        a callable that takes in a deep copy of the currently stored 2D tensor of sample softmax probabilities, a deep
        copy of the 1D tensor of their target labels and the current training epoch number and returns a new 2D tensor
        of sample softmax probabilities that overwrites the current one in `self.soft_probs`. Training epoch numbering
        starts from 0. Note that the class targets are not the real class targets but the order in which classes appear
        in the list of classes seen by the single subnetwork accessible through
        `strategy.model.subnetworks_classes[id_of_the_single_subnetwork]`. If None, the softmax probabilities
        currently stored in `self.soft_probs` are not modified during training.
        """

        self.device: Optional[Union[Literal["cpu", "cuda"], torch.device]] = device
        """
        device where to store the tensors of sample softmax probabilities, their respective indices and target labels
        in `self.soft_probs`. If None, the tensors are stored on the device specified by the
        `device` attribute of the :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` strategy.
        """

        self._exps: Dict[Literal["train", "val", "test"], List] = {v: [] for v in self._dataset_type}
        """
        a dictionary containing `train`, `val`, `test` or any combination of them as keys, depending on the values
        provided at initialisation, and lists as values. Each list contains the current training, validation or test
        experience, depending on the respective key. All lists must contain either one experience or be empty. The
        lists are emptied after the new experience training ends and are populated when a new training experience
        arrives unless it is the first experience.
        """

        self._soft_probs_metric: ListAccumulator = ListAccumulator()
        """
        metric for keeping track of the softmax probabilities for the samples in a dataset
        """

        self._target_metric: ListAccumulator = ListAccumulator()
        """
        metric for keeping track of the target labels for the samples in a dataset
        """

        self._indices_metric: ListAccumulator = ListAccumulator()
        """
        metric for keeping track of the dataset indices for the samples in a dataset
        """

        self._eval_called: bool = False
        """
        boolean flag indicating whether `strategy.eval` was called by this plugin
        """

    def before_training(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate, **kwargs):
        """
        Callback called before the overall training process starts.

        Check whether `use_incremental_classifier_prob_vecs = True` was provided as a keyword argument when `train` of
        the :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` strategy was invoked.
        :param strategy: a strategy
        :param kwargs: some keyword arguments
        """
        if "use_incremental_classifier_prob_vecs" not in kwargs.keys():
            raise RuntimeError("`use_incremental_classifier_prob_vecs` must be provided as a keyword argument when "
                               "`train` of the `DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` strategy "
                               "is invoked")
        if kwargs["use_incremental_classifier_prob_vecs"] is False:
            raise RuntimeError("The value of the `use_incremental_classifier_prob_vecs` keyword argument, provided "
                               "when `train` of the `DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` "
                               "strategy was invoked, must be True")

    def before_eval(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate, **kwargs):
        """
        Callback called before the overall evaluation process starts.

        Check whether `use_incremental_classifier_prob_vecs = True` was provided as a keyword argument when `eval` of
        the :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` strategy was invoked.
        :param strategy: a strategy
        :param kwargs: some keyword arguments
        """
        if "use_incremental_classifier_prob_vecs" not in kwargs.keys():
            raise RuntimeError("`use_incremental_classifier_prob_vecs` must be provided as a keyword argument when "
                               "`eval` of the `DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` strategy "
                               "is invoked")
        if kwargs["use_incremental_classifier_prob_vecs"] is False:
            raise RuntimeError("The value of the `use_incremental_classifier_prob_vecs` keyword argument, provided "
                               "when `eval` of the `DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` "
                               "strategy was invoked, must be True")

    def before_training_exp(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
                            **kwargs):
        """
        Callback called before training on the new experience

        This callback appends the new training experience to `self._exps` if it is not the first training experience
        and `train` is a key in `self._dataset_type`. The first training experience is assumed to be the experience
        with ID equal to 0.
        If `val` and/or `test` are keys in `self._dataset_type`, the validation and/or test experiences with the same ID
        as the new training experience are appended to `self._exps` if the ID is not equal to 0.
        The new validation and/or test experiences are taken from `strategy._eval_streams`.
        A :class:`RuntimeError` is raised if `strategy._eval_streams` does not
        contain the new validation and/or test experiences.

        .note::
            This callback raises a :class:`RuntimeError` if not all values in `self.soft_probs` are all None

        .note::
            This callback raises a :class:`RuntimeError` if not all lists in `self._exps` are empty.
        :param strategy: a strategy
        :param kwargs: some keyword arguments
        """
        if strategy.experience.origin_stream.name == "train":  # if it is a training experience

            # if not all values in `self.soft_probs` are all None, then raise a :class:`RuntimeError`
            if not all([val is None for val in self.soft_probs.values()]):
                raise RuntimeError("All values in `self.soft_probs` must be None before training on a new experience")
            # if not all lists in `self._exps` are empty, then raise a :class:`RuntimeError`
            if not all([len(val) == 0 for val in self._exps.values()]):
                raise RuntimeError("All lists in `self._exps` must be empty before training on a new experience")

            new_exp_id = strategy.experience.current_experience  # get the ID of the training experience
            # if it is not the first training experience
            if new_exp_id > 0:
                # append the new training, validation or test experience, or any combination of them, according to the
                # keys in `self._exps`
                for dataset_type, ls_exp in self._exps.items():
                    if dataset_type == "train":
                        ls_exp.append(strategy.experience)  # append the new training experience
                    else:
                        # retrieve the val or test experience according to `dataset_type` with the same ID as the new
                        # training experience from `strategy._eval_streams`
                        # `strategy._eval_streams` is a list of lists
                        rspctive_exp = [exp for exp_list in strategy._eval_streams for exp in exp_list
                                        if exp.origin_stream.name == dataset_type and exp.current_experience == new_exp_id]
                        if len(rspctive_exp) == 0:
                            raise RuntimeError(f"There is no respective {dataset_type} experience in "
                                               f"`strategy._eval_streams` for the new training experience")
                        ls_exp.append(rspctive_exp[0])  # append the new val or test experience

    def after_train_datasets_adaptation(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
                                        **kwargs):
        """
        Callback called after the train datasets adaptation

        This callback verifies that there is a single subnetwork by checking that `strategy.adapted_dataset` contains
        the dataset of a single subnetwork. If there are multiple subnetworks, this callback raises
        a :class`RuntimeError`.

        The ID of the single subnetwork is added into `self._subnetwork_ids`.
        :param strategy: a strategy
        :param kwargs: some keyword arguments
        """
        if len(strategy.adapted_dataset) > 1:
            raise RuntimeError("This plugin only works when there is a single subnetwork")
        self._subnetwork_ids.add(int(list(strategy.adapted_dataset.keys())[0]))
        
    def before_training_epoch(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate, **kwargs):
        """
        Callback called before each training epoch.

        If the current epoch number is 0, this callback calls `strategy.eval` to collect the softmax probabilities,
        the dataset indices and target labels of the samples in the experiences contained in `self._exps`. Note that
        `self._exps` contains no experiences when the current training experience is the first one, i.e. the one with
        ID equal to 0. The softmax probabilities, the dataset indices and target labels are collected
        for the samples in the train, validation, test datasets, or any combination of these, depending on the keys in
        `self._dataset_type`. The `after_eval_iteration` callback is used to capture this data.
        During the `strategy.eval` calls performed by this callback, the metrics in `EvaluationPlugin` are not
        computed as they would be meaningless and their computation would only be a computational overhead.

        If `self.scheduler` is not None, the softmax probabilities in `self.soft_probs` are modified according to the
        new 2D tensor output by the scheduler callable.

        .note::
            If there is one or more samples in the new experience (whose softmax probabilities must be computed)
            with a "dataset_indices" DataAttribute identical to another sample in the same experience,
            a :class:`RuntimeError` is raised.
        :param strategy: a strategy
        :param kwargs: some keyword arguments
        """
        curr_epoch: int = strategy.curr_epoch  # current epoch number
        # if the current epoch is 0, then collect the softmax probabilities, the dataset indices and target labels
        # of the samples in the experiences contained in `self._exps`
        if curr_epoch == 0:
            self._eval_called = True  # indicate that the next `strategy.eval` calls are performed by this plugin
            # get the evaluator
            evaluator: EvaluationPlugin = [plg for plg in strategy.plugins if isinstance(plg, EvaluationPlugin)][0]
            # this will deactivate the evaluator, i.e. metrics are not computed during the next eval calls.
            evaluator.active = False
            for dataset_type, exps in self._exps.items():
                for exp in exps:
                    strategy.eval(exp, **kwargs)  # perform the eval phase for this experience
                    # after conclusion of the eval phases, get the softmax probs, the respective indices and target
                    # labels of the samples in the experience in `exps`
                    soft_probs = self._soft_probs_metric.result()
                    indices = self._indices_metric.result()
                    target = self._target_metric.result()
                    self._soft_probs_metric.reset()  # reset the metric to its initial state
                    self._indices_metric.reset()  # reset the metric to its initial state
                    self._target_metric.reset()  # reset the metric to its initial state
                    # concatenate all the 2d tensors of softmax probs vertically and move to the appropriate device.
                    # To avoid any possible issues, the resulting tensor is cloned
                    soft_probs = torch.cat(soft_probs, dim=0).to(self._get_device(strategy.device)).detach().clone()
                    # concatenate all the 1d tensors of indices and move to the appropriate device. To avoid any
                    # possible issues, the resulting tensor is cloned
                    indices = torch.cat(indices).to(self._get_device(strategy.device)).detach().clone()
                    # concatenate all the 1d tensors of target labels and move to the appropriate device. To avoid any
                    # possible issues, the resulting tensor is cloned
                    target = torch.cat(target).to(self._get_device(strategy.device)).detach().clone()
                    # store the softmax probs, the indices and the targets of the samples in `self.soft_probs`
                    self.soft_probs[dataset_type] = (soft_probs, indices, target)
                    if len(torch.unique(self.soft_probs[dataset_type][1])) < len(self.soft_probs[dataset_type][1]):
                        raise RuntimeError(
                            "One or more samples in the new experience have a `dataset_indices` DataAttribute "
                            "identical to another sample in the same experience")
            self._eval_called = False  # reset the flag to false
            # active the evaluator back again, so that metrics are computed when eval is not called by this plugin
            evaluator.active = True

        # modify the softmax probabilities according to the current epoch number by using the provided scheduler in
        # `self.scheduler`
        if self.scheduler is not None:
            for key, val in self.soft_probs.items():
                if val is not None:
                    self.soft_probs[key] = (
                        self.scheduler(val[0].detach().clone(), val[2].detach().clone(), curr_epoch), val[1], val[2])

        # move the tensors in `self.soft_probs` (if any) to the appropriate device; in case the scheduler changes their
        # device
        self._to_device(self._get_device(strategy.device))

    def after_eval_iteration(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate, **kwargs):
        """
        Callback called after each eval iteration.

        Capture the softmax probabilities, the dataset indices and target labels of the samples in the current
        mini-batch.
        This callback gets executed only if the current eval phase was called by this plugin.
        This is achieved by looking at the value of `self._eval_called`. This callback **should not** be executed
        when the current eval phase was not called by this plugin.

        .note::
            It is *assumed* that the output of the single subnetwork's incremental classifier is the raw logits before
            passing through the softmax layer
        :param strategy: a strategy
        :param kwargs: some keyword arguments
        """
        if self._eval_called:
            curr_sub_id = strategy.curr_sub_id  # store the current state of the strategy's subnetwork id
            subs_id = list(strategy.mb_output.keys())
            if len(subs_id) > 1:
                raise RuntimeError("This plugin must be used only when there is one subnetwork")
            strategy.curr_sub_id = subs_id[0]
            if "dataset_indices" not in strategy.use_in_getitem_indices.keys():
                raise RuntimeError("The dataset of the current experience must have the `dataset_indices` "
                                   "DataAttribute")
            self._soft_probs_metric.update(strategy.mb_output_incremental_classifier.softmax(dim=1))
            self._indices_metric.update(strategy.mb_dataset_indices)
            self._target_metric.update(strategy.mb_y_incremental_classifier)
            strategy.curr_sub_id = curr_sub_id  # re-store the state of the strategy's subnetwork id

    def after_training_exp(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate, **kwargs):
        """
        Callback called after training on the new experience.

        This callback empties `self.soft_probs` and `self._exps`

        :param strategy:  a strategy
        :param kwargs: some keyword arguments
        """
        self.soft_probs = {v: None for v in self._dataset_type}
        self._exps = {v: [] for v in self._dataset_type}

    def _get_incremental_classifier_prob_vecs(self, dataset_indices: torch.Tensor,
                                              dataset_type: Literal["train", "val", "test"], subnetwork_id: int,
                                              num_classes: int) -> torch.Tensor:
        """
        Get the probability vectors of the given samples.

        .note::
            Safely assume that `dataset_indices` is a 1D tensor containing some elements, the probability vectors of
            all the given samples are in the storage, `dataset_type` is contained in `self._dataset_type` and
            `subnetwork_id` is contained in `self._subnetwork_ids`.
            These checks are performed by `get_incremental_classifier_prob_vecs`,  which successively invokes this
            method.

        .note:
            The returned tensor does not need to be allocated to the same device of `dataset_indices` because
            `get_incremental_classifier_prob_vecs` will do it.

        :param dataset_indices: a 1D tensor of the `dataset_indices` DataAttribute of the samples whose probability
            vectors must be retrieved.
        :param dataset_type: `train`, `val` or `test`. The type of dataset the given samples belong to.
        :param subnetwork_id: the id of the subnetwork the classes of the samples are allocated to. Note that all the
            classes of the samples *must* be allocated to the same subnetwork.
        :param num_classes: the number of classes assigned to the incremental classifier of the subnetwork with id
            `subnetwork_id`, i.e. the number of output nodes of the incremental classifier of the subnetwork with id
            `subnetwork_id`. If the size of the probability vectors returned does not match this number,
            a :class:`RuntimeError` is raised by `get_incremental_classifier_prob_vecs`.
        :return: a 2D tensor of probability vectors for the given samples.
        """
        # get the stored softmax probabilities and indices for the given dataset type
        soft_probs, stored_indices, _ = self.soft_probs[dataset_type]
        # move `dataset_indices` to the same device as `soft_probs`
        dataset_indices = dataset_indices.to(soft_probs.device)
        matches = (stored_indices.unsqueeze(0) == dataset_indices.unsqueeze(1)).int()
        indices = matches.argmax(dim=1)
        return soft_probs[indices]

    def _contains_incremental_classifier_prob_vecs(self, dataset_indices: torch.Tensor,
                                                   dataset_type: Literal["train", "val", "test"],
                                                   subnetwork_id: int) -> torch.Tensor:
        """
        Check whether the probability vectors of the given samples are in the storage.

        .note::
            Safely assume that `dataset_indices` is a 1D tensor containing some elements, `dataset_type` is contained
            in `self._dataset_type` and `subnetwork_id` is contained in `self._subnetwork_ids`.
            These checks are performed by `contains_incremental_classifier_prob_vecs`, which successively invokes this
            method.

        .note:
            The returned tensor does not need to be allocated to the same device of `dataset_indices` because
            `contains_incremental_classifier_prob_vecs` will do it.

        :param dataset_indices: a 1D tensor of the `dataset_indices` DataAttribute of the samples whose probability
            vectors must be checked for presence.
        :param dataset_type: `train`, `val` or `test`. The type of dataset the given samples belong to.
        :param subnetwork_id: the id of the subnetwork the classes of the samples are allocated to. Note that all the
            classes of the samples *must* be allocated to the same subnetwork.
        :return: a 1D tensor of boolean values of same size as `dataset_indices` indicating whether the probability
            vectors of the respective samples are in the storage.
        """
        value = self.soft_probs[dataset_type]
        # if no softmax probabilities have been collected, return a tensor of same size as `dataset_indices` containing
        # only False
        if value is None:
            return torch.zeros(len(dataset_indices), dtype=torch.bool)
        else:  # check whether each element in dataset_indices is in the storage
            # `dataset_indices` is moved to the same device as `value[1]`
            return torch.isin(dataset_indices.to(value[1].device), value[1])

    def _get_device(self, device: Union[Literal["cpu", "cuda"], torch.device]):
        """
        Get a device. If `self.device` is not None, `self.device` is returned. Otherwise, the provided device is
        returned.
        :return: `self.device` if `self.device` is not None. The provided device, otherwise.
        """
        _device = self.device
        if _device is None:
            _device = device
        return _device

    def _to_device(self, device: Union[Literal["cpu", "cuda"], torch.device]):
        """
        Move the tensors of softmax probabilities, their respective indices and target labels in `self.soft_probs` to
        the given device.
        :param device: device where to store the tensors of softmax probabilities, their respective indices and target
            labels
        """
        for dataset_type, value in self.soft_probs.items():
            if value is not None:
                self.soft_probs[dataset_type] = (value[0].to(device), value[1].to(device), value[2].to(device))


class InjectClassSamplesNextExperiencePluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate(
    ProbVecsStorageDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
    DynamicNetworkInitFeatureExtractorIncrementalOutlierPlugin, supports_distributed=True):
    """
    Inject class samples of successive experience plugin
    for :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`.

    This plugin injects *all* the samples of *all* the classes contained in the successive training experience into the
    dataset of the current training experience prior to start training on the current experience. This injection is
    performed for each current training experience except for the last experience as there is no successive experience.
    The user can specify to skip some experiences, i.e. for some experiences samples are not injected from the
    successive training experience.
    The targets of the samples injected from the successive experience are transformed into targets of classes contained
    in the current or previous experiences as specified by the user through a dictionary. This plugin uses the
    dictionary provided by the user to update the underlying strategy's `target_to_target` dictionary attribute. It is
    assumed that the strategy's `target_to_target` attribute would be an empty dictionary at any time if this plugin
    were not used. Read the doc of the `target_to_target` dictionary
    in :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` for more info about this dictionary.
    Substantially, this plugin concatenates the dataset of the current training experience with the
    dataset of the successive training experience, transforming the targets as outlined above, so that the subsequent
    dataloaders built for each subnetwork contain these samples. Therefore, prior to perform training on the successive
    experience, the subnetworks will be biased toward classifying the samples in the successive experience as one of the
    classes previously encountered.

    For more fine-grained control, additionally, this plugin allows to specify, through a callable, the specific
    probability vectors of each sample injected from the successive experience to be used for computing the loss during
    training of the current experience. `use_incremental_classifier_prob_vecs = True` *must* be provided as a
    keyword argument when invoking `train` of the
    underlying :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` strategy when the callable is
    provided. Otherwise, a :class:`RuntimeError` is raised by this plugin. When the above-mentioned keyword argument is
    provided, the strategy will use the probability vectors returned by the callable for the samples injected from the
    next training experience to compute the loss rather than using the standard one-hot encoding vectors. By default,
    the strategy will use standard one-hot encoding vectors for the other samples.

    This plugin implements the `before_training`, `after_train_dataset_adaptation`, `after_train_datasets_adaptation`
    and `after_training_exp` callbacks.

    .note::
        This plugin does not modify the original train dataset instance of the current and successive experience
        in-place. Therefore, such instances contain the original samples. As a consequence, the periodic evaluations
        performed by :class:`PeriodicEvalPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` during the
        training of the current experience do not take the injected samples into account. Only the evaluations
        performed during the actual training, such as the training epoch average loss, take the injected samples
        into account. Bear this information in mind when analysing the evaluation metrics data.

    .note::
        This class subclasses :class:`ProbVecsStorageDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`.
        Have a look at its documentation for more information.

    .note::
        A :class:`RuntimeError` is raised if `self.get_incremental_classifier_prob_vecs` or
        `self._get_incremental_classifier_prob_vecs` are invoked and no callable specifying the probability vectors
        of each sample in the successive training experience was provided at initialisation

    .note::
        This plugin *assumes* that the "dataset_indices" DataAttribute is present in the datasets of the training
        experiences.

    .note::
        This plugin *assumes* that there are no samples in the next training experience with a "dataset_indices"
        DataAttribute identical to another sample in the same experience.
        If this is not satisfied, a :class:`RuntimeError` is raised.

    .note::
        When :class:`ReplayPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` or
        :class:`ReplayNoOutlierPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` are inserted into
        the `plugins` list of the underlying strategy, this plugin *must* be inserted before them into the `plugins`
        list. If this is not done, the replay buffer might be populated with samples belonging to the successive
        experience.
    """
    def __init__(self, target_map: Dict[int, int], skip_exps: Optional[Iterable[int]] = None,
                 prob_vec: Optional[Callable[[torch.Tensor, int, int], torch.Tensor]] = None):
        """
        Create a new InjectClassSamplesNextExperiencePluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate
        :param target_map: a dictionary determining how the targets of the samples injected from the successive
            training experiences are mapped into the targets in the current or previously encountered experiences.
            The targets *must* be the same ones returned by the underlying continual learning benchmark. The targets of
            the classes contained in each experience can be accessed through `experience.classes_in_this_experience`.
            A :class:`RuntimeError` is raised if this dictionary does not contain the mapping for all
            targets contained in the successive experiences or if the respective value of a mapping is not a target in
            the current or previously encountered experiences.
        :param skip_exps: an iterable containing the IDs of the experiences for which samples from the successive
            training experience *must not* be injected into. If None, samples from the successive training experiences
            are injected into all the current experiences except for the last experience as there is no successive
            experience. Default is None.
        :param prob_vec: a callable that takes in a tensor of `dataset_indices` DataAttributes, the ID of the subnetwork
            the given samples are allocated to and the size each probability vector must have and returns a 2D tensor
            containing the probability vectors of each sample. Safely assume that the tensor of `dataset_indices`
            DataAttributes refers to samples that belong to the successive training experience.
            `use_incremental_classifier_prob_vecs = True` *must* be provided as a keyword argument when invoking `train`
            of the underlying :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` strategy when this
            callable is provided. Otherwise, a :class`RuntimeError` will be raised. If None, a :class:`RuntimeError` is
            raised if `self.get_incremental_classifier_prob_vecs` or `self._get_incremental_classifier_prob_vecs` are
            invoked. Default is None.
        """
        # specify that probability vectors are stored for the samples in the train, validation and test datasets
        # although this plugin only outputs probability vectors for the samples injected from the dataset of the
        # successive training experience.
        super().__init__(dataset_type=["train", "val", "test"])

        self._target_map: Dict[int, int] = target_map
        """
        a dictionary determining how the targets of the samples injected from the successive
        training experiences are mapped into the targets in the current or previously encountered training experiences.
        The targets *must* be the same ones returned by the underlying continual learning benchmark. The targets of
        the classes contained in each experience can be accessed through `experience.classes_in_this_experience`.
        A :class:`RuntimeError` is raised if this dictionary does not contain the mapping for all
        targets contained in the successive experiences or if the respective value of a mapping is not a target in
        the current or previously encountered experiences.
        """

        self._skip_exps: Iterable[int] = skip_exps if skip_exps is not None else []
        """
        an iterable containing the IDs of the experiences for which samples from the successive training experience
        *must not* be injected into.
        """

        self._prob_vec: Optional[Callable[[torch.Tensor, int, int], torch.Tensor]] = prob_vec
        """
        a callable that takes in a tensor of `dataset_indices` DataAttributes, the ID of the subnetwork
        the given samples are allocated to and the size each probability vector must have and returns a 2D tensor
        containing the probability vectors of each sample. Safely assume that the tensor of `dataset_indices`
        DataAttributes refers to samples that belong to the successive training experience.
        `use_incremental_classifier_prob_vecs = True` *must* be provided as a keyword argument when invoking `train`
        of the underlying :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` strategy when this
        callable is provided. Otherwise, a :class`RuntimeError` will be raised. If None, a :class:`RuntimeError` is
        raised if `self.get_incremental_classifier_prob_vecs` or `self._get_incremental_classifier_prob_vecs` are
        invoked.
        """

        self._successive_dataset_indices: Optional[torch.Tensor] = None
        """
        a tensor containing the `dataset_indices` DataAttribute of the samples contained in the successive training
        experience (if any). It is set to None after the training on the current experience ends.
        """

    def before_training(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate, **kwargs):
        """
        Callback called before the overall training process starts.

        If `self._prob_vec` is not None, check whether `use_incremental_classifier_prob_vecs = True` was provided as a
        keyword argument when `train` of the :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
        strategy was invoked.
        :param strategy: a strategy
        :param kwargs: some keyword arguments
        """
        if self._prob_vec is not None:
            if "use_incremental_classifier_prob_vecs" not in kwargs.keys():
                raise RuntimeError("When the callable of probability vectors is provided, "
                                   "`use_incremental_classifier_prob_vecs` must be provided as a keyword argument when "
                                   "`train` of the `DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` "
                                   "strategy is invoked")
            if kwargs["use_incremental_classifier_prob_vecs"] is False:
                raise RuntimeError("When the callable of probability vectors is provided, the value of the "
                                   "`use_incremental_classifier_prob_vecs` keyword argument, provided "
                                   "when `train` of the `DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` "
                                   "strategy was invoked, must be True")

    def after_train_dataset_adaptation(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
                                       **kwargs):
        """
        Callback called after the `adapted_dataset` attribute of the underlying strategy is initialised with the train
        dataset of the current experience. It concatenates the dataset of the current training experience with the
        dataset of the successive training experience (if any), transforming the targets of the samples in the
        successive training experience as determined by `self._target_map`, and stores it in the `adapted_dataset`
        attribute of the underlying strategy. It updates `strategy.target_to_target` with the targets in the successive
        training experience as keys and their respective values in `self._target_map`.
        This way, the subsequent dataloaders built for each subnetwork contain these samples. Therefore, prior to
        perform training on the successive experience, the subnetworks will be biased toward classifying the samples in
        the successive experience as one of the classes previously encountered.

        The `dataset_indices` DataAttribute of the samples in the successive training experience (if any) are stored in
        `self._successive_dataset_indices`.

        .note::
            It does not modify the original train dataset instance of the current and successive experience in-place.
            Therefore, such instances contain the original samples. As a consequence, the periodic evaluations performed
            by :class:`PeriodicEvalPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` during the
            training of the current experience do not take the injected samples into account. Only the evaluations
            performed during the actual training, such as the training epoch average loss, take the injected samples
            into account. Bear this information in mind when analysing the evaluation metrics data.

        .note::
            It raises a :class:`RuntimeError` if the `dataset_indices` DataAttribute is not present in the dataset of
            the current and successive training experience (if any).

        .note::
            It raises a :class:`RuntimeError` if `self._target_map` does not contain the mapping for all
            targets contained in the successive experiences or if the respective value of a mapping is not a target in
            the current or previously encountered experiences.

        .note::
            It raises a :class:`RuntimeError` if `strategy.target_to_target` is not an empty dictionary prior to
            updating it.

        :param strategy: a strategy
        :param kwargs: some keyword arguments
        """
        if "dataset_indices" not in strategy.use_in_getitem_indices.keys():
            raise RuntimeError("The dataset of the current training experience must have the `dataset_indices` "
                               "DataAttribute")
        curr_exp_id = strategy.experience.current_experience  # get the ID of the current training experience
        # get the number of training experiences in the current benchmark
        n_exps = len(strategy.experience.benchmark.train_stream)
        # if the current experience is not the last one, i.e. there exists a successive training experience, and the
        # current experience is not to be skipped
        if curr_exp_id + 1 < n_exps and curr_exp_id not in self._skip_exps:
            next_exp = strategy.experience.benchmark.train_stream[curr_exp_id + 1]  # get the next training experience
            # get the classes in the next training experience
            next_exp_classes = next_exp.classes_in_this_experience
            if not all([cls in self._target_map.keys() for cls in next_exp_classes]):
                raise RuntimeError("The provided dictionary does not contain the mapping for all targets contained in "
                                   "the successive experiences")
            # get the targets that the targets in the next training experience map to
            targets_mapped = [self._target_map[cls] for cls in next_exp_classes]
            # get the targets present in the current and previous training experiences
            previous_targets = strategy.experience.classes_seen_so_far
            if not all([t in previous_targets for t in targets_mapped]):
                raise RuntimeError("The provided dictionary maps some targets in the successive training experience to "
                                   "targets that are not present in the current or previous training experiences")
            # get the training dataset of the next experience and sets the train transformation as default
            next_dataset = next_exp.dataset.train()
            if not hasattr(next_dataset, "dataset_indices"):
                raise RuntimeError("The dataset of the next training experience must have the `dataset_indices` "
                                   "DataAttribute")
            # transform the targets in the successive training experience as determined by `self._target_map`
            next_dataset = next_dataset.update_data_attribute("targets",
                                                    [self._target_map[int(cls)] for cls in next_dataset.targets.data])
            # concat the training dataset of the current training experience with the one of the next train experience
            strategy.adapted_dataset = strategy.adapted_dataset.concat(next_dataset)
            # set the train transformation as default
            strategy.adapted_dataset = strategy.adapted_dataset.train()
            if not len(strategy.target_to_target) == 0:
                raise RuntimeError("`strategy.target_to_target` must be an empty dictionary prior to updating it")
            # update this dictionary with the targets present in the next training experience and the previous targets
            # they map to
            strategy.target_to_target = {t: val for t, val in self._target_map.items() if t in next_exp_classes}
            # save the `dataset_indices` DataAttribute of the samples in the next training experience
            self._successive_dataset_indices = torch.tensor(list(next_dataset.dataset_indices), dtype=torch.int64)
            if len(torch.unique(self._successive_dataset_indices)) < len(self._successive_dataset_indices):
                raise RuntimeError(
                    "One or more samples in the next training experience have a `dataset_indices` DataAttribute "
                    "identical to another sample in the same experience")

    def after_train_datasets_adaptation(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
                                        **kwargs):
        """
        Callback called after the train datasets adaptation.

        This callback appends the ID of the subnetworks present in `strategy.adapted_datasets.keys()` into
        `self._subnetwork_ids`.
        :param strategy: a strategy
        :param kwargs: some keyword arguments
        """
        self._subnetwork_ids.update([int(sub_id) for sub_id in strategy.adapted_dataset.keys()])

    def after_training_exp(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate, **kwargs):
        """
        Callback called after training on the current experience ends.

        It removes the samples injected from the successive training experience (if any) from the datasets of each
        subnetwork in `strategy.adapted_dataset`. It sets `self._successive_dataset_indices` to None.
        It sets `strategy.target_to_target` to an empty dictionary.
        :param strategy: a strategy
        :param kwargs: some keyword arguments
        """
        # `self._successive_dataset_indices` is None when there is no successive training experience or the current
        # training experience was to be skipped
        if self._successive_dataset_indices is not None:
            strategy.adapted_dataset = {sub_id: dataset.subset([i for i, data_index in enumerate(dataset.dataset_indices.data)
                                                                if int(data_index) not in self._successive_dataset_indices])
                                        for sub_id, dataset in strategy.adapted_dataset.items()}
        self._successive_dataset_indices = None
        strategy.target_to_target = {}

    def _get_incremental_classifier_prob_vecs(self, dataset_indices: torch.Tensor,
                                              dataset_type: Literal["train", "val", "test"], subnetwork_id: int,
                                              num_classes: int) -> torch.Tensor:
        """
        Get the probability vectors of the given samples using `self._prob_vec`

        .note::
            Safely assume that `dataset_indices` is a 1D tensor containing some elements and all these elements refer to
            samples injected from the next training experience, `dataset_type` is `train` and
            `subnetwork_id` is contained in `self._subnetwork_ids`.
            Some of these checks are performed by `get_incremental_classifier_prob_vecs`,  which successively invokes
            this method.

        .note:
            The returned tensor does not need to be allocated to the same device of `dataset_indices` because
            `get_incremental_classifier_prob_vecs` will do it.

        :param dataset_indices: a 1D tensor of the `dataset_indices` DataAttribute of the samples whose probability
            vectors must be retrieved.
        :param dataset_type: `train`, `val` or `test`. The type of dataset the given samples belong to.
        :param subnetwork_id: the id of the subnetwork the classes of the samples are allocated to. Note that all the
            classes of the samples *must* be allocated to the same subnetwork.
        :param num_classes: the number of classes assigned to the incremental classifier of the subnetwork with id
            `subnetwork_id`, i.e. the number of output nodes of the incremental classifier of the subnetwork with id
            `subnetwork_id`. If the size of the probability vectors returned does not match this number,
            a :class:`RuntimeError` is raised by `get_incremental_classifier_prob_vecs`.
        :return: a 2D tensor of probability vectors for the given samples.
        """
        if self._prob_vec is None:
            raise RuntimeError("It is impossible to retrieve the probability vectors of the given samples because no "
                               "callable was provided at initialisation")
        return self._prob_vec(dataset_indices, subnetwork_id, num_classes)

    def _contains_incremental_classifier_prob_vecs(self, dataset_indices: torch.Tensor,
                                                   dataset_type: Literal["train", "val", "test"],
                                                   subnetwork_id: int) -> torch.Tensor:
        """
        Check whether the given samples are samples injected from the training dataset of the next experience.

        .note::
            Safely assume that `dataset_indices` is a 1D tensor containing some elements, `dataset_type` is contained
            in `self._dataset_type` and `subnetwork_id` is contained in `self._subnetwork_ids`.
            These checks are performed by `contains_incremental_classifier_prob_vecs`, which successively invokes this
            method.

        .note:
            The returned tensor does not need to be allocated to the same device of `dataset_indices` because
            `contains_incremental_classifier_prob_vecs` will do it.

        :param dataset_indices: a 1D tensor of the `dataset_indices` DataAttribute of the samples whose probability
            vectors must be checked for presence.
        :param dataset_type: `train`, `val` or `test`. The type of dataset the given samples belong to.
        :param subnetwork_id: the id of the subnetwork the classes of the samples are allocated to. Note that all the
            classes of the samples *must* be allocated to the same subnetwork.
        :return: a 1D tensor of boolean values of same size as `dataset_indices` indicating whether the probability
            vectors of the respective samples are in the storage.
        """
        # if the samples belong to a train dataset and `self._successive_dataset_indices` is not None,
        # check whether the provided samples are samples injected from the successive training experience
        if dataset_type == "train" and self._successive_dataset_indices is not None:
            return torch.isin(dataset_indices.to(self._successive_dataset_indices.device),
                              self._successive_dataset_indices)
        else:
            # return a tensor of same size as `dataset_indices` containing only False if the samples are not samples
            # injected from the next training experience
            return torch.zeros(len(dataset_indices), dtype=torch.bool)


class GradientSchedulerPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate(
    DynamicNetworkInitFeatureExtractorIncrementalOutlierPlugin, supports_distributed=True):
    """
    Gradient scheduler plugin for :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`.

    This plugin schedules a periodic computation of gradients for each class at distinct optimisation points during the
    SGD training path. Specifically, the gradients of **all** the samples of **all** the classes encountered so far
    (both the classes in the current experience and the classes in the previously encountered experiences) are computed
    at distinct optimisation points. One can decide whether to compute the gradients for the samples in the
    train, validation or test datasets. Only one type of dataset (train, validation or test) can be selected.

    Given that computing the full-gradient of the loss is expensive, it requires a forward and backward pass,
    we compute the gradients by means of three different approximations.
    The first approximation computes the gradient of the standard multi-class cross entropy loss w.r.t the
    logits (raw scores before passing through the softmax layers) as described in
    "Coresets for Data-efficient Training of Machine Learning Models" and
    "Towards Sustainable Learning: Coresets for Data-efficient Deep Learning", which only requires a forward pass. The
    second approximation, which only requires a forward pass but is slightly more expensive to compute in comparison to
    the first approximation, is a better approximation as it computes the gradient of the standard multi-class cross
    entropy loss w.r.t the weights and biases in the classification head. The last approximation computes the gradient
    of the sum of the standard multi-class cross entropy loss between current logits and targets and the mean
    squared error loss between past and current logits w.r.t the weights and biases in the classification head.


    For all approximations, it is assumed that the last layer of the model is a classification head with Softmax
    activation function. Under this assumption, for a given sample, the gradient of the multi-class cross entropy loss
    w.r.t the logits is Y_hat - Y, the gradient of the multi-class cross entropy loss w.r.t the weights in the
    classification head is (Y_hat - Y) H^T, the gradient of the mean squared error loss w.r.t the weights in the
    classification head is 2/c (Z - T) H^T, the gradient of the multi-class cross entropy loss w.r.t the biases in the
    classification head is Y_hat - Y, the gradient of the mean squared error loss w.r.t the biases in the classification
    head is 2/c (Z - T), where Y_hat is the vector of softmax probabilities, Y is the one-hot encoding vector of the
    true class, H is the embedding vector of the given sample , also known as feature vector, being fed into the
    classification head, Z is the current logits vector output of the classification head prior to
    apply the softmax function, T is the past logits vector output and c is the dimensionality of T. If T exists and c
    is smaller than c', the dimensionality of Z, then Z-T is equal to (Z[:c] - T) || [0]*(c'-c), where || is the
    concatenation operator and [0]*(c'-c) is a vector containing c'-c zeros. Note that both (Y_hat - Y) H^T and
    2/c (Z - T) H^T are the outer product between two vectors resulting in a matrix. Note that for both the calculations
    of the gradient of the mean squared error loss w.r.t the weights and biases in
    the classification head, if the past logit vector T does not exist, then 2/c (Z-T) is set equal to [0]*c'.
    Consequently, the gradients of the mean squared error loss w.r.t the weights and biases in the classification head
    are a zero matrix and a zero vector, respectively, as if the mean squared error loss term for the given sample were
    discarded or if the past logit vector were equivalent to the current one. Finally, note that the gradients of the
    sum of the stand standard multi-class cross entropy loss and the mean squared error loss is just the sum of the
    two respective gradients, i.e. the gradients w.r.t the weights is
    (Y_hat - Y) H^T + 2/c (Z - T) H^T = [(Y_hat - Y) + 2/c (Z - T)]H^T and the gradients w.r.t the biases is
    (Y_hat - Y) + 2/c (Z - T).

    .note::
        This plugin *assumes* that the "dataset_indices" DataAttribute is present in the datasets of the **training**,
        **validation** and **test** experiences

    .note::
        If the third gradient approximation is used, an instance
        of :class:`LogitsStorageDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` *must* be in the `plugins`
        attribute of the strategy

    .note::
        This plugin metric *only* works when there is *only* one subnetwork

    .note::
        If the gradients must be computed for the samples in the validation or test datasets, every time the training
        process for a training experience with a given ID starts, the respective validation or test experience must be
        contained in `strategy._eval_streams`. Otherwise, a :class:`RuntimeError` is raised.

    .note::
        The class targets used in this plugin are not the real class targets but the order in which
        classes appear in the list of classes seen by the single subnetwork accessible through
        `strategy.model.subnetworks_classes[id_of_the_single_subnetwork]
    """
    def __init__(self, dataset_type: Literal["train", "val", "test"],
                 gradient_approx: Literal["logits", "class_head", "ce_mse_class_head"] = "logits",
                 frequency: Union[Tuple[int, int], int] = 0,
                 frequency_mode: Literal["epoch", "iteration", "both"] = "epoch", do_initial: bool = False,
                 save_space_gradients: bool = False):
        """
        Create a new GradientSchedulerPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate
        :param dataset_type: `train`, `val` or `test`. Decides whether to compute the gradients for the samples in the
            train, validation or test datasets.
        :param gradient_approx: the approximation to use for the gradient computation. `logits` to compute the gradient
            of the multi-class cross entropy loss w.r.t the logits. `class_head` to compute the gradient of the
            multi-class cross entropy loss w.r.t the weights and biases in the classification head.
            `ce_mse_class_head` to compute the gradient of the sum of the multi-class cross entropy loss and the mean
            squared error loss w.r.t the weights and biases in the classification head. If the latter is chosen,
            an instance of :class:`LogitsStorageDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` *must* be
            in the `plugins` attribute of the strategy. Default is `logits`.
        :param frequency: the frequency used to compute the gradients at different points in the optimisation
            space during the SGD training path. If a single integer number, this is the frequency used across epochs,
            iterations or both of them according to the value of `frequency_mode`. 0 means the gradients are
            computed only at the end of the SGD training path, i.e. the gradients are computed at the convergence
            point of the SGD training path. Values > 0 mean that the gradients are computed every `frequency`
            epochs, iterations or both of them according to `frequency_mode` and at the end of the SGD training path.
            -1 means the gradients are never computed during the SGD training path. When -1, `do_initial` must
            be set to True; otherwise a :class:`ValueError` is raised.
            If a 2-tuple, `frequency_mode` must be `both` and the first and second value must refer to the frequency
            used across epochs and iterations, respectively. Both values must be strictly positive (> 0) and the
            gradients are also computed at the end of the SGD training path.
            If both values are not strictly positive, a :class:`ValueError` is raised.
            If a 2-tuple and `frequency_mode` is not `both`, a :class:`ValueError` is raised. Default is 0.
        :param frequency_mode: `epoch`, `iteration` or `both`. Decides whether the computation of the gradients
            during the SGD training path should execute every `frequency` epochs, iterations or both.
            Default is `epoch`.
        :param do_initial: whether to compute the gradients before the SGD training path starts for the current
            experience, i.e. the gradients are computed using the initial weights. When `frequency` is set to -1,
            this parameter must be set to True; otherwise a :class:`ValueError` is raised. Default is False.
        :param save_space_gradients: if True and `gradient_approx` is `class_head`, the gradients of the multi-class
            cross entropy loss w.r.t the biases in the classification head and the sample embeddings (feature vectors)
            are stored to lower the memory footprint rather than storing the gradients of the multi-class
            cross entropy loss w.r.t. the weights and biases in the classification head. Note that it is possible to
            derive the gradient of the multi-class cross entropy loss w.r.t. the weights and biases in the
            classification head for a given sample by computing the outer product between the gradient of
            the multi-class cross entropy loss w.r.t the biases and the sample embedding and concatenating this with the
            gradient w.r.t. the biases. If True and `gradient_approx` is `ce_mse_class_head`, the gradients of the
            multi-class cross entropy loss and the mean squared error loss w.r.t the biases in the classification head
            are stored separately along with the sample embeddings (feature vectors) to lower the memory footprint. Note
            that it is possible to derive the gradient of the sum of the multi-class cross entropy loss and the mean
            squared error loss w.r.t. the weights and biases in the classification head for a given sample by first
            computing the sum between the gradients of the multi-class cross entropy loss and the mean squared error
            loss w.r.t the biases, denoted as G, subsequently compute the outer product between G and the sample
            embedding and finally concatenating the result of the outer product with G. Default is False.
        """
        if dataset_type not in ["train", "val", "test"]:
            raise ValueError("dataset_type must be `train`, `val` or `test`")
        if gradient_approx not in ["logits", "class_head", "ce_mse_class_head"]:
            raise ValueError("gradient_approx must be either `logits`, `class_head` or `ce_mse_class_head`")
        if frequency_mode not in ["epoch", "iteration", "both"]:
            raise ValueError("frequency_mode must be either `epoch` or `iteration` or `both`")
        if isinstance(frequency, tuple):
            if not frequency_mode == "both":
                raise ValueError("frequency must be a single integer number when frequency_mode is not both")
            if not len(frequency) == 2:
                raise ValueError("frequency must be a 2-tuple")
            if not (frequency[0] > 0 and frequency[1] > 0):
                raise ValueError("Both values of frequency must be strictly positive integers")
        else:
            if frequency < -1:
                raise ValueError("The frequency must be greater than or equal to -1")
            if frequency == -1 and do_initial is False:
                raise ValueError("do_initial must be True when frequency is set to -1")

        super().__init__()

        self.dataset_type: Literal["train", "val", "test"] = dataset_type
        """
        `train`,`val or `test`. Decides whether to compute the gradients for the samples in the
        train, validation or test datasets.
        """

        self.gradient_approx: Literal["logits", "class_head", "ce_mse_class_head"] = gradient_approx
        """
        the approximation to use for the gradient computation. `logits` to compute the gradient of the multi-class cross
        entropy loss w.r.t the logits. `class_head` to compute the gradient of the multi-class cross entropy loss w.r.t
        the weights and biases in the classification head. `ce_mse_class_head` to compute the gradient of the sum of the
        multi-class cross entropy loss and the mean squared error loss w.r.t the weights and biases in the
        classification head. If the latter, an instance
        of :class:`LogitsStorageDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` *must* be in the `plugins`
        attribute of the strategy.
        """

        self.frequency: Union[Tuple[int, int], int] = frequency
        """
        the frequency used to compute the gradients at different points in the optimisation
        space during the SGD training path. If a single integer number, this is the frequency used across epochs,
        iterations or both of them according to the value of `self.frequency_mode`. 0 means the gradients are
        computed only at the end of the SGD training path, i.e. the gradients are computed at the convergence
        point of the SGD training path. Values > 0 mean that the gradients are computed every `frequency`
        epochs, iterations or both of them according to `self.frequency_mode` and at the end of the SGD training path.
        -1 means the gradients are never computed during the SGD training path. When -1, `self.do_initial` must
        be set to True.
        If a 2-tuple, `self.frequency_mode` must be `both` and the first and second value must refer to the frequency
        used across epochs and iterations, respectively. Both values must be strictly positive (> 0) and the
        gradients are also computed at the end of the SGD training path.
        """

        self.frequency_mode: Literal["epoch", "iteration", "both"] = frequency_mode
        """
        `epoch`, `iteration` or `both`. Decides whether the computation of the gradients
        during the SGD training path should execute every `self.frequency` epochs, iterations or both.
        """

        self.do_initial: bool = do_initial
        """
        whether to compute the gradients before the SGD path starts for the current experience, i.e. the gradients
        are computed using the initial weights. It must be True when `self.frequency` is -1
        """

        self.save_space_gradients: bool = save_space_gradients
        """
        if True and `self.gradient_approx` is `class_head`, the gradients of the multi-class
        cross entropy loss w.r.t the biases in the classification head and the sample embeddings (feature vectors)
        are stored to lower the memory footprint rather than storing the gradients of the multi-class
        cross entropy loss w.r.t. the weights and biases in the classification head. Note that it is possible to
        derive the gradient of the multi-class cross entropy loss w.r.t. the weights and biases in the
        classification head for a given sample by computing the outer product between the gradient of
        the multi-class cross entropy loss w.r.t the biases and the sample embedding and concatenating this with the
        gradient w.r.t. the biases. If True and `self.gradient_approx` is `ce_mse_class_head`, the gradients of the
        multi-class cross entropy loss and the mean squared error loss w.r.t the biases in the classification head
        are stored separately along with the sample embeddings (feature vectors) to lower the memory footprint. Note
        that it is possible to derive the gradient of the sum of the multi-class cross entropy loss and the mean
        squared error loss w.r.t. the weights and biases in the classification head for a given sample by first
        computing the sum between the gradients of the multi-class cross entropy loss and the mean squared error
        loss w.r.t the biases, denoted as G, subsequently compute the outer product between G and the sample
        embedding and finally concatenating the result of the outer product with G.
        """

        self.all_gradients: Dict[int, List[Tuple[
            Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], np.ndarray],
            np.ndarray]]] = defaultdict(list)
        """
        a dictionary with target labels as keys and lists as values. Each element in the list is a distinct tuple
        computed at different time steps. Each tuple contains two elements. If `self.save_space_gradients` is True
        and `self.gradient_approx` is `class_head`, the first element is a tuple of two elements, where the first
        element is an array of gradients of the multi-class cross entropy loss w.r.t. the biases in the classification
        head and the second element is an array of embeddings (feature vectors). If `self.save_space_gradients` is True
        and `self.gradient_approx` is `ce_mse_class_head`, the first element is a tuple of three elements, where the
        first element is an array of gradients of the multi-class cross entropy loss w.r.t. the biases in the
        classification head, the second element is an array of gradients of the mean squared error loss w.r.t the biases
        in the classification head and the third element is an array of embeddings (feature vectors).
        Otherwise, if `self.save_space_gradients` is False or `self.gradient_approx` is `logits`, the first element is
        an array of gradients. The second element is always an array of respective indices. The target labels are not
        the real target labels but the order in which classes appear in the list of classes seen by the unique
        subnetwork accessible through `strategy.model.subnetworks_classes[sub_id].
        """

        self._exps: List = []
        """
        a list of the experiences encountered so far. They are training experiences, validation experiences or test
        experiences according to `self.dataset_type`. The order of the experiences in this list  reflects the order
        with which these experiences were encountered
        """

        self._gradient_metric: GradientLossWRTInputLastLayer = GradientLossWRTInputLastLayer()
        """
        the standalone metric for keeping track of the gradient of the multi-class cross entropy loss w.r.t. the logits
        for the samples in a dataset.
        """

        self._gradient_class_head_metric: GradientLossWRTWeightsBiasesClassificationHead = (
            GradientLossWRTWeightsBiasesClassificationHead())
        """
        the standalone metric for keeping track of the gradient of the multi-class cross entropy loss w.r.t. the weights
        and biases in the classification head.
        """

        self._gradient_ce_mse_class_head_metric: GradientCrossEntropyLossMSELossWRTWeightsBiasesClassificationHead = (
            GradientCrossEntropyLossMSELossWRTWeightsBiasesClassificationHead())
        """
        the standalone metric for keeping track of the gradient of the sum of the multi-class cross entropy loss and the
        mean squared error loss w.r.t. the weights and biases in the classification head.
        """

        self._eval_called: bool = False
        """boolean flag indicating whether `strategy.eval` was called by this plugin"""

    def before_training_exp(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
                            **kwargs):
        """
        Callback called before training on the new experience

        This callback appends the new training experience to `self._exps` if it is a training experience
        never encountered before and `self.dataset_type` is set to `train`.
        If `self.dataset_type` is `val` or `test`, the validation or test experience, respectively, with the same ID as
        the new training experience is appended to `self._exps`. The new validation or test experience is taken from
        `strategy._eval_streams`. A :class:`RuntimeError` is raised if `strategy._eval_streams` does not contain the
        new validation or test experience
        :param strategy: a strategy
        :param kwargs: some keyword arguments
        """
        if strategy.experience.origin_stream.name == "train":  # if it is a training experience
            new_exp_id = strategy.experience.current_experience  # get the ID of the training experience
            # if this experience has not been encountered before
            if new_exp_id not in [exp.current_experience for exp in self._exps]:
                if self.dataset_type == "train":
                    self._exps.append(strategy.experience)  # append the new training experience
                else:
                    # retrieve the val or test experience according to `self.dataset_type` with the same ID as the new
                    # training experience from `strategy._eval_streams`
                    # `strategy._eval_streams` is a list of lists
                    rspctive_exp = [exp for exp_list in strategy._eval_streams for exp in exp_list
                                if exp.origin_stream.name == self.dataset_type and exp.current_experience == new_exp_id]
                    if len(rspctive_exp) == 0:
                        raise RuntimeError(f"There is no respective {self.dataset_type} experience in "
                                           f"`strategy._eval_streams` for the new training experience")
                    self._exps.append(rspctive_exp[0])

    def after_train_datasets_adaptation(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
                              **kwargs):
        """
        Callback called after the train datasets adaptation

        This callback verifies that there is a single subnetwork by checking that `strategy.adapted_dataset` contains
        the dataset of a single subnetwork. If there are multiple subnetworks, this callback raises
        a :class`RuntimeError`.
        :param strategy: a strategy
        :param kwargs: some keyword arguments
        """
        if len(strategy.adapted_dataset) > 1:
            raise RuntimeError("This plugin only works when there is a single subnetwork")

    def before_training_subnetwork(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
                                   **kwargs):
        """
        Callback called before the training loop of the single subnetwork.

        Compute the gradients at the current optimisation point before the training loop starts if
        `self.do_initial` is True

        :param strategy: a strategy
        :param kwargs: some keyword arguments
        """
        if self.do_initial:
            self._compute_gradients(strategy, **kwargs)

    def after_training_epoch(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate, **kwargs):
        """
        Callback called at the end of each training epoch.

        If `self.frequency_mode` is `epoch` or `both`, the computation of the gradients is performed
        according to the value of `self.frequency` and `strategy.clock.train_exp_epochs`.
        """
        if self.frequency_mode in ["epoch", "both"]:
            frequency = self.frequency
            # if it is a tuple then get the first element
            if isinstance(frequency, tuple):
                frequency = frequency[0]
            self._maybe_compute_gradients(strategy, strategy.clock.train_exp_epochs, frequency, **kwargs)

    def after_training_iteration(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
                                 **kwargs):
        """
        Callback called at the end of each training iteration.

        If `self.frequency_mode` is `iteration` or `both`, the computation of the gradients is performed
        according to the value of `self.frequency` and `strategy.clock.train_exp_iterations`.
        """
        if self.frequency_mode in ["iteration", "both"]:
            frequency = self.frequency
            # if it is a tuple then get the second element
            if isinstance(frequency, tuple):
                frequency = frequency[1]
            self._maybe_compute_gradients(strategy, strategy.clock.train_exp_iterations, frequency, **kwargs)

    def after_training_subnetwork(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
                                  **kwargs):
        """
        Callback called at the end of the training loop of the single subnetwork.

        Compute the gradients at the end of the training loop of the single subnetwork.
        The computation of the gradients is not performed by this callback if `self.frequency` is -1 or
        the last periodic computation of the gradients was already performed at the convergence point.
        In contrast, the computation of the gradientss is performed if the last periodic computation was not
        performed at the convergence point or `self.frequency` is 0.
        """
        compute_final: bool = False
        if isinstance(self.frequency, tuple):
            counter = (strategy.clock.train_exp_epochs, strategy.clock.train_exp_iterations)
            if counter[0] % self.frequency[0] != 0 and counter[1] % self.frequency[1] != 0:
                compute_final = True
        elif self.frequency == 0:
            compute_final = True
        elif self.frequency > 0:
            if self.frequency_mode == "epoch":
                counter = strategy.clock.train_exp_epochs
            elif self.frequency_mode == "iteration":
                counter = strategy.clock.train_exp_iterations
            else:
                counter = (strategy.clock.train_exp_epochs, strategy.clock.train_exp_iterations)

            if isinstance(counter, tuple):
                if counter[0] % self.frequency != 0 and counter[1] % self.frequency != 0:
                    compute_final = True
            elif counter % self.frequency != 0:
                compute_final = True

        if compute_final:
            self._compute_gradients(strategy, **kwargs)

    def _maybe_compute_gradients(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
                                 counter, frequency: int, **kwargs):
        """
        Maybe compute the gradients for the single subnetwork.
        The computation is performed according to the counter and frequency provided.
        :param strategy: strategy
        :param counter: counter
        :param frequency: frequency
        :param kwargs: custom arguments
        """
        if frequency > 0 and counter % frequency == 0:
            self._compute_gradients(strategy, **kwargs)

    def _compute_gradients(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate, **kwargs):
        """
        Compute the gradients for the single subnetwork.

        This method calls `strategy.eval` to compute the gradients for each experience in `self._exps`.
        The `after_eval_iteration` callback is used to capture the gradients either of the multi-class cross entropy
        loss w.r.t the logits or w.r.t the weights and biases in the classification head, or of the sum of the
        multi-class cross entropy loss and mean squared error loss w.r.t the weights and biases in the classification
        head.

        During the `strategy.eval` calls performed by this callback, the metrics in `EvaluationPlugin` are not
        computed as they would be meaningless and their computation would only be a computational overhead.

        :param strategy: a strategy
        :param kwargs: custom arguments
        """
        self._eval_called = True  # indicate that the next `strategy.eval` calls are performed by this plugin
        # get the evaluator
        evaluator: EvaluationPlugin = [plg for plg in strategy.plugins if isinstance(plg, EvaluationPlugin)][0]
        # this will deactivate the evaluator, i.e. metrics are not computed during the next eval calls.
        evaluator.active = False
        for exp in self._exps:
            strategy.eval(exp, **kwargs)  # start the eval phase
            # after conclusion of the eval phase, get the collected gradients, targets and indices. They were
            # collected by the `after_eval_iteration` callback
            if self.gradient_approx == "logits":
                gradients, targets, indices = self._gradient_metric.result()
                self._gradient_metric.reset()  # reset the gradient metric to its initial state
            elif self.gradient_approx == "class_head":
                if self.save_space_gradients:
                    # also get the gradients wrt the biases and the embeddings
                    gradients, targets, indices, gradients_biases, embeds = self._gradient_class_head_metric.result(
                        retrieve_grad_biases_embeddings=True)
                else:  # only get the normal gradients
                    gradients, targets, indices = self._gradient_class_head_metric.result()

                self._gradient_class_head_metric.reset()  # reset to its initial state
            else:
                if self.save_space_gradients:
                    # also get the gradients wrt the biases and the embeddings
                    gradients, targets, indices, ce_gradients_biases, mse_gradients_biases, embeds = (
                        self._gradient_ce_mse_class_head_metric.result(retrieve_grad_biases_embeddings=True))
                else:  # only get the normal gradients
                    gradients, targets, indices = self._gradient_ce_mse_class_head_metric.result()

                self._gradient_ce_mse_class_head_metric.reset()  # reset to its initial state

            # get the unique targets and sort them into ascending order
            unique_targets = torch.unique(targets, sorted=True)
            for t in unique_targets:
                t = t.item()  # get the python value rather than a pytorch scalar tensor
                t_indices = indices[targets == t]  # get the indices of the given target
                indx_order = torch.argsort(t_indices)
                t_indices = t_indices[indx_order]  # sort the indices into ascending order
                # convert into numpy array, it should be unnecessary to call detach
                t_indices = t_indices.cpu().numpy()

                # if space needs to be saved and `self.gradient_approx` is `class_head` or `ce_mse_class_head`,
                # then store the gradients wrt the biases and the embeddings
                if self.gradient_approx in ["class_head", "ce_mse_class_head"] and self.save_space_gradients:
                    t_embeds = embeds[targets == t]  # get the embeddings of the given target
                    # sort the embeddings according to their respective sample index into ascending order
                    t_embeds = t_embeds[indx_order]
                    t_embeds = t_embeds.cpu().detach().numpy()  # convert into numpy array
                    if self.gradient_approx == "class_head":
                        t_gradients_biases = gradients_biases[targets == t]  # get the grads biases of the given target
                        # sort the gradients biases according to their respective sample index into ascending order
                        t_gradients_biases = t_gradients_biases[indx_order]
                        t_gradients_biases = t_gradients_biases.cpu().detach().numpy()  # convert into numpy array
                        self.all_gradients[t].append(((t_gradients_biases, t_embeds), t_indices))
                    else:
                        # get the cross entropy grads biases of the given target
                        t_ce_gradients_biases = ce_gradients_biases[targets == t]
                        # get the mean squared error grads biases of the given target
                        t_mse_gradients_biases = mse_gradients_biases[targets == t]
                        # sort the cross entropy gradients biases according to their respective sample index into
                        # ascending order
                        t_ce_gradients_biases = t_ce_gradients_biases[indx_order]
                        # sort the mean squared error gradients biases according to their respective sample index into
                        # ascending order
                        t_mse_gradients_biases = t_mse_gradients_biases[indx_order]
                        t_ce_gradients_biases = t_ce_gradients_biases.cpu().detach().numpy()  # convert into numpy array
                        # convert into numpy array
                        t_mse_gradients_biases = t_mse_gradients_biases.cpu().detach().numpy()
                        self.all_gradients[t].append(((t_ce_gradients_biases, t_mse_gradients_biases, t_embeds),
                                                      t_indices))

                else:  # otherwise, save the full gradient
                    t_gradients = gradients[targets == t]  # get the gradients of the given target
                    # sort the gradients according to their respective sample index into ascending order
                    t_gradients = t_gradients[indx_order]
                    t_gradients = t_gradients.cpu().detach().numpy()  # convert into numpy array
                    self.all_gradients[t].append((t_gradients, t_indices))

        self._eval_called = False  # reset the flag to false
        # active the evaluator back again, so that metrics are computed when eval is not called by this plugin
        evaluator.active = True

    def after_eval_iteration(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate, **kwargs):
        """
        Callback called after each eval iteration.

        Capture the gradients of the multi-class cross entropy loss w.r.t the logits or w.r.t the weights and biases in
        the classification head, or the gradients of the sum of the multi-class cross entropy loss and the mean squared
        error loss w.r.t the weights and biases in the classification head, the targets and the indices of the samples
        in the current mini-batch.
        This callback gets executed only if the current eval phase was called by this plugin.
        This can be achieved by looking at the value of `self._eval_called`. This callback **should not** be executed
        when the current eval phase was not called by this plugin.
        :param strategy: a strategy
        :param kwargs: some keyword arguments
        """
        if self._eval_called:
            curr_sub_id = strategy.curr_sub_id  # store the current state of the strategy's subnetwork id
            subs_id = list(strategy.mb_output.keys())
            if len(subs_id) > 1:
                raise RuntimeError("This plugin must be used only when there is one subnetwork")
            strategy.curr_sub_id = subs_id[0]
            num_classes = len(strategy.model.subnetworks_classes[strategy.curr_sub_id])
            if "dataset_indices" not in strategy.use_in_getitem_indices.keys():
                raise RuntimeError("The dataset of the current training experience must have the `dataset_indices` "
                                   "DataAttribute")
            indices = strategy.mb_dataset_indices
            if self.gradient_approx == "logits":
                self._gradient_metric.update(strategy.mb_output_incremental_classifier,
                                             strategy.mb_y_incremental_classifier, num_classes, indices=indices)
            elif self.gradient_approx == "class_head":
                self._gradient_class_head_metric.update(strategy.mb_output_incremental_classifier,
                                                        strategy.mb_y_incremental_classifier,
                                                        strategy.mb_feature_incremental_classifier,
                                                        num_classes, indices=indices)
            else:
                self._gradient_ce_mse_class_head_metric.update(strategy.mb_output_incremental_classifier,
                                                               strategy.mb_y_incremental_classifier,
                                                               strategy.mb_feature_incremental_classifier,
                                                               num_classes,
                                                               strategy.mb_has_incremental_classifier_logits,
                                                               strategy.mb_incremental_classifier_logits,
                                                               indices=indices)
            strategy.curr_sub_id = curr_sub_id  # re-store the state of the strategy's subnetwork id


class ClassWiseReplayFullDataAvgFullGradientSchedulerPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate(
    DynamicNetworkInitFeatureExtractorIncrementalOutlierPlugin, supports_distributed=True):
    """
    Class-wise replay and full-data average full-gradient scheduler plugin
    for :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`.

    This plugin schedules a periodic computation of *average* full-gradients (gradients wrt all parameters) for each
    class at distinct optimisation points during the SGD training path. For each class, two different average
    full-gradients are computed at each optimization point: (1) the average full-gradient across *all* training
    samples of that class and (2) the average full-gradient across the training samples of that class currently stored
    in the replay buffer. If a class has no training samples currently stored in the replay buffer then no gradients are
    tracked for it. Therefore, during the first incremental step, no gradients are tracked because no class has
    samples currently stored in the replay buffer.

    .note::
        This plugin *assumes* that the "dataset_indices" DataAttribute is present in the datasets of the **training**
        experiences

    .note::
        This plugin metric *only* works when there is *only* one subnetwork.

    .note::
        The full-gradients are computed wrt *all* parameters in `strategy.model`

    .note::
        This plugin *assumes* that the loss function in use computes the average value across the samples in the
        current batch

    .note::
        This plugin *assumes* that no same class is present in distinct experiences

    .note::
        The class targets used in this plugin are not the real class targets but the order in which
        classes appear in the list of classes seen by the single subnetwork accessible through
        `strategy.model.subnetworks_classes[id_of_the_single_subnetwork]
    """
    def __init__(self, storage_policy: ClassBalancedBufferDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
                 mb_size: Optional[int] = None, frequency: Union[Tuple[int, int], int] = 0,
                 frequency_mode: Literal["epoch", "iteration", "both"] = "epoch", do_initial: bool = False):
        """
        Create a new
        ClassWiseReplayFullDataAvgFullGradientSchedulerPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate
        :param storage_policy: the policy that controls how to add new exemplars in the external memory buffer. It must
        be an instance of :class:`ClassBalancedBufferDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`;
        otherwise a :class:`ValueError` is raised. It *must* be the same instance used in the underlying strategy.
        :param mb_size: mini-batch size used when computing the average full-gradients for each class. If None,
            `strategy.eval_mb_size` is used. Default is None.
        :param frequency: the frequency used to compute the full-gradients at different points in the optimisation
            space during the SGD training path. If a single integer number, this is the frequency used across epochs,
            iterations or both of them according to the value of `frequency_mode`. 0 means the full-gradients are
            computed only at the end of the SGD training path, i.e. the full-gradients are computed at the convergence
            point of the SGD training path. Values > 0 mean that the full-gradients are computed every `frequency`
            epochs, iterations or both of them according to `frequency_mode` and at the end of the SGD training path.
            -1 means the full-gradients are never computed during the SGD training path. When -1, `do_initial` must
            be set to True; otherwise a :class:`ValueError` is raised.
            If a 2-tuple, `frequency_mode` must be `both` and the first and second value must refer to the frequency
            used across epochs and iterations, respectively. Both values must be strictly positive (> 0) and the
            full-gradients are also computed at the end of the SGD training path.
            If both values are not strictly positive, a :class:`ValueError` is raised.
            If a 2-tuple and `frequency_mode` is not `both`, a :class:`ValueError` is raised. Default is 0.
        :param frequency_mode: `epoch`, `iteration` or `both`. Decides whether the computation of the full-gradients
            during the SGD training path should execute every `frequency` epochs, iterations or both.
            Default is `epoch`.
        :param do_initial: whether to compute the full-gradients before the SGD training path starts for the current
            experience, i.e. the gradients are computed using the initial weights. When `frequency` is set to -1,
            this parameter must be set to True; otherwise a :class:`ValueError` is raised. Default is False.
        """
        if not isinstance(storage_policy, ClassBalancedBufferDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate):
            raise ValueError("storage_policy must be an instance of "
                             "ClassBalancedBufferDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate")
        if frequency_mode not in ["epoch", "iteration", "both"]:
            raise ValueError("frequency_mode must be either `epoch` or `iteration` or `both`")
        if isinstance(frequency, tuple):
            if not frequency_mode == "both":
                raise ValueError("frequency must be a single integer number when frequency_mode is not both")
            if not len(frequency) == 2:
                raise ValueError("frequency must be a 2-tuple")
            if not (frequency[0] > 0 and frequency[1] > 0):
                raise ValueError("Both values of frequency must be strictly positive integers")
        else:
            if frequency < -1:
                raise ValueError("The frequency must be greater than or equal to -1")
            if frequency == -1 and do_initial is False:
                raise ValueError("do_initial must be True when frequency is set to -1")

        super().__init__()

        self.storage_policy: ClassBalancedBufferDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate = storage_policy
        """
        the policy that controls how to add new exemplars in the external memory buffer.
        It *must* be the same instance used in the underlying strategy.
        """

        self.mb_size: Optional[int] = mb_size
        """
        mini-batch size used when computing the average full-gradients for each class. If None,
        `strategy.eval_mb_size` is used.
        """

        self.frequency: Union[Tuple[int, int], int] = frequency
        """
        the frequency used to compute the full-gradients at different points in the optimisation
        space during the SGD training path. If a single integer number, this is the frequency used across epochs,
        iterations or both of them according to the value of `self.frequency_mode`. 0 means the full-gradients are
        computed only at the end of the SGD training path, i.e. the full-gradients are computed at the convergence
        point of the SGD training path. Values > 0 mean that the full-gradients are computed every `frequency`
        epochs, iterations or both of them according to `self.frequency_mode` and at the end of the SGD training path.
        -1 means the full-gradients are never computed during the SGD training path. When -1, `self.do_initial` must
        be set to True.
        If a 2-tuple, `self.frequency_mode` must be `both` and the first and second value must refer to the frequency
        used across epochs and iterations, respectively. Both values must be strictly positive (> 0) and the
        full-gradients are also computed at the end of the SGD training path.
        """

        self.frequency_mode: Literal["epoch", "iteration", "both"] = frequency_mode
        """
        `epoch`, `iteration` or `both`. Decides whether the computation of the full-gradients
        during the SGD training path should execute every `self.frequency` epochs, iterations or both.
        """

        self.do_initial: bool = do_initial
        """
        whether to compute the full-gradients before the SGD path starts for the current experience, i.e. the
        full-gradients are computed using the initial weights. It must be True when `self.frequency` is -1
        """

        self.all_gradients: Dict[int, List[np.ndarray]] = defaultdict(list)
        """
        a dictionary with target labels as keys and lists as values. Each element in the list is a distinct 2d array
        computed at different time steps. Each 2d array has size (2, number of network parameters), where the first row
        contains the average full-gradient across *all* training samples of that class and the second row contains the 
        the average full-gradient across the training samples of that class currently stored in the replay buffer.
        The target labels are not the real target labels but the order in which classes appear in the list of classes
        seen by the unique subnetwork accessible through `strategy.model.subnetworks_classes[sub_id]`.
        """

        self._exps: Dict[FrozenSet[int], CLExperience] = {}
        """
        a dictionary containing the training experiences encountered so far as values. The respective key of each
        experience is a set (frozenset because it is hashable) containing the target labels in the training dataset of
        that experience. The target labels are those output by the underlying CL benchmark
        """

        self._current_target_label_dataset_indices: Optional[Union[int, Set[int]]] = None
        """
        always None except when `strategy.eval_with_grad` is called by `self._compute_gradients`. When not None, 
        it is either an integer or a set of integers. If an integer, it is the target label for the class whose average
        full-gradient needs to be computed across all of its training samples. If a set of integers, it is a set
        containing the `dataset_indices` DataAttribute of those samples whose average full-gradient needs to be
        computed. These are all samples belonging to a specific class that are currently stored in the memory buffer.
        """

        self._average_full_gradient: Optional[Tuple[np.ndarray, int]] = None
        """
        always None except when `strategy.eval_with_grad` is called by `self._compute_gradients`. When not None, it is a
        2-tuple where the first element is a 1D numpy array of the average full-gradient computed so far for the samples
        specified by `self._current_target_label_dataset_indices` and the second element is the number of samples used
        for computing the average so far.
        """

        self._eval_called: bool = False
        """boolean flag indicating whether `strategy.eval_with_grad` or `strategy.eval` was called by this plugin"""

    def before_training_exp(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
                            **kwargs):
        """
        Callback called before training on the new experience

        This callback adds the new training experience, along with its set of target labels, to `self._exps` if it is a
        training experience never encountered before.
        :param strategy: a strategy
        :param kwargs: some keyword arguments
        """
        if strategy.experience.origin_stream.name == "train":  # if it is a training experience
            new_exp_id = strategy.experience.current_experience  # get the ID of the training experience
            # if this experience has not been encountered before
            if new_exp_id not in [exp.current_experience for exp in self._exps.values()]:
                key = frozenset(strategy.experience.dataset.targets)
                # append the new training experience along with its set of target labels
                self._exps[key] = strategy.experience

    def after_train_datasets_adaptation(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
                                        **kwargs):
        """
        Callback called after the train datasets adaptation

        This callback verifies that there is a single subnetwork by checking that `strategy.adapted_dataset` contains
        the dataset of a single subnetwork. If there are multiple subnetworks, this callback raises
        a :class`RuntimeError`.
        :param strategy: a strategy
        :param kwargs: some keyword arguments
        """
        if len(strategy.adapted_dataset) > 1:
            raise RuntimeError("This plugin only works when there is a single subnetwork")

    def before_training_subnetwork(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
                                   **kwargs):
        """
        Callback called before the training loop of the single subnetwork.

        Compute the full-gradients at the current optimisation point before the training loop starts if
        `self.do_initial` is True

        :param strategy: a strategy
        :param kwargs: some keyword arguments
        """
        if self.do_initial:
            self._compute_gradients(strategy, **kwargs)

    def after_training_epoch(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate, **kwargs):
        """
        Callback called at the end of each training epoch.

        If `self.frequency_mode` is `epoch` or `both`, the computation of the full-gradients is performed
        according to the value of `self.frequency` and `strategy.clock.train_exp_epochs`.
        """
        if self.frequency_mode in ["epoch", "both"]:
            frequency = self.frequency
            # if it is a tuple then get the first element
            if isinstance(frequency, tuple):
                frequency = frequency[0]
            self._maybe_compute_gradients(strategy, strategy.clock.train_exp_epochs, frequency, **kwargs)

    def after_training_iteration(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
                                 **kwargs):
        """
        Callback called at the end of each training iteration.

        If `self.frequency_mode` is `iteration` or `both`, the computation of the full-gradients is performed
        according to the value of `self.frequency` and `strategy.clock.train_exp_iterations`.
        """
        if self.frequency_mode in ["iteration", "both"]:
            frequency = self.frequency
            # if it is a tuple then get the second element
            if isinstance(frequency, tuple):
                frequency = frequency[1]
            self._maybe_compute_gradients(strategy, strategy.clock.train_exp_iterations, frequency, **kwargs)

    def after_training_subnetwork(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
                                  **kwargs):
        """
        Callback called at the end of the training loop of the single subnetwork.

        Compute the full-gradients at the end of the training loop of the single subnetwork.
        The computation of the full-gradients is not performed by this callback if `self.frequency` is -1 or
        the last periodic computation of the full-gradients was already performed at the convergence point.
        In contrast, the computation of the full-gradients is performed if the last periodic computation was not
        performed at the convergence point or `self.frequency` is 0.
        """
        compute_final: bool = False
        if isinstance(self.frequency, tuple):
            counter = (strategy.clock.train_exp_epochs, strategy.clock.train_exp_iterations)
            if counter[0] % self.frequency[0] != 0 and counter[1] % self.frequency[1] != 0:
                compute_final = True
        elif self.frequency == 0:
            compute_final = True
        elif self.frequency > 0:
            if self.frequency_mode == "epoch":
                counter = strategy.clock.train_exp_epochs
            elif self.frequency_mode == "iteration":
                counter = strategy.clock.train_exp_iterations
            else:
                counter = (strategy.clock.train_exp_epochs, strategy.clock.train_exp_iterations)

            if isinstance(counter, tuple):
                if counter[0] % self.frequency != 0 and counter[1] % self.frequency != 0:
                    compute_final = True
            elif counter % self.frequency != 0:
                compute_final = True

        if compute_final:
            self._compute_gradients(strategy, **kwargs)

    def _maybe_compute_gradients(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
                                 counter, frequency: int, **kwargs):
        """
        Maybe compute the full-gradients for the single subnetwork.
        The computation is performed according to the counter and frequency provided.
        :param strategy: strategy
        :param counter: counter
        :param frequency: frequency
        :param kwargs: custom arguments
        """
        if frequency > 0 and counter % frequency == 0:
            self._compute_gradients(strategy, **kwargs)

    def _compute_gradients(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate, **kwargs):
        """
        Compute the full-gradients for the single subnetwork.

        This method calls `strategy.eval_with_grad` to compute the full-gradients for each class that has
        samples stored in `self.storage_policy`. For each class having samples stored in `self.storage_policy`, the
        average full gradient across all of its training samples and that across its samples stored in
        `self.storage_policy` are computed.


        The `after_eval_iteration` callback is used to capture the average full-gradients of the current mini-batch.


        During the `strategy.eval_with_grad` calls performed by this callback, the metrics in `EvaluationPlugin` are not
        computed as they would be meaningless and their computation would only be a computational overhead.

        .note::
            This plugin *assumes* that the loss function in use computes the average value across the samples in the
            current batch

        .note::
            This plugin *assumes* that no same class is present in distinct experiences

        :param strategy: a strategy
        :param kwargs: custom arguments
        """
        self._eval_called = True  # indicate that the next `strategy.eval_with_grad` calls are performed by this plugin
        # get the evaluator
        evaluator: EvaluationPlugin = [plg for plg in strategy.plugins if isinstance(plg, EvaluationPlugin)][0]
        # this will deactivate the evaluator, i.e. metrics are not computed during the next eval calls.
        evaluator.active = False
        if self.mb_size is not None:
            # If not None,  store the eval_mb_size of the strategy and set it to the value of `self.mb_size`
            eval_mb_size = strategy.eval_mb_size
            strategy.eval_mb_size = self.mb_size

        for target_label, buffer in self.storage_policy.buffer_groups.items():
            dataset = buffer.buffer
            # if there are actually more than or equal to 1 exemplars stored for this class
            if len(dataset) > 0:
                # the `dataset_indices` DataAttribute of the sample of the given class in the memory buffer
                dataset_indices = set(getattr(dataset, "dataset_indices", None))
                if dataset_indices is None:
                    raise RuntimeError("The dataset of the memory buffer of the given class must have the "
                                       "`dataset_indices` DataAttribute")
                # a list containing the experiences containing the given class
                exps = [exp for target_set, exp in self._exps.items() if target_label in target_set]
                if len(exps) == 0:
                    raise RuntimeError("No experience in `self._exps` contains the given class")
                elif len(exps) > 1:
                    raise RuntimeError("Multiple experiences in `self._exps` contain the given class")
                # the unique experience containing the given class
                target_exp = exps[0]
                # a list that will contain the average gradient across all training samples of the given class in the
                # first index; the average gradient only across the samples currently stored in the memory buffer in the
                # second index
                avg_full_gradients: List[np.ndarray] = []
                # `strategy_eval_with_grad` is called twice. The first time to compute the average gradient across all
                # training samples of the given class; the second time to compute the average gradient only across the
                # samples currently stored in the memory buffer
                for n in range(2):
                    # set which target label full-gradients need to computed for if n is equal to 0; otherwise, set which
                    # exemplars of the given class are currently stored in the memory buffer if n is equal to 1
                    self._current_target_label_dataset_indices = target_label if n == 0 else dataset_indices
                    # start the eval phase enabled with gradient computation
                    strategy.eval_with_grad(target_exp, **kwargs)
                    # add the computed average full-gradient to `avg_full_gradients`
                    avg_full_gradients.append(self._average_full_gradient[0])
                    self._current_target_label_dataset_indices = None  # reset to None
                    self._average_full_gradient = None  # reset to None

                # get the classes seen by the single subnetwork
                seen_classes = list(strategy.model.subnetworks_classes.values())[0]
                # get the index in which the given target label appears in the list of classes seen by the single
                # subnetwork
                target_index = seen_classes.index(target_label)
                # add the average full-gradients
                self.all_gradients[target_index].append(np.stack(avg_full_gradients, axis=0))

        self._eval_called = False  # reset the flag to false
        # active the evaluator back again, so that metrics are computed when eval is not called by this plugin
        evaluator.active = True
        if self.mb_size is not None:
            # restore to its previous value
            strategy.eval_mb_size = eval_mb_size

    def after_eval_dataset_adaptation(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
                                      **kwargs):
        """
        Callback called after each eval dataset adaptation.
        This callback gets executed only if the current eval phase was called by this plugin.
        This is achieved by looking at the value of `self._eval_called`. This callback **should not** be executed
        when the current eval phase was not called by this plugin.

        This callback modifies `strategy.adapted_dataset` by only retaining specific samples according to the value of
        `self._current_target_label_dataset_indices`. If `self._current_target_label_dataset_indices` is a target label,
        only those samples belonging to the given target label are retained in `strategy.adapted_dataset`. Otherwise,
        if `self._current_target_label_dataset_indices` is a set of `dataset_indices` DataAttribute, only those samples
        with the given `dataset_indices` DataAttribute are retained.
        :param strategy: a strategy
        :param kwargs: some keyword arguments
        """
        if self._eval_called:
            if self._current_target_label_dataset_indices is None:
                raise RuntimeError("`self._current_target_label_dataset_indices` cannot be None")
            elif isinstance(self._current_target_label_dataset_indices, int):
                # only preserve those samples with the given target label
                target_label = self._current_target_label_dataset_indices
                strategy.adapted_dataset = strategy.adapted_dataset.subset(
                    [i for i, cls in enumerate(strategy.adapted_dataset.targets.data) if int(cls) == target_label]
                )
            elif isinstance(self._current_target_label_dataset_indices, set):
                # only preserve those samples with specific `dataset_indices` DataAttributes
                dataset_indices = self._current_target_label_dataset_indices
                strategy.adapted_dataset = strategy.adapted_dataset.subset(
                    [i for i, d_index in enumerate(strategy.adapted_dataset.dataset_indices.data)
                     if int(d_index) in dataset_indices]
                )
            else:
                raise RuntimeError("`self._current_target_label_dataset_indices` is of an unexpected type")

    def after_eval_iteration(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate, **kwargs):
        """
        Callback called after each eval iteration.
        This callback gets executed only if the current eval phase was called by this plugin.
        This can be achieved by looking at the value of `self._eval_called`. This callback **should not** be executed
        when the current eval phase was not called by this plugin.

        Capture the average full-gradient of the samples in the current mini-batch and updates
        `self._average_full_gradient` accordingly

        .note::
            It is *assumed* that the loss function in use computes the average value across the samples in the
            current batch
        :param strategy: a strategy
        :param kwargs: some keyword arguments
        """
        if self._eval_called:
            curr_sub_id = strategy.curr_sub_id  # store the current state of the strategy's subnetwork id
            subs_id = list(strategy.mb_output.keys())
            if len(subs_id) > 1:
                raise RuntimeError("This plugin must be used only when there is one subnetwork")
            strategy.curr_sub_id = subs_id[0]
            # get the number of samples in the current mini-batch
            n_samples = len(strategy.mb_y_incremental_classifier)
            # get the loss
            loss = strategy.loss[strategy.curr_sub_id]
            # grads is a tuple of tensors, one tensor for each model parameter provided as input
            grads = torch.autograd.grad(loss, list(strategy.model.parameters()), materialize_grads=True)
            # flatten each grad in grads, convert it to a numpy array and then concatenate all of them in a single 1d
            # array
            avg_grad = np.concatenate([torch.flatten(grad).detach().cpu().numpy() for grad in grads])
            if self._average_full_gradient is None:
                self._average_full_gradient = (avg_grad, n_samples)
            else:
                avg_grad_so_far, n_samples_so_far = self._average_full_gradient
                avg_grad_so_far = (avg_grad_so_far * n_samples_so_far + avg_grad * n_samples) / (n_samples_so_far + n_samples)
                n_samples_so_far = n_samples + n_samples_so_far
                self._average_full_gradient = (avg_grad_so_far, n_samples_so_far)

            strategy.curr_sub_id = curr_sub_id  # re-store the state of the strategy's subnetwork id


class GradientBasedReplayBufferSelectionSGDPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate(
    BufferDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
    DynamicNetworkInitFeatureExtractorIncrementalOutlierPlugin,
    supports_distributed=True):
    """
    Gradient-based replay buffer selection over SGD optimisation parameters storage policy and plugin
    for :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`.

    This storage policy stores samples for replay, equally divided over classes. There is a separate buffer for each
    class. The number of classes can be fixed up front or adaptive, based on the 'adaptive_size' attribute.
    When adaptive, the memory is equally divided over all the unique classes observed so far. Replay-buffer selection
    is framed as a gradient-based optimization problem over randomly sampled optimisation parameters as described in the
    first progression review report. In this storage policy, the SGD training path is used as sampling policy.
    This storage policy is also a plugin for :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` that
    implements several callbacks to capture the gradients of each sample along the SGD training path.

    This plugin schedules a periodic computation of gradient-based distance kernels for each class at distinct
    optimisation points during the SGD training path. A gradient-based distance kernel is a square symmetric matrix
    where the ith, jth element is the distance between the gradients of the ith and jth samples of the given class at
    the given optimisation point. Currently, the supported metrics for computing the distance between two gradient
    vectors are: `euclidean` and `mahalanobis`. Given that computing the full-gradient is expensive, it requires a
    forward and backward pass, we compute the gradient by means of three different approximations.
    The first approximation computes the gradient of the standard multi-class cross entropy loss w.r.t the
    logits (raw scores before passing through the softmax layers) as described in
    "Coresets for Data-efficient Training of Machine Learning Models" and
    "Towards Sustainable Learning: Coresets for Data-efficient Deep Learning", which only requires a forward pass. The
    second approximation, which only requires a forward pass but is slightly more expensive to compute in comparison to
    the first approximation, is a better approximation as it computes the gradient of the standard multi-class cross
    entropy loss w.r.t the weights and biases in the classification head. The last approximation computes the gradient
    of the sum of the standard multi-class cross entropy loss between current logits and targets and the mean
    squared error loss between past and current logits w.r.t the weights and biases in the classification head.
    For all approximations, it is assumed that the last layer of the model is a classification head with Softmax
    activation function. Under this assumption, for a given sample, the gradient of the multi-class cross entropy loss
    w.r.t the logits is Y_hat - Y, the gradient of the multi-class cross entropy loss w.r.t the weights in the
    classification head is (Y_hat - Y) H^T, the gradient of the mean squared error loss w.r.t the weights in the
    classification head is 2/c (Z - T) H^T, the gradient of the multi-class cross entropy loss w.r.t the biases in the
    classification head is Y_hat - Y, the gradient of the mean squared error loss w.r.t the biases in the classification
    head is 2/c (Z - T), where Y_hat is the vector of softmax probabilities, Y is the one-hot encoding vector of the
    true class, H is the embedding vector of the given sample , also known as feature vector, being fed into the
    classification head, Z is the current logits vector output of the classification head prior to
    apply the softmax function, T is the past logits vector output and c is the dimensionality of T. If T exists and c
    is smaller than c', the dimensionality of Z, then Z-T is equal to (Z[:c] - T) || [0]*(c'-c), where || is the
    concatenation operator and [0]*(c'-c) is a vector containing c'-c zeros. Note that both (Y_hat - Y) H^T and
    2/c (Z - T) H^T are the outer product between two vectors resulting in a matrix. Note that for both the calculations
    of the gradient of the mean squared error loss w.r.t the weights and biases in
    the classification head, if the past logit vector T does not exist, then 2/c (Z-T) is set equal to [0]*c'.
    Consequently, the gradients of the mean squared error loss w.r.t the weights and biases in the classification head
    are a zero matrix and a zero vector, respectively, as if the mean squared error loss term for the given sample were
    discarded or if the past logit vector were equivalent to the current one. Finally, note that the gradients of the
    sum of the stand standard multi-class cross entropy loss and the mean squared error loss is just the sum of the
    two respective gradients, i.e. the gradients w.r.t the weights is
    (Y_hat - Y) H^T + 2/c (Z - T) H^T = [(Y_hat - Y) + 2/c (Z - T)]H^T and the gradients w.r.t the biases is
    (Y_hat - Y) + 2/c (Z - T).

    When the `update` method of this storage policy is called, all the gradient-based distance kernels that have been
    collected during the training of the current experience are used to compute an aggregate similarity kernel for each
    class. One can choose between two types of aggregate similarity kernels: the minimal similarity kernel and the
    mean similarity kernel. If the minimal similarity kernel is selected, for each class, the ith, jth element in
    the respective minimal similarity kernel is the minimal similarity between the gradients of the ith and jth samples
    of the given class encountered during the SGD training path at selected optimisation points where a gradient-based
    distance kernel was computed. In contrast, if the mean similarity kernel is selected, for each class,
    the ith, jth element in the respective mean similarity kernel is the mean similarity between the gradients
    of the ith and jth samples of the given class encountered during the SGD training path at selected optimisation
    points where a gradient-based distance kernel was computed. The minimal similarity kernel is
    computed by first computing a maximal distance kernel and then transforming it into a minimal similarity kernel.
    Each ith, jth value in the maximal distance kernel is the maximum distance between the gradients of the ith and jth
    samples of the given class encountered during the SGD training path at selected optimisation points where a
    gradient-based distance kernel was computed. In the case of `euclidean` and `mahalanobis`, the maximal distance
    kernel is converted into the minimal similarity kernel by setting the value of each ith, jth element to m - vij,
    where m is the largest distance in the maximal distance kernel and vij is the value of the ith, jth element in the
    maximal distance kernel. The mean similarity kernel is computed by first computing a mean distance kernel and
    then transforming it into a mean similarity kernel. Each ith, jth value in the mean distance kernel is the
    mean distance between the gradients of the ith and jth samples of the given class encountered during the
    SGD training path at selected optimisation points where a gradient-based distance kernel was computed. In the case
    of `euclidean` and `mahalanobis`, the mean distance kernel is converted into the mean similarity kernel by
    setting the value of each ith, jth element to m - vij, where m is the largest distance in the mean distance
    kernel and vij is the value of the ith, jth element in the mean distance kernel.

    The aggregate similarity kernel of each class is then used to select the samples that must be preserved in the
    replay buffer of the given class along with their marginal gains and weights as described in
    "Coresets for Data-efficient Training of Machine Learning Models" and
    "Towards Sustainable Learning: Coresets for Data-efficient Deep Learning". This procedure is implemented in
    :class:`coreset.coreset.CRAIGCoreset`. In alternative, one can use
    a :class:`SubsetGeneratorGradientBasedReplayBufferSelection` to divide the
    samples of a given class into `n` disjoint subsets, select for each subset the samples that must be preserved in
    the replay buffer along with their marginal gains and weights as described in
    "Coresets for Data-efficient Training of Machine Learning Models" and
    "Towards Sustainable Learning: Coresets for Data-efficient Deep Learning" and then concatenate the samples
    preserved for each subgroup together, forming the overall set of samples that is preserved for the given class.

    .note::
        This plugin *assumes* that the "dataset_indices" DataAttribute is present in the datasets of the training
        experiences

    .note::
        If the third gradient approximation is used, an instance
        of :class:`LogitsStorageDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` *must* be in the `plugins`
        attribute of the strategy

    .note::
        This plugin metric *only* works when there is *only* one subnetwork

    .note::
        The class targets used in this plugin are not the real class targets but the order in which
        classes appear in the list of classes seen by the single subnetwork accessible through
        `strategy.model.subnetworks_classes[id_of_the_single_subnetwork]

    .note::
        This class acts both as a plugin and a storage policy. Therefore, the same instance of this class must be
        provided to the list of plugins when
        instantiating :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` and to the `storage_policy`
        parameter when instantiating :class:`ReplayPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
        or :class:`ReplayNoOutlierPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
    """
    def __init__(self, max_size: int, adaptive_size: bool = True, total_num_classes: Optional[int] = None,
                 metric: Literal["euclidean", "mahalanobis"] = "euclidean",
                 gradient_approx: Literal["logits", "class_head", "ce_mse_class_head"] = "logits",
                 aggregate_similarity_kernel: Literal["min", "mean"] = "min", num_subsets: int = 1,
                 subset_generator: SubsetGeneratorGradientBasedReplayBufferSelection = RandomSubsetGenerator(),
                 frequency: Union[Tuple[int, int], int] = 0,
                 frequency_mode: Literal["epoch", "iteration", "both"] = "epoch", do_initial: bool = False,
                 recompute_coreset: Optional[bool] = False, store_all_distance_kernels: bool = False,
                 store_all_coresets: bool = False, store_all_gradients: bool = False,
                 compute_all_past_gradients: bool = False, save_space_gradients: bool = False,
                 coreset_mode: Literal["dense", "sparse", "clustered"] = "dense",
                 no_coreset_exps: Optional[Union[int, List[int]]] = None,
                 **kwargs):
        """
        Create a new GradientBasedReplayBufferSelectionPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate
        :param max_size: the maximum capacity of the replay buffer.
        :param adaptive_size: whether to divide the capacity of the replay buffer equally over all classes observed so
            far. Default is True.
        :param total_num_classes: if `adaptive_size` is False, the upfront total number of classes to divide capacity
            over. Default is None.
        :param metric: the metric to be used to compute the distance kernels. The metrics currently supported are:
            `euclidean` and `mahalanobis`. Default is `euclidean`.
        :param gradient_approx: the approximation to use for the gradient computation. `logits` to compute the gradient
            of the multi-class cross entropy loss w.r.t the logits. `class_head` to compute the gradient of the
            multi-class cross entropy loss w.r.t the weights and biases in the classification head.
            `ce_mse_class_head` to compute the gradient of the sum of the multi-class cross entropy loss and the mean
            squared error loss w.r.t the weights and biases in the classification head. If the latter is chosen,
            an instance of :class:`LogitsStorageDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` *must* be
            in the `plugins` attribute of the strategy. Default is `logits`.
        :param aggregate_similarity_kernel: `min` or `mean`. Use `min` to aggregate the distance-kernels of a class
            collected during the SGD training path into a minimal similarity kernel. Use `mean` to aggregate the
            distance-kernels of a class collected during the SGD training path into a mean similarity kernel.
            Default is `min`.
        :param num_subsets: the number of disjoint subsets to be used to select the samples that must be
            preserved in the replay buffer for each class. If 1, all samples of a given class are used to select the
            class coreset. If >1, the samples of a given class are divided into the given number of disjoint
            subsets according to the provided `subset_generator`, a coreset is computed for each subset and the class
            coreset is equal to the union of the coresets found for each subset. Default is 1.
        :param subset_generator: a subset generator that divides the samples of a given class into `num_subsets`
            disjoint subsets. Default is the random subset generator, which randomly divides the samples
            of a given class into `num_subsets` disjoint subsets.
        :param frequency: the frequency used to compute the distance kernels at different points in the optimisation
            space during the SGD training path. If a single integer number, this is the frequency used across epochs,
            iterations or both of them according to the value of `frequency_mode`. 0 means the distance kernels are
            computed only at the end of the SGD training path, i.e. the distance kernels are computed at the convergence
            point of the SGD training path. Values > 0 mean that the distance kernels are computed every `frequency`
            epochs, iterations or both of them according to `frequency_mode` and at the end of the SGD training path.
            -1 means the distance kernels are never computed during the SGD training path. When -1, `do_initial` must
            be set to True; otherwise a :class:`ValueError` is raised.
            If a 2-tuple, `frequency_mode` must be `both` and the first and second value must refer to the frequency
            used across epochs and iterations, respectively. Both values must be strictly positive (> 0) and the
            distance kernels are also computed at the end of the SGD training path.
            If both values are not strictly positive, a :class:`ValueError` is raised.
            If a 2-tuple and `frequency_mode` is not `both`, a :class:`ValueError` is raised. Default is 0.
        :param frequency_mode: `epoch`, `iteration` or `both`. Decides whether the computation of the distance kernels
            during the SGD training path should execute every `frequency` epochs, iterations or both.
            Default is `epoch`.
        :param do_initial: whether to compute the distance kernels before the SGD training path starts for the current
            experience, i.e. the distance kernels are computed using the initial weights. When `frequency` is set to -1,
            this parameter must be set to True; otherwise a :class:`ValueError` is raised. Default is False.
        :param recompute_coreset: whether the coreset of old classes should be recomputed during the current SGD path
            based on the samples currently stored in the replay buffer. This parameter must be used only when
            `adaptive_size` is True. Otherwise, this parameter must be set to None. Assuming `adaptive_size` is True,
            if True, the distance kernels for the old classes are computed based on the samples currently
            stored in the replay buffer. If False, the samples of old classes that are discarded to make space for the
            samples of new classes are the ones with the lowest facility location marginal gain as computed when the
            coreset of that class was found. Default is False.
        :param store_all_distance_kernels: if True, all distance kernels are stored in `self.all_distance_kernels`.
            Default is False.
        :param store_all_coresets: if True, all coresets are stored in `self.all_coresets`. Default is False.
        :param store_all_gradients: if True, all gradients are stored in `self.all_gradients`. Default is False.
        :param compute_all_past_gradients: if True, `store_all_gradients` is True and `recompute_coreset` is True,
            the gradients of all past samples are computed and stored. Otherwise, if False, `store_all_gradients` is
            True and `recompute_coreset` is True, only the gradients of those past samples currently stored in the
            replay buffer are computed and stored. Changing this value when `store_all_gradients` is False or
            `recompute_coreset` is False does not have any effect. Default is False.
        :param save_space_gradients: if True, `gradient_approx` is `class_head` and `store_all_gradients` is True,
            the gradients of the multi-class cross entropy loss w.r.t the biases in the classification head and the
            sample embeddings (feature vectors) are stored to lower the memory footprint rather than storing the
            gradients of the multi-class cross entropy loss w.r.t. the weights and biases in the classification head.
            Note that it is possible to derive the gradient of the multi-class cross entropy loss w.r.t. the weights
            and biases in the classification head for a given sample by computing the outer product between the gradient
            of the multi-class cross entropy loss w.r.t the biases and the sample embedding and concatenating this with
            the gradient w.r.t. the biases. If True, `gradient_approx` is `ce_mse_class_head` and `store_all_gradients`
            is True, the gradients of the multi-class cross entropy loss and the mean squared error loss w.r.t the
            biases in the classification head are stored separately along with the sample embeddings (feature vectors)
            to lower the memory footprint. Note that it is possible to derive the gradient of the sum of the multi-class
            cross entropy loss and the mean squared error loss w.r.t. the weights and biases in the classification head
            for a given sample by first computing the sum between the gradients of the multi-class cross entropy loss
            and the mean squared error loss w.r.t the biases, denoted as G, subsequently compute the outer product
            between G and the sample embedding and finally concatenating the result of the outer product with G.
            Default is False.
        :param coreset_mode: `dense`, `sparse` or `clustered`. It specifies whether :class:`FacilityLocationFunction`
            should operate in dense mode (using a dense similarity kernel), sparse mode (using a sparse similarity
            kernel) or clustered mode (evaluating over clusters). Check the documentation
            of :meth:`CRAIGCorset.select_coreset` for more info about this parameter. Default is `dense`.
        :param no_coreset_exps: the ID or a list of IDs of the experiences whose coresets should not be computed
            at any time step. These experiences are totally ignored as if they were never encountered.
            If the current training experience is an experience to be ignored, then no computation of distance kernels
            is scheduled during the SGD path and no update to the buffer is performed at the end of training. This holds
            because no new samples are added to the buffer. If the current experience is not to be ignored,
            `recompute_coreset` is True and experiences to be ignored were encountered in the past, the computation of
            the distance kernels is performed for all the past experiences + the current experience except for the past
            experiences to be ignored.
        :param kwargs: a set of keyword arguments to be used to compute the coresets. They are passed
            to :meth:`CRAIGCorset.select_coreset`. Therefore, have a look at the documentation
            of :meth:`CRAIGCorset.select_coreset` to check which keyword arguments can be passed. Note that the argument
            `metric` should not be passed.
        """
        if adaptive_size:
            if recompute_coreset is None:
                raise ValueError("recompute_coreset cannot be None when adaptive_size is True")
            if total_num_classes is not None:
                raise ValueError("total_num_classes must be None when adaptive_size is True")
        else:
            if recompute_coreset is not None:
                raise ValueError("recompute_coreset must be None when adaptive_size is False")
            if total_num_classes is None:
                raise ValueError("total_num_classes cannot be None when adaptive_size is False")
            if not total_num_classes > 0:
                raise ValueError("total_num_classes must be strictly positive when adaptive_size is False")
        if max_size <= 0:
            raise ValueError('The max_size must be greater than 0')
        if metric not in ["euclidean", "mahalanobis"]:
            raise ValueError("The only supported metrics are `euclidean` and `mahalanobis`")
        if gradient_approx not in ["logits", "class_head", "ce_mse_class_head"]:
            raise ValueError("gradient_approx must be either `logits`, `class_head` or `ce_mse_class_head`")
        if aggregate_similarity_kernel not in ["min", "mean"]:
            raise ValueError("aggregate_similarity_kernel must be either `min` or `mean`")
        if num_subsets < 1:
            raise ValueError("num_subsets must be greater than or equal to 1")
        if not isinstance(subset_generator, SubsetGeneratorGradientBasedReplayBufferSelection):
            raise ValueError("subset_generator must be of type SubsetGeneratorGradientBasedReplayBufferSelection")
        if frequency_mode not in ["epoch", "iteration", "both"]:
            raise ValueError("frequency_mode must be either `epoch` or `iteration` or `both`")
        if isinstance(frequency, tuple):
            if not frequency_mode == "both":
                raise ValueError("frequency must be a single integer number when frequency_mode is not both")
            if not len(frequency) == 2:
                raise ValueError("frequency must be a 2-tuple")
            if not (frequency[0] > 0 and frequency[1] > 0):
                raise ValueError("Both values of frequency must be strictly positive integers")
        else:
            if frequency < -1:
                raise ValueError("The frequency must be greater than or equal to -1")
            if frequency == -1 and do_initial is False:
                raise ValueError("do_initial must be True when frequency is set to -1")

        super().__init__(max_size)  # this calls the __init__ of `BufferDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`

        self.adaptive_size = adaptive_size
        """
        whether to divide the capacity of the replay buffer equally over all classes observed so far
        """

        self.total_num_groups = total_num_classes
        """
        up front total number of groups
        """

        self.total_num_classes = total_num_classes
        """
        up front total number of classes
        """

        self.seen_classes: Set[int] = set()
        """
        class targets observed so far. The target labels are not the real target labels but the order in which classes
        appear in the list of classes seen by the unique subnetwork accessible through
        `strategy.model.subnetworks_classes[sub_id]
        """

        self.buffer_groups: Dict[int, AvalancheDataset] = {}
        """
        a dictionary with target labels as keys and their respective buffer datasets as values. The target
        labels are not the real target labels but the order in which classes appear in the list of classes seen by the
        unique subnetwork accessible through `strategy.model.subnetworks_classes[sub_id].
        """

        self.metric: Literal["euclidean", "mahalanobis"] = metric
        """
        the metric to be used to compute the distance kernels. The only metrics currently supported are: `euclidean` and
        `mahalanobis`
        """

        self.gradient_approx: Literal["logits", "class_head", "ce_mse_class_head"] = gradient_approx
        """
        the approximation to use for the gradient computation. `logits` to compute the gradient of the multi-class cross
        entropy loss w.r.t the logits. `class_head` to compute the gradient of the multi-class cross entropy loss w.r.t
        the weights and biases in the classification head. `ce_mse_class_head` to compute the gradient of the sum of the
        multi-class cross entropy loss and the mean squared error loss w.r.t the weights and biases in the
        classification head. If the latter, an instance
        of :class:`LogitsStorageDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` *must* be in the `plugins`
        attribute of the strategy.
        """

        self.aggregate_similarity_kernel: Literal["min", "mean"] = aggregate_similarity_kernel
        """
        `min` or `mean`. Use `min` to aggregate the distance-kernels of a class collected during the SGD training path
        into a minimal similarity kernel. Use `mean` to aggregate the distance-kernels of a class collected during the
        SGD training path into a mean similarity kernel.
        """

        self.num_subsets: int = num_subsets
        """
        the number of disjoint subsets to be used to select the samples that must be preserved in the replay
        buffer for each class. If 1, all samples of a given class are used to select the class coreset. If >1, the
        samples of a given class are divided into the given number of disjoint subsets according to `self.num_subsets`,
        a coreset is computed for each subset and the class coreset is equal to the union of the coresets found for
        each random subset.
        """

        self.subset_generator: SubsetGeneratorGradientBasedReplayBufferSelection = subset_generator
        """
        a subset generator that divides the samples of a given class into `self.num_subsets` disjoint subsets
        """

        self.frequency: Union[Tuple[int, int], int] = frequency
        """
        the frequency used to compute the distance kernels at different points in the optimisation
        space during the SGD training path. If a single integer number, this is the frequency used across epochs,
        iterations or both of them according to the value of `self.frequency_mode`. 0 means the distance kernels are
        computed only at the end of the SGD training path, i.e. the distance kernels are computed at the convergence
        point of the SGD training path. Values > 0 mean that the distance kernels are computed every `frequency`
        epochs, iterations or both of them according to `self.frequency_mode` and at the end of the SGD training path.
        -1 means the distance kernels are never computed during the SGD training path. When -1, `self.do_initial` must
        be set to True.
        If a 2-tuple, `self.frequency_mode` must be `both` and the first and second value must refer to the frequency
        used across epochs and iterations, respectively. Both values must be strictly positive (> 0) and the
        distance kernels are also computed at the end of the SGD training path.
        """

        self.frequency_mode: Literal["epoch", "iteration", "both"] = frequency_mode
        """
        `epoch`, `iteration` or `both`. Decides whether the computation of the distance kernels
        during the SGD training path should execute every `self.frequency` epochs, iterations or both.
        """

        self.do_initial: bool = do_initial
        """
        whether to compute the distance kernels before the SGD path starts for the current experience, i.e. the distance
        kernels are computed using the initial weights. It must be True when `self.frequency` is -1
        """

        self.recompute_coreset: bool = recompute_coreset
        """
        whether the coreset of old classes should be recomputed during the current SGD path based on the samples
        currently stored in the replay buffer. Then, if True, distance kernels for the old classes are
        computed based on the samples currently stored in the replay buffer. If False, the samples of old classes that
        are discarded to make space for the samples of new classes are selected as the ones with the lowest facility
        location marginal gain as computed when the coreset of that class was found
        """

        self.store_all_distance_kernels: bool = store_all_distance_kernels
        """
        if True, all distance kernels are stored in `self.all_distance_kernels`. Otherwise, they are not stored.
        """

        self.store_all_coresets: bool = store_all_coresets
        """
        if True, all coresets are stored in `self.all_coresets`. Otherwise, they are not stored.
        """

        self.store_all_gradients: bool = store_all_gradients
        """
        if True, all gradients are stored in `self.all_gradients`. Otherwise, they are not stored.
        """

        self.compute_all_past_gradients: bool = compute_all_past_gradients
        """
        if True, `self.store_all_gradients` is True and `self.recompute_coreset` is True,
        the gradients of all past samples are computed and stored. Otherwise, if False, `self.store_all_gradients` is
        True and `self.recompute_coreset` is True, only the gradients of those past samples currently stored in the
        replay buffer are computed and stored. Changing this value when `self.store_all_gradients` is False or
        `self.recompute_coreset` is False does not have any effect.
        """

        self.save_space_gradients: bool = save_space_gradients
        """
        if True, `self.gradient_approx` is `class_head` and `self.store_all_gradients` is True,
        the gradients of the multi-class cross entropy loss w.r.t the biases in the classification head and the
        sample embeddings (feature vectors) are stored to lower the memory footprint rather than storing the
        gradients of the multi-class cross entropy loss w.r.t. the weights and biases in the classification head.
        Note that it is possible to derive the gradient of the multi-class cross entropy loss w.r.t. the weights
        and biases in the classification head for a given sample by computing the outer product between the gradient
        of the multi-class cross entropy loss w.r.t the biases and the sample embedding and concatenating this with
        the gradient w.r.t. the biases. If True, `self.gradient_approx` is `ce_mse_class_head` and
        `self.store_all_gradients` is True, the gradients of the multi-class cross entropy loss and the mean squared
        error loss w.r.t the biases in the classification head are stored separately along with the sample embeddings
        (feature vectors) to lower the memory footprint. Note that it is possible to derive the gradient of the sum of
        the multi-class cross entropy loss and the mean squared error loss w.r.t. the weights and biases in the
        classification head for a given sample by first computing the sum between the gradients of the multi-class
        cross entropy loss and the mean squared error loss w.r.t the biases, denoted as G, subsequently compute the
        outer product between G and the sample embedding and finally concatenating the result of the outer product with
        G. Default is False.
        """

        self.curr_distance_kernels: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
        """
        a dictionary with target labels as keys and a list of tuples as values. Each tuple contains a distance kernel
        as first element and an array of respective sample indices as second element. The ith, jth element in the
        distance kernel is the distance between the gradient of the ith and jth sample of the given class, where the ith
        and jth samples of the given class have the ith and jth index contained in the array of sample indices. The
        distance kernels here are used to compute the aggregate similarity kernels and subsequenly the coresets when
        `update` is called. At the end of `update`, this dictionary is emptied. The target labels are not the real
        target labels but the order in which classes appear in the list of classes seen by the unique subnetwork
        accessible through `strategy.model.subnetworks_classes[sub_id]
        """

        self.all_distance_kernels: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
        """
        a dictionary with target labels as keys and a list of tuples as values. Each tuple contains a distance kernel
        as first element and an array of respective sample indices as second element. The ith, jth element in the
        distance kernel is the distance between the gradient of the ith and jth sample of the given class, where the ith
        and jth samples of the given class have the ith and jth index contained in the array of sample indices. This
        dictionary stores all the distance kernels computed so far for each target label if
        `self.store_all_distance_kernels` is True. The target labels are not the real target labels but the order in
        which classes appear in the list of classes seen by the unique subnetwork accessible through
        `strategy.model.subnetworks_classes[sub_id].
        """

        self.curr_coreset: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        """
        a dictionary with target labels as keys and tuples as values holding the current coreset stored in the
        replay buffer. Each tuple contains three elements. The first element is an array of indices of samples of the
        given class stored in the replay buffer sorted according to their facility location marginal gain (samples with
        higher marginal gain are first). The second element is an array of respective marginal gains. The third element
        is an array of respective weights as described in "Coresets for Data-efficient Training of Machine Learning
        Models" and "Towards Sustainable Learning: Coresets for Data-efficient Deep Learning". The target
        labels are not the real target labels but the order in which classes appear in the list of classes seen by the
        unique subnetwork accessible through `strategy.model.subnetworks_classes[sub_id].
        """

        self.all_coresets: Dict[int, List[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = defaultdict(list)
        """
        a dictionary with target labels as keys and lists as values holding all the coresets computed so far for the
        given target label, including the coreset currently stored. Each element in the list is a distinct tuple
        computed at different time steps. Each tuple contains three elements. The first element is an array of indices
        of samples of the given class sorted according to their facility location marginal gain (samples with
        higher marginal gain are first), which at a certain time step were stored in the replay buffer. The second
        element is an array of respective marginal gains. The third element is an array of respective weights as
        described in "Coresets for Data-efficient Training of Machine Learning Models" and
        "Towards Sustainable Learning: Coresets for Data-efficient Deep Learning". The target
        labels are not the real target labels but the order in which classes appear in the list of classes seen by the
        unique subnetwork accessible through `strategy.model.subnetworks_classes[sub_id].
        """

        self.all_gradients: Dict[int, List[Tuple[
            Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], np.ndarray],
            np.ndarray]]] = defaultdict(list)
        """
        a dictionary with target labels as keys and lists as values. Each element in the list is a distinct tuple
        computed at different time steps. Each tuple contains two elements. If `self.save_space_gradients` is True
        and `self.gradient_approx` is `class_head`, the first element is a tuple of two elements, where the first
        element is an array of gradients of the multi-class cross entropy loss w.r.t. the biases in the classification
        head and the second element is an array of embeddings (feature vectors). If `self.save_space_gradients` is True
        and `self.gradient_approx` is `ce_mse_class_head`, the first element is a tuple of three elements, where the
        first element is an array of gradients of the multi-class cross entropy loss w.r.t. the biases in the
        classification head, the second element is an array of gradients of the mean squared error loss w.r.t the biases
        in the classification head and the third element is an array of embeddings (feature vectors).
        Otherwise, if `self.save_space_gradients` is False or `self.gradient_approx` is `logits`, the first element is
        an array of gradients. The second element is always an array of respective indices. The target labels are not
        the real target labels but the order in which classes appear in the list of classes seen by the unique
        subnetwork accessible through `strategy.model.subnetworks_classes[sub_id].
        """

        self.coreset_mode = coreset_mode
        """
        `dense`, `sparse` or `clustered`. It specifies whether :class:`FacilityLocationFunction` should operate in
        dense mode (using a dense similarity kernel), sparse mode (using a sparse similarity kernel) or clustered mode
        (evaluating over clusters). Check the documentation of :meth:`CRAIGCorset.select_coreset` for more info about
        this parameter.
        """

        exps_no_coreset = []
        if isinstance(no_coreset_exps, int):
            exps_no_coreset.append(no_coreset_exps)
        elif isinstance(no_coreset_exps, list):
            exps_no_coreset = no_coreset_exps

        self.exps_no_coreset: List[int] = exps_no_coreset
        """
        list of experience IDs whose coresets should not be computed at any time step. These experiences are
        totally ignored as if they were never encountered. If the current training experience is an experience to be
        ignored, then no computation of distance kernels is scheduled during the SGD path and no update to the buffer is
        performed at the end of training. This holds because no new samples are added to the buffer. If the current
        experience is not to be ignored, `recompute_coreset` is True and experiences to be ignored were encountered in
        the past, the computation of the distance kernels is performed for all the past experiences + the current
        experience except for the past experiences to be ignored.
        """

        self.kwargs = kwargs
        """
        a set of keyword arguments to be used to compute the coresets. They are passed 
        to :meth:`CRAIGCorset.select_coreset`. Therefore, have a look at the documentation
        of :meth:`CRAIGCorset.select_coreset` to check which keyword arguments can be passed. The argument `metric`
        should not be passed.
        """

        self._train_exps: List = []
        """
        a list of the training experiences encountered so far. The order of the training experiences in this list 
        reflects the order with which these experiences were encountered
        """

        self._train_datasets: List[AvalancheDataset] = []
        """
        a list of the training datasets of the experiences encountered so far. The order of the training datasets in
        this list reflects the order with which the respective experiences were encountered
        """

        self._gradient_metric: GradientLossWRTInputLastLayer = GradientLossWRTInputLastLayer()
        """
        the standalone metric for keeping track of the gradient of the multi-class cross entropy loss w.r.t. the logits
        for the samples in a dataset.
        """

        self._gradient_class_head_metric: GradientLossWRTWeightsBiasesClassificationHead = (
            GradientLossWRTWeightsBiasesClassificationHead())
        """
        the standalone metric for keeping track of the gradient of the multi-class cross entropy loss w.r.t. the weights
        and biases in the classification head.
        """

        self._gradient_ce_mse_class_head_metric: GradientCrossEntropyLossMSELossWRTWeightsBiasesClassificationHead = (
            GradientCrossEntropyLossMSELossWRTWeightsBiasesClassificationHead())
        """
        the standalone metric for keeping track of the gradient of the sum of the multi-class cross entropy loss and the
        mean squared error loss w.r.t. the weights and biases in the classification head.
        """

        self._eval_called: bool = False
        """boolean flag indicating whether `strategy.eval` was called by this plugin"""

    @property
    def buffer_datasets(self) -> List[AvalancheDataset]:
        """
        Get a list of the weighted datasets of each class. The weighted dataset of a class is its dataset
        where each sample is repeated a certain number of times according to its weight in the current coreset.
        :return a list of weighted datasets, one for each class
        """
        weighted_datasets = []
        for target, buffer in self.buffer_groups.items():
            indxs, weights = self.curr_coreset[target][0], self.curr_coreset[target][2]
            weighted_datasets.append(self._create_weighted_dataset(buffer, indxs=indxs, weights=weights))
        return weighted_datasets

    @property
    def buffer(self) -> AvalancheDataset:
        """
        Get all the weighted datasets of each class concatenated together. The weighted dataset of a class is its
        dataset where each sample is repeated a certain number of times according to its weight in the current coreset.
        """
        return concat_datasets(self.buffer_datasets)

    @buffer.setter
    def buffer(self, new_buffer):
        assert NotImplementedError(
            "Cannot set `self.buffer` for this class. "
            "You should modify `self.buffer_groups instead."
        )

    def buffer_subnetwork(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
                          sub_id: str) -> AvalancheDataset:
        """
        Get the rehearsal buffer for the single subnetwork.
        :param strategy: a :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` strategy
        :param sub_id: ID of the single subnetwork
        :return: a rehearsal dataset of past data for the single subnetwork. If there is no rehearsal buffer for the
        single subnetwork, an empty dataset is returned.
        """
        if not len(strategy.model.subnetworks_classes) == 1:
            raise RuntimeError("This plugin only works when there is a single subnetwork")
        if sub_id not in strategy.model.subnetworks_classes.keys():
            raise RuntimeError("There is no subnetwork with the provided ID")
        return self.buffer

    def resize(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate, new_size: int):
        """
        Update the maximum size of the replay buffer. If the new size is larger than or equal to the current total
        number of  samples stored in the replay buffer, then nothing is performed apart from updating
        `self.max_size`. Otherwise, if the new size is smaller than the current total number of samples stored in the
        replay buffer, then an equal number of samples is removed from each class to reach the desired size. The samples
        that are removed for each class are the ones with the lowest marginal gain.
        """
        self.max_size = new_size
        # if the new size is smaller than the number of samples stored in the replay buffer
        if self.max_size < sum([len(g) for g in self.buffer_groups.values()]):
            lens = self.get_group_lengths(len(self.buffer_groups))
            new_coreset: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
            for ll, target in zip(lens, self.buffer_groups.keys()):
                indx, marg_gains, weights = self.curr_coreset[target]
                new_coreset[target] = (indx[:ll], marg_gains[:ll], weights[:ll])
                curr_dataset = self.buffer_groups[target]
                self.buffer_groups[target] = self._subset_dataset_indices(curr_dataset, indx[:ll])
            self.curr_coreset = new_coreset
            # store the new coreset if required
            if self.store_all_coresets:
                self._store_curr_coreset()

    def get_group_lengths(self, num_groups):
        """Compute groups lengths given the number of groups `num_groups`."""
        if self.adaptive_size:
            lengths = [self.max_size // num_groups for _ in range(num_groups)]
            # distribute remaining size among experiences.
            rem = self.max_size - sum(lengths)
            for i in range(rem):
                lengths[i] += 1
        else:
            lengths = [
                self.max_size // self.total_num_groups for _ in range(num_groups)
            ]
        return lengths

    def _store_curr_coreset(self):
        """
        Store the current coreset in `self.curr_coreset` into `self.all_coresets`
        """
        for t, coreset in self.curr_coreset.items():
            self.all_coresets[t].append(coreset)

    def _create_weighted_dataset(self, dataset: AvalancheDataset, indxs: np.ndarray, weights: np.ndarray):
        """
        Create a dataset where each sample is contained a certain number of times according to its weight
        from a dataset containing each sample only once.
        :param dataset: a dataset containing each sample only once
        :param indxs: an array of indices. These indices must be equivalent to the indices in the `dataset_indices`
            DataAttribute of `dataset`
        :param weights: an array of weights. Each weight is the number of times the respective sample must be contained
            in the returned dataset
        :return: a dataset where each sample is contained a certain number of times according to its weight
        """
        all_subdatasets = []
        for i, s_index in enumerate(dataset.dataset_indices.data):
            s_index = int(s_index)
            single_sample_dataset = dataset.subset([i])  # a dataset only containing the current sample
            weight = int(weights[indxs == s_index][0])
            all_subdatasets.append(self._duplicate_dataset_with_single_sample(single_sample_dataset, weight))
        return concat_datasets(all_subdatasets)

    def _duplicate_dataset_with_single_sample(self, dataset: AvalancheDataset, n: int):
        """
        Given a dataset with a single sample, create a dataset of the same type containing the given sample n times.
        :param dataset: a dataset
        :param n: the number of times the single sample must be repeated
        :return: a dataset with the single sample repeated n times
        """
        if not len(dataset) == 1:
            raise ValueError("The dataset must contain a single sample")
        if n <= 0:
            raise ValueError("n must be strictly positive")
        if n == 1:
            return dataset
        return concat_datasets([dataset for _ in range(n)])

    def update(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate, **kwargs):
        """
        Update the buffer with new data. The buffer is updated only if the strategy is in training mode.

        .note::
            The data is retrieved from `strategy.adapted_dataset`.
        :param strategy: a :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` strategy
        :return: None
        """
        if strategy.is_training:  # it does not make sense to update the buffer at evaluation time
            # when in training mode, `strategy.adapted_dataset` is a dictionary containing subnetwork IDs as keys and
            # the respective datasets as values. This plugin assumes that there is a single subnetwork
            if not len(strategy.adapted_dataset) == 1:
                raise RuntimeError("This plugin only works when there is a single subnetwork")
            adapted_datasets = list(strategy.adapted_dataset.values())[0]
            self.update_from_dataset(adapted_datasets, strategy)

    def update_from_dataset(self, new_data: AvalancheDataset,
                            strategy: Optional[DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate] = None
                            ):
        """
        Update the replay buffer according to the distance kernels computed in `self.curr_distance_kernels`.
        If `self.curr_distance_kernels` is empty, then no update is performed on the buffer.
        `self.curr_distance_kernels` is empty when the current training experience is to be ignored; in this case no
        distance kernels were computed during training.
        If the distance kernels of a past class have not been computed, i.e. that class is not a key
        in `self.curr_distance_kernels`, its coreset and buffer is reduced in size if necessary. The samples with the
        lowest marginal gains are discarded. In contrast, if the distance kernels of a class have been computed,
        the coreset of that class is computed from its aggregate similarity kernel as described in
        "Coresets for Data-efficient Training of Machine Learning Models" and
        "Towards Sustainable Learning: Coresets for Data-efficient Deep Learning" and according to `num_samples`.
        Its buffer is updated accordingly.
        :param new_data: dataset of the current experience
        :param strategy: a strategy
        """
        # `self.curr_distance_kernels` is empty when the current training experience is to be ignored; in this case no
        # distance kernels were computed during training and so there is no need of updating the replay buffer
        if len(self.curr_distance_kernels) == 0:
            return
        # Update the set of classes seen so far. `self.curr_distance_kernels.keys()` contains the new classes observed
        # in the current experience and, if `self.adaptive_size` is True and `self.recompute_coreset` is True, also all
        # the old classes observed in the past
        self.seen_classes.update(self.curr_distance_kernels.keys())

        # associate lengths to classes. `class_to_len` is a dictionary containing all the class labels observed so far
        # as keys and the respective buffer sizes as values.
        lens = self.get_group_lengths(len(self.seen_classes))
        class_to_len: Dict[int, int] = {}
        for target, ll in zip(self.seen_classes, lens):
            class_to_len[target] = ll

        # get the classes observed in the past whose distance kernels have not been computed during the training of the
        # current experience. This set of classes is not empty when `self.adaptive_size` is False or when
        # `self.adaptive_size` is True and `self.recompute_coreset` is False.
        no_distance_kernels_classes = self.seen_classes - set(self.curr_distance_kernels.keys())
        for target in no_distance_kernels_classes:
            length = class_to_len[target]  # get the new size of the buffer for the current class
            # if the new size is smaller than the size of the current buffer, which only occurs when
            # `self.adaptive_size` is True and `self.recompute_coreset` is False, then discard the samples
            # that have the lowest facility location marginal gain as computed when the coreset of the given class was
            # found.
            if length < len(self.buffer_groups[target]):
                indxs, marg_gains, weights = self.curr_coreset[target]  # get the current coreset
                # discard the samples with the lowest marginal gain
                self.curr_coreset[target] = (indxs[:length], marg_gains[:length], weights[:length])
                curr_buffer = self.buffer_groups[target]  # get the current buffer
                # discard from the buffer the samples with the lowest marginal gain
                self.buffer_groups[target] = self._subset_dataset_indices(curr_buffer, indxs[:length])

        # compute the coreset for each class whose distance kernels have been computed during the training of the
        # current experience.
        for target, tuples_list in self.curr_distance_kernels.items():
            # the order of the sample indices used to compute all the distance kernels for the current class. The order
            # should not change across the list if `update` is called at the end of training experience, as it is
            # performed by the replay plugins. Thus, we get this order from the first tuple in the list
            indxs = tuples_list[0][1]
            # check whether the order of the sample indices used to compute all the distance kernels for the current
            # class are equal across all the distance kernels
            if not all([np.array_equal(t2, indxs) for _, t2 in tuples_list]):
                raise RuntimeError("The order of the sample indices must be equal across all the computed distance "
                                   "kernels")
            dist_kernels_list = [t1 for t1, _ in tuples_list]  # get the distance kernels only
            # all the 2D distance kernels are stacked in the same 3D array along the first dimension
            stack_dist_kernels = np.stack(dist_kernels_list, axis=0)
            if self.aggregate_similarity_kernel == "min":  # if `min` compute the maximal distance kernel
                aggr_dist_kernel = np.max(stack_dist_kernels, axis=0)
            else:  # otherwise, compute the mean distance kernel
                aggr_dist_kernel = np.mean(stack_dist_kernels, axis=0)
            if self.metric not in ["euclidean", "mahalanobis"]:
                raise RuntimeError("The only supported metrics are: `euclidean` and `mahalanobis`")
            highest_value = np.max(aggr_dist_kernel)  # get the highest value in the aggregate distance kernel
            # convert the aggregate distance kernel into the aggregate similarity kernel, which in the case of euclidean
            # and mahalanobis is performed by computing the highest value m in the aggregate distance kernel and then
            # replacing each value in the aggregate distance kernel by m - vij, where vij is the ith, jth element in the
            # aggregate distance kernel
            aggr_sim_kernel = highest_value - aggr_dist_kernel
            if len(indxs) < self.num_subsets:
                raise RuntimeError("The number of random subsets to be used to compute the coreset of the given class "
                                   "is greater than the number of samples in this class")
            if class_to_len[target] < self.num_subsets:
                raise RuntimeError("The number of random subsets to be used to compute the coreset of the given class "
                                   "is greater than the size of the buffer of this class")
            # get the subsets and the respective coreset sizes
            subsets, coreset_size_subsets = self.subset_generator.generate_subsets(n_subsets=self.num_subsets,
                                                                                   tot_coreset_size=class_to_len[target],
                                                                                   indices=indxs)
            actual_indxs_ls = []  # list containing all the indexes to be preserved for this class
            mar_gains_ls = []  # list containing all the respective mar gains of all the indexes to be preserved
            weights_ls = []  # list containing all the respective weights of all the indexes to be preserved
            for i, subset in enumerate(subsets):
                # get the rows and columns indexes from the given random subset. By using this index, it is possible to
                # extract from the aggregate similarity kernel, the aggregate similarity kernel for the given random
                # subset
                kernel_index = np.ix_(subset, subset)
                core_indxs, mar_gains, weights = CRAIGCoreset().select_coreset(b=coreset_size_subsets[i],
                                                                               sim_kernel=aggr_sim_kernel[kernel_index],
                                                                               mode=self.coreset_mode,
                                                                               **self.kwargs)
                # convert the coreset indexes first into the actual indexes in the random subset and then into the
                # actual indexes which are stored in indxs
                actual_indxs = indxs[subset[core_indxs]]

                actual_indxs_ls.append(actual_indxs)  # append into the list
                mar_gains_ls.append(mar_gains)  # append into the list
                weights_ls.append(weights)  # append into the list

            actual_indxs = np.concatenate(actual_indxs_ls)  # the indexes to be preserved for this class
            mar_gains = np.concatenate(mar_gains_ls)  # the respective mar gains
            weights = np.concatenate(weights_ls)   # the respective weights
            # sort actual_indxs and consequently also mar_gains and weights so that the indices are sorted according to
            # their facility location marginal gain (sample indices with higher marginal gain are first)
            desc_order = np.argsort(mar_gains)[::-1]
            # update the coreset of this class in `self.curr_coreset`
            self.curr_coreset[target] = (actual_indxs[desc_order], mar_gains[desc_order], weights[desc_order])
            # if this class is an old class whose coreset has been recomputed, i.e. `self.recompute_coreset` is True
            # and `self.adaptive_size` is True, then create a subset of the dataset currently stored in the buffer.
            # Otherwise, the class is a new class and the subset must be created out of `new_data`.
            if target in self.buffer_groups.keys():
                dataset = self.buffer_groups[target]
            else:
                dataset = new_data
            self.buffer_groups[target] = self._subset_dataset_indices(dataset, actual_indxs)

        # empty self.curr_distance_kernels
        self.curr_distance_kernels = defaultdict(list)
        # add the new coreset to `self.all_coresets` if required
        if self.store_all_coresets:
            self._store_curr_coreset()

    def _subset_dataset_indices(self, dataset: AvalancheDataset, indices: Iterable[int]) -> AvalancheDataset:
        """
        Compute a subset of the given dataset by only retaining those samples whose "dataset_indices" DataAttribute is
        contained in the provided iterable of integers.
        :param dataset: a dataset
        :param indices: an iterable of indices whose samples need to be retained
        :return: a subset that contains samples whose "dataset_indices" DataAttribute is contained in the provided
            iterable of integers
        """
        return dataset.subset([i for i, s_index in enumerate(dataset.dataset_indices.data) if int(s_index) in indices])

    def before_training_exp(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
                              **kwargs):
        """
        Callback called before training on the new experience

        This callback appends the new experience to `self._train_exps` if it is a training experience never encountered
        before.
        :param strategy: a strategy
        :param kwargs: some keyword arguments
        """
        if strategy.experience.origin_stream.name == "train":  # if it is a training experience
            # if this experience has not been encountered before, then append it to `self._train_exps`
            if strategy.experience.current_experience not in [exp.current_experience for exp in self._train_exps]:
                self._train_exps.append(strategy.experience)

    def after_train_datasets_adaptation(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
                              **kwargs):
        """
        Callback called after the train datasets adaptation

        This callback verifies that there is a single subnetwork by checking that `strategy.adapted_dataset` contains
        the dataset of a single subnetwork. If there are multiple subnetworks, this callback raises
        a :class`RuntimeError`. Otherwise, the dataset of the single subnetwork is added to ` self._train_datasets`.
        :param strategy: a strategy
        :param kwargs: some keyword arguments
        """
        if len(strategy.adapted_dataset) > 1:
            raise RuntimeError("This plugin only works when there is a single subnetwork")
        adapted_dataset = list(strategy.adapted_dataset.values())[0]
        self._train_datasets.append(adapted_dataset)

    def before_training_subnetwork(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
                                   **kwargs):
        """
        Callback called before the training loop of the single subnetwork.

        Compute the distance kernels at the current optimisation point before the training loop starts if
        `self.do_initial` is True

        :param strategy: a strategy
        :param kwargs: some keyword arguments
        """
        if self.do_initial:
            self._compute_dist_kernels(strategy, **kwargs)

    def after_training_epoch(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate, **kwargs):
        """
        Callback called at the end of each training epoch.

        If `self.frequency_mode` is `epoch` or `both`, the computation of the distance kernels is performed
        according to the value of `self.frequency` and `strategy.clock.train_exp_epochs`.
        """
        if self.frequency_mode in ["epoch", "both"]:
            frequency = self.frequency
            # if it is a tuple then get the first element
            if isinstance(frequency, tuple):
                frequency = frequency[0]
            self._maybe_compute_dist_kernels(strategy, strategy.clock.train_exp_epochs, frequency, **kwargs)

    def after_training_iteration(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
                                 **kwargs):
        """
        Callback called at the end of each training iteration.

        If `self.frequency_mode` is `iteration` or `both`, the computation of the distance kernels is performed
        according to the value of `self.frequency` and `strategy.clock.train_exp_iterations`.
        """
        if self.frequency_mode in ["iteration", "both"]:
            frequency = self.frequency
            # if it is a tuple then get the second element
            if isinstance(frequency, tuple):
                frequency = frequency[1]
            self._maybe_compute_dist_kernels(strategy, strategy.clock.train_exp_iterations, frequency, **kwargs)

    def after_training_subnetwork(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
                                  **kwargs):
        """
        Callback called at the end of the training loop of the single subnetwork.

        Compute the distance kernels at the end of the training loop of the single subnetwork.
        The computation of the distance kernels is not performed by this callback if `self.frequency` is -1 or
        the last periodic computation of the distance kernels was already performed at the convergence point.
        In contrast, the computation of the distance kernels is performed if the last periodic computation was not
        performed at the convergence point or `self.frequency` is 0.
        """
        compute_final: bool = False
        if isinstance(self.frequency, tuple):
            counter = (strategy.clock.train_exp_epochs, strategy.clock.train_exp_iterations)
            if counter[0] % self.frequency[0] != 0 and counter[1] % self.frequency[1] != 0:
                compute_final = True
        elif self.frequency == 0:
            compute_final = True
        elif self.frequency > 0:
            if self.frequency_mode == "epoch":
                counter = strategy.clock.train_exp_epochs
            elif self.frequency_mode == "iteration":
                counter = strategy.clock.train_exp_iterations
            else:
                counter = (strategy.clock.train_exp_epochs, strategy.clock.train_exp_iterations)

            if isinstance(counter, tuple):
                if counter[0] % self.frequency != 0 and counter[1] % self.frequency != 0:
                    compute_final = True
            elif counter % self.frequency != 0:
                compute_final = True

        if compute_final:
            self._compute_dist_kernels(strategy, **kwargs)

    def _maybe_compute_dist_kernels(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
                                    counter, frequency: int, **kwargs):
        """
        Maybe compute the distance kernels for the single subnetwork.
        The computation is performed according to the counter and frequency provided.
        :param strategy: strategy
        :param counter: counter
        :param frequency: frequency
        :param kwargs: custom arguments
        """
        if frequency > 0 and counter % frequency == 0:
            self._compute_dist_kernels(strategy, **kwargs)

    def _compute_dist_kernels(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate, **kwargs):
        """
        Compute the distance kernels for the single subnetwork.
        The distance kernels are only computed if the current experience is not to be ignored; that is, it is not in
        `self.exps_no_coreset`. The following follows if the current experience is not to be ignored.
        If `self.adaptive_size` is True and `self.recompute_coreset` is False or `self.adaptive_size` is False, the
        distance kernels are only computed for the classes in the current training experience, which is the
        last one in `self._train_exps`.
        Otherwise, if `self.adaptive_size` is True and `self.recompute_coreset` is True, the distance
        kernels are also computed for all classes encountered in past training experiences but only on those
        samples that have been retained in the replay buffer. Note that the distance kernels are not computed for the
        past training experiences to be ignored.

        This method calls `strategy.eval` to compute the distance kernels.
        The `after_eval_iteration` callback is used to capture the gradients either of the multi-class cross entropy
        loss w.r.t the logits or w.r.t the weights and biases in the classification head, or of the sum of the
        multi-class cross entropy loss and mean squared error loss w.r.t the weights and biases in the classification
        head and then these gradients are used to compute the distance kernels.

        During the `strategy.eval` calls performed by this callback, the metrics in `EvaluationPlugin` are not
        computed as they would be meaningless and their computation would only be a computational overhead.

        :param strategy: a strategy
        :param kwargs: custom arguments
        """
        # only compute the distance kernels if the current experience is not to be ignored
        if self._train_exps[-1].current_experience not in self.exps_no_coreset:
            if self.adaptive_size:
                if self.recompute_coreset:
                    exps = self._train_exps  # gather all past training experience
                else:
                    # only gather the current train experience, which is the last one in `self._train_exps`
                    exps = [self._train_exps[-1]]
            else:
                # only gather the current train experience, which is the last one in `self._train_exps`
                exps = [self._train_exps[-1]]

            self._compute_dist_kernels_for_each_exp(exps, strategy, **kwargs)

    def _compute_dist_kernels_for_each_exp(self, exps: List,
                                           strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
                                           **kwargs):
        """
        Compute the distance kernels for the single subnetwork for each of the provided experiences. The distance
        kernels are not computed for those experiences that are to be ignored, i.e. they are in `self.exps_no_coreset`.

        This method calls `strategy.eval` to compute the distance kernels.
        The `after_eval_iteration` callback is used to capture the gradients either of the multi-class cross entropy
        loss w.r.t the logits or w.r.t the weights and biases in the classification head, or of the sum of the
        multi-class cross entropy loss and mean squared error loss w.r.t the weights and biases in the classification
        head and then these gradients are used to compute the distance kernels.

        During the `strategy.eval` calls performed by this callback, the metrics in `EvaluationPlugin` are not
        computed as they would be meaningless and their computation would only be a computational overhead.

        :param exps: a list of experiences
        :param strategy: a strategy
        :param kwargs: a set of keyword arguments
        """
        self._eval_called = True  # indicate that the next `strategy.eval` calls are performed by this plugin
        # get the evaluator
        evaluator: EvaluationPlugin = [plg for plg in strategy.plugins if isinstance(plg, EvaluationPlugin)][0]
        # this will deactivate the evaluator, i.e. metrics are not computed during the next eval calls.
        evaluator.active = False
        for exp in exps:
            # compute the distance kernels only if the current experience is not in `self.exps_no_coreset`
            if exp.current_experience not in self.exps_no_coreset:
                strategy.eval(exp, **kwargs)  # start the eval phase
                # after conclusion of the eval phase, get the collected gradients, targets and indices. They were
                # collected by the `after_eval_iteration` callback
                if self.gradient_approx == "logits":
                    gradients, targets, indices = self._gradient_metric.result()
                    self._gradient_metric.reset()  # reset the gradient metric to its initial state
                elif self.gradient_approx == "class_head":
                    if self.save_space_gradients and self.store_all_gradients:
                        # also get the gradients wrt the biases and the embeddings
                        gradients, targets, indices, gradients_biases, embeds = self._gradient_class_head_metric.result(
                            retrieve_grad_biases_embeddings=True)
                    else:  # only get the normal gradients
                        gradients, targets, indices = self._gradient_class_head_metric.result()

                    self._gradient_class_head_metric.reset()  # reset to its initial state
                else:
                    if self.save_space_gradients and self.store_all_gradients:
                        # also get the gradients wrt the biases and the embeddings
                        gradients, targets, indices, ce_gradients_biases, mse_gradients_biases, embeds = (
                            self._gradient_ce_mse_class_head_metric.result(retrieve_grad_biases_embeddings=True))
                    else:  # only get the normal gradients
                        gradients, targets, indices = self._gradient_ce_mse_class_head_metric.result()

                    self._gradient_ce_mse_class_head_metric.reset()  # reset to its initial state

                # get the unique targets and sort them into ascending order
                unique_targets = torch.unique(targets, sorted=True)
                for t in unique_targets:
                    t = t.item()  # get the python value rather than a pytorch scalar tensor
                    t_gradients = gradients[targets == t]  # get the gradients of the given target
                    t_indices = indices[targets == t]  # get the indices of the given target
                    indx_order = torch.argsort(t_indices)
                    # sort the gradients according to their respective sample index into ascending order
                    t_gradients = t_gradients[indx_order]
                    t_indices = t_indices[indx_order]  # sort the indices into ascending order
                    t_gradients = t_gradients.cpu().detach().numpy()  # convert into numpy array
                    # convert into numpy array, it should be unnecessary to call detach
                    t_indices = t_indices.cpu().numpy()

                    if self.store_all_gradients:
                        # if space needs to be saved and `self.gradient_approx` is `class_head` or `ce_mse_class_head`,
                        # then store the gradients wrt the biases and the embeddings
                        if self.gradient_approx in ["class_head", "ce_mse_class_head"] and self.save_space_gradients:
                            t_embeds = embeds[targets == t]  # get the embeddings of the given target
                            # sort the embeddings according to their respective sample index into ascending order
                            t_embeds = t_embeds[indx_order]
                            t_embeds = t_embeds.cpu().detach().numpy()  # convert into numpy array
                            if self.gradient_approx == "class_head":
                                # get the grads biases of the given target
                                t_gradients_biases = gradients_biases[targets == t]
                                # sort the gradients biases according to their respective sample index into ascending
                                # order
                                t_gradients_biases = t_gradients_biases[indx_order]
                                # convert into numpy array
                                t_gradients_biases = t_gradients_biases.cpu().detach().numpy()
                                self.all_gradients[t].append(((t_gradients_biases, t_embeds), t_indices))
                            else:
                                # get the cross entropy grads biases of the given target
                                t_ce_gradients_biases = ce_gradients_biases[targets == t]
                                # get the mean squared error grads biases of the given target
                                t_mse_gradients_biases = mse_gradients_biases[targets == t]
                                # sort the cross entropy gradients biases according to their respective sample index
                                # into ascending order
                                t_ce_gradients_biases = t_ce_gradients_biases[indx_order]
                                # sort the mean squared error gradients biases according to their respective sample
                                # index into ascending order
                                t_mse_gradients_biases = t_mse_gradients_biases[indx_order]
                                # convert into numpy array
                                t_ce_gradients_biases = t_ce_gradients_biases.cpu().detach().numpy()
                                # convert into numpy array
                                t_mse_gradients_biases = t_mse_gradients_biases.cpu().detach().numpy()
                                self.all_gradients[t].append(((t_ce_gradients_biases, t_mse_gradients_biases, t_embeds),
                                                              t_indices))
                        else:  # otherwise, save the full gradient
                            self.all_gradients[t].append((t_gradients, t_indices))

                    if self.compute_all_past_gradients and self.store_all_gradients:
                        # if the given target is a target encountered in a past experience, since
                        # self.compute_all_past_gradients and self.store_all_gradients are True, `t_gradients` and
                        # `t_indices` contain gradients and indices of samples that are not in the replay buffer. These
                        # must be filtered out because the distance kernels of past targets must contain only samples
                        # that have been preserved in the replay buffer
                        if t in self.curr_coreset.keys():
                            # get the indices of the given target that are currently stored in the replay buffer
                            ixs = self.curr_coreset[t][0]
                            mask = np.isin(t_indices, ixs)
                            t_gradients = t_gradients[mask]  # get the grads of only the samples stored in the buffer
                            t_indices = t_indices[mask]  # get the indices of only the samples stored in the buffer

                    # compute the distance kernel of the given gradients. The ith, jth element of such a kernel is the
                    # distance between the ith and jth gradient vector in t_gradients
                    distance_kernel = pairwise_distances(X=t_gradients, metric=self.metric)
                    self.curr_distance_kernels[t].append((distance_kernel, t_indices))
                    if self.store_all_distance_kernels:
                        self.all_distance_kernels[t].append((distance_kernel, t_indices))
        self._eval_called = False  # reset the flag to false
        # active the evaluator back again, so that metrics are computed when eval is not called by this plugin
        evaluator.active = True

    def after_eval_dataset_adaptation(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
                                      **kwargs):
        """
        Callback called after each eval dataset adaptation.

        This callback gets executed only if the current eval phase was called by this plugin.
        This can be achieved by looking at the value of `self._eval_called`. This callback **should not** be executed
        when the current eval phase was not called by this plugin.

        If the training experience currently being evaluated is the last one in `self._train_exps`, which means that a
        coreset still must be computed for the classes in this experience, this callback does nothing. This callback
        also does nothing when the training experience currently being evaluated is not the last one in
        `self._train_exps`,`self.compute_all_past_gradients` is True and `self.store_all_gradients` is True.
        Otherwise, this callback creates a subset of the dataset stored in `strategy.adapted_dataset` that only
        includes those past samples that have been preserved in the replay buffer. Since older training experiences are
        evaluated by this plugin only when `self.recompute_coreset` is True, we want to only compute the distance
        kernels for those samples that we still store in the replay buffer.
        :param strategy: a strategy
        :param kwargs: some keyword arguments
        """
        if self._eval_called:
            # if the training experience currently being evaluated is not the last one
            if not strategy.experience.current_experience == self._train_exps[-1].current_experience:
                if not (self.compute_all_past_gradients and self.store_all_gradients):
                    cls_in_exp = strategy.experience.classes_in_this_experience  # get the classes in this experience
                    if not len(strategy.model.subnetworks_classes) == 1:
                        raise RuntimeError("This plugin only works when there is a single subnetwork")
                    # get the classes of the single subnetwork
                    sub_cls = list(strategy.model.subnetworks_classes.values())[0]
                    # convert the classes in this experience into the order with which they appear in the single
                    # subnetwork
                    order = [sub_cls.index(cls) for cls in cls_in_exp]
                    # all sample indexes currently stored in the replay buffer for the classes in this experience
                    all_indxs = np.concatenate([ind for t, (ind, _, _) in self.curr_coreset.items() if t in order])
                    # create a subset of `strategy.adapted_dataset` that only stores the samples stored in the replay
                    # buffer
                    strategy.adapted_dataset = self._subset_dataset_indices(strategy.adapted_dataset, all_indxs)

    def after_eval_iteration(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate, **kwargs):
        """
        Callback called after each eval iteration.

        Capture the gradients of the multi-class cross entropy loss w.r.t the logits or w.r.t the weights and biases in
        the classification head, or the gradients of the sum of the multi-class cross entropy loss and the mean squared
        error loss w.r.t the weights and biases in the classification head, the targets and the indices of the samples
        in the current mini-batch.
        This callback gets executed only if the current eval phase was called by this plugin.
        This can be achieved by looking at the value of `self._eval_called`. This callback **should not** be executed
        when the current eval phase was not called by this plugin.
        :param strategy: a strategy
        :param kwargs: some keyword arguments
        """
        if self._eval_called:
            curr_sub_id = strategy.curr_sub_id  # store the current state of the strategy's subnetwork id
            subs_id = list(strategy.mb_output.keys())
            if len(subs_id) > 1:
                raise RuntimeError("This plugin must be used only when there is one subnetwork")
            strategy.curr_sub_id = subs_id[0]
            num_classes = len(strategy.model.subnetworks_classes[strategy.curr_sub_id])
            if "dataset_indices" not in strategy.use_in_getitem_indices.keys():
                raise RuntimeError("The dataset of the current training experience must have the `dataset_indices` "
                                   "DataAttribute")
            indices = strategy.mb_dataset_indices
            if self.gradient_approx == "logits":
                self._gradient_metric.update(strategy.mb_output_incremental_classifier,
                                             strategy.mb_y_incremental_classifier, num_classes, indices=indices)
            elif self.gradient_approx == "class_head":
                self._gradient_class_head_metric.update(strategy.mb_output_incremental_classifier,
                                                        strategy.mb_y_incremental_classifier,
                                                        strategy.mb_feature_incremental_classifier,
                                                        num_classes, indices=indices)
            else:
                self._gradient_ce_mse_class_head_metric.update(strategy.mb_output_incremental_classifier,
                                                               strategy.mb_y_incremental_classifier,
                                                               strategy.mb_feature_incremental_classifier,
                                                               num_classes,
                                                               strategy.mb_has_incremental_classifier_logits,
                                                               strategy.mb_incremental_classifier_logits,
                                                               indices=indices)
            strategy.curr_sub_id = curr_sub_id  # re-store the state of the strategy's subnetwork id


class GradientBasedReplayBufferSelectionSGDFutureExpPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate(
    GradientBasedReplayBufferSelectionSGDPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
    supports_distributed=True):
    """
    Gradient-based replay buffer selection over SGD optimisation parameters for future experiments storage policy and
    plugin for :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`.

    This class inherits
    from :class:`GradientBasedReplayBufferSelectionSGDPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`.

    This storage policy and plugin behaves as follows: all samples in the datasets of previous training experiences are
    stored in the buffer before encountering a given experience. We call such experience the significant experience.
    In other words, a coreset with all the samples is stored along with the marginal gain and weight of each sample set
    to 0 and 1, respectively. When the significant experience is encountered, during the training of such experience,
    this plugin schedules a periodic computation of gradient-based distance kernels for each past class at distinct
    optimisation points during the SGD training path. Note that the distance kernels of a given class are computed over
    all samples of the given class and also note that the distance kernels are not computed for the classes in the
    significant experience.
    At the end of the training process of the significant experience, when `update` is called, a coreset of a given
    size is selected as described
    in :class:`GradientBasedReplayBufferSelectionSGDPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
    using the computed distance kernels. The memory buffer is equally divided over all the unique classes observed
    in the experiences prior to the significant experience.

    This class allows you to perform this type of experiments. Assume a checkpoint of the state of a given strategy
    template is saved after the training on each experience ends. Thus, a checkpoint is saved immediately prior to
    starting training on the significant experience and after it. The state of the attributes `self.buffer_groups` and
    `self.curr_coreset` in the checkpoint saved after training on the significant experience can be injected into the
    checkpoint saved immediately before starting the training on the significant experience. This way, you can
    re-perform the training process on the significant experience using a coreset computed previously rather than using
    all the past samples. Note that the state of the attributes `self.buffer_groups` and
    `self.curr_coreset` in the checkpoint saved immediately before training on the significant experience can be
    arbitrarily modified as you like. As an example you can randomly take a subset of all past samples using reservoir
    sampling.

    .note::
        When injecting something different into `self.buffer_groups` and `self.curr_coreset` in the checkpoint saved
        immediately before training on the significant experience and keep training from this checkpoint, set the
        attribute `self.retraining` to True. This way, at the end of the training process on the significant experience,
        no update is performed on the buffer when `update` is called.

    .note::
        The periodic computation of gradient-based distance kernels for each past class at distinct
        optimisation points during the SGD training path on the significant experience occurs both when
        `self.retraining` is True or False.
        Note that the distance kernels of a given class are computed over all samples of the given class and also note
        that the distance kernels are not computed for the classes in the significant experience.

    .note::
        This plugin *assumes* that the "dataset_indices" DataAttribute is present in the datasets of the training
        experiences

    .note::
        This plugin metric *only* works when there is *only* one subnetwork

    .note::
        The class targets used in this plugin are not the real class targets but the order in which
        classes appear in the list of classes seen by the single subnetwork accessible through
        `strategy.model.subnetworks_classes[id_of_the_single_subnetwork]

    .note::
        This class acts both as a plugin and a storage policy. Therefore, the same instance of this class must be
        provided to the list of plugins when
        instantiating :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` and to the `storage_policy`
        parameter when instantiating :class:`ReplayPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
        or :class:`ReplayNoOutlierPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`

    .warning::
        This plugin assumes that the significant experience will be the last encountered experience. If another
        experience arrives after the significant experience, a :class:`RuntimeError` will be raised.
    """
    def __init__(self, exp: int, max_size: int, metric: Literal["euclidean", "mahalanobis"] = "euclidean",
                 gradient_approx: Literal["logits", "class_head"] = "logits",
                 aggregate_similarity_kernel: Literal["min", "mean"] = "min", num_subsets: int = 1,
                 subset_generator: SubsetGeneratorGradientBasedReplayBufferSelection = RandomSubsetGenerator(),
                 frequency: Union[Tuple[int, int], int] = 0,
                 frequency_mode: Literal["epoch", "iteration", "both"] = "epoch", do_initial: bool = False,
                 store_all_distance_kernels: bool = False, store_all_coresets: bool = False,
                 store_all_gradients: bool = False, save_space_gradients: bool = False,
                 coreset_mode: Literal["dense", "sparse", "clustered"] = "dense",
                 **kwargs):
        """
        Create a new
        GradientBasedReplayBufferSelectionSGDFutureExpPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate
        :param exp: the ID of the significant experience
        :param max_size: size of the coreset after the training process on the significant experience ends
        :param metric: the metric to be used to compute the distance kernels. The only metrics currently
            supported are `euclidean` and `mahalanobis`. Default is `euclidean`.
        :param gradient_approx: the approximation to use for the gradient computation. `logits` to compute the gradient
            w.r.t the logits. `class_head` to compute the gradient w.r.t the weights and biases in the classification
            head. Default is `logits`.
        :param aggregate_similarity_kernel: `min` or `mean`. Use `min` to aggregate the distance-kernels of a class
            collected during the SGD training path into a minimal similarity kernel. Use `mean` to aggregate the
            distance-kernels of a class collected during the SGD training path into a mean similarity kernel.
            Default is `min`.
        :param num_subsets: the number of disjoint subsets to be used to select the samples that must be
            preserved in the replay buffer for each class. If 1, all samples of a given class are used to select the
            class coreset. If >1, the samples of a given class are divided into the given number of disjoint
            subsets according to the provided `subset_generator`, a coreset is computed for each subset and the class
            coreset is equal to the union of the coresets found for each random subset. Default is 1.
        :param subset_generator: a subset generator that divides the samples of a given class into `num_subsets`
            disjoint subsets. Default is the random subset generator, which randomly divides the samples
            of a given class into `num_subsets` disjoint subsets.
        :param frequency: the frequency used to compute the distance kernels at different points in the optimisation
            space during the SGD training path. If a single integer number, this is the frequency used across epochs,
            iterations or both of them according to the value of `frequency_mode`. 0 means the distance kernels are
            computed only at the end of the SGD training path, i.e. the distance kernels are computed at the convergence
            point of the SGD training path. Values > 0 mean that the distance kernels are computed every `frequency`
            epochs, iterations or both of them according to `frequency_mode` and at the end of the SGD training path.
            -1 means the distance kernels are never computed during the SGD training path. When -1, `do_initial` must
            be set to True; otherwise a :class:`ValueError` is raised.
            If a 2-tuple, `frequency_mode` must be `both` and the first and second value must refer to the frequency
            used across epochs and iterations, respectively. Both values must be strictly positive (> 0) and the
            distance kernels are also computed at the end of the SGD training path.
            If both values are not strictly positive, a :class:`ValueError` is raised.
            If a 2-tuple and `frequency_mode` is not `both`, a :class:`ValueError` is raised. Default is 0.
        :param frequency_mode: `epoch`, `iteration` or `both`. Decides whether the computation of the distance kernels
            during the SGD training path should execute every `frequency` epochs, iterations or both.
            Default is `epoch`.
        :param do_initial: whether to compute the distance kernels before the SGD training path of the significant
            experience starts, i.e. the distance kernels are computed using the initial weights. When `frequency` is
            set to -1, this parameter must be set to True; otherwise a :class:`ValueError` is raised. Default is False.
        :param store_all_distance_kernels: if True, all distance kernels are stored in `self.all_distance_kernels`.
            Default is False.
        :param store_all_coresets: if True, all coresets are stored in `self.all_coresets`. Default is False.
        :param store_all_gradients: if True, all gradients are stored in `self.all_gradients`. Default is False.
        :param save_space_gradients: if True and `gradient_approx` is `class_head` and `store_all_gradients` is True,
            the gradients of the loss w.r.t the biases in the classification head and the sample embeddings
            (feature vectors) are stored to lower the memory footprint rather than storing the gradients of the loss
            w.r.t. the weights and biases in the classification head. Note that it is possible to derive the
            gradient of the loss w.r.t. the weights and biases in the classification head for a given sample by
            computing the outer product between the gradient of the loss w.r.t the biases and the sample embedding and
            concatenating this with the gradient w.r.t. the biases. Default is False.
        :param coreset_mode: `dense`, `sparse` or `clustered`. It specifies whether :class:`FacilityLocationFunction`
            should operate in dense mode (using a dense similarity kernel), sparse mode (using a sparse similarity
            kernel) or clustered mode (evaluating over clusters). Check the documentation
            of :meth:`CRAIGCorset.select_coreset` for more info about this parameter. Default is `dense`.
        :param kwargs: a set of keyword arguments to be used to compute the coresets. They are passed
            to :meth:`CRAIGCorset.select_coreset`. Therefore, have a look at the documentation
            of :meth:`CRAIGCorset.select_coreset` to check which keyword arguments can be passed. Note that the argument
            `metric` should not be passed.
        """
        super().__init__(max_size=max_size, adaptive_size=True, total_num_classes=None, metric=metric,
                         gradient_approx=gradient_approx, aggregate_similarity_kernel=aggregate_similarity_kernel,
                         num_subsets=num_subsets, subset_generator=subset_generator,
                         frequency=frequency, frequency_mode=frequency_mode, do_initial=do_initial,
                         recompute_coreset=False, store_all_distance_kernels=store_all_distance_kernels,
                         store_all_coresets=store_all_coresets, store_all_gradients=store_all_gradients,
                         compute_all_past_gradients=False, save_space_gradients=save_space_gradients,
                         coreset_mode=coreset_mode, no_coreset_exps=None,
                         **kwargs)

        self.exp: int = exp
        """
        ID of the significant experience
        """

        self.retraining: bool = False
        """
        boolean flag that indicates whether at the end of the training process on the significant experience,
        an update of the buffer should occur when `update` is called.
        """

    def before_training_exp(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate, **kwargs):
        """
        Callback called before training on the new experience

        This callback appends the new experience to `self._train_exps` if it is a training experience never encountered
        before. If the significant experience has already been encountered, a :class:`RuntimeError` will be raised
        because this plugin assumes that the significant experience will be the last encountered experience.
        :param strategy: a strategy
        :param kwargs: a set of keyword arguments
        """
        # if the significant experience has already been encountered, then raise an exception
        if self.exp in [exp.current_experience for exp in self._train_exps]:
            raise RuntimeError("This plugin assumes that the significant experience is the last encountered experience")
        super().before_training_exp(strategy, **kwargs)

    def before_training_subnetwork(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
                                   **kwargs):
        """
        Callback called before the training loop of the single subnetwork.

        Compute the distance kernels at the current optimisation point before the training loop starts if
        the ID of the last experience in `self._train_exps` is `exp` and  `self.do_initial` is True

        :param strategy: a strategy
        :param kwargs: some keyword arguments
        """
        if self._train_exps[-1].current_experience == self.exp:
            super().before_training_subnetwork(strategy, **kwargs)

    def after_training_epoch(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate, **kwargs):
        """
        Callback called at the end of each training epoch.

        If `self.frequency_mode` is `epoch` and the ID of the last experience in `self._train_exps` is `exp`, the
        computation of the distance kernels is performed according to the value of `self.frequency` and
        `strategy.clock.train_exp_epochs`.
        """
        if self._train_exps[-1].current_experience == self.exp:
            super().after_training_epoch(strategy, **kwargs)

    def after_training_iteration(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
                                 **kwargs):
        """
        Callback called at the end of each training iteration.

        If `self.frequency_mode` is `iteration` and the ID of the last experience in `self._train_exps` is `exp`, the
        computation of the distance kernels is performed according to the value of `self.frequency` and
        `strategy.clock.train_exp_iterations`.
        """
        if self._train_exps[-1].current_experience == self.exp:
            super().after_training_iteration(strategy, **kwargs)

    def after_training_subnetwork(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
                                  **kwargs):
        """
        Callback called at the end of the training loop of the single subnetwork.

        Compute the distance kernels at the end of the training loop of the single subnetwork.
        The computation of the distance kernels is not performed by this callback if the last periodic
        computation of the distance kernels was already performed at the convergence point or the ID of the last
        experience in `self._train_exps` is not `exp`. In contrast, the computation
        of the distance kernels is performed if the last periodic computation was not performed at the convergence
        point or `self.frequency` is 0 and the ID of the last experience in `self._train_exps` is `exp`
        """
        if self._train_exps[-1].current_experience == self.exp:
            super().after_training_subnetwork(strategy, **kwargs)

    def _compute_dist_kernels(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate, **kwargs):
        """
        Compute the distance kernels for the single subnetwork.

        The distance kernels are computed for all classes encountered in past training experiences over all samples.
        The distance kernels are not computed for the classes in the current training experience. The current training
        experience is the last one in `self._train_exps`.

        This method calls `strategy.eval` to compute the distance kernels. The `after_eval_iteration`
        callback is used to capture the gradients either w.r.t the logits or w.r.t the weights and biases in the
        classification head and then these gradients are used to compute the distance kernels.

        During the `strategy.eval` calls performed by this callback, the metrics in `EvaluationPlugin` are not
        computed as they would be meaningless and their computation would only be a computational overhead.

        :param strategy: a strategy
        :param kwargs: custom arguments
        """
        exps = self._train_exps[:-1]  # get all experiences except the current one
        super()._compute_dist_kernels_for_each_exp(exps, strategy, **kwargs)

    def after_eval_dataset_adaptation(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
                                      **kwargs):
        """
        Callback called after each eval dataset adaptation.

        This callback overrides the one in the superclass. Unlike the method in the superclass, this method performs
        nothing.
        :param strategy: a strategy
        :param kwargs: some keyword arguments
        """
        pass

    def update_from_dataset(self, new_data: AvalancheDataset, strategy: Optional[
                            DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate] = None):
        """
        Update the replay buffer according to the distance kernels computed in `self.curr_distance_kernels`.

        If `self.curr_distance_kernels` is empty, which means that the last encountered experience is not the
        significant experience, then all the samples in `new_data` are stored. Otherwise, if
        `self.curr_distance_kernels` is not empty, which means that the last encountered experience is the significant
        experience, the coreset of each class in `self.curr_distance_kernels` is computed from its aggregate similarity
        kernel as described in "Coresets for Data-efficient Training of Machine Learning Models" and
        "Towards Sustainable Learning: Coresets for Data-efficient Deep Learning" only if `self.retraining` is False.
        If `self.retraining` is True and `self.curr_distance_kernels` is not empty, nothing is performed.

        :param new_data: dataset of the current experience
        :param strategy: a strategy
        """
        # if `self.curr_distance_kernels` is empty, which means that the last encountered experience is not the
        # significant experience, then all the samples in `new_data` are stored
        if len(self.curr_distance_kernels) == 0:
            if not len(strategy.model.subnetworks_classes) == 1:
                raise RuntimeError("This plugin only works when there is a single subnetwork")
            # the classes seen by the unique subnetwork
            sub_id_cls = list(strategy.model.subnetworks_classes.values())[0]
            targets = new_data.targets  # get the targets DataAttribute from `new_data`

            # Get sample idxs per class
            cl_idxs: Dict[int, List[int]] = defaultdict(list)
            for idx, target in enumerate(targets):
                # Conversion to int may fix issues when target is a single-element torch.tensor
                target = int(target)
                # convert the target label into the order with which it appears in `sub_id_cls`
                order = sub_id_cls.index(target)
                cl_idxs[order].append(idx)

            # store all the samples of each class in the buffer and update the current coreset
            for order, idxs in cl_idxs.items():
                self.buffer_groups[order] = new_data.subset(idxs)
                # get the indices from the `dataset_indices` DataAttribute and convert it into a numpy array
                np_indices = np.asarray(self.buffer_groups[order].dataset_indices, dtype=np.int64)
                # set the marginal gains of each sample to 0
                mar_gains = np.zeros(len(idxs), dtype=np.float64)
                # set the weight of each sample to 1
                weights = np.ones(len(idxs), dtype=np.int64)
                # store the coreset of the current class into `self.curr_coreset`
                self.curr_coreset[order] = (np_indices, mar_gains, weights)

            # add the new coreset to `self.all_coresets` if required
            if self.store_all_coresets:
                self._store_curr_coreset()
        else:
            # if `self.curr_distance_kernels` is not empty, which means that the last encountered experience is the
            # significant experience, the coreset of each class in `self.curr_distance_kernels` is computed from its
            # aggregate similarity kernel only if `self.retraining` is False
            if not self.retraining:
                super().update_from_dataset(new_data, strategy)


class ForgettingEventsSubsetGenerator(SubsetGeneratorGradientBasedReplayBufferSelection,
                                      DynamicNetworkInitFeatureExtractorIncrementalOutlierPlugin,
                                      supports_distributed=True):
    """
    Subset generator based on the number of forgetting events
    for class:`training.plugins.GradientBasedReplayBufferSelectionSGDPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`.
    A forgetting event occurs when a sample transitions from being correctly classified to being misclassified.
    This definition of a forgetting event is taken from
    "An Empirical Study of Example Forgetting during Deep Neural Network Learning".

    When `generate_subsets` is called, the number of forgetting events that have been collected during training are used
    to divide the samples into the given number of subsets. The subsets are created by first sorting the samples in
    descending order based on their number of forgetting events. After sorting, disjoint subsets are constructed
    sequentially from this ordered list.

    The `after_training_iteration` callback collects the number of forgetting events during training.

    The `after_training_exp` callback stores the number of forgetting events collected so far if the attribute
    `store_all_forg_events` is True and resets the state of the forgetting metric.

    .note::
        This plugin *assumes* that the "dataset_indices" DataAttribute is present in the datasets of the training
        experiences

    .note::
        This plugin metric *only* works when there is *only* one subnetwork

    .note::
        The `after_training_exp` callback, called at the end of each training experience, resets the state of the
        forgetting metric. Thus, whatever piece of code that calls `generate_subsets`, must call it before the
        forgetting metric's state is reinitialised. If a plugin calls `generate_subsets` within the
        `after_training_exp` callback, then that plugin must come before this plugin in the `plugins` list.

    .note::
        This class acts both as a plugin and a subset generator. Therefore, the same instance of this class must be
        provided to the list of plugins when
        instantiating :class:`training.templates.DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
        and :class:`training.plugins.GradientBasedReplayBufferSelectionSGDPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`.
    """
    def __init__(self, store_all_forg_events: bool = False):
        """
        Create a new ForgettingEventsSubsetGenerator
        :param store_all_forg_events: (optional) whether to store all the forgetting events collected during each
            training experience. Default is False.
        """
        super().__init__()

        self.all_forg_events: List[Tuple[np.ndarray, np.ndarray]] = []
        """
        a list of tuples. The first element of each tuple is an array of indices of the samples whose number of
        forgetting events have been tracked of during the given training experience sorted into descending order
        according to the respective number of forgetting events. The second element is an array of respective number
        of forgetting events.
        """

        self.store_all_forg_events: bool = store_all_forg_events
        """
        whether to store all the forgetting events collected during each training experience
        """

        self._forgetting_metric: ForgettingEvents = ForgettingEvents()
        """
        the standalone metric for keeping track of the number of forgetting events for the samples in a dataset
        """

    def _generate_more_than_one_subsets(self, n_subsets: int, tot_coreset_size: int,
                                        indices: np.ndarray) -> Tuple[List[np.ndarray], List[int]]:
        """
        Generate a given number of subsets from the number of forgetting events collected so far for each sample in
        `indices`. The subsets are created by first sorting the samples in `indices` in descending order based on their
        number of forgetting events. After sorting, the disjoint subsets are constructed sequentially from this ordered
        list.
        :param n_subsets: the number of subsets to generate. It is assumed that `n_subsets` is greater than 1.
        :param tot_coreset_size: the total coreset size
        :param indices: the indices of the samples that must be divided into `n_subsets` disjoint subsets. The index of
            a given sample is the index to be used to retrieve the given sample from the underlying dataset.
        :return: a list containing the subsets and a list of respective coreset sizes. Each subset is encoded as an
            array of indices where each index refers to the position of a given sample in `indices`. The sum of the
            coreset sizes for each subset is equal to `tot_coreset_size`.
        """
        # get the size of the coreset for each subset
        size_coreset_subsets = sg.distribute_quantity(tot_coreset_size, n_subsets)
        # get the size of each subset
        size_subsets = sg.distribute_quantity(len(indices), n_subsets)
        # `samples_order` contains all samples tracked so far sorted into descending order based on their number of
        # forgetting events
        samples_order, _ = self._forgetting_metric.result()
        # filter out all samples that are not in `indices`
        samples_order = samples_order[np.isin(samples_order, indices)]
        # if the lengths are different then some samples in `indices` have not been tracked by the forgetting events
        # metric
        if not len(samples_order) == len(indices):
            raise RuntimeError("The length of samples_order must match the length of indices.")
        # for each sample in `samples_order` get the index where it appears in `indices`
        samples_indexes = np.asarray([np.where(indices == s)[0][0] for s in samples_order])

        subsets = []
        for size in size_subsets:
            subsets.append(samples_indexes[:size])
            samples_indexes = samples_indexes[size:]

        return subsets, size_coreset_subsets

    def after_train_datasets_adaptation(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
                                        **kwargs):
        """
        Callback called after the train datasets adaptation

        This callback verifies that there is a single subnetwork by checking that `strategy.adapted_dataset` contains
        the dataset of a single subnetwork. If there are multiple subnetworks, this callback raises
        a :class`RuntimeError`
        :param strategy: a strategy
        :param kwargs: some keyword arguments
        """
        if len(strategy.adapted_dataset) > 1:
            raise RuntimeError("This plugin only works when there is a single subnetwork")

    def after_training_iteration(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
                                 **kwargs):
        """
        Callback called after each training iteration.
        It updates the number of forgetting events.
        :param strategy: a strategy
        :param kwargs: some keyword arguments
        """
        if "dataset_indices" not in strategy.use_in_getitem_indices.keys():
            raise RuntimeError("The dataset of the current training experience must have the `dataset_indices` "
                               "DataAttribute")
        self._forgetting_metric.update(predicted_y=strategy.mb_output_incremental_classifier,
                                       true_y=strategy.mb_y_incremental_classifier,
                                       indices=strategy.mb_dataset_indices)

    def after_training_exp(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate, **kwargs):
        """
        Callback called at the end of each training experience.
        It resets the state of the forgetting metric and stores the number of forgetting events collected so far if
        `self.store_all_forg_events` is True.
        :param strategy: a strategy
        :param kwargs: some keyword arguments
        """
        if self.store_all_forg_events:
            self.all_forg_events.append(self._forgetting_metric.result())
        self._forgetting_metric.reset()


class FreezeUnfreezeSubnetWeightsExceptNewOutputNodesDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate(
    DynamicNetworkInitFeatureExtractorIncrementalOutlierPlugin, supports_distributed=True):
    """
    Freeze and unfreeze subnetwork weights except for the weights and biases of the newly allocated output nodes
    plugin for :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`.

    This plugin freezes all the weights of each :class:`IncrementalClassifierOutlierDetectorWithInitFeatureExtractor`
    subnetwork, except for the weights and biases of the newly allocated output nodes in the classifier head of their
    incremental classifier, prior to start training on a new experience. The number of newly allocated output nodes in
    each subnetwork's incremental classifier is equal to the number of new classes contained in the new experience that
    need to be trained on the given subnetwork. This plugin also allows to unfreeze all the weights of each subnetwork
    after a given number of training epochs. At the end of training, all the weights of a subnetwork are **always**
    unfrozen.

    This plugin also allows, during the freezing phase, to treat all the samples of the new classes contained in the
    new experience that need to be trained on a given subnetwork as if they belonged to a single class,
    effectively performing the training process of a given subnetwork during the freezing phase as if only a single
    new class in the new experience needs to be trained on the given subnetwork.
    Specifically, all the samples of the new classes contained in the new experience that need to be trained on a
    given subnetwork are treated as if they belonged to the class corresponding to the first newly
    allocated output node in the classifier head of the given subnetwork's incremental classifier. Such output node is
    called the proxy output node. The loss function is only applied to the output of the old output nodes and the proxy
    output node. Hence, among all the newly allocated output nodes, only the weights and biases of the proxy output node
    are actually updated during the training process of the freezing phase as no gradients are computed for the other
    ones. The weights and biases of the other newly allocated output nodes are set to the value of the weights and
    biases of the proxy output node, respectively, after every training iteration.

    This plugin allows to perform some checks during the training of a subnetwork on the new experience. The checks
    performed are the following:
        - At the end of training if no unfreezing operation was performed during training or when the unfreezing
          operation is performed, this plugin checks whether the weights of a subnetwork that were supposed to be frozen
          throughout training have actually stayed unchanged. Moreover, it checks whether the weights and biases of the
          new output nodes in the classifier head of the subnetwork's incremental classifier have changed from their
          initial values.
        - At the end of training if the unfreezing operation was performed during training, this plugin checks whether
          all the weights of a subnetwork have changed from their values collected right when the unfreezing operation
          was carried.
    If one of the above checks is not satisfied, a :class:`RuntimeError` is raised.

    .note::
        Obviously, this plugin performs nothing on subnetworks that are not going to be trained during the new
        experience, i.e. they are allocated with no classes. This plugin also performs nothing on subnetworks that are
        going to be trained but were not being trained on any class or did not exist prior to the new experience.
        Additionally, this plugin performs nothing on subnetworks that will be trained on a set of classes that are not
        new, i.e. the network has already been trained on that set of classes prior to the new experience.

    .note::
        All the weights of an :class:`IncrementalClassifierOutlierDetectorWithInitFeatureExtractor` subnetwork, except
        for the weights and biases in the classifier head of its incremental classifier, are frozen by setting their
        `requires_grad` attribute to False. If specified, these weights are unfrozen after a given number of training
        epochs by setting their `requires_grad` attribute back to True. Since from the start of the new experience, the
        subnetwork's optimizer is assumed to contain all the weights of the subnetwork, it will optimize this set of
        weights as soon as their `requires_grad` attribute is set back to True.

    .note::
        The weights and biases of the old output nodes in the classifier head of a subnetwork's incremental classifier
        are frozen during a new experience by storing their values prior to start training on the new experience and
        resetting them back to those values after every training iteration. If specified, these weights are
        unfrozen after a given number of training epochs by simply not resetting them after each training iteration.
        This technique is used because the weights and biases of both the old and new output nodes are in the same
        tensor and PyTorch only allows to set the `requires_grad` attribute for the whole tensor. If the `requires_grad`
        attribute of the tensor of the weights and biases in the classifier head of a subnetwork's incremental
        classifier were set to False, it would prevent the weights ond biases of the new output nodes to be optimised.

    .note::
        If one opts for treating all the samples of the new classes contained in the new experience that need to be
        trained on a given subnetwork as if they belonged to a single class during the freezing phase, this plugin
        currently supports only the :class:`nn.CrossEntropyLoss` and
        the :class:`CrossEntropyLossMSELossLogits` loss functions.
        If the `_criterion` attribute in the :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
        strategy does not store an instance of such loss functions, a :class:`RuntimeError` will be raised.

    .note::
        This plugin, prior to start the training process of the current subnetwork, replaces the forward method of the
        loss function stored in the `_criterion` attribute of
        the :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` strategy with an alternative forward
        method that preprocesses the inputs differently,
        if one opts for treating all the samples of the new classes contained in the new experience that need to be
        trained on a given subnetwork as if they belonged to a single class during the freezing phase. When the
        unfreezing operation is performed or at the end of training if the unfreezing operations was not carried during
        training, the original forward method is set back. A :class:`RuntimeError` will be raised when the original
        forward method is set back if the instance it is bound to is different from the instance currently
        stored in the `_criterion` attribute of
        the :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` strategy. The instance the original
        forward method is bound to is the instance that was previously stored in the `_criterion` attribute.

    .note::
        If one opts for treating all the samples of the new classes contained in the new experience that need to be
        trained on a given subnetwork as if they belonged to a single class during the freezing phase, this plugin does
        not actually change the targets of the samples; it only replaces the forward method of the loss function
        with an alternative forward method that preprocesses the inputs differently. Therefore, care must be taken when
        analysing the results of the metrics computed during the freezing phase. If one wants to check how many samples
        of new classes are correctly classified as belonging to the class of the proxy output node, the confusion matrix
        should be checked instead of the class accuracies or the overall accuracy.

    .note::
        Optionally, one can decide whether to reset the state of a subnetwork's optimizer prior to start training on
        the new experience.

    .note::
        Optionally, one can decide whether to reset the state of a subnetwork's optimizer when (and if) the unfreezing
        operation is performed.

    .warning::
        This plugin assumes that all the weights of a subnetwork are included in its optimizer
    """
    def __init__(self, unfreeze: Optional[int] = None, single_new_class: bool = False,
                 reset_opt_state_before_training: bool = False, reset_opt_state_unfreeze: bool = False,
                 check: bool = False, device: Union[str, torch.device] = torch.device("cpu")):
        """
        Create a new
        FreezeUnfreezeSubnetWeightsExceptNewOutputNodesDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate
        :param unfreeze: epoch after which to unfreeze all the weights of a subnetwork. Epoch counting starts from 0.
            If None, the unfreezing operation is not performed during training. Note that the unfreezing operation is
            also not performed when the actual number of training epochs is less than or equal to this. At the end of
            training, all the weights of a subnetwork are **always** unfrozen. Default is None.
        :param single_new_class: whether to treat all the samples of the new classes contained in the new experience
            that need to be trained on a given subnetwork as if they belonged to a single class during the freezing
            phase, effectively performing the training process of a given subnetwork during the freezing phase as if
            only a single new class in the new experience needs to be trained on the given subnetwork.
            If True, only the :class:`nn.CrossEntropyLoss` and
            the :class:`CrossEntropyLossMSELossLogits` loss functions are currently supported.
            If the `_criterion` attribute in the :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
            strategy does not store an instance of such loss functions, a :class:`RuntimeError` will be raised. For
            more information on how the training process of a given subnetwork during the freezing phase is performed
            when this is True, read the main docstring of this plugin. Default is False.
        :param reset_opt_state_before_training: whether to reset the state of a subnetwork's optimizer prior to start
            training on the new experience. Default is False.
        :param reset_opt_state_unfreeze: whether to reset the state of a subnetwork's optimizer when (and if)
            the unfreezing operation is performed. Default is False.
        :param check: whether to perform some checks during training. The checks performed are the following:
            at the end of training if no unfreezing operation was performed during training or when the unfreezing
            operation is performed, check whether the weights of a subnetwork that were supposed to be frozen
            throughout training have actually stayed unchanged. Moreover, check whether the weights and biases of the
            new output nodes in the classifier head of the subnetwork's incremental classifier have changed from their
            initial values. At the end of training if the unfreezing operation was performed during training, check
            whether all the weights of a subnetwork have changed from their values collected right when the unfreezing
            operation was carried. A :class:`RuntimeError` is raised if one of these checks is not satisfied.
            Default is False.
        :param device: device where to store a copy of a subnetwork's weights used to perform the checks if `check` is
            True and the values of the weights and biases of the output nodes in the classifier head
            of a subnetwork's incremental classifier prior to start training on the new experience. Default is `cpu`.
        """
        if unfreeze is not None:
            if not isinstance(unfreeze, int):
                raise ValueError("`unfreeze` must be None or an integer number greater than or equal to 0")
            else:
                if not unfreeze >= 0:
                    raise ValueError("`unfreeze must be an integer number greater than or equal to 0`")

        super().__init__()

        self._unfreeze: Optional[int] = unfreeze
        """
        epoch after which to unfreeze all the weights of a subnetwork. Epoch counting starts from 0.
        If None, the unfreezing operation is not performed. Note that the unfreezing operation is also not performed
        when the actual number of training epochs is less than or equal to this. At the end of training, all the weights
        of a subnetwork are **always** unfrozen
        """

        self._single_new_class: bool = single_new_class
        """
        whether to treat all the samples of the new classes contained in the new experience
        that need to be trained on a given subnetwork as if they belonged to a single class during the freezing
        phase, effectively performing the training process of a given subnetwork during the freezing phase as if
        only a single new class in the new experience needs to be trained on the given subnetwork.
        If True, only the :class:`nn.CrossEntropyLoss` and
        the :class:`CrossEntropyLossMSELossLogits` loss functions are currently supported.
        If the `_criterion` attribute in the :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
        strategy does not store an instance of such loss functions, a :class:`RuntimeError` will be raised. For
        more information on how the training process of a given subnetwork during the freezing phase is performed
        when this is True, read the main docstring of this plugin.
        """

        self._reset_opt_state_before_training: bool = reset_opt_state_before_training
        """
        whether to reset the state of a subnetwork's optimizer prior to start training on the new experience
        """

        self._reset_opt_state_unfreeze: bool = reset_opt_state_unfreeze
        """
        whether to reset the state of a subnetwork's optimizer when (and if) the unfreezing operation is performed
        """

        self._check: bool = check
        """
        whether to perform some checks during training. The checks performed are the following:
        at the end of training if no unfreezing operation was performed during training or when the unfreezing
        operation is performed, check whether the weights of a subnetwork that were supposed to be frozen
        throughout training have actually stayed unchanged. Moreover, check whether the weights and biases of the
        new output nodes in the classifier head of the subnetwork's incremental classifier have changed from their
        initial values. At the end of training if the unfreezing operation was performed during training, check
        whether all the weights of a subnetwork have changed from their values collected right when the unfreezing
        operation was carried. A :class:`RuntimeError` is raised if one of these checks is not satisfied
        """

        self._device: Union[str, torch.device] = device
        """
        device where to store a copy of a subnetwork's weights used to perform the checks if `self._check` is
        True and the values of the weights and biases of the output nodes in the classifier head of a subnetwork's
        incremental classifier prior to start training on the new experience
        """

        self._subnetworks_classes_before_model_adaptation: Dict[str, int] = {}
        """
        a dictionary containing subnetwork IDs as keys and the corresponding number of classes seen so far as values.
        This dictionary must be computed before adapting
        the :class:`DynamicNetworkWithInitFeatureExtractorAndIncrementalOutlierSubNetworks` model to the new experience.
        This way, it is possible to know the number of classes seen by each subnetwork, and consequently the number of
        output nodes in the classifier head of its incremental classifier, prior to being adapted to the new experience.
        Also, this way, new subnetworks instantiated due to the new experience that did not exist before 
        won't be included in the dictionary. The subnetworks that have seen no classes so far before the adaptation to
        the new experience are filtered out. After the adaptation to the new experience, the subnetworks in this
        dictionary that will not be trained on new classes during the new experience are filtered out. This dictionary
        is emptied at the end of each training experience.
        """

        self._subnetwork_old_weights: Dict[str, torch.Tensor] = {}
        """
        a dictionary containing the name of parameters as keys and the respective old parameter values as values for the
        subnetwork currently being trained. If `self._check` is False, it only includes the weights and biases in the
        classifier head of the subnetwork's incremental classifier (note that both the values of the weights and biases
        of the old output nodes prior to start training on the new experience and the initial values of the weights and
        biases of the new output nodes are included). The old parameter values are the values of the given parameter
        prior to starting on the current experience. When (and if) the unfreezing operation is performed, the initial
        values of the weights and biases of the new output nodes are replaced with the ones at the time when the
        unfreezing operation is performed. The tensors of the parameters in this dictionary are placed into 
        `self._device`. It is emptied at the end of the subnetwork's training process. 
        """

        self._loss_forward_method: Optional[Union[Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                                                  Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                                                            torch.Tensor], torch.Tensor]]] = None
        """
        always None if `self._single_new_class` is False. If `self._single_new_class` is True, it stores the original
        forward method of the loss function stored in the `_criterion` attribute of
        the :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` strategy prior to start the training
        process of the current subnetwork if the current subnetwork is not new, was previously trained on some classes
        and is going to be trained on a set of new classes. Note that only the :class:`nn.CrossEntropyLoss` and
        the :class:`CrossEntropyLossMSELossLogits` loss functions are currently supported.
        The forward method of the former loss function takes two tensors as input and returns a tensor. The forward
        method of the latter loss takes five tensors as input and returns a tensor. It is reset back to None when
        the unfreezing operation is performed or at the end of the training process of the current subnetwork if the
        unfreezing operation was not performed during training 
        """

        self._name_classifier_head_weights: str = "subnetworks.incremental_classifier.classifier.classifier.weight"
        """
        the name of the weights tensor in the classifier head of a subnetwork's incremental classifier
        """

        self._name_classifier_head_biases: str = "subnetworks.incremental_classifier.classifier.classifier.bias"
        """
        the name of the biases tensor in the classifier head of a subnetwork's incremental classifier
        """

    def before_train_datasets_adaptation(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
                                         **kwargs):
        """
        Callback called before adapting the model to the current experience, updating the optimizers and setting up the
        training dataset for each subnetwork for the current experience.

        This callback populates `self._subnetworks_classes_before_model_adaptation`, filtering out those subnetworks
        that have seen no classes so far
        :param strategy: a :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` strategy
        :param kwargs: some keyword arguments
        """
        self._subnetworks_classes_before_model_adaptation = {sub_id: len(classes_seen) for sub_id, classes_seen in
                                                             strategy.model.subnetworks_classes.items()
                                                             if len(classes_seen) > 0}

    def after_train_datasets_adaptation(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
                                        **kwargs):
        """
        Callback called after adapting the model to the current experience, updating the optimizers and setting up the
        training dataset for each subnetwork for the current experience.

        This callback filters out those subnetworks in `self._subnetworks_classes_before_model_adaptation` that are not
        trained on new classes during the current experience
        :param strategy: a :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` strategy
        :param kwargs: some keyword arguments
        """
        self._subnetworks_classes_before_model_adaptation = {sub_id: n_old_classes for sub_id, n_old_classes in
                                                             self._subnetworks_classes_before_model_adaptation.items()
                                                             if len(strategy.model.subnetworks_classes[sub_id]) > n_old_classes}

    def before_training_subnetwork(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
                                   **kwargs):
        """
        Callback called before training a subnetwork on the current experience.

        This callback performs the following operations only if the subnetwork that is going to be trained is not new,
        was previously trained on some classes and is going to be trained on a set of new classes.

        The state of the subnetwork's optimizer is reset if `self._reset_opt_state_before_training` is True.

        If `self._single_new_class` is True, the forward method of the loss function stored in the `_criterion`
        attribute of the :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` strategy is stored in
        `self._loss_forward_method`. Also, the forward method of the loss function is replaced with a function that is
        identical to the original forward method except for the fact that it computes the loss by taking into account
        the output of only the old output nodes and the proxy output node, i.e. the first newly allocated
        output node, and treats all the samples of new classes as if they belonged to the class of the
        proxy output node. Only the :class:`nn.CrossEntropyLoss` and
        the :class:`CrossEntropyLossMSELossLogits` loss functions are currently supported.
        If the `_criterion` attribute in the :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
        strategy does not store an instance of such loss functions, a :class:`RuntimeError` will be raised.

        For each parameter in the subnetwork, its `requires_grad` attribute is set to False if it is neither the weights
        nor the biases in the classifier head of the subnetwork's incremental classifier.
        If `self._check` is True, all the parameters are deep copied into `self._subnetwork_old_weights`. Otherwise,
        only the weights and the biases in the classifier head of the subnetwork's incremental
        classifier are deep copied (they are deep copied also when `self._check` is True).
        :param strategy: a :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` strategy
        :param kwargs: some keyword arguments
        """
        curr_sub_id = strategy.curr_sub_id  # get the ID of the subnetwork that is going to be trained
        # if the subnetwork is not new, was previously trained on some classes and is going to be trained on a set of
        # new classes
        if curr_sub_id in self._subnetworks_classes_before_model_adaptation.keys():
            # reset the state of the subnetwork's optimizer if required
            if self._reset_opt_state_before_training:
                self._reset_optimizer_state(strategy.optimizer[curr_sub_id])

            if self._single_new_class:
                if not isinstance(strategy._criterion, (nn.CrossEntropyLoss, CrossEntropyLossMSELossLogits)):
                    raise RuntimeError("The `_criterion` attribute of the strategy must store either the "
                                       "`nn.CrossEntropyLoss` or the "
                                       "`training.loss_functions.CrossEntropyLossMSELossLogits` loss functions")
                if self._loss_forward_method is not None:
                    raise RuntimeError("`self._loss_forward_method` is supposed to be None prior to start training of "
                                       "a subnetwork")
                # store the original forward method of the loss function
                self._loss_forward_method = strategy._criterion.forward
                # the number of old output nodes in the class. head of the subnetwork's incremental classifier
                n_old_output_nodes = self._subnetworks_classes_before_model_adaptation[curr_sub_id]

                if isinstance(strategy._criterion, nn.CrossEntropyLoss):
                    # replace the forward method of `strategy._criterion`
                    self._replace_cross_entropy_loss_forward(strategy._criterion, n_old_output_nodes)
                else:
                    # replace the forward method of `strategy._criterion`
                    self._replace_cross_entropy_mean_squared_err_loss_forward(strategy._criterion, n_old_output_nodes)

            for name, param in strategy.model.subnetworks[curr_sub_id].named_parameters():
                # if `param` is the weights or the biases in the classifier head of the subnetwork's incremental
                # classifier or `self._check` is True, add a deep copy of `param` and its name into
                # `self._subnetwork_old_weights`.
                if name in [self._name_classifier_head_weights, self._name_classifier_head_biases] or self._check:
                    # create an empty tensor with the same shape of `param` and its device set to `self._device`
                    param_copy = torch.empty_like(param, device=self._device)
                    with torch.no_grad():  # it avoids any unnecessary gradient tracking
                        # move in-place the values of `param` into `param_copy`, effectively computing a deep copy
                        param_copy.copy_(param)
                    self._subnetwork_old_weights[name] = param_copy
                # set the `requires_grad` attribute to False if `param` is neither the weights
                # nor the biases in the classifier head of the subnetwork's incremental classifier
                if name not in [self._name_classifier_head_weights, self._name_classifier_head_biases]:
                    param.requires_grad = False

    def after_training_subnetwork(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
                                  **kwargs):
        """
        Callback called after training a subnetwork on the current experience.

        This callback performs the following operations only if the subnetwork that has just been trained is not new,
        was previously trained on some classes and has just been trained on a set of new classes.

        All the weights of the subnetwork are unfrozen and `self._subnetwork_old_weights` is emptied.

        If no unfreezing operation was performed during training and `self._single_new_class` is True, replace the
        forward method of the loss in `strategy._criterion` with the original forward method stored in
        `self._loss_forward_method`. Additionally, set `self._loss_forward_method` back to None.

        If `self._check` is True, if no unfreezing operation was performed during training, check whether the weights
        of the subnetwork that were supposed to be frozen throughout training have actually stayed unchanged. Moreover,
        check whether the weights and biases of the new output nodes in the classifier head of the subnetwork's
        incremental classifier have changed from their initial values. If the unfreezing operation was performed during
        training, check whether all the weights of the subnetwork have changed from their values collected
        right when the unfreezing operation was carried.
        :param strategy: a :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` strategy
        :param kwargs: some keyword arguments
        """
        curr_sub_id = strategy.curr_sub_id  # get the ID of the subnetwork that has just been trained
        # if the subnetwork is not new, was previously trained on some classes and has just been trained on a set of
        # new classes
        if curr_sub_id in self._subnetworks_classes_before_model_adaptation.keys():
            if self._check:
                # if the unfreezing operation has not been performed
                if self._unfreeze is None or strategy.curr_epoch <= self._unfreeze:
                    # check whether the weights that were supposed to be frozen, actually stayed frozen.
                    # Also, check if the weights and biases of the new output nodes in the classifier head of the
                    # subnetwork's incremental classifier have changed throughout training from their initial values.
                    # Raise a RuntimeError if the checks are not satisified
                    self._check_frozen_weights_not_changed_new_output_nodes_changed(
                        strategy.model.subnetworks[curr_sub_id], curr_sub_id)
                else:  # if the unfreezing operation has been performed
                    # if not all weights have changed since the unfreezing operation, raise a RuntimeError
                    self._check_all_weights_change(strategy.model.subnetworks[curr_sub_id])

            # if the unfreezing operation has not been performed and `self._single_new_class` is True
            if (self._unfreeze is None or strategy.curr_epoch <= self._unfreeze) and self._single_new_class:
                # replace the forward method of the loss in `strategy._criterion` with the original forward method
                # stored in `self._loss_forward_method`. Additionally, set `self._loss_forward_method` back to None.
                self._set_forward_method(strategy._criterion)

            self._unfreeze_model(strategy.model.subnetworks[curr_sub_id])
            self._subnetwork_old_weights = {}

    def before_training_epoch(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate, **kwargs):
        """
        Callback called before each epoch training.

        This callback performs the following operations only if the subnetwork that is being trained is not new,
        was previously trained on some classes and is being trained on a set of new classes.

        If the current epoch is equal to the epoch in `self._unfreeze` + 1: unfreeze all the weights of the subnetwork,
        reset the state of the subnetwork's optimizer if `self._reset_opt_state_unfreeze` is True, replace the forward
        method of the loss in `strategy._criterion` with the original forward method stored in
        `self._loss_forward_method` and set `self._loss_forward_method` back to None,  and if `self._check`
        is True, check whether the weights of the subnetwork that were supposed to be frozen throughout training have
        actually stayed unchanged. Moreover, check whether the weights and biases of the new output nodes in the
        classifier head of the subnetwork's incremental classifier have changed from their initial values.

        :param strategy: a :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` strategy
        :param kwargs: some keyword arguments
        """
        curr_sub_id = strategy.curr_sub_id  # get the ID of the subnetwork that is being trained
        # if the subnetwork is not new, was previously trained on some classes and is being trained on a set of
        # new classes
        if curr_sub_id in self._subnetworks_classes_before_model_adaptation.keys():
            # if the unfreezing operation must be performed
            if self._unfreeze is not None and strategy.curr_epoch == self._unfreeze + 1:
                # unfreeze all the weights of the model
                self._unfreeze_model(strategy.model.subnetworks[curr_sub_id])
                if self._reset_opt_state_unfreeze:
                    self._reset_optimizer_state(strategy.optimizer[curr_sub_id])
                if self._single_new_class:
                    # replace the forward method of the loss in `strategy._criterion` with the original forward method
                    # stored in `self._loss_forward_method`. Additionally, set `self._loss_forward_method` back to None.
                    self._set_forward_method(strategy._criterion)
                if self._check:
                    # check whether the weights that were supposed to be frozen, actually stayed frozen.
                    # Also, check if the weights and biases of the new output nodes in the classifier head of the
                    # subnetwork's incremental classifier have changed throughout training from their initial values.
                    # A RuntimeError is raised if these checks are not satisfied
                    self._check_frozen_weights_not_changed_new_output_nodes_changed(
                        strategy.model.subnetworks[curr_sub_id], curr_sub_id)
                    n_old_output_nodes = self._subnetworks_classes_before_model_adaptation[curr_sub_id]
                    # a dictionary containing the parameter names as keys and their respective current values as values
                    weight_dict = dict(strategy.model.subnetworks[curr_sub_id].named_parameters())
                    # set the value of the weights of the new output nodes in `self._subnetwork_old_weights[name]` to
                    # the current ones
                    self._subnetwork_old_weights[self._name_classifier_head_weights][n_old_output_nodes:].copy_(
                        weight_dict[self._name_classifier_head_weights][n_old_output_nodes:]
                    )
                    # set the value of the biases of the new output nodes in `self._subnetwork_old_weights[name]` to
                    # the current ones
                    self._subnetwork_old_weights[self._name_classifier_head_biases][n_old_output_nodes:].copy_(
                        weight_dict[self._name_classifier_head_biases][n_old_output_nodes:]
                    )

    def after_update(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate, **kwargs):
        """
        Callback called after each optimizer's step call.

        This callback performs the following operations only if the subnetwork that is being trained is not new,
        was previously trained on some classes and is being trained on a set of new classes.

        Reset the weights and biases of the old output nodes in the classifier head of the subnetwork's incremental
        classifier to their values prior to start training on the new experience if `self._unfreeze` is None or the
        current epoch number is less than or equal to `self._unfreeze`. If `self._unfreeze` is None or the
        current epoch number is less than or equal to `self._unfreeze`, and `self._single_new_class` is True, set the
        weights and biases of all the newly allocated output nodes to the value of the weights and biases, respectively,
        of the proxy output node, i.e. the first newly allocated output node.

        :param strategy: a :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` strategy
        :param kwargs: some keyword arguments
        """
        curr_sub_id = strategy.curr_sub_id  # get the ID of the subnetwork that is being trained
        # if the subnetwork is not new, was previously trained on some classes and is being trained on a set of
        # new classes
        if curr_sub_id in self._subnetworks_classes_before_model_adaptation.keys():
            # if the unfreezing operation has not been performed yet
            if self._unfreeze is None or strategy.curr_epoch <= self._unfreeze:
                # a dictionary of the current parameters of the subnetwork
                param_dict = dict(strategy.model.subnetworks[curr_sub_id].named_parameters())
                # the number of old output nodes in the class. head of the subnetwork's incremental classifier
                n_old_output_nodes = self._subnetworks_classes_before_model_adaptation[curr_sub_id]
                with torch.no_grad():  # it avoids any unnecessary gradient tracking
                    # reset the weights of the old output nodes
                    param_dict[self._name_classifier_head_weights][:n_old_output_nodes].copy_(
                        self._subnetwork_old_weights[self._name_classifier_head_weights][:n_old_output_nodes]
                    )
                    # reset the biases of the old output nodes
                    param_dict[self._name_classifier_head_biases][:n_old_output_nodes].copy_(
                        self._subnetwork_old_weights[self._name_classifier_head_biases][:n_old_output_nodes]
                    )
                    if self._single_new_class:
                        # set the weights of all the newly allocated output nodes to the value of the weights
                        # of the proxy output node
                        param_dict[self._name_classifier_head_weights][n_old_output_nodes + 1:].copy_(
                            param_dict[self._name_classifier_head_weights][n_old_output_nodes: n_old_output_nodes + 1]
                        )
                        # set the biases of all the newly allocated output nodes to the value of the bias
                        # of the proxy output node
                        param_dict[self._name_classifier_head_biases][n_old_output_nodes + 1:].copy_(
                            param_dict[self._name_classifier_head_biases][n_old_output_nodes: n_old_output_nodes + 1]
                        )

    def after_training_exp(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate, **kwargs):
        """
        Callback called after training on the current experience.

        It empties `self._subnetworks_classes_before_model_adaptation`
        :param strategy: a :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` strategy
        :param kwargs: some keyword arguments
        """
        self._subnetworks_classes_before_model_adaptation = {}

    def _check_frozen_weights_not_changed_new_output_nodes_changed(self,
                                                  subnet: IncrementalClassifierOutlierDetectorWithInitFeatureExtractor,
                                                  curr_sub_id: str):
        """
        Check whether the current values of the weights and biases of the new output nodes in the classifier head of the
        current subnetwork's incremental classifier have changed throughout training from their initial values stored in
        `self._subnetwork_old_weights`. Moreover, check whether all the other weights are equal to the ones stored in
        `self._subnetwork_old_weights`. If these checks are not satisfied, :class:`RuntimeError` is raised.
        :param subnet: current subnetwork
        :param curr_sub_id: ID of the current subnetwork
        """
        for name, param in subnet.named_parameters():
            if name in [self._name_classifier_head_weights, self._name_classifier_head_biases]:
                with torch.no_grad():
                    # check whether the old values of the weights or biases of the old output nodes match
                    # the current values and whether the current values of the weights or biases of the new
                    # output nodes are different from their initial values
                    param = param.to(self._device)
                    n_old_output_nodes = self._subnetworks_classes_before_model_adaptation[curr_sub_id]
                    if not self._check_tensor_equal(param[:n_old_output_nodes],
                                                    self._subnetwork_old_weights[name][:n_old_output_nodes]):
                        raise RuntimeError("The weights or biases of the old output nodes, supposed to be "
                                           "frozen throughout training, have changed")
                    if self._check_tensor_equal(param[n_old_output_nodes:],
                                                self._subnetwork_old_weights[name][n_old_output_nodes:]):
                        raise RuntimeError("The weights or biases of the new output nodes, supposed to "
                                           "change throughout training, have not changed")
            else:
                with torch.no_grad():
                    # check whether the old values of this parameter match the current values
                    if not self._check_tensor_equal(param.to(self._device),
                                                    self._subnetwork_old_weights[name]):
                        raise RuntimeError("One or more parameters, supposed to be frozen throughout training, have "
                                           "changed")

    def _check_all_weights_change(self, subnet: IncrementalClassifierOutlierDetectorWithInitFeatureExtractor):
        """
        Check whether the current values of all the weights in the current subnetwork are different from the ones
        stored in  `self._subnetwork_old_weights`. A :class:`RuntimeError` is raised if one or more weights have not
        changed.
        :param subnet: current subnetwork
        """
        for name, param in subnet.named_parameters():
            with torch.no_grad():
                if self._check_tensor_equal(param.to(self._device), self._subnetwork_old_weights[name]):
                    raise RuntimeError("One or more parameters, supposed to change, have not changed after the "
                                       "unfreezing operation")

    def _set_forward_method(self, loss: Union[nn.CrossEntropyLoss, CrossEntropyLossMSELossLogits]):
        """
        Set the forward method of the loss to the forward method stored in `self._loss_forward_method`. Subsequently,
        set `self._loss_forward_method` to None.
        .note::
            A :class:`RuntimeError` is raised if the forward method stored in `self._loss_forward_method` is not bound
            to the `loss` instance.
        :param loss: an instance of :class:`nn.CrossEntropyLoss` or :class:`CrossEntropyLossMSELossLogits`
        """
        if not isinstance(loss, (nn.CrossEntropyLoss, CrossEntropyLossMSELossLogits)):
            raise RuntimeError("`loss` must be an instance of either `nn.CrossEntropyLoss` or "
                               "`training.loss_functions.CrossEntropyLossMSELossLogits`")
        if self._loss_forward_method is None:
            raise RuntimeError("The attribute `self._loss_forward_method` cannot be None")

        if self._loss_forward_method.__self__ is not loss:
            raise RuntimeError("`self._loss_forward_method` is bound to an instance that is not `loss`")

        loss.forward = self._loss_forward_method
        self._loss_forward_method = None  # set it to None

    @staticmethod
    def _unfreeze_model(model: nn.Module):
        """
        Unfreezes all the parameters of a model
        :param model: a model
        """
        for param in model.parameters():
            param.requires_grad = True

    @staticmethod
    def _reset_optimizer_state(optimizer: Optimizer):
        """
        Reset the state of an optimizer
        :param optimizer: an optimizer
        """
        optimizer.state = defaultdict(dict)

    @staticmethod
    def _check_tensor_equal(t1: torch.Tensor, t2: torch.Tensor) -> bool:
        """
        Check whether two tensors are equal.
        :param t1: a tensor
        :param t2: a tensor
        :return: True if the two tensors are equal. False, otherwise.
        """
        return bool(torch.eq(t1, t2).all())

    @staticmethod
    def _replace_cross_entropy_loss_forward(ce_loss: nn.CrossEntropyLoss, n_old_output_nodes: int):
        """
        Replace the forward method of an instance of :class:`nn.CrossEntropyLoss` with a function identical to the
        original forward method except for the fact that it computes the loss by taking into account
        the output of only the old output nodes and the proxy output node, i.e. the first newly allocated output node,
        and treats all the samples of new classes as if they belonged to the class of the proxy output node.
        :param ce_loss: a :class:`nn.CrossEntropyLoss` instance
        :param n_old_output_nodes: the number of old output nodes
        """
        if not isinstance(ce_loss, nn.CrossEntropyLoss):
            raise ValueError("`ce_loss` must be an instance of `nn.CrossEntropyLoss`")

        def new_forward(self, input, target):
            # replace the target of new classes with the target of the proxy output node
            target[target > n_old_output_nodes] = n_old_output_nodes
            # apply the original forward method to the output of only the old output nodes and the proxy output node
            # and to the target tensor where the target of new classes is replaced with the target of the proxy output
            # node
            return nn.CrossEntropyLoss.forward(self, input[:, :n_old_output_nodes + 1], target)

        # replace the original forward method with `new_forward`
        ce_loss.forward = new_forward.__get__(ce_loss)

    @staticmethod
    def _replace_cross_entropy_mean_squared_err_loss_forward(ce_mse_loss: CrossEntropyLossMSELossLogits,
                                                             n_old_output_nodes: int):
        """
        Replace the forward method of an instance of :class:`CrossEntropyLossMSELossLogits` with a function identical to
        the original forward method except for the fact that it computes the loss by taking into account
        the output of only the old output nodes and the proxy output node, i.e. the first newly allocated output node,
        and treats all the samples of new classes as if they belonged to the class of the proxy output node.
        :param ce_mse_loss: a :class:`CrossEntropyLossMSELossLogits` instance
        :param n_old_output_nodes: the number of old output nodes
        """
        if not isinstance(ce_mse_loss, CrossEntropyLossMSELossLogits):
            raise ValueError("`ce_mse_loss` must be an instance of "
                             "`training.loss_functions.CrossEntropyLossMSELossLogits`")

        def new_forward(self, current_logits_no_past_logits, current_logits_past_logits,
                        past_logits, targets_no_past_logits, targets_past_logits):
            # replace the target of new classes in `targets_no_past_logits` with the target of the proxy output node
            targets_no_past_logits[targets_no_past_logits > n_old_output_nodes] = n_old_output_nodes
            # replace the target of new classes in `targets_past_logits` with the target of the proxy output node
            # this is done although there should not be samples of new classes that have past logits
            targets_past_logits[targets_past_logits > n_old_output_nodes] = n_old_output_nodes

            if not current_logits_no_past_logits.numel() == 0:  # if not empty
                # preserve the output of only the old output nodes and the proxy output node
                current_logits_no_past_logits = current_logits_no_past_logits[:, :n_old_output_nodes + 1]
            if not current_logits_past_logits.numel() == 0:  # if not empty
                # preserve the output of only the old output nodes and the proxy output node
                current_logits_past_logits = current_logits_past_logits[:, :n_old_output_nodes + 1]
            if not past_logits.numel() == 0:  # if not empty
                # preserve the output of only the old output nodes and the proxy output node
                past_logits = past_logits[:, :n_old_output_nodes + 1]

            return CrossEntropyLossMSELossLogits.forward(self, current_logits_no_past_logits,
                                                         current_logits_past_logits, past_logits,
                                                         targets_no_past_logits, targets_past_logits)

        # replace the original forward method with `new_forward`
        ce_mse_loss.forward = new_forward.__get__(ce_mse_loss)


class LrSchedulerPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate(
    DynamicNetworkInitFeatureExtractorIncrementalOutlierPlugin, supports_distributed=True):
    """
    Learning rate scheduler plugin for :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`.

    This plugin takes in a PyTorch scheduler from `torch.optim.lr_scheduler` and during training of each subnetwork
    performs calls to the `step()` method to adjust the learning rate accordingly.

    ..note:
        Any learning rate scheduler from `torch.optim.lr_scheduler` can be used apart
        from :class:`torch.optim.lr_scheduler.ReduceLROnPlateau`. The latter cannot be used because the validation loss
        must be provided to the `step()` method while all the other schedulers do not accept any argument when calling
        the `step()` method.
    """
    def __init__(self, scheduler: Type[LRScheduler], step_calls: Literal["epoch", "iteration"] = "epoch",
                 reset_lrs: bool = True,
                 state_scheduler: Optional[
                     Callable[
                         [int, LrSchedulerPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate],
                         None
                     ]
                 ] = None,
                 **kwargs):
        """
        Create a new LrSchedulerPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate
        :param scheduler: a class subclassing :class:`LRScheduler`. Note that a class must be provided, not an instance.
        :param step_calls: when to perform the calls to the `step()` method. Either `epoch` or `iteration` to
            perform the calls to the `step()` method after each epoch or iteration, respectively. Default is `epoch`.
        :param reset_lrs: whether to reset the learning rate of the optimizer of each subnetwork to its initial value
            after the training of the current experience ends. Default is True.
        :param state_scheduler: a callable that takes in the ID of the current training experience and a reference to
            this instance and returns None. If not None, this callable is called before adapting the model to the
            current experience, updating the optimizers and setting up the training dataset for each subnetwork for the
            current training experience. This callable can be used to modify the state of this instance prior to start
            training on each experience. Default is None.
        :param kwargs: a set of keyword arguments to be passed at construction time when instantiating the scheduler.
            Note that the keyword arguments that can be passed are any arguments accepted by the constructor of the
            given scheduler apart from the `optimizer` one. This plugin will pass the optimizer itself.
        """
        super().__init__()

        if not issubclass(scheduler, LRScheduler):
            raise ValueError("`scheduler` must be a subclass of `LRScheduler`")
        if step_calls not in ["epoch", "iteration"]:
            raise ValueError("`step_calls` must be either epoch or iteration")

        self.scheduler: Type[LRScheduler] = scheduler
        """
        a class subclassing :class:`LRScheduler`
        """

        self.step_calls: Literal["epoch", "iteration"] = step_calls
        """
        when to perform the calls to the `step()` method. Either `epoch` or `iteration` to
        perform the calls to the `step()` method after each epoch or iteration, respectively.
        """

        self.reset_lrs: bool = reset_lrs
        """
        whether to reset the learning rate of the optimizer of each subnetwork to its initial value
        after the training of the current experience ends
        """

        self.kwargs = kwargs
        """
        a set of keyword arguments to be passed at construction time when instantiating the scheduler
        """

        self.scheduler_instncs: Dict[str, LRScheduler] = {}
        """
        a dictionary containing the subnetwork IDs as keys and the respective scheduler instances as values
        """

        self._state_scheduler: Optional[
            Callable[[int, LrSchedulerPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate], None]
        ] = state_scheduler
        """
        a callable that takes in the ID of the current training experience and a reference to
        this instance and returns None. If not None, this callable is called before adapting the model to the
        current experience, updating the optimizers and setting up the training dataset for each subnetwork for the
        current training experience. This callable can be used to modify the state of this instance prior to start
        training on each experience.
        """

        self._init_lrs: Dict[str, List[float]] = defaultdict(list)
        """
        a dictionary containing the subnetwork IDs as keys and a list of initial learning rates as values, one for
        each param group 
        """

    def before_train_datasets_adaptation(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
                                         **kwargs):
        """
        Callback called before adapting the model to the current experience, updating the optimizers and setting up the
        training dataset for each subnetwork for the current experience.

        This callback calls `self._state_scheduler` if it is not None.
        :param strategy: a strategy
        :param kwargs: some keyword arguments
        """
        if self._state_scheduler is not None:
            self._state_scheduler(strategy.experience.current_experience, self)

    def after_train_datasets_adaptation(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
                                        **kwargs):
        """
        Callback called after adapting the model to the current experience, updating the optimizers and setting up the
        training dataset for each subnetwork for the current experience.

        This callback creates a scheduler instance for the optimizer of each subnetwork.
        :param strategy: a strategy
        :param kwargs: some keyword arguments to be passed
        """
        # creating a scheduler instance for each subnetwork's optimizer
        for sub_id, opt in strategy.optimizer.items():
            opt.__class__ = torch.optim.SGD
            # create an instance of the scheduler and pass the respective optimizer and any provided keyword arguments
            self.scheduler_instncs[sub_id] = self.scheduler(optimizer=opt, **self.kwargs)
            if self.reset_lrs:
                # store the initial learning rates of each param group of the current optimizer in
                # `self._init_lrs[sub_id]`
                for group in opt.param_groups:
                    self._init_lrs[sub_id].append(group['lr'])

    def after_training_epoch(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate, **kwargs):
        """
        Callback called after each training epoch.
        This callback calls the `step()` method of the scheduler of the current subnetwork
        if `self.step_calls` is `epoch`
        :param strategy: a strategy
        :param kwargs: some keyword arguments to be passed
        """
        if self.step_calls == "epoch":
            self.scheduler_instncs[strategy.curr_sub_id].step()

    def after_training_iteration(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
                                 **kwargs):
        """
        Callback called after each training iteration.
        This callback calls the `step()` method of the scheduler of the  current subnetwork if `self.step_calls` is
        `iteration`.
        :param strategy: a strategy
        :param kwargs: some keyword arguments to be passed
        """
        if self.step_calls == "iteration":
            self.scheduler_instncs[strategy.curr_sub_id].step()

    def after_training_exp(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate, **kwargs):
        """
        Callback called after the end of the current training experience.

        It resets the learning rates of the optimizer of each subnetwork to their initial values.
        :param strategy: a strategy
        :param kwargs: some keyword arguments
        """
        self.scheduler_instncs = {}  # empty the dictionary of scheduler instances
        if self.reset_lrs:
            # reset the initial learning rate of each scheduler
            for sub_id, opt in strategy.optimizer.items():
                for i, group in enumerate(opt.param_groups):
                    group['lr'] = self._init_lrs[sub_id][i]
            self._init_lrs = defaultdict(list)  # empty the dictionary of initial learning rates


