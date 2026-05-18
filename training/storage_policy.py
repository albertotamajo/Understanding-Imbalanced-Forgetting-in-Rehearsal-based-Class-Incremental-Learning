"""
This module includes a set of storage policy buffers.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Set, TYPE_CHECKING
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from numpy import inf
from avalanche.training.storage_policy import (ClassBalancedBuffer, ExemplarsBuffer, ReservoirSamplingBuffer,
                                               ExemplarsSelectionStrategy, ParametricBuffer, BalancedExemplarsBuffer,
                                               RandomExemplarsSelectionStrategy)
from avalanche.benchmarks.utils.data_loader import collate_from_data_or_kwargs
from avalanche.benchmarks.utils import concat_datasets, AvalancheDataset, classification_subset

if TYPE_CHECKING:
    from avalanche.training.templates import SupervisedTemplate
    from training.templates import DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate


class BufferDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate(ExemplarsBuffer):
    """
    ABC for all rehearsal buffers used in the
    template :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`.

    It subclasses :class:`ExemplarsBuffer`, which is the ABC for all rehearsal buffers used in Avalanche.
    """
    @abstractmethod
    def buffer_subnetwork(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
                          sub_id: str) -> AvalancheDataset:
        """
        Get the  rehearsal buffer for a given subnetwork

        .note::
            If there is no rehearsal buffer for a given subnetwork, an empty dataset must be returned
        :param strategy: a :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` strategy
        :param sub_id: ID of a subnetwork
        :return: a rehearsal dataset of past data for the given subnetwork
        """
        ...


class ClassBalancedBufferDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate(
    ClassBalancedBuffer,
    BufferDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate
):
    """
    Store samples for replay, equally divided over classes.

    There is a separate buffer updated by reservoir sampling for each class.
    It should be called in the 'after_training_exp' phase.
    The number of classes can be fixed up front or adaptive, based on the 'adaptive_size' attribute.
    When adaptive, the memory is equally divided over all the unique observed classes so far.
    """
    def __init__(self, max_size: int, adaptive_size: bool = True, total_num_classes: Optional[int] = None,
                 seed: Optional[int] = None):
        """
        Create a new ClassBalancedBufferDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate

        :param max_size: the max capacity of the replay memory.
        :param adaptive_size: True if mem_size is divided equally over all observed experiences (keys in replay_mem).
        :param total_num_classes: if adaptive size is False, the fixed number of classes to divide capacity over.
        :param seed: a valid int used to initialize the random number generator before selecting a subset of samples
            to keep for each class when `self.update` or `self.update_from_dataset` is called.
            The random number generator seed and the random subsampling operations are set and performed,
            respectively, within a `torch.random.fork_rng` context. Hence, the primary state of torch's random generator
            is not affected. If None, no random number generator seed is set, preventing results from being replicated.
            Default is None.
        """
        super().__init__(max_size=max_size, adaptive_size=adaptive_size, total_num_classes=total_num_classes)

        self._seed: Optional[int] = seed
        """
        a valid int used to initialize the random number generator before selecting a subset of samples
        to keep for each class when `self.update` or `self.update_from_dataset` is called.
        The random number generator seed and the random subsampling operations are set and performed,
        respectively, within a `torch.random.fork_rng` context. Hence, the primary state of torch's random generator
        is not affected. If None, no random number generator seed is set, preventing results from being replicated.
        """

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
            # the respective datasets as values
            adapted_datasets = list(strategy.adapted_dataset.values())  # put the datasets of each subnet into a list
            adapted_datasets = concat_datasets(adapted_datasets)   # the datasets are concatenated into a single dataset
            self.update_from_dataset(adapted_datasets, strategy)

    def update_from_dataset(self, new_data: AvalancheDataset,
                            strategy: Optional[DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate] = None):
        """
        Update buffer using `new_data`.
        """
        if len(new_data) == 0:
            return

        # if a seed was provided at initialisation, then True; otherwise False
        enabled: bool = True if self._seed is not None else False

        # it forks the random number generator of Pytorch if `enabled` is True
        with torch.random.fork_rng(devices=("cpu", "cuda"), enabled=enabled):
            if enabled:
                torch.manual_seed(self._seed)  # set seed if `enabled` is True

            targets = getattr(new_data, "targets", None)
            assert targets is not None

            # Get sample idxs per class
            cl_idxs: Dict[int, List[int]] = defaultdict(list)
            for idx, target in enumerate(targets):
                # Conversion to int may fix issues when target
                # is a single-element torch.tensor
                target = int(target)
                cl_idxs[target].append(idx)

            # Make AvalancheSubset per class
            cl_datasets = {}
            for c, c_idxs in cl_idxs.items():
                cl_datasets[c] = classification_subset(new_data, indices=c_idxs)
            classes_in_new_data = list(cl_datasets.keys())
            classes_in_new_data.sort()  # it is sorted for replication purposes

            # Update seen classes
            self.seen_classes.update(classes_in_new_data)
            seen_classes = list(self.seen_classes)
            seen_classes.sort()  # it is sorted for replication purposes

            # associate lengths to classes
            lens = self.get_group_lengths(len(seen_classes))
            class_to_len = {}
            for class_id, ll in zip(seen_classes, lens):
                class_to_len[class_id] = ll

            # update buffers with new data
            for class_id in classes_in_new_data:
                new_data_c = cl_datasets[class_id]
                ll = class_to_len[class_id]
                if class_id in self.buffer_groups:
                    old_buffer_c = self.buffer_groups[class_id]
                    old_buffer_c.update_from_dataset(new_data_c)
                    old_buffer_c.resize(strategy, ll)
                else:
                    new_buffer = ReservoirSamplingBuffer(ll)
                    new_buffer.update_from_dataset(new_data_c)
                    self.buffer_groups[class_id] = new_buffer

            # resize buffers
            for class_id in seen_classes:
                self.buffer_groups[class_id].resize(strategy, class_to_len[class_id])

    def buffer_subnetwork(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
                          sub_id: str) -> AvalancheDataset:
        """
        Get the  rehearsal buffer for a given subnetwork.
        :param strategy: a :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` strategy
        :param sub_id: ID of a subnetwork
        :return: a rehearsal dataset of past data for the given subnetwork. If there is no rehearsal buffer for a
        given subnetwork, an empty dataset is returned.
        """
        sub_classes = strategy.model.subnetworks_classes[sub_id]
        return concat_datasets([self.buffer_groups[cls].buffer
                                for cls in self.buffer_groups.keys() if cls in sub_classes])


class HerdingClassBalancedBufferDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate(
    ParametricBuffer, BufferDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate):
    """
    Store samples for replay, equally divided over classes.

    There is a separate buffer for each class updated by using the herding selection strategy as described in iCaRL.

    When a class is first encountered, the herding selection strategy selects a subset of samples by greedily selecting
    at each iteration the remaining exemplar that makes the center of already selected exemplars as close as possible to
    the center of all elements (in the feature space). Later, when the buffer of a class must be reduced in size, the
    last selected samples are removed to match the new size.

    It should be called in the 'after_training_exp' phase.
    The number of classes can be fixed up front or adaptive, based on the 'adaptive_size' attribute.
    When adaptive, the memory is equally divided over all the unique classes observed so far.
    """
    def __init__(self, max_size: int, adaptive_size: bool = True, total_num_classes: Optional[int] = None):
        """
        Create a new HerdingClassBalancedBufferDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate
        :param max_size: the overall maximum capacity of the replay memory
        :param adaptive_size: True if `max_size` is divided equally over all unique classes observed so far.
        :param total_num_classes: if `adaptive_size` is False, the fixed number of classes to divide capacity over.
        """
        BalancedExemplarsBuffer.__init__(self, max_size=max_size, adaptive_size=adaptive_size,
                                         total_num_groups=total_num_classes)
        self.groupby = "class"
        self.selection_strategy: ExemplarsSelectionStrategy = (
            HerdingExemplarsSelectionStrategyDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate())
        self.seen_groups: Set[int] = set()

    def update(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate, **kwargs):
        """
        Update the replay memory by using the samples of the classes contained in the current experience.
        :param strategy: a strategy
        :param kwargs: some keyword arguments
        """
        if strategy.is_training:  # it does not make sense to update the buffer at evaluation time
            # when in training mode, `strategy.adapted_dataset` is a dictionary containing subnetwork IDs as keys and
            # the respective datasets as values
            adapted_datasets = list(strategy.adapted_dataset.values())  # put the datasets of each subnet into a list
            adapted_datasets = concat_datasets(adapted_datasets)   # the datasets are concatenated into a single dataset
            # split `adapted_datasets` into as many disjoint subsets as the number of unique classes in
            # `adapted_datasets`
            new_groups = self._make_groups(strategy, adapted_datasets)
            self.seen_groups.update(new_groups.keys())

            # associate lengths to classes
            lens = self.get_group_lengths(len(self.seen_groups))
            group_to_len = {}
            for group_id, ll in zip(self.seen_groups, lens):
                group_to_len[group_id] = ll

            # update buffers with new data
            for group_id, new_data_g in new_groups.items():
                ll = group_to_len[group_id]
                if group_id in self.buffer_groups.keys():
                    old_buffer_g: SelectionStrategyExemplarsBuffer = self.buffer_groups[group_id]
                    old_buffer_g.update_from_dataset(new_data_g, strategy)
                    old_buffer_g.resize(strategy, ll)
                else:
                    new_buffer = SelectionStrategyExemplarsBuffer(max_size=ll,
                                                                  selection_strategy=self.selection_strategy)
                    new_buffer.update_from_dataset(new_data_g, strategy)
                    self.buffer_groups[group_id] = new_buffer

            # resize buffers
            for group_id, class_buf in self.buffer_groups.items():
                self.buffer_groups[group_id].resize(strategy, group_to_len[group_id])

    def buffer_subnetwork(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
                          sub_id: str) -> AvalancheDataset:
        """
        Get the  rehearsal buffer for a given subnetwork.
        :param strategy: a :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` strategy
        :param sub_id: ID of a subnetwork
        :return: a rehearsal dataset of past data for the given subnetwork. If there is no rehearsal buffer for a
            given subnetwork, an empty dataset is returned.
        """
        sub_classes = strategy.model.subnetworks_classes[sub_id]
        return concat_datasets([self.buffer_groups[cls].buffer
                                for cls in self.buffer_groups.keys() if cls in sub_classes])


class SelectionStrategyExemplarsBuffer(ExemplarsBuffer):
    """
    A single buffer that stores samples for replay using an :class:`ExemplarsSelectionStrategy` to select a subset of
    exemplars from a dataset.

    `update_from_dataset` updates the single buffer using the selection strategy on the concatenation of the samples in
    the new dataset and the samples already in the buffer (if any).

    `resize` sets the new maximum capacity of the single buffer. If the new maximum capacity is less than the number of
    samples in the single buffer, the last samples in the single buffer are removed to match the  new capacity.

    .note::
        Calling the `update` method causes a :class:`NotImplementedError` to be raised.
    """
    def __init__(self, max_size: int, selection_strategy: Optional[ExemplarsSelectionStrategy] = None):
        """
        Create a new SelectionStrategyExemplarsBuffer
        :param max_size: the maximum capacity of the single buffer
        :param selection_strategy: the strategy used to select a subset of exemplars from a dataset.
            If None, a :class:`RandomExemplarsSelectionStrategy` is used. Default is None.
        """
        super().__init__(max_size=max_size)
        ss = selection_strategy or RandomExemplarsSelectionStrategy()
        self._selection_strategy: ExemplarsSelectionStrategy = ss

    def update(self, strategy: SupervisedTemplate, **kwargs):
        raise NotImplementedError("This method cannot be called")

    def update_from_dataset(self, new_data: AvalancheDataset, strategy: SupervisedTemplate):
        """
        Update the single buffer using the selection strategy on the concatenation of the samples in the new dataset and
        the samples already in the buffer (if any).
        :param new_data: a new dataset
        :param strategy: a strategy
        """
        # concatenate the samples in the new dataset with the samples already in the single buffer
        self.buffer = new_data.concat(self.buffer)
        # get the sorted list of indices from the selection strategy
        idx = self._selection_strategy.make_sorted_indices(strategy=strategy, data=self.buffer)
        # store in the buffer `self.max_size` samples. The samples in the buffer are the first  `self.max_size` samples
        # as indicated in `idx` and in the same order.
        self.buffer = self.buffer.subset(idx[: self.max_size])

    def resize(self, strategy: SupervisedTemplate, new_size: int):
        """
        Resize the single buffer. Set the new maximum capacity of the single buffer. If the new maximum capacity is less
        than the number of samples in the single buffer, the last samples in the single buffer are removed to match the
        new capacity.
        :param strategy: a strategy
        :param new_size: the new maximum capacity of the single buffer
        """
        self.max_size = new_size  # set the new size
        if self.max_size < len(self.buffer):  # if the number of samples in the buffer is greater than the new size
            # remove from the buffer the last samples so that to match the new size
            self.buffer = self.buffer.subset(list(range(self.max_size)))


class FeatureBasedExemplarsSelectionStrategyDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate(
    ExemplarsSelectionStrategy, ABC):
    """
    ABC to select exemplars from their features
    for :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`.

    Subclasses *must* implement the `make_sorted_indices_from_features` method.

    .note::
        This selection strategy *only* works when there is *only* one subnetwork
    """
    @torch.no_grad()
    def make_sorted_indices(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
                            data: AvalancheDataset, num_workers=0, pin_memory=None, **kwargs) -> List[int]:
        """
        Compute the features for each sample in `data` and return a sorted list of indices for the samples in `data`
        based on the samples' features. The samples in `data` whose indices are the last will be the first to be removed
        when the size of the buffer is reduced.
        :param strategy: a :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` strategy
        :param data: an :class:`AvalancheDataset` dataset
        :param num_workers: (optional) number of thread workers for the data loading. Default is 0.
        :param pin_memory: (optional) If True, the data loader will copy Tensors into CUDA pinned memory before
            returning them. Defaults to True.
        :return: a sorted list of indices for the samples in `data`. The samples in `data` whose indices are the last
            will be the first to be removed when the size of the buffer is reduced.
        """
        if len(strategy.model.subnetworks_classes) > 1:
            raise RuntimeError("This selection strategy only works when there is a single subnetwork")
        # get the ID of the single subnetwork
        single_subnet_id = list(strategy.model.subnetworks_classes.keys())[0]
        # save the current state of the strategy
        curr_state = strategy.save_train_state()
        # set the model in the strategy to eval mode
        strategy.model.eval()
        # a new dataset with eval transformation loaded. This is done because eval transformations usually don't have
        # any augmentation procedure. It does not affect the original dataset.
        data = data.eval()

        other_dataloader_args = strategy.obtain_common_dataloader_parameters(
            batch_size=strategy.eval_mb_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=pin_memory,
            persistent_workers=False,
        )
        collate_from_data_or_kwargs(data, other_dataloader_args)
        dataloader = DataLoader(data, **other_dataloader_args)
        # concatenate the features of all samples in `data` into a 2D tensor in the same order as they appear in `data`
        features = torch.cat([strategy.model(
                                x.to(strategy.device),
                                subnetwork_ids=single_subnet_id,
                                use_init_feature_extractor=True)[single_subnet_id]["incremental_classifier"][1]
                              for x, *_ in dataloader],
                             dim=0)
        # restore previous state
        strategy.load_train_state(curr_state)

        return self.make_sorted_indices_from_features(features)

    @abstractmethod
    def make_sorted_indices_from_features(self, features: torch.Tensor) -> List[int]:
        """
        Return a sorted list of indices for the features in `features`. The samples whose respective features in
        `features` are the last in the sorted list of indices *must* be the first to be removed when the size of the
        buffer is reduced.
        :param features: a 2D tensor of feature vectors
        :return: a sorted list of indices for the features in `features`. The samples whose respective features in
            `features` are the last in the sorted list of indices *must* be the first to be removed when the size of the
            buffer is reduced.
        """


class HerdingExemplarsSelectionStrategyDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate(
    FeatureBasedExemplarsSelectionStrategyDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate):
    """
    The herding selection strategy as described in iCaRL
    for :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`.

    It is a greedy algorithm that at each iteration selects the remaining exemplar that makes the center of already
    selected exemplars as close as possible to the center of all elements (in the feature space).

    .note::
        This selection strategy *only* works when there is *only* one subnetwork
    """

    def make_sorted_indices_from_features(self, features: torch.Tensor) -> List[int]:
        """
        Return a sorted list of indices for the features in `features` by using the herding selection strategy as
        described in iCaRL. It is a greedy algorithm that at each iteration selects the remaining exemplar that makes
        the center of already selected exemplars as close as possible to the center of all elements (in the feature
        space). The samples whose respective features in `features` are the last in the sorted list of indices *must* be
        the first to be removed when the size of the buffer is reduced.
        :param features: a 2D tensor of feature vectors
        :return: a sorted list of indices for the features in `features`. The samples whose respective features in
            `features` are the last in the sorted list of indices *must* be the first to be removed when the size of the
            buffer is reduced.
        """
        selected_indices: List[int] = []
        center = features.mean(dim=0)
        current_center = center * 0

        for i in range(len(features)):
            # Compute distances with real center
            candidate_centers = current_center * i / (i + 1) + features / (i + 1)
            distances = pow(candidate_centers - center, 2).sum(dim=1)
            distances[selected_indices] = inf

            # Select best candidate
            new_index = distances.argmin().tolist()
            selected_indices.append(new_index)
            current_center = candidate_centers[new_index]

        return selected_indices
