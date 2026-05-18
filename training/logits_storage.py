"""
This module includes the abstract class to implement a logit storage
for :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal, Iterable, Set, Union, TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from training.templates import DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate


class LogitsStorageDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate(ABC):
    """
    ABC for logit storage used in :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` strategies.

    This abstract class defines the `get_incremental_classifier_logits` and `contains_incremental_classifier_logits`
    methods, which handle retrieving the logits of specific samples and checking if logits for certain samples are
    stored, respectively, using their `dataset_indices` DataAttribute. The logits of a sample are its raw output scores
    computed at a certain time step by the incremental classifier of the subnetwork allocated to the class of the
    sample. `logits_storage`, a cached property in :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
    strategies, automatically looks for a plugin that implements this abstract class within the `plugins` attribute of
    the strategy and selects the first such plugin in the list. If no such a plugin exists in the `plugins` attribute of
    the strategy, a :class:`RuntimeError` is raised if `logits_storage` is invoked. `logits_storage` is a cached
    property because it locates such a plugin the first time is invoked and then caches it. As long as the `plugins`
    attribute of the strategy remains unchanged, the cached result will be returned. If `plugins` changes size or some
    plugin instances are replaced with others, the lookup will be run again and the new result be cached, replacing the
    previous value.

    Subclasses must implement the `_get_incremental_classifier_logits` and `_contains_incremental_classifier_logits`
    abstract methods.

    .note::
        Logits of different samples may vary in size because they are computed at different time steps.
        Additionally, their sizes might be smaller than the size of the logit outputs of the current subnetwork's
        incremental classifier. Therefore, to ensure consistency, the logits must be padded with NaN values when
        returned by `_get_incremental_classifier_logits` so that their sizes match the size of the logit outputs of the
        current subnetwork's incremental classifier.

    .note::
        A :class:`ValueError` is raised if the `get_incremental_classifier_logits` and/or
        `contains_incremental_classifier_logits` methods are invoked on the samples of an experience whose type
        (train, val or test) is a type for which an instance of this class does not store logits.

    .note::
        The samples which the `get_incremental_classifier_logits` and/or `contains_incremental_classifier_logits`
        methods are invoked on must have the "dataset_indices" DataAttribute.
    """

    def __init__(self, dataset_type: Union[Literal["train", "val", "test"], Iterable[Literal["train", "val", "test"]]]):
        """
        Create a new LogitsStorageDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate

        :param dataset_type: train`, `val`, `test` or an iterable containing any combination of them. Decides whether
            to store the logits for the samples in the train, validation, test datasets, or any combination of these.
        """
        if isinstance(dataset_type, str):
            dataset_type: Set[str] = {dataset_type}
        else:
            dataset_type: Set = set(dataset_type)

        if len(dataset_type) == 0:
            raise ValueError("`dataset_type` must not be an empty iterable")
        if len(dataset_type - {"train", "val", "test"}) > 0:
            raise ValueError("`dataset_type` must be `train`, `val`, `test` or an iterable only containing them")

        # if the above checks have been passed, then `dataset_type` is a set containing `train`, `val`, `test` or any
        # combination of them
        self._dataset_type: Set[Literal["train", "val", "test"]] = dataset_type
        """
        a set containing `train`, `val`, `test` or any combination of them
        """

    def get_incremental_classifier_logits(self, dataset_indices: torch.Tensor,
                                          dataset_type: Literal["train", "val", "test"]) -> torch.Tensor:
        """
        Get the logits of the given samples.

        :param dataset_indices: a 1D tensor of the `dataset_indices` DataAttribute of the samples whose logits must be
            retrieved. A :class:`ValueError` is raised if the logits of one or multiple samples are not in the storage.
        :param dataset_type: the type of dataset the given samples belong to.  It could be `train`, `val` or `test`.
            A :class:`ValueError` is raised if it is not a dataset type for which this instance stores logits.
        :return: a 2D tensor of logits for the given samples. The size of the logit vectors matches the size of the
            logit outputs of the current subnetwork's incremental classifier even when they have smaller size.
            This is achieved by padding them with NaN values. A 2D tensor with no elements is returned if
            `dataset_indices` is a tensor with no elements. The returned tensor is allocated to the same device of
            `dataset_indices`.
        """
        self._check_input(dataset_indices, dataset_type)
        if dataset_indices.numel() == 0:
            return torch.tensor([[]], device=dataset_indices.device, dtype=torch.float32)
        if not bool(self._contains_incremental_classifier_logits(dataset_indices, dataset_type).all()):
            raise ValueError("The logits of one or more samples are not in the storage")
        return self._get_incremental_classifier_logits(dataset_indices, dataset_type).to(dataset_indices.device)

    def contains_incremental_classifier_logits(self, dataset_indices: torch.Tensor,
                                               dataset_type: Literal["train", "val", "test"]) -> torch.Tensor:
        """
        Check whether the logits of the given samples are in the storage.

        :param dataset_indices: a 1D tensor of the `dataset_indices` DataAttribute of the samples whose logits must be
            checked for presence.
        :param dataset_type: the type of dataset the given samples belong to.  It could be `train`, `val` or `test`.
            A :class:`ValueError` is raised if it is not a dataset type for which this instance stores logits.
        :return: a 1D tensor of boolean values of same size as `dataset_indices` indicating whether the logits of
            the respective samples are in the storage. A 1D tensor with no elements is returned if `dataset_indices` is
            a tensor with no elements. The returned tensor is allocated to the same device of `dataset_indices`
        """
        self._check_input(dataset_indices, dataset_type)
        if dataset_indices.numel() == 0:
            return torch.tensor([], dtype=torch.bool, device=dataset_indices.device)
        return self._contains_incremental_classifier_logits(dataset_indices, dataset_type).to(dataset_indices.device)

    def _check_input(self, dataset_indices, dataset_type):
        """
        Check whether `dataset_indices` is a 1D tensor and `dataset_type` is contained in `self._dataset_type`.
        """
        if (not isinstance(dataset_indices, torch.Tensor)) or (not dataset_indices.dim() == 1):
            raise ValueError("`dataset_indices` must be a 1D tensor")
        if dataset_type not in self._dataset_type:
            raise ValueError("`dataset_type` is not a dataset for which logits are stored")

    @abstractmethod
    def _get_incremental_classifier_logits(self, dataset_indices: torch.Tensor,
                                           dataset_type: Literal["train", "val", "test"]) -> torch.Tensor:
        """
        Get the logits of the given samples.
        .note::
            Safely assume that `dataset_indices` is a 1D tensor containing some elements, the logits of all the given
            samples are in the storage and `dataset_type` is contained in `self._dataset_type`.
            These checks are performed by `get_incremental_classifier_logits`,  which successively invokes this method.
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

    @abstractmethod
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
