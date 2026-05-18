"""
This module includes the abstract class to implement a probability vectors storage
for :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal, Iterable, Set, Union, TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from training.templates import DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate


class ProbVecsStorageDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate(ABC):
    """
    ABC for probability vectors storage used in :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
    strategies.

    This abstract class defines the `get_incremental_classifier_prob_vecs` and
    `contains_incremental_classifier_prob_vecs` methods, which handle retrieving the probability vectors of specific
    samples and checking if probability vectors for certain samples are stored, respectively, using their
    `dataset_indices` DataAttribute. The probability vector of a sample is a vector whose size matches the number of
    classes assigned to the incremental classifier of the subnetwork associated with the sample's class. This vector
    represents probabilities that sum to 1. A one-hot encoding vector is a special, degenerate case of a probability
    vector where only one class has a probability of 1, and all others are 0.
    `prob_vecs_storage`, a cached property in :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
    strategies, automatically looks for a plugin that implements this abstract class within the `plugins` attribute of
    the strategy and selects the first such plugin in the list. If no such a plugin exists in the `plugins` attribute of
    the strategy, a :class:`RuntimeError` is raised if `prob_vecs_storage` is invoked. `prob_vecs_storage` is a cached
    property because it locates such a plugin the first time is invoked and then caches it. As long as the `plugins`
    attribute of the strategy remains unchanged, the cached result will be returned. If `plugins` changes size or some
    plugin instances are replaced with others, the lookup will be run again and the new result be cached, replacing the
    previous value.

    Subclasses must implement the `_get_incremental_classifier_prob_vecs` and
    `_contains_incremental_classifier_prob_vecs` abstract methods.

    .note::
        A :class:`ValueError` is raised if the `get_incremental_classifier_prob_vecs` and/or
        `contains_incremental_classifier_prob_vecs` methods are invoked on the samples of an experience whose type
        (train, val or test) is a type for which an instance of this class does not store probability vectors.

    .note::
        The samples which the `get_incremental_classifier_prob_vecs` and/or `contains_incremental_classifier_prob_vecs`
        methods are invoked on must have the "dataset_indices" DataAttribute.
    """

    def __init__(self, dataset_type: Union[Literal["train", "val", "test"], Iterable[Literal["train", "val", "test"]]]):
        """
        Create a new ProbVecsStorageDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate

        :param dataset_type: train`, `val`, `test` or an iterable containing any combination of them. Decides whether
            to store the probability vectors for the samples in the train, validation, test datasets, or any
            combination of these.
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
        a set containing `train`, `val`, `test` or any combination of them. The probability vectors are only stored for
        the samples in the given types of dataset. 
        """

        self._subnetwork_ids: Set[int] = set()
        """
        a set containing subnetwork IDs. The probability vectors are only stored for the samples that belong to classes
        allocated to the given subnetworks.
        """

    def get_incremental_classifier_prob_vecs(self, dataset_indices: torch.Tensor,
                                             dataset_type: Literal["train", "val", "test"], subnetwork_id: int,
                                             num_classes: int) -> torch.Tensor:
        """
        Get the probability vectors of the given samples.

        :param dataset_indices: a 1D tensor of the `dataset_indices` DataAttribute of the samples whose probability
            vectors must be retrieved. A :class:`ValueError` is raised if the probability vectors of one or multiple
            samples are not in the storage.
        :param dataset_type: the type of dataset the given samples belong to.  It could be `train`, `val` or `test`.
            A :class:`ValueError` is raised if it is not a dataset type for which this instance stores probability
            vectors.
        :param subnetwork_id: the id of the subnetwork the classes of the samples are allocated to. Note that all the
            classes of the samples *must* be allocated to the same subnetwork. A :class:`ValueError` is raised if the
            probability vectors for the samples belonging to classes allocated to this subnetwork are not stored.
        :param num_classes: the number of classes assigned to the incremental classifier of the subnetwork with id
            `subnetwork_id`, i.e. the number of output nodes of the incremental classifier of the subnetwork with id
            `subnetwork_id`. If the size of the probability vectors stored does not match this number,
            a :class:`RuntimeError` is raised.
        :return: a 2D tensor of probability vectors for the given samples. The size of the probability vectors is equal
            to `num_classes`. A 2D tensor with no elements of size (0, `num_classes`) is returned if `dataset_indices`
            is a tensor with no elements. The returned tensor is allocated to the same device of `dataset_indices`.
        """
        self._check_input(dataset_indices, dataset_type, subnetwork_id)
        if dataset_indices.numel() == 0:
            return torch.empty(0, num_classes, device=dataset_indices.device, dtype=torch.float32)
        if not bool(self._contains_incremental_classifier_prob_vecs(dataset_indices, dataset_type, subnetwork_id).all()):
            raise ValueError("The probability vectors of one or more samples are not in the storage")

        prob_vecs = self._get_incremental_classifier_prob_vecs(dataset_indices, dataset_type, subnetwork_id,
                                                               num_classes).to(dataset_indices.device)
        if not prob_vecs.shape[1] == num_classes:
            raise RuntimeError("The size of the probability vectors stored does not match the number of output nodes "
                               "of the incremental classifier of the subnetwork with id `subnetwork_id`")
        return prob_vecs

    def contains_incremental_classifier_prob_vecs(self, dataset_indices: torch.Tensor,
                                                  dataset_type: Literal["train", "val", "test"],
                                                  subnetwork_id: int) -> torch.Tensor:
        """
        Check whether the probability vectors of the given samples are in the storage.

        :param dataset_indices: a 1D tensor of the `dataset_indices` DataAttribute of the samples whose probability
            vectors must be checked for presence.
        :param dataset_type: the type of dataset the given samples belong to.  It could be `train`, `val` or `test`.
            A :class:`ValueError` is raised if it is not a dataset type for which this instance stores probability
            vectors.
        :param subnetwork_id: the id of the subnetwork the classes of the samples are allocated to. Note that all the
            classes of the samples *must* be allocated to the same subnetwork. A :class:`ValueError` is raised if the
            probability vectors for the samples belonging to classes allocated to this subnetwork are not stored.
        :return: a 1D tensor of boolean values of same size as `dataset_indices` indicating whether the probability
            vectors of the respective samples are in the storage. A 1D tensor with no elements is returned if
            `dataset_indices` is a tensor with no elements.
            The returned tensor is allocated to the same device of `dataset_indices`
        """
        self._check_input(dataset_indices, dataset_type, subnetwork_id)
        if dataset_indices.numel() == 0:
            return torch.tensor([], dtype=torch.bool, device=dataset_indices.device)
        return self._contains_incremental_classifier_prob_vecs(dataset_indices, dataset_type, subnetwork_id).to(
            dataset_indices.device)

    def _check_input(self, dataset_indices, dataset_type, subnetwork_id):
        """
        Check whether `dataset_indices` is a 1D tensor, `dataset_type` is contained in `self._dataset_type` and
        `subnetwork_id` is contained in `self._subnetwork_ids`.
        """
        if (not isinstance(dataset_indices, torch.Tensor)) or (not dataset_indices.dim() == 1):
            raise ValueError("`dataset_indices` must be a 1D tensor")
        if dataset_type not in self._dataset_type:
            raise ValueError("`dataset_type` is not a dataset for which probability vectors are stored")
        if subnetwork_id not in self._subnetwork_ids:
            raise ValueError("`subnetwork_id` is not a subnetwork for which probability vectors of samples that belong "
                             "to classes allocated to it are stored")

    @abstractmethod
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

    @abstractmethod
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
