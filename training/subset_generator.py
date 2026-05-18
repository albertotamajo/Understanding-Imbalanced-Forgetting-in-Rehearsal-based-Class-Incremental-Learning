"""
This module contains some subset generators
for :class:`training.plugins.GradientBasedReplayBufferSelectionSGDPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
and some utilities
"""
from abc import ABC, abstractmethod
from typing import List, Tuple
import random

import numpy as np


def distribute_quantity(qnt: int, n_groups: int) -> List[int]:
    """
    Distribute a given quantity over a certain number of groups
    :param qnt: a quantity
    :param n_groups: number of groups
    :return: a list containing the quantity distributed over each group
    """
    qnt_groups = [qnt // n_groups for _ in range(n_groups)]
    # distribute remaining quantity among groups
    rem = qnt - sum(qnt_groups)
    for i in range(rem):
        qnt_groups[i] += 1
    return qnt_groups


class SubsetGeneratorGradientBasedReplayBufferSelection(ABC):
    """
    ABC for subset generators
    of :class:`training.plugins.GradientBasedReplayBufferSelectionSGDPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`.

    Classes must implement the abstract method `_generate_more_than_one_subsets`.
    """
    def __init__(self):
        super().__init__()

    def generate_subsets(self, n_subsets: int, tot_coreset_size: int,
                         indices: np.ndarray) -> Tuple[List[np.ndarray], List[int]]:
        """
        Generate a given number of subsets. If `n_subsets` is 1, two one-element lists are returned where the first list
        contains an array of indices from 0 up to len(indices) -1 and the second list contains `tot_coreset_size`. If
        `n_subsets` is greater than 1, `self._generate_more_than_one_subsets` is used to generate the subsets.
        :param n_subsets: the number of subsets to generate
        :param tot_coreset_size: the total coreset size
        :param indices: the indices of the samples that must be divided into `n_subsets` disjoint subsets. The index of
            a given sample is the index to be used to retrieve the given sample from the underlying dataset.
        :return: a list containing the subsets and a list of respective coreset sizes. Each subset is encoded as an
            array of indices where each index refers to the position of a given sample in `indices`. The sum of the
            coreset sizes for each subset is equal to `tot_coreset_size`.
        """
        if n_subsets == 1:
            subset = list(range(len(indices)))
            return [np.asarray(subset)], [tot_coreset_size]
        else:
            return self._generate_more_than_one_subsets(n_subsets=n_subsets, tot_coreset_size=tot_coreset_size,
                                                        indices=indices)

    @abstractmethod
    def _generate_more_than_one_subsets(self, n_subsets: int, tot_coreset_size: int,
                                        indices: np.ndarray) -> Tuple[List[np.ndarray], List[int]]:
        """
        Generate a given number of subsets.
        :param n_subsets: the number of subsets to generate. Implementations can assume that this value is always
            greater than 1.
        :param tot_coreset_size: the total coreset size
        :param indices: the indices of the samples that must be divided into `n_subsets` disjoint subsets. The index of
            a given sample is the index to be used to retrieve the given sample from the underlying dataset.
        :return: a list containing the subsets and a list of respective coreset sizes. Each subset is encoded as an
            array of indices where each index refers to the position of a given sample in `indices`. The sum of the
            coreset sizes for each subset is equal to `tot_coreset_size`.
        """


class RandomSubsetGenerator(SubsetGeneratorGradientBasedReplayBufferSelection):
    """
    Random subset generator
    for class:`training.plugins.GradientBasedReplayBufferSelectionSGDPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`.
    """
    def __init__(self):
        super().__init__()

    def _generate_more_than_one_subsets(self, n_subsets: int, tot_coreset_size: int,
                                        indices: np.ndarray) -> Tuple[List[np.ndarray], List[int]]:
        """
        Generate a given number of random subsets.
        :param n_subsets: the number of subsets to generate. It is assumed that `n_subsets` is greater than 1.
        :param tot_coreset_size: the total coreset size
        :param indices: the indices of the samples that must be divided into `n_subsets` disjoint subsets. The index of
            a given sample is the index to be used to retrieve the given sample from the underlying dataset.
        :return: a list containing the subsets and a list of respective coreset sizes. Each subset is encoded as an
            array of indices where each index refers to the position of a given sample in `indices`. The sum of the
            coreset sizes for each subset is equal to `tot_coreset_size`.
        """
        # get the size of the coreset for each subset
        size_coreset_subsets = distribute_quantity(tot_coreset_size, n_subsets)
        # get the size of each subset
        size_subsets = distribute_quantity(len(indices), n_subsets)
        samples_indexes = list(range(len(indices)))
        # shuffle the list of indexes
        random.shuffle(samples_indexes)

        random_subsets = []
        for size in size_subsets:
            random_subsets.append(np.asarray(samples_indexes[:size]))
            samples_indexes = samples_indexes[size:]

        return random_subsets, size_coreset_subsets
