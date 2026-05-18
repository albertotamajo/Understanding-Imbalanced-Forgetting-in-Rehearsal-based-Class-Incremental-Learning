"""
This module includes a set of selection accuracy plugin metrics and respective
helper methods
"""
from __future__ import annotations

import copy
from typing import List, TYPE_CHECKING

from avalanche.evaluation.metrics.accuracy import AccuracyPluginMetric
import torch

if TYPE_CHECKING:
    from training.templates import DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate


class SelectionAccuracyPluginMetric(AccuracyPluginMetric):
    """
    Base class for all selection accuracy plugin metrics.
    This plugin metric computes the running selection accuracy of a
    :class:`DynamicNetworkWithInitFeatureExtractorAndIncrementalOutlierSubNetworks` network.
    The selection accuracy is defined as the accuracy of the network in the selection of the correct subnetwork for a
    given sample based on the logits output of the outlier detector of each subnetwork.

    This plugin *only* works during the *evaluation phase*.

    This plugin metric *must* be used *only* with strategies that are subclasses
    of :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
    """

    def __init__(self, reset_at, emit_at):
        super().__init__(reset_at=reset_at, emit_at=emit_at, mode="eval")

    def update(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate):
        """
        Update the underlying accuracy metric. For each sample in the current mini-batch, the logit of the outlier
        detector of each subnetwork is used to compute the selection accuracy.

        .note::
            The outlier detector of each subnetwork is trained to output low-valued logits for samples of classes that
            have undertaken training on it. Therefore, the logits are negated before updating the underlying accuracy
            metric.

        .note::
            This method expects each sample in the mini-batch to have the targets_task_labels DataAttribute to be set
            to the ID of the subnetwork that has undertaken training on the respective class
        :param strategy: strategy
        :return: None
        """
        # `strategy.mb_output` stores a dictionary containing all the IDs of the subnetworks in the network as keys and
        # dictionaries as values. Each dictionary has the following form:
        # {"incremental_classifier": (output1, output2), "outlier_detector": output}, where output1 is the output of
        # the incremental classifier (the logits) while output2 is the feature vector, also known as embedding, that
        # precedes the final linear classifier layer in the incremental classifier.
        sub_ids = list(strategy.mb_output.keys())  # get a list of the subnetwork IDs in the current mini-batch output

        # If there is only one subnetwork, then the selection accuracy is always 100%
        if len(sub_ids) == 1:
            # two dummy tensors are provided to get a 100% accuracy in the current mini-batch
            self._metric.update(torch.as_tensor([0, 1], device=strategy.device),
                                torch.as_tensor([0, 1], device=strategy.device))
        else:
            curr_sub_id = strategy.curr_sub_id  # store the current state of the strategy's subnetwork id
            # concatenate the outputs of each subnetwork's outlier detector along the first dimension for the same sample
            mb_output = torch.cat(tensors=[strategy.mb_output_outlier_detector for strategy.curr_sub_id in sub_ids], dim=1)
            mb_output = - mb_output  # negate all the logits values because the outlier detector of each subnetwork is
            # trained to output low-valued logits for samples of classes that have undertaken training on the respective
            # subnetwork
            strategy.curr_sub_id = curr_sub_id  # re-store the state of the strategy's subnetwork id
            mb_task_id = copy.deepcopy(strategy.mb_task_id).cpu()
            sub_ids = [int(sub_id) for sub_id in sub_ids]  # convert the subnetwork IDs into int from strings storing int
            mb_task_id.apply_(lambda x: sub_ids.index(x))  # convert the task id into its index in `sub_ids`. This way the
            # indices in the tensor of true task ids match the order in `mb_output`
            mb_task_id = mb_task_id.to(strategy.device)  # move to the device of the strategy
            self._metric.update(mb_output, mb_task_id)


class ExperienceSelectionAccuracy(SelectionAccuracyPluginMetric):
    """
    At the end of each experience, this metric reports the selection accuracy over all patterns seen in
    that experience.

    This plugin metric *only* works at evaluation time.

    This plugin metric *must* be used *only* with strategies that are subclasses
    of :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
    """
    def __init__(self):
        """
        Create a new ExperienceSelectionAccuracy metric
        """
        super().__init__(reset_at="experience", emit_at="experience")

    def __str__(self):
        return "Selection_Accuracy_Exp"


def selection_accuracy_metrics(experience=False) -> List[SelectionAccuracyPluginMetric]:
    """
    Helper method that can be used to obtain the desired set of selection accuracy plugin metrics.
    :param experience: If True, will return a metric able to log the selection accuracy on each evaluation experience.
        Default is False
    :return: A list of selection accuracy plugin metrics.
    """
    metrics: List[SelectionAccuracyPluginMetric] = []
    if experience:
        metrics.append(ExperienceSelectionAccuracy())
    return metrics
