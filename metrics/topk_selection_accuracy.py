"""
This module includes a set of top-k selection accuracy plugin metrics and respective
helper methods
"""
from __future__ import annotations

import copy
from typing import List, TYPE_CHECKING

from avalanche.evaluation.metrics.topk_acc import TopkAccuracyPluginMetric
import torch

if TYPE_CHECKING:
    from training.templates import DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate


class TopkSelectionAccuracyPluginMetric(TopkAccuracyPluginMetric):
    """
    Base class for all top-k selection accuracy plugin metrics.
    This plugin metric computes the running top-k selection accuracy of a
    :class:`DynamicNetworkWithInitFeatureExtractorAndIncrementalOutlierSubNetworks` network.

    The top-k selection accuracy is defined as the top-k accuracy of the network in the selection of the correct
    subnetwork for a given sample based on the logits output of the outlier detector of each subnetwork.

    It computes a dictionary of the following form: `{0: top-k selection accuracy}`. The key 0 has no meaning.

    This plugin *only* works during the *evaluation phase*.

    This plugin metric *must* be used *only* with strategies that are subclasses
    of :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
    """
    def __init__(self, reset_at, emit_at, top_k: int):
        super().__init__(reset_at=reset_at, emit_at=emit_at, mode="eval", top_k=top_k)

    def update(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate):
        """
        Update the underlying top-k selection accuracy metric. For each sample in the current mini-batch, the logit of
        the outlier detector of each subnetwork is used to compute the top-k selection accuracy.

        .note::
            The outlier detector of each subnetwork is trained to output low-valued logits for samples of classes that
            have undertaken training on it. Therefore, the logits are negated before updating the underlying accuracy
            metric.

        .note::
            This method expects each sample in the mini-batch to have the targets_task_labels DataAttribute to be set
            to the ID of the subnetwork that has undertaken training on the respective class
        :param strategy: strategy
        :return:
        """
        assert strategy.experience is not None
        # `strategy.mb_output` stores a dictionary containing all the IDs of the subnetworks in the network as keys and
        # dictionaries as values. The dictionary of each subnetwork has the following form:
        # {"incremental_classifier": (output1, output2), "outlier_detector": output}, where output1 is the output of the
        # incremental classifier (the logits) while output2 is the feature vector, also known as embedding, that
        # precedes the final linear classifier layer in the incremental classifier.
        sub_ids = list(strategy.mb_output.keys())  # get a list of the subnetwork IDs in the current mini-batch output

        # if the number of subnetworks is less than top_k, the underlying metric raises an error. In order to avoid this
        # problem, the value of top_k is temporarily set to the number of subnets. This way the topk selection  accuracy
        # is 100%, which is what one would expect when computing the topk selection accuracy on a set of class whose
        # number is less than k.
        reset_top_k = None  # stores the original value of `top_k` if the number of subnets is less than `top_k`
        if len(sub_ids) < self._metric.top_k:
            reset_top_k = self._metric.top_k
            self._metric.top_k = len(sub_ids)

        # If there is only one subnetwork, then the topk selection accuracy is always 100% regardless of the value of k
        if len(sub_ids) == 1:
            # two dummy tensors are provided to get a 100% accuracy in the current mini-batch
            self._metric.update(torch.as_tensor([0, 1], device=strategy.device),
                                torch.as_tensor([0, 1], device=strategy.device), 0)
        else:
            curr_sub_id = strategy.curr_sub_id  # store the current state of the strategy's subnetwork id
            # concatenate the outputs of each subnetwork's outlier detector along the first dimension for the same
            # sample
            mb_output = torch.cat(tensors=[strategy.mb_output_outlier_detector for strategy.curr_sub_id in sub_ids],
                                  dim=1)
            mb_output = - mb_output  # negate all the logits values because the outlier detector of each subnetwork is
            # trained to output low-valued logits for samples of classes that have undertaken training on the respective
            # subnetwork
            strategy.curr_sub_id = curr_sub_id  # re-store the state of the strategy's subnetwork id
            mb_task_id = copy.deepcopy(strategy.mb_task_id).cpu()
            # convert the subnetwork IDs into int from strings storing int
            sub_ids = [int(sub_id) for sub_id in sub_ids]
            # convert the task id into its index in `sub_ids`. This way the indices in the tensor of true task ids match
            # the order in `mb_output`
            mb_task_id.apply_(lambda x: sub_ids.index(x))
            mb_task_id = mb_task_id.to(strategy.device)  # move to the device of the strategy
            self._metric.update(mb_output, mb_task_id, 0)

        # reset the original value of top-k
        if reset_top_k is not None:
            self._metric.top_k = reset_top_k


class ExperienceTopkSelectionAccuracy(TopkSelectionAccuracyPluginMetric):
    """
    At the end of each experience, this metric reports the top-k selection accuracy over all patterns seen in
    that experience. This plugin metric only works at eval time.
    """
    def __init__(self, top_k: int):
        """
        Create a new ExperienceTopkSelectionAccuracy metric
        :param top_k: integer number to define the value of k.
        """
        super().__init__(reset_at="experience", emit_at="experience", top_k=top_k)
        self.top_k = top_k

    def __str__(self):
        return f"Top{self.top_k}_Selection_Accuracy_Exp"


def topk_selection_accuracy_metrics(top_k=3, experience=False) -> List[TopkSelectionAccuracyPluginMetric]:
    """
    Helper method that can be used to obtain the desired set of top-k selection accuracy plugin metrics.
    :param top_k: integer number to define the value of k. Default is 3
    :param experience: If True, will return a metric able to log the top-k selection accuracy on each evaluation
        experience. Default is False
    :return: A list of top-k selection accuracy plugin metrics.
    """
    metrics: List[TopkSelectionAccuracyPluginMetric] = []
    if experience:
        metrics.append(ExperienceTopkSelectionAccuracy(top_k=top_k))
    return metrics
