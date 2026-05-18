"""
This module includes a set of incremental classifier loss per subnetwork plugin metrics and respective
helper methods
"""
from __future__ import annotations

from typing import List, TYPE_CHECKING

from avalanche.evaluation.metrics.loss import LossPerTaskPluginMetric
from torch import Tensor

if TYPE_CHECKING:
    from training.templates import DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate


class LossPerSubnetworkIncrementalClassifierPluginMetric(LossPerTaskPluginMetric):
    """
    Base class for all incremental classifier loss per subnetwork plugin metrics.

    This plugin metric computes the running loss of the incremental classifier of a given subnetwork.
    It computes a dictionary of <subnetwork ID, incremental classifier loss value> pairs.

    This plugin metric *must* be used *only* with strategies that are subclasses
    of :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
    """

    def _update(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate):
        classes = strategy.model.subnetworks_classes[strategy.curr_sub_id]
        incremental_classifier_loss = strategy.incremental_classifier_loss
        # when in evaluation mode, `strategy.incremental_classifier_loss` is a dictionary containing the subnetwork
        # IDs as keys and their respective loss tensors as values. Otherwise, it is a tensor
        incremental_classifier_loss = incremental_classifier_loss if isinstance(incremental_classifier_loss, Tensor) \
            else incremental_classifier_loss[strategy.curr_sub_id]
        if len(classes) == 1:  # if in binary classification mode
            self._loss.update(loss=incremental_classifier_loss,
                              patterns=len(strategy.mb_y_incremental_classifier),
                              task_label=int(strategy.curr_sub_id))
        else:
            # if in the current mini-batch there are samples that belong to classes that the current subnetwork has
            # undertaken training on, then update the underlying metric
            if not strategy.mb_output_masked_incremental_classifier.numel() == 0:
                self._loss.update(loss=incremental_classifier_loss,
                                  patterns=len(strategy.mb_y_masked_incremental_classifier),
                                  task_label=int(strategy.curr_sub_id))

    def update(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate):
        if self._mode == "train":  # if in training mode
            self._update(strategy)
        else:  # if in evaluation mode
            # when in evaluation mode, `strategy.incremental_classifier_loss` is a dictionary containing the subnetwork
            # IDs as keys and their respective loss tensors as values.
            curr_sub_id = strategy.curr_sub_id  # store the current state of the strategy's subnetwork id
            for strategy.curr_sub_id in strategy.mb_output.keys():
                self._update(strategy)
            strategy.curr_sub_id = curr_sub_id  # re-store the state of the strategy's subnetwork id


class MiniBatchLossPerSubnetworkIncrementalClassifier(LossPerSubnetworkIncrementalClassifierPluginMetric):
    """
    The minibatch incremental classifier loss per subnetwork metric.
    This metric computes the average incremental classifier loss per subnetwork over patterns from a single minibatch.
    It reports the result after each iteration.
    If a more coarse-grained logging is needed, consider using :class:`EpochLossPerSubnetworkIncrementalClassifier`
    instead.

    This plugin metric *only* works at training time.

    This plugin metric *must* be used *only* with strategies that are subclasses
    of :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
    """
    def __init__(self):
        """
        Create an instance of the MiniBatchLossPerSubnetworkIncrementalClassifier metric.
        """
        super().__init__(reset_at="iteration", emit_at="iteration", mode="train")

    def __str__(self):
        return "Loss_Incremental_Classifier_MB"


class EpochLossPerSubnetworkIncrementalClassifier(LossPerSubnetworkIncrementalClassifierPluginMetric):
    """
    The average incremental classifier loss per subnetwork over a single training epoch.

    The incremental classifier loss per subnetwork will be logged after each training epoch by computing the loss on
    the predicted patterns during the epoch divided by the overall number of patterns encountered in that epoch.

    This plugin metric *only* works at training time.

    This plugin metric *must* be used *only* with strategies that are subclasses
    of :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
    """
    def __init__(self):
        """
        Create an instance of the EpochLossPerSubnetworkIncrementalClassifier metric.
        """
        super().__init__(reset_at="epoch", emit_at="epoch", mode="train")

    def __str__(self):
        return "Loss_Incremental_Classifier_Epoch"


class ExperienceLossPerSubnetworkIncrementalClassifier(LossPerSubnetworkIncrementalClassifierPluginMetric):
    """
    At the end of each experience, this metric reports the average incremental classifier loss per subnetwork over all
    patterns seen in that experience.

    This plugin metric *only* works at evaluation time.

    This plugin metric *must* be used *only* with strategies that are subclasses
    of :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
    """

    def __init__(self):
        """
        Create an instance of ExperienceLossPerSubnetworkIncrementalClassifier metric
        """
        super().__init__(reset_at="experience", emit_at="experience", mode="eval")

    def __str__(self):
        return "Loss_Incremental_Classifier_Exp"


def incremental_classifier_loss_per_subnetwork_metrics(minibatch=False, epoch=False, experience=False)\
        -> List[LossPerSubnetworkIncrementalClassifierPluginMetric]:
    """
    Helper method that can be used to obtain the desired set of incremental classifier loss per subnetwork plugin
    metrics.
    :param minibatch: If True, will return a metric able to log the minibatch incremental classifier loss per subnetwork
        at training time. Default is False
    :param epoch: If True, will return a metric able to log the epoch incremental classifier loss per subnetwork at
        training time. Default is False
    :param experience: If True, will return a metric able to log the incremental classifier loss per subnetwork on each
        evaluation experience. Default is False
    :return: A list of incremental classifier loss per subnetwork plugin metrics.
    """
    metrics: List[LossPerSubnetworkIncrementalClassifierPluginMetric] = []
    if minibatch:
        metrics.append(MiniBatchLossPerSubnetworkIncrementalClassifier())
    if epoch:
        metrics.append(EpochLossPerSubnetworkIncrementalClassifier())
    if experience:
        metrics.append(ExperienceLossPerSubnetworkIncrementalClassifier())
    return metrics
