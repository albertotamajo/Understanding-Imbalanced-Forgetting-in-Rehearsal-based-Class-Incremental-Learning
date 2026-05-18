"""
This module includes a set of outlier detector loss per subnetwork plugin metrics and respective
helper methods
"""
from __future__ import annotations

from typing import List, TYPE_CHECKING

from avalanche.evaluation.metrics.loss import LossPerTaskPluginMetric
from torch import Tensor

if TYPE_CHECKING:
    from training.templates import DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate


class LossPerSubnetworkOutlierDetectorPluginMetric(LossPerTaskPluginMetric):
    """
    Base class for all outlier detector loss per subnetwork plugin metrics.
    This plugin metric computes the running loss of the outlier detector of a given subnetwork.
    It computes a dictionary of <subnetwork ID, outlier detector loss value> pairs.

    This plugin metric *must* be used *only* with strategies that are subclasses
    of :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
    """

    def _update(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate):
        outlier_detector_loss = strategy.outlier_detector_loss
        # when in evaluation mode, `strategy.outlier_detector_loss` is a dictionary containing the subnetwork
        # IDs as keys and their respective loss tensors as values. Otherwise, it is a tensor
        outlier_detector_loss = outlier_detector_loss if isinstance(outlier_detector_loss, Tensor) \
            else outlier_detector_loss[strategy.curr_sub_id]
        self._loss.update(loss=outlier_detector_loss, patterns=len(strategy.mb_y_outlier_detector),
                          task_label=int(strategy.curr_sub_id))

    def update(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate):
        if self._mode == "train":  # if in training mode
            self._update(strategy)
        else:  # if in evaluation mode
            # when in evaluation mode, `strategy.outlier_detector_loss` is a dictionary containing the subnetwork
            # IDs as keys and their respective loss tensors as values.
            curr_sub_id = strategy.curr_sub_id  # store the current state of the strategy's subnetwork id
            for strategy.curr_sub_id in strategy.mb_output.keys():
                self._update(strategy)
            strategy.curr_sub_id = curr_sub_id  # re-store the state of the strategy's subnetwork id


class MiniBatchLossPerSubnetworkOutlierDetector(LossPerSubnetworkOutlierDetectorPluginMetric):
    """
    The minibatch outlier detector loss per subnetwork metric.
    This metric computes the average outlier detector loss per subnetwork over patterns from a single minibatch.
    It reports the result after each iteration.
    If a more coarse-grained logging is needed, consider using :class:`EpochLossPerSubnetworkOutlierDetector`
    instead.

    This plugin metric *only* works at training time.

    This plugin metric *must* be used *only* with strategies that are subclasses
    of :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
    """
    def __init__(self):
        """
        Create an instance of the MiniBatchLossPerSubnetworkOutlierDetector metric.
        """
        super().__init__(reset_at="iteration", emit_at="iteration", mode="train")

    def __str__(self):
        return "Loss_Outlier_Detector_MB"


class EpochLossPerSubnetworkOutlierDetector(LossPerSubnetworkOutlierDetectorPluginMetric):
    """
    The average outlier detector loss per subnetwork over a single training epoch.

    The outlier detector loss per subnetwork will be logged after each training epoch by computing the loss on
    the predicted patterns during the epoch divided by the overall number of patterns encountered in that epoch.

    This plugin metric *only* works at training time.

    This plugin metric *must* be used *only* with strategies that are subclasses
    of :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
    """
    def __init__(self):
        """
        Create an instance of the EpochLossPerSubnetworkOutlierDetector metric.
        """
        super().__init__(reset_at="epoch", emit_at="epoch", mode="train")

    def __str__(self):
        return "Loss_Outlier_Detector_Epoch"


class ExperienceLossPerSubnetworkOutlierDetector(LossPerSubnetworkOutlierDetectorPluginMetric):
    """
    At the end of each experience, this metric reports the average outlier detector loss per subnetwork over all
    patterns seen in that experience.

    This plugin metric *only* works at evaluation time.

    This plugin metric *must* be used *only* with strategies that are subclasses
    of :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
    """
    def __init__(self):
        """
        Create an instance of ExperienceLossPerSubnetworkOutlierDetector metric
        """
        super().__init__(reset_at="experience", emit_at="experience", mode="eval")

    def __str__(self):
        return "Loss_Outlier_Detector_Exp"


def outlier_detector_loss_per_subnetwork_metrics(minibatch=False, epoch=False, experience=False)\
        -> List[LossPerSubnetworkOutlierDetectorPluginMetric]:
    """
    Helper method that can be used to obtain the desired set of outlier detector loss per subnetwork plugin
    metrics.
    :param minibatch: If True, will return a metric able to log the minibatch outlier detector loss per subnetwork
        at training time. Default is False
    :param epoch: If True, will return a metric able to log the epoch outlier detector loss per subnetwork at
        training time. Default is False
    :param experience: If True, will return a metric able to log the outlier detector loss per subnetwork on each
        evaluation experience. Default is False
    :return: A list of outlier detector loss per subnetwork plugin metrics.
    """
    metrics: List[LossPerSubnetworkOutlierDetectorPluginMetric] = []
    if minibatch:
        metrics.append(MiniBatchLossPerSubnetworkOutlierDetector())
    if epoch:
        metrics.append(EpochLossPerSubnetworkOutlierDetector())
    if experience:
        metrics.append(ExperienceLossPerSubnetworkOutlierDetector())
    return metrics
