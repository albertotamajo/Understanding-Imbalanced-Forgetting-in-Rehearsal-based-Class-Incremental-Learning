"""
This module includes a set of incremental classifier accuracy per subnetwork plugin metrics and respective
helper methods
"""
from __future__ import annotations

from typing import List, TYPE_CHECKING

from avalanche.evaluation.metrics.accuracy import AccuracyPerTaskPluginMetric
import torch

if TYPE_CHECKING:
    from training.templates import DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate


class AccuracyPerSubnetworkIncrementalClassifierPluginMetric(AccuracyPerTaskPluginMetric):
    """
    Base class for all incremental classifier accuracy per subnetwork plugin metrics.
    This plugin metric computes the running accuracy of the incremental classifier of a given subnetwork.
    It computes a dictionary of <subnetwork ID, incremental classifier accuracy value> pairs.

    This plugin metric *must* be used *only* with strategies that are subclasses
    of :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
    """

    def _update(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate):
        classes = strategy.model.subnetworks_classes[strategy.curr_sub_id]
        if len(classes) == 1:  # if in binary classification mode
            # The underlying metric does not support computing the accuracy from the logit output of a network having
            # a single output node. However, it supports computing the accuracy when providing the labels directly.
            # The labels (predictions) are computed by passing the logit output through the sigmoid operator and then
            # applying a threshold at 0.5. Note that when in binary classification mode, the incremental classifier is
            # trained to output high values for outlier samples and low values for the single class the incremental
            # classifier must be trained on. The target of the single class is 0 while the outlier samples have target
            # 1.
            predictions = (torch.sigmoid(strategy.mb_output_incremental_classifier) > 0.5).int().flatten()
            mb_y_incremental_classifier = strategy.mb_y_incremental_classifier
            mb_y_incremental_classifier[mb_y_incremental_classifier == -1] = 1
            self._metric.update(predictions, mb_y_incremental_classifier, int(strategy.curr_sub_id))
        else:
            # if in the current mini-batch there are samples that belong to classes that the current subnetwork has
            # undertaken training on, then update the underlying metric
            if not strategy.mb_output_masked_incremental_classifier.numel() == 0:
                self._metric.update(strategy.mb_output_masked_incremental_classifier,
                                    strategy.mb_y_masked_incremental_classifier, int(strategy.curr_sub_id))

    def update(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate):
        if self._mode == "train":  # if in training mode
            self._update(strategy)
        else:  # if in evaluation mode
            # when in evaluation mode,`strategy.mb_output` stores a dictionary containing all the IDs of the subnetworks
            # in the network as keys and respective outputs as values. The output of each subnetwork is a dictionary of
            # the following form: {"incremental_classifier": (output1, output2), "outlier_detector": output},
            # where output1 is the output of the incremental classifier (the logits) while output2 is the
            # feature vector, also known as embedding, that precedes the final linear classifier layer in the
            # incremental classifier.
            curr_sub_id = strategy.curr_sub_id  # store the current state of the strategy's subnetwork id
            for strategy.curr_sub_id in strategy.mb_output.keys():
                self._update(strategy)
            strategy.curr_sub_id = curr_sub_id  # re-store the state of the strategy's subnetwork id


class EpochAccuracyPerSubnetworkIncrementalClassifier(AccuracyPerSubnetworkIncrementalClassifierPluginMetric):
    """
    The average incremental classifier accuracy per subnetwork over a single training epoch.

    The average incremental classifier accuracy per subnetwork will be logged after each training epoch by computing
    the number of correctly predicted patterns during the epoch divided by the overall number of patterns encountered
    in that epoch.

    This plugin metric *only* works at training time.

    This plugin metric *must* be used *only* with strategies that are subclasses
    of :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
    """
    def __init__(self):
        """
        Create an instance of the EpochAccuracyPerSubnetworkIncrementalClassifier metric
        """
        super().__init__(reset_at="epoch", emit_at="epoch", mode="train")

    def __str__(self):
        return "Top1_Acc_Incremental_Classifier_Epoch"


class ExperienceAccuracyPerSubnetworkIncrementalClassifier(AccuracyPerSubnetworkIncrementalClassifierPluginMetric):
    """
    At the end of each experience, this plugin metric reports the average incremental classifier accuracy per
    subnetwork over all patterns seen in that experience.

    This plugin metric *only* works at training time.

    This plugin metric *must* be used *only* with strategies that are subclasses
    of :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
    """
    def __init__(self):
        """
        Create an instance of ExperienceAccuracyPerSubnetworkIncrementalClassifier metric
        """
        super().__init__(reset_at="experience", emit_at="experience", mode="eval")

    def __str__(self):
        return "Top1_Acc_Incremental_Classifier_Exp"


def incremental_classifier_accuracy_per_subnetwork_metrics(epoch=False, experience=False)\
        -> List[AccuracyPerSubnetworkIncrementalClassifierPluginMetric]:
    """
    Helper method that can be used to obtain the desired set of incremental classifier accuracy per subnetwork plugin
    metrics.
    :param epoch: If True, will return a metric able to log the epoch incremental classifier accuracy per subnetwork at
        training time. Default is False
    :param experience: If True, will return a metric able to log the incremental classifier accuracy per subnetwork on
        each evaluation experience. Default is False
    :return: A list of incremental classifier accuracy per subnetwork plugin metrics.
    """
    metrics: List[AccuracyPerSubnetworkIncrementalClassifierPluginMetric] = []
    if epoch:
        metrics.append(EpochAccuracyPerSubnetworkIncrementalClassifier())
    if experience:
        metrics.append(ExperienceAccuracyPerSubnetworkIncrementalClassifier())
    return metrics

