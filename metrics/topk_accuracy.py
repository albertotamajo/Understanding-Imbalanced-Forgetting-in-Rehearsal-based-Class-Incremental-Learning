"""
This module includes a set of incremental classifier top-k accuracy per subnetwork plugin metrics and respective
helper methods
"""
from __future__ import annotations

from typing import List, TYPE_CHECKING

from avalanche.evaluation.metrics.topk_acc import TopkAccuracyPluginMetric
import torch

if TYPE_CHECKING:
    from training.templates import DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate


class TopkAccuracyPerSubnetworkIncrementalClassifierPluginMetric(TopkAccuracyPluginMetric):
    """
    Base class for all incremental classifier top-k accuracy per subnetwork plugin metrics.
    This plugin metric computes the running top-k accuracy of the incremental classifier of a given subnetwork.
    It computes a dictionary of <subnetwork ID, incremental classifier top-k accuracy value> pairs.

    This plugin metric *must* be used *only* with strategies that are subclasses
    of :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
    """
    def _update(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate):
        classes = strategy.model.subnetworks_classes[strategy.curr_sub_id]
        # if the number of classes is less than top_k, the underlying metric raises an error. In order to avoid this
        # problem, the value of top_k is temporarily set to the number of classes. This way the topk accuracy is 100%,
        # which is what one would expect when computing the topk accuracy on a set of classes whose number is less than
        # k. If the number of classes is 1, then the subnetwork is in binary classification mode and the number of
        # effective classes is 2 rather than 1.
        reset_top_k = None  # stores the original value of `top_k` if the number of classes is less than `top_k`
        if len(classes) < self._metric.top_k:
            reset_top_k = self._metric.top_k
            self._metric.top_k = len(classes) if len(classes) != 1 else 2

        if len(classes) == 1:  # if in binary classification mode.
            # The underlying metric does not support computing the topk accuracy from the logit output of a network
            # having a single output node. However, it supports computing the topk accuracy when providing the labels
            # directly.
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

        # reset the original value of top-k
        if reset_top_k is not None:
            self._metric.top_k = reset_top_k

    def update(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate):
        assert strategy.experience is not None
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


class EpochTopkAccuracyPerSubnetworkIncrementalClassifier(TopkAccuracyPerSubnetworkIncrementalClassifierPluginMetric):
    """
    The average top-k incremental classifier accuracy per subnetwork over a single training epoch.

    The top-k incremental classifier accuracy  per subnetwork will be logged after each training epoch by computing
    the number of correctly predicted patterns during the epoch divided by
    the overall number of patterns encountered in that epoch.

    This plugin metric *only* works at training time.

    This plugin metric *must* be used *only* with strategies that are subclasses
    of :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
    """
    def __init__(self, top_k: int):
        """
        Create an instance of the EpochTopkAccuracyPerSubnetworkIncrementalClassifier metric.
        :param top_k: integer number to define the value of k.
        """
        super().__init__(reset_at="epoch", emit_at="epoch", mode="train", top_k=top_k)
        self.top_k = top_k

    def __str__(self):
        return f"Top{self.top_k}_Acc_Incremental_Classifier_Epoch"


class ExperienceTopkAccuracyPerSubnetworkIncrementalClassifier(TopkAccuracyPerSubnetworkIncrementalClassifierPluginMetric):
    """
    At the end of each experience, this plugin metric reports
    the average top-k incremental classifier accuracy per subnetwork over all patterns seen in that experience.

    This metric *only* works at evaluation time.

    This plugin metric *must* be used *only* with strategies that are subclasses
    of :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
    """
    def __init__(self, top_k):
        """
        Create an instance of the ExperienceTopkAccuracyPerSubnetworkIncrementalClassifier metric.
        :param top_k: integer number to define the value of k.
        """
        super().__init__(reset_at="experience", emit_at="experience", mode="eval", top_k=top_k)
        self.top_k = top_k

    def __str__(self):
        return f"Top{self.top_k}_Acc_Incremental_Classifier_Exp"


def incremental_classifier_topk_accuracy_per_subnetwork_metrics(top_k=3, epoch=False, experience=False)\
        -> List[TopkAccuracyPerSubnetworkIncrementalClassifierPluginMetric]:
    """
    Helper method that can be used to obtain the desired set of top-k incremental classifier accuracy per subnetwork
    plugin metrics.
    :param top_k: integer number to define the value of k. Default is 3
    :param epoch: If True, will return a metric able to log the epoch top-k incremental classifier accuracy at training
        time. Default is False
    :param experience: If True, will return a metric able to log the top-k incremental classifier accuracy per
        subnetwork on each evaluation experience. Default is False
    :return: A list of top-k incremental classifier accuracy per subnetwork plugin metrics.
    """
    metrics: List[TopkAccuracyPerSubnetworkIncrementalClassifierPluginMetric] = []
    if epoch:
        metrics.append(EpochTopkAccuracyPerSubnetworkIncrementalClassifier(top_k=top_k))
    if experience:
        metrics.append(ExperienceTopkAccuracyPerSubnetworkIncrementalClassifier(top_k=top_k))
    return metrics
