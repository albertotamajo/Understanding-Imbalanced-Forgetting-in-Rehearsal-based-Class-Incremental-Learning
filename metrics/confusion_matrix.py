"""
This module includes a set of incremental classifier confusion matrix per subnetwork plugin metrics and respective
helper methods
"""
from __future__ import annotations

from typing import Dict, Optional, Literal, Union, List, TYPE_CHECKING
from collections import defaultdict

from avalanche.evaluation import Metric
from avalanche.evaluation.metrics.confusion_matrix import ConfusionMatrix
from avalanche.evaluation.metric_definitions import GenericPluginMetric
import torch
from torch import Tensor

if TYPE_CHECKING:
    from training.templates import DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate


class ConfusionMatrixPerSubnetworkIncrementalClassifier(Metric[Dict[int, Tensor]]):
    """
    The standalone incremental classifier confusion matrix per subnetwork metric.
    The metric computes a dictionary of <subnetwork ID, incremental classifier confusion matrix> pairs.
    update/result/reset methods are all subnetwork ID-aware.

    Instances of this metric keep track of the confusion matrix of the incremental classifier of multiple subnetworks
    by receiving a triplet of "ground truth" and "prediction" Tensors describing the labels of a
    minibatch and a "subnetwork ID" Tensor describing the subnetwork ID of each sample or only a single subnetwork ID.
    The "ground truth" and "prediction" tensors can both contain plain labels or one-hot/logit vectors.

    Beware that by default the confusion matrix size of each incremental classifier will depend on the value of the
    maximum label as detected by looking at both the ground truth and predictions Tensors. When passing one-hot/logit
    vectors, this metric will try to infer the number of classes from the vector sizes. Otherwise, the maximum label
    value encountered in the truth/prediction Tensors will be used.

    If the user sets the `num_classes`, then the confusion matrix of each subnetwork will always be of size
    `num_classes, num_classes`.
    Whenever a prediction or label tensor is provided as logits, only the first `num_classes` units will be considered
    in the confusion matrix computation. If they are provided as numerical labels, each of them has to be smaller
    than `num_classes`.

    The reset method will bring the metric to its initial state. By default this metric in its initial state will
    return an empty dictionary.
    """

    def __init__(self, num_classes: Optional[int] = None, normalize: Optional[Literal["true", "pred", "all"]] = None):
        """
        Create an instance of the standalone incremental classifier confusion matrix per subnetwork metric.

        By default, this metric in its initial state will return an empty dictionary.
        The metric can be updated by using the `update` method while the running confusion matrix of multiple
        subnetworks can be retrieved using the `result` method.

        :param num_classes: The number of classes. Defaults to None, which means that the number of classes will be
            inferred from ground truth and prediction Tensors (see class description for more details). If not None, the
            confusion matrix of each subnetwork will always be of size `num_classes, num_classes` and only the first
            `num_classes` values of output logits or target logits will be considered in the update. If the output or
            targets are provided as numerical labels, there can be no label greater than `num_classes`.
        :param normalize: how to normalize confusion matrix. None to not normalize
        """
        self.num_classes: Optional[int] = num_classes
        """the number of classes to keep track of for the confusion matrix of each subnetwork"""

        self.normalize: Optional[Literal["true", "pred", "all"]] = normalize
        """how to normalize the confusion matrix of each subnetwork"""

        self._confusion_matrix: Dict[int, ConfusionMatrix] = defaultdict(lambda: ConfusionMatrix(num_classes=num_classes,
                                                                                                 normalize=normalize))
        """dictionary of <subnetwork IDs, confusion matrix metric> pairs"""

    @torch.no_grad()
    def update(self, true_y: Tensor, predicted_y: Tensor, sub_ids: Union[int, Tensor]) -> None:
        """
        Update the running confusion matrix given the true, predicted labels and subnetwork IDs.
        :param true_y: the ground truth. Both labels and one-hot vectors are supported.
        :param predicted_y: the prediction. Both labels and logit vectors are supported.
        :param sub_ids: the subnetwork IDs for each sample or a single subnetwork ID
        :return: None.
        """
        if len(true_y) != len(predicted_y):
            raise ValueError("Size mismatch for true_y and predicted_y tensors")

        if isinstance(sub_ids, Tensor) and len(sub_ids) != len(true_y):
            raise ValueError("Size mismatch for true_y and sub_ids tensors")

        if isinstance(sub_ids, int):
            self._confusion_matrix[sub_ids].update(true_y, predicted_y)
        elif isinstance(sub_ids, Tensor):
            for pred, true, t in zip(predicted_y, true_y, sub_ids):
                if isinstance(t, Tensor):
                    t = t.item()
                self._confusion_matrix[t].update(true.unsqueeze(0), pred.unsqueeze(0))
        else:
            raise ValueError(
                f"Task label type: {type(sub_ids)}, "
                f"expected int or Tensor"
            )

    def result(self, sub_id: Optional[int] = None) -> Dict[int, Tensor]:
        """
        Retrieve the dictionary of running confusion matrices.
        Calling this method will not change the internal state of the metric.
        :param sub_id: if None, return the entire dictionary of confusion matrices for each subnetwork. Otherwise,
            return the dictionary `{sub_id: confusion matrix}`.
        :return: A dict of running confusion matrices for each subnetwork.
        """
        assert sub_id is None or isinstance(sub_id, int)

        if sub_id is None:
            return {k: v.result() for k, v in self._confusion_matrix.items()}
        else:
            return {sub_id: self._confusion_matrix[sub_id].result()}

    def reset(self, sub_id: Optional[int] = None) -> None:
        """
        Reset the dictionary of running confusion matrices.
        :param sub_id: if None, reset the entire dictionary. Otherwise, reset the value associated to `sub_id`.
        :return: None.
        """
        assert sub_id is None or isinstance(sub_id, int)
        if sub_id is None:
            self._confusion_matrix = defaultdict(lambda: ConfusionMatrix(num_classes=self.num_classes,
                                                                         normalize=self.normalize))
        else:
            self._confusion_matrix[sub_id].reset()


class ConfusionMatrixPerSubnetworkIncrementalClassifierPluginMetric(GenericPluginMetric[Dict[int, Tensor],
                                                                    ConfusionMatrixPerSubnetworkIncrementalClassifier]):
    """
    Base class for all incremental classifier confusion matrix per subnetwork plugin metrics.
    This plugin metric computes the running confusion matrix of a given subnetwork.
    It computes a dictionary of <subnetwork ID, confusion matrix> pairs.

    This plugin metric *must* be used *only* with strategies that are subclasses
    of :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
    """
    def __init__(self, reset_at, emit_at, mode, num_classes: Optional[int] = None,
                 normalize: Optional[Literal["true", "pred", "all"]] = None):
        """
        Create a new incremental classifier confusion matrix per subnetwork plugin metric.
        :param reset_at: when to reset the underlying metric
        :param emit_at: when to emit the underlying metric
        :param mode: when to use this plugin metric. Either "train" or "eval"
        :param num_classes: the number of classes in the confusion matrices. Defaults to None, which means that the
            number of classes will be inferred from ground truth and prediction Tensors
            (see class :class:`ConfusionMatrixPerSubnetworkIncrementalClassifier` description for more details).
            If not None, the confusion matrix of each subnetwork will always be of size `num_classes, num_classes`
            and only the first `num_classes` values of output logits or target logits will be considered in the update.
            If the output or targets are provided as numerical labels, there can be no label greater than `num_classes`.
        :param normalize: how to normalize confusion matrix. None to not normalize
        """
        super().__init__(ConfusionMatrixPerSubnetworkIncrementalClassifier(num_classes, normalize), reset_at=reset_at,
                         emit_at=emit_at, mode=mode)

    def _update(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate):
        classes = strategy.model.subnetworks_classes[strategy.curr_sub_id]
        if len(classes) == 1:  # if in binary classification mode
            # The underlying metric does not support computing the confusion matrix from the logit output of a network
            # having a single output node. However, it supports computing it when providing the labels directly.
            # The labels (predictions) are computed by passing the logit output through the sigmoid operator and then
            # applying a threshold at 0.5. Note that when in binary classification mode, the incremental classifier is
            # trained to output high values for outlier samples and low values for the single class the incremental
            # classifier must be trained on. The target of the single class is 0 while the outlier samples have target
            # 1.
            predictions = (torch.sigmoid(strategy.mb_output_incremental_classifier) > 0.5).int().flatten()
            mb_y_incremental_classifier = strategy.mb_y_incremental_classifier
            mb_y_incremental_classifier[mb_y_incremental_classifier == -1] = 1
            self._metric.update(mb_y_incremental_classifier, predictions, int(strategy.curr_sub_id))
        else:
            # if in the current mini-batch there are samples that belong to classes that the current subnetwork has
            # undertaken training on, then update the underlying confusion matrix
            if not strategy.mb_output_masked_incremental_classifier.numel() == 0:
                self._metric.update(strategy.mb_y_masked_incremental_classifier,
                                    strategy.mb_output_masked_incremental_classifier, int(strategy.curr_sub_id))

    def update(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate):
        if self._mode == "train":  # if in training mode
            self._update(strategy)
        else:
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


class ExperienceConfusionMatrixPerSubnetworkIncrementalClassifier(ConfusionMatrixPerSubnetworkIncrementalClassifierPluginMetric):
    """
    At the end of each experience, this metric reports the confusion matrix per subnetwork over all patterns seen in
    that experience.

    This plugin metric *only* works at evaluation time.

    This plugin metric *must* be used *only* with strategies that are subclasses
    of :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
    """
    def __init__(self, num_classes: Optional[int] = None, normalize: Optional[Literal["true", "pred", "all"]] = None):
        """
        Create an instance of ExperienceConfusionMatrixPerSubnetworkIncrementalClassifier metric
        :param num_classes: the number of classes in the confusion matrices. Defaults to None, which means that the
            number of classes will be inferred from ground truth and prediction Tensors
            (see class :class:`ConfusionMatrixPerSubnetworkIncrementalClassifier` description for more details).
            If not None, the
            confusion matrix of each subnetwork will always be of size `num_classes, num_classes` and only the first
            `num_classes` values of output logits or target logits will be considered in the update. If the output or
            targets are provided as numerical labels, there can be no label greater than `num_classes`.
        :param normalize: how to normalize confusion matrix. None to not normalize
        """
        super().__init__(reset_at="experience", emit_at="experience", mode="eval", num_classes=num_classes,
                         normalize=normalize)

    def __str__(self):
        return "Confusion_Matrix_Incremental_Classifier_Exp"


def incremental_classifier_confusion_matrix_per_subnetwork_metrics(experience=False, num_classes: Optional[int] = None,
                                                                   normalize: Optional[Literal["true", "pred", "all"]] = None)\
        -> List[ConfusionMatrixPerSubnetworkIncrementalClassifierPluginMetric]:
    """
    Helper method that can be used to obtain the desired set of incremental classifier confusion matrix per subnetwork
    plugin metrics.
    :param experience: If True, will return a metric able to log the incremental classifier confusion matrix per
        subnetwork on each evaluation experience. Default is False
    :param num_classes: the number of classes in the confusion matrices. Defaults to None, which means that the
        number of classes will be inferred from ground truth and prediction Tensors (see
        class :class:`ConfusionMatrixPerSubnetworkIncrementalClassifier` description for more details). If not None, the
        confusion matrix of each subnetwork will always be of size `num_classes, num_classes` and only the first
        `num_classes` values of output logits or target logits will be considered in the update. If the output or
        targets are provided as numerical labels, there can be no label greater than `num_classes`.
    :param normalize: how to normalize confusion matrix. None to not normalize
    :return: A list of incremental classifier confusion matrix per subnetwork plugin metrics.
    """
    metrics: List[ConfusionMatrixPerSubnetworkIncrementalClassifierPluginMetric] = []
    if experience:
        metrics.append(ExperienceConfusionMatrixPerSubnetworkIncrementalClassifier(num_classes, normalize))
    return metrics
