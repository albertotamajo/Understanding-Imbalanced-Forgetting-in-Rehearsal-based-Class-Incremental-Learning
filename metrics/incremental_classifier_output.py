from __future__ import annotations
from typing import Dict, List, Literal, TYPE_CHECKING
from collections import defaultdict
import torch

from metrics.list_accumulator import ListAccumulatorDictionary

from avalanche.evaluation import GenericPluginMetric
from avalanche.evaluation.metric_results import MetricValue
from avalanche.evaluation.metric_utils import get_metric_name

if TYPE_CHECKING:
    from training.templates import DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate


class OutputIncrementalClassifierPluginMetric(GenericPluginMetric[Dict[int, Dict[int, List[torch.Tensor]]],
    ListAccumulatorDictionary]):
    """
    Base class for all output incremental classifier plugin metrics.

    This plugin metric accumulates the outputs of the incremental classifier of a given subnetwork sample-wise.
    It computes a dictionary of <subnetwork ID, <sample index, list of output tensors>> pairs. The sample index is the
    index to be used to fetch the given sample from the underlying dataset. The tensors in the list of output tensors
    of each sample are [1xd] tensors, where d is the number of output nodes.

    This plugin metric *must* be used *only* with strategies that are subclasses
    of :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`.

    .warning::
        This plugin metric assumes that the dataset used has the "dataset_indices" DataAttribute with
        use_in_getitem=True. The latter stores the dataset index of each sample. If this DataAttribute is not present or
        does not have use_in_getitem=True , calling `strategy.mb_dataset_indices` will  raise an :class:`AssertionError`
    """
    def __init__(self, reset_at, emit_at, mode):
        """
        Initialise a new OutputIncrementalClassifierPluginMetric
        """
        # the metric parameter accepts a metric as argument; however, a defaultdict of ListAccumulatorDictionary metrics
        # is provided instead. This will not be an issue because some methods are overridden so to work with the
        # defaultdict
        super().__init__(metric=defaultdict(ListAccumulatorDictionary), reset_at=reset_at, emit_at=emit_at, mode=mode)

    def reset(self) -> None:
        self._metric = defaultdict(ListAccumulatorDictionary)

    def result(self) -> Dict[int, Dict[int, List[torch.Tensor]]]:
        return {sub_id: metric.result() for sub_id, metric in self._metric.items()}

    def _package_result(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate):
        metric_value = self.result()
        add_exp = self._emit_at == "experience"
        plot_x_position = strategy.clock.train_iterations
        metric_name = get_metric_name(self, strategy, add_experience=add_exp, add_task=True)
        return [MetricValue(self, metric_name, metric_value, plot_x_position)]

    def _update(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate):
        curr_sub_id = int(strategy.curr_sub_id)
        mb_output = strategy.mb_output_incremental_classifier
        mb_indices = strategy.mb_dataset_indices
        for output, mb_index in zip(mb_output, mb_indices):
            if isinstance(mb_index, torch.Tensor):
                mb_index = mb_index.item()
            self._metric[curr_sub_id].update(key=int(mb_index), x=output.unsqueeze(0))

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


class ExperienceOutputIncrementalClassifier(OutputIncrementalClassifierPluginMetric):
    """
    At the end of each experience, this plugin metric emits all the outputs of the incremental classifier of a given
    subnetwork in that experience sample-wise.

    This plugin metric *must* be used *only* with strategies that are subclasses
    of :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
    """
    def __init__(self, mode: Literal["train", "eval"]):
        """
        Create an instance of ExperienceAccuracyPerSubnetworkIncrementalClassifier metric
        :param mode: the mode of this plugin metric. Either `train` or `eval`.
        """
        super().__init__(reset_at="experience", emit_at="experience", mode=mode)

    def __str__(self):
        return "Output_Incremental_Classifier_Exp"


def incremental_classifier_output_metrics(experience_train: bool = False, experience_eval: bool = False) \
        -> List[OutputIncrementalClassifierPluginMetric]:
    """
    Helper method that can be used to obtain the desired set of output incremental classifier plugin metrics.
    :param experience_train: if True, the :class:`ExperienceOutputIncrementalClassifier` with mode=train is added.
        Default is False.
    :param experience_eval: if True, the :class:`ExperienceOutputIncrementalClassifier` with mode=eval is added.
        Default is False.
    :return: A list of output incremental classifier plugin metrics
    """
    metrics: List[OutputIncrementalClassifierPluginMetric] = []
    if experience_train:
        metrics.append(ExperienceOutputIncrementalClassifier(mode="train"))
    if experience_eval:
        metrics.append(ExperienceOutputIncrementalClassifier(mode="eval"))
    return metrics
