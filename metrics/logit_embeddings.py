"""
This module includes a set of metrics and plugin metrics for keeping track of the predicted raw logits and embeddings
(feature vectors) for the samples in a dataset.
"""

from __future__ import annotations
from typing import Tuple, List, Union, Iterable, Literal, Set, Optional, TYPE_CHECKING
from avalanche.evaluation import Metric, GenericPluginMetric
import torch

if TYPE_CHECKING:
    from training.templates import DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate


class LogitEmbeddingMetric(Metric[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    """
    The standalone metric for keeping track of the predicted raw logits and embeddings (feature vectors) for the samples
    in a dataset.

    The update method receives a tensor of predicted raw logits (these must be raw logits that did not pass through the
    softmax layer), a tensor of true target labels and a tensor of embeddings (feature vectors).
    It appends the predicted logits together with the embeddings into the `logit_embedding` attribute.
    The corresponding target labels are appended into the `targets` attribute. This method
    also optionally accepts a tensor of dataset indices. The index of a given sample is the index to be used to
    retrieve the given sample from the underlying dataset. If the tensor of dataset indices is not provided, the tensor
    of indices is set to a tensor containing -1s as indices by default. The dataset indices are appended into the
    `indices` attribute.

    The result method returns the predicted raw logits, the feature embeddings collected so far, their respective target
    labels and indices. The order of the predicted raw logits, the feature embeddings and the respective target labels
    and indices reflects the order of the calls to the update method.

    The reset method will bring the metric to its initial state. By default, this metric in its initial state will
    return a tuple with four empty tensors.
    """

    def __init__(self, device: Optional[Union[str, torch.device]] = None):
        """
        Create a new LogitEmbeddingMetric
        :param device: device where to store the collected tensors. If None, the collected tensors are not moved from
            their original devices. Default is None.
        """

        self.logit_embedding: List[Tuple[torch.Tensor, torch.Tensor]] = []
        """
        a list of tuples. The first element of each tuple is a tensor encoding the predicted raw logits for several
        samples. The second element of each tuple is a tensor encoding the embeddings, also known as feature vectors,
        for the respective samples.
        """

        self.targets: List[torch.Tensor] = []
        """a list of tensors encoding the target labels for several samples"""

        self.indices: List[torch.Tensor] = []
        """
        a list of tensors encoding the dataset indices for several samples. The index of a sample is the index to be
        used to retrieve the given sample from the underlying dataset
        """

        self._device: Optional[Union[str, torch.device]] = device
        """
        device where to store the collected tensors. If None, the collected tensors are not moved from their original
        devices.
        """

    def update(self, predicted_y: torch.Tensor, true_y: torch.Tensor, embeddings: torch.Tensor,
               indices: Optional[torch.Tensor] = None) -> None:
        """
        Append the predicted raw logits together with the embeddings (feature vectors) into the `logit_embedding`
        attribute. Additionally, append the targets and indices into the `targets` and `indices` attributes,
        respectively. If `self._device` is not None, the tensors are moved to `self._device`; otherwise, they are not
        moved from their original devices.
        :param predicted_y: the predicted raw logits, they must be the raw logits before passing through the softmax
            layer
        :param true_y: the ground truth target labels
        :param embeddings: the embeddings, also known as feature vectors, being fed into the classification head
        :param indices: (optional) the dataset indices. Each index is the index to be used to retrieve the given sample
            from the underlying dataset. If None, the indices tensor is set to a tensor containing -1s.
            Default is None.
        """
        if self._device is not None:
            predicted_y = predicted_y.to(self._device)
            embeddings = embeddings.to(self._device)
            true_y = true_y.to(self._device)
            indices = indices if indices is None else indices.to(self._device)
        self.logit_embedding.append((predicted_y, embeddings))
        self.targets.append(true_y)
        if indices is None:
            indices = -torch.ones(len(true_y), dtype=torch.int64, device=predicted_y.device)
        self.indices.append(indices)

    def reset(self) -> None:
        """
        Reset the metric to its initial state. The `logit_embedding`, `targets` and `indices` attributes are emptied.
        :return: None
        """
        self.logit_embedding = []
        self.targets = []
        self.indices = []

    def result(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieve the predicted raw logits, embeddings collected so far and their respective targets and indices
        :return: a tensor of predicted raw logits, a tensor of respective embeddings, a tensor of respective targets and
            a tensor of respective indices. If `self._device` is not None, the tensors are allocated to `self._device`;
            otherwise, they are not moved from their original devices. If no tensors have been collected so far, the
            returned empty tensors will be allocated to `self._device` if it is not None; otherwise, they are allocated
            to `cpu`.
        """
        # if nothing was collected, just return four empy tensors
        if len(self.logit_embedding) == 0:
            device = torch.device("cpu") if self._device is None else self._device
            return (torch.tensor([], dtype=torch.float32, device=device),
                    torch.tensor([], dtype=torch.float32, device=device),
                    torch.tensor([], dtype=torch.int64, device=device),
                    torch.tensor([], dtype=torch.int64, device=device))

        return (torch.cat([logit for logit, _ in self.logit_embedding], dim=0),
                torch.cat([embed for _, embed in self.logit_embedding], dim=0),
                torch.cat(self.targets, dim=0), torch.cat(self.indices, dim=0))


class LogitEmbeddingPluginMetric(GenericPluginMetric[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    LogitEmbeddingMetric]):
    """
    This plugin metric collects the predicted raw logits and embeddings (feature vectors) over the samples in a given
    experience. At the end of an experience, this plugin outputs the collected predicted raw logits and embeddings along
    with the associated targets and indices. The index of a predicted raw logit is the index of the respective sample to
    be used to retrieve the given sample from the underlying dataset.

    The order of the predicted raw logits, embeddings, targets and indices reflects the order used to iterate over the
    dataset of the given experience.

    .note::
        This plugin metric *only* works at eval time.

    .note::
        This plugin *assumes* that the "dataset_indices" DataAttribute is present in the datasets of the experiences

    .note::
        The class targets collected by this plugin metric are not the real class targets but the order in which
        classes appear in the list of classes seen by the single subnetwork accessible through
        `strategy.model.subnetworks_classes[id_of_the_single_subnetwork]

    .warning::
        This plugin metric *only* works when there is *only* one subnetwork
    """

    def __init__(self,
                 experience_type: Union[Literal["train", "val", "test"], Iterable[Literal["train", "val", "test"]]],
                 device: Optional[Union[str, torch.device]] = None):
        """
        Create a new LogitEmbeddingPluginMetric
        :param experience_type: `train`, `val`, `test` or an iterable containing any combination of them.
            Decides whether to collect the predicted raw logits and embeddings (feature vectors) over the samples in the
            train, validation, test experiences, or any combination of them.
        :param device: device where to store the collected tensors. If None, the collected tensors are not moved from
            their original devices. Default is None.
        """
        if isinstance(experience_type, str):
            experience_type = {experience_type}
        else:
            experience_type = set(experience_type)
        if len(experience_type - {"train", "val", "test"}) > 0:
            raise ValueError("`experience_type` must be `train`, `val`, `test` or an iterable containing any "
                             "combination of them")

        super().__init__(LogitEmbeddingMetric(device=device), reset_at="experience", emit_at="experience", mode="eval")

        self._experience_type: Set[Literal["train", "val", "test"]] = experience_type
        """
        a set containing `train`, `val`, `test` or any combination of them.
        Decides whether to collect the predicted raw logits and embeddings (feature vectors) over the samples in the
        train, validation, test experiences, or any combination of them.
        """

    def reset(self) -> None:
        self._metric.reset()

    def result(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._metric.result()

    def update(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate):
        curr_sub_id = strategy.curr_sub_id  # store the current state of the strategy's subnetwork id
        subs_id = list(strategy.mb_output.keys())
        if len(subs_id) > 1:
            raise RuntimeError("This plugin must be used only when there is one subnetwork")
        strategy.curr_sub_id = subs_id[0]
        if "dataset_indices" not in strategy.use_in_getitem_indices.keys():
            raise RuntimeError("The dataset of the current training experience must have the `dataset_indices` "
                               "DataAttribute")
        indices = strategy.mb_dataset_indices
        self._metric.update(strategy.mb_output_incremental_classifier, strategy.mb_y_incremental_classifier,
                            strategy.mb_feature_incremental_classifier, indices=indices)
        strategy.curr_sub_id = curr_sub_id  # re-store the state of the strategy's subnetwork id

    def before_eval_exp(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate):
        if strategy.experience.origin_stream.name in self._experience_type:
            super().before_eval_exp(strategy)

    def after_eval_exp(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate):
        if strategy.experience.origin_stream.name in self._experience_type:
            return super().after_eval_exp(strategy)

    def after_eval_iteration(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate):
        if strategy.experience.origin_stream.name in self._experience_type:
            super().after_eval_iteration(strategy)

    def __str__(self):
        return "LogitEmbedding"
