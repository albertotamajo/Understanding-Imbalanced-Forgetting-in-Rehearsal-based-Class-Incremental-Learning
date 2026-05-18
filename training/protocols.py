"""
This module includes a set of protocols
"""

from __future__ import annotations

from typing import Optional, Iterable, Dict, Union, Sequence, Tuple, Literal, Any, TYPE_CHECKING
from functools import lru_cache
import copy

from models.dynamic_optimizers import update_optimizer
from training.logits_storage import LogitsStorageDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate
from training.prob_vecs_storage import ProbVecsStorageDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate
from training.loss_functions import CrossEntropyLossMSELossLogits

from avalanche.training.utils import at_task_boundary
from avalanche.training.templates.strategy_mixin_protocol import SGDStrategyProtocol, TMBinput, TMBoutput
from avalanche.models.dynamic_optimizers import reset_optimizer
from avalanche.benchmarks import CLExperience, OnlineCLExperience
from avalanche.training.templates.update_type.sgd_update import SGDUpdate
from avalanche.training.templates.problem_type.supervised_problem import SupervisedProblem
import torch
from torch import Tensor
import torch.nn as nn
from torch.optim import Optimizer

if TYPE_CHECKING:
    from models.dynamic_networks import DynamicNetworkWithInitFeatureExtractorAndIncrementalOutlierSubNetworks


class BatchObservationDynamicNetworkInitFeatureExtractorIncrementalOutlier(SGDStrategyProtocol):
    model: DynamicNetworkWithInitFeatureExtractorAndIncrementalOutlierSubNetworks
    optimizer: Dict[str, Optimizer]
    optimizer_type: Optimizer

    def __init__(self):
        """
        Create a new BatchObservationDynamicNetworkInitFeatureExtractorIncrementalOutlier
        """
        self.optimized_param_id: Dict[str, Dict[str, Tensor]] = {}
        """
        A dictionary containing subnetwork IDs as keys and corresponding currently optimized parameters. The currently
        optimized parameters are stored in a dictionary containing parameter names as keys and corresponding tensors.
        """

    def model_adaptation(self,
                         model: Optional[
                             DynamicNetworkWithInitFeatureExtractorAndIncrementalOutlierSubNetworks
                         ] = None) -> DynamicNetworkWithInitFeatureExtractorAndIncrementalOutlierSubNetworks:
        """
        Adapt the model to the current experience
        """
        if model is None:
            model = self.model

        if self.experience is None:
            raise RuntimeError("The model cannot be adapted to the current experience because the latter is None")

        # For training:
        if isinstance(self.experience, OnlineCLExperience) and self.is_training:
            if self.experience.access_task_boundaries:
                self._model_adaptation(model, self.experience.origin_experience)
            else:
                self._model_adaptation(model, self.experience)
        else:
            self._model_adaptation(model, self.experience)
        # the whole model is passed to the correct device in case new parts of the model created during the adaptation
        # stage were not passed to the correct device
        return model.to(self.device)

    def make_optimizer(self, reset_optimizer_state=False, **kwargs):
        """
        Optimizers initialization. Called before each training experience to configure the optimizers.
        The set of existing optimizers is updated according to the changes in the respective subnetworks and/or
        new optimizers are created if new subnetwork are created as a result of the adaptation to the new experience
        :param reset_optimizer_state: whether to reset the state of the optimizer, defaults to False
        """
        sub_ids = list(self.model.subnetworks.keys())  # collect all the subnetwork IDs
        for sub_id in sub_ids:
            if sub_id not in self.optimized_param_id.keys():
                # if a new subnetwork is created, its respective optimizer needs to be created. A deep copy of the
                # optimizer stored in `self.optimizer_type` is performed and the parameters of the respective
                # subnetwork are inserted into the optimizer. Afterwards, the optimizer is added to the dictionary
                # stored in `self.optimizer`
                optimizer = copy.deepcopy(self.optimizer_type)
                self.optimized_param_id[sub_id] = reset_optimizer(optimizer, self.model.subnetworks[sub_id])
                self.optimizer[sub_id] = optimizer
            else:
                # if the subnetwork is not new, update the respective optimizer by adding the new subnetwork's
                # parameters if there are any.
                self.optimized_param_id[sub_id] = update_optimizer(self.optimizer[sub_id],
                                                                   dict(self.model.subnetworks[
                                                                            sub_id].named_parameters()),
                                                                   self.optimized_param_id[sub_id],
                                                                   reset_state=reset_optimizer_state)

    def check_model_and_optimizer(self, reset_optimizer_state=False, **kwargs):
        """
        Adapt the model to the current experience and update the optimizers.
        :param reset_optimizer_state: True to reset the optimizer states, False otherwise. Default is False
        :param kwargs:
        :return:
        """
        # If strategy has access to the task boundaries, and the current
        # sub-experience is the first sub-experience in the online stream,
        # then adapt the model with the full origin experience:
        if self.experience is None:
            raise RuntimeError(
                "The model cannot be adapted to the current experience (as a result the optimiser cannot "
                "be updated either) because the current experience is None")

        if at_task_boundary(self.experience):
            self.model = self.model_adaptation()
            self.make_optimizer(reset_optimizer_state=reset_optimizer_state, **kwargs)
        else:
            self.model = self.model_adaptation()
            self.make_optimizer(reset_optimizer_state=reset_optimizer_state, **kwargs)

    def _model_adaptation(self, model: DynamicNetworkWithInitFeatureExtractorAndIncrementalOutlierSubNetworks,
                          experience: CLExperience):
        """
        Adapt the given model to the experience. This method is called by the method
        :meth:`BatchObservationDynamicNetworkInitFeatureExtractorIncrementalOutlier.model_adaptation`
        This method *must not* be called.
        :param model: a :class:`DynamicNetworkWithInitFeatureExtractorAndIncrementalOutlierSubNetworks` instance
        :param experience: experience
        :return:
        """
        model.adaptation(experience)


class SGDUpdateDynamicNetworkInitFeatureExtractorIncrementalOutlier(SGDUpdate):
    optimizer: Dict[str, Optimizer]
    curr_sub_id: str
    curr_train_dataloader: Iterable[TMBinput]
    incremental_classifier_loss: Tensor
    outlier_detector_loss: Tensor

    def forward(self,
                subnetwork_ids: Optional[Union[Sequence[str], str, Literal["all"]]] = None,
                use_init_feature_extractor: bool = True
                ) -> TMBoutput:
        ...

    def criterion(self, use_incremental_classifier_prob_vecs: bool = False, **kwargs):
        ...

    def training_epoch(self, **kwargs):
        """
        Training epoch.

        The structure of the training epoch is:
            - before_training_iteration

                - before_forward
                - forward
                - after_forward
                - before_backward
                - backward
                - after_backward
                - before_update
                - optimizer_step
                - after_update

            - after_training_iteration
        :param kwargs:
        :return:
        """
        for self.mbatch in self.curr_train_dataloader:
            if self._stop_training:
                break

            self._unpack_minibatch()
            self._before_training_iteration(**kwargs)

            self.optimizer[self.curr_sub_id].zero_grad()
            self.incremental_classifier_loss = self._make_empty_loss()
            self.outlier_detector_loss = self._make_empty_loss()
            self.loss = self._make_empty_loss()

            # Forward
            self._before_forward(**kwargs)
            self.mb_output = self.forward(subnetwork_ids=self.curr_sub_id)
            # this attribute stores a dictionary
            # {self.curr_sub_id: {"incremental_classifier": (output1, output2), "outlier_detector": output}},
            # where output1 is the output of the incremental classifier (the logits) while output2 is the
            # feature vector, also known as embedding, that precedes the final linear classifier layer in the
            # incremental classifier.
            self._after_forward(**kwargs)

            # Loss & Backward
            self.loss += self.criterion(**kwargs)

            self._before_backward(**kwargs)
            self.backward()
            self._after_backward(**kwargs)

            # Optimization step
            self._before_update(**kwargs)
            self.optimizer_step()
            self._after_update(**kwargs)

            self._after_training_iteration(**kwargs)


class SupervisedProblemDynamicNetworkInitFeatureExtractorIncrementalOutlier(SupervisedProblem):
    model: DynamicNetworkWithInitFeatureExtractorAndIncrementalOutlierSubNetworks
    _outlier_criterion: nn.Module
    outlier_criterion_weight: float
    curr_sub_id: str
    incremental_classifier_loss: Union[Dict[str, Tensor], Tensor]
    outlier_detector_loss: Union[Dict[str, Tensor], Tensor]
    use_in_getitem_indices: Dict[str, int] = {}
    target_to_target: Dict[Any, Any]

    @lru_cache(maxsize=1)
    def _logits_storage(self, plugins: Tuple)\
            -> Optional[LogitsStorageDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate]:
        """
        Get the first :class:`LogitsStorageDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` instance in
        `plugins`. If such an instance does not exist, None is returned.

        .note::
            This method caches the result of the last call and reuses the cached value if the provided `plugins`
            parameter is unchanged from the previous call.
        """
        for plg in plugins:
            if isinstance(plg, LogitsStorageDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate):
                return plg
        return None

    @lru_cache(maxsize=1)
    def _prob_vecs_storage(self, plugins: Tuple) \
            -> Optional[ProbVecsStorageDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate]:
        """
        Get the first :class:`ProbVecsStorageDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` instance in
        `plugins`. If such an instance does not exist, None is returned.

        .note::
            This method caches the result of the last call and reuses the cached value if the provided `plugins`
            parameter is unchanged from the previous call.
        """
        for plg in plugins:
            if isinstance(plg, ProbVecsStorageDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate):
                return plg
        return None

    @property
    def logits_storage(self) -> LogitsStorageDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate:
        """
        The first :class:`LogitsStorageDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` instance in
        `self.plugins`. If such an instance does not exist, a :class:`RuntimeError` is raised.

        .note::
            The first :class:`LogitsStorageDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` instance in
            `self.plugins` is located only once and then cached. As long as `self.plugins` remains unchanged, the
            cached result will be returned. If `self.plugins` changes size or some plugin instances are replaced with
            others, the lookup will be run again and the new result be cached, replacing the previous value.
        """
        res = self._logits_storage(tuple(self.plugins))
        if res is None:
            raise RuntimeError("No `LogitsStorageDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` "
                               "instance exists in `self.plugins`.")
        return res

    @property
    def prob_vecs_storage(self) -> ProbVecsStorageDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate:
        """
        The first :class:`ProbVecsStorageDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` instance in
        `self.plugins`. If such an instance does not exist, a :class:`RuntimeError` is raised.

        .note::
            The first :class:`ProbVecsStorageDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` instance in
            `self.plugins` is located only once and then cached. As long as `self.plugins` remains unchanged, the
            cached result will be returned. If `self.plugins` changes size or some plugin instances are replaced with
            others, the lookup will be run again and the new result be cached, replacing the previous value.
        """
        res = self._prob_vecs_storage(tuple(self.plugins))
        if res is None:
            raise RuntimeError("No `ProbVecsStorageDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` "
                               "instance exists in `self.plugins`.")
        return res

    @property
    def mb_has_incremental_classifier_logits(self) -> Tensor:
        """
        A boolean tensor indicating whether the samples in the current mini-batch have a logit vector stored in
        `self.logits_storage`. The logits of a sample are its raw output scores computed at a certain time step by the
        incremental classifier of the subnetwork allocated to the class of the sample.

        .note::
            The boolean tensor is computed over **all** samples in the mini-batch, including even those
            samples that belong to classes that the incremental classifier has not been trained on

        .note::
            `self.logits_storage` and `self.mb_dataset_indices` are required by this property.
        """
        return self.logits_storage.contains_incremental_classifier_logits(
            dataset_indices=self.mb_dataset_indices, dataset_type=self.experience.origin_stream.name)

    @property
    def mb_masked_has_incremental_classifier_logits(self) -> Tensor:
        """
        A boolean tensor indicating whether the samples in the current mini-batch that belong to classes the incremental
        classifier has been trained on have a logit vector stored in `self.logits_storage`. The logits of a sample are
        its raw output scores computed at a certain time step by the incremental classifier of the subnetwork
        allocated to the class of the sample.

        .note::
            It **only** includes the samples that belong to classes that the incremental classifier has been trained on

        .note::
            This method expects each sample to have the targets_task_labels DataAttribute to be set to the ID of the
            subnetwork that has been trained on the respective class. The targets_task_labels DataAttribute stores the
            subnetwork IDs as integer numbers not as integer numbers wrapped into a string.

        .note::
            `self.logits_storage` and `self.mb_dataset_indices` are required by this property.
        """
        mask = self.mb_task_id == int(self.curr_sub_id)  # the targets_task_labels DataAttribute of each sample is
        # the ID of the subnetwork that has been trained on the respective class
        return self.logits_storage.contains_incremental_classifier_logits(
            dataset_indices=self.mb_dataset_indices[mask], dataset_type=self.experience.origin_stream.name)

    @property
    def mb_masked_has_incremental_classifier_prob_vecs(self) -> Tensor:
        """
        A boolean tensor indicating whether the samples in the current mini-batch that belong to classes the incremental
        classifier has been trained on have a probability vector stored in `self.prob_vecs_storage`.
        The probability vector of a sample is a vector whose size matches the number of classes assigned to the
        incremental classifier of the subnetwork associated with the sample's class. This vector represents
        probabilities that sum to 1. A one-hot encoding vector is a special, degenerate case of a probability vector
        where only one class has a probability of 1, and all others are 0.

        .note::
            It **only** includes the samples that belong to classes that the incremental classifier has been trained on

        .note::
            This method expects each sample to have the targets_task_labels DataAttribute to be set to the ID of the
            subnetwork that has been trained on the respective class. The targets_task_labels DataAttribute stores the
            subnetwork IDs as integer numbers not as integer numbers wrapped into a string.

        .note::
            `self.prob_vecs_storage` and `self.mb_dataset_indices` are required by this property.
        """
        curr_sub_id = int(self.curr_sub_id)
        # the targets_task_labels DataAttribute of each sample is the ID of the subnetwork that has been trained on the
        # respective class
        mask = self.mb_task_id == curr_sub_id
        return self.prob_vecs_storage.contains_incremental_classifier_prob_vecs(
            dataset_indices=self.mb_dataset_indices[mask], dataset_type=self.experience.origin_stream.name,
            subnetwork_id=curr_sub_id)

    @property
    def mb_incremental_classifier_logits(self) -> Tensor:
        """
        The logits of the samples in the current mini-batch. It is a 2D tensor containing as many rows as the number of
        samples in the current mini-batch having a logit vector stored in `self.logits_storage`. If there are no such
        samples, an empty 2D tensor is returned. If not empty, the number of columns
        (the size of the individual logit vectors) matches the size of the logit outputs of the
        current subnetwork's incremental classifier. This is achieved by padding logit vectors of smaller size with NaN
        values. The logits of a sample are its raw output scores computed at a certain time step by the incremental
        classifier of the subnetwork allocated to the class of the sample.

        .note::
            It includes **all** logits (if they exist) of the samples in the current mini-batch, even of those that
            belong to classes that the incremental classifier has not been trained on

        .note::
            `self.mb_has_incremental_classifier_logits` and `self.mb_dataset_indices` are required by this property.
        """
        logits = self.logits_storage.get_incremental_classifier_logits(
            dataset_indices=self.mb_dataset_indices[self.mb_has_incremental_classifier_logits],
            dataset_type=self.experience.origin_stream.name
        )
        if not logits.numel() == 0:  # if not empty
            # if the number of columns in `logits` does not match the respective number in the logit
            # outputs of the current subnetwork's incremental classifier
            if not logits.shape[1] == self.mb_output_incremental_classifier.shape[1]:
                raise RuntimeError("The size of the individual logit vectors taken from `self.logits_storage` does not "
                                   "match the size of the logit outputs of the current subnetwork's incremental "
                                   "classifier")
        return logits

    @property
    def mb_masked_incremental_classifier_logits(self) -> Tensor:
        """
        The logits of the samples in the current mini-batch that belong to classes the incremental classifier has been
        trained on. It is a 2D tensor containing as many rows as the number of samples in the current mini-batch, which
        belong to classes the incremental classifier has been trained on, having a logit vector stored in
        `self.logits_storage`.
        If there are no such samples, an empty 2D tensor is returned. If not empty, the number of columns
        (the size of the individual logit vectors) matches the size of the logit outputs of the
        current subnetwork's incremental classifier. This is achieved by padding logit vectors of smaller size with NaN
        values. The logits of a sample are its raw output scores computed at a certain time step by the incremental
        classifier of the subnetwork allocated to the class of the sample.

        .note::
            It **only** includes the samples that belong to classes that the incremental classifier has been trained on

        .note::
            This method expects each sample to have the targets_task_labels DataAttribute to be set to the ID of the
            subnetwork that has been trained on the respective class. The targets_task_labels DataAttribute stores the
            subnetwork IDs as integer numbers not as integer numbers wrapped into a string.

        .note::
            `self.mb_masked_has_incremental_classifier_logits` and `self.mb_dataset_indices` are required by this
            property.
        """
        mask = self.mb_task_id == int(self.curr_sub_id)  # the targets_task_labels DataAttribute of each sample is
        # the ID of the subnetwork that has been trained on the respective class
        logits = self.logits_storage.get_incremental_classifier_logits(
            dataset_indices=self.mb_dataset_indices[mask][self.mb_masked_has_incremental_classifier_logits],
            dataset_type=self.experience.origin_stream.name
        )
        if not logits.numel() == 0:  # if not empty
            # if the number of columns in `logits` does not match the respective number in the logit
            # outputs of the current subnetwork's incremental classifier
            if not logits.shape[1] == self.mb_output_incremental_classifier.shape[1]:
                raise RuntimeError("The size of the individual logit vectors taken from `self.logits_storage` does not "
                                   "match the size of the logit outputs of the current subnetwork's incremental "
                                   "classifier")
        return logits

    @property
    def mb_masked_incremental_classifier_prob_vecs(self) -> Tensor:
        """
        The probability vectors of the samples in the current mini-batch that belong to classes the incremental
        classifier has been trained on. It is a 2D tensor containing as many rows as the number of samples in the
        current mini-batch that belong to classes the incremental classifier has been trained on. The number of columns
        (the size of the individual probability vectors) matches the number of classes assigned to the
        current subnetwork's incremental classifier. If a sample does not have a probability vector stored in
        `self.prob_vecs_storage`, the one-hot encoding vector of its class is used.
        If there are no samples in the current mini-batch that belong to classes the incremental
        classifier has been trained on, an empty 2D tensor of size (0, num_classes) is returned, where num_classes is
        equal to the number of classes assigned to the current subnetwork's incremental classifier.
        The probability vector of a sample is a vector whose size matches the number of classes assigned to the
        incremental classifier of the subnetwork associated with the sample's class. This vector represents
        probabilities that sum to 1. A one-hot encoding vector is a special, degenerate case of a probability vector
        where only one class has a probability of 1, and all others are 0.

        .note::
            It **only** includes the samples that belong to classes that the incremental classifier has been trained on

        .note::
            This method expects each sample to have the targets_task_labels DataAttribute to be set to the ID of the
            subnetwork that has been trained on the respective class. The targets_task_labels DataAttribute stores the
            subnetwork IDs as integer numbers not as integer numbers wrapped into a string.

        .note::
            `self.prob_vecs_storage` and `self.mb_dataset_indices` are required by this property.
        """
        curr_sub_id = int(self.curr_sub_id)
        # the targets_task_labels DataAttribute of each sample is the ID of the subnetwork that has been trained on the
        # respective class
        mask = self.mb_task_id == curr_sub_id
        # the number of classes assigned to the current subnetwork's incremental classifier
        num_classes = len(self.model.subnetworks_classes[str(curr_sub_id)])
        # a 2d tensor containing the one-hot encoding vectors of the samples in the current mini-batch that belong to
        # classes the incremental classifier has been trained on. The one-hot encoding vector of each sample is selected
        # according to the class the given sample belongs to
        one_hot_encoding_vecs = torch.eye(num_classes, device=mask.device,
                                          dtype=self.mb_output_incremental_classifier.dtype
                                          )[self.mb_y_masked_incremental_classifier]
        # the probability vectors of the samples in the current mini-batch that belong to classes the incremental
        # classifier has been trained on and have a probability vector stored in `self.prob_vecs`
        prob_vecs = self.prob_vecs_storage.get_incremental_classifier_prob_vecs(
            dataset_indices=self.mb_dataset_indices[mask][self.mb_masked_has_incremental_classifier_prob_vecs],
            dataset_type=self.experience.origin_stream.name, subnetwork_id=curr_sub_id, num_classes=num_classes
        )
        # replace the one-hot encoding vectors of the samples that have a probability vector stored in
        # `self.prob_vec_storage` with the respective probability vector
        one_hot_encoding_vecs[self.mb_masked_has_incremental_classifier_prob_vecs] = prob_vecs

        return one_hot_encoding_vecs

    @property
    def mb_masked_logits_incremental_classifier_prob_vecs(self) -> Tensor:
        """
        The probability vectors of the samples in the current mini-batch that belong to classes the incremental
        classifier has been trained on and have a logit vector stored in `self.logits_storage`.
        It is a 2D tensor containing as many rows as the number of samples in the current mini-batch that belong to
        classes the incremental classifier has been trained on and have a logit vector stored in `self.logits_storage`.
        The number of columns (the size of the individual probability vectors) matches the number of classes assigned to
        the current subnetwork's incremental classifier. If a sample does not have a probability vector stored in
        `self.prob_vecs_storage`, the one-hot encoding vector of its class is used.
        If there are no samples in the current mini-batch that belong to classes the incremental
        classifier has been trained on and have a logit vector stored in `self.logits_storage`, an empty 2D tensor of
        size (0, num_classes) is returned, where num_classes is equal to the number of classes assigned to the current
        subnetwork's incremental classifier.
        The probability vector of a sample is a vector whose size matches the number of classes assigned to the
        incremental classifier of the subnetwork associated with the sample's class. This vector represents
        probabilities that sum to 1. A one-hot encoding vector is a special, degenerate case of a probability vector
        where only one class has a probability of 1, and all others are 0.

        .note::
            It **only** includes the samples that belong to classes that the incremental classifier has been trained on
            and have a logit vector stored in `self.logits_storage`
        .note::
            This method expects each sample to have the targets_task_labels DataAttribute to be set to the ID of the
            subnetwork that has been trained on the respective class. The targets_task_labels DataAttribute stores the
            subnetwork IDs as integer numbers not as integer numbers wrapped into a string.

        .note::
            `self.prob_vecs_storage`, `self.mb_masked_has_incremental_classifier_logits` and
            `self.mb_dataset_indices` are required by this property.
        """
        return self.mb_masked_incremental_classifier_prob_vecs[self.mb_masked_has_incremental_classifier_logits]

    @property
    def mb_masked_no_logits_incremental_classifier_prob_vecs(self) -> Tensor:
        """
        The probability vectors of the samples in the current mini-batch that belong to classes the incremental
        classifier has been trained on and do not have a logit vector stored in `self.logits_storage`.
        It is a 2D tensor containing as many rows as the number of samples in the current mini-batch that belong to
        classes the incremental classifier has been trained on and do not have a logit vector stored in
        `self.logits_storage`.
        The number of columns (the size of the individual probability vectors) matches the number of classes assigned to
        the current subnetwork's incremental classifier. If a sample does not have a probability vector stored in
        `self.prob_vecs_storage`, the one-hot encoding vector of its class is used.
        If there are no samples in the current mini-batch that belong to classes the incremental
        classifier has been trained on and do not have a logit vector stored in `self.logits_storage`,
        an empty 2D tensor of size (0, num_classes) is returned, where num_classes is equal to the number of classes
        assigned to the current subnetwork's incremental classifier.
        The probability vector of a sample is a vector whose size matches the number of classes assigned to the
        incremental classifier of the subnetwork associated with the sample's class. This vector represents
        probabilities that sum to 1. A one-hot encoding vector is a special, degenerate case of a probability vector
        where only one class has a probability of 1, and all others are 0.

        .note::
            It **only** includes the samples that belong to classes that the incremental classifier has been trained on
            and do not have a logit vector stored in `self.logits_storage`
        .note::
            This method expects each sample to have the targets_task_labels DataAttribute to be set to the ID of the
            subnetwork that has been trained on the respective class. The targets_task_labels DataAttribute stores the
            subnetwork IDs as integer numbers not as integer numbers wrapped into a string.

        .note::
            `self.prob_vecs_storage`, `self.mb_masked_has_incremental_classifier_logits` and
            `self.mb_dataset_indices` are required by this property.
        """
        return self.mb_masked_incremental_classifier_prob_vecs[~self.mb_masked_has_incremental_classifier_logits]

    @property
    def mb_task_id(self) -> Tensor:
        """Current mini-batch task labels."""
        mbatch = self.mbatch
        assert mbatch is not None
        assert len(mbatch) >= 3
        assert "targets_task_labels" in self.use_in_getitem_indices.keys()
        index = self.use_in_getitem_indices["targets_task_labels"]
        return mbatch[index]

    @property
    def mb_dataset_indices(self) -> Tensor:
        """
        Current mini-batch dataset indices. The index of each sample is the index to be used to retrieve the sample from
        the underlying dataset, i.e. underlying_dataset[index] retrieves the respective sample.
        """
        mbatch = self.mbatch
        assert mbatch is not None
        assert len(mbatch) >= 3
        assert "dataset_indices" in self.use_in_getitem_indices.keys()
        index = self.use_in_getitem_indices["dataset_indices"]
        return mbatch[index]

    @property
    def mb_output_incremental_classifier(self) -> Tensor:
        """
        Current mini-batch output of the incremental classifier.

        .note::
            It includes all the samples in the mini-batch output, even those samples that belong to classes that
            the incremental classifier has not been trained on
        """
        mb_output = self.mb_output
        assert mb_output is not None
        return mb_output[self.curr_sub_id]["incremental_classifier"][0]

    @property
    def mb_feature_incremental_classifier(self) -> Tensor:
        """
        Current mini-batch of feature vectors of the incremental classifier.

        The feature vectors, also known as embeddings, are the vectors that precede the final linear classifier layer
        in the incremental classifier.

        .note::
            It includes all the samples in the mini-batch of feature vectors, even those samples that belong to classes
            that the incremental classifier has not been trained on
        """
        mb_output = self.mb_output
        assert mb_output is not None
        return mb_output[self.curr_sub_id]["incremental_classifier"][1]

    @property
    def mb_output_masked_incremental_classifier(self) -> Tensor:
        """
        Current mini-batch output of the incremental classifier. It only includes the samples in the mini-batch output
        that belong to classes the incremental classifier has been trained on

        .note::
            This method expects each sample to have the targets_task_labels DataAttribute to be set to the ID of the
            subnetwork that has been trained on the respective class. The targets_task_labels DataAttribute stores the
            subnetwork IDs as integer numbers not as integer numbers wrapped into a string.
        """
        mask = self.mb_task_id == int(self.curr_sub_id)  # the targets_task_labels DataAttribute of each sample is
        # the ID of the subnetwork that has been trained on the respective class
        return self.mb_output_incremental_classifier[mask]

    @property
    def mb_output_masked_logits_incremental_classifier(self) -> Tensor:
        """
        Current mini-batch output of the incremental classifier. It only includes the samples in the mini-batch output
        that belong to classes the incremental classifier has been trained on and have a logit vector stored in
        `self.logits_storage`.

        .note::
            This method expects each sample to have the targets_task_labels DataAttribute to be set to the ID of the
            subnetwork that has been trained on the respective class. The targets_task_labels DataAttribute stores the
            subnetwork IDs as integer numbers not as integer numbers wrapped into a string.

        .note::
            `self.mb_masked_has_incremental_classifier_logits` is required by this property
        """
        return self.mb_output_masked_incremental_classifier[self.mb_masked_has_incremental_classifier_logits]

    @property
    def mb_output_masked_no_logits_incremental_classifier(self) -> Tensor:
        """
        Current mini-batch output of the incremental classifier. It only includes the samples in the mini-batch output
        that belong to classes the incremental classifier has been trained on and do not have a logit vector stored in
        `self.logits_storage`.

        .note::
            This method expects each sample to have the targets_task_labels DataAttribute to be set to the ID of the
            subnetwork that has been trained on the respective class. The targets_task_labels DataAttribute stores the
            subnetwork IDs as integer numbers not as integer numbers wrapped into a string.

        .note::
            `self.mb_masked_has_incremental_classifier_logits` is required by this property
        """
        return self.mb_output_masked_incremental_classifier[~self.mb_masked_has_incremental_classifier_logits]

    @property
    def mb_feature_masked_incremental_classifier(self) -> Tensor:
        """
        Current mini-batch of feature vectors of the incremental classifier. It only includes the samples in the
        mini-batch of feature vectors that belong to classes the incremental classifier has been trained on

        .note::
            This method expects each sample to have the targets_task_labels DataAttribute to be set to the ID of the
            subnetwork that has been trained on the respective class. The targets_task_labels DataAttribute stores the
            subnetwork IDs as integer numbers not as integer numbers wrapped into a string.
        """
        mask = self.mb_task_id == int(self.curr_sub_id)  # the targets_task_labels DataAttribute of each sample is
        # the ID of the subnetwork that has been trained on the respective class
        return self.mb_feature_incremental_classifier[mask]

    @property
    def mb_feature_masked_logits_incremental_classifier(self) -> Tensor:
        """
        Current mini-batch of feature vectors of the incremental classifier. It only includes the samples in the
        mini-batch of feature vectors that belong to classes the incremental classifier has been trained on and
        have a logit vector stored in `self.logits_storage`.

        .note::
            This method expects each sample to have the targets_task_labels DataAttribute to be set to the ID of the
            subnetwork that has been trained on the respective class. The targets_task_labels DataAttribute stores the
            subnetwork IDs as integer numbers not as integer numbers wrapped into a string.

        .note::
            `self.mb_masked_has_incremental_classifier_logits` is required by this property
        """
        return self.mb_feature_masked_incremental_classifier[self.mb_masked_has_incremental_classifier_logits]

    @property
    def mb_feature_masked_no_logits_incremental_classifier(self) -> Tensor:
        """
        Current mini-batch of feature vectors of the incremental classifier. It only includes the samples in the
        mini-batch of feature vectors that belong to classes the incremental classifier has been trained on and
        do not have a logit vector stored in `self.logits_storage`.

        .note::
            This method expects each sample to have the targets_task_labels DataAttribute to be set to the ID of the
            subnetwork that has been trained on the respective class. The targets_task_labels DataAttribute stores the
            subnetwork IDs as integer numbers not as integer numbers wrapped into a string.

        .note::
            `self.mb_masked_has_incremental_classifier_logits` is required by this property
        """
        return self.mb_feature_masked_incremental_classifier[~self.mb_masked_has_incremental_classifier_logits]

    @property
    def mb_output_outlier_detector(self) -> Tensor:
        """
        Current mini-batch output of the outlier detector. It includes all samples.
        """
        mb_output = self.mb_output
        assert mb_output is not None
        return mb_output[self.curr_sub_id]["outlier_detector"]

    @property
    def mb_y(self):
        """
        Current mini-batch target. If a target in the current mini-batch is present as a key in the `target_to_target`
        dictionary, it is replaced with its respective value in the `target_to_target` dictionary.
        """
        mb_y = super().mb_y
        if len(self.target_to_target) == 0:
            return mb_y
        else:
            mb_y = copy.deepcopy(mb_y).cpu()
            mb_y.apply_(lambda x: self.target_to_target[x] if x in self.target_to_target.keys() else x)
            return mb_y.to(self.device)

    @property
    def mb_y_incremental_classifier(self) -> Tensor:
        """
        Current mini-batch target of the incremental classifier.

        .note::
            It includes all samples in the mini-batch, even those samples that belong to classes that
            the current network has not been trained on.
            The target of these samples is set to -1. For those classes that the current subnetwork has been trained on,
            the target indices are computed according to the order in which
            classes appear in the list of classes seen by the current subnetwork accessible through
            `self.model.subnetworks_classes[self.curr_sub_id]`

        .note::
            The output is a 1D tensor
        """
        incremental_classifier_targets = copy.deepcopy(self.mb_y).cpu()
        classes = self.model.subnetworks_classes[self.curr_sub_id]
        incremental_classifier_targets.apply_(lambda x: classes.index(x) if x in classes else -1)
        return incremental_classifier_targets.to(self.device)

    @property
    def mb_y_masked_incremental_classifier(self) -> Tensor:
        """
        Current mini-batch target of the incremental classifier. It only includes the samples in the mini-batch that
        belong to classes the current subnetwork has been trained on.

        .note::
            Refer to method
            :meth:`SupervisedProblemDynamicNetworkInitFeatureExtractorIncrementalOutlier.mb_y_incremental_classifier`
            for more information about how the target indices are computed. This method expects each sample to have the
            targets_task_labels DataAttribute to be set to the ID of the subnetwork that has been trained on the
            respective class. The targets_task_labels DataAttribute stores the subnetwork IDs as integer numbers
            not as integer numbers wrapped into a string.

        .note::
            The output is a 1D tensor
        """
        mask = self.mb_task_id == int(self.curr_sub_id)  # the targets_task_labels DataAttribute of each sample is
        # the ID of the subnetwork that has been trained on the respective class
        return self.mb_y_incremental_classifier[mask]

    @property
    def mb_y_masked_logits_incremental_classifier(self) -> Tensor:
        """
        Current mini-batch target of the incremental classifier. It only includes the samples in the mini-batch that
        belong to classes the current subnetwork has been trained on and have a logit vector stored in
        `self.logits_storage`.

        .note::
            Refer to method
            :meth:`SupervisedProblemDynamicNetworkInitFeatureExtractorIncrementalOutlier.mb_y_incremental_classifier`
            for more information about how the target indices are computed. This method expects each sample to have the
            targets_task_labels DataAttribute to be set to the ID of the subnetwork that has been trained on the
            respective class. The targets_task_labels DataAttribute stores the subnetwork IDs as integer numbers
            not as integer numbers wrapped into a string.

        .note::
            `self.mb_masked_has_incremental_classifier_logits` is required by this property

        .note::
            The output is a 1D tensor
        """
        return self.mb_y_masked_incremental_classifier[self.mb_masked_has_incremental_classifier_logits]

    @property
    def mb_y_masked_no_logits_incremental_classifier(self) -> Tensor:
        """
        Current mini-batch target of the incremental classifier. It only includes the samples in the mini-batch that
        belong to classes the current subnetwork has been trained on and do not have a logit vector stored in
        `self.logits_storage`.

        .note::
            Refer to method
            :meth:`SupervisedProblemDynamicNetworkInitFeatureExtractorIncrementalOutlier.mb_y_incremental_classifier`
            for more information about how the target indices are computed. This method expects each sample to have the
            targets_task_labels DataAttribute to be set to the ID of the subnetwork that has been trained on the
            respective class. The targets_task_labels DataAttribute stores the subnetwork IDs as integer numbers
            not as integer numbers wrapped into a string.

        .note::
            `self.mb_masked_has_incremental_classifier_logits` is required by this property

        .note::
            The output is a 1D tensor
        """
        return self.mb_y_masked_incremental_classifier[~self.mb_masked_has_incremental_classifier_logits]

    @property
    def mb_y_outlier_detector(self) -> Tensor:
        """
        Current mini-batch target of the outlier detector. Samples of classes that the current subnetwork has been
        trained on are represented with 0 while samples of classes that the current subnetwork has not been trained
        on are represented with 1.

        .note::
            This method expects each sample to have the targets_task_labels DataAttribute to be set to the ID of the
            subnetwork that has been trained on the respective class. The targets_task_labels DataAttribute stores the
            subnetwork IDs as integer numbers not as integer numbers wrapped into a string.

        .note::
            The output is a 1D tensor
        """
        outlier_detector_targets = copy.deepcopy(self.mb_y).to(self.device)
        outlier_detector_targets[self.mb_task_id == int(self.curr_sub_id)] = 0
        outlier_detector_targets[self.mb_task_id != int(self.curr_sub_id)] = 1
        return outlier_detector_targets

    def criterion(self, use_incremental_classifier_prob_vecs: bool = False, **kwargs):
        """
        Compute the incremental classifier loss function, outlier detector loss function and subnetwork loss function.

        If in training mode, the attribute `self.mb_output`, which stores the current
        mini-batch output, *must* only contain the output of the current training subnetwork. The loss is computed on
        that output.

        If in evaluation mode, `self.mb_output` can store the outputs of an arbitrary number of subnetworks and the
        loss is computed for each of them.

        If a subnetwork has only one output node in the incremental classifier then the incremental classifier is in
        binary classification mode (it is trained to recognise the only class it has seen from all the other classes or
        evaluated on this if in evaluation mode). Therefore, the incremental classifier is fed with classes it has not
        seen too. On the other hand, if a subnetwork has more than one output node in the incremental classifier then
        the incremental classifier is trained/evaluated on the seen classes only.

        The outlier detector of a subnetwork always receives the entire batch of data.

        The subnetwork loss function is generally computed as follows if `use_incremental_classifier_prob_vecs` is
        False:

        `incremental_classifier_criterion(incremental_classifier_output, incremental_classifier_targets) +
        outlier_criterion_weight * outlier_criterion(outlier_detector_output, outlier_detector_target)` where
        `incremental_classifier_targets` is accessed through the `mb_y_masked_incremental_classifier` property.

        Otherwise, if `use_incremental_classifier_prob_vecs` is True, the subnetwork loss function is generally computed
        as follows:

        `incremental_classifier_criterion(incremental_classifier_output, incremental_classifier_prob_vecs) +
        outlier_criterion_weight * outlier_criterion(outlier_detector_output, outlier_detector_target)` where
        `incremental_classifier_prob_vecs` is accessed through the `mb_masked_incremental_classifier_prob_vecs`
        property.

        Note that if `use_incremental_classifier_prob_vecs` is True, an instance
        of :class:`ProbVecsStorageDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` *must* be
        present within the sequence of plugins in the `self.plugins` attribute.


        If the incremental classifier criterion is an instance of :class:`CrossEntropyLossMSELossLogits`, the output of
        the incremental classifier of each subnetwork *must* be raw logits before passing through a softmax layer and
        an instance of :class:`LogitsStorageDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` *must* be
        present within the sequence of plugins in the `self.plugins` attribute. The subnetwork loss function is
        computed as follows:

        `incremental_classifier_criterion(incremental_classifier_output_no_past_logits,
        incremental_classifier_output_past_logits, past_logits, incremental_classifier_targets_no_past_logits,
        incremental_classifier_targets_past_logits) +
        outlier_criterion_weight * outlier_criterion(outlier_detector_output, outlier_detector_target)`

        where
            -  `incremental_classifier_output_no_past_logits` is the output of the incremental classifier for the
               samples that belong to classes the incremental classifier has been trained on and have no past logits.
               It is is accessible through the `mb_output_masked_no_logits_incremental_classifier` property
            -  `incremental_classifier_output_past_logits` is the output of the incremental classifier for the
               samples that belong to classes the incremental classifier has been trained on and have past logits.
               It is accessible through the `mb_output_masked_logits_incremental_classifier` property
            - `past_logits` the past logits of the samples that belong to classes the incremental classifier has been
              trained on and have past logits. It is accessible through the `mb_masked_incremental_classifier_logits`
              property
            - `incremental_classifier_targets_no_past_logits` is the targets for the samples that belong to classes the
              incremental classifier has been trained on and have no past logits, if
              `use_incremental_classifier_prob_vecs` is False. Otherwise, it is the probability vectors for the samples
              that belong to classes the incremental classifier has been trained on and have no past logits.
              It is accessible through the `mb_y_masked_no_logits_incremental_classifier` property if
              `use_incremental_classifier_prob_vecs` is False. Otherwise, through the
              `mb_masked_no_logits_incremental_classifier_prob_vecs` property.
            - `incremental_classifier_targets_past_logits` is the targets for the samples that belong to classes the
              incremental classifier has been trained on and have past logits, if
              `use_incremental_classifier_prob_vecs` is False. Otherwise, it is the probability vectors for the samples
              that belong to classes the incremental classifier has been trained on and have past logits.
              It is accessible through the `mb_y_masked_logits_incremental_classifier` property if
              `use_incremental_classifier_prob_vecs` is False. Otherwise, through the
              `mb_masked_logits_incremental_classifier_prob_vecs` property.


        TODO should the outlier criterion weight be used when in binary classification mode?

        .note::
            The incremental classifier and outlier detector losses are also stored separately in
            `self.incremental_classifier_loss` and `self.outlier_detector_loss`, respectively. If in training mode,
            both losses are tensors storing a single scalar. If in evaluation mode, both attributes are dictionaries
            containing the subnetwork IDs as keys and their respective loss tensors as values.

        .note::
            This method expects the outlier criterion stored in `self._outlier_criterion` to be a criterion that
            requires the target tensor to be of the same shape as the input tensor.

        ..warning::
            Both in training mode and evaluation mode and when the current subnetwork has been trained on more than one
            class, it might happen that the current mini-batch does not include samples that belong to classes that a
            given subnetwork has been trained on. In this case, if in training mode, the incremental
            classifier loss is not incremented (therefore, it should be 0). If in evaluation mode, the incremental
            classifier loss is set to 0. Loss metrics *must* take this into account to avoid updating when this occurs.

        :return a loss tensor if in training mode. A dictionary containing the subnetwork IDs as keys and the respective
        loss tensors as values if in evaluation mode.
        """
        def compute_losses(use_incremental_classifier_prob_vecs: bool):
            """
            Compute the incremental classifier and outlier detector losses for the current subnetwork.

            The outlier detector loss is computed using the outlier detector criterion
            `self._outlier_criterion` on the whole mini batch output of the outlier detector.

            The incremental classifier loss is computed differently according to the number of classes the current
            subnetwork has seen. If the current subnetwork has seen only one class, the incremental classifier loss
            is computed on the entire mini batch using the outlier criterion (Binary Cross Entropy). Otherwise, the
            incremental classifier loss is computed using the incremental classifier criterion `self._criterion` on only
            the samples of the mini batch that belong to classes that the current subnetwork has seen.

            The subnetwork loss function is generally computed as follows if `use_incremental_classifier_prob_vecs` is
            False:

            `incremental_classifier_criterion(incremental_classifier_output, incremental_classifier_targets) +
            outlier_criterion_weight * outlier_criterion(outlier_detector_output, outlier_detector_target)` where
            `incremental_classifier_targets` is accessed through the `mb_y_masked_incremental_classifier` property.

            Otherwise, if `use_incremental_classifier_prob_vecs` is True, the subnetwork loss function is generally
            computed as follows:

            `incremental_classifier_criterion(incremental_classifier_output, incremental_classifier_prob_vecs) +
            outlier_criterion_weight * outlier_criterion(outlier_detector_output, outlier_detector_target)` where
            `incremental_classifier_prob_vecs` is accessed through the `mb_masked_incremental_classifier_prob_vecs`
            property.

            Note that if `use_incremental_classifier_prob_vecs` is True, an instance
            of :class:`ProbVecsStorageDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` *must* be
            present within the sequence of plugins in the `self.plugins` attribute.

            If `self.criterion` is an instance of :class:`CrossEntropyLossMSELossLogits`, the output of the incremental
            classifier of each subnetwork *must* be raw logits before passing through a softmax layer and
            an instance of :class:`LogitsStorageDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` *must* be
            present within the sequence of plugins in the `self.plugins` attribute. The subnetwork loss function is
            computed as follows:

            `incremental_classifier_criterion(incremental_classifier_output_no_past_logits,
            incremental_classifier_output_past_logits, past_logits, incremental_classifier_targets_no_past_logits,
            incremental_classifier_targets_past_logits) +
            outlier_criterion_weight * outlier_criterion(outlier_detector_output, outlier_detector_target)`

            where
            -  `incremental_classifier_output_no_past_logits` is the output of the incremental classifier for the
               samples that belong to classes the incremental classifier has been trained on and have no past logits.
               It is is accessible through the `mb_output_masked_no_logits_incremental_classifier` property
            -  `incremental_classifier_output_past_logits` is the output of the incremental classifier for the
               samples that belong to classes the incremental classifier has been trained on and have past logits.
               It is accessible through the `mb_output_masked_logits_incremental_classifier` property
            - `past_logits` the past logits of the samples that belong to classes the incremental classifier has been
              trained on and have past logits. It is accessible through the `mb_masked_incremental_classifier_logits`
              property
            - `incremental_classifier_targets_no_past_logits` is the targets for the samples that belong to classes the
              incremental classifier has been trained on and have no past logits, if
              `use_incremental_classifier_prob_vecs` is False. Otherwise, it is the probability vectors for the samples
              that belong to classes the incremental classifier has been trained on and have no past logits.
              It is accessible through the `mb_y_masked_no_logits_incremental_classifier` property if
              `use_incremental_classifier_prob_vecs` is False. Otherwise, through the
              `mb_masked_no_logits_incremental_classifier_prob_vecs` property.
            - `incremental_classifier_targets_past_logits` is the targets for the samples that belong to classes the
              incremental classifier has been trained on and have past logits, if
              `use_incremental_classifier_prob_vecs` is False. Otherwise, it is the probability vectors for the samples
              that belong to classes the incremental classifier has been trained on and have past logits.
              It is accessible through the `mb_y_masked_logits_incremental_classifier` property if
              `use_incremental_classifier_prob_vecs` is False. Otherwise, through the
              `mb_masked_logits_incremental_classifier_prob_vecs` property.

            .note::
                This method expects the outlier criterion stored in `self._outlier_criterion` to be a criterion that
                requires the target tensor to be of the same shape as the input tensor.

            :return: incremental classifier loss, outlier detector loss
            """
            # unsqueeze and set dtype to match the shape of the input and its dtype
            mb_y_outlier_detector = self.mb_y_outlier_detector.unsqueeze(1).to(self.mb_output_outlier_detector.dtype)
            outlier_detector_loss = self._outlier_criterion(self.mb_output_outlier_detector, mb_y_outlier_detector)

            classes = self.model.subnetworks_classes[self.curr_sub_id]
            if len(classes) == 1:
                mb_y_incremental_classifier = self.mb_y_incremental_classifier
                mb_y_incremental_classifier[mb_y_incremental_classifier == -1] = 1
                # unsqueeze and set dtype to match the shape of the input and its dtype
                mb_y_incremental_classifier = mb_y_incremental_classifier.unsqueeze(1).to(
                    self.mb_output_incremental_classifier.dtype)
                incremental_classifier_loss = self._outlier_criterion(self.mb_output_incremental_classifier,
                                                                      mb_y_incremental_classifier)
            else:
                # If there are no samples that belong to classes that have undertaken training on the current subnetwork,
                # then the incremental classifier loss is set to 0
                if self.mb_output_masked_incremental_classifier.numel() == 0:
                    incremental_classifier_loss = torch.zeros_like(outlier_detector_loss)
                else:
                    if isinstance(self._criterion, CrossEntropyLossMSELossLogits):
                        if use_incremental_classifier_prob_vecs:
                            mb_y_masked_no_logits_target_probs = self.mb_masked_no_logits_incremental_classifier_prob_vecs
                            mb_y_masked_logits_target_probs = self.mb_masked_logits_incremental_classifier_prob_vecs
                        else:
                            mb_y_masked_no_logits_target_probs = self.mb_y_masked_no_logits_incremental_classifier
                            mb_y_masked_logits_target_probs = self.mb_y_masked_logits_incremental_classifier

                        incremental_classifier_loss = self._criterion(
                            self.mb_output_masked_no_logits_incremental_classifier,
                            self.mb_output_masked_logits_incremental_classifier,
                            self.mb_masked_incremental_classifier_logits,
                            mb_y_masked_no_logits_target_probs,
                            mb_y_masked_logits_target_probs
                        )
                    else:
                        if use_incremental_classifier_prob_vecs:
                            mb_y_masked_target_probs = self.mb_masked_incremental_classifier_prob_vecs
                        else:
                            mb_y_masked_target_probs = self.mb_y_masked_incremental_classifier

                        incremental_classifier_loss = self._criterion(self.mb_output_masked_incremental_classifier,
                                                                      mb_y_masked_target_probs)
            return incremental_classifier_loss, outlier_detector_loss

        if self.is_training:  # if in training mode
            if not (len(self.mb_output) == 1 and self.curr_sub_id in self.mb_output.keys()):
                raise RuntimeError("The attribute self.mb_output must have only one key during training and the latter"
                                   "must be the ID of the subnetwork currently being trained")
            incremental_classifier_loss, outlier_detector_loss = compute_losses(use_incremental_classifier_prob_vecs)
            self.incremental_classifier_loss += incremental_classifier_loss
            self.outlier_detector_loss += outlier_detector_loss
            return self.incremental_classifier_loss + self.outlier_criterion_weight * self.outlier_detector_loss
        else:  # if in evaluation mode
            incremental_classifier_loss = {}
            outlier_detector_loss = {}
            loss = {}
            for self.curr_sub_id in self.mb_output.keys():
                incremental_classifier_loss_, outlier_detector_loss_ = compute_losses(
                    use_incremental_classifier_prob_vecs)
                incremental_classifier_loss[self.curr_sub_id] = incremental_classifier_loss_
                outlier_detector_loss[self.curr_sub_id] = outlier_detector_loss_
                loss[self.curr_sub_id] = (incremental_classifier_loss_ +
                                          self.outlier_criterion_weight * outlier_detector_loss_)
            self.incremental_classifier_loss = incremental_classifier_loss
            self.outlier_detector_loss = outlier_detector_loss
            return loss

    def forward(self, subnetwork_ids: Optional[Union[Sequence[str], str, Literal["all"]]] = None,
                use_init_feature_extractor: bool = True):
        """Compute the model's output given the current mini-batch."""
        return self.model(self.mb_x, subnetwork_ids, use_init_feature_extractor)
