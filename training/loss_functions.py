"""
This module includes a set of loss functions
"""
from typing import Union, Tuple, Dict, Any, Optional
import torch
import torch.nn as nn


class CrossEntropyLossMSELossLogits(nn.Module):
    """
    This criterion combines the cross entropy loss between input logits and targets, and the mean squared error loss
    between past and current logits. This is the loss used in https://arxiv.org/pdf/2004.07211

    The cross entropy loss is computed using :class:`nn.CrossEntropyLoss` while the mean squared error loss is computed
    using :class:`nn.MSELoss``

    It takes the following inputs:
        - the current logits of the samples that have no past logits (`current_logits_no_past_logits`) and their
          targets (`targets_no_past_logits`)
        - the current logits of the samples that have past logits (`current_logits_past_logits`),
          their targets (`targets_past_logits`) and their past logits (`past_logits`)
    The targets can be either 1D tensors of integer numbers indicating the class the samples belong to or 2D tensors of
    probability vectors where each row sums up to 1 and has the same size as each logit vector.
    A one-hot encoding vector is a special, degenerate case of a probability vector where only one class has a
    probability of 1, and all others are 0.

    The dimension of past logits must match the dimension of the current logits. If the past logits of certain samples
    have smaller dimension, they must be padded with NaN values.

    If a single weight is provided for the cross entropy loss, this criterion performs the following calculation:

    .. math::
        weight_1 * cross\_entropy\_loss(all\_current\_logits, all\_targets) +
        weight_2 * mse\_loss(current\_logits\_past\_logits, past\_logits)


    If two separate weights are provided for the cross entropy loss, this criterion performs the following calculation:

    .. math::
        weight_1 * cross\_entropy\_loss(current\_logits\_no\_past\_logits, targets\_no\_past\_logits) +
        weight_2 * cross\_entropy\_loss(current\_logits\_past\_logits, targets\_past\_logits) +
        weight_3 * mse\_loss(current\_logits\_past\_logits, past\_logits)

    One of these sets {`current_logits_no_past_logits`, `targets_no_past_logits`} or {`current_logits_past_logits`,
    `targets_past_logits`, `past_logits`} can contain all empty tensors but not both. If one of the two sets contains
    all empty tensors, the calculation of the loss discards those terms relying on those tensors.
    """
    def __init__(self, cross_entropy_weight: Union[float, Tuple[float, float]] = 1., mse_weight: float = 1.,
                 kwargs_cross_entropy: Optional[Dict[str, Any]] = None, kwargs_mse: Optional[Dict[str, Any]] = None):
        """
        Create a new CrossEntropyLossMSELossLogits
        :param cross_entropy_weight: weight for the cross entropy loss. If a tuple with two weights is provided, the
             cross entropy loss is computed separately for the samples that do not have past logits and the samples that
             have past logits. The cross entropy loss for the first type of samples is weighted by the first weight in
             the tuple while the same loss is weighted by the second weight in the tuple for the second type of samples.
             Default is 1.
        :param mse_weight: weight for the mean squared error loss. Default is 1.
        :param kwargs_cross_entropy: a dictionary containing the keyword arguments for :class:`nn.CrossEntropyLoss`. If
            None, no keyword arguments are passed. Default is None.
        :param kwargs_mse: a dictionary containing the keyword arguments for :class:`nn.MSELoss`. If
            None, no keyword arguments are passed. Default is None.
        """
        if isinstance(cross_entropy_weight, tuple):
            if not len(cross_entropy_weight) == 2:
                raise ValueError("`cross_entropy_weight` must be a single float number or a tuple with two float "
                                 "numbers")
        if kwargs_cross_entropy is None:
            kwargs_cross_entropy = {}
        if kwargs_mse is None:
            kwargs_mse = {}

        super().__init__()

        self.cross_entropy_weight: Union[float, Tuple[float, float]] = cross_entropy_weight
        """
        weight for the cross entropy loss. If a tuple with two weights, the cross entropy loss is computed separately
        for the samples that do not have past logits and the samples that have past logits. The cross entropy loss for
        the first type of samples is weighted by the first weight in the tuple while the same loss is weighted by the
        second weight in the tuple for the second type of samples.
        """

        self.mse_weight: float = mse_weight
        """
        weight for the mean squared error loss.
        """

        self._ce_loss: nn.CrossEntropyLoss = nn.CrossEntropyLoss(**kwargs_cross_entropy)
        """
        the cross entropy loss.
        """

        self._mse_loss: nn.MSELoss = nn.MSELoss(**kwargs_mse)
        """
        the mean squared error loss.
        """

    def forward(self, current_logits_no_past_logits: torch.Tensor, current_logits_past_logits: torch.Tensor,
                past_logits: torch.Tensor, targets_no_past_logits: torch.Tensor, targets_past_logits: torch.Tensor):
        """
        Compute the loss

        .note::
            One of these sets {`current_logits_no_past_logits`, `targets_no_past_logits`} or
            {`current_logits_past_logits`, `targets_past_logits`, `past_logits`} can contain just empty tensors but not
            both. If one of the two sets contains just empty tensors, the calculation of the loss discards those terms
            relying on those tensors.

        .note::
            If both {`current_logits_no_past_logits`, `targets_no_past_logits`}
            and {`current_logits_past_logits`, `targets_past_logits`, `past_logits`} contain non-empty tensors and
            `self.cross_entropy_weight` is a single float number, `targets_no_past_logits` and `targets_past_logits`
            must be both 1D tensors of integer numbers indicating the class the respective samples belong to or 2D
            tensors of probability vectors where each row has the same size as each logit vector.
            Otherwise, a :class:`ValueError` is raised.

        :param current_logits_no_past_logits: current logits of the samples that have no past logits
        :param current_logits_past_logits: current logits of the samples that have past logits.
        :param past_logits: past logits. The dimension of past logits must match the dimension of the current logits.
            If the past logits of certain samples have smaller dimension, they must be padded with NaN values.
        :param targets_no_past_logits: targets of the samples that have no past logits. It can be either a 1D tensor of
        integer numbers indicating the class the respective samples belong to or a 2D tensor of probability vectors
        where each row sums up to 1 and has the same size as each logit vector. A one-hot encoding vector is a special,
        degenerate case of a probability vector where only one class has a probability of 1, and all others are 0.
        :param targets_past_logits:  targets of the samples that have past logits. It can be either a 1D tensor of
        integer numbers indicating the class the respective samples belong to or a 2D tensor of probability vectors
        where each row sums up to 1 and has the same size as each logit vector. A one-hot encoding vector is a special,
        degenerate case of a probability vector where only one class has a probability of 1, and all others are 0.
        :return: the loss
        """
        # a list containing the current logits and targets of the samples with no past logits
        ls_no_past_logits = [current_logits_no_past_logits, targets_no_past_logits]
        # a list containing the current logits, past logits and targets of the samples with past logits
        ls_past_logits = [current_logits_past_logits, past_logits, targets_past_logits]

        # if all the tensors are empty
        if self._check_empty(*ls_no_past_logits, *ls_past_logits):
            raise ValueError("All the tensors provided are empty")
        # if only one between `ls_no_past_logits` and `ls_past_logits` contains all empty tensors
        elif self._check_empty(*ls_no_past_logits) ^ self._check_empty(*ls_past_logits):
            # if `ls_past_logits` contains all empty tensors
            if self._check_empty(*ls_past_logits):
                return self._loss_no_past_logits(current_logits_no_past_logits, targets_no_past_logits)
            else:  # if `ls_no_past_logits` contains all empty tensors
                return self._loss_past_logits(current_logits_past_logits, targets_past_logits, past_logits)
        # if all the tensors are not empty
        elif self._check_non_empty(*ls_no_past_logits, *ls_past_logits):
            return self._loss_all(current_logits_no_past_logits, current_logits_past_logits, past_logits,
                                  targets_no_past_logits, targets_past_logits)
        else:
            raise ValueError("The tensors provided have an invalid configuration. The sets "
                             "{`current_logits_no_past_logits`, `targets_no_past_logits`} and "
                             "{`current_logits_past_logits`, `targets_past_logits`, `past_logits`} must contain "
                             "exclusively non-empty tensors or only one between them can contain exclusively "
                             "empty tensors")

    def _loss_no_past_logits(self, current_logits_no_past_logits: torch.Tensor,
                             targets_no_past_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss using only the logits and targets of the samples that have no past logits
        :param current_logits_no_past_logits: current logits of the samples that have no past logits
        :param targets_no_past_logits: targets of the samples that have no past logits. It can be either a 1D tensor of
        integer numbers indicating the class the respective samples belong to or a 2D tensor of probability vectors
        where each row sums up to 1 and has the same size as each logit vector. A one-hot encoding vector is a special,
        degenerate case of a probability vector where only one class has a probability of 1, and all others are 0.
        :return: the loss
        """
        weight = self.cross_entropy_weight
        if isinstance(weight, tuple):
            weight = weight[0]
        return weight * self._ce_loss(current_logits_no_past_logits, targets_no_past_logits)

    def _loss_past_logits(self, current_logits_past_logits: torch.Tensor, targets_past_logits: torch.Tensor,
                          past_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss using only the logits, targets and past logits of the samples that have past logits
        :param current_logits_past_logits: current logits of the samples that have past logits.
        :param targets_past_logits: targets of the samples that have past logits. It can be either a 1D tensor of
        integer numbers indicating the class the respective samples belong to or a 2D tensor of probability vectors
        where each row sums up to 1 and has the same size as each logit vector. A one-hot encoding vector is a special,
        degenerate case of a probability vector where only one class has a probability of 1, and all others are 0.
        :param past_logits: past logits. The dimension of past logits must match the dimension of the current logits.
            If the past logits of certain samples have smaller dimension, they must be padded with NaN values.
        :return: the loss
        """
        if not current_logits_past_logits.shape[1] == past_logits.shape[1]:
            raise ValueError("The dimension of past logits does not match the dimension of the current logits")
        ce_weight = self.cross_entropy_weight
        if isinstance(ce_weight, tuple):
            ce_weight = ce_weight[1]
        mask = ~torch.isnan(past_logits)  # only select the non-NaN values
        return (ce_weight * self._ce_loss(current_logits_past_logits, targets_past_logits) +
                self.mse_weight * self._mse_loss(current_logits_past_logits[mask], past_logits[mask]))

    def _loss_all(self, current_logits_no_past_logits: torch.Tensor, current_logits_past_logits: torch.Tensor,
                  past_logits: torch.Tensor, targets_no_past_logits: torch.Tensor, targets_past_logits: torch.Tensor):
        """
        Compute the loss using the logits and targets of the samples that have no past logits, and the logits, targets
        and past logits of the samples that have past logits.

        .note::
            If `self.cross_entropy_weight` is a single float number, `targets_no_past_logits` and `targets_past_logits`
            must be both 1D tensors of integer numbers indicating the class the respective samples belong to or 2D
            tensors of probability vectors where each row has the same size as each logit vector.
            Otherwise, a :class:`ValueError` is raised.

        :param current_logits_no_past_logits: current logits of the samples that have no past logits
        :param current_logits_past_logits: current logits of the samples that have past logits.
        :param past_logits: past logits. The dimension of past logits must match the dimension of the current logits.
            If the past logits of certain samples have smaller dimension, they must be padded with NaN values.
        :param targets_no_past_logits: targets of the samples that have no past logits. It can be either a 1D tensor of
        integer numbers indicating the class the respective samples belong to or a 2D tensor of probability vectors
        where each row sums up to 1 and has the same size as each logit vector. A one-hot encoding vector is a special,
        degenerate case of a probability vector where only one class has a probability of 1, and all others are 0.
        :param targets_past_logits:  targets of the samples that have past logits. It can be either a 1D tensor of
        integer numbers indicating the class the respective samples belong to or a 2D tensor of probability vectors
        where each row sums up to 1 and has the same size as each logit vector. A one-hot encoding vector is a special,
        degenerate case of a probability vector where only one class has a probability of 1, and all others are 0.
        :return: the loss
        """
        if not current_logits_past_logits.shape[1] == past_logits.shape[1]:
            raise ValueError("The dimension of past logits does not match the dimension of the current logits")
        # if `self.cross_entropy_weight` is a single float number
        if not isinstance(self.cross_entropy_weight, tuple):
            # if the two target tensors do not have the same dimensionality
            if not targets_no_past_logits.dim() == targets_past_logits.dim():
                raise ValueError("If `self.cross_entropy_weight` is a single float number, `targets_no_past_logits` "
                                 "and `targets_past_logits` must be both 1D tensors of integer numbers indicating the "
                                 "class the respective samples belong to or 2D tensors of probability vectors "
                                 "where each row has the same size as each logit vector")
            # if the two target tensors are both 2d tensors, they must have the same number of columns
            if targets_no_past_logits.dim() == 2 and (not targets_no_past_logits.shape[1] ==
                                                      targets_past_logits.shape[1]):
                raise ValueError("If `self.cross_entropy_weight` is a single float number, and "
                                 "`targets_no_past_logits` and `targets_past_logits` are both 2D tensors, they must "
                                 "have the same number of columns")
        mask = ~torch.isnan(past_logits)  # only select the non-NaN values
        loss = self.mse_weight * self._mse_loss(current_logits_past_logits[mask], past_logits[mask])
        if isinstance(self.cross_entropy_weight, tuple):
            loss += (self.cross_entropy_weight[0] * self._ce_loss(current_logits_no_past_logits, targets_no_past_logits)
                     + self.cross_entropy_weight[1] * self._ce_loss(current_logits_past_logits, targets_past_logits))
        else:
            all_logits = torch.cat((current_logits_no_past_logits, current_logits_past_logits), dim=0)
            all_targets = torch.cat((targets_no_past_logits, targets_past_logits), dim=0)
            loss += self.cross_entropy_weight * self._ce_loss(all_logits, all_targets)
        return loss

    @staticmethod
    def _check_empty(*args):
        """
        Check whether all the tensors provided are empty
        :param args: some tensors
        :return: True if all tensors are empty. False, otherwise.
        """
        return all([arg.numel() == 0 for arg in args])

    @staticmethod
    def _check_non_empty(*args):
        """
        Check whether all the tensors provided are not empty
        :param args: some tensors
        :return: True if all tensors are not empty. False, otherwise.
        """
        return all([arg.numel() != 0 for arg in args])


class CrossEntropyKnowledgeDistilLoss(nn.Module):
    """
    This criterion combines the cross entropy loss between input logits and targets and the cross entropy
    loss between input logits and probability vectors computed from past logits. The latter is the typical knowledge
    distillation loss commonly used in Continual Learning. It was first proposed in https://arxiv.org/abs/1606.09282.

    Both the cross entropy loss and knowledge distillation loss terms are computed using :class:`nn.CrossEntropyLoss`.

    It takes the following inputs:
        - the current logits of the samples that have no past logits (`current_logits_no_past_logits`) and their
          targets (`targets_no_past_logits`)
        - the current logits of the samples that have past logits (`current_logits_past_logits`),
          their targets (`targets_past_logits`) and their past logits (`past_logits`)
    The targets can be either 1D tensors of integer numbers indicating the class the samples belong to or 2D tensors of
    probability vectors where each row sums up to 1 and has the same size as each logit vector.
    A one-hot encoding vector is a special, degenerate case of a probability vector where only one class has a
    probability of 1, and all others are 0.

    The dimension of past logits must match the dimension of the current logits. If the past logits of certain samples
    have smaller dimension, they must be padded with NaN values.

    If a single weight is provided for the cross entropy loss term, this criterion performs the following calculation:

    .. math::
        weight_1 * cross\_entropy\_loss(all\_current\_logits, all\_targets) +
        weight_2 * knowledge\_distil\_loss(current\_logits\_past\_logits, prob\_vec(past\_logits))


    If two separate weights are provided for the cross entropy loss term, this criterion performs the following
    calculation:

    .. math::
        weight_1 * cross\_entropy\_loss(current\_logits\_no\_past\_logits, targets\_no\_past\_logits) +
        weight_2 * cross\_entropy\_loss(current\_logits\_past\_logits, targets\_past\_logits) +
        weight_3 * knowledge\_distil\_loss(current\_logits\_past\_logits, prob\_vec(past\_logits))

    One of these sets {`current_logits_no_past_logits`, `targets_no_past_logits`} or {`current_logits_past_logits`,
    `targets_past_logits`, `past_logits`} can contain all empty tensors but not both. If one of the two sets contains
    all empty tensors, the calculation of the loss discards those terms relying on those tensors.
    """
    def __init__(self, cross_entropy_weight: Union[float, Tuple[float, float]] = 1.,
                 knowledge_distil_weight: float = 1., kwargs_cross_entropy: Optional[Dict[str, Any]] = None,
                 kwargs_knowledge_distil: Optional[Dict[str, Any]] = None):
        """
        Create a new CrossEntropyKnowledgeDistilLoss
        :param cross_entropy_weight: weight for the cross entropy loss. If a tuple with two weights is provided, the
             cross entropy loss is computed separately for the samples that do not have past logits and the samples that
             have past logits. The cross entropy loss for the first type of samples is weighted by the first weight in
             the tuple while the same loss is weighted by the second weight in the tuple for the second type of samples.
             Default is 1.
        :param knowledge_distil_weight: weight for the knowledge distillation loss. Default is 1.
        :param kwargs_cross_entropy: a dictionary containing the keyword arguments for the
            :class:`nn.CrossEntropyLoss` that is applied for the cross entropy loss. If None, no keyword arguments are
            passed. Default is None.
        :param kwargs_knowledge_distil: a dictionary containing the keyword arguments for the
            :class:`nn.CrossEntropyLoss` that is applied for the knowledge distillation loss. If None, no keyword
            arguments are passed. Default is None.
        """
        if isinstance(cross_entropy_weight, tuple):
            if not len(cross_entropy_weight) == 2:
                raise ValueError("`cross_entropy_weight` must be a single float number or a tuple with two float "
                                 "numbers")
        if kwargs_cross_entropy is None:
            kwargs_cross_entropy = {}
        if kwargs_knowledge_distil is None:
            kwargs_knowledge_distil = {}

        super().__init__()

        self.cross_entropy_weight: Union[float, Tuple[float, float]] = cross_entropy_weight
        """
        weight for the cross entropy loss. If a tuple with two weights, the cross entropy loss is computed separately
        for the samples that do not have past logits and the samples that have past logits. The cross entropy loss for
        the first type of samples is weighted by the first weight in the tuple while the same loss is weighted by the
        second weight in the tuple for the second type of samples.
        """

        self.knowledge_distil_weight: float = knowledge_distil_weight
        """
        weight for the knowledge distillation loss.
        """

        self._ce_loss: nn.CrossEntropyLoss = nn.CrossEntropyLoss(**kwargs_cross_entropy)
        """
        the cross entropy loss.
        """

        self._knowledge_distil_loss: nn.CrossEntropyLoss = nn.CrossEntropyLoss(**kwargs_knowledge_distil)
        """
        the knowledge distillation loss.
        """

    def forward(self, current_logits_no_past_logits: torch.Tensor, current_logits_past_logits: torch.Tensor,
                past_logits: torch.Tensor, targets_no_past_logits: torch.Tensor, targets_past_logits: torch.Tensor):
        """
        Compute the loss

        .note::
            One of these sets {`current_logits_no_past_logits`, `targets_no_past_logits`} or
            {`current_logits_past_logits`, `targets_past_logits`, `past_logits`} can contain just empty tensors but not
            both. If one of the two sets contains just empty tensors, the calculation of the loss discards those terms
            relying on those tensors.

        .note::
            If both {`current_logits_no_past_logits`, `targets_no_past_logits`}
            and {`current_logits_past_logits`, `targets_past_logits`, `past_logits`} contain non-empty tensors and
            `self.cross_entropy_weight` is a single float number, `targets_no_past_logits` and `targets_past_logits`
            must be both 1D tensors of integer numbers indicating the class the respective samples belong to or 2D
            tensors of probability vectors where each row has the same size as each logit vector.
            Otherwise, a :class:`ValueError` is raised.

        :param current_logits_no_past_logits: current logits of the samples that have no past logits
        :param current_logits_past_logits: current logits of the samples that have past logits.
        :param past_logits: past logits. The dimension of past logits must match the dimension of the current logits.
            If the past logits of certain samples have smaller dimension, they must be padded with NaN values.
        :param targets_no_past_logits: targets of the samples that have no past logits. It can be either a 1D tensor of
        integer numbers indicating the class the respective samples belong to or a 2D tensor of probability vectors
        where each row sums up to 1 and has the same size as each logit vector. A one-hot encoding vector is a special,
        degenerate case of a probability vector where only one class has a probability of 1, and all others are 0.
        :param targets_past_logits:  targets of the samples that have past logits. It can be either a 1D tensor of
        integer numbers indicating the class the respective samples belong to or a 2D tensor of probability vectors
        where each row sums up to 1 and has the same size as each logit vector. A one-hot encoding vector is a special,
        degenerate case of a probability vector where only one class has a probability of 1, and all others are 0.
        :return: the loss
        """
        # a list containing the current logits and targets of the samples with no past logits
        ls_no_past_logits = [current_logits_no_past_logits, targets_no_past_logits]
        # a list containing the current logits, past logits and targets of the samples with past logits
        ls_past_logits = [current_logits_past_logits, past_logits, targets_past_logits]

        # if all the tensors are empty
        if self._check_empty(*ls_no_past_logits, *ls_past_logits):
            raise ValueError("All the tensors provided are empty")
        # if only one between `ls_no_past_logits` and `ls_past_logits` contains all empty tensors
        elif self._check_empty(*ls_no_past_logits) ^ self._check_empty(*ls_past_logits):
            # if `ls_past_logits` contains all empty tensors
            if self._check_empty(*ls_past_logits):
                return self._loss_no_past_logits(current_logits_no_past_logits, targets_no_past_logits)
            else:  # if `ls_no_past_logits` contains all empty tensors
                return self._loss_past_logits(current_logits_past_logits, targets_past_logits, past_logits)
        # if all the tensors are not empty
        elif self._check_non_empty(*ls_no_past_logits, *ls_past_logits):
            return self._loss_all(current_logits_no_past_logits, current_logits_past_logits, past_logits,
                                  targets_no_past_logits, targets_past_logits)
        else:
            raise ValueError("The tensors provided have an invalid configuration. The sets "
                             "{`current_logits_no_past_logits`, `targets_no_past_logits`} and "
                             "{`current_logits_past_logits`, `targets_past_logits`, `past_logits`} must contain "
                             "exclusively non-empty tensors or only one between them can contain exclusively "
                             "empty tensors")

    def _loss_no_past_logits(self, current_logits_no_past_logits: torch.Tensor,
                             targets_no_past_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss using only the logits and targets of the samples that have no past logits
        :param current_logits_no_past_logits: current logits of the samples that have no past logits
        :param targets_no_past_logits: targets of the samples that have no past logits. It can be either a 1D tensor of
        integer numbers indicating the class the respective samples belong to or a 2D tensor of probability vectors
        where each row sums up to 1 and has the same size as each logit vector. A one-hot encoding vector is a special,
        degenerate case of a probability vector where only one class has a probability of 1, and all others are 0.
        :return: the loss
        """
        weight = self.cross_entropy_weight
        if isinstance(weight, tuple):
            weight = weight[0]
        return weight * self._ce_loss(current_logits_no_past_logits, targets_no_past_logits)

    def _loss_past_logits(self, current_logits_past_logits: torch.Tensor, targets_past_logits: torch.Tensor,
                          past_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss using only the logits, targets and past logits of the samples that have past logits
        :param current_logits_past_logits: current logits of the samples that have past logits.
        :param targets_past_logits: targets of the samples that have past logits. It can be either a 1D tensor of
        integer numbers indicating the class the respective samples belong to or a 2D tensor of probability vectors
        where each row sums up to 1 and has the same size as each logit vector. A one-hot encoding vector is a special,
        degenerate case of a probability vector where only one class has a probability of 1, and all others are 0.
        :param past_logits: past logits. The dimension of past logits must match the dimension of the current logits.
            If the past logits of certain samples have smaller dimension, they must be padded with NaN values.
        :return: the loss
        """
        if not current_logits_past_logits.shape[1] == past_logits.shape[1]:
            raise ValueError("The dimension of past logits does not match the dimension of the current logits")
        ce_weight = self.cross_entropy_weight
        if isinstance(ce_weight, tuple):
            ce_weight = ce_weight[1]
        mask = torch.isnan(past_logits)  # select the NaN values
        past_logits = past_logits.masked_fill(mask, -1e9)  # mask the NaN values with a very large negative number
        # convert the past logits into probabilities by applying softmax. The values with -1e9 have 0 probability
        past_probs = past_logits.softmax(dim=1)
        # mask irrelevant logits with a very large negative number so that they are effectively ignored
        masked_current_logits_past_logits = current_logits_past_logits.masked_fill(mask, -1e9)
        return (ce_weight * self._ce_loss(current_logits_past_logits, targets_past_logits) +
                self.knowledge_distil_weight * self._knowledge_distil_loss(masked_current_logits_past_logits,
                                                                           past_probs))

    def _loss_all(self, current_logits_no_past_logits: torch.Tensor, current_logits_past_logits: torch.Tensor,
                  past_logits: torch.Tensor, targets_no_past_logits: torch.Tensor, targets_past_logits: torch.Tensor):
        """
        Compute the loss using the logits and targets of the samples that have no past logits, and the logits, targets
        and past logits of the samples that have past logits.

        .note::
            If `self.cross_entropy_weight` is a single float number, `targets_no_past_logits` and `targets_past_logits`
            must be both 1D tensors of integer numbers indicating the class the respective samples belong to or 2D
            tensors of probability vectors where each row has the same size as each logit vector.
            Otherwise, a :class:`ValueError` is raised.

        :param current_logits_no_past_logits: current logits of the samples that have no past logits
        :param current_logits_past_logits: current logits of the samples that have past logits.
        :param past_logits: past logits. The dimension of past logits must match the dimension of the current logits.
            If the past logits of certain samples have smaller dimension, they must be padded with NaN values.
        :param targets_no_past_logits: targets of the samples that have no past logits. It can be either a 1D tensor of
        integer numbers indicating the class the respective samples belong to or a 2D tensor of probability vectors
        where each row sums up to 1 and has the same size as each logit vector. A one-hot encoding vector is a special,
        degenerate case of a probability vector where only one class has a probability of 1, and all others are 0.
        :param targets_past_logits:  targets of the samples that have past logits. It can be either a 1D tensor of
        integer numbers indicating the class the respective samples belong to or a 2D tensor of probability vectors
        where each row sums up to 1 and has the same size as each logit vector. A one-hot encoding vector is a special,
        degenerate case of a probability vector where only one class has a probability of 1, and all others are 0.
        :return: the loss
        """
        if not current_logits_past_logits.shape[1] == past_logits.shape[1]:
            raise ValueError("The dimension of past logits does not match the dimension of the current logits")
        # if `self.cross_entropy_weight` is a single float number
        if not isinstance(self.cross_entropy_weight, tuple):
            # if the two target tensors do not have the same dimensionality
            if not targets_no_past_logits.dim() == targets_past_logits.dim():
                raise ValueError("If `self.cross_entropy_weight` is a single float number, `targets_no_past_logits` "
                                 "and `targets_past_logits` must be both 1D tensors of integer numbers indicating the "
                                 "class the respective samples belong to or 2D tensors of probability vectors "
                                 "where each row has the same size as each logit vector")
            # if the two target tensors are both 2d tensors, they must have the same number of columns
            if targets_no_past_logits.dim() == 2 and (not targets_no_past_logits.shape[1] ==
                                                      targets_past_logits.shape[1]):
                raise ValueError("If `self.cross_entropy_weight` is a single float number, and "
                                 "`targets_no_past_logits` and `targets_past_logits` are both 2D tensors, they must "
                                 "have the same number of columns")
        mask = torch.isnan(past_logits)  # select the NaN values
        past_logits = past_logits.masked_fill(mask, -1e9)  # mask the NaN values with a very large negative number
        # convert the past logits into probabilities by applying softmax. The values with -1e9 have 0 probability
        past_probs = past_logits.softmax(dim=1)
        # mask irrelevant logits with a very large negative number so that they are effectively ignored
        masked_current_logits_past_logits = current_logits_past_logits.masked_fill(mask, -1e9)
        loss = self.knowledge_distil_weight * self._knowledge_distil_loss(masked_current_logits_past_logits, past_probs)
        if isinstance(self.cross_entropy_weight, tuple):
            loss += (self.cross_entropy_weight[0] * self._ce_loss(current_logits_no_past_logits, targets_no_past_logits)
                     + self.cross_entropy_weight[1] * self._ce_loss(current_logits_past_logits, targets_past_logits))
        else:
            all_logits = torch.cat((current_logits_no_past_logits, current_logits_past_logits), dim=0)
            all_targets = torch.cat((targets_no_past_logits, targets_past_logits), dim=0)
            loss += self.cross_entropy_weight * self._ce_loss(all_logits, all_targets)
        return loss

    @staticmethod
    def _check_empty(*args):
        """
        Check whether all the tensors provided are empty
        :param args: some tensors
        :return: True if all tensors are empty. False, otherwise.
        """
        return all([arg.numel() == 0 for arg in args])

    @staticmethod
    def _check_non_empty(*args):
        """
        Check whether all the tensors provided are not empty
        :param args: some tensors
        :return: True if all tensors are not empty. False, otherwise.
        """
        return all([arg.numel() != 0 for arg in args])
