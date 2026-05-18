"""
This module includes a set of gradient plugin metrics and respective helper methods
"""
from __future__ import annotations
from typing import Tuple, List, Optional, Union, TYPE_CHECKING
from avalanche.evaluation import Metric, GenericPluginMetric
import torch

if TYPE_CHECKING:
    from training.templates import DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate


class GradientLossWRTInputLastLayer(Metric[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """
    The standalone metric for keeping track of the gradient of the loss w.r.t. the input to the last layer for the
    samples in a dataset. This metric assumes that the last layer of a model is a Softmax layer and that the loss used
    is the standard cross entropy loss used in multi-class classifications problems. Under these assumptions, for a
    given sample, the gradient of the loss w.r.t the input to the last layer is Y_hat - Y, where Y_hat is the vector of
    softmax probabilities and Y is the one-hot encoding vector of the true class.

    The update method receives a tensor of predicted logits (these must be raw logits that did not pass through the
    softmax layer) and true target labels, uses them for computing the gradient of the loss w.r.t. the input to the
    last layer and then appends these gradients into the `gradients` attribute and the corresponding target labels into
    the `targets` attribute. This method also optionally accepts a tensor of dataset indices. The index of a given
    sample is the index to be used to retrieve the given sample from the underlying dataset. If the tensor of dataset
    indices is not provided, the tensor of indices is set to a tensor containing -1s as indices by default.

    The result method returns the gradients computed so far, their respective target labels and indices. The order of
    the gradients, the respective target labels and indices reflects the order of the calls to the update method.

    The reset method will bring the metric to its initial state. By default, this metric in its initial state will
    return a tuple with three empty tensors.
    """
    def __init__(self, device: Optional[Union[str, torch.device]] = None):
        """
        Create a new GradientLossWRTInputLastLayer
        :param device: device where to store the collected tensors. If None, the collected tensors are not moved from
            their original devices. Default is None.
        """
        self.gradients: List[torch.Tensor] = []
        """
        a list of tensors encoding the gradient of the loss w.r.t. the input to the last layer for several samples
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

    def update(self, predicted_y: torch.Tensor, true_y: torch.Tensor, num_classes: int,
               indices: Optional[torch.Tensor] = None) -> None:
        """
        Compute and add the gradients of the loss w.r.t the input to the last layer and their respective targets and
        indices. If `self._device` is not None, the tensors are moved to `self._device`; otherwise, they are not
        moved from their original devices.
        :param predicted_y: the predicted raw logits, they must be the raw logits before passing through the softmax
            layer
        :param true_y: the ground truth target labels
        :param num_classes: number of classes
        :param indices: (optional) the dataset indices. Each index is the index to be used to retrieve the given sample
            from the underlying dataset. If None, the indices tensor is set to a tensor containing -1s.
            Default is None.
        """
        if self._device is not None:
            predicted_y = predicted_y.to(self._device)
            true_y = true_y.to(self._device)
            indices = indices if indices is None else indices.to(self._device)
        gradients = predicted_y.softmax(dim=1) - torch.eye(num_classes, device=predicted_y.device)[true_y]
        self.gradients.append(gradients)
        self.targets.append(true_y)
        if indices is None:
            indices = -torch.ones(len(true_y), dtype=torch.int64, device=predicted_y.device)
        self.indices.append(indices)

    def reset(self) -> None:
        """
        Reset the metric to its initial state. The `gradients`, `targets` and `indices` attributes are emptied.
        :return: None
        """
        self.gradients = []
        self.targets = []
        self.indices = []

    def result(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieve the gradients computed so far and their respective targets and indices
        :return: a tensor of gradients, a tensor of respective targets, a tensor of indices.
            If `self._device` is not None, the tensors are allocated to `self._device`;
            otherwise, they are not moved from their original devices. If no tensors have been collected so far, the
            returned empty tensors will be allocated to `self._device` if it is not None; otherwise, they are allocated
            to `cpu`.
        """
        # if the gradients, targets and indices attributes are empty, just return three empy tensors
        if len(self.gradients) == 0:
            device = torch.device("cpu") if self._device is None else self._device
            return (torch.tensor([], dtype=torch.float32, device=device),
                    torch.tensor([], dtype=torch.int64, device=device),
                    torch.tensor([], dtype=torch.int64, device=device))
        return torch.cat(self.gradients, dim=0), torch.cat(self.targets, dim=0), torch.cat(self.indices, dim=0)


class GradientLossWRTWeightsBiasesClassificationHead(Metric[Union[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
]]):
    """
    The standalone metric for keeping track of the gradient of the loss w.r.t. the weights and biases in the
    classification head for the samples in a dataset.
    This metric assumes that the last layer of a model is a classification head with softmax activation function and
    that the loss used is the standard multi-class cross entropy loss. Under these assumptions, for a given sample,
    the gradient of the loss w.r.t the weights in the classification head is (Y_hat - Y) H^T, where Y_hat is the vector
    of softmax probabilities, Y is the one-hot encoding vector of the true class and H is the embedding vector of the
    given sample, also known as feature vector, being fed into the classification head. Note that (Y_hat - Y) H^T is the
    outer product between two vectors resulting in a matrix.  Also, under these assumptions, for a given sample, the
    gradient of the loss w.r.t the biases in the classification head is Y_hat - Y.

    The update method receives a tensor of predicted logits (these must be raw logits that did not pass through the
    softmax layer), a tensor of true target labels and a tensor of embeddings (feature vectors). It computes the
    gradient of the loss w.r.t. the biases in the classification head and appends it together with the embeddings into
    the `gradients` attribute. The corresponding target labels are appended into the `targets` attribute. This method
    also optionally accepts a tensor of dataset indices. The index of a given sample is the index to be used to
    retrieve the given sample from the underlying dataset. If the tensor of dataset indices is not provided, the tensor
    of indices is set to a tensor containing -1s as indices by default. The dataset indices are appended into the
    `indices` attribute.

    The result method computes and returns the gradients of the loss w.r.t. the weights and biases in the
    classification head collected so far, their respective target labels and indices. The order of
    the gradients, the respective target labels and indices reflects the order of the calls to the update method.

    The reset method will bring the metric to its initial state. By default, this metric in its initial state will
    return a tuple with three or five empty tensors.
    """

    def __init__(self, device: Optional[Union[str, torch.device]] = None):
        """
        Create a new GradientLossWRTWeightsBiasesClassificationHead
        :param device: device where to store the collected tensors. If None, the collected tensors are not moved from
            their original devices. Default is None.
        """

        self.gradients: List[Tuple[torch.Tensor, torch.Tensor]] = []
        """
        a list of tuples. The first element of each tuple is a tensor encoding the gradient of the loss w.r.t the
        biases in the classification head for several samples. The second element of each tuple is a tensor encoding
        the embeddings, also known as feature vectors, for the respective samples. Note that the
        gradient of the loss w.r.t the biases in the classification head is equivalent to the gradient of the loss
        w.r.t the raw score outputs (raw logits).
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

    def update(self, predicted_y: torch.Tensor, true_y: torch.Tensor, embeddings: torch.Tensor, num_classes: int,
               indices: Optional[torch.Tensor] = None) -> None:
        """
        Compute the gradients of the loss w.r.t the biases in the classification head and append them together with the
        embeddings into the `gradients` attribute. Additionally, append the targets and indices into the `targets` and
        `indices` attributes, respectively. If `self._device` is not None, the tensors are moved to `self._device`;
        otherwise, they are not moved from their original devices.
        :param predicted_y: the predicted raw logits, they must be the raw logits before passing through the softmax
            layer
        :param true_y: the ground truth target labels
        :param embeddings: the embeddings, also known as feature vectors, being fed into the classification head
        :param num_classes: number of classes
        :param indices: (optional) the dataset indices. Each index is the index to be used to retrieve the given sample
            from the underlying dataset. If None, the indices tensor is set to a tensor containing -1s.
            Default is None.
        """
        if self._device is not None:
            predicted_y = predicted_y.to(self._device)
            embeddings = embeddings.to(self._device)
            true_y = true_y.to(self._device)
            indices = indices if indices is None else indices.to(self._device)
        gradients_biases = predicted_y.softmax(dim=1) - torch.eye(num_classes, device=predicted_y.device)[true_y]
        self.gradients.append((gradients_biases, embeddings))
        self.targets.append(true_y)
        if indices is None:
            indices = -torch.ones(len(true_y), dtype=torch.int64, device=predicted_y.device)
        self.indices.append(indices)

    def reset(self) -> None:
        """
        Reset the metric to its initial state. The `gradients`, `targets` and `indices` attributes are emptied.
        :return: None
        """
        self.gradients = []
        self.targets = []
        self.indices = []

    def result(self, retrieve_grad_biases_embeddings: bool = False) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ]:
        """
        Retrieve the gradients collected so far and their respective targets and indices
        :param retrieve_grad_biases_embeddings: (optional) whether to additionally retrieve the gradients w.r.t the
            biases and the respective embeddings. Default is False.
        :return: a tensor of gradients w.r.t the weights and biases in the classification head, a tensor of respective
            targets, a tensor of respective indices if `retrieve_grad_biases_embeddings` is False. Otherwise,
            a tensor of gradients w.r.t the weights and biases in the classification head, a tensor of respective
            targets, a tensor of respective indices, a tensor of respective gradients w.r.t the biases in the
            classification head, a tensor of respective embeddings. If `self._device` is not None, the tensors are
            allocated to `self._device`; otherwise, they are not moved from their original devices.
            If no tensors have been collected so far, the returned empty tensors will be allocated to `self._device` if
            it is not None; otherwise, they are allocated to `cpu`.
        """
        # if nothing was collected, just return three or five empy tensors according to
        # the value of `retrieve_grad_biases_embeddings`
        if len(self.gradients) == 0:
            device = torch.device("cpu") if self._device is None else self._device
            out = (torch.tensor([], dtype=torch.float32, device=device),
                   torch.tensor([], dtype=torch.int64, device=device),
                   torch.tensor([], dtype=torch.int64, device=device),
                   torch.tensor([], dtype=torch.float32, device=device),
                   torch.tensor([], dtype=torch.float32, device=device))
            if not retrieve_grad_biases_embeddings:  # only get the first three elements
                out = out[:3]
            return out

        gradients_weights_biases = []
        # compute the gradients wrt the weights and biases from each tuple in `self.gradients`
        for gradients_biases, embeddings in self.gradients:
            gradients_weights = torch.bmm(gradients_biases.unsqueeze(2), embeddings.unsqueeze(1))
            gradients_weights = gradients_weights.reshape(len(gradients_biases), -1)
            gradients_weights_biases.append(torch.cat((gradients_weights, gradients_biases), dim=1))

        if not retrieve_grad_biases_embeddings:
            return (torch.cat(gradients_weights_biases, dim=0), torch.cat(self.targets, dim=0),
                    torch.cat(self.indices, dim=0))
        else:
            return (torch.cat(gradients_weights_biases, dim=0), torch.cat(self.targets, dim=0),
                    torch.cat(self.indices, dim=0),
                    torch.cat([grad_biases for grad_biases, _ in self.gradients], dim=0),
                    torch.cat([embeds for _, embeds in self.gradients], dim=0))


class GradientCrossEntropyLossMSELossWRTWeightsBiasesClassificationHead(Metric[Union[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
]]):
    """
    The standalone metric for keeping track of the gradient of the loss w.r.t. the weights and biases in the
    classification head for the samples in a dataset. This metric assumes that the last layer of a model is a
    classification head with softmax activation function. The loss is the sum of the standard multi-class cross
    entropy loss between current logits and targets and the mean squared error loss between past and current logits:

    .. math::
        cross\_entropy\_loss(current\_logits, targets) + mse\_loss(current\_logits, past\_logits)

    Under these assumptions, for a given sample, the gradient of the multi-class cross entropy loss w.r.t the weights
    in the classification head is (Y_hat - Y) H^T while the gradient of the mean squared error loss w.r.t the weights
    in the classification head is 2/c (Z - T) H^T, where Y_hat is the vector of softmax probabilities, Y is the one-hot
    encoding vector of the true class, H is the embedding vector of the given sample , also known as feature vector,
    being fed into the classification head, Z is the current logits vector output of the classification head prior to
    apply the softmax function, T is the past logits vector output and c is the dimensionality of T. If T exists and c
    is smaller than c', the dimensionality of Z, then Z-T is equal to (Z[:c] - T) || [0]*(c'-c),
    where || is the concatenation operator and [0]*(c'-c) is a vector containing c'-c zeros.
    Note that both (Y_hat - Y) H^T and 2/c (Z - T) H^T are the outer product between two vectors resulting in a matrix.
    Therefore, for a given sample, the gradient of the total loss w.r.t the weights in the classification head is
    (Y_hat - Y) H^T + 2/c (Z - T) H^T = [(Y_hat - Y) + 2/c (Z - T)]H^T.
    Also, under these assumptions, for a given sample, the gradient of the multi-class cross entropy loss w.r.t the
    biases in the classification head is Y_hat - Y while the gradient of the mean squared error loss w.r.t the biases
    in the classification head is 2/c (Z - T). Therefore, for a given sample, the gradient of the total loss w.r.t the
    biases in the classification head is (Y_hat - Y) + 2/c (Z - T).
    Note that for both the calculations of the gradient of the mean squared error loss w.r.t the weights and biases in
    the classification head, if the past logit vector T does not exist, then 2/c (Z-T) is set equal to [0]*c'.
    Consequently, the gradients of the mean squared error loss w.r.t the weights and biases in the classification head
    are a zero matrix and a zero vector, respectively, as if the mean squared
    error loss term for the given sample were discarded or if the past logit vector were equivalent to the current one.

    The update method receives a tensor of predicted logits (these must be raw logits that did not pass through
    the softmax layer), a tensor of true target labels, a tensor of embeddings (feature vectors), a boolean tensor
    indicating whether each logits vector in the tensor of predicted logits has a respective past logits vector
    and, finally, a tensor of past logits containing as many past logits vectors as the number of True elements in the
    boolean tensor and in the same order. If the size of past logits vectors is smaller than the size of the
    predicted logits, they must be padded with NaN values to match the size of the predicted logits.
    It computes the gradient of the multi-class cross entropy loss w.r.t. the biases in the classification head,
    the gradient of the mean squared error loss w.r.t. the biases in the classification head and appends them together
    with the embeddings into the `gradients` attribute. The corresponding target labels are appended into the `targets`
    attribute.
    This method also optionally accepts a tensor of dataset indices. The index of a given sample is the index to be used
    to retrieve the given sample from the underlying dataset. If the tensor of dataset indices is not provided,
    the tensor of indices is set to a tensor containing -1s as indices by default. The dataset indices are appended into
    the `indices` attribute.

    The result method computes and returns the gradients of the loss w.r.t. the weights and biases in the
    classification head collected so far, their respective target labels and indices. The order of
    the gradients, the respective target labels and indices reflects the order of the calls to the update method.

    The reset method will bring the metric to its initial state. By default, this metric in its initial state will
    return a tuple with three or six empty tensors.
    """
    def __init__(self, device: Optional[Union[str, torch.device]] = None):
        """
        Create a new GradientCrossEntropyLossMSELossWRTWeightsBiasesClassificationHead
        :param device: device where to store the collected tensors. If None, the collected tensors are not moved from
            their original devices. Default is None.
        """

        self.gradients: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        """
        a list of tuples. The first element of each tuple is a tensor encoding the gradient of the multi-class cross
        entropy loss w.r.t the biases in the classification head for several samples. The second element of each tuple
        is a tensor encoding the gradient of the mean squared error loss w.r.t the biases in the classification head for
        the respective samples. The third element of each tuple is a tensor encoding the embeddings, also known as
        feature vectors, for the respective samples. Note that the gradient of the multi-class cross entropy loss w.r.t
        the biases in the classification head is equivalent to the gradient of the multi-class cross entropy loss
        w.r.t the raw score outputs (raw logits). Also, the gradient of the mean squared error loss w.r.t the biases in
        the classification head is equivalent to the gradient of the mean squared error loss w.r.t the raw score outputs
        (raw logits).
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

    def update(self, predicted_y: torch.Tensor, true_y: torch.Tensor, embeddings: torch.Tensor, num_classes: int,
               has_past_logits: torch.Tensor, past_logits: torch.Tensor,
               indices: Optional[torch.Tensor] = None) -> None:
        """
        Compute the gradient of the multi-class cross entropy loss w.r.t. the biases in the classification head,
        the gradient of the mean squared error loss w.r.t. the biases in the classification head and append them
        together with the embeddings into the `gradients` attribute. Additionally, append the targets and indices into
        the `targets` and `indices` attributes, respectively. If `self._device` is not None, the tensors are moved to
        `self._device`; otherwise, they are not moved from their original devices.
        :param predicted_y: the predicted raw logits, they must be the raw logits before passing through the softmax
            layer
        :param true_y: the ground truth target labels
        :param embeddings: the embeddings, also known as feature vectors, being fed into the classification head
        :param num_classes: number of classes
        :param has_past_logits: boolean values indicating whether each logits vector in `predicted_y` has a respective
            past logits vector in `past_logits`. A :class:`ValueError` is raised if the number of elements in this
            tensor is not equal to the number of rows in `predicted_y`.
        :param past_logits: the past logits vectors. This tensor *must* store as many past logits vectors as the number
            of True elements in `has_past_logits` and in the same order. Otherwise, a :class:`ValueError` is raised.
            Additionally, if the size of past logits vectors is smaller than the size of the predicted logits in
            `predicted_y`, they must be padded with NaN values to match the size of the predicted logits.
            A :class:`ValueError` is raised if the size of the past logits vectors does not match the size of the logits
            vectors in `predicted_y`.
        :param indices: (optional) the dataset indices. Each index is the index to be used to retrieve the given sample
            from the underlying dataset. If None, the indices tensor is set to a tensor containing -1s.
            Default is None.
        """
        if not len(has_past_logits) == len(predicted_y):
            raise ValueError("The number of elements in `has_past_logits` must be equal to the number of rows in "
                             "`predicted_y`")
        if bool(torch.any(has_past_logits)):
            if not torch.sum(has_past_logits) == len(past_logits):
                raise ValueError("`past_logits` must store as many past logits vectors as the number of True elements "
                                 "in `has_past_logits`")
            if not past_logits.shape[1] == predicted_y.shape[1]:
                raise ValueError("The size of the past logits vectors in `past_logits` must match the size of the "
                                 "logits vectors in `predicted_y`")
        else:
            if not past_logits.numel() == 0:
                raise ValueError("`past_logits` must be empty when there is no True value in `has_past_logits`")
        # the gradient of the multi-class cross entropy loss wrt the biases in the classification head
        ce_loss_gradients_biases = predicted_y.softmax(dim=1) - torch.eye(num_classes, device=predicted_y.device)[true_y]
        # the gradient of the mean squared error loss wrt the biases in the classification head. It is initialised as
        # a zero matrix
        mse_loss_gradients_biases = torch.zeros_like(ce_loss_gradients_biases)
        # if there exists one or multiple logits vectors in `predicted_y` that have a respective past logits vector in
        # `past_logits`, then update `mse_loss_gradients_biases`
        if bool(torch.any(has_past_logits)):
            # the difference between the predicted logits and the past logits. Differences between a number and a NaN
            # value always return NaN
            diff = predicted_y[has_past_logits] - past_logits
            # the number of non-NaN values in each row of `diff`
            not_NaN = torch.sum(~torch.isnan(diff), dim=1, keepdim=True)
            # the actual gradients of the mse loss wrt the biases
            actual_grads = (2 / not_NaN) * diff
            # replace NaN values with 0
            actual_grads = torch.nan_to_num(actual_grads, nan=0.0)
            # move the actual gradients into `mse_loss_gradients_biases`
            mse_loss_gradients_biases[has_past_logits] = actual_grads

        if self._device is not None:
            ce_loss_gradients_biases = ce_loss_gradients_biases.to(self._device)
            mse_loss_gradients_biases = mse_loss_gradients_biases.to(self._device)
            embeddings = embeddings.to(self._device)
            true_y = true_y.to(self._device)
            indices = indices if indices is None else indices.to(self._device)
        self.gradients.append((ce_loss_gradients_biases, mse_loss_gradients_biases, embeddings))
        self.targets.append(true_y)
        if indices is None:
            indices = -torch.ones(len(true_y), dtype=torch.int64, device=ce_loss_gradients_biases.device)
        self.indices.append(indices)

    def reset(self) -> None:
        """
        Reset the metric to its initial state. The `gradients`, `targets` and `indices` attributes are emptied.
        :return: None
        """
        self.gradients = []
        self.targets = []
        self.indices = []

    def result(self, retrieve_grad_biases_embeddings: bool = False) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ]:
        """
        Retrieve the gradients collected so far and their respective targets and indices
        :param retrieve_grad_biases_embeddings: (optional) whether to additionally retrieve the gradients of the
            multi-class cross entropy loss w.r.t the biases, the gradients of the mean squared error loss w.r.t the
            biases and the respective embeddings. Default is False.
        :return: a tensor of gradients of the total loss w.r.t the weights and biases in the classification head,
            a tensor of respective targets and a tensor of respective indices if `retrieve_grad_biases_embeddings` is
            False. Otherwise, a tensor of gradients of the total loss w.r.t the weights and biases in the classification
            head, a tensor of respective targets, a tensor of respective indices, a tensor of respective gradients of
            the multi-class cross entropy loss w.r.t the biases in the classification head, a tensor of respective
            gradients of the mean squared error loss w.r.t the biases in the classification head and a tensor of
            respective embeddings. If `self._device` is not None, the tensors are allocated to `self._device`;
            otherwise, they are not moved from their original devices.
            If no tensors have been collected so far, the returned empty tensors will be allocated to `self._device` if
            it is not None; otherwise, they are allocated to `cpu`.
        """
        # if nothing was collected, just return three or six empy tensors according to
        # the value of `retrieve_grad_biases_embeddings`
        if len(self.gradients) == 0:
            device = torch.device("cpu") if self._device is None else self._device
            out = (torch.tensor([], dtype=torch.float32, device=device),
                   torch.tensor([], dtype=torch.int64, device=device),
                   torch.tensor([], dtype=torch.int64, device=device),
                   torch.tensor([], dtype=torch.float32, device=device),
                   torch.tensor([], dtype=torch.float32, device=device),
                   torch.tensor([], dtype=torch.float32, device=device))
            if not retrieve_grad_biases_embeddings:  # only get the first three elements
                out = out[:3]
            return out

        gradients_weights_biases = []
        # compute the gradients wrt the weights and biases from each tuple in `self.gradients`
        for ce_loss_gradients_biases, mse_loss_gradients_biases, embeddings in self.gradients:
            gradients_biases = ce_loss_gradients_biases + mse_loss_gradients_biases
            gradients_weights = torch.bmm(gradients_biases.unsqueeze(2), embeddings.unsqueeze(1))
            gradients_weights = gradients_weights.reshape(len(gradients_biases), -1)
            gradients_weights_biases.append(torch.cat((gradients_weights, gradients_biases), dim=1))

        if not retrieve_grad_biases_embeddings:
            return (torch.cat(gradients_weights_biases, dim=0), torch.cat(self.targets, dim=0),
                    torch.cat(self.indices, dim=0))
        else:
            return (torch.cat(gradients_weights_biases, dim=0), torch.cat(self.targets, dim=0),
                    torch.cat(self.indices, dim=0),
                    torch.cat([ce_loss_grad_biases for ce_loss_grad_biases, _, _ in self.gradients], dim=0),
                    torch.cat([mse_loss_grad_biases for _, mse_loss_grad_biases, _ in self.gradients], dim=0),
                    torch.cat([embeds for _, _, embeds in self.gradients], dim=0))


class GradientLossWRTInputLastLayerPluginMetric(GenericPluginMetric[Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    GradientLossWRTInputLastLayer]):
    """
    This plugin metric collects the gradients w.r.t. the input to the last layer over the samples in a given
    training experience. At the end of a training experience, this plugin outputs the collected gradients along with
    the associated targets and indices. The index of a gradient is the index of the respective training sample to be
    used to retrieve the given sample from the underlying training dataset. This plugin metric assumes that the
    DataAttribute with name `dataset_indices` contains the indices of each sample in the dataset of the current training
    experience. This plugin also assumes that the latter DataAttribute has the property use_in_getitem=True.
    If the latter DataAttribute is not in the dataset of the current training experience or it is in the dataset but its
    property use_in_getitem=False , the index of each gradient is set to -1.

    The order of the gradients, the targets and indices reflects the order used to iterate over the dataset of the
    training experience.

    This plugin metric assumes that the last layer of a model is a Softmax layer and that the loss used
    is the standard cross entropy loss used in multi-class classifications problems. Under these assumptions, for a
    given sample, the gradient of the loss w.r.t the input to the last layer is Y_hat - Y, where Y_hat is the vector of
    softmax probabilities and Y is the one-hot encoding vector of the true class.

    This plugin metric *only* works at eval time and on training experiences.

    .note::
        This plugin metric does not work on test or val experiences

    .warning::
        This plugin metric *only* works when there is *only* one subnetwork
    """
    def __init__(self):
        super().__init__(GradientLossWRTInputLastLayer(), reset_at="experience", emit_at="experience", mode="eval")

    def reset(self) -> None:
        self._metric.reset()

    def result(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._metric.result()

    def update(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate):
        curr_sub_id = strategy.curr_sub_id  # store the current state of the strategy's subnetwork id
        subs_id = list(strategy.mb_output.keys())
        if len(subs_id) > 1:
            raise RuntimeError("This plugin must be used only when there is one subnetwork")
        strategy.curr_sub_id = subs_id[0]
        num_classes = len(strategy.model.subnetworks_classes[strategy.curr_sub_id])
        # if the DataAttribute "dataset_indices" is present and its property use_in_getitem=True, use that to retrieve
        # the dataset indices. Otherwise, just set the indices to -1 by default
        indices = None
        if "dataset_indices" in strategy.use_in_getitem_indices.keys():
            indices = strategy.mb_dataset_indices
        self._metric.update(strategy.mb_output_incremental_classifier, strategy.mb_y_incremental_classifier,
                            num_classes, indices=indices)
        strategy.curr_sub_id = curr_sub_id  # re-store the state of the strategy's subnetwork id

    def before_eval_exp(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate):
        if strategy.experience.origin_stream.name == "train":
            super().before_eval_exp(strategy)

    def after_eval_exp(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate):
        if strategy.experience.origin_stream.name == "train":
            return super().after_eval_exp(strategy)

    def after_eval_iteration(self, strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate):
        if strategy.experience.origin_stream.name == "train":
            super().after_eval_iteration(strategy)

    def __str__(self):
        return "GradientLossWRTInputLastLayer"


def gradient_metrics(grad_wrt_input_last_layer: bool = False) -> List:
    """
    Helper method that can be used to obtain the desired set of gradient plugin metrics.
    :param grad_wrt_input_last_layer: if True, the :class:`GradientLossWRTInputLastLayerPluginMetric` is added.
        Default is False.
    :return: A list of gradient plugin metrics
    """
    metrics: List = []
    if grad_wrt_input_last_layer:
        metrics.append(GradientLossWRTInputLastLayerPluginMetric())
    return metrics


