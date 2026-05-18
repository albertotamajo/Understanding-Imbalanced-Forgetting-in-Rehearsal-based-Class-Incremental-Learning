"""
This module includes a set of forgetting events metrics
"""
from typing import Tuple, Dict

import numpy as np
from avalanche.evaluation import Metric
import torch


class ForgettingEvents(Metric[Tuple[np.ndarray, np.ndarray]]):
    """
    The standalone metric for keeping track of the number of forgetting events for the samples in a dataset.
    A forgetting event occurs when a sample transitions from being correctly classified to being misclassified.
    This definition of a forgetting event is taken from
    "An Empirical Study of Example Forgetting during Deep Neural Network Learning".

    The update method receives a tensor of predictions (they could be either raw logits or softmax probabilities),
    a tensor of true target labels, a tensor of dataset indices and updates the number of forgetting events of the
    respective samples. The index of a given sample is the index to be used to retrieve the given sample from the
    underlying dataset.

    The result method returns an array of indices of the samples whose number of forgetting events have been tracked of
    sorted into descending order according to the respective number of forgetting events and an array of respective
    number of forgetting events.

    The reset method will bring the metric to its initial state. By default, this metric in its initial state will
    return a tuple with two empty arrays.
    """

    def __init__(self):
        """
        Create a new ForgettingEvents
        """

        self._num_forgetting_events: Dict[int, int] = {}
        """
        a dictionary containing the indices of samples as keys and the respective number of forgetting events as values
        """

        self._corr_classified: Dict[int, bool] = {}
        """
        a dictionary containing the indices of samples as keys and the respective classification states as values. The
        classification state of a given sample is a boolean value indicating whether the last time the sample was
        encountered, it was correctly classified (True) or misclassified (False).
        """

    def update(self, predicted_y: torch.Tensor, true_y: torch.Tensor, indices: torch.Tensor):
        """
        Update the number of forgetting events
        :param predicted_y: the predictions (either raw logits or softmax probabilities)
        :param true_y: the ground truth target labels
        :param indices: the dataset indices. Each index is the index to be used to retrieve the given sample
            from the underlying dataset.
        """
        predicted_labels = torch.argmax(predicted_y, dim=1)
        # a tensor with boolean values. True when the predicted label matches the true label; False otherwise.
        corr_predictions = torch.eq(predicted_labels, true_y)
        for i, pred in zip(indices, corr_predictions):
            if isinstance(i, torch.Tensor):
                i = int(i.item())
            if isinstance(pred, torch.Tensor):
                pred = bool(pred.item())
            # if the given sample has never been met before
            if i not in self._num_forgetting_events.keys():
                self._num_forgetting_events[i] = 0
            else:
                # if the given sample was correctly classified before and now not anymore, update its number of
                # forgetting events
                if self._corr_classified[i] is True and pred is False:
                    self._num_forgetting_events[i] += 1

            # update the classification state of the given sample
            self._corr_classified[i] = pred

    def reset(self):
        """
        Reset the metric to its initial state. The `_num_forgetting_events` and `_corr_classified` attributes are
        emptied.
        """
        self._num_forgetting_events = {}
        self._corr_classified = {}

    def result(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve the indices of the samples whose number of forgetting events have been tracked of and the respective
        number of forgetting events.
        :return: an array of indices of the samples whose number of forgetting events have been tracked of sorted into
            descending order according to the respective number of forgetting events and an array of respective number
            of forgetting events.
        """
        if len(self._num_forgetting_events) == 0:  # if `_num_forgetting_events` is empty, return two empty arrays
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        indices = np.asarray(list(self._num_forgetting_events.keys()), dtype=np.int64)
        forgetting_events = np.asarray(list(self._num_forgetting_events.values()), dtype=np.int64)
        # get the indices to sort `indices` and `forgetting_events` into descending order according to the number of
        # forgetting events
        desc_order = np.argsort(forgetting_events)[::-1]
        return indices[desc_order], forgetting_events[desc_order]


