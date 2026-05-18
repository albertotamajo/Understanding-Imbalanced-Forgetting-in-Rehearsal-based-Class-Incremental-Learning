from typing import List, Any, Dict
from collections import defaultdict

from avalanche.evaluation import Metric


class ListAccumulator(Metric[List[Any]]):
    """
    The standalone metric for accumulating elements into a list.

    The update method receives an element and appends it to the accumulator list.

    The result method returns the accumulator list. The order of the elements in the accumulator list  reflects the
    order of the calls to the update method.

    The reset method will bring the metric to its initial state. By default, this metric in its initial state will
    return an empty list.
    """
    def __init__(self):
        """
        Initialise a new ListAccumulator
        """
        self._accumulator: List[Any] = []
        """the accumulator list"""

    def update(self, x: Any):
        """
        Update the accumulator list by appending an element to it
        :param x: an element
        """
        self._accumulator.append(x)

    def reset(self):
        """
        Reset the accumulator list by setting it to an empty list
        """
        self._accumulator = []

    def result(self) -> List[Any]:
        """
        Get the accumulator list
        :return: the accumulator list
        """
        return self._accumulator


class ListAccumulatorDictionary(Metric[Dict[Any, List[Any]]]):
    """
    The standalone metric for accumulating elements into a list for each key. This metric computes a dictionary of
    <key, accumulator list of elements> pairs.

    The update method receives a key and an element and appends the element to the accumulator list of the respective
    key.

    The result method returns a dictionary of <key, accumulator list of elements> pairs. The order of the elements in
    the accumulator list of each key reflects the order of the calls to the update method.

    The reset method will bring the metric to its initial state. By default, this metric in its initial state will
    return an empty dictionary.
    """
    def __init__(self):
        """
        Initialise a new ListAccumulatorDictionary
        """
        self._accumulator_dictionary = defaultdict(ListAccumulator)
        """a dictionary containing some keys along with their accumulator lists as values"""

    def update(self, key: Any, x: Any):
        """
        Update the accumulator dictionary by appending the provided element to the accumulator list of the given key
        :param key: a key
        :param x: an element
        """
        self._accumulator_dictionary[key].update(x)

    def reset(self):
        """
        Reset the accumulator dictionary by setting it to an empty dictionary
        :return:
        """
        self._accumulator_dictionary = defaultdict(ListAccumulator)

    def result(self) -> Dict[Any, List[Any]]:
        """
        Get the accumulator dictionary
        :return: a dictionary containing some keys along with their accumulator lists
        """
        return {key: value.result() for key, value in self._accumulator_dictionary.items()}
