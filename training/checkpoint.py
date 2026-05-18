"""
This module includes a set of utilities to save and load checkpoints
"""
from copy import copy

import dill
import torch


def save_checkpoint(strategy, fname, exclude=None):
    """
    Save the strategy state into a file.

    The file can be loaded using `maybe_load_checkpoint`.

    For efficiency, the user can specify some attributes to `exclude`.
    For example, if the optimizer is static and doesn't change during training,
    it can be safely excluded. These helps to speed up the serialization.

    WARNING: the method cannot be used inside the training and evaluation
    loops of the strategy.

    .note::
        This function is identical to the Avalanche's `save_checkpoint` function with the only difference that it
        does not save the state of `RNGManager`.

    :param strategy: strategy to serialize.
    :param fname: name of the file.
    :param exclude: List[string] list of attributes to remove before the
        serialization.
    :return:
    """
    if exclude is None:
        exclude = []
    ended_experience_counter = strategy.clock.train_exp_counter

    strategy = copy(strategy)
    for attr in exclude:
        delattr(strategy, attr)

    checkpoint_data = {
        "strategy": strategy,
        "exp_counter": ended_experience_counter,
    }
    torch.save(checkpoint_data, fname, pickle_module=dill)
