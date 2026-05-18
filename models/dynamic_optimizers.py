"""
Utilities to handle optimizer's update when using dynamic architectures.
Dynamic Modules (e.g. multi-head) can change their parameters dynamically
during training, which usually requires to update the optimizer to learn
the new parameters or freeze the old ones.
"""
from collections import defaultdict

from avalanche.models.dynamic_optimizers import compare_keys


def update_optimizer(optimizer, new_params, optimized_params, reset_state=False):
    """
    Update the optimizer by adding new parameters,
    removing removed parameters, and adding new parameters
    to the optimizer, for instance after model has been adapted
    to a new task. The state of the optimizer can also be reset,
    it will be reset for the modified parameters.

    Newly added parameters are added by default to parameter group 0

    .note::
        This function is identical to :func:`avalanche.models.dynamic_optimizers.update_optimizer` except for
        an additional line of code that fixes an error of the original function. When an already existing parameter
        changes size, i.e. growing IncrementalClassifier, the older parameter is removed from the state of the
        optimizer. If this is not done, an error is raised when creating the state dictionary of the optimizer because
        the size of `optimizer.param_groups[0]['params']` does not match the size of `optimizer.state`, the latter
        having a bigger size.

    :param new_params: Dict (name, param) of new parameters
    :param optimized_params: Dict (name, param) of
        currently optimized parameters (returned by reset_optimizer)
    :param reset_state: Wheter to reset the optimizer's state (i.e momentum).
        Defaults to False.
    :return: Dict (name, param) of optimized parameters
    """
    not_in_new, in_both, not_in_old = compare_keys(optimized_params, new_params)
    # Change reference to already existing parameters
    # i.e growing IncrementalClassifier
    for key in in_both:
        old_p_hash = optimized_params[key]
        new_p = new_params[key]
        # Look for old parameter id in current optimizer
        found = False
        for group in optimizer.param_groups:
            for i, curr_p in enumerate(group["params"]):
                if id(curr_p) == id(old_p_hash):
                    found = True
                    if id(curr_p) != id(new_p):
                        group["params"][i] = new_p
                        optimized_params[key] = new_p
                        optimizer.state.pop(curr_p)
                        optimizer.state[new_p] = {}
                    break
        if not found:
            raise Exception(
                f"Parameter {key} expected but " "not found in the optimizer"
            )

    # Remove parameters that are not here anymore
    # This should not happend in most use case
    keys_to_remove = []
    for key in not_in_new:
        old_p_hash = optimized_params[key]
        found = False
        for i, group in enumerate(optimizer.param_groups):
            keys_to_remove.append([])
            for j, curr_p in enumerate(group["params"]):
                if id(curr_p) == id(old_p_hash):
                    found = True
                    keys_to_remove[i].append((j, curr_p))
                    optimized_params.pop(key)
                    break
        if not found:
            raise Exception(
                f"Parameter {key} expected but " "not found in the optimizer"
            )

    for i, idx_list in enumerate(keys_to_remove):
        for j, p in sorted(idx_list, key=lambda x: x[0], reverse=True):
            del optimizer.param_groups[i]["params"][j]
            if p in optimizer.state:
                optimizer.state.pop(p)

    # Add newly added parameters (i.e Multitask, PNN)
    # by default, add to param groups 0
    for key in not_in_old:
        new_p = new_params[key]
        optimizer.param_groups[0]["params"].append(new_p)
        optimized_params[key] = new_p
        optimizer.state[new_p] = {}

    if reset_state:
        optimizer.state = defaultdict(dict)

    return optimized_params