"""
This module includes a set of classes used to run experiments and a set of plugins
"""
from __future__ import annotations

import datetime
import dill
from typing import Optional, Literal, List, Iterable, Union, Sequence, Dict, Any, TYPE_CHECKING
import os
from abc import ABC

from training.checkpoint import save_checkpoint
from training.plugins import GradientBasedReplayBufferSelectionSGDPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate

from avalanche.benchmarks import ClassificationScenario
from avalanche.benchmarks.utils import concat_datasets
from avalanche.training.determinism import RNGManager
from avalanche.training.checkpoint import maybe_load_checkpoint
import torch
import torch.distributed as dist
import numpy as np
from sklearn.metrics import pairwise_distances

if TYPE_CHECKING:
    from training.templates import DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate
    from avalanche.core import BasePlugin


class Runner:
    """
    This class allows running continual learning experiments using strategies that subclass
    :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`.

    The basic way of running non-distributed continual learning experiments using strategies that
    subclass :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` is::
        python YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)  # running training script in the cmd

        # Inside the training script
        strategy = # instantiate a :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` strategy
        benchmark = # instantiate a classification scenario
        runner = Runner(strategy, benchmark)
        runner.run()

    The basic way of running distributed continual learning experiments on one node using strategies that
    subclass :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` is::
        torchrun
            --standalone
            --nnodes=1
            --nproc-per-node=$NUM_TRAINERS(GPUs)
            YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...) # running training script in the cmd
        OR
        python -m torch.distributed.launch
            --use-env
            --nnodes=1
            --nproc-per-node=$NUM_TRAINERS(GPUs)
            YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...) # running training script in the cmd

        # Inside the training script
        device = Runner.init_distributed()
        strategy = # instantiate a :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` strategy
            setting the strategy's device to `device`
        benchmark = # instantiate a classification scenario
        runner = Runner(strategy, benchmark)
        runner.run()
    """

    def __init__(self,
                 strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
                 benchmark: ClassificationScenario,
                 ):
        """
        Create a new Runner
        :param strategy: a :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` strategy
        :param benchmark: a classification scenario instance.
        """
        self.strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate = strategy
        """a :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` strategy"""

        self.benchmark: ClassificationScenario = benchmark
        """a classification scenario"""

    def run(self, save_checkpoint_path: str, load_checkpoint: Optional[str] = None,
            device_checkpoint: Optional[Union[str, torch.device]] = None, seed: Union[List[int], int] = 1234,
            eval_streams_type: Union[Literal["train", "val", "test"], Iterable[Literal["train", "val", "test"]]] = "test",
            experience_stop: Optional[int] = None, plugins: Optional[Sequence[RunnerPlugin]] = None, **kwargs):
        """
        Run an experiment.

        The training loop iterates over each experience in `self.benchmark`. Each iteration uses `self.strategy`
        to train on the current experience and then uses `self.strategy` to evaluate on all the past and current
        training, test and (if available) validation experiences.

        At the end of each iteration, a checkpoint is saved.

        .note::
            If running a distributed training, a checkpoint is saved at the end of each iteration only by the process
            with rank 0. This happens because only the process with rank 0 is used to perform the evaluations
            on all the past and current training and evaluation experiences. Therefore, only the process with rank 0
            has all the evaluation metrics at the end of an experience. It is redundant for the other processes to
            perform the evaluations too as they would perform the same operations as the process with rank 0.

        :param save_checkpoint_path: path where to store a checkpoint. It can be any string. After each iteration, the
            following string '_experience_{experience number}.pth' is appended to this string and a checkpoint saved.
            Therefore, after each iteration, a checkpoint with a different path is saved.
        :param load_checkpoint: (optional) path where to load checkpoint. If None, no checkpoint is loaded.
            Default is None.
        :param device_checkpoint: (optional) device where to move the strategy deserialised from the loaded checkpoint.
            If None, the deserialised strategy will be moved to the same device of the strategy stored in
            `self.strategy`. Note that issues are encountered when moving the strategy to CUDA and setting `num_workers`
            in the dataloader to a number greater than 0. Therefore, it is recommended to deserialise the strategy to
            CPU and then let the strategy itself move the underlying model to CUDA. Default is None.
        :param seed: (optional) initial seed of all random number generators or a list of seeds, one per experience in
            the benchmark. Python (`random` module), NumPy, and PyTorch global generators are managed. If only one seed
            is provided (a single integer number), the seed is set once prior to running the experiment. Otherwise, if a
            list of seeds is provided, the respective seed is set prior to running each experience. If the length of
            the provided list of seeds does not match the number of experiences in the benchmark, a :class:`ValueError`
            is raised. Default is 1234.
        :param eval_streams_type: (optional) `train`, `val`, `test` or an iterable containing one or any combination of
            them.
            If `train` is present, all the past train experiences and the current train experience are provided to the
            `strategy.train` method.
            If `val` is present, all the past validation experiences and the current one are provided to the
            `strategy.train` method. If no validation stream is present in the `benchmark`, a :class:`RuntimeError`
            is raised.
            If `test` is present, all the past test experiences and the current one are provided to the
            `strategy.train` method.
            This is useful when some periodic evaluation after n epochs or n iterations is
            performed during the training. This happens if the attribute `eval_every` of the strategy is greater than or
            equal to 0. The periodic evaluation will be performed on the experiences described by this argument.
            Default is `test`.
        :param experience_stop: (optional) ID of the experience after which training stops. Note that the training is
            not performed on this last experience. If None, the training is performed for all experiences in the
            training stream. Default is None.
        :param plugins: a sequence of :class:`RunnerPlugin` plugins to "inject" additional code during the execution of
            this method. Default is None.
        :param kwargs: some keyword arguments that are propagated
            through :meth:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate.train`
            and :meth:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate.eval`
        :return: None
        """
        if isinstance(seed, int):
            RNGManager.set_random_seeds(seed)
        elif isinstance(seed, list):
            # if the number of seeds provided does not match the number of experiences in the benchmark
            if len(seed) != len(self.benchmark.train_stream):
                raise ValueError("The number of seeds provided does not match the number of experiences in the "
                                 "benchmark")
        else:
            raise ValueError("The seed must be either an integer or a list of integers")

        if plugins is None:
            plugins = []

        self._before_loading_checkpoint(plugins)
        exp_counter = 0  # starting experience
        if load_checkpoint is not None:
            if device_checkpoint is None:
                device_checkpoint = self.strategy.device
            self.strategy, exp_counter = maybe_load_checkpoint(self.strategy, load_checkpoint,
                                                               map_location=device_checkpoint)
        self._after_loading_checkpoint(plugins)

        if eval_streams_type == "train" or eval_streams_type == "val" or eval_streams_type == "test":
            eval_streams_type = [eval_streams_type]
        # remove duplicated elements
        eval_streams_type = set(eval_streams_type)
        for e_type in eval_streams_type:
            if e_type not in ["train", "val", "test"]:
                raise ValueError(
                    "The eval_streams_type argument can only be `train`, `val` and `test` or an iterable containing "
                    "any combination of them")

        print("Starting experiment...")
        for index, experience in enumerate(self.benchmark.train_stream[exp_counter:]):  # starting from the starting exp
            curr_exp = exp_counter + index
            # stop the training if the stop experience has been reached
            if experience_stop is not None and curr_exp >= experience_stop:
                break
            if isinstance(seed, list):
                RNGManager.set_random_seeds(seed[curr_exp])

            print("Start of experience: ", experience.current_experience)
            print("Current Classes: ", experience.classes_in_this_experience)

            print("Training...")

            # according to the values in eval_streams_type, all the past val experiences and the current one
            # or all the past train experiences and the current one or all the test experiences and the current one or
            # a combination of any of them are provided to the `train` method in case some periodic evaluation after n
            # epochs or n iterations is performed during the training. This happens if the attribute `eval_every` of
            # the strategy is greater than or equal to 0.
            eval_streams = []
            for eval_stream_type in eval_streams_type:
                if eval_stream_type == "train":
                    eval_streams.append(self.benchmark.train_stream[:curr_exp + 1])
                elif eval_stream_type == "val":
                    if not hasattr(self.benchmark, "val_stream"):
                        raise RuntimeError("The provided benchmark does not have a validation stream")
                    eval_streams.append(self.benchmark.val_stream[:curr_exp + 1])
                else:
                    eval_streams.append(self.benchmark.test_stream[:curr_exp + 1])

            self.strategy.train(experience, eval_streams=eval_streams, **kwargs)
            print('Training completed')

            # If running a distributed training only the process with rank 0 performs the evaluations and can save the
            # checkpoint.
            if (dist.is_initialized() and dist.get_rank() == 0) or (not dist.is_initialized()):
                print('Evaluating on the previous training experiences and current training experience...')
                self.strategy.eval(self.benchmark.train_stream[:curr_exp + 1], **kwargs)
                print("Evaluation completed")

                if hasattr(self.benchmark, "val_stream"):
                    print('Evaluating on the previous validation experiences and current validation experience...')
                    self.strategy.eval(self.benchmark.val_stream[:curr_exp + 1], **kwargs)
                    print("Evaluation completed")

                print('Evaluating on the previous test experiences and current test experience...')
                self.strategy.eval(self.benchmark.test_stream[:curr_exp + 1], **kwargs)
                print("Evaluation completed")

                print("Saving checkpoint...")
                save_checkpoint(self.strategy, f"{save_checkpoint_path}_experience_{curr_exp}.pth")
                print("Checkpoint saved")

        print("Experiment completed")

    @staticmethod
    def init_distributed(backend: Optional[Literal["gloo", "mpi", "nccl"]] = None, use_cuda=True,
                         set_cuda_device=True, seed: int = 1234):
        """
        Initialise a distributed process.

        .warning::
            This method retrieves the local rank of a process from the LOCAL_RANK environment variable rather than
            from a `--local-rank` cmd argument. Therefore, either `torch.run` or `torch.distributed.launch` with the
            `--use-env` flag must be used to start the distributed process. Note that `torch.distributed.launch` is now
            deprecated in favour of `torch.run`.
        :param backend: (optional) a distributed backend. Either "gloo", "mpi" or "nccl". If None, the backend defaults
            to "nccl" if using CUDA, "gloo" otherwise. Default is None.
        :param use_cuda: (optional) whether to use cuda or not. Default is True
        :param set_cuda_device: (optional) If True and `use_cuda=True`, set the default CUDA device of the process
            by calling `torch.cuda.set_device`. Default is True
        :param seed: (optional) initial seed of all random number generators. Python (`random` module), NumPy, and
            PyTorch global generators are managed. Default is 1234.
        :return: the default device used by the process
        """
        if dist.is_initialized():
            raise RuntimeError("Distributed process already initialized")

        use_cuda = use_cuda and torch.cuda.is_available()

        if backend is None:
            if use_cuda:
                backend = "nccl"
            else:
                backend = "gloo"

        if backend == "nccl" and not use_cuda:
            raise RuntimeError("Bad configuration: using NCCL, but you set use_cuda=False!")

        local_rank = os.environ.get("LOCAL_RANK", None)
        if local_rank is None:
            raise RuntimeError("Torch distributed could not be initialized (missing environment configuration)")
        local_rank = int(local_rank)
        dist.init_process_group(backend=backend, timeout=datetime.timedelta(seconds=10800))
        if use_cuda and local_rank >= 0:
            ref_device = torch.device(f"cuda:{local_rank}")
            if set_cuda_device:
                torch.cuda.set_device(ref_device)
        else:
            ref_device = torch.device("cpu")

        RNGManager.set_random_seeds(seed)
        return ref_device

    #########################################################
    # Plugin Triggers                                       #
    #########################################################
    def _before_loading_checkpoint(self, plugins: Sequence[RunnerPlugin], **kwargs):
        self.trigger_plugins(plugins, "before_loading_checkpoint", **kwargs)

    def _after_loading_checkpoint(self, plugins: Sequence[RunnerPlugin], **kwargs):
        self.trigger_plugins(plugins, "after_loading_checkpoint", **kwargs)

    def trigger_plugins(self, plugins: Sequence[RunnerPlugin], event: str, **kwargs):
        """
        Call the given callback on each of the provided plugins
        :param plugins: a sequence of plugins
        :param event: name of the callback event
        """
        for p in plugins:
            if hasattr(p, event):
                getattr(p, event)(self, **kwargs)


class RunnerPlugin(ABC):
    """
    ABC for :class:`Runner` plugins.

    A plugin is simply an object implementing some callbacks.
    Plugins are called automatically during :meth:`Runner.run`.

    Callbacks provide access to the state of a :class`Runner` object before/after a given phase.
    In general, for a given phase, this plugin provides two functions `before_{phase}` and `after_{phase}`, called
    before and after the given phase, respectively.
    Therefore, plugins can "inject" additional code during the execution of :meth:`Runner.run` by implementing
    callbacks. Each callback receives the :class:`Runner` object as argument allowing it to gain access to the state of
    the object.
    """
    def __init__(self):
        super().__init__()

    def before_loading_checkpoint(self, runner: Runner, *args, **kwargs):
        """Called before loading a checkpoint by :class:`Runner`."""
        pass

    def after_loading_checkpoint(self, runner: Runner, *args, **kwargs):
        """Called after loading a checkpoint by :class:`Runner`."""
        pass


class InjectRunnerPlugin(RunnerPlugin):
    """
    This plugin allows to inject a given object into the `strategy` attribute of a :class:`Runner` object as a
    replacement for another after loading a checkpoint.
    """
    def __init__(self, injection: Any, locs: Union[str, Iterable[str]], update_attributes: bool = False, *args):
        """
        Create a new InjectRunnerPlugin
        :param injection: an object (it must not be of primitive type) to be injected into the `strategy` attribute of
            a :class:`Runner` object
        :param locs: a location or multiple locations where the injection object must be injected as a replacement for
            another. Note that all locations must actually refer to the same object in memory. The object must not be of
            primitive type. Suppose an object to be replaced is the first element in the `plugins` attribute of the
            strategy, then "plugins[0]" must be provided.
        :param update_attributes: whether the attributes and their respective values in the replaced object must be
            inserted into the injection object. If an attribute does not exist in the injection object, it and its
            respective value are added to the injection object. If an attribute exists in the injected object, its value
            is replaced with the respective value in the replaced object. Default is False.
        :param args: series of strings denoting which attributes in the replaced object must not be inserted into the
            injection object. Only used when `update_attributes` is True.
        """
        super().__init__()

        self._injection: Any = injection
        """
        an object to be injected into the `strategy` attribute of a :class:`Runner` object
        """

        if isinstance(locs, str):
            locs = [locs]
        self._locs: Iterable[str] = locs
        """
        a location or multiple locations where the injection object must be injected as a replacement for another.
        """

        self._update_attributes: bool = update_attributes
        """
        whether the attributes and their respective values in the replaced object must be inserted into the injection
        object.
        """

        self._no_update_attributes = args
        """
        an empty tuple or a tuple containing some strings denoting which attributes in the replaced object must not be
        inserted into the injection object. Only used when `self._update_attributes` is True`.
        """

    def after_loading_checkpoint(self, runner: Runner, *args, **kwargs):
        """
        Callback called after loading a checkpoint.

        It injects the injection object into the provided locations. If the attributes in the replaced object can be
        inserted into the injection object, they are inserted into the injection object except for those attributes that
        are in `self._no_update_attributes`
        :param runner: a :class:`Runner` instance
        """

        # collect all objects the locations point to
        objs = []
        for loc in self._locs:
            exec(f"objs.append(runner.strategy.{loc})")
        # check whether all objects are stored into the same memory address
        if not all([obj is objs[0] for obj in objs]):
            raise RuntimeError("The locations must refer to the same object in memory")

        if self._update_attributes:
            # get all attributes of the replaced object
            attrs = objs[0].__dict__
            # discard all attributes that are in `self.no_update_attributes`
            attrs = {x: y for x, y in attrs.items() if x not in self._no_update_attributes}

            # update the attributes of the injection object
            self._injection.__dict__.update(attrs)

        # replace the locations with the injection object
        for loc in self._locs:
            exec(f"runner.strategy.{loc} = self._injection")


class AddUnsharedAttributesRunnerPlugin(RunnerPlugin):
    """
    This plugin allows to add the unshared attributes present in an object into another object after loading a
    checkpoint. The second object must be inside the `strategy` attribute of a :class:`Runner` object after loading a
    checkpoint.
    """
    def __init__(self, obj: Any, loc: str):
        """
        Create a new AddUnsharedAttributesRunnerPlugin
        :param obj: an object (it must not be of primitive type) whose unshared attributes must be added to the object
            in `loc` after loading a checkpoint.
        :param loc: the location of the object inside the `strategy` attribute of a :class:`Runner` object
            that is added with the unshared attributes in `obj` after loading a checkpoint. The object must not be of
            primitive type.
            Suppose the object is the first element in the `plugins` attribute of the strategy, then "plugins[0]"
            must be provided.
        """
        super().__init__()

        self._obj: Any = obj
        """
        an object whose unshared attributes must be added to the object in `loc` after loading a checkpoint.
        The object must not be of primitive type.
        """

        self._loc: str = loc
        """
        the location of the object inside the `strategy` attribute of a :class:`Runner` object
        that is added with the unshared attributes in `obj` after loading a checkpoint.
        The object must not be of primitive type.
        """

    def after_loading_checkpoint(self, runner: Runner, *args, **kwargs):
        """
        Callback called after loading a checkpoint.

        It adds the unshared attributes present in `self._obj` into the object referenced in `loc`.
        :param runner: a :class:`Runner` instance
        """
        # get the object referenced in `loc`
        obj_ref = []
        exec(f"obj_ref.append(runner.strategy.{self._loc})")
        obj_ref = obj_ref[0]
        # get the set of attributes in `obj_ref`
        attr_obj_ref = set(obj_ref.__dict__.keys())
        # get the set of attributes in `self._obj`
        attr_obj = set(self._obj.__dict__.keys())
        # get the set of attributes in `self._obj` not in `obj_ref`
        attr_not_in_obj_ref = attr_obj - attr_obj_ref
        obj_ref.__dict__.update(
            {key: val for key, val in self._obj.__dict__.items() if key in attr_not_in_obj_ref}
        )


class InsertPluginRunnerPlugin(RunnerPlugin):
    """
    This plugin allows to insert a :class:`BasePlugin` into a given index of the `plugins` list in the `strategy`
    attribute of a :class:`Runner` object. Note that the other elements in `plugins` are shifted to the right to make
    space for the new element.
    """
    def __init__(self, index: int, plugin: BasePlugin):
        """
        Create a new InsertPluginRunnerPlugin
        :param index: index where the plugin is inserted; the other elements in `plugins` are shifted to the right to
            make space for the new element
        :param plugin: the plugin to be inserted
        """
        super().__init__()

        self._index: int = index
        """index where the plugin is inserted"""

        self._plugin: BasePlugin = plugin
        """the plugin to be inserted"""

    def after_loading_checkpoint(self, runner: Runner, *args, **kwargs):
        """
        Callback called after loading a checkpoint.

        Insert `self._plugin` into the `self._index` index of the `plugins` list in the `strategy`
        attribute of `runner`.
        :param runner: a :class:`Runner` instance
        """
        runner.strategy.plugins.insert(self._index, self._plugin)


class RecomputeCoresetGradientBasedReplayBufferSelectionSGDRunnerPlugin(RunnerPlugin):
    """
    This plugin allows to recompute the coreset stored
    in :class:`GradientBasedReplayBufferSelectionSGDPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
    after loading a strategy's checkpoint. Therefore, the strategy **must** have an instance
    of :class:`GradientBasedReplayBufferSelectionSGDPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
    in the `plugins` attribute.

    It undoes the actions performed by the `update_from_dataset` method
    in :class:`GradientBasedReplayBufferSelectionSGDPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
    the last time it was called, resetting the state of the class as it was immediately before calling
    `update_from_dataset`. Note that the `update_from_dataset` method is the method that computes the coreset.
    The attributes that must be reset to their previous state are: `curr_distance_kernels`, `curr_coreset`,
    `buffer_groups`, `all_coresets`, `all_distance_kernels` and `seen_classes`. To reset the state of
    `curr_distance_kernels`, `store_all_distance_kernels` or `store_all_gradients` must be True.
    If `store_all_gradients` is True and `store_all_distance_kernels` is False, the only way to recover the distance
    kernels is to recompute them from the gradients stored in `all_gradients`.`store_all_coresets`
    must be True to reset the state of `curr_coreset`. The current coreset must also be removed from `all_coresets`.
    Afterwards, rather than using all the distance kernels in
    `curr_distance_kernels`, this plugin uses a different number of distance kernels, according to the values of
    `n_dist_kernels` and `direction` provided at construction, to recompute the coreset.
    Additionally, one can make further modifications to how the coreset is computed by modifying the values of some
    attributes
    in :class:`GradientBasedReplayBufferSelectionSGDPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`.
    The following attributes can be modified:
    - `num_subsets`: to change the number of disjoint subsets to be used to select the samples that must be preserved
        in the replay buffer for each class
    -`subset_generator`: to change the subset generator that divides the samples of a given class into
        `self.num_subsets` disjoint subsets
    -`metric`: to use a different metric to compute the distance kernels. Only used when `store_all_gradients` is True.
        Even if `store_all_distance_kernels` is True, all the distance kernels stored in `all_distance_kernels` are
        ignored and they are recomputed with a different metric from the gradients stored in `all_gradients`. The old
        distance kernels in `all_distance_kernels` are replaced with the newly computed ones.
    -`max_size`: to change the maximum capacity of the replay buffer. It can **only** be used when only one training
        experience was encountered during training
    - `aggregate_similarity_kernel`: to change how the distance kernels of a class are aggregated into a single
        aggregate similarity kernel
    """
    def __init__(self, n_dist_kernels: int, direction: Literal["first", "last"], **kwargs):
        """
        Create a new RecomputeCoresetGradientBasedReplayBufferSelectionSGDRunnerPlugin
        :param n_dist_kernels: number of distance kernels to use for the re-computation of the coreset
        :param direction: either `first` to use the first `n_dist_kernels` or `last` to use the last `n_dist_kernels`
            for the re-computation of the coreset
        :param kwargs: some keyword arguments. The keys must be in
            {`num_subsets`, `subset_generator`, `metric`, `max_size`, `aggregate_similarity_kernel`}.
            Use `num_subsets` to change the number of disjoint subsets to be used to select the samples that must be
            preserved in the replay buffer for each class. Use `subset_generator` to change the subset generator that
            divides the samples of a given class into a given number of disjoint subsets. Use `metric` to use a
            different metric to compute the distance kernels. **Only** used when `store_all_gradients` is True.
            Even if `store_all_distance_kernels` is True, all the distance kernels stored in `all_distance_kernels` are
            ignored and they are recomputed with a different metric from the gradients stored in `all_gradients`.
            The old distance kernels in `all_distance_kernels` are replaced with the newly computed ones.
            Use `max_size` to change the maximum capacity of the replay buffer. It can **only** be used when only one
            training experience was encountered during training. Use `aggregate_similarity_kernel` to change how the
            distance kernels of a class are aggregated into a single aggregate similarity kernel.
        """
        super().__init__()

        self.n_dist_kernels: int = n_dist_kernels
        """
        number of distance kernels to use for the re-computation of the coreset
        """

        self.direction: Literal["first", "last"] = direction
        """
        either `first` to use the first `self.n_dist_kernels` or `last` to use the last `self.n_dist_kernels`
        for the re-computation of the coreset
        """

        # check whether all keys in the provided keyword arguments are in
        # {`num_subsets`, `subset_generator`, `metric`, `max_size`, `aggregate_similarity_kernel`}
        if not all([key in ["num_subsets", "subset_generator", "metric", "max_size", "aggregate_similarity_kernel"]
                    for key in kwargs.keys()]):
            raise ValueError("At least one key of the provided keyword arguments is not in "
                             "{`num_subsets`, `subset_generator`, `metric`, `max_size`, `aggregate_similarity_kernel`}")

        self.kwargs = kwargs
        """
        some keyword arguments. The keys are in {`num_subsets`, `subset_generator`, `metric` `max_size`,
        `aggregate_similarity_kernel`}. Use `num_subsets` to change the number of disjoint subsets to be used to select
        the samples that must be preserved in the replay buffer for each class. Use `subset_generator` to change the
        subset generator that divides the samples of a given class into a given number of disjoint subsets.
        Use `metric` to use a different metric to compute the distance kernels. Only used when `store_all_gradients`
        is True. Even if `store_all_distance_kernels` is True, all the distance kernels stored in `all_distance_kernels`
        are ignored and they are recomputed with a different metric from the gradients stored in `all_gradients`.
        The old distance kernels in `all_distance_kernels` are replaced with the newly computed ones.
        Use `max_size` to change the maximum capacity of the replay buffer. It can **only** be used when only one
        training experience was encountered during training. Use `aggregate_similarity_kernel` to change how the
        distance kernels of a class are aggregated into a single aggregate similarity kernel.
        """

    def after_loading_checkpoint(self, runner: Runner, *args, **kwargs):
        """
        Callback called after loading a checkpoint.

        This callback recomputes the coreset stored
        in :class:`GradientBasedReplayBufferSelectionSGDPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
        after loading a strategy's checkpoint. Therefore, the strategy **must** have an instance
        of :class:`GradientBasedReplayBufferSelectionSGDPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
        in the `plugins` attribute. It undoes the actions performed by the `update_from_dataset` method
        in :class:`GradientBasedReplayBufferSelectionSGDPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
        the last time it was called, resetting the state of the class as it was immediately before calling
        `update_from_dataset`. Note that the `update_from_dataset` method is the method that computes the coreset.
        :param runner: a :class:`Runner` instance
        """
        plgs = [plg for plg in runner.strategy.plugins if isinstance(plg, GradientBasedReplayBufferSelectionSGDPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate)]
        if len(plgs) == 0:
            raise RuntimeError("The strategy does not have an instance of "
                               "`GradientBasedReplayBufferSelectionSGDPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` "
                               "in the `plugins` attribute")
        coreset_plg: GradientBasedReplayBufferSelectionSGDPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate = plgs[0]
        if not (coreset_plg.store_all_distance_kernels or coreset_plg.store_all_gradients):
            raise RuntimeError("`store_all_distance_kernels` or `store_all_gradients` must be True")
        if not coreset_plg.store_all_coresets:
            raise RuntimeError("`store_all_coresets` must be True")
        if "max_size" in self.kwargs.keys():
            if not len(coreset_plg._train_exps) == 1:
                raise RuntimeError("The number of training experiences encountered during training must be 1 when "
                                   "`max_size` is provided as a keyword argument")

        # get the last encountered experience
        last_exp = coreset_plg._train_exps[-1]

        # only reset the state of the class to what it was immediately before calling `update_from_dataset` if the last
        # encountered experience is not to be ignored. If it is to be ignored, no coreset updates were actually
        # performed
        if last_exp.current_experience not in coreset_plg.exps_no_coreset:
            cls_in_last_exp = last_exp.classes_in_this_experience  # get the classes in the last experience
            if not len(runner.strategy.model.subnetworks_classes) == 1:
                raise RuntimeError("This plugin expects a single subnetwork")
            # get the classes seen by the single subnetwork
            sub_cls = list(runner.strategy.model.subnetworks_classes.values())[0]
            # convert the classes in the last experience into the order with which they appear in the single
            # subnetwork
            order_last_exp = [sub_cls.index(cls) for cls in cls_in_last_exp]
            # reset `seen_classes`
            self._reset_seen_classes(plg=coreset_plg, cls=order_last_exp)
            # reset `curr_coreset`
            self._reset_curr_coreset(plg=coreset_plg, cls=order_last_exp)
            # reset `all_coresets`
            self._reset_all_coresets(plg=coreset_plg, cls=order_last_exp)
            # reset `buffer_groups`
            self._reset_buffer_groups(plg=coreset_plg, cls=order_last_exp)

            # modify the values of some attributes by using the keyword arguments in `self.kwargs`
            for key, val in self.kwargs.items():
                if not hasattr(coreset_plg, key):
                    raise RuntimeError(f"The attribute `{key}` does not exist")
                if not key == "metric":
                    setattr(coreset_plg, key, val)
                else:
                    # only change the `metric` attribute if `store_all_gradients` is True
                    if coreset_plg.store_all_gradients:
                        setattr(coreset_plg, key, val)

            if not coreset_plg.store_all_gradients:
                # `curr_distance_kernels` is reset to the state it was before calling `update_from_dataset` from the
                # kernels in `all_distance_kernels`
                self._reset_curr_distance_kernels_from_all_distance_kernels(plg=coreset_plg, cls=order_last_exp)
            else:
                # `curr_distance_kernels` is reset to the state it was before calling `update_from_dataset` from the
                # gradients in `all_gradients`
                self._reset_curr_distance_kernels_from_all_gradients(plg=coreset_plg, cls=order_last_exp)

            # filter out `curr_distance_kernels`
            for target in coreset_plg.curr_distance_kernels.keys():
                t_dist_kernels = coreset_plg.curr_distance_kernels[target]
                if self.direction == "first":
                    coreset_plg.curr_distance_kernels[target] = t_dist_kernels[:self.n_dist_kernels]
                else:
                    coreset_plg.curr_distance_kernels[target] = t_dist_kernels[-self.n_dist_kernels:]

            # recompute the coreset
            coreset_plg.update_from_dataset(new_data=coreset_plg._train_datasets[-1])

    def _reset_seen_classes(self,
                            plg: GradientBasedReplayBufferSelectionSGDPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
                            cls: List[int]):
        """
        Reset the state of `seen_classes` to what it was before calling `update_from_dataset`.
        :param plg: a :class:`GradientBasedReplayBufferSelectionSGDPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
            plugin
        :param cls: a list of the target labels encountered in the last experience. The target labels must not be the
            real target labels but the order in which classes appear in the list of classes seen by the unique
            subnetwork
        """
        plg.seen_classes.difference_update(set(cls))

    def _reset_curr_coreset(self,
                            plg: GradientBasedReplayBufferSelectionSGDPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
                            cls: List[int]):
        """
        Reset the state of `curr_coreset` to what it was before calling `update_from_dataset`.
        .note::
            This method assumes that `all_coresets` has not been reset yet to the state it was before calling
            `update_from_dataset`
        :param plg: a :class:`GradientBasedReplayBufferSelectionSGDPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
            plugin
        :param cls: a list of the target labels encountered in the last experience. The target labels must not be the
            real target labels but the order in which classes appear in the list of classes seen by the unique
            subnetwork
        """
        # remove from `curr_coreset` the coreset of the classes encountered in the last experience
        for target in cls:
            plg.curr_coreset.pop(target)

        # reset the state of `curr_coreset` to what it was before calling `update_from_dataset` by resetting the coreset
        # of old classes
        for target in plg.curr_coreset.keys():
            plg.curr_coreset[target] = plg.all_coresets[target][-2]

    def _reset_all_coresets(self,
                            plg: GradientBasedReplayBufferSelectionSGDPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
                            cls: List[int]):
        """
        Reset the state of `all_coresets` to what it was before calling `update_from_dataset`.
        :param plg: a :class:`GradientBasedReplayBufferSelectionSGDPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
            plugin
        :param cls: a list of the target labels encountered in the last experience. The target labels must not be the
            real target labels but the order in which classes appear in the list of classes seen by the unique
            subnetwork
        """
        # remove the key-value pairs in `all_coresets` of classes that are in the last encountered experience
        for target in cls:
            plg.all_coresets.pop(target)

        # remove the current coreset from `all_coresets` for all classes encountered in past experiences
        for coreset_ls in plg.all_coresets.values():
            coreset_ls.pop()

    def _reset_buffer_groups(self,
                             plg: GradientBasedReplayBufferSelectionSGDPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
                             cls: List[int]):
        """
        Reset the state of `buffer_groups` to what it was before calling `update_from_dataset`.
        .note::
            This method assumes that `curr_coreset` is reset to the state it was before calling `update_from_dataset`
        :param plg: a :class:`GradientBasedReplayBufferSelectionSGDPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
            plugin
        :param cls: a list of the target labels encountered in the last experience. The target labels must not be the
            real target labels but the order in which classes appear in the list of classes seen by the unique
            subnetwork
        """
        # remove from `buffer_groups` the buffers of the classes encountered in the last experience
        for target in cls:
            plg.buffer_groups.pop(target)

        # reset the state of `buffer_groups` to what it was before calling `update_from_dataset`
        for target, (indxs, _, _) in plg.curr_coreset.items():
            plg.buffer_groups[target] = concat_datasets(
                [plg._subset_dataset_indices(ds, indxs) for ds in plg._train_datasets]
            )

    def _reset_curr_distance_kernels_from_all_distance_kernels(self,
                                                               plg: GradientBasedReplayBufferSelectionSGDPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
                                                               cls: List[int]):
        """
        Reset the state of `curr_distance_kernels` to what it was before calling `update_from_dataset` from the kernels
        in `all_distance_kernels`.
        .note::
            This method assumes that the state of `all_coresets` is the one before calling `update_from_dataset`
        :param plg: a :class:`GradientBasedReplayBufferSelectionSGDPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
            plugin
        :param cls: a list of the target labels encountered in the last experience. The target labels must not be the
            real target labels but the order in which classes appear in the list of classes seen by the unique
            subnetwork
        """
        if plg.adaptive_size and plg.recompute_coreset:
            # `curr_distance_kernels` needs to be populated with the distance kernels of past classes as well
            cls += list(plg.all_coresets.keys())
        for target in cls:
            t_all_distance_kernels = plg.all_distance_kernels[target]
            # get the number of indices tracked in the last distance kernel
            n_indices = len(t_all_distance_kernels[-1][1])
            # get the number of distance kernels with the same number of tracked indices
            n_kernels = sum([int(len(indx) == n_indices) for _, indx in t_all_distance_kernels])
            # add in `curr_distance_kernels` the last `n_kernels` for the given target
            plg.curr_distance_kernels[target].extend(t_all_distance_kernels[-n_kernels:])

    def _reset_curr_distance_kernels_from_all_gradients(self,
                                                        plg: GradientBasedReplayBufferSelectionSGDPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
                                                        cls: List[int]):
        """
        Reset the state of `curr_distance_kernels` to what it was before calling `update_from_dataset` from the
        gradients in `all_gradients`
        .note::
            This method assumes that the states of `curr_coreset` and `all_coresets` are the ones before calling
            `update_from_dataset`
        :param plg: a :class:`GradientBasedReplayBufferSelectionSGDPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
            plugin
        :param cls: a list of the target labels encountered in the last experience. The target labels must not be the
            real target labels but the order in which classes appear in the list of classes seen by the unique
            subnetwork
        """
        if plg.adaptive_size and plg.recompute_coreset:
            # `curr_distance_kernels` needs to be populated with the distance kernels of past classes as well
            cls += list(plg.all_coresets.keys())
        for target in cls:
            t_all_gradients = plg.all_gradients[target]
            if plg.gradient_approx == "class_head" and plg.save_space_gradients:
                # get the size of the gradient wrt the biases in the last gradient item
                size = (t_all_gradients[-1][0][0]).shape[1]
                # get the number of gradient items with the same size of gradient wrt biases
                n_gradients = sum([int(size == grad_bias.shape[1]) for (grad_bias, _), _ in t_all_gradients])
                # get the last `n_gradients` gradient items
                last_gradients_biases_embeds = t_all_gradients[-n_gradients:]
                # convert the grads biases, embeds pairs in `last_gradients_biases_embeds` into the grads wrt
                # the weights and biases
                last_gradients = []
                for (grads_biases, embeds), indices in last_gradients_biases_embeds:
                    grads_biases_expand = np.expand_dims(grads_biases, axis=2)
                    embeds_expand = np.expand_dims(embeds, axis=1)
                    grads_weights = np.matmul(grads_biases_expand, embeds_expand).reshape(len(grads_biases), -1)
                    grads_weights_biases = np.concatenate((grads_weights, grads_biases), axis=1)
                    last_gradients.append((grads_weights_biases, indices))
            else:
                # get the size of the gradient in the last gradient item
                size = (t_all_gradients[-1][0]).shape[1]
                # get the number of gradient items with the same size of gradient
                n_gradients = sum([int(size == grad.shape[1]) for grad, _ in t_all_gradients])
                # get the last `n_gradients` gradient items
                last_gradients = t_all_gradients[-n_gradients:]

            if plg.compute_all_past_gradients:
                # if the given target is a target encountered in a past experience, since
                # plg.compute_all_past_gradients and plg.store_all_gradients are True,
                # the grads, indices pairs in `last_gradients` contain gradients and indices of samples that are
                # not in the state of the replay buffer before calling `update_from_dataset`. These
                # must be filtered out because the distance kernels of past targets must contain only samples
                # that have been preserved in the replay buffer
                if target in plg.curr_coreset.keys():
                    # get the indices of the given target that are stored in the state of the replay buffer
                    # before calling `update_from_dataset`
                    ixs = plg.curr_coreset[target][0]
                    corr_last_gradients = []
                    for grads, indices in last_gradients:
                        mask = np.isin(indices, ixs)
                        # get the grads of only the samples stored in the state of the replay buffer before
                        # calling `update_from_dataset`
                        grads = grads[mask]
                        # get the indices of only the samples stored in the state of the replay buffer before
                        # calling `update_from_dataset`
                        indices = indices[mask]
                        corr_last_gradients.append((grads, indices))
                    last_gradients = corr_last_gradients

            # compute the distance kernel for each of the grads, indices pair in `last_gradients` and add them
            # into `curr_distance_kernels`
            for grads, indices in last_gradients:
                distance_kernel = pairwise_distances(X=grads, metric=plg.metric)
                plg.curr_distance_kernels[target].append((distance_kernel, indices))

            if plg.store_all_distance_kernels:
                # replace the distance kernels in `all_distance_kernels` with the new distance kernels
                dist_kernels = plg.curr_distance_kernels[target]
                plg.all_distance_kernels[target][-len(dist_kernels):] = dist_kernels


class InjectCoresetFromCheckpointGradientBasedReplayBufferSelectionSGDRunnerPlugin(RunnerPlugin):
    """
    This plugin allows to inject the coreset stored
    in the :class:`GradientBasedReplayBufferSelectionSGDPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
    plugin of a strategy loaded from a checkpoint into
    the :class:`GradientBasedReplayBufferSelectionSGDPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
    plugin of the strategy loaded by :meth:`Runner.run`.
    Therefore, both strategies **must** have an instance
    of :class:`GradientBasedReplayBufferSelectionSGDPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
    in the `plugins` attribute.

    The injection consists of replacing the values of the attributes `buffer_groups` and `curr_coreset` in the instance
    of :class:`GradientBasedReplayBufferSelectionSGDPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
    in the strategy loaded by :meth:`Runner.run` with the values of the respective attributes in the instance
    of :class:`GradientBasedReplayBufferSelectionSGDPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
    in the strategy loaded from the other checkpoint.

    If `store_all_coresets` is True in the instance
    of :class:`GradientBasedReplayBufferSelectionSGDPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
    in the strategy loaded by :meth:`Runner.run`, the old coreset is replaced with the new coreset in `all_coresets`.
    """

    def __init__(self, checkpoint: str):
        """
        Create a new InjectCoresetFromCheckpointGradientBasedReplayBufferSelectionSGDRunnerPlugin
        :param checkpoint: checkpoint to load
        """
        super().__init__()
        ckp = torch.load(checkpoint, pickle_module=dill, map_location="cpu")
        strategy: DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate = ckp["strategy"]
        plgs = [plg for plg in strategy.plugins if isinstance(plg,
            GradientBasedReplayBufferSelectionSGDPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate)]
        if len(plgs) == 0:
            raise RuntimeError("The strategy in the external checkpoint does not have an instance of "
                               "`GradientBasedReplayBufferSelectionSGDPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` "
                               "in the `plugins` attribute")
        coreset_plg: GradientBasedReplayBufferSelectionSGDPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate = plgs[0]
        self._buffer_groups = coreset_plg.buffer_groups  # get the buffer group from the external strategy
        self._curr_coreset = coreset_plg.curr_coreset  # get the current coreset from the external strategy

    def after_loading_checkpoint(self, runner: Runner, *args, **kwargs):
        """
        Callback called after loading a checkpoint

        Inject the values of the attributes `buffer_groups` and `curr_coreset` in the instance
        of :class:`GradientBasedReplayBufferSelectionSGDPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
        in the strategy loaded from the external checkpoint into the respective attributes
        in the instance
        of :class:`GradientBasedReplayBufferSelectionSGDPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
        in the strategy loaded by :meth:`Runner.run`.

        Additionally, if `store_all_coresets` is True in the instance
        of :class:`GradientBasedReplayBufferSelectionSGDPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
        in the strategy loaded by :meth:`Runner.run`, the old coreset is replaced with the new coreset in
        `all_coresets`.
        :param runner: a :class:`Runner` instance
        """
        plgs = [plg for plg in runner.strategy.plugins if isinstance(plg,
            GradientBasedReplayBufferSelectionSGDPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate)]
        if len(plgs) == 0:
            raise RuntimeError("The strategy does not have an instance of "
                               "`GradientBasedReplayBufferSelectionSGDPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` "
                               "in the `plugins` attribute")
        coreset_plg: GradientBasedReplayBufferSelectionSGDPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate = plgs[0]
        coreset_plg.buffer_groups = self._buffer_groups  # inject the buffer groups
        coreset_plg.curr_coreset = self._curr_coreset  # inject the current coreset
        # if `store_all_coresets` is True, replace the old coreset in `all_coresets` with the injected one
        if coreset_plg.store_all_coresets:
            # remove the old coreset for each target
            for t_coreset_list in coreset_plg.all_coresets.values():
                t_coreset_list.pop()
            # store the current coreset for each target
            coreset_plg._store_curr_coreset()


class AddAttributesGradientBasedReplayBufferSelectionSGDRunnerPlugin(RunnerPlugin):
    """
    This plugin allows to add/modify attributes in the instance
    of :class:`GradientBasedReplayBufferSelectionSGDPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
    in the `plugins` attribute of the strategy loaded by :meth:`Runner.run`.
    Therefore, the loaded strategy **must** have an instance
    of :class:`GradientBasedReplayBufferSelectionSGDPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
    in the `plugins` attribute.
    """
    def __init__(self, **kwargs):
        """
        Create a new AddAttributeGradientBasedReplayBufferSelectionSGDRunnerPlugin
        :param kwargs: some keyword arguments. The <key, value> pairs are added as attribute and value, respectively,
        to the instance of :class:`GradientBasedReplayBufferSelectionSGDPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
        in the `plugins` attribute of the strategy loaded by :meth:`Runner.run`. If a key is equal to the name of an
        attribute in the instance of :class:`GradientBasedReplayBufferSelectionSGDPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`,
        the value of that attribute is overwritten by the respective value of that key.
        """
        super().__init__()
        self._kwargs: Dict[str, Any] = kwargs

    def after_loading_checkpoint(self, runner: Runner, *args, **kwargs):
        """
        Callback called after loading a checkpoint

        Add the <key, value> pairs in `self._kwargs` as attribute and value, respectively, to the instance
        of :class:`GradientBasedReplayBufferSelectionSGDPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
        in the `plugins` attribute of the strategy loaded by :meth:`Runner.run`.
        :param runner: a :class:`Runner` instance
        """
        plgs = [plg for plg in runner.strategy.plugins if isinstance(plg,
                GradientBasedReplayBufferSelectionSGDPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate)]
        if len(plgs) == 0:
            raise RuntimeError("The strategy does not have an instance of "
                               "`GradientBasedReplayBufferSelectionSGDPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` "
                               "in the `plugins` attribute")
        coreset_plg: GradientBasedReplayBufferSelectionSGDPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate = plgs[0]
        for key, val in self._kwargs.items():
            setattr(coreset_plg, key, val)


