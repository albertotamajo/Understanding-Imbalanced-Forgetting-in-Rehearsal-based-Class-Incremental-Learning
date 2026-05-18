"""
This module includes a set of training templates
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Optional, Union, Sequence, Callable, Any, Iterable, List, Dict, Literal, Tuple, TYPE_CHECKING
from datetime import datetime


from training.protocols import (BatchObservationDynamicNetworkInitFeatureExtractorIncrementalOutlier,
                                SGDUpdateDynamicNetworkInitFeatureExtractorIncrementalOutlier,
                                SupervisedProblemDynamicNetworkInitFeatureExtractorIncrementalOutlier)

from metrics.class_accuracy import incremental_classifier_class_accuracy_per_subnetwork_metrics
from metrics.confusion_matrix import incremental_classifier_confusion_matrix_per_subnetwork_metrics
from metrics.incremental_classifier_accuracy import incremental_classifier_accuracy_per_subnetwork_metrics
from metrics.incremental_classifier_loss import incremental_classifier_loss_per_subnetwork_metrics
from metrics.loss import loss_per_subnetwork_metrics
from metrics.outlier_detector_accuracy import outlier_detector_accuracy_per_subnetwork_metrics
from metrics.outlier_detector_loss import outlier_detector_loss_per_subnetwork_metrics
from metrics.selection_accuracy import selection_accuracy_metrics
from metrics.topk_accuracy import incremental_classifier_topk_accuracy_per_subnetwork_metrics
from metrics.topk_selection_accuracy import topk_selection_accuracy_metrics
from metrics.gradient import gradient_metrics
from metrics.incremental_classifier_output import incremental_classifier_output_metrics

from training.plugins import (DynamicNetworkInitFeatureExtractorIncrementalOutlierPlugin,
                              PeriodicEvalPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate)

from benchmarks.utils.data_loader import MultiDatasetBalancedDataLoader

from avalanche.core import BasePlugin
from avalanche.training.utils import trigger_plugins
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.plugins.clock import Clock
from avalanche.benchmarks.utils.data_loader import collate_from_data_or_kwargs
from avalanche.benchmarks import CLExperience, CLStream
from avalanche.evaluation.metrics import timing_metrics
from avalanche.training.templates.base_sgd import TDatasetExperience, BaseSGDTemplate, PeriodicEval, TMBInput, TMBOutput
from avalanche.training.templates.base import (BaseTemplate, TExperienceType, _experiences_parameter_as_iterable)
from avalanche.logging import InteractiveLogger, BaseLogger, TextLogger
from avalanche.benchmarks.utils import AvalancheDataset, concat_datasets

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import Optimizer
from torch import Tensor
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from models.dynamic_networks import DynamicNetworkWithInitFeatureExtractorAndIncrementalOutlierSubNetworks
    from training.loss_functions import CrossEntropyLossMSELossLogits
    from training.logits_storage import LogitsStorageDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate


class BaseSGDTemplateInit(BaseSGDTemplate):  # TODO is there any better way to fix the issue with the original `__init__` method?
    """
    This class overrides the `__init__` method of :class:`BaseSGDTemplate` in Avalanche because there is an error in it.
    Specifically, `super().__init__()` should not be called. The rest of the original method is copied in this
    `__init__` method.
    """
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion=nn.CrossEntropyLoss(),
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: Optional[int] = 1,
        device: Union[str, torch.device] = "cpu",
        plugins: Optional[Sequence[BasePlugin]] = None,
        evaluator: Union[
            EvaluationPlugin, Callable[[], EvaluationPlugin]
        ] = default_evaluator,
        eval_every=-1,
        peval_mode="epoch",
    ):
        """Init.

        :param model: PyTorch model.
        :param optimizer: PyTorch optimizer.
        :param criterion: loss function.
        :param train_mb_size: mini-batch size for training.
        :param train_epochs: number of training epochs.
        :param eval_mb_size: mini-batch size for eval.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations. None to remove logging.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that
            `eval` is called every `eval_every` epochs and at the end of the
            learning experience.
        :param peval_mode: one of {'epoch', 'iteration'}. Decides whether the
            periodic evaluation during training should execute every
            `eval_every` epochs or iterations (Default='epoch').
        """

        BaseTemplate.__init__(self=self, model=model, device=device, plugins=plugins)

        self.optimizer: Optimizer = optimizer
        """ PyTorch optimizer. """

        self._criterion = criterion
        """ Criterion. """

        self.train_epochs: int = train_epochs
        """ Number of training epochs. """

        self.train_mb_size: int = train_mb_size
        """ Training mini-batch size. """

        self.eval_mb_size: int = train_mb_size if eval_mb_size is None else eval_mb_size
        """ Eval mini-batch size. """

        self.retain_graph: bool = False
        """ Retain graph when calling loss.backward(). """

        if evaluator is None:
            evaluator = EvaluationPlugin()
        elif callable(evaluator):
            evaluator = evaluator()

        self.plugins.append(evaluator)  # type: ignore
        self.evaluator: EvaluationPlugin = evaluator
        """ EvaluationPlugin used for logging and metric computations. """

        # Configure periodic evaluation.
        assert peval_mode in {"experience", "epoch", "iteration"}
        self.eval_every = eval_every
        peval = PeriodicEval(eval_every, peval_mode)
        self.plugins.append(peval)

        self.clock: Clock = Clock()
        """ Incremental counters for strategy events. """
        # WARNING: Clock needs to be the last plugin, otherwise
        # counters will be wrong for plugins called after it.
        self.plugins.append(self.clock)

        ###################################################################
        # State variables. These are updated during the train/eval loops. #
        ###################################################################

        self.adapted_dataset: Optional[AvalancheDataset] = None
        """ Data used to train. It may be modified by plugins. Plugins can 
        append data to it (e.g. for replay). 

        .. note::

            This dataset may contain samples from different experiences. If you 
            want the original data for the current experience  
            use :attr:`.BaseTemplate.experience`.
        """

        self.dataloader: Iterable[Any] = []
        """ Dataloader. """

        self.mbatch: Optional[TMBInput] = None
        """ Current mini-batch. """

        self.mb_output: Optional[TMBOutput] = None
        """ Model's output computed on the current mini-batch. """

        self.loss: Tensor = self._make_empty_loss()
        """ Loss of the current mini-batch. """

        self._stop_training = False


def default_evaluator_DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate() -> EvaluationPlugin:
    """
    Get the default evaluator for the template
    :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`.
    The metrics included in this default evaluator are:
        - incremental classifier class accuracy per subnetwork after each evaluation experience
        - incremental classifier confusion matrix per subnetwork after each evaluation experience
        - incremental classifier accuracy per subnetwork after each train epoch and evaluation experience
        - incremental classifier loss per subnetwork after each train minibatch, train epoch  and evaluation experience
        - loss per subnetwork after each train minibatch, train epoch and evaluation experience
        - outlier detector accuracy per subnetwork after each train epoch and evaluation experience
        - outlier detector loss per subnetwork after each train minibatch, train epoch  and evaluation experience
        - selection accuracy after each evaluation experience
        - incremental classifier top-3 accuracy per subnetwork after each evaluation experience
        - incremental classifier top-5 accuracy per subnetwork after each evaluation experience
        - elapsed time between train minibatch and train epochs
    The loggers included in this default evaluator are:
        - :class:`InteractiveLogger` if no distributed training is used
        - :class:`InteractiveLogger` and :class:`TextLogger` if distributed training is used. Each process appends text
        to a text file whose name is `log-rank{process rank}-{current time in ISO format}.txt`
    :return: the default evaluator for the template :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
    """
    if dist.is_initialized():
        # replace BaseLogger init method with a dummy one because the original one does not allow processes with rank
        # different to 0 to perform logging
        BaseLogger.__init__ = lambda self: print("")
        rank = dist.get_rank()
        time = datetime.now().isoformat(timespec="minutes").replace(":", "-")
        loggers = [InteractiveLogger(), TextLogger(open(f'log-rank{rank}-{time}.txt', 'a', buffering=1))]
    else:
        loggers = [InteractiveLogger()]
    return EvaluationPlugin(
        incremental_classifier_class_accuracy_per_subnetwork_metrics(experience=True),
        incremental_classifier_confusion_matrix_per_subnetwork_metrics(experience=True),
        incremental_classifier_accuracy_per_subnetwork_metrics(epoch=True, experience=True),
        incremental_classifier_loss_per_subnetwork_metrics(minibatch=True, epoch=True, experience=True),
        loss_per_subnetwork_metrics(minibatch=True, epoch=True, experience=True),
        outlier_detector_accuracy_per_subnetwork_metrics(epoch=True, experience=True),
        outlier_detector_loss_per_subnetwork_metrics(minibatch=True, epoch=True, experience=True),
        selection_accuracy_metrics(experience=True),
        incremental_classifier_topk_accuracy_per_subnetwork_metrics(top_k=3, experience=True),
        incremental_classifier_topk_accuracy_per_subnetwork_metrics(top_k=5, experience=True),
        topk_selection_accuracy_metrics(top_k=3, experience=True),
        topk_selection_accuracy_metrics(top_k=5, experience=True),
        timing_metrics(minibatch=True, epoch=True),
        loggers=loggers
    )


def single_subnetwork_evaluator_DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate() -> EvaluationPlugin:
    """
    Get the evaluator for the template :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` when there
    is only one subnetwork.
    The metrics included in this evaluator are:
        - incremental classifier class accuracy per subnetwork after each evaluation experience
        - incremental classifier confusion matrix per subnetwork after each evaluation experience
        - incremental classifier accuracy per subnetwork after each train epoch and evaluation experience
        - incremental classifier loss per subnetwork after each train minibatch, train epoch  and evaluation experience
        - incremental classifier top-3 accuracy per subnetwork after each evaluation experience
        - incremental classifier top-5 accuracy per subnetwork after each evaluation experience
        - elapsed time between train minibatch and train epochs
    The loggers included in this evaluator are:
        - :class:`InteractiveLogger`
    :return: the evaluator for the template :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` when
        there is only one subnetwork.
    """
    return EvaluationPlugin(
        incremental_classifier_class_accuracy_per_subnetwork_metrics(experience=True),
        incremental_classifier_confusion_matrix_per_subnetwork_metrics(experience=True),
        incremental_classifier_accuracy_per_subnetwork_metrics(epoch=True, experience=True),
        incremental_classifier_loss_per_subnetwork_metrics(minibatch=True, epoch=True, experience=True),
        incremental_classifier_topk_accuracy_per_subnetwork_metrics(top_k=3, experience=True),
        incremental_classifier_topk_accuracy_per_subnetwork_metrics(top_k=5, experience=True),
        timing_metrics(minibatch=True, epoch=True),
        loggers=[InteractiveLogger()]
    )


def single_subnetwork_evaluator_gradient_DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate() -> EvaluationPlugin:
    """
    Get the evaluator for the template :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` when there
    is only one subnetwork.
    The metrics included in this evaluator are:
        - incremental classifier class accuracy per subnetwork after each evaluation experience
        - incremental classifier confusion matrix per subnetwork after each evaluation experience
        - incremental classifier accuracy per subnetwork after each train epoch and evaluation experience
        - incremental classifier loss per subnetwork after each train minibatch, train epoch  and evaluation experience
        - incremental classifier top-3 accuracy per subnetwork after each evaluation experience
        - incremental classifier top-5 accuracy per subnetwork after each evaluation experience
        - gradient of the loss w.r.t. the input to the last layer for each sample after each training experience in
          eval mode
        - elapsed time between train minibatch and train epochs
    The loggers included in this evaluator are:
        - :class:`InteractiveLogger`
    :return: the evaluator for the template :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` when
        there is only one subnetwork.
    """
    return EvaluationPlugin(
        incremental_classifier_class_accuracy_per_subnetwork_metrics(experience=True),
        incremental_classifier_confusion_matrix_per_subnetwork_metrics(experience=True),
        incremental_classifier_accuracy_per_subnetwork_metrics(epoch=True, experience=True),
        incremental_classifier_loss_per_subnetwork_metrics(minibatch=True, epoch=True, experience=True),
        incremental_classifier_topk_accuracy_per_subnetwork_metrics(top_k=3, experience=True),
        incremental_classifier_topk_accuracy_per_subnetwork_metrics(top_k=5, experience=True),
        gradient_metrics(grad_wrt_input_last_layer=True),
        timing_metrics(minibatch=True, epoch=True),
        loggers=[InteractiveLogger()]
    )


def single_subnetwork_evaluator_gradient_output_DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate() -> EvaluationPlugin:
    """
    Get the evaluator for the template :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` when there
    is only one subnetwork.
    The metrics included in this evaluator are:
        - incremental classifier class accuracy per subnetwork after each evaluation experience
        - incremental classifier confusion matrix per subnetwork after each evaluation experience
        - incremental classifier accuracy per subnetwork after each train epoch and evaluation experience
        - incremental classifier loss per subnetwork after each train minibatch, train epoch  and evaluation experience
        - incremental classifier top-3 accuracy per subnetwork after each evaluation experience
        - incremental classifier top-5 accuracy per subnetwork after each evaluation experience
        - gradient of the loss w.r.t. the input to the last layer for each sample after each training experience in
          eval mode
        - incremental classifier outputs per subnetwork for each sample used during a training experience
        - elapsed time between train minibatch and train epochs
    The loggers included in this evaluator are:
        - :class:`InteractiveLogger`
    :return: the evaluator for the template :class:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` when
        there is only one subnetwork.
    """
    return EvaluationPlugin(
        incremental_classifier_class_accuracy_per_subnetwork_metrics(experience=True),
        incremental_classifier_confusion_matrix_per_subnetwork_metrics(experience=True),
        incremental_classifier_accuracy_per_subnetwork_metrics(epoch=True, experience=True),
        incremental_classifier_loss_per_subnetwork_metrics(minibatch=True, epoch=True, experience=True),
        incremental_classifier_topk_accuracy_per_subnetwork_metrics(top_k=3, experience=True),
        incremental_classifier_topk_accuracy_per_subnetwork_metrics(top_k=5, experience=True),
        gradient_metrics(grad_wrt_input_last_layer=True),
        incremental_classifier_output_metrics(experience_train=True),
        timing_metrics(minibatch=True, epoch=True),
        loggers=[InteractiveLogger()]
    )


class DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate(
    BatchObservationDynamicNetworkInitFeatureExtractorIncrementalOutlier,
    SupervisedProblemDynamicNetworkInitFeatureExtractorIncrementalOutlier,
    SGDUpdateDynamicNetworkInitFeatureExtractorIncrementalOutlier,
    BaseSGDTemplateInit):  # it inherits from a modified version of BaseSGDTemplate, which fixes the error in the `__init__` method
    """
    Base class for continual learning skeletons
    of :class:`DynamicNetworkWithInitFeatureExtractorAndIncrementalOutlierSubNetworks`.

    This template supports distributed training. Multiple distributed training processes can be spawned using
    `torch.distributed.run` or `torch.distributed.launch`. The basic script for using them for single-node multi-worker
    distributed training is::
        torchrun
            --standalone
            --nnodes=1
            --nproc-per-node=$NUM_TRAINERS
            YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)
        OR
        python -m torch.distributed.launch
            --use-env
            --nnodes=1
            --nproc-per-node=$NUM_TRAINERS(GPUs)
            YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)

    Note that `torch.distributed.launch` is now deprecated in favour of `torch.run`.
    In your training program, you must read from the LOCAL_RANK environment variable as demonstrated by the
    following code snippet::
        import os
        local_rank = int(os.environ["LOCAL_RANK"])
    If your training program uses GPUs, you should ensure that your code only runs on the GPU device of
    LOCAL_PROCESS_RANK. This can be done by::
        torch.cuda.set_device(local_rank)  # before your code runs

    Refer to `https://pytorch.org/docs/stable/elastic/run.html#launcher-api` for more details about how to start
    distributed training.


    **Training loop**
    The training loop is organized as follows::

        train
            before_training  # only once at the beginning of the training step
                # for each experience
                before_train_dataset_adaptation
                train_dataset_adaptation
                after_train_dataset_adaptation
                before_train_datasets_adaptation
                model_adaptation
                make_optimizer
                train_datasets_adaptation
                after_train_datasets_adaptation
                before_train_dataloader
                make_train_dataloader
                after_train_dataloader
                before_training_exp
                    # for each subnetwork that must be trained on the experience
                    before_training_subnetwork
                        # for each epoch
                        before_training_epoch
                            # for each iteration
                            before_training_iteration
                            before_forward
                            forward
                            after_forward
                            before_backward
                            backward
                            after_backward
                            before_update
                            optimizer_step
                            after_update
                            after_training_iteration
                        after_training_epoch
                    after_training_subnetwork
                after_training_exp
            after_training

    **Evaluation loop**
    The evaluation loop is organized as follows::

        eval
            before_eval  # only once at the beginning of the evaluation step
                # for each experience
                before_eval_dataset_adaptation
                eval_dataset_adaptation
                after_eval_dataset_adaptation
                make_eval_dataloader
                model_adaptation
                before_eval_exp
                    eval_epoch  # we have a single epoch in evaluation mode
                        # for each iteration
                        before_eval_iteration
                        before_eval_forward
                        after_eval_forward
                        after_eval_iteration
                after_eval_exp
            after_eval

    .note::
        When calling :meth:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate.eval` at the end of the
        training process on the current experience to evaluate on the current training/eval experience and/or all the
        previous training/eval experiences, only one process should perform the actual evaluation process if running a
        distributed training because otherwise each process in the distributed training process would perform the same
        redundant operations. For consistency, only the process with rank 0 should be used. Consequently, only the
        process with rank 0 must be used to save checkpoints.

    .note::
        If running a distributed training, when loading a single checkpoint from different processes, each process
        must use the utility  :func:`avalanche.training.checkpoint.maybe_load_checkpoint` by providing as parameter
        to the argument `map_location` "cuda:{rank of this process}"

    .note::
        If running a distributed training, the processes collect the more updated metrics from all the other processes
        during the training step without moving the internal tensor representations of each metric into the gpu device
        allocated to the receiving process. This, generally, should not be an issue, unless the metrics are aggregated
        at run time for analysis by a process. However, when loading a checkpoint with the utility
        :func:`avalanche.training.checkpoint.maybe_load_checkpoint` by providing an appropriate device
        as parameter to the argument `map_location`, all the metrics are mapped to the same device

    .note::
        The `train_dataset_adaptation` and `eval_dataset_adaptation` functions update the `use_in_getitem_indices`
        dictionary for each experience so to know which index must be used to retrieve the values of a given
        DataAttribute when a mini-batch is fetched at train and eval time, respectively. It is highly recommended that
        the datasets of all the training experiences and eval experiences of a given stream have the same
        DataAttributes to avoid undesired behaviour.


    .warning::
        If running a distributed process and all processes are started with the same random seed for each random number
        generator (NumPy, Torch, Python, etc...), which can be done straightforwardly  by using
        `RNGManager.set_random_seeds(seed)`, the processes will be in the same random state at any given time unless
        random operations are performed between `before_training_subnetwork` and `after_training_subnetwork`, both
        included. This happens because each process will train a different set of subnetworks.

    .warning::
        If running a distributed training, the script of each process must set the CUDA device used by
        the process by providing the rank of the process to `torch.cuda.set_device()`. The CUDA device must be set
        before calling the method :meth:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate.train` of this
        template. Preferably, it would be better to set the CUDA device ID before instantiating an instance of this
        template. The same CUDA device *must* be used when creating an instance of this template by setting the
        `device` parameter in :meth:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate.__init__` accordingly.

    .warning::
        When running a distributed training, ensure that all processes use the same random seeds. Otherwise,
        for example, each process might receive a different order of experiences.

    .warning::
        The ID of each subnetwork in :class:`DynamicNetworkWithInitFeatureExtractorAndIncrementalOutlierSubNetworks`
        must be an int wrapped into a string, such as "0" instead of 0

    .warning::
        When using this template, the attribute
        :attr:`DynamicNetworkWithInitFeatureExtractorAndIncrementalOutlierSubNetworks.init_feature_extractor_eval` must
        be always True. During instantiation, this template checks whether this attribute is set to True.

    TODO SOLVE THE USE_INIT_FEATURE_EXTRACTOR PROBLEM
    """
    PLUGIN_CLASS = DynamicNetworkInitFeatureExtractorIncrementalOutlierPlugin

    def __init__(self,
                 model: DynamicNetworkWithInitFeatureExtractorAndIncrementalOutlierSubNetworks,
                 optimizer: Optimizer,
                 criterion=nn.CrossEntropyLoss(),
                 outlier_criterion_weight: float = 1.,
                 train_mb_size: int = 1,
                 train_epochs: int = 1,
                 eval_mb_size: Optional[int] = 1,
                 device: Union[str, torch.device] = "cpu",
                 plugins: Optional[Sequence[BasePlugin]] = None,
                 evaluator: Union[
                     EvaluationPlugin, Callable[[], EvaluationPlugin]
                 ] = default_evaluator_DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
                 eval_every: int = -1,
                 peval_mode: Literal["epoch", "iteration"] = "epoch",
                 do_initial: bool = False
                 ):
        """
        Create a new DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate
        :param model: a :class:`DynamicNetworkWithInitFeatureExtractorAndIncrementalOutlierSubNetworks` model.
            The attribute :attr:`DynamicNetworkWithInitFeatureExtractorAndIncrementalOutlierSubNetworks.init_feature_extractor_eval`
            must be True. Otherwise, a ValueError is raised.
        :param optimizer: a PyTorch optimizer. The optimizer can be initialised with any model parameters. During the
            training procedure an optimizer identical to this will be used for each subnetwork with the appropriate
            learnable parameters.
        :param criterion: loss function to be used for the incremental classifier of each subnetwork.
            If :class:`CrossEntropyLossMSELossLogits`, the output of the incremental classifier of each subnetwork
            *must* be raw logits before passing though a softmax layer and an instance
            of :class:`LogitsStorageDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` *must* be provided
            within the sequence of plugins in `plugins`. Such an instance is retrieved by invoking the `logits_storage`
            property of this strategy. If no such a plugin exists in the `plugins` attribute of this strategy,
            a :class:`RuntimeError` is raised if `logits_storage` is invoked.
            The logit storage instance in `plugins` *must* store logits for the samples of all types of experiences
            (train, val or test) that this strategy will encounter. Otherwise, a :class:`ValueError` is raised by the
            logit storage when its public interface is invoked on the samples of those experiences.
            Furthermore, the "dataset_indices" DataAttribute *must* be present in the datasets of those experiences
            encountered by this strategy. Note that this strategy interacts with the public interface of the logit
            storage instance in `plugins` through some properties defined
            in :class:`SupervisedProblemDynamicNetworkInitFeatureExtractorIncrementalOutlier`.
            Default is :class:`nn.CrossEntropyLoss`. By default, the loss function to be used for the outlier detector
            of each subnetwork is the binary cross entropy. It can be changed by accessing the attribute
            `_outlier_criterion`. However, note that the outlier criterion is expected to be a criterion that requires
            the target tensor to be of the same shape as the input tensor.
        :param outlier_criterion_weight: (optional) weight to be used for the outlier detector loss function.
            Default is 1.
            The loss function of each subnetwork is equal to `criterion(output, targets) +
            outlier_criterion_weight * nn.BCEWithLogitsLoss(output, targets)`
        :param train_mb_size: (optional) mini-batch size for training. Default is 1.
        :param train_epochs: (optional) number of training epochs. Default is 1.
        :param eval_mb_size: (optional) mini-batch size for eval. default is 1.
        :param device: (optional) device used for the training and evaluation steps. Default is `cpu`.
        :param plugins: (optional) sequence of plugins that add specific behaviour to this template. Default is `None`
        :param evaluator: (optional) instance of :class:`EvaluationPlugin` for logging and metric computations.
            None to remove logging. The default evaluator is the :class:`EvaluationPlugin` instance output
            of :func:`default_evaluator_DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`
        :param eval_every: the frequency of the calls to `eval` inside the training loop of the currently trained
            subnetwork. -1 disables the evaluation. 0 means `eval` is called only at the end of the learning loop.
            Values > 0 mean that `eval` is called every `eval_every` epochs or iterations according to `peval_mode`
            and at the end of the learning loop. Default is `-1`.
        :param peval_mode: one of {'epoch', 'iteration'}. Decides whether the periodic evaluation during training
            should execute every `eval_every` epochs or iterations. Default is 'epoch'
        :param do_initial: (optional) whether to perform a periodic evaluation before the training loop of a subnetwork
            at the start of each experience.
            This parameter is used by
            the :class:`PeriodicEvalPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate`.
            Default is `False`.
        """
        super().__init__()
        BaseSGDTemplateInit.__init__(self, model, optimizer, criterion, train_mb_size, train_epochs, eval_mb_size,
                                    device, plugins, evaluator, eval_every, peval_mode)
        if not self.model.init_feature_extractor_eval:
            raise ValueError("Attribute `init_feature_extractor_eval` of the model must be True when using "
                             "this template")
        # Replace the PeriodicEval plugin placed inside plugins by BaseSGDTemplateInit with the periodicEval plugin of
        # this template. The old PeriodicEval plugin is placed in the penultimate index
        self.plugins[-2] = PeriodicEvalPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate(eval_every,
                                                                                                          peval_mode,
                                                                                                          do_initial=do_initial)
        self.target_to_target: Dict[Any, Any] = {}
        """
        A dictionary containing targets as keys and targets as values. This dictionary is used by the `mb_y` property
        to return the targets contained in the current mini-batch. If a target in the current mini-batch is present in
        this dictionary as a key, `mb_y` replaces the target with its respective value in this dictionary. It is an
        empty dictionary by default.  
        """

        self.adapted_dataset: Optional[Dict[str, AvalancheDataset], AvalancheDataset] = None
        """
        A dictionary containing subnetwork IDs as keys and corresponding training datasets as values during training.
        A single dataset during evaluation. The training dataset contains the samples of classes a subnetwork must be
        trained on in the new experience.
        """

        self.dataloader: Union[Dict[str, Iterable[Any]], Iterable[Any]] = {}
        """
        A dictionary containing the subnetwork IDs as keys and the corresponding dataloaders as values during the
        training step. A dataloader during the evaluation step.
        """

        self.optimizer: Dict[str, Optimizer] = {}
        """A dictionary containing the subnetwork IDs as keys and the corresponding optimizers as values"""

        self.optimizer_type: Optimizer = optimizer
        """
        A PyTorch optimizer. During the training procedure an optimizer identical to this will be used for each
        subnetwork with the appropriate learnable parameters
        """

        self.curr_epoch: Optional[int] = None
        """Current training epoch"""

        self.curr_sub_id: Optional[str] = None
        """ID of the subnetwork that is being trained or evaluated"""

        self.curr_train_dataloader: Iterable[Any] = []
        """Dataloader of the subnetwork currently being trained"""

        self._outlier_criterion = nn.BCEWithLogitsLoss()
        """Loss function for the outlier detector of each subnetwork"""

        self.outlier_criterion_weight: float = outlier_criterion_weight
        """Weight used for the outlier detector loss function"""

        self.loss: Union[Dict[str, Tensor], Tensor] = self._make_empty_loss()
        """ Loss tensor of the subnetwork currently being trained or a dictionary containing the
        subnetwork IDs as keys and corresponding loss tensors as values during the evaluation step
        """

        self.incremental_classifier_loss: Union[Dict[str, Tensor], Tensor] = self._make_empty_loss()
        """
        Incremental classifier loss tensor of the subnetwork currently being trained or a dictionary containing the
        subnetwork IDs as keys and corresponding incremental classifier loss tensors as values during the evaluation
        step
        """

        self.outlier_detector_loss:  Union[Dict[str, Tensor], Tensor] = self._make_empty_loss()
        """
        Outlier detector loss tensor of the subnetwork currently being trained or a dictionary containing the
        subnetwork IDs as keys and corresponding outlier detector loss tensors as values during the evaluation step.
        """

        self.mb_output: Optional[Dict[str, Dict[str, Union[Tensor, Tuple[Tensor, Tensor]]]]] = None
        """
        mini-batch output. A dictionary containing one or multiple subnetwork IDs as keys and corresponding output
        dictionaries as values. An output dictionary is a dictionary of the following form
        `{"incremental_classifier": (output tensor1, output tensor2), "outlier_detector": output tensor}`, where
        output tensor1 is the output of the incremental classifier (the logits) while output tensor2 is the
        feature vector, also known as embedding, that precedes the final linear classifier layer in the incremental
        classifier.
        """

        self.use_in_getitem_indices: Dict[str, int] = {}
        """
        a dictionary containing the name of DataAttribute instances that have use_in_getitem=True as keys and the
        corresponding indices as values. This dictionary is updated by the `train_dataset_adaptation` and
        `eval_dataset_adaptation` functions so to know which index must be used to retrieve the values of a given
        DataAttribute when a mini-batch is fetched at train and eval time, respectively. If the dictionary does not
        contain the name of a DataAttribute, then that DataAttribute is not present in the dataset of the current
        experience or does not have use_in_getitem=True. The indices are negative integers because of the way
        mini-batches are fetched.
        When a mini-batch is fetched, __getitem__ is called on the underlying dataset and then the values of the
        DataAttribute instances with use_in_getitem=True are appended. Using negative indexes ensures that the correct
        values of a given DataAttribute are retrieved regardless of the number of items that the __getitem__ function of
        the underlying dataset returns.  
        """

    def train(
            self,
            experiences: Union[TDatasetExperience, Iterable[TDatasetExperience]],
            eval_streams: Optional[
                Sequence[Union[TDatasetExperience, Iterable[TDatasetExperience]]]
            ] = None,
            **kwargs
    ):
        """
        Training loop.

        If experiences is a single element trains on it.
        If it is a sequence, trains the model on each experience in order.
        This is different from joint training on the entire stream.
        It returns a dictionary with last recorded value for each metric. If running a distributed training, all the
        processes will return the dictionary with the last recorded value for each metric, including the metrics
        recorded by all the other processes.

        .note::
            If running a distributed training, all the processes will collect at the end of the training process
            the dictionaries with all recorded metrics by the other processes. This way all the processes share the same
            metrics state.
            However, only the process with rank 0 must be used to save the state of this template at each checkpoint
            with the utility
            :fun:`avalanche.training.checkpoint.save_checkpoint` because only the process with rank 0 has the metrics
            about the evaluation phase.
            When loading this template state from different
            processes, this single checkpoint file needs to be used by using the utility
            :func:`avalanche.training.checkpoint.maybe_load_checkpoint` by providing as parameter to the argument
            `map_location` "cuda:{rank of this process}"

        .warning::
            If running a distributed training, the processes collect the more updated metrics from all the other processes
            without moving the internal tensor representations of each metric into the process gpu device. This,
            generally, should not be an issue, unless the metrics are aggregated at run time for analysis. When loading
            a checkpoint with the utility
            :func:`avalanche.training.checkpoint.maybe_load_checkpoint` by providing an appropriate device
            as parameter to the argument `map_location`, all the metrics are mapped to the same device

        :param experiences: single Experience or sequence.
        :param eval_streams: sequence of streams for evaluation. This sequence is used by
            the :class:`PeriodicEvalPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate` for periodic
            evaluation. If None: use training experiences for evaluation. Use [] if you do not want to evaluate during
            training. Experiences in `eval_streams` are grouped by stream name when calling `eval`. If you use multiple
            streams, they must have different names.
        """
        last_metrics = super().train(experiences, eval_streams, **kwargs)
        if dist.is_initialized():  # if running a distributed training
            world_size = dist.get_world_size()  # number of processes in the current distributed training
            if world_size > 1:  # if there are more than one process in the distributed training
                # Since it might occur that one process finishes the training much
                # earlier than the other processes and I do not know whether `dist.all_gather_object()` has
                # any timeouts, I avoid this problem by using `dist.barrier()`, which blocks all processes until the
                # whole group enters this function.
                dist.barrier()
                rank = dist.get_rank()
                metrics = (rank, {"all_metrics": self.evaluator.get_all_metrics(), "last_metrics": last_metrics})
                other_metrics = [None for _ in range(dist.get_world_size())]
                dist.all_gather_object(object_list=other_metrics, obj=metrics)
                for other_metric in other_metrics:
                    if other_metric[0] != rank:  # if this metric pair is not the one provided by this process
                        other_metric_dict = other_metric[1]
                        other_metric_all_metrics = other_metric_dict["all_metrics"]
                        other_metric_last_metrics = other_metric_dict["last_metrics"]
                        for key_metric, val in other_metric_last_metrics.items():
                            # if this key metric is not in the last_metrics dictionary of this process, then add it
                            if key_metric not in self.evaluator.get_last_metrics().keys():
                                self.evaluator.last_metric_results[key_metric] = val
                        for key_metric, val in other_metric_all_metrics.items():
                            # if this metric has never been used by this process, then add it. Otherwise, if this metric
                            # is in this process, replace the value of this process for this metric if the other process
                            # has more updated data for this metric
                            if ((key_metric not in self.evaluator.all_metric_results.keys())
                                    or (key_metric in self.evaluator.all_metric_results.keys() and
                                        len(val[0]) > len(self.evaluator.all_metric_results[key_metric][0]))):
                                self.evaluator.all_metric_results[key_metric] = val
                last_metrics = self.evaluator.get_last_metrics()  # update the last_metrics var with the current state
        return last_metrics

    def _before_training_exp(self, **kwargs):
        """
        Setup to train on a single experience.

        The structure of this setup is:

        - before_train_dataset_adaptation
        - train_dataset_adaptation
        - after_train_dataset_adaptation
        - before_train_datasets_adaptation
        - model_adaptation
        - make_optimizer
        - train_datasets_adaptation
        - after_train_datasets_adaptation
        - before_train_dataloader
        - make_train_dataloader
        - after_train_dataloader
        - before_training_exp
        """
        # Data Adaptation (e.g. add new samples/data augmentation)
        self._before_train_dataset_adaptation(**kwargs)
        self.train_dataset_adaptation(**kwargs)
        self._after_train_dataset_adaptation(**kwargs)

        self._before_train_datasets_adaptation(**kwargs)
        # Model Adaptation (e.g. freeze/add new units)
        self.check_model_and_optimizer(**kwargs)
        self.train_datasets_adaptation(**kwargs)
        self._after_train_datasets_adaptation(**kwargs)

        self._before_train_dataloader(**kwargs)
        self.make_train_dataloader(**kwargs)
        self._after_train_dataloader(**kwargs)

        BaseTemplate._before_training_exp(self, **kwargs)

    def train_dataset_adaptation(self, **kwargs):
        """
        Initialise the `adapted_dataset` attribute and update the `use_in_getitem_indices` attribute accordingly
        """
        super().train_dataset_adaptation(**kwargs)
        self._update_use_in_getitem_indices()

    def _update_use_in_getitem_indices(self):
        """
        Update the dictionary stored in the `use_in_getitem_indices` attribute according to the
        dataset stored in the `adapted_dataset` attribute.

        The name of the DataAttributes with use_in_getitem=True in the dataset of the `adapted_dataset` attribute along
        with their indices for the retrieval of their values on the mini-batches are stored in the
        `use_in_getitem_indices` attribute. The indices are negative integers because of the way mini-batches are
        fetched. When a mini-batch is fetched, __getitem__ is called on the underlying dataset and then the values of
        the DataAttribute instances with use_in_getitem=True are appended. Using negative indexes ensures that the
        correct values of a given DataAttribute are retrieved regardless of the number of items that the __getitem__
        function of the underlying dataset returns.
        """
        data_attributes: List[str] = []
        # looping over all DataAttribute instances in the dataset of the current experience
        for data_attribute in self.adapted_dataset._data_attributes.values():
            if data_attribute.use_in_getitem:  # if use_in_getitem=True for the current DataAttribute instance
                data_attributes.append(data_attribute.name)
        self.use_in_getitem_indices = dict(zip(data_attributes, range(-len(data_attributes), 0)))

    def train_datasets_adaptation(self, **kwargs):
        """
        Set up the training dataset for each subnetwork for the current experience.

        The training datasets are stored in `self.adapted_dataset` as a dictionary containing the subnetwork IDs as
        keys and the respective avalanche datasets as values. The dataset of each subnetwork has the
        `targets_task_labels` DataAttribute set to the respective subnetwork ID.

        .note::
            The ID of each subnetwork is a string containing an integer number only. However, the `targets_task_labels`
            DataAttribute of the dataset of each subnetwork stores the integer number in the form of an int not wrapped
            into a string

        .note::
            Not all subnetworks are trained on a new experience; it depends on whether the new experience
            contains classes that match with a given subnetwork
        """
        # a dictionary containing the subnetwork IDs as keys and all the corresponding classes seen from each
        # subnetwork as values
        subnetworks_classes = self.model.subnetworks_classes
        curr_classes = self.experience.classes_in_this_experience
        # include only the classes that are in the current experience
        subnetworks_classes = {sub_id: [cls for cls in classes if cls in curr_classes]
                               for sub_id, classes in subnetworks_classes.items()}
        # remove the subnetworks that do not undertake training in this experience
        subnetworks_classes = {sub_id: classes for sub_id, classes in subnetworks_classes.items() if len(classes) != 0}
        # create avalanche datasets for each subnetwork
        adapted_datasets = {sub_id: self.adapted_dataset.subset(
            [i for i, cls in enumerate(self.adapted_dataset.targets.data) if int(cls) in classes])
            for sub_id, classes in subnetworks_classes.items()}
        # change the targets_task_labels DataAttribute of each sample in the dataset of each subnetwork to the
        # corresponding subnetwork ID. The ID of each subnetwork is a string containing an integer number. The
        # targets_task_labels DataAttribute contains the respective integer number not the string.
        adapted_datasets = {sub_id: dataset.update_data_attribute("targets_task_labels",
                                                                  [int(sub_id) for _ in range(len(dataset))])
                            for sub_id, dataset in adapted_datasets.items()}
        self.adapted_dataset = adapted_datasets

    def make_train_dataloader(self, num_workers=0, shuffle=True, pin_memory=None, persistent_workers=False,
                              drop_last=False, termination_dataset=-2, oversample_small_datasets=False,
                              balance_large_datasets=True, **kwargs):
        """
        Data loader initialization. Called at the start of each learning experience after the dataset adaptation.

        For each subnetwork that undertakes training on the new experience, a :class:`MultiDatasetBalancedDataLoader`
        is created. This dataloader balances the data in each mini-batch from two datasets: the dataset the respective
        subnetwork must be trained on and all the other subnetworks' datasets. The first dataset is used
        for training the incremental classifier of the given subnetwork while both of them are used for the training of
        the outlier detector. The datasets are taken from `self.adapted_dataset`.

        The training dataloaders are stored in `self.dataloader` as a dictionary containing the subnetwork IDs as
        keys and the respective dataloaders as values.
        :param num_workers: (optional) number of thread workers for the data loading.
        :param shuffle: (optional) True if the data should be shuffled, False otherwise. Default is True.
        :param pin_memory: (optional) If True, the data loader will copy Tensors into CUDA
            pinned memory before returning them. Defaults to True.
        :param persistent_workers: (optional) If True, the data loader will not shut down the worker processes after a dataset
            has been consumed once. This allows to maintain the workers Dataset instances alive. Defaults to False
        :param drop_last: (optional) set to True to drop the last incomplete batch, if the dataset size is not divisible by the
            batch size. If False and the size of dataset is not divisible by the batch size, then the last batch will be
            smaller. Defaults to False
        :param termination_dataset: (optional) an integer number denoting the index of the dataset to be used for
            determining when to stop iterating (the iteration is stopped when the end of the respective dataset is hit).
            0 for stopping iterating when the end of the dataset the respective subnetwork must be trained on is hit.
            1 for stopping iterating when the end of the concatenation of all the other subnetworks' datasets is hit.
            -1 for stopping iterating when the end of the larger dataset is hit.
            -2 for stopping iterating when the end of the smaller dataset is hit.
            Default is -2.
        :param oversample_small_datasets: (optional) if True, the smaller dataset is oversampled to match the
            `termination_dataset`. Otherwise, once data from a dataset is completely iterated, the dataset will be
            skipped in the subsequent mini-batches. Default is False.
        :param balance_large_datasets: (optional) if True, the larger dataset is randomly reduced in size to match the
            `termination_dataset` while ensuring that each class has an identical or as similar as possible
            number of samples. Note that if True, the larger dataset is randomly reduced in size every
            time `__iter__` is called. Therefore, a different smaller dataset is created out of the larger dataset
            in each epoch. If False, the larger dataset is not reduced in size and the
            samples used for it in each dataloader iteration depends on the argument `shuffle`. If `shuffle` is False
            and `balance_large_datasets` is False, the same samples of the larger dataset are used for each dataloader
            iteration. Default is True.
        """
        if self.adapted_dataset is None:
            raise RuntimeError("It is impossible to build train dataloaders if the adapted dataset is None")

        other_dataloader_args = self._obtain_common_dataloader_parameters(
            batch_size=self.train_mb_size,
            num_workers=num_workers,
            shuffle=shuffle,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            drop_last=drop_last,
        )
        dataloader = {}
        for sub_id, dataset in self.adapted_dataset.items():
            outlier_dataset = concat_datasets([dataset2 for sub_id2, dataset2 in self.adapted_dataset.items()
                                               if sub_id2 != sub_id])  # concatenate all the other subnetworks' datasets
            if len(outlier_dataset) == 0:  # if empty
                collate_from_data_or_kwargs(dataset, other_dataloader_args)
                dataloader[sub_id] = DataLoader(dataset, **other_dataloader_args)
            else:
                dataloader[sub_id] = MultiDatasetBalancedDataLoader(datasets=[dataset, outlier_dataset],
                                                                    termination_dataset=termination_dataset,
                                                                    oversample_small_datasets=oversample_small_datasets,
                                                                    balance_large_datasets=balance_large_datasets,
                                                                    distributed_sampling=False,
                                                                    **other_dataloader_args)
        self.dataloader = dataloader

    def _train_exp(self, experience: CLExperience, eval_streams=None, **kwargs):
        """
        Training loop over a single experience.

        The structure of the training loop is:
            - before_training_subnetwork

                - before_training_epoch
                - training_epoch
                - after_training_epoch

            - after_training_subnetwork

        .note::
            If running a distributed training, this method distributes the
            training steps that must be carried out for this experience (one training step per subnetwork that must
            undertake training on the current experience) evenly over all processes in the distributed group. After all the
            training steps, all the processes converge to the same state by
            exchanging the latest weights of a given subnetwork and the respective optimizer state.
        :param experience: CL experience information.
        :param eval_streams: list of streams for evaluation.
            If None: use the training experience for evaluation.
            Use [] if you do not want to evaluate during training.
        :param kwargs: custom arguments.
        """
        sub_ids = self._get_sub_ids_train_exp()  # the IDs of the subnetworks that this process must train

        for self.curr_sub_id, self.curr_train_dataloader in self.dataloader.items():
            if self.curr_sub_id in sub_ids:  # if this process can train the current subnetwork
                self._before_training_subnetwork(**kwargs)
                for self.curr_epoch in range(self.train_epochs):
                    self._before_training_epoch(**kwargs)

                    if self._stop_training:  # Early stopping
                        self._stop_training = False
                        break

                    self.training_epoch(**kwargs)
                    self._after_training_epoch(**kwargs)

                self._after_training_subnetwork(**kwargs)

        self._load_subnetwork_and_optimizer_states_train_exp(sub_ids=sub_ids)

    def _get_sub_ids_train_exp(self):
        """
        Get the IDs of the subnetworks that the current process must train for the current experience.

        If running a distributed training, the training of subnetworks that need to learn from the current experience
        is evenly distributed across all processes in the distributed group. Otherwise, this process will perform
        the training of all subnetworks.
        :return: the subnetwork IDs that the current process must train
        """
        sub_ids = list(self.dataloader.keys())  # list of the ids of all subnetworks that need to undertake training
        sub_ids.sort()   # the list of IDs is sorted to ensure that every process has a list with the same order
        if dist.is_initialized():  # if running a distributed training
            world_size = dist.get_world_size()  # number of processes in the current distributed training
            if world_size > 1:  # if there are more than one process in the distributed training
                rank = dist.get_rank()  # rank of the current process
                # if there is fewer subnetworks to train than the number of processes or an equal number
                if len(sub_ids) <= world_size:
                    if rank < len(sub_ids):
                        sub_ids = [sub_ids[rank]]
                    else:
                        sub_ids = []
                else:  # if there is more subnetworks to train than the number of processes
                    sub_ids = [sub_ids[i] for i in range(rank, len(sub_ids), world_size)]
        return sub_ids

    def _load_subnetwork_and_optimizer_states_train_exp(self, sub_ids: List[str]):
        """
        If running a distributed training, load the state dictionary of the subnetworks and the state dictionary
        of their respective optimizers that the other processes in this distributed training group have carried
        training on/with for the current experience. This method allows this process to get the latest weights of the
        subnetworks that have been trained on the other processes. It also allows this process to get the latest states
        of the optimizers that have been used by the other processes for training the other subnetworks.

        .note::
            This method is called at the end of the training process on a new experience by
            :meth:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate._train_exp`. This way all the processes
            in the current distributed training converge to the same state.
        :param sub_ids: list of the IDs of the subnetworks that this process has carried training on for the current
            experience
        :return: None
        """
        if dist.is_initialized():  # if running a distributed training
            world_size = dist.get_world_size()  # number of processes in the current distributed training
            if world_size > 1:  # if there are more than one process in the distributed training
                # collect the state dictionary of the subnetworks trained on this process and their
                # respective optimizers.
                states_dictionary = {}
                for sub_id in sub_ids:
                    subnet_state_dict = self.model.subnetworks[sub_id].state_dict()
                    opt_state_dict = self.optimizer[sub_id].state_dict()
                    # update the `states_dictionary` of this process which will later be broadcast
                    # to all other processes
                    states_dictionary[sub_id] = {"subnet_state_dict": subnet_state_dict, "opt_state_dict": opt_state_dict}

                # Since it might occur that one process finishes the training on the current experience much
                # earlier than the other processes and I do not know whether `dist.all_gather_object()` has
                # any timeouts, I avoid this problem by using `dist.barrier()`, which blocks all processes until the
                # whole group enters this function.
                dist.barrier()

                other_states_dictionary = [None for _ in range(dist.get_world_size())]
                # after dist.all_gather_object(), other_states_dictionary is populated with the dictionary of
                # each process,
                # including the dictionary of this process
                dist.all_gather_object(object_list=other_states_dictionary, obj=states_dictionary)
                for other_state_dict in other_states_dictionary:
                    for sub_id, sub_state_dict in other_state_dict.items():
                        if sub_id in sub_ids:  # if this dictionary is the dictionary built by this process then skip it
                            break
                        # pass all the tensors in the state dict of this subnetwork from cpu to the gpu of this process
                        gpu_sub_state_dict = OrderedDict([(key, value.to(self.device))
                                                          for key, value in sub_state_dict["subnet_state_dict"].items()])
                        # load the state dictionary of the network
                        self.model.subnetworks[sub_id].load_state_dict(gpu_sub_state_dict)
                        # load the state dictionary of the optimizer
                        self.optimizer[sub_id].load_state_dict(sub_state_dict["opt_state_dict"])

    def optimizer_step(self):
        """Execute the optimizer step (weights update)."""
        self.optimizer[self.curr_sub_id].step()

    def _save_train_state(self) -> Dict[str, Any]:
        """
        Save the training state which may be modified by the eval loop.
        This currently includes: experience, adapted_dataset, dataloader,
        is_training, train/eval modes for each module, curr_sub_id, curr_epoch, use_in_getitem_indices, mbatch,
        mb_output, loss, incremental_classifier_loss, outlier_detector_loss
        """
        state = super()._save_train_state()
        new_state = {
            "curr_sub_id": self.curr_sub_id,
            "curr_epoch": self.curr_epoch,
            "use_in_getitem_indices": self.use_in_getitem_indices,
            "mbatch": self.mbatch,
            "mb_output": self.mb_output,
            "loss": self.loss,
            "incremental_classifier_loss": self.incremental_classifier_loss,
            "outlier_detector_loss": self.outlier_detector_loss
        }
        return {**state, **new_state}

    def _load_train_state(self, prev_state: Dict[str, Any]):
        """
        Load the training state from a dictionary.
        :param prev_state: dictionary of the previous state
        :return: None
        """
        super()._load_train_state(prev_state)
        self.curr_sub_id = prev_state["curr_sub_id"]
        self.curr_epoch = prev_state["curr_epoch"]
        self.use_in_getitem_indices = prev_state["use_in_getitem_indices"]
        self.mbatch = prev_state["mbatch"]
        self.mb_output = prev_state["mb_output"]
        self.loss = prev_state["loss"]
        self.incremental_classifier_loss = prev_state["incremental_classifier_loss"]
        self.outlier_detector_loss = prev_state["outlier_detector_loss"]

    def save_train_state(self) -> Dict[str, Any]:
        """
        Save the training state which may be modified by the eval loop.
        This currently includes: experience, adapted_dataset, dataloader,
        is_training, train/eval modes for each module, curr_sub_id, curr_epoch, use_in_getitem_indices, mbatch,
        mb_output, loss, incremental_classifier_loss, outlier_detector_loss.

        This method exposes publicly the protected
        method :meth:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate._save_train_state`
        """
        return self._save_train_state()

    def load_train_state(self, prev_state: Dict[str, Any]):
        """
        Load the training state from a dictionary.

        This method exposes publicly the protected
        method :meth:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate._load_train_state`
        :param prev_state: dictionary of the previous state
        :return: None
        """
        self._load_train_state(prev_state=prev_state)

    def _train_cleanup(self):
        """
        Clean up the state of this template after the training step
        :return: None
        """
        super()._train_cleanup()
        # reset for faster serialization
        self.dataloader = {}
        self.curr_sub_id = None
        self.curr_epoch = None
        self.curr_train_dataloader = []
        self.incremental_classifier_loss = self._make_empty_loss()
        self.outlier_detector_loss = self._make_empty_loss()
        self.use_in_getitem_indices = {}

    def _eval_cleanup(self):
        """
        Clean up the state of this template after the evaluation step
        :return: None
        """
        super()._eval_cleanup()
        # reset for faster serialization
        self.dataloader = {}
        self.curr_sub_id = None
        self.incremental_classifier_loss = self._make_empty_loss()
        self.outlier_detector_loss = self._make_empty_loss()
        self.use_in_getitem_indices = {}

    @torch.no_grad()
    def eval(self, exp_list: Union[CLExperience, CLStream], **kwargs):
        """
        Evaluate the current model on a series of experiences and return the last recorded value for each metric.

        .note::
            When calling this method at the end of the training process on the current experience to evaluate on the
            current training/eval experience and/or all the previous training/eval experiences, only one process should
            perform the actual evaluation process if running a distributed training because otherwise each process in
            the distributed training process would perform the same redundant operations. For consistency, only the
            process with rank 0 should be used.

        :param exp_list: CL experience information.
        :param kwargs: custom arguments.

        :return: dictionary containing last recorded value for each metric name.
        """
        return super().eval(exp_list, **kwargs)

    def eval_with_grad(self, exp_list: Union[CLExperience, CLStream], **kwargs):
        """
        Evaluate the current model on a series of experiences and return the last recorded value for each metric.

        .note::
            This method behaves identically to `self.eval` with the only difference that it is not wrapped with the
            `torch.no_grad` decorator. Therefore, when evaluation needs to be performed but gradients need to be
            tracked as well, this method should be called instead of `self.eval`.
        .note::
            When calling this method at the end of the training process on the current experience to evaluate on the
            current training/eval experience and/or all the previous training/eval experiences, only one process should
            perform the actual evaluation process if running a distributed training because otherwise each process in
            the distributed training process would perform the same redundant operations. For consistency, only the
            process with rank 0 should be used.

        :param exp_list: CL experience information.
        :param kwargs: custom arguments.

        :return: dictionary containing last recorded value for each metric name.
        """
        if not self._distributed_check:
            # Checks if the strategy elements are compatible with
            # distributed training
            self._check_distributed_training_compatibility()
            self._distributed_check = True

        # eval can be called inside the train method.
        # Save the shared state here to restore before returning.
        prev_train_state = self._save_train_state()
        self.is_training = False
        self.model.eval()

        experiences_list: Iterable[
            TExperienceType
        ] = _experiences_parameter_as_iterable(exp_list)
        self.current_eval_stream = experiences_list

        self._before_eval(**kwargs)
        for self.experience in experiences_list:
            self._before_eval_exp(**kwargs)
            self._eval_exp(**kwargs)
            self._after_eval_exp(**kwargs)

        self._after_eval(**kwargs)
        self._eval_cleanup()

        # restore previous shared state.
        self._load_train_state(prev_train_state)

        return self.evaluator.get_last_metrics()

    def eval_dataset_adaptation(self, **kwargs):
        """
        Initialize `self.adapted_dataset` for evaluation and update the `use_in_getitem_indices` attribute accordingly

        The `targets_task_labels` DataAttribute of each sample in the
        dataset is set to the ID of the subnetwork that has been trained on the respective class.

        .note::
            The ID of each subnetwork is a string containing an integer number only. However, the `targets_task_labels`
            DataAttribute stores the integer number in the form of an int not wrapped into a string
        """
        super().eval_dataset_adaptation(**kwargs)
        self._update_use_in_getitem_indices()
        subnetwork_classes = self.model.subnetworks_classes

        def from_targets_to_subnetwork_id(target):
            for sub_id, classes in subnetwork_classes.items():
                if target in classes:
                    return int(sub_id)  # the subnetwork ID is converted into an int
            #  if the target of the current sample has never been seen before, raise an error
            raise RuntimeError("The target of the current sample has never been seen before by any subnetwork")

        targets_task_labels = [from_targets_to_subnetwork_id(int(cls)) for cls in self.adapted_dataset.targets.data]
        self.adapted_dataset = self.adapted_dataset.update_data_attribute("targets_task_labels", targets_task_labels)

    def eval_epoch(self, subnetwork_ids: Optional[Union[str, Literal["all"]]] = "all", **kwargs):
        """
        Evaluation loop over the current `self.dataloader`.

        Each mini-batch is fed into one or multiple subnetworks according to the value of `subnetwork_ids`. Therefore,
        `self.mb_output` is a dictionary containing one or multiple IDs of the subnetworks in the network as keys and
        dictionaries as values. The dictionary of each subnetwork has the following form:
        `{"incremental_classifier": (output tensor1, output tensor2), "outlier_detector": output tensor}`, where
        output tensor1 is the output of the incremental classifier (the logits) while output tensor2 is the
        feature vector, also known as embedding, that precedes the final linear classifier layer in the
        incremental classifier. As a consequence, the loss of each subnetwork is stored in `self.loss` as a dictionary
        containing the subnetwork IDs as keys and the respective losses as values.
        :param subnetwork_ids: (optional) the ID of a subnetwork. The subnetwork with the corresponding ID is used to
            perform the forward pass at each mini-batch. If None, the
            method :meth:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate.forward_no_subnetwork_ids` is
            used. If the literal `all` is provided, the
            method :meth:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate.forward_all_subnetworks` is used
            (the forward pass is computed for all subnetworks). Default is `all`.

        .note::
            Refer to :meth:`SupervisedProblemDynamicNetworkInitFeatureExtractorIncrementalOutlier.criterion` for more
            information about how the incremental classifier and outlier detector losses of each subnetwork are stored
        """
        for self.mbatch in self.dataloader:
            self._unpack_minibatch()
            self._before_eval_iteration(**kwargs)

            self._before_eval_forward(**kwargs)
            self.mb_output = self.forward(subnetwork_ids=subnetwork_ids)
            # this attribute stores a dictionary containing the IDs of the subnetworks in the network as keys and
            # dictionaries as values. The dictionary of each subnetwork has the following form:
            # `{"incremental_classifier": (output tensor1, output tensor2), "outlier_detector": output tensor}`, where
            # output tensor1 is the output of the incremental classifier (the logits) while output tensor2 is the
            # feature vector, also known as embedding, that precedes the final linear classifier layer in the
            # incremental classifier.
            self._after_eval_forward(**kwargs)
            self.loss = self.criterion(**kwargs)

            self._after_eval_iteration(**kwargs)

    def obtain_common_dataloader_parameters(self, **kwargs):
        """
        Utility function that returns the dictionary of parameters to be passed to a dataloader.

        This method exposes publicly the protected
        method :meth:`DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate._obtain_common_dataloader_parameters`
        :param kwargs: the dataloader arguments
        :return: a dictionary of parameters to be passed to the DataLoader class or to one of the Avalanche dataloaders.
        """
        return self._obtain_common_dataloader_parameters(**kwargs)

    #########################################################
    # Plugin Triggers                                       #
    #########################################################
    def _before_train_datasets_adaptation(self, **kwargs):
        trigger_plugins(self, "before_train_datasets_adaptation", **kwargs)

    def _after_train_datasets_adaptation(self, **kwargs):
        trigger_plugins(self, "after_train_datasets_adaptation", **kwargs)

    def _before_train_dataloader(self, **kwargs):
        trigger_plugins(self, "before_train_dataloader", **kwargs)

    def _after_train_dataloader(self, **kwargs):
        trigger_plugins(self, "after_train_dataloader", **kwargs)

    def _before_training_subnetwork(self, **kwargs):
        trigger_plugins(self, "before_training_subnetwork", **kwargs)

    def _after_training_subnetwork(self, **kwargs):
        trigger_plugins(self, "after_training_subnetwork", **kwargs)
