"""
This module allows performing training on the CIFAR100 dataset with a ResNet-32 model in a class-incremental learning
setting consisting of 10 experiences of 10 classes each. It performs the third experience with different new classes
starting from the same second experience checkpoint produced by
`experiment28_cifar100_resnet32_10_increments_10_classes_2000.py`.
"""

import os
import dill
import random

from training.runner import Runner, RunnerPlugin
from training.templates import DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate
from benchmarks.classic import SplitCIFAR100
from training.storage_policy import ClassBalancedBufferDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate
from models.network_class_incremental import NetworkClassIncremental
from training.plugins import (
                              LrSchedulerPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
                              TrainMBSchedulerPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
                              ReplayNoOutlierPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate,
                              )
from metrics.class_accuracy import incremental_classifier_class_accuracy_per_subnetwork_metrics
from metrics.confusion_matrix import incremental_classifier_confusion_matrix_per_subnetwork_metrics
from metrics.incremental_classifier_accuracy import incremental_classifier_accuracy_per_subnetwork_metrics
from metrics.incremental_classifier_loss import incremental_classifier_loss_per_subnetwork_metrics
from metrics.topk_accuracy import incremental_classifier_topk_accuracy_per_subnetwork_metrics
from metrics.logit_embeddings import LogitEmbeddingPluginMetric
from convnets.cifar_resnet import resnet32
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import timing_metrics
from avalanche.logging import InteractiveLogger
random.seed(101)  # set the random see for the creation of the splits in SplitCIFAR100; TODO needs to be different for
# each experiment type

EXP_ID = 28
"""
Experiment ID
"""

N_EXPS = 10
"""
Number of experiences.
"""

N_CLASSES_PER_EXP = 10
"""
Number of classes per experience
"""

N_EXEMPLARS = 2000
"""
The number of exemplars to keep in the replay buffer.
"""

N_CLASSES_TOTAL = N_EXPS * N_CLASSES_PER_EXP
"""
Total number of classes
"""

N_CLASS_ORDERS = 40
"""
Number of class orders
"""

CLASS_ORDERS = [random.sample(range(0, N_CLASSES_TOTAL), N_CLASSES_TOTAL) for _ in range(N_CLASS_ORDERS)]
"""
The class orders that will be used for creating the splits in the SplitCIFAR100 benchmark.
"""

EXP_SEEDS = [[random.randint(0, 1300) for _ in range(N_EXPS)] for _ in range(N_CLASS_ORDERS)]
"""
The seeds that will be used to control the random operations during training for each different class order
"""

REHEARSAL_SEEDS = [random.randint(0, 1300) for _ in range(N_CLASS_ORDERS)]
"""
The seeds that will be used to create the rehearsal set for each different class order 
"""

CLASS_ORDERS = [CLASS_ORDERS[0][:N_CLASSES_PER_EXP*2] + random.sample(CLASS_ORDERS[0][N_CLASSES_PER_EXP*2:],
                                                                      N_CLASSES_TOTAL - N_CLASSES_PER_EXP*2)
                for _ in range(N_CLASS_ORDERS)]
"""
The class orders where the first and second experience classes are all the same while the classes for the other
experiences are different
"""

EXP_SEEDS = [EXP_SEEDS[0] for _ in range(N_CLASS_ORDERS)]
"""
The seeds that will be used to control the random operations during training for each different class order; they are
all equal
"""

REHEARSAL_SEEDS = [REHEARSAL_SEEDS[0] for _ in range(N_CLASS_ORDERS)]
"""
The seeds that will be used to create the rehearsal set for each different class order; they are all equal 
"""

CLASS_ORDERS = CLASS_ORDERS[:20]
EXP_SEEDS = EXP_SEEDS[:20]
REHEARSAL_SEEDS = REHEARSAL_SEEDS[:20]

TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor(),
        transforms.Normalize(mean=torch.tensor([0.5071, 0.4865, 0.4409]),
                             std=torch.tensor([0.2673, 0.2564, 0.2762]))
    ]
)
"""
The transformations to be used during training.
They are the same transformations used in https://arxiv.org/abs/2302.03648
"""

EVAL_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=torch.tensor([0.5071, 0.4865, 0.4409]),
                             std=torch.tensor([0.2673, 0.2564, 0.2762]))
    ]
)
"""
The transformations to be used during evaluation.
Obviously, they do not perform any sort of data augmentation.
They are the same transformations used in https://arxiv.org/abs/2302.03648
"""

LR = 0.1
"""
The initial learning rate for each experience. The same one as in https://arxiv.org/abs/2302.03648
"""

MOMENTUM = 0.9
"""
The momentum to be used for optimisation. The same one as in https://arxiv.org/abs/2302.03648
"""

WEIGHT_DECAY = 0.0005
"""
The weight decay to be used for optimization. The same one as in https://arxiv.org/abs/2302.03648
"""

INIT_TRAIN_MB_SIZE = 64
"""
The training mini-batch size during the first experience 
"""

NEXT_TRAIN_MB_SIZE = 128
"""
The training mini-batch size during all experiences except for the first one. 64 samples are sampled from the current
experience's dataset while the remaining half is sampled from the replay buffer.
The same one as in https://arxiv.org/abs/2302.03648
"""

EVAL_MB_SIZE = 22500
"""
The evaluation mini-batch size.
"""

TRAIN_EPOCHS = 170
"""
The number of training epochs. The same one as in https://arxiv.org/abs/2302.03648
"""

class ReplaceStoragePolicy(RunnerPlugin):
    def __init__(self, path: str):
        super().__init__()
        self.storage_policy = torch.load(path, pickle_module=dill, map_location="cpu")
        print(f"Buffer datasets: {[len(b) for b in self.storage_policy.buffer_datasets]}")

    def after_loading_checkpoint(self, runner: Runner, *args, **kwargs):
        runner.strategy.plugins[2].storage_policy = self.storage_policy


if __name__ == "__main__":

    for index, (class_order, exp_seeds, rehearsal_seed) in enumerate(zip(CLASS_ORDERS, EXP_SEEDS, REHEARSAL_SEEDS)):

        # If cuda is available, the default device is "cuda". Otherwise, "cpu" is used.
        # If this script is run using `torch.run` for distributed training, the device is returned by
        # `Runner.init_distributed`
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if os.environ.get("LOCAL_RANK", None) is not None:  # if LOCAL_RANK env var exists, then torch.run was used
            device = Runner.init_distributed()

        benchmark = SplitCIFAR100(
            n_experiences=N_EXPS,
            val_dataset=0.1,  # 10% of the samples in the train dataset are apportioned to validation
            seed=123,  # this seed is used to create the validation set so that to replicate results
            fixed_class_order=class_order,
            shuffle=False,
            train_transform=TRAIN_TRANSFORM,
            eval_transform=EVAL_TRANSFORM,
            dataset_root="cifar100",
            include_indices=True
        )

        strategy = DynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate(
            # a single ResNet32 model with a dynamic output layer for the class incremental setting
            model=NetworkClassIncremental(
                model=resnet32(),
                class_head_name="fc",
                num_features=64
            ),
            # the same optimizer as in https://arxiv.org/abs/2302.03648
            optimizer=SGD(
                params=nn.Linear(1, 1).parameters(),  # just some dummy parameters
                lr=LR,
                momentum=MOMENTUM,
                weight_decay=WEIGHT_DECAY
            ),
            # since we only care about the incremental classifier, the outlier criterion weight is set to 0 so that
            # it does not influence the training process
            outlier_criterion_weight=0.,
            train_mb_size=INIT_TRAIN_MB_SIZE,
            train_epochs=TRAIN_EPOCHS,
            eval_mb_size=EVAL_MB_SIZE,
            device=device,
            plugins=[
                # During the first experience (experience 0), the training mini-batch size is 64. From the second
                # experience (including) onwards, the training mini-batch size is set to 128, i.e. 64 for the
                # samples in the current experience and 64 for the samples stored in the replay buffer.
                TrainMBSchedulerPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate(
                    train_mb_sizes={1: NEXT_TRAIN_MB_SIZE}
                ),
                LrSchedulerPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate(
                    scheduler=CosineAnnealingLR,
                    step_calls="epoch",
                    reset_lrs=True,
                    # some keyword arguments for `CosineAnnealingLR`
                    T_max=TRAIN_EPOCHS  # the cosineannealing scheduler is defined over all the training epochs
                ),
                # handles a replay memory buffer. The same number of samples are stored across each observed class.
                # The samples for each class are stored (uniform) randomly.
                ReplayNoOutlierPluginDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate(
                    storage_policy=ClassBalancedBufferDynamicNetworkInitFeatureExtractorIncrementalOutlierTemplate(
                        max_size=N_EXEMPLARS,
                        # seeding the creation of the replay buffer for each class
                        seed=rehearsal_seed
                    )
                )
            ],
            evaluator=EvaluationPlugin(
                incremental_classifier_class_accuracy_per_subnetwork_metrics(experience=True),
                incremental_classifier_confusion_matrix_per_subnetwork_metrics(experience=True),
                incremental_classifier_accuracy_per_subnetwork_metrics(epoch=True, experience=True),
                incremental_classifier_loss_per_subnetwork_metrics(epoch=True, experience=True),
                incremental_classifier_topk_accuracy_per_subnetwork_metrics(top_k=3, experience=True),
                timing_metrics(minibatch=True, epoch=True),
                [LogitEmbeddingPluginMetric(experience_type=["train", "test"], device=torch.device("cpu"))],
                loggers=[InteractiveLogger()]
            ),
            eval_every=1,
            peval_mode="epoch",
            do_initial=True,  # perform evaluation at the start of each experience before training starts
        )

        runner = Runner(strategy, benchmark)
        runner.run(
            save_checkpoint_path=f"experiment{EXP_ID}_cifar100_resnet32_{N_EXPS}_increments_{N_CLASSES_PER_EXP}_classes_{index}_{N_EXEMPLARS}_diff_classes",
            load_checkpoint=f"experiment{EXP_ID}_cifar100_resnet32_{N_EXPS}_increments_{N_CLASSES_PER_EXP}_classes_0_{N_EXEMPLARS}_experience_1.pth",
            device_checkpoint=torch.device("cpu"),
            seed=exp_seeds,
            experience_stop=3,
            eval_streams_type=["train", "val", "test"],
            plugins=[
                ReplaceStoragePolicy(
                    path=f"experiment{EXP_ID}_cifar100_resnet32_{N_EXPS}_increments_{N_CLASSES_PER_EXP}_classes_0_{N_EXEMPLARS}_storage_policy_experience_1.pth"
                )
            ],
            num_workers=10,  # the number of workers used for dataloading
            reset_optimizer_state=True  # after each experience, the optimizer state is reset
        )
