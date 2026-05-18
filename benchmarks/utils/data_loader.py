"""
This module provides a set of custom data loaders
"""
from __future__ import annotations

from typing import Sequence, List, TYPE_CHECKING

from utils import reduce_size

from avalanche.benchmarks.utils.data_loader import (MultiDatasetDataLoader, MultiDatasetSampler,
                                                    _make_data_loader_with_batched_sampler, _make_data_loader)
from torch.utils.data import ConcatDataset
if TYPE_CHECKING:
    from avalanche.benchmarks.utils.data import AvalancheDataset


class MultiDatasetBalancedDataLoader(MultiDatasetDataLoader):
    """
    Dataloader that balances data from multiple datasets.

    This dataloader iterates in parallel multiple datasets which are used to create mini-batches by concatenating their
    data together. The number of examples from each dataset in each mini-batch is equal or differs at most by 1. It
    depends on whether the remainder of (batch size / number of datasets) is 0 or not.

    This dataloader takes inspiration from :class:`avalanche.benchmarks.utils.data_loader.GroupBalancedDataLoader`
    """
    def __init__(self, datasets: Sequence[AvalancheDataset], termination_dataset: int = -1,
                 oversample_small_datasets: bool = False, balance_large_datasets: bool = False, batch_size: int = 32,
                 distributed_sampling: bool = True, **kwargs):
        """
        Create a new MultiDatasetBalancedDataLoader.

        .note::
            If `oversample_small_datasets` is False and `balance_large_datasets` is True, it might happen that the
            last iteration of an epoch is unbalanced. If this is an issue, `drop_last = True` should be provided as a
            keyword argument. TODO I AM NOT ENTIRELY SURE ABOUT THIS. I SHOULD CHECK BETTER

        :param datasets: a sequence of :class:`AvalancheDataset`.
        :param termination_dataset: (optional) an integer number denoting the index of the dataset to be used for
            determining when to stop iterating (the iteration is stopped when the end of the respective dataset is hit).
            In alternative, -1 for stopping the iteration when the end of the largest dataset is hit or -2 for stopping
            the iteration when the end of the smallest dataset is hit. Default is -1.
        :param oversample_small_datasets: (optional) if True, smaller datasets (if any) are oversampled to match the
            `termination_dataset`. Otherwise, once data from a dataset is completely iterated, the dataset will be
            skipped in the subsequent mini-batches. Default is False.
        :param balance_large_datasets: (optional) if True, larger datasets (if any) are randomly reduced in size to
            match the `termination_dataset` while ensuring that each class has an identical or as similar as possible
            number of samples. If True, the datasets must have the `targets` data attribute; otherwise
            a :class:`ValueError` is raised. Note that if True, the larger datasets are randomly reduced in size every
            time `__iter__` is called. Therefore, different smaller datasets are created for each larger dataset
            in each epoch.
            If False, the larger datasets are not reduced in size and the samples used for each of them in each
            dataloader iteration depends on the argument `shuffle` of
            Pytorch :class:`torch.utils.data.dataloader.DataLoader`. Default is False.
        :param batch_size: (optional) the size of the batch. It must be greater than or equal to the number of datasets.
            Otherwise, a :class:`ValueError` is raised. Each batch is created by collating together mini-batches of
            the same size or that differ by at most 1 from each dataset. Default is 32.
        :param distributed_sampling: (optional) If True and a distributed training is being run, apply the
            Pytorch :class:`torch.utils.data.DistributedSampler`. Default is True.
        :param kwargs: data loader arguments used to instantiate the loader for each group separately.
            See Pytorch :class:`torch.utils.data.dataloader.DataLoader`.
        """
        # check if batch_size is larger than or equal to the number of datasets
        if batch_size < len(datasets):
            raise ValueError(f"The batch size ({batch_size}) must be greater than or equal to the number of datasets "
                             f"({len(datasets)})")
        # if balance_large_datasets is True, check if all datasets have the targets data attribute
        if balance_large_datasets:
            for dataset in datasets:
                if "targets" not in dataset._data_attributes:
                    raise ValueError("If balance_large_datasets is True, all datasets must have the `targets` data "
                                     "attribute")
        # check if termination_dataset is in the right range
        if not -2 <= termination_dataset < len(datasets):
            raise ValueError(f"The termination_dataset ({termination_dataset}) must be between -2 (included) and "
                             f"len(datasets) ({len(datasets)}) (excluded)")

        # divide the batch between all datasets in the group
        ds_batch_size = batch_size // len(datasets)
        remaining = batch_size % len(datasets)
        batch_sizes = []
        for _ in datasets:
            bs = ds_batch_size
            if remaining > 0:
                bs += 1
                remaining -= 1
            batch_sizes.append(bs)

        loaders_len = []
        for dataset, batch_size in zip(datasets, batch_sizes):
            # get the number of iterations necessary to fully cycle through this dataset given its batch size and
            # dataloader args
            loader_len = len(_make_data_loader(dataset, distributed_sampling, kwargs, batch_size,
                                               force_no_workers=True)[0])
            loaders_len.append(loader_len)

        # set the termination dataset to the index of the smallest dataset
        if termination_dataset == -2:
            termination_dataset = loaders_len.index(min(loaders_len))

        self._loaders_len: List[int] = loaders_len
        """
        list containing the number of iterations necessary to completely cycle through each dataset.
        """

        self.balance_large_datasets: bool = balance_large_datasets
        """
        boolean flag that indicates whether larger datasets (if any) need to be reduced in size to match the 
        `termination_dataset` while ensuring that each class has an identical or as similar as possible
        number of samples.
        """

        super().__init__(datasets=datasets, batch_sizes=batch_sizes, termination_dataset=termination_dataset,
                         oversample_small_datasets=oversample_small_datasets, distributed_sampling=distributed_sampling,
                         never_ending=False, **kwargs)

    def _get_loader(self):
        datasets = self.datasets
        if self.balance_large_datasets:
            termination_dataset_loader_len = self._loaders_len[self.termination_dataset]
            # make all the larger datasets as large as the termination dataset while ensuring balance of their classes
            datasets = []
            for dataset, loader_len, batch_size in zip(self.datasets, self._loaders_len, self.batch_sizes):
                # if the number of iterations necessary to cycle through this dataset is larger than the number of
                # iterations necessary to cycle through the termination dataset, then reduce the size of this dataset to
                # the number of iterations necessary to cycle through the termination dataset * the batch size of this
                # dataset
                if loader_len > termination_dataset_loader_len:
                    size = termination_dataset_loader_len * batch_size
                    dataset = reduce_size(dataset, size, balancing=True)
                datasets.append(dataset)

        samplers = self._create_samplers(
            datasets,
            self.batch_sizes,
            self.distributed_sampling,
            self.loader_kwargs,
        )

        overall_dataset = ConcatDataset(datasets)

        multi_dataset_batch_sampler = MultiDatasetSampler(
            overall_dataset.datasets,
            samplers,
            termination_dataset_idx=self.termination_dataset,
            oversample_small_datasets=self.oversample_small_datasets,
            never_ending=self.never_ending,
        )

        loader = _make_data_loader_with_batched_sampler(
            overall_dataset,
            batch_sampler=multi_dataset_batch_sampler,
            data_loader_args=self.loader_kwargs,
        )

        return loader

