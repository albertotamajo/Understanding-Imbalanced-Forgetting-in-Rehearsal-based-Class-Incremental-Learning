"""
This module includes PyTorch dataset implementations for some datasets
"""
import multiprocessing
from abc import ABC, abstractmethod
from typing import Dict, Any, Sequence, Tuple, Union, Optional, Callable

from torchvision.datasets import ImageNet


class SampleCacheMixin(ABC):
    """
    This mixin class enables Pytorch datasets to cache and uncache samples, their targets and other metadata on a
    dictionary.
    """

    def __init__(self, cache_mode: bool = False, cached_transform: Optional[Callable] = None,
                 cached_target_transform: Optional[Callable] = None):
        """
        Create a new SampleCacheMixin
        :param cache_mode: (optional) whether the dataset must cache the samples it retrieves when `__getitem__` is
            called. Default is False.
        :param cached_transform: (optional) a function/transform that takes in an PIL image and returns a transformed
            version. This transformation is applied before caching a given sample. Therefore, a transformed version of
            the sample is cached if a transformation is provided. If None, a sample is cached without applying any
            transformation. Default is None.
        :param cached_target_transform: (optional) a function/transform that takes in a target and transforms it.
            This transformation is applied before caching the target of a sample. Therefore, a transformed version of
            the target is cached if a transformation is provided. If None, the target of a sample is cached without
            applying any transformation. Default is None.
        """
        self.cache: Dict[int, Any] = {}
        """
        a dictionary containing indexes as keys and the corresponding cached samples, their targets and other metadata
        as values
        """

        self.cache_mode: bool = cache_mode
        """
        a boolean flag that indicates whether the dataset must cache the samples, their targets and other metadata
        it retrieves when `__getitem__` is called.
        """

        self.cached_transform = cached_transform
        """
        a function/transform applied before caching that takes in an PIL image and returns a transformed version.
        """

        self.cached_target_transform = cached_target_transform
        """
        a function/transform applied before caching that takes in a target and transforms it
        """

    @abstractmethod
    def cache_samples(self, indexes: Union[int, Sequence[int]], n_subprocesses: int = 0):
        """
        Cache dataset samples, their targets and other metadata. The samples, their targets and other metadata must be
        cached in the `cache` dictionary.
        :param indexes: a single index or a sequence of indexes of dataset samples that must be cached
        :param n_subprocesses: number of subprocesses to use for loading the samples, their targets and other metadata
            and then caching them. If n_subprocesses=0, the current process performs all the loading and caching
            operations. If n_subprocesses>0, an equivalent number of subprocesses is spawned which concurrently
            perform loading operations and caching operations. Default is 0.
        :return: None
        """
        pass

    @abstractmethod
    def uncache_samples(self, indexes: Optional[Union[int, Sequence[int]]] = None):
        """
        Uncache dataset samples, their targets and other metadata. The samples, their targets and other metadata must be
        uncached from the `cache` dictionary.
        :param indexes: (optional) a single index or a sequence of indexes of dataset samples that must be uncached. If
            None, all samples are uncached. Default is None.
        :return: None
        """
        pass


class ImageNetSampleCache(ImageNet, SampleCacheMixin):
    """
    `ImageNet <http://image-net.org/>`_ 2012 Classification Dataset.
    This dataset subclasses :class:`ImageNet` and :class:`SampleCacheMixin`.
    It has the same functionalities of :class:`ImageNet` with caching functionalities in addition.

        . note::
            Before using this class, it is required to download ImageNet 2012 dataset from
            `here <https://image-net.org/challenges/LSVRC/2012/2012-downloads.php>`_ and
            place the files ``ILSVRC2012_devkit_t12.tar.gz`` and ``ILSVRC2012_img_train.tar``
            or ``ILSVRC2012_img_val.tar`` based on ``split`` in the root directory.


        Attributes:
            cache (dict): Dict containing indexes as keys and the corresponding cached samples and their targets
            as values
            cache_mode (bool): a boolean flag that indicates whether the dataset must cache the samples and the targets
            it retrieves when `__getitem__` is called.
            classes (list): List of the class name tuples.
            class_to_idx (dict): Dict with items (class_name, class_index).
            wnids (list): List of the WordNet IDs.
            wnid_to_idx (dict): Dict with items (wordnet_id, class_index).
            imgs (list): List of (image path, class_index) tuples
            targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root: str, split: str = "train", **kwargs: Any) -> None:
        """
        Create a new ImageNetSampleCache
        :param root: root directory of the ImageNet Dataset
        :param split: the dataset split, supports ``train``, or ``val``
        :param cache_mode: (bool, optional) whether the samples retrieved when `__getitem__` is called must be cached.
            Default is False.
        :param cached_transform: (callable, optional) a function/transform that takes in an PIL image and returns a
            transformed version. This transformation is applied before caching a given sample. Therefore, a transformed
            version of the sample is cached if a transformation is provided. If None, a sample is cached without
            applying any transformation. Default is None.
        :param transform: (callable, optional) a function/transform applied after the `cached_transform`. Unlike
            `cached_transform`, this transformation is not applied before caching a given sample. Therefore, this
            transformation is always applied whenever a given sample is retrieved. This transformation is useful
            when some randomised transformations must be applied and it is important that every time a sample is
            retrieved a different transformation is applied to it. If None, no transformation is applied.
            Default is None.
        :param cached_target_transform: (callable, optional) a function/transform that takes in a target and transforms
            it. This transformation is applied before caching the target of a sample. Therefore, a transformed version
            of the target is cached if a transformation is provided. If None, the target of a sample is cached without
            applying any transformation. Default is None.
        :param target_transform: (callable, optional): a function/transform applied after the `cached_target_transform`.
            Unlike `cached_target_transform`, this transformation is not applied before caching a target. Therefore,
            this transformation is always applied whenever a given target is retrieved. This transformation is useful
            when some randomised transformations must be applied and it is important that every time a target is
            retrieved a different transformation is applied to it. If None, no transformation is applied.
            Default is None.
        :param loader: (callable, optional): a function to load an image given its path.
        """
        kwargs_super = {}
        kwargs_sample_cache_mixin = {}
        for key, val in kwargs.items():
            if key in ["cache_mode", "cached_transform", "cached_target_transform"]:
                kwargs_sample_cache_mixin[key] = val
            else:
                kwargs_super[key] = val
        super().__init__(root=root, split=split, **kwargs_super)
        SampleCacheMixin.__init__(self, **kwargs_sample_cache_mixin)

    def cache_samples(self, indexes: Union[int, Sequence[int]], n_subprocesses: int = 0):
        """
        TODO this method does not work on the cluster when n_subprocesses is greater than 0.
        TODO It works on my laptop,though
        Cache dataset samples and their targets. Before caching, the samples and targets are transformed using
        `cached_transform` and `cached_target_transform`, respectively.

        :param indexes: a single index or a sequence of indexes of dataset samples that must be cached
        :param n_subprocesses: number of subprocesses to use for loading the samples, their targets and other metadata
            and then caching them. If n_subprocesses=0, the current process performs all the loading and caching
            operations. If n_subprocesses>0, an equivalent number of subprocesses is spawned which concurrently
            perform loading operations and caching operations. Default is 0.
        :return: None
        """
        if n_subprocesses < 0:
            raise ValueError("n_subprocesses must be greater than or equal to 0")

        if isinstance(indexes, int):
            if indexes not in self.cache.keys():
                sample, target = self._load_sample(indexes, use_cached_transform=True)
                self.cache[indexes] = (sample, target)
        else:
            indexes = list(set(indexes))  # remove duplicates and then transform into a list
            indexes = [index for index in indexes if index not in self.cache.keys()]  # remove already cached indexes
            # if there are no left indexes, then just return. Otherwise, continue with the caching operations
            if len(indexes) == 0:
                return

            if n_subprocesses == 0:
                for ind in indexes:
                    sample, target = self._load_sample(ind, use_cached_transform=True)
                    self.cache[ind] = (sample, target)
            else:
                # create a process pool with the desired number of processes
                with multiprocessing.Pool(processes=n_subprocesses) as pool:
                    # map the get_sample_target function to the list of indexes, running them concurrently
                    results = pool.map(self._load_sample_exception_handled, indexes)
                # wait for all processes to finish and get the results
                pool.close()
                pool.join()
                # `results` is a list of pairs
                # (loaded image with cached transformation, target with cached transformation). If one sample could not
                # be loaded, the corresponding pair is (index, exception). if there is at lest one exception, an
                # exception is raised by this method.
                exception_list = [val2 for val1, val2 in results if isinstance(val2, Exception)]
                if len(exception_list) != 0:
                    raise exception_list[0]  # raise the first exception on the list
                # otherwise, the cache dictionary is updated using the pairs of results
                self.cache.update(dict(results))

    def uncache_samples(self, indexes: Optional[Union[int, Sequence[int]]] = None):
        """
        Uncache dataset samples and their targets.
        :param indexes: a single index or a sequence of indexes of dataset samples that must be uncached. If None,
            all samples are uncached. Default is None.
        :return: None
        """
        if indexes is None:
            self.cache = {}
        else:
            if isinstance(indexes, int):
                indexes = [indexes]

            indexes = list(set(indexes))  # remove duplicates and transform `indexes` into a list
            for ind in indexes:
                if ind in self.cache.keys():
                    self.cache.pop(ind)

    def _load_sample(self, index: int, use_cached_transform: bool = False) -> Tuple[Any, Any]:
        """
        Load a sample and its target
        :param index: index of the sample to load
        :param use_cached_transform: whether to process the sample and its target with the `cached_transform` and
            `cached_target_transform` respectively.
        :return: sample, target
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if use_cached_transform:
            if self.cached_transform is not None:
                sample = self.cached_transform(sample)
            if self.cached_target_transform is not None:
                target = self.cached_target_transform(target)
        return sample, target

    def _load_sample_exception_handled(self, index: int) -> Tuple[Any, Any]:
        """
        Load a sample and its target
        :param index: index of the sample to load
        :return:  sample, index or index, exception if an exception is raised when loading the sample
        """
        try:
            sample, target = self._load_sample(index, use_cached_transform=True)
            return index, (sample, target)
        except Exception as e:
            return index, e

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Get a sample and its target. If the sample and its target are stored in the cache, then they are retrieved from
        the cache and then transformations (if any) are applied to them. Otherwise, they are loaded externally and
        subsequently cached transformations (if any) followed by transformations (if any) are applied to them. If cache
        mode is active `cache_mode==True`, the sample and its target are cached after applying cached transformations to
        them.
        :param index: index of the sample that must be retrieved
        :return: (sample, target)
        """
        if index in self.cache.keys():
            sample, target = self.cache[index]
        else:
            sample, target = self._load_sample(index, use_cached_transform=True)
            if self.cache_mode:
                self.cache[index] = (sample, target)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target
