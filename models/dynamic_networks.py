"""
This module includes a set of dynamic networks that subclass Avalanche :class:`DynamicModule`.

Dynamic Modules are Avalanche modules that can be incrementally expanded to allow architectural modifications
(multi-head classifiers, progressive networks, ...). Compared to PyTorch Modules, they provide an additional method,
`model_adaptation`, which adapts the model given the current experience. Avalanche strategies call this method to
adapt the architecture *before* processing each experience. Strategies also update the optimizer automatically.
"""

from typing import Union, Any, Literal, Optional, Sequence, Dict, List, Tuple

import torch
from torch import Tensor
from torch.nn import Linear
import torch.nn as nn
import numpy as np

from avalanche.models.dynamic_modules import DynamicModule
from avalanche.benchmarks.scenarios import CLExperience


class DynamicNetworkWithInitFeatureExtractor(DynamicModule):
    """
    A dynamic network that has an initial feature extractor and a number of subsequent subnetworks.

    An input is fed into the initial feature extractor, whose output is in turn fed into one or multiple subsequent
    subnetworks.

    The subnetworks are stored in a :class:`nn.ModuleDict`. Therefore, new subnetworks can be added to this network or
    old ones can be removed straightforwardly by managing this dictionary.

    The subnetwork IDs *must* be of type string because :class:`nn.ModuleDict` only supports string keys.

    Subclasses must override the methods :meth:`DynamicNetworkWithInitFeatureExtractor.train_adaptation`
    and :meth:`DynamicNetworkWithInitFeatureExtractor.eval_adaptation` to implement a custom adaptation of this dynamic
    module at training and evaluation time, respectively,  given the current experience.

    The method
    :meth:`DynamicNetworkWithInitFeatureExtractor.forward_no_subnetwork_ids` should be overridden if a custom forward
    process needs to be implemented when the subnetwork IDs are not provided at training or inference time. By default,
    the forward pass is computed for all subnetworks when no subnetwork IDs are provided.

    Dynamic Modules are Avalanche modules that can be incrementally expanded to allow architectural modifications
    (multi-head classifiers, progressive networks, ...). Compared to PyTorch Modules, they provide an additional method,
    `model_adaptation`, which adapts the model given the current experience. Avalanche strategies call this method to
    adapt the architecture *before* processing each experience. Strategies also update the optimizer automatically.

    .warning::
        The subnetwork IDs can be any arbitrary string but "all"

    .warning::
        When overriding :meth`DynamicNetworkWithInitFeatureExtractor.train_adaptation`
        and :meth`DynamicNetworkWithInitFeatureExtractor.eval_adaptation`, remember to pass the new parts of
        this network into the same device of the other components if the network is expanded.
    """

    def __init__(self, init_feature_extractor: nn.Module):
        """
        Initializes a new DynamicNetworkWithInitFeatureExtractor
        :param init_feature_extractor: the initial feature extractor
        """
        super().__init__()
        self.init_feature_extractor: nn.Module = init_feature_extractor
        """the initial feature extractor"""

        self.subnetworks: nn.ModuleDict = nn.ModuleDict()
        """the dictionary of subnetworks. New subnetworks *must* be added in here"""

    def train_adaptation(self, experience: CLExperience):
        """
        Module's adaptation at training time.
        Avalanche strategies automatically call this method *before* training
        on each experience. You should **never** use this data to **train** the module's parameters.

        .warning::
            remember to pass the new parts of this network into the same device of the other components if
            the network is expanded.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def eval_adaptation(self, experience: CLExperience):
        """
        Module's adaptation at evaluation time.
        Avalanche strategies automatically call this method *before* evaluating
        on each experience. You should **never** use this data to **train** the module's parameters.
        This method receives the experience's data at evaluation time because some models might need it for adaptation.

        .warning::
            remember to pass the new parts of this network into the same device of the other components if
            the network is expanded.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def forward(self, x: Tensor, subnetwork_ids: Optional[Union[Sequence[str], str, Literal["all"]]] = None,
                use_init_feature_extractor: bool = True) -> Dict[str, Any]:
        """
        Compute the forward pass given the input tensor.
        :param x: input tensor
        :param subnetwork_ids: (optional) subnetwork IDs for each sample or only one subnetwork ID.
            If None, the method :meth:`DynamicNetworkWithInitFeatureExtractor.forward_no_subnetwork_ids` is used.
            If the literal `all` is provided,
            the method :meth:`DynamicNetworkWithInitFeatureExtractor.forward_all_subnetworks`
            is used. Default is None
        :param use_init_feature_extractor: (optional) whether to use the initial_feature extractor or not during the
            forward pass. Default is True
        :return: a dictionary with subnetwork IDs as keys and the output of the corresponding subnetwork as value if
            multiple subnetwork IDs are provided. The order of the samples in each subnetwork output matches the order
            of the corresponding samples in the input tensor. If one subnetwork ID is provided, the output is the output
            of the method :meth:`DynamicNetworkWithInitFeatureExtractor.forward_single_subnetwork`.
            If `subnetworks_ids` is None, the output is the output of the
            method :meth:`DynamicNetworkWithInitFeatureExtractor.forward_no_subnetwork_ids`.
            If `subnetworks_ids` is `all`, the output is the output of
            the method :meth:`DynamicNetworkWithInitFeatureExtractor.forward_all_subnetworks`
        """
        if subnetwork_ids is None:
            return self.forward_no_subnetwork_ids(x, use_init_feature_extractor)
        elif subnetwork_ids == "all":
            return self.forward_all_subnetworks(x, use_init_feature_extractor)
        elif subnetwork_ids in self.subnetworks.keys():  # if only one subnetwork ID is provided and it is an actual ID
            return self.forward_single_subnetwork(x, subnetwork_ids, use_init_feature_extractor)
        else:  # if a subnetwork ID is provided for each sample
            # convert the subnetwork_ids sequence of str into a numpy array of str
            subnetwork_ids = np.asarray(subnetwork_ids).flatten()  # flatten it if it is not a 1D array
            if not x.shape[0] == len(subnetwork_ids):
                raise ValueError("The number of subnetwork IDs does not match the number of samples")
            out = {}
            unique_subnetwork_ids = np.unique(subnetwork_ids)
            for subnetwork_id in unique_subnetwork_ids:
                subnetwork_mask = subnetwork_ids == subnetwork_id
                out.update(self.forward_single_subnetwork(x[subnetwork_mask], subnetwork_id,
                                                          use_init_feature_extractor))
            return out

    def forward_single_subnetwork(self, x: Tensor, subnetwork_id: str,
                                  use_init_feature_extractor: bool = True) -> Dict[str, Any]:
        """
        Compute the forward pass given the input tensor and the subnetwork ID
        :param x: input tensor
        :param subnetwork_id: id of the subnetwork for the forward pass
        :param use_init_feature_extractor: (optional) whether to use the initial_feature extractor or not.
            Default is True
        :return: a dictionary containing the subnetwork ID as key and the output of the subnetwork as value
        """
        if use_init_feature_extractor:
            x = self.init_feature_extractor(x)
        return {subnetwork_id: self.subnetworks[subnetwork_id](x)}

    def forward_all_subnetworks(self, x: Tensor, use_init_feature_extractor: bool = True) -> Dict[str, Any]:
        """
        Compute the forward pass given the input tensor for all subnetworks
        :param x: input tensor
        :param use_init_feature_extractor: whether to use the initial_feature extractor or not. Default is True
        :return: a dictionary containing all subnetwork IDs as keys and the output of the
            corresponding subnetwork as value.
        """
        out = {}
        for subnetwork_id in self.subnetworks.keys():
            out.update(self.forward_single_subnetwork(x, subnetwork_id, use_init_feature_extractor))
        return out

    def forward_no_subnetwork_ids(self, x: Tensor, use_init_feature_extractor: bool = True) -> Dict[str, Any]:
        """
        Compute the forward pass when the subnetwork IDs for each sample are not provided.
        By default, the forward pass is computed for all subnetworks.
        :param x: input tensor
        :param use_init_feature_extractor: whether to use the initial_feature extractor or not. Default is True
        :return: the output is the output of the
            method :meth:`DynamicNetworkWithInitFeatureExtractor.forward_all_subnetworks`
        """
        return self.forward_all_subnetworks(x, use_init_feature_extractor)


class IncrementalClassifierHead(DynamicModule):
    """
    Classification head that incrementally adds units whenever new classes are
    encountered. New output units are added *only* at training time (when `self.training` is True).
    This class overrides the Avalanche class :class:`DynamicModule`.

    The classifier head is initialised
    *only after* calling the method :meth:`IncrementalClassifierHead.adaptation` on a training experience.
    Therefore, before calling the :meth:`IncrementalClassifierHead.forward` method,
    the method :meth:`IncrementalClassifierHead.adaptation` *must* be called to adapt the classifier head
    to a training experience. Otherwise, a :class:`RuntimeError` is raised.

    The order of the output nodes in the classifier head matches the order of the classes in
    the `seen_classes` attribute.
    """

    def __init__(self, in_features: int):
        """
        Initialise a new IncrementalClassifierHead
        :param in_features: size of each input sample
        """
        super().__init__()
        self.in_features: int = in_features
        """size of each input sample"""

        self.classifier: Linear = Linear(in_features=in_features, out_features=1)
        """
        classifier head. At initialisation, the classifier head is a linear layer with only one output node. After the
        first call to `adapt`, the linear layer is modified according to the given experience.
        """

        self.seen_classes: List[Any] = []
        """classes seen so far"""

    @torch.no_grad()
    def train_adaptation(self, experience: CLExperience):
        """
        Adaptation of the classification head on a new training experience.
        If `n` new classes are encountered, `n` new nodes are added to the classification head.
        Otherwise, if no new classes are encountered, the classification head is not modified.

        New output nodes are appended to the previous ones using the order in which the new classes appear
        on a new experience. This order is determined by the `classes_in_this_experience` attribute.

        Avalanche strategies automatically call this method *before* training or evaluating on each experience.
        :param experience: experience
        :return: None
        """
        device = self._adaptation_device
        old_nclasses = len(self.seen_classes)
        curr_classes = experience.classes_in_this_experience
        new_classes = [cls for cls in curr_classes if cls not in self.seen_classes]
        new_nclasses = len(new_classes)
        self.seen_classes.extend(new_classes)

        # update classifier weights
        if new_nclasses > 0:
            classifier = Linear(self.in_features, old_nclasses + new_nclasses).to(device)
            if old_nclasses > 0:
                # copy the weights and biases of the previous nodes into the corresponding nodes of the new linear layer
                old_w, old_b = self.classifier.weight, self.classifier.bias
                classifier.weight[:old_nclasses] = old_w
                classifier.bias[:old_nclasses] = old_b
            self.classifier = classifier

    def forward(self, x: Tensor) -> Tensor:
        """
        Compute the forward pass on the given input tensor. Before calling this method, the method
        :meth:`IncrementalClassifierHead.adaptation` *must* be called to adapt the classifier head to an experience.
        Otherwise, a :class:`RuntimeError` will be raised
        :param x: input tensor
        :return: output tensor
        """
        if len(self.seen_classes) == 0:  # if this network has never seen any classes
            raise RuntimeError("The classifier head has not been initialised yet. "
                               "Before calling the forward method, the "
                               "adaptation method must be called to adapt the classifier head to an experience.")
        return self.classifier(x)


class NetworkWithIncrementalClassifierHead(DynamicModule):
    """
    A network that includes an instance of the class :class:`IncrementalClassifierHead` as the classifier head in the
    last layer.

    This network comprises a backbone network and a classifier head. The backbone network takes inputs, processes them
    and its outputs are fed into the incremental classifier head for discrimination.

    .note::
        During the adaptation stage of this network, if the backbone network is an instance of a class that subclasses
        :class:`DynamicModule`, the backbone network is adapted to the new experience too
    """

    def __init__(self, network: nn.Module, in_features: int):
        """
        Initialise a new NetworkWithIncrementalClassifierHead
        :param network: backbone network. The output of this network is fed into the incremental classifier head
        :param in_features: size of each input sample for the incremental classifier head
        """
        super().__init__()
        self.network: nn.Module = network
        """the backbone network. The output of this network is fed into the incremental classifier head"""

        self.classifier: IncrementalClassifierHead = IncrementalClassifierHead(in_features)
        """the incremental classifier head"""

    def adaptation(self, experience: CLExperience):
        """
        Adaptation of the network on a new experience. The classifier head is adapted to the new experience.
        If the backbone network is a subclass of :class:`DynamicModule`, the backbone network is adapted to the new
        experience too.
        :param experience: experience
        :return: None
        """
        if issubclass(self.network.__class__, DynamicModule):
            self.network.adaptation(experience)
        self.classifier.adaptation(experience)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Compute the forward pass given the input tensor
        :param x: input tensor
        :return: the output tensor of the classification head (the logits), the output of the backbone network
            (the feature vector, also known as embedding, that precedes the classification head)
        """
        x = self.network(x)
        out = self.classifier(x)
        return out, x


class IncrementalClassifierOutlierDetectorWithInitFeatureExtractor(DynamicNetworkWithInitFeatureExtractor):
    """
    A network that has an initial feature extractor and two subsequent subnetworks. The two subsequent networks take as
    input the output of the initial feature extractor. One subnetwork, the incremental classifier, classifies the input
    samples while the other subnetwork, the outlier detector, detects whether a sample is an outlier or not.

    The incremental classifier *must* be an instance of the class :class:`NetworkWithIncrementalClassifierHead` while
    the outlier detector *must* be a PyTorch module (:class:`nn.Module`) with a linear layer with *only one* output
    node in the last layer.

    The ID of the incremental classifier is `incremental_classifier` while the ID of the outlier detector
    is `outlier_detector`.

    This class is a subclass of the class :class:`DynamicNetworkWithInitFeatureExtractor`.

    .note::
        During the adaptation stage, besides the incremental classifier, the initial feature extractor and outlier
        detector are also adapted to the new experience if they are instances of a class that subclasses `DynamicModule`

    .warning::
        During the adaptation stage, new parts *must* be added to the components in this network
        (initial feature extractor, incremental classifier and outlier detector) with caution because the number
        of output features of the initial feature extractor and input features of the incremental classifier and
        outlier detector *must* match. Also, the outlier detector must have *only one* node in its output layer

    .warning::
        No additional subnetworks *must* be added to this network besides the incremental classifier and
        outlier detector
    """
    def __init__(self, init_feature_extractor: nn.Module, incremental_classifier: NetworkWithIncrementalClassifierHead,
                 outlier_detector: nn.Module):
        """
        Initialise the IncrementalClassifierOutlierDetectorWithInitFeatureExtractor
        :param init_feature_extractor: initial feature extractor
        :param incremental_classifier: incremental classifier
        :param outlier_detector: outlier detector
        """
        super().__init__(init_feature_extractor)
        if not issubclass(incremental_classifier.__class__, NetworkWithIncrementalClassifierHead):
            raise ValueError("The incremental classifier must be an instance of NetworkWithIncrementalClassifierHead")
        last_module_outlier_detector = list(outlier_detector.modules())[-1]
        if not isinstance(last_module_outlier_detector, Linear):
            raise ValueError("The last layer of the outlier detector must be a linear layer")
        if not last_module_outlier_detector.out_features == 1:
            raise ValueError("The final linear layer of the outlier detector must have only one node")
        # add the incremental classifier and outlier detector to the network
        self.subnetworks.update({"incremental_classifier": incremental_classifier,
                                 "outlier_detector": outlier_detector})

    def train_adaptation(self, experience: CLExperience):
        """
        Adaptation of the network at training time. The incremental classifier is adapted to the new experience.
        If the initial feature extractor is a subclass of :class:`DynamicModule`, it is
        adapted to the new experience too. If the outlier detector is a subclass of :class:`DynamicModule`, it is
        adapted to the new experience too.

        .warning::
            Note that during the adaptation stage, new parts *must* be added to the components in this network
            (initial feature extractor, incremental classifier and outlier detector) with caution because the number
            of output features of the initial feature extractor and input features of the incremental classifier and
            outlier detector *must* match. Also, the outlier detector must have *only one* node in its output layer
        :param experience: experience
        :return: None
        """
        if issubclass(self.init_feature_extractor.__class__, DynamicModule):
            self.init_feature_extractor.adaptation(experience)
        if issubclass(self.subnetworks["outlier_detector"].__class__, DynamicModule):
            self.subnetworks["outlier_detector"].adaptation(experience)
        self.subnetworks["incremental_classifier"].adaptation(experience)

    def eval_adaptation(self, experience: CLExperience):
        """
        Adaptation of the network at evaluation time. Nothing is performed.
        :param experience:
        :return: None
        """
        pass

    def forward(self, x: Tensor,
                subnetwork_ids: Optional[Union[Sequence[Literal["incremental_classifier", "outlier_detector"]],
                                               Literal["all", "incremental_classifier", "outlier_detector"]]] = "all",
                use_init_feature_extractor: bool = True) -> Dict[str, Any]:
        """
        Compute the forward pass given the input tensor.
        :param x: input tensor
        :param subnetwork_ids: (optional) `incremental_classifier` or `outlier_detector` for each sample or only one
            of them.
            If None,
            the method :meth:`IncrementalClassifierOutlierDetectorWithInitFeatureExtractor.forward_no_subnetwork_ids`
            is used.
            If the literal `all` is provided,
            the method :meth:`IncrementalClassifierOutlierDetectorWithInitFeatureExtractor.forward_all_subnetworks`
            is used. Default is `all`, thus for a given sample both the incremental classifier head and outlier detector
            are used.
        :param use_init_feature_extractor: (optional) whether to use the initial_feature extractor or not during the
            forward pass. Default is True
        :return: a dictionary with `incremental_classifier` and/or `outlier_detector` as keys and the output of the
            corresponding subnetwork as value if
            multiple subnetwork IDs are provided. The order of the samples in each subnetwork output matches the order
            of the corresponding samples in the input tensor. If one subnetwork ID is provided, the output is the output
            of
            the method :meth:`IncrementalClassifierOutlierDetectorWithInitFeatureExtractor.forward_single_subnetwork`.
            If `subnetworks_ids` is None, the output is the output of the
            method :meth:`IncrementalClassifierOutlierDetectorWithInitFeatureExtractor.forward_no_subnetwork_ids`.
            If `subnetworks_ids` is `all`, the output is the output of
            the method :meth:`IncrementalClassifierOutlierDetectorWithInitFeatureExtractor.forward_all_subnetworks`
        """
        return super().forward(x, subnetwork_ids, use_init_feature_extractor)


class DynamicNetworkWithInitFeatureExtractorAndIncrementalOutlierSubNetworks(DynamicNetworkWithInitFeatureExtractor):
    """
    A dynamic network that has an initial feature extractor and a number of subsequent
    :class:`IncrementalClassifierOutlierDetectorWithInitFeatureExtractor` subnetworks.

    An input is fed into the initial feature extractor, whose output is in turn fed into one or multiple subsequent
    subnetworks.
    Subclasses must override the methods:
        - :meth:`DynamicNetworkWithInitFeatureExtractorAndIncrementalOutlierSubNetworks._create_init_feature_extractor`
        - :meth:`DynamicNetworkWithInitFeatureExtractorAndIncrementalOutlierSubNetworks._create_subnetwork`
        - :meth:`DynamicNetworkWithInitFeatureExtractorAndIncrementalOutlierSubNetworks._create_subnetworks_experiences`

    This class allows the initial feature extractor to be always in eval mode, even when
    calling :meth:`DynamicNetworkWithInitFeatureExtractorAndIncrementalOutlierSubNetworks.train`.
    This is useful when only the subnetworks must be trained and the initial feature extractor is always kept
    frozen during training.
    This way, when calling :meth:`DynamicNetworkWithInitFeatureExtractorAndIncrementalOutlierSubNetworks.train`, the
    initial feature extractor will not be moved to the training mode accidentally.

    This class is a subclass of the class :class:`DynamicNetworkWithInitFeatureExtractor`
    """
    def __init__(self, init_feature_extractor_eval: bool = False):
        """
        Initialise a new DynamicNetworkWithInitFeatureExtractorAndIncrementalOutlierSubNetworks
        :param init_feature_extractor_eval: (optional) If True, the initial feature extractor is *always* in eval mode,
            even when calling :meth:`DynamicNetworkWithInitFeatureExtractorAndIncrementalOutlierSubNetworks.train`.
            This is useful when only the subnetworks must be trained and the initial feature extractor is always kept
            frozen during training. This way, when
            calling :meth:`DynamicNetworkWithInitFeatureExtractorAndIncrementalOutlierSubNetworks.train`, the initial
            feature extractor will not be moved to the training mode accidentally. Default is False.
        """
        # initially the initial feature extractor is set to a dummy linear layer
        super().__init__(init_feature_extractor=Linear(1, 1))

        self._init_feature_extractor_initialised: bool = False
        """boolean flag that indicates whether the initial feature extractor has been initialised"""

        self.init_feature_extractor_eval: bool = init_feature_extractor_eval
        """boolean flag the determines whether the initial feature extractor must be always in eval mode"""

        self._initialise_init_feature_extractor()  # the initial feature extractor is set to a PyTorch module thereafter

    def train_adaptation(self, experience: CLExperience):
        """
        Adaptation of the network at training time on the given experience.

        This method uses the method
        :meth:`DynamicNetworkWithInitFeatureExtractorAndIncrementalOutlierSubNetworks_create_subnetworks_experiences`
        to create a dictionary of subnetwork IDs and corresponding experiences. For each entry in this dictionary,
        this method first creates a subnetwork with the corresponding subnetwork ID if it does not exist yet, then it
        adapts the subnetwork to the corresponding experience.
        :param experience: experience
        :return:
        """
        subnetwork_experiences = self._create_subnetworks_experiences(experience)
        for subnetwork_id, exp in subnetwork_experiences.items():
            if subnetwork_id in self.subnetworks.keys():
                self.subnetworks[subnetwork_id].adaptation(exp)
            else:
                # the new subnetwork is passed to the device which stores the whole network
                self.subnetworks[subnetwork_id] = self._create_subnetwork().to(self._adaptation_device)
                self.subnetworks[subnetwork_id].adaptation(exp)

    def eval_adaptation(self, experience: CLExperience):
        """
        Adaptation of the network at evaluation time. Nothing is performed.
        :param experience:
        :return: None
        """
        pass

    @property
    def subnetworks_classes(self) -> Dict[str, List[Any]]:
        """
        Get the classes seen from each subnetwork
        :return: a dictionary containing the subnetwork IDs as keys and the corresponding classes seen from each
            subnetwork as values
        """
        out = {}
        for subnetwork_id, subnetwork in self.subnetworks.items():
            out[subnetwork_id] = subnetwork.subnetworks["incremental_classifier"].classifier.seen_classes
        return out

    def train(self, mode: bool = True):
        """
        Override this method so that if :attr:`init_feature_extractor_eval` is True, then the initial feature extractor
        is always in eval mode.
        :param mode: whether to set training mode (``True``) or evaluation mode (``False``). Default: ``True``.
        :return: this instance
        """
        super().train(mode)
        if self.init_feature_extractor_eval:
            self.init_feature_extractor.eval()
        return self

    def _initialise_init_feature_extractor(self):
        """
        Initialise the initial feature extractor. This method assigns the attribute
        :attr:`DynamicNetworkWithInitFeatureExtractor.init_feature_extractor` of the class
        :class:`DynamicNetworkWithInitFeatureExtractor` to the output of the method
        :meth:`DynamicNetworkWithInitFeatureExtractorAndIncrementalOutlierSubNetworks._create_init_feature_extractor`.

        The latter attribute is assigned only if the initial feature extractor has not been initialised yet as indicated
        by the boolean
        flag :attr:`DynamicNetworkWithInitFeatureExtractorAndIncrementalOutlierSubNetworks._init_feature_extractor_initialised`.

        This method *must* not be called.

        .note::
            If :attr:`DynamicNetworkWithInitFeatureExtractorAndIncrementalOutlierSubNetworks.init_feature_extractor_eval`
            is True, the initial feature extractor is set to eval mode.
        :return:
        """
        if self._init_feature_extractor_initialised is False:
            self.init_feature_extractor = self._create_init_feature_extractor().to(self._adaptation_device)
            if self.init_feature_extractor_eval:
                self.init_feature_extractor.eval()  # set the initial feature extractor to eval mode
            self._init_feature_extractor_initialised = True  # set the initial feature extractor flag to True

    def _create_init_feature_extractor(self) -> nn.Module:
        """
        Create the initial feature extractor. This method is called by the method
        :meth:`DynamicNetworkWithInitFeatureExtractorAndIncrementalOutlierSubNetworks._initialise_init_feature_extractor`
        at initialisation time.

        Subclasses *must* override this method.

        This method *must* not be called.

        .note::
            It is not necessary to pass the initial feature extractor into the device storing the whole network. The
            method `DynamicNetworkWithInitFeatureExtractorAndIncrementalOutlierSubNetworks._initialise_init_feature_extractor`
            will do it.

        .note::
            It is not necessary to move the initial feature extractor to eval mode
            if :attr:`DynamicNetworkWithInitFeatureExtractorAndIncrementalOutlierSubNetworks.init_feature_extractor_eval`
            is True. The
            method `DynamicNetworkWithInitFeatureExtractorAndIncrementalOutlierSubNetworks._initialise_init_feature_extractor`
            will do it.
        :return: the initial feature extractor
        """
        raise NotImplementedError("Subclasses must implement this method")

    def _create_subnetwork(self) -> IncrementalClassifierOutlierDetectorWithInitFeatureExtractor:
        """
        Create a subnetwork. The subnetwork *must* be a subclass of the class
        :class:`IncrementalClassifierOutlierDetectorWithInitFeatureExtractor`

        Subclasses *must* override this method.

        This method *must* not be called.

        .note::
            It is not necessary to pass the subnetwork into the device storing the whole network. The
            method `DynamicNetworkWithInitFeatureExtractorAndIncrementalOutlierSubNetworks.train_adaptation`
            will do it.
        :return: the subnetwork
        """
        raise NotImplementedError("Subclasses must implement this method")

    def _create_subnetworks_experiences(self, experience: CLExperience) -> Dict[str, CLExperience]:
        """
        From a given experience, create different experiences for each subnetwork that must undertake training on the
        new experience. This method is used by the method
        :meth:`DynamicNetworkWithInitFeatureExtractorAndIncrementalOutlierSubNetworks.train_adaptation` for adaptation
        on a new experience at training time.

        If a subnetwork does not yet exist, its subnetwork ID and corresponding experience *must* be included too.

        This method *must* not be called.
        :param experience: experience
        :return: dictionary containing the subnetwork IDs as key and the corresponding experiences as value
        """
        raise NotImplementedError("Subclasses must implement this method")
