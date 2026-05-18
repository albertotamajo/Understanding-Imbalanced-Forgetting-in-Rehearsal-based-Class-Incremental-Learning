from typing import Dict, Union, Optional

from models.dynamic_networks import (
    DynamicNetworkWithInitFeatureExtractorAndIncrementalOutlierSubNetworks,
    IncrementalClassifierOutlierDetectorWithInitFeatureExtractor,
    NetworkWithIncrementalClassifierHead
)
from avalanche.benchmarks.scenarios import CLExperience
import torch
import torch.nn as nn
import timm


class OneOutput(nn.Module):
    def __init__(self):
        super().__init__()
        # this is used because the outlier detector must have a final linear layer with output size 1
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        x = torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device)
        # the linear layer is used here
        return self.linear(x)


class NetworkClassIncremental(DynamicNetworkWithInitFeatureExtractorAndIncrementalOutlierSubNetworks):
    """
    A single network for class incremental settings (classes can repeat across different experiences) based on a
    specific deep learning architecture. This network consists of a backbone network based on a specific deep learning
    architecture followed by a linear classification head that incrementally adds units whenever new classes are
    encountered.

    Although this class is a subclass
    of :class:`DynamicNetworkWithInitFeatureExtractorAndIncrementalOutlierSubNetworks`, it models a single network
    backbone followed by a dynamic linear classification head.
    This is achieved by:
        - setting the initial feature extractor to the identity function
        - assigning every new experience to the subnetwork with id 0, i.e. only one subnetwork is created during the
          training phase
        - assigning the feature extractor of subnetwork 0 to the identity function
        - assigning the outlier detector of subnetwork 0 to a dummy :class:`nn.Module` that always outputs 1
        - assigning the incremental classifier to a specific deep learning architecture by instantiating
          a :class:`NetworkWithIncrementalClassifierHead` whose backbone network is a specific deep learning
          architecture and the final linear classifier is a dynamic one
    """
    def __init__(self, model: Union[nn.Module, str], class_head_name: Optional[str] = None,
                 num_features: Optional[int] = None, **kwargs):
        """
        Create a new NetworkClassIncremental
        :param model: a deep learning architecture model or a valid model identifier from `timm.list_models` to use as
            backbone. `timm.create_model` is used to instantiate a backbone network of the specified deep learning
            architecture if a model identifier is provided.
        :param class_head_name: name of the attribute storing the final linear classifier head of the backbone model.
            The linear classifier head is replaced by a :class:`nn.Identity`. A :class:`RuntimeError` is raised if such
            an attribute does not exist. If None and `model` is a model identifier, `num_classes=0` is provided
            as keyword argument when calling `timm.create_model`. When passing such an argument, *some* timm models
            automatically replace the final linear classifier with a :class:`nn.Identity`. Ensure this is what actually
            happens when instantiating your backbone network. A :class:`ValueError` is raised if it is None and `model`
            is a deep learning model. Default is None.
        :param num_features: the size of the feature vectors returned by the backbone network prior to being fed into
            the linear classifier head. If None, the size of the feature vectors is
            retrieved by looking at the `num_features` attribute of `model`. Note that *not all* timm models have such
            an attribute. If `model` is a deep learning model, it *should* have such an attribute.
            If such an attribute does not exist, a :class:`RuntimeError` is raised. Default is None.
        :param kwargs: some keyword arguments to be passed to `timm.create_model` when instantiating the backbone
            network. They are not used if `model` is a deep learning model.
        """
        if isinstance(model, nn.Module) and class_head_name is None:
            raise ValueError("`class_head_name` must be provided when `model` is an `nn.Module` instance")
        # just set the initial feature extractor to be always in eval mode, in the end the init feature extractor is
        # just the identity function
        super().__init__(init_feature_extractor_eval=True)

        self._class_head_name: Optional[str] = class_head_name
        """
        name of the attribute storing the final linear classifier head of the backbone model.
        The linear classifier head is replaced by a :class:`nn.Identity`. A :class:`RuntimeError` is raised if such
        an attribute does not exist. If None and a model identifier is provided at initialisation,
        `num_classes=0` is provided as keyword argument when calling `timm.create_model`. When passing such an argument,
        *some* timm models automatically replace the final linear classifier with a :class:`nn.Identity`.
        Ensure this is what actually happens when instantiating your backbone network.
        """

        self._num_features: Optional[int] = num_features
        """
        the size of the feature vectors returned by the backbone network prior to being fed into
        the linear classifier head. If None, the size of the feature vectors is
        retrieved by looking at the `num_features` attribute of the backbone network. Note that *not all* timm models
        have such an attribute. If a deep learning model is directly provided at initialisation, it *should* have such
        an attribute. If such an attribute does not exist, a :class:`RuntimeError` is raised.
        """

        self._kwargs = kwargs
        """
        some keyword arguments to be passed to `timm.create_model` when instantiating the backbone network.
        They are not used if a deep learning model is provided directly at initialisation.
        """

        self.subnetworks["0"] = self._create_subnetwork(model=model)  # add at construction the unique subnetwork

    def _create_init_feature_extractor(self) -> nn.Module:
        """
        Create the initial feature extractor. The initial feature extractor is just the identity function
        :return: the initial feature extractor, that is the identity function
        """
        return nn.Identity()

    def _create_subnetwork(self, model: Optional[Union[nn.Module, str]] = None) \
            -> IncrementalClassifierOutlierDetectorWithInitFeatureExtractor:
        """
        Create a subnetwork.
        A subnetwork is an instance of the class :class:`IncrementalClassifierOutlierDetectorWithInitFeatureExtractor`.
        The subnetwork consists of a backbone network based on a specific deep learning architecture followed by a
        linear classification head that incrementally adds units whenever new classes are encountered.
        This is achieved by:
            - assigning the feature extractor of the subnetwork to the identity function
            - assigning the outlier detector of the subnetwork to a dummy :class:`nn.Module` that always outputs 1
            - assigning the incremental classifier to a specific deep learning architecture by instantiating
              a :class:`NetworkWithIncrementalClassifierHead` whose backbone network is a specific deep learning
              architecture and the final linear classifier is a dynamic one
        :param model: a deep learning architecture model or a valid model identifier from `timm.list_models` to use as
            backbone. If None, a :class:`ValueError` is raised. Default is None.
        :return: a subnetwork instance of :class:`IncrementalClassifierOutlierDetectorWithInitFeatureExtractor`
        """
        if model is None:
            raise ValueError("A subnetwork cannot be created if `model` is None")

        if isinstance(model, nn.Module):
            backbone = model
        else:
            kwargs = self._kwargs
            if self._class_head_name is None:
                kwargs = {**kwargs, **{"num_classes": 0}}
            backbone = timm.create_model(model_name=model, **kwargs)
        if self._class_head_name is not None:
            if not hasattr(backbone, self._class_head_name):
                raise RuntimeError(f"Attribute {self._class_head_name} does not exist in the backbone instance")
            setattr(backbone, self._class_head_name, nn.Identity())
        num_features = self._num_features
        if num_features is None:
            if not hasattr(backbone, "num_features"):
                raise RuntimeError(f"Attribute `num_features` does not exist in the backbone instance")
            num_features = backbone.num_features
        incremental_classifier = NetworkWithIncrementalClassifierHead(network=backbone, in_features=num_features)
        outlier_detector = OneOutput()
        return IncrementalClassifierOutlierDetectorWithInitFeatureExtractor(nn.Identity(), incremental_classifier,
                                                                            outlier_detector)

    def _create_subnetworks_experiences(self, experience: CLExperience) -> Dict[str, CLExperience]:
        """
        Assign every experience to subnetwork 0.
        :param experience: experience
        :return: a dictionary containing a single entry <"0", the current experience>
        """
        return {"0": experience}
