from typing import Dict

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


class DynamicResNet18ClassIncremental(DynamicNetworkWithInitFeatureExtractorAndIncrementalOutlierSubNetworks):
    """
    A dynamic network based on the ResNet18 architecture for class incremental settings.
    Although this class is a subclass of :class:`DynamicNetworkWithInitFeatureExtractorAndIncrementalOutlierSubNetworks`,
    it models a single ResNet18 architecture with a dynamic output layer for the class incremental settings. This is
    achieved by:
        - setting the initial feature extractor to the identity function
        - assigning every new experience to the subnetwork with id 0, i.e. only one subnetwork is created during the training phase
        - assigning the feature extractor of subnetwork 0 to the identity function
        - assigning the outlier detector of subnetwork 0 to a dummy :class:`nn.Module` that always outputs 1
        - assigning the incremental classifier to a ResNet18 architecture with a :class:`NetworkWithIncrementalClassifierHead` as output layer
    """
    def __init__(self):
        """
        Create a new DynamicResNet18ClassIncremental
        :param weights_path: path to the file containing the weights of the model
        """
        # just set the initial feature extractor to be always in eval mode, in the end the init feature extractor is
        # just the identity function
        super().__init__(init_feature_extractor_eval=True)
        self.subnetworks["0"] = self._create_subnetwork()  # add at construction the unique subnetwork

    def _create_init_feature_extractor(self) -> nn.Module:
        """
        Create the initial feature extractor. The initial feature extractor is just the identity function
        :return: the initial feature extractor, that is the identity function
        """
        return nn.Identity()

    def _create_subnetwork(self) -> IncrementalClassifierOutlierDetectorWithInitFeatureExtractor:
        """
        Create a subnetwork.
        A subnetwork is an instance of the class :class:`IncrementalClassifierOutlierDetectorWithInitFeatureExtractor`.
        The subnetwork is just a ResNet18 architecture with
        a :class:`NetworkWithIncrementalClassifierHead` as output layer. This is achieved by:
            - assigning the feature extractor of the subnetwork to the identity function
            - assigning the outlier detector of the subnetwork to a dummy :class:`nn.Module` that always outputs 1
            - assigning the incremental classifier to a ResNet18 architecture
              with a :class:`NetworkWithIncrementalClassifierHead` as output layer

        :return: a subnetwork instance of :class:`IncrementalClassifierOutlierDetectorWithInitFeatureExtractor`
        """
        resnet18 = timm.create_model("resnet18", pretrained=False)
        resnet18.fc = nn.Identity()
        incremental_classifier = NetworkWithIncrementalClassifierHead(network=resnet18, in_features=512)
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

