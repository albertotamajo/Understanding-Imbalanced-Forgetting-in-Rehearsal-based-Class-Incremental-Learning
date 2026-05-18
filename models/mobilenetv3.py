from typing import Optional, Dict, Tuple
import copy
from collections import defaultdict

from models.dynamic_networks import (
    DynamicNetworkWithInitFeatureExtractorAndIncrementalOutlierSubNetworks,
    IncrementalClassifierOutlierDetectorWithInitFeatureExtractor,
    NetworkWithIncrementalClassifierHead
)
import torch
import torch.nn as nn
import timm
from timm.models.mobilenetv3 import MobileNetV3
from avalanche.benchmarks.scenarios import CLExperience


class MobileNetV3Large:
    """
    Class with static methods to deal with the MobileNetV3-Large network
    """
    @staticmethod
    def create_instance(weights_path: Optional[str] = None, num_classes: int = 1000,
                        slice_indexes: Optional[Tuple[int, int]] = None) -> MobileNetV3:
        """
        Create a mobilenetv3-large network or a sliced-up version of it.
        :param weights_path: (optional) path to the file containing the weights of the network or None for random
            initialization. Default is None.
        :param num_classes: (optional) number of classes to classify (number of nodes in the classifier head).
            Default is 1000.
        :param slice_indexes: (optional) tuple with two elements: start_index and end_index. These two indexes are used
            to select only those layers falling within the specified range, starting from the provided start index
            (inclusive) and ending at the specified end index (exclusive) as illustrated in the method
            :meth:`MobileNetV3Large.slice_network`. The network is sliced up after loading the external weights.
            If None, the mobilenetv3-large network is not sliced up. Default is None.
        :return: a mobilenetv3-large network or a sliced-up version of it.
        """
        if not (weights_path is None or isinstance(weights_path, str)):
            raise TypeError("Input weights_path must be None or of type str")
        if not isinstance(num_classes, int):
            raise TypeError("Input num_classes must be of type int")

        def load_model_weights(model: nn.Module, weights_path: str) -> nn.Module:
            """
            Loads the weights into a PyTorch model
            :param model: PyTorch model
            :param weights_path: path to the file containing the weights
            :return: the PyTorch model with loaded weights
            """
            state = torch.load(weights_path, map_location='cpu')
            for key in model.state_dict():
                if 'num_batches_tracked' in key:
                    continue
                p = model.state_dict()[key]
                if key in state:
                    ip = state[key]
                    if p.shape == ip.shape:
                        p.data.copy_(ip.data)  # Copy the data of parameters
                    else:
                        print("Could not load layer: {}, mismatch shape {} ,{}".format(key, (p.shape), (ip.shape)))
                else:
                    print("Could not load layer: {}, not in checkpoint".format(key))
            return model

        mobilenetv3_large = timm.create_model('mobilenetv3_large_100', pretrained=False,
                                              num_classes=num_classes)
        if weights_path is not None:
            mobilenetv3_large = load_model_weights(mobilenetv3_large, weights_path)
        if slice_indexes is not None:
            mobilenetv3_large = MobileNetV3Large.slice_network(slice_indexes[0], slice_indexes[1],
                                                               mobilenetv3_large)
        return mobilenetv3_large

    @staticmethod
    def slice_network(start_index: int, end_index: int, mobilenetv3_large: MobileNetV3) -> MobileNetV3:
        """
        Slice up a MobileNetV3-Large network. The selected layers are those falling within the
        specified range, starting from the provided start index (inclusive) and ending at the specified end
        index (exclusive).
        The MobileNetV3-Large architecture comprises 20 layers, indexed from 0 to 19.
        .note::
            The remaining layers are not removed; they are set to the identity function.
        :param start_index: starting layer index (included)
        :param end_index: ending layer index (excluded)
        :param mobilenetv3_large: an instance of a mobilenetv3-large network
        :return: a deep copy of the input mobilenetv3-large instance including only the layers between the specified
            start index (included) and end index (excluded). The deep copy is moved to the same device of the original
            network. The device of the original network is assumed to be the device that holds the first parameter of
            the original network.
        """
        layers = {0: ["conv_stem", "bn1"],
                  1: ["blocks.0.0"],
                  2: ["blocks.1.0"],
                  3: ["blocks.1.1"],
                  4: ["blocks.2.0"],
                  5: ["blocks.2.1"],
                  6: ["blocks.2.2"],
                  7: ["blocks.3.0"],
                  8: ["blocks.3.1"],
                  9: ["blocks.3.2"],
                  10: ["blocks.3.3"],
                  11: ["blocks.4.0"],
                  12: ["blocks.4.1"],
                  13: ["blocks.5.0"],
                  14: ["blocks.5.1"],
                  15: ["blocks.5.2"],
                  16: ["blocks.6.0"],
                  17: ["global_pool"],
                  18: ["conv_head", "act2", "flatten"],
                  19: ["classifier"]
                  }
        if not 0 <= start_index <= 19:
            raise ValueError("The start index must be between 0 and 19; both included")
        if not 0 <= end_index <= 20:
            raise ValueError("The end index must be between 0 and 20; both included")
        if not start_index <= end_index:
            raise ValueError("The start index must be less than or equal to the end index")
        if not isinstance(mobilenetv3_large, MobileNetV3):
            raise TypeError("Input mobilenetv3_large must be of type timm.models.mobilenetv3.MobileNetV3")
        device = next(mobilenetv3_large.parameters()).device
        mobilenetv3_large = copy.deepcopy(mobilenetv3_large).to(device)
        layers_to_delete = [layer for layer in range(20) if layer not in range(start_index, end_index)]
        for layer in layers_to_delete:
            for module_name in layers[layer]:
                if module_name.startswith("blocks"):
                    _, first_index, second_index = module_name.split(".")
                    mobilenetv3_large.blocks[int(first_index)][int(second_index)] = nn.Identity()
                else:
                    if module_name == "conv_stem":
                        mobilenetv3_large.conv_stem = nn.Identity()
                    elif module_name == "bn1":
                        mobilenetv3_large.bn1 = nn.Identity()
                    elif module_name == "global_pool":
                        mobilenetv3_large.global_pool = nn.Identity()
                    elif module_name == "conv_head":
                        mobilenetv3_large.conv_head = nn.Identity()
                    elif module_name == "act2":
                        mobilenetv3_large.act2 = nn.Identity()
                    elif module_name == "flatten":
                        mobilenetv3_large.flatten = nn.Identity()
                    elif module_name == "classifier":
                        mobilenetv3_large.classifier = nn.Identity()
        return mobilenetv3_large


class DynamicMobileNetV3LargeHardCodedImageNetCoarseClasses(
        DynamicNetworkWithInitFeatureExtractorAndIncrementalOutlierSubNetworks):
    """
    A dynamic network based on the MobileNetV3-Large architecture. The first n layers of the MobileNetV3-Large are
    allocated to the initial feature extractor, while the remaining 20 - n layers are allocated to each subnetwork.

    Each subnetwork comprises a feature extractor and two parallel networks: an incremental classifier and outlier
    detector. The incremental classifier is trained to classify the classes it has seen while the outlier detector is
    trained to recognise classes that have been trained on the respective subnetwork from all the other classes. The
    feature extractor of each subnetwork is allocated m layers of the MobileNetV3-Large architecture while the
    incremental classifier and outlier detector are allocated the subsequent 20 - n - m layers.
    Each subnetwork is a subclass of :class:`IncrementalClassifierOutlierDetectorWithInitFeatureExtractor`.

    An input is fed into the initial feature extractor, whose output is in turn fed into one or multiple subsequent
    subnetworks. The output of a subnetwork is a dictionary with `incremental_classifier` and/or `outlier_detector` as
    keys and the output of the corresponding subnetwork as value.

    At training time, the network is adapted to a new experience according to the provided
    ImageNet hard-coded coarse classes. The incremental classifier of each subnetwork is trained only on the classes
    belonging to a specific coarse class.
    """
    def __init__(self, n_init_feature_extractor_layers: int, n_subnetwork_feature_extractor_layers: int,
                 imagenet_coarse_classes: dict, init_feature_extractor_weights_path: Optional[str] = None,
                 init_feature_extractor_eval: bool = False):
        """
        Instantiate a new DynamicMobileNetV3LargeHardCodedImageNetCoarseClasses.

        The first `n` layers of the MobileNetV3-large architecture are allocated to the initial feature extractor,
        where `n` is the `n_init_feature_extractor_layers` parameter. The subsequent `m` layers of the
        MobileNetV3-large are allocated to the feature extractor of each subnetwork, where `m` is the
        `n_subnetwork_feature_extractor_layers` parameter. The remaining layers are allocated to the incremental
        classifier and outlier detector of each subnetwork.

        The number of layers of the MobileNetV3-large architecture is 20 (19 without the classifier head); thus,
        the sum of the layers of the initial feature extractor and the feature extractor of
        each subnetwork cannot be more than or equal to 20.
        :param n_init_feature_extractor_layers: number of layers of the initial feature extractor. It must be between 0
            and 19, both excluded.
        :param n_subnetwork_feature_extractor_layers: number of layers of the feature extractor of each subnetwork. It
            must be between 0 and 19, both excluded.
        :param imagenet_coarse_classes: a dictionary containing the classes of ImageNet1K as keys and another dictionary
            as values. The dictionary of each class *must* contain a key named `coarse_class` whose corresponding value
            is the coarse class the respective ImageNet1K class belongs to. Although the classes of Imagenet1K are
            denoted as integers, the keys of this dictionary *must* be integers wrapped as strings, such as "0" instead
            of 0. The name of a coarse class must be an integer number. The name of each subnetwork in this dynamic
            network will be the name of a coarse class as provided in this dictionary but wrapped as a string. All the
            classes belonging to the same coarse class are trained in the same subnetwork.
        :param init_feature_extractor_weights_path: path to the file containing the weights of the initial feature
            extractor or None for random initialization. Default is None
        :param init_feature_extractor_eval: (optional) If True, the initial feature extractor is *always* in eval mode,
            even when calling :meth:`DynamicMobileNetV3LargeHardCodedImageNetCoarseClasses.train`.
            This is useful when only the subnetworks must be trained and the initial feature extractor is always kept
            frozen during training. This way, when
            calling :meth:`DynamicNetworkWithInitFeatureExtractorAndIncrementalOutlierSubNetworks.train`, the initial
            feature extractor will not be moved to the training mode accidentally. Default is False.
        """
        if not 0 < n_init_feature_extractor_layers < 19:
            raise ValueError("The number of layers of the initial feature extractor must be between "
                             "0 and 19 both excluded")
        if not 0 < n_subnetwork_feature_extractor_layers < 19:
            raise ValueError("The number of layers of the feature extractor of each subnetwork must be between "
                             "0 and 19 both excluded")
        if not n_init_feature_extractor_layers + n_subnetwork_feature_extractor_layers < 20:
            raise ValueError("The number of layers of the MobileNetV3-large model is 20 "
                             "(19 without the classifier head), but the sum of the layers of the initial feature "
                             "extractor and the feature extractor of each subnetwork is more than or equal to 20")
        self.n_init_feature_extractor_layers = n_init_feature_extractor_layers
        """number of layers of the initial feature extractor"""

        self.n_subnetwork_feature_extractor_layers = n_subnetwork_feature_extractor_layers
        """number of layers of the feature extractor of each subnetwork"""

        self.imagenet_coarse_classes = imagenet_coarse_classes
        """
        a dictionary containing the classes of ImageNet1K as keys and another dictionary
        as values. The dictionary of each class *must* contain a key named `coarse_class` whose corresponding value is
        the coarse class the respective ImageNet1K class belongs to
        """

        self.init_feature_extractor_weights_path = init_feature_extractor_weights_path
        """path to the file containing the weights of the initial feature extractor"""

        super().__init__(init_feature_extractor_eval)

    def _create_init_feature_extractor(self) -> nn.Module:
        """
        Create the initial feature extractor.

        The initial feature extractor is the first `n` layers of the MobileNetV3-large architecture, where `n` is
        equal to `self._n_init_feature_extractor_layers`. The weights of the initial feature extractor are initialised
        according to `self.init_feature_extractor_weights_path`.
        :return: the first n layers of the MobileNetV3-large architecture
        """
        return MobileNetV3Large.create_instance(self.init_feature_extractor_weights_path,
                                                slice_indexes=(0, self.n_init_feature_extractor_layers))

    def _create_subnetwork(self) -> IncrementalClassifierOutlierDetectorWithInitFeatureExtractor:
        """
        Create a subnetwork.

        A subnetwork is an instance of the class :class:`IncrementalClassifierOutlierDetectorWithInitFeatureExtractor`.
        Each subnetwork comprises a feature extractor and two subsequent networks: an incremental classifier and an
        outlier detector.

        The number of layers of the feature extractor of each subnetwork is determined
        by `self.n_subnetwork_feature_extractor_layers`. The remaining layers are allocated to the incremental
        classifier and outlier detector.
        :return: a subnetwork instance of :class:`IncrementalClassifierOutlierDetectorWithInitFeatureExtractor`
        """
        end_index_sub_feature_extractor = self.n_init_feature_extractor_layers + self.n_subnetwork_feature_extractor_layers
        sub_feature_extractor = MobileNetV3Large.create_instance(slice_indexes=(self.n_init_feature_extractor_layers,
                                                                                end_index_sub_feature_extractor))
        if end_index_sub_feature_extractor < 19:  # if the incremental classifier and outlier detector of each
                                                  # subnetwork comprises part of the MobileNetV3-large architecture
                                                  # beyond a mere final linear layer
            incremental_classifier = NetworkWithIncrementalClassifierHead(MobileNetV3Large.create_instance(
                slice_indexes=(end_index_sub_feature_extractor, 19)), 1280)
            outlier_detector = MobileNetV3Large.create_instance(slice_indexes=(end_index_sub_feature_extractor, 20),
                                                                num_classes=1)
        else:  # if the incremental classifier and outlier detector of each subnetwork is just a final linear layer
            incremental_classifier = NetworkWithIncrementalClassifierHead(nn.Identity(), 1280)
            outlier_detector = nn.Linear(in_features=1280, out_features=1)

        return IncrementalClassifierOutlierDetectorWithInitFeatureExtractor(sub_feature_extractor,
                                                                            incremental_classifier,
                                                                            outlier_detector)

    def _create_subnetworks_experiences(self, experience: CLExperience) -> Dict[str, CLExperience]:
        """
        From a given experience, create different experiences for each subnetwork. According to the dictionary stored
        in the attribute `self.imagenet_coarse_classes`, each subnetwork will undertake training only on a subset of
        classes (the classes belonging to the same coarse class). The experience of each subnetwork is just a shallow
        copy of the original experience whose attribute `classes_in_this_experience` is modified so that to contain
        only the classes that should be trained on the respective subnetwork.

        .note::
            It is not necessary to perform a deep copy of the original experience, which is a rather slow operation,
            because setting the `classes_in_this_experience` attribute in the experience of each subnetwork to point to
            another list of classes will not affect the original experience. Additionally, the adaptation stage of
            each subnetwork, which is performed successively, will only read from the `classes_in_this_experience`
            attribute without modifying any internal object. Therefore, the original experience will not be affected.
        :param experience: experience
        :return: a dictionary containing the subnetwork IDs (equivalent to the coarse classes IDs in the attribute
        `self.imagenet_coarse_classes`) as keys and the respective experiences as values.
        """
        subnetworks_classes = defaultdict(list)
        classes_in_experience = experience.classes_in_this_experience
        for class_in_experience in classes_in_experience:
            # the coarse class is transformed into a string wrapping an int in case it is an int
            coarse_class = str(self.imagenet_coarse_classes[str(class_in_experience)]["coarse_class"])
            subnetworks_classes[coarse_class].append(class_in_experience)

        subnetworks_experiences: Dict[str, CLExperience] = {}
        for coarse_class, classes in subnetworks_classes.items():
            coarse_class_experience = copy.copy(experience)
            coarse_class_experience.classes_in_this_experience = classes
            subnetworks_experiences[coarse_class] = coarse_class_experience

        return subnetworks_experiences

