from typing import Dict, Tuple, Protocol, List, Sequence, Literal, Union, Iterable, Any, Optional
from numbers import Number
from itertools import chain, combinations
import importlib
import os

from scipy import stats
from scipy.stats import spearmanr
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


##################################Stylize seaborn############################
# Set seaborn-white style
sns.set_style("white")

# NeurIPS style settings
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7
})

# Colorblind palette
colors = sns.color_palette("colorblind")
##################################Stylize seaborn############################


##################################Lists of keys for modules############################
N_REHEARSAL_PER_CLASS = [40, 100, 200]
"""
Number of rehearsal samples per class stored at the end of each incremental step
"""

CIFAR100_10_CLS_1 = [("cifar100", "resnet32", 10, 10, r, 1) for r in N_REHEARSAL_PER_CLASS]
"""
All the keys for the modules containing the second experience of experiments run with cifar100, resnet32, 10 increments
of 10 classes each.
"""

CIFAR100_10_CLS_2 = [("cifar100", "resnet32", 10, 10, r, 2) for r in N_REHEARSAL_PER_CLASS]
"""
All the keys for the modules containing the third experience of experiments run with cifar100, resnet32, 10 increments
of 10 classes each.
"""

CIFAR100_5_CLS_1 = [("cifar100", "resnet32", 5, 20, r, 1) for r in N_REHEARSAL_PER_CLASS]
"""
All the keys for the modules containing the second experience of experiments run with cifar100, resnet32, 5 increments
of 20 classes each.
"""

CIFAR100_5_CLS_2 = [("cifar100", "resnet32", 5, 20, r, 2) for r in N_REHEARSAL_PER_CLASS]
"""
All the keys for the modules containing the third experience of experiments run with cifar100, resnet32, 5 increments
of 20 classes each.
"""

TINYIN_10_CLS_1 = [("tinyimagenet", "resnet18", 10, 20, r, 1) for r in N_REHEARSAL_PER_CLASS]
"""
All the keys for the modules containing the second experience of experiments run with tinyimagenet, resnet18,
10 increments of 20 classes
each.
"""

TINYIN_10_CLS_2 = [("tinyimagenet", "resnet18", 10, 20, r, 2) for r in N_REHEARSAL_PER_CLASS]
"""
All the keys for the modules containing the third experience of experiments run with tinyimagenet, resnet18,
10 increments of 20 classes
each.
"""

TINYIN_5_CLS_1 = [("tinyimagenet", "resnet18", 5, 40, r, 1) for r in N_REHEARSAL_PER_CLASS]
"""
All the keys for the modules containing the second experience of experiments run with tinyimagenet, resnet18,
5 increments of 40 classes each.
"""

TINYIN_5_CLS_2 = [("tinyimagenet", "resnet18", 5, 40, r, 2) for r in N_REHEARSAL_PER_CLASS]
"""
All the keys for the modules containing the third experience of experiments run with tinyimagenet, resnet18,
5 increments of 40 classes each.
"""

ALL_CIFAR_1 = CIFAR100_10_CLS_1 + CIFAR100_5_CLS_1
"""
All the keys for the modules containing the second experience of experiments run with cifar100.
"""

ALL_CIFAR_2 = CIFAR100_10_CLS_2 + CIFAR100_5_CLS_2
"""
All the keys for the modules containing the third experience of experiments run with cifar100.
"""

ALL_TINYIN_1 = TINYIN_10_CLS_1 + TINYIN_5_CLS_1
"""
All the keys for the modules containing the second experience of experiments run with tinyimagenet.
"""

ALL_TINYIN_2 = TINYIN_10_CLS_2 + TINYIN_5_CLS_2
"""
All the keys for the modules containing the third experience of experiments run with tinyimagenet.
"""

BENCHMARKS = [("ALL_CIFAR_1", ALL_CIFAR_1), ("ALL_TINYIN_1", ALL_TINYIN_1),
              ("ALL_CIFAR_2", ALL_CIFAR_2), ("ALL_TINYIN_2", ALL_TINYIN_2)]
##################################Lists of keys for modules############################


##################################Classes############################
class ExperimentModule(Protocol):
    """
    A protocol for a module containing data about experiments.
    """

    EXP_ID: int
    """
    Experiments ID
    """

    DATASET: str
    """
    Name of the dataset
    """

    NETWORK: str
    """
    Name of the network architecture
    """

    N_EXPS: int
    """
    Number of experiences actually performed in the experiments.
    """

    N_EXPS_TOTAL: int
    """
    Number of experiences that could be run theoretically
    """

    N_CLASSES_PER_EXP: int
    """
    Number of classes per experience
    """

    N_REHEARSAL: int
    """
    Total number of samples in the rehearsal buffer after the first experience
    """

    N_CLASS_ORDERS: int
    """
    Number of class orders in the experiments
    """

    N_SEEDS: int
    """
    Number of seeds
    """

    N_EPOCHS: int
    """
    Number of epochs
    """

    MB_SIZE_SECOND_EXP: int
    """
    Mini-batch size during the second experience
    """

    N_TRAINING_EXAMPLES_PER_CLASS: int
    """
    Number of training examples per class
    """

    N_THEORY_MEASURES: int
    """
    Number of measures derived from the theoretical analysis
    """

    DATA_PATH: str
    """
    Path to the extracted data
    """

    accuracy: Dict[int, List[Tuple[np.ndarray, np.ndarray]]]
    """
    a dictionary containing the class order indexes as keys and respective lists as values. Each list contains as many
    elements as the number of experiences. Each element is a 2-tuple where both elements are 2D numpy arrays of size NxC,
    where N is equal to the number of seeds and C is equal to the number of unique classes encountered up to the current
    experience. The ith, jth element in the first numpy array is the training accuracy of class j at the end of the current
    experience for the ith seed. The ith, jth element in the second numpy array is the test accuracy of class j at the end
    of the current experience for the ith seed. 
    """

    forgetting: Dict[int, List[Tuple[np.ndarray, np.ndarray]]]
    """
    a dictionary containing the class order indexes as keys and respective lists as values. Each list contains as many
    elements as the number of experiences - 1; this is because the first experience is skipped. Each element is a 2-tuple
    where both elements are 2D numpy arrays of size NxC, where N is equal to the number of seeds and C is equal to the
    number of unique classes encountered up to the previous experience. The ith, jth element in the first numpy array is the
    training forgetting of class j at the end of the current experience relative to its training accuracy at the end of the
    previous experience for the ith seed. The ith, jth element in the second numpy array is the test forgetting of class j
    at the end of the current experience relative to its test accuracy at the end of the previous experience for the ith
    seed. 
    """

    exemplar_indices: Dict[int, List[np.ndarray]]
    """
    a dictionary containing the class order indexes as keys and respective lists as values. Each list contains as many
    elements as the number of experiences. Each element is a 2D numpy array of size NxC, where N is equal to the number of
    seeds and C is equal to the number of exemplars in the rehearsal buffer after each experience. The ith row in each numpy
    array contains the indices of all the samples selected for rehearsal at the end of the current experience for the ith
    seed.
    """

    conf_matrix: Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]
    """
    a dictionary containing the class order indexes as keys and respective dictionaries as values. The latter dictionaries
    contain the experience indexes as keys and respective 4-tuples as values. All elements in the 4-tuples are 3D numpy
    arrays of size NxCxC, where N is equal to the number of seeds and C is equal to the number of unique classes encountered
    up to the current experience. Specifically, all elements in the 4-tuples contain N confusion matrices, one for each
    seed. While the confusion matrices in the first and second element of the 4-tuples are computed prior to perform
    training on the given experience, the confusion matrices in the third and fourth element of the 4-tuples are computed
    after performing training on the given experience. The confusion matrices in the first and third elements are training
    confusion matrices. The confusion matrices in the second and fourth elements are test confusion matrices.
    """

    softmax_prob: Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray]]]
    """
    a dictionary containing the class order indexes as keys and respective dictionaries as values. The latter dictionaries
    contain the experience indexes as keys and respective 2-tuples as values. The first experience, the experience with
    index 0, is skipped. Both elements in the 2-tuples are 2D numpy arrays of size NxC, where N is equal to the number of
    seeds and C is equal to the number of unique classes encountered up to the current experience.
    The ith, jth element in the first array is the average softmax probability for class j computed across all training
    samples in the current experience (only the new classes) for the ith seed prior to perform training on the current
    experience.
    The ith, jth element in the second array is the average softmax probability for class j computed across all test
    samples in the current experience (only the new classes) for the ith seed prior to perform training on the current
    experience.
    """

    logits: Dict[int, Dict[int, np.ndarray]]
    """
    a dictionary containing the class order indexes as keys and respective dictionaries as values. The latter dictionaries
    contain the experience indexes as keys and respective numpy arrays as values. The first experience, the experience with
    index 0, is skipped. Each array is a 2D numpy array of size NxC, where N is equal to the number of seeds and C is equal
    to the number of unique classes encountered up to the current experience.
    The ith, jth element in each array is the average softmax probability for class j computed across all training
    samples in the current experience (only the new classes) for the ith seed prior to perform training on the current
    experience.
    """

    theory_measures: Dict[int, np.ndarray]
    """
    a dictionary containing the class order indexes as keys and respective arrays as values. Each array is a 4D numpy array
    of size TxExNxC, where T is equal to the number of theoretical measures derived from the theoretical analysis (11),
    E is equal to the number of training epochs for the second/third experience +1---this is because it
    contains one element computed prior to perform training on the second/third experience---N is equal to the number
    of seeds, C is equal to the number of old classes during the second/third experience.
        - The first theoretical measure is (f_c @ b_c) / ||f_c||
        - The second theoretical measure is (f_c @ b_c^{others}) / ||f_c||
        - The third theoretical measure is (f_c @ f^{bias}) / ||f_c||
        - The fourth theoretical measure is (f_c @ f^{bias-except-self}) / ||f_c||
        - The fifth theoretical measure is (f_c @ f^{no-bias}) / ||f_c||
        - The sixth theoretical measure is (f_c(c-row) @ b_c^{others}(c-row)) / ||f_c(c-row)||
        - The seventh theoretical measure is (f_c(c-row) @ f^{bias}(c-row)) / ||f_c(c-row)||
        - The eighth theoretical measure is (f_c(c-row) @ f^{bias-except-self}(c-row)) / ||f_c(c-row)||
        - The ninth theoretical measure is (f_c(c-row) @ f^{no-bias}(c-row)) / ||f_c(c-row)||
        - The tenth theoretical measure is ||f_c||
        - The eleventh theoretical measure is ||f_c(c-row)||

    where
        - f_c is the average gradient of class c computed using its full training dataset
        - b_c = r_c - f_c, where r_c is the average gradient of class c computed using its rehearsal set
        - b_c^{others} is the sum of the bias components of the other old classes (all old classes except for c)
        - f^{bias} is the sum of the average gradients of all old classes computed using their full training dataset (it 
          includes class c)
        - f^{bias-except-self} is the sum of the average gradients of the other old classes (all old classes except for c)
          computed using their full training dataset
        - f^{no-bias} is the sum of the average gradients of the new classes computes using their full training dataset
        - f_c(c-row) is the average gradient of class c computed using its full training dataset wrt only its class-specific
          parameters (final-layer row of class c + its bias)
        - b_c^{others}(c-row) is the sum of the bias components of the other old classes (all old classes except for c) wrt
          only the class-specific parameters of c (final-layer row of class c + its bias)
        - f^{bias}(c-row) is the sum of the average gradients of all old classes computed using their full training dataset
          (it includes class c) wrt only the class-specific parameters of c (final-layer row of class c + its bias)
        - f^{bias-except-self}(c-row) is the sum of the average gradients of the other old classes (all old classes except
          for c) computed using their full training dataset wrt only the class-specific parameters of c (final-layer row of
          class c + its bias)
        - f^{no-bias}(c-row) is the sum of the average gradients of the new classes computes using their full training
          dataset wrt only the class-specific parameters of c (final-layer row of class c + its bias)
    """
##################################Classes############################

##################################Collect the experiment modules############################


modules_dict: Dict[Tuple[str, str, int, int, int, int], ExperimentModule] = {}
"""
A dictionary containing 6-tuples as keys and respective modules as values.
The first element in each  6-tuple is the name of the dataset used by the respective module;
the second element is the name of the network architecture;
the third element is the number of total experiences;
the fourth element is the number of classes per experience;
the fifth element is the total number of rehearsal samples stored at the end of each incremental step per class
the sixth element is the experience index
"""
# fill `modules_dict` with the experiment modules present in this folder
for name in os.listdir():
    if name.endswith('.py') and "diff_classes" in name:
        name = name.removesuffix(".py")
        splits = name.split('_')
        if len(splits) in [11, 12]:
            key = (splits[1], splits[2], int(splits[3]), int(splits[5]), int(splits[7]) // int(splits[5]))
            if len(splits) == 11:
                key = key + (1,)
            else:
                key = key + (2,)
            module: ExperimentModule = importlib.import_module(name)
            modules_dict[key] = module
##################################Collect the experiment modules############################


##################################Utilities############################
def collect_measures(label_measures: Sequence[str],
                     module: ExperimentModule) -> np.ndarray:
    """
    Collect the given measures from the given module.
    :param label_measures: a sequence of measures that need to be collected
    :param module: a module that contains the given measures
    :return: a 3D numpy array of size NxCxM, where N is the number of training runs used in the module, C is the number
        of classes used in the module and M is the number of measures collected. Note that N=OxS, where O is the number
        of class orders used in the module and S is the number of seeds used per class order in the module.
    """
    if len(label_measures) == 0:
        raise ValueError("`label_measures cannot be empty`")
    if len(set(label_measures)) != len(label_measures):
        raise ValueError("`label_measures cannot contain duplicates`")

    measures_list: List[np.ndarray] = []
    # proportion1
    prop1 = -(((module.N_REHEARSAL / module.N_CLASSES_PER_EXP) / module.N_REHEARSAL) / 2)
    if module.N_EXPS == 3:  # if it is a module for the third experience
        prop1 = prop1 / 2
    # proportion2
    prop2 = -((module.N_TRAINING_EXAMPLES_PER_CLASS / (module.N_TRAINING_EXAMPLES_PER_CLASS * module.N_CLASSES_PER_EXP))/2)
    # labels, with same order, for the 11 theoretical measures
    theory_measures_names = ["SBI", "CBI", "PCI", "PCI_EXS", "NDI", "CBI-S", "PCI-S", "PCI_EXS-S", "NDI-S"]
    for class_order in range(module.N_CLASS_ORDERS):
        for seed in range(module.N_SEEDS):
            measures = []
            # append the theoretical measures in label_measures
            for l in label_measures:
                if l == "CF":
                    class_forgetting = (module.forgetting[class_order][-1][1][seed]).copy()
                    accuracy_first_exp = module.accuracy[class_order][0][1][seed]
                    class_forgetting[:module.N_CLASSES_PER_EXP] = (class_forgetting[:module.N_CLASSES_PER_EXP] /
                                                                   accuracy_first_exp)
                    if module.N_EXPS == 3:
                        accuracy_second_exp = module.accuracy[class_order][1][1][seed]
                        class_forgetting[module.N_CLASSES_PER_EXP:] = (class_forgetting[module.N_CLASSES_PER_EXP:] /
                                                                       accuracy_second_exp[module.N_CLASSES_PER_EXP:])
                    measure = class_forgetting
                elif l in theory_measures_names:
                    index = theory_measures_names.index(l)
                    measure = module.theory_measures[class_order][index, :, seed, :].sum(axis=0)
                    if l in ["NDI", "NDI-S"]:
                        measure = measure * prop2
                    else:
                        measure = measure * prop1
                else:
                    raise ValueError("One of the provided label measures is not one of the theoretical measures")
                measures.append(measure)

            measures = np.column_stack(measures)
            measures_list.append(measures)

    measures_list = np.stack(measures_list, axis=0)
    return measures_list


def fisher_z_transform(x) -> np.ndarray:
    """
    Apply the fisher z-transform element-wise.
    :param x: an array-like input
    :return: a transformed numpy array
    """
    return np.arctanh(x)


def inverse_fisher_z_transform(x) -> np.ndarray:
    """
    Apply the inverse fisher z-transform element-wise.
    :param x: an array-like input
    :return: a transformed numpy array
    """
    return np.tanh(x)

##################################Utilities############################


if __name__ == "__main__":
    spearmans_dict = {}
    for key in ALL_CIFAR_1 + ALL_CIFAR_2 + ALL_TINYIN_1 + ALL_TINYIN_2:
        measures = collect_measures(["SBI", "NDI-S"], modules_dict[key])
        _, n_classes, _ = measures.shape
        spearmans = []
        for n in range(n_classes):
            spearmans.append(spearmanr(measures[:, n, 0], measures[:, n, 1])[0])
        spearmans_dict[key] = spearmans


    spearman_means_stds = {key: (inverse_fisher_z_transform(fisher_z_transform(value).mean()), np.asarray(value).std())
                           for key, value in spearmans_dict.items()}

    rows = []
    for key, values in spearmans_dict.items():
        for v in values:
            rows.append({"Key": str(key), r"$\rho$": v})

    df_long = pd.DataFrame(rows)

    plt.figure(figsize=(7.1, 8.8))
    sns.boxplot(x=r"$\rho$", y="Key", data=df_long, palette="tab10", showfliers=False)
    sns.swarmplot(data=df_long, x=r"$\rho$", y="Key",
                  edgecolor="black", alpha=0.5, color="black")
    plt.grid(axis='x', color='lightgray', linestyle='--', linewidth=0.5, alpha=0.9)
    plt.grid(axis='y', color='lightgray', linestyle='-', linewidth=0.5, alpha=0.5)
    plt.xticks([-0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.yticks([str(key) for key in spearmans_dict.keys()],
               [ rf"${int(100 / key[2])}\% \,\text{{Cls.}},\, {int(key[4] / 500 * 100)}\% \, \text{{Reh.}}, \, \hat{{\Omega}}_{{\mathrm{{{'C100' if key[0] == 'cifar100' else 'TIN'}}}}}^{{{key[-1]+1}}}$" for key in spearmans_dict.keys()])
    plt.ylabel("")
    plt.axvline(x=0, color="red", linestyle='--', linewidth=0.5, alpha=0.9)
    plt.tight_layout()
    plt.savefig("controlled_exps_distributions.pdf", bbox_inches='tight', pad_inches=0.02)
    plt.show()
