"""
This module allows extracting data from the experiments run by
`experiment28_cifar100_resnet32_5_increments_20_classes_4000.py`
"""

from typing import Dict, List, Tuple, Any
import dill
import pickle
import math
import torch
import numpy as np
from scipy.spatial.distance import cosine

EXP_ID = 28
"""
Experiment ID
"""

N_EXPS = 2
"""
Number of experiences actually performed in the experiments.
"""

N_EXPS_TOTAL = 5
"""
Number of experiences that could be run theoretically
"""

N_CLASSES_PER_EXP = 20
"""
Number of classes per experience in the experiments
"""

N_CLASS_ORDERS = 40
"""
Number of class orders in the experiments 
"""

N_SEEDS = 1
"""
Number of seeds in the experiments per class order
"""

TRAIN_EPOCHS = 170
"""
Number of training epochs in the experiments
"""

N_EXEMPLARS = 4000
"""
Number of exemplars in the rehearsal buffer after each experience
"""

N_FEATURES = 64
"""
Number of features in the backbone network used in the experiments
"""

N_THEORY_MEASURES = 11
"""
Number of measures derived from the theoretical analysis
"""

GRADIENT_SIZE_SECOND_EXP = (N_FEATURES*N_CLASSES_PER_EXP*2)+(N_CLASSES_PER_EXP*2)
"""
Size of the final-layer gradient during the second experience; it includes the number of parameters in the final-layer
weight matrix + the number of bias elements.
"""

CHECKPOINT_PATH = (f"experiment{EXP_ID}_cifar100_resnet32_{N_EXPS_TOTAL}_increments_{N_CLASSES_PER_EXP}" +
                   "_classes_{index_class_order}_{n_exemplars}_experience_{index_exp}.pth")
"""
Path of the checkpoints of the experiments
"""

SAVE_PATH = f"experiment{EXP_ID}_cifar100_resnet32_{N_EXPS_TOTAL}_increments_{N_CLASSES_PER_EXP}_classes_{N_EXEMPLARS}_data.pkl"
"""
Path where to save all the extracted data
"""

INDEX_REPLAY_PLUGIN = 2
"""
Index of the replay plugin
"""

INDEX_EVAL_PLUGIN = 3
"""
Index of the evaluation plugin
"""

accuracy: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {i: [(np.zeros((N_SEEDS, N_CLASSES_PER_EXP*(j+1))),
                                                                 np.zeros((N_SEEDS, N_CLASSES_PER_EXP*(j+1))))
                                                                for j in range(N_EXPS)]
                                                            for i in range(N_CLASS_ORDERS)}
"""
a dictionary containing the class order indexes as keys and respective lists as values. Each list contains as many
elements as the number of experiences. Each element is a 2-tuple where both elements are 2D numpy arrays of size NxC,
where N is equal to the number of seeds and C is equal to the number of unique classes encountered up to the current
experience. The ith, jth element in the first numpy array is the training accuracy of class j at the end of the current
experience for the ith seed. The ith, jth element in the second numpy array is the test accuracy of class j at the end
of the current experience for the ith seed. 
"""

forgetting: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {i: [(np.zeros((N_SEEDS, N_CLASSES_PER_EXP*j)),
                                                                   np.zeros((N_SEEDS, N_CLASSES_PER_EXP*j)))
                                                                  for j in range(1, N_EXPS)]
                                                              for i in range(N_CLASS_ORDERS)}
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

exemplar_indices: Dict[int, List[np.ndarray]] = {i: [np.zeros((N_SEEDS, N_EXEMPLARS), dtype=np.int64)
                                                     for _ in range(N_EXPS)]
                                                 for i in range(N_CLASS_ORDERS)}
"""
a dictionary containing the class order indexes as keys and respective lists as values. Each list contains as many
elements as the number of experiences. Each element is a 2D numpy array of size NxC, where N is equal to the number of
seeds and C is equal to the number of exemplars in the rehearsal buffer after each experience. The ith row in each numpy
array contains the indices of all the samples selected for rehearsal at the end of the current experience for the ith
seed.
"""

avg_gradients_full_data: Dict[int, np.ndarray] = {i: np.zeros((TRAIN_EPOCHS + 1, N_SEEDS, N_CLASSES_PER_EXP*2,
                                                               GRADIENT_SIZE_SECOND_EXP))
                                                  for i in range(N_CLASS_ORDERS)}
"""
a dictionary containing the class order indexes as keys and respective arrays as values. Each array is a 4D numpy array
of size ExNxCxL, where E is equal to the number of training epochs for the second experience +1---this is because it
contains one element computed prior to perform training on the second experience----N is equal to the number of seeds,
C is equal to the total number of classes during the second experience and L is equal to the size of the avg gradient
computed for the given class during the second experience for the given epoch and seed using the full training dataset
of the given class
"""

avg_gradients_replay_data: Dict[int, np.ndarray] = {i: np.zeros((TRAIN_EPOCHS + 1, N_SEEDS, N_CLASSES_PER_EXP,
                                                                 GRADIENT_SIZE_SECOND_EXP))
                                                    for i in range(N_CLASS_ORDERS)}
"""
a dictionary containing the class order indexes as keys and respective arrays as values. Each array is a 4D numpy array
of size ExNxCxL, where E is equal to the number of training epochs for the second experience +1---this is because it
contains one element computed prior to perform training on the second experience---N is equal to the number of seeds,
C is equal to the number of old classes during the second experience and L is equal to the size of the avg gradient
computed for the given class during the second experience for the given epoch and seed using the replay training dataset
of the given class
"""

theory_measures: Dict[int, np.ndarray] = {i: np.zeros((N_THEORY_MEASURES, TRAIN_EPOCHS + 1, N_SEEDS, N_CLASSES_PER_EXP))
                                          for i in range(N_CLASS_ORDERS)}
"""
a dictionary containing the class order indexes as keys and respective arrays as values. Each array is a 4D numpy array
of size TxExNxC, where T is equal to the number of theoretical measures derived from the theoretical analysis (11),
E is equal to the number of training epochs for the second experience +1---this is because it
contains one element computed prior to perform training on the second experience---N is equal to the number of seeds,
C is equal to the number of old classes during the second experience.
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

conf_matrix: Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = \
    {i: {j: (
              np.zeros((N_SEEDS, N_CLASSES_PER_EXP*(j+1), N_CLASSES_PER_EXP*(j+1)), dtype=np.int64),
              np.zeros((N_SEEDS, N_CLASSES_PER_EXP*(j+1), N_CLASSES_PER_EXP*(j+1)), dtype=np.int64),
              np.zeros((N_SEEDS, N_CLASSES_PER_EXP*(j+1), N_CLASSES_PER_EXP*(j+1)), dtype=np.int64),
              np.zeros((N_SEEDS, N_CLASSES_PER_EXP*(j+1), N_CLASSES_PER_EXP*(j+1)), dtype=np.int64)
            )
         for j in range(N_EXPS)}
     for i in range(N_CLASS_ORDERS)}
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

softmax_prob: Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray]]] = \
    {i: {j: (
              np.zeros((N_SEEDS, N_CLASSES_PER_EXP*(j+1))),
              np.zeros((N_SEEDS, N_CLASSES_PER_EXP*(j+1)))
            )
         for j in range(1, N_EXPS)}
     for i in range(N_CLASS_ORDERS)}
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


logits: Dict[int, Dict[int, np.ndarray]] = \
    {i: {j: np.zeros((N_SEEDS, N_CLASSES_PER_EXP*(j+1)))
         for j in range(1, N_EXPS)}
     for i in range(N_CLASS_ORDERS)}
"""
a dictionary containing the class order indexes as keys and respective dictionaries as values. The latter dictionaries
contain the experience indexes as keys and respective numpy arrays as values. The first experience, the experience with
index 0, is skipped. Each array is a 2D numpy array of size NxC, where N is equal to the number of seeds and C is equal
to the number of unique classes encountered up to the current experience.
The ith, jth element in each array is the average logir for class j computed across all training
samples in the current experience (only the new classes) for the ith seed prior to perform training on the current
experience.
"""


def insert_accuracy(metric_results: Dict[str, Any], index_class_order: int, index_seed: int, index_exp: int):
    train_acc, test_acc = accuracy[index_class_order][index_exp]
    for t in range(train_acc.shape[1]):
        exp = int(math.floor(t/N_CLASSES_PER_EXP))
        train_str = f'Class_Accuracy_Incremental_Classifier_Exp/eval_phase/train_stream/Task000/Exp00{exp}/{t}'
        test_str = f'Class_Accuracy_Incremental_Classifier_Exp/eval_phase/test_stream/Task000/Exp00{exp}/{t}'
        # insert the train accuracy
        train_acc[index_seed, t] = metric_results[train_str][1][-1]
        # insert the test accuracy
        test_acc[index_seed, t] = metric_results[test_str][1][-1]


def fill_forgetting_from_accuracy():
    for index_class_order, accuracies in accuracy.items():
        for index_exp, ((train_acc_0, test_acc_0), (train_acc_1, test_acc_1)) in enumerate(
                zip(accuracies, accuracies[1:])):
            train_forgetting, test_forgetting = forgetting[index_class_order][index_exp]
            train_forgetting[:, :] = train_acc_0 - train_acc_1[:, :train_acc_0.shape[1]]
            test_forgetting[:, :] = test_acc_0 - test_acc_1[:, :test_acc_0.shape[1]]


def insert_exemplar_indices(storage_policy, index_class_order: int, index_seed: int, index_exp: int):
    exemplar_indices[index_class_order][index_exp][index_seed] = np.asarray(list(storage_policy.buffer.dataset_indices))


def insert_conf_matrix(metric_results: Dict[str, Any], index_class_order: int, index_seed: int, index_exp: int):
    train_conf_matrix_prior, test_conf_matrix_prior, train_conf_matrix_after, test_conf_matrix_after = (
        conf_matrix)[index_class_order][index_exp]
    train_conf_matrix_prior, test_conf_matrix_prior, train_conf_matrix_after, test_conf_matrix_after = (
        train_conf_matrix_prior[index_seed], test_conf_matrix_prior[index_seed], train_conf_matrix_after[index_seed],
        test_conf_matrix_after[index_seed])
    for i in range(index_exp + 1):
        start_index = N_CLASSES_PER_EXP * i
        end_index = N_CLASSES_PER_EXP * (i+1)
        train_str = f'Confusion_Matrix_Incremental_Classifier_Exp/eval_phase/train_stream/Task000/Exp00{i}'
        test_str = f'Confusion_Matrix_Incremental_Classifier_Exp/eval_phase/test_stream/Task000/Exp00{i}'
        train_conf_matrix_prior[start_index: end_index] = [e for e in metric_results[train_str][1]
                                                           if len(e) == N_CLASSES_PER_EXP * (index_exp + 1)][0][start_index: end_index]
        test_conf_matrix_prior[start_index: end_index] = [e for e in metric_results[test_str][1]
                                                          if len(e) == N_CLASSES_PER_EXP * (index_exp + 1)][0][start_index: end_index]
        train_conf_matrix_after[start_index: end_index] = metric_results[train_str][1][-1][start_index: end_index]
        test_conf_matrix_after[start_index: end_index] = metric_results[test_str][1][-1][start_index: end_index]


def insert_softmax_prob(metric_results: Dict[str, Any], index_class_order: int, index_seed: int, index_exp: int):
    if index_exp > 0:
        train_softmax_prob, test_softmax_prob = softmax_prob[index_class_order][index_exp]
        train_str = f'LogitEmbedding/eval_phase/train_stream/Task000/Exp00{index_exp}'
        test_str = f'LogitEmbedding/eval_phase/test_stream/Task000/Exp00{index_exp}'
        train_logits, *_ = metric_results[train_str][1][0]
        train_probs = train_logits.softmax(dim=1).cpu().numpy()
        test_logits, *_ = metric_results[test_str][1][0]
        test_probs = test_logits.softmax(dim=1).cpu().numpy()
        train_softmax_prob[index_seed] = np.mean(train_probs, axis=0)
        test_softmax_prob[index_seed] = np.mean(test_probs, axis=0)


def insert_logits(metric_results: Dict[str, Any], index_class_order: int, index_seed: int, index_exp: int):
    if index_exp > 0:
        train_logits_ = logits[index_class_order][index_exp]
        train_str = f'LogitEmbedding/eval_phase/train_stream/Task000/Exp00{index_exp}'
        train_logits, *_ = metric_results[train_str][1][0]
        train_logits = train_logits.cpu().numpy()
        train_logits_[index_seed] = np.mean(train_logits, axis=0)


def comp_grads_wrt_weights_biases(grads_biases: np.ndarray, embeds: np.ndarray) -> np.ndarray:
    """
    Compute the gradients wrt the weights and biases in the classification head from the gradients wrt the biases and
    the sample embeddings (feature vectors) for different samples
    :param grads_biases: 2d array where each row is a gradient wrt the biases in the classification for different
        samples
    :param embeds: 2d array where each row is the embedding (feature vector) for different samples
    :return: 2d array where each row is the gradient wrt the weights and biases in the classification head for different
        samples
    """
    grads_biases_expand = np.expand_dims(grads_biases, axis=2)
    embeds_expand = np.expand_dims(embeds, axis=1)
    grads_weights = np.matmul(grads_biases_expand, embeds_expand).reshape(len(grads_biases), -1)
    return np.concatenate((grads_weights, grads_biases), axis=1)


def insert_avg_grads(metric_results: Dict[str, Any], index_class_order: int, index_seed: int, index_exp: int):
    """
    It must be used only after the exemplar indices for the given class order, seed and experience-1 have been filled in
    exemplar_indices
    """
    if index_exp > 0:
        for i in range(index_exp + 1):
            train_str = f'LogitEmbedding/eval_phase/train_stream/Task000/Exp00{i}'
            start_index = [l for l, e in enumerate(metric_results[train_str][1])
                           if e[0].shape[1] == N_CLASSES_PER_EXP * (index_exp + 1)][0]
            end_index = start_index + TRAIN_EPOCHS + 1
            for n, (train_logit, train_embed, train_targets, train_indices) in enumerate(
                        metric_results[train_str][1][start_index: end_index]):
                train_soft_prob = train_logit.softmax(dim=1).cpu().numpy()
                train_embed = train_embed.cpu().numpy()
                train_targets = train_targets.cpu().numpy()
                train_indices = train_indices.cpu().numpy()
                train_grads_biases = train_soft_prob - np.eye(train_soft_prob.shape[1])[train_targets]
                train_grads_weights_biases = comp_grads_wrt_weights_biases(train_grads_biases, train_embed)
                for t in range(N_CLASSES_PER_EXP * i, N_CLASSES_PER_EXP * (i + 1)):
                    t_train_grads_weights_biases = train_grads_weights_biases[train_targets == t]
                    avg_gradients_full_data[index_class_order][n][index_seed, t] = t_train_grads_weights_biases.mean(
                        axis=0)
                    if i < index_exp:
                        t_train_indices = train_indices[train_targets == t]
                        exemplars_indexes = exemplar_indices[index_class_order][index_exp - 1][index_seed]
                        t_rehearsed_train_grads_weights_biases = t_train_grads_weights_biases[np.isin(t_train_indices,
                                                                                                      exemplars_indexes)]
                        avg_gradients_replay_data[index_class_order][n][index_seed, t] = t_rehearsed_train_grads_weights_biases.mean(axis=0)


def insert_theory_measures():
    for class_order in range(N_CLASS_ORDERS):
        for epoch in range(TRAIN_EPOCHS + 1):
            for seed in range(N_SEEDS):
                avg_full_gradients = avg_gradients_full_data[class_order][epoch, seed]
                avg_replay_gradients = avg_gradients_replay_data[class_order][epoch, seed]
                bias = avg_replay_gradients - avg_full_gradients[:N_CLASSES_PER_EXP]
                # Compute L2 norm  of each avg full gradient
                avg_full_gradients_norms = np.linalg.norm(avg_full_gradients, axis=1)
                for t in range(N_CLASSES_PER_EXP):
                    class_avg_grad = avg_full_gradients[t]
                    class_avg_grad_norm = avg_full_gradients_norms[t]
                    class_t_avg_grad = np.concatenate((
                        class_avg_grad[N_FEATURES * t: N_FEATURES * (t+1)],
                        class_avg_grad[N_FEATURES*N_CLASSES_PER_EXP:][t:(t+1)]
                    ))
                    class_t_avg_grad_norm = np.linalg.norm(class_t_avg_grad)


                    theory_measures[class_order][0, epoch, seed, t] = ((class_avg_grad @ bias[t])/
                                                                       class_avg_grad_norm)

                    sum_others_bias = bias[np.arange(N_CLASSES_PER_EXP) != t].sum(axis=0)
                    t_sum_others_bias = np.concatenate((
                        sum_others_bias[N_FEATURES * t: N_FEATURES * (t+1)],
                        sum_others_bias[N_FEATURES*N_CLASSES_PER_EXP:][t:(t+1)]
                    ))
                    theory_measures[class_order][1, epoch, seed, t] = ((class_avg_grad @ sum_others_bias)/
                                                                       class_avg_grad_norm)
                    theory_measures[class_order][5, epoch, seed, t] = ((class_t_avg_grad @ t_sum_others_bias) /
                                                                       class_t_avg_grad_norm)

                    sum_old_class_gradients = avg_full_gradients[:N_CLASSES_PER_EXP].sum(axis=0)
                    t_sum_old_class_gradients = np.concatenate((
                        sum_old_class_gradients[N_FEATURES * t: N_FEATURES * (t + 1)],
                        sum_old_class_gradients[N_FEATURES * N_CLASSES_PER_EXP:][t:(t + 1)]
                    ))
                    theory_measures[class_order][2, epoch, seed, t] = ((class_avg_grad @ sum_old_class_gradients)/
                                                                       class_avg_grad_norm)
                    theory_measures[class_order][6, epoch, seed, t] = ((class_t_avg_grad @ t_sum_old_class_gradients) /
                                                                       class_t_avg_grad_norm)

                    sum_old_class_gradients_except_self = (avg_full_gradients[:N_CLASSES_PER_EXP][
                        np.arange(N_CLASSES_PER_EXP) != t]).sum(axis=0)
                    t_sum_old_class_gradients_except_self = np.concatenate((
                        sum_old_class_gradients_except_self[N_FEATURES * t: N_FEATURES * (t + 1)],
                        sum_old_class_gradients_except_self[N_FEATURES * N_CLASSES_PER_EXP:][t:(t + 1)]
                    ))
                    theory_measures[class_order][3, epoch, seed, t] = ((class_avg_grad @ sum_old_class_gradients_except_self)/
                                                                       class_avg_grad_norm)
                    theory_measures[class_order][7, epoch, seed, t] = (
                                (class_t_avg_grad @ t_sum_old_class_gradients_except_self) /
                                class_t_avg_grad_norm)

                    sum_new_class_gradients = avg_full_gradients[N_CLASSES_PER_EXP:].sum(axis=0)
                    t_sum_new_class_gradients = np.concatenate((
                        sum_new_class_gradients[N_FEATURES * t: N_FEATURES * (t + 1)],
                        sum_new_class_gradients[N_FEATURES * N_CLASSES_PER_EXP:][t:(t + 1)]
                    ))
                    theory_measures[class_order][4, epoch, seed, t] = ((class_avg_grad @ sum_new_class_gradients)/
                                                                       class_avg_grad_norm)
                    theory_measures[class_order][8, epoch, seed, t] = ((class_t_avg_grad @ t_sum_new_class_gradients)/
                                                                       class_t_avg_grad_norm)

                    theory_measures[class_order][9, epoch, seed, t] = class_avg_grad_norm

                    theory_measures[class_order][10, epoch, seed, t] = class_t_avg_grad_norm


if __name__ == "__main__":
    for index_class_order in range(N_CLASS_ORDERS):
        for index_seed in range(N_SEEDS):
            for index_exp in range(N_EXPS):
                # load checkpoint
                ckp = torch.load(CHECKPOINT_PATH.format(index_class_order=index_class_order, n_exemplars=N_EXEMPLARS,
                                                        index_exp=index_exp), pickle_module=dill, map_location="cpu")
                # extract strategy
                strategy = ckp["strategy"]

                # get the storage policy of the replay plugin
                storage_policy = strategy.plugins[INDEX_REPLAY_PLUGIN].storage_policy
                # insert the exemplar indices
                insert_exemplar_indices(storage_policy, index_class_order, index_seed, index_exp)

                # get the dictionary containing all the metric results
                all_metric_results = strategy.plugins[INDEX_EVAL_PLUGIN].all_metric_results
                # insert the accuracies
                insert_accuracy(all_metric_results, index_class_order, index_seed, index_exp)
                # insert the confusion matrices
                insert_conf_matrix(all_metric_results, index_class_order, index_seed, index_exp)
                # insert the softmax probabilities
                insert_softmax_prob(all_metric_results, index_class_order, index_seed, index_exp)
                # insert the logits
                insert_logits(all_metric_results, index_class_order, index_seed, index_exp)
                # insert the avg grads
                insert_avg_grads(all_metric_results, index_class_order, index_seed, index_exp)

    # fill `forgetting` using the accuracies in `accuracy`
    fill_forgetting_from_accuracy()
    insert_theory_measures()
    # create dictionary encapsulating all the extracted data
    data = dict(accuracy=accuracy, forgetting=forgetting, exemplar_indices=exemplar_indices,
                theory_measures=theory_measures, conf_matrix=conf_matrix,
                softmax_prob=softmax_prob, logits=logits)
    # save dictionary encapsulating all the extracted data
    with open(SAVE_PATH, 'wb') as file:
        pickle.dump(data, file)