from typing import Dict, List, Tuple
import pickle
import numpy as np

EXP_ID = 28
"""
Experiments ID
"""

DATASET = "cifar100"
"""
Name of the dataset
"""

NETWORK = "resnet32"
"""
Name of the network architecture
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
Number of classes per experience
"""

N_REHEARSAL = 4000
"""
Total number of samples in the rehearsal buffer
"""

N_CLASS_ORDERS = 20
"""
Number of class orders in the experiments
"""

N_SEEDS = 1
"""
Number of seeds
"""

N_EPOCHS = 170
"""
Number of epochs
"""

MB_SIZE_SECOND_EXP = 128
"""
Mini-batch size during the second experience
"""

N_TRAINING_EXAMPLES_PER_CLASS = 450
"""
Number of training examples per class
"""

N_THEORY_MEASURES = 11
"""
Number of measures derived from the theoretical analysis
"""

DATA_PATH = f"experiment{EXP_ID}_{DATASET}_{NETWORK}_{N_EXPS_TOTAL}_increments_{N_CLASSES_PER_EXP}_classes_{N_REHEARSAL}_diff_classes_data.pkl"
"""
Path to the extracted data
"""


with open(DATA_PATH, "rb") as file:
    data = pickle.load(file)

accuracy: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = data["accuracy"]
"""
a dictionary containing the class order indexes as keys and respective lists as values. Each list contains as many
elements as the number of experiences. Each element is a 2-tuple where both elements are 2D numpy arrays of size NxC,
where N is equal to the number of seeds and C is equal to the number of unique classes encountered up to the current
experience. The ith, jth element in the first numpy array is the training accuracy of class j at the end of the current
experience for the ith seed. The ith, jth element in the second numpy array is the test accuracy of class j at the end
of the current experience for the ith seed. 
"""

forgetting: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = data["forgetting"]
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

exemplar_indices: Dict[int, List[np.ndarray]] = data["exemplar_indices"]
"""
a dictionary containing the class order indexes as keys and respective lists as values. Each list contains as many
elements as the number of experiences. Each element is a 2D numpy array of size NxC, where N is equal to the number of
seeds and C is equal to the number of exemplars in the rehearsal buffer after each experience. The ith row in each numpy
array contains the indices of all the samples selected for rehearsal at the end of the current experience for the ith
seed.
"""

conf_matrix: Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = data["conf_matrix"]
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

softmax_prob: Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray]]] = data["softmax_prob"]
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

logits: Dict[int, Dict[int, np.ndarray]] = data["logits"]
"""
a dictionary containing the class order indexes as keys and respective dictionaries as values. The latter dictionaries
contain the experience indexes as keys and respective numpy arrays as values. The first experience, the experience with
index 0, is skipped. Each array is a 2D numpy array of size NxC, where N is equal to the number of seeds and C is equal
to the number of unique classes encountered up to the current experience.
The ith, jth element in each array is the average softmax probability for class j computed across all training
samples in the current experience (only the new classes) for the ith seed prior to perform training on the current
experience.
"""

theory_measures: Dict[int, np.ndarray] = data["theory_measures"]
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

