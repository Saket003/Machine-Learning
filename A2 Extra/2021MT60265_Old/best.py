from typing import Union
from svm_binary import Trainer as T
from svm_multiclass import Trainer_OVO as T_ovo, Trainer_OVA as T_ova
from kernel import rbf,linear

def best_classifier_two_class()->T:
    """Return the best classifier for the two-class classification problem."""
    kernel = linear
    c = 1
    trainer = T(kernel,c)
    return trainer

def best_classifier_multi_class()->Union[T_ovo,T_ova]:
    """Return the best classifier for the multi-class classification problem."""
    kernel = rbf
    c = 1
    n_classes = 10
    trainer = T_ova(kernel,c,n_classes,gamma = 0.1)
    return trainer
