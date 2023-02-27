import numpy as np
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
import torch.optim as optim

from model import Model

RANDOM_STATE = 21


def nested_cv(data, labels, num_folds_outer, num_folds_inner, hyperparams):
    """ Performs nested cross-validation with hyperparameter tuning using grid search. """

    outer_scores = []
    kf_outer = StratifiedKFold(n_splits=num_folds_outer, shuffle=True, random_state=RANDOM_STATE)

    for train_index_outer, test_index_outer in kf_outer.split(data, labels):
        X_train_outer, data_test_outer = data[train_index_outer], data[test_index_outer]
        labels_train_outer, labels_test_outer = labels[train_index_outer], labels[test_index_outer]

        best_score_inner = -np.inf
        best_hyperparams_inner = None
        kf_inner = StratifiedKFold(n_splits=num_folds_inner, shuffle=True, random_state=RANDOM_STATE)

        for hyperparams_inner in hyperparams:
            score_inner = 0
            for train_index_inner, val_index_inner in kf_inner.split(X_train_outer, labels_train_outer):
                data_train_inner, data_val_inner = X_train_outer[train_index_inner], X_train_outer[
                    val_index_inner]
                labels_train_inner, labels_val_inner = labels_train_outer[train_index_inner], labels_train_outer[
                    val_index_inner]

                model = Model(**hyperparams_inner)

                model.train()

                model.eval()
                output = model(data_val_inner)
                pred = output.argmax(dim=1)
                score_inner += (pred == labels_val_inner).sum().item()

            score_inner /= len(X_train_outer)
            if score_inner > best_score_inner:
                best_score_inner = score_inner
                best_hyperparams_inner = hyperparams_inner

        model = Model(**best_hyperparams_inner)

        model.train()

        model.eval()
        output = model(data_test_outer)
        pred = output.argmax(dim=1)
        score_outer = (pred == labels_test_outer).sum().item() / len(data_test_outer)
        outer_scores.append(score_outer)

    return np.mean(outer_scores)
