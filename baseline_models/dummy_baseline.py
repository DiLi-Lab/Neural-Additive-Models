from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV

from model import Model


class DummyBaseline(Model):
    name = 'dummy_baseline'

    def __init__(self, strategy, root: str | Path, split_criterion: str, **kwargs):
        super().__init__(root, split_criterion, **kwargs)
        self.strategy = strategy

    def train(self, X_train, y_train):
        dummy_clf = DummyClassifier(strategy=self.strategy)
        dummy_clf.fit(X_train, np.array(y_train, dtype=int))

        params = dummy_clf.get_params()

        return params, dummy_clf

    def predict(self, X_test):
        raise NotImplementedError

    def train_hp_tuning(
            self,
            X_train,
            y_train,
            cv_splits,
            grid_search_verbosity=1,
    ):
        param_grid = {'strategy': [self.strategy]}

        dummy_clf = GridSearchCV(
            estimator=DummyClassifier(),
            param_grid=param_grid,
            verbose=grid_search_verbosity,
            return_train_score=True,
            cv=cv_splits,
        )

        dummy_clf.fit(X_train, y_train)

        best_parameters = dummy_clf.best_params_

        return best_parameters, dummy_clf


