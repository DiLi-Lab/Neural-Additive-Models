from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV

from baseline_models.baseline_model import Model


class DummyBaseline(Model):
    name = 'dummy_baseline'

    def __init__(self, strategy, root: str | Path, split_criterion: str, **kwargs):
        super().__init__(root, split_criterion, **kwargs)
        self.strategy = strategy

    def train(self, x_train, y_train, params):
        dummy_clf = DummyClassifier(strategy=self.strategy)
        dummy_clf.fit(x_train, np.array(y_train, dtype=int))

        self.param_grid = dummy_clf.get_params()

        return dummy_clf

    def train_hp_tuning(
            self,
            x_train,
            y_train,
            x_val,
            y_val,
    ):
        return {'strategy': self.strategy}
