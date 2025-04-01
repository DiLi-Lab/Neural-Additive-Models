from __future__ import annotations

from pathlib import Path

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from baseline_models.baseline_model import Model


class SVCls(Model):

    name = 'svc'

    def __init__(
            self,
            root: str | Path,
            split_criterion: str,
            param_grid: dict = None,
            **kwargs,
    ):
        super().__init__(root, split_criterion, param_grid, **kwargs)
        self.model = None

        if param_grid is None:
            self.param_grid = {
                    'C': [1, 50, 100, 500, 700, 900, 1000, 1200],
                    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                    'degree': [3, 4, 5, 6],
                    'gamma': ['scale', 'auto'],
                    'random_state': [21],
                    'coef0': [0.0, 0.5, 1.5],
                }

    def get_model(self, params):
        return SVC(**params, probability=True)

