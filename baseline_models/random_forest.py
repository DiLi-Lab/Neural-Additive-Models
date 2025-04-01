from __future__ import annotations

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from baseline_models.baseline_model import Model


class RandomForest(Model):

    name = 'rf'

    def __init__(
            self,
            root: str | Path,
            split_criterion: str,
            param_grid: dict = None,
            **kwargs,
    ):
        super().__init__(root, split_criterion, param_grid, **kwargs)

        if param_grid is None:
            self.param_grid = {
                'n_estimators': [50, 100, 500, 700, 1000, 1200],
                'max_features': ['sqrt', 'log2', None],
                'max_depth': [2, 4, 8, 16, 32, 64, None],
                'criterion': ['entropy', 'gini', 'log_loss'],
                'random_state': [21],
                'n_jobs': [-1]
            }

    def get_model(self, params):
        return RandomForestClassifier(**params)


