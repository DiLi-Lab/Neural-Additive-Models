from __future__ import annotations

from pathlib import Path

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

from baseline_models.baseline_model import Model


class GradientBoostingCls(Model):

    name = 'gradient-boosting'

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
                'max_depth': [3, 8, 16, 32, 64, None],
                'criterion': ['friedman_mse', 'squared_error'],
                'loss': ['log_loss', 'exponential'],
                'random_state': [21],
            }

    def get_model(self, params):
        return GradientBoostingClassifier(**params)


