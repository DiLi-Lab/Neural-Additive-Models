from __future__ import annotations

from datetime import datetime
from pathlib import Path

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

from model import Model


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

    def train(self, X_train, y_train):
        pass

    def predict(self, X_test):
        pass

    def train_hp_tuning(
            self,
            X_train,
            y_train,
            cv_splits,
            grid_search_verbosity=1,
    ):

        message = f"Using custom param grid: {self.param_grid}"

        if self.param_grid is None:
            self.param_grid = {
                'n_estimators': [50, 100, 500, 700, 1000, 1200],
                'max_features': ['sqrt', 'log2', None],
                'max_depth': [3, 8, 16, 32, 64, None],
                'criterion': ['friedman_mse', 'squared_error'],
                'loss': ['log_loss', 'exponential'],
                'random_state': [21],
            }

            message = f'Using default param grid: {self.param_grid}'

        self.write_to_logfile(message)

        rf_clf = GridSearchCV(
            estimator=GradientBoostingClassifier(),
            param_grid=self.param_grid,
            verbose=grid_search_verbosity,
            return_train_score=True,
            cv=cv_splits,
        )

        rf_clf.fit(X_train, y_train)

        best_parameters = rf_clf.best_params_

        return best_parameters, rf_clf


