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

    def train(self, X_train, y_train):

        self.model = SVC(
            **self.param_grid,
            probability=True,
        )

        return self.model.fit(X_train, y_train)


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

        self.param_grid = {
            'C': [1, 50, 100, 500, 700, 900, 1000, 1200],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'degree': [3, 4, 5, 6],
            'gamma': ['scale', 'auto'],
            'random_state': [21],
            'coef0': [0.0, 0.5, 1.5],
        }

        message = f'Using default param grid: {self.param_grid}'

        self.write_to_logfile(message)

        rf_clf = GridSearchCV(
            estimator=SVC(random_state=self.random_state, probability=True),
            param_grid=self.param_grid,
            verbose=grid_search_verbosity,
            return_train_score=True,
            cv=cv_splits,
        )

        rf_clf.fit(X_train, y_train)

        best_parameters = rf_clf.best_params_

        self.param_grid = best_parameters

        return best_parameters, rf_clf


