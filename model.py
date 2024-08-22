from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path


class Model:
    name = None
    best_score = 0
    best_params = None

    def __init__(
            self,
            root: str | Path,
            split_criterion: str,
            param_grid: dict = None,
            random_state: int = 21,
            **kwargs
    ) -> None:
        """
        :param root: root path of the project
        :param split_criterion: the name of the split criterion for splitting the data into train and test
        :param name: name of the model
        :param param_grid: hyperparameter grid for hyperparameter tuning

        """

        self.random_state = random_state
        self.param_grid = param_grid
        self.hp_tuning = True if param_grid is not None else False
        self.split_criterion = split_criterion

        self.result_folder = self._create_result_folder(
            root=root, split_criterion=self.split_criterion, hp_tuning=self.hp_tuning
        )
        self.logfile_path = self.result_folder / f'{self.name}_log.txt'

        with open(self.logfile_path, 'w', encoding='utf8') as lf:
            lf.write(f'{self.name} log params\n')

    def train(self, X_train, y_train):
        raise NotImplementedError

    def predict(self, X_test):
        raise NotImplementedError

    def train_hp_tuning(
            self,
            X_train,
            y_train,
            cv_splits,
            grid_search_verbosity=1,
    ):
        raise NotImplementedError

    def write_to_logfile(self, message: str) -> None:
        with open(self.logfile_path, 'a', encoding='utf8') as lf:
            lf.write(f'{message}\n')

    def _create_result_folder(self, root: str | Path, split_criterion: str,
                              hp_tuning=False
                              ) -> Path:
        today = datetime.now().strftime('%Y-%m-%d-%H:%M')

        full_path = (Path(root) /
                     f'{self.name}{"_hp_tuning" if hp_tuning else ""}_split-{split_criterion}_{today}')
        self.result_folder = full_path

        # create results folder if it does not exist
        if not os.path.isdir(full_path):
            os.makedirs(full_path)

        return full_path

    def save_best_params(self):
        with open(self.result_folder / 'best_params.txt', 'w', encoding='utf8') as f:
            f.write(str(self.best_params))

