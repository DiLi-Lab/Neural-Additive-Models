from __future__ import annotations

import argparse
from datetime import datetime
import json
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
from sklearn import metrics
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from baseline_models.gradient_boosting_cls import GradientBoostingCls
from baseline_models.svc import SVCls
from data.extract_features import get_combined_features
from baseline_models.random_forest import RandomForest
from baseline_models.dummy_baseline import DummyBaseline
from data.potec import Potec
from neural_additive_models import data_utils


def evaluate_potec_expert_clf(
        root_path: str | Path,
        potec_folder: str,
        split_criterion_str: str,
        label: str,
        random_state: int,
        num_cv_folds_outer: int = 5,
        num_cv_folds_inner: int = 3,
        grid_search_verbosity: int = 1,
        hp_tuning: bool = False,
):
    random.seed(random_state)
    np.random.seed(random_state)
    today = datetime.now().strftime('%Y-%m-%d-%H:%M')
    result_folder = root_path / 'results_baselines' / f'{today}_label-{label}_split-{split_criterion_str}{"_hp-tuning" if hp_tuning else ""}'

    if not result_folder.exists():
        result_folder.mkdir(parents=True)

    # just use those if you don't want to run hp tuning
    params_rf = {'n_estimators': 100, 'max_features': 'sqrt', 'max_depth': 4, 'criterion': 'entropy',
                 'random_state': 21, 'n_jobs': -1}
    params_gb = {'n_estimators': 500, 'max_features': 'log2', 'max_depth': 16, 'criterion': 'friedman_mse',
                 'loss': 'log_loss', 'random_state': 21}
    params_svc = {'C': 1, 'kernel': 'sigmoid', 'degree': 3, 'gamma': 'scale', 'random_state': 21, 'coef0': 0.0}

    rf_model = RandomForest(root=result_folder, split_criterion=split_criterion_str,
                            param_grid=params_rf if not hp_tuning else None)
    dummy_bsl = DummyBaseline(strategy='most_frequent', root=result_folder, split_criterion=split_criterion_str)
    gbcl = GradientBoostingCls(root=result_folder, split_criterion=split_criterion_str,
                               param_grid=params_gb if not hp_tuning else None)
    svc = SVCls(root=result_folder, split_criterion=split_criterion_str,
                param_grid=params_svc if not hp_tuning else None)

    baseline_models = [
        rf_model,
        svc,
        gbcl,
        dummy_bsl,
    ]

    metric_dict = {}
    for model in baseline_models:
        metric_dict[model.name] = defaultdict(list)

    data_split_dict = {}

    print(potec_folder)

    data_x, data_y, column_names, split_criterion = data_utils.load_dataset('PoTeC', split_criterion_str,
                                                                            potec_folder, str(result_folder), label)

    # 10% of the data are excluded and used for hp tuning
    (data_x, data_y), (hp_tuning_x, hp_tuning_y), (split_criterion, split_hp_tuning) = data_utils.get_train_test_fold(
        data_x, data_y,
        fold_num=1,
        num_folds=10,
        stratified=True,
        group_split=split_criterion
    )

    print(f'# class 1: {np.count_nonzero(data_y == 1)}')
    print(f'# class 0: '
          f'{np.count_nonzero(data_y == 0)}')

    outer_fold = 1

    outer_kf = StratifiedGroupKFold(n_splits=num_cv_folds_outer)

    for train_index, test_index in outer_kf.split(data_x, data_y, groups=split_criterion):
        print(f'\n{"*" * 50}')
        print(f'***************** Outer fold {outer_fold} *******************')
        print(f'{"*" * 50}')

        X_train, X_test = data_x[train_index], data_x[test_index]
        y_train, y_test = data_y[train_index], data_y[test_index]

        outer_train_ids = [split_criterion[i] for i in train_index]
        outer_test_ids = [split_criterion[i] for i in test_index]

        data_split_dict[f'outer_fold_{outer_fold}'] = {
            'outer_train_ids': outer_train_ids,
            'outer_test_ids': outer_test_ids,
            'train_index': train_index,
            'test_index': test_index,
            'y_train': y_train,
            'y_test': y_test,
        }

        # standardize/normalize the data
        scaler = StandardScaler()
        # fit data on training set
        X_train_std = scaler.fit_transform(X_train)
        # standardize test data as well
        X_test_std = scaler.transform(X_test)
        x_hp_tuning_std = scaler.transform(hp_tuning_x)

        for model in baseline_models:
            print(f'\n--------- Running {model.name} on fold {outer_fold} ---------')

            # if hp tuning, test hps on first fold on separate validation fold
            if hp_tuning and outer_fold == 1:
                best_params = model.train_hp_tuning(X_train_std, y_train, x_hp_tuning_std, hp_tuning_y)
                model.param_grid = best_params

                # train model again using best params
                best_model = model.train(X_train_std, y_train, best_params)

            else:
                best_params = model.param_grid
                best_model = model.train(X_train_std, y_train, best_params)

            pred_proba = best_model.predict_proba(X_test_std)
            fpr, tpr, _ = metrics.roc_curve(np.array(y_test, dtype=int), pred_proba[:, 1], pos_label=1)
            auc = metrics.auc(fpr, tpr)
            acc = best_model.score(X_test_std, np.array(y_test, dtype=int))
            print(f' -- test AUC score on fold {outer_fold}: {auc}')
            print(f' -- test accuracy score on fold {outer_fold}: {acc}')

            model.write_to_logfile(f'Parameters on fold {outer_fold}: {best_params}\n'
                                   f'Test AUC score with best params: {auc}\n'
                                   f'Test accuracy score with best params: {acc}\n\n')

            metric_dict[model.name]['auc'].append(auc)
            metric_dict[model.name]['accuracy'].append(acc)
            metric_dict[model.name]['fprs'].append(fpr)
            metric_dict[model.name]['tprs'].append(tpr)
            metric_dict[model.name]['proba'].append(pred_proba)
            metric_dict[model.name]['label'].append(y_test)
            metric_dict[model.name]['best_params'].append(best_params)

        outer_fold += 1

    for model in baseline_models:
        model.save_best_params()

        print(f'\n{"*" * 50}')
        print(f'*************** {model.name} RESULTS *****************')
        final_mean_score = round(np.array(metric_dict[model.name]['accuracy']).mean(), 4)
        print(f' -- accuracy mean score over all folds : {final_mean_score}')
        aucs = np.array(metric_dict[model.name]['auc'])
        final_mean_score = round(aucs.mean(), 3)
        se = round(np.std(aucs, ddof=1) / np.sqrt(np.size(aucs)), 3)
        print(f' -- auc mean score over all folds : {final_mean_score} (SE: {se})')

    with open(result_folder / 'metric_dict.txt', 'w', encoding='utf8') as f:
        f.write(str(metric_dict))

    with open(result_folder / 'data_split_dict.txt', 'w', encoding='utf8') as f:
        f.write(str(data_split_dict))


def parse_args() -> dict:
    parser = argparse.ArgumentParser(description='Run baseline models')
    parser.add_argument(
        '--config',
        type=str,
        help='Path to config file',
        default='evaluation_configs/config_baseline_hp_tuning_2_labels_new-reader-split_label_expert_cls.json',
    )

    parser.add_argument(
        '--data-folder',
        type=str,
        help='Path to the potec folder',
    )

    parser.add_argument(
        '--hp-tuning',
        action='store_true',
        help='Whether to run hyperparameter tuning',
    )
    args = parser.parse_args()
    return vars(args)


if __name__ == '__main__':
    arguments = parse_args()
    root = Path(__file__).parent
    config_path = root / arguments['config']
    config = json.load(open(config_path, 'r'))

    if arguments['data_folder'] is not None:
        config['potec_folder'] = arguments['data_folder']

    evaluate_potec_expert_clf(root_path=root, hp_tuning=arguments['hp_tuning'], **config)
