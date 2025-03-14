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
    params_rf = {'criterion': 'entropy', 'max_depth': 2, 'max_features': 'sqrt', 'n_estimators': 500, 'n_jobs': -1,
                 'random_state': 21}
    params_gb = {'criterion': 'squared_error', 'loss': 'log_loss', 'max_depth': 32, 'max_features': 'sqrt', 'n_estimators': 50, 'random_state': 21}
    params_svc = {'C': 1, 'coef0': 0.0, 'degree': 3, 'gamma': 'auto', 'kernel': 'poly', 'random_state': 21}

    rf_model = RandomForest(root=result_folder, split_criterion=split_criterion_str, param_grid=params_rf)
    dummy_bsl = DummyBaseline(strategy='most_frequent', root=result_folder, split_criterion=split_criterion_str)
    gbcl = GradientBoostingCls(root=result_folder, split_criterion=split_criterion_str, param_grid=params_gb)
    svc = SVCls(root=result_folder, split_criterion=split_criterion_str, param_grid=params_svc)

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

    X, y, feature_names, split_criterion = data_utils.load_dataset('PoTeC', split_criterion_str,
                                                                   potec_folder, str(result_folder), label)

    print(f'# expert reading (experts reading text in their expert domain): {np.count_nonzero(y == 1)}')
    print(f'# non-expert reading (beginners reading text OR experts reading text in non-expert domain): '
          f'{np.count_nonzero(y == 0)}')

    outer_fold = 1
    best_score = 0

    outer_kf = StratifiedGroupKFold(n_splits=num_cv_folds_outer)
    inner_kf = StratifiedGroupKFold(n_splits=num_cv_folds_inner)
    hp_tuning_kf = StratifiedGroupKFold(n_splits=10)

    # exclude the hp test set from
    hp_splits = list(hp_tuning_kf.split(X, y, groups=split_criterion))
    hp_split = hp_splits[0]
    hp_test_index = hp_split[1]

    # exclude the hp test set from the outer folds
    X = np.delete(X, hp_test_index, axis=0)
    y = np.delete(y, hp_test_index, axis=0)

    print(len(hp_test_index))
    print(len(X))
    print(len(y))

    for train_index, test_index in outer_kf.split(X, y, groups=split_criterion):
        print(f'\n{"*" * 50}')
        print(f'***************** Outer fold {outer_fold} *******************')
        print(f'{"*" * 50}')

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

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

        cv_list = list(inner_kf.split(X_train_std, y_train, groups=outer_train_ids))

        # log inner fold properties
        for inner_fold_idx, (inner_train_indices, inner_test_indices) in enumerate(cv_list):
            inner_train_ids = [outer_train_ids[i] for i in inner_train_indices]
            inner_test_ids = [outer_train_ids[i] for i in inner_test_indices]
            y_train_inner, y_test_inner = y_train[inner_train_ids], y_train[inner_test_ids]

            data_split_dict[f'outer_fold_{outer_fold}'][f'inner_fold_{inner_fold_idx}'] = {
                f'inner_train_{split_criterion_str}_ids': inner_train_ids,
                f'inner_test_{split_criterion_str}_ids': inner_test_ids,
                'inner_train_index': inner_train_indices,
                'inner_test_index': inner_test_indices,
                'y_train_inner': y_train_inner,
                'y_test_inner': y_test_inner,
            }

        for model in baseline_models:
            print(f'\n--------- Running {model.name} on fold {outer_fold} ---------')

            # if hp tuning, test hps on first fold

            if hp_tuning and outer_fold == 1:
                # delete current test set indices from hp tuning set


                best_params, best_model = model.train_hp_tuning(
                    X_train_std,
                    y_train,
                    cv_splits=cv_list,
                    grid_search_verbosity=grid_search_verbosity,
                )

                current_best_score = best_model.best_score_

                if model.best_score < current_best_score:
                    model.best_score = current_best_score
                    model.best_params = best_params
                    print(f' -- new best params for {model.name}: {model.best_params}')

                metric_dict[model.name]['cv_results'].append(best_model.cv_results_)


            else:
                best_params = model.param_grid
                best_model = model.train(X_train_std, y_train)

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
