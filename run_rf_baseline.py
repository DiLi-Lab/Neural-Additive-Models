import argparse
from datetime import datetime
import json
import random

import numpy as np
from sklearn import metrics
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler

from extract_features_rf import get_combined_features
from utils import results_utils, data_utils


def run_rf_baseline(
        data_folder: str,
        split_criterion_str: str,
        label_col: str,
        random_state: int,
        num_cv_folds: int = 5,
        grid_search_verbosity: int = 1,
        param_grid: dict = None,
        hp_tuning: bool = False,
):
    random.seed(random_state)
    np.random.seed(random_state)

    data = data_utils.load_data(data_folder)
    results_path = results_utils.create_results_folders(
        script=f'rf_{"hp_tuning" if hp_tuning else ""}_new_{split_criterion_str}')

    X, feature_names = get_combined_features(data)
    y = np.array([df[label_col].iloc[0] for df in data])
    split_criterion = [df[split_criterion_str].iloc[0] for df in data]

    logger = results_utils.get_logger(results_path)

    metric_dict = {
        'rf': {'auc': [],
               'fprs': [],
               'tprs': [],
               'proba': [],
               'label': []},
    }

    aucs = []
    fprs = []
    tprs = []
    rf_accs = []
    baseline_accs = []
    outer_fold = 1
    best_score = 0
    best_hparams = None

    logger.info(
        f'Grid search for random forest baseline. New {split_criterion_str} split. Started at {datetime.now()}')
    logger.info(f'Grid search parameters: {param_grid}')

    kf = StratifiedGroupKFold(n_splits=5)

    for train_index, test_index in kf.split(X, y, groups=split_criterion):
        logger.info(f'Outer fold {outer_fold}')
        logger.debug(f'Outer fold {outer_fold}')

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        outer_train_ids = [split_criterion[i] for i in train_index]
        outer_test_ids = [split_criterion[i] for i in test_index]

        logger.info(f'Outer train {split_criterion_str}-ids: {sorted(set(outer_train_ids))}')
        logger.info(f'Outer test {split_criterion_str}-ids: {sorted(set(outer_test_ids))}')

        logger.debug(f'Outer train data indices: {train_index}')
        logger.debug(f'Outer test data indices: {test_index}')
        logger.debug(f'Outer train data labels: {y_train}')
        logger.debug(f'Outer test data labels: {y_test}')
        logger.debug(f'Inner train {split_criterion_str}-ids: {outer_train_ids}')
        logger.debug(f'Inner test {split_criterion_str}-ids: {outer_train_ids}')

        inner_kf = StratifiedGroupKFold(n_splits=num_cv_folds)

        # standardize/normalize the data
        scaler = StandardScaler()
        # fit data on training set
        X_train_std = scaler.fit_transform(X_train)
        # standardize test data as well
        X_test_std = scaler.transform(X_test)

        cv_list = list(inner_kf.split(X_train_std, y_train, groups=outer_train_ids))

        for inner_fold_idx, (train, test) in enumerate(cv_list):
            inner_train_ids = [outer_train_ids[i] for i in train]
            inner_test_ids = [outer_train_ids[i] for i in test]

            logger.info(f'Inner fold {inner_fold_idx + 1}')
            logger.debug(f'Inner fold {inner_fold_idx + 1}')

            logger.info(f'Inner train {split_criterion_str}-ids: {sorted(set(inner_train_ids))}')
            logger.info(f'Inner test {split_criterion_str}-ids: {sorted(set(inner_test_ids))}')

            logger.debug(f'Inner train data indices on outer train data: {train}')
            logger.debug(f'Inner test data indices on outer train data: {test}')
            logger.debug(f'Inner train data labels on outer train data: {y_train[train]}')
            logger.debug(f'Inner test data labels on outer train data: {y_train[test]}')
            logger.debug(f'Inner train {split_criterion_str}-ids: {inner_train_ids}')
            logger.debug(f'Inner test {split_criterion_str}-ids: {inner_test_ids}')

        best_params, rf = evaluate_rf_baseline(
            X_train_std,
            y_train,
            cv_splits=cv_list,
            grid_search_verbosity=grid_search_verbosity,
            param_grid=param_grid,
        )

        pred_proba = rf.predict_proba(X_test_std)

        fpr, tpr, _ = metrics.roc_curve(np.array(y_test, dtype=int), pred_proba[:, 1], pos_label=1)
        auc = metrics.auc(fpr, tpr)
        metric_dict['rf']['auc'].append(auc)
        metric_dict['rf']['fprs'].append(fpr)
        metric_dict['rf']['tprs'].append(tpr)
        metric_dict['rf']['proba'].append(pred_proba)
        metric_dict['rf']['label'].append(y_test)

        # get best score from estimator on that fold and compare to old best score
        current_best_score = rf.best_score_
        logger.info(f'Best score on inner fold: {current_best_score}')
        logger.debug(f'All CV results on fold {outer_fold}: {rf.cv_results_}')

        if current_best_score > best_score:
            logger.info(f'New best score!')

            best_score = current_best_score
            best_hparams = rf.best_params_
            logger.info(f'New best parameters on fold {outer_fold}: {rf.best_params_}')
        else:
            logger.info(f'Best parameters still the same on fold {outer_fold}: {best_hparams}')

        aucs.append(auc)
        fprs.append(fpr)
        tprs.append(tpr)

        # dummy result baseline
        d_metrics, d_acc = dummy_baseline(X_train_std, X_test_std, y_train, y_test)
        metric_dict['dummy'] = d_metrics

        baseline_accs.append(d_acc)
        rf_accs.append(rf.score(X_test, np.array(y_test, dtype=int)))
        logger.info('Baseline: ' + str(baseline_accs[-1]) + ' vs. ' + str(rf_accs[-1]))
        logger.info(f'Test AUC-ROC on fold {outer_fold} with best hparams: {auc}')
        outer_fold += 1

    final_mean_score = round(np.array(metric_dict['rf']['auc']).mean(), 4)
    print(f'Final mean score: {final_mean_score}')
    logger.info(f'Final AUC-ROC score (mean over 5 folds): {final_mean_score}')

    with open(f'{results_path}/metric_dict.txt', 'w', encoding='utf8') as f:
        f.write(str(metric_dict))


def dummy_baseline(X_train, X_test, y_train, y_test):
    # baseline
    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(X_train, np.array(y_train, dtype=int))
    dummy_proba = dummy_clf.predict_proba(X_test)
    fpr, tpr, _ = metrics.roc_curve(np.array(y_test, dtype=int), dummy_proba[:, 1], pos_label=1)
    auc = metrics.auc(fpr, tpr)

    dummy_dict = {'auc': [], 'fprs': [], 'tprs': [], 'proba': [], 'label': []}

    dummy_dict['auc'].append(auc)
    dummy_dict['fprs'].append(fpr)
    dummy_dict['tprs'].append(tpr)
    dummy_dict['proba'].append(dummy_proba)
    dummy_dict['label'].append(y_test)

    dummy_acc = dummy_clf.score(X_test, np.array(y_test, dtype=int))

    return dummy_dict, dummy_acc


def evaluate_rf_baseline(
        X_train,
        y_train,
        cv_splits: list,
        grid_search_verbosity: int = 1,
        param_grid: dict = None
):
    """
    :input:     X_train: training data (matrix)
                y_train: array with labels for train data
                param_grid: grid for grid search


    :return:    y_pred: predicted values for predicting on X_test
                best_parameters: dictionary containing the best hyper parameter combination (if param_grid is empty,
                best_parameters is an empty dict)
    """

    if param_grid is None:
        param_grid = {
            'n_estimators': [50, 100, 300, 400, 500, 700, 900, 1000, 1200],
            'max_features': ['sqrt', 'log2', None],
            'max_depth': [2, 4, 8, 16, 32, 64, None],
            'criterion': ['entropy', 'gini', 'log_loss'],
            'random_state': [21],
            'n_jobs': [-1]
        }

    rf_clf = GridSearchCV(
        estimator=RandomForestClassifier(),
        param_grid=param_grid,
        verbose=grid_search_verbosity,
        return_train_score=True,
        cv=cv_splits,
    )

    rf_clf.fit(X_train, y_train)

    best_parameters = rf_clf.best_params_

    return best_parameters, rf_clf


def parse_args() -> dict:
    parser = argparse.ArgumentParser(description='Run RF baseline')
    parser.add_argument(
        '--config',
        type=str,
        help='Path to config file',
        default='config_baseline_hp_tuning.json'
    )
    args = parser.parse_args()
    return vars(args)


if __name__ == '__main__':
    arguments = parse_args()
    config = json.load(open(arguments['config'], 'r'))

    # param_grid = {'criterion': ['gini'], 'max_depth': [32], 'max_features': [None], 'n_estimators': [700],
    #               'n_jobs': [-1], 'random_state': [21]}

    run_rf_baseline(**config, param_grid=None)
