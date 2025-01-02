#!/usr/bin/env python3

# coding=utf-8
# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Training script for Neural Additive Models.

"""
import datetime
import operator
import os
from collections import defaultdict, OrderedDict
from itertools import product
from typing import Tuple, List, Dict, Any
from absl import app
from absl import flags
import numpy as np
import yaml
import tensorflow.compat.v1 as tf
from sklearn import metrics
from tqdm import tqdm

from neural_additive_models import data_utils, graph_builder
from neural_additive_models.graph_builder import calculate_metric, sigmoid

gfile = tf.io.gfile
DatasetType = data_utils.DatasetType

FLAGS = flags.FLAGS

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

flags.DEFINE_integer('training_epochs', 100,
                     'The number of epochs to run training for.')
flags.DEFINE_float('learning_rate', 1e-2, 'Hyperparameter: learning rate.')
flags.DEFINE_float('output_regularization', 0.0, 'Hyperparameter: feature reg')
flags.DEFINE_float('l2_regularization', 0.0, 'Hyperparameter: l2 weight decay')
flags.DEFINE_integer('batch_size', 28, 'Hyperparameter: batch size.')
flags.DEFINE_string('logdir', f'{THIS_DIR}/results_nam/logs_{datetime.datetime.now().strftime("%Y-%m-%d-%H:%M")}',
                    'Path to dir where to store summaries.')
flags.DEFINE_string('dataset_name', 'PoTeC',
                    'Name of the dataset to load for training.')
flags.DEFINE_string('dataset_folder', '/Users/debor/repos/PoTeC-data',
                    'Folder where dataset is. How to load it must be defined in data_utils.py')
flags.DEFINE_float('decay_rate', 0.995, 'Hyperparameter: Optimizer decay rate')
flags.DEFINE_float('dropout', 0.5, 'Hyperparameter: Dropout rate')
flags.DEFINE_integer(
    'data_split', 2, 'Dataset split index to use for splitting the training set into val and train. '
                     'Possible values are 1 to `FLAGS.num_splits`.')
flags.DEFINE_integer('tf_seed', 1, 'seed for tf.')
flags.DEFINE_float('feature_dropout', 0.0,
                   'Hyperparameter: Prob. with which features are dropped')
flags.DEFINE_integer(
    'num_basis_functions', 1000, 'Number of basis functions '
                                 'to use in a FeatureNN for a real-valued feature.')
flags.DEFINE_integer('units_multiplier', 2, 'Number of basis functions for a '
                                            'categorical feature')
flags.DEFINE_boolean(
    'cross_val', True, 'Boolean flag indicating whether to '
                       'perform cross validation or not.')
flags.DEFINE_integer(
    'max_checkpoints_to_keep', 1, 'Indicates the maximum '
                                  'number of recent checkpoint files to keep.')
flags.DEFINE_integer(
    'save_checkpoint_every_n_epochs', 10, 'Indicates the '
                                          'number of epochs after which an checkpoint is saved')
flags.DEFINE_integer('n_models', 1, 'the number of models to train.')
flags.DEFINE_integer('num_splits', 5, 'Number of data splits to use')
flags.DEFINE_integer('fold_num', 1, 'Index of the fold to be used')
flags.DEFINE_string(
    'activation', 'exu', 'Activation function to used in the '
                         'hidden layer. Possible options: (1) relu, (2) exu')
flags.DEFINE_boolean(
    'regression', False, 'Boolean flag indicating whether we '
                         'are solving a regression task or a classification task.')
flags.DEFINE_boolean('debug', False, 'Debug mode. Log additional things')
flags.DEFINE_boolean('shallow', False, 'Whether to use shallow or deep NN.')
flags.DEFINE_boolean('use_dnn', False, 'Deep NN baseline.')
flags.DEFINE_integer('early_stopping_epochs', 60, 'Early stopping epochs')
flags.DEFINE_string('group_by', 'reader_id', 'Specifies the group label to split by for GroupKFold')
flags.DEFINE_string('config_path', None, 'Define where best config is stored.')
flags.DEFINE_boolean('all_folds', True, 'Specifies whether to run all folds or just one fold')
flags.DEFINE_boolean('hp_tuning', False, 'Specifies whether to run hyperparameter tuning or not')
flags.DEFINE_integer('gpu', 7, 'Specifies which gpu to use')
_N_FOLDS = 5
GraphOpsAndTensors = graph_builder.GraphOpsAndTensors
EvaluationMetric = graph_builder.EvaluationMetric


@flags.multi_flags_validator(['data_split', 'cross_val'],
                             message='Data split should not be used in '
                                     'conjunction with cross validation')
def data_split_with_cross_validation(flags_dict):
    return (flags_dict['data_split'] == 1) or (not flags_dict['cross_val'])


def _get_train_and_lr_decay_ops(
        graph_tensors_and_ops,
        early_stopping):
    """Returns training and learning rate decay ops."""
    train_ops = [
        g['train_op']
        for n, g in enumerate(graph_tensors_and_ops)
        if not early_stopping[n]
    ]
    lr_decay_ops = [
        g['lr_decay_op']
        for n, g in enumerate(graph_tensors_and_ops)
        if not early_stopping[n]
    ]
    return train_ops, lr_decay_ops


def _update_latest_checkpoint(checkpoint_dir,
                              best_checkpoint_dir):
    """Updates the latest checkpoint in `best_checkpoint_dir` from `checkpoint_dir`."""
    for filename in gfile.glob(os.path.join(best_checkpoint_dir, 'model.*')):
        gfile.remove(filename)
    for name in gfile.glob(os.path.join(checkpoint_dir, 'model.*')):
        gfile.copy(
            name,
            os.path.join(best_checkpoint_dir, os.path.basename(name)),
            overwrite=True)


def _create_computation_graph(
        x_train, y_train, x_validation,
        y_validation, hp_grid: Dict[str, Any]
):
    """Build the computation graph."""
    graph_tensors_and_ops = []
    metric_scores = []
    for n in range(FLAGS.n_models):
        graph_tensors_and_ops_n, metric_scores_n = graph_builder.build_graph(
            x_train=x_train,
            y_train=y_train,
            x_test=x_validation,
            y_test=y_validation,
            regression=FLAGS.regression,
            use_dnn=FLAGS.use_dnn,
            trainable=True,
            name_scope=f'model_{n}',
            **hp_grid,
        )
        graph_tensors_and_ops.append(graph_tensors_and_ops_n)
        metric_scores.append(metric_scores_n)
    return graph_tensors_and_ops, metric_scores


def _create_graph_saver(graph_tensors_and_ops,
                        logdir, num_steps_per_epoch):
    """Create saving hook(s) as well as model and checkpoint directories."""
    saver_hooks, model_dirs, best_checkpoint_dirs = [], [], []
    save_steps = num_steps_per_epoch * FLAGS.save_checkpoint_every_n_epochs
    # The MonitoredTraining Session counter increments by `n_models`
    save_steps = save_steps * FLAGS.n_models
    for n in range(FLAGS.n_models):
        scaffold = tf.train.Scaffold(
            saver=tf.train.Saver(
                var_list=graph_tensors_and_ops[n]['nn_model'].trainable_variables,
                save_relative_paths=True,
                max_to_keep=FLAGS.max_checkpoints_to_keep))
        model_dirs.append(os.path.join(logdir, 'model_{}').format(n))
        best_checkpoint_dirs.append(os.path.join(model_dirs[-1], 'best_checkpoint'))
        gfile.makedirs(best_checkpoint_dirs[-1])
        saver_hook = tf.train.CheckpointSaverHook(
            checkpoint_dir=model_dirs[-1], save_steps=save_steps, scaffold=scaffold)
        saver_hooks.append(saver_hook)
    return saver_hooks, model_dirs, best_checkpoint_dirs


def _update_metrics_and_checkpoints(sess,
                                    epoch,
                                    metric_scores,
                                    curr_best_epoch,
                                    best_validation_metric,
                                    best_train_metric,
                                    model_dir,
                                    best_checkpoint_dir,
                                    metric_name='RMSE'):
    """Update metric scores and latest checkpoint."""
    # Minimize RMSE and maximize AUROC
    compare_metric = operator.lt if FLAGS.regression else operator.gt
    # Calculate the AUROC/RMSE on the validation split
    validation_metric = metric_scores['test'](sess)
    if FLAGS.debug:
        tf.logging.info('Epoch %d %s Val %.4f', epoch, metric_name,
                        validation_metric)
    # Update the best validation metric and the corresponding train metric
    if compare_metric(validation_metric, best_validation_metric):
        curr_best_epoch = epoch
        best_validation_metric = validation_metric
        best_train_metric = metric_scores['train'](sess)
        # copy the checkpoints files *.meta *.index, *.data* each time
        # there is a better result
        _update_latest_checkpoint(model_dir, best_checkpoint_dir)
    return curr_best_epoch, best_validation_metric, best_train_metric


def training(x_train, y_train, x_validation,
             y_validation, x_test, y_test,
             logdir, hyperparameters) -> Tuple[float, float, List[float], dict]:
    """Trains the Neural Additive Model (NAM).

  Args:
    x_train: Training inputs.
    y_train: Training labels.
    x_validation: Validation inputs.
    y_validation: Validation labels.
    x_test: Test inputs.
    y_test: Test labels.
    logdir: dir to save the checkpoints.
    hyperparameters: Hyperparameters for the NAM model.

  Returns:
    Best train and validation evaluation metric obtained during NAM training averaged across all models.
    And all the test metrics for all the models.
  """
    tf.logging.info('Started training with logdir %s', logdir)
    print(f'Started training with logdir {logdir}')
    batch_size = min(FLAGS.batch_size, x_train.shape[0])
    num_steps_per_epoch = x_train.shape[0] // batch_size
    # Keep track of the best validation RMSE/AUROC and train AUROC score which
    # corresponds to the best validation metric score.
    if FLAGS.regression:
        best_train_metric = np.inf * np.ones(FLAGS.n_models)
        best_validation_metric = np.inf * np.ones(FLAGS.n_models)
    else:
        best_train_metric = np.zeros(FLAGS.n_models)
        best_validation_metric = np.zeros(FLAGS.n_models)
    # Set to a large value to avoid early stopping initially during training
    curr_best_epoch = np.full(FLAGS.n_models, np.inf)
    # Boolean variables to indicate whether the training of a specific model has
    # been early stopped.
    early_stopping = [False] * FLAGS.n_models
    # Classification: AUROC, Regression : RMSE Score
    metric_name = 'RMSE' if FLAGS.regression else 'AUROC'
    tf.reset_default_graph()
    with tf.Graph().as_default():
        tf.compat.v1.set_random_seed(hyperparameters['tf_seed'])
        # Setup your training.
        hp_copy = hyperparameters.copy()
        hp_copy.pop('tf_seed')
        graph_tensors_and_ops, metric_scores = _create_computation_graph(
            x_train, y_train, x_validation, y_validation, hp_copy)

        train_ops, lr_decay_ops = _get_train_and_lr_decay_ops(
            graph_tensors_and_ops, early_stopping)
        global_step = tf.train.get_or_create_global_step()
        increment_global_step = tf.assign(global_step, global_step + 1)
        saver_hooks, model_dirs, best_checkpoint_dirs = _create_graph_saver(
            graph_tensors_and_ops, logdir, num_steps_per_epoch)

        # Define test set inference in the graph
        test_predictions = []
        for n in range(FLAGS.n_models):
            model = graph_tensors_and_ops[n]['nn_model']
            test_predictions.append(model(x_test, training=False))  # Create the test prediction ops

        if FLAGS.debug:
            summary_writer = tf.summary.FileWriter(os.path.join(logdir, 'tb_log'))

        with tf.train.MonitoredSession(hooks=saver_hooks) as sess:
            for n in range(FLAGS.n_models):
                sess.run([
                    graph_tensors_and_ops[n]['iterator_initializer'],
                    graph_tensors_and_ops[n]['running_vars_initializer']
                ])
            for epoch in range(1, FLAGS.training_epochs + 1):
                if not all(early_stopping):
                    for _ in range(num_steps_per_epoch):
                        sess.run(train_ops)  # Train the network
                    # Decay the learning rate by a fixed ratio every epoch
                    sess.run(lr_decay_ops)
                else:
                    tf.logging.info('All models early stopped at epoch %d', epoch)
                    break

                for n in range(FLAGS.n_models):
                    if early_stopping[n]:
                        sess.run(increment_global_step)
                        continue
                    # Log summaries
                    if FLAGS.debug:
                        global_summary, global_step = sess.run([
                            graph_tensors_and_ops[n]['summary_op'],
                            graph_tensors_and_ops[n]['global_step']
                        ])
                        summary_writer.add_summary(global_summary, global_step)

                    if epoch % FLAGS.save_checkpoint_every_n_epochs == 0:
                        (curr_best_epoch[n], best_validation_metric[n],
                         best_train_metric[n]) = _update_metrics_and_checkpoints(
                            sess, epoch, metric_scores[n], curr_best_epoch[n],
                            best_validation_metric[n], best_train_metric[n], model_dirs[n],
                            best_checkpoint_dirs[n], metric_name
                        )
                        if curr_best_epoch[n] + FLAGS.early_stopping_epochs < epoch:
                            tf.logging.info('Early stopping at epoch {}'.format(epoch))
                            print('Early stopping at epoch {}'.format(epoch))
                            early_stopping[n] = True  # Set early stopping for model `n`.
                            train_ops, lr_decay_ops = _get_train_and_lr_decay_ops(
                                graph_tensors_and_ops, early_stopping)
                    # Reset running variable counters
                    sess.run(graph_tensors_and_ops[n]['running_vars_initializer'])

            test_metrics = []
            # Inference on the test set unless we're tuning the hyperparameters
            metric_dict = {}
            if not FLAGS.hp_tuning:
                for n in range(FLAGS.n_models):
                    preds = sess.run(test_predictions[n])

                    if not FLAGS.regression:
                        probas = sigmoid(preds)
                        tpr, fpr, _ = metrics.roc_curve(np.array(y_test, dtype=int), probas, pos_label=1)

                    metric = calculate_metric(y_test, preds, FLAGS.regression)
                    test_metrics.append(metric)

                    metric_dict[f'model_{n}'] = {
                        'preds': preds,
                        'tpr': tpr,
                        'fpr': fpr,
                        f'{"RSME" if FLAGS.regression else "AUROC"}': metric
                    }

                    if not FLAGS.regression:
                        metric_dict[f'model_{n}']['probas'] = probas
            else:
                test_metrics = [np.NaN] * FLAGS.n_models

    tf.logging.info('Finished training on one fold.')
    print('Finished training on one fold.')
    for n in range(FLAGS.n_models):
        tf.logging.info(
            f'Model {n}: Best Epoch {curr_best_epoch[n]}, Individual {metric_name}: Train {best_train_metric[n]}, '
            f'Validation {best_validation_metric[n]}, Test {test_metrics[n]}')
        print(f'Model {n}: Best Epoch {curr_best_epoch[n]}, Individual {metric_name}: Train {best_train_metric[n]}, '
              f'Validation {best_validation_metric[n]}, Test {test_metrics[n]}')

    # returns the mean over all models for train and val score for the best checkpoint and
    # the test scores for all models individually
    return np.mean(best_train_metric), np.mean(best_validation_metric), test_metrics, metric_dict


def create_test_train_fold(
        fold_num,
        data_x,
        data_y,
        split_criterion,
):
    """Splits the dataset into training and held-out test set."""
    # Get the training and test set based on the StratifiedKFold split
    (x_train_all, y_train_all), (x_test, y_test), (groups_train, groups_test) = data_utils.get_train_test_fold(
        data_x,
        data_y,
        fold_num=fold_num,
        num_folds=_N_FOLDS,
        stratified=not FLAGS.regression,
        group_split=split_criterion)

    data_gen = data_utils.split_training_dataset(
        x_train_all,
        y_train_all,
        FLAGS.num_splits,
        group_split=list(groups_train),
        stratified=not FLAGS.regression)

    test_dataset = (x_test, y_test)

    return data_gen, test_dataset


def single_split_training(data_gen,
                          test_data,
                          logdir,
                          fold):
    """
    Uses a specific (training, validation) split for NAM training.
        data_gen: Iterator that generates (x_train, y_train), (x_validation, y_validation)
        test_data: (x_test, y_test)
        logdir: Directory to save the model checkpoints.
        fold: Fold number.
    """
    for _ in range(FLAGS.data_split):
        (x_train, y_train), (x_validation, y_validation) = next(data_gen)
    curr_logdir = os.path.join(logdir, 'fold_{}',
                               'split_{}').format(fold,
                                                  FLAGS.data_split)

    if FLAGS.hp_tuning and fold == 1:
        print(f'Hyperparameter tuning on fold {fold}')
        # Perform hyperparameter tuning on the first fold
        # This is done only once for the first fold, the hps are defined in the flags on top of the file
        # TODO: move this to config file or somewhere else
        hyper_parameters = {
            'learning_rate': [1e-3, 1e-4],
            'output_regularization': [0.0, 0.1],
            'l2_regularization': [0.0, 0.1],
            'batch_size': [32],
            'decay_rate': [0.99, 0.995],
            'dropout': [0.0, 0.5, 0.7],
            'feature_dropout': [0.0, 0.1],
            'num_basis_functions': [1000, 2000],
            'units_multiplier': [2, 4, 8],
            'shallow': [True, False],
            'tf_seed': [1],
        }

        # create file where to log the results_baselines for the tuning
        hp_file_name = f'{curr_logdir}/hp_tuning_results.txt'
        os.makedirs(os.path.dirname(hp_file_name), exist_ok=True)
        with open(hp_file_name, 'w', encoding='utf8') as f:
            f.write('Hyperparameter tuning results\n')
            f.write(f'Hyperparameters tested: {hyper_parameters}\n')
            f.write(f'Metric: {"AUROC" if not FLAGS.regression else "RMSE"}\n')

        # Generate all combinations of hyperparameters as dictionaries
        hp_combinations = [dict(zip(hyper_parameters.keys(), values)) for values in product(*hyper_parameters.values())]

        best_val_score = np.inf if FLAGS.regression else 0
        best_train_score = None
        best_hps = None

        print(f'Testing {len(hp_combinations)} hyperparameter combinations')

        for hp_grid in tqdm(hp_combinations):

            train_score, val_score, test_scores, metric_dict = training(x_train, y_train, x_validation, y_validation,
                                                                        *test_data, curr_logdir, hp_grid)

            if FLAGS.regression and val_score < best_val_score:
                best_val_score = val_score
                best_train_score = train_score
                best_hps = hp_grid
                with open(hp_file_name, 'a', encoding='utf8') as f:
                    f.write(f'Current best hyperparameters: {best_hps}\n')

            elif not FLAGS.regression and val_score > best_val_score:
                best_val_score = val_score
                best_train_score = train_score
                best_hps = hp_grid
                with open(hp_file_name, 'a', encoding='utf8') as f:
                    f.write(f'Current best hyperparameters: {best_hps}\n')
            else:
                with open(hp_file_name, 'a', encoding='utf8') as f:
                    f.write(f'Tested hyperparameters: {hp_grid}\n')

            with open(hp_file_name, 'a', encoding='utf8') as f:
                f.write(f'Train score: {train_score}\n')
                f.write(f'Validation score: {val_score}\n')

        with open(hp_file_name, 'a', encoding='utf8') as f:
            f.write(f'Final best hyperparameters: {best_hps}\n')

        # repeat training one more time for this fold with the best hyperparameters
        FLAGS.hp_tuning = False
        _, _, best_test_scores, metric_dict = training(x_train, y_train, x_validation, y_validation, *test_data,
                                                       curr_logdir, best_hps)

        with open(hp_file_name, 'a', encoding='utf8') as f:
            f.write(f'Train score: {best_train_score}\n')
            f.write(f'Validation score: {best_val_score}\n')
            f.write(f'Test scores for best hyperparameters: {best_test_scores}')

        print(f'Finished hyperparameter tuning on fold {fold}')
        print(f'Best hyperparameters: {best_hps}')

        return best_hps, best_train_score, best_val_score, best_test_scores, metric_dict

    else:
        if FLAGS.config_path is not None:

            print('++ Loading best config.')
            with open(f'{FLAGS.config_path}.yaml', 'r') as file:
                hps = yaml.safe_load(file)

        else:
            hps = {
                'learning_rate': FLAGS.learning_rate,
                'output_regularization': FLAGS.output_regularization,
                'l2_regularization': FLAGS.l2_regularization,
                'batch_size': FLAGS.batch_size,
                'decay_rate': FLAGS.decay_rate,
                'dropout': FLAGS.dropout,
                'feature_dropout': FLAGS.feature_dropout,
                'num_basis_functions': FLAGS.num_basis_functions,
                'units_multiplier': FLAGS.units_multiplier,
                'shallow': FLAGS.shallow,
                'tf_seed': FLAGS.tf_seed,
            }
            # hps = {'learning_rate': 0.01, 'output_regularization': 0.0, 'l2_regularization': 0.0, 'batch_size': 28,
            #        'decay_rate': 0.995, 'dropout': 0.5, 'feature_dropout': 0.1, 'num_basis_functions': 1000,
            #        'units_multiplier': 4, 'shallow': True, 'tf_seed': 2}

        train_score, val_score, test_scores, metric_dict = training(x_train, y_train, x_validation, y_validation,
                                                                    *test_data, curr_logdir, hps)
        return hps, train_score, val_score, test_scores, metric_dict


def main(argv):

    GPU = FLAGS.gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    config = tf.ConfigProto(log_device_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True
    tf_session = tf.Session(config=config)
    del argv  # Unused
    tf.logging.set_verbosity(tf.logging.INFO)

    data_x, data_y, column_names, split_criterion = data_utils.load_dataset(FLAGS.dataset_name, FLAGS.group_by,
                                                                            FLAGS.dataset_folder)
    test_scores_all_models = []
    all_metrics = OrderedDict()

    if FLAGS.all_folds:
        print(f'Dataset: {FLAGS.dataset_name}, Size: {data_x.shape[0]}')
        tf.logging.info('Dataset: %s, Size: %d', FLAGS.dataset_name, data_x.shape[0])

        for fold in range(1, _N_FOLDS + 1):
            tf.logging.info('Cross-val fold: %d/%d', fold, _N_FOLDS)
            print(f'Cross-val fold: {fold}/{_N_FOLDS}')
            data_gen, test_data = create_test_train_fold(fold, data_x, data_y, split_criterion)
            _, _, _, test_scores, metric_dict = single_split_training(data_gen, test_data, FLAGS.logdir, fold)

            test_scores_all_models.append(test_scores)

            all_metrics[f'fold_{fold}'] = metric_dict

            # dump it after each fold and overwrite the file
            with open(f'{FLAGS.logdir}/metrics_dict.txt', 'w', encoding='utf8') as f:
                f.write(str(all_metrics))

    else:
        tf.logging.info('Dataset: %s, Size: %d', FLAGS.dataset_name, data_x.shape[0])
        tf.logging.info('Cross-val fold: %d/%d', FLAGS.fold_num, _N_FOLDS)
        print(f'Dataset: {FLAGS.dataset_name}, Size: {data_x.shape[0]}')
        print(f'Cross-val fold: {FLAGS.fold_num}/{_N_FOLDS}')
        data_gen, test_data = create_test_train_fold(FLAGS.fold_num, data_x, data_y, split_criterion)
        _, _, _, test_scores_all_models, metric_dict = single_split_training(data_gen, test_data, FLAGS.logdir, FLAGS.fold_num)

        all_metrics[f'fold_{FLAGS.fold_num}'] = metric_dict

    print(f'Test scores for all models for each fold: {test_scores_all_models}')
    print(f'Mean test scores for all models: {np.mean(test_scores_all_models, axis=0)}')

    with open(f'{FLAGS.logdir}/metrics_dict.txt', 'w', encoding='utf8') as f:
        f.write(str(all_metrics))


if __name__ == '__main__':
    flags.mark_flag_as_required('logdir')
    flags.mark_flag_as_required('training_epochs')
    app.run(main)
