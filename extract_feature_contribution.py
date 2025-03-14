import argparse
import os

import joblib
import numpy as np
import tensorflow.compat.v2 as tf
import yaml
from absl import app
from matplotlib import pyplot as plt, patches

from nam_train import FLAGS
from neural_additive_models import graph_builder, data_utils

tf.enable_v2_behavior()

import os.path as osp

import tensorflow as tf

from textwrap import wrap


def inverse_min_max_scaler(x, min_val, max_val):
    return (x + 1) / 2 * (max_val - min_val) + min_val


def load_nam(argv):
    del argv

    logs_model = 'in_paper/2025-01-21-21:29_label-all_tq_correct_split-reader_id'
    assert logs_model is not None, "Please provide the path to load the model on line 30."

    # expert_cls_label, all_tq_correct, all_bq_correct
    label = "all_tq_correct"
    assert label is not None, "Please provide the label on line 34."

    dir_this_file = osp.dirname(osp.abspath(__file__))
    FLAGS.logdir = f'{dir_this_file}/feature_analysis/{logs_model}/'

    # get path of this repo
    this_repo_path = osp.dirname(osp.abspath(__file__))

    # make dir
    if not tf.io.gfile.exists(FLAGS.logdir):
        tf.io.gfile.makedirs(FLAGS.logdir)

    dataset_name = 'PoTeC'

    dataset = data_utils.load_potec_data(FLAGS.group_by, FLAGS.dataset_folder, FLAGS.logdir, label)

    col_min_max = load_col_min_max(dataset)

    data_x, data_y, feature_names, split_criterion = data_utils.reformat_data(dataset, dataset_name)

    (x_train_all, y_train_all), (test_x, test_y), (groups_x, groups_y) = data_utils.get_train_test_fold(
        data_x,
        data_y,
        fold_num=1,
        num_folds=5,
        stratified=not FLAGS.regression,
        group_split=split_criterion)

    data_gen = data_utils.split_training_dataset(
        x_train_all,
        y_train_all,
        n_splits=20,
        stratified=not FLAGS.regression,
        group_split=list(groups_x)
    )

    (x_train, y_train), _ = next(data_gen)

    tf.compat.v1.reset_default_graph()

    # load the hps used for this model
    hp_path = osp.join(this_repo_path, 'results_nam', f'{logs_model}', 'hps.json')
    with open(hp_path, 'r') as file:
        hps = yaml.safe_load(file)

    nn_model = graph_builder.create_nam_model(
        x_train=x_train,
        dropout=hps['dropout'],
        feature_dropout=hps['feature_dropout'],
        num_basis_functions=hps['num_basis_functions'],
        activation=hps['activation'],
        trainable=False,
        shallow=hps['shallow'],
        name_scope='model_0',
        units_multiplier=hps['units_multiplier'],
    )

    _ = nn_model(x_train[:1])
    nn_model.summary()

    model_logdir = f'{this_repo_path}/results_nam/{logs_model}/fold_1/split_1/'
    ckpt_dir = osp.join(model_logdir, 'model_0', 'best_checkpoint')
    ckpt_files = sorted(tf.io.gfile.listdir(ckpt_dir))
    ckpt = osp.join(ckpt_dir, ckpt_files[0].split('.data')[0])
    ckpt_reader = tf.train.load_checkpoint(ckpt)
    variables = tf.train.list_variables(ckpt)

    # Print variable names and shapes
    # for var_name, shape in variables:
    # print(f"Variable Name: {var_name}, Shape: {shape}")

    for var in nn_model.variables:
        tensor_name = var.name.split(':', 1)[0].replace('nam', 'model_0/nam')
        value = ckpt_reader.get_tensor(tensor_name)
        var.assign(value)

    test_predictions = get_test_predictions(nn_model, test_x)

    num_features = data_x.shape[1]
    single_features = np.split(data_x, num_features, axis=1)
    test_features = np.split(test_x, num_features, axis=1)
    unique_features = [np.unique(x, axis=0) for x in single_features]

    # gets the predictions from each feature NN for each individual feature value in the dataset
    feature_predictions = get_feature_predictions(nn_model, single_features)
    test_feature_importances = get_feature_predictions(nn_model, test_features)

    test_metric = graph_builder.calculate_metric(
        test_y, test_predictions, regression=FLAGS.regression)
    metric_str = 'RMSE' if FLAGS.regression else 'AUROC'
    print(f'{metric_str}: {test_metric}')

    single_features_scaled = {}
    unique_features_scaled = {}
    for i, col in enumerate(feature_names):
        min_val, max_val = col_min_max[col]
        unique_features_scaled[col] = inverse_min_max_scaler(
            unique_features[i][:, 0], min_val, max_val)
        single_features_scaled[col] = inverse_min_max_scaler(
            single_features[i][:, 0], min_val, max_val)

    # map all the feature names to the predictions of each unique value of that feature
    feature_predictions_per_value = {col: predictions for col, predictions in zip(feature_names, feature_predictions)}

    all_indices = {}
    mean_pred = {}

    for i, col in enumerate(feature_names):
        x_i = data_x[:, i]
        all_indices[col] = np.searchsorted(unique_features[i][:, 0], x_i, 'left')
    # get the predicted value of each feature for each data point and average it per feature
    for feature in feature_names:
        mean_pred[feature] = np.mean([feature_predictions_per_value[feature][i] for i in all_indices[feature]])

    joblib_dir = f'{FLAGS.logdir}/data_for_plots/'
    if not os.path.exists(joblib_dir):
        os.makedirs(joblib_dir)

    # save feature_predictions, feature_names, test_x
    test_feature_importances = np.array(test_feature_importances)
    path = osp.join(joblib_dir, f'test_feature_importances_{label}.joblib')
    joblib.dump(test_feature_importances, path, compress=3, protocol=2)

    feature_names = np.array(feature_names)
    path = osp.join(joblib_dir, f'feature_names_{label}.joblib')
    joblib.dump(feature_names, path, compress=3, protocol=2)

    test_x = np.array(test_x)
    path = osp.join(joblib_dir, f'test_x_{label}.joblib')
    joblib.dump(test_x, path, compress=3, protocol=2)


def load_col_min_max(dataset):
    if 'full' in dataset:
        dataset = dataset['full']
    x = dataset['X']
    col_min_max = {}
    for col in x:
        unique_vals = x[col].unique()
        col_min_max[col] = (np.min(unique_vals), np.max(unique_vals))

    return col_min_max


def partition(lst, batch_size):
    lst_len = len(lst)
    index = 0
    while index < lst_len:
        x = lst[index: batch_size + index]
        index += batch_size
        yield x


def generate_predictions(gen, nn_model):
    y_pred = []
    while True:
        try:
            x = next(gen)
            pred = nn_model(x).numpy()
            y_pred.extend(pred)
        except:
            break
    return y_pred


def get_test_predictions(nn_model, x_test, batch_size=256):
    batch_size = min(batch_size, x_test.shape[0])
    generator = partition(x_test, batch_size)
    return generate_predictions(generator, nn_model)


def get_feature_predictions(nn_model, unique_features):
    """
    calls the NN for each unique value for all features and gets the output from this feature NN
    """
    feature_predictions = []
    for feature_idx, feature_val in enumerate(unique_features):
        f_preds = nn_model.feature_nns[feature_idx](feature_val, training=nn_model._false)
        feature_predictions.append(f_preds)
    return feature_predictions


if __name__ == '__main__':
    app.run(load_nam)
