import argparse

import numpy as np
import tensorflow.compat.v2 as tf
from absl import app
from matplotlib import pyplot as plt

from nam_train import FLAGS
from neural_additive_models import graph_builder, data_utils

tf.enable_v2_behavior()

import os.path as osp

import tensorflow as tf


def inverse_min_max_scaler(x, min_val, max_val):
    return (x + 1) / 2 * (max_val - min_val) + min_val


def load_nam(argv):
    """
    Train the NAM model on the full data.
    """

    del argv

    logs_model = '2024-12-22-10:53'

    FLAGS.logdir = f'/Users/debor/repos/PoTeC/feature_analysis/{logs_model}/'

    # make dir
    if not tf.io.gfile.exists(FLAGS.logdir):
        tf.io.gfile.makedirs(FLAGS.logdir)

    dataset_name = 'PoTeC'

    dataset = data_utils.load_potec_data(FLAGS.group_by, FLAGS.dataset_folder, FLAGS.logdir)

    col_min_max = load_col_min_max(dataset)

    data_x, data_y, column_names, split_criterion = data_utils.reformat_data(dataset, dataset_name)

    (x_train_all, y_train_all), test_dataset, _ = data_utils.get_train_test_fold(
        data_x,
        data_y,
        fold_num=1,
        num_folds=5,
        stratified=not FLAGS.regression,
        group_split=split_criterion)

    data_gen = data_utils.split_training_dataset(
        x_train_all, y_train_all,
        n_splits=20, stratified=not FLAGS.regression)
    (x_train, y_train), _ = next(data_gen)

    tf.compat.v1.reset_default_graph()

    nn_model = graph_builder.create_nam_model(
        x_train=x_train,
        dropout=0.7,
        feature_dropout=0.2,
        num_basis_functions=2000,
        activation='relu',
        trainable=False,
        shallow=False,
        name_scope='model_0',
        units_multiplier=8
    )

    _ = nn_model(x_train[:1])
    nn_model.summary()

    logdir = f'/Users/debor/repos/PoTeC/results_nam/logs_{logs_model}/fold_1/split_1/'
    ckpt_dir = osp.join(logdir, 'model_0', 'best_checkpoint')
    ckpt_files = sorted(tf.io.gfile.listdir(ckpt_dir))
    ckpt = osp.join(ckpt_dir, ckpt_files[0].split('.data')[0])
    ckpt_reader = tf.train.load_checkpoint(ckpt)
    # variables = tf.train.list_variables(ckpt)

    # Print variable names and shapes
    # for var_name, shape in variables:
    #  print(f"Variable Name: {var_name}, Shape: {shape}")

    for var in nn_model.variables:
        tensor_name = var.name.split(':', 1)[0].replace('nam', 'model_0/nam')
        value = ckpt_reader.get_tensor(tensor_name)
        var.assign(value)

    test_predictions = get_test_predictions(nn_model, test_dataset[0])
    unique_features = compute_features(data_x.copy())
    feature_predictions = get_feature_predictions(nn_model, unique_features)

    test_metric = graph_builder.calculate_metric(
        test_dataset[1], test_predictions, regression=FLAGS.regression)
    metric_str = 'RMSE' if FLAGS.regression else 'AUROC'
    print(f'{metric_str}: {test_metric}')

    NUM_FEATURES = data_x.shape[1]
    SINGLE_FEATURES = np.split(data_x, NUM_FEATURES, axis=1)
    UNIQUE_FEATURES = [np.unique(x, axis=0) for x in SINGLE_FEATURES]

    SINGLE_FEATURES_ORIGINAL = {}
    UNIQUE_FEATURES_ORIGINAL = {}
    for i, col in enumerate(column_names):
        min_val, max_val = col_min_max[col]
        UNIQUE_FEATURES_ORIGINAL[col] = inverse_min_max_scaler(
            UNIQUE_FEATURES[i][:, 0], min_val, max_val)
        SINGLE_FEATURES_ORIGINAL[col] = inverse_min_max_scaler(
            SINGLE_FEATURES[i][:, 0], min_val, max_val)

    COL_NAMES = {}
    COL_NAMES[dataset_name] = {x: x for x in column_names}

    avg_hist_data = {col: predictions for col, predictions in zip(column_names, feature_predictions)}

    ALL_INDICES = {}
    MEAN_PRED = {}

    for i, col in enumerate(column_names):
        x_i = data_x[:, i]
        ALL_INDICES[col] = np.searchsorted(UNIQUE_FEATURES[i][:, 0], x_i, 'left')
    for col in column_names:
        MEAN_PRED[col] = np.mean([avg_hist_data[col][i] for i in ALL_INDICES[col]])

    print(MEAN_PRED)

    x1, x2 = compute_mean_feature_importance(avg_hist_data, MEAN_PRED)
    cols = [COL_NAMES[dataset_name][x] for x in x1]
    fig = plot_mean_feature_importance(x2, cols, dataset_name)


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


def get_feature_predictions_old(nn_model, features, batch_size=256):
    """Get feature predictions for unique values for each feature."""
    unique_feature_pred, unique_feature_gen = [], []
    for i, feature in enumerate(features):
        batch_size = min(batch_size, feature.shape[0])
        generator = partition(feature, batch_size)
        feature_pred = lambda x: nn_model.feature_nns[i](
            x, training=nn_model._false)  # pylint: disable=protected-access
        unique_feature_gen.append(generator)
        unique_feature_pred.append(feature_pred)

    feature_predictions = [
        generate_predictions(generator, feature_pred) for
        feature_pred, generator in zip(unique_feature_pred, unique_feature_gen)
    ]
    feature_predictions = [np.array(x) for x in feature_predictions]
    return feature_predictions


def get_feature_predictions(nn_model, unique_features):
    feature_predictions = []
    for c, i in enumerate(unique_features):
        f_preds = nn_model.feature_nns[c](i, training=nn_model._false)
        feature_predictions.append(f_preds)
    return feature_predictions


def compute_features(x_data):
    """
    Compute the unique features for each column in the dataset.
    x_data (np.ndarray): preloaded features of the dataset.
    """
    single_features = np.split(x_data, x_data.shape[1], axis=1)
    unique_features = [np.unique(f, axis=0) for f in single_features]
    return unique_features


def compute_mean_feature_importance(avg_hist_data, mean_pred):
    mean_abs_score = {}
    for k in avg_hist_data:
        mean_abs_score[k] = np.mean(np.abs(avg_hist_data[k] - mean_pred[k]))
    x1, x2 = zip(*mean_abs_score.items())
    return x1, x2


def plot_mean_feature_importance(x2, cols, dataset_name, width=0.3, num_top_features=5):
    """
    Plots the mean feature importance for the top features, ensuring x-axis labels fit.

    Parameters:
        x2 (list or array-like): Feature importance values.
        cols (list): Feature names corresponding to the importance values.
        dataset_name (str): Name of the dataset (used in the title).
        width (float): Bar width for the plot.
        num_top_features (int): Number of top features to display.

    Returns:
        matplotlib.figure.Figure: The generated plot.
    """
    # Ensure inputs are numpy arrays for consistent behavior
    x2 = np.array(x2)
    num_features = len(x2)

    # Handle cases where num_top_features exceeds the number of available features
    num_top_features = min(num_top_features, num_features)

    # Sort indices based on x2 importance values in descending order
    x2_indices = np.argsort(-x2)
    cols_sorted = [cols[i] for i in x2_indices]
    x2_sorted = x2[x2_indices]

    # Select the top features
    cols_top = cols_sorted[:num_top_features]
    x2_top = x2_sorted[:num_top_features]

    # Adjust figure size based on the number of features to ensure labels fit
    fig_height = max(4, 0.5 * num_top_features)
    fig, ax = plt.subplots(figsize=(6, fig_height))
    ind = np.arange(num_top_features)  # x locations for the groups

    # Plot the bars
    ax.bar(ind, x2_top, width, label='Feature Importance', color='skyblue')

    # Add labels, title, and legend
    ax.set_xticks(ind)
    ax.set_xticklabels(cols_top, rotation=45, ha='right')  # Rotate labels for better fit
    ax.set_ylabel('Mean Absolute Score')
    ax.set_title(f'Overall Importance: {dataset_name}')
    ax.legend(loc='upper right')

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()

    return fig


def parse_args() -> dict:
    """
    Parse the arguments for the evaluation script.
    """
    # create an argument parser
    parser = argparse.ArgumentParser(description='Analyse feature activation')
    parser.add_argument(
        '--hps',
        type=str,
        help='Path to json file containing best hyperparameters',
    )


if __name__ == '__main__':
    app.run(load_nam)
