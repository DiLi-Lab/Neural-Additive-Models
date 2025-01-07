import argparse
import json

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


def inverse_min_max_scaler(x, min_val, max_val):
    return (x + 1) / 2 * (max_val - min_val) + min_val


def load_nam(argv):
    del argv

    logs_model = '2025-01-07-14:03_label-all_bq_correct_split-reader_id'

    FLAGS.logdir = f'/Users/debor/repos/PoTeC/feature_analysis/{logs_model}/'

    # get path og this repo
    this_repo_path = osp.dirname(osp.abspath(__file__))

    # make dir
    if not tf.io.gfile.exists(FLAGS.logdir):
        tf.io.gfile.makedirs(FLAGS.logdir)

    dataset_name = 'PoTeC'

    dataset = data_utils.load_potec_data(FLAGS.group_by, FLAGS.dataset_folder, FLAGS.logdir, FLAGS.label)

    col_min_max = load_col_min_max(dataset)

    data_x, data_y, column_names, split_criterion = data_utils.reformat_data(dataset, dataset_name)

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
        dropout=0.7,
        feature_dropout=0.2,
        num_basis_functions=2000,
        activation='relu',
        trainable=False,
        shallow=False,
        name_scope='model_0',
        units_multiplier=8,
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
    #for var_name, shape in variables:
       #print(f"Variable Name: {var_name}, Shape: {shape}")

    for var in nn_model.variables:
        tensor_name = var.name.split(':', 1)[0].replace('nam', 'model_0/nam')
        value = ckpt_reader.get_tensor(tensor_name)
        var.assign(value)

    test_predictions = get_test_predictions(nn_model, test_x)
    unique_features = compute_features(data_x.copy())
    feature_predictions = get_feature_predictions(nn_model, unique_features)

    test_metric = graph_builder.calculate_metric(
        test_y, test_predictions, regression=FLAGS.regression)
    metric_str = 'RMSE' if FLAGS.regression else 'AUROC'
    print(f'{metric_str}: {test_metric}')

    num_features = data_x.shape[1]
    single_features = np.split(data_x, num_features, axis=1)
    unique_features = [np.unique(x, axis=0) for x in single_features]

    single_features_original = {}
    unique_features_original = {}
    for i, col in enumerate(column_names):
        min_val, max_val = col_min_max[col]
        unique_features_original[col] = inverse_min_max_scaler(
            unique_features[i][:, 0], min_val, max_val)
        single_features_original[col] = inverse_min_max_scaler(
            single_features[i][:, 0], min_val, max_val)

    col_names = {dataset_name: {x: x for x in column_names}}

    avg_hist_data = {col: predictions for col, predictions in zip(column_names, feature_predictions)}

    all_indices = {}
    mean_pred = {}

    feature_label_mapping = {}

    for i, col in enumerate(column_names):
        x_i = data_x[:, i]
        all_indices[col] = np.searchsorted(unique_features[i][:, 0], x_i, 'left')
    for col in column_names:
        mean_pred[col] = np.mean([avg_hist_data[col][i] for i in all_indices[col]])

    print(mean_pred)

    x1, x2 = compute_mean_feature_importance(avg_hist_data, mean_pred)
    cols = [col_names[dataset_name][x] for x in x1]

    num_top_features = 10

    fig = plot_mean_feature_importance(x2, cols, dataset_name, num_top_features=num_top_features)
    fig.savefig(osp.join(FLAGS.logdir, f'mean_feature_importance_top_{num_top_features}.png'))

    colors = [[0.9, 0.4, 0.5], [0.5, 0.9, 0.4], [0.4, 0.5, 0.9], [0.9, 0.5, 0.9]]
    num_cols = 6
    n_blocks = 20

    MIN_Y = None
    MAX_Y = None

    NUM_ROWS = int(np.ceil(num_features / num_cols))
    fig = plt.figure(num=None, figsize=(num_cols * 4.5, NUM_ROWS * 4.5),
                     facecolor='w', edgecolor='k')

    MIN_Y, MAX_Y = plot_all_hist(avg_hist_data, NUM_ROWS, num_cols, colors[2],
                                 col_names=column_names, feature_label_mapping=feature_label_mapping,
                                 unique_features=unique_features_original,
                                 mean_predictions=mean_pred, dataset_name=dataset_name, categorical_names=column_names,
                                 min_y=MIN_Y, max_y=MAX_Y, feature_to_use=column_names)
    shade_by_density_blocks(avg_hist_data, NUM_ROWS, num_cols, unique_features_original, column_names,
                            n_blocks=n_blocks, feature_to_use=column_names)

    # This is for plotting individual plots when there are multiple models
    """
    for pred in feature_predictions:
      model_hist = {col: pred[0, i] for i, col in enumerate(column_names)}
      plot_all_hist(model_hist, NUM_ROWS, num_cols,
                    color_base=[0.3, 0.4, 0.9, 0.2], alpha=0.06,
                    linewidth=0.1, min_y=MIN_Y, max_y=MAX_Y, feature_to_use=features)
    """
    plt.subplots_adjust(hspace=0.23)
    plt.show()


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


def shade_by_density_blocks(hist_data,
                            num_rows,
                            num_cols,
                            unique_features,
                            categorical_names,
                            feature_to_use=None,
                            n_blocks=5, color=[0.9, 0.5, 0.5],
                            ):
    hist_data_pairs = list(hist_data.items())
    hist_data_pairs.sort(key=lambda x: x[0])
    min_y = np.min([np.min(a[1]) for a in hist_data_pairs])
    max_y = np.max([np.max(a[1]) for a in hist_data_pairs])
    min_max_dif = max_y - min_y
    min_y = min_y - 0.01 * min_max_dif
    max_y = max_y + 0.01 * min_max_dif

    if feature_to_use:
        hist_data_pairs = [v for v in hist_data_pairs if v[0] in feature_to_use]

    for i, (name, pred) in enumerate(hist_data_pairs):

        # unique_x_data, single_feature_data, pred = data
        unique_x_data = unique_features[name]
        single_feature_data = unique_features[name]
        ax = plt.subplot(num_rows, num_cols, i + 1)
        min_x = np.min(unique_x_data)
        max_x = np.max(unique_x_data)
        x_n_blocks = min(n_blocks, len(unique_x_data))
        if name in categorical_names:
            min_x -= 0.5
            max_x += 0.5
        segments = (max_x - min_x) / x_n_blocks
        density = np.histogram(single_feature_data, bins=x_n_blocks)
        normed_density = density[0] / np.max(density[0])
        rect_params = []
        for p in range(x_n_blocks):
            start_x = min_x + segments * p
            end_x = min_x + segments * (p + 1)
            # start_insert_index = min(
            #     max(0, np.searchsorted(density[1], start_x) - 1), x_n_blocks - 1)
            # end_insert_index = min(
            #     max(0, np.searchsorted(density[1], end_x) - 1), x_n_blocks - 1)
            # d = (normed_density[start_insert_index] + normed_density[end_insert_index])/2
            d = min(1.0, 0.01 + normed_density[p])
            rect_params.append((d, start_x, end_x))

        for param in rect_params:
            alpha, start_x, end_x = param
            rect = patches.Rectangle((start_x, min_y - 1), end_x - start_x,
                                     max_y - min_y + 1, linewidth=0.01,
                                     edgecolor=color, facecolor=color, alpha=alpha)
            ax.add_patch(rect)


def plot_all_hist(hist_data, num_rows, num_cols, color_base, col_names, feature_label_mapping,
                  unique_features,
                  categorical_names, mean_predictions, dataset_name,
                  linewidth=3.0, min_y=None, max_y=None, alpha=1.0,
                  feature_to_use=None
                  ):
    hist_data_pairs = list(hist_data.items())
    hist_data_pairs.sort(key=lambda x: x[0])
    if min_y is None:
        min_y = np.min([np.min(a) for _, a in hist_data_pairs])
    if max_y is None:
        max_y = np.max([np.max(a) for _, a in hist_data_pairs])
    min_max_dif = max_y - min_y
    min_y = min_y - 0.01 * min_max_dif
    max_y = max_y + 0.01 * min_max_dif
    col_mapping = col_names[dataset_name]
    feature_mapping = feature_label_mapping[dataset_name]

    total_mean_bias = 0

    if feature_to_use:
        hist_data_pairs = [v for v in hist_data_pairs if v[0] in feature_to_use]

    for i, (name, pred) in enumerate(hist_data_pairs):
        mean_pred = mean_predictions[name]  # np.mean(pred)
        total_mean_bias += mean_pred
        unique_x_data = unique_features[name]
        plt.subplot(num_rows, num_cols, i + 1)

        if name in categorical_names:
            unique_x_data = np.round(unique_x_data, decimals=1)
            if len(unique_x_data) <= 2:
                step_loc = "mid"
            else:
                step_loc = "post"
            unique_plot_data = np.array(unique_x_data) - 0.5
            unique_plot_data[-1] += 1
            plt.step(unique_plot_data, pred - mean_pred, color=color_base,
                     linewidth=linewidth, where=step_loc, alpha=alpha)

            if name in feature_mapping:
                labels, rot = feature_mapping[name]
            else:
                labels = unique_x_data
                rot = None
            plt.xticks(unique_x_data, labels=labels, fontsize='x-large', rotation=rot)
        else:
            plt.plot(unique_x_data, pred - mean_pred, color=color_base,
                     linewidth=linewidth, alpha=alpha)
            plt.xticks(fontsize='x-large')

        plt.ylim(min_y, max_y)
        plt.yticks(fontsize='x-large')
        min_x = np.min(unique_x_data)
        max_x = np.max(unique_x_data)
        if name in categorical_names:
            min_x -= 0.5
            max_x += 0.5
        plt.xlim(min_x, max_x)
        if i % num_cols == 0:
            plt.ylabel('House Price Contribution', fontsize='x-large')
        plt.xlabel(col_mapping[name], fontsize='x-large')
    return min_y, max_y


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
