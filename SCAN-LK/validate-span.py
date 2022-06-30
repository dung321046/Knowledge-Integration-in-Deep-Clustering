"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import argparse
import statistics

import numpy as np
import torch
from sklearn.metrics.cluster import normalized_mutual_info_score
from termcolor import colored

from utils.common_config import get_val_transformations, \
    get_val_dataset, get_val_dataloader, \
    get_model
from utils.config import create_config
from utils.evaluate_utils import get_predictions

FLAGS = argparse.ArgumentParser(description='SCAN Loss')
FLAGS.add_argument('--config_env', help='Location of path config file')
FLAGS.add_argument('--config_exp', help='Location of experiments config file')


def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return w[row_ind, col_ind].sum() * 1.0 / y_pred.size


def produce_clustering(model, model_path, val_dataloader, p):
    model_checkpoint = torch.load(model_path, map_location='cpu')
    model.module.load_state_dict(model_checkpoint['model'])
    predictions = get_predictions(p, val_dataloader, model)[0]
    return predictions


PATH = "C:\\Users\\dung3\\PycharmProjects\\Unsupervised-Classification\\"


def get_mean_and_std(arr):
    if len(arr) == 0:
        return -1000000, 0.0
    if len(arr) == 1:
        return arr[0], 0.0
    if type(arr[0]) == str:
        return arr[0]
    return statistics.mean(arr), np.std(np.asarray(arr))


def print_sat(np_predictions, span_groups):
    for group in range(3):
        from collections import Counter
        c = Counter([np_predictions[g] for g in span_groups[group]["group"]])
        topC = c.most_common(2)
        print("Group ", group, " #Sat:", topC[0][1] + topC[1][1])


def main():
    args = FLAGS.parse_args()
    p = create_config(args.config_env, args.config_exp)
    print(colored(p, 'red'))

    # CUDNN
    torch.backends.cudnn.benchmark = True

    # Data
    print(colored('Get dataset and dataloaders', 'blue'))
    val_transformations = get_val_transformations(p)
    from data.stl import STL10
    dataset = STL10(transform=val_transformations, split='test', download=True)
    val_dataset = get_val_dataset(p, val_transformations, to_neighbors_dataset=True)
    val_dataloader = get_val_dataloader(p, val_dataset)
    span_targets = [[0, 1], [3, 5], [4, 6]]
    span_groups = []
    for span_target in span_targets:
        span_groups.append({"group": [], "target": span_target})
    for i, y in enumerate(val_dataset.dataset.labels):
        for group, span_target in enumerate(span_targets):
            if y in span_target:
                span_groups[group]["group"].append(i)
    for group in range(3):
        print("Group ", group, " #Con:", len(span_groups[group]["group"]))
    print('Validation transforms:', val_transformations)
    model = get_model(p, p['pretext_model'])
    # print(model)
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    predictions = produce_clustering(model,
                                     PATH + r"results\stl-10\scan-baseline\model.pth.tar",
                                     val_dataloader, p)
    print("Init")
    np_predictions = np.asarray(predictions["predictions"])
    print_sat(np_predictions, span_groups)
    init_prediction = predictions['predictions']
    print("ACC(SCAN):", acc(val_dataset.dataset.labels, init_prediction))
    print("NMI(SCAN):",
          normalized_mutual_info_score(val_dataset.dataset.labels, init_prediction, average_method="arithmetic"))
    accs = []
    nmis = []
    # Model
    # print(colored('Get model', 'blue'))
    model = get_model(p, p['pretext_model'])
    # print(model)
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    predictions = produce_clustering(model, PATH + r"results\stl-10\scan-span\model.pth.tar",
                                     val_dataloader, p)
    np_predictions = np.asarray(predictions["predictions"])
    print_sat(np_predictions, span_groups)
    accs.append(acc(val_dataset.dataset.labels, predictions['predictions']))
    nmis.append(normalized_mutual_info_score(val_dataset.dataset.labels, predictions['predictions'],
                                             average_method="arithmetic"))


if __name__ == "__main__":
    main()
