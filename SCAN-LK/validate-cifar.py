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

from utils.common_config import get_train_transformations, get_val_transformations, \
    get_val_dataset_henry, get_val_dataloader, \
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


def show_value(s, arr):
    a, b = get_mean_and_std(arr)
    print(s + ": {:.2f} $\\pm$ {:.2f}".format(100 * a, 100 * b))


def main():
    args = FLAGS.parse_args()
    p = create_config(args.config_env, args.config_exp)
    print(colored(p, 'red'))

    # CUDNN
    torch.backends.cudnn.benchmark = True

    # Data
    print(colored('Get dataset and dataloaders', 'blue'))
    train_transformations = get_train_transformations(p)
    val_transformations = get_val_transformations(p)
    from data.cifar import CIFAR10
    dataset = CIFAR10(transform=val_transformations, download=True)
    val_dataset = get_val_dataset_henry(dataset, p['topk_neighbors_train_path'], to_neighbors_dataset=True)
    val_dataloader = get_val_dataloader(p, val_dataset)
    print('Train transforms:', train_transformations)
    print('Validation transforms:', val_transformations)
    model = get_model(p, p['pretext_model'])
    # print(model)
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    predictions = produce_clustering(model,
                                     PATH + r"results\cifar-10\scan-baseline\model.pth.tar",
                                     val_dataloader, p)
    init_prediction = predictions['predictions']
    print("ACC(SCAN):", acc(val_dataset.dataset.targets, init_prediction))
    print("NMI(SCAN):",
          normalized_mutual_info_score(val_dataset.dataset.targets, init_prediction, average_method="arithmetic"))
    accs = []
    nmis = []
    for test in range(0, 3):
        # Model
        # print(colored('Get model', 'blue'))
        model = get_model(p, p['pretext_model'])
        # print(model)
        model = torch.nn.DataParallel(model)
        model = model.cuda()
        from utils.constraint_utils import read_ml_cl, count_violated_cons
        ml_ind1, ml_ind2, cl_ind1, cl_ind2 = read_ml_cl(
            PATH + r"constraints\cifar-10-pw1000-0.5\test" + str(test).zfill(2))
        print("#Violated PW constraint on ground-truth:",
              count_violated_cons(val_dataset.dataset.targets, ml_ind1, ml_ind2, cl_ind1, cl_ind2))
        predictions = produce_clustering(model,
                                         PATH + r"results\cifar-10\scan-test" + str(test).zfill(
                                             2) + "-1000r-0.1\model.pth.tar",
                                         val_dataloader, p)
        accs.append(acc(val_dataset.dataset.targets, predictions['predictions']))
        nmis.append(normalized_mutual_info_score(val_dataset.dataset.targets, predictions['predictions'],
                                                 average_method="arithmetic"))
        print("ACC:", accs[-1])
        print("NMI:", nmis[-1])
        print("Initial #Violates:", count_violated_cons(init_prediction, ml_ind1, ml_ind2, cl_ind1, cl_ind2))
        print("#Violates",
              count_violated_cons(predictions['predictions'], ml_ind1, ml_ind2, cl_ind1, cl_ind2))
    show_value("acc", accs)
    show_value("nmi", nmis)


if __name__ == "__main__":
    main()
