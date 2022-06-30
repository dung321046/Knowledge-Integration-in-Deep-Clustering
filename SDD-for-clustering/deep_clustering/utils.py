'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
    
'''
import random

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from scipy.linalg import norm
from sklearn.metrics import adjusted_rand_score

ari = adjusted_rand_score

random.seed(2)


def weights_xavier_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform(m.weight.data)
        nn.init.constant(m.bias.data, 0)


def buildNetwork(layers, activation="relu", dropout=0):
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i - 1], layers[i]))
        if activation == "relu":
            net.append(nn.ReLU())
        elif activation == "sigmoid":
            net.append(nn.Sigmoid())
        if dropout > 0:
            net.append(nn.Dropout(dropout))
    return nn.Sequential(*net)


class Dataset(data.Dataset):
    def __init__(self, data, labels, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.data = data
        self.labels = labels
        if torch.cuda.is_available():
            self.data = self.data.cuda()
            self.labels = self.labels.cuda()

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        # img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


def masking_noise(data, frac):
    """
    data: Tensor
    frac: fraction of unit to be masked out
    """
    data_noise = data.clone()
    rand = torch.rand(data.size())
    data_noise[rand < frac] = 0
    return data_noise


def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return w[row_ind, col_ind].sum() * 1.0 / y_pred.size


def group_acc(groups, total_cases, y_pred):
    l = 0.0
    num_cor = 0.0
    for i, group in enumerate(groups):
        sub_y = [y_pred[idx] for idx in group]
        l += len(sub_y)
        num_cor += acc(np.asarray(sub_y), np.asarray(total_cases[i])) * len(sub_y)
    return num_cor / l


def permutation(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return ind


def detect_wrong(y_true, y_pred):
    """
    Simulating instance difficulty constraints. Require scikit-learn installed
    
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        A mask vector M =  1xn which indicates the difficulty degree
        We treat k-means as weak learner and set low confidence (0.1) for incorrect instances.
        Set high confidence (1) for correct instances.
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    mapping_dict = {}
    for pair in ind:
        mapping_dict[pair[0]] = pair[1]
    wrong_preds = []
    for i in range(y_pred.size):
        if mapping_dict[y_pred[i]] != y_true[i]:
            wrong_preds.append(-0.1)  # low confidence -0.1 set for k-means weak learner
        else:
            wrong_preds.append(1)
    return np.array(wrong_preds)


def transitive_closure(ml_ind1, ml_ind2, cl_ind1, cl_ind2, n):
    """
    This function calculate the total transtive closure for must-links and the full entailment
    for cannot-links. 
    
    # Arguments
        ml_ind1, ml_ind2 = instances within a pair of must-link constraints
        cl_ind1, cl_ind2 = instances within a pair of cannot-link constraints
        n = total training instance number

    # Return
        transtive closure (must-links)
        entailment of cannot-links
    """
    ml_graph = dict()
    cl_graph = dict()
    for i in range(n):
        ml_graph[i] = set()
        cl_graph[i] = set()

    def add_both(d, i, j):
        d[i].add(j)
        d[j].add(i)

    for (i, j) in zip(ml_ind1, ml_ind2):
        add_both(ml_graph, i, j)

    def dfs(i, graph, visited, component):
        visited[i] = True
        for j in graph[i]:
            if not visited[j]:
                dfs(j, graph, visited, component)
        component.append(i)

    visited = [False] * n
    for i in range(n):
        if not visited[i]:
            component = []
            dfs(i, ml_graph, visited, component)
            for x1 in component:
                for x2 in component:
                    if x1 != x2:
                        ml_graph[x1].add(x2)
    for (i, j) in zip(cl_ind1, cl_ind2):
        add_both(cl_graph, i, j)
        for y in ml_graph[j]:
            add_both(cl_graph, i, y)
        for x in ml_graph[i]:
            add_both(cl_graph, x, j)
            for y in ml_graph[j]:
                add_both(cl_graph, x, y)
    ml_res_set = set()
    cl_res_set = set()
    for i in ml_graph:
        for j in ml_graph[i]:
            if j != i and j in cl_graph[i]:
                raise Exception('inconsistent constraints between %d and %d' % (i, j))
            if i <= j:
                ml_res_set.add((i, j))
            else:
                ml_res_set.add((j, i))
    for i in cl_graph:
        for j in cl_graph[i]:
            if i <= j:
                cl_res_set.add((i, j))
            else:
                cl_res_set.add((j, i))
    ml_res1, ml_res2 = [], []
    cl_res1, cl_res2 = [], []
    for (x, y) in ml_res_set:
        ml_res1.append(x)
        ml_res2.append(y)
    for (x, y) in cl_res_set:
        cl_res1.append(x)
        cl_res2.append(y)
    return np.array(ml_res1), np.array(ml_res2), np.array(cl_res1), np.array(cl_res2)


def generate_random_pair(y, num):
    """
    Generate random pairwise constraints.
    """
    ml_ind1, ml_ind2 = [], []
    cl_ind1, cl_ind2 = [], []
    while num > 0:
        tmp1 = random.randint(0, y.shape[0] - 1)
        tmp2 = random.randint(0, y.shape[0] - 1)
        if tmp1 == tmp2:
            continue
        if y[tmp1] == y[tmp2]:
            ml_ind1.append(tmp1)
            ml_ind2.append(tmp2)
        else:
            cl_ind1.append(tmp1)
            cl_ind2.append(tmp2)
        num -= 1
    return np.array(ml_ind1), np.array(ml_ind2), np.array(cl_ind1), np.array(cl_ind2)


def generate_mnist_triplets(y, num):
    """
    Generate random triplet constraints
    """
    # To download the trusted_embedding for mnist data, run the script download_model.sh
    # Or you can create your own truseted embedding by running our pairwise constraints model
    # with 100000 randomly generated constraints.
    mnist_embedding = np.load("../model/mnist_triplet_embedding.npy")
    anchor_inds, pos_inds, neg_inds = [], [], []
    while num > 0:
        tmp_anchor_index = random.randint(0, y.shape[0] - 1)
        tmp_pos_index = random.randint(0, y.shape[0] - 1)
        tmp_neg_index = random.randint(0, y.shape[0] - 1)
        pos_distance = norm(mnist_embedding[tmp_anchor_index] - mnist_embedding[tmp_pos_index], 2)
        neg_distance = norm(mnist_embedding[tmp_anchor_index] - mnist_embedding[tmp_neg_index], 2)
        # 35 is selected by grid search which produce human trusted positive/negative pairs
        if neg_distance <= pos_distance + 35:
            continue
        anchor_inds.append(tmp_anchor_index)
        pos_inds.append(tmp_pos_index)
        neg_inds.append(tmp_neg_index)
        num -= 1
    return np.array(anchor_inds), np.array(pos_inds), np.array(neg_inds)


def generate_sensitive_triplets(y, num, r=0.5, models="MNIST"):
    """
    Generate random triplet constraints
    """
    # To download the trusted_embedding for mnist data, run the script download_model.sh
    # Or you can create your own truseted embedding by running our pairwise constraints model
    # with 100000 randomly generated constraints.
    if models == "MNIST":
        trusted_embedding = np.load("../model/mnist_triplet_embedding.npy")
    else:
        trusted_embedding = np.load("../model/fashion_triplet_embedding.npy")
    anchor_inds, pos_inds, neg_inds = [], [], []
    # num1 is the triplet within a cluster
    num1 = int(num * r)
    num2 = num - num1
    while num1 + num2 > 0:
        tmp_anchor_index = random.randint(0, y.shape[0] - 1)
        tmp_pos_index = random.randint(0, y.shape[0] - 1)
        tmp_neg_index = random.randint(0, y.shape[0] - 1)
        pos_distance = norm(trusted_embedding[tmp_anchor_index] - trusted_embedding[tmp_pos_index], 2)
        neg_distance = norm(trusted_embedding[tmp_anchor_index] - trusted_embedding[tmp_neg_index], 2)
        # 35 is selected by grid search which produce human trusted positive/negative pairs
        if neg_distance < pos_distance:
            tmp_pos_index, tmp_neg_index = tmp_neg_index, tmp_pos_index
        if y[tmp_anchor_index] == y[tmp_pos_index] and y[tmp_anchor_index] == y[tmp_neg_index]:
            if num1 > 0:
                anchor_inds.append(tmp_anchor_index)
                pos_inds.append(tmp_pos_index)
                neg_inds.append(tmp_neg_index)
                num1 -= 1
        elif y[tmp_anchor_index] == y[tmp_pos_index] or y[tmp_anchor_index] != y[tmp_neg_index]:
            if num2 > 0:
                anchor_inds.append(tmp_anchor_index)
                pos_inds.append(tmp_pos_index)
                neg_inds.append(tmp_neg_index)
                num2 -= 1
    return np.array(anchor_inds), np.array(pos_inds), np.array(neg_inds)


def generate_triplet_constraints_continuous(y, num):
    """
    Generate random triplet constraints
    """
    # To download the trusted_embedding for mnist data, run the script download_model.sh
    # Or you can create your own truseted embedding by running our pairwise constraints model
    # with 100000 randomly generated constraints.
    fashion_embedding = np.load("../model/fashion_triplet_embedding.npy")
    anchor_inds, pos_inds, neg_inds = [], [], []
    while num > 0:
        tmp_anchor_index = random.randint(0, y.shape[0] - 1)
        tmp_pos_index = random.randint(0, y.shape[0] - 1)
        tmp_neg_index = random.randint(0, y.shape[0] - 1)
        pos_distance = norm(fashion_embedding[tmp_anchor_index] - fashion_embedding[tmp_pos_index], 2)
        neg_distance = norm(fashion_embedding[tmp_anchor_index] - fashion_embedding[tmp_neg_index], 2)
        # 80 is selected by grid search which produce human trusted positive/negative pairs
        if neg_distance <= pos_distance + 80:
            continue
        anchor_inds.append(tmp_anchor_index)
        pos_inds.append(tmp_pos_index)
        neg_inds.append(tmp_neg_index)
        num -= 1
    return np.array(anchor_inds), np.array(pos_inds), np.array(neg_inds)


def generate_random_triplets(y, num, ratio=0.5):
    anchor_inds, pos_inds, neg_inds = [], [], []
    num1 = int(num * ratio)
    while num1 > 0:
        tmp_anchor_index = random.randint(0, y.shape[0] - 1)
        tmp_pos_index = random.randint(0, y.shape[0] - 1)
        tmp_neg_index = random.randint(0, y.shape[0] - 1)
        if tmp_anchor_index != tmp_pos_index and tmp_pos_index != tmp_neg_index and tmp_neg_index != tmp_anchor_index:
            if y[tmp_anchor_index] == y[tmp_pos_index] and y[tmp_anchor_index] != y[tmp_neg_index]:
                # if y[tmp_anchor_index] == y[tmp_pos_index]:
                anchor_inds.append(tmp_anchor_index)
                pos_inds.append(tmp_pos_index)
                neg_inds.append(tmp_neg_index)
                num1 -= 1
    num2 = int(num * (1 - ratio))
    while num2 > 0:
        tmp_anchor_index = random.randint(0, y.shape[0] - 1)
        tmp_pos_index = random.randint(0, y.shape[0] - 1)
        tmp_neg_index = random.randint(0, y.shape[0] - 1)
        if y[tmp_anchor_index] != y[tmp_pos_index] and y[tmp_pos_index] != y[tmp_neg_index] \
                and y[tmp_neg_index] != y[tmp_anchor_index]:
            anchor_inds.append(tmp_anchor_index)
            pos_inds.append(tmp_pos_index)
            neg_inds.append(tmp_neg_index)
            num2 -= 1
    return np.array(anchor_inds), np.array(pos_inds), np.array(neg_inds)


def generate_random_pairwise_graph(y, num, ratio=0.5):
    n = y.shape[0]
    indexes = np.random.permutation(n)[:num]
    pw_graph = np.full((num, num), 0)
    for i in range(num):
        for j in range(i):
            if np.random.random() < ratio:
                if y[indexes[i]] == y[indexes[j]]:
                    print("ML:", i, j)
                    pw_graph[i][j] = 1
                    pw_graph[j][i] = 1
                else:
                    print("CL:", i, j)
                    pw_graph[i][j] = -1
                    pw_graph[j][i] = -1
    return indexes, pw_graph


def count_violated_cons(y, ml_ind1, ml_ind2, cl_ind1, cl_ind2):
    num_vcon = 0
    for i in range(len(ml_ind1)):
        if y[ml_ind1[i]] != y[ml_ind2[i]]:
            num_vcon += 1
    for i in range(len(cl_ind1)):
        if y[cl_ind1[i]] == y[cl_ind2[i]]:
            num_vcon += 1
    return num_vcon


def count_violated_neighbor_cons(y, ids, clusters):
    num_vcon = 0
    for i in ids:
        if y[i] not in clusters:
            num_vcon += 1
    return num_vcon


def check_pairwise_conjunction(ml, cl, y):
    for pair in ml:
        if y[pair[0]] != y[pair[1]]:
            return False
    for pair in cl:
        if y[pair[0]] == y[pair[1]]:
            return False
    return True


def is_satisfy_complex_constraint(y, case):
    mlp, clp, mlq, clq = case[0], case[1], case[2], case[3]
    p = check_pairwise_conjunction(mlp, clp, y)
    q = check_pairwise_conjunction(mlq, clq, y)
    if p and not q:
        return False
    return True
