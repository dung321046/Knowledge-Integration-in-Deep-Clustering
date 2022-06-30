import numpy as np
import scipy.sparse as sp
from pysdd.sdd import SddManager, Vtree
from torch.utils.data import Dataset

from sdd_clustering.convert_to_T import *

prj_path = "/home/henry/codes/SDCN/"

import torch


# prj_path = ""

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def load_graph(dataset, k):
    if k:
        path = prj_path + 'graph/{}{}_graph.txt'.format(dataset, k)
    else:
        path = prj_path + 'graph/{}_graph.txt'.format(dataset)
    if dataset == 'reut':
        data = np.load(prj_path + 'data/{}.npy'.format(dataset))
    else:
        data = np.loadtxt(prj_path + 'data/{}.txt'.format(dataset))
    n, _ = data.shape
    idx = np.array([i for i in range(n)], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(path, dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(n, n), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj


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


class load_data(Dataset):
    def __init__(self, dataset):
        if dataset == 'reut':
            self.x = np.load(prj_path + 'data/{}.npy'.format(dataset))
            self.y = np.load(prj_path + 'data/{}_label.npy'.format(dataset))
        else:
            self.x = np.loadtxt(prj_path + 'data/{}.txt'.format(dataset), dtype=float)
            self.y = np.loadtxt(prj_path + 'data/{}_label.txt'.format(dataset), dtype=int)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])), \
               torch.from_numpy(np.array(self.y[idx])), \
               torch.from_numpy(np.array(idx))



def load_sdd_complex(path):
    vtree = Vtree.from_file(bytes(path + ".vtree", encoding="utf8"))
    mgr = SddManager.from_vtree(vtree)
    root = mgr.read_sdd_file(bytes(path + ".sdd", encoding="utf8"))
    return mgr, root
