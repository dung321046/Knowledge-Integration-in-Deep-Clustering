import sys

sys.path.append("../..")
import numpy as np
from sklearn.cluster import KMeans
import torch
from deep_clustering.utils_setup import *

from deep_clustering.utils import acc


def load_ae(name):
    args = define_args()
    args.data = name
    args.epochs = 40
    # Load model
    idec, X, y, k = load_model(args)
    np.set_printoptions(formatter={'float': '{:0.3f}'.format})
    y = y.data.cpu().numpy()
    kmeans = KMeans(k, n_init=20, random_state=0)
    data = idec.encodeBatch(X)
    y_pred = kmeans.fit_predict(data.data.cpu().numpy())
    idec.mu.data.copy_(torch.Tensor(kmeans.cluster_centers_))
    with torch.no_grad():
        _, q, __ = idec.forward(X)
    q = q.data.cpu().numpy()
    print("Acc:", acc(y_pred, y))
    return q, len(y), k, y


def load_ae_with_y_pred(name):
    args = define_args()
    if "[data]" in args.pretrain:
        args.pretrain = args.pretrain.replace("[data]", args.data)
    args.data = name
    args.epochs = 40
    # Load model
    idec, X, y, k = load_model(args)
    # np.set_printoptions(formatter={'float': '{:0.4f}'.format})
    y = y.data.cpu().numpy()
    with torch.no_grad():
        _, q, __ = idec.forward(X)
    q = q.data.cpu().numpy()
    y_pred = np.argmax(q, axis=1)
    print("Acc:", acc(y_pred, y))
    return q, len(y), k, y, y_pred
