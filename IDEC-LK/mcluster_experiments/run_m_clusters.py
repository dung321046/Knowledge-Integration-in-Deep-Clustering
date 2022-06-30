import sys

import torch

sys.path.append("..")
import numpy as np
from lib.neighbor_group_cs_learner import NeighborCS
from experiments.utils_setup import *
from lib.utils import permutation
from experiments.utils_setup import default_parser
import math


def find_clusters(label, permu, k):
    ans = []
    for l in label:
        for i in range(k):
            if permu[i] == l:
                ans.append(i)
    return ans


def freq_clusters(p):
    clusters = dict()
    for i in p:
        if i not in clusters:
            clusters[i] = 1
        else:
            clusters[i] += 1
    return dict(sorted(clusters.items(), key=lambda item: -item[1]))


def find_most_likely(probs):
    return np.argsort(-np.sum(probs, axis=0))


def convert(permu, y_pred):
    ans = []
    for y in y_pred:
        ans.append(permu[y])
    return ans


if __name__ == "__main__":
    parser = default_parser("m-cluster group constraints")
    args = parser.parse_args()
    # Change model to idec
    args.pretrain = "../model/idec_[data]_weights.pt"
    # Load data
    k, X, y, Xtest, ytest = load_data(args.data)
    if "[data]" in args.pretrain:
        args.pretrain = args.pretrain.replace("[data]", str.lower(args.data))
    y = y.data.cpu().numpy()
    # Make the code reproducible
    import random

    np.random.seed(1)
    random.seed(1)

    stat = []
    lambda_c = 0.1
    add_recon = False
    # Deep logic constrained clustering
    neighbor_cs = NeighborCS(input_dim=784, z_dim=10, n_clusters=10,
                             encodeLayer=[500, 500, 2000], decodeLayer=[2000, 500, 500], activation="relu", dropout=0,
                             lambda_c=lambda_c, add_recon=add_recon)
    neighbor_cs.load_model(args.pretrain)
    groups = []
    inp_prefix = "test_set/MNIST-100-0.75/"
    # inp_prefix = "test_set/MNIST-R-50-0.5/"
    pairs = np.loadtxt(inp_prefix + "pairs.txt", dtype=int)
    if len(pairs.shape) == 1:
        pairs = [pairs]
    fs = []
    fs_test = []
    sizes = []
    for pair in pairs:
        f = np.loadtxt(inp_prefix + str(pair[0]) + "-" + str(pair[1]) + ".txt", dtype=int)
        fs.append(f)
        size = math.floor(len(f) / 2)
        groups.append(f[:, 0][:size])
        sizes.append(size)
        ftest = np.loadtxt(inp_prefix + str(pair[0]) + "-" + str(pair[1]) + "-test.txt", dtype=int)
        fs_test.append(ftest)
    print("Training samples:", sizes)
    Z = neighbor_cs.encodeBatch(X)
    q = neighbor_cs.soft_assign(Z)
    y_pred = torch.argmax(q, dim=1).data.cpu().numpy()
    permu = permutation(y, y_pred)
    q = q.data.cpu().numpy()
    clusters_in_groups = []
    for gid, group in enumerate(groups):
        clusters_in_groups.append(freq_clusters(y_pred[group]))
        # print(find_clusters(pairs[gid], permu, k))
        # print("Using Q:", find_most_likely(q[group])[:2])
    import timeit

    time_training = timeit.default_timer()
    out_prefix = inp_prefix + "lambdaC-" + str(lambda_c) + "-epoch-" + str(args.epochs) + "-" + str(add_recon)
    import os

    if not os.path.exists(out_prefix):
        os.makedirs(out_prefix)
    file_prob_name = out_prefix + "/" + "prob.tsv"
    y_pred = neighbor_cs.fit(stat, file_prob_name, groups, clusters_in_groups, X, y, lr=args.lr,
                             num_epochs=args.epochs, tol=-1)

    np.savetxt(fname=out_prefix + "/training-sizes.txt", X=np.array(sizes), fmt="%d")
    for i, pair in enumerate(pairs):
        np.savetxt(fname=out_prefix + "/" + str(pair[0]) + "-" + str(pair[1]) + "-after.txt",
                   X=np.insert(fs[i], 3, convert(permu, y_pred[fs[i][:, 0]]), axis=1), fmt="%d")
    y_test_pred = neighbor_cs.predict(Xtest)
    for i, pair in enumerate(pairs):
        np.savetxt(fname=out_prefix + "/" + str(pair[0]) + "-" + str(pair[1]) + "-after-test.txt",
                   X=np.insert(fs_test[i], 3, convert(permu, y_test_pred[fs_test[i][:, 0]]), axis=1), fmt="%d")
    time_training = float(timeit.default_timer() - time_training)
    final_stat = list(stat[-1])
    final_stat.append(time_training)
    save_tsv(out_prefix + "/" + "stat.tsv", stat)
    # f neighbor_cs.save_model(out_prefix + "/" + "model.pt")
