import sys

sys.path.append("..")
from experiments.utils_setup import *
import numpy as np
from lib.idec import IDEC
from lib.utils import permutation

import matplotlib.pyplot as plt


def plot_two_columns(column_names, a, b, group):
    labels = range(10)

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, a, width, label=column_names[0])
    rects2 = ax.bar(x + width / 2, b, width, label=column_names[1])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Frequency')
    ax.set_title('Cluster distribution of group ' + str(group) + ' in MNIST dataset')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()

    plt.show()


def display(num_row, num_col, tname, images, labels, y_pred, cluster_pair, path):
    fig, axes = plt.subplots(num_row, num_col, figsize=(1.2 * num_col, 1.4 * num_row))
    fig.canvas.set_window_title(tname)
    for i in range(min(len(images), num_row * num_col)):
        ax = axes[i // num_col, i % num_col]
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.imshow(images[i].data.cpu(), cmap=plt.cm.binary)
        if i < 50:
            ax.spines['bottom'].set_color('green')
            ax.spines['top'].set_color('green')
            ax.spines['left'].set_color('green')
            ax.spines['right'].set_color('green')
        if y_pred[i] not in cluster_pair:
            ax.set_title(labels[i], color='r', y=1.0, pad=-7)
        else:
            ax.set_title(labels[i], color='b', y=1.0, pad=-7)
    for i in range(len(images), num_row * num_col):
        ax = axes[i // num_col, i % num_col]
        fig.delaxes(ax)
    # plt.tight_layout()
    # plt.show()
    plt.savefig(path + tname + ".png", dpi=400)


def get_idx(y, y_pred, permu, pair, correct, incorrect, low, high, selected_idx):
    ans = []
    while correct + incorrect > 0:
        i = np.random.randint(low, high)
        if i not in selected_idx:
            if y[i] in pair:
                if y[i] == permu[y_pred[i]]:
                    if correct > 0:
                        ans.append(i)
                        correct -= 1
                        selected_idx.add(i)
                else:
                    if incorrect > 0:
                        ans.append(i)
                        incorrect -= 1
                        selected_idx.add(i)
    return ans


if __name__ == "__main__":
    parser = default_parser("Generation for group-2-cluster constraints")
    args = parser.parse_args()
    # Change model to idec
    args.pretrain = "../model/idec_[data]_weights.pt"
    # Load data
    k, X, y, Xtest, ytest = load_data(args.data)
    if "[data]" in args.pretrain:
        args.pretrain = args.pretrain.replace("[data]", args.data)
    y = y.data.cpu().numpy()
    ytest = ytest.data.cpu().numpy()
    if args.data == "MNIST":
        trusted_embedding = np.load("../model/mnist_triplet_embedding.npy")
    else:
        trusted_embedding = np.load("../model/fashion_triplet_embedding.npy")
    n = len(y)
    np.random.seed(1)
    if args.data == "Reuters":
        dlcc = IDEC(input_dim=2000, z_dim=10, n_clusters=4,
                    encodeLayer=[500, 500, 2000], decodeLayer=[2000, 500, 500], activation="relu", dropout=0)
    else:
        dlcc = IDEC(input_dim=784, z_dim=10, n_clusters=10,
                    encodeLayer=[500, 500, 2000], decodeLayer=[2000, 500, 500], activation="relu", dropout=0)
    dlcc.load_model(args.pretrain)
    Z = dlcc.encodeBatch(X)
    q = dlcc.soft_assign(Z)
    y_pred = np.argmax(q.data.cpu().numpy(), axis=1)
    Z = dlcc.encodeBatch(Xtest)
    q = dlcc.soft_assign(Z)
    y_test_pred = np.argmax(q.data.cpu().numpy(), axis=1)
    permu = permutation(y, y_pred)

    number_constraints = 100
    ratio = 0.75
    prefix = "test_set/" + args.data + "-" + str(number_constraints) + "-" + str(ratio) + "/"
    # prefix = "test_set/" + args.data + "-" + str(number_constraints) + "/"
    import os

    if not os.path.exists(prefix):
        os.makedirs(prefix)
    #pairs = [(3, 9), (6, 8), (0, 9), (2, 5)]
    pairs = [(3, 9), (6, 8), (1, 7), (2, 5)]
    groups = dict()
    selected_idx = set()
    for pair in pairs:
        correct = int(number_constraints * ratio)
        incorrect = int(number_constraints * (1 - ratio))
        groups[pair] = []
        groups[pair].extend(get_idx(y, y_pred, permu, pair, correct, incorrect, 0, n, selected_idx))
        groups[pair].extend(get_idx(y, y_pred, permu, pair, correct, incorrect, 0, n, selected_idx))
        # groups[pair].extend(get_idx(y, y_pred, permu, pair, selected_idx))

    extracted_pairs = []
    for p in pairs:
        extracted_pairs.append(p)
        images = []
        labels = []
        stat = []
        sub_y_pred = []
        for t in range(len(groups[p])):
            idx = groups[p][t]
            images.append(X[idx].reshape((28, 28)))
            sub_y_pred.append(permu[y_pred[idx]])
            stat.append([idx, y[idx], sub_y_pred[-1]])
            labels.append(str(y[idx]) + "-" + str(sub_y_pred[-1]))

        print(p, ": ", len(groups[p]))
        # plot_two_columns(["Ground-truth", "IDEC"], true_distributes[k], init_distributes[cluster_pair], cluster_pair)
        str_pair = str(p[0]) + "-" + str(p[1])
        display(10, 10, str_pair + "-all-images", images, labels, sub_y_pred, p, prefix)
        fname = str_pair + ".txt"
        np.savetxt(fname=prefix + fname, X=np.asarray(stat), fmt="%d")
        fname = str_pair + "-test.txt"
        stat = []
        for idx in range(len(ytest)):
            if ytest[idx] in p:
                stat.append([idx, ytest[idx], y_test_pred[idx]])
        np.savetxt(fname=prefix + fname, X=np.asarray(stat), fmt="%d")
    np.savetxt(fname=prefix + "pairs.txt", X=np.asarray(extracted_pairs), fmt="%d")
