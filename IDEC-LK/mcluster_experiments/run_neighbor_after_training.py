import sys

sys.path.append("..")
from experiments.utils_setup import *
import numpy as np
from lib.idec import IDEC
from lib.utils import permutation

import matplotlib.pyplot as plt

num_improv = 0
num_unimpr = 0


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


def display(num_row, num_col, tname, images, labels, y_pred, y_aftr, cluster_pair, training_size, path):
    fig, axes = plt.subplots(num_row, num_col, figsize=(1.1 * num_col, 1.5 * num_row))
    fig.canvas.set_window_title(tname)
    for i in range(min(len(images), num_row * num_col)):
        ax = axes[i // num_col, i % num_col]
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.imshow(images[i].data.cpu(), cmap=plt.cm.binary)
        if i < training_size:
            ax.spines['bottom'].set_color('green')
            ax.spines['top'].set_color('green')
            ax.spines['left'].set_color('green')
            ax.spines['right'].set_color('green')
        if y_pred[i] not in cluster_pair:
            if y_aftr[i] not in cluster_pair:
                ax.set_title(labels[i], color='m', y=1.0, pad=-7)
                if i >= training_size:
                    global num_unimpr
                    num_unimpr += 1
            else:
                ax.set_title(labels[i], color='g', y=1.0, pad=-7)
                if i >= training_size:
                    global num_improv
                    num_improv += 1
        else:
            if y_aftr[i] in cluster_pair:
                ax.set_title(labels[i], y=1.0, pad=-7)
            else:
                ax.set_title(labels[i], color='r', y=1.0, pad=-7)
    for i in range(len(images), num_row * num_col):
        ax = axes[i // num_col, i % num_col]
        fig.delaxes(ax)
    # plt.tight_layout()
    # plt.show()
    plt.savefig(path + tname + ".png")


if __name__ == "__main__":
    parser = default_parser()
    args = parser.parse_args()
    # Load data
    k, X, y, Xtest, ytest = load_data(args.data)
    if "[data]" in args.pretrain:
        args.pretrain = args.pretrain.replace("[data]", str.lower(args.data))
    y = y.data.cpu().numpy()
    n = len(y)
    print("-", len(ytest))
    formu = "B"
    lambda_c = "0.001"
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
    permu = permutation(y, y_pred)
    l = 0
    # pairs = [[3, 9], [7, 9], [2, 7]]
    # pairs = [[2, 3], [3, 9], [7, 9]]
    # prefix = "test_set/neighbor-k10/"
    # prefix = "test_set/neighbor0.001-50/"
    config = "lambdaC-0.1-epoch-200-False/"
    # config = "lambdaC-0.2-epoch-50-True/"
    inp_prefix = "test_set/MNIST-100-0.75/"
    # inp_prefix = "test_set/MNIST-R-50-0.5/"
    out_prefix = inp_prefix + config
    pairs = np.loadtxt(inp_prefix + "pairs.txt", dtype=int)
    if len(pairs.shape) == 1:
        pairs = [pairs]
    sizes = np.loadtxt(out_prefix + "training-sizes.txt", dtype=int)
    if len(sizes.shape) == 0:
        sizes = [sizes]
    num_sat = 0
    num_unsat = 0
    num_unsat_before = 0.0
    x_arr = []
    column_sat = []
    column_unsat = []
    for i, pair in enumerate(pairs):
        stat = np.loadtxt(out_prefix + str(pair[0]) + "-" + str(pair[1]) + "-after.txt",
                          dtype=int)
        images = []
        sub_y_pred = []
        sub_y_aftr = []
        labels = []
        for t in stat:
            images.append(X[t[0]].reshape((28, 28)))
            sub_y_pred.append(t[2])
            sub_y_aftr.append(t[3])
            labels.append(str(t[1]) + "-" + str(t[2]) + "-" + str(t[3]))
        # if k in labels:
        #     plot_two_columns(["Ground-truth", "IDEC"], true_distributes[k], init_distributes[k], k)
        str_pair = str(pair[0]) + "-" + str(pair[1])
        display(10, 10, str_pair + "-after-training", images, labels, sub_y_pred, sub_y_aftr, pair, sizes[i],
                out_prefix)
        # fname = str(k[0]) + "-" + str(k[1]) + "-MNIST.txt"
        # np.savetxt(fname=fname, X=np.asarray(groups[k]), fmt="%d")
        stat2 = np.loadtxt(out_prefix + str(pair[0]) + "-" + str(pair[1]) + "-after-test.txt",
                           dtype=int)
        num_no_change = 0
        num_good = 0
        num_worse = 0
        num_same = 0
        for t in stat2:
            if t[2] not in pair:
                if t[3] not in pair:
                    num_no_change += 1
                else:
                    num_good += 1
            else:
                if t[3] not in pair:
                    num_worse += 1
                else:
                    num_same += 1
            if t[3] not in pair:
                num_unsat += 1
            else:
                num_sat += 1
            if t[2] not in pair:
                num_unsat_before += 1
        print("Test:", num_worse, num_good, num_no_change, num_same)
        x_arr.append(str(pair[0]) + "-" + str(pair[1]) + "-before")
        x_arr.append(str(pair[0]) + "-" + str(pair[1]) + "-after")
        column_sat.extend([num_same + num_worse, num_good + num_same])
        column_unsat.extend([num_no_change + num_good, num_no_change + num_worse])
    fig, ax = plt.subplots()
    plt.bar(x_arr, column_sat, label="#Satified")
    plt.bar(x_arr, column_unsat, label="#Unsatified", bottom=column_sat)
    plt.legend(bbox_to_anchor=(0.25, -0.28), loc='upper left')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("tmp.png")

    print(num_unsat, num_sat)
    print(num_improv)
    # print(num_improv / (num_unimpr + num_improv))
    print(1 - num_unsat / num_unsat_before)
