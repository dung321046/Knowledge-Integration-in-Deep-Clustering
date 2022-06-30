import sys

sys.path.append("..")
from experiments.utils_setup import *
import numpy as np
from lib.complex_cs_learner import ComplexCS
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
    fig, axes = plt.subplots(num_row, num_col, figsize=(1.2 * num_col, 2.5 * num_row))
    fig.canvas.set_window_title(tname)
    for i in range(min(len(images), num_row * num_col)):
        ax = axes[i // num_col, i % num_col]
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.imshow(images[i], cmap=plt.cm.binary)
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
    plt.savefig(path + tname + ".png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pairwise Constraints Program')
    parser.add_argument('--lr', type=float, default=0.001, metavar='N',
                        help='learning rate for training (default: 0.001)')
    parser.add_argument('--batch-constraint-size', type=int, default=128, metavar='N',
                        help='input batch size for constrained learning (default: 128)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--pretrain', type=str, default="./model/idec_mnist.pt", metavar='N',
                        help='directory for pre-trained weights')
    parser.add_argument('--data', type=str, default="MNIST", metavar='N', help='dataset(MNIST, Fashion, Reuters)')
    parser.add_argument('--use_pretrain', type=bool, default=True)
    args = parser.parse_args()

    # Load data
    mnist_train = MNIST('./dataset/mnist', train=True, download=True)
    mnist_test = MNIST('./dataset/mnist', train=False)
    X = mnist_train.train_data
    y = mnist_train.train_labels
    test_X = mnist_test.test_data
    test_y = mnist_test.test_labels
    trusted_embedding = np.load("../model/mnist_triplet_embedding.npy")
    k = 10
    # Set parameters
    if args.data == "Fashion":
        fashionmnist_train = FashionMNIST('./dataset/fashion_mnist', train=True, download=True)
        fashionmnist_test = FashionMNIST('./dataset/fashion_mnist', train=False)
        X = fashionmnist_train.train_data
        y = fashionmnist_train.train_labels
        test_X = fashionmnist_test.test_data
        test_y = fashionmnist_test.test_labels
        args.pretrain = "./model/idec_fashion.pt"
        trusted_embedding = np.load("../model/fashion_triplet_embedding.npy")
    elif args.data == "Reuters":
        reuters_train = Reuters('./dataset/reuters', train=True, download=False)
        reuters_test = Reuters('./dataset/reuters', train=False)
        X = reuters_train.train_data
        y = reuters_train.train_labels
        test_X = reuters_test.test_data
        test_y = reuters_test.test_labels
        args.pretrain = "./model/idec_reuters.pt"
        k = 4
    # Print Network Structure
    # print(idec)
    y = y.data.cpu().numpy()
    n = len(y)
    formu = "B"
    lambda_c = "0.001"
    if args.data == "Reuters":
        dlcc = ComplexCS(input_dim=2000, z_dim=10, n_clusters=4,
                         encodeLayer=[500, 500, 2000], decodeLayer=[2000, 500, 500], activation="relu", dropout=0,
                         lambda_c=lambda_c, formu=formu)
    else:
        dlcc = ComplexCS(input_dim=784, z_dim=10, n_clusters=10,
                         encodeLayer=[500, 500, 2000], decodeLayer=[2000, 500, 500], activation="relu", dropout=0,
                         lambda_c=lambda_c, formu=formu)
    dlcc.load_model(args.pretrain)
    Z = dlcc.encodeBatch(X)
    q = dlcc.soft_assign(Z)
    y_pred = np.argmax(q.data.cpu().numpy(), axis=1)
    permu = permutation(y, y_pred)
    from sklearn.neighbors import NearestNeighbors

    neighbor_size = 10
    min_freq = 1
    prefix = "test_set/" + args.data + "-" + str(neighbor_size) + "-" + str(min_freq) + "/"
    import os

    if not os.path.exists(prefix):
        os.makedirs(prefix)
    # print(trusted_embedding.shape)
    nbrs = NearestNeighbors(n_neighbors=neighbor_size, algorithm='ball_tree').fit(trusted_embedding)
    distances, indices = nbrs.kneighbors(trusted_embedding)
    from collections import Counter

    freqs = dict()
    error_freqs = Counter()
    groups = dict()
    true_distributes = dict()
    init_distributes = dict()
    for i in range(n):
        c = Counter()
        c2 = Counter()
        for j in indices[i]:
            c[y[j]] += 1
            c2[permu[y_pred[j]][1]] += 1
        freq = c.most_common()
        if freq[0][1] < neighbor_size:
            for pair in freq:
                if pair[0] != y[i] and pair[1] > min_freq:
                    group_id = (min(pair[0], y[i]), max(pair[0], y[i]))
                    if permu[y_pred[i]][1] not in group_id:
                        error_freqs[group_id] += 1
                    if group_id not in freqs:
                        groups[group_id] = [i]
                        freqs[group_id] = 1
                        true_distributes[group_id] = [0] * k
                        true_distributes[group_id][y[i]] = 1
                        init_distributes[group_id] = [0] * k
                        init_distributes[group_id][permu[y_pred[i]][1]] = 1
                    else:
                        groups[group_id].append(i)
                        freqs[group_id] += 1
                        true_distributes[group_id][y[i]] += 1
                        init_distributes[group_id][permu[y_pred[i]][1]] += 1
            # print(c.most_common(), " - ", c2.most_common())
            # if freq[0][0] != y[i]:
            #     print("___________", y[i])
    # ordered_dict = dict(sorted(freqs.items(), key=lambda item: -item[1]))
    # print("Size:", ordered_dict)
    print("Error:", error_freqs.most_common())
    error_pers = dict()
    for a in error_freqs:
        error_pers[a] = error_freqs[a] * 1.0 / freqs[a]
    ordered_error_per = sorted(error_pers.items(), key=lambda item: -item[1])
    print("Group by Error Percentages:", ordered_error_per)
    l = 0
    extracted_pairs = []
    for a in ordered_error_per:
        p = a[0]
        l += 1
        if l > 10:
            break
        extracted_pairs.append(p)
        images = []
        labels = []
        stat = []
        sub_y_pred = []
        for t in range(len(groups[p])):
            idx = groups[p][t]
            images.append(X[idx].reshape((28, 28)))
            sub_y_pred.append(permu[y_pred[idx]][1])
            stat.append([idx, y[idx], sub_y_pred[-1]])
            labels.append(str(y[idx]) + "-" + str(sub_y_pred[-1]))

        print(p, ": ", len(groups[p]), " - Error:", error_freqs[p])
        # plot_two_columns(["Ground-truth", "IDEC"], true_distributes[k], init_distributes[cluster_pair], cluster_pair)
        str_pair = str(p[0]) + "-" + str(p[1])
        display(10, 10, str_pair + "-all-images", images, labels, sub_y_pred, p, prefix)
        fname = str_pair + ".txt"
        np.savetxt(fname=prefix + fname, X=np.asarray(stat), fmt="%d")
    np.savetxt(fname=prefix + "pairs.txt", X=np.asarray(extracted_pairs), fmt="%d")
