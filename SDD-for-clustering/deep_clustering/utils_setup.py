import argparse

from deep_clustering.datasets import MNIST, FashionMNIST, Reuters
from deep_clustering.dcc import DCC


def define_args():
    parser = argparse.ArgumentParser(description='Triplet Constraints Example')
    parser.add_argument('--lr', type=float, default=0.001, metavar='N',
                        help='learning rate for training (default: 0.001)')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--pretrain', type=str, default="../model/idec_[data]_weights.pt", metavar='N',
                        help='directory for pre-trained weights')
    parser.add_argument('--data', type=str, default="MNIST", metavar='N', help='dataset(MNIST, Fashion)')
    parser.add_argument('--use_pretrain', type=bool, default=True)
    return parser.parse_args()


def load_model(args):
    dim = 784
    k = 10
    if args.data == "Fashion":
        fashionmnist_train = FashionMNIST('../dataset/fashion_mnist', train=True, download=True)
        X = fashionmnist_train.train_data
        y = fashionmnist_train.train_labels
        args.pretrain = "../model/idec_fashion.pt"
    elif args.data == "Reuters":
        dim = 2000
        k = 4
        reuters_train = Reuters('../dataset/reuters', train=True, download=False)
        X = reuters_train.train_data
        y = reuters_train.train_labels
        args.pretrain = "../model/idec_reuters.pt"
    else:
        mnist_train = MNIST('../dataset/mnist', train=True, download=True)
        X = mnist_train.train_data
        y = mnist_train.train_labels
    idec = DCC(input_dim=dim, z_dim=10, n_clusters=k,
               encodeLayer=[500, 500, 2000], decodeLayer=[2000, 500, 500], activation="relu", dropout=0)
    if args.use_pretrain:
        idec.load_model(args.pretrain)
    return idec, X, y, k


import csv


def save_training_log(filename, stat):
    with open(filename, "w") as file:
        writer = csv.writer(file, delimiter='\t')
        for row in stat:
            writer.writerow(['{:0.4f}'.format(row[0]), '{:0.4f}'.format(row[1]), '{:0.4f}'.format(row[2]),
                             '{:0.4f}'.format(row[3]), '{:0.4f}'.format(row[4])])


def save_tsv(filename, table):
    with open(filename, "w") as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerows(table)
