import argparse
import csv

from lib.datasets import MNIST, FashionMNIST, Reuters


def data_parser(title="Select dataset"):
    parser = argparse.ArgumentParser(description=title)
    parser.add_argument('--data', type=str, default="MNIST", help='dataset(MNIST, Fashion, Reuters)')
    return parser


def clustering_parser(title="Autoencoder setup"):
    parser = data_parser(title=title)
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=500, metavar='N',
                        help='number of epochs to train (default: 500)')
    return parser


def default_parser(title='General setup'):
    parser = data_parser(title)
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate for training (default: 0.001)')
    parser.add_argument('--batch-size', type=int, default=256, help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train (default: 200)')
    parser.add_argument('--pretrain', type=str, default="../model/sdae_[data]_weights.pt", metavar='N',
                        help='file path for pre-trained weights')
    parser.add_argument('-np''--np', action="store_true", help='no-pretrain')
    return parser


def load_data(dataname):
    k = 10
    if dataname == "Fashion":
        data_train = FashionMNIST('../dataset/fashion_mnist', train=True, download=False)
        data_test = FashionMNIST('../dataset/fashion_mnist', train=False)
    elif dataname == "Reuters":
        data_train = Reuters('../dataset/reuters', train=True)
        data_test = Reuters('../dataset/reuters', train=False)
        k = 4
    else:
        data_train = MNIST('../dataset/mnist', train=True, download=False)
        data_test = MNIST('../dataset/mnist', train=False)
    X = data_train.train_data
    y = data_train.train_labels
    Xtest = data_test.test_data
    ytest = data_test.test_labels
    return k, X, y, Xtest, ytest


def load_dataset(dataname):
    k = 10
    if dataname == "Fashion":
        data_train = FashionMNIST('../dataset/fashion_mnist', train=True, download=True)
        data_test = FashionMNIST('../dataset/fashion_mnist', train=False)
    elif dataname == "Reuters":
        data_train = Reuters('../dataset/reuters', train=True)
        data_test = Reuters('../dataset/reuters', train=False)
        k = 4
    else:
        data_train = MNIST('../dataset/mnist', train=True, download=True)
        data_test = MNIST('../dataset/mnist', train=False)
    dim = len(data_train.train_data[0])
    return k, dim, data_train, data_test


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
