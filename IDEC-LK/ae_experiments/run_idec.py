import sys

sys.path.append("..")
from experiments.utils_setup import *

import torch
import random
import numpy as np
import os
from lib.idec import IDEC


def draw(file, stat):
    import matplotlib.pyplot as plt

    x = []
    nmi = []
    acc = []
    # cluster_loss, recon_loss = [], []
    total_loss = []
    for i, s in enumerate(stat):
        x.append(i)
        nmi.append(s["nmi"])
        acc.append(s["acc"])
        # cluster_loss.append(s["cluster-loss"])
        # recon_loss.append(s["recon-loss"])
        total_loss.append(s["cluster-loss"] + s["recon-loss"])
    fig, ax = plt.subplots()
    plt.plot(x, nmi, label="NMI")
    plt.plot(x, acc, label="Acc")
    ax.set_ylabel("Clustering quality")
    ax2 = ax.twinx()
    # plt.plot(x, cluster_loss, label="Cluster loss")
    # plt.plot(x, recon_loss, label="Recon loss")
    plt.plot(x, total_loss, color='r', label="Loss")
    ax2.set_ylabel("Loss value")
    ax.legend(bbox_to_anchor=(1.2, 1), loc='upper left')
    ax2.legend(bbox_to_anchor=(1.2, 0.5), loc='upper left')
    plt.tight_layout()
    plt.savefig(file)


if __name__ == "__main__":
    parser = clustering_parser("Improved Deep Clustering")
    parser.add_argument('--lr', type=float, default=0.001, metavar='N',
                        help='learning rate for training (default: 0.001)')
    parser.add_argument('--pretrain', type=str, default="../model/sdae_[data]_weights.pt", metavar='N',
                        help='file path for pre-trained weights')
    parser.add_argument('--np', action="store_true", help='no-pretrain')
    args = parser.parse_args()
    if "[data]" in args.pretrain:
        args.pretrain = args.pretrain.replace("[data]", args.data)

    # Load data
    k, X, y, test_X, test_y = load_data(args.data)
    input_dim = len(X[0])
    for seed in range(5):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        idec = IDEC(input_dim=input_dim, z_dim=10, n_clusters=k,
                    encodeLayer=[500, 500, 2000], decodeLayer=[2000, 500, 500], activation="relu", dropout=0)
        if not args.np:
            idec.load_model(args.pretrain)
            use_km = True
        else:
            use_km = False
        # Construct constraints (here is the baseline so no constraints are provided).
        stat = []
        # Train the clustering model
        train_acc, train_nmi, epo = idec.fit(stat, X, y, lr=args.lr, batch_size=args.batch_size, num_epochs=args.epochs,
                                             use_kmeans=use_km, update_interval=1, tol=1 * 1e-3)

        # Test on the test data
        test_acc, test_nmi = idec.predict(test_X, test_y)
        folder_name = "./model/idec_" + args.data
        if not args.np:
            folder_name += "-sdae"
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)
        import json

        idec.save_model(folder_name + "/" + str(args.lr) + "_" + str(seed) + ".pt")
        with open(folder_name + "/" + str(args.lr) + "_" + str(seed) + ".json", "w") as stat_json:
            json.dump(stat, stat_json)
        draw(folder_name + "/" + str(args.lr) + "_" + str(seed) + ".png", stat)
        # Print the result
        print("Training Accuracy:", train_acc)
        print("Training NMI;", train_nmi)
        print("Training Epochs:", epo)
        print("Test Accuracy:", test_acc)
        print("Test NMI:", test_nmi)
