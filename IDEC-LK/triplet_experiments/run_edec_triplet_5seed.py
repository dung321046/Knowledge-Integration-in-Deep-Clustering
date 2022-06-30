import datetime
import os
import random
import sys
import timeit

import numpy as np

sys.path.append("..")
from lib.triplet_cs_learner import TripletCS

from experiments.utils import read_triplet
from experiments.utils_setup import *
from lib.utils import count_violated_triplet_cons

if __name__ == "__main__":
    parser = default_parser('Triplet Constraints Program')
    parser.add_argument('--batch-csize', type=int, default=85,
                        help='input batch size for triplet constraints (default: 85)')
    parser.add_argument('--formu', type=str, default="A",
                        help='formulation for calculating constrained losses (default: A) [A,B]')
    parser.add_argument('--loss-method', type=str, default="batch-sdd",
                        help='method for calculating the loss (default: batch-sdd) [sdd, drct, batch-sdd]')
    parser.add_argument('--lambda-c', type=float, default=0.01,
                        help='the weight for pairwise losses (default: 0.01)')
    args = parser.parse_args()

    # formu = args.formu
    formu = "B"
    # "sdd", "batch-sdd", "drct"
    loss_type = args.loss_method
    # coefficient for constrained loss
    lambda_c = args.lambda_c
    setup_str = formu + "5seed-" + str(args.batch_csize) + "-" + loss_type + "-" + str(lambda_c)
    print("Training with:", setup_str)

    # Load data
    k, X, y, Xtest, ytest = load_data(args.data)
    input_dim = len(X[0])
    if "[data]" in args.pretrain:
        args.pretrain = args.pretrain.replace("[data]", str.lower(args.data))
    y = y.data.cpu().numpy()
    # Construct constraints
    folder_name = "../test_set/Triplet-" + args.data

    for triplet_num in [1000]:
        stat_pw = []
        for test in range(5):
            test_folder = folder_name + "/" + str(triplet_num) + "/test" + str(test).zfill(2)
            anchor, positive, negative = read_triplet(test_folder)
            print("#Violated triplet constraint on ground-truth:",
                  count_violated_triplet_cons(y, anchor, positive, negative))
            for seed in range(5):
                np.random.seed(seed)
                random.seed(seed)
                # Deep logic constrained clustering
                tripletcs = TripletCS(input_dim=input_dim, z_dim=10, n_clusters=k,
                                      encodeLayer=[500, 500, 2000], decodeLayer=[2000, 500, 500], activation="relu",
                                      dropout=0,
                                      lambda_c=lambda_c, formu=formu, loss_type=loss_type)
                tripletcs.load_model(args.pretrain)
                case_folder = test_folder + "/" + setup_str + "_" + str(seed)
                if os.path.exists(case_folder):
                    print("Setup has been run and save at:", case_folder)
                    # continue
                else:
                    os.makedirs(case_folder)
                stat = []
                start = timeit.default_timer()
                # Train Neural Network
                tripletcs.fit(stat, anchor, positive, negative, X, y, lr=args.lr,
                              batch_csize=args.batch_csize, num_epochs=args.epochs, tol=1e-4)
                training_time = (float)(timeit.default_timer() - start)

                final_stat = list(stat[-1])
                final_stat.append(training_time)
                final_stat.append(test)
                stat_pw.append(final_stat)
                save_tsv(case_folder + "/stat.tsv", stat)
                # pwcs.save_model(case_folder + "/model.pt")
        dt = datetime.datetime.now()
        # seq = str(int(dt.strftime("%d%H%M%S")))
        save_tsv(folder_name + "/" + str(triplet_num) + "/" + setup_str + ".tsv", stat_pw)
