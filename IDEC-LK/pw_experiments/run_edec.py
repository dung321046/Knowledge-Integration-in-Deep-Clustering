import datetime
import os
import random
import sys
import timeit

import numpy as np

sys.path.append("..")

from lib.pairwise_cs_learner import PairwiseCS
from experiments.utils import read_ml_cl
from experiments.utils_setup import *
from lib.utils import transitive_closure, count_violated_cons

if __name__ == "__main__":
    parser = default_parser('Pairwise Constraints Program')
    parser.add_argument('--batch-csize', type=int, default=128,
                        help='input batch size for pw constraints (default: 128)')
    parser.add_argument('--formu', type=str, default="A",
                        help='formulation for calculating constrained losses (default: A) [A,B]')
    parser.add_argument('--loss-method', type=str, default="batch-sdd",
                        help='method for calculating the loss (default: batch-sdd) [sdd, drct, batch-sdd]')
    parser.add_argument('--lambda-c', type=float, default=0.01,
                        help='the weight for pairwise losses (default: 0.01)')
    args = parser.parse_args()

    formu = args.formu
    # "sdd", "batch-sdd", "drct"
    loss_type = args.loss_method
    # coefficient for constrained loss
    lambda_c = args.lambda_c
    setup_str = formu + "200-" + str(args.batch_csize) + "-" + loss_type + "-" + str(lambda_c)
    print("Training with:", setup_str)

    # Load data
    k, X, y, Xtest, ytest = load_data(args.data)
    input_dim = len(X[0])
    if "[data]" in args.pretrain:
        args.pretrain = args.pretrain.replace("[data]", str.lower(args.data))
    y = y.data.cpu().numpy()
    # Construct constraints

    np.random.seed(1)
    random.seed(1)
    folder_name = "../test_set/Pw-" + args.data

    for pairwise_num in [10, 100, 500, 1000]:
        stat_pw = []
        for test in range(5):
            # Deep logic constrained clustering
            pwcs = PairwiseCS(input_dim=input_dim, z_dim=10, n_clusters=k,
                              encodeLayer=[500, 500, 2000], decodeLayer=[2000, 500, 500], activation="relu", dropout=0,
                              lambda_c=lambda_c, formu=formu, loss_type=loss_type)
            pwcs.load_model(args.pretrain)
            test_folder = folder_name + "/" + str(pairwise_num) + "/test" + str(test).zfill(2)

            case_folder = test_folder + "/" + setup_str
            if os.path.exists(case_folder):
                print("Setup has been run and save at:", case_folder)
                continue
            else:
                os.makedirs(case_folder)
            ml_ind1, ml_ind2, cl_ind1, cl_ind2 = read_ml_cl(test_folder)
            print("#Violated PW constraint on ground-truth:",
                  count_violated_cons(y, ml_ind1, ml_ind2, cl_ind1, cl_ind2))
            ml_ind1, ml_ind2, cl_ind1, cl_ind2 = transitive_closure(ml_ind1, ml_ind2, cl_ind1, cl_ind2, X.shape[0])

            stat = []

            start = timeit.default_timer()
            # Train Neural Network
            pwcs.fit(stat, ml_ind1, ml_ind2, cl_ind1, cl_ind2, X, y, lr=args.lr,
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
        save_tsv(folder_name + "/" + str(pairwise_num) + "/" + setup_str + ".tsv", stat_pw)
