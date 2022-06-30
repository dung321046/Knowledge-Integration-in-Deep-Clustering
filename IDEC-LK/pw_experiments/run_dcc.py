import os
import sys

sys.path.append("..")
import torch.utils.data
import numpy as np
import timeit
from lib.dcc import IDEC
from experiments.utils import read_ml_cl
from experiments.utils_setup import *
from lib.utils import transitive_closure, count_violated_cons
import random

if __name__ == "__main__":
    parser = default_parser('Pairwise Constraints Program with DCC')
    parser.add_argument('--update-interval', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 1)')
    parser.add_argument('--without_pretrain', action='store_false')
    parser.add_argument('--without_kmeans', action='store_false')
    parser.add_argument('--noisy', type=float, default=0.0, metavar='N',
                        help='noisy constraints rate for training (default: 0.0)')
    parser.add_argument('--plotting', action='store_true')
    args = parser.parse_args()

    # Load data
    k, X, y, Xtest, ytest = load_data(args.data)
    input_dim = len(X[0])
    if "[data]" in args.pretrain:
        args.pretrain = args.pretrain.replace("[data]", str.lower(args.data))
    # Set parameters
    ml_penalty, cl_penalty = 0.1, 1
    np.random.seed(1)
    random.seed(1)
    folder_name = "../test_set/Pw-" + args.data
    setup_str = "DCC200"
    if args.without_pretrain:
        setup_str += "-pt"
    else:
        setup_str += "-np"

    for pairwise_num in [10, 100, 500, 1000]:
        stat_pw = []
        for test in range(5):
            idec = IDEC(input_dim=input_dim, z_dim=10, n_clusters=k,
                        encodeLayer=[500, 500, 2000], decodeLayer=[2000, 500, 500], activation="relu", dropout=0)
            model_tag = "Raw"
            if args.without_pretrain:
                model_tag = "Pretrain"
                idec.load_model(args.pretrain)

            init_tag = "Random"
            if args.without_kmeans:
                init_tag = "KMeans"
            test_folder = folder_name + "/" + str(pairwise_num) + "/test" + str(test).zfill(2)

            case_folder = test_folder + "/" + setup_str
            if os.path.exists(case_folder):
                # print("Setup has been run and save at:", case_folder)
                # continue
                pass
            else:
                os.makedirs(case_folder)
            ml_ind1, ml_ind2, cl_ind1, cl_ind2 = read_ml_cl(test_folder)
            print("#Violated PW constraint on ground-truth:",
                  count_violated_cons(y, ml_ind1, ml_ind2, cl_ind1, cl_ind2))
            ml_ind1, ml_ind2, cl_ind1, cl_ind2 = transitive_closure(ml_ind1, ml_ind2, cl_ind1, cl_ind2, X.shape[0])
            plotting_dir = ""
            anchor, positive, negative = np.array([]), np.array([]), np.array([])
            instance_guidance = torch.zeros(X.shape[0]).cuda()
            use_global = False

            start = timeit.default_timer()
            # Train Neural Network
            train_acc, train_nmi, vcon, delta, epo = idec.fit(anchor, positive, negative, ml_ind1, ml_ind2, cl_ind1,
                                                              cl_ind2,
                                                              instance_guidance, use_global, ml_penalty, cl_penalty, X,
                                                              y,
                                                              lr=args.lr, batch_size=args.batch_size,
                                                              num_epochs=args.epochs,
                                                              update_interval=args.update_interval, tol=1 * 1e-3,
                                                              use_kmeans=args.without_kmeans, plotting=plotting_dir)
            training_time = (float)(timeit.default_timer() - start)
            stat_pw.append([train_nmi, train_acc, 0.0, vcon, delta, training_time])
            # Report Results
            print("ACC:", train_acc)
            print("NMI;", train_nmi)
            print("Epochs:", epo)
            print("ML Closure:", ml_ind1.shape[0])
            print("CL Closure:", cl_ind1.shape[0])
        save_tsv(folder_name + "/" + str(pairwise_num) + "/" + setup_str + ".tsv", stat_pw)
