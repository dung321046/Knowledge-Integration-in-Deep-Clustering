import os
import random
import sys

import numpy as np

sys.path.append("..")
from experiments.utils_setup import *


def generate_random_pair(y, num):
    """
    Generate random pairwise constraints.
    """
    ml_ind1, ml_ind2 = [], []
    cl_ind1, cl_ind2 = [], []
    while num > 0:
        tmp1 = random.randint(0, y.shape[0] - 1)
        tmp2 = random.randint(0, y.shape[0] - 1)
        if tmp1 == tmp2:
            continue
        if y[tmp1] == y[tmp2]:
            ml_ind1.append(tmp1)
            ml_ind2.append(tmp2)
        else:
            cl_ind1.append(tmp1)
            cl_ind2.append(tmp2)
        num -= 1
    return np.array(ml_ind1), np.array(ml_ind2), np.array(cl_ind1), np.array(cl_ind2)


if __name__ == "__main__":
    parser = data_parser("Generate pairwise constraints")
    args = parser.parse_args()
    # Load data
    k, X, y, test_X, test_y = load_data(args.data)

    folder_name = "../test_set/Pw-" + args.data
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    for pairwise_num in [10, 100, 500, 1000]:
        print("Generate ", pairwise_num, " pairwise constraints for each test.")
        folder_prefix_path = folder_name + "/" + str(pairwise_num) + "/"
        os.mkdir(folder_prefix_path)
        folder_prefix_path += "test"
        for seed in range(10):
            np.random.seed(seed)
            random.seed(seed)
            folder_path = folder_prefix_path + str(seed).zfill(2)
            os.mkdir(folder_path)
            ml_ind1, ml_ind2, cl_ind1, cl_ind2 = generate_random_pair(y, pairwise_num)
            np.savetxt(os.path.join(folder_path, "ml.txt"), np.column_stack((ml_ind1, ml_ind2)), fmt='%s')
            np.savetxt(os.path.join(folder_path, "cl.txt"), np.column_stack((cl_ind1, cl_ind2)), fmt='%s')
