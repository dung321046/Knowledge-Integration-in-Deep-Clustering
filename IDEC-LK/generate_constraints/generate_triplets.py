import os
import random
import sys

import numpy as np

sys.path.append("..")
from experiments.utils_setup import *
from lib.utils import generate_random_triplets

if __name__ == "__main__":
    parser = data_parser("Generate pairwise constraints")
    args = parser.parse_args()
    # Load data
    k, X, y, test_X, test_y = load_data(args.data)

    folder_name = "../test_set/Triplet-" + args.data
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    for triplet_num in [10, 100, 500, 1000, 10000]:
        print("Generate ", triplet_num, " triplet constraints for each test.")
        folder_prefix_path = folder_name + "/" + str(triplet_num) + "/"
        os.mkdir(folder_prefix_path)
        folder_prefix_path += "test"
        for seed in range(10):
            np.random.seed(seed)
            random.seed(seed)
            folder_path = folder_prefix_path + str(seed).zfill(2)
            os.mkdir(folder_path)
            anchors, postives, negatives = generate_random_triplets(y, triplet_num)
            # triplets = np.stack((anchors, postives, negatives), axis=1)
            np.savetxt(os.path.join(folder_path, "triplet.txt"), np.column_stack((anchors, postives, negatives)),
                       fmt='%s')
            # np.savetxt(os.path.join(folder_path, "cl.txt"), np.column_stack((cl_ind1, cl_ind2)), fmt='%s')
