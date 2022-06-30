import argparse
import os.path
import sys

sys.path.append("..")
import numpy as np
from sklearn import preprocessing

from sdd_clustering import convert_to_T
from sdd_clustering import model_a
from sdd_clustering import model_b
from sdd_clustering.utils import ProbCalculator, weight_convert_b


def define_args():
    parser = argparse.ArgumentParser(description='Construct constraint formulation')
    parser.add_argument('--formu', type=str, default="A", metavar='A/B',
                        help='Two available formulations: A/B  (default: A)')
    parser.add_argument('-c', '--c', action="store_true", help='Include checking with direct calculation')
    parser.add_argument('--type', type=str, default="ml",
                        help='Select constraint type: ml (must-link), cl (cannot-link), triplet, relative, m-clusters  (default: ml)')
    parser.add_argument('--k', type=int, default=4, help='Input number of cluster  (default: 4)')
    parser.add_argument('--out_path', type=str, default="./", metavar='PATH', help='Folder to save outputs')
    return parser


def generate_cases(n, k):
    p = np.random.random((n, k))
    p = preprocessing.normalize(p, axis=1, norm="l1")
    ans = [p]
    for i in range(4):
        # p = np.zeros((n, k))
        # p = np.random.randint(2, size=(n, k))
        p = np.random.choice([0, 1], size=n * k, p=[.7, .3])
        p = np.reshape(p, (n, k))
        for j in range(n):
            p[j][np.random.randint(k)] = 1.0
        p = preprocessing.normalize(p, axis=1, norm="l1")
        ans.append(p)
    return ans


def get_m_clusters():
    input_string = input('Enter clusters (numbered from 1 to k) that appear in the group:')
    return [int(a) for a in input_string.split()]


if __name__ == "__main__":
    par = define_args()
    args = par.parse_args()
    con_type = args.type
    k = args.k
    path = os.path.join(args.out_path, con_type + args.formu)
    fname = con_type + args.formu + "-" + str(k)
    if con_type == "triplet" or con_type == "relative":
        n = 3
    elif con_type == "m-clusters":
        n = 1
    else:
        n = 2
    # Create clustering model
    if args.formu == "A":
        mgr, root = model_a.clustering_model(n, k, [], -1, True)
    else:
        mgr, root = model_b.clustering_model(n, k, [], -1)
    # Adding constraint
    if con_type == "triplet" or con_type == "relative":
        if args.formu == "A":
            root = root.conjoin(model_a.f_triplet(0, 1, 2, k, mgr))
        else:
            root = root.conjoin(model_b.f_triplet(0, 1, 2, k, mgr))
    elif con_type == "ml":
        if args.formu == "A":
            root = root.conjoin(model_a.f_must_link(0, 1, k, mgr))
        else:
            root = root.conjoin(model_b.f_must_link(0, 1, k, mgr))
    elif con_type == "cl":
        if args.formu == "A":
            root = root.conjoin(model_a.f_cannot_link(0, 1, k, mgr))
        else:
            root = root.conjoin(model_b.f_cannot_link(0, 1, k, mgr))
    elif con_type == "m-clusters":
        clusters = get_m_clusters()
        temp = mgr.false()
        for cluster in clusters:
            if args.formu == "A":
                temp = temp.disjoin(mgr.literal(cluster))
            else:
                temp = temp.disjoin(model_b.pos_point(0, cluster - 1, k, mgr))
        root = root.conjoin(temp)
    # Optimize tree
    root.ref()
    mgr.minimize()
    # Save files
    convert_to_T.save_model(path, fname, mgr, root)
    # Calculate WMC directly or by loading arithmetics tree
    if args.c:
        cases = generate_cases(n, k)
        loaded_root = convert_to_T.load_t(os.path.join(path, fname + ".txt"))
        for p in cases:
            if args.formu == "A":
                prob = ProbCalculator(p.flatten())
            else:
                prob = ProbCalculator(weight_convert_b(p, n, k))
            print("Input       :\n", p)
            print("WMC - direct:", prob.calculate(root))
            print("WMC - file  :", prob.calculate(root))
