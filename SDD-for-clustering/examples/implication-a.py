import os
import timeit
import warnings

from sdd_clustering.model_a import *
from sdd_clustering.utils import *

warnings.filterwarnings("ignore")


def get_f_from_ml_cl(ml, cl, mgr):
    p = mgr.true()
    for pair in ml:
        p = p.conjoin(f_must_link(pair[0], pair[1], k, mgr))
    for pair in cl:
        p = p.conjoin(f_cannot_link(pair[0], pair[1], k, mgr))
    return p


def get_negf_from_ml_cl(ml, cl, mgr):
    p = mgr.false()
    for pair in ml:
        p = p.disjoin(f_cannot_link(pair[0], pair[1], k, mgr))
    for pair in cl:
        p = p.disjoin(f_must_link(pair[0], pair[1], k, mgr))
    return p


def reformat(pair_arr):
    if pair_arr.shape == (2,):
        return [pair_arr]
    return pair_arr


from sklearn import preprocessing

PREFIX = "../generate_constraints/"

# dataset = "MNIST"
# v = 6
# k = 10
dataset = "Reuters"
v = 10
m = 10
k = 4
stats = []
for test in range(5):
    folder_path = PREFIX + dataset + "-complexR-" + str(v) + "-" + str(m) + "/test" + str(test).zfill(2)
    np.random.seed(test)
    p = np.random.random((v, k))
    p = preprocessing.normalize(p, axis=1, norm="l1")

    for group in range(m):
        ids = np.loadtxt(os.path.join(folder_path, str(group).zfill(2)) + "-indexes", dtype=int)
        mlp = np.loadtxt(os.path.join(folder_path, str(group).zfill(2)) + "-mlp", dtype=int)
        clp = np.loadtxt(os.path.join(folder_path, str(group).zfill(2)) + "-clp", dtype=int)
        mlq = np.loadtxt(os.path.join(folder_path, str(group).zfill(2)) + "-mlq", dtype=int)
        clq = np.loadtxt(os.path.join(folder_path, str(group).zfill(2)) + "-clq", dtype=int)
        mlp, clp = reformat(mlp), reformat(clp)
        mlq, clq = reformat(mlq), reformat(clq)

        time_training = timeit.default_timer()

        mgr, root = clustering_model(len(ids), k, [], -1)
        # root.ref()
        negp = get_negf_from_ml_cl(mlp, clp, mgr)
        q = get_f_from_ml_cl(mlq, clq, mgr)
        # constraint = negp.disjoin(q)
        # constraint.ref()
        # mgr.minimize()
        root = root.conjoin(negp.disjoin(q))
        # constraint.deref()
        root.ref()
        mgr.minimize()
        time_training = (float)(timeit.default_timer() - time_training)
        stats.append([mgr.model_count(root), mgr.size(), time_training])

        # probb = ProbCalculator(p.flatten())
        # pb = probb.calculate(root)
        # print("Prob:", pb)
        print(stats[-1])
        if not os.path.exists(folder_path + "/sdd-a"):
            os.makedirs(folder_path + "/sdd-a")
        vtree = mgr.vtree()
        vtree.save(bytes(os.path.join(folder_path + "/sdd-a", str(group).zfill(2)) + ".vtree", encoding="utf8"))
        mgr.save(bytes(os.path.join(folder_path + "/sdd-a", str(group).zfill(2)) + ".sdd", encoding="utf8"), root)
import csv

with open(dataset + "-" + str(v) + "-A", "w") as file:
    writer = csv.writer(file, delimiter='\t')
    writer.writerows(stats)
