from deep_clustering.load_dcc import load_ae_with_y_pred

from sdd_clustering import model_b
from sdd_clustering.utils import *


# from deep_utils import load_ae
def generate_ml_cl(start, f, ids, length, y, forbiden_pairs=set()):
    '''
    :param start: start indexes of chained pairwises
    :param f: value of the generated function based on y = True/False
    :param v: the number of input samples
    :param y: ground truth clusters  corresponding to input samples
    :return: The logic function: conjunction (and) of multiple Together/Apart constraints
    '''
    chained_pair = set()
    n = len(y)
    x_cur = start
    while len(chained_pair) < length:
        # if np.random.random() < 0.5 and len(ids) > 2:
        if np.random.random() < 0.7 and len(ids) > 2:
            x_nxt = ids[np.random.randint(len(ids))]
        else:
            x_nxt = np.random.randint(n)
        if x_nxt == x_cur:
            continue
        if (x_cur, x_nxt) not in chained_pair and (x_nxt, x_cur) not in chained_pair:
            if (x_cur, x_nxt) not in forbiden_pairs and (x_nxt, x_cur) not in forbiden_pairs:
                chained_pair.add((x_cur, x_nxt))
                x_cur = x_nxt
                if x_cur not in ids:
                    ids.append(x_cur)
    error = -1
    chained_pair = list(chained_pair)
    if not f:
        error = np.random.randint(len(chained_pair))
    ml1 = []
    cl1 = []
    for i in range(len(chained_pair)):
        u, v = chained_pair[i][0], chained_pair[i][1]
        if (y[u] == y[v]) ^ (i == error):
            ml1.append((ids.index(u), ids.index(v)))
        else:
            cl1.append((ids.index(u), ids.index(v)))
    return ml1, cl1, chained_pair


def check_pairwise_conjunction(ml, cl, y):
    for pair in ml:
        if y[pair[0]] != y[pair[1]]:
            return False
    for pair in cl:
        if y[pair[0]] == y[pair[1]]:
            return False
    return True


def is_satisfy_complex_constraint(mlp, clp, mlq, clq, y):
    p = check_pairwise_conjunction(mlp, clp, y)
    q = check_pairwise_conjunction(mlq, clq, y)
    if p and not q:
        return False
    return True


def get_f_from_ml_cl(ml, cl, mgr):
    p = mgr.true()
    for pair in ml:
        p = p.conjoin(model_b.f_must_link(pair[0], pair[1], k, mgr))
    for pair in cl:
        p = p.conjoin(model_b.f_cannot_link(pair[0], pair[1], k, mgr))
    return p


def get_negf_from_ml_cl(ml, cl, mgr):
    p = mgr.false()
    for pair in ml:
        p = p.disjoin(model_b.f_cannot_link(pair[0], pair[1], k, mgr))
    for pair in cl:
        p = p.disjoin(model_b.f_must_link(pair[0], pair[1], k, mgr))
    return p


np.set_printoptions(formatter={'float': '{: 0.4f}'.format})

# Number of constraints - m
m = 100
casePTQT = 80
# Complex level - v
v = 4
dataset = "MNIST"

# v = 2
# dataset = "Reuters"

total_p, total_n, k, total_y, total_y_pred = load_ae_with_y_pred(dataset)

import os

# Complex R2 means that P = 80% T, 20% F

testset_path = dataset + "-complexR2-" + str(v) + "-" + str(m)

if not os.path.exists(testset_path):
    os.makedirs(testset_path)
for test in range(5):
    print("Generating test: ", test)
    folder_path = testset_path + "/test" + str(test).zfill(2)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    np.random.seed(test)
    cur_num = 0
    while cur_num < m:
        if np.random.random() < 0.8 and casePTQT > 0:
            P = 1
            casePTQT -= 1
        else:
            P = 0
        # P = np.random.randint(2)
        if P == 1:
            Q = 1
        else:
            Q = np.random.randint(2)
        start = np.random.randint(total_n)
        ids = [start]
        mlp, clp, pairs = generate_ml_cl(start, P, ids, v - 1, total_y)
        # print(ids)
        mlq, clq, _ = generate_ml_cl(start, Q, ids, 1, total_y, forbiden_pairs=pairs)
        # print("P:", mlp, clp)
        # print("Q:", mlq, clq)

        mgr, root = model_b.clustering_model(len(ids), k, [], -1)
        negp = get_negf_from_ml_cl(mlp, clp, mgr)
        q = get_f_from_ml_cl(mlq, clq, mgr)

        root = root.conjoin(negp.disjoin(q))
        root.ref()
        mgr.minimize()
        if mgr.model_count(root) == 1:
            continue
        # P_check = check_pairwise_conjunction(mlp, clp, y)
        # Q_check = check_pairwise_conjunction(mlq, clq, y)
        y_pred = total_y_pred[ids]
        if is_satisfy_complex_constraint(mlp, clp, mlq, clq, y_pred) and np.random.random() > 0.1:
            continue

        # print(total_y_pred[ids])
        np.savetxt(os.path.join(folder_path, str(cur_num).zfill(2)) + "-indexes", ids, fmt='%s')
        np.savetxt(os.path.join(folder_path, str(cur_num).zfill(2)) + "-mlp", mlp, fmt='%s')
        np.savetxt(os.path.join(folder_path, str(cur_num).zfill(2)) + "-clp", clp, fmt='%s')
        np.savetxt(os.path.join(folder_path, str(cur_num).zfill(2)) + "-mlq", mlq, fmt='%s')
        np.savetxt(os.path.join(folder_path, str(cur_num).zfill(2)) + "-clq", clq, fmt='%s')
        cur_num += 1
        # print(P, Q)
        # print(P_pred, Q_pred)
        # print("__", check_complex_constraint(mlp, clp, mlq, clq, y))
        # print(P, Q)
        # print(check_complex_constraint(mlp, clp, mlq, clq, y))
