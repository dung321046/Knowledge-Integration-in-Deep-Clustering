import numpy as np

from deep_clustering.load_dcc import load_ae_with_y_pred

def generate_ml_cl(start, f, v, length, y):
    '''
    :param start: start indexes of chained pairwises
    :param f: value of the generated function based on y = True/False
    :param v: the number of input samples
    :param y: ground truth clusters  corresponding to input samples
    :return: The logic function: conjunction (and) of multiple Together/Apart constraints
    '''
    chained_pair = set()
    x_cur = start
    for l in range(length):
        x_nxt = (x_cur + np.random.randint(v - 1) + 1) % v
        if (x_cur, x_nxt) not in chained_pair and (x_nxt, x_cur) not in chained_pair:
            chained_pair.add((x_cur, x_nxt))
            x_cur = x_nxt
    error = -1
    chained_pair = list(chained_pair)
    if not f:
        error = np.random.randint(len(chained_pair))
    ml1 = []
    cl1 = []
    for i in range(len(chained_pair)):
        if (y[chained_pair[i][0]] == y[chained_pair[i][1]]) ^ (i == error):
            ml1.append((chained_pair[i][0], chained_pair[i][1]))
        else:
            cl1.append((chained_pair[i][0], chained_pair[i][1]))
    return ml1, cl1


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


np.set_printoptions(formatter={'float': '{: 0.4f}'.format})

# Number of constraints - m
m = 10

# Complex level - v
# v = 6
# dataset = "MNIST"

v = 10
dataset = "Reuters"

total_p, total_n, k, total_y, total_y_pred = load_ae_with_y_pred(dataset)

import os

testset_path = dataset + "-complexR-" + str(v) + "-" + str(m)

if not os.path.exists(testset_path):
    os.makedirs(testset_path)
for test in range(10):
    print("Generating test: ", test)
    folder_path = testset_path + "/test" + str(test).zfill(2)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    np.random.seed(test)
    cur_num = 0
    while cur_num < m:
        ids = np.random.permutation(total_n)[:v]
        y = total_y[ids]
        y_pred = total_y_pred[ids]
        P = np.random.randint(2)
        if P == 1:
            Q = 1
        else:
            Q = np.random.randint(2)
        start = np.random.randint(v)
        # Pair of P and Q could be the same???
        mlp, clp = generate_ml_cl(start, P, v, v // 2, y)
        mlq, clq = generate_ml_cl(start, Q, v, v - v // 2, y)
        # P_check = check_pairwise_conjunction(mlp, clp, y)
        # Q_check = check_pairwise_conjunction(mlq, clq, y)
        # if not is_satisfy_complex_constraint(mlp, clp, mlq, clq, y_pred):
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
