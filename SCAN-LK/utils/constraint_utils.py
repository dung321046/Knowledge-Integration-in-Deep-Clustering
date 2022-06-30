import os
import random

import numpy as np


def generate_random_pair(y, num, r=0.5):
    """
    Generate random pairwise constraints.
    """
    ml_ind1, ml_ind2 = [], []
    cl_ind1, cl_ind2 = [], []
    n = len(y)
    maxcl = int(num * r)
    ncl = 0
    while num > 0:
        tmp1 = random.randint(0, n - 1)
        tmp2 = random.randint(0, n - 1)
        if tmp1 == tmp2:
            continue
        if y[tmp1] == y[tmp2]:
            ml_ind1.append(tmp1)
            ml_ind2.append(tmp2)
        else:
            if ncl >= maxcl:
                continue
            cl_ind1.append(tmp1)
            cl_ind2.append(tmp2)
            ncl += 1
        num -= 1
    return np.array(ml_ind1), np.array(ml_ind2), np.array(cl_ind1), np.array(cl_ind2)


def generate_pw_constraints(path, labels, pairwise_num=10000, r=0.5):
    for t in range(10):
        random.seed(t)
        ml_ind1, ml_ind2, cl_ind1, cl_ind2 = generate_random_pair(labels, pairwise_num, r=0.5)
        test_path = os.path.join(path, "test" + str(t).zfill(2))
        os.mkdir(test_path)
        np.savetxt(os.path.join(test_path, "ml.txt"), np.column_stack((ml_ind1, ml_ind2)), fmt='%s')
        np.savetxt(os.path.join(test_path, "cl.txt"), np.column_stack((cl_ind1, cl_ind2)), fmt='%s')


def read_ml_cl(folder_name):
    ml = np.loadtxt(folder_name + "\\ml.txt", dtype=int, ndmin=2)
    cl = np.loadtxt(folder_name + "\\cl.txt", dtype=int, ndmin=2)
    if len(ml) > 0:
        ml1 = ml[:, 0]
        ml2 = ml[:, 1]
    else:
        ml1 = []
        ml2 = []
    if len(cl) > 0:
        cl1 = cl[:, 0]
        cl2 = cl[:, 1]
    else:
        cl1 = []
        cl2 = []
    return ml1, ml2, cl1, cl2


def transitive_closure(ml_ind1, ml_ind2, cl_ind1, cl_ind2, n):
    """
    This function calculate the total transtive closure for must-links and the full entailment
    for cannot-links.

    # Arguments
        ml_ind1, ml_ind2 = instances within a pair of must-link constraints
        cl_ind1, cl_ind2 = instances within a pair of cannot-link constraints
        n = total training instance number

    # Return
        transtive closure (must-links)
        entailment of cannot-links
    """
    ml_graph = dict()
    cl_graph = dict()
    for i in range(n):
        ml_graph[i] = set()
        cl_graph[i] = set()

    def add_both(d, i, j):
        d[i].add(j)
        d[j].add(i)

    for (i, j) in zip(ml_ind1, ml_ind2):
        add_both(ml_graph, i, j)

    def dfs(i, graph, visited, component):
        visited[i] = True
        for j in graph[i]:
            if not visited[j]:
                dfs(j, graph, visited, component)
        component.append(i)

    visited = [False] * n
    for i in range(n):
        if not visited[i]:
            component = []
            dfs(i, ml_graph, visited, component)
            for x1 in component:
                for x2 in component:
                    if x1 != x2:
                        ml_graph[x1].add(x2)
    for (i, j) in zip(cl_ind1, cl_ind2):
        add_both(cl_graph, i, j)
        for y in ml_graph[j]:
            add_both(cl_graph, i, y)
        for x in ml_graph[i]:
            add_both(cl_graph, x, j)
            for y in ml_graph[j]:
                add_both(cl_graph, x, y)
    ml_res_set = set()
    cl_res_set = set()
    for i in ml_graph:
        for j in ml_graph[i]:
            if j != i and j in cl_graph[i]:
                raise Exception('inconsistent constraints between %d and %d' % (i, j))
            if i <= j:
                ml_res_set.add((i, j))
            else:
                ml_res_set.add((j, i))
    for i in cl_graph:
        for j in cl_graph[i]:
            if i <= j:
                cl_res_set.add((i, j))
            else:
                cl_res_set.add((j, i))
    ml_res1, ml_res2 = [], []
    cl_res1, cl_res2 = [], []
    for (x, y) in ml_res_set:
        ml_res1.append(x)
        ml_res2.append(y)
    for (x, y) in cl_res_set:
        cl_res1.append(x)
        cl_res2.append(y)
    return np.array(ml_res1), np.array(ml_res2), np.array(cl_res1), np.array(cl_res2)


def count_violated_cons(y, ml_ind1, ml_ind2, cl_ind1, cl_ind2):
    num_vcon = 0
    for i in range(len(ml_ind1)):
        if y[ml_ind1[i]] != y[ml_ind2[i]]:
            num_vcon += 1
    for i in range(len(cl_ind1)):
        if y[cl_ind1[i]] == y[cl_ind2[i]]:
            num_vcon += 1
    return num_vcon
