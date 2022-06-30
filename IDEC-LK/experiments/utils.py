import random

import numpy as np


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


import os


def read_ml_cl(folder_name):
    ml = np.loadtxt(folder_name + "/ml.txt", dtype=int, ndmin=2)
    cl = np.loadtxt(folder_name + "/cl.txt", dtype=int, ndmin=2)
    if len(ml) > 0:
        ml1 = ml[:, 0]
        ml2 = ml[:, 1]
    else:
        ml1 = []
        ml2 = []
    if len(cl) > 0:
        cl1 = []
        cl2 = []
    else:
        cl1 = cl[:, 0]
        cl2 = cl[:, 1]
    return ml1, ml2, cl1, cl2


def read_triplet(folder_name):
    triplet = np.loadtxt(folder_name + "/triplet.txt", dtype=int, ndmin=2)
    return triplet[:, 0], triplet[:, 1], triplet[:, 2]


def read_hard_ml_cl(folder_name, y):
    '''

    :param folder_name:folder contain ml and cl files
    :return: ml, cl that not satisfied yet
    '''
    ml = np.loadtxt(os.path.join(folder_name, "ml.txt"), dtype=int)
    cl = np.loadtxt(os.path.join(folder_name, "cl.txt"), dtype=int)
    ml1 = []
    ml2 = []
    for i in range(len(ml)):
        u = ml[i][0]
        v = ml[i][1]
        if y[u] != y[v]:
            ml1.append(u)
            ml2.append(v)
    cl1 = []
    cl2 = []
    for i in range(len(cl)):
        u = cl[i][0]
        v = cl[i][1]
        if y[u] == y[v]:
            cl1.append(u)
            cl2.append(v)
    return ml1, ml2, cl1, cl2


def random_vertex(used_vertices, new_vertices, c_index):
    if len(used_vertices) == 0:
        return new_vertices[random.randint(0, len(new_vertices) - 1)]
    if len(new_vertices) == 0:
        return used_vertices[random.randint(0, len(used_vertices) - 1)]
    if random.random() >= c_index:
        return new_vertices[random.randint(0, len(new_vertices) - 1)]
    return used_vertices[random.randint(0, len(used_vertices) - 1)]


def generate_random_pair_compact(y, num, c_index):
    """
    Generate random pairwise constraints.
    """
    ml_ind1, ml_ind2 = [], []
    cl_ind1, cl_ind2 = [], []
    used_vertices = []
    new_vertices = []
    for i in range(y.shape[0]):
        new_vertices.append(i)
    while num > 0:
        tmp1 = random_vertex(used_vertices, new_vertices, c_index)
        tmp2 = random_vertex(used_vertices, new_vertices, c_index)
        if tmp1 == tmp2:
            continue
        if tmp1 in new_vertices:
            new_vertices.remove(tmp1)
            used_vertices.append(tmp1)
        if tmp2 in new_vertices:
            new_vertices.remove(tmp2)
            used_vertices.append(tmp2)
        if y[tmp1] == y[tmp2]:
            ml_ind1.append(tmp1)
            ml_ind2.append(tmp2)
        else:
            cl_ind1.append(tmp1)
            cl_ind2.append(tmp2)
        num -= 1
    print("C-index:", c_index, " #Vertex:", len(used_vertices))
    return np.array(ml_ind1), np.array(ml_ind2), np.array(cl_ind1), np.array(cl_ind2)


def generate_all_pair(y, groups):
    """
    Generate all pairwises for each group.
    """
    ml_ind1, ml_ind2 = [], []
    cl_ind1, cl_ind2 = [], []
    for group in groups:
        for i in range(len(group)):
            for j in range(i):
                if y[group[i]] == y[group[j]]:
                    ml_ind1.append(group[i])
                    ml_ind2.append(group[j])
                else:
                    cl_ind1.append(group[i])
                    cl_ind2.append(group[j])
    return np.array(ml_ind1), np.array(ml_ind2), np.array(cl_ind1), np.array(cl_ind2)
