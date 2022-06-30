from pysdd.sdd import SddManager


def neg_point(u, h, k, mgr):
    f = -mgr.literal(u * k + h + 1)
    for i in range(h):
        f = f.disjoin(mgr.literal(u * k + i + 1))
    return f


def pos_point(u, h, k, mgr):
    f = mgr.literal(u * k + h + 1)
    for i in range(h):
        f = f.conjoin(-mgr.literal(u * k + i + 1))
    return f


def clustering_model(n, k, p, eps):
    mgr = SddManager(var_count=n * k)
    root = mgr.true()
    for i in range(n):
        root = root.conjoin(mgr.literal(i * k + k))
    if eps > 0.0:
        for i in range(n):
            for j in range(k):
                if p[i][j] < eps:
                    root = root.conjoin(neg_point(i, j, k, mgr))
                elif p[i][j] > 1 - eps:
                    root = root.conjoin(pos_point(i, j, k, mgr))
    return mgr, root


def neg_both_active(u, v, i, k, mgr):
    f = mgr.literal(-u * k - i - 1).disjoin(-mgr.literal(v * k + i + 1))
    for t in range(i):
        f = mgr.disjoin(f, mgr.literal(u * k + t + 1))
        f = mgr.disjoin(f, mgr.literal(v * k + t + 1))
    return f


def both_active(u, v, t, k, mgr):
    f = mgr.conjoin(mgr.literal(u * k + t + 1), mgr.literal(v * k + t + 1))
    for i in range(t):
        f = mgr.conjoin(f, -mgr.literal(u * k + i + 1))
        f = mgr.conjoin(f, -mgr.literal(v * k + i + 1))
    return f


def one_active_one_neg(u, v, i, k, mgr):
    f = mgr.literal(u * k + i + 1)
    for t in range(i):
        f = f.conjoin(-mgr.literal(u * k + t + 1))
    g = -mgr.literal(v * k + i + 1)
    for t in range(i):
        g = g.disjoin(mgr.literal(v * k + t + 1))
    return mgr.disjoin(f, g)


def f_cannot_link(u, v, k, mgr):
    # f = mgr.true()
    # for i in range(k):
    #     f = f.conjoin(neg_both_active(u, v, i, k, mgr))

    # Second way
    # f = mgr.disjoin(mgr.literal(u * k + k - 1).conjoin(-mgr.literal(v * k + k - 1)),
    #                 mgr.literal(v * k + k - 1).conjoin(-mgr.literal(u * k + k - 1)))
    # for i in range(k - 2, 0, -1):
    #     f = f.conjoin(-mgr.literal(u * k + i))
    #     f = f.conjoin(-mgr.literal(v * k + i))
    #     f = f.disjoin(mgr.literal(u * k + i).conjoin(-mgr.literal(v * k + i)))
    #     f = f.disjoin(mgr.literal(v * k + i).conjoin(-mgr.literal(u * k + i)))
    # Third way
    f = mgr.true()
    for i in range(k):
        f = f.conjoin(mgr.disjoin(neg_point(u, i, k, mgr), neg_point(v, i, k, mgr)))
    return f


def f_must_link(u, v, k, mgr):
    f = mgr.true()
    for i in range(k):
        f = f.conjoin(one_active_one_neg(u, v, i, k, mgr))
        f = f.conjoin(one_active_one_neg(v, u, i, k, mgr))
    # Second way
    # In fact, f = mgr.conjoin(mgr.literal(u * k + k), mgr.literal(v * k + k))
    # f = mgr.true()
    # for i in range(k - 1, 0, -1):
    #     f = f.conjoin(-mgr.literal(u * k + i))
    #     f = f.conjoin(-mgr.literal(v * k + i))
    #     f = f.disjoin(mgr.literal(u * k + i).conjoin(mgr.literal(v * k + i)))
    return f


def f_triplet(a, p, n, k, mgr):
    return f_must_link(a, p, k, mgr).disjoin(f_cannot_link(a, n, k, mgr))


def partition_model(n, k, y, p, esp):
    mgr, root = clustering_model(n, k, p, esp)
    ccroot = list(range(n))
    cc = dict(zip(ccroot, [{i} for i in range(n)]))
    for i in range(n):
        for j in range(i):
            if y[i] == y[j]:
                if ccroot[i] != ccroot[j]:
                    tmp = ccroot[j]
                    for t in cc[ccroot[j]]:
                        ccroot[t] = ccroot[i]
                        cc[ccroot[i]].add(t)
                    del cc[tmp]
    cl_set = set()
    for i in range(n):
        for j in range(i):
            if y[i] == y[j]:
                if ccroot[i] == i or ccroot[j] == j:
                    root = root.conjoin(f_must_link(i, j, k, mgr))
            elif (ccroot[i], ccroot[j]) not in cl_set:
                root = root.conjoin(f_cannot_link(i, j, k, mgr))
                cl_set.add((ccroot[i], ccroot[j]))
                cl_set.add((ccroot[j], ccroot[i]))
    root.ref()
    mgr.minimize()
    return mgr, root
