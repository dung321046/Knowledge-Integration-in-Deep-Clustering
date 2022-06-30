from pysdd.sdd import SddManager, SddNode


def f_cannot_link(u, v, k, mgr):
    f = mgr.true()
    for i in range(k):
        f = f.conjoin((-mgr.literal(u * k + i + 1)).disjoin(-mgr.literal(v * k + i + 1)))
    return f


def f_must_link(u, v, k, mgr):
    f = mgr.true()
    for i in range(k):
        # model = model.conjoin(mgr.disjoin(mgr.conjoin(-mgr.literal(u * k + i + 1), -mgr.literal(v * k + i + 1)),
        #                                   mgr.conjoin(mgr.literal(u * k + i + 1), mgr.literal(v * k + i + 1))))
        # The second case is slightly better in compiled time.
        f = f.conjoin(mgr.literal(u * k + i + 1).disjoin(-mgr.literal(v * k + i + 1)))
        f = f.conjoin(mgr.literal(v * k + i + 1).disjoin(-mgr.literal(u * k + i + 1)))
    return f


def f_triplet(a, p, n, k, mgr):
    return f_must_link(a, p, k, mgr).disjoin(f_cannot_link(a, n, k, mgr))


def f_relative(a, b, c, k, mgr):
    all_diff = f_cannot_link(a, b, k, mgr).conjoin(f_cannot_link(b, c, k, mgr).conjoin(f_cannot_link(c, a, k, mgr)))
    return f_must_link(a, b, k, mgr).disjoin(all_diff)


def add_must_link(u, v, k, mgr, model):
    for i in range(k):
        # model = model.conjoin(mgr.disjoin(mgr.conjoin(-mgr.literal(u * k + i + 1), -mgr.literal(v * k + i + 1)),
        #                                   mgr.conjoin(mgr.literal(u * k + i + 1), mgr.literal(v * k + i + 1))))
        # The second case is slightly better in compiled time.
        model = model.conjoin(mgr.literal(u * k + i + 1).disjoin(-mgr.literal(v * k + i + 1)))
        model = model.conjoin(mgr.literal(v * k + i + 1).disjoin(-mgr.literal(u * k + i + 1)))
    return model


def add_partition_constraint(n, k, y, mgr: SddManager, root: SddNode):
    ccroot = list(range(n))
    cc = dict(zip(ccroot, [{i} for i in range(n)]))
    for i in range(n):
        for j in range(i):
            if y[i] == y[j] and ccroot[i] != ccroot[j]:
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
                    root = add_must_link(i, j, k, mgr, root)
            elif (ccroot[i], ccroot[j]) not in cl_set:
                # root = add_cannot_link(ccroot[i], ccroot[j], k, mgr, fclustering)
                # For quick test i,j is better.
                root = root.conjoin(f_cannot_link(i, j, k, mgr))
                cl_set.add((ccroot[i], ccroot[j]))
                cl_set.add((ccroot[j], ccroot[i]))
    return root


def clustering_model(n, k, q, eps, include_c=True):
    mgr = SddManager(var_count=n * k)
    root = mgr.true()
    A = []
    for i in range(n):
        A.append([])
        for j in range(k):
            A[i].append(mgr.literal(i * k + j + 1))
    if include_c:
        # Equation %\ref{eq:clust1}%
        for i in range(n):
            f = A[i][0]
            for j in range(1, k):
                f = f.disjoin(A[i][j])
            root = root.conjoin(f)
        # Equation %\ref{eq:clust2}%
        for i in range(n):
            for j in range(k):
                for t in range(j):
                    root = root.conjoin(mgr.disjoin(-A[i][j], -A[i][t]))
    if eps > 0.0:
        for i in range(n):
            for j in range(k):
                if q[i][j] < eps:
                    root = root.conjoin(-A[i][j])
                elif q[i][j] > 1 - eps:
                    root = root.conjoin(A[i][j])
    return mgr, root


def create_clustering_with_partition_con(n, k, y, q, eps, include_c=True):
    mgr, root = clustering_model(n, k, q, eps, include_c)
    root = add_partition_constraint(n, k, y, mgr, root)
    root.ref()
    mgr.minimize()
    return mgr, root
