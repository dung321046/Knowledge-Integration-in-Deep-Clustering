from pysdd.sdd import SddManager


def clustering_model(n, k):
    mgr = SddManager(var_count=n * k)
    # vtree = Vtree.from_file("clustering.vtree")
    # mgr = SddManager(vtree=vtree)
    a = []
    for i in range(n):
        a.append([])
        for j in range(k):
            a[i].append(mgr.literal(i * k + j + 1))
    exist_at_1 = []
    for i in range(n):
        f = mgr.disjoin(a[i][0], a[i][1])
        for j in range(2, k):
            f = mgr.disjoin(f, a[i][j])
        mgr.global_minimize_cardinality(f)
        exist_at_1.append(f)
    final_exist = mgr.conjoin(exist_at_1[0], exist_at_1[1])
    for i in range(2, n):
        final_exist = mgr.conjoin(final_exist, exist_at_1[i])

    # Only 1 assignment
    for i in range(n):
        for j in range(k):
            for t in range(j):
                final_exist = mgr.conjoin(final_exist,
                                          mgr.disjoin(-mgr.literal(i * k + j + 1), -mgr.literal(i * k + t + 1)))
    # Non-empty cluster
    # for i in range(k):
    #     or_cluster = mgr.disjoin(mgr.literal(i + 1), mgr.literal(i + k + 1))
    #     for j in range(2, n):
    #         or_cluster = mgr.disjoin(or_cluster, mgr.literal(j * k + i + 1))
    #     final_exist = mgr.conjoin(final_exist, or_cluster)
    return mgr, final_exist


def add_cannot_link(u, v, mgr, model):
    for i in range(k):
        model = mgr.conjoin(model, mgr.disjoin(-mgr.literal(u * k + i + 1), -mgr.literal(v * k + i + 1)))
    return mgr, model


def add_must_link(u, v, mgr, model):
    for i in range(k):
        model = mgr.conjoin(model, mgr.disjoin(mgr.conjoin(-mgr.literal(u * k + i + 1), -mgr.literal(v * k + i + 1)),
                                               mgr.conjoin(mgr.literal(u * k + i + 1), mgr.literal(v * k + i + 1))))
    return mgr, model


def travel(node):
    if node.node_size() == 0:
        print("Leaf: ", node)
        return
    t = node.elements()
    print(node)
    for i in range(len(t)):
        if t[i][1].is_false():
            continue
        travel(t[i][0])
        travel(t[i][1])
    return


class ProbCalculator:
    def __init__(self, probs):
        self.probs = probs

    def calculate(self, node):
        if node.node_size() == 0:
            print("Leaf: ", node)
            if node.literal > 0:
                return self.probs[node.literal - 1]
            return 1.0
        t = node.elements()
        print(node)
        ans = 0.0
        for i in range(len(t)):
            if t[i][1].is_false():
                continue
            u = self.calculate(t[i][0])
            v = self.calculate(t[i][1])
            ans += u * v
        return ans


def calculate_pw_prob(n, k, y, q):
    mgr, fclustering = clustering_model(n, k)
    fclustering.ref()
    mgr.minimize_limited()
    fclustering.deref()
    for i in range(n):
        for j in range(i):
            if y[i] == y[j]:
                mgr, fclustering = add_must_link(i, j, mgr, fclustering)
            else:
                mgr, fclustering = add_cannot_link(i, j, mgr, fclustering)
    prob_cal = ProbCalculator(q)
    return prob_cal.calculate(fclustering)


if __name__ == "__main__":

    import random
    import os
    import sys

    sys.path.append(os.pardir)
    n = 5
    k = 3
    random.seed(1)
    mgr, fclustering = clustering_model(n, k)
    fclustering.ref()
    mgr.minimize_limited()
    fclustering.deref()
    # g = Source(fclustering.dot())
    # g.render(view=True)
    # mgr, fclustering = add_cannot_link(0, 1, mgr, fclustering)
    y = [random.randrange(k) for i in range(n)]
    print(y)
    # for i in range(n):
    #     for j in range(i):
    #         if y[i] == y[j]:
    #             mgr, fclustering = add_must_link(i, j, mgr, fclustering)
    #         else:
    #             mgr, fclustering = add_cannot_link(i, j, mgr, fclustering)
    cc = dict()
    ccroot = dict()
    for i in range(n):
        for j in range(i):
            if y[i] == y[j]:
                if i in ccroot:
                    if j not in ccroot:
                        ccroot[j] = ccroot[i]
                        cc[ccroot[i]].add(j)
                    elif ccroot[i] != ccroot[j]:
                        for t in cc[ccroot[j]]:
                            ccroot[t] = ccroot[i]
                            cc[ccroot[i]].add(t)
                        # cc.pop(ccroot[j])
                        del cc[ccroot[j]]
                else:
                    if j in ccroot:
                        ccroot[i] = ccroot[j]
                        cc[ccroot[j]].add(i)
                    else:
                        ccroot[i] = ccroot[j] = i
                        cc[i] = set([i, j])
    for i in range(n):
        if i not in ccroot:
            ccroot[i] = i
    cl_set = set()
    for i in range(n):
        for j in range(i):
            if y[i] == y[j]:
                if ccroot[i] == i or ccroot[j] == j:
                    mgr, fclustering = add_must_link(i, j, mgr, fclustering)
                    fclustering.ref()
                    mgr.minimize_limited()
                    print("ML:", i, j)
            elif (ccroot[i], ccroot[j]) not in cl_set:
                mgr, fclustering = add_cannot_link(i, j, mgr, fclustering)
                # mgr, fclustering = add_cannot_link(ccroot[i], ccroot[j], mgr, fclustering)
                fclustering.ref()
                mgr.minimize_limited()
                cl_set.add((ccroot[i], ccroot[j]))
                cl_set.add((ccroot[j], ccroot[i]))
                print("CL:", i, j, ccroot[i], ccroot[j])
    # g = Source(final_exist.dot())
    # g.render(view=True)
    # fclustering.ref()
    # mggr.minimize_limited()
    # mgr.global_minimize_cardinality(fclustering)
    # vtree = mgr.vtree()
    # print(fclustering.dot())
    # fclustering.deref()
    # prob_cal = ProbCalculator([0.8, 0.2, 0.3, 0.7, 0.0, 1.0])
    # print("Prob:", prob_cal.calculate(fclustering))
    # g = Source(fclustering.dot())
    # g.render(view=True)
    fclustering.ref()
    # mgr.minimize_limited()
    mgr.minimize()
    print(mgr.size())
    print(mgr.model_count(fclustering))
    print(mgr.count())
    mgr.minimum_cardinality(fclustering)
    print(mgr.size())
    print(mgr.model_count(fclustering))
    print(mgr.count())
