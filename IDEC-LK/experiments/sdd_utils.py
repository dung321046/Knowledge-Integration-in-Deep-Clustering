from pysdd.sdd import SddManager


def clustering_model(n, k, q):
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
        exist_at_1.append(f)
    clustering_model = mgr.conjoin(exist_at_1[0], exist_at_1[1])
    for i in range(2, n):
        clustering_model = mgr.conjoin(clustering_model, exist_at_1[i])

    # Only 1 assignment
    for i in range(n):
        for j in range(k):
            for t in range(j):
                clustering_model = mgr.conjoin(clustering_model,
                                               mgr.disjoin(-mgr.literal(i * k + j + 1), -mgr.literal(i * k + t + 1)))
    for i in range(n):
        for j in range(k):
            if q[i][j] < 0.0001:
                clustering_model = mgr.conjoin(clustering_model, -mgr.literal(i * k + j + 1))
            elif q[i][j] > 1 - 0.0001:
                clustering_model = mgr.conjoin(clustering_model, mgr.literal(i * k + j + 1))
    # Non-empty cluster
    # for i in range(k):
    #     or_cluster = mgr.disjoin(mgr.literal(i + 1), mgr.literal(i + k + 1))
    #     for j in range(2, n):
    #         or_cluster = mgr.disjoin(or_cluster, mgr.literal(j * k + i + 1))
    #     final_exist = mgr.conjoin(final_exist, or_cluster)
    return mgr, clustering_model


def add_cannot_link(u, v, k, mgr, model):
    for i in range(k):
        model = mgr.conjoin(model, mgr.disjoin(-mgr.literal(u * k + i + 1), -mgr.literal(v * k + i + 1)))
    return mgr, model


def add_must_link(u, v, k, mgr, model):
    for i in range(k):
        # model = mgr.conjoin(model, mgr.disjoin(mgr.conjoin(-mgr.literal(u * k + i + 1), -mgr.literal(v * k + i + 1)),
        #                                        mgr.conjoin(mgr.literal(u * k + i + 1), mgr.literal(v * k + i + 1))))
        model = mgr.conjoin(model, mgr.disjoin(mgr.literal(u * k + i + 1), -mgr.literal(v * k + i + 1)))
        model = mgr.conjoin(model, mgr.disjoin(-mgr.literal(u * k + i + 1), mgr.literal(v * k + i + 1)))
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
    # fclustering.ref()
    mgr.minimize_limited()
    for i in range(n):
        for j in range(i):
            if y[i] == y[j]:
                mgr, fclustering = add_must_link(i, j, mgr, fclustering)
            else:
                mgr, fclustering = add_cannot_link(i, j, mgr, fclustering)
    prob_cal = ProbCalculator(q)
    return prob_cal.calculate(fclustering)


def save_models(n, k, y, filename, q):
    mgr, fclustering = clustering_model(n, k, q)
    # fclustering.ref()
    # mgr.minimize()
    cc = dict()
    ccroot = dict()
    for i in range(n):
        ccroot[i] = i
        cc[i] = set([i])
    for i in range(n):
        for j in range(i):
            if y[i] == y[j]:
                if ccroot[i] != ccroot[j]:
                    tmp = ccroot[j]
                    for t in cc[ccroot[j]]:
                        ccroot[t] = ccroot[i]
                        cc[ccroot[i]].add(t)
                    # cc.pop(ccroot[j])
                    del cc[tmp]
    cl_set = set()
    for i in range(n):
        for j in range(i):
            if y[i] == y[j]:
                if ccroot[i] == i or ccroot[j] == j:
                    mgr, fclustering = add_must_link(i, j, k, mgr, fclustering)
            elif (ccroot[i], ccroot[j]) not in cl_set:
                mgr, fclustering = add_cannot_link(i, j, k, mgr, fclustering)
                cl_set.add((ccroot[i], ccroot[j]))
                cl_set.add((ccroot[j], ccroot[i]))
    fclustering.ref()
    # mgr.minimum_cardinality(fclustering)
    # mgr.minimize()
    mgr.minimize_limited()
    # fclustering.deref()
    vtree = mgr.vtree()
    vtree.save(bytes(filename + ".vtree", encoding="utf8"))
    mgr.save(bytes(filename, encoding="utf8"), fclustering)

    # from graphviz import Source
    # g = Source(fclustering.dot())
    # g.render(view=True)


class ProbCalculator:

    def __init__(self, probs):
        self.probs = probs
        self.cal_nodes = dict()

    def calculate_re(self, node):
        if node.id in self.cal_nodes:
            return self.cal_nodes[node.id]
        if node.node_size() == 0:
            if node.literal > 0:
                return self.probs[node.literal - 1]
            return 1.0 - self.probs[- node.literal - 1]
        t = node.elements()
        ans = 0.0
        for i in range(len(t)):
            if t[i][1].is_false():
                continue
            u = self.calculate_re(t[i][0])
            v = self.calculate_re(t[i][1])
            ans += u * v
        self.cal_nodes[node.id] = ans
        return ans

    def calculate(self, node):
        if node.is_false():
            return 0.0
        if node.is_true():
            return 1.0
        return self.calculate_re(node)
