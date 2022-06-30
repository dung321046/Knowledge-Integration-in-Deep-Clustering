from pysdd.sdd import SddManager


class SDD_Case_Constructor():
    def __init__(self, n, k, check_func, mgr: SddManager, p, eps):
        self.n = n
        self.n_clusters = k
        self.condition_func = check_func
        self.mgr = mgr
        self.root = self.mgr.false()
        self.p = p
        self.eps = eps

    def pos_point(self, u, h, k, mgr):
        if h == k:
            f = mgr.true()
        else:
            f = mgr.literal(u * k + h + 1)
        for i in range(h):
            f = f.conjoin(-mgr.literal(u * k + i + 1))
        return f

    def build(self, assigned):
        m = len(assigned)
        if m == self.n:
            # print("Case:", assigned)
            return self.mgr.true()
        # if m == 0:
        #     wcs_prob = self.root
        #     self.root.ref()
        # else:
        wcs_prob = self.mgr.false()
        for i in range(self.n_clusters):
            if self.condition_func(assigned, i):
                assigned.append(i)
                sub_case = self.pos_point(m, i, self.n_clusters, self.mgr)
                sub_case = sub_case.conjoin(self.build(assigned))
                wcs_prob = wcs_prob.disjoin(sub_case)
                assigned.pop()
        return wcs_prob

    def build_eps(self, assigned):
        m = len(assigned)
        if m == self.n:
            # print("Case:", assigned)
            return self.mgr.true()
        # if m == 0:
        #     wcs_prob = self.root
        #     self.root.ref()
        # else:
        wcs_prob = self.mgr.false()
        for i in range(self.n_clusters):
            if self.p[m][i] > self.eps and self.condition_func(assigned, i):
                assigned.append(i)
                sub_case = self.pos_point(m, i, self.n_clusters, self.mgr)
                sub_case = sub_case.conjoin(self.build(assigned))
                wcs_prob = wcs_prob.disjoin(sub_case)
                assigned.pop()
        return wcs_prob


def create_model(n, k, y, p, eps):
    mgr = SddManager(var_count=n * k, auto_gc_and_minimize=False)

    def is_satisfied(assigned, v):
        next = len(assigned)
        for i in range(next):
            if assigned[i] == v and y[i] != y[next]:
                return False
            if assigned[i] != v and y[i] == y[next]:
                return False
        return True

    sdd_constructor = SDD_Case_Constructor(n, k, is_satisfied, mgr, p, eps)
    if eps > 0.0:
        root = sdd_constructor.build_eps([])
    else:
        root = sdd_constructor.build([])
    root.ref()
    mgr.minimize()
    return mgr, root

#
# mgr, root = create_model(2, 3, [1, 2])
# print("Initial:", mgr.count())
#
# from graphviz import Source
#
# # g = Source(root.dot())
# # g.render(view=True)
# root.ref()
# mgr.minimize()
# print("Optimize:", mgr.count())
# g = Source(root.dot())
# g.render(view=True)
