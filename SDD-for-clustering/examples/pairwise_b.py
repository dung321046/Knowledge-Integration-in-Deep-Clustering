from sklearn import preprocessing

from sdd_clustering import model_b
from sdd_clustering.utils import *

n = 2
k = 10
p = np.random.random((n, k))
p = preprocessing.normalize(p, axis=1, norm="l1")
print(p)
mgr, root = model_b.clustering_model(n, k, [], -1)
# all_disjoin = model_b.f_cannot_link(0, 1, k, mgr).conjoin(
#     model_b.f_cannot_link(1, 2, k, mgr).conjoin(model_b.f_cannot_link(2, 0, k, mgr)))
# root = mgr.disjoin(model_b.f_must_link(0, 1, k, mgr), all_disjoin)
# root = mgr.disjoin(model_b.f_must_link(0, 1, k, mgr), model_b.f_cannot_link(2, 0, k, mgr))
root = root.conjoin(model_b.f_cannot_link(0, 1, k, mgr))
root.ref()
mgr.minimize()
probb = ProbCalculator(weight_convert_b(p, n, k))
pb = probb.calculate(root)
print("Prob:     ", pb)
ans = 0.0
t = 0.0
for i in range(k):
    ans += p[0][i] * p[1][i]
    t += p[0][i] * p[1][i]
    # for j in range(k):
    #     if i != j:
    #         ans += p[0][i] * p[1][j]  * (1 - p[2][i] - p[2][j])
# u = 0.0
# for i in range(k):
#     for j in range(k):
#         for z in range(k):
#             # if (i != j) and (j != z) and (z != i):
#             #     u += p[0][i] * p[1][j] * p[2][z]
#             if (i != j) and (z != i):
#                 u += p[0][i] * p[1][j] * p[2][z]
#                 ans += p[0][i] * p[1][j] * p[2][z]
print("Prob-true:", ans)
print("t        :", t)
# print("u        :", u)
#
# mgr, root = model_b.clustering_model(n, k, [], -1)
# all_disjoin = model_b.f_cannot_link(0, 1, k, mgr).conjoin(
#     model_b.f_cannot_link(1, 2, k, mgr).conjoin(model_b.f_cannot_link(2, 0, k, mgr)))
# all_disjoin.ref()
# mgr.minimize()
# probb = ProbCalculator(weight_convert_b(p, n, k))
# pb = probb.calculate(all_disjoin)
#
# print("u: ", pb)
#
# mgr, root = model_b.clustering_model(n, k, [], -1)
# root = model_b.f_must_link(0, 1, k, mgr)
# root.ref()
# mgr.minimize()
# probb = ProbCalculator(weight_convert_b(p, n, k))
# pb = probb.calculate(root)
# print("t: ", pb)
# from graphviz import Source
#
# g = Source(root.dot())
# g.render(view=True)
vtree = mgr.vtree()
vtree.save(bytes("cl-k-10b.vtree", encoding="utf8"))
mgr.save(bytes("cl-k-10b", encoding="utf8"), root)
