from sklearn import preprocessing

from sdd_clustering import model_b
from sdd_clustering.utils import *

num_triplet = 4
n = 1 + num_triplet * 2
k = 4
p = np.random.random((n, k))
p = preprocessing.normalize(p, axis=1, norm="l1")
print(p)
mgr, root = model_b.clustering_model(n, k, [], -1)
# all_disjoin = model_b.f_cannot_link(0, 1, k, mgr).conjoin(
#     model_b.f_cannot_link(1, 2, k, mgr).conjoin(model_b.f_cannot_link(2, 0, k, mgr)))
# root = mgr.disjoin(model_b.f_must_link(0, 1, k, mgr), all_disjoin)
root = mgr.true()
for i in range(num_triplet):
    root = root.conjoin(
        mgr.disjoin(model_b.f_must_link(0, i * 2 + 1, k, mgr), model_b.f_cannot_link(0, i * 2 + 2, k, mgr)))
root.ref()
mgr.minimize()
probb = ProbCalculator(weight_convert_b(p, n, k))
pb = probb.calculate(root)
print("Prob:     ", pb)
ans = 0.0

for i in range(k):
    tmp = 1.0
    for t in range(num_triplet):
        tmp *= p[t * 2 + 1][i] + (1 - p[t * 2 + 1][i]) * (1 - p[t * 2 + 2][i])
    ans += p[0][i] * tmp
print("Prob-true:", ans)
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
vtree.save(bytes("group-" + str(num_triplet) + "-triplet-k-10b.vtree", encoding="utf8"))
mgr.save(bytes("group-" + str(num_triplet) + "-triplet-k-10b", encoding="utf8"), root)
