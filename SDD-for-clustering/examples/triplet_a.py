from sdd_clustering import model_a

n = 3
k = 3
mgr, root = model_a.clustering_model(n, k, [], -1, True)
root = root.conjoin(model_a.f_must_link(0, 1, k, mgr).disjoin(model_a.f_cannot_link(0, 2, k, mgr)))
root.ref()
mgr.minimize()
from graphviz import Source

g = Source(root.dot())
g.render(view=True)
vtree = mgr.vtree()
vtree.save(bytes("triplet-k-" + str(k) + ".vtree", encoding="utf8"))
mgr.save(bytes("triplet-k-" + str(k), encoding="utf8"), root)
