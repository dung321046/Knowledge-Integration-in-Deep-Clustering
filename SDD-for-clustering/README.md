# SDD-for-clustering

1. Constructing pairwise, triplet, relative and m-clusters group constraint

```bash
cd examples
python construct_constraint_formulation.py --type [pw,triplet, ...] --formu [A/B]
```
2. Generate implication constraints
```bash
cd generate_constraints
python generate_implication_horn.py 
```
3. Construct implication sdd
```bash
cd examples
python implication-a.py
python implication-b.py  
```

