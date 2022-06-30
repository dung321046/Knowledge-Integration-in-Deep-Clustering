# Deep-Clustering-by-Constraint-Score-Maximisation

The formulation given in the paper is the formulation B.

## Enviroment
```bash
pip install requirements.txt
```

## Models
See: ae_experiments/README.md
https://drive.google.com/file/d/10YNJZhoou2CiRkkr7gR6J-486pMAKsXx/view?usp=sharing

## Pairwise
1. Generate constraints
```bash
cd generate_constraints
python generate_pairwise.py --data [MNIST/Fashion/Reuters]
```
2. Run experiments
Run five times over five constraint sets 
```bash
cd pw_experiments
python run_edec.py --data [MNIST/Fashion/Reuters] --formu [A/B]
```
Run five times over one constraint sets 
```bash
cd pw_experiments
python run_edec_5seed.py --data [MNIST/Fashion/Reuters] --formu [A/B]
```
3. Run reports
```bash
cd reports
python pairwise_report.py
```

## Triplet
Similar to pairwise, the files are in ./triplet_experiments folder.

## m-clusters group
1. Generate constraints
```bash
cd mcluster_experiments
python generate-group-2-clusters.py
```
2. Run 
```bash
cd mcluster_experiments
python run_m_clusters.py
```
3. Report
```bash
cd mcluster_experiments
python run_neighbor_after_training.py
```
## Implication
1. Generate: see instruction in SDD-for-clustering
2. Run performance of IDEC and EDEC, respectively.
```bash
cd experiments
python run_baseline_implication.py
python run_edec_implication.py
```

