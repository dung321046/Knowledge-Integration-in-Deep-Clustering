## Stacked Denoising Autoencoder (SDAE)

1. Run SDAE (create 5 models)
```bash
python run_sdae.py --data [MNIST, Fashion, Reuters]
```
2. Run and report K-means using SDAE
```bash
python evaluate_sdae_kmeans.py --data [MNIST, Fashion, Reuters]
```
3. Select the first model and move it to Deep-Clustering-by-Constraint-Score-Maximisation/model
```bash
cd ae_experiments
cp sdae_[data]/0.pt ../model/sdae_[data]_weights.pt
```
4. Report and check performance of ```../model/sdae_[data]_weights.pt``` models (pretrains for Cop-Kmeans, DCC, and EDEC) on all three dataset
```bash
python evaluate_pretrain_models.py
```

## Improved Deep Embedded Clustering (IDEC) 
1. Run IDEC with SDAE pretrains
```bash
python evaluate_pretrain_models.py
```
2. Report
Results provide by author at: https://www.overleaf.com/read/nqtkxfnwhsqp

