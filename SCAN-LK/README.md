# SCAN-LK


## Pw constraints:

1. Generate

```
python generate_pw.py --config_env configs/env.yml --config_exp configs/scan/scan_cifar10.yml
python generate_pw.py --config_env configs/env.yml --config_exp configs/scan/scan_stl10.yml
```

2. Run model

```
python scan-pw.py --config_env configs/env.yml --config_exp configs/scan/scan_cifar10.yml
python scan-pw.py --config_env configs/env.yml --config_exp configs/scan/scan_stl10.yml
```

## Span-limited constraints:

1. Generate and run model

```
python scan-train-span.py --config_env configs/env.yml --config_exp configs/scan/scan_stl10.yml
```
