# Temporal FIN with dropout, train users 1-26, test users 27-36

Written by `bilin_fin_temporal_dropout.py`; seeds [42, 0, 1, 2, 3], 3017 test windows, 60 epochs, scored after the last epoch. Activity columns are F1, mean over seeds.

| model | accuracy | macro F1 | Jogging | Walking | Upstairs | Downstairs | Sitting | Standing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FIN + 40, pretrained (frozen), dropout 0.5 | 0.9314 ± 0.0031 | 0.8900 ± 0.0052 | 0.9768 | 0.9650 | 0.7876 | 0.8476 | 0.8879 | 0.8748 |
| FIN + 40, random init, dropout 0.5 | 0.9285 ± 0.0028 | 0.8850 ± 0.0058 | 0.9722 | 0.9630 | 0.7896 | 0.8570 | 0.8678 | 0.8607 |
