# MotionSense -> WISDM, early stopping

Written by `cross_dataset_early_stopping.py m2w`; seeds [42, 0, 1, 2, 3]. Validation: users [21, 22, 23, 24] of the training dataset; target: 3017 windows.

| model | accuracy, early stopping | macro F1, early stopping | accuracy, last epoch | stopping epoch |
|---|---:|---:|---:|---:|
| FIN + 40, pretrained | 0.8644 ± 0.0247 | 0.8171 ± 0.0151 | 0.8579 ± 0.0260 | 35, 23, 54, 45, 23 |
| FIN + 40, random init | 0.8618 ± 0.0259 | 0.8056 ± 0.0168 | 0.8144 ± 0.0146 | 11, 9, 45, 20, 7 |
| CNN + 40 statistical | 0.8298 ± 0.0133 | 0.7991 ± 0.0074 | 0.7578 ± 0.0504 | 26, 54, 13, 60, 58 |
