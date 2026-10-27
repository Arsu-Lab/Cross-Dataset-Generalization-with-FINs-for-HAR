# WISDM -> MotionSense, early stopping

Written by `cross_dataset_early_stopping.py w2m`; seeds [42, 0, 1, 2, 3]. Validation: users [21, 22, 23, 24, 25, 26] of the training dataset; target: 5193 windows.

| model | accuracy, early stopping | macro F1, early stopping | accuracy, last epoch | stopping epoch |
|---|---:|---:|---:|---:|
| FIN + 40, pretrained | 0.9260 ± 0.0084 | 0.9092 ± 0.0145 | 0.9360 ± 0.0067 | 26, 27, 11, 42, 13 |
| FIN + 40, random init | 0.8950 ± 0.0061 | 0.8942 ± 0.0048 | 0.9028 ± 0.0073 | 15, 14, 17, 49, 13 |
| CNN + 40 statistical | 0.9308 ± 0.0064 | 0.9180 ± 0.0076 | 0.9254 ± 0.0111 | 35, 19, 52, 38, 57 |
