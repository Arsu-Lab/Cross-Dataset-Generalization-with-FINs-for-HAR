# WISDM -> MotionSense

Written by `cross_dataset.py w2m`; seeds [42, 0, 1, 2, 3], scored after the last epoch. Activity columns are recall, mean over seeds.

## Test: MotionSense

| model | accuracy | macro F1 | Jogging | Walking | Upstairs | Downstairs | Sitting | Standing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FIN + 40, pretrained | 0.9497 ± 0.0036 | 0.9438 ± 0.0032 | 0.997 | 0.983 | 0.928 | 0.847 | 0.920 | 0.974 |
| FIN + 40, random init | 0.9288 ± 0.0073 | 0.9280 ± 0.0079 | 0.994 | 0.978 | 0.940 | 0.851 | 0.851 | 0.958 |
| CNN + 40 statistical | 0.9464 ± 0.0041 | 0.9379 ± 0.0044 | 0.955 | 0.931 | 0.931 | 0.925 | 0.942 | 0.980 |

## Test: WISDM 27-36

| model | accuracy | macro F1 | Jogging | Walking | Upstairs | Downstairs | Sitting | Standing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FIN + 40, pretrained | 0.9336 ± 0.0024 | 0.8948 ± 0.0034 | 0.979 | 0.989 | 0.728 | 0.810 | 0.855 | 0.965 |
| FIN + 40, random init | 0.9307 ± 0.0026 | 0.8896 ± 0.0046 | 0.979 | 0.986 | 0.749 | 0.787 | 0.850 | 0.935 |
| CNN + 40 statistical | 0.9274 ± 0.0027 | 0.8881 ± 0.0041 | 0.979 | 0.982 | 0.682 | 0.853 | 0.812 | 0.963 |
