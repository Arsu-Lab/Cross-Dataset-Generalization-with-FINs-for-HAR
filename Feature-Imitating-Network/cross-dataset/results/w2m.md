# WISDM -> MotionSense

Written by `cross_dataset.py w2m`; seeds [42, 0, 1, 2, 3], scored after the last epoch. Activity columns are recall, mean over seeds.

## Test: MotionSense

| model | accuracy | macro F1 | Jogging | Walking | Upstairs | Downstairs | Sitting | Standing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FIN + 40, pretrained | 0.9497 ± 0.0036 | 0.9438 ± 0.0032 | 0.997 | 0.983 | 0.928 | 0.847 | 0.920 | 0.974 |
| FIN + 40, random init | 0.9272 ± 0.0093 | 0.9285 ± 0.0086 | 0.996 | 0.976 | 0.939 | 0.870 | 0.836 | 0.963 |
| CNN + 40 statistical | 0.9452 ± 0.0039 | 0.9367 ± 0.0047 | 0.946 | 0.929 | 0.938 | 0.924 | 0.940 | 0.980 |

## Test: WISDM 27-36

| model | accuracy | macro F1 | Jogging | Walking | Upstairs | Downstairs | Sitting | Standing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FIN + 40, pretrained | 0.9336 ± 0.0024 | 0.8948 ± 0.0034 | 0.979 | 0.989 | 0.728 | 0.810 | 0.855 | 0.965 |
| FIN + 40, random init | 0.9315 ± 0.0038 | 0.8907 ± 0.0065 | 0.978 | 0.987 | 0.750 | 0.797 | 0.842 | 0.939 |
| CNN + 40 statistical | 0.9289 ± 0.0035 | 0.8913 ± 0.0047 | 0.978 | 0.981 | 0.707 | 0.849 | 0.812 | 0.963 |
