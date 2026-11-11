# MotionSense -> WISDM

Written by `cross_dataset.py m2w`; seeds [42, 0, 1, 2, 3], scored after the last epoch. Activity columns are recall, mean over seeds.

## Test: WISDM, all users

| model | accuracy | macro F1 | Jogging | Walking | Upstairs | Downstairs | Sitting | Standing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FIN + 40, pretrained | 0.8766 ± 0.0208 | 0.8318 ± 0.0140 | 0.923 | 0.893 | 0.694 | 0.787 | 0.942 | 0.897 |
| FIN + 40, random init | 0.8100 ± 0.0065 | 0.7611 ± 0.0055 | 0.912 | 0.774 | 0.760 | 0.521 | 0.943 | 0.870 |
| CNN + 40 statistical | 0.7488 ± 0.0343 | 0.7523 ± 0.0208 | 0.737 | 0.711 | 0.808 | 0.718 | 0.925 | 0.883 |

## Test: MotionSense 19-24

| model | accuracy | macro F1 | Jogging | Walking | Upstairs | Downstairs | Sitting | Standing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FIN + 40, pretrained | 0.9555 ± 0.0055 | 0.9408 ± 0.0058 | 0.987 | 0.912 | 0.960 | 0.806 | 0.999 | 0.995 |
| FIN + 40, random init | 0.9316 ± 0.0041 | 0.9166 ± 0.0044 | 0.982 | 0.859 | 0.908 | 0.777 | 0.996 | 0.989 |
| CNN + 40 statistical | 0.9414 ± 0.0047 | 0.9241 ± 0.0044 | 0.980 | 0.882 | 0.938 | 0.806 | 1.000 | 0.980 |
