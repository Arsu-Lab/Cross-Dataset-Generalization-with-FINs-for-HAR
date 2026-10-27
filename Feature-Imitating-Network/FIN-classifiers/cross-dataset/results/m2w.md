# MotionSense -> WISDM

Written by `cross_dataset.py m2w`; seeds [42, 0, 1, 2, 3], scored after the last epoch. Activity columns are recall, mean over seeds.

## Test: WISDM 27-36

| model | accuracy | macro F1 | Jogging | Walking | Upstairs | Downstairs | Sitting | Standing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FIN + 40, pretrained | 0.8709 ± 0.0294 | 0.8248 ± 0.0199 | 0.916 | 0.904 | 0.690 | 0.755 | 0.868 | 0.922 |
| FIN + 40, random init | 0.8038 ± 0.0059 | 0.7661 ± 0.0053 | 0.936 | 0.757 | 0.730 | 0.547 | 0.888 | 0.901 |
| CNN + 40 statistical | 0.7399 ± 0.0269 | 0.7447 ± 0.0167 | 0.691 | 0.718 | 0.758 | 0.804 | 0.854 | 0.896 |

## Test: MotionSense 19-24

| model | accuracy | macro F1 | Jogging | Walking | Upstairs | Downstairs | Sitting | Standing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FIN + 40, pretrained | 0.9555 ± 0.0055 | 0.9408 ± 0.0058 | 0.987 | 0.912 | 0.960 | 0.806 | 0.999 | 0.995 |
| FIN + 40, random init | 0.9316 ± 0.0041 | 0.9166 ± 0.0044 | 0.982 | 0.859 | 0.908 | 0.777 | 0.996 | 0.989 |
| CNN + 40 statistical | 0.9473 ± 0.0126 | 0.9296 ± 0.0147 | 0.982 | 0.906 | 0.906 | 0.844 | 1.000 | 0.980 |
