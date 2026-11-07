# MotionSense -> WISDM

Written by `cross_dataset.py m2w`; seeds [42, 0, 1, 2, 3], scored after the last epoch. Activity columns are recall, mean over seeds.

## Test: WISDM, all users

| model | accuracy | macro F1 | Jogging | Walking | Upstairs | Downstairs | Sitting | Standing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FIN + 40, pretrained | 0.8766 ± 0.0208 | 0.8318 ± 0.0140 | 0.923 | 0.893 | 0.694 | 0.787 | 0.942 | 0.897 |
| FIN + 40, random init | 0.8100 ± 0.0065 | 0.7611 ± 0.0055 | 0.912 | 0.774 | 0.760 | 0.521 | 0.943 | 0.870 |
| CNN + 40 statistical | 0.7262 ± 0.0460 | 0.7409 ± 0.0245 | 0.712 | 0.667 | 0.805 | 0.766 | 0.924 | 0.873 |

## Test: WISDM 27-36

| model | accuracy | macro F1 | Jogging | Walking | Upstairs | Downstairs | Sitting | Standing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FIN + 40, pretrained | 0.8709 ± 0.0294 | 0.8248 ± 0.0199 | 0.916 | 0.904 | 0.690 | 0.755 | 0.868 | 0.922 |
| FIN + 40, random init | 0.8038 ± 0.0059 | 0.7661 ± 0.0053 | 0.936 | 0.757 | 0.730 | 0.547 | 0.888 | 0.901 |
| CNN + 40 statistical | 0.7122 ± 0.0511 | 0.7343 ± 0.0228 | 0.676 | 0.655 | 0.793 | 0.786 | 0.857 | 0.890 |

## Test: MotionSense 19-24

| model | accuracy | macro F1 | Jogging | Walking | Upstairs | Downstairs | Sitting | Standing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FIN + 40, pretrained | 0.9555 ± 0.0055 | 0.9408 ± 0.0058 | 0.987 | 0.912 | 0.960 | 0.806 | 0.999 | 0.995 |
| FIN + 40, random init | 0.9316 ± 0.0041 | 0.9166 ± 0.0044 | 0.982 | 0.859 | 0.908 | 0.777 | 0.996 | 0.989 |
| CNN + 40 statistical | 0.9328 ± 0.0149 | 0.9163 ± 0.0124 | 0.977 | 0.848 | 0.934 | 0.819 | 0.999 | 0.978 |
