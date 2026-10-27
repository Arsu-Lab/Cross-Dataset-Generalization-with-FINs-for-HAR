# Where the paper's numbers come from

Ignatov's CNN everywhere = as his code runs it (hstack(x, y, z) reshaped to (200, 3), dropout 85 %).
FIN everywhere: the pretrained FIN stays frozen, the random-init FIN is trained with the head from the first epoch,
both heads have dropout 0.5.

## Generalization to unseen subjects (scarcity)

7 random training users, 2 validation, 27 test; seeds 0-4.

| paper item | file |
|---|---|
| `tab:scarcity-classical` (RF, XGBoost, log. regression, 1-NN, SVM; 40 stat. / 3 FIN feat.) | `data-scarcity/baselines_7_users.ipynb` |
| `tab:scarcity-neural` (CNN + 40 / + 3 FIN feat., FIN and FIN + 40 pretrained / random init) | `data-scarcity/data_scarcity.ipynb`, table cell, column "accuracy, last epoch" |
| `fig:scarcity-curves` (`scarcity_panels.eps`) | last cell of `data_scarcity.ipynb`: writes `data-scarcity/data_scarcity_curves.npz` and `data-scarcity/scarcity_panels.*` (`plot_panels` of `paper-figures/scarcity_curves.py`) |

## Generalization to other datasets (cross-dataset)

W→M: WISDM users 1-26 → all 24 MotionSense users; M→W: all MotionSense → WISDM users 27-36; flipped copies in
training; 5 seeds (42, 0, 1, 2, 3).

| paper item | file |
|---|---|
| `tab:cross-dataset` / `tab:w2m-classical`: random forest, XGBoost, log. regression, 1-NN (40 stat. / 33 temp.) | `baseline/results.md` (scripts in `baseline/cross-dataset/`) |
| `tab:cross-dataset` / `tab:w2m-neural`: CNN (+40 stat.), FIN + 40 pretrained, FIN + 40 random init | `FIN-classifiers/cross-dataset/results/{w2m,m2w}.md`, by `python cross_dataset.py w2m` / `m2w` |
| `fig:m2w-curves` (`m2w_curves.eps`), `w2m_curves`, `*_confusion` | the same runs, `FIN-classifiers/cross-dataset/results/` |
