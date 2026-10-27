"""bilin_fin_temporal_plus_original.py with the FIN setup of the scarcity and cross-dataset sections: dropout 0.5
after the head's hidden layer; the pretrained FIN stays frozen for all epochs, the random-init FIN trains with the
head from the first epoch. Train users 1-26, test users 27-36, seeds 42, 0, 1, 2, 3, scored after the last epoch.
Writes temporal_dropout_results.md and the per-epoch test accuracy (temporal_dropout_curves.npz) in this folder.

    python bilin_fin_temporal_dropout.py
"""
import os
import sys

os.environ["KERAS_BACKEND"] = "torch"
HERE = os.path.dirname(os.path.abspath(__file__))
FIN = os.path.dirname(HERE)
sys.path[:0] = [HERE, FIN, os.path.join(FIN, "data-processing")]

import keras
import numpy as np
import torch
from keras import layers, ops
from sklearn.metrics import f1_score

import config as cfg
import wisdm_windowing as ww
from bilin_fin_temporal_plus_original import RhythmFIN, ignatov_features, targets

OUT = os.path.join(HERE, "temporal_dropout_results.md")
CURVES = os.path.join(HERE, "temporal_dropout_curves.npz")
MODELS = ("FIN + 40, pretrained (frozen), dropout 0.5", "FIN + 40, random init, dropout 0.5")


class Head(keras.Model):
    """FIN embedding of each axis + the 40 features -> Dense 128 -> dropout -> softmax."""

    def __init__(self, fin, dropout, **kw):
        super().__init__(**kw)
        self.fin = fin
        self.hidden = layers.Dense(cfg.HEAD_WIDTH, activation="relu")
        self.drop = layers.Dropout(dropout)
        self.out = layers.Dense(cfg.N_CLASSES, activation="softmax")

    def call(self, inputs, training=None):
        x, stats = inputs
        parts = [self.fin.embed(x[:, i, :]) for i in range(len(cfg.AXES))]
        return self.out(self.drop(self.hidden(ops.concatenate(parts + [stats], axis=1)), training=training))


class TestAccuracy(keras.callbacks.Callback):
    """Test accuracy after every epoch; restores the RNG so training is unaffected."""

    def __init__(self, x, y):
        super().__init__()
        self.x, self.y, self.curve = x, y, []

    def on_epoch_end(self, epoch, logs=None):
        rng = torch.random.get_rng_state()
        self.curve.append(float(np.mean(self.model.predict(self.x, batch_size=1024, verbose=0).argmax(1) == self.y)))
        torch.random.set_rng_state(rng)


def main():
    data = ww.load_splits()
    xtr = np.concatenate([data["train"][0], data["val"][0]])  # users 1-26
    ytr = np.concatenate([data["train"][1], data["val"][1]])
    xte, yte = data["test"]                                     # users 27-36

    ftr, fte = ignatov_features(xtr), ignatov_features(xte)
    mu, sd = ftr.mean(0), ftr.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    ftr, fte = ((ftr - mu) / sd).astype(np.float32), ((fte - mu) / sd).astype(np.float32)
    axes = xtr.reshape(-1, cfg.SEGMENT)
    y = targets(axes)
    y = ((y - y.mean(0)) / y.std(0)).astype(np.float32)

    rows, curves = {m: [] for m in MODELS}, {m: [] for m in MODELS}
    for seed in cfg.SEEDS:
        keras.utils.set_random_seed(seed)
        pretrained = RhythmFIN()
        pretrained.compile(optimizer=keras.optimizers.Adam(cfg.FIN_LR), loss="mse")
        pretrained.fit(axes, y, epochs=cfg.FIN_EPOCHS, batch_size=cfg.BATCH_SIZE, verbose=0)
        pretrained.trainable = False

        keras.utils.set_random_seed(seed)
        random = RhythmFIN()
        random(np.zeros((1, cfg.SEGMENT), np.float32))

        for name, fin in zip(MODELS, (pretrained, random)):
            keras.utils.set_random_seed(seed)
            model = Head(fin, 0.5)
            model.compile(optimizer=keras.optimizers.Adam(cfg.HEAD_LR), loss="sparse_categorical_crossentropy")
            track = TestAccuracy([xte, fte], yte)
            model.fit([xtr, ftr], ytr, epochs=cfg.HEAD_EPOCHS, batch_size=cfg.BATCH_SIZE, verbose=0, callbacks=[track])
            curves[name].append(track.curve)
            pred = model.predict([xte, fte], batch_size=1024, verbose=0).argmax(axis=1)
            acc, f1 = np.mean(pred == yte), f1_score(yte, pred, average="macro")
            per_class = f1_score(yte, pred, average=None, labels=range(cfg.N_CLASSES))
            rows[name].append([acc, f1, *per_class])
            print(f"seed {seed} {name}: accuracy {acc:.4f}, macro F1 {f1:.4f}", flush=True)

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("# Temporal FIN with dropout, train users 1-26, test users 27-36\n\n"
                 f"Written by `bilin_fin_temporal_dropout.py`; seeds {list(cfg.SEEDS)}, {len(yte)} test windows, "
                 f"{cfg.HEAD_EPOCHS} epochs, scored after the last epoch. Activity columns are F1, mean over seeds.\n\n"
                 "| model | accuracy | macro F1 | " + " | ".join(cfg.ACTIVITIES) + " |\n"
                 "|---|---:|---:|" + "---:|" * cfg.N_CLASSES + "\n")
        for name, r in rows.items():
            r = np.array(r)
            m, s = r.mean(0), r.std(0, ddof=1)
            fh.write(f"| {name} | {m[0]:.4f} ± {s[0]:.4f} | {m[1]:.4f} ± {s[1]:.4f} | "
                     + " | ".join(f"{v:.4f}" for v in m[2:]) + " |\n")
    np.savez(CURVES, seeds=np.array(cfg.SEEDS), pretrained=np.array(curves[MODELS[0]]),
             random_init=np.array(curves[MODELS[1]]))
    print("wrote", OUT, CURVES)


if __name__ == "__main__":
    main()
