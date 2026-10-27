"""Cross-dataset test with early stopping on held-out users of the training dataset (no target labels are used).

m2w: trained on MotionSense users 1-20, early stopping on MotionSense users 21-24, tested on WISDM users 27-36.
w2m: trained on WISDM users 1-20, early stopping on WISDM users 21-26, tested on all 24 MotionSense users.
The training windows are used with their flipped copies (180 degrees about y), the validation windows as they are.
Same models and training as cross_dataset.py (FIN + 40 pretrained: frozen; FIN + 40 random init: trained with the
head; both with dropout 0.5; Ignatov's CNN as in his code), up to 60 epochs; each model is scored on the target at
the epoch with the best validation accuracy and, for comparison, after the last epoch.

    python cross_dataset_early_stopping.py m2w     # writes results/early_stopping_m2w.md
    python cross_dataset_early_stopping.py w2m
"""
import os
import sys

os.environ["KERAS_BACKEND"] = "torch"
HERE = os.path.dirname(os.path.abspath(__file__))
FIN = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, FIN, os.path.join(FIN, "FIN-classifiers"), os.path.join(FIN, "data-processing")]

import argparse

import numpy as np

import config as cfg
from cross_dataset import CNN_LR, DIRECTIONS, FLIP, HEAD_DROPOUT, MODELS, OUT

VAL_USERS = {"m2w": range(21, 25), "w2m": range(21, 27)}


def load(direction):
    """(train windows with flipped copies, labels), (validation windows, labels), (target windows, labels)."""
    import motionsense_windowing as ms
    import wisdm_windowing as ww

    w_windows, w_users, w_labels = ww.window_raw()
    m_windows, m_users, m_labels = ms.window_raw()
    if direction == "m2w":
        src, target = (m_windows, m_users, m_labels), (w_windows[w_users >= 27], w_labels[w_users >= 27])
    else:
        src, target = (w_windows[w_users <= 26], w_users[w_users <= 26], w_labels[w_users <= 26]), (m_windows, m_labels)
    x, users, y = src
    val = np.isin(users, VAL_USERS[direction])
    xtr, ytr = x[~val], y[~val]
    return (np.concatenate([xtr, xtr * FLIP]), np.concatenate([ytr, ytr])), (x[val], y[val]), target


def run(direction, seeds):
    import keras
    import torch
    from keras import layers, ops
    from sklearn.metrics import f1_score

    from bilin_fin_temporal_plus_original import RhythmEnsemble, RhythmFIN, ignatov_features, targets

    class Track(keras.callbacks.Callback):
        """Validation and target accuracy after every epoch and the weights of the best validation epoch."""

        def __init__(self, val, target):
            super().__init__()
            self.val, self.target = val, target
            self.acc = {"val": [], "target": []}
            self.best_weights = None

        def on_epoch_end(self, epoch, logs=None):
            rng = torch.random.get_rng_state()
            for name, (x, y) in (("val", self.val), ("target", self.target)):
                self.acc[name].append(float(np.mean(self.model.predict(x, batch_size=1024, verbose=0).argmax(1) == y)))
            torch.random.set_rng_state(rng)
            if self.acc["val"][-1] > max(self.acc["val"][:-1], default=-1):
                self.best_weights = self.model.get_weights()

    class DropoutEnsemble(RhythmEnsemble):
        def __init__(self, fin, **kw):
            super().__init__(fin, **kw)
            self.drop = layers.Dropout(HEAD_DROPOUT)

        def call(self, inputs, training=None):
            x, stats = inputs
            h = ops.concatenate([self.fin.embed(x[:, i, :]) for i in range(len(cfg.AXES))] + [stats], axis=1)
            for layer in self.hidden:
                h = self.drop(layer(h), training=training)
            return self.out(h)

    def ignatov_cnn(n_features):
        init = dict(kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.01),
                    bias_initializer=keras.initializers.Constant(0.01))
        l2 = keras.regularizers.L2(5e-4 / 2 / 200)
        signal = keras.Input((cfg.SEGMENT, len(cfg.AXES)))
        features = keras.Input((n_features,))
        h = layers.Conv1D(196, 12, padding="same", activation="relu", **init)(signal)
        h = layers.MaxPooling1D(4, padding="same")(h)
        h = layers.Concatenate()([layers.Flatten()(h), features])
        h = layers.Dense(1024, activation="relu", kernel_regularizer=l2, bias_regularizer=l2, **init)(h)
        h = layers.Dropout(0.85)(h)
        return keras.Model([signal, features], layers.Dense(cfg.N_CLASSES, activation="softmax", **init)(h))

    def centred(x):  # as Ignatov's code: per-axis mean removed, hstack(x, y, z) reshaped to (time, 3)
        x = x - x.mean(axis=2, keepdims=True)
        return x.reshape(len(x), cfg.SEGMENT, len(cfg.AXES)).astype(np.float32)

    (xtr, ytr), (xva, yva), (xte, yte) = load(direction)
    ftr = ignatov_features(xtr)
    mu, sd = ftr.mean(0), np.where(ftr.std(0) < 1e-8, 1.0, ftr.std(0))
    std = lambda f: ((f - mu) / sd).astype(np.float32)
    ftr, fva, fte = std(ftr), std(ignatov_features(xva)), std(ignatov_features(xte))
    print(f"{DIRECTIONS[direction]}: training on {len(xtr)} windows (with the flipped copies), validation "
          f"{len(xva)} windows of users {list(VAL_USERS[direction])}, target {len(xte)} windows", flush=True)

    rows = {m: [] for m in MODELS}
    for seed in seeds:
        for name in MODELS:
            keras.utils.set_random_seed(seed)
            if name == "ignatov_cnn":
                model, lr, inp = ignatov_cnn(ftr.shape[1]), CNN_LR, lambda x, f: [centred(x), f]
            else:
                fin = RhythmFIN()
                if name == "fin_frozen_dropout":
                    axes = xtr.reshape(-1, cfg.SEGMENT)
                    y = targets(axes)
                    fin.compile(optimizer=keras.optimizers.Adam(cfg.FIN_LR), loss="mse")
                    fin.fit(axes, ((y - y.mean(0)) / y.std(0)).astype(np.float32), epochs=cfg.FIN_EPOCHS,
                            batch_size=cfg.BATCH_SIZE, verbose=0)
                    fin.trainable = False
                keras.utils.set_random_seed(seed)
                model, lr, inp = DropoutEnsemble(fin), cfg.HEAD_LR, lambda x, f: [x, f]
            track = Track((inp(xva, fva), yva), (inp(xte, fte), yte))
            model.compile(optimizer=keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy")
            model.fit(inp(xtr, ftr), ytr, epochs=cfg.HEAD_EPOCHS, batch_size=cfg.BATCH_SIZE, verbose=0,
                      callbacks=[track])
            last = model.predict(inp(xte, fte), batch_size=1024, verbose=0).argmax(1)
            model.set_weights(track.best_weights)
            best = model.predict(inp(xte, fte), batch_size=1024, verbose=0).argmax(1)
            epoch = int(np.argmax(track.acc["val"])) + 1
            rows[name].append((np.mean(best == yte), f1_score(yte, best, average="macro"), np.mean(last == yte), epoch))
            print(f"seed {seed}, {name}: early stopping at epoch {epoch}: {rows[name][-1][0]:.4f}, "
                  f"last epoch {rows[name][-1][2]:.4f}", flush=True)

    ms = lambda v: f"{np.mean(v):.4f} ± {np.std(v, ddof=1):.4f}"
    lines = [f"# {DIRECTIONS[direction]}, early stopping", "",
             f"Written by `cross_dataset_early_stopping.py {direction}`; seeds {seeds}. Validation: users "
             f"{list(VAL_USERS[direction])} of the training dataset; target: {len(yte)} windows.", "",
             "| model | accuracy, early stopping | macro F1, early stopping | accuracy, last epoch | stopping epoch |",
             "|---|---:|---:|---:|---:|"]
    for name, r in rows.items():
        acc, f1, last, ep = zip(*r)
        lines.append(f"| {MODELS[name][0]} | {ms(acc)} | {ms(f1)} | {ms(last)} | {', '.join(map(str, ep))} |")
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, f"early_stopping_{direction}.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("saved", out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("direction", choices=list(DIRECTIONS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(cfg.SEEDS))
    args = parser.parse_args()
    run(args.direction, args.seeds)


if __name__ == "__main__":
    main()
