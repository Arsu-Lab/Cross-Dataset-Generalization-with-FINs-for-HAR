
import os
import sys

os.environ["KERAS_BACKEND"] = "torch"
HERE = os.path.dirname(os.path.abspath(__file__))
FIN = os.path.dirname(HERE)
sys.path[:0] = [FIN, os.path.join(FIN, "FIN-classifiers"), os.path.join(FIN, "data-processing")]

import argparse

import numpy as np

import config as cfg

FORMATS = ("eps",)

SEEDS = (0, 1, 2, 3, 4)
N_TRAIN, N_VAL = 7, 2
EPOCHS = 150
N_SYNTH = 30000
CURVES = os.path.join(HERE, "scarcity_curves.npz")
NETWORKS = ("fin_random", "fin_warped", "fin_real_warped", "ignatov_cnn", "stats_random", "stats_real_warped")
TREES = ("random_forest", "xgboost")


def train(epochs, n_synth, seeds, path):
    import keras
    import torch
    from keras import layers, ops
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier

    import wisdm_windowing as ww
    from augmentation import make
    from bilin_fin_temporal_plus_original import RhythmEnsemble, RhythmFIN, ignatov_features, targets

    class TestAccuracy(keras.callbacks.Callback):
        """Test accuracy after every epoch; restores the RNG so training is unaffected."""

        def __init__(self, x, y):
            super().__init__()
            self.x, self.y, self.curve = x, y, []

        def on_epoch_end(self, epoch, logs=None):
            rng = torch.random.get_rng_state()
            pred = self.model.predict(self.x, batch_size=1024, verbose=0).argmax(axis=1)
            torch.random.set_rng_state(rng)
            self.curve.append(float(np.mean(pred == self.y)))

    class FINClassifier(keras.Model):
        """FIN embedding of x, y and z -> Dense 128 -> softmax."""

        def __init__(self, fin, **kw):
            super().__init__(**kw)
            self.fin = fin
            self.hidden = layers.Dense(cfg.HEAD_WIDTH, activation="relu")
            self.out = layers.Dense(cfg.N_CLASSES, activation="softmax")

        def call(self, x):
            h = ops.concatenate([self.fin.embed(x[:, i, :]) for i in range(len(cfg.AXES))], axis=1)
            return self.out(self.hidden(h))

    def ignatov_cnn(n_features):
        """Earlier port of Ignatov's CNN (ClassicML/cnn_wisdm.py) with a (time, axis) input and 15 % dropout; the
        paper uses his code's layout and dropout, see data-scarcity/data_scarcity.ipynb."""
        init = dict(kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.01),
                    bias_initializer=keras.initializers.Constant(0.01))
        l2 = keras.regularizers.L2(5e-4 / 2 / 200)  # his summed loss over a batch of 200, per window
        signal = keras.Input((cfg.SEGMENT, len(cfg.AXES)))
        features = keras.Input((n_features,))
        h = layers.Conv1D(196, 12, padding="same", activation="relu", **init)(signal)
        h = layers.MaxPooling1D(4, padding="same")(h)
        h = layers.Concatenate()([layers.Flatten()(h), features])
        h = layers.Dense(1024, activation="relu", kernel_regularizer=l2, bias_regularizer=l2, **init)(h)
        h = layers.Dropout(0.15)(h)
        return keras.Model([signal, features], layers.Dense(cfg.N_CLASSES, activation="softmax", **init)(h))

    def pretrain(axes, seed):
        keras.utils.set_random_seed(seed)
        fin = RhythmFIN()
        y = targets(axes)
        fin.compile(optimizer=keras.optimizers.Adam(cfg.FIN_LR), loss="mse")
        fin.fit(axes, ((y - y.mean(0)) / y.std(0)).astype(np.float32),
                epochs=cfg.FIN_EPOCHS, batch_size=cfg.BATCH_SIZE, verbose=0)
        return fin.get_weights()

    def fin(weights, seed):
        keras.utils.set_random_seed(seed)
        model = RhythmFIN()
        if weights is not None:
            model(np.zeros((1, cfg.SEGMENT), np.float32))
            model.set_weights(weights)
            model.trainable = False
        return model

    def curve(model, x, lr, tr, te):
        pick = (lambda m: [a[m] for a in x]) if isinstance(x, list) else (lambda m: x[m])
        track = TestAccuracy(pick(te), labels[te])
        model.compile(optimizer=keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy")
        model.fit(pick(tr), labels[tr], epochs=epochs, batch_size=cfg.BATCH_SIZE, verbose=0, callbacks=[track])
        return track.curve

    windows, users, labels = ww.window_raw()
    candidates = np.setdiff1d(np.unique(users), [30])
    stats = ignatov_features(windows)
    xyz = windows.transpose(0, 2, 1)
    centred = (xyz - xyz.mean(axis=1, keepdims=True)).astype(np.float32)  # Ignatov's per-window, per-axis mean removal
    out = {name: [] for name in NETWORKS + TREES}
    for seed in seeds:
        picked = np.random.default_rng(seed).choice(candidates, N_TRAIN + N_VAL, replace=False)
        tr, va = np.isin(users, picked[:N_TRAIN]), np.isin(users, picked[N_TRAIN:])
        te = ~(tr | va)
        real = windows[tr].reshape(-1, cfg.SEGMENT)
        warp = make(n_synth, seed, real)
        w_warped = pretrain(warp, seed)
        w_real_warped = pretrain(np.concatenate([real, warp]), seed)
        mu, sd = stats[tr].mean(0), stats[tr].std(0)
        f = ((stats - mu) / np.where(sd < 1e-8, 1.0, sd)).astype(np.float32)

        builders = {
            "fin_random": lambda: (FINClassifier(fin(None, seed)), windows, cfg.HEAD_LR),
            "fin_warped": lambda: (FINClassifier(fin(w_warped, seed)), windows, cfg.HEAD_LR),
            "fin_real_warped": lambda: (FINClassifier(fin(w_real_warped, seed)), windows, cfg.HEAD_LR),
            "ignatov_cnn": lambda: (ignatov_cnn(f.shape[1]), [centred, f], 5e-4),
            "stats_random": lambda: (RhythmEnsemble(fin(None, seed)), [windows, f], cfg.HEAD_LR),
            "stats_real_warped": lambda: (RhythmEnsemble(fin(w_real_warped, seed)), [windows, f], cfg.HEAD_LR),
        }
        for name, build in builders.items():
            keras.utils.set_random_seed(seed)
            model, x, lr = build()
            out[name].append(curve(model, x, lr, tr, te))
            print(f"seed {seed}, {name}: last-epoch test accuracy {out[name][-1][-1]:.4f}", flush=True)
        trees = {"random_forest": RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1),
                 "xgboost": XGBClassifier(n_estimators=100, random_state=seed, eval_metric="mlogloss")}
        for name, model in trees.items():
            out[name].append(float(np.mean(model.fit(stats[tr], labels[tr]).predict(stats[te]) == labels[te])))
            print(f"seed {seed}, {name}: test accuracy {out[name][-1]:.4f}", flush=True)
    np.savez(path, seeds=np.array(seeds), **{k: np.array(v) for k, v in out.items()})
    print("saved", path)


def plot(path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "mathtext.fontset": "stix",
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 6.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.2,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    })
    data = np.load(path)
    panels = [
        ("(a) raw window", [("fin_real_warped", "FIN, pretrained (real + warped)", "#2a78d6"),
                            ("fin_warped", "FIN, pretrained (warped)", "#1baf7a"),
                            ("fin_random", "FIN, random init", "#eb6834")]),
        ("(b) raw window + 40 statistical features", [
            ("stats_real_warped", "FIN + 40, pretrained (real + warped)", "#2a78d6"),
            ("stats_random", "FIN + 40, random init", "#eb6834"),
            ("ignatov_cnn", "Ignatov CNN + 40", "#7a4fc9")]),
    ]
    trees = [("random_forest", "Random forest (40)", (0, (4, 2))), ("xgboost", "XGBoost (40)", (0, (1, 1.5)))]

    fig, axes = plt.subplots(1, 2, figsize=(4.72, 2.1), sharey=True)  # 12 cm wide, the ESANN text width
    for ax, (title, curves) in zip(axes, panels):
        for key, label, color in curves:
            mean = data[key].mean(0)
            epochs = np.arange(1, len(mean) + 1)
            ax.plot(epochs, mean, color=color, label=label)
        for key, label, style in trees:
            ax.axhline(data[key].mean(), color="#52514e", linewidth=0.8, linestyle=style, label=label)
        ax.set_title(title, loc="left")
        ax.set_xlabel("epoch")
        ax.set_xlim(1, epochs[-1])
        ax.grid(axis="y", color="#e1e0d9", linewidth=0.4)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, loc="lower right", handlelength=2.2)
    axes[0].set_ylabel("test accuracy")
    fig.tight_layout(pad=0.3, w_pad=1.2)
    for ext in FORMATS:
        out = f"{os.path.splitext(path)[0]}.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=300)
        print("saved", out)


PAIR_EPOCHS = 30  # EPOCHS shows the whole training


def plot_pair(path):
    """Our FIN on the raw window only, pretrained on warped windows, against the same FIN with random weights and
    the weakest model of the table (Ignatov's CNN + 40 statistical features). Mean over the draws, first
    PAIR_EPOCHS epochs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "mathtext.fontset": "stix",
        "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.linewidth": 0.6, "lines.linewidth": 1.3, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    })
    data = np.load(path)
    fig, ax = plt.subplots(figsize=(3.15, 1.95))  # 8 cm wide
    for key, label, color in (("fin_warped", "FIN, pretrained", "#2a78d6"),
                              ("fin_random", "FIN, random init", "#eb6834"),
                              ("ignatov_cnn", "CNN + 40 statistical", "#7a4fc9")):
        mean = data[key].mean(0)[:PAIR_EPOCHS]
        epochs = np.arange(1, len(mean) + 1)
        ax.plot(epochs, mean, color=color, label=label)
    ax.set_xlabel("epoch")
    ax.set_ylabel("test accuracy")
    ax.set_xlim(1, epochs[-1])
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(pad=0.3)
    for ext in FORMATS:
        out = os.path.join(os.path.dirname(path), f"scarcity_pair.{ext}")
        fig.savefig(out, bbox_inches="tight", dpi=300)
        print("saved", out)


def std_bands(ax, x, bands, alpha=0.18):
    """Mean +- s.d. regions as opaque fills that look like alpha-blended ones (EPS has no transparency): every
    overlap of several bands is filled with the colour those bands would blend to over white.
    bands: list of (low, high, colour), drawn in this order."""
    from itertools import combinations

    from matplotlib.colors import to_rgb

    for k in range(1, len(bands) + 1):
        for subset in combinations(bands, k):
            low = np.max([b[0] for b in subset], axis=0)
            high = np.min([b[1] for b in subset], axis=0)
            colour = np.ones(3)
            for b in subset:
                colour = colour * (1 - alpha) + np.array(to_rgb(b[2])) * alpha
            ax.fill_between(x, low, high, where=high > low, interpolate=True, color=colour, linewidth=0, zorder=0.5)


def plot_best(path, stop=None, fin3=False, full=False, xmax=None):
    """The three curves of plot_pair over all epochs, our FIN stopped at epoch `stop` (default: its best epoch, the
    peak of the mean test curve; the validation curves are not saved) and continued as a dashed line at that accuracy. The best epoch, and every
    epoch at which another curve reaches the dashed line, is marked on the x axis. Shaded: mean +- s.d.
    over the seeds (the held FIN keeps its s.d. at `stop`). fin3: also Ignatov's CNN with the FIN's 3 features instead of the 40,
    saved as scarcity_epoch<stop>_fin3.*. full: our FIN is not stopped, every curve runs to the last epoch
    (scarcity_full.*). xmax: show only the first xmax epochs (file name gets _<xmax>ep)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "mathtext.fontset": "stix",
        "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.linewidth": 0.6, "lines.linewidth": 1.3, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    })
    data = np.load(path)
    fig, ax = plt.subplots(figsize=(3.15, 1.95))  # 8 cm wide
    ours = data["fin_warped"].mean(0)
    n = len(ours)
    best = n if full else stop or int(ours.argmax()) + 1
    acc = ours[best - 1]
    marks = {} if full else {best: "#2a78d6"}  # epoch -> tick colour
    x = np.arange(1, n + 1)
    sd = data["fin_warped"].std(0, ddof=1)
    held_mean = np.where(x <= best, ours, acc)
    held_sd = np.where(x <= best, sd, sd[best - 1])
    bands = [(held_mean - held_sd, held_mean + held_sd, "#2a78d6")]
    others = [("fin_random", "FIN, random init", "#eb6834"), ("ignatov_cnn", "CNN + 40 statistical", "#7a4fc9")]
    if fin3:
        others.append(("ignatov_cnn_fin3", "CNN + 3 FIN features", "#1baf7a"))
    for key, _, color in others:
        m, d = data[key].mean(0), data[key].std(0, ddof=1)
        bands.append((m - d, m + d, color))
    std_bands(ax, x, bands)
    ax.plot(np.arange(1, best + 1), ours[:best], color="#2a78d6", label="FIN, pretrained")
    if not full:
        ax.plot([best, n], [acc, acc], color="#2a78d6", linestyle=(0, (3, 2)), linewidth=1.0)
        ax.text((best + n) / 2, acc + 0.004, f"{acc:.3f}", color="#2a78d6", fontsize=7, ha="center", va="bottom")
    for key, label, color in others:
        mean = data[key].mean(0)
        ax.plot(np.arange(1, n + 1), mean, color=color, label=label)
        if full:
            continue
        above = mean[best - 1:] >= acc  # reaching the dashed line, from below
        for i in np.nonzero(above & ~np.concatenate([[False], above[:-1]]))[0]:
            marks[best + int(i)] = color

    xmax = xmax or n
    base = (1, 50, 100, 150) if xmax > 60 else (1, *range(10, xmax + 1, 10))
    gap = 12 if xmax > 60 else 4
    ticks = [t for t in base if t <= xmax and all(abs(t - m) > gap for m in marks)]
    ax.set_xticks(sorted(ticks + list(marks)))
    for label in ax.get_xticklabels():
        epoch = int(label.get_text().replace("−", "-"))
        if epoch in marks:
            label.set_color(marks[epoch])
    ax.set_xlabel("epoch")
    ax.set_ylabel("test accuracy")
    ax.set_xlim(1, xmax)
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.4, zorder=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(pad=0.3)
    for ext in FORMATS:
        out = os.path.join(os.path.dirname(path), ("scarcity_full" if full else f"scarcity_epoch{stop}" if stop else "scarcity_best") + ("_fin3" if fin3 else "") + (f"_{xmax}ep" if xmax < n else "") + f".{ext}")
        fig.savefig(out, bbox_inches="tight", dpi=300)
        print("saved", out)
    print(f"best epoch {best}: {acc:.4f}; marked epochs {sorted(marks)}")


PANEL = [("fin40_pretrained", "FIN, pretrained", "#2a78d6"),
         ("fin40_random", "FIN, random init", "#eb6834"),
         ("ignatov_cnn", "CNN", "#7a4fc9")]


def plot_panels(path, xmax=50):
    """The figure of the paper (tab:scarcity-neural as curves): test accuracy per epoch, mean +- s.d. over the
    draws, first xmax epochs, of the curves data-scarcity/data_scarcity.ipynb saves, with the 40 statistical
    features: FIN + 40 pretrained / random init and Ignatov's CNN + 40. Saves scarcity_panels.* next to the .npz."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "mathtext.fontset": "stix",
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 6.5, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.2, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    })
    data = np.load(path)
    x = np.arange(1, xmax + 1)
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    stats = [(data[k][:, :xmax].mean(0), data[k][:, :xmax].std(0, ddof=1)) for k, _, _ in PANEL]
    std_bands(ax, x, [(m - s, m + s, c) for (m, s), (_, _, c) in zip(stats, PANEL)])
    for (m, _), (_, label, color) in zip(stats, PANEL):
        ax.plot(x, m, color=color, label=label)
    ax.set_xlabel("epoch")
    ax.set_ylabel("test accuracy")
    ax.set_xlim(1, xmax)
    ax.set_ylim(bottom=0.6)
    ax.set_xticks([1, *range(10, xmax + 1, 10)])
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.4, zorder=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right", handlelength=1.8)
    fig.tight_layout(pad=0.3)
    for ext in FORMATS:
        out = os.path.join(os.path.dirname(path), f"scarcity_panels.{ext}")
        fig.savefig(out, bbox_inches="tight", dpi=600 if ext == "png" else None)
        print("saved", out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["train", "plot", "pair", "best", "panels"])
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--n-synth", type=int, default=N_SYNTH)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--stop", type=int, help="best: hold our FIN from this epoch on (default: its best epoch)")
    parser.add_argument("--xmax", type=int, help="best, panels: show only the first XMAX epochs")
    parser.add_argument("--full", action="store_true", help="best: do not stop our FIN, draw every curve to the end")
    parser.add_argument("--fin3", action="store_true", help="best: also draw Ignatov's CNN with the 3 FIN features")
    parser.add_argument("--curves", default=CURVES,
                        help="the .npz to save the curves to / plot from; the figures go next to it")
    args = parser.parse_args()
    if args.step == "train":
        train(args.epochs, args.n_synth, args.seeds, args.curves)
    elif args.step == "plot":
        plot(args.curves)
    elif args.step == "pair":
        plot_pair(args.curves)
    elif args.step == "panels":
        plot_panels(args.curves, args.xmax or 50)
    else:
        plot_best(args.curves, args.stop, args.fin3, args.full, args.xmax)


if __name__ == "__main__":
    main()
