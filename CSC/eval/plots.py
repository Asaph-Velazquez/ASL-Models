"""graficas rapidas"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(log_dirs, out_png):
    """Lee los *_log.csv de cada fase y grafica sus metricas."""
    found = []
    for d in log_dirs:
        for f in sorted(Path(d).glob("*_log.csv")):
            found.append(f)
    if not found:
        return None
    fig, axes = plt.subplots(1, len(found), figsize=(5 * len(found), 4), squeeze=False)
    for ax, f in zip(axes[0], found):
        with open(f, newline="") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            continue
        steps = [int(float(r["step"])) for r in rows]
        for k in rows[0]:
            if k == "step":
                continue
            ax.plot(steps, [float(r[k]) for r in rows], label=k)
        ax.set_title(f.stem)
        ax.set_xlabel("epoca")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return out_png


def confusion_matrix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def plot_confusion_matrix(cm, class_names, confusable_pairs, out_png):
    """Matriz de confusion con los pares confundibles del vocabulario RESALTADOS en rojo
    (COLD/HOT, A/S/T/E, numeros vs letras...)."""
    n = len(class_names)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    size = max(8, n * 0.28)
    fig, ax = plt.subplots(figsize=(size, size))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    idx = {g: i for i, g in enumerate(class_names)}
    for a, b in confusable_pairs:
        if a in idx and b in idx:
            for i, j in ((idx[a], idx[b]), (idx[b], idx[a])):
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           edgecolor="red", linewidth=1.6))
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=90, fontsize=6)
    ax.set_yticklabels(class_names, fontsize=6)
    ax.set_xlabel("prediccion")
    ax.set_ylabel("real")
    ax.set_title("Matriz de confusion (rojo = pares confundibles declarados en el CSV)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    return out_png


def skeleton_animation(npy_path, layout, out_gif, max_frames=80):
    """Animacion 2D del esqueleto (manos + torso) de un clip de landmarks [T, D] o [T, 3D]."""
    from matplotlib.animation import FuncAnimation, PillowWriter
    arr = np.load(npy_path).astype(np.float32)
    D = layout["D"]
    arr = arr[:, :D]  # si trae movimiento (pos|vel|acel), solo la posicion
    if len(arr) > max_frames:
        arr = arr[np.linspace(0, len(arr) - 1, max_frames).round().astype(int)]

    blocks = {name: layout["offsets"][name] for name in ("lh", "rh", "pose")}
    fig, ax = plt.subplots(figsize=(4, 4))
    scats = {name: ax.plot([], [], "o", ms=3, label=name)[0] for name in blocks}
    allxy = np.concatenate([arr[:, a:b].reshape(len(arr), -1, 3)[:, :, :2].reshape(-1, 2)
                            for a, b in blocks.values()])
    lo, hi = np.nanpercentile(allxy, 1), np.nanpercentile(allxy, 99)
    pad = 0.1 * (hi - lo + 1e-6)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(hi + pad, lo - pad)  # y invertida (coordenadas de imagen)
    ax.legend(fontsize=7)
    ax.set_title(Path(npy_path).stem)

    def update(t):
        for name, (a, b) in blocks.items():
            pts = arr[t, a:b].reshape(-1, 3)
            scats[name].set_data(pts[:, 0], pts[:, 1])
        return list(scats.values())

    anim = FuncAnimation(fig, update, frames=len(arr), blit=True)
    anim.save(out_gif, writer=PillowWriter(fps=12))
    plt.close(fig)
    return out_gif
