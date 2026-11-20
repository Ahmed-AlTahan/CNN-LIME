"""Results and figures, the same for every model.

Everything goes to outputs/<model>/:
    weights/   seed<N>.pth                      best-on-validation weights of each run (saved by engine.py)
    figures/   inputs, curves, confusion_test, samples_test, attention_test, scenes/scene_<id>.png
    results/   summary.json, history_seed<N>.csv, classification_report.txt,
               per_scene_accuracy.csv, predictions_test.csv"""

import csv
import dataclasses
import json
import math
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

import config
import data
from engine import device, evaluate, out_path, set_seed


def output_dir(model_name):
    path = os.path.join(config.OUTPUT_DIR, model_name)
    os.makedirs(path, exist_ok=True)
    return path


def _save_figure(fig, out_dir, name, dpi=150):
    path = out_path(out_dir, "figures", name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.show()
    print(f"Saved {path}")


def _thumbnail(path, size=800):
    img = Image.open(path).convert("RGB")
    img.thumbnail((size, size))
    return img


# ---------------------------------------------------------------- before / during training

def plot_inputs(train_ds, recipe, out_dir):
    """12 augmented training images, exactly as the network sees them."""
    set_seed(0)
    images, labels = next(iter(data.make_train_loader(train_ds, recipe, 0)))
    x = data.augment(images[:12].to(device), recipe).cpu()
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    for ax, img, lbl in zip(axes.flatten(), x, labels[:12]):
        ax.imshow(data.denormalize(img)); ax.set_title(config.CLASS_NAMES[int(lbl)]); ax.axis("off")
    fig.suptitle("Augmented training inputs", fontweight="bold")
    _save_figure(fig, out_dir, "inputs.png", dpi=100)


def plot_curves(runs, out_dir):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4.5))
    for r in runs:
        h = np.array(r["history"])
        line, = a1.plot(h[:, 0], h[:, 3], label=f"val, seed {r['seed']}")
        a1.plot(h[:, 0], h[:, 1], ":", color=line.get_color())
        a2.plot(h[:, 0], h[:, 4], color=line.get_color(), label=f"val, seed {r['seed']}")
        a2.plot(h[:, 0], h[:, 2], ":", color=line.get_color())
    a2.axhline(100 / config.NUM_CLASSES, ls="--", c="gray", label="chance")
    a1.set_title("Loss (solid = val, dotted = train)"); a1.set_xlabel("epoch"); a1.legend()
    a2.set_title("Accuracy % (solid = val, dotted = train)"); a2.set_xlabel("epoch"); a2.legend()
    _save_figure(fig, out_dir, "curves.png")


# ---------------------------------------------------------------- test set

def test_summary(runs, test_loader, recipe, out_dir):
    """Evaluates every run once on the test set, prints mean +- std over the seeds and saves
    results/summary.json. Returns the run with the best VALIDATION accuracy (never chosen by test)."""
    print("seed | best val acc | test acc")
    for r in runs:
        _, r["test_acc"], r["preds"], r["labels"] = evaluate(r["model"], test_loader)
        print(f"{r['seed']:4d} | {r['best_val_acc']:11.2f}% | {r['test_acc']:7.2f}%")
    accs = np.array([r["test_acc"] for r in runs])
    best = max(runs, key=lambda r: r["best_val_acc"])
    print("========================================")
    print(f"TEST accuracy over {len(runs)} seed(s): {accs.mean():.2f}% +- {accs.std():.2f}% "
          f"(min {accs.min():.2f}%, max {accs.max():.2f}%)")
    print("========================================")
    print(f"\nDetailed analysis below: seed {best['seed']} "
          f"(best validation accuracy {best['best_val_acc']:.2f}%, test {best['test_acc']:.2f}%)")

    summary = {
        "model": recipe.model,
        "test_acc_mean": round(float(accs.mean()), 2),
        "test_acc_std": round(float(accs.std()), 2),
        "analysed_seed": best["seed"],
        "runs": [{"seed": r["seed"], "best_epoch": r["best_epoch"], "best_val_acc": round(float(r["best_val_acc"]), 2),
                  "test_acc": round(float(r["test_acc"]), 2)} for r in runs],
        "recipe": dataclasses.asdict(recipe),
    }
    path = out_path(out_dir, "results", "summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {path}")
    return best


def analyse(run, test_ds, out_dir, title):
    """Per-class report, prediction counts, per-scene accuracy, predictions CSV, confusion matrix,
    sample grid, and the two worst scenes plus the best scene. Returns {scene: accuracy}."""
    preds, labels = run["preds"], run["labels"]
    report = classification_report(labels, preds, labels=list(range(config.NUM_CLASSES)),
                                   target_names=config.CLASS_NAMES, digits=4, zero_division=0)
    pred_counts = np.bincount(preds, minlength=config.NUM_CLASSES)
    true_counts = np.bincount(labels, minlength=config.NUM_CLASSES)
    counts = [f"{'class':>14} | true | predicted"]
    for c, name in enumerate(config.CLASS_NAMES):
        flag = "  <-- over-predicted" if pred_counts[c] > 1.3 * true_counts[c] else ""
        counts.append(f"{name:>14} | {true_counts[c]:4d} | {pred_counts[c]:9d}{flag}")
    text = f"{title} - test set, seed {run['seed']}, accuracy {run['test_acc']:.2f}%\n\n{report}\n" + "\n".join(counts)
    print(text)
    with open(out_path(out_dir, "results", "classification_report.txt"), "w") as f:
        f.write(text + "\n")

    scenes = np.array(test_ds.scenes)
    scene_acc = {int(s): float((preds[scenes == s] == labels[scenes == s]).mean()) for s in np.unique(scenes)}
    v = np.array(list(scene_acc.values()))
    ranked = sorted(scene_acc, key=scene_acc.get)
    print(f"\nPer-scene accuracy: mean {v.mean()*100:.1f}% | std {v.std()*100:.1f}% "
          f"| min {v.min()*100:.1f}% | max {v.max()*100:.1f}%")
    print("Worst scenes:", ", ".join(f"{s}: {scene_acc[s]*100:.0f}%" for s in ranked[:5]))
    with open(out_path(out_dir, "results", "per_scene_accuracy.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene", "accuracy"])
        w.writerows((s, round(scene_acc[s] * 100, 2)) for s in ranked)

    with open(out_path(out_dir, "results", "predictions_test.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["img_path", "scene", "true", "pred"])
        for smp, p in zip(test_ds.samples, preds):
            w.writerow([smp["img_path"], smp["scene"], smp["label"], int(p)])
    print(f"Saved results to {os.path.join(out_dir, 'results')}")

    plot_confusion(preds, labels, f"{title} - test set, seed {run['seed']}, acc {run['test_acc']:.1f}%", out_dir)
    plot_samples(test_ds, preds, out_dir)
    for s in dict.fromkeys(ranked[:2] + ranked[-1:]):          # 2 worst + best, without repeats
        plot_scene(test_ds, preds, s, out_dir)
    return scene_acc


def plot_confusion(preds, labels, title, out_dir):
    cm = confusion_matrix(labels, preds, labels=list(range(config.NUM_CLASSES)))
    fig = plt.figure(figsize=(12, 10))
    plt.imshow(cm, cmap="Blues"); plt.colorbar()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j]:
                plt.text(j, i, cm[i, j], ha="center", va="center",
                         color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.xticks(range(config.NUM_CLASSES), config.CLASS_NAMES, rotation=60, ha="right")
    plt.yticks(range(config.NUM_CLASSES), config.CLASS_NAMES)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title(title)
    _save_figure(fig, out_dir, "confusion_test.png", dpi=200)


def plot_samples(test_ds, preds, out_dir, n=16):
    """n random test images with true and predicted class (green = correct, red = wrong)."""
    idxs = random.Random(0).sample(range(len(test_ds)), min(n, len(test_ds)))
    cols, rows = 4, math.ceil(len(idxs) / 4)
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    for ax in axes.flatten():
        ax.axis("off")
    for ax, i in zip(axes.flatten(), idxs):
        smp = test_ds.samples[i]
        ax.imshow(_thumbnail(smp["img_path"]))
        ax.set_title(f"true: {config.CLASS_NAMES[smp['label']]}\npred: {config.CLASS_NAMES[preds[i]]}"
                     f" | scene {smp['scene']}", fontsize=10,
                     color="green" if smp["label"] == preds[i] else "red")
    fig.suptitle("Sample test predictions (green = correct, red = wrong)", fontweight="bold")
    _save_figure(fig, out_dir, "samples_test.png")


def plot_scene(test_ds, preds, scene, out_dir):
    """All 15 images of one test scene with their predictions."""
    idxs = sorted((i for i, s in enumerate(test_ds.scenes) if s == scene),
                  key=lambda i: test_ds.samples[i]["label"])
    fig, axes = plt.subplots(3, 5, figsize=(18, 10))
    for ax in axes.flatten():
        ax.axis("off")
    for ax, i in zip(axes.flatten(), idxs):
        smp = test_ds.samples[i]
        ax.imshow(_thumbnail(smp["img_path"]))
        ax.set_title(f"true: {config.CLASS_NAMES[smp['label']]}\npred: {config.CLASS_NAMES[preds[i]]}",
                     fontsize=9, color="green" if smp["label"] == preds[i] else "red")
    acc = np.mean([test_ds.samples[i]["label"] == preds[i] for i in idxs]) * 100
    fig.suptitle(f"Scene {scene}  ({acc:.0f}% correct)", fontsize=14, fontweight="bold")
    _save_figure(fig, out_dir, os.path.join("scenes", f"scene_{scene}.png"))


@torch.no_grad()
def plot_attention(run, test_ds, out_dir, n_right=6, n_wrong=6):
    """AttentionCNN only: where the model looks on correct and wrong test images."""
    model, preds, labels = run["model"].eval(), run["preds"], run["labels"]
    rng = random.Random(1)
    right = [i for i in range(len(preds)) if preds[i] == labels[i]]
    wrong = [i for i in range(len(preds)) if preds[i] != labels[i]]
    idxs = rng.sample(right, min(n_right, len(right))) + rng.sample(wrong, min(n_wrong, len(wrong)))

    x = data.normalize(torch.stack([test_ds.images[i] for i in idxs]).to(device))
    _, scores = model(x, return_attention=True)
    g = int(math.isqrt(scores.size(1)))
    maps = scores.softmax(1).view(-1, g, g).cpu().numpy()

    size = test_ds.images[0].shape[-1]
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    for ax in axes.flatten():
        ax.axis("off")
    for ax, i, heat in zip(axes.flatten(), idxs, maps):
        ax.imshow(test_ds.images[i].permute(1, 2, 0).numpy())
        ax.imshow(heat, cmap="jet", alpha=0.4, extent=(0, size, size, 0), interpolation="bilinear")
        ax.set_title(f"true: {config.CLASS_NAMES[labels[i]]}\npred: {config.CLASS_NAMES[preds[i]]}",
                     color="green" if preds[i] == labels[i] else "red", fontsize=10)
    fig.suptitle("Learned attention (red = where the model looks)", fontweight="bold")
    _save_figure(fig, out_dir, "attention_test.png", dpi=100)
