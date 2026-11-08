import argparse
import csv
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader

import config
from dataset import SimpleFischertechnikDataset, get_transforms
from model import FischertechnikCNN


def evaluate_model(split="test"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | evaluating split: '{split}'")

    # ---- Dataset: SAME split settings as train.py (taken from config.py) ----
    _, eval_tf = get_transforms()
    dataset = SimpleFischertechnikDataset(
        image_dir=config.IMAGE_DIR, split=split, val_ratio=config.VAL_RATIO,
        test_ratio=config.TEST_RATIO, random_split=config.RANDOM_SPLIT,
        seed=config.SEED, transform=eval_tf)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"Loaded {len(dataset)} samples from {len(dataset.scene_ids)} scenes.")

    # ---- Model ----
    model = FischertechnikCNN(num_classes=config.NUM_CLASSES, pretrained=False).to(device)
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
    model.eval()

    # ---- Inference (shuffle=False, so order matches dataset.samples) ----
    preds, labels_all = [], []
    with torch.no_grad():
        for images, labels in loader:
            out = model(images.to(device))
            preds.extend(out.argmax(dim=1).cpu().numpy())
            labels_all.extend(labels.numpy())
    preds, labels_all = np.array(preds), np.array(labels_all)
    scenes = np.array([s["scene"] for s in dataset.samples])

    # ---- Metrics ----
    print("\n========================================")
    print(f"Overall {split} accuracy: {accuracy_score(labels_all, preds) * 100:.2f}%")
    print("========================================\n")
    print(classification_report(labels_all, preds, digits=4))

    # How often is each class predicted vs how often it really occurs? (class 12 analysis)
    pred_counts = np.bincount(preds, minlength=config.NUM_CLASSES)
    true_counts = np.bincount(labels_all, minlength=config.NUM_CLASSES)
    print("Class | true count | predicted count")
    for c in range(config.NUM_CLASSES):
        flag = "  <-- over-predicted" if pred_counts[c] > 1.3 * true_counts[c] else ""
        print(f"{c:5d} | {true_counts[c]:10d} | {pred_counts[c]:15d}{flag}")

    # Accuracy per scene: does the background influence the result?
    scene_acc = {s: float((preds[scenes == s] == labels_all[scenes == s]).mean())
                 for s in np.unique(scenes)}
    vals = np.array(list(scene_acc.values()))
    print(f"\nPer-scene accuracy: mean {vals.mean()*100:.1f}% | std {vals.std()*100:.1f}% "
          f"| min {vals.min()*100:.1f}% | max {vals.max()*100:.1f}%")
    worst = sorted(scene_acc.items(), key=lambda kv: kv[1])[:5]
    print("Worst scenes (scene index: accuracy):",
          ", ".join(f"{s}: {a*100:.0f}%" for s, a in worst))

    # Save every prediction (useful later for choosing images to explain with LIME)
    with open(f"predictions_{split}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["img_path", "scene", "true", "pred"])
        for smp, p in zip(dataset.samples, preds):
            w.writerow([smp["img_path"], smp["scene"], smp["label"], int(p)])
    print(f"Saved predictions_{split}.csv")

    # ---- Confusion matrix ----
    cm = confusion_matrix(labels_all, preds, labels=list(range(config.NUM_CLASSES)))
    plt.figure(figsize=(12, 10))
    ticks = [str(i) for i in range(config.NUM_CLASSES)]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True,
                xticklabels=ticks, yticklabels=ticks)
    plt.xlabel("Predicted Workpiece Class", fontsize=12, fontweight="bold")
    plt.ylabel("True Workpiece Class", fontsize=12, fontweight="bold")
    plt.title(f"15x15 Confusion Matrix - Fischertechnik ResNet18 ({split} set)\n"
              "(Diagonal represents correct predictions)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_15x15_{split}.png", dpi=300)
    print(f"Saved confusion_matrix_15x15_{split}.png")
    plt.close()

    # ---- Sample grid: 16 random samples across ALL scenes (not just the first scene) ----
    idxs = random.Random(0).sample(range(len(dataset)), min(16, len(dataset)))
    mean, std = np.array(config.MEAN), np.array(config.STD)
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for ax, i in zip(axes.flatten(), idxs):
        img_tensor, true_lbl = dataset[i]
        img = np.clip(std * img_tensor.permute(1, 2, 0).numpy() + mean, 0, 1)
        ax.imshow(img); ax.axis("off")
        ok = int(true_lbl) == int(preds[i])
        ax.set_title(f"True: {int(true_lbl)} | Pred: {int(preds[i])} | scene {scenes[i]}",
                     fontsize=9, color="green" if ok else "red")
    plt.suptitle("Sample Predictions (Green = Correct, Red = Incorrect)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"sample_predictions_grid_{split}.png", dpi=300)
    print(f"Saved sample_predictions_grid_{split}.png")
    plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["val", "test"])
    evaluate_model(ap.parse_args().split)
