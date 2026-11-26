"""LIME explanations of the trained models (Ribeiro et al., 2016), implemented here.
The superpixels come from SLIC (Achanta et al., 2012) in scikit-image.

One explanation of one image and one class:
  1. Split the 512x512 image (exactly what the model sees) into superpixels with SLIC.
  2. Make n_samples copies. In each copy a random subset of the superpixels is hidden by filling it
     with the mean colour of the whole image. Not black (it would look like black plastic) and not
     each superpixel's own mean colour (the lime default, which would keep the workpiece's colour).
     The first copy is the original image.
  3. The model gives the probability of the explained class for every copy.
  4. Every copy is weighted by its similarity to the original: cosine distance between its on/off
     vector and the all-on vector, exponential kernel (as in the lime package).
  5. A weighted Ridge regression from "which superpixels are visible" to "probability" is fitted.
     Its coefficient for a superpixel is that superpixel's importance:
     > 0 supports the class, < 0 speaks against it.
  The R^2 of this regression ("fidelity") says how well the linear model imitates the network
  around this image.

Step 1 (this file): explanations of a few chosen test images, as figures and a table.
Outputs: outputs/lime/<model>/ -> saved to results/lime/<model>/ (GitHub) like every other result."""

import csv
import json
import os
import random
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.segmentation import mark_boundaries, slic
from sklearn.linear_model import Ridge

import config
import data
import engine
import env
from engine import device, out_path, rel


def output_dir(model_name):
    path = os.path.join(config.OUTPUT_DIR, "lime", model_name)
    os.makedirs(path, exist_ok=True)
    return path


# ---------------------------------------------------------------- model and examples

def _model_results(model_name, name):
    return os.path.join(config.RESULTS_DIR, model_name, "results", f"{model_name}_{name}")


def load_trained(model_name):
    """The run that is analysed everywhere else: the seed with the best validation accuracy,
    read from results/<model>/results/<model>_summary.json."""
    with open(_model_results(model_name, "summary.json")) as f:
        summary = json.load(f)
    seed = summary["analysed_seed"]
    run = next(r for r in summary["runs"] if r["seed"] == seed)
    path = env.find_weights(model_name, seed)
    model = engine.load_model(model_name, path)
    print(f"{model_name}, seed {seed}: validation {run['best_val_acc']:.2f}%, test {run['test_acc']:.2f}% "
          f"| weights: {rel(path)}")
    return model, seed


def choose_examples(model_name, n_wrong=4):
    """Pairs of test images: a wrongly classified one and a correctly classified one of the same material
    from another scene, so right and wrong can be compared directly. Read from the test predictions
    of the analysed run (results/<model>/results/<model>_predictions_test.csv).

    Wrong images, at most one per scene and one per (true, predicted) pair, in this order:
      1. metal confused with metal (aluminium, brass, steel)
      2. black plastic involved (true or predicted)
      3. all other mistakes"""
    with open(_model_results(model_name, "predictions_test.csv")) as f:
        rows = [dict(img_path=os.path.join(config.ROOT, r["img_path"]), scene=int(r["scene"]),
                     true=int(r["true"]), pred=int(r["pred"])) for r in csv.DictReader(f)]
    metals = {config.CLASS_NAMES.index(c) for c in ("aluminium", "brass", "steel")}
    black = config.CLASS_NAMES.index("black plastic")

    def priority(r):
        if r["true"] in metals and r["pred"] in metals:
            return 0
        return 1 if black in (r["true"], r["pred"]) else 2

    wrong, scenes, pairs = [], set(), set()
    for r in sorted((r for r in rows if r["true"] != r["pred"]), key=lambda r: (priority(r), r["scene"])):
        if len(wrong) < n_wrong and r["scene"] not in scenes and (r["true"], r["pred"]) not in pairs:
            wrong.append(r); scenes.add(r["scene"]); pairs.add((r["true"], r["pred"]))

    rng, examples = random.Random(0), []
    for w in wrong:
        candidates = [r for r in rows if r["true"] == r["pred"] == w["true"] and r["scene"] not in scenes]
        right = rng.choice(candidates)
        scenes.add(right["scene"])
        examples += [dict(right, kind="correct"), dict(w, kind="wrong")]

    print(f"{'#':>2} | {'kind':7} | scene | {'true':14} | {'predicted':14} | image")
    for i, e in enumerate(examples, 1):
        print(f"{i:2d} | {e['kind']:7} | {e['scene']:5d} | {config.CLASS_NAMES[e['true']]:14} | "
              f"{config.CLASS_NAMES[e['pred']]:14} | {os.path.basename(e['img_path'])}")
    return examples


# ---------------------------------------------------------------- LIME

def superpixels(image_hwc, s):
    """SLIC superpixels of an H x W x 3 uint8 image: H x W array of ids 0..n-1."""
    return slic(image_hwc, n_segments=s.n_segments, compactness=s.compactness, start_label=0)


@torch.no_grad()
def predict_proba(model, images_uint8):
    """Class probabilities for a B x 3 x H x W uint8 batch on the model's device."""
    with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
        out = model(data.normalize(images_uint8))
    return out.float().softmax(1)


@torch.no_grad()
def lime_explain(model, image, target, s, segments=None):
    """LIME for one image (3 x H x W uint8 tensor) and one class (steps 1-5 in the module docstring).
    Returns dict(target, prob, weights (one per superpixel), intercept, fidelity, segments)."""
    if segments is None:
        segments = superpixels(image.permute(1, 2, 0).numpy(), s)
    n = int(segments.max()) + 1

    rng = np.random.default_rng(s.seed)
    masks = rng.integers(0, 2, size=(s.n_samples, n)).astype(np.float32)   # 1 = superpixel visible
    masks[0] = 1                                                            # copy 0 = the original

    img = image.to(device)
    fill = img.float().mean(dim=(1, 2), keepdim=True).round().to(torch.uint8)   # mean colour of the image
    seg = torch.from_numpy(segments).long().to(device)
    probs = []
    for i in range(0, s.n_samples, s.batch_size):
        visible = torch.from_numpy(masks[i:i + s.batch_size]).to(device).bool()[:, seg]   # B x H x W
        copies = torch.where(visible.unsqueeze(1), img.unsqueeze(0), fill.unsqueeze(0))
        probs.append(predict_proba(model, copies)[:, target].cpu())
    probs = torch.cat(probs).numpy()

    on = masks.sum(1)
    distance = 1 - on / (np.sqrt(on) * np.sqrt(n) + 1e-12)          # cosine distance to the all-on vector
    kernel = np.sqrt(np.exp(-distance ** 2 / s.kernel_width ** 2))
    # exact solution; "svd" also avoids a bug of old scikit-learn versions with new scipy versions
    linear = Ridge(alpha=s.ridge_alpha, solver="svd").fit(masks, probs, sample_weight=kernel)
    return dict(target=int(target), prob=float(probs[0]), weights=linear.coef_,
                intercept=float(linear.intercept_), fidelity=float(linear.score(masks, probs, sample_weight=kernel)),
                segments=segments)


# ---------------------------------------------------------------- figures and table

def _heatmap(exp):
    return exp["weights"][exp["segments"]]                    # importance of every pixel's superpixel


def _top_superpixels(exp, k):
    order = np.argsort(exp["weights"])[::-1][:k]
    return [int(i) for i in order if exp["weights"][i] > 0]


def plot_explanation(image_hwc, exps, title, s, path):
    """One row per explained class: image | superpixels | top-k superpixels | importance heatmap."""
    fig, axes = plt.subplots(len(exps), 4, figsize=(20, 5.3 * len(exps)), squeeze=False)
    for row, exp in zip(axes, exps):
        name = config.CLASS_NAMES[exp["target"]]
        top = _top_superpixels(exp, s.top_k)
        keep = np.isin(exp["segments"], top)
        dimmed = np.where(keep[..., None], image_hwc, (0.25 * image_hwc + 0.75 * 255).astype(np.uint8))
        heat = _heatmap(exp)
        vmax = np.abs(exp["weights"]).max() or 1.0

        row[0].imshow(image_hwc)
        row[0].set_title("input (512x512, as the model sees it)")
        row[1].imshow(mark_boundaries(image_hwc, exp["segments"], color=(1, 1, 0)))
        row[1].set_title(f"{exp['segments'].max() + 1} superpixels (SLIC)")
        row[2].imshow(mark_boundaries(dimmed, keep.astype(int), color=(1, 0, 0)))
        row[2].set_title(f"top {len(top)} superpixels for '{name}'")
        row[3].imshow(image_hwc.mean(2), cmap="gray")
        im = row[3].imshow(heat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, alpha=0.6)
        row[3].set_title(f"importance for '{name}' (p = {exp['prob']:.2f}, fidelity R2 = {exp['fidelity']:.2f})")
        fig.colorbar(im, ax=row[3], fraction=0.046, pad=0.02, label="red = supports, blue = against")
        for ax in row:
            ax.axis("off")
    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.show()
    plt.close(fig)


def explain_examples(model, examples, s, out_dir, model_name):
    """Explains every example: the predicted class, and for a wrong prediction also the true class.
    Saves one figure per example, an overview figure, lime_examples.csv and lime_summary.json."""
    for sub in ("figures", "results"):                          # a re-run replaces the old files
        shutil.rmtree(os.path.join(out_dir, sub), ignore_errors=True)
    rows, overview = [], []
    for i, e in enumerate(examples, 1):
        t0 = time.time()
        image = data.load_image(e["img_path"])
        pred = int(predict_proba(model, image.unsqueeze(0).to(device)).argmax(1))
        if pred != e["pred"]:
            print(f"  note: example {i} is now predicted as {config.CLASS_NAMES[pred]} "
                  f"(test run: {config.CLASS_NAMES[e['pred']]}); explaining the current prediction.")
        targets = [pred] + ([e["true"]] if e["true"] != pred else [])
        exps, segments = [], None
        for t in targets:
            exps.append(lime_explain(model, image, t, s, segments))
            segments = exps[-1]["segments"]                     # same superpixels for both classes
        seconds = time.time() - t0

        hwc = image.permute(1, 2, 0).numpy()
        true, name = config.CLASS_NAMES[e["true"]], config.CLASS_NAMES[pred]
        tag = f"{i:02d}_{e['kind']}_{true.replace(' ', '-')}_scene{e['scene']}"
        title = (f"Example {i} ({e['kind']}) - scene {e['scene']} - true: {true} | predicted: {name}"
                 f"  [{os.path.basename(e['img_path'])}]")
        plot_explanation(hwc, exps, title, s, out_path(out_dir, "figures", f"lime_{tag}.png"))
        overview.append((hwc, exps[0], f"{i}. {e['kind']} | true: {true}\npred: {name} (p={exps[0]['prob']:.2f})",
                         pred == e["true"]))
        for exp in exps:
            top = _top_superpixels(exp, s.top_k)
            rows.append([i, e["kind"], os.path.relpath(e["img_path"], config.ROOT), e["scene"], true, name,
                         config.CLASS_NAMES[exp["target"]], round(exp["prob"], 4), round(exp["fidelity"], 4),
                         int(segments.max()) + 1, " ".join(map(str, top)),
                         " ".join(f"{exp['weights'][j]:.4f}" for j in top), round(seconds, 1)])
        print(f"example {i}/{len(examples)} done in {seconds:.0f}s")

    _plot_overview(overview, model_name, out_dir)
    with open(out_path(out_dir, "results", "lime_examples.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["example", "kind", "img_path", "scene", "true", "pred", "explained_class", "prob",
                    "fidelity_r2", "n_superpixels", "top_superpixels", "top_weights", "seconds"])
        w.writerows(rows)
    with open(out_path(out_dir, "results", "lime_summary.json"), "w") as f:
        json.dump({"model": model_name, "examples": len(examples), "settings": vars(s),
                   "mean_fidelity_r2": round(float(np.mean([r[8] for r in rows])), 4),
                   "environment": env.info()}, f, indent=2)
    print(f"Saved figures and tables to {rel(out_dir)}")
    return rows


def _plot_overview(items, model_name, out_dir):
    """All examples at a glance: image and importance heatmap for the predicted class."""
    cols = 4
    rows = int(np.ceil(len(items) / 2))
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.8 * rows), squeeze=False)
    for ax in axes.flatten():
        ax.axis("off")
    for k, (hwc, exp, label, correct) in enumerate(items):
        a, b = axes[k // 2, 2 * (k % 2)], axes[k // 2, 2 * (k % 2) + 1]
        vmax = np.abs(exp["weights"]).max() or 1.0
        a.imshow(hwc)
        a.set_title(label, fontsize=10, color="green" if correct else "red")
        b.imshow(hwc.mean(2), cmap="gray")
        b.imshow(_heatmap(exp), cmap="RdBu_r", vmin=-vmax, vmax=vmax, alpha=0.6)
        b.set_title("LIME: red = supports the prediction", fontsize=10)
    fig.suptitle(f"LIME explanations - {model_name} (green = correct, red = wrong)", fontsize=15, fontweight="bold")
    fig.tight_layout()
    path = out_path(out_dir, "figures", "lime_overview.png")
    fig.savefig(path, dpi=100)
    plt.show()
    plt.close(fig)
    print(f"Saved {rel(path)}")
