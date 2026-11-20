"""Training and evaluation, the same for every model. Runs are reproducible:
the same recipe and seed give the same result."""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # deterministic GPU matmuls; set before CUDA starts

import copy
import csv
import math
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

import config
import data
import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True, warn_only=True)    # warn instead of failing if an op has no deterministic version


def out_path(out_dir, sub, name):
    """Path of an output file: <out_dir>/<sub>/<name>, sub = 'weights', 'figures' or 'results'."""
    folder = os.path.join(out_dir, sub)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, name)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader):
    """Loss, accuracy (%), predictions and labels of a model on a loader."""
    model.eval()
    loss_sum, preds, labels_all = 0.0, [], []
    for images, labels in loader:
        images, labels = data.normalize(images.to(device)), labels.to(device)
        out = model(images)
        loss_sum += F.cross_entropy(out.float(), labels).item() * labels.size(0)
        preds.append(out.argmax(1).cpu()); labels_all.append(labels.cpu())
    preds, labels_all = torch.cat(preds).numpy(), torch.cat(labels_all).numpy()
    return loss_sum / len(labels_all), 100.0 * (preds == labels_all).mean(), preds, labels_all


def load_model(name, path):
    """A trained model from a saved .pth file, in eval mode."""
    model = models.build_model(name, config.NUM_CLASSES, pretrained=False).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.eval()


def sanity_check(recipe, train_ds, steps=150):
    """A fresh model must overfit one scene (15 images) to ~100%. If it cannot, the pipeline is
    broken; if it can, any remaining problem is generalisation. Uses a throw-away model."""
    set_seed(0)
    model = models.build_model(recipe.model, config.NUM_CLASSES).to(device)
    images, labels = next(iter(data.make_train_loader(train_ds, recipe, 0)))
    n = config.SCENE_SIZE                       # a full batch needs too much memory without AMP
    images, labels = data.normalize(images[:n].to(device)), labels[:n].to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for step in range(1, steps + 1):
        opt.zero_grad()
        out = model(images)
        loss = F.cross_entropy(out, labels)
        loss.backward(); opt.step()
        if step == 1 or step % 25 == 0:
            acc = 100.0 * (out.argmax(1) == labels).float().mean().item()
            print(f"step {step:4d} | loss {loss.item():.4f} | batch acc {acc:.1f}%")
    del model, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _lr_factor(recipe, steps_per_epoch):
    total, warmup = recipe.epochs * steps_per_epoch, recipe.warmup_epochs * steps_per_epoch

    def factor(step):
        if step < warmup:
            return (step + 1) / warmup
        if not recipe.cosine:
            return 1.0
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))
    return factor


def train_one_run(recipe, seed, train_ds, val_loader, out_dir):
    """One full training run from scratch (new model, seeded data order, new optimizer).
    Keeps the weights of the epoch with the best validation accuracy and saves them."""
    set_seed(seed)
    train_loader = data.make_train_loader(train_ds, recipe, seed)
    model = models.build_model(recipe.model, config.NUM_CLASSES).to(device)

    Opt = torch.optim.AdamW if recipe.optimizer == "adamw" else torch.optim.Adam
    optimizer = Opt(model.parameters(), lr=recipe.lr, weight_decay=recipe.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_factor(recipe, len(train_loader)))
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    init_loss, init_acc, _, _ = evaluate(model, val_loader)
    print(f"Before training: val loss {init_loss:.3f} (chance ~{math.log(config.NUM_CLASSES):.3f}) "
          f"| val acc {init_acc:.1f}%")

    history = []
    best_val_acc, best_val_loss, best_weights, best_epoch, bad_epochs = -1.0, 1e9, None, 0, 0
    for epoch in range(1, recipe.epochs + 1):
        t0 = time.time()
        model.train()
        loss_sum, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = data.augment(images.to(device), recipe), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                out = model(images)
                loss = F.cross_entropy(out.float(), labels, label_smoothing=recipe.label_smoothing)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update(); scheduler.step()
            loss_sum += loss.item() * labels.size(0)
            total += labels.size(0)
            correct += (out.argmax(1) == labels).sum().item()

        val_loss, val_acc, _, _ = evaluate(model, val_loader)
        tr_loss, tr_acc = loss_sum / total, 100.0 * correct / total
        history.append((epoch, tr_loss, tr_acc, val_loss, val_acc))
        print(f"[seed {seed}] Epoch [{epoch:3d}/{recipe.epochs}] | Train Loss {tr_loss:.4f} "
              f"| Train Acc {tr_acc:6.2f}% | Val Loss {val_loss:.4f} | Val Acc {val_acc:6.2f}% "
              f"| lr {scheduler.get_last_lr()[0]:.2e} | {time.time() - t0:.0f}s")

        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc, best_val_loss, best_epoch, bad_epochs = val_acc, val_loss, epoch, 0
            best_weights = copy.deepcopy(model.state_dict())
            print("   ---> New best validation accuracy, weights kept.")
        else:
            bad_epochs += 1
            if bad_epochs >= recipe.patience:
                print(f"\nEarly stopping: no validation improvement for {recipe.patience} epochs.")
                break

    model.load_state_dict(best_weights)
    path = out_path(out_dir, "weights", f"seed{seed}.pth")
    torch.save(model.state_dict(), path)
    with open(out_path(out_dir, "results", f"history_seed{seed}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        w.writerows(history)
    print(f"[seed {seed}] Best validation accuracy: {best_val_acc:.2f}% (epoch {best_epoch}) -> saved {path}")
    return dict(seed=seed, model=model, history=history, best_val_acc=best_val_acc,
                best_epoch=best_epoch, path=path)


def train_all(recipe, train_ds, val_loader, out_dir):
    """One run per seed in recipe.seeds."""
    n = models.count_parameters(models.build_model(recipe.model, config.NUM_CLASSES, pretrained=False))
    print(f"{recipe.model}: {n / 1e6:.2f} M parameters | device: {device}")
    runs = []
    for seed in recipe.seeds:
        print(f"\n==================== Training run with seed {seed} ====================")
        runs.append(train_one_run(recipe, seed, train_ds, val_loader, out_dir))
    return runs
