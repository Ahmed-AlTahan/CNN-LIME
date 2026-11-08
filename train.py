import copy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import config
from dataset import SimpleFischertechnikDataset, get_transforms
from model import FischertechnikCNN


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
    return 100.0 * correct / total if total else 0.0


def train_model():
    # ---- Hyperparameters ----
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    PATIENCE = 10          # epochs without VALIDATION improvement before stopping

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    set_seed(config.SEED)

    # ---- Data: scene-wise train / val / test ----
    train_tf, eval_tf = get_transforms()
    common = dict(image_dir=config.IMAGE_DIR, val_ratio=config.VAL_RATIO,
                  test_ratio=config.TEST_RATIO, random_split=config.RANDOM_SPLIT,
                  seed=config.SEED)
    SimpleFischertechnikDataset.check_scene_grouping(config.IMAGE_DIR)

    train_ds = SimpleFischertechnikDataset(split="train", transform=train_tf, **common)
    val_ds = SimpleFischertechnikDataset(split="val", transform=eval_tf, **common)
    test_ds = SimpleFischertechnikDataset(split="test", transform=eval_tf, **common)

    # Safety net: no scene may appear in two splits
    s = [set(d.scene_ids) for d in (train_ds, val_ds, test_ds)]
    assert not (s[0] & s[1] or s[0] & s[2] or s[1] & s[2]), "Scene leakage between splits!"
    print(f"Scenes   -> train: {len(s[0])} | val: {len(s[1])} | test: {len(s[2])}")
    print(f"Images   -> train: {len(train_ds)} | val: {len(val_ds)} | test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)

    # ---- Model, loss, optimizer ----
    model = FischertechnikCNN(num_classes=config.NUM_CLASSES, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc, patience_counter, best_weights = 0.0, 0, None

    print("\nStarting training loop...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            total += labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()

        val_acc = accuracy(model, val_loader, device)   # decisions use VALIDATION only
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Train Loss: {running_loss/total:.4f} "
              f"| Train Acc: {100*correct/total:.2f}% | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print("   ---> New best validation accuracy! Weights saved.")
        else:
            patience_counter += 1
            print(f"   ---> No improvement. Patience counter: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch+1} "
                      f"(validation accuracy did not improve for {PATIENCE} epochs).")
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)
    torch.save(model.state_dict(), config.MODEL_PATH)

    # ---- Test set: evaluated ONCE, after all decisions are made ----
    test_acc = accuracy(model, test_loader, device)
    print("\n========================================")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"FINAL TEST accuracy (reported once): {test_acc:.2f}%")
    print("========================================")
    print(f"Weights saved to '{config.MODEL_PATH}'. Do not tune anything on the test set anymore.")


if __name__ == "__main__":
    train_model()
