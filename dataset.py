import os
import random

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import config


def get_transforms():
    """Returns (train_transform, eval_transform). Use eval_transform for val and test."""
    norm = transforms.Normalize(mean=config.MEAN, std=config.STD)
    size = (config.IMG_SIZE, config.IMG_SIZE)
    train_tf = transforms.Compose([
        transforms.Resize(size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        norm,
    ])
    eval_tf = transforms.Compose([transforms.Resize(size), transforms.ToTensor(), norm])
    return train_tf, eval_tf


def _read_label_line(txt_path):
    """Returns (class_id, x, y, w, h) from the first line of a YOLO label file, or None."""
    with open(txt_path, "r") as f:
        line = f.readline().strip()
    if not line:
        return None
    p = line.split()
    return int(p[0]), *[float(v) for v in p[1:5]]


class SimpleFischertechnikDataset(Dataset):
    def __init__(self, image_dir, split="train", val_ratio=0.15, test_ratio=0.15,
                 random_split=True, seed=42, transform=None):
        """
        Scene-wise train/val/test split. One scene = 15 consecutive sorted images
        (one per material), so all 15 images of a scene always land in the SAME split.

        Args:
            image_dir: path to the 'all_images' folder.
            split: 'train', 'val' or 'test'.
            val_ratio / test_ratio: fraction of SCENES used for validation / testing.
            random_split: True = shuffle scenes, False = sequential blocks.
            seed: seed for the shuffle (uses a local RNG, global random state is untouched).
        """
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val' or 'test', got '{split}'")
        self.transform = transform
        self.split = split

        image_files = sorted(f for f in os.listdir(image_dir) if f.endswith(".jpg"))
        if len(image_files) % 15 != 0:
            print(f"WARNING: {len(image_files)} images is not divisible by 15; "
                  f"the last {len(image_files) % 15} are ignored.")
        num_scenes = len(image_files) // 15

        num_test = round(num_scenes * test_ratio)
        num_val = round(num_scenes * val_ratio)

        scene_indices = list(range(num_scenes))
        if random_split:
            random.Random(seed).shuffle(scene_indices)

        scene_splits = {
            "test": scene_indices[:num_test],
            "val": scene_indices[num_test:num_test + num_val],
            "train": scene_indices[num_test + num_val:],
        }
        self.scene_ids = sorted(scene_splits[split])

        self.samples = []
        for scene_idx in self.scene_ids:
            for i in range(scene_idx * 15, scene_idx * 15 + 15):
                img_path = os.path.join(image_dir, image_files[i])
                txt_path = img_path.rsplit(".", 1)[0] + ".txt"
                if not os.path.exists(txt_path):
                    continue
                parsed = _read_label_line(txt_path)
                if parsed is None:
                    continue
                self.samples.append({
                    "img_path": img_path,
                    "txt_path": txt_path,
                    "label": parsed[0],
                    "scene": scene_idx,   # used for per-scene analysis in evaluate.py
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["img_path"]).convert("RGB")  # full uncropped image
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(sample["label"], dtype=torch.long)

    @staticmethod
    def check_scene_grouping(image_dir, max_center_spread=0.05):
        """Sanity check for the '15 consecutive files = 1 scene' assumption.
        Every block must contain each class 0..14 exactly once, and the bounding-box centre
        should barely move inside a block (same scene, only the material changes).
        Returns the list of suspicious block indices (empty = assumption holds)."""
        files = sorted(f for f in os.listdir(image_dir) if f.endswith(".jpg"))
        bad = []
        for b in range(len(files) // 15):
            rows = []
            for f in files[b * 15:(b + 1) * 15]:
                txt = os.path.join(image_dir, f.rsplit(".", 1)[0] + ".txt")
                rows.append(_read_label_line(txt) if os.path.exists(txt) else None)
            if any(r is None for r in rows):
                bad.append((b, "missing or empty label file")); continue
            if sorted(r[0] for r in rows) != list(range(15)):
                bad.append((b, "classes are not exactly 0..14")); continue
            spread = max(max(r[1] for r in rows) - min(r[1] for r in rows),
                         max(r[2] for r in rows) - min(r[2] for r in rows))
            if spread > max_center_spread:
                bad.append((b, f"bbox centre spread {spread:.3f}"))
        print(f"Scene check: {len(files) // 15} blocks, {len(bad)} suspicious.")
        for b, why in bad[:10]:
            print(f"  block {b}: {why}")
        return bad
