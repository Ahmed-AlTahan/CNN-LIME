"""Everything about the data: labels, the scene-wise split, the in-memory image cache,
scene-grouped batches and the GPU augmentation.

Only the class id of each label file is used. The bounding-box coordinates are read
only by `check_scene_grouping`, a sanity check of the dataset; no model ever sees them."""

import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import config


# ---------------------------------------------------------------- labels and split

def read_label(txt_path):
    """First line of a YOLO label file as (class_id, cx, cy, w, h), or None if it is empty."""
    with open(txt_path) as f:
        line = f.readline().strip()
    if not line:
        return None
    p = line.split()
    return int(p[0]), *map(float, p[1:5])


def _image_files():
    return sorted(f for f in os.listdir(config.IMAGE_DIR) if f.endswith(".jpg"))


def _label_path(img_path):
    return img_path.rsplit(".", 1)[0] + ".txt"


def scene_split(num_scenes):
    """Scene ids of each split. A scene always goes to one split as a whole, so a test
    image never shares its background with a training image."""
    ids = list(range(num_scenes))
    random.Random(config.SPLIT_SEED).shuffle(ids)
    n_test = round(num_scenes * config.TEST_RATIO)
    n_val = round(num_scenes * config.VAL_RATIO)
    return {"test": sorted(ids[:n_test]),
            "val": sorted(ids[n_test:n_test + n_val]),
            "train": sorted(ids[n_test + n_val:])}


def load_samples(split):
    """[{img_path, label, scene}] for 'train', 'val' or 'test'. Scene k = sorted images 15k..15k+14."""
    files, n = _image_files(), config.SCENE_SIZE
    if len(files) % n:
        print(f"WARNING: {len(files)} images is not a multiple of {n}; "
              f"the last {len(files) % n} are ignored.")
    samples = []
    for scene in scene_split(len(files) // n)[split]:
        for f in files[scene * n:(scene + 1) * n]:
            img_path = os.path.join(config.IMAGE_DIR, f)
            txt = _label_path(img_path)
            label = read_label(txt) if os.path.exists(txt) else None
            if label is not None:
                samples.append({"img_path": img_path, "label": label[0], "scene": scene})
    return samples


def check_scene_grouping(max_center_spread=0.05):
    """Checks the '15 consecutive files = 1 scene' assumption: every block must hold each class
    exactly once, and the workpiece position should barely move inside a block.
    Returns the suspicious blocks (empty list = assumption holds)."""
    files, n = _image_files(), config.SCENE_SIZE
    bad = []
    for b in range(len(files) // n):
        rows = []
        for f in files[b * n:(b + 1) * n]:
            txt = _label_path(os.path.join(config.IMAGE_DIR, f))
            rows.append(read_label(txt) if os.path.exists(txt) else None)
        if any(r is None for r in rows):
            bad.append((b, "missing or empty label file"))
        elif sorted(r[0] for r in rows) != list(range(config.NUM_CLASSES)):
            bad.append((b, f"classes are not exactly 0..{config.NUM_CLASSES - 1}"))
        else:
            spread = max(max(r[i] for r in rows) - min(r[i] for r in rows) for i in (1, 2))
            if spread > max_center_spread:
                bad.append((b, f"workpiece centre moves by {spread:.3f}"))
    print(f"Scene check: {len(files) // n} blocks, {len(bad)} suspicious.")
    for b, why in bad[:10]:
        print(f"  block {b}: {why}")
    return bad


# ---------------------------------------------------------------- images

def load_image(path, size=None):
    """Whole image resized to size x size, as a uint8 tensor 3 x H x W."""
    size = size or config.IMG_SIZE
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    return torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1)


class CachedImageDataset(Dataset):
    """One split. Every image is decoded once and kept in RAM as uint8 (~2 GB for all images),
    so JPEGs are not decoded again every epoch."""

    def __init__(self, split):
        self.split = split
        self.samples = load_samples(split)
        self.scenes = [s["scene"] for s in self.samples]
        self.labels = torch.tensor([s["label"] for s in self.samples])
        with ThreadPoolExecutor(8) as ex:
            self.images = list(ex.map(lambda s: load_image(s["img_path"]), self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.images[i], self.labels[i]


def load_splits():
    """(train, val, test) datasets, with a check that no scene appears in two splits."""
    check_scene_grouping()
    t0 = time.time()
    splits = [CachedImageDataset(s) for s in ("train", "val", "test")]
    tr, va, te = (set(d.scenes) for d in splits)
    assert not (tr & va or tr & te or va & te), "Scene leakage between splits!"
    print(f"Loaded in {time.time() - t0:.0f}s")
    print(f"Scenes -> train: {len(tr)} | val: {len(va)} | test: {len(te)}")
    print(f"Images -> train: {len(splits[0])} | val: {len(splits[1])} | test: {len(splits[2])}")
    return splits


# ---------------------------------------------------------------- batches

class SceneBatchSampler:
    """Each batch = all images of k randomly chosen scenes. Inside such a batch every material
    appears on the same backgrounds, so the background cannot explain the label."""

    def __init__(self, scenes, k, seed):
        groups = {}
        for i, sc in enumerate(scenes):
            groups.setdefault(sc, []).append(i)
        self.groups, self.k, self.rng = list(groups.values()), k, random.Random(seed)

    def __iter__(self):
        order = self.groups[:]
        self.rng.shuffle(order)
        for b in range(0, len(order) - self.k + 1, self.k):
            yield [i for g in order[b:b + self.k] for i in g]

    def __len__(self):
        return len(self.groups) // self.k


def make_train_loader(ds, recipe, seed):
    """A new loader for every run, with its own seeded order, so a run can be repeated exactly."""
    if recipe.scenes_per_batch:
        return DataLoader(ds, batch_sampler=SceneBatchSampler(ds.scenes, recipe.scenes_per_batch, seed))
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=recipe.batch_size, shuffle=True, generator=g)


def make_eval_loader(ds, batch_size):
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


# ---------------------------------------------------------------- normalisation and augmentation

def _normalize_float(x):
    mean = torch.tensor(config.MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(config.STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def normalize(x_uint8):
    """uint8 batch -> normalised float batch (the model input)."""
    return _normalize_float(x_uint8.float() / 255)


def denormalize(x):
    """Normalised 3 x H x W tensor -> H x W x 3 numpy image in [0, 1], for plotting."""
    return np.clip(np.array(config.STD) * x.permute(1, 2, 0).cpu().numpy() + np.array(config.MEAN), 0, 1)


def augment(x_uint8, recipe):
    """Random brightness/contrast, horizontal flip, rotation, zoom and shift on a uint8 batch
    (on its own device). Returns the normalised batch. Hue and saturation are never changed,
    because colour is what separates many of the classes."""
    B, dev = x_uint8.size(0), x_uint8.device

    def uniform(*shape):          # uniform in [-1, 1]
        return torch.rand(*shape, device=dev) * 2 - 1

    x = x_uint8.float() / 255
    b = 1 + uniform(B, 1, 1, 1) * recipe.brightness
    c = 1 + uniform(B, 1, 1, 1) * recipe.contrast
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (((x - m) * c + m) * b).clamp(0, 1)

    # flip + rotation + zoom + shift as one affine warp
    ang = uniform(B) * math.radians(recipe.rotation)
    scale = 1 + uniform(B) * recipe.zoom
    flip = torch.where(torch.rand(B, device=dev) < 0.5, -1.0, 1.0)
    cos, sin = torch.cos(ang) / scale, torch.sin(ang) / scale
    A = torch.stack([torch.stack([cos * flip, -sin], 1),
                     torch.stack([sin * flip, cos], 1)], 1)                 # B x 2 x 2
    t = uniform(B, 2) * recipe.shift
    grid = F.affine_grid(torch.cat([A, t.unsqueeze(2)], 2), list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return _normalize_float(x)
