"""Settings shared by all notebooks: data location, scene-wise split, classes,
and the training recipe of every model."""

from dataclasses import dataclass, field, replace  # noqa: F401  (replace: change a recipe in a notebook)

# ---- Data ----
IMAGE_DIR = "data/15workpieces_same_position/all_images"
OUTPUT_DIR = "outputs"            # outputs/<model>/{weights, figures, results}/
DRIVE_DIR = "/content/drive/MyDrive/thesis_results"   # same structure on Google Drive: <DRIVE_DIR>/<model>/
IMG_SIZE = 512                    # whole image, only resized (no cropping)
SCENE_SIZE = 15                   # 15 consecutive sorted images = 1 scene (one per material)

# ---- Scene-wise split: 166 scenes -> 116 train / 25 val / 25 test ----
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SPLIT_SEED = 123                  # fixed: all models use exactly the same scenes

# ---- Classes (index = class id in the .txt label files) ----
CLASS_NAMES = ["white plastic", "aluminium", "brass", "steel", "orange acryl", "yellow acryl",
               "blue acryl", "red acryl", "red plastic", "blue plastic", "yellow plastic",
               "gray plastic", "black plastic", "light wood", "dark wood"]
NUM_CLASSES = len(CLASS_NAMES)

# ImageNet normalisation (needed by the pretrained ResNet18, used by all models for comparability)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


@dataclass
class Recipe:
    """How one model is trained. Every seed in `seeds` is one full, reproducible training run."""
    model: str                    # key in models.MODELS
    epochs: int
    lr: float
    optimizer: str = "adamw"      # "adam" or "adamw"
    weight_decay: float = 0.0
    warmup_epochs: int = 0        # linear learning-rate warm-up
    cosine: bool = False          # cosine decay after the warm-up (False = constant learning rate)
    label_smoothing: float = 0.0
    patience: int = 20            # early stopping: epochs without validation improvement
    batch_size: int = 32          # random batches; also the batch size for validation / test
    scenes_per_batch: int = 0     # > 0: each training batch = all 15 images of this many scenes
    # augmentation (GPU, training only); colour hue / saturation are never changed
    rotation: float = 10.0        # +- degrees
    shift: float = 0.0            # +- fraction of the image size
    zoom: float = 0.0             # +- fraction
    brightness: float = 0.2       # +- fraction
    contrast: float = 0.2         # +- fraction
    seeds: list = field(default_factory=lambda: [1, 2, 3])


# Models 1 and 2 are trained from scratch with one identical recipe, so only the network differs.
_FROM_SCRATCH = dict(epochs=80, lr=1e-3, optimizer="adamw", weight_decay=0.05, warmup_epochs=3,
                     cosine=True, label_smoothing=0.1, patience=20, batch_size=30,
                     scenes_per_batch=4, shift=0.1, zoom=0.1)

RECIPES = {
    "simple_cnn": Recipe("simple_cnn", **_FROM_SCRATCH),
    "attention_cnn": Recipe("attention_cnn", **_FROM_SCRATCH),
    # Same settings as the original ResNet18 training (train.py): Adam 1e-4, constant lr, patience 10.
    "resnet18": Recipe("resnet18", epochs=100, lr=1e-4, optimizer="adam", patience=10, batch_size=32),
    # Same recipe as ResNet18, so only the depth of the network differs.
    "resnet50": Recipe("resnet50", epochs=100, lr=1e-4, optimizer="adam", patience=10, batch_size=32),
}
