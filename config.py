"""Shared settings. train.py and evaluate.py must use identical split settings,
otherwise the evaluation would run on different scenes than the ones held out in training."""

IMAGE_DIR = "./data/15workpieces_same_position/all_images"
MODEL_PATH = "fischertechnik_cnn.pth"   # written by train.py, read by evaluate.py

NUM_CLASSES = 15
IMG_SIZE = 512
BATCH_SIZE = 32

# Scene-wise split (166 scenes -> about 116 train / 25 val / 25 test)
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SPLIT = True
# NEW seed on purpose: with seed 42 the old test scenes were already used for early
# stopping, so a different seed gives a test set that was never used for any decision.
SEED = 123

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
