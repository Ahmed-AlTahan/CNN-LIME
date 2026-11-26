# Fischertechnik workpiece material classification

Classifies the material of the workpiece (15 classes) in photos of the Fischertechnik factory model.
Five CNNs are compared on the same data and the same scene-wise split.

| # | Model | Notebook | Results |
|---|---|---|---|
| 1 | Simple CNN, trained from scratch | [01_simple_cnn.ipynb](notebooks/01_simple_cnn.ipynb) | [results/simple_cnn](results/simple_cnn) |
| 2 | Attention CNN, trained from scratch | [02_attention_cnn.ipynb](notebooks/02_attention_cnn.ipynb) | [results/attention_cnn](results/attention_cnn) |
| 3 | ResNet18, pretrained on ImageNet | [03_resnet18.ipynb](notebooks/03_resnet18.ipynb) | [results/resnet18](results/resnet18) |
| 4 | ResNet50, pretrained on ImageNet | [04_resnet50.ipynb](notebooks/04_resnet50.ipynb) | [results/resnet50](results/resnet50) |
| 5 | ConvNeXt-Tiny, pretrained on ImageNet | [05_convnext_tiny.ipynb](notebooks/05_convnext_tiny.ipynb) | [results/convnext_tiny](results/convnext_tiny) |
| - | LIME explanations of a trained model (step 1: a few test images) | [06_lime.ipynb](notebooks/06_lime.ipynb) | [results/lime](results/lime) |

A results folder appears once its notebook has been run and saved.

## Repository layout

```
src/
  config.py     locations, split, class names, training recipe of each model
  data.py       labels, scene-wise split, image cache, scene-grouped batches, GPU augmentation
  models.py     SimpleCNN, AttentionCNN, Pretrained (ResNet18 / ResNet50 / ConvNeXt-Tiny)
  engine.py     training (reproducible, one run per seed), evaluation, loading saved weights
  report.py     test summary over seeds, per-class / per-scene analysis, all figures
  explain.py    LIME (own implementation, SLIC superpixels from scikit-image), example choice, LIME figures
  env.py        Colab or own computer, dataset download, saving results to GitHub / Google Drive
notebooks/      one notebook per model; they only call the code in src/
results/        figures and result files of every model (no weights), written by the notebooks;
                LIME results in results/lime/<model>/
requirements.txt
```

## Data

* 166 scenes x 15 images = 2490 images. A scene is one camera position, photographed once per material,
  so the 15 images of a scene differ only in the workpiece.
* Each `.jpg` has a YOLO `.txt` label; only the class id is used, never the bounding box.
* Every model gets the whole image resized to 512x512 (no cropping).
* Split by scene, seed 123: 116 / 25 / 25 scenes (1740 / 375 / 375 images), identical for all models.
* The dataset is downloaded automatically into `data/` by the first notebook cell (not stored in git).

## Running the notebooks

The same notebooks run in **Colab** and on **an own computer with an NVIDIA GPU** (e.g. the university
computer); the first cell detects which one it is.

**Colab**
1. Open the notebook in Colab, Runtime -> Change runtime type -> GPU.
2. Run all. The first cell clones this repository (or pulls the newest code) and downloads the dataset.

**Own computer (one-time setup)**
```
git clone https://github.com/Ahmed-AlTahan/CNN-LIME.git
cd CNN-LIME
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # GPU build, see pytorch.org
pip install -r requirements.txt
```
Then open a notebook from `notebooks/` in Jupyter or VS Code and Run all. The first cell pulls the newest code
from GitHub and downloads the dataset the first time.

Only one notebook should train on a GPU at a time; two at once can run out of GPU memory.
To change a setting for an experiment, edit the notebook's settings cell, e.g.
`recipe = config.replace(recipe, seeds=[1, 2, 3, 4, 5])`. The defaults are in `config.RECIPES`.

## Outputs

Each model writes everything to `outputs/<model>/` on the machine it runs on. Every file name starts with the
model name, so a file is identifiable even when it is downloaded or listed without its folder.

```
<model>/
  weights/   <model>_seed1.pth, ...                best-on-validation weights of each run
  figures/   <model>_inputs.png, _curves.png, _confusion_test.png, _samples_test.png,
             _attention_test.png (attention CNN only), scenes/<model>_scene_<id>.png
  results/   <model>_summary.json                  test accuracy per seed, mean +- std, recipe, where it ran
             <model>_train_log.txt                 the full training printout
             <model>_history_seed<N>.csv           loss and accuracy per epoch
             <model>_classification_report.txt     per-class precision / recall of the analysed run
             <model>_per_scene_accuracy.csv, <model>_predictions_test.csv
```

The last notebook cell keeps them permanently:

| Where | What | When |
|---|---|---|
| GitHub, `results/<model>/` | figures + results (no weights) | always (Colab: needs the `GITHUB_TOKEN` secret, see below) |
| Google Drive, `MyDrive/thesis_results/<model>/` | everything, including the weights | Colab; own computer once rclone is set up (below) |
| `outputs/<model>/` on the computer | everything, including the weights | own computer |

A re-run of a model replaces that model's files only.

**Pushing from Colab (once):** create a GitHub personal access token (GitHub -> Settings -> Developer settings ->
Fine-grained tokens, repository `CNN-LIME`, permission *Contents: read and write*). In Colab, open the key icon
(Secrets) in the left sidebar, add `GITHUB_TOKEN` with the token as value and enable notebook access.
On an own computer the normal git login is used.

**Google Drive on an own computer (once):** install [rclone](https://rclone.org) (no admin rights needed) and
connect it to Google Drive as a remote named `gdrive`; a browser window opens to sign in:
```
mkdir -p ~/.local/bin && cd /tmp && curl -LO https://downloads.rclone.org/rclone-current-linux-amd64.zip \
  && unzip -o rclone-current-linux-amd64.zip && cp rclone-*-linux-amd64/rclone ~/.local/bin/
~/.local/bin/rclone config create gdrive drive scope=drive
~/.local/bin/rclone lsd gdrive:          # check: lists the folders in your Drive
```
The last notebook cell then uploads `outputs/<model>/` (with weights) to `MyDrive/thesis_results/<model>/`.
If the upload fails, retry it in the notebook with `env.save_to_drive_rclone(out_dir)`.
On a shared computer, remove the Drive login when you are done: `~/.local/bin/rclone config delete gdrive`.

## LIME explanations

`notebooks/06_lime.ipynb` explains predictions of a trained model with LIME (Ribeiro et al., 2016):
SLIC superpixels (about 170 per 512x512 image), 1000 copies with random superpixels hidden (filled with the
image's mean colour), the model's probability for every copy, and a weighted Ridge regression whose
coefficients are the importances of the superpixels. The settings are `config.LIME`.

**Step 1** (current): ConvNeXt-Tiny on 8 test images - 4 wrong predictions, each paired with a correct
image of the same material from another scene; for wrong predictions both the predicted and the true class are
explained. Outputs in `results/lime/<model>/`: one figure per image, an overview figure,
`<model>_lime_examples.csv` (top superpixels, probability, fidelity R2 per explanation) and `<model>_lime_summary.json`.

The notebook uses the analysed run of the model (best validation accuracy, from its `summary.json`). Its weights
are taken from `outputs/<model>/weights/`, or else from Google Drive (`thesis_results/<model>/weights/`).
Run it on a GPU: about 2 s per explanation, on a CPU about 10 minutes.

## Loading a trained model (e.g. for LIME)

```python
import sys; sys.path.insert(0, "src")
import data, engine
model = engine.load_model("resnet50", "outputs/resnet50/weights/resnet50_seed1.pth")
x = data.normalize(data.load_image("path/to/image.jpg").unsqueeze(0).to(engine.device))
pred = model(x).argmax(1).item()
```
Weights from Colab are on Google Drive in `thesis_results/<model>/weights/`.
