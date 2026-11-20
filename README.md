# Fischertechnik workpiece material classification

Classifies the material of the workpiece (15 classes) in photos of the Fischertechnik factory model.
Four CNNs are compared on the same data and the same scene-wise split.

| # | Model | Notebook |
|---|---|---|
| 1 | Simple CNN, trained from scratch | [notebooks/01_simple_cnn.ipynb](notebooks/01_simple_cnn.ipynb) |
| 2 | Attention CNN, trained from scratch | [notebooks/02_attention_cnn.ipynb](notebooks/02_attention_cnn.ipynb) |
| 3 | ResNet18, pretrained on ImageNet | [notebooks/03_resnet18.ipynb](notebooks/03_resnet18.ipynb) |
| 4 | ResNet50, pretrained on ImageNet | [notebooks/04_resnet50.ipynb](notebooks/04_resnet50.ipynb) |

## Repository layout

```
src/
  config.py     data location, split, class names, training recipe of each model
  data.py       labels, scene-wise split, image cache, scene-grouped batches, GPU augmentation
  models.py     SimpleCNN, AttentionCNN, PretrainedResNet (ResNet18 / ResNet50)
  engine.py     training (reproducible, one run per seed), evaluation, loading saved weights
  report.py     test summary over seeds, per-class / per-scene analysis, all figures
notebooks/      one Colab notebook per model; they only call the code in src/
```

## Data

* 166 scenes x 15 images = 2490 images. A scene is one camera position, photographed once per material,
  so the 15 images of a scene differ only in the workpiece.
* Each `.jpg` has a YOLO `.txt` label; only the class id is used, never the bounding box.
* Every model gets the whole image resized to 512x512 (no cropping).
* Split by scene, seed 123: 116 / 25 / 25 scenes (1740 / 375 / 375 images), identical for all models.

## Running a notebook in Colab

1. Open the notebook in Colab, choose Runtime -> Change runtime type -> GPU.
2. Run all. The first cell clones this repository (branch set in `BRANCH`) and downloads the dataset.
3. The last cell copies the outputs to Google Drive.

To change a setting for an experiment, edit the notebook's settings cell, e.g.
`recipe = config.replace(recipe, seeds=[1, 2, 3, 4, 5])`. The defaults are in `config.RECIPES`.

## Outputs

Each model writes to its own folder, `outputs/<model>/` in Colab, copied to
`MyDrive/thesis_results/<model>/` on Google Drive:

```
<model>/
  weights/   seed1.pth, seed2.pth, ...          best-on-validation weights of each run
  figures/   inputs.png, curves.png, confusion_test.png, samples_test.png,
             attention_test.png (attention CNN only), scenes/scene_<id>.png
  results/   summary.json                       test accuracy per seed, mean +- std, the recipe used
             history_seed<N>.csv                loss and accuracy per epoch
             classification_report.txt          per-class precision / recall of the analysed run
             per_scene_accuracy.csv, predictions_test.csv
```

`<model>` is `simple_cnn`, `attention_cnn`, `resnet18` or `resnet50`. Outputs are not stored in git.

## Loading a trained model (e.g. for LIME)

```python
import sys; sys.path.insert(0, "src")
import data, engine
model = engine.load_model("resnet18", "outputs/resnet18/weights/seed1.pth")
x = data.normalize(data.load_image("path/to/image.jpg").unsqueeze(0).to(engine.device))
pred = model(x).argmax(1).item()
```
