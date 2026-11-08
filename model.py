import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class FischertechnikCNN(nn.Module):

  def __init__(self, num_classes=15, pretrained=True):
    super().__init__()
    # pretrained=True for training; pretrained=False when loading your own saved weights
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    self.base_model = resnet18(weights=weights)

    # Replace the final fully connected layer to match the 15 classes
    in_features = self.base_model.fc.in_features
    self.base_model.fc = nn.Sequential(
        nn.Dropout(0.5), nn.Linear(in_features, num_classes)
    )

  def forward(self, x):
    return self.base_model(x)
