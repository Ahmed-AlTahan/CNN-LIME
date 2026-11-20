"""The three networks. Layer names are unchanged from the original code,
so weights saved by the earlier runs still load."""

import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class SimpleCNN(nn.Module):
    """Model 1 (the original from-scratch CNN): 4 x [conv 3x3 -> BatchNorm -> ReLU -> max-pool],
    global average pooling, then Linear(256,128) -> ReLU -> Dropout(0.5) -> Linear(128,15)."""

    def __init__(self, num_classes=15):
        super().__init__()

        def block(cin, cout):
            return [nn.Conv2d(cin, cout, kernel_size=3, padding=1), nn.BatchNorm2d(cout),
                    nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2)]

        self.features = nn.Sequential(*block(3, 32), *block(32, 64), *block(64, 128), *block(128, 256),
                                      nn.AdaptiveAvgPool2d((1, 1)))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(256, 128), nn.ReLU(),
                                        nn.Dropout(0.5), nn.Linear(128, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


def _conv_bn_relu(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class AttentionCNN(nn.Module):
    """Model 2: 5 stages of 2 x (conv 3x3 -> BatchNorm -> ReLU), 512x512 input -> 16x16 feature map.
    A 1x1 conv scores every cell, a softmax over the 256 cells gives attention weights, and the
    class is predicted from the attention-weighted features. Learned from the class labels only."""

    def __init__(self, num_classes=15, widths=(32, 64, 128, 256, 256), dropout=0.3):
        super().__init__()
        layers = [_conv_bn_relu(3, widths[0], stride=2), _conv_bn_relu(widths[0], widths[0]),
                  nn.MaxPool2d(2)]                                           # 512 -> 128
        cin = widths[0]
        for i, w in enumerate(widths[1:]):
            layers += [_conv_bn_relu(cin, w), _conv_bn_relu(w, w)]
            if i < len(widths) - 2:                                          # no pool after the last stage
                layers.append(nn.MaxPool2d(2))
            cin = w
        self.features = nn.Sequential(*layers)                               # 128 -> 16x16
        self.attention = nn.Conv2d(cin, 1, kernel_size=1)                    # one score per cell
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(cin, num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01); nn.init.zeros_(m.bias)
        nn.init.zeros_(self.attention.weight); nn.init.zeros_(self.attention.bias)  # start = plain average

    def forward(self, x, return_attention=False):
        f = self.features(x)                                  # B x C x h x w
        scores = self.attention(f).flatten(1).float()         # B x (h*w)
        w = scores.softmax(1).to(f.dtype)
        pooled = (f.flatten(2) * w.unsqueeze(1)).sum(2)       # attention-weighted features
        logits = self.classifier(pooled)
        return (logits, scores) if return_attention else logits


class ResNet18(nn.Module):
    """Model 3: ResNet18 pretrained on ImageNet, last layer replaced by Dropout(0.5) -> Linear(512,15).
    pretrained=True for training; pretrained=False when loading saved weights."""

    def __init__(self, num_classes=15, pretrained=True):
        super().__init__()
        self.base_model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.base_model.fc = nn.Sequential(nn.Dropout(0.5),
                                           nn.Linear(self.base_model.fc.in_features, num_classes))

    def forward(self, x):
        return self.base_model(x)


MODELS = {"simple_cnn": SimpleCNN, "attention_cnn": AttentionCNN, "resnet18": ResNet18}


def build_model(name, num_classes=15, pretrained=True):
    if name == "resnet18":
        return ResNet18(num_classes, pretrained=pretrained)
    return MODELS[name](num_classes)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
