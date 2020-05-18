import torch.nn as nn
from torchvision import models

class NsfwClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self._backbone = self._init_backbone(num_classes)

    @staticmethod
    def _init_backbone(num_classes: int) -> nn.Module:
        backbone = models.resnet50(pretrained=True)

        # Attach new classification head
        last_layer_features_num = backbone.fc.in_features
        backbone.fc = nn.Linear(last_layer_features_num, num_classes)
        return backbone

    def forward(self, inputs):
        return self._backbone(inputs)

