import torch.nn as nn
from torchvision import models

class NsfwClassifier(nn.Module):
    def __init__(self, num_classes=1, freeze_backbone=False):
        super().__init__()
        self._backbone = self._init_backbone(num_classes, freeze_backbone)

    @staticmethod
    def _init_backbone(num_classes: int, freeze_backbone: bool) -> nn.Module:
        backbone = models.resnet50(pretrained=True)

        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False

        # Attach new classification head
        last_layer_features_num = backbone.fc.in_features
        backbone.fc = nn.Linear(last_layer_features_num, num_classes)
        return backbone

    def forward(self, inputs):
        return self._backbone(inputs).flatten()

