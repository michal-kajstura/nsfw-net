import torch.nn as nn
from torchvision import models


class NsfwClassifier(nn.Module):
    def __init__(self, freeze_backbone: bool = False):
        super().__init__()
        self._backbone = self._init_backbone(freeze_backbone)

    @staticmethod
    def _init_backbone(freeze_backbone: bool) -> nn.Module:
        backbone = models.resnet101(pretrained=True)

        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False

        # Attach new classification head
        last_layer_features_num = backbone.fc.in_features
        head = nn.Linear(last_layer_features_num, 1)
        nn.init.kaiming_normal_(head.weight)
        backbone.fc = head
        return backbone

    def forward(self, inputs):
        return self._backbone(inputs).flatten()

