import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights

class ResNet50Binary(nn.Module):
    """
    ResNet-50d model for binary classification.
    3×3 stride-1 stem, no max-pool, optional freezing of early layers.
    """
    def __init__(self, pretrained: bool = True, dropout: float = 0.2, freeze_early: bool = True):
        super(ResNet50Binary, self).__init__()

        # Load ResNet-50 backbone with ImageNet weights
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = models.resnet50(weights=weights)

        # Replace stem with 3×3 conv (stride 1) and remove max-pool
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()

        # Replace classifier with dropout and single-logit output
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 1)
        )

        # Freeze layer1 and layer2 if freeze_early is True
        if freeze_early:
            for p in self.backbone.layer1.parameters():
                p.requires_grad = False
            for p in self.backbone.layer2.parameters():
                p.requires_grad = False

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor (batch_size, 3, H, W).
        Returns:
            Tensor: Raw logits (batch_size, 1).
        """
        return self.backbone(x)

def create_resnet50(pretrained: bool = True, dropout: float = 0.4, freeze_early: bool = True):
    """
    Creates a ResNet-50d binary classifier.

    Args:
        pretrained (bool): Use ImageNet weights.
        dropout (float): Dropout before the classifier.
        freeze_early (bool): Freeze layer1 and layer2 if True.
    Returns:
        nn.Module: ResNet-50d model.
    """
    return ResNet50Binary(pretrained=pretrained, dropout=dropout, freeze_early=freeze_early)
