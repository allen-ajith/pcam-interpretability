import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights

class ResNet50Binary(nn.Module):
    """
    ResNet-50d model for binary classification.
    Uses a 3×3 stride-1 stem, removes the max-pool, and
    swaps in a single-logit classifier head.
    """
    def __init__(self, pretrained: bool = True, dropout: float = 0.2):
        super(ResNet50Binary, self).__init__()

        # Load ImageNet-pretrained backbone
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = models.resnet50(weights=weights)

        # 3×3 stride-1 stem + no early max-pool
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1,
                                        padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()

        # Dropout + single-logit classifier
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 1)   # Binary output (logits)
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): (batch_size, 3, H, W), already resized & normalized.
        Returns:
            Tensor: Raw logits, shape (batch_size, 1).
        """
        return self.backbone(x)

def create_resnet50(pretrained: bool = True, dropout: float = 0.2):
    """
    Creates a ResNet-50d binary classifier for PCam.

    Args:
        pretrained (bool): Use ImageNet weights if True.
        dropout (float): Dropout probability before the head.
    """
    return ResNet50Binary(pretrained=pretrained, dropout=dropout)