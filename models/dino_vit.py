import torch
import torch.nn as nn
from transformers import AutoModelForImageClassification, AutoImageProcessor

class DinoVitBinary(nn.Module):
    """
    DINO ViT-B/16 model for binary classification.
    Replaces the final classifier head with a single output neuron.
    """
    def __init__(self, pretrained=True):
        super(DinoVitBinary, self).__init__()
        model_name = "facebook/dino-vitb16"
        
        if pretrained:
            self.backbone = AutoModelForImageClassification.from_pretrained(model_name)
        else:
            self.backbone = AutoModelForImageClassification.from_pretrained(model_name, ignore_mismatched_sizes=True)

        # Replace classifier for binary output
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Linear(in_features, 1)  # Binary output (logits)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (batch_size, 3, 224, 224).
        Returns:
            Tensor: Raw logits (before sigmoid), shape (batch_size, 1).
        """
        return self.backbone(x).logits

def create_dino_vit(pretrained=True):
    """
    Creates a DINO ViT-B/16 model for binary classification.

    Args:
        pretrained (bool): Whether to use self-supervised pretrained weights.
    Returns:
        nn.Module: DINO ViT binary classifier model.
    """
    return DinoVitBinary(pretrained=pretrained)
