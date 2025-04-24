import torch
import torch.nn as nn
from transformers import SwinForImageClassification, AutoImageProcessor

class SwinBaseBinary(nn.Module):
    """
    Swin Transformer Base model for binary classification.
    Replaces the final classifier head with a single output neuron.
    """
    def __init__(self, pretrained=True):
        super(SwinBaseBinary, self).__init__()
        model_name = "microsoft/swin-base-patch4-window7-224"
        
        if pretrained:
            self.backbone = SwinForImageClassification.from_pretrained(model_name)
        else:
            self.backbone = SwinForImageClassification.from_pretrained(model_name, ignore_mismatched_sizes=True)

        # Replace the classifier head for binary output (1 output neuron)
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Linear(in_features, 1)  # Raw logits (no sigmoid here)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (batch_size, 3, 224, 224).
        Returns:
            Tensor: Raw logits (before sigmoid), shape (batch_size, 1).
        """
        return self.backbone(x).logits

def create_swin_base(pretrained=True):
    """
    Creates a Swin-Base model for binary classification.
    
    Args:
        pretrained (bool): Whether to use ImageNet pretrained weights.
    Returns:
        nn.Module: Swin-Base binary classifier model.
    """
    return SwinBaseBinary(pretrained=pretrained)
