import torch
import torch.nn as nn
import torchvision.models as models

class ResNet50Binary(nn.Module):
    """
    ResNet-50 model for binary classification 
    Replaces the final fully connected layer with a single output neuron.
    """
    def __init__(self, pretrained=True):
        super(ResNet50Binary, self).__init__()
        # Load the pretrained ResNet-50 backbone
        self.backbone = models.resnet50(pretrained=pretrained)
        # Replace the final fully connected layer for binary output
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)
    
    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (batch_size, 3, 96, 96).
        Returns:
            Tensor: Raw logits (before sigmoid), shape (batch_size, 1).
        """
        return self.backbone(x)

def create_resnet50(pretrained=True):
    """
    Args:
        pretrained (bool): Whether to use ImageNet pretrained weights.
    Returns:
        nn.Module: ResNet-50 binary classifier model.
    """
    return ResNet50Binary(pretrained=pretrained)
