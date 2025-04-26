import torch
import torch.nn as nn
from transformers import AutoModelForImageClassification, AutoImageProcessor


class DinoVitBinary(nn.Module):
    """
    DINO ViT-S/16 model for binary classification

    Forward returns raw logits.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        model_name = "facebook/dino-vits16"
        self.backbone = AutoModelForImageClassification.from_pretrained(
            model_name,
            num_labels=1,
            ignore_mismatched_sizes=not pretrained,
        )

        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Linear(in_features, 1)

        self.processor = AutoImageProcessor.from_pretrained(model_name)

    def forward(self, x, output_attentions=False):
        """
        Args:
            x (Tensor or image list): Input images.
            output_attentions (bool): Whether to return attention maps.
        Returns:
            Either raw logits, or (logits, attentions) if output_attentions=True.
        """
        if isinstance(x, torch.Tensor):
            outputs = self.backbone(pixel_values=x, output_attentions=output_attentions)
        else:  # For image lists or ndarrays
            inputs = self.processor(x, return_tensors="pt").to(next(self.backbone.parameters()).device)
            outputs = self.backbone(**inputs, output_attentions=output_attentions)

        if output_attentions:
            return outputs.logits, outputs.attentions
        else:
            return outputs.logits



def create_dino_vit(pretrained: bool = True) -> nn.Module:
    """
        A DINO ViT-S/16 model with a single-logit head.
    """
    return DinoVitBinary(pretrained=pretrained)
