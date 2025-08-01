import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50

# DeepLabV3 customized model for Unknown Object Segmentation (UOS)
class DeepLabUOS(nn.Module):
    def __init__(self, n_classes=7, normalize_uos=True):
        super().__init__()
        base = deeplabv3_resnet50(weights=None, num_classes=n_classes)
        self.backbone = base
        self.sigmoid = nn.Sigmoid()
        # enable/disable normalization
        self.normalize_uos = normalize_uos

    def forward(self, x):
        eps = 1e-6
        # Temperature scaling for logits
        T = 2
        logits = self.backbone(x)["out"]
        # Scale logits to avoid instability
        logits_scaled = logits / T
        # Apply softmax to get probabilities
        probs = torch.softmax(logits_scaled, dim=1)

        # Objectness = max probabilities on foreground classes (Human, Vehicle, Objects)
        objectness = probs[:, [2, 3, 5], :, :].max(dim=1, keepdim=True)[0]
        # Unknown Score (US) = sum of probabilities of non-foreground classes
        us = torch.exp(torch.sum(torch.log(1 - probs + eps), dim=1, keepdim=True))
        # Log-space Unknown Objectness Score (UOS) = objectness * US
        uos = objectness * us

        # Optional Normalization 0–1 to avoid compressed UOS values
        if self.normalize_uos:
            uos_min, uos_max = uos.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0], \
                               uos.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
            uos = (uos - uos_min) / (uos_max - uos_min + eps)

        return {"logits": logits, "probs": probs,"us": us, "objectness": objectness, "uos": uos}