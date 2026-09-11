"""capas 4A/4B: cabezas de clasificacion y CTC sobre el mismo encoder compartido"""
import torch
import torch.nn as nn

from models.spatial_encoder import build_spatial_encoder
from models.temporal_encoder import build_temporal_encoder


class SignEncoder(nn.Module):
    """capa 2 (espacial) + capa 3 (temporal)"""

    def __init__(self, model_cfg, input_dim):
        super().__init__()
        self.spatial = build_spatial_encoder(model_cfg["spatial"], input_dim)
        self.temporal = build_temporal_encoder(model_cfg["temporal"], self.spatial.output_dim)
        self.output_dim = self.temporal.output_dim

    def forward(self, x, lengths=None):
        return self.temporal(self.spatial(x), lengths)


class ClassificationModel(nn.Module):
    """capa 4A: softmax con pooling temporal enmascarado"""

    def __init__(self, model_cfg, input_dim, num_classes):
        super().__init__()
        self.encoder = SignEncoder(model_cfg, input_dim)
        self.head = nn.Linear(self.encoder.output_dim, num_classes)

    def forward(self, x, lengths):
        h = self.encoder(x, lengths)                       # [B, T, H]
        mask = (torch.arange(h.shape[1], device=h.device)[None, :]
                < lengths[:, None].to(h.device)).float().unsqueeze(-1)
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return self.head(pooled)                           # [B, C]


class CTCModel(nn.Module):
    """capa 4B: logits por frame para CTC (blank = 0)"""

    def __init__(self, model_cfg, input_dim, num_ctc_classes):
        super().__init__()
        self.encoder = SignEncoder(model_cfg, input_dim)
        self.head = nn.Linear(self.encoder.output_dim, num_ctc_classes)

    def forward(self, x, lengths=None):
        return self.head(self.encoder(x, lengths))         # [B, T, C+1]


def transfer_encoder(dst_model, ckpt_path, device="cpu"):
    """carga solo los pesos del encoder desde un checkpoint de otra fase"""
    state = torch.load(ckpt_path, map_location=device)
    src = state.get("model", state)
    enc = {k[len("encoder."):]: v for k, v in src.items() if k.startswith("encoder.")}
    missing, unexpected = dst_model.encoder.load_state_dict(enc, strict=False)
    return {"cargados": len(enc) - len(missing), "faltantes": len(missing),
            "inesperados": len(unexpected)}
