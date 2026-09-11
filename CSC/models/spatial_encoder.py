"""capa 2: encoder espacial, este es el MLP"""
import torch
import torch.nn as nn


class MLPSpatialEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims=(256, 256), dropout=0.2):
        super().__init__()
        layers, d = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        self.net = nn.Sequential(*layers)
        self.output_dim = d

    def forward(self, x):  # [B, T, D] -> [B, T, H]
        return self.net(x)


class GCNSpatialEncoder(nn.Module):
    """gcn simple sobre el grafo de articulaciones"""

    def __init__(self, input_dim, hidden_dims=(256, 256), dropout=0.2, n_coords=9):
        super().__init__()
        assert input_dim % n_coords == 0, "input_dim debe ser nodos * 9 (pos+vel+acel en xyz)"
        self.n_nodes = input_dim // n_coords
        self.n_coords = n_coords
        A = torch.eye(self.n_nodes)
        # cadena simple entre nodos consecutivos
        for i in range(self.n_nodes - 1):
            A[i, i + 1] = A[i + 1, i] = 1.0
        A = A / A.sum(1, keepdim=True)
        self.register_buffer("A", A)
        dims = [n_coords] + list(hidden_dims)
        self.gcs = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1))
        self.drop = nn.Dropout(dropout)
        self.output_dim = hidden_dims[-1]

    def forward(self, x):  # [B, T, D]
        B, T, D = x.shape
        h = x.view(B, T, self.n_nodes, self.n_coords)
        for gc in self.gcs:
            h = torch.einsum("nm,btmc->btnc", self.A, h)
            h = self.drop(torch.relu(gc(h)))
        return h.mean(dim=2)  # [B, T, H] pooling sobre nodos


def build_spatial_encoder(cfg, input_dim):
    t = cfg.get("type", "mlp")
    if t == "gcn":
        return GCNSpatialEncoder(input_dim, tuple(cfg["hidden_dims"]), cfg["dropout"])
    return MLPSpatialEncoder(input_dim, tuple(cfg["hidden_dims"]), cfg["dropout"])
