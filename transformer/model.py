import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class SignLanguageTransformer(nn.Module):
    def __init__(self, 
                 num_classes,
                 d_model=256,
                 nhead=8,
                 num_layers=4,
                 dim_feedforward=512,
                 dropout=0.1,
                 max_seq_len=100,
                 feature_dim=150):
        super().__init__()
        
        self.d_model = d_model
        self.feature_dim = feature_dim
        
        # Proyección de KPCA features a embeddings
        self.input_projection = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # Clasificador
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x, mask=None):
        # Proyección
        x = self.input_projection(x)
        
        # Positional encoding
        x = self.pos_encoder(x)
        
        # Transformer
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # Global pooling
        x = x.mean(dim=1)
        
        # Clasificación
        logits = self.classifier(x)
        
        return logits