"""capa 3: encoder temporal (bi-lstm por defecto, tcn opcional)"""
import torch
import torch.nn as nn


class BiLSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True,
                            bidirectional=True, dropout=dropout if num_layers > 1 else 0.0)
        self.output_dim = hidden_dim * 2

    def forward(self, x, lengths=None):  # [B, T, H] -> [B, T, 2H]
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False)
            out, _ = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True,
                                                      total_length=x.shape[1])
        else:
            out, _ = self.lstm(x)
        return out


class TCNBlock(nn.Module):
    def __init__(self, c_in, c_out, dilation, dropout):
        super().__init__()
        pad = dilation
        self.conv1 = nn.Conv1d(c_in, c_out, 3, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(c_out, c_out, 3, padding=pad, dilation=dilation)
        self.drop = nn.Dropout(dropout)
        self.res = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        h = self.drop(torch.relu(self.conv1(x)))
        h = self.drop(torch.relu(self.conv2(h)))
        return torch.relu(h + self.res(x))


class TCNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=2, dropout=0.2):
        super().__init__()
        blocks, c = [], input_dim
        for i in range(num_layers):
            blocks.append(TCNBlock(c, hidden_dim, dilation=2 ** i, dropout=dropout))
            c = hidden_dim
        self.net = nn.Sequential(*blocks)
        self.output_dim = hidden_dim

    def forward(self, x, lengths=None):  # [B, T, H]
        return self.net(x.transpose(1, 2)).transpose(1, 2)


def build_temporal_encoder(cfg, input_dim):
    t = cfg.get("type", "bilstm")
    if t == "tcn":
        return TCNEncoder(input_dim, cfg["hidden_dim"], cfg["num_layers"], cfg["dropout"])
    return BiLSTMEncoder(input_dim, cfg["hidden_dim"], cfg["num_layers"], cfg["dropout"])
