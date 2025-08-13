import torch
import torch.nn as nn
from src.base.model import BaseModel


class MLP(BaseModel):
    def __init__(self, hidden_dim=128, num_layers=2, dropout=0.1, **args):
        super(MLP, self).__init__(**args)
        self.dropout = nn.Dropout(dropout)

        input_size = self.input_dim * self.seq_len
        output_size = self.output_dim * self.horizon

        layers = []
        last = input_size
        for _ in range(max(0, num_layers - 1)):
            layers.append(nn.Linear(last, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            last = hidden_dim
        layers.append(nn.Linear(last, output_size))
        self.mlp = nn.Sequential(*layers)

    def forward(self, input, label=None):  # (b, t, n, f)
        b, t, n, f = input.shape
        x = input.reshape(b * n, t * f)
        y = self.mlp(x)
        y = y.view(b, n, self.horizon, self.output_dim).permute(0, 2, 1, 3)
        return y

