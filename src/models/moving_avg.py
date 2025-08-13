import torch
import torch.nn as nn
from src.base.model import BaseModel


class MovingAvg(BaseModel):
    def __init__(self, ma_window=3, **args):
        super(MovingAvg, self).__init__(**args)
        self.ma_window = ma_window

    def forward(self, input, label=None):  # (b, t, n, f)
        b, t, n, f = input.shape
        w = min(self.ma_window, t)
        x = input[:, -w:, :, :].mean(dim=1, keepdim=True)
        y = x.expand(b, self.horizon, n, self.output_dim)
        return y

