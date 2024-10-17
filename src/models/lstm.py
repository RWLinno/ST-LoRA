import torch.nn as nn
import torch.nn.functional as F
from src.base.model import BaseModel

class LSTM(BaseModel):
    def __init__(self, init_dim, hid_dim, end_dim, layer, dropout, **args):
        super(LSTM, self).__init__(**args)   
        self.start_conv = nn.Conv2d(in_channels=self.input_dim, 
                                    out_channels=init_dim, 
                                    kernel_size=(1,1))

        self.lstm = nn.LSTM(input_size=init_dim, hidden_size=hid_dim, num_layers=layer, batch_first=True, dropout=dropout)
        
        self.end_linear1 = nn.Linear(hid_dim, end_dim)
        self.end_linear2 = nn.Linear(end_dim, self.horizon)


    def forward(self, input, label=None):  # (b, t, n, f)
        x = input.transpose(1, 3)
        b, f, n, t = x.shape

        x = x.transpose(1,2).reshape(b*n, f, 1, t)
        x = self.start_conv(x).squeeze().transpose(1, 2)

        out, _ = self.lstm(x)
        x = out[:, -1, :]

        x = F.relu(self.end_linear1(x))
        x = self.end_linear2(x)
        x = x.reshape(b, n, t, 1).transpose(1, 2)
        return x

'''
import mxnet as mx
from mxnet import nd
from mxnet.gluon import rnn
from mxnet.gluon import nn as mnn

class LSTM_m(mnn.Block):

    # This is a LSTM extension model, which accepts the last Tr samples and yiels next Tp samples, (num, N, T) -> (num, N, Tp).

    def __init__(self, init_dim, hid_dim, end_dim, layer, dropout, Tp=3, num_layers=7, **kwargs):
        super(LSTM_m, self).__init__(**kwargs)
        self.init_dim = init_dim
        self.hid_dim = hid_dim
        self.end_dim = end_dim
        self.layer = layer
        self.dropout = dropout
        self.Tp = Tp
        self.lstm = rnn.LSTM(hidden_size=Tp, num_layers=num_layers) #default layout, 'TNC' = 'sequence length, batch size, feature dimensions'
        self.fc = mnn.Dense(Tp, activation='relu',flatten=False) #map elements of ndarray from [0,1] to [0,\inf]

    def forward(self, x):
        num, N, T = x.shape
        out = x.transpose(0,1) #(sequence_length, batch_size, input_size) <-> (N, num, T)
        out = self.lstm(out) #(sequence_length, batch_size, num_hidden) <-> (N, num, Tp)
        out = self.fc(out) #(N, num, Tp)
        out = nd.transpose(out, axes=(1,0,2)) #(num, N, Tp)
        return out

'''