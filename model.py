import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import random

class NALL(nn.Linear):
    # LoRA implemented in a dense layer
    def __init__(self, in_features, out_features, r=10, lora_alpha=8, lora_dropout=0.3,**kwargs):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)),requires_grad=True)
        self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)),requires_grad=True)
        self.ranknum = nn.Parameter(self.weight.new_zeros(1), requires_grad=False)
        self.lora_alpha = lora_alpha
        self.r = r
        self.lora_dropout = nn.Dropout(p=lora_dropout)
        self.scaling = self.lora_alpha / self.r
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False  # how can I do better with this?

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5.))
        nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        if mode:
            self.weight.data -= (self.lora_B @ self.lora_A) * self.scaling
        else:
            self.weight.data += (self.lora_B @ self.lora_A) * self.scaling

    def forward(self, x: torch.Tensor):
        result = F.linear(x, self.weight, bias=self.bias)            
        result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) \
                    * self.scaling  / (self.ranknum+1e-5)
        return result

# def replace_layer_with_lora(module):
#     for name, child in module.named_children():
#         if isinstance(child, nn.Linear):
#             setattr(module, name, lora.Linear(child.in_features, child.out_features))
#         elif isinstance(child, (nn.Conv2d, nn.Conv1d)):
#             setattr(module, name, lora.Conv2d(child.in_channels, child.out_channels, child.kernel_size))
#         elif isinstance(child, nn.ModuleList):
#             for _, subchild in enumerate(child):
#                 subchild = replace_layer_with_lora(subchild)
#         child.requires_grad_(requires_grad=False)
#     return module

class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,v->ncvl',(x, A))
        return x.contiguous()
    
class GCN(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(GCN, self).__init__()
        self.nconv = nconv()
        c_in = (order * support_len + 1) * c_in
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
           #print("shape:",x.shape,a.shape)
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = F.dropout(h, self.dropout, training=self.training)
        h = F.batch_norm(h, h.mean(dim=0), h.std(dim=0), training=self.training) #
        return h


class Linear_Adapter(nn.Module):
    def __init__(self, input_dim, output_dim, horizon, hidden_dim, num_layers, supports,
                 lora_r=8, lora_alpha=16, lora_dropout=0.3,linear=False,lagcn=False):
        super(Linear_Adapter, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.supports = supports
        self.lagcn = lagcn
        # self.nall_start = NALL(in_features=input_dim,out_features=self.hidden_dim,r=lora_r,lora_alpha=lora_alpha,lora_dropout=lora_dropout)
        self.start_conv = torch.nn.Conv2d(in_channels=input_dim, out_channels=hidden_dim * horizon, kernel_size=(1,1),padding=(0,0),stride=(1,1),bias=True)
        self.nalls = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.bn = nn.ModuleList()
        for _ in range(num_layers):
            if linear:
                self.nalls.append(nn.Linear(in_features=self.hidden_dim,
                                out_features=self.hidden_dim,
                                bias=True)
                )
            else:
                self.nalls.append(NALL(in_features=self.hidden_dim * horizon ,
                                    out_features=self.hidden_dim * horizon,
                                    r=lora_r,
                                    lora_alpha=lora_alpha,
                                    lora_dropout=lora_dropout)
                )
            if self.lagcn:
                self.gconv.append(GCN(c_in=self.hidden_dim, c_out=self.hidden_dim, dropout=lora_dropout, support_len=len(self.supports)))
            self.bn.append(nn.BatchNorm1d(self.hidden_dim*horizon))

        self.batch_norm1 = nn.BatchNorm1d(self.hidden_dim)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(lora_dropout)
        self.nall_end = NALL(in_features=self.hidden_dim * horizon, out_features=output_dim,r=lora_r,lora_alpha=lora_alpha,lora_dropout=lora_dropout)
        self.end_conv = torch.nn.Conv1d(in_channels=hidden_dim*horizon, out_channels=horizon, kernel_size=1, bias=True)
        # self.batch_norm2 = nn.BatchNorm1d(self.output_dim)

    def forward(self, x, label=None):
        B, T, N, D = x.shape
        x = x.transpose(1, 3)
        # x = x.reshape(B * T * N, D)
        x = x.transpose(1, 2).reshape(B * N, D, 1, T)
        x = self.start_conv(x).squeeze().transpose(1, 2)
        # x = self.batch_norm1(x.unsqueeze(-1))
        # x = self.leaky_relu(x)
        # x = self.dropout(x)
        #print("x shape:",x.shape)
        for i in range(len(self.nalls)):
            x = self.nalls[i](x).transpose(1, 2)
            # print("x shape:",x.shape)
            x = self.leaky_relu(x)
            x = self.dropout(x)
            x = self.bn[i](x).transpose(1, 2)
        # print("x shape:",x.shape) #[19648,12,240]
        # x = self.end_conv(x.transpose(1,2)).squeeze()
        # x = F.relu(x[:, -1, :])
        x = self.nall_end(x)
        # print("x shape:",x.shape) #[19648,12]
        # x = self.end_conv(x)
        # x = self.leaky_relu(x)
        # x = self.batch_norm2(x)
        output = x.reshape(B, T, N, self.output_dim)

        return output

class LAST(nn.Module):
    def __init__(self, device, node_num,input_dim, output_dim, horizon, model, supports,
                 frozen=False, lagcn=False, embed_dim=12, num_layers=4, num_blocks=1,
                 la_dropout=0.3, last_lr=1e-4,last_weight_decay=1e-5, last_pool_type='absmin'):
        super(LAST, self).__init__()
        self.device = device
        self.num_node = node_num
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.horizon = horizon
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.pre_model = model
        self.supports = supports
        self.dropout = nn.Dropout(la_dropout)
        self.lr = last_lr # 2e-4 default -> 1e-3
        self.weight_decay = last_weight_decay #5e-5 default -> 1e-4
        self.linear = False
        self.lagcn = lagcn
        self.pool_type = last_pool_type
        if frozen:
            for name, value in self.pre_model.named_parameters():
                    value.requires_grad_(requires_grad=False)

        self.embed_dim = embed_dim
        self.adapter = nn.ModuleList()        
        #self.gconv = nn.ModuleList()
        self.bn = nn.ModuleList()
        for _ in range(num_blocks):
            self.adapter.append(Linear_Adapter( input_dim=self.output_dim,
                                                output_dim=self.output_dim,
                                                horizon = self.horizon,
                                                hidden_dim=self.output_dim * self.embed_dim,
                                                num_layers=self.num_layers,
                                                supports=self.supports,
                                                lora_dropout=la_dropout,
                                                linear = self.linear,
                                                lagcn=lagcn)
                    )
            self.bn.append(nn.BatchNorm1d(self.output_dim))
    
        if self.lagcn:
            self.gconv = GCN(c_in=self.input_dim, c_out=self.output_dim, dropout=0.1, support_len=len(self.supports))
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

    def forward(self, x, label=None, iter=None):
        B,T,N,D = x.shape
        # output = self.pre_model(x,label,iter) #dcrnn
        output = self.pre_model(x,iter)         
        tunings = []
        for i in range(self.num_blocks):
            tmp =  output * F.softmax(F.leaky_relu(self.adapter[i](output)[:, :, :, -self.output_dim:]),dim=-1)
            tunings.append(tmp)
        tuning_stack = torch.stack(tunings, dim=0)
        
        if not self.pool_type in ['mean', 'min', 'max', 'absmin']:
            self.pool_type = random.choice(['mean', 'min', 'max'])
        if self.pool_type == 'mean':
            output = output + torch.mean(tuning_stack, dim=0) 
        elif self.pool_type == 'min':
            output = output + torch.min(tuning_stack, dim=0) 
        elif self.pool_type == 'max':
            output = output + torch.max(tuning_stack, dim=0)
        elif self.pool_type == 'absmin':
            abs_values = torch.abs(tuning_stack)
            closest_to_zero_index = torch.argmin(abs_values, dim=0)
            closest_to_zero_values = torch.take(tuning_stack, closest_to_zero_index)
            output = output + closest_to_zero_values
        
        if self.lagcn: 
            output += F.softmax(F.relu(self.gconv(x, self.supports)[:,-self.horizon:,:,-self.output_dim:]), dim=-1)
            

        return output

    def backward(self, loss):
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)  # 添加梯度裁剪
            # 检查梯度裁剪后的梯度是否过大
        total_norm = 0
        for p in self.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)

        if total_norm > 1.0:  # 如果梯度过大，降低学习率
            self.scheduler.step(total_norm)
        
        self.optimizer.step()
        self.optimizer.zero_grad()