#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn

from typing import Dict

from .layers import LoRALayer

# 将模型中的参数设置为是否可训练
def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    for n, p in model.named_parameters():
        if 'lora_' not in n:
            p.requires_grad = False 
    if bias == 'none':
        return
    elif bias == 'all':   # 所有参数中包含 'lora_' 或 'bias' 的参数都被设置为可训练
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    elif bias == 'lora_only':  #仅包含在 LoRALayer 模块中的参数被设置为可训练
        for m in model.modules():
            if isinstance(m, LoRALayer) and \
                hasattr(m, 'bias') and \
                m.bias is not None:
                    m.bias.requires_grad = True
    else:
        raise NotImplementedError

# 用于获取模型的状态字典
def lora_state_dict(model: nn.Module, bias: str = 'none') -> Dict[str, torch.Tensor]:
    my_state_dict = model.state_dict()
    if bias == 'none': # 返回所有参数中包含 'lora_' 的状态字典
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k}
    elif bias == 'all': # 返回所有参数中包含 'lora_' 或 'bias' 的状态字典
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k or 'bias' in k}
    elif bias == 'lora_only': # 返回仅包含在 LoRALayer 模块中的参数和对应的偏置项的状态字典。
        to_return = {}
        for k in my_state_dict:
            if 'lora_' in k:
                to_return[k] = my_state_dict[k]
                bias_name = k.split('lora_')[0]+'bias'
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    else:
        raise NotImplementedError
