import torch
import torch.nn as nn
from typing import Iterable, Optional, Set
from .layers import Linear as LoraLinear, Conv1d as LoraConv1d, Conv2d as LoraConv2d


def _should_wrap(name: str, include: Optional[Set[str]], exclude: Optional[Set[str]]) -> bool:
    if exclude:
        for pat in exclude:
            if pat and pat in name:
                return False
    if include:
        for pat in include:
            if pat and pat in name:
                return True
        return False
    return True


def apply_lora_to_module(
    model: nn.Module,
    target_modules: Iterable[str] = ("Linear", "Conv1d", "Conv2d"),
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    include_names: Optional[Iterable[str]] = None,
    exclude_names: Optional[Iterable[str]] = None,
    merge_weights: bool = True,
) -> nn.Module:
    include = set(include_names) if include_names else None
    exclude = set(exclude_names) if exclude_names else None

    def replace(module: nn.Module, prefix: str = ""):
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            replace(child, full_name)

            if isinstance(child, nn.Linear) and "Linear" in target_modules and _should_wrap(full_name, include, exclude):
                wrapped = LoraLinear(
                    child.in_features,
                    child.out_features,
                    r=r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    bias=(child.bias is not None),
                    merge_weights=merge_weights,
                )
                # copy weights
                with torch.no_grad():
                    wrapped.weight.copy_(child.weight)
                    if child.bias is not None:
                        wrapped.bias.copy_(child.bias)
                setattr(module, name, wrapped)

            elif isinstance(child, nn.Conv1d) and "Conv1d" in target_modules and _should_wrap(full_name, include, exclude):
                wrapped = LoraConv1d(
                    child.in_channels,
                    child.out_channels,
                    child.kernel_size,
                    r=r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    stride=child.stride,
                    padding=child.padding,
                    dilation=child.dilation,
                    groups=child.groups,
                    bias=(child.bias is not None),
                    merge_weights=merge_weights,
                )
                with torch.no_grad():
                    wrapped.conv.weight.copy_(child.weight)
                    if child.bias is not None:
                        wrapped.conv.bias.copy_(child.bias)
                setattr(module, name, wrapped)

            elif isinstance(child, nn.Conv2d) and "Conv2d" in target_modules and _should_wrap(full_name, include, exclude):
                wrapped = LoraConv2d(
                    child.in_channels,
                    child.out_channels,
                    child.kernel_size,
                    r=r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    stride=child.stride,
                    padding=child.padding,
                    dilation=child.dilation,
                    groups=child.groups,
                    bias=(child.bias is not None),
                    merge_weights=merge_weights,
                )
                with torch.no_grad():
                    wrapped.conv.weight.copy_(child.weight)
                    if child.bias is not None:
                        wrapped.conv.bias.copy_(child.bias)
                setattr(module, name, wrapped)

    replace(model)
    return model


