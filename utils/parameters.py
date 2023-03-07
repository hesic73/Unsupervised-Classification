import torch
from torch import Tensor
from torch import nn
from torch.nn import Module
from torch.nn.parameter import Parameter

from typing import Callable


def freeze_parameters(m: Module):
    for n, p in m.named_parameters():
        p.requires_grad = False
        
def unfreeze_parameters(m:Module):
    for n, p in m.named_parameters():
        p.requires_grad = True
