"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import ResNet,BasicBlock
import torch

from typing import Optional

def resnet50():
    backbone = models.__dict__['resnet50']()
    backbone.fc = nn.Identity()
    return {'backbone': backbone, 'dim': 2048}


def resnet18(pretrained_model_path:Optional[str]=None):
    model=ResNet(BasicBlock, [2, 2, 2, 2])
    if pretrained_model_path is not None:
        state_dict=torch.load(pretrained_model_path)
        model.load_state_dict(state_dict)
    model.fc = nn.Identity()
    return {'backbone': model, 'dim': 512}
