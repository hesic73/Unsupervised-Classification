from PIL import Image, ImageOps
import random
import torchvision.transforms.functional as TF
from torch import Tensor
import torch

import numbers


class CircularCenterCrop:
    def __init__(self, r, size) -> None:
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

        self.mask = torch.empty(self.size, dtype=torch.bool)

        cx = 1.0*self.size[0]/2
        cy = 1.0*self.size[1]/2

        for x in range(self.size[0]):
            for y in range(self.size[1]):
                self.mask[x, y] = (x-cx)**2+(y-cy)**2 <= r*r

    def __call__(self, img: Tensor):
        pass


class DisCreteRandomRotation:
    def __init__(self, angles) -> None:
        self.angles = angles

    def __call__(self, img):
        img = TF.rotate(img, random.choice(self.angles))
        return img


class RandomColorise:
    def __init__(self, p: float = 0.5) -> None:
        # torchvision.transforms.RandomApply doesn't accept such kind of Transforms
        assert 0 <= p <= 1
        self.p = p
        pass

    def __call__(self, img: Image.Image) -> Image.Image:
        toss = random.uniform(0, 1)
        if toss > self.p:
            return toss
