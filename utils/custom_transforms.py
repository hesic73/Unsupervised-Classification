from PIL import Image, ImageOps
import random
import torchvision.transforms.functional as TF


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
