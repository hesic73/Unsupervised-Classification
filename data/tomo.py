import torch
import os
import numpy as np
from PIL import Image, ImageOps

from torch.utils.data import Dataset
from utils.mypath import MyPath


class TomoDataset(Dataset):
    """miaomiaomiao
    """
    def __init__(self,
                 root: str = MyPath.db_root_dir('tomo'),
                 split: str = "train",
                 transform=None) -> None:
        assert split in ['train', 'test']
        super().__init__()
        self.root = root
        self.transform = transform
        self.img_size = (30, 30)
        arrs = []

        for file in os.listdir(root):
            file = os.path.join(root, file)
            img = Image.open(file)
            img = ImageOps.autocontrast(img, cutoff=5)
            arr = np.array(img)
            if arr.shape[:2] == self.img_size:
                arrs.append(arr)
                arrs.append(np.array(img.rotate(90)))
                arrs.append(np.array(img.rotate(180)))
                arrs.append(np.array(img.rotate(270)))

        self.data = np.stack(arrs, axis=0)

    def __getitem__(self, index: int):
        img = self.data[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        out = {
            'image': img,
            'target': 0,
            'meta': {
                'im_size': self.img_size,
                'index': index,
                'class': 'unknown'
            }
        }

        return out

    def __len__(self):
        return len(self.data)