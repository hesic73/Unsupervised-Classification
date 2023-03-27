import os
import sys
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from utils.mypath import MyPath
from torch import Tensor

from typing import Tuple, Optional, Callable

from torchvision.transforms import Compose, CenterCrop


class Proteasome(Dataset):

    # Attention: Here I assume train_set and val_set have the same number of classes,
    # which isn't the case in the original .npy files. So I shuffled the arrays.

    def __init__(self,
                 root: str = MyPath.db_root_dir('proteasome-12'),
                 train: bool = True,
                 transform: Optional[Callable[[Image.Image], Tensor]] = None,
                 autocontrast: bool = False,
                 centercrop: bool = False,
                 **kwargs):

        super(Proteasome, self).__init__()
        self.root = root
        self.transform = transform
        self.train = train  # training set or test set
        self.autocontrast = autocontrast
        self.centercrop = CenterCrop(
            kwargs['centercrop_size']) if centercrop else None

        if train:
            data_filename = 'train_data.npy'
            labels_filename = 'train_labels.npy'
        else:
            data_filename = 'test_data.npy'
            labels_filename = 'test_labels.npy'

        # (n,80,80)
        self.pictures = np.load(os.path.join(self.root, data_filename))
        # (n,)
        self.labels = np.load(os.path.join(self.root, labels_filename))

        if self.pictures.dtype == np.float32 or self.pictures.dtype == np.float64:
            self.pictures = (self.pictures * 255).astype(np.uint8)

        self.n = self.pictures.shape[0]

        # (n,80,80,1)
        # self.pictures = np.expand_dims(self.pictures, axis=-1)
        # (n,80,80,3)
        # self.pictures = self.pictures.repeat(3, axis=-1)

        self.labels = torch.from_numpy(self.labels.reshape(-1, 1))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            dict: {'image': image, 'target': index of target class, 'meta': dict}
        """
        img, target = self.pictures[index], self.labels[index]
        img_size = (img.shape[0], img.shape[1])
        img = Image.fromarray(img)

        if self.autocontrast:
            img = ImageOps.autocontrast(img, cutoff=5)

        if self.centercrop is not None:
            img = self.centercrop(img)
            img_size = img.size

        if self.transform is not None:
            img = self.transform(img)

        out = {
            'image': img,
            'target': target,
            'meta': {
                'im_size': img_size,
                'index': index,
                'class': target[0]
            }
        }

        return out

    def get_image(self, index):
        img = self.data[index]
        return img

    def __len__(self):
        return len(self.pictures)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train else "Test")
