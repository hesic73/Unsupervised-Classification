import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import numpy as np
from PIL import Image

import argparse
from argparse import Namespace
import typing
from tqdm import tqdm

from models.models import ContrastiveModel
from utils.common_config import get_model, get_val_transformations
from utils.config import create_config


class CustomDataset(Dataset):
    def __init__(self, data_path: str, transforms: transforms.Compose) -> None:
        self.data = np.load(data_path)
        self.data = np.expand_dims(self.data, axis=-1)
        self.data = self.data.repeat(3, axis=-1)
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        img = Image.fromarray(self.data[index])
        return self.transforms(img)


def get_dataloader(data_path: str,
                   transforms: transforms.Compose) -> DataLoader:
    dataset = CustomDataset(data_path, transforms)
    return DataLoader(dataset, batch_size=256)


def get_args() -> Namespace:
    parser = argparse.ArgumentParser(description='用训练好的模型提取特征')
    parser.add_argument(
        '--config_env',
        type=str,
        default=
        'custom/configs/env.yml',
        help=
        'no use here.(But it must be a valid yml file for compatibility with other settings.)'
    )
    parser.add_argument(
        '--config_exp',
        type=str,
        default=
        "custom/configs/extract_features.yml",
        help='Config file for the experiment')
    parser.add_argument(
        '--model_path',
        type=str,
        default=
        "/Share/UserHome/tzhao/2023/SCAN/results/proteasome-topaz-denoise/proteasome/pretext/checkpoint.pth.tar",
        help='path of the trained_model')
    parser.add_argument(
        '--data_path',
        type=str,
        default=
        "/Share/UserHome/tzhao/2023/sicheng/GraduationDesign/data/data_denoised.npy",
        help='path of the .npy file')
    parser.add_argument(
        '--save_path',
        type=str,
        default=
        "/Share/UserHome/tzhao/2023/sicheng/GraduationDesign/data/extracted_features.pt",
    )

    args = parser.parse_args()

    return args


@torch.no_grad()
def inference(model: ContrastiveModel, dataloader: DataLoader, save_path: str):
    model.cuda()
    outputs = []
    for x in tqdm(dataloader, "extracting features..."):
        x = x.cuda(non_blocking=True)
        y = model(x)
        outputs.append(y.detach().clone().cpu())

    outputs = torch.cat(outputs)
    print(outputs.shape)
    torch.save(outputs, save_path)
    print(f"save at {save_path}.")


if __name__ == "__main__":
    torch.cuda.set_device(1)
    args = get_args()
    config = create_config(args.config_env, args.config_exp)
    model = get_model(config)
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    model.eval()
    dataloader = get_dataloader(args.data_path,
                                get_val_transformations(config))
    inference(model, dataloader, args.save_path)
