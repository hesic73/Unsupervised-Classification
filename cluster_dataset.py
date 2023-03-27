"""
classify images based on the results of DeepDPM
"""

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, ConcatDataset
import os

import argparse
from argparse import Namespace
from tqdm import tqdm

from utils.common_config import \
    get_val_transformations, \
    get_train_dataset, get_val_dataset
from utils.config import create_exp_config
from utils.collate import collate_custom


def get_args() -> Namespace:
    parser = argparse.ArgumentParser(description='用DeepDPM的结果对图像进行分类')
    parser.add_argument('--config_exp',
                        type=str,
                        default="custom/tomo/baseline.yml")
    parser.add_argument(
        '--output_dir',
        type=str,
        default="/Share/UserHome/tzhao/2023/sicheng/GraduationDesign/data/tomo_deepDPM_results",
    )
    parser.add_argument(
        '--clusters_path',
        type=str,
        default="/Share/UserHome/tzhao/2023/sicheng/GraduationDesign/DeepDPM/results/tomo_clusters.npy",
    )

    args = parser.parse_args()

    return args


def get_dataset(p) -> DataLoader:
    t = None
    train_dataset = get_train_dataset(p, t)
    val_dataset = get_val_dataset(p, t)
    print(
        f"train/val dataset lengths: {len(train_dataset)}/{len(val_dataset)}")
    dataset = ConcatDataset([train_dataset, val_dataset])
    return dataset


def mkdir_if_not_exist(dir: str):
    if os.path.exists(dir):
        return
    os.mkdir(dir)


if __name__ == "__main__":
    args = get_args()
    clusters = np.load(args.clusters_path, allow_pickle=True).item()
    
    config = create_exp_config(args.config_exp)
    dataset = get_dataset(config)

    for id, indexes in clusters.items():
        subdir = os.path.join(args.output_dir, f"{id:>2d}")
        mkdir_if_not_exist(subdir)

        for index in tqdm(indexes, desc=f"{id:>2d}"):
            img = dataset.__getitem__(index)['image']
            img.save(os.path.join(subdir, f"{index:>5}.png"))
