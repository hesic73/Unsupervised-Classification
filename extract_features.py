import torch
from torch.utils.data import DataLoader, ConcatDataset
import os

import argparse
from argparse import Namespace
from tqdm import tqdm

from models.models import ContrastiveModel
from utils.common_config import get_model, \
    get_val_transformations, \
     get_train_dataset, get_val_dataset
from utils.config import create_exp_config
from utils.collate import collate_custom


def get_args() -> Namespace:
    parser = argparse.ArgumentParser(description='用训练好的模型提取特征')
    parser.add_argument('--config_exp',
                        type=str,
                        default="custom/cng/baseline.yml")
    parser.add_argument(
        '--model_path',
        type=str,
        default="root_dir/SimCLR_CNG/new_baseline/cng/pretext/model.pth.tar",
        help='path of the trained_model')
    parser.add_argument(
        '--output_dir',
        type=str,
        default=
        "/Share/UserHome/tzhao/2023/sicheng/GraduationDesign/data/cng_features",
    )

    args = parser.parse_args()

    return args

def get_dataloader(p)->DataLoader:
    t=get_val_transformations(p)
    train_dataset=get_train_dataset(p,t)
    val_dataset=get_val_dataset(p,t)
    print(
        f"train/val dataset lengths: {len(train_dataset)}/{len(val_dataset)}")
    dataset=ConcatDataset([train_dataset,val_dataset])
    return DataLoader(dataset,
               num_workers=p['num_workers'],
               batch_size=p['batch_size'],
               pin_memory=True,
               collate_fn=collate_custom,
               shuffle=False)


@torch.no_grad()
def inference(model: ContrastiveModel, dataloader: DataLoader, output_dir: str):
    model.cuda()
    features=[]
    labels=[]
    for b in tqdm(dataloader, "extracting features..."):
        x=b['image']
        l=b['target']
        x = x.cuda(non_blocking=True)
        y = model(x)
        features.append(y.detach().clone().cpu())
        labels.append(l.detach().clone())

    features = torch.cat(features)
    print(features.shape)
    labels = torch.cat(labels)
    print(labels.shape)
    print(f"save at {output_dir}.")
    torch.save(features, os.path.join(output_dir,"train_data.pt"))
    torch.save(labels, os.path.join(output_dir,"train_labels.pt"))


if __name__ == "__main__":
    args = get_args()
    config = create_exp_config(args.config_exp)
    model = get_model(config)
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    model.eval()

    dataloader=get_dataloader(config)

    inference(model, dataloader, args.output_dir)
