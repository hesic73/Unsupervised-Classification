"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import argparse
import os
import torch
import numpy as np
import time

from utils.config import create_config
from utils.common_config import get_criterion, get_model, get_train_dataset,\
    get_val_dataset, get_train_dataloader,\
    get_val_dataloader, get_train_transformations,\
    get_val_transformations, get_optimizer,\
    adjust_learning_rate
from utils.evaluate_utils import contrastive_evaluate
from utils.memory import MemoryBank
from utils.train_utils import simclr_train
from utils.utils import fill_memory_bank
from termcolor import colored


import wandb
from typing import Optional
from torch.utils.data import DataLoader

# Parser
parser = argparse.ArgumentParser(description='SimCLR')
parser.add_argument('--config_env',
                    help='Config file for the environment')
parser.add_argument('--config_exp',
                    help='Config file for the experiment')
parser.add_argument(
    '--run_name', type=str, default=None, help='wandb run\'s name')

parser.add_argument(
    '--wandb_mode', type=str, default=None, choices=['online', 'offline', 'disabled'], help='wandb mode')

args = parser.parse_args()


def main():

    # Retrieve config file
    p = create_config(args.config_env, args.config_exp)
    print(colored(p, 'red'))

    # Model
    print(colored('Retrieve model', 'blue'))
    model = get_model(p)
    print('Model is {}'.format(model.__class__.__name__))
    print('Model parameters: {:.2f}M'.format(
        sum(p.numel() for p in model.parameters()) / 1e6))
    print(model)
    model = model.cuda()

    # CUDNN
    print(colored('Set CuDNN benchmark', 'blue'))
    torch.backends.cudnn.benchmark = True

    # Dataset
    print(colored('Retrieve dataset', 'blue'))
    train_transforms = get_train_transformations(p)
    print('Train transforms:', train_transforms)
    val_transforms = get_val_transformations(p)
    print('Validation transforms:', val_transforms)
    train_dataset = get_train_dataset(p, train_transforms, to_augmented_dataset=True,
                                      split='train+unlabeled')  # Split is for stl-10
    val_dataset = get_val_dataset(p, val_transforms)
    train_dataloader = get_train_dataloader(p, train_dataset)
    val_dataloader = get_val_dataloader(p, val_dataset)
    print('Dataset contains {}/{} train/val samples'.format(len(train_dataset), len(val_dataset)))

    # Memory Bank
    print(colored('Build MemoryBank', 'blue'))
    # Dataset w/o augs for knn eval
    base_dataset = get_train_dataset(p, val_transforms, split='train')
    base_dataloader = get_val_dataloader(p, base_dataset)
    memory_bank_base = MemoryBank(len(base_dataset),
                                  p['model_kwargs']['features_dim'],
                                  p['num_classes'], p['criterion_kwargs']['temperature'])
    memory_bank_base.cuda()
    memory_bank_val = MemoryBank(len(val_dataset),
                                 p['model_kwargs']['features_dim'],
                                 p['num_classes'], p['criterion_kwargs']['temperature'])
    memory_bank_val.cuda()

    # Criterion
    print(colored('Retrieve criterion', 'blue'))
    criterion = get_criterion(p)
    print('Criterion is {}'.format(criterion.__class__.__name__))
    criterion = criterion.cuda()

    # Optimizer and scheduler
    print(colored('Retrieve optimizer', 'blue'))
    optimizer = get_optimizer(p, model)
    print(optimizer)

    # Checkpoint
    if os.path.exists(p['pretext_checkpoint']):
        print(colored('Restart from checkpoint {}'.format(
            p['pretext_checkpoint']), 'blue'))
        checkpoint = torch.load(p['pretext_checkpoint'], map_location='cpu')
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.load_state_dict(checkpoint['model'])
        model.cuda()
        start_epoch = checkpoint['epoch']

    else:
        print(colored('No checkpoint file at {}'.format(
            p['pretext_checkpoint']), 'blue'))
        start_epoch = 0
        model = model.cuda()

    # Training
    print(colored('Starting main loop', 'blue'))

    run = wandb.init(project="SimCLR",
                     name=args.run_name, mode=args.wandb_mode)

    for epoch in range(start_epoch, p['epochs']):
        print(colored('Epoch %d/%d' % (epoch, p['epochs']), 'yellow'))
        print(colored('-'*15, 'yellow'))

        # Adjust lr
        lr = adjust_learning_rate(p, optimizer, epoch)

        # Train
        loss = simclr_train(train_dataloader, model,
                            criterion, optimizer, epoch)

        run.log({'loss': loss, 'lr': lr}, step=epoch)

        if epoch % 10 == 0:
            fill_memory_bank(val_dataloader, model, memory_bank_val)
            _, acc_val_5 = memory_bank_val.mine_nearest_neighbors(5)

            fill_memory_bank(base_dataloader, model, memory_bank_base)
            _, acc_train_20 = memory_bank_base.mine_nearest_neighbors(20)

            top1 = contrastive_evaluate(
                val_dataloader, model, memory_bank_base)

            run.log({'val top5 acc': acc_val_5*100,
                    'train top20 acc': acc_train_20*100, 'kNN eval': top1}, step=epoch)

            # Checkpoint
            print('Checkpoint ...')
            torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(),
                        'epoch': epoch + 1}, p['pretext_checkpoint'])

    # Save final model
    torch.save(model.state_dict(), p['pretext_model'])

    # Mine the topk nearest neighbors at the very end (Train)
    # These will be served as input to the SCAN loss.
    print(colored('Fill memory bank for mining the nearest neighbors (train) ...', 'blue'))
    fill_memory_bank(base_dataloader, model, memory_bank_base)
    topk = 20
    print('Mine the nearest neighbors (Top-%d)' % (topk))
    indices, acc = memory_bank_base.mine_nearest_neighbors(topk)
    print('Accuracy of top-%d nearest neighbors on train set is %.2f' %
          (topk, 100*acc))

    run.log({f"top-{topk} accuracy": 100*acc})
    np.save(p['topk_neighbors_train_path'], indices)

    # Mine the topk nearest neighbors at the very end (Val)
    # These will be used for validation.
    print(colored('Fill memory bank for mining the nearest neighbors (val) ...', 'blue'))
    fill_memory_bank(val_dataloader, model, memory_bank_val)
    topk = 5
    print('Mine the nearest neighbors (Top-%d)' % (topk))
    indices, acc = memory_bank_val.mine_nearest_neighbors(topk)
    print('Accuracy of top-%d nearest neighbors on val set is %.2f' %
          (topk, 100*acc))
    np.save(p['topk_neighbors_val_path'], indices)


if __name__ == '__main__':
    main()
