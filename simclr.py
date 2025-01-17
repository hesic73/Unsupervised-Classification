"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import argparse
import os
import torch
import numpy as np
from pprint import pprint

from utils.config import create_exp_config
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


def get_args():
    # Parser
    parser = argparse.ArgumentParser(description='SimCLR')
    parser.add_argument('--config_exp', help='Config file for the experiment')
    parser.add_argument(
        '--root_dir', help='root directory for saving checkpoints etc')
    parser.add_argument('--project', type=str,
                        default=None, help='wandb project\'s name')
    parser.add_argument('--run_name', type=str,
                        default=None, help='wandb run\'s name')
    parser.add_argument('--wandb_mode', type=str, default=None,
                        choices=['online', 'offline', 'disabled'], help='wandb mode. It seems on slurm nodes online mode doesn\'t work')
    parser.add_argument('--manually_load_model', type=str,
                        default=None, help='load model msnually.')

    args = parser.parse_args()
    return args


def main(config):

    # Model
    print(colored('Retrieve model', 'blue'))
    model = get_model(config)
    print('Model is {}'.format(model.__class__.__name__))
    print('Model parameters: {:.2f}M'.format(
        sum(p.numel() for p in model.parameters()) / 1e6))
    if config.manually_load_model is not None:
        checkpoint = torch.load(config.manually_load_model, map_location='cpu')
        model.load_state_dict(checkpoint)
    # print(model)
    model = model.cuda()

    # CUDNN
    print(colored('Set CuDNN benchmark', 'blue'))
    torch.backends.cudnn.benchmark = True

    # Dataset
    print(colored('Retrieve dataset', 'blue'))
    train_transforms = get_train_transformations(config)
    print('Train transforms:', train_transforms)
    val_transforms = get_val_transformations(config)
    print('Validation transforms:', val_transforms)
    train_dataset = get_train_dataset(config, train_transforms, to_augmented_dataset=True,
                                      split='train+unlabeled')  # Split is for stl-10
    val_dataset = get_val_dataset(config, val_transforms)
    train_dataloader = get_train_dataloader(config, train_dataset)
    val_dataloader = get_val_dataloader(config, val_dataset)
    print('Dataset contains {}/{} train/val samples'.format(len(train_dataset), len(val_dataset)))

    # Memory Bank
    print(colored('Build MemoryBank', 'blue'))
    # Dataset w/o augs for knn eval
    base_dataset = get_train_dataset(config, val_transforms, split='train')
    base_dataloader = get_val_dataloader(config, base_dataset)
    memory_bank_base = MemoryBank(len(base_dataset),
                                  config['model_kwargs']['features_dim'],
                                  config['num_classes'], p['criterion_kwargs']['temperature'])
    memory_bank_base.cuda()
    memory_bank_val = MemoryBank(len(val_dataset),
                                 config['model_kwargs']['features_dim'],
                                 config['num_classes'], p['criterion_kwargs']['temperature'])
    memory_bank_val.cuda()

    # Criterion
    print(colored('Retrieve criterion', 'blue'))
    criterion = get_criterion(p)
    print('Criterion is {}'.format(criterion.__class__.__name__))
    criterion = criterion.cuda()

    # Optimizer and scheduler
    print(colored('Retrieve optimizer', 'blue'))
    optimizer = get_optimizer(config, model)
    print(optimizer)

    # Checkpoint
    if os.path.exists(config['pretext_checkpoint']):
        print(colored('Restart from checkpoint {}'.format(
            config['pretext_checkpoint']), 'blue'))
        checkpoint = torch.load(p['pretext_checkpoint'], map_location='cpu')
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.load_state_dict(checkpoint['model'])
        model.cuda()
        start_epoch = checkpoint['epoch']

    else:
        print(colored('No checkpoint file at {}'.format(
            config['pretext_checkpoint']), 'blue'))
        start_epoch = 0
        model = model.cuda()

    # Training
    print(colored('Starting main loop', 'blue'))

    run = wandb.init(project="SimCLR" if config.project is None else config.project, config=config,
                     name=config.run_name, mode=config.wandb_mode)

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
            print('Fill memory bank for kNN...')
            fill_memory_bank(base_dataloader, model, memory_bank_base)

            print('Evaluate ...')
            top1 = contrastive_evaluate(
                val_dataloader, model, memory_bank_base)
            print('Result of kNN evaluation is %.2f' % (top1))

            _, acc1 = memory_bank_base.mine_nearest_neighbors(1)
            print(
                'Accuracy of top-{} nearest neighbors on validation set is {:.2f}'.format(1, 100*acc1))

            _, acc5 = memory_bank_base.mine_nearest_neighbors(5)
            print(
                'Accuracy of top-{} nearest neighbors on validation set is {:.2f}'.format(5, 100*acc5))

            _, acc20 = memory_bank_base.mine_nearest_neighbors(20)
            print(
                'Accuracy of top-{} nearest neighbors on validation set is {:.2f}'.format(20, 100*acc20))

            run.log({'Acc-top5': acc5, 'Acc-top20': acc20,
                    'Acc-top1': acc1, 'kNN-eval': top1}, step=epoch)

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
    args = get_args()
    p = create_exp_config(args.config_exp, args.root_dir)
    p.update({
        "run_name": args.run_name,
        "wandb_mode": args.wandb_mode,
             "manually_load_model": args.manually_load_model,
             "project": args.project})
    pprint(p)
    main(p)
