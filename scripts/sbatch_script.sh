#!/bin/bash

#SBATCH -o outputs/job%j.txt
#SBATCH --error errors/job%j.txt 
#SBATCH -p GPU2
#SBATCH --qos=normal
#SBATCH -J simclr
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:1
#SBATCH --mail-user hesicheng2001@163.com  
#SBATCH --chdir /Share/UserHome/tzhao/2023/sicheng/GraduationDesign/Unsupervised-Classification

date +%c
hostname
pwd
which python
RUN_NAME=resnet18_256
python simclr.py --config_env custom/configs/env.yml \
--config_exp custom/configs/proteasome-256.yml \
--run_name $RUN_NAME \
--root_dir root_dir/$RUN_NAME \
--wandb_mode offline
echo 跑完了喵