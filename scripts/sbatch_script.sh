#!/bin/bash

#SBATCH -o outputs/job%j.txt
#SBATCH --error errors/job%j.txt 
#SBATCH -p GPU2
#SBATCH --qos=normal
#SBATCH -J SimCLR
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:1
#SBATCH --mail-user hesicheng2001@163.com  
#SBATCH --chdir /Share/UserHome/tzhao/2023/sicheng/GraduationDesign/Unsupervised-Classification

RUN_NAME=baseline
CONFIG_FILE=baseline.yml

date +%c
hostname
pwd
which python

python simclr.py \
--config_exp custom/configs/$CONFIG_FILE \
--run_name $RUN_NAME \
--root_dir root_dir/$RUN_NAME \
--wandb_mode offline
echo 跑完了喵