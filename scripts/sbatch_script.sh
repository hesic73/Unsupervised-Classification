#!/bin/bash

#SBATCH -o outputs/job%j.txt
#SBATCH --error errors/job%j.txt 
#SBATCH -p GPU2
#SBATCH --qos=normal
#SBATCH -J hsc_job
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:1
#SBATCH --mail-user hesicheng2001@163.com  
#SBATCH --chdir /Share/UserHome/tzhao/2023/sicheng/GraduationDesign/Unsupervised-Classification

PROJECT=SimCLR
RUN_NAME=baseline_finetune
CONFIG_FILE=./custom/proteasome/baseline_finetune.yml

date +%c
hostname
pwd
which python

python simclr.py \
--manually_load_model ./root_dir/baseline/proteasome-12/pretext/model.pth.tar \
--config_exp $CONFIG_FILE \
--project $PROJECT \
--run_name $RUN_NAME \
--root_dir root_dir/$PROJECT/$RUN_NAME \
--wandb_mode offline
echo 跑完了喵