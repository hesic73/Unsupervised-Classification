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

PROJECT=SimCLR_proteasome12
RUN_NAME=new_baseline
CONFIG_FILE=custom/proteasome/baseline.yml

date +%c
hostname
pwd
which python

echo $PROJECT/$RUN_NAME

python simclr.py \
--config_exp $CONFIG_FILE \
--project $PROJECT \
--run_name $RUN_NAME \
--root_dir root_dir/$PROJECT/$RUN_NAME \
--wandb_mode offline 
# --manually_load_model ./root_dir/baseline/proteasome-12/pretext/model.pth.tar \
echo 跑完了喵