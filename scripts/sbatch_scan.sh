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
#SBATCH --nodelist=gpu002
#SBATCH --chdir /Share/UserHome/tzhao/2023/sicheng/GraduationDesign/Unsupervised-Classification

PROJECT=SimCLR_CNG
RUN_NAME=baseline
CONFIG_FILE=custom/scan/cng.yml

date +%c
hostname
pwd
which python

echo Project/name: $PROJECT/$RUN_NAME

python scan.py \
--config_exp $CONFIG_FILE \
--root_dir root_dir/$PROJECT/$RUN_NAME 


echo 跑完了喵

date +%c
