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

cd /Share/UserHome/tzhao/2023/sicheng/GraduationDesign/Unsupervised-Classification
echo 开始跑了喵
python simclr.py --config_env custom/configs/env.yml \
--config_exp custom/configs/exp_env.yml
echo 跑完了喵