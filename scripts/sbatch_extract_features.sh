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

which python
./scripts/extract_features.sh
echo 跑完了喵