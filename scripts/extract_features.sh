#!/bin/bash

#SBATCH -o outputs/job%j.txt
#SBATCH --error errors/job%j.txt 
#SBATCH -p GPU1
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH -J extract_features
#SBATCH --nodes=1 
#SBATCH --mail-user hesicheng2001@163.com  
#SBATCH --chdir /Share/UserHome/tzhao/2023/sicheng/GraduationDesign/Unsupervised-Classification

conda activate scan
python extract_features.py