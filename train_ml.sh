#!/bin/bash 
#SBATCH --job-name=xgboost
#SBATCH --requeue
#SBATCH --array=1-1
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpuhost[015,016]
#SBATCH --cpus-per-task=8
#SBATCH -o /work/scratch-pw3/marc/errq/xgboost%A_%a.out
#SBATCH -e /work/scratch-pw3/marc/errq/xgboost%A_%a.err
#SBATCH -t 23:59:59
#SBATCH --mem=256000

mamba activate xgboost
cd /home/users/marcyin/UK_crop_map/
# ~/mambaforge/envs/xgboost/bin/python -u train_ml.py

~/mambaforge/envs/xgboost/bin/python -u train_curf.py
