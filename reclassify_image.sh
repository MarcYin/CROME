#!/bin/bash 
#SBATCH --job-name=classify
#SBATCH --requeue
#SBATCH --array=1-35
#SBATCH -p short-serial
#SBATCH --cpus-per-task=1
#SBATCH -o /work/scratch-pw3/marc/errq/classify%A_%a.out
#SBATCH -e /work/scratch-pw3/marc/errq/classify%A_%a.err
#SBATCH -t 23:59:59
#SBATCH --mem=63000

source activate xgboost
cd /home/users/marcyin/UK_crop_map
python -u reclassify_image.py $SLURM_ARRAY_TASK_ID

