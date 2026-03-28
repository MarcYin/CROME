#!/bin/bash 
#SBATCH --job-name=uk_s2
#SBATCH --requeue
#SBATCH --array=1-35%8
#SBATCH -p standard
#SBATCH --account=nceo_generic
#SBATCH --qos=short
#SBATCH --cpus-per-task=1
#SBATCH -o /work/scratch-pw3/marc/errq/uk_s2%A_%a.out
#SBATCH -e /work/scratch-pw3/marc/errq/uk_s2%A_%a.err
#SBATCH -t 3:59:59
#SBATCH --mem=63000

source activate xee_beam
cd /home/users/marcyin/UK_crop_map
python -u get_monthly_composite.py $SLURM_ARRAY_TASK_ID

