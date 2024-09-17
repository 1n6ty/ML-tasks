#!/bin/bash

#SBATCH --job-name=spine_segmentation
#SBATCH --error=spine_segmentation-%j.err
#SBATCH --output=spine_segmentation-%j.log
#SBATCH --time=14-00:00:00
#SBATCH --ntasks=2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=250000

srun python3 --cpus-per-task=32 --ntasks=1 train_model.py --mode side > training_side.log &
srun python3 --cpus-per-task=32 --ntasks=1 train_model.py --mode frontal > training_frontal.log &

wait