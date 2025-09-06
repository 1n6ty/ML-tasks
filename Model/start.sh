#!/bin/bash

#SBATCH --job-name=spine_segmentation
#SBATCH --error=spine_segmentation-%j.err
#SBATCH --output=spine_segmentation-%j.log
#SBATCH --time=14-00:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=250000

srun --cpus-per-task=32 --ntasks=1 python3 train_model.py train_conf.toml > training_side.log &

wait