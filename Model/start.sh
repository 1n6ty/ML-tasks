#!bin/bash

#SBATCH --job-name={task}
#SBATCH --error={task}-%j.err
#SBATCH --output={task}-%j.log
#SBATCH --time={17-00:00:00}
#SBATCH --ntasks={16}
#SBATCH --nodes={1}
#SBATCH --cpus-per-task={2}

srun python3 train_model.py &

wait