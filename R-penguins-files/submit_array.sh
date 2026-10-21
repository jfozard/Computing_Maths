#!/bin/bash
#SBATCH --job-name=r_sim_array
#SBATCH --array=1-10  # Run 10 tasks, TASK_ID will go from 1 to 10
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --partition PenguinPartition
#SBATCH --time=00:10:00 # 10 minutes per task (adjust as needed)
#SBATCH --output=parallel_%A_%a.out # Job_ID, Task_ID
#SBATCH --error=parallel_error_%A_%a.err

Rscript parallel_seeds.R "${SLURM_ARRAY_TASK_ID}"
