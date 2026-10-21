#!/bin/bash
#SBATCH --partition=PenguinPartition       # Partition to run on (-p_
#SBATCH --job-name=R       # Name of the job (-J)
#SBATCH --nodes=1                 # Number of nodes to use (-N)
#SBATCH --ntasks-per-node=1       # Number of tasks (processes) per node 
#SBATCH --cpus-per-task=8         # Number of CPU cores per task (-c)
#SBATCH --mem=8G                  # Memory per node (e.g., 8GB). Can also specify per CPU with --mem-per-cpu
#SBATCH --time=00:05:00           # Wall clock time limit (e.g., 5 minutes)
Rscript mclapply-test.R
