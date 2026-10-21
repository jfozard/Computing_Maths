## Computing Resources


#### **The Penguins 🐧**
* Small cluster, suitable for prototyping, smaller jobs, and getting familiar with a cluster environment.
* 5 machines with >=72 CPUs, 512GB ram.
* To access - ask Mariusz / Cyrus.
* Documentation - http://ma-info.lancs.ac.uk/ . Grp-Penguins

#### **HEC (High-End Computing) Cluster**
* Lancaster Univesity main computing resource. Typically offers more cores, memory, and shared access via a job scheduler like SLURM.
* Multiple partitions - some 16 core 64GB, some up to 128 core 512GB.
* Also some GPUs - 8 3xV100 32GB / 192GB / 32 core GPU machines. Machines with 3xL40s 48gb and two machines with 2xH200.
* To access - Supervisor applies at ISS HEC page (pretty fast and trivial) 
* Documentation - https://lancaster-hec.readthedocs.io/en/latest/index.html


## Using the Penguins

### Logging in

Log in using `ssh username@icefloe-fe.lancs.ac.uk`

Can run small tasks on login node, but for real work need to submit a job.

### Copying files on/off the Pengins

#### scp

To copy a small number of files
`scp local_file username@icefloe-fe.lancs.ac.uk:remote_file`

To copy a whole directory to your Penguins home directory
`scp -r local_directory username@icefloe-fe.lancs.ac.uk:`

#### rsync

This is a bit more robust for copying many files

Copy a directory from the Penguins to your laptop

`rsync -ahvzP usename@icefloe-fe.lancs.ac.uk:remote_dir/ local_dir`
Trailing slash means copy the contents of this directory


Copy a directory from your laptop to the Penguins

`rsync -ahvzP local_dir/ usename@icefloe-fe.lancs.ac.uk:remote_dir`




### Submitting jobs

#### Single jobs

```bash
#!/bin/bash
#SBATCH --partition=PenguinPartition       # Partition to run on (-p_
#SBATCH --job-name=R       # Name of the job (-J)
#SBATCH --nodes=1                 # Number of nodes to use (-N)
#SBATCH --ntasks-per-node=1       # Number of tasks (processes) per node 
#SBATCH --cpus-per-task=8         # Number of CPU cores per task (-c)
#SBATCH --mem=8G                  # Memory per node (e.g., 8GB). Can also specify per CPU with --mem-per-cpu
#SBATCH --time=00:05:00           # Wall clock time limit (e.g., 5 minutes)
Rscript mclapply-test.R
```

mclapply-test.R
```R
library(parallel)
# Use the CPUs allocated to this Slurm task, not all CPUs on the node.
n_cores <- as.integer(Sys.getenv("SLURM_CPUS_PER_TASK", "1"))
n_simulations <- 10
n_sample_per_sim <- 1000

# Function to be executed in parallel
run_one_simulation <- function(sim_id, base_seed) {
  set.seed(base_seed + sim_id) # Ensure each simulation gets a unique seed
  rnorm(n_sample_per_sim)
}

base_simulation_seed <- 1234

results_list <- mclapply(
 1:n_simulations,
 function(i) { run_one_simulation(i, base_simulation_seed) },
 mc.cores = n_cores
)
results_mclapply <- do.call(rbind, results_list)
head(results_mclapply)
summary(results_mclapply)

write.csv(results_mclapply, "independent_sim_results_mclapply.csv", row.names = FALSE)
```

Submit job with
`sbatch submit_single.sh`


#### Job arrays in R

```bash
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
```

```R
args <- commandArgs(trailingOnly = TRUE)
idx <- as.numeric(args[1])


RNGkind("L'Ecuyer-CMRG")
set.seed(42)

s <- .Random.seed
if (idx > 1) {
  for (i in 2:idx) {
    s <- parallel::nextRNGStream(s)
  }
}
.Random.seed <- s

df <- data.frame(data = rnorm(1000))
write.csv(df, paste0("output_", idx, ".csv"), row.names = FALSE)
```

Submit job with ``sbatch submit_array.sh``

You then need to combine all the results from the output files into one
```R
# Read in simulation-ID order.
combined <- data.frame()

for (i in 1:10) {
  df <- read.csv(paste0("output_", i, ".csv"))
  combined <- rbind(combined, df)
}

write.csv(combined, "combined_results.csv", row.names = FALSE)
```

#### Single job but using the same seeds as the array job

mclapply-test.R
```R
library(parallel)

n_cores <- as.integer(Sys.getenv("SLURM_CPUS_PER_TASK", "1"))
n_simulations <- 10L
n_sample_per_sim <- 1000L

# Match the array job's RNG and initial seed.
RNGkind("L'Ecuyer-CMRG")
set.seed(42)

seeds <- vector("list", n_simulations)
seeds[[1L]] <- .Random.seed

if (n_simulations > 1L) {
  for (i in 2:n_simulations) {
    seeds[[i]] <- nextRNGStream(seeds[[i - 1L]])
  }
}

run_one_simulation <- function(sim_id) {
  assign(".Random.seed", seeds[[sim_id]], envir = .GlobalEnv)
  rnorm(n_sample_per_sim)
}

results_list <- mclapply(
  seq_len(n_simulations),
  run_one_simulation,
  mc.cores = n_cores,
  mc.set.seed = FALSE
)

# One row per simulation, one column per draw.
results_mclapply <- do.call(rbind, results_list)

write.csv(
  results_mclapply,
  "independent_sim_results_mclapply.csv",
  row.names = FALSE
)
```
