
# "Penguins, HEC and other compute resources"
---

# First steps

Getting results quicker is great - but only if you're solving the right problem, know that the method is correct, and ideally can reproduce your simulations. Before reaching for more CPU time - which costs money and has an environmental impact - you need to ensure that your results are trustworthy and meaningful.


## 1  Getting the Correct Result

### 1.1 Integration & Regression Tests

Run the entire code, and verify the overall behavior of your program.

* **Analytical benchmarks** - e.g. closed‑form OLS coefficients
  $\hat\beta=(X^{\top}X)^{-1}X^{\top}y.$
* **Approximate solutions** - behaviour known in some limit of small/large parameter values. 
* **Reference software** - compare to *scipy*, *Stan*, other R packages, etc.
* **Convergence studies** - verify error proportional step‑size\${}^p\$ when \$p\$ is known.

### 1.2 Unit Tests

If it's hard to test the overall behaviour of the code - e.g. the problem is too complex, ensure that the code
is split up into functions - ideally pure, with have no side-effects - and write tests.
srun --pty bash -i
Different languages have different testing frameworks

| Language | Framework            |
| -------- | -------------------- |
| R        | `testthat`           |
| Python   | `pytest`, `unittest` |

#### Example in R (How to test a single function?)



### 1.3 Tracking Code and Parameters

If you're going to spend large amounts of compute on a problem, you only want to do it a small number of times. (Cost / environment.) 
Plus, you may need to re-run code (for a later paper submission).
Hence it is important to track exactly what code was run, and which options were used.

#### Version control

- Use *git* - I can run a training course if people are interested.
- Preferably commit the code before every long run, consider using tags for special runs (`git tag -a v1.0 -m "Version 1.0 for paper submission"`)

#### Tracking parameters

- Rather than supply on the command-line, read script parameters from a configuration file (.csv or .yaml), or just include a file defining the parameters.
- If you're using slurm submission scripts, commit the script to version control.

This all requires quite a bit of discipline.

For ML - consider using experiment tracking services (wandb, mlflow).

#### Keep track of environments (package versions)

- In Python, tools such as uv or poetry.
- In R, at least keep a list of package versions.

### 2 Avoid very slow sequential code

Optimising code is usually quite a poor return on your time. 

### 2.1 Language pitfalls

- Both *R* and *Python
  - Use vectorized operations, e.g. `numpy` or vectorized R operations
  - Exploit compiled back-ends (`numpy`, `pandas`, `data.table`)
  - Avoid for-loops, especially nested for-loops.

- *Python*
  - Avoid for-loops as much as possible, also heavy use of Python lists

- *R*
  - Avoid allocation in for-loops; pre-allocate result.

### 2.2 Algorithmic pitfalls

- Sparse > dense when large proportion (>95%) of entries are zeros (`Matrix`, `scipy.sparse`).
- Choose stiff vs. non‑stiff ODE solvers appropriately.
- Use hash maps (Python `dict`, R environments / `hashtab`) for \$\mathcal O(1)\$ lookup instead of repeated linear searches.
- Consider the running time of the overall algorithm.

## Parallel code

### 3.1 Levels of parallelization

Parallelism can occur at various scales:

1.  **CPU Pipelining/Superscalar/Instruction-Level Parallelism**: Modern CPUs execute multiple instructions from a single computation stream simultaneously (and often out of order). 
2.  **Vectorized Operations (SIMD)**: Single Instruction, Multiple Data. Operations like `numpy` array additions or R vector operations apply one instruction to multiple data elements at once (e.g., using AVX extensions on CPUs, or on GPUs with CUDA).
3.  **Multithreading**: Multiple threads run on different cores of a single CPU, sharing the same memory space. Examples include OpenMP, and R's `mclapply` on Unix-like systems. Hard in Python at present because of GIL. 
4.  **Multiprocessing**: Multiple processes, each with its own private memory space (and R/Python interpreter), run concurrently on the same machine. Usually processes communicate by pipes, sockets or files. These can however, share common memory buffers, e.g. Python `multiprocessing` with shared memory.
5.  **Distributed Computing**: Multiple processes, on distinct physical machines. Data needs to be exchanged over the network (e.g. MPI) or through shared files. Some HPCs have better interconnections than others. 

### Embarassingly parallel problems

Many scientific and statistical problems are **embarrassingly parallel** (also known as "pleasantly parallel"). 
Tasks can be broken down into many independent sub-tasks that require almost no communication or dependency between them.

* Running the same simulation with different random seeds.
* Bootstrapping or permutation tests.
* Cross-validation folds.
* Parameter sweeps (evaluating a function over a grid of parameters).
* Processing independent chunks of data.

As these don't require communication between tasks, they're easy wins for parallelization.

### Amdahl's Law

If parts of the task are inherently serial, you can't speed these up by working in parallel.
Use profilers such as `profvis` (R) or `cProfile`/`austin`/`scalene` (Python) to view which parts of code are actually slow.

## 4 Computing Resources


#### **The Penguins 🐧**
* Small cluster, suitable for prototyping, smaller jobs, and getting familiar with a cluster environment.
* 5 machines with >=72 CPUs, 512GB ram.
* To access - ask Mariusz / Cyrus.
* Documentation - http://ma-info.lancs.ac.uk/ . Grp-Penguins```python
from multiprocessing import Pool
import os

def worker(x):
    return x * x

if __name__ == '__main__':
    print(len(os.sched_getaffinity(0)))
    with Pool(8) as p:
        results = p.map(worker, range(100))
    print(results)

#### **HEC (High-End Computing) Cluster**
* Lancaster Univesity main computing resource. Typically offers more cores, memory, and shared access via a job scheduler like SLURM.
* Multiple partitions - some 16 core 64GB, some up to 64 core 256GB. Also 8 3xV100 32GB / 192GB / 32 core GPU machines.
* To access - Supervisor applies at ISS HEC page (pretty fast and trivial) 
* Documentation - https://lancaster-hec.readthedocs.io/en/latest/index.html

#### **Bede (Tier 2 HPC)**
* Larger cluster primarily for GPU-accelerated workloads. (Many V100s and a few H100s.)
* To access - talk to me. (Requires supervisor approval; takes a while to get access.)
* Documentation - https://bede-documentation.readthedocs.io/

All these machines run Linux (or a Unix variant) and use SLURM for job submission and management.

See also HeX

## 5 Using Clusters

### 5.1 How clusters work

Typically login via ssh to the login (or head or job submission) nodes.
Usually no direct access to compute nodes.
(HEC: ssh username@wayland-2022.hec.lancs.ac.uk Penguins: ssh  username@icefloe-fe.lancs.ac.uk)

Login nodes are shared between all users - overuse (OOM) can make the cluster unusable.
Typically only for installing software, compiling code (this can be tricky), and submitting jobs.
Sometimes only login nodes have access to external internet.

Shared filesystems for storing programs and data. Also  per-node fast local storage available.

### 5.2 SLURM Submission scripts

Some servers (elsewhere) are free-for-alls - ssh in, run job, play nicely.

SLURM is a way of allocating appropriate resources to multiple users, avoiding contention or OOM.
Database storing completed and currently running jobs.

For each job, need to request resources. Usually done in a batch file, e.g.


HEC example

```bash
#!/bin/bash
#SBATCH --partition=parallel        # Partition to run on (-p_
#SBATCH --job-name=my_R_job       # Name of the job (-J)
#SBATCH --nodes=1                 # Number of nodes to use (-N)
#SBATCH --ntasks-per-node=1       # Number of tasks (processes) per node 
#SBATCH --cpus-per-task=1         # Number of CPU cores per task (-c)
#SBATCH --mem=1G                  # Memory per node (e.g., 1GB). Can also specify per CPU with --mem-per-cpu
#SBATCH --time=00:05:00           # Wall clock time limit (e.g., 5 minutes)
source /etc/profile

hostname
```


Consists of a list of SLURM directive (command line options to sbatch/srun), followed by the commands to run.
Most of these SLURM directives have short and long forms: `-p -j -n` etc.


On HEC no need to specify memory if <500M (but this is commonly enforced on other HPCs)
Also default times for each partition.

If we want to spawn multiple processes need to use srun

```bash
#!/bin/bash
#SBATCH --partition=parallel        # Partition to run on (-p_
#SBATCH --job-name=my_R_job       # Name of the job (-J)
#SBATCH --nodes=2                 # Number of nodes to use (-N)
#SBATCH --ntasks-per-node=1       # Number of tasks (processes) per node 
#SBATCH --cpus-per-task=1         # Number of CPU cores per task (-c)
#SBATCH --mem=1G                  # Memory per node (e.g., 1GB). Can also specify per CPU with --mem-per-cpu
#SBATCH --time=00:05:00           # Wall clock time limit (e.g., 5 minutes)
source /etc/profile

hostname
```
Will only output a single line

```bash
#!/bin/bash
#SBATCH --partition=parallel        # Partition to run on (-p_
#SBATCH --job-name=my_R_job       # Name of the job (-J)
#SBATCH --nodes=2                 # Number of nodes to use (-N)
#SBATCH --ntasks-per-node=1       # Number of tasks (processes) per node 
#SBATCH --cpus-per-task=1         # Number of CPU cores per task (-c)
#SBATCH --mem=1G                  # Memory per node (e.g., 1GB). Can also specify per CPU with --mem-per-cpu
#SBATCH --time=00:05:00           # Wall clock time limit (e.g., 5 minutes)
source /etc/profile

srun hostname
```
runs multiple jobs.


Can do all sorts of extra things
```bash
#SBATCH --mail-type=END,FAIL 
#SBATCH --mail-user=YOUR_EMAIL_ADDRESS_HERE
#SBATCH --output=logs/my_job_%j.out # Standard output file (%j expands to jobID)
#SBATCH --error=logs/my_job_%j.err  # Standard error file
```

Penguin example
```
#!/bin/bash
#SBATCH --partition Penguin
```
plus all the other usual stuff




### 5.3 Using software

#### HEC

See list with `module avail` e.g. `module add R/4.3.1`
Will probably need to add libraries
For Python - `module add miniforge/20240923` and create conda environment, or `module add opence` for outdated versions of ML stack.

#### Penguins 

For Python - `pyenv install 3.11`. Create venv `python -m venv env; source ./env/bin/activate`. Install packages with `pip`
Alternatively manage packages with `poetry` or `uv`

Installing poetry on Penguins

Install pyenv - instructions at https://github.com/pyenv/pyenv
Set global python to non-system version

Install pipx
```
python -m install --user pipx
```
Add install directory to path
```
export PATH=~/.local/bin:$PATH
```
Install poetry
```
pipx install poetry
```

Poetry
```
poetry init
```
Change pyproject.toml (optional)
```
[tool.poetry]
package-mode = false
```


```
poetry install --no-root
eval $(poetry env activate)
```


Installing uv on Penguins


R version 4.1.2 installed. Instructions to install packages locally on ma-info.lancs.ac.uk

### 5.4 Other slurm commands

Interactive jobs
`srun --pty bash -i`

Interactive job requesting GPU on HEC

`sinfo` - show all available queues

`squeue` - show all running and waiting jobs
`squeue -u $(whoami)` - show all my jobs

`sacct` - show in-progress jobs

`scancel <job_id>` - cancel a running / queued job.


### 5.5 Storage on the HEC

| File Area | Quota | Backup Policy | File Retention policy | Location | Environment var |
|-----------|-------|---------------|-----------------------|----------|-----------------|
| home | 10G | Nightly for 90 days | For the lifetime of the account | global | $HOME |
| storage | 200G | None | For the lifetime of the account | global | $global_storage |
| scratch | 10T | None | Files automatically deleted after 4 weeks | global | $global_scratch |
| temp | Unlimited | None | Files automatically deleted at the end of the job | local | $TMPDIR |

Check quotas using `gpfsquota`. All global storage is a little slow because of network.
Try not to make many small files.

### 5.6 SLURM array jobs

Running too many jobs breaks the SLURM scheduler.

sbatch script run_sim_array.sh
```bash
#!/bin/bash
#SBATCH --job-name=r_sim_array
#SBATCH --array=1-200  # Run 200 tasks, TASK_ID will go from 1 to 200
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:30:00 # 30 minutes per task (adjust as needed)
#SBATCH --output=slurm_logs/sim_output_%A_%a.out # Job_ID, Task_ID
#SBATCH --error=slurm_logs/sim_error_%A_%a.err

TASK_ID=${SLURM_ARRAY_TASK_ID}

python task.py
```
Note that there are a number of other environment variables in a slurm script - 
https://docs.hpc.shef.ac.uk/en/latest/referenceinfo/scheduler/SLURM/SLURM-environment-variables.html#gsc.tab=0


## 6. Parallel examples

### 6.1 Multiprocessing in Python

For, e.g. multiprocessing on a single node, can use the form

HEC
```bash
#!/bin/bash
#SBATCH --partition=parallel       # Partition to run on (-p_
#SBATCH --job-name=python       # Name of the job (-J)
#SBATCH --nodes=1                 # Number of nodes to use (-N)
#SBATCH --ntasks-per-node=1       # Number of tasks (processes) per node 
#SBATCH --cpus-per-task=8         # Number of CPU cores per task (-c)
#SBATCH --mem=1G                  # Memory per node (e.g., 1GB). Can also specify per CPU with --mem-per-cpu
#SBATCH --time=00:05:00           # Wall clock time limit (e.g., 5 minutes)
source /etc/profile
module activate miniforge

python multiprocessing_test.py
```

Penguins
```bash
#!/bin/bash
#SBATCH --partition=PenguinPartition       # Partition to run on (-p_
#SBATCH --job-name=python       # Name of the job (-J)
#SBATCH --nodes=1                 # Number of nodes to use (-N)
#SBATCH --ntasks-per-node=1       # Number of tasks (processes) per node 
#SBATCH --cpus-per-task=8         # Number of CPU cores per task (-c)
#SBATCH --mem=1G                  # Memory per node (e.g., 1GB). Can also specify per CPU with --mem-per-cpu
#SBATCH --time=00:05:00           # Wall clock time limit (e.g., 5 minutes)
source start-pyenv

python multiprocessing_test.py
```

Where multiprocessing_test.py is
```python
from multiprocessing import Pool
import os

def worker(x):
    return x * x

if __name__ == '__main__':
    print(len(os.sched_getaffinity(0)))
    with Pool(8) as p:
        results = p.map(worker, range(100))
    print(results)
```

### 6.2 Job arrays in Python

HEC
```bash
#!/bin/bash
#SBATCH --partition=parallel
#SBATCH --job-name=python_array
#SBATCH --array=1-200  # Run 200 tasks, TASK_ID will go from 1 to 200
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00 # 30 minutes per task (adjust as needed)
#SBATCH --output=array_output_%A_%a.out # Job_ID, Task_ID
#SBATCH --error=array_error_%A_%a.err

# Load python module
module activate miniforge

# Get the SLURM array task ID
TASK_ID=${SLURM_ARRAY_TASK_ID}

# Call the R script, passing the task ID as an argument
# The R script will use this ID to set its seed and name its output file
python array_test.py ${TASK_ID} ${TASK_ID}.dat
```

Penguins
```bash
#!/bin/bash
#SBATCH --partition=PenguinPartition
#SBATCH --job-name=python_array
#SBATCH --array=1-200  # Run 200 tasks, TASK_ID will go from 1 to 200
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00 # 30 minutes per task (adjust as needed)
#SBATCH --output=array_output_%A_%a.out # Job_ID, Task_ID
#SBATCH --error=array_error_%A_%a.err

# Init Pyenv
source ${HOME}/start-pyenv

# Get the SLURM array task ID
TASK_ID=${SLURM_ARRAY_TASK_ID}

# Call the R script, passing the task ID as an argument
# The R script will use this ID to set its seed and name its output file
eval $(poetry env activate)
python array_test.py ${TASK_ID} ${TASK_ID}.dat
```

array_test.py
```Python
import os
import sys

def worker(x):
    return x*x

if __name__=="__main__":
    idx = int(sys.argv[1])
    output_file = sys.argv[2]
    results = worker(idx)
    with open(output_file, 'w') as f:
        print(results, file=f)

```

### 6.3 mclapply in R

Penguins

```bash
#!/bin/bash
#SBATCH --partition=PenguinPartition       # Partition to run on (-p_
#SBATCH --job-name=R       # Name of the job (-J)
#SBATCH --nodes=1                 # Number of nodes to use (-N)
#SBATCH --ntasks-per-node=1       # Number of tasks (processes) per node 
#SBATCH --cpus-per-task=8         # Number of CPU cores per task (-c)
#SBATCH --mem=1G                  # Memory per node (e.g., 1GB). Can also specify per CPU with --mem-per-cpu
#SBATCH --time=00:05:00           # Wall clock time limit (e.g., 5 minutes)

RScript mclapply-test.R
```


mclapply-test.R
```R
library(parallel)
# Don't use parallel::detectCores(); either hard-code or use parallely::availableCores() and parallely::availableWorkers()
n_cores <- 8
n_simulations <- 200
n_sample_per_sim <- 1000

# Function to be executed in parallel
run_one_simulation <- function(sim_id, base_seed) {
  set.seed(base_seed + sim_id) # Ensure each simulation gets a unique seed
  true_mu <- rnorm(1, mean = 0, sd = 1)
  sim_sample <- rnorm(n_sample_per_sim, mean = true_mu, sd = 1)
  # Return a named vector or list for easier combination
  c(sim_id = sim_id, estimated_mu = mean(sim_sample), sample_sd = sd(sim_sample), true_mu = true_mu)
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

foreach-test.R
```R
library(foreach)
library(doParallel)
library(parallel) # For detectCores

# Simulation parameters
n_simulations <- 200
n_sample_per_sim <- 1000

# Function to be executed in parallel
run_one_simulation <- function(sim_id, base_seed) {
  set.seed(base_seed + sim_id) # Ensure each simulation gets a unique seed
  true_mu <- rnorm(1, mean = 0, sd = 1)
  sim_sample <- rnorm(n_sample_per_sim, mean = true_mu, sd = 1)
  # Return a named vector or list for easier combination
  c(sim_id = sim_id, estimated_mu = mean(sim_sample), sample_sd = sd(sim_sample), true_mu = true_mu)
}

# Setup parallel backend
n_cores <- 8
cl <- makeCluster(n_cores)
registerDoParallel(cl) # Register the cluster for foreach

base_simulation_seed <- 1000 # A base seed for reproducibility

results_foreach <- foreach(
  i = 1:n_simulations,
  .combine = rbind  # Combine results into a data frame
) %dopar% {
  run_one_simulation(sim_id = i, base_seed = base_simulation_seed)
}

# Stop the cluster
stopCluster(cl)

# Convert to data frame for easier analysis
results_df_foreach <- as.data.frame(results_foreach)
head(results_df_foreach)
summary(results_df_foreach)
write.csv(results_df_foreach, "independent_sim_results_foreach.csv", row.names = FALSE)

```


### 6.4 Job arrays in R

Penguins
```bash
#!/bin/bash
#SBATCH --partition=PenguinPartition
#SBATCH --job-name=r_sim_array
#SBATCH --array=1-200  # Run 200 tasks, TASK_ID will go from 1 to 200
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:30:00 # 30 minutes per task (adjust as needed)
#SBATCH --output=slurm_logs/sim_output_%A_%a.out # Job_ID, Task_ID
#SBATCH --error=slurm_logs/sim_error_%A_%a.err


# Create log and results directories if they don't exist
mkdir -p slurm_logs
mkdir -p slurm_results

# Get the SLURM array task ID
TASK_ID=${SLURM_ARRAY_TASK_ID}

# Call the R script, passing the task ID as an argument
# The R script will use this ID to set its seed and name its output file
Rscript single_simulation.R ${TASK_ID}
```

single_simulation.R
```R
args <- commandArgs(trailingOnly = TRUE)
task_id <- as.integer(args[1])
n_sample_per_sim <- 1000
base_seed <- 2000 # A different base seed for this set of simulations

set.seed(base_seed + task_id) # Ensure each task has a unique seed!

true_mu <- rnorm(1, mean = 0, sd = 1)
sim_sample <- rnorm(n_sample_per_sim, mean = true_mu, sd = 1)

result_data <- data.frame(
  task_id = task_id,
  true_mu = true_mu,
  estimated_mu = mean(sim_sample),
  sample_sd = sd(sim_sample)
)

output_dir <- "slurm_results"
dir.create(output_dir, showWarnings = FALSE) # Create dir if it doesn't exist

output_file_csv <- file.path(output_dir, paste0("result_task_", task_id, ".csv"))
write.csv(result_data, file = output_file_csv, row.names = FALSE)

cat("Task", task_id, "completed. Result saved to", output_file_csv, "\n")
```


combine_results.R
```R
results_dir <- "slurm_results"
csv_files <- list.files(results_dir, pattern = "\\.csv$", full.names = TRUE)
data_list <- lapply(
  csv_files,
  read.csv,
)
combined_results <- do.call(rbind, data_list)
head(combined_results)
summary(combined_results)
write.csv(combined_results, file = "combined_results.csv", row.names = FALSE)
```


## 7. Other useful things

### tmux

Keeping sessions alive using tmux / screen
(Otherwise terminated if ssh connection drops)

Run `tmux`
To detatch session `C-b d`
To reattach `tmux ls`, `tmux attach -t 0`

### vscode ssh

Install vscode Remote-ssh extension



















