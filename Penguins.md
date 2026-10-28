# Using the Penguins cluster

## Getting started

Apply for an account by emailing support@helpdesk.maths.lancs.ac.uk
 
Documentation at http://ma-info.lancs.ac.uk/

Need a way to log into machine. Built into Linux and Mac - on Windows should have `PuTTY`. 

Within the university (e.g. on LU eduroam) / when connected via the VPN, login with `ssh username@icefloe-fe.maths.lancs.ac.uk`

Here username is the short username (sometimes `lastname(firstinitial)` used to e.g. log into the portal).

## Using software

### Python

#### Managing Python Versions through pyenv

Software on the penguins isn't actively managed.

First install `pyenv` in your own home directory - `curl -fsSL https://pyenv.run | bash`

Add the text provided to your `.bashrc` file. Then `source ~/.bashrc`

Then install a reasonably recent Python version:
`pyenv install 3.14`.  
Choose that version
`pyenv global 3.14`
and test which version is being used
`python --version`

Create a venv `python -m venv my_venv` - this is a virtual environment into which you can install packages. Make a different venv for each project.

To use the venv - `source ./my_venv/bin/activate`. 
You can then install packages with `pip` - e.g. `pip install numpy scipy`


#### Managing Python through uv

Installing uv on Penguins
`curl -LsSf https://astral.sh/uv/install.sh | sh`

You can install python versions using `uv python install` or `uv python install 3.14`

### R

R version 4.3.3 installed. Instructions to install packages locally on `ma-info.lancs.ac.uk`

Install packages into a local directory. At the command line:
```bash
mkdir -p ~/R/library
```
Then in R:
```R
install.packages("evd", lib="~/R/library")
.libPaths(c("~/R/library", .libPaths()))
library(evd)
```


Can manage R versions using tools such as `Rig` `https://rig.r-lib.org/` - see `https://rig.r-lib.org/tutorial-user.html`

Manage R environments using `renv` `https://rstudio.github.io/renv/articles/renv.html`


## Running jobs

### SLURM batch files

SLURM is a way of allocating appropriate resources to multiple users, avoiding contention or OOM.

For each job, need to request resources. Usually done in a batch file, e.g.

``` bash
#!/bin/bash
#SBATCH --partition PenguinPartition
#SBATCH --nodes=1                 # Number of nodes to use (-N)
#SBATCH --ntasks-per-node=1       # Number of tasks (processes) per node 
#SBATCH --cpus-per-task=1         # Number of CPU cores per task (-c)
#SBATCH --mem=4G                  # Memory per node (e.g., 14B). Can also specify per CPU with --mem-per-cpu
#SBATCH --time=00:05:00           # Wall clock time limit (e.g., 5 minutes)

hostname
```

Once ssh'd into login node `sbatch <name of batch file>` will submit the job to the queue.

To use more cpu cores for your task, set `#SBATCH --cpus-per-task=10`. 

For parallel jobs across multiple nodes, we need to set `#SBATCH --nodes=2` and/or `#SBATCH --ntasks-per-node=2` (or to some larger number) and use `srun` to launch commands - but this requires the command to be aware that it is being run in parallel. Note that if you use  `#SBATCH --nodes=2` or  `#SBATCH --ntasks-per-node=2`without `srun`,
you allocate all the nodes/tasks but only run one task!

### Job arrays

Job arrays are more suitable if you want to run the same task multiple times with slightly different options (e.g. a rng seed)

```bash
#!/bin/bash
#SBATCH --partition=PenguinPartition
#SBATCH --job-name=simulation
#SBATCH --array=1-100%10    # 100 jobs in the array - at most 10 at once.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G    # This is memory per task in the array
#SBATCH --time=00:05:00
#SBATCH --output=simulation-%A_%a.out

echo "$SLURM_ARRAY_TASK_ID"
$HOME/my_venv/bin/python simulation.py "$SLURM_ARRAY_TASK_ID"
```

The script is run 100 times, each time with a different value for `$SLURM_ARRAY_TASK_ID`

To use this in a script, parse the command line variable in the script - e.g.
```R
id <- as.integer(commandArgs(trailingOnly = TRUE)[1])

set.seed(id)
result <- mean(rnorm(1000))  # Replace with your simulation.

saveRDS(result, sprintf("result-%03d.rds", id))
```
or
```Python
import sys
import numpy as np
import numpy.random as npr
from pathlib import Path

task_id = int(sys.argv[1])
base_seed = 1234

seed = npr.SeedSequence([task_id, base_seed])
rng = npr.default_rng(seed)

samples = rng.normal(size=1_000)

output = Path("results")
output.mkdir(exist_ok=True)

np.savez(
    output / f"simulation-{task_id:03d}.npz",
    mean=samples.mean(),
    task_id=task_id,
    base_seed=base_seed,
)
```

### Running parallel computations

```bash
#!/bin/bash
#SBATCH --partition=PenguinPartition
#SBATCH --job-name=simulation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=4G # Memory per node - shared by all cpus!
#SBATCH --time=00:05:00

echo "Your commands here"
```

#### Python

Some libraries will take advantage of multiple cores - e.g. `numpy` will for some matrix operations. `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` and `OMP_NUM_THREADS` are the standard environment variable that controls how many threads this will use. However much of the code will still only run in serial. Jax will take advantage of all cores, but this depends a bit on how it is used.

To explicitly exploit multiple cores usually requires writing multithreaded/multiprocess code.

#### Python

```Python
import os
from multiprocessing import Pool

import numpy as np


def simulate(simulation_id):
    rng = np.random.default_rng([simulation_id, 12345])
    return rng.normal(size=100_000).mean()


if __name__ == "__main__":
    nworkers = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))

    with Pool(processes=nworkers) as pool:
        results = pool.map(simulate, range(20))

    np.save("results.npy", results)
```
```bash
#!/bin/bash
#SBATCH --partition=PenguinPartition
#SBATCH --job-name=simulation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=4G
#SBATCH --time=00:05:00

# Libraries like numpy try to spawn lots of threads - limit this.
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

cd "$HOME/my-project"
$HOME/my_venv/bin/python simulation.py
```

#### R

There are a number of different options in R - the `parallel` library and `mclapply` is one good option
```R
# parallel.R
library(parallel)

nworkers <- as.integer(Sys.getenv("SLURM_CPUS_PER_TASK", "1"))

# Generate reproducible RNG streams for each simulation.
RNGkind("L'Ecuyer-CMRG")
set.seed(12345)

simulate <- function(i) {
  mean(rnorm(100000))
}

results <- mclapply(
  1:1000,
  simulate,
  mc.cores = nworkers
)

saveRDS(results, "results-r.rds")
```
```bash
#!/bin/bash
#SBATCH --partition=PenguinPartition
#SBATCH --job-name=simulation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=4G
#SBATCH --time=00:05:00

cd "$HOME/my-project"
Rscript parallel.py
```
These simulations will be reproducible *provided* the same number of workers is used. It's possible to manually set the seed for each thread.


## Other slurm commands

Interactive jobs
`srun --pty bash -i`

`sinfo` - show all available queues

`squeue` - show all running and waiting jobs
`squeue --me` - show all my jobs
`scancel <job_id>` - cancel a running / queued job.

