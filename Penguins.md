# Using the Penguins cluster

## Getting started

Apply for an account by emailing support@helpdesk.maths.lancs.ac.uk
 
Documentation at http://ma-info.lancs.ac.uk/

Need a way to log into machine.
Built into Linux and Mac - on Windows should have `PuTTY`. 

Within the university (e.g. on LU eduroam) / when connected via the VPN, login with `ssh username@icefloe-fe.maths.lancs.ac.uk`

## Running jobs

SLURM is a way of allocating appropriate resources to multiple users, avoiding contention or OOM.
Database storing completed and currently running jobs.

For each job, need to request resources. Usually done in a batch file, e.g.

``` bash
#!/bin/bash
#SBATCH --partition PenguinPartition
#SBATCH --nodes=1                 # Number of nodes to use (-N)
#SBATCH --ntasks-per-node=1       # Number of tasks (processes) per node 
#SBATCH --cpus-per-task=1         # Number of CPU cores per task (-c)
#SBATCH --mem=1G                  # Memory per node (e.g., 1GB). Can also specify per CPU with --mem-per-cpu
#SBATCH --time=00:05:00           # Wall clock time limit (e.g., 5 minutes)

hostname
```

Once ssh'd into login node `sbatch <name of batch file>` will submit the job to the queue.

For parallel jobs, we need to use `srun`

``` bash
#!/bin/bash
#SBATCH --partition PenguinPartition
#SBATCH --job-name=my_R_job       # Name of the job (-J)
#SBATCH --nodes=2                 # Number of nodes to use (-N)
#SBATCH --ntasks-per-node=1       # Number of tasks (processes) per node 
#SBATCH --cpus-per-task=1         # Number of CPU cores per task (-c)
#SBATCH --mem=1G                  # Memory per node (e.g., 1GB). Can also specify per CPU with --mem-per-cpu
#SBATCH --time=00:05:00           # Wall clock time limit (e.g., 5 minutes)

srun hostname
```

## Using software

### Python

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

### R

R version 4.1.2 installed. Instructions to install packages locally on ma-info.lancs.ac.uk

## Other slurm commands

Interactive jobs
`srun --pty bash -i`

`sinfo` - show all available queues

`squeue` - show all running and waiting jobs
`squeue -u $(whoami)` - show all my jobs

`sacct` - show in-progress jobs
`scancel <job_id>` - cancel a running / queued job.

