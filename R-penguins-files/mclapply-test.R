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

