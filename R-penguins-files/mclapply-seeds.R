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

print(paste0("Simulated ", n_simulations, " with ", n_sample_per_sim))
