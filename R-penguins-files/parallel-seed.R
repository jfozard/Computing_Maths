args <- commandArgs(trailingOnly = TRUE)
idx <- as.numeric(args[1])

n_samples <- 1000

RNGkind("L'Ecuyer-CMRG")
set.seed(42)

s <- .Random.seed
if (idx > 1) {
  for (i in 2:idx) {
    s <- parallel::nextRNGStream(s)
  }
}
.Random.seed <- s

df <- data.frame(data = rnorm(n_samples))

filename <-  paste0("output_", idx, ".csv")
write.csv(df, filename, row.names = FALSE)
print(paste0("Simulated ", n_samples, " and written to ", filename))
