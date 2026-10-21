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
