# Read in simulation-ID order.
combined <- data.frame()

for (i in 1:10) {
  df <- read.csv(paste0("output_", i, ".csv"))
  combined <- rbind(combined, df)
}

write.csv(combined, "combined_results.csv", row.names = FALSE)
