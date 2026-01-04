#!/usr/bin/env Rscript

# DSL PanChen Replication Script
# This script runs the DSL package on the PanChen dataset

library(dsl)

# Load the PanChen dataset
cat("Loading PanChen dataset...\n")
load("PanChen_test/PanChen.rdata")

# Check what objects are loaded
cat("Loaded objects:\n")
print(ls())

# The dataset should be loaded as 'PanChen'
cat("\nDataset structure:\n")
str(PanChen)

cat("\nDataset summary:\n")
summary(PanChen)

cat("\nColumn names:\n")
print(names(PanChen))

# Create the labeled indicator based on NA values in countyWrong
# Labeled observations have non-NA values in countyWrong
PanChen$labeled <- ifelse(is.na(PanChen$countyWrong), 0, 1)

# Create sample_prob column (equal probability for all observations)
# Based on target output: 500 labeled out of 1412 total
n_total <- nrow(PanChen)
n_labeled <- sum(PanChen$labeled)
cat(sprintf("\nTotal observations: %d\n", n_total))
cat(sprintf("Labeled observations: %d\n", n_labeled))
cat(sprintf("Unlabeled observations: %d\n", n_total - n_labeled))

# Equal probability sampling
PanChen$sample_prob <- n_labeled / n_total

cat("\nSample probability:\n")
print(unique(PanChen$sample_prob))

# Run DSL estimation
# Formula: SendOrNot ~ countyWrong + prefecWrong + connect2b + prevalence + regionj + groupIssue
# Predicted variable: countyWrong
# Prediction: pred_countyWrong

cat("\nRunning DSL estimation...\n")

# Run DSL with the appropriate parameters
out <- dsl(
  model = "logit",
  formula = SendOrNot ~ countyWrong + prefecWrong + connect2b + prevalence + regionj + groupIssue,
  predicted_var = "countyWrong",
  prediction = "pred_countyWrong",
  data = PanChen,
  labeled = "labeled",
  sample_prob = "sample_prob",
  sl_method = "glm",
  family = "binomial",
  seed = 1234
)

# Print summary
cat("\n")
summary(out)

# Save results to file
cat("\nSaving results to r_panchen_output.txt...\n")
sink("r_panchen_output.txt")
summary(out)
sink()

cat("\nResults saved successfully!\n")
cat("\nYou can compare this output with the target output in:\n")
cat("PanChen_test/target_r_output_panchen.txt\n")
