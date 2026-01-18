#!/usr/bin/env Rscript

# Check if devtools is installed
if (!requireNamespace("devtools", quietly = TRUE)) {
  cat("Installing devtools...\n")
  install.packages("devtools", repos = "https://cloud.r-project.org")
} else {
  cat("devtools already installed\n")
}

# Load and verify
library(devtools)
cat("devtools version:", as.character(packageVersion("devtools")), "\n")
