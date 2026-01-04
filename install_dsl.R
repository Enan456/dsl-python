#!/usr/bin/env Rscript

library(devtools)

# Install DSL package from GitHub
cat("Installing DSL package from naoki-egami/dsl...\n")
devtools::install_github("naoki-egami/dsl", force = TRUE)

# Verify installation
cat("\nVerifying DSL package installation...\n")
library(dsl)
cat("DSL package version:", as.character(packageVersion("dsl")), "\n")
cat("DSL package loaded successfully!\n")
