# run_toy_pipeline.R
# Toy data test for end-to-end verification of quantile-GuidedPLS pipeline

set.seed(42)

source("src/functions_pipeline.R")

cat("============================================\n")
cat("  Toy Pipeline: Quantile-GuidedPLS\n")
cat("============================================\n\n")

# --- Generate toy data ---
n_source <- 200
n_target <- 150
p_source <- 10
p_target <- 8
n_regions <- 3
n_latent <- 2

cat("--- Generating toy data ---\n")
cat(sprintf("  Source: %d x %d, Target: %d x %d, Regions: %d\n",
            n_source, p_source, n_target, p_target, n_regions))

# Shared latent variables
Z_source <- matrix(rnorm(n_source * n_latent), n_source, n_latent)
Z_target <- matrix(rnorm(n_target * n_latent), n_target, n_latent)

# Loadings
W_source <- matrix(runif(n_latent * p_source, -1, 1), n_latent, p_source)
W_target <- matrix(runif(n_latent * p_target, -1, 1), n_latent, p_target)

# Expression with noise
X <- Z_source %*% W_source + matrix(rnorm(n_source * p_source, sd = 0.3),
                                     n_source, p_source)
Y <- Z_target %*% W_target + matrix(rnorm(n_target * p_target, sd = 0.3),
                                     n_target, p_target)

# Apply monotone transforms (simulate modality-specific nonlinearity)
X <- abs(X)^1.5  # power transform
Y <- exp(Y * 0.5)  # exponential transform

colnames(X) <- paste0("lipid_", seq_len(p_source))
colnames(Y) <- paste0("gene_", seq_len(p_target))

cat(sprintf("  X range: [%.2f, %.2f]\n", min(X), max(X)))
cat(sprintf("  Y range: [%.2f, %.2f]\n", min(Y), max(Y)))

# Anatomy: assign regions based on latent dim 1
breaks_s <- quantile(Z_source[, 1], probs = c(0, 1/3, 2/3, 1))
breaks_t <- quantile(Z_target[, 1], probs = c(0, 1/3, 2/3, 1))
region_source <- cut(Z_source[, 1], breaks = breaks_s, labels = FALSE,
                     include.lowest = TRUE)
region_target <- cut(Z_target[, 1], breaks = breaks_t, labels = FALSE,
                     include.lowest = TRUE)

region_names <- paste0("region_", seq_len(n_regions))

# One-hot encoding
A_source <- matrix(0, n_source, n_regions)
A_target <- matrix(0, n_target, n_regions)
colnames(A_source) <- region_names
colnames(A_target) <- region_names
for (i in seq_len(n_source)) A_source[i, region_source[i]] <- 1
for (i in seq_len(n_target)) A_target[i, region_target[i]] <- 1

cat(sprintf("  Source region counts: %s\n",
            paste(colSums(A_source), collapse = ", ")))
cat(sprintf("  Target region counts: %s\n",
            paste(colSums(A_target), collapse = ", ")))

# --- Save input data ---
dir.create("data/toy", recursive = TRUE, showWarnings = FALSE)
write.csv(X, "data/toy/source_exp.csv", row.names = FALSE)
write.csv(Y, "data/toy/target_exp.csv", row.names = FALSE)
write.csv(A_source, "data/toy/source_anatomy.csv", row.names = FALSE)
write.csv(A_target, "data/toy/target_anatomy.csv", row.names = FALSE)
cat("\nInput data saved to data/toy/\n")

# --- Step 1: Test quantile transform alone ---
cat("\n============================================\n")
cat("  Step 1: Quantile Transform Test\n")
cat("============================================\n\n")

qt_x <- quantile_transform(X)
qt_y <- quantile_transform(Y)

cat(sprintf("  X_q summary: min=%.4f, median=%.4f, max=%.4f\n",
            min(qt_x$X_q), median(qt_x$X_q), max(qt_x$X_q)))
cat(sprintf("  Y_q summary: min=%.4f, median=%.4f, max=%.4f\n",
            min(qt_y$X_q), median(qt_y$X_q), max(qt_y$X_q)))

# --- Step 2: Test inverse mapping alone ---
cat("\n============================================\n")
cat("  Step 2: Inverse Mapping Test\n")
cat("============================================\n\n")

inv_iso <- learn_inverse_isotonic(qt_y$fits)
Y_reconstructed <- inverse_transform(qt_y$X_q, inv_iso)

# Reconstruction should be near-perfect for training data
recon_error <- mean(abs(Y - Y_reconstructed))
cat(sprintf("  Isotonic reconstruction MAE: %.6f (should be ~0)\n", recon_error))

# Test with RF if available
tryCatch({
    inv_rf <- learn_inverse_rf(qt_y$fits, ntree = 50)
    Y_recon_rf <- inverse_transform(qt_y$X_q, inv_rf)
    recon_error_rf <- mean(abs(Y - Y_recon_rf))
    cat(sprintf("  RF reconstruction MAE: %.6f\n", recon_error_rf))
}, error = function(e) {
    cat(sprintf("  RF skipped: %s\n", e$message))
})

# --- Step 3: Full pipeline ---
cat("\n============================================\n")
cat("  Step 3: Full Pipeline\n")
cat("============================================\n\n")

result <- run_pipeline(
    X = X, Y = Y,
    A_source = A_source, A_target = A_target,
    r = min(5, n_latent), k = 5,
    inverse_method = "isotonic",
    outdir = "output/toy"
)

# --- Summary ---
cat("\n============================================\n")
cat("  Summary\n")
cat("============================================\n\n")

cat("Output shapes:\n")
cat(sprintf("  warped_quantile:     %d x %d\n",
            nrow(result$warped_quantile), ncol(result$warped_quantile)))
cat(sprintf("  warped_raw:          %d x %d\n",
            nrow(result$warped_raw), ncol(result$warped_raw)))

cat(sprintf("\n  warped_quantile range: [%.4f, %.4f]\n",
            min(result$warped_quantile), max(result$warped_quantile)))
cat(sprintf("  warped_raw range:      [%.4f, %.4f]\n",
            min(result$warped_raw), max(result$warped_raw)))

# Cross-correlation between warped source and target (random baseline)
cat("\nCross-modal correlations (warped source lipids vs target genes):\n")
for (s in colnames(result$warped_raw)[1:3]) {
    for (t in colnames(Y)[1:2]) {
        cc <- cor(result$warped_raw[, s], Y[, t])
        cat(sprintf("  %s vs %s: pearson=%.4f\n", s, t, cc))
    }
}

cat("\nToy pipeline completed successfully.\n")
cat("Intermediate results saved to output/toy/\n")
