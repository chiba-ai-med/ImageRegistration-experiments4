# run_variance_study.R
# EnsembleOT variance/sensitivity study on baseline synthetic data.
#
# 1. run_level        — per-run cluster_preservation spread
# 2. ensemble_size    — mean ± std as n_runs grows
# 3. cluster_count    — accuracy/time trade-off vs k
# 4. reproducibility  — sanity check that fixed random_state gives bit-exact

set.seed(123)

source("src/functions_pipeline.R")
source("src/functions_gwot.R")
library("ggplot2")
library("jsonlite")

if (Sys.getenv("GWOT_PYTHON", unset = "") == "") {
    py <- file.path(getwd(), ".venv_ot", "bin", "python")
    if (file.exists(py)) Sys.setenv(GWOT_PYTHON = py)
}
python_bin <- Sys.getenv("GWOT_PYTHON", unset = "python3")

# --- Minimal synthetic generator (subset of benchmark's) ---
gen <- function(seed = 42) {
    set.seed(seed)
    n_source <- 300; n_target <- 200
    p_source <- 20;  p_target <- 15
    n_latent <- 3;   n_regions <- 4
    cluster_source <- sample(seq_len(n_regions), n_source, replace = TRUE)
    cluster_target <- sample(seq_len(n_regions), n_target, replace = TRUE)
    cl_centers <- matrix(rnorm(n_regions * n_latent, sd = 2), n_regions, n_latent)
    Zs <- cl_centers[cluster_source, ] + matrix(rnorm(n_source*n_latent, sd=0.5), n_source, n_latent)
    Zt <- cl_centers[cluster_target, ] + matrix(rnorm(n_target*n_latent, sd=0.5), n_target, n_latent)
    Ws <- matrix(rnorm(n_latent * p_source), n_latent, p_source)
    Wt <- matrix(rnorm(n_latent * p_target), n_latent, p_target)
    X <- Zs %*% Ws + matrix(rnorm(n_source*p_source, sd=0.3), n_source, p_source)
    Y <- Zt %*% Wt + matrix(rnorm(n_target*p_target, sd=0.3), n_target, p_target)
    X <- sign(X) * abs(X)^1.5
    Y <- sign(Y) * abs(Y)^1.5
    colnames(X) <- paste0("src_feat_", seq_len(p_source))
    colnames(Y) <- paste0("tgt_feat_", seq_len(p_target))
    list(X = X, Y = Y, n_regions = n_regions,
         cluster_source = cluster_source, cluster_target = cluster_target)
}

# Cluster-mean preservation metric
eval_cluster_preservation <- function(warped, syn) {
    true_cm <- do.call(rbind, lapply(seq_len(syn$n_regions), function(r) {
        idx <- which(syn$cluster_source == r)
        if (length(idx) == 0) rep(NA_real_, ncol(syn$X))
        else colMeans(syn$X[idx, , drop = FALSE])
    }))
    warped_cm <- do.call(rbind, lapply(seq_len(syn$n_regions), function(r) {
        idx <- which(syn$cluster_target == r)
        if (length(idx) == 0) rep(NA_real_, ncol(warped))
        else colMeans(warped[idx, , drop = FALSE])
    }))
    cors <- sapply(seq_len(syn$n_regions), function(r) {
        if (any(is.na(warped_cm[r, ])) || any(is.na(true_cm[r, ])))
            return(NA_real_)
        cor(warped_cm[r, ], true_cm[r, ])
    })
    mean(cors, na.rm = TRUE)
}

# --- Prepare data on disk ---
cat("=== Variance Study: generating synthetic baseline ===\n")
syn <- gen(seed = 42)
workdir <- "output/variance"
dir.create(workdir, recursive = TRUE, showWarnings = FALSE)
write.csv(syn$X, file.path(workdir, "X.csv"), row.names = FALSE)
write.csv(syn$Y, file.path(workdir, "Y.csv"), row.names = FALSE)

# --- Invoke variance_study.py ---
cat("\n=== Calling variance_study.py ===\n")
args <- c(
    "src/variance_study.py",
    "--source", file.path(workdir, "X.csv"),
    "--target", file.path(workdir, "Y.csv"),
    "--outdir", workdir,
    "--n-clusters", "20",
    "--epsilon", "0.05",
    "--run-level-n-runs", "30",
    "--ensemble-sizes", "1,3,5,10,20,50",
    "--ensemble-n-seeds", "10",
    "--cluster-ks", "5,10,20,40,80",
    "--cluster-n-seeds", "5",
    "--cluster-n-runs", "5",
    "--repro-n-repeats", "3",
    "--repro-n-runs", "5"
)
status <- system2(python_bin, args)
if (status != 0) stop("variance_study.py failed")

# --- Evaluate each warped CSV ---
cat("\n=== Evaluating warped outputs ===\n")
read_warped <- function(p) as.matrix(read.csv(p, check.names = FALSE))

# 1. run_level
run_level_dir <- file.path(workdir, "run_level")
run_level_files <- sort(list.files(run_level_dir, pattern = "^warped_run_.*\\.csv$", full.names = TRUE))
run_level_df <- data.frame(
    run = seq_along(run_level_files) - 1,
    cluster_preservation = sapply(run_level_files, function(p)
        eval_cluster_preservation(read_warped(p), syn))
)
write.csv(run_level_df, file.path(workdir, "run_level_eval.csv"), row.names = FALSE)
cat(sprintf("[run_level] mean=%.4f sd=%.4f min=%.4f max=%.4f n=%d\n",
            mean(run_level_df$cluster_preservation),
            sd(run_level_df$cluster_preservation),
            min(run_level_df$cluster_preservation),
            max(run_level_df$cluster_preservation),
            nrow(run_level_df)))

# 2. ensemble_size
es_dir <- file.path(workdir, "ensemble_size")
es_files <- list.files(es_dir, pattern = "^warped_size.*\\.csv$", full.names = TRUE)
es_parsed <- regmatches(basename(es_files),
                        regexec("warped_size(\\d+)_seed(\\d+)\\.csv", basename(es_files)))
es_df <- do.call(rbind, lapply(seq_along(es_files), function(i) {
    m <- es_parsed[[i]]
    data.frame(
        n_runs = as.integer(m[2]),
        seed = as.integer(m[3]),
        cluster_preservation = eval_cluster_preservation(read_warped(es_files[i]), syn)
    )
}))
es_timing <- fromJSON(file.path(es_dir, "ensemble_size.json"))
es_df <- merge(es_df, es_timing[, c("n_runs", "seed", "elapsed_sec")],
               by = c("n_runs", "seed"))
write.csv(es_df, file.path(workdir, "ensemble_size_eval.csv"), row.names = FALSE)

es_summary <- aggregate(cbind(cluster_preservation, elapsed_sec) ~ n_runs,
                        data = es_df,
                        FUN = function(x) c(mean = mean(x), sd = sd(x)))
cat("[ensemble_size] summary (mean ± sd):\n"); print(es_summary)

# 3. cluster_count
cc_dir <- file.path(workdir, "cluster_count")
cc_files <- list.files(cc_dir, pattern = "^warped_k.*\\.csv$", full.names = TRUE)
cc_parsed <- regmatches(basename(cc_files),
                        regexec("warped_k(\\d+)_seed(\\d+)\\.csv", basename(cc_files)))
cc_df <- do.call(rbind, lapply(seq_along(cc_files), function(i) {
    m <- cc_parsed[[i]]
    data.frame(
        n_clusters = as.integer(m[2]),
        seed = as.integer(m[3]),
        cluster_preservation = eval_cluster_preservation(read_warped(cc_files[i]), syn)
    )
}))
cc_timing <- fromJSON(file.path(cc_dir, "cluster_count.json"))
cc_df <- merge(cc_df, cc_timing[, c("n_clusters", "seed", "elapsed_sec")],
               by = c("n_clusters", "seed"))
write.csv(cc_df, file.path(workdir, "cluster_count_eval.csv"), row.names = FALSE)
cc_summary <- aggregate(cbind(cluster_preservation, elapsed_sec) ~ n_clusters,
                        data = cc_df,
                        FUN = function(x) c(mean = mean(x), sd = sd(x)))
cat("[cluster_count] summary:\n"); print(cc_summary)

# 4. reproducibility
repro_json <- fromJSON(file.path(workdir, "reproducibility", "reproducibility.json"))
cat(sprintf("[reproducibility] identical=%s (hashes: %s)\n",
            repro_json$identical,
            paste(substr(repro_json$hashes, 1, 10), collapse = ", ")))

# --- Plots ---
plotdir <- "plot/variance"
dir.create(plotdir, recursive = TRUE, showWarnings = FALSE)

g1 <- ggplot(run_level_df, aes(y = cluster_preservation, x = "")) +
    geom_boxplot(fill = "#ffcc00", width = 0.3) +
    geom_jitter(width = 0.05, size = 1.2, alpha = 0.6) +
    theme_minimal() +
    labs(title = "Run-level variance (fixed seed, 30 runs)",
         x = NULL, y = "Cluster-mean preservation")
ggsave(file.path(plotdir, "run_level.png"), g1, width = 5, height = 5, dpi = 150)

g2 <- ggplot(es_df, aes(x = factor(n_runs), y = cluster_preservation)) +
    geom_boxplot(fill = "#66c2a5") +
    theme_minimal() +
    labs(title = "Ensemble size vs accuracy",
         x = "n_runs (ensemble size)", y = "Cluster-mean preservation")
ggsave(file.path(plotdir, "ensemble_size.png"), g2, width = 7, height = 5, dpi = 150)

g2b <- ggplot(es_df, aes(x = factor(n_runs), y = elapsed_sec)) +
    geom_boxplot(fill = "#fc8d62") +
    theme_minimal() +
    labs(title = "Ensemble size vs wall-clock",
         x = "n_runs", y = "Elapsed (sec)")
ggsave(file.path(plotdir, "ensemble_size_time.png"), g2b, width = 7, height = 5, dpi = 150)

g3 <- ggplot(cc_df, aes(x = factor(n_clusters), y = cluster_preservation)) +
    geom_boxplot(fill = "#8da0cb") +
    theme_minimal() +
    labs(title = "Cluster count sweep (n_runs=5)",
         x = "n_clusters", y = "Cluster-mean preservation")
ggsave(file.path(plotdir, "cluster_count.png"), g3, width = 7, height = 5, dpi = 150)

g3b <- ggplot(cc_df, aes(x = factor(n_clusters), y = elapsed_sec)) +
    geom_boxplot(fill = "#e78ac3") +
    theme_minimal() +
    labs(title = "Cluster count vs wall-clock",
         x = "n_clusters", y = "Elapsed (sec)")
ggsave(file.path(plotdir, "cluster_count_time.png"), g3b, width = 7, height = 5, dpi = 150)

cat("\n=== Variance study complete. Plots in plot/variance/ ===\n")
