# run_cluster_sweep.R
# Sweep n_clusters for EnsembleOT (kmeans vs random_voronoi)
# on both synthetic and real data.
#
# Measures: accuracy (cluster_preservation or Pearson), elapsed_sec, peak_rss_mb

source("src/functions_pipeline.R")
source("src/functions_gwot.R")
library("ggplot2")
library("jsonlite")

if (Sys.getenv("GWOT_PYTHON", unset = "") == "") {
    py <- file.path(getwd(), ".venv_ot", "bin", "python")
    if (file.exists(py)) Sys.setenv(GWOT_PYTHON = py)
}

set.seed(123)
dir.create("output/cluster_sweep", recursive = TRUE, showWarnings = FALSE)
dir.create("plot/cluster_sweep",   recursive = TRUE, showWarnings = FALSE)

N_CLUSTERS_GRID <- c(10, 20, 50, 100, 200)
N_RUNS <- 5

# ============================================================
# Part 1: Synthetic data
# ============================================================
cat("============================================\n")
cat("  Cluster Sweep: Synthetic Data\n")
cat("============================================\n\n")

# generate_synthetic is defined inline (from run_synthetic_benchmark.R)
generate_synthetic <- function(n_source = 300, n_target = 200,
                               p_source = 20, p_target = 15,
                               n_latent = 3, n_regions = 4,
                               noise_sd = 0.5, dropout_rate = 0.0,
                               cluster_imbalance = FALSE,
                               monotone_transform = "power",
                               seed = 42) {
    set.seed(seed)
    if (cluster_imbalance) {
        prob_source <- c(0.5, 0.2, 0.2, 0.1)[seq_len(n_regions)]
        prob_target <- c(0.1, 0.2, 0.2, 0.5)[seq_len(n_regions)]
    } else {
        prob_source <- rep(1 / n_regions, n_regions)
        prob_target <- rep(1 / n_regions, n_regions)
    }
    prob_source <- prob_source / sum(prob_source)
    prob_target <- prob_target / sum(prob_target)
    cluster_source <- sample(seq_len(n_regions), n_source, replace = TRUE, prob = prob_source)
    cluster_target <- sample(seq_len(n_regions), n_target, replace = TRUE, prob = prob_target)
    cluster_centers <- matrix(rnorm(n_regions * n_latent, sd = 2), n_regions, n_latent)
    Z_source <- cluster_centers[cluster_source, ] +
        matrix(rnorm(n_source * n_latent, sd = 0.5), n_source, n_latent)
    Z_target <- cluster_centers[cluster_target, ] +
        matrix(rnorm(n_target * n_latent, sd = 0.5), n_target, n_latent)
    W_source <- matrix(rnorm(n_latent * p_source), n_latent, p_source)
    W_target <- matrix(rnorm(n_latent * p_target), n_latent, p_target)
    X_raw <- Z_source %*% W_source +
        matrix(rnorm(n_source * p_source, sd = noise_sd), n_source, p_source)
    Y_raw <- Z_target %*% W_target +
        matrix(rnorm(n_target * p_target, sd = noise_sd), n_target, p_target)
    apply_transform <- function(M, type) {
        if (type == "power") sign(M) * abs(M)^1.5
        else if (type == "exp") exp(M * 0.3)
        else if (type == "sigmoid") 1 / (1 + exp(-M))
        else M
    }
    X <- apply_transform(X_raw, monotone_transform)
    Y <- apply_transform(Y_raw, monotone_transform)
    if (dropout_rate > 0) {
        X <- X * matrix(rbinom(n_source * p_source, 1, 1 - dropout_rate), n_source, p_source)
        Y <- Y * matrix(rbinom(n_target * p_target, 1, 1 - dropout_rate), n_target, p_target)
    }
    colnames(X) <- paste0("src_feat_", seq_len(p_source))
    colnames(Y) <- paste0("tgt_feat_", seq_len(p_target))
    region_names <- paste0("region_", seq_len(n_regions))
    A_source <- matrix(0, n_source, n_regions, dimnames = list(NULL, region_names))
    A_target <- matrix(0, n_target, n_regions, dimnames = list(NULL, region_names))
    for (i in seq_len(n_source)) A_source[i, cluster_source[i]] <- 1
    for (i in seq_len(n_target)) A_target[i, cluster_target[i]] <- 1
    list(X = X, Y = Y, A_source = A_source, A_target = A_target,
         Z_source = Z_source, Z_target = Z_target,
         cluster_source = cluster_source, cluster_target = cluster_target,
         W_source = W_source, W_target = W_target,
         params = list(n_source = n_source, n_target = n_target,
                       p_source = p_source, p_target = p_target,
                       n_latent = n_latent, n_regions = n_regions,
                       noise_sd = noise_sd, dropout_rate = dropout_rate,
                       cluster_imbalance = cluster_imbalance,
                       monotone_transform = monotone_transform))
}

syn <- generate_synthetic(
    n_source = 300, n_target = 200,
    p_source = 20, p_target = 15,
    n_latent = 3, n_regions = 4,
    noise_sd = 0.3, dropout_rate = 0.0,
    cluster_imbalance = FALSE,
    monotone_transform = "power",
    seed = 42
)

# True cluster means for evaluation
true_src_cluster_means <- do.call(rbind, lapply(
    seq_len(syn$params$n_regions), function(r) {
        idx <- which(syn$cluster_source == r)
        if (length(idx) == 0) rep(NA_real_, ncol(syn$X))
        else colMeans(syn$X[idx, , drop = FALSE])
    }))

cluster_preservation <- function(warped) {
    if (is.null(warped)) return(NA_real_)
    if (ncol(warped) != ncol(true_src_cluster_means)) return(NA_real_)
    warped_cm <- do.call(rbind, lapply(
        seq_len(syn$params$n_regions), function(r) {
            idx <- which(syn$cluster_target == r)
            if (length(idx) == 0) rep(NA_real_, ncol(warped))
            else colMeans(warped[idx, , drop = FALSE])
        }))
    cors <- sapply(seq_len(syn$params$n_regions), function(r) {
        if (any(is.na(warped_cm[r, ])) || any(is.na(true_src_cluster_means[r, ])))
            return(NA_real_)
        cor(warped_cm[r, ], true_src_cluster_means[r, ])
    })
    mean(cors, na.rm = TRUE)
}

# Sweep configurations: backend × clustering_method × quantile
configs <- list(
    list(label = "GW",  backend = "ensembleot",     fgw = FALSE),
    list(label = "FGW", backend = "ensembleot_fgw",  fgw = TRUE)
)
clustering_methods <- c("kmeans", "random_voronoi")
quantile_modes <- c(FALSE, TRUE)

syn_results <- data.frame()

for (cfg in configs) {
    for (cm in clustering_methods) {
        for (qt in quantile_modes) {
            for (nc in N_CLUSTERS_GRID) {
                method_label <- sprintf("%s_%s_%s_k%d",
                    cfg$label,
                    ifelse(cm == "kmeans", "km", "rv"),
                    ifelse(qt, "qt", "raw"),
                    nc)
                cat(sprintf("\n>>> Synthetic: %s\n", method_label))

                extra <- if (cfg$fgw) {
                    list(A_source = syn$A_source, A_target = syn$A_target, alpha = 0.5)
                } else list()

                res <- tryCatch({
                    common <- list(
                        X = syn$X, Y = syn$Y, backend = cfg$backend,
                        n_clusters = min(nc, nrow(syn$X), nrow(syn$Y)),
                        n_runs = N_RUNS,
                        clustering_method = cm,
                        epsilon = 0.05,
                        outdir = NULL
                    )
                    if (qt) {
                        do.call(run_pipeline_gwot_quantile, c(common, extra))
                    } else {
                        do.call(run_pipeline_gwot_raw, c(common, extra))
                    }
                }, error = function(e) {
                    cat(sprintf("  ERROR: %s\n", e$message))
                    NULL
                })

                if (!is.null(res)) {
                    w <- if (qt) res$warped_quantile else res$warped_exp
                    cp <- cluster_preservation(w)
                    syn_results <- rbind(syn_results, data.frame(
                        type = cfg$label,
                        clustering = cm,
                        quantile = qt,
                        n_clusters = nc,
                        cluster_preservation = cp,
                        elapsed_sec = res$elapsed_sec,
                        peak_rss_mb = if (!is.null(res$peak_rss_mb)) res$peak_rss_mb else NA_real_,
                        stringsAsFactors = FALSE
                    ))
                }
            }
        }
    }
}

cat("\n\n=== Synthetic Sweep Results ===\n")
print(syn_results)
write.csv(syn_results, "output/cluster_sweep/synthetic_results.csv", row.names = FALSE)

# ============================================================
# Part 2: Real data
# ============================================================
cat("\n\n============================================\n")
cat("  Cluster Sweep: Real Data\n")
cat("============================================\n\n")

DATASETS <- list(
    kidney = list(
        source_csv = "/home/koki/dev/ImageRegistration-experiments3/data/kidney/kidney_maldi.csv",
        target_csv = "/home/koki/dev/ImageRegistration-experiments3/data/kidney/kidney_xenium.csv",
        region_col = "region",
        source_markers = c("FA.22.6"),
        target_markers = c("Slc27a2")
    ),
    `251208` = list(
        source_csv = "/home/koki/dev/ImageRegistration-experiments3/data/251208/data/df_sl12.csv",
        target_csv = "/home/koki/dev/ImageRegistration-experiments3/data/251208/data/df_st32.csv",
        region_col = "ccf_region_name",
        source_markers = c(
            "HexCer.36.1.O2.1", "HexCer.36.2.O2", "HexCer.38.2.O2",
            "HexCer.40.0.O2", "HexCer.40.1.O2", "HexCer.40.2.O2",
            "HexCer.41.1.O2", "HexCer.42.1.O2", "HexCer.42.2.O2",
            "HexCer.44.2.O2",
            "SM.31.1.O2", "SM.34.1.O2", "SM.35.1.O2",
            "SM.36.1.O2", "SM.36.2.O2", "SM.38.1.O2",
            "SM.41.2.O2", "SM.42.1.O2", "SM.42.2.O2", "SM.42.3.O2"),
        target_markers = c("Mog", "Sox10")
    )
)

load_dataset <- function(info) {
    src <- read.csv(info$source_csv, check.names = TRUE)
    tgt <- read.csv(info$target_csv, check.names = TRUE)
    drop_cols <- c("X", "Y", info$region_col)
    src_feat <- as.matrix(src[, !(colnames(src) %in% drop_cols)])
    tgt_feat <- as.matrix(tgt[, !(colnames(tgt) %in% drop_cols)])
    src_region <- as.character(src[[info$region_col]])
    tgt_region <- as.character(tgt[[info$region_col]])
    regions <- sort(unique(c(
        src_region[!is.na(src_region)],
        tgt_region[!is.na(tgt_region)]
    )))
    onehot <- function(v) {
        v[is.na(v)] <- regions[1]
        idx <- match(v, regions)
        M <- matrix(0, length(v), length(regions))
        colnames(M) <- paste0("region_", seq_along(regions))
        M[cbind(seq_along(v), idx)] <- 1
        M
    }
    list(X = src_feat, Y = tgt_feat,
         A_source = onehot(src_region), A_target = onehot(tgt_region),
         target_all = tgt_feat)
}

eval_markers <- function(warped_exp, target_all, source_markers, target_markers) {
    cors <- c()
    for (s in source_markers) {
        if (!(s %in% colnames(warped_exp))) next
        for (t in target_markers) {
            if (!(t %in% colnames(target_all))) next
            r <- tryCatch(
                cor(warped_exp[, s], target_all[, t], method = "pearson"),
                error = function(e) NA_real_
            )
            cors <- c(cors, abs(r))
        }
    }
    if (length(cors) == 0) NA_real_ else mean(cors, na.rm = TRUE)
}

real_results <- data.frame()

for (ds_name in names(DATASETS)) {
    cat(sprintf("\n########################################\n"))
    cat(sprintf("  Dataset: %s\n", ds_name))
    cat(sprintf("########################################\n\n"))

    info <- DATASETS[[ds_name]]
    data <- load_dataset(info)
    n_x <- nrow(data$X); n_y <- nrow(data$Y)

    for (cfg in configs) {
        for (cm in clustering_methods) {
            for (qt in quantile_modes) {
                for (nc in N_CLUSTERS_GRID) {
                    actual_nc <- min(nc, n_x, n_y)
                    method_label <- sprintf("%s_%s_%s_k%d",
                        cfg$label,
                        ifelse(cm == "kmeans", "km", "rv"),
                        ifelse(qt, "qt", "raw"),
                        nc)
                    cat(sprintf("\n>>> %s: %s\n", ds_name, method_label))

                    extra <- if (cfg$fgw) {
                        list(A_source = data$A_source, A_target = data$A_target, alpha = 0.5)
                    } else list()

                    res <- tryCatch({
                        common <- list(
                            X = data$X, Y = data$Y, backend = cfg$backend,
                            n_clusters = actual_nc,
                            n_runs = N_RUNS,
                            clustering_method = cm,
                            epsilon = 0.05,
                            outdir = NULL
                        )
                        if (qt) {
                            do.call(run_pipeline_gwot_quantile, c(common, extra))
                        } else {
                            do.call(run_pipeline_gwot_raw, c(common, extra))
                        }
                    }, error = function(e) {
                        cat(sprintf("  ERROR: %s\n", e$message))
                        NULL
                    })

                    if (!is.null(res)) {
                        w <- if (qt) res$warped_quantile else res$warped_exp
                        pearson <- eval_markers(w, data$target_all,
                                               info$source_markers, info$target_markers)
                        real_results <- rbind(real_results, data.frame(
                            dataset = ds_name,
                            type = cfg$label,
                            clustering = cm,
                            quantile = qt,
                            n_clusters = nc,
                            mean_abs_pearson = pearson,
                            elapsed_sec = res$elapsed_sec,
                            peak_rss_mb = if (!is.null(res$peak_rss_mb)) res$peak_rss_mb else NA_real_,
                            stringsAsFactors = FALSE
                        ))
                    }
                }
            }
        }
    }
}

cat("\n\n=== Real Data Sweep Results ===\n")
print(real_results)
write.csv(real_results, "output/cluster_sweep/real_results.csv", row.names = FALSE)

# ============================================================
# Plots
# ============================================================
cat("\n\nGenerating plots...\n")

# Helper: method label
make_label <- function(df) {
    paste0(df$type, "_",
           ifelse(df$clustering == "kmeans", "km", "rv"), "_",
           ifelse(df$quantile, "qt", "raw"))
}

# Synthetic plots
if (nrow(syn_results) > 0) {
    syn_results$method <- make_label(syn_results)

    g1 <- ggplot(syn_results, aes(x = n_clusters, y = cluster_preservation,
                                   color = method, linetype = method)) +
        geom_line(linewidth = 0.8) + geom_point(size = 2) +
        theme_minimal() +
        labs(title = "Synthetic: Cluster preservation vs n_clusters",
             x = "n_clusters", y = "Cluster preservation") +
        scale_x_continuous(breaks = N_CLUSTERS_GRID)
    ggsave("plot/cluster_sweep/synthetic_accuracy.png", g1, width = 10, height = 6, dpi = 150)

    g2 <- ggplot(syn_results, aes(x = n_clusters, y = elapsed_sec,
                                   color = method, linetype = method)) +
        geom_line(linewidth = 0.8) + geom_point(size = 2) +
        theme_minimal() +
        labs(title = "Synthetic: Elapsed time vs n_clusters",
             x = "n_clusters", y = "Elapsed (sec)") +
        scale_x_continuous(breaks = N_CLUSTERS_GRID)
    ggsave("plot/cluster_sweep/synthetic_timing.png", g2, width = 10, height = 6, dpi = 150)

    if (any(!is.na(syn_results$peak_rss_mb))) {
        g3 <- ggplot(syn_results[!is.na(syn_results$peak_rss_mb), ],
                     aes(x = n_clusters, y = peak_rss_mb,
                         color = method, linetype = method)) +
            geom_line(linewidth = 0.8) + geom_point(size = 2) +
            theme_minimal() +
            labs(title = "Synthetic: Peak RSS vs n_clusters",
                 x = "n_clusters", y = "Peak RSS (MB)") +
            scale_x_continuous(breaks = N_CLUSTERS_GRID)
        ggsave("plot/cluster_sweep/synthetic_memory.png", g3, width = 10, height = 6, dpi = 150)
    }
}

# Real data plots
if (nrow(real_results) > 0) {
    real_results$method <- make_label(real_results)

    g4 <- ggplot(real_results, aes(x = n_clusters, y = mean_abs_pearson,
                                    color = method, linetype = method)) +
        geom_line(linewidth = 0.8) + geom_point(size = 2) +
        facet_wrap(~ dataset, scales = "free_y") +
        theme_minimal() +
        labs(title = "Real data: |Pearson| vs n_clusters",
             x = "n_clusters", y = "Mean |Pearson|") +
        scale_x_continuous(breaks = N_CLUSTERS_GRID)
    ggsave("plot/cluster_sweep/real_accuracy.png", g4, width = 12, height = 6, dpi = 150)

    g5 <- ggplot(real_results, aes(x = n_clusters, y = elapsed_sec,
                                    color = method, linetype = method)) +
        geom_line(linewidth = 0.8) + geom_point(size = 2) +
        facet_wrap(~ dataset, scales = "free_y") +
        theme_minimal() +
        labs(title = "Real data: Elapsed time vs n_clusters",
             x = "n_clusters", y = "Elapsed (sec)") +
        scale_x_continuous(breaks = N_CLUSTERS_GRID)
    ggsave("plot/cluster_sweep/real_timing.png", g5, width = 12, height = 6, dpi = 150)

    if (any(!is.na(real_results$peak_rss_mb))) {
        g6 <- ggplot(real_results[!is.na(real_results$peak_rss_mb), ],
                     aes(x = n_clusters, y = peak_rss_mb,
                         color = method, linetype = method)) +
            geom_line(linewidth = 0.8) + geom_point(size = 2) +
            facet_wrap(~ dataset, scales = "free_y") +
            theme_minimal() +
            labs(title = "Real data: Peak RSS vs n_clusters",
                 x = "n_clusters", y = "Peak RSS (MB)") +
            scale_x_continuous(breaks = N_CLUSTERS_GRID)
        ggsave("plot/cluster_sweep/real_memory.png", g6, width = 12, height = 6, dpi = 150)
    }
}

cat("\nCluster sweep complete. Results in output/cluster_sweep/, plots in plot/cluster_sweep/.\n")
