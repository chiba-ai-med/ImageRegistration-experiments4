# run_synthetic_benchmark.R
# Synthetic benchmark with controllable noise, dropout, sample/cluster imbalance

set.seed(123)

source("src/functions_pipeline.R")
source("src/functions_gwot.R")

# Default to the project-local venv if GWOT_PYTHON isn't set.
if (Sys.getenv("GWOT_PYTHON", unset = "") == "") {
    # NB: do NOT normalizePath — venv's python is a symlink to the base
    # interpreter, and invoking via the resolved real path bypasses venv
    # site-packages activation.
    default_py <- file.path(getwd(), ".venv_ot", "bin", "python")
    if (file.exists(default_py)) Sys.setenv(GWOT_PYTHON = default_py)
}

library("ggplot2")
if (!requireNamespace("jsonlite", quietly = TRUE)) {
    install.packages("jsonlite", repos = "https://cloud.r-project.org")
}

cat("============================================\n")
cat("  Synthetic Benchmark: Quantile-GuidedPLS\n")
cat("============================================\n\n")

# --- Synthetic data generator ---

#' Generate synthetic cross-modal data from shared latent structure
#'
#' @param n_source number of source samples
#' @param n_target number of target samples
#' @param p_source source features
#' @param p_target target features
#' @param n_latent latent dimensions
#' @param n_regions number of anatomy regions
#' @param noise_sd noise standard deviation
#' @param dropout_rate fraction of entries set to zero
#' @param cluster_imbalance if TRUE, source/target have different cluster proportions
#' @param monotone_transform type of nonlinear transform ("power", "exp", "sigmoid", "none")
#' @param seed random seed
generate_synthetic <- function(n_source = 300, n_target = 200,
                               p_source = 20, p_target = 15,
                               n_latent = 3, n_regions = 4,
                               noise_sd = 0.5, dropout_rate = 0.0,
                               cluster_imbalance = FALSE,
                               monotone_transform = "power",
                               seed = 42) {
    set.seed(seed)

    # Cluster assignments (latent structure)
    if (cluster_imbalance) {
        # Different proportions per modality
        prob_source <- c(0.5, 0.2, 0.2, 0.1)[seq_len(n_regions)]
        prob_target <- c(0.1, 0.2, 0.2, 0.5)[seq_len(n_regions)]
    } else {
        prob_source <- rep(1 / n_regions, n_regions)
        prob_target <- rep(1 / n_regions, n_regions)
    }
    prob_source <- prob_source / sum(prob_source)
    prob_target <- prob_target / sum(prob_target)

    cluster_source <- sample(seq_len(n_regions), n_source,
                             replace = TRUE, prob = prob_source)
    cluster_target <- sample(seq_len(n_regions), n_target,
                             replace = TRUE, prob = prob_target)

    # Cluster centers in latent space
    cluster_centers <- matrix(rnorm(n_regions * n_latent, sd = 2),
                              n_regions, n_latent)

    # Latent variables
    Z_source <- cluster_centers[cluster_source, ] +
        matrix(rnorm(n_source * n_latent, sd = 0.5), n_source, n_latent)
    Z_target <- cluster_centers[cluster_target, ] +
        matrix(rnorm(n_target * n_latent, sd = 0.5), n_target, n_latent)

    # Loadings
    W_source <- matrix(rnorm(n_latent * p_source), n_latent, p_source)
    W_target <- matrix(rnorm(n_latent * p_target), n_latent, p_target)

    # Raw expression
    X_raw <- Z_source %*% W_source +
        matrix(rnorm(n_source * p_source, sd = noise_sd), n_source, p_source)
    Y_raw <- Z_target %*% W_target +
        matrix(rnorm(n_target * p_target, sd = noise_sd), n_target, p_target)

    # Monotone transform (modality-specific)
    apply_transform <- function(M, type) {
        if (type == "power") {
            sign(M) * abs(M)^1.5
        } else if (type == "exp") {
            exp(M * 0.3)
        } else if (type == "sigmoid") {
            1 / (1 + exp(-M))
        } else {
            M
        }
    }
    X <- apply_transform(X_raw, monotone_transform)
    Y <- apply_transform(Y_raw, monotone_transform)

    # Dropout
    if (dropout_rate > 0) {
        mask_x <- matrix(rbinom(n_source * p_source, 1, 1 - dropout_rate),
                         n_source, p_source)
        mask_y <- matrix(rbinom(n_target * p_target, 1, 1 - dropout_rate),
                         n_target, p_target)
        X <- X * mask_x
        Y <- Y * mask_y
    }

    colnames(X) <- paste0("src_feat_", seq_len(p_source))
    colnames(Y) <- paste0("tgt_feat_", seq_len(p_target))

    # One-hot anatomy
    region_names <- paste0("region_", seq_len(n_regions))
    A_source <- matrix(0, n_source, n_regions, dimnames = list(NULL, region_names))
    A_target <- matrix(0, n_target, n_regions, dimnames = list(NULL, region_names))
    for (i in seq_len(n_source)) A_source[i, cluster_source[i]] <- 1
    for (i in seq_len(n_target)) A_target[i, cluster_target[i]] <- 1

    list(
        X = X, Y = Y,
        A_source = A_source, A_target = A_target,
        Z_source = Z_source, Z_target = Z_target,
        cluster_source = cluster_source,
        cluster_target = cluster_target,
        W_source = W_source, W_target = W_target,
        params = list(n_source = n_source, n_target = n_target,
                      p_source = p_source, p_target = p_target,
                      n_latent = n_latent, n_regions = n_regions,
                      noise_sd = noise_sd, dropout_rate = dropout_rate,
                      cluster_imbalance = cluster_imbalance,
                      monotone_transform = monotone_transform)
    )
}


#' Evaluate latent space quality
#'
#' @param gpls_out guidedPLS output
#' @param Z_true_source true latent variables (source)
#' @param Z_true_target true latent variables (target)
#' @return data.frame with latent correlation metrics
evaluate_latent <- function(gpls_out, Z_true_source, Z_true_target) {
    scoreX1 <- apply(gpls_out$scoreX1, 2, scale)
    scoreX2 <- apply(gpls_out$scoreX2, 2, scale)

    n_dim <- min(ncol(Z_true_source), ncol(scoreX1))

    # Canonical correlation between estimated and true latent
    source_cors <- sapply(seq_len(n_dim), function(d) {
        max(abs(cor(Z_true_source[, d], scoreX1)))
    })
    target_cors <- sapply(seq_len(n_dim), function(d) {
        max(abs(cor(Z_true_target[, d], scoreX2)))
    })

    data.frame(
        dim = seq_len(n_dim),
        source_latent_cor = source_cors,
        target_latent_cor = target_cors
    )
}


#' Compute monotonicity violation rate
#'
#' Fraction of pairs where rank order is not preserved
#'
#' @param x predicted values
#' @param y true values
#' @return violation rate in [0, 1]
monotonicity_violation <- function(x, y) {
    n <- length(x)
    if (n < 2) return(0)

    # Sample pairs to avoid O(n^2) for large n
    max_pairs <- min(10000, choose(n, 2))
    idx <- combn(min(n, 200), 2)

    violations <- sum(apply(idx, 2, function(ij) {
        i <- ij[1]; j <- ij[2]
        sign(x[i] - x[j]) != sign(y[i] - y[j])
    }))

    violations / ncol(idx)
}


# --- Run benchmark scenarios ---
scenarios <- list(
    baseline = list(noise_sd = 0.3, dropout_rate = 0.0,
                    cluster_imbalance = FALSE, monotone_transform = "power"),
    high_noise = list(noise_sd = 1.0, dropout_rate = 0.0,
                      cluster_imbalance = FALSE, monotone_transform = "power"),
    dropout = list(noise_sd = 0.3, dropout_rate = 0.3,
                   cluster_imbalance = FALSE, monotone_transform = "power"),
    imbalanced = list(noise_sd = 0.3, dropout_rate = 0.0,
                      cluster_imbalance = TRUE, monotone_transform = "power"),
    exp_transform = list(noise_sd = 0.3, dropout_rate = 0.0,
                         cluster_imbalance = FALSE, monotone_transform = "exp"),
    sigmoid_transform = list(noise_sd = 0.3, dropout_rate = 0.0,
                             cluster_imbalance = FALSE, monotone_transform = "sigmoid"),
    no_transform = list(noise_sd = 0.3, dropout_rate = 0.0,
                        cluster_imbalance = FALSE, monotone_transform = "none")
)

all_results <- data.frame()
dir.create("output/benchmark", recursive = TRUE, showWarnings = FALSE)
dir.create("plot/benchmark", recursive = TRUE, showWarnings = FALSE)

for (scenario_name in names(scenarios)) {
    cat(sprintf("\n\n########################################\n"))
    cat(sprintf("  Scenario: %s\n", scenario_name))
    cat(sprintf("########################################\n\n"))

    params <- scenarios[[scenario_name]]

    # Generate data
    syn <- generate_synthetic(
        n_source = 300, n_target = 200,
        p_source = 20, p_target = 15,
        n_latent = 3, n_regions = 4,
        noise_sd = params$noise_sd,
        dropout_rate = params$dropout_rate,
        cluster_imbalance = params$cluster_imbalance,
        monotone_transform = params$monotone_transform,
        seed = 42
    )

    outdir <- file.path("output/benchmark", scenario_name)

    # --- Run with quantile transform ---
    cat("\n>>> With quantile transform:\n")
    result_qt <- tryCatch({
        run_pipeline(
            X = syn$X, Y = syn$Y,
            A_source = syn$A_source, A_target = syn$A_target,
            r = 3, k = 5,
            inverse_method = "isotonic",
            outdir = file.path(outdir, "quantile")
        )
    }, error = function(e) {
        cat(sprintf("  ERROR: %s\n", e$message))
        NULL
    })

    # --- Run without quantile transform (direct GuidedPLS) ---
    cat("\n>>> Without quantile transform (baseline):\n")
    result_raw <- tryCatch({
        gpls_raw <- run_guidedpls(
            syn$X, syn$Y, syn$A_source, syn$A_target
        )
        warp_raw <- knn_warping(
            gpls_raw, syn$X, syn$A_source, r = 3, k = 5
        )
        list(gpls_out = gpls_raw, warped_raw = warp_raw$warped_exp)
    }, error = function(e) {
        cat(sprintf("  ERROR: %s\n", e$message))
        NULL
    })

    # --- Evaluate latent space ---
    if (!is.null(result_qt)) {
        lat <- evaluate_latent(result_qt$gpls_out, syn$Z_source, syn$Z_target)
        cat("\nLatent correlations (quantile):\n")
        print(lat)
    }

    # --- GW-OT / FGW routes (C-J) ---
    gwot_runs <- list(
        list(label = "pot_gw_raw",             backend = "pot_entropic",    quantile = FALSE, fgw = FALSE),
        list(label = "pot_gw_quantile",        backend = "pot_entropic",    quantile = TRUE,  fgw = FALSE),
        list(label = "ensembleot_gw_raw",      backend = "ensembleot",      quantile = FALSE, fgw = FALSE),
        list(label = "ensembleot_gw_quantile", backend = "ensembleot",      quantile = TRUE,  fgw = FALSE),
        list(label = "pot_fgw_raw",            backend = "pot_fgw_entropic",quantile = FALSE, fgw = TRUE),
        list(label = "pot_fgw_quantile",       backend = "pot_fgw_entropic",quantile = TRUE,  fgw = TRUE),
        list(label = "ensembleot_fgw_raw",     backend = "ensembleot_fgw",  quantile = FALSE, fgw = TRUE),
        list(label = "ensembleot_fgw_quantile",backend = "ensembleot_fgw",  quantile = TRUE,  fgw = TRUE),
        # random_voronoi variants (EnsembleOT only)
        list(label = "ensembleot_gw_raw_rv",      backend = "ensembleot",      quantile = FALSE, fgw = FALSE, clustering_method = "random_voronoi"),
        list(label = "ensembleot_gw_quantile_rv", backend = "ensembleot",      quantile = TRUE,  fgw = FALSE, clustering_method = "random_voronoi"),
        list(label = "ensembleot_fgw_raw_rv",     backend = "ensembleot_fgw",  quantile = FALSE, fgw = TRUE,  clustering_method = "random_voronoi"),
        list(label = "ensembleot_fgw_quantile_rv",backend = "ensembleot_fgw",  quantile = TRUE,  fgw = TRUE,  clustering_method = "random_voronoi")
    )
    gwot_results <- list()
    for (gr in gwot_runs) {
        cat(sprintf("\n>>> GW-OT route: %s\n", gr$label))
        extra <- if (gr$fgw) list(A_source = syn$A_source, A_target = syn$A_target, alpha = 0.5) else list()
        cm <- if (!is.null(gr$clustering_method)) gr$clustering_method else "kmeans"
        res <- tryCatch({
            common <- list(
                X = syn$X, Y = syn$Y, backend = gr$backend,
                n_clusters = min(20, nrow(syn$X), nrow(syn$Y)),
                n_runs = 5, clustering_method = cm, epsilon = 0.05,
                outdir = file.path(outdir, gr$label)
            )
            if (gr$quantile) {
                do.call(run_pipeline_gwot_quantile, c(common, extra))
            } else {
                do.call(run_pipeline_gwot_raw, c(common, extra))
            }
        }, error = function(e) {
            cat(sprintf("  ERROR (%s): %s\n", gr$label, e$message))
            NULL
        })
        gwot_results[[gr$label]] <- res
    }

    # --- Collect summary metrics ---
    # Cluster-mean preservation: compare cluster-wise means of warped source
    # features (grouped by target cluster labels) against true source cluster
    # means. Higher = warping sends target points close to the correct source
    # region. Cross-modal safe (uses only source features on both sides).
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
        # Diagonal correlation: region r warped-mean vs region r true-mean.
        cors <- sapply(seq_len(syn$params$n_regions), function(r) {
            if (any(is.na(warped_cm[r, ])) || any(is.na(true_src_cluster_means[r, ])))
                return(NA_real_)
            cor(warped_cm[r, ], true_src_cluster_means[r, ])
        })
        mean(cors, na.rm = TRUE)
    }
    summarize <- cluster_preservation

    routes <- list()
    if (!is.null(result_qt)) routes[["quantile"]]    <- list(w = result_qt$warped_raw, t = NA_real_)
    if (!is.null(result_raw)) routes[["raw"]]        <- list(w = result_raw$warped_raw, t = NA_real_)
    for (gr in gwot_runs) {
        res <- gwot_results[[gr$label]]
        if (is.null(res)) next
        w <- if (gr$quantile) res$warped_quantile else res$warped_exp
        routes[[gr$label]] <- list(w = w, t = res$elapsed_sec)
    }
    for (nm in names(routes)) {
        all_results <- rbind(all_results, data.frame(
            scenario = scenario_name,
            method = nm,
            cluster_preservation = summarize(routes[[nm]]$w),
            elapsed_sec = routes[[nm]]$t,
            stringsAsFactors = FALSE
        ))
    }

    # Save synthetic data
    saveRDS(syn, file.path(outdir, "synthetic_data.rds"))
}

# --- Summary table ---
cat("\n\n============================================\n")
cat("  Benchmark Summary\n")
cat("============================================\n\n")
print(all_results)

write.csv(all_results, "output/benchmark/summary.csv", row.names = FALSE)

# --- Comparison plot ---
if (nrow(all_results) > 0) {
    g <- ggplot(all_results, aes(x = scenario, y = cluster_preservation,
                                  fill = method)) +
        geom_bar(stat = "identity", position = position_dodge(width = 0.8)) +
        theme_minimal() +
        theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
        labs(title = "Warping routes: Self-Consistency",
             x = "Scenario", y = "Cluster-mean preservation (Pearson)",
             fill = "Method")

    ggsave("plot/benchmark/comparison.png", g, width = 12, height = 6, dpi = 150)

    # Timing plot (GW-OT routes only)
    timing_df <- all_results[!is.na(all_results$elapsed_sec), ]
    if (nrow(timing_df) > 0) {
        g2 <- ggplot(timing_df, aes(x = scenario, y = elapsed_sec,
                                    fill = method)) +
            geom_bar(stat = "identity", position = position_dodge(width = 0.8)) +
            theme_minimal() +
            theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
            labs(title = "GW-OT routes: wall-clock time",
                 x = "Scenario", y = "Elapsed (sec)", fill = "Method")
        ggsave("plot/benchmark/timing.png", g2, width = 12, height = 6, dpi = 150)
    }
    cat("\nPlot saved to plot/benchmark/comparison.png\n")
}

# --- Step 5: Compare isotonic vs RF inverse ---
cat("\n\n============================================\n")
cat("  Isotonic vs Random Forest Inverse\n")
cat("============================================\n\n")

syn <- generate_synthetic(seed = 42)

# Same-modality test: quantile transform Y, then inverse, compare
qt_y <- quantile_transform(syn$Y)

inv_iso <- learn_inverse_isotonic(qt_y$fits)
inv_rf <- tryCatch(
    learn_inverse_rf(qt_y$fits, ntree = 100),
    error = function(e) NULL
)

Y_recon_iso <- inverse_transform(qt_y$X_q, inv_iso)
recon_metrics_iso <- sapply(seq_len(ncol(syn$Y)), function(j) {
    c(mae = mean(abs(syn$Y[, j] - Y_recon_iso[, j])),
      pearson = cor(syn$Y[, j], Y_recon_iso[, j]),
      mono_viol = monotonicity_violation(Y_recon_iso[, j], syn$Y[, j]))
})

cat("Isotonic inverse reconstruction:\n")
cat(sprintf("  Mean MAE:     %.6f\n", mean(recon_metrics_iso["mae", ])))
cat(sprintf("  Mean Pearson: %.6f\n", mean(recon_metrics_iso["pearson", ])))
cat(sprintf("  Mean Mono Violation: %.6f\n", mean(recon_metrics_iso["mono_viol", ])))

if (!is.null(inv_rf)) {
    Y_recon_rf <- inverse_transform(qt_y$X_q, inv_rf)
    recon_metrics_rf <- sapply(seq_len(ncol(syn$Y)), function(j) {
        c(mae = mean(abs(syn$Y[, j] - Y_recon_rf[, j])),
          pearson = cor(syn$Y[, j], Y_recon_rf[, j]),
          mono_viol = monotonicity_violation(Y_recon_rf[, j], syn$Y[, j]))
    })

    cat("\nRandom Forest inverse reconstruction:\n")
    cat(sprintf("  Mean MAE:     %.6f\n", mean(recon_metrics_rf["mae", ])))
    cat(sprintf("  Mean Pearson: %.6f\n", mean(recon_metrics_rf["pearson", ])))
    cat(sprintf("  Mean Mono Violation: %.6f\n", mean(recon_metrics_rf["mono_viol", ])))

    # Save comparison
    comparison <- data.frame(
        method = c("isotonic", "rf"),
        mean_mae = c(mean(recon_metrics_iso["mae", ]),
                     mean(recon_metrics_rf["mae", ])),
        mean_pearson = c(mean(recon_metrics_iso["pearson", ]),
                         mean(recon_metrics_rf["pearson", ])),
        mean_mono_violation = c(mean(recon_metrics_iso["mono_viol", ]),
                                mean(recon_metrics_rf["mono_viol", ]))
    )
    print(comparison)
    write.csv(comparison, "output/benchmark/inverse_comparison.csv",
              row.names = FALSE)
}

cat("\nBenchmark complete. Results in output/benchmark/\n")
