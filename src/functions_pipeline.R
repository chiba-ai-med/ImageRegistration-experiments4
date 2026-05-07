# functions_pipeline.R
# End-to-end pipeline: quantile transform -> GuidedPLS -> kNNWarping -> inverse mapping

library("guidedPLS")
library("RANN")

source("src/functions_quantile.R")
source("src/functions_inverse.R")

#' GuidedPLS wrapper on quantile-transformed data
#'
#' @param X_q quantile-transformed source expression (n_source x p_source)
#' @param Y_q quantile-transformed target expression (n_target x p_target)
#' @param A_source source anatomy one-hot matrix (n_source x n_regions)
#' @param A_target target anatomy one-hot matrix (n_target x n_regions)
#' @param cortest logical, perform correlation test (default TRUE)
#' @param verbose logical (default TRUE)
#' @return guidedPLS output object
run_guidedpls <- function(X_q, Y_q, A_source, A_target,
                          cortest = TRUE, verbose = TRUE) {
    cat("[run_guidedpls] Running GuidedPLS...\n")
    cat(sprintf("  Source: %d x %d, Target: %d x %d\n",
                nrow(X_q), ncol(X_q), nrow(Y_q), ncol(Y_q)))
    cat(sprintf("  Anatomy: %d regions\n", ncol(A_source)))

    out <- guidedPLS(
        X1 = as.matrix(X_q),
        X2 = as.matrix(Y_q),
        Y1 = as.matrix(A_source),
        Y2 = as.matrix(A_target),
        cortest = cortest,
        verbose = verbose
    )

    cat("[run_guidedpls] Done.\n")
    cat(sprintf("  Score dimensions: X1=%d, X2=%d\n",
                ncol(out$scoreX1), ncol(out$scoreX2)))
    out
}


#' kNN warping in latent space
#'
#' @param out guidedPLS output
#' @param source_exp source expression matrix (original or quantile scale)
#' @param source_anatomy source anatomy one-hot matrix
#' @param r number of PLS dimensions to use
#' @param k number of nearest neighbors
#' @return list with warped_exp and warped_anatomy
knn_warping <- function(out, source_exp, source_anatomy, r = 18, k = 11) {
    scoreX1 <- apply(out$scoreX1, 2, scale)
    scoreX2 <- apply(out$scoreX2, 2, scale)
    r <- min(r, ncol(scoreX1), ncol(scoreX2))

    cat(sprintf("[knn_warping] Using r=%d dims, k=%d neighbors\n", r, k))

    nn <- nn2(
        data  = scoreX1[, seq(r)],
        query = scoreX2[, seq(r)],
        k = k
    )
    idx <- nn$nn.idx[, 1:k]

    warped_exp <- t(apply(idx, 1, function(idxs) {
        colMeans(source_exp[idxs, , drop = FALSE])
    }))

    warped_anatomy <- apply(idx, 1, function(idxs) {
        tmp <- colMeans(source_anatomy[idxs, , drop = FALSE])
        names(tmp)[which.max(tmp)]
    })

    cat(sprintf("[knn_warping] Warped expression: %d x %d\n",
                nrow(warped_exp), ncol(warped_exp)))

    list(warped_exp = warped_exp, warped_anatomy = warped_anatomy)
}


#' End-to-end quantile-guided PLS pipeline
#'
#' @param X source expression matrix (n_source x p_source)
#' @param Y target expression matrix (n_target x p_target)
#' @param A_source source anatomy one-hot (n_source x n_regions)
#' @param A_target target anatomy one-hot (n_target x n_regions)
#' @param r PLS dimensions for kNN warping
#' @param k neighbors for kNN warping
#' @param inverse_method "isotonic" or "rf"
#' @param outdir directory to save intermediate results (NULL = no save)
#' @return list with all intermediate and final results
run_pipeline <- function(X, Y, A_source, A_target,
                         r = 18, k = 11,
                         inverse_method = "isotonic",
                         outdir = NULL) {

    cat("=== Quantile-GuidedPLS Pipeline ===\n\n")

    # --- Step 1: Quantile transform ---
    cat("--- Step 1: Quantile Transform ---\n")
    qt_source <- quantile_transform(X)
    qt_target <- quantile_transform(Y)

    if (!is.null(outdir)) {
        dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
        saveRDS(qt_source, file.path(outdir, "qt_source.rds"))
        saveRDS(qt_target, file.path(outdir, "qt_target.rds"))
    }

    # --- Step 2: Learn inverse mapping on target side ---
    cat("\n--- Step 2: Learn Inverse Mapping (target) ---\n")
    if (inverse_method == "isotonic") {
        inv_models <- learn_inverse_isotonic(qt_target$fits)
    } else if (inverse_method == "rf") {
        inv_models <- learn_inverse_rf(qt_target$fits)
    } else {
        stop(paste("Unknown inverse_method:", inverse_method))
    }

    if (!is.null(outdir)) {
        saveRDS(inv_models, file.path(outdir, "inv_models.rds"))
    }

    # --- Step 3: GuidedPLS on quantile-transformed data ---
    cat("\n--- Step 3: GuidedPLS ---\n")
    gpls_out <- run_guidedpls(
        qt_source$X_q, qt_target$X_q,
        A_source, A_target
    )

    if (!is.null(outdir)) {
        saveRDS(gpls_out, file.path(outdir, "gpls_out.rds"))
    }

    # --- Step 4: kNN warping in quantile space ---
    cat("\n--- Step 4: kNN Warping (quantile space) ---\n")
    # Warp source quantile values to target locations
    warp_out <- knn_warping(
        gpls_out, qt_source$X_q, as.matrix(A_source),
        r = r, k = k
    )
    warped_q <- warp_out$warped_exp  # n_target x p_source (quantile scale)

    if (!is.null(outdir)) {
        write.csv(warped_q, file.path(outdir, "warped_quantile.csv"),
                  row.names = FALSE)
    }

    # --- Step 5: Inverse transform to target's original scale ---
    # Note: warped_q has source features, but we want target-scale values.
    # The inverse mapping is learned from target features.
    # This step is meaningful when source and target share the same features,
    # or when we want to map a subset of features.
    #
    # For cross-modal case (different features), we also provide:
    # - warped_q: the quantile-scale warped values (source features at target locations)
    # - For evaluation, compare warped source markers with target markers directly

    # If features match between source and target, apply inverse transform
    shared_features <- intersect(colnames(warped_q), names(inv_models))
    if (length(shared_features) > 0) {
        cat(sprintf("\n--- Step 5: Inverse Transform (%d shared features) ---\n",
                    length(shared_features)))
        warped_q_shared <- warped_q[, shared_features, drop = FALSE]
        inv_shared <- inv_models[shared_features]
        warped_original <- inverse_transform(warped_q_shared, inv_shared)
    } else {
        cat("\n--- Step 5: Skipped (no shared features, cross-modal) ---\n")
        warped_original <- NULL
    }

    # Also provide raw (non-quantile) warping for comparison
    cat("\n--- Step 4b: kNN Warping (original space) ---\n")
    warp_raw <- knn_warping(
        gpls_out, X, as.matrix(A_source),
        r = r, k = k
    )

    if (!is.null(outdir)) {
        write.csv(warp_raw$warped_exp, file.path(outdir, "warped_original.csv"),
                  row.names = FALSE)
        if (!is.null(warped_original)) {
            write.csv(warped_original, file.path(outdir, "warped_inv_transform.csv"),
                      row.names = FALSE)
        }
        save(gpls_out, warp_out, warp_raw,
             file = file.path(outdir, "pipeline.RData"))
    }

    cat("\n=== Pipeline Complete ===\n")

    list(
        qt_source = qt_source,
        qt_target = qt_target,
        inv_models = inv_models,
        gpls_out = gpls_out,
        warped_quantile = warped_q,
        warped_original_scale = warped_original,
        warped_raw = warp_raw$warped_exp,
        warped_anatomy = warp_out$warped_anatomy
    )
}


#' Compute evaluation metrics
#'
#' @param predicted numeric vector
#' @param actual numeric vector
#' @return named list of metrics
compute_metrics <- function(predicted, actual) {
    list(
        rmse = sqrt(mean((predicted - actual)^2)),
        mae = mean(abs(predicted - actual)),
        pearson = cor(predicted, actual, method = "pearson"),
        spearman = cor(predicted, actual, method = "spearman")
    )
}


#' Evaluate warped expression against target
#'
#' @param warped_exp matrix (n_target x n_features)
#' @param target_exp matrix (n_target x n_features)
#' @param source_markers character vector of source marker names
#' @param target_markers character vector of target marker names
#' @return data.frame with metrics per marker pair
evaluate_warping <- function(warped_exp, target_exp,
                             source_markers, target_markers) {
    results <- data.frame()

    for (s in source_markers) {
        for (t in target_markers) {
            if (!(s %in% colnames(warped_exp))) next
            if (!(t %in% colnames(target_exp))) next

            m <- compute_metrics(warped_exp[, s], target_exp[, t])
            results <- rbind(results, data.frame(
                source_marker = s,
                target_marker = t,
                rmse = m$rmse,
                mae = m$mae,
                pearson = m$pearson,
                spearman = m$spearman,
                stringsAsFactors = FALSE
            ))
        }
    }

    # Order preservation: fraction of pairs where rank order is maintained
    cat(sprintf("[evaluate_warping] %d marker pairs evaluated\n", nrow(results)))
    cat(sprintf("  Mean Pearson: %.4f, Mean Spearman: %.4f\n",
                mean(results$pearson, na.rm = TRUE),
                mean(results$spearman, na.rm = TRUE)))

    results
}
