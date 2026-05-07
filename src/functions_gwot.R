# functions_gwot.R
# R wrapper around src/run_gwot.py (POT / EnsembleOT GW-OT warping).

#' Warp source features to target locations via GW-OT.
#'
#' Writes X, Y as temporary CSVs, invokes src/run_gwot.py, reads back the
#' warped matrix of shape (n_target, p_source).
#'
#' @param X source feature matrix (n_source x p_source)
#' @param Y target feature matrix (n_target x p_target)
#' @param backend "pot" | "pot_entropic" | "ensembleot"
#' @param python path to python interpreter (default: Sys.getenv)
#' @param n_clusters cluster count per domain (ensembleot only)
#' @param n_runs ensemble size (ensembleot only)
#' @param epsilon entropic regularization
#' @param outdir optional directory to persist intermediates
#' @return list(warped_exp, elapsed_sec, backend)
gwot_warping <- function(X, Y, backend = "ensembleot",
                         python = NULL,
                         n_clusters = 20, n_runs = 5,
                         clustering_method = "kmeans",
                         epsilon = 0.05,
                         random_state = 0,
                         A_source = NULL, A_target = NULL,
                         alpha = 0.5,
                         coords_source = NULL, coords_target = NULL,
                         outdir = NULL) {
    stopifnot(backend %in% c("pot", "pot_entropic", "ensembleot",
                              "pot_fgw_entropic", "ensembleot_fgw"))
    is_fgw <- grepl("fgw", backend)
    if (is_fgw && (is.null(A_source) || is.null(A_target))) {
        stop(sprintf("backend=%s requires A_source and A_target", backend))
    }
    if (is.null(python)) {
        python <- Sys.getenv("GWOT_PYTHON", unset = "python3")
    }

    X <- as.matrix(X)
    Y <- as.matrix(Y)
    if (is.null(colnames(X))) {
        colnames(X) <- paste0("src_", seq_len(ncol(X)))
    }
    if (is.null(colnames(Y))) {
        colnames(Y) <- paste0("tgt_", seq_len(ncol(Y)))
    }

    workdir <- if (is.null(outdir)) tempfile("gwot_") else outdir
    dir.create(workdir, recursive = TRUE, showWarnings = FALSE)
    src_csv <- file.path(workdir, "source.csv")
    tgt_csv <- file.path(workdir, "target.csv")
    out_csv <- file.path(workdir, sprintf("warped_%s.csv", backend))
    timing_json <- file.path(workdir, sprintf("timing_%s.json", backend))

    write.csv(X, src_csv, row.names = FALSE)
    write.csv(Y, tgt_csv, row.names = FALSE)

    extra_args <- character(0)
    if (!is.null(coords_source) && !is.null(coords_target)) {
        sc_csv <- file.path(workdir, "source_coords.csv")
        tc_csv <- file.path(workdir, "target_coords.csv")
        write.csv(as.matrix(coords_source), sc_csv, row.names = FALSE)
        write.csv(as.matrix(coords_target), tc_csv, row.names = FALSE)
        extra_args <- c(extra_args,
                        "--source-coords", sc_csv,
                        "--target-coords", tc_csv)
    }
    if (is_fgw) {
        ax_csv <- file.path(workdir, "source_anatomy.csv")
        ay_csv <- file.path(workdir, "target_anatomy.csv")
        write.csv(as.matrix(A_source), ax_csv, row.names = FALSE)
        write.csv(as.matrix(A_target), ay_csv, row.names = FALSE)
        extra_args <- c("--source-anatomy", ax_csv,
                        "--target-anatomy", ay_csv,
                        "--alpha", as.character(alpha))
    }

    script <- normalizePath("src/run_gwot.py", mustWork = TRUE)
    args <- c(
        script,
        "--source", src_csv,
        "--target", tgt_csv,
        "--out", out_csv,
        "--backend", backend,
        "--n-clusters", as.character(n_clusters),
        "--n-runs", as.character(n_runs),
        "--clustering-method", clustering_method,
        "--epsilon", as.character(epsilon),
        "--random-state", as.character(random_state),
        "--timing", timing_json,
        extra_args
    )
    cat(sprintf("[gwot_warping] backend=%s, calling: %s\n", backend, python))
    status <- system2(python, args, stdout = TRUE, stderr = TRUE)
    if (!is.null(attr(status, "status")) && attr(status, "status") != 0) {
        cat(paste(status, collapse = "\n"), "\n")
        stop(sprintf("gwot_warping failed (backend=%s)", backend))
    }

    warped <- as.matrix(read.csv(out_csv, check.names = FALSE))
    # Free all intermediate CSVs to avoid filling disk.
    try(file.remove(src_csv, tgt_csv, out_csv), silent = TRUE)
    if (is_fgw) try(file.remove(ax_csv, ay_csv), silent = TRUE)
    if (!is.null(coords_source)) try(file.remove(sc_csv, tc_csv), silent = TRUE)
    elapsed <- NA_real_
    peak_rss_mb <- NA_real_
    if (file.exists(timing_json)) {
        tj <- tryCatch(
            jsonlite::fromJSON(timing_json),
            error = function(e) NULL
        )
        if (!is.null(tj)) {
            elapsed <- tj$elapsed_sec
            if (!is.null(tj$peak_rss_mb)) peak_rss_mb <- tj$peak_rss_mb
        }
    }
    cat(sprintf("[gwot_warping] done: warped %d x %d, elapsed=%.3fs, RSS=%.0fMB\n",
                nrow(warped), ncol(warped),
                ifelse(is.na(elapsed), -1, elapsed),
                ifelse(is.na(peak_rss_mb), -1, peak_rss_mb)))

    list(warped_exp = warped, elapsed_sec = elapsed,
         peak_rss_mb = peak_rss_mb, backend = backend)
}


#' GW-OT pipeline (no quantile): raw X, Y -> warped source features.
run_pipeline_gwot_raw <- function(X, Y, backend,
                                  n_clusters = 20, n_runs = 5,
                                  clustering_method = "kmeans",
                                  epsilon = 0.05, outdir = NULL,
                                  A_source = NULL, A_target = NULL,
                                  alpha = 0.5,
                                  coords_source = NULL, coords_target = NULL,
                                  ...) {
    gwot_warping(
        X, Y, backend = backend,
        n_clusters = n_clusters, n_runs = n_runs,
        clustering_method = clustering_method,
        epsilon = epsilon, outdir = outdir,
        A_source = A_source, A_target = A_target, alpha = alpha,
        coords_source = coords_source, coords_target = coords_target, ...
    )
}


#' Quantile + GW-OT + inverse pipeline (mirrors run_pipeline() for routes D/F).
run_pipeline_gwot_quantile <- function(X, Y, backend,
                                       inverse_method = "isotonic",
                                       n_clusters = 20, n_runs = 5,
                                       clustering_method = "kmeans",
                                       epsilon = 0.05, outdir = NULL,
                                       A_source = NULL, A_target = NULL,
                                       alpha = 0.5,
                                       coords_source = NULL, coords_target = NULL,
                                       ...) {
    qt_source <- quantile_transform(X)
    qt_target <- quantile_transform(Y)

    if (inverse_method == "isotonic") {
        inv_models <- learn_inverse_isotonic(qt_target$fits)
    } else if (inverse_method == "rf") {
        inv_models <- learn_inverse_rf(qt_target$fits)
    } else {
        stop(paste("Unknown inverse_method:", inverse_method))
    }

    gw <- gwot_warping(
        qt_source$X_q, qt_target$X_q, backend = backend,
        n_clusters = n_clusters, n_runs = n_runs,
        clustering_method = clustering_method,
        epsilon = epsilon, outdir = outdir,
        A_source = A_source, A_target = A_target, alpha = alpha,
        coords_source = coords_source, coords_target = coords_target, ...
    )
    warped_q <- gw$warped_exp

    shared <- intersect(colnames(warped_q), names(inv_models))
    warped_original <- NULL
    if (length(shared) > 0) {
        warped_original <- inverse_transform(
            warped_q[, shared, drop = FALSE],
            inv_models[shared]
        )
    }

    list(
        warped_quantile = warped_q,
        warped_original_scale = warped_original,
        elapsed_sec = gw$elapsed_sec,
        backend = backend,
        qt_source = qt_source,
        qt_target = qt_target
    )
}
