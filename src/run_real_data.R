# run_real_data.R
# Real-data comparison (kidney, 251208) across routes A-F.
#
# A: GuidedPLS + kNN (raw)
# B: Quantile + GuidedPLS + kNN (+ inverse if possible)
# C: POT GW-OT (raw)                       -- OOM skip on large n
# D: Quantile + POT GW-OT (+ inverse)
# E: EnsembleOT GW-OT (raw)
# F: Quantile + EnsembleOT GW-OT (+ inverse)
#
# Evaluation: marker-pair Pearson correlation between warped source marker
# values (at target locations) and the corresponding target marker.

source("src/functions_pipeline.R")
source("src/functions_gwot.R")
library("ggplot2")
library("jsonlite")

if (Sys.getenv("GWOT_PYTHON", unset = "") == "") {
    py <- file.path(getwd(), ".venv_ot", "bin", "python")
    if (file.exists(py)) Sys.setenv(GWOT_PYTHON = py)
}

# --- Dataset loaders ------------------------------------------------------
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
    cat(sprintf("[load] source: %s\n", info$source_csv))
    src <- read.csv(info$source_csv, check.names = TRUE)
    cat(sprintf("       target: %s\n", info$target_csv))
    tgt <- read.csv(info$target_csv, check.names = TRUE)

    # Drop coords and region; use numeric feature columns.
    drop_cols <- c("X", "Y", info$region_col)
    src_feat <- as.matrix(src[, !(colnames(src) %in% drop_cols)])
    tgt_feat <- as.matrix(tgt[, !(colnames(tgt) %in% drop_cols)])

    # One-hot anatomy on the union of regions present in both sides.
    src_region <- as.character(src[[info$region_col]])
    tgt_region <- as.character(tgt[[info$region_col]])
    # Drop NAs from region vector before building the union.
    regions <- sort(unique(c(
        src_region[!is.na(src_region)],
        tgt_region[!is.na(tgt_region)]
    )))
    onehot <- function(v) {
        v[is.na(v)] <- regions[1]  # map NA to first region bucket
        idx <- match(v, regions)
        M <- matrix(0, length(v), length(regions))
        colnames(M) <- paste0("region_", seq_along(regions))
        M[cbind(seq_along(v), idx)] <- 1
        M
    }
    A_src <- onehot(src_region)
    A_tgt <- onehot(tgt_region)

    cat(sprintf("       X=%d x %d, Y=%d x %d, %d regions\n",
                nrow(src_feat), ncol(src_feat),
                nrow(tgt_feat), ncol(tgt_feat), length(regions)))

    list(X = src_feat, Y = tgt_feat, A_source = A_src, A_target = A_tgt,
         target_all = tgt_feat)
}

# --- Marker-pair Pearson evaluation ---------------------------------------
eval_markers <- function(warped_exp, target_all,
                         source_markers, target_markers) {
    rows <- list()
    for (s in source_markers) {
        if (!(s %in% colnames(warped_exp))) next
        for (t in target_markers) {
            if (!(t %in% colnames(target_all))) next
            r <- tryCatch(
                cor(warped_exp[, s], target_all[, t], method = "pearson"),
                error = function(e) NA_real_
            )
            rows[[length(rows) + 1]] <- data.frame(
                source_marker = s, target_marker = t, pearson = r
            )
        }
    }
    if (length(rows) == 0) return(data.frame())
    do.call(rbind, rows)
}

# --- Route runners --------------------------------------------------------
route_guidedpls <- function(data, quantile, r = 18, k = 11) {
    t0 <- Sys.time()
    if (quantile) {
        out <- run_pipeline(
            X = data$X, Y = data$Y,
            A_source = data$A_source, A_target = data$A_target,
            r = r, k = k, inverse_method = "isotonic",
            outdir = NULL
        )
        warped <- out$warped_raw  # source-space warped, n_target x p_source
    } else {
        gp <- run_guidedpls(data$X, data$Y, data$A_source, data$A_target)
        w  <- knn_warping(gp, data$X, data$A_source, r = r, k = k)
        warped <- w$warped_exp
    }
    elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
    list(warped_exp = warped, elapsed_sec = elapsed)
}

route_gwot <- function(data, backend, quantile,
                       n_clusters = 50, n_runs = 5, epsilon = 0.05,
                       alpha = 0.5, clustering_method = "kmeans",
                       outdir = NULL) {
    is_fgw <- grepl("fgw", backend)
    extra <- if (is_fgw) {
        list(A_source = data$A_source, A_target = data$A_target, alpha = alpha)
    } else list()
    if (quantile) {
        out <- do.call(run_pipeline_gwot_quantile, c(list(
            X = data$X, Y = data$Y, backend = backend,
            inverse_method = "isotonic",
            n_clusters = n_clusters, n_runs = n_runs,
            clustering_method = clustering_method,
            epsilon = epsilon, outdir = outdir
        ), extra))
        list(warped_exp = out$warped_quantile, elapsed_sec = out$elapsed_sec)
    } else {
        out <- do.call(run_pipeline_gwot_raw, c(list(
            X = data$X, Y = data$Y, backend = backend,
            n_clusters = n_clusters, n_runs = n_runs,
            clustering_method = clustering_method,
            epsilon = epsilon, outdir = outdir
        ), extra))
        list(warped_exp = out$warped_exp, elapsed_sec = out$elapsed_sec)
    }
}

# --- Main loop ------------------------------------------------------------
dir.create("output/real_data", recursive = TRUE, showWarnings = FALSE)
dir.create("plot/real_data",  recursive = TRUE, showWarnings = FALSE)

all_marker <- data.frame()
all_timing <- data.frame()

only <- Sys.getenv("REAL_DATA_ONLY", unset = "")
ds_names <- if (nzchar(only)) strsplit(only, ",")[[1]] else names(DATASETS)
for (ds_name in ds_names) {
    cat(sprintf("\n########################################\n"))
    cat(sprintf("  Dataset: %s\n", ds_name))
    cat(sprintf("########################################\n\n"))
    info <- DATASETS[[ds_name]]
    data <- load_dataset(info)

    # Scale n_runs, n_clusters based on size; keep POT as-is.
    n_x <- nrow(data$X); n_y <- nrow(data$Y)
    n_clusters <- min(50, n_x, n_y)

    outdir_ds <- file.path("output/real_data", ds_name)

    routes <- list(
        A_guidedpls_raw      = function() route_guidedpls(data, quantile = FALSE),
        B_guidedpls_quantile = function() route_guidedpls(data, quantile = TRUE),
        C_pot_raw            = function() route_gwot(data, "pot_entropic", FALSE,
                                                     n_clusters = n_clusters,
                                                     outdir = file.path(outdir_ds, "pot_raw")),
        D_pot_quantile       = function() route_gwot(data, "pot_entropic", TRUE,
                                                     n_clusters = n_clusters,
                                                     outdir = file.path(outdir_ds, "pot_quantile")),
        E_ensembleot_raw     = function() route_gwot(data, "ensembleot", FALSE,
                                                     n_clusters = n_clusters,
                                                     outdir = file.path(outdir_ds, "ensembleot_raw")),
        F_ensembleot_quant   = function() route_gwot(data, "ensembleot", TRUE,
                                                     n_clusters = n_clusters,
                                                     outdir = file.path(outdir_ds, "ensembleot_quantile")),
        G_pot_fgw_raw        = function() route_gwot(data, "pot_fgw_entropic", FALSE,
                                                     n_clusters = n_clusters,
                                                     outdir = file.path(outdir_ds, "pot_fgw_raw")),
        H_pot_fgw_quant      = function() route_gwot(data, "pot_fgw_entropic", TRUE,
                                                     n_clusters = n_clusters,
                                                     outdir = file.path(outdir_ds, "pot_fgw_quantile")),
        I_ensembleot_fgw_raw = function() route_gwot(data, "ensembleot_fgw", FALSE,
                                                     n_clusters = n_clusters,
                                                     outdir = file.path(outdir_ds, "ensembleot_fgw_raw")),
        J_ensembleot_fgw_quant = function() route_gwot(data, "ensembleot_fgw", TRUE,
                                                     n_clusters = n_clusters,
                                                     outdir = file.path(outdir_ds, "ensembleot_fgw_quantile")),
        K_ensembleot_gw_raw_rv = function() route_gwot(data, "ensembleot", FALSE,
                                                     n_clusters = n_clusters,
                                                     clustering_method = "random_voronoi",
                                                     outdir = file.path(outdir_ds, "ensembleot_gw_raw_rv")),
        L_ensembleot_gw_quant_rv = function() route_gwot(data, "ensembleot", TRUE,
                                                     n_clusters = n_clusters,
                                                     clustering_method = "random_voronoi",
                                                     outdir = file.path(outdir_ds, "ensembleot_gw_quantile_rv")),
        M_ensembleot_fgw_raw_rv = function() route_gwot(data, "ensembleot_fgw", FALSE,
                                                     n_clusters = n_clusters,
                                                     clustering_method = "random_voronoi",
                                                     outdir = file.path(outdir_ds, "ensembleot_fgw_raw_rv")),
        N_ensembleot_fgw_quant_rv = function() route_gwot(data, "ensembleot_fgw", TRUE,
                                                     n_clusters = n_clusters,
                                                     clustering_method = "random_voronoi",
                                                     outdir = file.path(outdir_ds, "ensembleot_fgw_quantile_rv"))
    )

    only_routes <- Sys.getenv("REAL_DATA_ROUTES", unset = "")
    if (nzchar(only_routes)) {
        keep <- strsplit(only_routes, ",")[[1]]
        routes <- routes[names(routes) %in% keep]
    }
    for (rn in names(routes)) {
        cat(sprintf("\n--- Route %s ---\n", rn))
        res <- tryCatch(routes[[rn]](),
                        error = function(e) {
                            cat(sprintf("  SKIP (%s): %s\n", rn, e$message))
                            NULL
                        })
        if (is.null(res)) {
            all_timing <- rbind(all_timing, data.frame(
                dataset = ds_name, route = rn,
                elapsed_sec = NA_real_, status = "SKIP",
                stringsAsFactors = FALSE))
            next
        }
        all_timing <- rbind(all_timing, data.frame(
            dataset = ds_name, route = rn,
            elapsed_sec = res$elapsed_sec, status = "OK",
            stringsAsFactors = FALSE))

        mk <- eval_markers(res$warped_exp, data$target_all,
                           info$source_markers, info$target_markers)
        if (nrow(mk) > 0) {
            mk$dataset <- ds_name
            mk$route   <- rn
            all_marker <- rbind(all_marker, mk)
            cat(sprintf("  mean |Pearson| = %.4f over %d marker pairs\n",
                        mean(abs(mk$pearson), na.rm = TRUE), nrow(mk)))
        }
    }
}

suffix <- Sys.getenv("REAL_DATA_SUFFIX", unset = "")
out_marker <- sprintf("output/real_data/marker_pairs%s.csv", suffix)
out_timing <- sprintf("output/real_data/timing%s.csv", suffix)
out_summary <- sprintf("output/real_data/summary%s.csv", suffix)
write.csv(all_marker, out_marker, row.names = FALSE)
write.csv(all_timing, out_timing, row.names = FALSE)

cat("\n\n=== Summary (mean |Pearson| per dataset × route) ===\n")
if (nrow(all_marker) > 0) {
    summ <- aggregate(abs(pearson) ~ dataset + route, data = all_marker, mean)
    colnames(summ)[3] <- "mean_abs_pearson"
    print(summ)
    write.csv(summ, out_summary, row.names = FALSE)
}
cat("\n=== Timing ===\n")
print(all_timing)

# --- Plot ----------------------------------------------------------------
if (nrow(all_marker) > 0) {
    g <- ggplot(all_marker, aes(x = route, y = abs(pearson), fill = route)) +
        geom_boxplot(alpha = 0.7) +
        geom_jitter(width = 0.1, alpha = 0.5, size = 0.8) +
        facet_wrap(~ dataset, scales = "free_x") +
        theme_minimal() +
        theme(axis.text.x = element_text(angle = 45, hjust = 1),
              legend.position = "none") +
        labs(title = "Real data: marker-pair correlation by route",
             x = "Route", y = "|Pearson|")
    ggsave("plot/real_data/marker_pairs.png", g, width = 12, height = 6, dpi = 150)
}
if (nrow(all_timing) > 0) {
    tplot <- all_timing[all_timing$status == "OK", ]
    if (nrow(tplot) > 0) {
        g2 <- ggplot(tplot, aes(x = route, y = elapsed_sec, fill = route)) +
            geom_bar(stat = "identity") +
            facet_wrap(~ dataset, scales = "free_y") +
            theme_minimal() +
            theme(axis.text.x = element_text(angle = 45, hjust = 1),
                  legend.position = "none") +
            labs(title = "Real data: wall-clock time",
                 x = "Route", y = "Elapsed (sec, log scale)") +
            scale_y_log10()
        ggsave("plot/real_data/timing.png", g2, width = 12, height = 6, dpi = 150)
    }
}

cat("\nDone. Outputs in output/real_data/, plots in plot/real_data/.\n")
