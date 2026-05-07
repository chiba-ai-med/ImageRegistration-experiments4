# run_realdata_comparison.R
# Compare Quantile-GuidedPLS (experiments4) vs plain GuidedPLS (experiments3)
# on real 251208 and kidney datasets

set.seed(42)

source("src/functions_pipeline.R")

library("ggplot2")

EXP3_DIR <- "../ImageRegistration-experiments3"

# Marker definitions (same as experiments3)
get_markers <- function(dataset) {
    if (dataset == "251208") {
        list(
            source = c(
                "HexCer.36.1.O2.1", "HexCer.36.2.O2", "HexCer.38.2.O2",
                "HexCer.40.0.O2", "HexCer.40.1.O2", "HexCer.40.2.O2",
                "HexCer.41.1.O2", "HexCer.42.1.O2", "HexCer.42.2.O2",
                "HexCer.44.2.O2",
                "SM.31.1.O2", "SM.34.1.O2", "SM.35.1.O2",
                "SM.36.1.O2", "SM.36.2.O2", "SM.38.1.O2",
                "SM.41.2.O2", "SM.42.1.O2", "SM.42.2.O2", "SM.42.3.O2"),
            target = c("Mog", "Sox10"))
    } else if (dataset == "kidney") {
        list(
            source = c("FA.22.6"),
            target = c("Slc27a2"))
    }
}

# Correlation evaluation
cor_combination <- function(warped_exp, target_all_exp,
    source_markers, target_markers) {
    results <- data.frame()
    for (s in source_markers) {
        for (t in target_markers) {
            if (!(s %in% colnames(warped_exp))) next
            if (!(t %in% colnames(target_all_exp))) next
            cc <- cor(warped_exp[, s], target_all_exp[, t], method = "pearson")
            results <- rbind(results, data.frame(
                source = s, target = t, pearson = cc,
                stringsAsFactors = FALSE))
        }
    }
    results
}

all_comparison <- data.frame()

for (dataset in c("251208", "kidney")) {
    cat(sprintf("\n########################################\n"))
    cat(sprintf("  Dataset: %s\n", dataset))
    cat(sprintf("########################################\n\n"))

    # Load data
    data_dir <- file.path(EXP3_DIR, "data", dataset)
    X <- as.matrix(read.csv(file.path(data_dir, "source", "all_exp.csv"), header = TRUE))
    Y <- as.matrix(read.csv(file.path(data_dir, "target", "all_exp.csv"), header = TRUE))
    A_source <- as.matrix(read.csv(file.path(data_dir, "source", "anatomy.csv"), header = TRUE))
    A_target <- as.matrix(read.csv(file.path(data_dir, "target", "anatomy.csv"), header = TRUE))

    cat(sprintf("  Source: %d x %d\n", nrow(X), ncol(X)))
    cat(sprintf("  Target: %d x %d\n", nrow(Y), ncol(Y)))
    cat(sprintf("  Anatomy regions: %d\n", ncol(A_source)))

    markers <- get_markers(dataset)
    outdir <- file.path("output", dataset)

    # --- Quantile-GuidedPLS ---
    cat("\n>>> Quantile-GuidedPLS:\n")
    result_qt <- run_pipeline(
        X = X, Y = Y,
        A_source = A_source, A_target = A_target,
        r = 18, k = 11,
        inverse_method = "isotonic",
        outdir = file.path(outdir, "quantile_guidedpls")
    )

    # Evaluate quantile pipeline (raw warped values)
    cc_qt <- cor_combination(result_qt$warped_raw, Y,
        markers$source, markers$target)
    cc_qt$method <- "quantile_guidedpls"

    # --- Plain GuidedPLS (same params, no quantile transform) ---
    cat("\n>>> Plain GuidedPLS:\n")
    gpls_plain <- run_guidedpls(X, Y, A_source, A_target)
    warp_plain <- knn_warping(gpls_plain, X, A_source, r = 18, k = 11)

    # Save plain results too
    dir.create(file.path(outdir, "plain_guidedpls"), recursive = TRUE, showWarnings = FALSE)
    write.csv(warp_plain$warped_exp, file.path(outdir, "plain_guidedpls", "warped.txt"),
              row.names = FALSE)

    cc_plain <- cor_combination(warp_plain$warped_exp, Y,
        markers$source, markers$target)
    cc_plain$method <- "plain_guidedpls"

    # --- Load experiments3 result for reference ---
    exp3_warped_file <- file.path(EXP3_DIR, "output", dataset, "guidedpls", "warped.txt")
    if (file.exists(exp3_warped_file)) {
        exp3_warped <- read.csv(exp3_warped_file, header = TRUE)
        cc_exp3 <- cor_combination(exp3_warped, Y,
            markers$source, markers$target)
        cc_exp3$method <- "experiments3_guidedpls"
    } else {
        cc_exp3 <- data.frame()
    }

    # Combine
    cc_all <- rbind(cc_qt, cc_plain, cc_exp3)
    cc_all$dataset <- dataset
    all_comparison <- rbind(all_comparison, cc_all)

    # Print comparison table
    cat(sprintf("\n--- %s: Comparison ---\n", dataset))
    if (nrow(cc_exp3) > 0) {
        merged <- merge(
            cc_qt[, c("source", "target", "pearson")],
            cc_plain[, c("source", "target", "pearson")],
            by = c("source", "target"), suffixes = c("_qt", "_plain"))
        merged <- merge(merged,
            cc_exp3[, c("source", "target", "pearson")],
            by = c("source", "target"))
        names(merged)[5] <- "pearson_exp3"
        merged$diff_qt_vs_plain <- merged$pearson_qt - merged$pearson_plain
        print(merged)
        cat(sprintf("\nMean Pearson: qt=%.4f, plain=%.4f, exp3=%.4f\n",
            mean(merged$pearson_qt, na.rm = TRUE),
            mean(merged$pearson_plain, na.rm = TRUE),
            mean(merged$pearson_exp3, na.rm = TRUE)))
        cat(sprintf("Mean diff (qt - plain): %.4f\n",
            mean(merged$diff_qt_vs_plain, na.rm = TRUE)))
    }
}

# --- Save results ---
dir.create("output", showWarnings = FALSE)
write.csv(all_comparison, "output/realdata_comparison.csv", row.names = FALSE)

# --- Summary plot ---
dir.create("plot", showWarnings = FALSE)

g <- ggplot(all_comparison, aes(x = method, y = pearson, fill = method)) +
    geom_boxplot(alpha = 0.7) +
    facet_wrap(~dataset, scales = "free") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    labs(title = "Quantile-GuidedPLS vs Plain GuidedPLS",
         x = "Method", y = "Pearson CC") +
    scale_fill_manual(values = c(
        "quantile_guidedpls" = "#E41A1C",
        "plain_guidedpls" = "#377EB8",
        "experiments3_guidedpls" = "#4DAF4A"))

ggsave("plot/realdata_comparison.png", g, width = 10, height = 6, dpi = 150)
cat("\nPlot saved to plot/realdata_comparison.png\n")
cat("Results saved to output/realdata_comparison.csv\n")
