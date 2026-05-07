# functions_quantile.R
# Feature-wise quantile transform and its inverse preparation

#' Feature-wise quantile transform
#'
#' Each column is mapped to [0,1] via empirical CDF (average rank for ties).
#' Returns transformed matrix and per-column fit info for inverse mapping.
#'
#' @param X numeric matrix (n_samples x n_features)
#' @return list with:
#'   - X_q: quantile-transformed matrix (same shape)
#'   - fits: list of per-column data frames (value, quantile) sorted by value
quantile_transform <- function(X) {
    stopifnot(is.matrix(X) || is.data.frame(X))
    X <- as.matrix(X)
    n <- nrow(X)
    p <- ncol(X)

    X_q <- matrix(NA, nrow = n, ncol = p)
    colnames(X_q) <- colnames(X)
    fits <- vector("list", p)
    names(fits) <- colnames(X)

    for (j in seq_len(p)) {
        x <- X[, j]
        # Average rank, then scale to [0,1]
        r <- rank(x, ties.method = "average")
        q <- (r - 0.5) / n  # Hazen plotting position

        X_q[, j] <- q

        # Store unique (value, quantile) pairs for inverse mapping
        df <- data.frame(value = x, quantile = q)
        df <- df[order(df$value), ]
        # Deduplicate: average quantile for identical values
        df <- aggregate(quantile ~ value, data = df, FUN = mean)
        df <- df[order(df$value), ]
        fits[[j]] <- df
    }

    cat(sprintf("[quantile_transform] Input: %d x %d -> Output: %d x %d\n",
                n, p, nrow(X_q), ncol(X_q)))
    cat(sprintf("  Quantile range: [%.4f, %.4f]\n", min(X_q), max(X_q)))

    list(X_q = X_q, fits = fits)
}


#' Apply a previously fitted quantile transform to new data
#'
#' Maps new values through the stored empirical CDF via linear interpolation.
#'
#' @param X_new numeric matrix (n_samples x n_features)
#' @param fits list of per-column (value, quantile) data frames from quantile_transform
#' @return quantile-transformed matrix
quantile_transform_apply <- function(X_new, fits) {
    stopifnot(is.matrix(X_new) || is.data.frame(X_new))
    X_new <- as.matrix(X_new)
    p <- ncol(X_new)
    stopifnot(p == length(fits))

    X_q <- matrix(NA, nrow = nrow(X_new), ncol = p)
    colnames(X_q) <- colnames(X_new)

    for (j in seq_len(p)) {
        fit <- fits[[j]]
        # Linear interpolation, clamp to [0,1]
        X_q[, j] <- approx(
            x = fit$value, y = fit$quantile,
            xout = X_new[, j],
            rule = 2  # clamp outside range
        )$y
    }

    X_q
}
