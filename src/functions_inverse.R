# functions_inverse.R
# Inverse mapping: quantile values -> original scale

#' Learn isotonic inverse mapping for each feature
#'
#' For each column, fits isotonic regression: quantile -> original value.
#' This ensures monotonicity of the inverse mapping.
#'
#' @param fits list of per-column (value, quantile) data frames from quantile_transform
#' @return list of per-column isoreg objects
learn_inverse_isotonic <- function(fits) {
    inv_models <- vector("list", length(fits))
    names(inv_models) <- names(fits)

    for (j in seq_along(fits)) {
        df <- fits[[j]]
        # isotonic regression: quantile -> value (should be monotone increasing)
        iso <- isoreg(df$quantile, df$value)
        inv_models[[j]] <- list(
            type = "isotonic",
            model = iso,
            q_range = range(df$quantile),
            v_range = range(df$value)
        )
    }

    cat(sprintf("[learn_inverse_isotonic] Fitted %d features\n", length(fits)))
    inv_models
}


#' Learn random forest inverse mapping for each feature
#'
#' For each column, fits random forest: quantile -> original value.
#'
#' @param fits list of per-column (value, quantile) data frames from quantile_transform
#' @param ntree number of trees (default 100)
#' @return list of per-column randomForest objects
learn_inverse_rf <- function(fits, ntree = 100) {
    if (!requireNamespace("randomForest", quietly = TRUE)) {
        stop("Package 'randomForest' is required. Install with: install.packages('randomForest')")
    }

    inv_models <- vector("list", length(fits))
    names(inv_models) <- names(fits)

    for (j in seq_along(fits)) {
        df <- fits[[j]]
        rf <- randomForest::randomForest(
            x = data.frame(quantile = df$quantile),
            y = df$value,
            ntree = ntree
        )
        inv_models[[j]] <- list(
            type = "rf",
            model = rf,
            q_range = range(df$quantile),
            v_range = range(df$value)
        )
    }

    cat(sprintf("[learn_inverse_rf] Fitted %d features (ntree=%d)\n",
                length(fits), ntree))
    inv_models
}


#' Apply inverse mapping to quantile-scale matrix
#'
#' @param Q_matrix numeric matrix of quantile values (n_samples x n_features)
#' @param inv_models list of inverse models from learn_inverse_*
#' @return matrix in original value scale
inverse_transform <- function(Q_matrix, inv_models) {
    stopifnot(is.matrix(Q_matrix) || is.data.frame(Q_matrix))
    Q_matrix <- as.matrix(Q_matrix)
    n <- nrow(Q_matrix)
    p <- ncol(Q_matrix)
    stopifnot(p == length(inv_models))

    X_inv <- matrix(NA, nrow = n, ncol = p)
    colnames(X_inv) <- names(inv_models)

    for (j in seq_len(p)) {
        mod <- inv_models[[j]]
        q_vals <- Q_matrix[, j]

        # Clamp to training range
        q_vals <- pmax(q_vals, mod$q_range[1])
        q_vals <- pmin(q_vals, mod$q_range[2])

        if (mod$type == "isotonic") {
            # Use stepfun from isoreg for prediction
            sf <- as.stepfun(mod$model)
            X_inv[, j] <- sf(q_vals)
        } else if (mod$type == "rf") {
            X_inv[, j] <- predict(mod$model,
                newdata = data.frame(quantile = q_vals))
        } else {
            stop(paste("Unknown inverse model type:", mod$type))
        }
    }

    cat(sprintf("[inverse_transform] Mapped %d x %d quantile values to original scale\n",
                n, p))
    cat(sprintf("  Output range: [%.4f, %.4f]\n", min(X_inv, na.rm = TRUE),
                max(X_inv, na.rm = TRUE)))

    X_inv
}
