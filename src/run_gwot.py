"""GW-OT warping bridge: POT / EnsembleOT backends.

Reads source / target feature CSVs, solves a Gromov-Wasserstein coupling,
and writes a warped source-feature matrix of shape (n_target, p_source).

Usage:
    python run_gwot.py --source X.csv --target Y.csv --out warped.csv \
        --backend {pot,ensembleot} [--n-clusters 20] [--n-runs 5] \
        [--epsilon 0.05] [--timing timing.json]
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import ot
import ot.gromov


def _load(path: str) -> tuple[np.ndarray, list[str]]:
    import csv
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [[float(v) for v in row] for row in reader]
    return np.asarray(rows, dtype=np.float64), header


def _save(path: str, mat: np.ndarray, header: list[str]) -> None:
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in mat:
            w.writerow(row.tolist())


def _cost(Z: np.ndarray) -> np.ndarray:
    C = ot.dist(Z, Z, metric="sqeuclidean")
    m = C.max()
    if m > 0:
        C = C / m
    return C


def run_pot(X: np.ndarray, Y: np.ndarray, epsilon: float,
            solver: str) -> np.ndarray:
    """Return warped source features at target locations: (n_y, p_x)."""
    Cx = _cost(X)
    Cy = _cost(Y)
    a = np.full(X.shape[0], 1.0 / X.shape[0])
    b = np.full(Y.shape[0], 1.0 / Y.shape[0])
    if solver == "entropic_gw":
        T = ot.gromov.entropic_gromov_wasserstein(
            Cx, Cy, a, b, loss_fun="square_loss",
            epsilon=epsilon, max_iter=1000, tol=1e-6,
        )
    else:
        T = ot.gromov.gromov_wasserstein(
            Cx, Cy, a, b, loss_fun="square_loss",
            max_iter=1000, tol=1e-6,
        )
    T = np.asarray(T)
    col_sum = T.sum(axis=0, keepdims=True)
    col_sum[col_sum == 0] = 1.0
    # warped[j, :] = sum_i T[i,j]/sum_i * X[i, :]
    warped = (T.T @ X) / col_sum.T
    return warped


def run_ensembleot(X: np.ndarray, Y: np.ndarray, epsilon: float,
                   n_clusters: int, n_runs: int,
                   random_state: int,
                   clustering_method: str = "kmeans",
                   coords_x: np.ndarray | None = None,
                   coords_y: np.ndarray | None = None) -> np.ndarray:
    from ensembleot import run_ensemble_gw, make_mean_operator
    ops = run_ensemble_gw(
        X, Y,
        n_clusters_x=min(n_clusters, X.shape[0]),
        n_clusters_y=min(n_clusters, Y.shape[0]),
        n_runs=n_runs,
        clustering_method=clustering_method,
        solver_method="entropic_gw",
        epsilon=epsilon,
        random_state=random_state,
        coords_x=coords_x,
        coords_y=coords_y,
    )
    return _apply_mean(make_mean_operator(ops), X)


def _apply_mean(mean_op, X: np.ndarray) -> np.ndarray:
    warped = mean_op.apply_transpose_to_features(X)
    ones = np.ones((X.shape[0], 1))
    col_mass = mean_op.apply_transpose_to_features(ones)
    col_mass[col_mass == 0] = 1.0
    return warped / col_mass


def _cluster_anatomy_dist(A_x: np.ndarray, A_y: np.ndarray):
    """Build a cross_feature_cost_fn closure for EnsembleOT FGW.

    For each cluster k, computes the mean anatomy one-hot (a probability
    distribution over regions) and returns the K_x × K_y squared-Euclidean
    distance between those distributions.
    """
    def fn(*, X, Y, centers_x, centers_y, labels_x, labels_y,
           n_clusters_x, n_clusters_y, seed, metric, **_):
        Px = np.zeros((n_clusters_x, A_x.shape[1]))
        for k in range(n_clusters_x):
            mask = labels_x == k
            if mask.any():
                Px[k] = A_x[mask].mean(axis=0)
        Py = np.zeros((n_clusters_y, A_y.shape[1]))
        for k in range(n_clusters_y):
            mask = labels_y == k
            if mask.any():
                Py[k] = A_y[mask].mean(axis=0)
        return ot.dist(Px, Py, metric="sqeuclidean")
    return fn


def run_ensembleot_fgw(X, Y, A_x, A_y, epsilon, n_clusters, n_runs,
                       alpha, random_state,
                       clustering_method: str = "kmeans",
                       coords_x: np.ndarray | None = None,
                       coords_y: np.ndarray | None = None):
    from ensembleot import run_ensemble_fgw, make_mean_operator
    ops = run_ensemble_fgw(
        X, Y,
        n_clusters_x=min(n_clusters, X.shape[0]),
        n_clusters_y=min(n_clusters, Y.shape[0]),
        n_runs=n_runs,
        clustering_method=clustering_method,
        solver_method="entropic_fgw",
        alpha=alpha,
        epsilon=epsilon,
        random_state=random_state,
        cross_feature_cost_fn=_cluster_anatomy_dist(A_x, A_y),
        coords_x=coords_x,
        coords_y=coords_y,
    )
    return _apply_mean(make_mean_operator(ops), X)


def run_pot_fgw(X, Y, A_x, A_y, epsilon, alpha):
    """Sample-level FGW via POT. M is pairwise anatomy distance (n_x × n_y).

    Will OOM for large datasets — let the caller handle the exception.
    """
    Cx = _cost(X)
    Cy = _cost(Y)
    M = ot.dist(A_x.astype(np.float64), A_y.astype(np.float64),
                metric="sqeuclidean")
    if M.max() > 0:
        M = M / M.max()
    a = np.full(X.shape[0], 1.0 / X.shape[0])
    b = np.full(Y.shape[0], 1.0 / Y.shape[0])
    T = ot.gromov.entropic_fused_gromov_wasserstein(
        M, Cx, Cy, a, b, loss_fun="square_loss",
        alpha=alpha, epsilon=epsilon, max_iter=1000, tol=1e-6,
    )
    T = np.asarray(T)
    col_sum = T.sum(axis=0, keepdims=True)
    col_sum[col_sum == 0] = 1.0
    return (T.T @ X) / col_sum.T


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--backend", required=True,
                   choices=["pot", "pot_entropic", "ensembleot",
                            "pot_fgw_entropic", "ensembleot_fgw"])
    p.add_argument("--source-anatomy", default=None,
                   help="One-hot anatomy CSV for source (required for *_fgw).")
    p.add_argument("--target-anatomy", default=None,
                   help="One-hot anatomy CSV for target (required for *_fgw).")
    p.add_argument("--alpha", type=float, default=0.5,
                   help="FGW trade-off (POT semantics).")
    p.add_argument("--n-clusters", type=int, default=20)
    p.add_argument("--n-runs", type=int, default=5)
    p.add_argument("--clustering-method", default="kmeans",
                   choices=["kmeans", "random_voronoi"],
                   help="Clustering method for EnsembleOT.")
    p.add_argument("--source-coords", default=None,
                   help="Spatial coordinates CSV for source (2 cols: X,Y).")
    p.add_argument("--target-coords", default=None,
                   help="Spatial coordinates CSV for target (2 cols: X,Y).")
    p.add_argument("--epsilon", type=float, default=0.05)
    p.add_argument("--random-state", type=int, default=0)
    p.add_argument("--timing", default=None)
    args = p.parse_args()

    X, src_header = _load(args.source)
    Y, _ = _load(args.target)

    needs_anatomy = args.backend in ("pot_fgw_entropic", "ensembleot_fgw")
    A_x = A_y = None
    if needs_anatomy:
        if not args.source_anatomy or not args.target_anatomy:
            raise SystemExit(f"backend {args.backend} requires --source-anatomy and --target-anatomy")
        A_x, _ = _load(args.source_anatomy)
        A_y, _ = _load(args.target_anatomy)

    # Spatial coordinates for clustering (optional; used by random_voronoi)
    coords_x = coords_y = None
    if args.source_coords:
        coords_x, _ = _load(args.source_coords)
    if args.target_coords:
        coords_y, _ = _load(args.target_coords)

    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t0 = time.perf_counter()
    if args.backend == "pot":
        warped = run_pot(X, Y, args.epsilon, solver="gw")
    elif args.backend == "pot_entropic":
        warped = run_pot(X, Y, args.epsilon, solver="entropic_gw")
    elif args.backend == "ensembleot":
        warped = run_ensembleot(
            X, Y, args.epsilon,
            n_clusters=args.n_clusters,
            n_runs=args.n_runs,
            random_state=args.random_state,
            clustering_method=args.clustering_method,
            coords_x=coords_x,
            coords_y=coords_y,
        )
    elif args.backend == "pot_fgw_entropic":
        warped = run_pot_fgw(X, Y, A_x, A_y, args.epsilon, args.alpha)
    elif args.backend == "ensembleot_fgw":
        warped = run_ensembleot_fgw(
            X, Y, A_x, A_y, args.epsilon,
            n_clusters=args.n_clusters,
            n_runs=args.n_runs,
            alpha=args.alpha,
            random_state=args.random_state,
            clustering_method=args.clustering_method,
            coords_x=coords_x,
            coords_y=coords_y,
        )
    else:
        raise SystemExit(f"unknown backend {args.backend}")
    elapsed = time.perf_counter() - t0

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    _save(args.out, warped, src_header)

    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is in KB on Linux
    peak_rss_mb = rss_after / 1024.0

    info = {
        "backend": args.backend,
        "elapsed_sec": elapsed,
        "peak_rss_mb": peak_rss_mb,
        "n_source": int(X.shape[0]),
        "n_target": int(Y.shape[0]),
        "p_source": int(X.shape[1]),
        "p_target": int(Y.shape[1]),
    }
    if args.timing:
        Path(args.timing).parent.mkdir(parents=True, exist_ok=True)
        with open(args.timing, "w") as f:
            json.dump(info, f, indent=2)
    print(json.dumps(info))
    return 0


if __name__ == "__main__":
    sys.exit(main())
