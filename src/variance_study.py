"""EnsembleOT variance study.

Runs four experiments on a single (X, Y) pair:

  1. run_level        — fixed random_state, n_runs trials, per-run warped
                         matrix of shape (n_target, p_source). Output:
                         warped_run_<i>.csv
  2. ensemble_size    — sweep n_runs ∈ sizes, n_seeds independent outer
                         seeds per size. Mean operator per (size, seed).
                         Output: ensemble_size.csv (summary) +
                         warped_size<N>_seed<S>.csv
  3. cluster_count    — sweep n_clusters_{x,y} ∈ ks, n_seeds seeds each.
                         Output: cluster_count.csv + warped_k<K>_seed<S>.csv
  4. reproducibility  — run three times with identical random_state; write
                         hashes + a boolean identical flag.

The X / Y features carried through the operators are the source features
(we produce warped source features at target locations via
apply_transpose_to_features). Cluster-preservation evaluation is left to
the R driver — this script only outputs the warped matrices and timings.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from ensembleot import (
    run_ensemble_gw,
    make_mean_operator,
)


def _load(path: str) -> tuple[np.ndarray, list[str]]:
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [[float(v) for v in row] for row in reader]
    return np.asarray(rows, dtype=np.float64), header


def _save(path: str, mat: np.ndarray, header: list[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in mat:
            w.writerow(row.tolist())


def _warped_from_op(op, X: np.ndarray) -> np.ndarray:
    # T.T @ X with row-normalization so each target row is a convex
    # combination of source rows (mass-agnostic average).
    w = op.apply_transpose_to_features(X)
    ones = np.ones((X.shape[0], 1))
    mass = op.apply_transpose_to_features(ones)
    mass[mass == 0] = 1.0
    return w / mass


def experiment_run_level(X, Y, header, outdir, n_runs, n_clusters, epsilon,
                         random_state):
    t0 = time.perf_counter()
    ops = run_ensemble_gw(
        X, Y, n_clusters_x=n_clusters, n_clusters_y=n_clusters,
        n_runs=n_runs, solver_method="entropic_gw",
        epsilon=epsilon, random_state=random_state,
    )
    elapsed_all = time.perf_counter() - t0
    per_run = []
    for i, op in enumerate(ops):
        w = _warped_from_op(op, X)
        _save(f"{outdir}/warped_run_{i:03d}.csv", w, header)
        per_run.append({
            "i": i,
            "seed": int(op.meta.get("seed", -1)),
            **{f"metric_{k}": v for k, v in op.meta.get("metrics", {}).items()
               if isinstance(v, (int, float))},
        })
    with open(f"{outdir}/run_level_meta.json", "w") as f:
        json.dump({"n_runs": n_runs, "n_clusters": n_clusters,
                   "epsilon": epsilon, "elapsed_all_sec": elapsed_all,
                   "per_run": per_run}, f, indent=2)
    print(f"[run_level] {n_runs} ops in {elapsed_all:.2f}s")


def experiment_ensemble_size(X, Y, header, outdir, sizes, n_seeds,
                             n_clusters, epsilon):
    summary_rows = []
    for size in sizes:
        for seed in range(n_seeds):
            t0 = time.perf_counter()
            ops = run_ensemble_gw(
                X, Y, n_clusters_x=n_clusters, n_clusters_y=n_clusters,
                n_runs=size, solver_method="entropic_gw",
                epsilon=epsilon, random_state=1000 + seed,
            )
            mop = make_mean_operator(ops)
            w = _warped_from_op(mop, X)
            elapsed = time.perf_counter() - t0
            _save(f"{outdir}/warped_size{size}_seed{seed}.csv", w, header)
            summary_rows.append({
                "n_runs": size, "seed": seed, "elapsed_sec": elapsed,
                "n_clusters": n_clusters,
            })
            print(f"[ensemble_size] size={size} seed={seed} {elapsed:.2f}s")
    with open(f"{outdir}/ensemble_size.json", "w") as f:
        json.dump(summary_rows, f, indent=2)


def experiment_cluster_count(X, Y, header, outdir, ks, n_seeds, n_runs,
                             epsilon):
    summary_rows = []
    for k in ks:
        for seed in range(n_seeds):
            k_eff = min(k, X.shape[0], Y.shape[0])
            t0 = time.perf_counter()
            ops = run_ensemble_gw(
                X, Y, n_clusters_x=k_eff, n_clusters_y=k_eff,
                n_runs=n_runs, solver_method="entropic_gw",
                epsilon=epsilon, random_state=2000 + seed,
            )
            mop = make_mean_operator(ops)
            w = _warped_from_op(mop, X)
            elapsed = time.perf_counter() - t0
            _save(f"{outdir}/warped_k{k}_seed{seed}.csv", w, header)
            summary_rows.append({
                "n_clusters": k, "n_clusters_eff": k_eff,
                "seed": seed, "elapsed_sec": elapsed, "n_runs": n_runs,
            })
            print(f"[cluster_count] k={k_eff} seed={seed} {elapsed:.2f}s")
    with open(f"{outdir}/cluster_count.json", "w") as f:
        json.dump(summary_rows, f, indent=2)


def experiment_reproducibility(X, Y, header, outdir, n_runs, n_clusters,
                               epsilon, random_state, n_repeats):
    hashes = []
    for rep in range(n_repeats):
        ops = run_ensemble_gw(
            X, Y, n_clusters_x=n_clusters, n_clusters_y=n_clusters,
            n_runs=n_runs, solver_method="entropic_gw",
            epsilon=epsilon, random_state=random_state,
        )
        mop = make_mean_operator(ops)
        w = _warped_from_op(mop, X)
        _save(f"{outdir}/warped_repeat{rep}.csv", w, header)
        h = hashlib.sha256(w.tobytes()).hexdigest()
        hashes.append(h)
        print(f"[reproducibility] rep={rep} hash={h[:12]}")
    identical = len(set(hashes)) == 1
    with open(f"{outdir}/reproducibility.json", "w") as f:
        json.dump({"hashes": hashes, "identical": identical,
                   "random_state": random_state, "n_runs": n_runs}, f, indent=2)
    print(f"[reproducibility] identical={identical}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--n-clusters", type=int, default=20)
    p.add_argument("--epsilon", type=float, default=0.05)
    # run_level
    p.add_argument("--run-level-n-runs", type=int, default=30)
    p.add_argument("--run-level-seed", type=int, default=0)
    # ensemble_size
    p.add_argument("--ensemble-sizes", default="1,3,5,10,20,50")
    p.add_argument("--ensemble-n-seeds", type=int, default=10)
    # cluster_count
    p.add_argument("--cluster-ks", default="5,10,20,40,80")
    p.add_argument("--cluster-n-seeds", type=int, default=5)
    p.add_argument("--cluster-n-runs", type=int, default=5)
    # reproducibility
    p.add_argument("--repro-n-repeats", type=int, default=3)
    p.add_argument("--repro-n-runs", type=int, default=5)
    p.add_argument("--repro-seed", type=int, default=42)
    # skip switches
    p.add_argument("--skip-run-level", action="store_true")
    p.add_argument("--skip-ensemble-size", action="store_true")
    p.add_argument("--skip-cluster-count", action="store_true")
    p.add_argument("--skip-reproducibility", action="store_true")
    args = p.parse_args()

    X, src_header = _load(args.source)
    Y, _ = _load(args.target)
    print(f"X={X.shape} Y={Y.shape}")
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    if not args.skip_run_level:
        experiment_run_level(
            X, Y, src_header, f"{args.outdir}/run_level",
            n_runs=args.run_level_n_runs,
            n_clusters=args.n_clusters,
            epsilon=args.epsilon,
            random_state=args.run_level_seed,
        )
    if not args.skip_ensemble_size:
        experiment_ensemble_size(
            X, Y, src_header, f"{args.outdir}/ensemble_size",
            sizes=[int(s) for s in args.ensemble_sizes.split(",")],
            n_seeds=args.ensemble_n_seeds,
            n_clusters=args.n_clusters,
            epsilon=args.epsilon,
        )
    if not args.skip_cluster_count:
        experiment_cluster_count(
            X, Y, src_header, f"{args.outdir}/cluster_count",
            ks=[int(s) for s in args.cluster_ks.split(",")],
            n_seeds=args.cluster_n_seeds,
            n_runs=args.cluster_n_runs,
            epsilon=args.epsilon,
        )
    if not args.skip_reproducibility:
        experiment_reproducibility(
            X, Y, src_header, f"{args.outdir}/reproducibility",
            n_runs=args.repro_n_runs,
            n_clusters=args.n_clusters,
            epsilon=args.epsilon,
            random_state=args.repro_seed,
            n_repeats=args.repro_n_repeats,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
