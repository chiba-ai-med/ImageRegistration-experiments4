# ImageRegistration-experiments4

**Quantile-GuidedPLS**: クオンタイル逆変換と GuidedPLS を組み合わせたクロスモーダル空間アライメント手法のプロトタイプ。

## 目的

[experiments3](https://github.com/chiba-ai-med/) の通常 GuidedPLS との比較実験。Feature-wise quantile transform による非線形スケール正規化が GuidedPLS の輸送精度を改善するかを検証する。

> **Note**: 効果が確認できなければ本リポジトリは削除する予定のプロトタイプ。

## 手法の流れ

```
Source X, Target Y (異なるモダリティ)
    │
    ▼
1. Feature-wise Quantile Transform → X_q, Y_q ∈ [0,1]
   (各列を empirical CDF で [0,1] に写像、average rank)
    │
    ▼
2. Target 側の Inverse Mapping を学習
   (quantile value → original value の単調回帰: isotonic / RF)
    │
    ▼
3. GuidedPLS(X_q, Y_q, A_source, A_target)
   (anatomy ガイド付き潜在空間推定)
    │
    ▼
4. kNNWarping (r=18, k=11)
   (潜在空間でソース → ターゲット輸送)
    │
    ▼
5. Inverse Transform
   (輸送されたランク値をターゲットの元スケールに復元)
```

## ディレクトリ構成

```
src/
  functions_quantile.R         # Feature-wise quantile transform ([0,1] mapping)
  functions_inverse.R          # Inverse mapping (isotonic regression / random forest)
  functions_pipeline.R         # End-to-end pipeline + evaluation metrics
  functions_gwot.R             # Gromov-Wasserstein OT 関連
  run_toy_pipeline.R           # Toy data での動作確認
  run_synthetic_benchmark.R    # Synthetic benchmark (noise / dropout / imbalance)
  run_real_data.R              # 実データでの実行
  run_realdata_comparison.R    # 実データ比較
  run_cluster_sweep.R          # クラスタ数スイープ
  run_variance_study.R         # 分散の影響評価
  run_gwot.py / variance_study.py  # GWOT 関連 Python スクリプト
```

データ（`data/`）と結果（`output/`, `plot/`, `paper/`）は `.gitignore` で除外している。

## 実行方法

```bash
# Toy data で動作確認
Rscript src/run_toy_pipeline.R

# Synthetic benchmark
Rscript src/run_synthetic_benchmark.R
```

## 依存パッケージ

- [`guidedPLS`](https://github.com/rikenbit/guidedPLS) — メイン手法（再実装しない）
- `RANN` — kNN 探索
- `randomForest` — RF inverse (optional)
- `ggplot2`, `dplyr` — 可視化・データ操作

## experiments3 との比較

同一データ (251208, kidney) に対して:

| 手法 | パイプライン |
|------|------|
| experiments3 | `GuidedPLS(X, Y) → kNNWarping → warped_exp` |
| experiments4 | `GuidedPLS(X_q, Y_q) → kNNWarping → inverse_transform → warped_exp` |

評価指標: Pearson CC（マーカーペア間の相関係数）。

## ライセンス・所属

Artificial Intelligence Medicine, Graduate School of Medicine, Chiba University ([chiba-ai-med](https://github.com/chiba-ai-med))
