# ImageRegistration-experiments4

Quantile-GuidedPLS: クオンタイル逆変換とGuidedPLSを組み合わせたクロスモーダル空間アライメント手法のプロトタイプ。

## 目的

experiments3（通常のGuidedPLS）との比較実験。Quantile transformによる非線形スケール正規化が、GuidedPLSの輸送精度を改善するかを検証する。

**効果がなければこのリポジトリは削除する。**

## アーキテクチャ

```
src/
  functions_quantile.R    # Feature-wise quantile transform ([0,1] mapping)
  functions_inverse.R     # Inverse mapping (isotonic regression / random forest)
  functions_pipeline.R    # End-to-end pipeline + evaluation metrics
  run_toy_pipeline.R      # Toy data での動作確認
  run_synthetic_benchmark.R  # Synthetic benchmark (noise/dropout/imbalance)
data/toy/                 # Toy data
output/toy/               # Toy pipeline 結果
output/benchmark/         # Benchmark 結果
plot/benchmark/           # Benchmark プロット
```

## 手法の流れ

```
Source X, Target Y (異なるモダリティ)
    ↓
1. Feature-wise Quantile Transform → X_q, Y_q ∈ [0,1]
   (各列を empirical CDF で [0,1] に写像、average rank)
    ↓
2. Target側の Inverse Mapping を学習
   (quantile value → original value の単調回帰: isotonic or RF)
    ↓
3. GuidedPLS(X_q, Y_q, A_source, A_target)
   (anatomy ガイド付き潜在空間推定)
    ↓
4. kNNWarping(r=18, k=11)
   (潜在空間でソース→ターゲット輸送)
    ↓
5. Inverse Transform
   (輸送されたランク値をターゲットの元スケールに復元)
```

## コマンド

```bash
# Toy data で動作確認
Rscript src/run_toy_pipeline.R

# Synthetic benchmark
Rscript src/run_synthetic_benchmark.R
```

## 依存パッケージ

- `guidedPLS` (rikenbit/guidedPLS) - メイン手法、再実装しない
- `RANN` - kNN探索
- `ggplot2`, `dplyr` - 可視化・データ操作
- `randomForest` - RF inverse (optional)

## experiments3 との比較方法

同一データ（251208, kidney）に対して:
- experiments3: GuidedPLS(X, Y) → kNNWarping → warped_exp
- experiments4: GuidedPLS(X_q, Y_q) → kNNWarping → inverse_transform → warped_exp

評価指標: Pearson CC（マーカーペア間の相関係数）
