# 次世代生成アルゴリズム修正概要

## 修正日
2026年1月20日

## 問題の要約
Interactive Evolutionary Computation (IEC) の次世代生成アルゴリズムに、以下の重大な問題がありました:

1. **seed値が適切に管理されていない** (指摘された既知の問題)
2. **音声生成時にseedが使用されていない**
3. **条件付けベクトル（テキストエンベディング）が不適切に変異**

これらの問題により、IECの進化が不安定で再現性がなく、選択した個体の特徴が次世代に正確に継承されませんでした。

## 修正内容

### 1. 交叉関数 (`crossover_slerp`) の修正
**ファイル**: `audioldm/iec.py`

- ✅ 子個体に新しいseedを生成して設定
- ✅ 親のseedをメタデータに記録
- ✅ 条件付けベクトルを補間せず、親1のものを使用

### 2. 変異関数 (`mutate_gaussian`) の修正
**ファイル**: `audioldm/iec.py`

- ✅ 変異後の個体に新しいseedを生成して設定
- ✅ 親のseedをメタデータに記録
- ✅ 条件付けベクトルを変異させない（削除）

### 3. 音声生成関数 (`_generate_audio_from_genotype`) の修正
**ファイル**: `audioldm/iec_pipeline.py`

- ✅ 遺伝子型のseedを使用してランダム状態を設定
- ✅ PyTorchとNumPyの両方のランダム状態を制御
- ✅ 再現可能な音声生成を実現

### 4. 初期化関数 (`initialize_population`, `initialize_random`) の修正
**ファイル**: `audioldm/iec_pipeline.py`, `audioldm/iec.py`

- ✅ seedを使用して潜在ノイズを生成（`torch.Generator`使用）
- ✅ seed範囲を32ビット (`0, 2**32 - 1`) に統一
- ✅ 完全に再現可能な初期化を実現

## テスト結果

全てのテストが成功:

```
✅ テスト1: seedの一貫性テスト
✅ テスト2: 条件付けベクトルの保持テスト
✅ テスト3: 個体群初期化時のseed使用テスト
✅ テスト4: seedベースの潜在ノイズ再生成テスト
✅ テスト5: 交叉の性質テスト
```

## 効果

### Before（修正前）
- ❌ 再現性なし: 同じ遺伝子型から異なる音声
- ❌ 進化が不安定: 選択した特徴が継承されない
- ❌ デバッグ困難: 問題の原因特定が難しい

### After（修正後）
- ✅ 完全な再現性: 遺伝子型から常に同じ音声
- ✅ 安定した進化: 特徴が確実に次世代に継承
- ✅ トレーサビリティ: seedとメタデータで追跡可能

## 使用方法

修正後のIECシステムは既存のAPIと互換性があります:

```python
from audioldm.iec_pipeline import AudioLDM_IEC

# システムの初期化
iec = AudioLDM_IEC(
    model_name="audioldm-s-full-v2",
    population_size=6,
    duration=5.0
)

# 初期個体群を生成
results = iec.initialize_population(prompt="爆発音")

# 次世代を生成（選択した個体の特徴が確実に継承される）
next_gen = iec.evolve_population(selected_indices=[0, 2, 4])
```

## 関連ファイル

- `BUGFIX_NEXT_GENERATION.md` - 詳細な技術レポート
- `scripts/test_next_generation_fix.py` - テストスクリプト
- `audioldm/iec.py` - 遺伝的操作の実装
- `audioldm/iec_pipeline.py` - IECパイプラインの実装

## 注意事項

この修正は、既存のIECセッションの履歴ファイルとの互換性に影響を与える可能性があります。新しいセッションから使用することを推奨します。
