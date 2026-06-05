# 次世代生成アルゴリズムの問題修正レポート

**日付**: 2026年1月20日  
**修正対象**: Interactive Evolutionary Computation (IEC) の次世代生成アルゴリズム

---

## 発見された問題点

### 1. **seed値の不適切な処理** ⚠️ 重大

#### 問題の詳細
- **ファイル**: `audioldm/iec.py`
- **関数**: `crossover_slerp()`, `mutate_gaussian()`
- **症状**: 
  - 交叉・変異後の子個体に`seed`値が設定されていない（`None`のまま）
  - 遺伝子型（潜在ノイズ）が変更されているにもかかわらず、seedが更新されない
  - 再現性が失われ、同じ遺伝子型から異なる音声が生成される可能性

#### 修正内容
```python
# crossover_slerp関数
# 修正前: seedが設定されていない
child = AudioGenotype(
    latent_noise=child_latent,
    conditioning=child_conditioning,
    # seed が None のまま
)

# 修正後: 新しいseedを生成
child_seed = np.random.randint(0, 2**32 - 1)
child = AudioGenotype(
    latent_noise=child_latent,
    conditioning=child_conditioning,
    seed=child_seed,  # 新しいseedを設定
    metadata={
        "parent1_seed": parent1.seed,  # 親のseedも記録
        "parent2_seed": parent2.seed,
        ...
    }
)

# mutate_gaussian関数も同様に修正
mutant.seed = np.random.randint(0, 2**32 - 1)
mutant.metadata["parent_seed"] = individual.seed
```

#### 影響
- **修正前**: 同じ遺伝子型でも生成のたびに異なる音声が生成される可能性
- **修正後**: seedが適切に管理され、遺伝子型と生成結果の対応が保証される

---

### 2. **音声生成時にseed値が使用されていない** ⚠️ 重大

#### 問題の詳細
- **ファイル**: `audioldm/iec_pipeline.py`
- **関数**: `_generate_audio_from_genotype()`
- **症状**:
  - 遺伝子型が持つ`seed`値が全く使用されていない
  - ランダム状態が管理されていないため、再現性がない
  - 同じ遺伝子型から毎回異なる音声が生成される可能性

#### 修正内容
```python
# 修正前: seedが無視されている
with torch.no_grad():
    x_T = genotype.latent_noise.to(self.device)
    # seedを使わずに生成処理

# 修正後: seedを使用してランダム状態を設定
with torch.no_grad():
    # 遺伝子型のseedを使用してランダム状態を設定（再現性のため）
    if genotype.seed is not None:
        torch.manual_seed(genotype.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(genotype.seed)
        np.random.seed(genotype.seed % (2**32))
    
    x_T = genotype.latent_noise.to(self.device)
    # 以降の生成処理
```

#### 影響
- **修正前**: IECの進化が不安定で、選択した個体の特徴が次世代に正確に継承されない
- **修正後**: 遺伝子型に基づいた再現可能な音声生成が保証される

---

### 3. **条件付けベクトル（テキストエンベディング）の不適切な変異** ⚠️ 中程度

#### 問題の詳細
- **ファイル**: `audioldm/iec.py`
- **関数**: `mutate_gaussian()`, `crossover_slerp()`
- **症状**:
  - テキストエンベディング（プロンプトの意味を表すベクトル）にランダムノイズを加えている
  - プロンプトの意味が変わってしまい、ユーザーの意図と異なる進化が起きる
  - IECの目的はプロンプトの意味を保ちながら音声特徴を進化させることなので、これは不適切

#### 修正内容
```python
# mutate_gaussian関数
# 修正前: conditioningにもノイズを加えている
if mutant.conditioning is not None and np.random.random() < mutation_rate * 0.5:
    cond_noise = torch.randn_like(mutant.conditioning) * (mutation_strength * 0.5)
    mutant.conditioning = mutant.conditioning + cond_noise

# 修正後: conditioningは変異させない
# 条件付けベクトルは変異させない
# (テキストエンベディングを変異させると、元のプロンプトの意味が変わってしまうため)
# mutant.conditioningはそのまま保持

# crossover_slerp関数
# 修正前: 両親のconditioningを補間
if parent1.conditioning is not None and parent2.conditioning is not None:
    child_conditioning = slerp(parent1.conditioning, parent2.conditioning, alpha)

# 修正後: 親1のconditioningをそのまま使用
if parent1.conditioning is not None:
    child_conditioning = parent1.conditioning.clone()
elif parent2.conditioning is not None:
    child_conditioning = parent2.conditioning.clone()
```

#### 理由
- テキストエンベディングは高度に最適化された意味表現であり、線形/球面補間しても意味的に有効なベクトルになるとは限らない
- IECでは「プロンプトの意味は固定」して「音声の特徴（潜在ノイズ）を進化」させるべき
- 潜在ノイズの多様性だけで十分な探索空間が得られる

#### 影響
- **修正前**: 進化の過程でプロンプトの意味が徐々にずれていく可能性
- **修正後**: プロンプトの意味を保ちながら、音響的特徴のみを進化させることができる

---

### 4. **seed値の範囲の不一致** ⚠️ 軽微

#### 問題の詳細
- **ファイル**: `audioldm/iec.py`, `audioldm/iec_pipeline.py`
- **症状**: seed値の生成範囲が統一されていない
  - `iec_pipeline.py`: `np.random.randint(0, 2**31 - 1)` （31ビット）
  - `iec.py`: `np.random.randint(0, 2**32 - 1)` （32ビット）

#### 修正内容
- 全ての箇所で`np.random.randint(0, 2**32 - 1)`に統一
- PyTorchの`manual_seed()`は32ビット符号なし整数を受け付けるため、32ビット範囲を使用

#### 影響
- **修正前**: 微妙な不整合が存在
- **修正後**: 一貫したseed管理

---

### 5. **初期化時のseed使用の欠如** ⚠️ 中程度

#### 問題の詳細
- **ファイル**: `audioldm/iec_pipeline.py`
- **関数**: `initialize_population()`
- **症状**:
  - seedを生成しているが、それを使って潜在ノイズを生成していない
  - `torch.randn()`がグローバルなランダム状態を使用するため、再現性がない

#### 修正内容
```python
# 修正前: seedを使わずにランダムノイズ生成
seed = np.random.randint(0, 2**31 - 1)
latent_noise = torch.randn((1,) + self.latent_shape, device=self.device)

# 修正後: seedを使用してランダムノイズ生成
seed = np.random.randint(0, 2**32 - 1)
generator = torch.Generator(device=self.device).manual_seed(seed)
latent_noise = torch.randn(
    (1,) + self.latent_shape, 
    device=self.device, 
    generator=generator
)
```

#### 影響
- **修正前**: 同じseedでも異なる初期個体群が生成される
- **修正後**: seedに基づいた完全に再現可能な初期化

---

## 修正によって解決される問題

### Before（修正前）
1. **再現性の欠如**: 同じ遺伝子型から異なる音声が生成される
2. **進化の不安定性**: 選択した個体の特徴が次世代に正確に継承されない
3. **意味のずれ**: プロンプトの意味が進化の過程で変化する
4. **デバッグ困難**: 問題の原因特定が極めて難しい

### After（修正後）
1. **完全な再現性**: 遺伝子型（seed + 潜在ノイズ）から常に同じ音声が生成される
2. **安定した進化**: 選択した個体の特徴が確実に次世代に継承される
3. **意味の保持**: プロンプトの意味を保ちながら音響特徴のみを進化させる
4. **デバッグ可能**: seedとメタデータによる完全なトレーサビリティ

---

## 技術的背景

### なぜseed値の線形補間が問題か

```python
# ❌ 間違った方法（指摘された問題）
parent1_seed = 42
parent2_seed = 100
child_seed = int(0.5 * parent1_seed + 0.5 * parent2_seed)  # = 71

# この方法の問題:
# 1. seedはランダム生成の「開始点」であり、線形補間に意味がない
# 2. 潜在ノイズを補間しているなら、それに対応するseedは存在しない
# 3. 再現性が壊れる
```

### 正しいアプローチ

```python
# ✅ 正しい方法
# 1. 潜在ノイズを補間（球面線形補間）
child_latent = slerp(parent1.latent_noise, parent2.latent_noise, 0.5)

# 2. 新しい子個体には新しいseedを割り当てる
child_seed = np.random.randint(0, 2**32 - 1)

# 3. 親のseedはメタデータとして保存（トレーサビリティのため）
child.metadata = {
    "parent1_seed": parent1.seed,
    "parent2_seed": parent2.seed,
    "child_seed": child_seed,
}
```

### 潜在ノイズとseedの関係

```
seed → Random Generator → 潜在ノイズ → 拡散モデル → 音声
 ↓                          ↓
 記録                      実際の遺伝子

進化操作:
- 交叉: 2つの潜在ノイズを補間 → 新しいseedを生成
- 変異: 潜在ノイズにノイズ追加 → 新しいseedを生成
- 再生成: seed → 潜在ノイズを再生成 → 同じ音声が得られる
```

---

## テスト方法

### 1. 再現性のテスト
```python
# 同じ遺伝子型から同じ音声が生成されることを確認
genotype = population[0]
audio1 = pipeline._generate_audio_from_genotype(genotype, "explosion")
audio2 = pipeline._generate_audio_from_genotype(genotype, "explosion")
assert np.allclose(audio1, audio2), "再現性が保証されていない"
```

### 2. 進化の一貫性テスト
```python
# 選択した個体の特徴が次世代に継承されることを確認
selected = [0, 1]  # 最初の2個体を選択
next_gen = pipeline.evolve_population(selected)

# エリート個体が変化していないことを確認
elite = next_gen[0][0]
original = population[0]
assert torch.allclose(elite.latent_noise, original.latent_noise)
```

### 3. seedの一貫性テスト
```python
# 全ての個体がseedを持つことを確認
for genotype, _ in generation_results:
    assert genotype.seed is not None, f"seedが設定されていない: {genotype.id}"
```

---

## 今後の改善提案

1. **seedベースの遺伝子型再構築**
   - seedから潜在ノイズを完全に再生成できるようにする
   - ストレージ効率の向上（潜在ノイズは大きいが、seedは4バイト）

2. **進化履歴の可視化**
   - 親子関係とseedを使った系統樹の生成
   - 各世代の特徴変化の追跡

3. **高度な交叉戦略**
   - 複数点交叉（潜在空間の異なる次元で異なる補間率）
   - 適応的補間率（個体の類似度に応じてalphaを調整）

4. **条件付けベクトルの多様化**
   - 複数プロンプトの補間（将来的に導入する場合）
   - プロンプトウェイトの進化

---

## まとめ

この修正により、IECシステムの**再現性**、**安定性**、**トレーサビリティ**が大幅に向上しました。特に、seed値の適切な管理により、ユーザーが選択した個体の特徴を確実に次世代に継承できるようになり、効果的な対話型進化が可能になりました。

**重要**: この修正は、既存のIECセッションの履歴との互換性に影響を与える可能性があります。新しいセッションから使用することを推奨します。
