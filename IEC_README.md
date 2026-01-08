# AudioLDM-IEC: 対話型進化的効果音生成システム

## 概要

AudioLDM-IECは、AudioLDM（Audio Latent Diffusion Model）と対話型進化計算（Interactive Evolutionary Computation, IEC）を組み合わせた、革新的な効果音生成システムです。

従来のテキストベース生成では困難だった「言語化できない理想の音」を、聴覚フィードバックと進化計算によって探索・生成します。

## 🎯 主な特徴

- **主観的評価ベース**: ユーザーが実際に音を聴いて評価するため、言語化困難な微細なニュアンスを反映可能
- **球面線形補間（Slerp）**: 高次元潜在空間での滑らかな交叉により、自然な音の遷移を実現
- **適応的突然変異**: 世代数と収束状況に応じて自動調整し、探索と収束のバランスを最適化
- **履歴管理**: 過去の世代に戻って別の選択を試せる「タイムトラベル」機能
- **Gradio UI**: 直感的なWebインターフェースで、コーディング不要で利用可能

## 📁 ファイル構成

```
audioldm/
├── iec.py                  # IECコアロジック（遺伝子型、交叉、突然変異）
├── iec_pipeline.py         # AudioLDM統合パイプライン
└── iec_gradio.py          # Gradio Webインターフェース

scripts/
├── launch_iec_gradio.py   # Gradio UI起動スクリプト
└── run_iec_cli.py         # CLI版IEC実行スクリプト
```

## 🚀 使い方

### 1. Gradio Webインターフェース版（推奨）

最も簡単な方法です。ブラウザで直感的に操作できます。

```bash
# デフォルト設定で起動
python scripts/launch_iec_gradio.py

# カスタム設定で起動
python scripts/launch_iec_gradio.py --population_size 8 --duration 3.0 --port 8080

# 公開リンクを生成（Gradio Share）
python scripts/launch_iec_gradio.py --share
```

起動後、ブラウザで `http://localhost:7860` にアクセスします。

#### Gradio UIの使い方

1. **初期化**
   - プロンプトを入力（例: "bomb"）
   - 「🎲 初期個体群を生成」をクリック
   - 6つの音声が生成されます

2. **選択と進化**
   - 生成された音声を聴く
   - 気に入った音声をチェックボックスで選択（複数可）
   - 「🧬 次世代を生成」をクリック
   - 選択した音声の特徴を受け継いだ次世代が生成されます

3. **パラメータ調整**
   - **突然変異率**: 変化の起こりやすさ（0.3推奨）
   - **突然変異強度**: 変化の大きさ（0.15推奨）
   - **エリート保存数**: 優秀個体を無変更で残す数（1推奨）

4. **その他の機能**
   - 🔙 1世代戻る: 探索が行き詰まった時に使用
   - 💾 セッション保存: 履歴と音声ファイルを保存

### 2. CLI（コマンドライン）版

ターミナルで対話的に実行します。

```bash
# プロンプトを指定
python scripts/run_iec_cli.py --prompt "a crash of thunder" --population_size 4 --duration 3.0

# ランダム初期化
python scripts/run_iec_cli.py --population_size 6 --max_generations 10
```

実行すると、音声が生成され、ターミナルで個体番号を入力して選択します。

```
--- 第0世代 ---
  [0] gen000_ind00.wav
  [1] gen000_ind01.wav
  [2] gen000_ind02.wav
  [3] gen000_ind03.wav

選択する個体番号 (例: 0 2 3): 1 3
```

### 3. Pythonスクリプトから利用

```python
from audioldm import AudioLDM_IEC

# システムの初期化
iec_system = AudioLDM_IEC(
    model_name="audioldm-s-full",
    population_size=6,
    duration=5.0
)

# 初期個体群を生成
results = iec_system.initialize_population(prompt="bomb sound")

# 結果を保存
saved_paths = iec_system.save_generation_audio(
    results,
    output_dir="./output/my_session",
    prefix="generation_0"
)

# 次世代を生成（個体0と2を選択）
results = iec_system.evolve_population(selected_indices=[0, 2])
```

## 🧬 アルゴリズムの詳細

### 遺伝子型（Genotype）

音声波形そのものではなく、以下を遺伝子として保持します：

- **潜在ノイズ**: 拡散モデルの初期ノイズベクトル
- **条件付けベクトル**: テキストエンベディング（オプション）
- **メタデータ**: 生成パラメータ、世代番号、親個体のID

### 交叉（Crossover）

**球面線形補間（Slerp: Spherical Linear Interpolation）** を使用：

```python
def slerp(v0, v1, t):
    """
    v0とv1を球面上でパラメータtに応じて補間
    高次元空間で滑らかな遷移を実現
    """
    theta = arccos(dot(v0, v1))
    return sin((1-t)*theta)/sin(theta) * v0 + sin(t*theta)/sin(theta) * v1
```

線形補間と比較して、潜在空間の幾何学的構造を保ちながら補間できます。

### 突然変異（Mutation）

ガウシアンノイズを付加：

```python
mutant = individual + N(0, σ²)
```

- 突然変異率: 変異が起こる確率（デフォルト: 0.3）
- 突然変異強度: ノイズの標準偏差（デフォルト: 0.15）

### 選択戦略

- **エリート保存**: 最良個体を次世代に無変更で残す
- **ユーザー選択**: ユーザーが主観的に選択した個体を親とする

## 📊 パラメータガイド

| パラメータ | 推奨値 | 説明 | 効果 |
|-----------|--------|------|------|
| `population_size` | 6 | 1世代の個体数 | 多いと多様性↑、計算時間↑ |
| `duration` | 3.0-5.0秒 | 音声の長さ | 長いと生成時間↑ |
| `mutation_rate` | 0.3 | 突然変異率 | 高いと探索範囲広がる |
| `mutation_strength` | 0.15 | 突然変異強度 | 高いと大きく変化 |
| `elite_count` | 1-2 | エリート保存数 | 良い個体を確実に保存 |

### パラメータチューニングのコツ

- **初期世代で多様性が低い場合**: `variation_strength` を上げる（0.5-0.7）
- **収束が遅い場合**: `mutation_rate` を下げる（0.1-0.2）
- **局所解に陥った場合**: `mutation_strength` を一時的に上げる（0.3-0.5）
- **良い音が見つかったら**: `elite_count` を2にして確実に保存

## 🎓 研究背景

本システムは、以下の研究課題に取り組むプロトタイプです：

### 課題
- テキストでは記述困難な「理想の音」の言語化の壁
- 擬音語やニュアンスの記述限界

### アプローチ
- 主観的評価（聴覚フィードバック）による探索
- 潜在空間での進化計算（Latent Variable Evolution）
- 球面線形補間による滑らかな音の遷移

### 懸念事項と対策

#### ユーザー疲労（User Fatigue）
**問題**: 音声は時間軸メディアのため、評価に時間がかかる

**対策**:
- 個体数を6個程度に抑える
- 音声長を3-5秒に制限
- エリート保存で良い個体を確実に残す
- ロールバック機能で探索の袋小路を回避

#### 推論レイテンシ
**問題**: 拡散モデルの推論に時間がかかる

**対策**:
- FP16精度での推論
- DDIM samplerによる高速化（200ステップ）
- GPU必須（VRAM 8GB以上推奨）

## 📈 今後の拡張予定

- [ ] 映像リファレンスの導入（CLIP条件付け）
- [ ] 適応的パラメータ調整の自動化
- [ ] マルチモーダル評価（映像との同期評価）
- [ ] 長時間音声への対応
- [ ] リアルタイム生成の高速化（LCM等）

## 🔧 トラブルシューティング

### GPUメモリ不足

```bash
# 個体数を減らす
python scripts/launch_iec_gradio.py --population_size 4

# 音声長を短くする
python scripts/launch_iec_gradio.py --duration 3.0
```

### 音声が生成されない

1. AudioLDMのチェックポイントが正しくダウンロードされているか確認
2. `~/.cache/audioldm` ディレクトリを確認

### 生成された音が似たり寄ったりになる

- 突然変異率を上げる（0.5程度）
- ロールバックして別の選択を試す
- 初期変異強度を上げて再起動

## 📚 参考文献

- AudioLDM: Text-to-Audio Generation with Latent Diffusion Models (Liu et al., 2023)
- Interactive Evolutionary Computation (Takagi, 2001)
- Spherical Linear Interpolation for Deep Learning (White, 2016)

## 📝 ライセンス

本プロジェクトは研究プロトタイプです。AudioLDMの元のライセンスに従います。

## 🙏 謝辞

- AudioLDMの開発者の皆様
- 対話型進化計算の先駆的研究者の皆様
