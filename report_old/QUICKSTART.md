# 🚀 AudioLDM-IEC クイックスタートガイド

## 対話型進化的効果音生成システムを5分で始める

### 1️⃣ 必要な依存関係をインストール

```bash
# Gradioをインストール（IEC機能に必要）
pip install gradio>=4.0.0

# AudioLDMの依存関係は既にインストール済みと仮定
```

### 2️⃣ コア機能のテスト（オプションだが推奨）

```bash
python scripts/test_iec_core.py
```

全てのテストが成功すれば、システムは正常に動作しています。

### 3️⃣ Webインターフェースを起動

```bash
python scripts/launch_iec_gradio.py
```

ブラウザで自動的に開くか、手動で `http://localhost:7860` にアクセス。

### 4️⃣ 音を進化させる

1. **プロンプトを入力**（例：「爆発音」「雷の音」「ドアが閉まる音」）
2. **「🎲 初期個体群を生成」をクリック**
3. **6つの音声が生成される → 聴く**
4. **気に入った音を選択**（複数選択可）
5. **「🧬 次世代を生成」をクリック**
6. **満足するまで 4-5 を繰り返す**
7. **「💾 セッション保存」で結果を保存**

### 📊 パラメータの調整

- **突然変異率** (0.3推奨): 高いと変化が激しい
- **突然変異強度** (0.15推奨): 高いと大きく変化
- **エリート保存数** (1推奨): 優秀個体を確実に保存

### 💡 Tips

- 探索が行き詰まったら **「🔙 1世代戻る」** を使う
- 初期世代で多様性が低い場合は、**初期変異強度** を上げる
- 良い音が見つかったら、エリート保存数を2にして確実に保存

## CLI版を使う場合

```bash
python scripts/run_iec_cli.py --prompt "雷の音" --population_size 4 --duration 3.0
```

ターミナルで個体番号を入力して選択（例：`0 2 3`）

## トラブルシューティング

### GPUメモリ不足の場合

```bash
python scripts/launch_iec_gradio.py --population_size 4 --duration 3.0
```

### ポートが使用中の場合

```bash
python scripts/launch_iec_gradio.py --port 8080
```

## 📚 詳細ドキュメント

- **使い方の詳細**: `IEC_README.md`
- **実装の詳細**: `IMPLEMENTATION_REPORT.md`
- **研究背景**: `研究計画書.md`

## 🎓 研究プロトタイプとして使う

```python
from audioldm import AudioLDM_IEC

# システム初期化
iec = AudioLDM_IEC(population_size=6, duration=5.0)

# 初期個体群生成
results = iec.initialize_population(prompt="爆発音")

# 音声保存
paths = iec.save_generation_audio(results, "./output/my_experiment")

# 次世代生成（個体0と2を選択）
results = iec.evolve_population(selected_indices=[0, 2])

# 履歴保存
iec.population.save_history("./output/history.json")
```

---

**すぐに始めるには**:
```bash
python scripts/launch_iec_gradio.py
```

それだけです！ブラウザで音の進化を楽しんでください 🎵
